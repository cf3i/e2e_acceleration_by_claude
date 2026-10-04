# 最终报告：profile 驱动的 LLM Agent 优化（Qwen3.5-27B，单卡 H100）

本项目用来学习一条完整的优化流水线：**profile → 用 Amdahl 定律选下一步 → 实现 → 验证正确性 → 再 profile**。
约束：
- 不写自定义 kernel，只做上层优化；
- 可以安装已有的库，也可以改变已有 kernel 的启动方式。

每一步的证据、估算和实测都记录在 `OPTIMIZATION_LOG.md` 与 `results/<stage>/README.md` 中。

## 1. 场景与度量

**Agent**
- 求解 GSM8K 数学题：模型先调用 `python` 工具写程序算出答案，看到输出后给出 `\boxed{N}`；
- 每轮最多生成 512 token，最多 6 轮；
- 固定 128 题，greedy 解码。

**模型：Qwen3.5-27B**
- 混合架构：48 层 Gated DeltaNet（线性注意力，循环状态）+ 16 层 full attention；
- bf16 权重 54 GB。

**硬件与环境**
- 1 张 H100 80GB（SLURM `freecycle-h100`）；
- 每个实验都从 `slurm/submit.sh` 提交，它对已 commit 的代码做快照。

**指标**
- 主指标：128 题的 **makespan**（墙钟时间）；
- 同时看 accuracy、各部分时间占比、task 延迟分布。

**两个阶段**
- Phase 1：单请求推理引擎，agent 串行执行；
- Phase 2：多 agent 并发（batching、工具调用与解码重叠、prefix reuse 等）。

## 2. 结果总览

| stage | 改动 | makespan (s) | 累计加速 | accuracy |
|---|---|---|---|---|
| stage00_baseline | HF eager，DynamicCache | 1673.6 | 1.0× | 0.930 |
| p1_s04 | + StaticCache + CUDA graph 解码（含 p1_s01–s03：关 cuDNN SDPA、fla kernel） | 660.8 | 2.5× | 0.938 |
| p1_s07 | + async readback、torch.compile、GQA-folded 解码注意力 | 474.8 | 3.5× | 0.938 |
| p2_s01 | slot cache + batch-32 解码 graph + continuous batching | 60.8 | 27.5× | 0.938 |
| p2_s02 | + 跨轮 / 共享前缀复用（单独无收益，是下一步的前提） | 60.7 | 27.6× | 0.938 |
| p2_s03 | + 短 suffix 作为 forced token 走解码 graph | 46.7 | 35.8× | 0.938 |
| p2_s04 | + 工具异步执行；slot 状态 suspend/resume | 38.7 | 43.2× | 0.938 |
| p2_s05 | + fla 解码 kernel 原地写循环状态 | 35.7 | 46.9× | 0.938 |
| p2_s06t | + FP8 线性层（torchao，per-tensor） | 28.6 | 58.5× | 0.938 |
| p2_s07 | + 64 slot，每个 batch bucket 一个 graph，live 行压缩 | 24.8 | 67.4× | 0.938 |
| p2_s08 | + 循环状态 bf16 存储 | 23.6 | 71.0× | 0.938 |
| p2_s09 | + 调优 fla 解码 kernel 的 (BV, warps) | 21.5 | 77.7× | 0.930 |
| **p2_s10** | + 所有 suffix 都走 forced token（force_max 512） | **21.3**（3 次重复中位数 20.3） | **78.5×**（82×） | 0.938 |

**对照：vLLM 0.30（同一 workload）**

| 精度 | vLLM | 我们 |
|---|---|---|
| fp8 | 20.6 s | 20.3 s |
| bf16 | 28.9 s | 28.2 s |

我们与 vLLM 持平。在同一条 751-token 关键链上，我们每 token 快约 3%。详见 `results/ref_vllm_fp8/README.md`。

## 3. 每一步为什么选它（Amdahl 链）

**Phase 1：单请求解码，从 host-bound 到 GPU-bound，再到 memory-bound**

1. **baseline**
   - 证据：GPU busy 只有 21%，cuDNN SDPA 每个新形状都要建 plan（冷启动 438 ms/tok）。
   - 动作：p1_s01 关掉 cuDNN SDPA；p1_s02 换用 fla kernel（prefill 快 3.8×，解码没变）。
2. **解码受 host 端 Python dispatch 限制**（59 ms/tok，GPU 理论下限 16 ms）
   - CUDA graph：31.5 ms/tok，2.5×。
3. **graph 内部的瓶颈**（nsys `--cuda-graph-trace=node`）
   - 2648 个小 elementwise kernel：torch.compile 融合，25.6 ms；
   - `repeat_kv` + 4096 slot 的 masked 注意力：改成两次 GQA-folded GEMM，21.4 ms/tok。
4. **Phase 1 收尾**
   - 21.4 ms/tok，是 roofline 的 75%；GEMM 占一步的 88%（读 54 GB 权重）。
   - 单请求已接近带宽极限，剩下的杠杆只有降低精度，或者让一次权重读取服务多个请求。

**Phase 2：多 agent**

- **p2_s01 continuous batching**
  - 证据：一步时间几乎全是读权重，与 batch 大小无关。
  - 32 行一起解码：7.8×。
  - 新的分布：prefill 46%、解码 40%、工具 13%。
- **p2_s02 prefix reuse**
  - 预计 −20 s，实测 **0 s**：prefill token 减少了 89%，时间却没变。
  - 原因：eager prefill 每次调用约 107 ms，几乎与长度无关（host-bound）。
- **p2_s03 forced token**
  - 把复用之后剩下的短 suffix 塞进解码 graph，prefill 调用次数从 256 降到 7。
  - 模拟器预计 47.7 s，实测 46.7 s。
- **p2_s04 异步工具**
  - 工具时间占 17.7%，且阻塞所有行。
  - 等待工具的 slot 保存/恢复自己的线性状态：38.7 s。
- **p2_s05 原地状态写回**
  - profile 中 copy_ 占一步的 10.7%：35.7 s。
- **p2_s06 FP8**
  - GEMM 占一步的 78%。
  - per-row（CUTLASS rowwise）在 M=32 时慢，per-tensor（cuBLASLt）更快：28.6 s。
- **p2_s07 bucket graph + 64 slot**
  - 32 行的 graph 平均只有 16.9 行是活的，而且最长的 agent 可能很晚才开始。
  - 每个 bucket 一个 graph，并压缩 live 行：24.8 s。
  - 此时步数（990）≈ 最长 agent 链：从这里开始受**关键路径**限制。
- **p2_s08 / p2_s09 循环状态**
  - 只改成 bf16 存储，kernel 从带宽瓶颈变成 launch 瓶颈，只快 8%；
  - 再扫描 (BV, warps)，recurrent 从 7.7 ms 降到 4.1 ms（B=64）：21.5 s。
- **p2_s10 全部 forced**
  - 最后 4 次 eager prefill 被去掉：21.3 s。
  - 此时 run-to-run 的链长波动（±70 步 ≈ ±1.4 s）已经与单步优化的收益相当，**收益递减，结束**。

## 4. 学到的东西（流水线本身）

**1. Amdahl 的 p 要在"成本驱动因素"上测，而不是在"工作量"上测。**
- p2_s02 把 prefill token 减少了 89%，收益为 0：成本是**调用次数**（host 开销），不是 token 数。
- p1_s02 同理：fla 让 prefill 快了 3.8×，但解码不变，因为解码的成本是 dispatch，不是数学运算。

**2. 估算会怎么错：**

| stage | 估算 | 实测 | 原因 |
|---|---|---|---|
| p1_s01 | 2.2× | 1.17× | profile 太短，过度放大了冷启动效应。要在真实 workload 上量 p。 |
| p1_s05 | −13% | −1.3% | torch.profiler 把 `cudaGraphLaunch` 的 host 时间放大了。用不带 instrumentation 的 A/B 复核。 |
| p2_s02 | −20 s | 0 | 见第 1 条。 |
| p1_s04, p1_s06, p1_s07, p2_s03, p2_s04 | — | 误差 < 5% | 证据来自正确的 profile（graph 节点级 nsys、步进模拟器）时，估算是可靠的。 |

**3. 先建模拟器，再写代码。**
- `bench/sim_steps.py` 用实测 turns 加每个 bucket 的步时间回放调度；
- p2_s03、p2_s04、p2_s07、p2_s10 都先用它估算收益，再决定做不做。

**4. 正确性要有自动化检查，并且知道它在保证什么。**
- `check_equiv` / `check_batch` 比较 greedy 输出是否 bit 一致；
- 改用 per-tensor FP8 和多 bucket graph 后，batch 不变性消失（行之间耦合、GEMM 选的 kernel 随 M 变化）；
- 这不是 bug，但必须在 `--one-bucket` + per-row 的配置下验证（20/20 一致），才能把它和真正的 slot 串扰区分开。

**5. 瓶颈会转移，单一指标会骗人。**
- "每 token 时间"在 Phase 2 后期已经不是关键：关键路径（最长 agent 链）决定 makespan；
- 更多 slot 或 longest-first 调度都不能缩短它（模拟器验证过）。

**6. 不写 kernel 也能做 kernel 层面的事。**
- 改已有 kernel 的写回位置（p2_s05）和 launch 配置（p2_s09），同一个 kernel、同样的数学，快了 1.9×。

**7. 噪声底线。**
- 到最后，一次运行的方差（fp8 batch 耦合导致链长变化）就已经与一步优化的收益相当，这时需要多次重复取中位数。

## 5. 现在的瓶颈与（范围之外的）下一步

p2_s10 的情况：
- makespan ≈ 最长 agent（task 423）的约 1000 个 token-step；
- 尾部 B=8 时一步 15.1 ms：FP8 GEMM 10.3 ms（HBM 带宽的 77%），约 1900 个小 inductor kernel 2.5 ms，lm_head 与注意力 1.3 ms。

再往下走需要以下手段之一：

| 方向 | 说明 |
|---|---|
| **缩短关键路径** | speculative decoding（MTP 头或 draft 模型）。混合模型需要循环状态的回滚或分支，这是最大的杠杆。 |
| **更少的权重字节** | int4 权重（AWQ/GPTQ + Marlin），尾部的步时间预计约 −30–40%。 |
| **kernel 工作** | 小 M 的 FP8 GEMM、融合 Gated-DeltaNet 解码层、减少约 1900 个小 kernel。本项目约束之外。 |
| **工程化** | 持久化编译缓存，把加载时间从 580 s 降下来；通用调度器。 |

## 6. 复现

```bash
slurm/submit.sh bench stage00_baseline --engine naive --runner serial
slurm/submit.sh bench p2_s10_force_all --engine p2_s10 --runner continuous_async
slurm/submit.sh profile p2_s10_force_all --engine p2_s10           # nsys + per-bucket timings
slurm/submit.sh vllm ref_vllm_fp8 --quantization fp8               # vLLM venv: /mnt/data/u/feic/pjs/venvs/vllm
python bench/report.py                                              # results/summary.csv
```
