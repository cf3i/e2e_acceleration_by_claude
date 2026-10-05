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

## 5. Phase 2 结束时的瓶颈与（当时范围之外的）下一步

Phase 3 做了其中的 kernel 工作，见第 7 节。

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
slurm/submit.sh profile_batch p2_s10_force_all --engine p2_s10     # nsys + per-bucket timings
slurm/submit.sh vllm ref_vllm_fp8 --quantization fp8               # vLLM venv: /mnt/data/u/feic/pjs/venvs/vllm
python bench/report.py                                              # results/summary.csv
```


---

# 7. Phase 3：系统优化逼近物理极限

**约束变化**
- 允许写 kernel，但只做**通用**优化：要对生产环境里各种 prompt 都成立，不做针对 workload 的特化；Phase 2 的 forced-token prefill 被真正的 chunked prefill + paged KV 取代。
- 精度固定：FP8 per-tensor 线性层，bf16 lm_head / KV / 循环状态，greedy。
- 不做 MTP / speculative decoding：优化的是系统，不改优化对象。
- 去掉 task 423（512 token 封顶、一直答错的复读 outlier），127 题。
- 新增第二个 workload `gsm8k_longctx_128`：1k–7k token 的长 prompt，没有共享前缀，用来检验通用性。

## 7.1 结果

| stage（commit） | 主要内容 | 127-task (s) | longctx (s) | accuracy (127 / longctx) |
|---|---|---|---|---|
| p3_s00_rebaseline（p2_s10） | Phase 2 终点，去掉 423 | 18.19 | — | 0.945 / — |
| v3 引擎 dev（3252fc5） | 自写 forward + 异步调度 + chunked prefill + paged KV | 10.95（dev） | 51.5（dev） | |
| p3_s01_kernels（2defad5） | + attention / silu kernel、合并小 M FP8 GEMM、runner、PDL、GDN 融合 | 8.86 | 40.64 | 0.945 / 0.945 |
| p3_s02_gdnchunk | + GDN chunk 配置 | 8.77 | 39.61 | |
| p3_s04_gdnfuse（0abefc7） | + w/u 不落 HBM | 8.47 | 40.83 | |
| p3_s05_kkt（0b442cc） | kkt 改为 tf32 求逆 → **数值事故** | 8.99 | 77.8 | 0.945 / **0.417** |
| p3_s06_kktfix（097cd7f） | 对角块 fp32 前代修复 | 9.45 | 40.46 | 0.945 / 0.953 |
| p3_s07_fa3conv（321c65f） | + 双流、FA3 prefill attention、decode conv 融合 | 8.46 | 39.00 | 0.945 / 0.937 |
| p3_s08_final（49b693b） | + no workspace、gdn 调度、Aq 移出链、decode 预取 | 9.98（task 428 变长，见下） | 38.21 | 0.945 / 0.937 |
| **p3_s08_final_replay**（同代码，按 p3_s07 的轨迹 replay） | 只测系统速度 | **8.36** | **38.17** | （replay 不测 accuracy） |
| vLLM 0.30 fp8（同 workload） | | 17.54 | 91.96 | 0.945 / 0.953 |

- 127-task：18.19 → 8.36 s（2.2×）；longctx：51.5 → 38.2 s（1.35×）。从项目开始算，127-task 的 makespan 是 baseline（1673.6 s，含 task 423）的约 1/200。
- 对 vLLM：127-task 快 2.1×，longctx 快 2.4×，accuracy 相同或在噪声内。
- p3_s08 正常 run 的 127-task 是 9.98 s：每步的时间与 replay 完全一样（B=1 都是 11.0 ms），但 task 428 这次生成得更长，单它就跑了 9.86 s，最后有 151 个只剩它一行的 step。
  - 原因：per-tensor FP8 下 batch 组成会改变数值，从而改变轨迹；这次 47 个 task 的文本变了，accuracy 不变。
  - makespan = 最慢 task 的延迟，所以一个 task 的轨迹就能让结果偏差 ±1.5 s。系统速度要看 replay。
- 正式 run 之间的差异不能直接读成优化收益：
  - 不同节点之间有 2–3% 的差异（例如 p3_s03 8.98 s vs p3_s02 8.77 s）；
  - 舍入变化会改变某个 task 的生成长度：p3_s06 的 9.45 s 是 task 428 从 391 token 变成了 581 token，系统并没有变慢。
- 每一步的真实收益来自同 GPU 的 replay A/B（见 7.3），记录在 `OPTIMIZATION_LOG.md` 的 Phase 3 表里。

## 7.2 做了什么（按收益排序）

**A. 引擎架构（收益最大：18.2 → 约 9 s）**
- 自己写 forward，不再依赖 HF；token 预算调度，chunked prefill 与 decode 行混在一个 step 里；paged KV；每个 step 都是 CUDA graph（按行数分 bucket）。
- 异步调度：CPU 准备下一步时 GPU 还在跑上一步；runner 去掉每段计时的 device sync，tokenization 改成增量。
- 结果：127-task 的 GPU ledger 是 8.36 s busy、0.07 s idle，host 端已经做完。

**B. kernel（约 9 → 8.4 s；longctx 44.6 → 38.7 s）**
- 小 M 的 FP8 GEMM：同一输入的线性层合并（gate|up、qkv|z|b|a），用 Triton 实现，每层少几次 launch、读一次输入。
- 所有 kernel 都开 PDL（programmatic dependent launch），kernel 之间的空隙降到约 0.5 µs。
- GDN（48 层线性注意力）：
  - chunk 递推与输出融合，w/u 不落 HBM；
  - kkt 和三角求逆放到 tensor core 上，对角块保留 fp32 前代；
  - intra-chunk 的 q·kᵀ 移出串行链；
  - decode 时 conv 融进 GEMM epilogue，l2norm 融进 gdn_decode；
  - gdn_decode 寄存器预取。
- attention：prefill 用 FA3（自己从源码编译，paged、varlen），decode 用 split-KV。
- mixed step 双流：prefill 行的 GDN / attention 和 decode 行并发执行（graph 内并行分支）。
- cuBLASLt 不分配 workspace：选 static-schedule kernel，graph 里不再有 memset 节点打断 PDL 链（longctx −1.1%）。

**C. 没有收益或负收益的尝试（都有测量，见 log）**
- 自写大 GEMM 加融合 epilogue：精度对齐 cuBLAS 时，Triton 684 µs、CUTLASS 567 µs，cuBLAS 504 µs。fast-accum 更快，但误差是 3 倍，算换精度。
- nano-batching：M 切小后 GEMM 的效率损失比藏住的访存时间还多。
- chunk kernel 的 producer/consumer 拆分、L2 预取下一个 GEMM 的权重、提前 trigger、split-K、各种 config 扫描。

## 7.3 学到的东西

1. **数值正确性要用"病态"输入测，并且由长上下文 accuracy 把关。**
   - 0b442cc 用 (I+Y)⁻¹ = (I−Y)(I+Y²)(I+Y⁴)(I+Y⁸) 在 tf32 下求逆。随机 key 下误差和 fla 一样，127-task accuracy 也没变。
   - 但真实的 chunk 内 key 高度相关，这个乘积公式数值不稳定：longctx accuracy 从 0.945 掉到 0.417。
   - 127-task 上它只表现为"某个 task 变长了"，被我当成舍入噪声放过了。
   - 之后的规矩：改变数值的改动必须先跑 longctx accuracy；单元测试加了相关 key 的 case。能做到 bit-exact 的改动尽量做到 bit-exact。
2. **测量方法本身是一项工程。**
   - 节点间 2–3% 的差异、轨迹变化带来的 ±1 s，都比单项优化的收益大。
   - 为此做了 `--replay`：强制按录制好的 token 序列走，同 GPU 交替跑 A/B，重复性做到 ±0.005 s（127）/ ±0.05 s（longctx）。
   - accuracy 仍然用正常的 run 测。longctx 的 accuracy 本身有 ±0.02 的噪声：同一份代码跑出过 0.937 到 0.961。
3. **功耗墙。**
   - longctx 的 mixed step 跑在功耗上限附近（约 630–690 W，SM 频率约 1.6 GHz，不是 1.98 GHz），cuBLAS FP8 约 1.35 PFLOPS。
   - 后果：
     - 孤立 micro 的结论在模型里不一定成立，访存 kernel 必须在模型里调参；
     - 两个分支各自都填满 GPU 时，并发几乎不省时间；
     - 真正的下限是每个 step 的能量，而不只是每个 kernel 的时间。
4. **"能写 kernel"不等于"能比库快"。**
   - cuBLAS 的 FP8 GEMM 在精确累加下是我们用 Triton 和 CUTLASS 都追不上的。
   - 所有"融合进 GEMM epilogue"的方案，前提都是先有一个同样快的 GEMM。
5. **大收益来自架构，kernel 收益是递减的。**
   - 引擎重写贡献了约 2×；之后几十个 kernel 改动加起来约 7%（127-task）/ 13%（longctx）。
   - 最后几个小时的改动都在 1% 以内，信号是：到墙了。

## 7.4 离物理极限还有多远

**127-task**
- SOL 模型：一步 ≈ 26.9 GB 权重 / 3.0 TB/s + B ×（state + KV 字节）/ 3.0 TB/s，即约 9.0 + 0.063·B ms。按最长链 647 token 算，makespan SOL ≈ **7.1 s**。现在 8.36 s（replay），即 SOL 的约 85%。
- 剩下的差距主要在小 batch decode：B ≤ 16 的 step 占 GPU 时间的一半，每步约 11 ms，下限约 8.4 ms。
  - GEMM 约为读带宽下限的 93%；
  - 每层约 20 µs 是延迟型小 kernel（add_rmsnorm ×2、gated_norm、silu、gdn_decode），只能靠更少的 kernel 来消除。

**longctx**
- mixed step（约 2048 token）约 110 ms，其中：
  - cuBLAS GEMM 75 ms，已在功耗墙下的峰值；
  - silu / rmsnorm 约为带宽的 90%；
  - FA3 6.1 ms；
  - GDN 约 10 ms：串行链 + 双流争用，比理想值多约 2 ms。
- 只算 GEMM 算力的下限约 21 s，我们 38.2 s。差距主要是这些访存 / 延迟型部分；在功耗上限下它们无法与 GEMM 重叠（nano-batching 实测是负收益）。

## 7.5 再往下走需要什么（都是"好几天"量级）

| 方向 | 预期 | 难点 |
|---|---|---|
| 小 batch decode 的 persistent megakernel | 127-task 的小 batch step 最多快 15–20%，整体约 8–10% | 跨层调度、grid 级同步、与 graph/PDL 共存 |
| GDN 的 persistent kernel（chain 与 decode tile 统一调度） | mixed step 约 2% | Triton 表达力不够，需要 CUDA |
| 精确累加且与 cuBLAS 一样快的 FP8 GEMM + 融合 epilogue | mixed step 约 3% | 需要手写 Hopper mainloop（分段 promote 以省寄存器） |
| 约束之外：MTP / speculative、更低精度 | 最大的杠杆 | 改变了优化对象，本项目不做 |

## 7.6 复现

```bash
slurm/submit.sh bench p3_s08_final --engine p3_a1 --runner v3 --exclude-tasks 423
slurm/submit.sh bench p3_s08_final_longctx --engine p3_a1 --runner v3 --exclude-tasks 423 --data gsm8k_longctx_128
# 同 GPU 速度 A/B（replay 同一轨迹）：
scratch/mktree.sh <revA>; scratch/mktree.sh <revB>
EXTRA='--replay <abs results dir>' bash /mnt/data/u/feic/tmp_ab/ab.sh <data> <rounds> <revA> <revB>
```
