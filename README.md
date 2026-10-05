# 单卡 H100 上的 LLM Agent 端到端加速：Qwen3.5-27B，同一批题从 1471 s 到 8.4 s

这个仓库完整记录了一次 **profile 驱动的端到端优化**。场景是一个会调用 Python 工具的数学 agent，在单张 H100 上做完一批题目。

- 代码、实验日志、每一步的测量结果都在仓库里。
- 优化工作由 Claude（Claude Code）在 SLURM 集群上完成，人只负责定目标、约束和方向。
- 本项目的出发点是学习：完整走一遍 **profile → 用 Amdahl 定律选下一步 → 实现 → 验证正确性 → 再 profile**，并记录每一次估算对在哪里、错在哪里。

## 先看懂这几个词

| 词 | 意思 |
|---|---|
| **makespan** | 从开始到**这一批题全部做完**的墙钟时间。就像一个班 127 个人同时考试，从发卷到最后一个人交卷的时间。所有对比都只看它，越小越好。 |
| **task** | 一道题。agent 做一道题要和模型来回好几轮：模型写代码 → 执行 Python → 模型看结果 → 给答案。 |
| **127-task**（主测试） | 127 道 GSM8K 小学数学应用题，每道题的 prompt 很短（几十个词）。原本是 128 道；其中 task 423 每次都会让模型复读到长度上限，而且答错，单这一道就决定了整批的时间，所以 Phase 3 起把它去掉了。 |
| **longctx**（通用性测试） | **同样的 128 道题**，但每道题前面塞了 1k–7k token 的长参考材料（若干道已解答的例题），且每道题的材料都不同。用来检查"长 prompt、没有可复用前缀"时优化是否依然成立。 |
| **accuracy** | 答对的比例，用来保证"变快"没有以"变笨"为代价。 |
| **baseline** | 起点：直接用 HuggingFace transformers 跑模型，一道题做完再做下一道。 |
| **vLLM** | 业界常用的开源推理引擎，作为外部参照：用它跑同一批题，看我们比它快还是慢。 |

**对比是不是在同一个东西上？** 是的。同一张表里，每个数字都是**同一批题、同一个模型、同一张 GPU、同样的解码规则**（greedy、每轮最多 512 token、最多 6 轮），只是推理系统不同。127-task 和 longctx 是两个不同的测试，它们的数字之间不能比较，只能各自纵向比较。

## 一句话结果

**主测试（127-task：127 道短题）**

| 系统 | makespan | 相对 baseline | accuracy |
|---|---|---|---|
| baseline（HF transformers，逐题串行） | 1470.6 s ¹ | 1× | 0.937 |
| Phase 1 结束（单请求引擎优化） | 457.4 s ¹ | 3.2× | 0.945 |
| Phase 2 结束（多 agent 并发） | 18.19 s | 81× | 0.945 |
| **Phase 3 结束（自写引擎 + kernel）** | **8.36 s** | **176×** | 0.945 |
| 参照：vLLM 0.30（同精度 fp8） | 17.54 s | — | 0.945 |

¹ Phase 1 及之前是在含 task 423 的 128 题上跑的：baseline 1673.6 s，Phase 1 结束 474.8 s。这两个阶段逐题串行，makespan 等于各题耗时之和，所以减去 task 423 自己的耗时（203.0 s / 17.4 s），就得到 127 题的数字。

**通用性测试（longctx：同样的题 + 1k–7k token 长 prompt）**

| 系统 | makespan | accuracy |
|---|---|---|
| Phase 3 开始（v3 引擎初版） | 51.5 s | — |
| **Phase 3 结束** | **38.2 s** | 0.937 |
| 参照：vLLM 0.30 fp8 | 91.96 s | 0.953（同一份代码 run 之间 ±0.02） |

结论：同一批题，最终比 vLLM 快 **2.1×**（短题）和 **2.4×**（长 prompt），accuracy 相同或差异在噪声范围内。

---

## 目录

1. [场景与度量](#1-场景与度量)
2. [优化轨迹](#2-优化轨迹)
3. [方法论](#3-方法论)
4. [主要经验](#4-主要经验)
5. [离物理极限还有多远](#5-离物理极限还有多远)
6. [仓库结构](#6-仓库结构)
7. [复现](#7-复现)
8. [延伸阅读](#8-延伸阅读)

---

## 1. 场景与度量

**Agent**
- 求解 GSM8K 数学题：模型先调用 `python` 工具写程序算答案，看到输出后给出 `\boxed{N}`。
- 使用 Qwen 原生 tool-call 格式，非 thinking 模式，greedy 解码。
- 每轮最多生成 512 token，最多 6 轮。
- 128 道固定题目（`data/gsm8k_128.jsonl`），开始时全部可用，相当于离线批处理。

**模型：Qwen3.5-27B**
- 混合架构：48 层 Gated DeltaNet（线性注意力，带循环状态）+ 16 层 full attention（GQA）。
- 词表 248k；bf16 权重约 54 GB。

**硬件**
- 1 × H100 80GB（SLURM 可抢占分区）。
- 实测会撞功耗墙：大 GEMM 时约 630–690 W，SM 频率约 1.6 GHz。

**指标**
- 主指标：整批题目的 **makespan**（墙钟时间）。
- 护栏：accuracy 不能下降（基线 0.93–0.95）。
- 辅助观察：各部分时间占比、task 延迟分布、GPU busy/idle。

**两个 workload（Phase 3 起）**

| 名称 | 内容 | 特点 |
|---|---|---|
| `gsm8k_128`（127-task） | 短题目（平均约 240 字符）；所有题共享 system prompt 和工具定义 | 生成（decode）为主，最长那道题的推理链决定 makespan |
| `gsm8k_longctx_128` | 同样的题，每题前加 1k–7k token 的已解答例题，各题不同 | prefill 为主，检验优化是否通用 |

Phase 3 去掉了 task 423，所以是 127 题。这道题会一直复读到 512 token 封顶，而且总是答错，一个 outlier 就决定了整个 makespan。

---

## 2. 优化轨迹

三个阶段的约束逐步放宽：

| 阶段 | 允许做什么 | 目标 |
|---|---|---|
| Phase 1 | 不写 kernel；可以装库、改 launch 方式 | 单请求（batch=1）推理尽量快 |
| Phase 2 | 同上 | 多 agent 并发，最小化 makespan |
| Phase 3 | **可以写 kernel**，但只做通用优化（不针对 workload 特化）；精度固定（FP8 per-tensor 线性层）；**不做 MTP / speculative decoding** | 逼近物理极限 |

### Phase 1：单请求，从 host-bound 到 memory-bound（1673.6 → 474.8 s）

| stage | 证据（profile 看到了什么） | 改动 | makespan |
|---|---|---|---|
| stage00_baseline | — | HF eager + DynamicCache | 1673.6 s |
| p1_s01 / s02 | GPU busy 只有 21%；cuDNN SDPA 每遇到新形状就建 plan（冷启动 438 ms/tok） | 关闭 cuDNN SDPA；GDN 层改用 fla / causal-conv1d kernel | — |
| p1_s03 / s04 | decode 59 ms/tok，而 GPU 理论下限是 16 ms：受 Python dispatch 限制 | StaticCache + CUDA graph 解码 | 660.8 s（2.5×） |
| p1_s05–s07 | graph 内有 2648 个小 elementwise kernel；`repeat_kv` + masked attention 浪费 | async readback、torch.compile 融合、GQA-folded 解码注意力 | **474.8 s**（3.5×） |

Phase 1 结束时是 21.4 ms/tok，达到 roofline 的 75%，GEMM（读 54 GB 权重）占 88%。单请求已经没什么可榨的了。

### Phase 2：多 agent 并发（474.8 → 20.3 s）

| stage | 证据 | 改动 | makespan |
|---|---|---|---|
| p2_s01 | 一步时间几乎全花在读权重上，与 batch 大小无关 | slot cache + batch-32 decode graph + continuous batching | 60.8 s（7.8×） |
| p2_s02 | prefill 占 46% | 跨轮 / 共享前缀复用 | 60.7 s（**预测 −20 s，实测 0**，见经验 1） |
| p2_s03 | eager prefill 每次调用约 107 ms，几乎与长度无关 | 短 suffix 当作 forced token 走 decode graph | 46.7 s |
| p2_s04 | 工具执行占 17.7%，且阻塞所有行 | 工具异步执行，slot 状态 suspend/resume | 38.7 s |
| p2_s05 | 每步里 `copy_` 占 10.7% | fla 解码 kernel 原地写循环状态 | 35.7 s |
| p2_s06t | GEMM 占 78% | FP8 线性层（torchao per-tensor，cuBLASLt） | 28.6 s |
| p2_s07 | 32 行的 graph 平均只有 16.9 行在跑 | 64 slot，每个 batch bucket 一个 graph，压缩 live 行 | 24.8 s |
| p2_s08 / s09 | 循环状态读写受带宽限制，之后又变成受 launch 限制 | 循环状态改 bf16；扫描 kernel 的 (BV, warps) | 21.5 s |
| p2_s10 | 还剩最后 4 次 eager prefill | 所有 suffix 都走 forced token | **21.3 s**（3 次中位数 20.3 s） |

Phase 2 结束时与 vLLM 持平（fp8：20.3 s vs 20.6 s）。此时 makespan 由最长 agent 链决定，run 之间的链长波动（±1.4 s）已经与单步优化的收益相当。

### Phase 3：自写引擎 + kernel，逼近物理极限（18.19 → 8.36 s）

| step | 改动 | 127-task | longctx |
|---|---|---|---|
| p3_s00 | 重新定基线（p2_s10 去掉 task 423） | 18.19 s | — |
| v3 引擎 | 自写 forward（不再依赖 HF）+ 异步 token-budget 调度器 + **chunked prefill 与 decode 混在一个 step** + paged KV + 每个 step 都是 CUDA graph | 10.95 s | 51.5 s |
| kernels | 128 行 prefill attention tile、单 pass silu_mul_quant、split-KV decode attention | 9.95 s | 47.4 s |
| 合并 GEMM | 小 M 的 FP8 GEMM 按输入合并（gate\|up、qkv\|z\|b\|a），用 Triton 实现 | 9.51 s | 45.0 s |
| runner | 计时不再做 device sync；tokenization 改为增量 | 8.96 s | 44.6 s |
| PDL | 所有 kernel 开启 programmatic dependent launch，kernel 间空隙约 0.5 µs | 8.95 s | 42.2 s |
| GDN 融合 | chunk 递推与输出融合，w/u 不落 HBM，kkt 和三角求逆放上 tensor core | 8.6–8.8 s | 39.6 s |
| 双流 | mixed step 里 prefill 行的 GDN/attention 与 decode 行在 graph 内并发 | 8.43 s | 39.3 s |
| FA3 + conv 融合 | prefill attention 改用 FA3（从源码编译，paged varlen）；decode 的 conv 融进 GEMM epilogue | 8.38 s | 38.9 s |
| 收尾 | cuBLASLt 不分配 workspace、GDN 调度顺序、intra-chunk 注意力移出串行链、gdn_decode 寄存器预取 | **8.36 s** | **38.17 s** |
| 对照 vLLM 0.30 fp8 | | 17.54 s | 91.96 s |

- 前面几行是 dev 测量；后面几行是同 GPU 的 replay A/B（见 3.3）。
- 正式的 `p3_s08_final` 跑出来是 9.98 s。每步时间与 replay 完全一致，但 FP8 的 batch 耦合让 task 428 这次生成得更长（9.86 s）。详见 `results/FINAL_REPORT.md` §7.1。

**Phase 3 中失败或负收益的尝试**（每一项都有测量，见 `OPTIMIZATION_LOG.md`）：
- **kkt 改用 tf32 乘积公式求逆**：随机输入上和参考实现一样准，但真实 key 高度相关，longctx accuracy 从 0.945 跌到 **0.417**。后来对角块改用 fp32 前代修复。
- **自写大 GEMM 加融合 epilogue**：精确累加下 Triton 684 µs、CUTLASS 567 µs，cuBLAS 是 504 µs，追不上。
- **Nano-batching**（把 M 切小来重叠访存与计算）：GEMM 效率的损失比藏住的访存时间还多。
- **其他**：L2 预取下一个 GEMM 的权重、chunk kernel 拆成 producer/consumer、split-K、提前 trigger。

---

## 3. 方法论

### 3.1 循环

1. **Profile**：三层工具。
   - L1：分段计时；
   - L2：`torch.profiler`；
   - L3：nsys `--cuda-graph-trace=node`，加上自写的关键路径归因（`scratch/tl_mixed.py`）。
2. **Amdahl 选择**：对每个候选，估算它占的比例 `p` 和局部加速 `s`，预期整体加速为 `1/((1-p)+p/s)`；按"收益/工作量"排序。
3. **只做一个改动**，然后做正确性检查：
   - `bench/check_equiv.py` / `check_batch.py`：greedy 输出 bit 一致；
   - `bench/test_v3_kernels.py`：kernel 对照参考实现；
   - longctx accuracy。
4. **测量**：e2e bench，同 GPU replay A/B。
5. 把预估、实测和偏差原因记进 `OPTIMIZATION_LOG.md`。

### 3.2 先建模拟器再写代码

`bench/sim_steps.py` 用实测的 turn 序列，加上每个 bucket 的 step 时间，回放调度过程。p2_s03、s04、s07、s10 都是先模拟、再决定做不做；预估与实测的误差都在 5% 以内。

### 3.3 Replay A/B：把"系统速度"和"轨迹噪声"分开

到了 Phase 3，单项优化的收益只有 0.5–2%，但噪声比这个还大：
- 不同节点之间有 2–3% 的速度差；
- per-tensor FP8 下，batch 组成会影响数值，进而改变某个 task 的生成长度。一个 task 就能让 makespan 偏 ±1.5 s。

解决办法是 `run_bench.py --replay <results dir>`：强制按录制好的 token 序列执行，在同一张 GPU 上交替跑 A/B。重复性做到 127-task ±0.005 s、longctx ±0.05 s。accuracy 仍然用正常 run 测。

---

## 4. 主要经验

1. **Amdahl 的 p 要在"成本驱动因素"上测，而不是在"工作量"上测。** prefix reuse 让 prefill token 减少了 89%，收益却是 0：成本来自 eager 调用的**次数**（host 开销），而不是 token 数。
2. **大收益来自架构，kernel 收益递减。**
   - Phase 2 的 continuous batching：7.8×；Phase 3 的引擎重写：约 2×。
   - 之后几十个 kernel 改动加起来只有约 7%（127-task）/ 13%（longctx）。
   - 最后几个改动都在 1% 以内，这就是到墙的信号。
3. **数值改动必须用病态输入加长上下文 accuracy 把关。**
   - tf32 求逆在随机输入的单元测试和 127-task 上都"看起来没问题"，却让 longctx accuracy 跌到 0.417。
   - 能做到 bit-exact 的改动尽量做到 bit-exact。
4. **测量方法本身就是一项工程。** 当噪声大于信号时，先把测量做对（replay A/B），再谈优化。
5. **功耗墙会改变结论。**
   - H100 在 FP8 大 GEMM 下撞功耗上限，频率降到约 1.6 GHz。
   - 孤立的 micro-benchmark 结论在模型里不一定成立；两个都能填满 GPU 的分支并发几乎不省时间；真正的下限是每个 step 的**能量**。
6. **"能写 kernel"不等于"能比库快"。** 精确累加的 FP8 GEMM 上，Triton 和 CUTLASS 都追不上 cuBLAS。所有"融合进 GEMM epilogue"的想法，前提都是先有一个同样快的 GEMM。
7. **瓶颈会转移，单一指标会骗人。** Phase 2 后期，"每 token 时间"已经不重要，最长 agent 链（关键路径）才决定 makespan。

---

## 5. 离物理极限还有多远

**127-task：约达到 SOL 的 85%**
- SOL 模型：一步 ≈ 26.9 GB 权重 / 3.0 TB/s + B ×（循环状态 + KV 字节）/ 3.0 TB/s ≈ 9.0 + 0.063·B ms。
- 按最长链 647 token 算，SOL ≈ **7.1 s**；我们是 8.36 s。
- 剩余差距在小 batch decode（B ≤ 16）：每步约 11 ms，下限约 8.4 ms。
  - GEMM 已经达到读带宽的约 93%；
  - 每层约有 20 µs 花在延迟型小 kernel 上（rmsnorm、gated_norm、silu、gdn_decode）。

**longctx**
- 约 2048 token 的 mixed step 约 110 ms，其中 cuBLAS GEMM 75 ms，已经是功耗墙下的峰值。
- 只算 GEMM 算力的下限约 21 s，我们是 38.2 s。差距在访存/延迟型部分，在功耗墙下它们无法和 GEMM 重叠。

**再往下走需要什么**（都是"好几天"量级的工作）

| 方向 | 预期 |
|---|---|
| 小 batch decode 的 persistent megakernel | 整体约 −8–10% |
| GDN persistent kernel（chain 与 decode tile 统一调度） | mixed step 约 −2% |
| 精确累加、与 cuBLAS 一样快的 FP8 GEMM + 融合 epilogue | mixed step 约 −3% |
| 约束之外：MTP / speculative decoding、更低精度 | 最大的杠杆，但改变了优化对象 |

---

## 6. 仓库结构

```
.
├── README.md                 本文件
├── PLAN.md                   最初的计划：场景、指标、迭代循环、各阶段退出条件（英文）
├── OPTIMIZATION_LOG.md       逐步优化日志：每一步的证据、Amdahl 预估、实测、失败尝试（英文，最详细）
├── agentopt/                 主体代码
│   ├── agent.py, tools.py    agent 循环、python 工具（子进程执行）
│   ├── runner.py             串行 / continuous / 异步工具等 runner
│   ├── engines/              Phase 1–2 的引擎（基于 HF transformers 打补丁）
│   │   ├── naive.py, static.py, batch.py      baseline、StaticCache+graph、批处理引擎
│   │   ├── slot_cache.py, inplace_state.py    slot 状态管理、原地状态写回
│   │   ├── decode_attention.py, quant.py      GQA-folded 解码注意力、FP8 量化
│   │   └── hybrid_prefill.py, hf_patches.py  GDN prefill 从 slot 已有状态续接；显式选择 fla 快路径
│   └── v3/                   Phase 3 自写引擎
│       ├── engine.py         token-budget 调度、chunked prefill、paged KV、bucket graph、双流
│       ├── model.py          Qwen3.5 forward（不依赖 HF）
│       └── kernels.py        Triton kernel：FP8 GEMM、GDN chunk/decode、attention、融合 norm/act 等
├── bench/                    基准与分析
│   ├── run_bench.py          e2e 基准（makespan / accuracy；--replay 做轨迹回放）
│   ├── run_vllm.py           vLLM 对照
│   ├── microbench.py, micro_*.py          微基准
│   ├── sim_steps.py          调度模拟器
│   ├── check_equiv.py, check_batch.py     正确性检查（greedy 输出一致性）
│   ├── test_v3_kernels.py, test_v3_e2e.py kernel / 引擎测试
│   ├── analyze_nsys.py, analyze_steps.py, prof_v3.py, profile_*.py   profile 分析
│   └── report.py             汇总 results/ 生成 summary.csv
├── slurm/                    SLURM 作业脚本；submit.sh 对已 commit 的代码做快照再提交
├── scratch/                  Phase 3 的实验脚本（kernel 原型、floor 测量、A/B 工具、CUTLASS GEMM 尝试）
├── data/                     两个 workload 的题目
└── results/
    ├── FINAL_REPORT.md       最终报告（中文，含三个阶段的结果、Amdahl 链、经验、SOL 分析）
    ├── summary.csv           所有 stage 的汇总
    └── <stage>/              每个 stage 的 metrics.json、README.md、transcripts.jsonl、日志
```

---

## 7. 复现

### 环境

| 组件 | 版本 |
|---|---|
| Python | 3.13 |
| PyTorch | 2.11.0 + CUDA 12.8 |
| Triton | 3.6.0 |
| transformers | 5.5.4（Phase 1–2 引擎和 tokenizer） |
| flash-linear-attention (fla) | 0.5.2 |
| causal-conv1d | 1.7.0 |
| torchao | 0.18.0（Phase 2 FP8） |
| FlashAttention 3 | 从源码编译（`flash-attention/hopper`） |
| vLLM（对照） | 0.30，单独的 venv |

还需要：
- 模型 `Qwen/Qwen3.5-27B`（HuggingFace）；
- 一张 Hopper GPU（kernel 用到了 TMA、PDL、FP8）。

> 注意：`slurm/*.sbatch` 中的分区、qos，以及 `scratch/` 里的 `/mnt/data/...` 绝对路径都是原集群上的，换环境需要修改。

### 命令

```bash
# baseline 与 Phase 2 终点
slurm/submit.sh bench stage00_baseline --engine naive --runner serial
slurm/submit.sh bench p2_s10_force_all --engine p2_s10 --runner continuous_async

# Phase 3 终点（两个 workload）
slurm/submit.sh bench p3_s08_final --engine p3_a1 --runner v3 --exclude-tasks 423
slurm/submit.sh bench p3_s08_final_longctx --engine p3_a1 --runner v3 --exclude-tasks 423 --data gsm8k_longctx_128

# vLLM 对照
slurm/submit.sh vllm ref_vllm_fp8 --quantization fp8

# 同 GPU 的 replay A/B：比较两个 revision 的系统速度
scratch/mktree.sh <revA>; scratch/mktree.sh <revB>
EXTRA='--replay <abs results dir>' bash scratch/ab.sh <data> <rounds> <revA> <revB>

# 汇总
python bench/report.py        # -> results/summary.csv
```

不用 SLURM 时可以直接跑 `python bench/run_bench.py --stage <name> --engine p3_a1 --runner v3 --exclude-tasks 423`。

---

## 8. 延伸阅读

| 想看什么 | 去哪里 |
|---|---|
| 完整故事、每一步为什么选它、经验 | [`results/FINAL_REPORT.md`](results/FINAL_REPORT.md) |
| 每一步的证据、预估与实测、所有失败尝试 | [`OPTIMIZATION_LOG.md`](OPTIMIZATION_LOG.md) |
| 某个 stage 的原始数据 | `results/<stage>/metrics.json`、`results/<stage>/README.md` |
| 与 vLLM 的逐 token 对比 | [`results/ref_vllm_fp8/README.md`](results/ref_vllm_fp8/README.md) |
| Phase 3 的 kernel 实现 | [`agentopt/v3/kernels.py`](agentopt/v3/kernels.py) |
