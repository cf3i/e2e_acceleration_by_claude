# PLAN: profile-driven optimization of an LLM agent on 1×H100

## Scenario (fixed for all stages)
- Offline batch math agent: 128 fixed GSM8K test questions (`data/gsm8k_128.jsonl`, seed 0), all available at t=0.
- ReAct-style agent with Qwen native tool-call format, one `python` tool (stdout returned); 27B uses the tool on ~97% of tasks unprompted.
- Model: Qwen3.5-27B (hybrid: 48 Gated-DeltaNet linear-attn layers + 16 full-attn GQA layers, 248k vocab), bf16 ≈ 54 GB.
  (Switched from Qwen3.5-2B: it rarely used tools and fell into greedy repetition loops.)
- Non-thinking mode, greedy decoding, ≤6 turns/task, ≤512 new tokens/turn.
- Hardware: 1×H100 80GB, SLURM partition freecycle-h100 (qos freecycle, preemptible → --requeue).

## Two benchmarks
| | Microbench (`bench/microbench.py`, ~5 min) | End-to-end (`bench/run_bench.py`) |
|---|---|---|
| measures | single-request prefill ms @ ctx 512/1024/2048; decode ms/token (cold + warm); % of HBM roofline | makespan of the 128-task agent workload; accuracy; tok/s; latency p50/p90 |
| role | fast inner loop, local speedup `s` | ground truth, overall speedup |

Guardrail: accuracy within 2 points of baseline; output-preserving changes should also keep greedy outputs (near-)identical.

## Iteration loop (both phases)
1. profile (`slurm/profile.sbatch`): L1 section timers, L2 torch.profiler, L3 nsys (+NVTX).
2. Amdahl: for each candidate estimate fraction `p` and local speedup `s` → `1/((1-p)+p/s)`; pick best gain/effort.
3. Implement ONE change → microbench (phase 1) / e2e bench → record in `results/<stage>/` + `OPTIMIZATION_LOG.md` → git tag.

Stage naming: `stage00_baseline` (shared start), `p1_sNN_<name>` (phase 1), `p2_sNN_<name>` (phase 2).

## Phase 1 — single-request inference engine (batch = 1)
Goal: make one generate() call as fast as possible. Measured mainly by the microbench; e2e run at phase end.
Physical ceiling: decode must stream all weights per token → 54 GB / 3.35 TB/s ≈ **16 ms/token**.
Candidate directions (order decided by profiles, not this list):
- fast-path kernels for linear attention (`flash-linear-attention`, `causal-conv1d` — existing libraries)
- the cold-shape slowdown seen in probes (suspected per-shape attention backend plan building)
- static / preallocated cache instead of HF DynamicCache growth
- `torch.compile` and CUDA Graphs for the decode step (remove launch + Python overhead)
- host-sync removal (per-token `.item()`), GPU-side stop checks
Exit: decode ≥ ~75–80% of roofline (~20–21 ms/token), or 3 stages <5% gain, or only kernel work remains.

## Phase 2 — multi-agent system (serving + agent layers)
Goal: minimize makespan of the 128-task workload. Measured by the e2e bench.
Expected ceiling of phase 1 alone is ~3× (58 → ~20 ms/token); larger wins need the GPU to serve many agents at once.
Candidate directions:
- batching across agents (static → continuous batching), within the ~25 GB left for KV + recurrent state
- cross-turn cache reuse (append-only history; hybrid model keeps KV *and* recurrent state) and shared-prefix reuse
- memory headroom (e.g. FP8 weights) to allow larger batches
- tool execution: persistent worker pool, overlap tool calls with GPU generation
- reference ceiling: vLLM on the same workload
Exit: 3 stages <5% gain, or close to the vLLM reference, or only kernel work remains.

## Scope
No custom kernels. Existing libraries and framework/system-level changes are allowed.
Final deliverable: `results/FINAL_REPORT.md` — speedup curve per stage, predicted vs. measured Amdahl gains, lessons.
