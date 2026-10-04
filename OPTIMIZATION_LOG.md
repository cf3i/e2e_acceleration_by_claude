# Optimization log

Each entry: profile evidence → Amdahl estimate → change → measured result → decision.

## Summary
| stage | phase | change | decode ms/tok (warm, ctx 1024) | makespan (s) | speedup vs base | accuracy | notes |
|---|---|---|---|---|---|---|---|
| stage00_baseline | - | HF eager, cuDNN SDPA, torch fallback for Gated-DeltaNet, DynamicCache | 63.0 (cold 438) | 1673.6 | 1.00 | 0.930 | GPU busy 21%; decode 87% / prefill 12% / tool 0.6% |
| p1_s01_no_cudnn_sdpa | p1 | `enable_cudnn_sdp(False)` | 59.9 (cold 59.4) | – | – | – | cold-shape cliff gone; prefill cold 1.3–2.2 s → 0.4–0.9 s |
| p1_s02_fla_fastpath | p1 | + fla / causal-conv1d kernels | 59.0 | – | – | – | prefill warm 412 → 110 ms @1024 (3.8×); decode unchanged; first-call Triton JIT 43 s |
| p1_s03_static_cache | p1 | + StaticCache (4096 slots) + static decode buffers/mask | 65.4 | – | – | – | 10% slower alone (masked attention over 4096 slots); prerequisite for graphs |
| p1_s04_cuda_graph | p1 | + CUDA-graph decode step | **31.5** | **660.8** | **2.53** | 0.9375 | 51% of roofline; GPU busy 0.83; 7/8 greedy outputs identical to p1_s02 (1 near-tie) |
| p1_s05_async_readback | p1 | + pipelined token readback | 31.1 | – | – | – | −1.3%; launch cost was profiler-inflated; 8/8 identical to p1_s04 |
| p1_s06_compile | p1 | + torch.compile decode (inside our graph) | 25.6 | – | – | – | 1.22×; elementwise 6.48 → 1.14 ms/step; 7/8 identical (near-tie) |
| p1_s07_gqa_decode_attn | p1 | + GQA-folded GEMM decode attention | **21.4** | **474.8** | **3.52** | 0.9375 | 75% of roofline; GEMM 92% of step; decode 92% of e2e |
| p2_s01_continuous_batching | p2 | slot cache + batch-32 decode graph + continuous runner | 30.4 ms/step (1.2 ms/tok amortized) | **60.8** | **27.5** | 0.9375 | ~25.5 live rows; prefill 46% / decode 40% / tool 13% |
| p2_s02_prefix_reuse | p2 | + cross-turn extend + shared-prefix snapshot | same | 60.7 | 27.6 | 0.9375 | prefill tokens −89% but prefill time unchanged (~107 ms/call fixed) |
| p2_s03_forced_prefill | p2 | + short suffixes fed as forced tokens through the decode graph | 30.1 ms/step | **46.7** | 35.8 | 0.9375 | prefill calls 256 → 7; 1222 steps; tool 17.7% |
| p2_s04_async_tools | p2 | + tool calls in a thread pool; suspend/resume slot states | 30.1 ms/step | **38.7** | 43.2 | 0.9375 | tool fully overlapped; decode 95% |
| p2_s05_inplace_state | p2 | + fla decode kernel writes recurrent state in place | 26.7 ms/step | **35.7** | 46.9 | 0.9375 | −3.2 ms/step state copy; GEMM 78% of step |
| p2_s06_fp8 | p2 | + FP8 linears (torchao, per-row scales) | 23.05 ms/step | 32.2 | 52.0 | 0.9375 | CUTLASS rowwise GEMM slow at M=32 |
| p2_s06t_fp8_tensor | p2 | FP8 with per-tensor scales (cuBLASLt) | **20.86 ms/step** | **28.6** | **58.5** | 0.9375 | GEMM 20.7 → 12.9 ms; kept |
| p2_s07_bucket_graphs | p2 | + 64 slots, one graph per batch bucket, live rows compacted | 26.9 ms/step @64, 16.2 @8 | **24.8** | 67.4 | 0.9375 | 990 steps ≈ longest chain → critical-path bound |
| p2_s08_bf16_state | p2 | + recurrent states stored in bf16 | 26.1 ms/step @64 | 23.6 | 71.0 | 0.9375 | recurrent kernel only −8% (launch-bound) |
| p2_s09_recurrent_launch | p2 | + tuned (BV, warps) launch of fla decode kernel | 23.1 ms/step @64, 15.7 @8 | **21.5** | 77.7 | 0.9297 | recurrent 7.7 → 4.1 ms @64; 1 borderline task flips |
| p2_s10_force_all | p2 | + all suffixes forced (force_max 512) | same | **21.3** | **78.5** | 0.9375 | eager prefill 1.9 → 0.6 s; chain +68 steps (noise) |

## stage00_baseline
- Profile: GPU busy 0.21; `_cudnn_attention_forward` 31% of CPU time (2.4 ms CPU / 13 µs GPU per call);
  ~5.9k kernels per generated token. Micro decode cold 438 vs warm 63 ms/tok.
- e2e: 1673.6 s, acc 0.930, decode 71.8 ms/tok (much closer to warm than the 4-task nsys run's 164 ms/tok).

## p1_s01_no_cudnn_sdpa — remove per-shape cuDNN SDPA plan building
- Evidence: cuDNN SDPA host time per new (q_len, kv_len); decode KV length is new at every step.
- Estimate: from the 4-task profile p≈0.57 of decode → ~2.2×. From the full e2e run (decode 71.8 vs warm ~60):
  p≈0.14 of e2e → ~1.17×. **The short profile over-weighted cold effects** — size Amdahl's p on the real workload.
- Result (micro): decode cold 438 → 59.4 ms/tok, warm 63.0 → 59.9. Kept.

## p1_s02_fla_fastpath — fla + causal-conv1d kernels for the 48 Gated-DeltaNet layers
- Evidence: baseline GPU time is thousands of tiny elementwise kernels from the torch fallback.
- Result (micro): prefill warm 3.8× faster (chunked delta rule in one Triton kernel instead of a Python loop
  over chunks); **decode unchanged (59.0)**. In decode the fallback was only a few ops per layer; the cost is the
  per-op host dispatch of the whole 64-layer forward, not the fallback math → decode is host/launch-bound.
- Caveat: first-call Triton JIT/autotune (43 s at the first shape, 11.8 s at a new bucket). One-time per process,
  but must be warmed before measuring (the e2e warmup task covers it).
- Kept (prefill is 12% of e2e: ~0.12 × (1 − 1/3.8) ≈ 9% e2e expected).

## p1_s03 / p1_s04 — StaticCache + CUDA-graph decode
- Evidence: decode ≈ 59 ms/tok vs 16 ms floor while GPU busy ≈ 21–27% → host-bound; a CUDA graph replays the
  whole step's kernels with one launch. StaticCache is the prerequisite (stable tensor addresses).
- Estimate: decode is ~87% of e2e; if decode → ~20 ms/tok (s≈3), overall ≈ 1/(0.13 + 0.87/3) ≈ 2.4×.
- Correctness: HF skips the attention mask for q_len = 1, which with a StaticCache would attend to unfilled
  slots → decode passes an explicit static 4D mask extended in-graph. `check_equiv`: p1_s03 and p1_s04 are
  identical to each other and 7/8 identical to p1_s02 (one divergence at token 42 inside a code comment).
- Result (micro, ctx 1024): p1_s03 65.4 ms/tok (slower: mem-efficient attention over 4096 masked slots, nothing
  else saved); **p1_s04 31.5 ms/tok** (1.87× vs p1_s02, 51% of roofline), flat across ctx 512–2048.
- e2e: **660.8 s (2.53× vs baseline)**, accuracy 0.9375; predicted ~700 s. Decode now 94% of e2e.
- Node-level nsys: GPU time per replay 30.3 ms ≈ the step → GPU-bound. GEMM 18.95 ms (85% of HBM BW),
  elementwise/copy 6.48 ms in 2648 tiny kernels, attention 3.92 ms (245 µs/layer over 4096 masked slots).
- Profile: GPU busy 0.29 → 0.83. `cudaGraphLaunch` costs ~4 ms CPU per replay (~3.6k nodes) and is serialized
  with the per-token blocking read → candidate p1_s05 (pipelined readback). The remaining GPU time (31.5 vs 16 ms
  floor) needs node-level graph tracing (`nsys --cuda-graph-trace=node`, re-profile submitted).

## p1_s05_async_readback — overlap graph launch with the previous step
- Evidence: torch.profiler showed `cudaGraphLaunch` ≈ 4 ms CPU per step, serialized with the blocking read.
- Estimate: up to 4/31.5 ≈ 13% of decode.
- Result: 31.5 → 31.1 ms/tok (−1.3%), 8/8 identical. **The 4 ms was profiler overhead**; host-side numbers from an
  instrumented run must be cross-checked with an uninstrumented A/B (the micro).

## p1_s06_compile — torch.compile (inductor) the decode forward, still inside our CUDA graph
- Evidence: node-level graph trace — 2648 elementwise/copy kernels per step, 6.48 ms (21% of the step).
- Estimate: fusion to a few hundred kernels → ~1.5 ms → step 30.3 → ~25 ms (≈1.2×).
- Result: **25.6 ms/tok (1.22×)**, matches the estimate; profile: elementwise 1.14 ms in 644 fused kernels,
  1485 kernels/step. 7/8 identical (near-tie flip from different bf16 rounding).
- Pitfall: inductor sets `TRITON_CACHE_DIR` to node-local /tmp when unset → fla kernels JIT-ed later missed the
  cache (cold prefill at a new length bucket 13 s). Pinned cache dirs in the engine.

## p1_s07_gqa_decode_attn — single-token attention as two GQA-folded GEMMs
- Evidence (p1_s06 profile): attention 4.45 ms/step (18%): HF `repeat_kv` 6× K/V copy (needed because a mask is
  passed) + `fmha_cutlassF` with one block per head over all 4096 slots, 237 us/layer.
- Estimate: → <1 ms, step ~21.5 ms.
- Result: **21.4 ms/tok** (−16%), 75% of the 16.1 ms floor, 8/8 identical. GEMM (~18.8 ms, 85% of HBM BW) is
  now ~88% of the step → phase-1 exit criterion met; the remaining levers are lower-precision weights (FP8,
  phase 2) or kernel work (out of scope).

## Phase 1 summary
1673.6 s → 474.8 s (3.52×), accuracy 0.930 → 0.9375; decode 63 (cold 438) → 21.4 ms/tok (25% → 75% of roofline).
What mattered, in order: CUDA graph (host-bound → GPU-bound, 1.87×), torch.compile fusion (1.22×), decode
attention rewrite (1.19×), cuDNN-SDPA off (cold cliff), fla kernels (prefill 3.8×). Two estimates were badly off
(p1_s01 from a too-short profile, p1_s05 from profiler-inflated host time) — both corrected by uninstrumented A/B.

## p2_s01_continuous_batching — 32 agents in flight, one decode graph
- Evidence: phase-1 end: decode 92% of e2e, and 88% of a step is the weight read → batch independent agents.
- Change: duck-typed slot cache (per-row KV write position + mask row), eager batch-1 prefill into a slot,
  CUDA-graph replay of the compiled step at batch 32, continuous admission.
- Result: **60.8 s (7.8× vs p1_s07, 27.5× vs baseline)**, acc 0.9375; 797 steps × 30.4 ms; check_batch 8/8.
  Task latency p50 3.2 → 11.5 s (throughput/latency trade).

## p2_s02_prefix_reuse — reuse previous turn + shared system prefix
- Evidence: 55% of prompt tokens re-prefill the previous turn; 352-token common prefix.
- Hybrid model: recurrent state only at the saved position → reuse at slot end (extend) or a snapshot.
- Estimate: ~−20 s (prefill tokens −89%). Result: **0 s** — prefill time 27.9 → 27.9 s, per-call median ~107 ms
  both ways. **Eager prefill is per-call (host) bound, not token bound**; the micro had shown it (105 ms @512 vs
  124 ms @1024). Measure p on the cost driver.
- Next: short suffixes as forced tokens through the decode graph (sim: ~47.7 s).

## p2_s03_forced_prefill — suffixes through the decode graph
- Evidence: prefill 46%, ~107 ms per eager call regardless of length; suffixes are short (mean 70 / 26 tokens).
- Change: `submit` queues the suffix; the graph feeds the queue head instead of the row's own output.
- Estimate (step simulator): ~47.7 s. Result: **46.7 s**, prefill calls 256 → 7, steps 797 → 1222.
- Next: tool 17.7% (inline tools block all rows).

## p2_s04_async_tools — tool calls off the decode loop
- Every replay updates all rows' linear states, so a waiting slot saves/restores its states (KV is safe).
- Result: **38.7 s** (predicted 39.6), check_batch suspend/resume 8/8 identical; decode 95%.

## p2_s05_inplace_state — no copy-back of recurrent states
- Evidence: `triton_poi_fused_copy_` 3.2 ms (10.7%) of a 29.8 ms step. Change: launch the same fla kernel with
  ht = h0 = cache buffer. Result: step 26.7 ms, **35.7 s**.

## p2_s06 / p2_s06t — FP8 weights
- Evidence: GEMM 78% of the step, bound by 54 GB of bf16 weights per step.
- Result: per-row 32.2 s, per-tensor **28.6 s**, accuracy unchanged both. GEMM fell 1.6× (not 2×): at M=32 the
  scale granularity picks the kernel (CUTLASS rowwise vs cuBLASLt), plus dynamic activation-quant kernels.
- Next: makespan is now set as much by step *count* as step time — 16.9 live rows per step in a 32-row graph
  (tail) and a ~1000-step longest chain → per-bucket graphs + compaction + 64 slots (sim ~22 s).

## p2_s07_bucket_graphs — tail and queueing
- Evidence: 16.9 live rows per step in a 32-row graph; the longest agent (~990 token-steps) can start late.
- Change: graphs for B ∈ {8,16,24,32,48,64} over [:B] views; logical slot → physical row map keeps live rows
  in [0, #live) via row swaps; S = 64. dynamo bug on the way (property reading cache.B not guarded).
- Result: **24.8 s**, 990 steps (chain 989), computed/live rows 1.89 → 1.12. check_batch with one bucket +
  per-row FP8: 20/20 identical; default 8/20 (bucket-dependent GEMMs + per-tensor FP8 couples rows).
- Lesson: per-tensor activation scales trade batch invariance for speed.

## p2_s08 / p2_s09 — recurrent state bytes, then the launch that can use them
- bf16 storage halved the bytes but the kernel went from bandwidth-bound to launch-bound (BV=8, 1 warp):
  −8% only (23.6 s). Sweep of (BV, warps, stages) in a micro → bf16 B=64 148 → 91 us/layer; **21.5 s**.
- Ablation: tuned launch on fp32 22.5 s. Each half alone gives part; together 3.3 s.

## p2_s10_force_all — last eager prefills
- 4 suffixes > 128 tokens → eager FP8 prefill 0.33 s each, stalling all rows. force_max 512 → **21.3 s**;
  same-turn simulation 21.5 → 20.2 s. Run-to-run chain variation (±70 steps) is now as big as one change.

## Phase 2 summary
474.8 s (p1_s07) → 21.3 s (22×), 1673.6 → 21.3 s overall (78.5×), accuracy 0.93–0.94 throughout.
Order of impact: continuous batching (7.8×), forced-token prefill (1.30×), FP8 (1.25×), async tools (1.21×),
bucket graphs + 64 slots (1.15×), state launch/bf16 (1.15×), in-place state (1.08×). Prefix reuse alone gave
0× (it enabled forced prefill). The run is now bound by its longest agent: ~1000 steps × 15–23 ms.

## Reference: vLLM 0.30 (same workload; `results/ref_vllm_fp8/README.md`)
- 3 repeats each. fp8: ours (p2_s10) 20.3 s median vs vLLM 20.6 s. bf16: ours (p2_s10_bf16) 28.2 s vs vLLM 28.9 s.
- Both are bound by task 423's chain. With the same 751-token chain, ours costs 26.5 vs 27.4 ms per chain token.
- vLLM admits all 128 agents, so its task p50 is 10.4 s vs ours 5.8 s (64 slots).
- vLLM loads faster: 223 s vs 580 s. Final write-up: `results/FINAL_REPORT.md`.
