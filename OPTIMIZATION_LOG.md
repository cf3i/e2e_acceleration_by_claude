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

# Phase 3 — systems optimization toward the physical limit
Fixed: FP8 per-tensor weights, bf16 lm_head / KV / recurrent state, greedy, no speculation (optimize the system,
not the model). Task 423 (a 750-token repetition outlier) excluded from both workloads. Two workloads:
gsm8k 127 tasks, and `gsm8k_longctx_128` (1k–7k-token prompts, no shared context). Rebaseline p2_s10: 18.19 s.
Dev numbers (`results/_dev_*`, dev job, ±0.1 s / ±1 s) in order; official runs via `slurm/submit.sh` at the end.

| step (commit) | 127-task | longctx |
|---|---|---|
| v3 engine: own forward, async token-budget scheduler, chunked prefill mixed with decode, paged KV, every step a CUDA graph (3252fc5) | 15.58 → 10.95 s | 51.5 s |
| kernels: 128-row prefill attention tiles, single-pass silu_mul_quant, split-KV decode attention (d375bad) | 9.95 s | 47.4 s |
| merged small-M FP8 GEMMs (gate\|up, qkv\|z\|b\|a, q\|k\|v) in Triton (802ce97) | 9.51 s | 45.0 s |
| runner: no device sync per timed section, incremental tokenization (624421f) | 8.96 s | 44.6 s |
| programmatic dependent launch on every kernel (b7bdbf8; off: 9.30 s) | 8.95 s | 42.2 s |
| fused GDN chunk recurrence + output, conv fast path (78bf72d) | 8.85 s | 41.65 s |
| small-M GEMM retune, graph-loop timing (2defad5) | 8.77 s | — |
| **official p3_s01_kernels / _longctx (2defad5)** | **8.86 s** (acc 0.945) | **40.64 s** (acc 0.945) |
| GDN chunk kernel BV 64 / 4 warps / 3 stages (e8ea498; official p3_s02_gdnchunk) | 8.77 s | 39.61 s |
| gdn_decode tiers tuned in-model under the power cap (3eaf6fe; official p3_s03_inmodel) | 8.67 s (A/B) | 39.8 s (A/B, noisy) |
| GDN prefill: recompute_w_u fused into the chunk recurrence (0abefc7; official p3_s04_gdnfuse 8.47 / 40.83 s) | 8.59 s (A/B) | mixed step −1.8 ms |
| GDN kkt + inverse on tensor cores, cumsum fused, no memsets (0b442cc; official p3_s05_kkt 8.99 s) | **broken**: longctx acc 0.417 (p3_s05_kkt_longctx 77.8 s) | unstable tf32 inverse, see below |
| kkt diagonal blocks by fp32 forward substitution (097cd7f; official p3_s06_kktfix) | 9.45 s (acc 0.945; task 428 now 581 tokens, see replay below) | 40.46 s (acc 0.953) |
| packed bf16 rounding (_rbf16), bit-exact (1e84d12) | mixed step 114.25 → 113.80 ms | |
| mixed steps: prefill-row GDN / attention kernels on a side stream, concurrent with decode rows (6b5d5f5) | replay A/B: 8.486 → 8.429 s | replay A/B: 39.87 → 39.32 s |
| replay A/B, 0abefc7 → 6b5d5f5 (same GPU, same trajectories, 2 rounds each) | 8.515 → 8.429 s (−1.0%) | 40.13 → 39.32 s (−2.0%) |
| decode-step causal conv1d fused into the gdn_in GEMM epilogue, q/k l2-norm into gdn_decode (321c65f) | replay A/B: 8.414 → 8.380 s | |
| prefill attention by FA3 (built from source, paged, varlen) on the side stream (321c65f) | | mixed step 116.2 → 114.3 ms; both: replay A/B 39.17 → 38.90 s |
| official p3_s07_fa3conv / _longctx (321c65f) | 8.458 s (acc 0.945) | 39.00 s (acc 0.937, see below) |
| vLLM 0.30 fp8, same workloads | 17.54 s | 91.96 s |

## Where the time is vs physics (2defad5 + 01fe367)
- Mixed step (2×1024 prefill + 48 decode rows @ ctx 4000): 114 ms. GEMMs 76 ms at ~1.31 PFLOPS = cuBLAS FP8
  practical peak on this (power-limited) GPU; silu/rmsnorm at 88–90% of practical HBM bandwidth; GDN prefill
  (fla intra kernels + fused chunk kernel) ~11 ms vs ~2–3 ms of bytes/FLOPs — the main remaining gap.
- Decode B=128 @ ctx 1000: 24.3 ms vs ~18.3 ms of bytes (weights 9.1 + state r/w 6.5 + KV 2.8 ms at 2.97 TB/s).
  gdn_decode 2.66 TB/s = the ceiling of its own access pattern (pure tile read-modify-write: 2.64–2.86 TB/s;
  flat in-place RMW 3.0 TB/s); attn_decode 2.8–3.0 TB/s; cuBLAS GEMMs 2.5–2.9 TB/s.
- Decode B=1: 11.1 ms vs 8.5–9.1 ms weight read; the rest is ~1.5 ms of latency-bound small kernels.

## Tried, no gain (reverted)
- L2 prefetch of the next weights (`cp.async.bulk.prefetch.L2` before gdc_wait): per-element slower, bulk neutral.
- Early `griddepcontrol.launch_dependents`: slower at B ≥ 64.
- Two streams in mixed steps (decode rows ∥ prefill rows): GDN kernels never overlapped, attention 0.9 of 4.5 ms;
  step time unchanged — both sides already fill every SM. (Phase 3 retry with the side stream at high priority
  and all kernels on PDL: kept, see 6b5d5f5 / "Streams" below.)
- Triton GEMMs for down/out projections or large-M wide GEMMs: slower than cuBLAS.
- gdn_decode with both reductions on the old state in one pass: same speed (bytes-bound, not reduction-bound).
- Lesson (benchmarks): Triton `do_bench` flushes L2 and leaves dirty lines that skew 30–60 µs kernels; time in
  CUDA graphs cycling over ≥ 2 weight copies instead. Warm up on a side stream only after `wait_stream`
  (otherwise the first launch races the input initialization — it looked like a kernel OOB bug).

## Measurement: official runs vs same-GPU A/B (p3_s03 / p3_s04)
- Official runs land on different nodes: p3_s03 8.98 s (h100-5) vs p3_s02 8.77 s (h100-2) although same-GPU A/B
  (alternating revisions, 2 rounds, dev job) gives e8ea498 8.74 / 3eaf6fe 8.67 / 0abefc7 8.59 s. p3_s04 official
  8.47 s. Node-to-node spread ~2–3% > one change: compare revisions only on one GPU (`/mnt/data/u/feic/tmp_ab/ab.sh`).
- longctx makespan depends on the scheduling trajectory (620–756 steps across runs of nearby revisions; one run
  even had a different accuracy 0.953 from a different batch composition): judge kernels by per-step GPU time
  (`scratch/prof_mixed.py`) instead.

## Measurement: trajectory replay (`run_bench.py --replay <results dir>`)
- The 127-task makespan is the latency of the slowest task: p3_s06_kktfix 9.45 s vs p3_s04 8.47 s is task 428
  generating 581 instead of 391 tokens (7.15 → 9.31 s end to end; total generated tokens +1.7%, steps 515 → 593,
  GPU time per step 16.3 → 15.7 ms). Any rounding change moves it by ±1 s, more than any one optimization.
- `--replay` forces a recorded run's generated tokens (turns.jsonl `gen_ids`, else its transcripts re-tokenized):
  the host sees the recorded tokens (stops, tool calls, next prompts all follow the recording), the device still
  feeds its own samples (same work per step). Same-GPU A/B with replay is reproducible to ±0.005 s (127-task) /
  ±0.05 s (longctx), and isolates system speed from numerics. Accuracy still comes from normal runs.

## Streams (6b5d5f5)
- In a mixed step, the prefill-row GDN kernels (kkt + chunk) and attention prefill run on a high-priority side
  stream, concurrent with gdn_decode / attn_decode (disjoint rows and state slots; parallel branches of the CUDA
  graph; bit-exact). Big steps (2k tokens): the sections shorten (gdn 246 → 227 us, attn 821 → 731 us per layer)
  but the step does not — power-capped, energy per step is the bound. Small steps (B=40, 40 prefill rows):
  20.6 → 20.0 ms. End to end (replay A/B): 127-task −0.7%, longctx −1.4%.

## The power wall (longctx)
- Mixed steps run at ~690 of 700 W (throttle reason SW power cap), SM clock ~1.55–1.6 GHz instead of 1.98.
  cuBLAS FP8 there: ~1.35 PFLOPS. Longctx compute floor ≈ 549k prompt tokens × 48.7 GFLOP / 1.35 PF ≈ 20 s +
  ~1.2 s attention; mixed steps now take ~33 s of the run (≈ 64% of floor efficiency, GEMMs themselves ≈ 95%).
- Consequence: memory-bound kernels must be tuned inside the model. gdn_decode: an isolated micro showed a tie
  between configs, in-model (lower clocks) the software-pipelined head loop (HPP 4, STAGES 3) wins: 128 rows
  7.31 → 6.87 ms per decode step, 56 decode rows of a mixed step 4.05 → 3.41 ms.
- attn_decode: num_warps / num_stages / nsplit all within ±2% in-model (KV-bandwidth bound); unchanged.

## GDN prefill: no w / u in HBM (0abefc7)
- fla computes u = A (beta v), w = A (beta e^g k) in a separate kernel, and the recurrence uses v_new = u − w h.
  Same thing: v_new = A (beta (v − e^g k h)) — one 64×64 dot per chunk inside the recurrence instead of two
  materialized [T, 48, 128] tensors (write + read ~74 MB per 2k-token layer) and a kernel. Same bf16 MMA inputs
  (state diff vs before: 1 bf16 ulp), accuracy unchanged in both A/B workloads.
- Per layer incl. kkt_solve: 1992 tokens 189 → 160 us, 4000: 350 → 289, 4×512: 177 → 142. The bit-identical
  variant (u, w computed per chunk in registers) is 20% slower than before: the extra dots sit on the serial chain.
- Masking A's upper triangle in-kernel instead of zero-filling (saves a memset) is slower (masked load stalls).
- The recurrence is latency-bound (~4.6 us per 64-token chunk at 96 programs ≈ 15% of SM MMA peak); more
  programs (BV 32) or warps do not help in Triton.

## Tried in phase 3 kernel work, no gain
- PDL + early `launch_dependents` + L2 bulk prefetch of the next GEMM's weights: all CTAs resident ~19 us early
  (globaltimer), but ≤ 2.5 us per GEMM gain and none end-to-end.
- Split-K for the N = 5120 linears at small M: ≤ 0.9 us. Rowwise-scaled merged GEMMs (bit-exact merging of
  linears with different per-tensor scales): 15–30% slower than per-linear cuBLAS. gdn state contiguous layout:
  same speed.
- Prefill attention (ours ~473–487 TFLOPS): TMA descriptors +3%, warp specialization (Triton 3.6 crashes at
  4 warps), lazy rescaling, manual QK^T pipelining: no gain. Reference FA3 (vLLM build, hdim 256 GQA 6, page 64):
  653 TFLOPS at 1992@3000, 626 at 2×1024@2000, 520 at 1992@0 — worth ~2 ms per mixed step (~1.4% longctx),
  needs a hand-written warp-specialized kernel.

## GDN kkt on tensor cores (0b442cc) — and the accuracy bug it shipped (fixed in 097cd7f)
- fla's fused kkt + solve_tril: forward substitution on 16×16 diagonal blocks (14 serial cross-lane reductions),
  ~48 us per layer at 2k tokens vs ~10 us for loading k and computing k k^T alone. 0b442cc inverted the diagonal
  blocks by (I+Y)^-1 = (I−Y)(I+Y²)(I+Y⁴)(I+Y⁸) (Y strictly lower ⇒ nilpotent) in tf32 on tensor cores, then
  tf32 block merges like fla; gate cumsum fused, no zero-fill. On random (nearly orthogonal) keys its error vs an
  fp64 inverse equalled fla's, and the 127-task accuracy stayed 0.945.
- **It was wrong.** Official p3_s05_kkt_longctx: accuracy **0.417** (0.945 before), 77.8 s (garbled answers, long
  generations). Real keys inside a chunk are correlated: with k_i·k_j ≈ 1 the powers Y², Y⁴, Y⁸ have entries up
  to ~C(15, 8) ≈ 6000 times the result, so tf32's 1e-3 relative error leaves errors of 2–15 on entries ≤ 1
  (scratch/kkt_adv.py; fla stays at 3e-3). The product formula is numerically unstable, not just imprecise.
  The 127-task run (short prompts) only showed it as a "trajectory change": task 428 grew from 391 to 517–564
  tokens, which I wrote off as rounding noise. Lessons: (1) random-input kernel tests miss conditioning problems —
  test_v3_kernels now has a correlated-keys gdn_prefill case, which fails on 0b442cc; (2) a numerics change that
  moves trajectories needs the long-context accuracy run before it is believed, not after.
- 097cd7f: diagonal blocks by fp32 forward substitution again (the four blocks interleaved in one loop), merges
  and k k^T blocks stay as in 0b442cc. Same error as fla on every case of kkt_adv. gdn_prefill per layer
  (fla kkt → ours): 2 chunks 27.2 → 25.5 us, 1992 tokens 163 → 143 us, 4×512 144 → 129 us (0b442cc's broken
  version: 20.7 / 132 / 116).


## Ledger and the last GEMM / decode sweeps (before 321c65f)
- GPU ledger (127-task, replay): 8.36 s busy, 0.07 s idle — the host side is done. longctx: 80% of GPU time is
  mixed 2048-token steps (~1992 prefill + ~55 decode rows), ~116 ms each, of which cuBLAS FP8 GEMMs ~73 ms at
  ~1.33 PFLOPS (the power-capped practical peak).
- Medium-M GEMMs (M 64–512), Triton with TMA / warp specialization / swap-AB vs the current dispatch: ≤ 2% better
  on a few shapes, worse on most; kept (Triton mm_fp8 for gdn_in at small M, cuBLAS for down / out). Distance to
  the weight-read floor at M=128: gate_up 68.7 vs 57.5 us, down 33.6–35.3 vs 28.8, out 14.2–14.9 vs 10.1.
- The two N=48 b / a GEMMs at large M merged into one call: ~3 us per layer. Not done.
- Decode B=1: 11.07 ms vs 8.41 ms speed-of-light; GEMMs ~93% of their read floor, ~1.5 ms of latency-bound
  small kernels (add_rmsnorm, gnorm, silu ~2.5–4 us per layer each). Early trigger + L2 prefetch: other −0.32 ms,
  GEMMs +0.27 ms, net −0.05 ms. Closing the rest needs a persistent megakernel.

## FA3 prefill attention + decode conv fusion (321c65f)
- FA3 (hopper, built from source with only hdim 256 bf16 forward, paged KV, varlen; `/mnt/data/u/feic/pjs/src/
  build_fa3.sh`): 1992@3000 301 vs 411 us (ours), 1992@6000 501 vs 738, 4×512@1000 113 vs 171. Called directly
  on our paged cache (page table rows of the prefill slots, `p_seqk` = lengths after the step, 0 for padding
  entries so the fixed-size graph batch works). Falls back to the Triton kernel if flash_attn_3 is missing.
- Conv fusion: in decode steps the causal conv1d (window of 3 + new token, silu) runs in the gdn_in merged GEMM
  epilogue on the qkv columns; one program owns each (slot, channel) window, so it is race-free in place. The q/k
  l2-norm moved into gdn_decode (NORM flag) — the only non-bit-exact part (reduction order, rare last-bit).
- Accuracy: 127-task 0.945 (unchanged); longctx 0.937 vs 0.953 for p3_s06 — tasks 304 (also wrong in p3_s04)
  and 428 now run out of tokens in a verbose first turn. Historical longctx range 0.9375–0.953.
- Not caused by either change: real (non-replay) longctx runs of the same tree on the dev GPU
  (`scratch/accenv.sh`): both on 0.953 (the official run of the same commit: 0.937), FA3 off 0.953, conv fusion
  off 0.945, both off 0.961. Run-to-run spread of the same code is ±2 tasks (batch composition differs between
  nodes and runs), so longctx accuracy differences under ~0.02 are noise; only collapses (0b442cc) are signals.
