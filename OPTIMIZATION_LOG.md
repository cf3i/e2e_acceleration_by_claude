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
