# p2_s06_fp8 (+ p2_s06t_fp8_tensor): FP8 linear layers (torchao)

p2_s05: GEMM was 78% of the batch-32 step (20.7 of 26.7 ms GPU), and decode was 95% of the makespan. The GEMMs
are bandwidth-bound on 54 GB of bf16 weights per step (M = 32 rows, so the weights are read once for very little
math).

Change (`agentopt/engines/quant.py`): torchao `Float8DynamicActivationFloat8WeightConfig` on every Linear except
`lm_head` (bf16 kept for logit precision on greedy near-ties; it is 2% of the weights). Weights go to E4M3 offline
and activations are quantized per step inside the compiled graph. No new kernel. A/B on the scale granularity:
- **row** (`p2_s06`): per-token activation × per-channel weight scales, using the CUTLASS sm90 rowwise `_scaled_mm`.
- **tensor** (`p2_s06t`): one scale each, using the plain cuBLASLt FP8 GEMM (`nvjet_qqtst_*`).

## Results (profile_batch: 100 timed steps, then 20 steps under nsys; e2e: 128 tasks, continuous_async runner)
| | step wall | GPU/step | GEMM/step | fla+conv | makespan | vs p2_s05 | accuracy | p50 / p90 |
|---|---|---|---|---|---|---|---|---|
| p2_s05 bf16 | 27.3 ms | 26.7 ms | 20.7 ms | 4.36 ms | 35.7 s | 1.00× | 0.9375 | 6.7 / 10.9 s |
| p2_s06 FP8 row | 23.05 ms | 21.8 ms | 15.4 ms | 4.33 ms | 32.2 s | 1.11× | 0.9375 | 5.9 / 9.6 s |
| **p2_s06t FP8 tensor** | **20.86 ms** | **20.1 ms** | **12.9 ms** | 4.34 ms | **28.6 s** | **1.25×** | **0.9375** | 5.4 / 8.8 s |

Peak memory is 51.2 GB e2e for both (the bf16 weights are freed). Accuracy is unchanged, but the transcripts are
not identical: FP8 agents generate 21117 / 21134 tokens vs 20265 for bf16 (+4%; some answers take a different
path). The 4% extra tokens are paid for inside the measured makespan.

## Why GEMM fell only 1.6×, not 2×
Halving the bytes should halve a bandwidth-bound GEMM. Two things get in the way:
- At M = 32 the CUTLASS rowwise kernel is not at the bandwidth roofline (46.6 us for the largest projection;
  cuBLASLt per-tensor takes 34.9 us for the same shape). Granularity is a *kernel choice*, not only an accuracy
  knob: per-tensor routes the GEMM to the faster library kernel.
- Dynamic activation quantization adds ~60 small amax/scale/cast kernels per step (~1.4 us each, fused by
  inductor into the neighbouring elementwise ops when possible).

The per-tensor variant is the one carried forward: same accuracy on this workload, 2.2 ms per step faster.

## Amdahl → next
The step is now 64% GEMM, 22% fla/conv (the fp32 recurrent states, untouched by FP8). In the e2e run, though, the
step count matters as much as the step time: 1251 steps for 21.1k tokens, i.e. 16.9 rows per step on average in
a 32-row graph. The tail (the last agents still running while most slots are empty) costs full 32-row steps, and
the longest agent chain (~1000 steps) puts a floor under the makespan whatever S is. Step simulator: per-size
graphs over compacted live rows plus 64 slots → ~22 s. → **p2_s07**.
