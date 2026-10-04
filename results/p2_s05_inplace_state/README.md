# p2_s05_inplace_state — the decode kernel writes the recurrent states straight into the cache

p2_s04: decode is 95% of the makespan. Batch-32 step profile (p2_s03 profile_batch, 29.8 ms GPU): a
`triton_poi_fused_copy_` of 66 us in each of the 48 Gated-DeltaNet layers = 3.2 ms (10.7%). HF's decode calls fla's
`fused_recurrent_gated_delta_rule(initial_state=cache, output_final_state=True)`, which writes the new state to a
fresh tensor, and the cache then `copy_`s it back: an extra read + write of 4.8 GB of fp32 states every step.

Change (`agentopt/engines/inplace_state.py`): launch the same fla kernel with `ht = h0 = the cache buffer` (each
program loads one (head, V-block) tile of h0 first and stores the same tile of ht last, so aliasing is safe) and
make `SlotCache.update_recurrent_state` skip its copy in decode mode. No new kernel.

Estimate: −3.2 ms/step → ~−10% of decode.

Correctness: `check_equiv` vs p2_s03 (job 297221) 7/8 identical (the known near-tie at token 14 of prompt 0;
inductor fuses the remaining elementwise ops differently once the copy is gone).

## Results
| step (profile_batch, job 297222) | e2e makespan (job 297224) | vs p2_s04 | accuracy | p50 / p90 latency |
|---|---|---|---|---|
| 29.8 → **26.7 ms GPU** (27.3 ms wall) | **35.7 s** | **1.08×** | 0.9375 | 6.7 / 10.9 s |

Step breakdown now: GEMM 20.95 ms (**78%**), fla/conv1d 4.36 ms (16%), inductor 1.23 ms (5%).
Breakdown e2e: decode 34.0 s (95%), prefill 1.2 s.

## Amdahl → next
GEMM is 78% of the step and 95% × 78% ≈ 74% of the makespan: 54 GB of bf16 weights read per step at 2.57 TB/s
(77% of HBM peak). FP8 weights halve the bytes → **p2_s06**.
