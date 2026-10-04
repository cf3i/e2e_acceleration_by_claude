# p2_s10_force_all: every turn's suffix as forced tokens (force_max 128 → 512)

p2_s09: 4 turns had a suffix longer than 128 tokens (2 long first-turn questions, 2 long tool results). Each one
went through eager batch-1 prefill at ~0.33 s (FP8 eager, with per-call dynamic-quant overhead, is slower than
the 0.107 s of bf16 eager). While it runs, all other rows wait, so 1.35 s sat on the critical path.

Change: preset only, `force_max=512`. Every suffix goes through the decode graph. The only eager prefill left
is the shared system prefix, once at the start of the run.

## Results (job 297340)
| | p2_s09 | p2_s10 |
|---|---|---|
| makespan | 21.5 s | **21.3 s** |
| accuracy | 0.9297 | 0.9375 |
| eager prefill calls / time | 5 / 1.94 s | **1 / 0.61 s** |
| non-decode time per step (sim-calibrated) | 2.89 ms | **1.53 ms** |
| decode steps / longest chain | 948 / 945 | 1015 / 1013 |
| p50 task latency | 6.5 s | 5.8 s |

The prefill saving is real (−1.3 s), but this run's longest chain is 68 steps longer. The transcript of
task 423 changed with the numerics, and 68 × ~20 ms is the same size as the saving. Replaying p2_s09's turns in
`sim_steps.py` with p2_s10's 1.53 ms/step overhead gives **20.2 s** for the same work: that is the effect of
this change alone. **At this point run-to-run variation of the critical path (±70 steps ≈ ±1.4 s) is as large
as a single optimization**, so differences below ~1 s should be read through the simulator, not one e2e run.

## Where the time goes now (bucket-8 profile, `profile/live8`)
The tail step (370 of 1015 steps) is 15.1 ms GPU:
- FP8 GEMMs 10.3 ms, at ~77% of HBM bandwidth on the weight read: this is the floor;
- ~1900 small inductor kernels 2.5 ms (FP8 dynamic-quant amax / scale / cast per linear + norms, ~1.3 us each);
- bf16 lm_head + attention GEMMs 1.3 ms;
- recurrent 0.55 ms; conv 0.16 ms.

Makespan ≈ 1000-step chain × (15–23 ms). The remaining levers are fewer bytes per step (int4 weights: a
quality risk that needs calibration) or fewer steps on the chain (speculative decoding, which on a hybrid
model also needs recurrent-state rollback). Both are out of scope for this learning exercise. **Diminishing
returns: the engine-level phase stops here.**
