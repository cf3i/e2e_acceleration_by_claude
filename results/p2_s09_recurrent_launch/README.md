# p2_s09_recurrent_launch: launch config of fla's decode kernel tuned per state dtype

p2_s08: the bf16-state kernel ran at 1.36 TB/s, limited by its launch configuration (BV = 8, 1 warp), not by
bytes.

Change: `bench/micro_recurrent.py` sweeps BV ∈ {8, 16, 32, 64} × num_warps ∈ {1, 2, 4, 8} × num_stages ∈ {1, 3}
for one decode step of one layer (in place, CUDA-graph timed, B ∈ {8, 32, 64}, fp32 and bf16). Then
`inplace_state.py` launches the same fla kernel with the best config per dtype (`recurrent_launch="tuned"`).
Same kernel and same math; only the tiling changes. This is what an autotuner would do, and fla does not
autotune this kernel.

## Micro (us per layer-step, `micro_recurrent.json`)
| B | fp32 default | fp32 best | bf16 default | bf16 best |
|---|---|---|---|---|
| 8 | 19.5 | 13.0 (BV 32, 1 warp) | 20.4 | 9.4 (BV 32, 1 warp) |
| 32 | 86.1 | 72.7 (BV 64, 4 warps) | 75.5 | 47.1 (BV 32, 1 warp) |
| 64 | 168.5 | 139.7 (BV 64, 4 warps) | 147.9 | **90.9** (BV 16/32, 1 warp) |

Chosen (robust across B): fp32 → (BV 32, 2 warps), bf16 → (BV 32, 1 warp). bf16 at B=64 reaches 2.2 TB/s.
The B=8 fp32 number above HBM peak is because 25 MB of state fits in the 50 MB L2.

## Results (e2e: 128 tasks, continuous_async)
| | recurrent / step (B=64) | step GPU (B=64) | makespan | accuracy | p50 / p90 |
|---|---|---|---|---|---|
| p2_s08 (bf16, default launch) | 7.11 ms | 25.5 ms | 23.6 s | 0.9375 | 7.4 / 11.1 s |
| p2_s09f (fp32, tuned) — ablation | ~6.8 ms (micro × 48) | – | 22.5 s | 0.9297 | 6.9 / 10.4 s |
| **p2_s09 (bf16, tuned)** | **4.13 ms** | **22.5 ms** | **21.5 s** | 0.9297 | 6.5 / 9.9 s |

Bucket replay: B=8 15.7, 16 16.7, 24 17.9, 32 18.7, 48 20.9, 64 23.1 ms (p2_s07: 16.2 … 26.9).

Accuracy: one extra task wrong (428) in **both** tuned runs, fp32 and bf16, and in neither untuned run. Its first
turn writes a long commented tool call that ends just under the 512-token cap. The tuned launch changes the
reduction order inside the kernel, the greedy path shifts slightly, and the turn now runs past the cap
(`no_answer`). It is a numerics flip of a borderline task (1/128), not a bf16 precision loss.

Ablation: of the 3.3 s from p2_s07 to p2_s09, the launch tuning on fp32 alone gives ~2.3 s, bf16 storage
alone ~1.2 s, and both together 3.3 s. The bf16 bytes pay off only with the retuned launch.

## Amdahl → next
`bench/sim_steps.py` on this run: 948 steps vs the longest chain of 945 token-steps (task 423: 122 forced +
239 generated, a tool call, then 72 forced + 512 generated). 96 or 128 slots or oracle longest-first admission
change < 0.2 s. Makespan ≈ chain × step time, so the levers are:
- non-decode time on the loop: 4 eager prefills (suffix > 128 tokens: long first questions or tool results)
  at ~0.33 s each (FP8 eager is slower than bf16 eager), 1.35 s, stalling every row → force_max 512
  (**p2_s10**);
- the tail step itself: 315 of 948 steps run in bucket 8 at 15.7 ms, 54% of the FP8 weight-read floor
  (~8.4 ms) → profile bucket 8.
