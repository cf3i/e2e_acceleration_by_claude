# p2_s07_bucket_graphs: 64 slots, one CUDA graph per batch bucket, live rows compacted

p2_s06t: decode was 92% of 28.6 s, but over 1251 steps only 16.9 rows per step were live on average in a 32-row
graph. Two separate losses:
- **Tail**: in the last part of the run most slots are empty, but every step still computes all 32 rows.
- **Queueing**: 128 tasks on 32 slots means the last tasks start late. The longest agent (task 423, ~990 token
  steps over 2 turns) can start in the second wave, so makespan > its chain.

Change (`batch.py`, `slot_cache.py`):
- One graph per bucket B in {8, 16, 24, 32, 48, 64}. Each computes rows [0, B) only: [:B] views of the KV /
  state storage, mask, positions and tokens. All graphs share one memory pool.
- Live sequences are kept in rows [0, #live). There is a logical slot -> physical row map. A sequence that
  becomes live is swapped into row #live; one that leaves (release / suspend) is swapped with the last live
  row. A swap moves one row's KV + states (~0.3 ms). `step()` replays the smallest bucket >= #live.
- KV and states are stored as one tensor per kind ([layers, S, ...]), so a swap is 4 copies, not 4 × 64.
- S = 64: 64 × (134 MB KV + 151 MB states) = 17.2 GiB cache.

Bug found on the way: the per-bucket state views were first exposed through a property reading `cache.B`.
dynamo did not guard on it reliably, so a frame traced for B = 64 was reused for B = 48 and crashed with
`conv_state must have shape (batch_size, ...)`. The fix uses plain attributes re-pointed by `set_rows(B)`.

## Correctness (`check_batch`, 20 prompts, staggered admission, suspend/resume, slots S-1-i → many swaps)
| config | alone vs concurrent |
|---|---|
| p2_s07 as is (job 297301) | 8/20 identical |
| `--one-bucket --opt fp8='"row"'` (job 297328) | **20/20 identical** |

Compaction, swaps and suspend/resume are exact: with a single graph and row-independent numerics, every
sequence matches running alone. The default config differs for two reasons, and both are numerics, not
cross-talk:
- **Bucket size changes the GEMM.** A sequence alone runs in bucket 8, concurrently in 16 / 24. cuBLAS picks a
  different kernel and reduction order per M.
- **Per-tensor FP8 couples rows.** The activation scale is the amax over the whole batch, so a row's
  quantization depends on the other rows. Batch invariance is what per-tensor FP8 trades for its speed. The
  benchmark accuracy is unaffected, but a request's exact output now depends on its neighbours.

## Results
| | p2_s06t (S=32, 1 graph) | **p2_s07** |
|---|---|---|
| makespan | 28.6 s | **24.8 s (1.15×)** |
| accuracy | 0.9375 | 0.9375 |
| decode steps | 1251 | **990** (longest chain: 989) |
| computed / live row-steps | 1.89 | **1.12** |
| p50 / p90 task latency | 5.4 / 8.8 s | 7.3 / 11.1 s |
| engine load (6 graphs compiled + captured) | ~120 s | 519 s |

Bucket replay times (`profile_batch`): B=8 16.2, 16 17.8, 24 19.5, 32 20.8, 48 24.0, 64 26.9 ms.
Per row: 64 rows cost 0.42 ms/row, 32 rows 0.65 ms/row.

The simulator predicted ~22 s. `bench/sim_steps.py` (step simulator on the run's own turns with the measured
bucket times) reproduces 24.8 s with 3.1 ms/step of non-decode time (eager prefills, templates). The gap came
from the S=64 step costing 26.9 ms, more than the scaled estimate.

**The run is now critical-path bound.** 990 steps ≈ task 423's 989-token chain, and simulating 96 or 128 slots
or longest-first admission changes nothing (±0.1 s). From here only shorter steps help, especially in the tail
(336 of the 990 steps are in bucket 8). An even longer chain would need speculative decoding.

## Amdahl → next
Step at B=64 (nsys): GEMM 14.6 ms (55%), **fla recurrent 7.8 ms (30%, was 19% at B=32)**, other 3.3 ms. The fp32
recurrent states (9.7 GB at 64 rows) are read and written every step, so this term scales with rows while the
weight read does not. → store the states in bf16 (**p2_s08**).
