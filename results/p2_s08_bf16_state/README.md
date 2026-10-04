# p2_s08_bf16_state: Gated-DeltaNet recurrent states stored in bf16

p2_s07: at B=64 the fla decode kernel took 7.7 ms of a 26.2 ms step (30%), and it grows with rows. It reads and
writes 9.7 GB of fp32 states per step (48 layers × 64 rows × 48 heads × 128 × 128 × 4 B).

Change: `state_dtype="bf16"`. The slot cache allocates the recurrent-state storage in bf16 (cache 17.2 → 12.7 GiB).
fla's kernels already load the state as `.to(float32)`, compute in fp32 and store as the buffer's dtype, so
this is only a storage change. Prefill's fp32 final state is cast when copied into the slot. No kernel change.

First run (job 297326): **34.6 s**. One turn's prefill took 10.6 s: the first extend-prefill with a bf16
`initial_state` JIT-compiled new Triton specializations of fla's chunk kernels in the middle of the run (a cold
effect again). Fix: the engine now runs one extend-prefill at load time. Re-run (job 297331) below.

## Results
| | p2_s07 | p2_s08 |
|---|---|---|
| recurrent kernel / step (B=64, nsys) | 7.74 ms | **7.11 ms (−8%)** |
| step GPU (B=64) | 26.2 ms | 25.5 ms |
| makespan | 24.8 s | **23.6 s (1.05×)** |
| accuracy | 0.9375 | 0.9375 (same 8 tasks wrong) |
| decode steps / gen tokens | 990 / 20879 | 947 / 21152 |

## Why halving the bytes saved only 8%
In fp32 the kernel ran at ~2.4 TB/s, close to the HBM roofline. In bf16 it ran at 1.36 TB/s: it was no longer
bandwidth bound. fla launches it with BV = 8 (a 128 × 8 tile of the state per program) and 1 warp, i.e.
64 × 48 × 16 = 49k tiny programs. With half the bytes per tile, per-program overhead (loading q / k / v / g /
beta, the reduction over K) dominates. The byte saving needs a matching launch config → **p2_s09**.
Part of the 1.2 s makespan gain is run-to-run variation in the transcript (43 fewer steps).
