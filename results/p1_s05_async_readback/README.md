# p1_s05_async_readback — overlap graph launch with the previous step

Step k+1 is enqueued before waiting for token k (token copied to pinned host memory + CUDA event); after a stop
token the one speculative step is discarded.

- Correctness (`equiv.json`): 8/8 identical to p1_s04.
- Micro: decode 31.5 → **31.1 ms/tok** (−1.3%). The ~4 ms `cudaGraphLaunch` seen under torch.profiler was mostly
  profiler overhead; without it, launch + readback is ~0.4 ms per step.
- Kept (free, output-identical), but below the 5% threshold — counts toward the phase-1 exit rule.
