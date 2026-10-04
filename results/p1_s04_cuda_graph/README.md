# p1_s04_cuda_graph — StaticCache + CUDA-graph-captured decode step

Engine `StaticEngine(cuda_graph=True)` (`agentopt/engines/static.py`) on top of p1_s02 (no cuDNN SDPA, fla fast path).
One decode step (forward + argmax + feed token back + advance position + extend static mask) is recorded once and
replayed per token.

- Correctness (`equiv.json`): identical to eager static (p1_s03); 7/8 identical to p1_s02, one near-tie divergence
  at token 42 inside a code comment.
- Micro (`micro.json`): decode **31.5 ms/tok** at ctx 512–2048 (p1_s02: 59.0) → 51% of the 16.1 ms roofline.
- **e2e (`metrics.json`): makespan 660.8 s (baseline 1673.6 s → 2.53×)**, accuracy 0.9375 (baseline 0.930),
  decode 30.6 ms/tok, prefill 4455 tok/s. Breakdown: decode 94.0%, prefill 4.3%, tool 1.6%.
  Predicted ~700 s from micro numbers.
- Profile (`profile/`, nsys with `--cuda-graph-trace=node`): GPU kernel time per replay 30.3 ms ≈ the whole step
  → GPU-bound now. Per replay: GEMM 18.95 ms (626 kernels; 85% of HBM bandwidth), torch elementwise/copy 6.48 ms
  (**2648 kernels**, avg 2.4 µs), mem-efficient attention 3.92 ms (16 calls × 245 µs over 4096 masked slots for
  16 MB of real KV), fla/conv1d 0.5 ms, reductions 0.43 ms.
- Note: the first profile run reported `cudaGraphLaunch` ≈ 4 ms CPU; that is inflated by profiler overhead
  (p1_s05 recovered only 0.4 ms by hiding it).

Next by Amdahl: fuse the elementwise kernels (torch.compile, ≈5 ms) → p1_s06; attention over the real length
(≈3 ms) → later; FP8 weights (≈9 ms of GEMM, changes numerics) → phase 2.
