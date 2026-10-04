# p1_s06_compile — torch.compile (inductor) the decode forward, inside our own CUDA graph

`torch.compile(model, dynamic=False)` (no inductor cudagraphs) is used only for the decode step; the compiled
step is then captured in the same CUDA graph as before. Prefill stays eager.

- Correctness (`equiv.json`, job 297169): 7/8 identical to p1_s05; one divergence at token 14 (both
  continuations are valid code — fused kernels change bf16 rounding, a near-tie flips).
- Micro (job 297180): decode 31.1 → **25.6 ms/tok** (1.22×, 63% of roofline). Compile + capture: 36 s per process
  with a warm inductor cache (58 s cold).
- Node-level profile (job 297181, `profile/nsys_summary.txt`): 1485 kernels / 24.9 ms GPU per replay
  (p1_s04: ~3.6k / 30.3 ms). Elementwise 6.48 ms → 1.14 ms (644 fused inductor kernels). Now:
  GEMM 18.82 ms (75%), **attention 4.45 ms (18%)** — `fmha_cutlassF` 237 us × 16 layers over all 4096 slots plus
  a fused `repeat_kv` copy (19.6 us × 32) → next target (p1_s07).

## Pitfall found: inductor redirects the Triton cache
The first micro (job 297170) showed cold prefill at ctx 2048 = **13 s** (p1_s05: 0.27 s), although prefill is
not compiled. Inductor's `CachingAutotuner` sets `TRITON_CACHE_DIR=/tmp/torchinductor_$USER/...` when unset, so
every fla Triton kernel JIT-ed afterwards (a new prefill-length bucket) missed `~/.triton/cache` and re-compiled
+ re-autotuned. Fix: the engine pins `TRITON_CACHE_DIR` and `TORCHINDUCTOR_CACHE_DIR` to persistent dirs before
compiling → cold 2048 prefill 265 ms, and compile time 58 → 36 s.
