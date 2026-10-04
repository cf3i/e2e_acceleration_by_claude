# p1_s02_fla_fastpath — fla + causal-conv1d kernels for the 48 Gated-DeltaNet layers

Micro (`micro.json`, ctx 1024): prefill warm 412 → **110 ms** (3.8×); decode warm 59.9 → 59.0 ms/tok (no change).
First call pays Triton JIT/autotune (43 s; 11.8 s more at the first ctx-2048 prefill) — one-time per process.

Profile (`profile/`) — why decode did not move:
- kernels per generated token 5906 → 3541 (−40%), but GPU busy stays 0.29;
- the new ops are expensive *on the host*: `LayerNormGatedFunction` 156 µs CPU/call, `FusedRecurrentFunction`
  149 µs, `causal_conv1d_update` 56 µs (Triton's Python launcher + autotune-cache lookup per launch);
- 48 layers × ~360 µs ≈ 17 ms host time per token, which cancels the saved torch launches.
Lesson: fewer kernels ≠ faster when host-bound; what matters is host time per step. Removing it entirely needs
CUDA graphs (p1_s04), which also makes the per-launch Triton overhead disappear.
