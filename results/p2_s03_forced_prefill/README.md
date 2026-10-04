# p2_s03_forced_prefill — short suffixes go through the batched decode graph, not eager prefill

p2_s02 lesson: an eager batch-1 prefill costs ~107 ms per call regardless of length (host-bound), and blocks all
32 rows. After prefix reuse the suffixes are short: first turn after the shared prefix mean 70 / p90 97 / max 150
tokens, next turn mean 26 / p90 38.

Change (`batch.py::submit`, `runner.py`):
- The decode graph takes each row's input from `where(force_on[r], force_tok[r], last_output[r])`; the host
  fills a pinned (2, S) buffer and copies it in (one async H2D) before each replay.
- A reusable suffix of ≤ `force_max` = 128 tokens is queued on the slot (after restoring the shared snapshot, or
  directly at the slot's end for the next turn) and fed one token per step; the row's outputs are ignored until
  the queue is empty, then its output is the turn's first token. Longer / non-reusable prompts: eager prefill.
- `step()` now returns `{slot: token}` for rows that are generating; turns start without a first token.

Prediction (step simulator over p2_s02's turns, 32 slots, 30.4 ms/step): 47.7 s.

Correctness (`check_batch`, job 297214): shared-snapshot + forced suffix vs full eager prefill **8/8 identical**;
extend (17 forced tokens) vs full 6/8 (near-ties at token 14 and 42); alone vs concurrent (forced rows mixed with
generating rows) 8/8 identical.

## End-to-end (job 297215)
| makespan | vs p2_s02 | accuracy | prefill calls | forced tokens | decode steps | p50 / p90 latency |
|---|---|---|---|---|---|---|
| **46.7 s** | **1.30×** | 0.9375 | 256 → 7 | 11.9k | 797 → 1222 | 11.3 → 8.6 / 20.1 → 14.3 s |

Breakdown: decode 36.8 s (78.9%, 30.1 ms/step), **tool 8.3 s (17.7%)**, prefill 1.1 s (2.3%).
Measured 46.7 s vs predicted 47.7 s. More steps (the suffix tokens are now row-steps), but no stalls and the 6
idle rows of an average step absorb them.

## Amdahl → next
- Tool 17.7%: tools run inline and the whole batch waits (~63 ms per call, 127 calls). Async tool workers:
  simulator → ~39.6 s. Needs the slot's Gated-DeltaNet state to survive while it waits (every replay updates all
  rows' states in place). → **p2_s04**.
- Decode step (profile_batch job 297216, 29.8 ms GPU): GEMM 20.9 ms (70%), fla recurrent 3.9 ms, **state copy
  `triton_poi_fused_copy_` 3.2 ms** (cache `copy_` of the 4.8 GB fp32 recurrent states every step), rest ~1.7 ms.
