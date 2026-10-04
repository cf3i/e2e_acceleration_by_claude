# p2_s02_prefix_reuse — reuse the previous turn and the shared system prefix (hybrid model)

p2_s01: prefill was 46% of the makespan (255 eager batch-1 prefills, ~110 ms each, every one stalls all 32
decoders). 55% of prompt tokens re-prefill the previous turn, and 352 tokens of every first-turn prompt are the
same system prompt + tool schema.

Change:
- Qwen3.5 is hybrid: KV can be reused at any prefix length, but a Gated-DeltaNet recurrent state exists only at
  the position where it was saved. So reuse is possible exactly (a) at a slot's current end (next turn of the
  same agent: turn k+1's prompt = turn k's prompt + generated tokens + `<|im_end|>` + tool message, verified
  token-exact) and (b) at a saved snapshot (the shared prefix, prefilled once).
- `slot_cache.py`: prefill can *extend* a slot (start > 0): KV written at [start, start+n), conv window and
  recurrent state continue from the slot's; `snapshot`/`restore` of a slot's prefix.
- `hybrid_prefill.py`: wraps the per-layer `causal_conv1d_fn` (prepend the previous K−1 inputs) and
  `chunk_gated_delta_rule` (`initial_state` = the slot's recurrent state) — HF's forward itself is unchanged.
- `batch.py` / `runner.py`: extend when the slot holds a prefix of the new prompt, else restore the shared
  snapshot (LCP of the first-turn prompts).

Correctness (`check_batch`, job 297205):
- shared-prefix restore vs full prefill: 8/8 identical; extend vs full prefill: 7/8 (one divergence at token 14,
  bf16 differences from processing the prompt in two chunks); alone vs concurrent: 8/8 identical.

## End-to-end (job 297207)
| makespan | vs p2_s01 | accuracy | prefill tokens | prefill time | decode | tool |
|---|---|---|---|---|---|---|
| **60.7 s** | **1.00×** | 0.9375 | 126.9k → **13.6k (−89%)** | 27.9 s → **27.9 s** | 24.2 s | 8.0 s |

Reused tokens: 67.2k from the previous turn, 46.5k from the shared snapshot.

## Why the estimate was wrong
Predicted ~−20 s from cutting 89% of the prefill tokens. Measured: 0 s. Median time per prefill call is the same
with and without reuse (turn 1: 107 → 108.5 ms, turn 2: 112 → 105.9 ms) although the calls now process ~53 tokens
instead of ~500. An eager batch-1 forward of this model costs ~105 ms **per call** almost independent of length
(the phase-1 micro already showed it: 105 ms @512, 124 ms @1024 — a big intercept, small slope): 64 layers of HF
Python + dispatch + ~10 fla Triton launches per Gated-DeltaNet layer, i.e. host-bound, the same disease the CUDA
graph cured for decode (eager decode was ~59 ms/token). **Amdahl's p has to be measured on the cost driver (calls),
not on the work (tokens).** The reuse machinery is still the prerequisite for the next step.

## Amdahl → next
- Prefill 46%: the remaining suffixes are short (first turn after the shared prefix: mean 70, p90 97 tokens;
  next turn: mean 26, p90 38). Feed them through the existing batched decode graph as *forced tokens* (the row
  ignores its output until the suffix is consumed) → no eager calls, no stall of the other 31 rows; a 70-token
  suffix costs 70 row-steps instead of ~3.5 steps × 32 rows. Step simulator on this run's turns: 60.7 s →
  **~47.7 s** (1303 steps × 30.4 ms + 8 s tool). → **p2_s03 forced-token prefill**.
- Tool 13% (inline, blocks the loop) → async tool workers after that.
