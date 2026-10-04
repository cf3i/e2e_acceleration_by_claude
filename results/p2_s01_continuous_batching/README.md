# p2_s01_continuous_batching — 32 agents in flight, one decode graph for all of them

Phase-1 end state: decode is 92% of e2e and 92% of a decode step is reading the 54 GB of weights. The workload
has 128 independent agents, so the same weight read can serve many sequences.

Change:
- `agentopt/engines/slot_cache.py`: a slot cache duck-typed to what HF's Qwen3.5 code calls. Each sequence
  owns a slot; in decode, row r = slot r writes its KV at its own position (scatter) and has its own mask row.
  (HF's StaticCache shares one write position across the batch, so rows cannot be at different lengths.)
- `agentopt/engines/batch.py`: batch-1 eager prefill into a slot; decode = one CUDA-graph replay of the compiled
  phase-1 step at batch 32. Idle rows are parked at a scratch position.
- `agentopt/runner.py::run_continuous`: each iteration admits tasks into free slots, runs one decode step for all
  slots, and finishes turns whose row emitted a stop token (detokenize → tool → next-turn prefill on the slot).

Correctness:
- `check_batch` (job 297192): 8/8 bit-identical decoding a prompt alone vs concurrently (staggered admission,
  other slots) → no cross-talk between rows.
- `check_equiv` (job 297193): 8/8 identical to p1_s07 (batch 1).

## End-to-end (job 297196)
| makespan | vs p1_s07 | vs baseline | accuracy | p50 / p90 task latency | peak mem |
|---|---|---|---|---|---|
| **60.8 s** | **7.8×** | **27.5×** | 0.9375 | 11.5 / 20.4 s | 59.0 GB (cache 8.6 GiB) |

Decode: 797 steps × 30.4 ms (vs 21.4 ms at batch 1) with ~25.5 live rows on average → 1.2 ms per generated
token amortized. Task latency goes up (3.2 → 11.5 s p50): the classic throughput/latency trade.

Breakdown (single-threaded loop, so sections are serial):
| prefill | decode | tool | template |
|---|---|---|---|
| 27.9 s (45.9%) | 24.2 s (39.8%) | 8.2 s (13.4%) | 0.4 s |

## Amdahl → next
- Prefill (46%): 255 batch-1 prefills, ~110 ms each, and every one stalls all 32 decoders. 55% of prompt tokens
  re-prefill the previous turn, and 352 of the ~400 first-turn tokens are a prefix shared by all tasks (system
  prompt + tool schema). Verified token-exact: turn k+1's prompt starts with turn k's prompt + generated tokens
  (40/40). → **p2_s02 prefix reuse**. Caveat: an eager batch-1 forward has ~50–60 ms of fixed host overhead
  (p1_s02 decode), so even tiny suffix prefills will cost that.
- Tool (13%): runs inline and blocks the decode loop → async tool workers later.
