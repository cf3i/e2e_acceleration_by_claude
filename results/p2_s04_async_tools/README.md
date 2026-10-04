# p2_s04_async_tools — tool calls in a thread pool, the batch keeps decoding

p2_s03: tool 8.3 s (17.7%). Each python tool call (~63 ms subprocess) ran inline, so all 32 rows waited.

Change:
- `runner.py::run_continuous(async_tools=True)` (runner `continuous_async`): a turn that ends in a tool call
  suspends its slot and submits the tool to an 8-thread pool; finished tools are collected between decode steps
  and the next turn is submitted on the same slot (forced suffix = tool result message).
- `batch.py::suspend / _resume`: each replay updates every row's Gated-DeltaNet conv/recurrent state in place,
  parked or not, so suspend saves the slot's linear states (~151 MB) and parks the row; resume copies them back,
  clears the scratch-position mask bit and moves the row back to its end. KV needs nothing (parked rows only
  write the scratch position).

Correctness (`check_batch`, job 297219): every odd prompt is suspended for 7 steps after 10 tokens, then resumed
by submitting its tokens + the pending token (1 forced token = what uninterrupted decode would feed):
**8/8 identical to running alone** → the state round-trip is exact.

## End-to-end (job 297220)
| makespan | vs p2_s03 | vs baseline | accuracy | p50 / p90 latency | decode steps |
|---|---|---|---|---|---|
| **38.7 s** | **1.21×** | **43.2×** | 0.9375 | 7.3 / 11.9 s | 1230 |

Breakdown: decode 36.9 s (**95.3%**), prefill 1.2 s, template 0.4 s; tool time (8.6 s summed) fully overlapped
(the loop never had to wait for a tool). Predicted 39.6 s.

## Amdahl → next
Decode is 95%, so the step itself (29.8 ms GPU) is the target:
- recurrent-state copy 3.2 ms (10.7%) → write states in place → **p2_s05** (expected ~−10%).
- GEMM 20.9 ms (70%), bound by reading 54 GB of bf16 weights → FP8 weights → **p2_s06**.
