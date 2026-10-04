# p1_s07_gqa_decode_attn — single-token attention as two GQA-folded GEMMs

Evidence (p1_s06 profile): attention 4.45 ms of a 24.9 ms step. With an explicit mask, HF's SDPA path expands
K/V 4 → 24 heads (`repeat_kv`, a 6× copy of the cache per layer per token), then the memory-efficient kernel
runs one thread block per head over all 4096 slots (237 us/layer, vs ~5 us to read the KV bytes at HBM BW).

Change (`agentopt/engines/decode_attention.py`, registered via HF `AttentionInterface`): for q_len = 1 the 6
query heads sharing a KV head are folded into the query axis: `softmax(mask(q(b,4,6,d) @ Kᵀ(b,4,d,L))) @ V` —
cuBLAS batched GEMMs parallel along L, no copy, fp32 softmax. Prefill still uses SDPA.

Estimate: attention 4.45 → <1 ms → step 24.9 → ~21.5 ms.

- Correctness (job 297184): **8/8 identical** to p1_s06.
- Micro (job 297185): decode 25.6 → **21.4 ms/tok** (−16%), **75% of roofline**, flat over ctx 512–2048.
- Node-level profile (job 297189): 20.98 ms GPU per replay; **GEMM 19.22 ms = 92%** (incl. the two attention
  bmm's), inductor 1.23 ms, fla/conv 0.51 ms. The `fmha_cutlassF` + repeat_kv copies are gone.

## End-to-end, end of phase 1 (job 297188, `metrics.json`)
| makespan | vs baseline | accuracy | decode ms/tok | prefill tok/s | p50 task latency | peak mem |
|---|---|---|---|---|---|---|
| **474.8 s** | **3.52×** | 0.9375 (same as p1_s04) | 21.6 | 4457 | 3.2 s | 50.7 GB |

Breakdown: decode 92.0%, prefill 6.0%, tool 1.8%.

## Phase-1 exit
Decode is at 75% of the 16.1 ms HBM floor and 92% of the step is weight GEMMs at ~85% of HBM bandwidth. At
batch 1 the only remaining lever is reading fewer bytes (FP8 weights) — or kernel work, out of scope. The
workload, however, has 128 independent agents: phase 2 amortizes the weight read over many sequences.
