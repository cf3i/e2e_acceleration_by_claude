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
