# stage00_baseline — plain HF transformers, batch 1, serial agents

Engine `naive` (`agentopt/engines/naive.py`): HF `Qwen3_5ForCausalLM`, bf16, SDPA attention (torch picks cuDNN),
torch fallback for the Gated-DeltaNet layers (no fla / causal-conv1d), DynamicCache, Python decode loop with
a host sync per token. Every turn re-prefills the whole conversation.

## Microbench (`micro.json`, job 297002)
Weights 53.8 GB → HBM floor 16.1 ms/token.

| ctx | prefill cold / warm (ms) | decode cold / warm (ms/tok) | roofline (warm) |
|---|---|---|---|
| 512  | 2229 / 371 | 443 / 63.4 | 25.3% |
| 1024 | 1301 / 445 | 438 / 63.0 | 25.5% |
| 2048 | 1572 / 710 | 440 / 63.8 | 25.2% |

"cold" = first time these shapes are seen. In decode the KV length grows by one every step, so in the agent
workload (every turn has a different prompt length) almost every step is cold → e2e decode ≈ the cold number
mixed with the warm one (nsys run: 164 ms/tok).

## Profile (`profile/`, job 297001; 1 task under torch.profiler, 4 tasks under nsys)
- L1 sections (nsys run, 4 tasks): decode 92.7%, prefill 6.9%, tool 0.4%, template/detokenize ≈ 0.
- GPU busy ratio **0.21**: the GPU idles ~79% of the time → host-bound, not bandwidth-bound.
- **`aten::_cudnn_attention_forward`: 31% of all CPU time**, 2.4 ms CPU per call but 13 µs of GPU work
  → cuDNN builds an execution plan for every new (q_len, kv_len) shape.
- ~5.9k kernel launches per generated token (`cudaLaunchKernel` 4.1M calls / 20 s in nsys); GPU time is
  59% GEMMs (nvjet) and the rest thousands of tiny elementwise/reduce kernels from the torch fallback of the
  linear-attention layers (copy_, mul, add, sum, RMSNorm mean/pow/rsqrt, depthwise conv).
- `profile_slurm.log` reports FAILED only because `analyze_nsys.py` crashed on the CUDA-API CSV column name
  (fixed in 4684213; `nsys_summary.txt` regenerated).

## Amdahl → next step
| candidate | p (share of e2e) | expected s | overall |
|---|---|---|---|
| cuDNN SDPA per-shape plan building | 4-task estimate: (164−63)/164 of decode ≈ 0.57 (full run: ≈0.14, see below) | ~∞ on that part | ~2.2× est. / ~1.17× real |
| linear-attn torch fallback (tiny kernels) | large share of the remaining warm 63 ms | ~1.5–2× | later |
| launch/Python overhead (CUDA Graphs) | GPU idle 79% | — | later |

→ **p1_s01**: disable cuDNN SDPA (`torch.backends.cuda.enable_cudnn_sdp(False)`), SDPA falls back to
flash / mem-efficient kernels that need no per-shape plan.

## End-to-end (`metrics.json`, job 297000)
| makespan | accuracy | gen tokens | decode ms/tok | prefill tok/s | p50 / p90 task latency |
|---|---|---|---|---|---|
| **1673.6 s** | 0.930 (119/128; 4 no_answer) | 20,520 | 71.8 | 611 | 9.7 / 17.0 s |

Breakdown: decode 87.0%, prefill 12.3%, tool 0.6%, template/detokenize ≈ 0. 254 turns, 126 tool calls,
126k prompt tokens of which 69k (55%) are re-prefills of the previous turn (phase-2 cache-reuse target).

**Lesson (Amdahl estimate vs reality):** the 4-task nsys run showed 164 ms/tok because almost every KV length was
new. Over 128 tasks the lengths repeat, cuDNN's plan cache warms up, and e2e decode is 71.8 ms/tok — only ~12 ms
above the warm 60 ms. So the cuDNN fix is worth ~(71.8−60)/71.8 × 87% ≈ 14% of e2e (≈1.17×), not the ~2.2×
estimated from the short profile. Profiles of short samples over-weight warmup/cold effects.
