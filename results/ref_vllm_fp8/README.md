# Reference: vLLM 0.30 on the same agent workload

This is not a stage. It checks where the hand-built engine (p2_s10) stands against a production server.
- `bench/run_vllm.py` runs one AsyncLLM in its own venv (torch 2.13+cu129, because the 570 driver cannot run cu130).
- Each of the 128 agents is an asyncio task: greedy decoding, max 512 tokens per turn, stop at `<|im_end|>`. Tools run in threads.
- AgentState, tools, scoring, the warmup task and the output files are the same as `run_bench.py`.
- vLLM defaults:
  - continuous batching with chunked prefill;
  - prefix caching, in hybrid 'align' mode: KV blocks plus Gated-DeltaNet state snapshots at block boundaries;
  - FULL_AND_PIECEWISE CUDA graphs, torch.compile, FlashInfer GDN prefill, Triton GDN decode;
  - `max_num_seqs` = 128, so every agent is admitted at once.
- `--quantization fp8` = dynamic per-tensor FP8 linears; the same scheme as our `fp8="tensor"`.
- The prefix cache is reset before every repeat.
- Our side is `ref_ours_p2s10_r3` (preset p2_s10) and `ref_ours_bf16_r3` (preset p2_s10_bf16, i.e. p2_s10 without FP8). Each ran 3 repeats in one process, like vLLM.

| | makespan, 3 repeats (s) | median | accuracy | task latency p50 / p90 (s) | load (s) |
|---|---|---|---|---|---|
| **ours fp8** (p2_s10) | 18.6 / 20.4 / 20.3 | **20.3** | 0.930 | **5.8 / 9.6** | 580 |
| vLLM fp8 | 19.9 / 20.7 / 20.6 | 20.6 | 0.938 | 10.4 / 14.1 | 223 |
| **ours bf16** (p2_s10_bf16) | 28.2 / 28.2 / 28.2 | **28.2** | 0.938 | 7.4 / 12.7 | 274 |
| vLLM bf16 | 29.1 / 28.8 / 28.9 | 28.9 | 0.938 | 13.5 / 17.7 | 123 |

## Reading it
- **Both systems are bound by the same critical path.** Task 423 is the last to finish in every run, so the makespan is roughly that task's latency.
- Its chain length depends on numerics. In the median fp8 repeats both systems happen to generate exactly 751 tokens for it, so time per chain token compares directly:

  | precision | ours | vLLM |
  |---|---|---|
  | fp8 | 26.5 ms | 27.4 ms |
  | bf16 | 34.5 ms (810 tokens) | 36.1 ms (798 tokens) |

  Ours is about 3–4% faster per token on the critical path. Our fp8 run-to-run spread (18.6–20.4 s) is larger than that difference. The likely cause is per-tensor FP8 scales coupling the rows of a batch (p2_s07 README), so the chain changes when the batch mix changes.
- **Same ingredients.** vLLM independently uses what the ladder found:
  - continuous batching;
  - hybrid prefix reuse — state snapshots at block boundaries, versus ours at the slot end plus the shared prefix;
  - CUDA graphs per batch size (35 sizes vs our 6);
  - compiled elementwise fusion;
  - FP8 per-tensor.

  At the same quantization the step costs end up within a few percent.
- **Latency vs throughput.** vLLM admits all 128 agents at once. Big early batches with prefill chunks inside raise per-token latency, so its task p50 is 10.4 s. Ours caps admission at 64 slots, giving a lower p50 (5.8 s). The makespan is the same because the late-starting long chain still finishes in time.
- **Load time.** Ours is 580 s with FP8: 6 compiled bucket graphs, extend-prefill JIT and torchao quantization, without a persistent compile cache. vLLM takes 223 s, or 123 s in bf16, and caches its compilation artifacts. This is a real gap for a server that restarts, but outside the makespan metric.
- What the hand-built engine does not have: a general scheduler (preemption, admission by KV budget), arbitrary-length prefill, multi-GPU, an API server. What vLLM does not have: our workload-specific shortcuts (forced suffix tokens through the decode graph, suspend/resume of slot states during tool calls). Neither moves the makespan much here, because the critical path dominates.
