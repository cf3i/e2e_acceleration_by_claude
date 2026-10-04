# PLAN: profile-driven optimization of a small LLM agent on 1×H100

## Scenario (fixed for all stages)
- Offline batch math agent: 128 fixed GSM8K test questions (`data/gsm8k_128.jsonl`, seed 0), all available at t=0.
- ReAct-style agent with Qwen native tool-call format, one `python` tool (stdout returned); 27B uses the tool on ~97% of tasks unprompted.
- Model: Qwen3.5-27B (hybrid: 48 Gated-DeltaNet linear-attn layers + 16 full-attn GQA layers, 248k vocab), bf16 ≈ 54 GB.
  (Switched from Qwen3.5-2B: it rarely used tools and fell into greedy repetition loops.)
- Non-thinking mode, greedy decoding, ≤6 turns/task, ≤512 new tokens/turn.

## Metrics
- Primary: makespan (wall time to finish all 128 tasks).
- Guardrail: accuracy must stay within 2 points of baseline; for output-preserving optimizations also track output agreement.
- Secondary: gen tok/s, decode ms/token, task latency p50/p90, GPU busy ratio, peak memory.

## Loop
1. profile (`slurm/profile.sbatch`): L1 section timers, L2 torch.profiler, L3 nsys (+ NVTX).
2. Amdahl breakdown → for each candidate estimate p (fraction) and s (local speedup): 1/((1-p)+p/s).
3. Implement ONE change. 4. bench (`slurm/bench.sbatch`). 5. Record in `results/stageNN_*/` and `OPTIMIZATION_LOG.md`, git tag.

Rules: profile and bench are separate runs; warmup; repeats=3 once a run is <10 min; partition freecycle-h100 (qos freecycle, preemptible → --requeue).
Scope: no custom kernels; existing libraries and framework/system-level changes are allowed.

## Stop criteria
3 consecutive stages <5% gain, or remaining bottleneck needs custom kernels, or close to a vLLM reference.
