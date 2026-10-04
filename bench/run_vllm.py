"""Reference point: the same 128-task GSM8K agent workload served by vLLM (not part of the optimization ladder).

Every agent is an asyncio task on one AsyncLLM: generate a turn (greedy, max 512 tokens, stop at <|im_end|>),
run the tool in a thread, submit the next turn. vLLM does the continuous batching, prefix caching, CUDA graphs
and (optionally) FP8. Same AgentState / tools / scoring / output files as `run_bench.py`, so `report.py` and the
comparison tables read it like any stage. Runs in its own venv (vLLM pins a different torch):

    python bench/run_vllm.py --stage ref_vllm_bf16
    python bench/run_vllm.py --stage ref_vllm_fp8 --quantization fp8
"""
import argparse
import asyncio
import json
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agentopt.agent import AgentState  # noqa: E402
from agentopt.tools import execute  # noqa: E402
from bench.run_bench import env_info, load_tasks, summarize  # noqa: E402

MODEL = "/mnt/data/u/feic/pjs/model/Qwen3.5-27B"


async def run_agents(llm, tok, tasks, sp, max_turns):
    stop_ids = set(sp.stop_token_ids)
    results, turns = [], []

    async def generate(ids):
        t0, t_first, out = time.perf_counter(), None, []
        async for o in llm.generate({"prompt_token_ids": ids}, sp, str(uuid.uuid4())):
            if t_first is None:
                t_first = time.perf_counter()
            out.extend(o.outputs[0].token_ids)  # DELTA outputs
        stopped = bool(out) and out[-1] in stop_ids
        return (out[:-1] if stopped else out), t_first - t0, time.perf_counter() - t_first, stopped

    async def agent(task):
        st, t_task = AgentState(task, max_turns=max_turns), time.perf_counter()
        while not st.done:
            ids = tok.encode(st.prompt_text(tok), add_special_tokens=False)
            gen, ttft, dec, stopped = await generate(ids)
            call = st.observe(tok.decode(gen, skip_special_tokens=False))
            tool_s = 0.0
            if call is not None:
                t0 = time.perf_counter()
                st.add_tool_result(await asyncio.to_thread(execute, *call))
                tool_s = time.perf_counter() - t0
            turns.append({"task_id": task["id"], "turn": st.turns, "prompt_len": len(ids), "gen_len": len(gen),
                          "prefill_s": ttft, "decode_s": dec, "tool_s": tool_s, "stopped": stopped})
        results.append({
            "task_id": task["id"], "gold": task["answer"], "pred": st.answer, "correct": st.correct,
            "turns": st.turns, "tool_calls": st.tool_calls, "finish_reason": st.finish_reason,
            "latency_s": time.perf_counter() - t_task, "messages": st.messages,
        })

    await asyncio.gather(*(agent(t) for t in tasks))
    order = {t["id"]: i for i, t in enumerate(tasks)}
    results.sort(key=lambda r: order[r["task_id"]])
    return results, turns


async def main_async(args):
    import torch
    import vllm
    from vllm import AsyncEngineArgs, SamplingParams
    from vllm.sampling_params import RequestOutputKind
    from transformers import AutoTokenizer
    from vllm.v1.engine.async_llm import AsyncLLM

    out = Path(args.out or ROOT / "results" / args.stage)
    out.mkdir(parents=True, exist_ok=True)
    tasks = load_tasks(args.num_tasks, set(args.exclude_tasks), args.data)
    eargs = dict(model=MODEL, dtype="bfloat16", max_model_len=args.max_model_len,
                 gpu_memory_utilization=args.gpu_mem, enable_prefix_caching=args.prefix_caching,
                 max_num_seqs=args.max_num_seqs, limit_mm_per_prompt={"image": 0, "video": 0},
                 quantization=args.quantization, seed=0)
    t0 = time.perf_counter()
    llm = AsyncLLM.from_engine_args(AsyncEngineArgs(**eargs))
    load_s = time.perf_counter() - t0
    print(f"vLLM {vllm.__version__} loaded in {load_s:.1f}s", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)  # same tokenizer / chat template as the main engines
    stop = [tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")]
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens, stop_token_ids=stop,
                        skip_special_tokens=False, detokenize=False, output_kind=RequestOutputKind.DELTA)

    await run_agents(llm, tok, tasks[:1], sp, args.max_turns)  # warmup, not measured
    reps = []
    for r in range(args.repeats):
        await llm.reset_prefix_cache()  # no cross-repeat hits; within a run the cache works as in serving
        t0 = time.perf_counter()
        results, turns = await run_agents(llm, tok, tasks, sp, args.max_turns)
        makespan = time.perf_counter() - t0
        m = summarize(results, turns, makespan)
        reps.append((m, results, turns))
        print(f"[rep {r}] makespan={makespan:.1f}s acc={m['accuracy']:.3f} gen_tok={m['gen_tokens']}", flush=True)

    med = sorted(range(len(reps)), key=lambda i: reps[i][0]["makespan_s"])[len(reps) // 2]
    m, results, turns = reps[med]
    # Agents overlap, so these are agent-seconds (sum over turns), not a partition of the makespan.
    agent_s = {k: sum(t[f"{k}_s"] for t in turns) for k in ("prefill", "decode", "tool")}
    metrics = {
        "stage": args.stage, "engine": "vllm", "engine_options": {**eargs, "vllm": vllm.__version__},
        "runner": "vllm_async", "config": {k: v for k, v in vars(args).items() if k not in ("stage", "out")},
        "load_s": load_s, "makespan_s_all": [x[0]["makespan_s"] for x in reps], **m,
        "agent_seconds": agent_s, "peak_mem_gb": None, "env": {**env_info(), "torch": torch.__version__},
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "breakdown.json").write_text(json.dumps(
        {k: {"seconds": v, "frac": v / m["makespan_s"], "count": len(turns)} for k, v in agent_s.items()}, indent=2))
    with open(out / "turns.jsonl", "w") as f:
        f.writelines(json.dumps(t) + "\n" for t in turns)
    with open(out / "transcripts.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in results)
    print(json.dumps({k: v for k, v in metrics.items() if k != "env"}, indent=2), flush=True)
    llm.shutdown()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--num-tasks", type=int, default=128)
    ap.add_argument("--data", default="gsm8k_128")
    ap.add_argument("--exclude-tasks", type=int, nargs="*", default=[])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--max-turns", type=int, default=6)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--max-num-seqs", type=int, default=128)
    ap.add_argument("--gpu-mem", type=float, default=0.92)
    ap.add_argument("--quantization", default=None, help="e.g. fp8 (dynamic per-tensor FP8 weights)")
    ap.add_argument("--no-prefix-caching", dest="prefix_caching", action="store_false")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
