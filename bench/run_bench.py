"""End-to-end benchmark: run the fixed GSM8K agent workload, write metrics/breakdown/transcripts.

    python bench/run_bench.py --stage stage00_baseline --engine naive --runner serial
"""
import argparse
import functools
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agentopt.engines import make_engine  # noqa: E402
from agentopt.runner import RUNNERS  # noqa: E402
from agentopt.timing import Timer  # noqa: E402


def load_tasks(n, exclude=(), data="gsm8k_128"):
    with open(ROOT / "data" / f"{data}.jsonl") as f:
        return [t for t in map(json.loads, f) if t["id"] not in exclude][:n]


def load_replay(run_dir, tok):
    """{(task id, turn): generated token ids} of a finished run (gen_ids, else its transcripts re-tokenized)."""
    run_dir = Path(run_dir)
    turns = [json.loads(l) for l in open(run_dir / "turns.jsonl")]
    if all("gen_ids" in t for t in turns):
        return {(t["task_id"], t["turn"]): t["gen_ids"] for t in turns}
    rep = {}
    for r in map(json.loads, open(run_dir / "transcripts.jsonl")):
        msgs = [m["content"] for m in r["messages"] if m["role"] == "assistant"]
        for i, c in enumerate(msgs, 1):
            rep[(r["task_id"], i)] = tok.encode(c, add_special_tokens=False)
    return rep


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def summarize(results, turns, makespan):
    gen_tokens = sum(t["gen_len"] for t in turns)
    prompt_tokens = sum(t["prompt_len"] for t in turns)
    # Tokens re-prefilled although already processed in the previous turn of the same task
    # (previous prompt + previous generation). Upper bound on what cross-turn cache reuse can save.
    redundant, prev = 0, {}
    for t in turns:
        p = prev.get(t["task_id"])
        if p is not None:
            redundant += min(p["prompt_len"] + p["gen_len"], t["prompt_len"])
        prev[t["task_id"]] = t
    decode_s = sum(t["decode_s"] for t in turns)
    prefill_s = sum(t["prefill_s"] for t in turns)
    lat = [r["latency_s"] for r in results]
    return {
        "makespan_s": makespan,
        "tasks": len(results),
        "accuracy": sum(r["correct"] for r in results) / len(results),
        "tasks_per_min": 60 * len(results) / makespan,
        "turns": len(turns),
        "tool_calls": sum(r["tool_calls"] for r in results),
        "gen_tokens": gen_tokens,
        "prompt_tokens": prompt_tokens,
        "redundant_prefill_tokens": redundant,
        "gen_tok_per_s_e2e": gen_tokens / makespan,
        "decode_ms_per_token": 1000 * decode_s / max(1, gen_tokens - len(turns)),
        "prefill_tok_per_s": prompt_tokens / max(1e-9, prefill_s),
        "task_latency_p50_s": pct(lat, 0.5),
        "task_latency_p90_s": pct(lat, 0.9),
        "finish_reasons": {k: sum(r["finish_reason"] == k for r in results) for k in {r["finish_reason"] for r in results}},
    }


def env_info():
    def sh(cmd):
        try:
            return subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=ROOT).stdout.strip()
        except Exception as e:  # noqa: BLE001
            return str(e)
    rev_file = ROOT / "REVISION"  # written by slurm/submit.sh into code snapshots
    git_commit = rev_file.read_text().strip() if rev_file.exists() else sh("git rev-parse HEAD")
    return {
        "host": os.uname().nodename,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "git_commit": git_commit,
        "snapshot": rev_file.exists(),
        "git_dirty": False if rev_file.exists() else bool(sh("git status --porcelain -- agentopt bench")),
        "argv": sys.argv,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--engine", default="naive")
    ap.add_argument("--runner", default="serial")
    ap.add_argument("--num-tasks", type=int, default=128)
    ap.add_argument("--data", default="gsm8k_128", help="data/<name>.jsonl (gsm8k_longctx_128: long, varied prompts)")
    ap.add_argument("--exclude-tasks", type=int, nargs="*", default=[], help="task ids to drop from the workload")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--max-turns", type=int, default=6)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None, help="default: results/<stage>")
    ap.add_argument("--replay", default=None, help="results dir: force its generated tokens (timing A/Bs that "
                    "follow one trajectory whatever the numerics; v3 runner; accuracy is then that run's)")
    args = ap.parse_args()

    out = Path(args.out or ROOT / "results" / args.stage)
    out.mkdir(parents=True, exist_ok=True)
    tasks = load_tasks(args.num_tasks, set(args.exclude_tasks), args.data)
    runner = RUNNERS[args.runner]

    t0 = time.perf_counter()
    engine = make_engine(args.engine, device=args.device)
    load_s = time.perf_counter() - t0
    print(f"engine loaded in {load_s:.1f}s", flush=True)
    if args.replay:
        runner = functools.partial(runner, replay=load_replay(args.replay, engine.tok))

    # Warmup (not measured): one full task.
    runner(engine, tasks[:1], Timer(), args.max_new_tokens, args.max_turns)

    reps = []
    for r in range(args.repeats):
        timer = Timer()
        t0 = time.perf_counter()
        results, turns = runner(engine, tasks, timer, args.max_new_tokens, args.max_turns)
        makespan = time.perf_counter() - t0
        m = summarize(results, turns, makespan)
        m["breakdown"] = timer.breakdown(makespan)
        if hasattr(engine, "stats"):
            m["engine_stats"] = dict(engine.stats)
        m["trace"] = [(a - t0, b - t0, *rest[:3], *(x - t0 for x in rest[3:]))
                      for a, b, *rest in getattr(engine, "trace", [])]
        if hasattr(engine, "gpu_times"):  # GPU busy time of the steps vs the makespan (the rest: GPU idle)
            g = engine.gpu_times()
            m["gpu_busy_s"] = sum(b - a for a, b in g)
            m["trace"] = [(*t[:5], a, b, *t[5:]) for t, (a, b) in zip(m["trace"], g)]
        reps.append((m, results, turns))
        print(f"[rep {r}] makespan={makespan:.1f}s acc={m['accuracy']:.3f} "
              f"gen_tok={m['gen_tokens']} decode={m['decode_ms_per_token']:.2f}ms/tok", flush=True)

    med_idx = sorted(range(len(reps)), key=lambda i: reps[i][0]["makespan_s"])[len(reps) // 2]
    m, results, turns = reps[med_idx]
    metrics = {
        "stage": args.stage, "engine": args.engine, "engine_options": getattr(engine, "options", {}),
        "runner": args.runner,
        "config": {k: v for k, v in vars(args).items() if k not in ("stage", "out")},
        "load_s": load_s,
        "makespan_s_all": [x[0]["makespan_s"] for x in reps],
        **{k: v for k, v in m.items() if k not in ("breakdown", "trace")},
        "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None,
        "env": env_info(),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "breakdown.json").write_text(json.dumps(m["breakdown"], indent=2))
    with open(out / "turns.jsonl", "w") as f:
        for t in turns:
            f.write(json.dumps(t) + "\n")
    if m["trace"]:  # median rep's decode steps, times relative to the start of the run
        with open(out / "steps.jsonl", "w") as f:
            for t in m["trace"]:
                f.write(json.dumps(dict(zip(("t0", "t1", "bucket", "live", "forcing", "g0", "g1", "t2"), t))) + "\n")
    with open(out / "transcripts.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(json.dumps({k: v for k, v in metrics.items() if k != "env"}, indent=2))
    print("breakdown (median rep):")
    for k, v in m["breakdown"].items():
        print(f"  {k:12s} {v['seconds']:9.2f}s  {100 * v['frac']:5.1f}%  n={v['count']}")


if __name__ == "__main__":
    main()
