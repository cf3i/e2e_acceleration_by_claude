"""End-to-end benchmark: run the fixed GSM8K agent workload, write metrics/breakdown/transcripts.

    python bench/run_bench.py --stage stage00_baseline --engine naive --runner serial
"""
import argparse
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


def load_tasks(n):
    with open(ROOT / "data/gsm8k_128.jsonl") as f:
        return [json.loads(line) for line in f][:n]


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
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--max-turns", type=int, default=6)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None, help="default: results/<stage>")
    args = ap.parse_args()

    out = Path(args.out or ROOT / "results" / args.stage)
    out.mkdir(parents=True, exist_ok=True)
    tasks = load_tasks(args.num_tasks)
    runner = RUNNERS[args.runner]

    t0 = time.perf_counter()
    engine = make_engine(args.engine, device=args.device)
    load_s = time.perf_counter() - t0
    print(f"engine loaded in {load_s:.1f}s", flush=True)

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
        **{k: v for k, v in m.items() if k != "breakdown"},
        "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None,
        "env": env_info(),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "breakdown.json").write_text(json.dumps(m["breakdown"], indent=2))
    with open(out / "turns.jsonl", "w") as f:
        for t in turns:
            f.write(json.dumps(t) + "\n")
    with open(out / "transcripts.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(json.dumps({k: v for k, v in metrics.items() if k != "env"}, indent=2))
    print("breakdown (median rep):")
    for k, v in m["breakdown"].items():
        print(f"  {k:12s} {v['seconds']:9.2f}s  {100 * v['frac']:5.1f}%  n={v['count']}")


if __name__ == "__main__":
    main()
