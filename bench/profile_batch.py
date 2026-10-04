"""Steady-state profile of one batched decode step (batch engines).

Fills all S slots with real first-turn prompts, times `--steps` uninstrumented steps, then runs `--trace-steps`
more inside a cudaProfilerApi range (for `nsys --capture-range=cudaProfilerApi --cuda-graph-trace=node`).

    python bench/profile_batch.py --stage p2_s03_forced_prefill --engine p2_s03
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agentopt.agent import AgentState  # noqa: E402
from agentopt.engines import make_engine  # noqa: E402
from bench.run_bench import env_info, load_tasks  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--trace-steps", type=int, default=20)
    args = ap.parse_args()
    out = ROOT / "results" / args.stage / "profile"
    out.mkdir(parents=True, exist_ok=True)

    engine = make_engine(args.engine)
    S, tok = engine.num_slots, engine.tok
    tasks = load_tasks(S)
    for s, t in enumerate(tasks):
        engine.prefill(s, tok.encode(AgentState(t).prompt_text(tok), add_special_tokens=False))
    for _ in range(10):
        engine.step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.steps):
        engine.step()
    step_ms = (time.perf_counter() - t0) / args.steps * 1e3
    print(f"{args.engine}: S={S}, step {step_ms:.2f} ms ({step_ms / S:.3f} ms per row)", flush=True)

    torch.cuda.profiler.start()
    for _ in range(args.trace_steps):
        engine.step()
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()
    (out / "batch_step.json").write_text(json.dumps(
        {"engine": args.engine, "options": engine.options, "num_slots": S, "step_ms": step_ms,
         "steps": args.steps, "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9, "env": env_info()}, indent=2))


if __name__ == "__main__":
    main()
