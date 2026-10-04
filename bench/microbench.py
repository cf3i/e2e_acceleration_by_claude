"""Phase-1 engine microbenchmark: single-request prefill / decode speed at agent-like context lengths.

Reports both "cold" (first time the engine sees these shapes) and "warm" (median of later reps),
and decode efficiency vs. the HBM-bandwidth roofline (every decode step must read all weights once).

    python bench/microbench.py --stage p1_stage00_baseline --engine naive
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agentopt.agent import AgentState  # noqa: E402
from agentopt.engines import make_engine  # noqa: E402
from agentopt.timing import Timer  # noqa: E402
from bench.run_bench import env_info, load_tasks  # noqa: E402

H100_HBM_BW = 3.35e12  # bytes/s, H100 SXM


def make_prompt(tok, length: int) -> list[int]:
    """Real agent prompts (system + tools + question) concatenated, truncated to `length` tokens."""
    ids = []
    for task in load_tasks(128):
        ids += tok.encode(AgentState(task).prompt_text(tok), add_special_tokens=False)
        if len(ids) >= length:
            return ids[:length]
    raise ValueError(f"not enough prompt tokens for length {length}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--engine", default="naive")
    ap.add_argument("--ctx", default="512,1024,2048")
    ap.add_argument("--decode-tokens", type=int, default=128)
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()
    out = ROOT / "results" / args.stage
    out.mkdir(parents=True, exist_ok=True)

    engine = make_engine(args.engine)
    weight_bytes = sum(p.numel() * p.element_size() for p in engine.model.parameters())
    floor_ms = 1000 * weight_bytes / H100_HBM_BW

    rows = []
    for L in map(int, args.ctx.split(",")):
        ids = make_prompt(engine.tok, L)
        runs = []
        for _ in range(1 + args.reps):  # first run = cold
            gen, st = engine.generate(ids, args.decode_tokens, Timer(), ignore_eos=True)
            runs.append({"prefill_ms": 1000 * st["prefill_s"],
                         "decode_ms_per_tok": 1000 * st["decode_s"] / max(1, len(gen) - 1)})
        warm = runs[1:]
        row = {
            "ctx": L,
            "prefill_ms_cold": runs[0]["prefill_ms"],
            "prefill_ms_warm": statistics.median(r["prefill_ms"] for r in warm),
            "decode_ms_per_tok_cold": runs[0]["decode_ms_per_tok"],
            "decode_ms_per_tok_warm": statistics.median(r["decode_ms_per_tok"] for r in warm),
        }
        row["prefill_tok_per_s_warm"] = L / (row["prefill_ms_warm"] / 1000)
        row["decode_bw_roofline_frac"] = floor_ms / row["decode_ms_per_tok_warm"]
        rows.append(row)
        print(json.dumps(row), flush=True)

    result = {
        "stage": args.stage, "engine": args.engine, "decode_tokens": args.decode_tokens, "reps": args.reps,
        "weight_gb": weight_bytes / 1e9, "decode_floor_ms_per_tok": floor_ms,
        "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30,
        "rows": rows, "env": env_info(),
    }
    (out / "micro.json").write_text(json.dumps(result, indent=2))
    print(f"weights {result['weight_gb']:.1f} GB -> bandwidth floor {floor_ms:.1f} ms/token")
    print(f"{'ctx':>6} {'prefill cold/warm ms':>22} {'decode cold/warm ms/tok':>25} {'roofline%':>10}")
    for r in rows:
        print(f"{r['ctx']:6d} {r['prefill_ms_cold']:10.1f} /{r['prefill_ms_warm']:9.1f} "
              f"{r['decode_ms_per_tok_cold']:12.1f} /{r['decode_ms_per_tok_warm']:10.1f} "
              f"{100 * r['decode_bw_roofline_frac']:9.1f}%")


if __name__ == "__main__":
    main()
