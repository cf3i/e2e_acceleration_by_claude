"""Correctness guardrail for engine changes: greedy outputs of candidate presets vs. a reference preset.

Runs the first-turn prompts of N tasks through each engine (one engine in memory at a time) and reports, per
candidate, how many outputs are token-identical and the first divergence position otherwise.

    python bench/check_equiv.py --stage p1_s04_cuda_graph --ref p1_s02 --cand p1_s03,p1_s04
"""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agentopt.agent import AgentState  # noqa: E402
from agentopt.engines import make_engine  # noqa: E402
from agentopt.timing import Timer  # noqa: E402
from bench.run_bench import env_info, load_tasks  # noqa: E402


def run(preset, prompts, max_new):
    engine = make_engine(preset)
    outs, t0 = [], time.perf_counter()
    for ids in prompts:
        gen, _ = engine.generate(ids, max_new, Timer())
        outs.append(gen)
    dt = time.perf_counter() - t0
    tok = engine.tok
    del engine
    gc.collect()
    torch.cuda.empty_cache()
    return outs, dt, tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--cand", required=True)
    ap.add_argument("--num-tasks", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from agentopt import MODEL_PATH
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    prompts = [tok.encode(AgentState(t).prompt_text(tok), add_special_tokens=False)
               for t in load_tasks(args.num_tasks)]

    ref, ref_s, _ = run(args.ref, prompts, args.max_new_tokens)
    report = {"ref": args.ref, "ref_seconds": ref_s, "num_prompts": len(prompts), "candidates": {}, "env": env_info()}
    for c in args.cand.split(","):
        outs, s, _ = run(c, prompts, args.max_new_tokens)
        rows = []
        for a, b in zip(ref, outs):
            div = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
            if div is None and len(a) != len(b):
                div = min(len(a), len(b))
            rows.append({"identical": div is None, "first_divergence": div, "ref_len": len(a), "cand_len": len(b)})
        report["candidates"][c] = {"seconds": s, "identical": sum(r["identical"] for r in rows), "rows": rows}
        print(f"{c}: {sum(r['identical'] for r in rows)}/{len(rows)} identical to {args.ref}, "
              f"{s:.1f}s vs ref {ref_s:.1f}s; divergences {[r['first_divergence'] for r in rows]}", flush=True)
        for r, a, b in zip(rows, ref, outs):
            if not r["identical"]:
                d = r["first_divergence"]
                print(f"  ref : ...{tok.decode(a[max(0, d - 10):d + 20])!r}\n  cand: ...{tok.decode(b[max(0, d - 10):d + 20])!r}")
    out = ROOT / "results" / args.stage
    out.mkdir(parents=True, exist_ok=True)
    (out / "equiv.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
