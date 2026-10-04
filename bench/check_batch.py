"""Slot-isolation check for batch engines: concurrent decoding must not change any sequence's output.

The decode graph always computes all S rows and rows never mix (GEMM rows, per-row attention / states), so a
prompt decoded alone on slot 0 and the same prompt decoded on another slot while other sequences join and leave
at different steps must give bit-identical tokens. Any difference = cross-talk (positions, masks, states).

    python bench/check_batch.py --stage p2_s01_continuous_batching --engine p2_s01
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agentopt.agent import AgentState  # noqa: E402
from agentopt.engines import make_engine  # noqa: E402
from agentopt.timing import Timer  # noqa: E402
from bench.run_bench import env_info, load_tasks  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--num-tasks", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--stagger", type=int, default=5, help="decode steps between admissions")
    args = ap.parse_args()

    engine = make_engine(args.engine)
    tok = engine.tok
    prompts = [tok.encode(AgentState(t).prompt_text(tok), add_special_tokens=False)
               for t in load_tasks(args.num_tasks)]

    t0 = time.perf_counter()
    alone = [engine.generate(ids, args.max_new_tokens, Timer())[0] for ids in prompts]
    alone_s = time.perf_counter() - t0

    # Concurrent: prompt i joins at step i * stagger on slot S-1-i (not slot 0), rows leave when done.
    S = engine.num_slots
    pending = list(enumerate(prompts))
    live, outs, step = {}, [None] * len(prompts), 0
    t0 = time.perf_counter()
    while pending or live:
        while pending and pending[0][0] * args.stagger <= step:
            i, ids = pending.pop(0)
            live[S - 1 - i] = (i, [engine.prefill(S - 1 - i, ids)])
        if live:
            out = engine.step()
            for slot in list(live):
                i, gen = live[slot]
                gen.append(out[slot])
                if gen[-1] in engine.stop_ids or len(gen) >= args.max_new_tokens:
                    outs[i] = gen[:-1] if gen[-1] in engine.stop_ids else gen
                    engine.release(slot)
                    del live[slot]
        step += 1
    conc_s = time.perf_counter() - t0

    rows = []
    for a, b in zip(alone, outs):
        div = next((k for k, (x, y) in enumerate(zip(a, b)) if x != y), None)
        if div is None and len(a) != len(b):
            div = min(len(a), len(b))
        rows.append({"identical": a == b, "first_divergence": div, "alone_len": len(a), "concurrent_len": len(b)})
    n_id = sum(r["identical"] for r in rows)
    print(f"{args.engine}: {n_id}/{len(rows)} identical alone vs concurrent; alone {alone_s:.1f}s, "
          f"concurrent {conc_s:.1f}s ({step} steps); divergences {[r['first_divergence'] for r in rows]}")
    for r, a, b in zip(rows, alone, outs):
        if not r["identical"]:
            k = r["first_divergence"]
            print("  alone:", repr(tok.decode(a[max(0, k - 10):k + 10])))
            print("  conc :", repr(tok.decode(b[max(0, k - 10):k + 10])))
    out = ROOT / "results" / args.stage
    out.mkdir(parents=True, exist_ok=True)
    (out / "batch_check.json").write_text(json.dumps(
        {"engine": args.engine, "identical": n_id, "rows": rows, "alone_s": alone_s, "concurrent_s": conc_s,
         "steps": step, "alone_tokens": alone, "env": env_info()}, indent=2))


if __name__ == "__main__":
    main()
