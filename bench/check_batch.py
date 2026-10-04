"""Slot-isolation check for batch engines: concurrent decoding must not change any sequence's output.

The decode graph always computes all S rows and rows never mix (GEMM rows, per-row attention / states), so a
prompt decoded alone on slot 0 and the same prompt decoded on another slot while other sequences join and leave
at different steps must give bit-identical tokens. Any difference = cross-talk (positions, masks, states).

With prefix reuse it also checks the two reuse paths against a full prefill (same tokens expected, up to
float differences from processing the prompt in two chunks):
  shared: fresh prefill on top of the restored shared-prefix snapshot;
  extend: prefill ids[:-17] first, then the full prompt extends the slot by 17 tokens.
With forced-token prefill (`force_max`) both reuse paths feed the suffix through the decode graph instead, and
the alone / concurrent runs start turns with `submit` (so forced rows mix with generating rows).

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

    def gen_submit(slot, ids):
        first = engine.submit(slot, ids)
        gen = [] if first is None else [first]
        while not gen or (gen[-1] not in engine.stop_ids and len(gen) < args.max_new_tokens):
            out = engine.step()
            if slot in out:
                gen.append(out[slot])
        engine.release(slot)
        return gen[:-1] if gen[-1] in engine.stop_ids else gen

    forced = engine.options.get("force_max", 0) > 0

    def gen_all(pre=None):
        outs = []
        for ids in prompts:
            if pre is not None:
                engine.prefill(0, pre(ids))
            if forced:
                outs.append(gen_submit(0, ids))
            else:
                outs.append(engine.generate(ids, args.max_new_tokens, Timer())[0])
        return outs

    def compare(a_list, b_list):
        rows = []
        for a, b in zip(a_list, b_list):
            div = next((k for k, (x, y) in enumerate(zip(a, b)) if x != y), None)
            if div is None and len(a) != len(b):
                div = min(len(a), len(b))
            rows.append({"identical": a == b, "first_divergence": div, "len_a": len(a), "len_b": len(b)})
        return rows

    report = {"engine": args.engine}
    reuse = engine.options.get("prefix_reuse", False)
    if reuse:
        engine.prefix_reuse, engine.force_max = False, 0
        full = gen_all()  # eager full prefill
        engine.prefix_reuse, engine.force_max = True, engine.options.get("force_max", 0)
        n = next((i for i, xs in enumerate(zip(*prompts)) if len(set(xs)) > 1), min(map(len, prompts)) - 1)
        engine.set_shared_prefix(prompts[0][:n])
        for name, pre in [("extend", lambda ids: ids[:-17]), ("shared", None)]:
            outs = gen_all(pre)
            rows = compare(full, outs)
            report[name] = rows
            print(f"{name} vs full prefill: {sum(r['identical'] for r in rows)}/{len(rows)} identical; "
                  f"divergences {[r['first_divergence'] for r in rows]}  (shared prefix {n} tokens)")
    t0 = time.perf_counter()
    alone = gen_all()
    alone_s = time.perf_counter() - t0

    # Concurrent: prompt i joins at step i * stagger on slot S-1-i (not slot 0), rows leave when done.
    S = engine.num_slots
    pending = list(enumerate(prompts))
    live, outs, step = {}, [None] * len(prompts), 0
    t0 = time.perf_counter()
    while pending or live:
        while pending and pending[0][0] * args.stagger <= step:
            i, ids = pending.pop(0)
            first = engine.submit(S - 1 - i, ids)
            live[S - 1 - i] = (i, [] if first is None else [first])
        if live:
            out = engine.step()
            for slot in list(live):
                i, gen = live[slot]
                if slot not in out:
                    continue
                gen.append(out[slot])
                if gen[-1] in engine.stop_ids or len(gen) >= args.max_new_tokens:
                    outs[i] = gen[:-1] if gen[-1] in engine.stop_ids else gen
                    engine.release(slot)
                    del live[slot]
        step += 1
    conc_s = time.perf_counter() - t0

    rows = compare(alone, outs)
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
        {**report, "identical": n_id, "rows": rows, "alone_s": alone_s, "concurrent_s": conc_s,
         "steps": step, "alone_tokens": alone, "env": env_info()}, indent=2))


if __name__ == "__main__":
    main()
