"""Whole-run profile from the decode-step timeline (`steps.jsonl`, written by run_bench for batch engines).

Splits the makespan into time inside `engine.step()` (per bucket; vs the bare graph replay time from
`profile*/batch_step*.json` when present), host time between steps (runner: detokenize, agent logic, template,
submit, tool bookkeeping) and time with no live row; shows how many rows were live over time and what the
last-finishing task (the critical path) did.

    python bench/analyze_steps.py results/<stage>
"""
import collections
import glob
import json
import sys
from pathlib import Path


def main():
    d = Path(sys.argv[1])
    m = json.loads((d / "metrics.json").read_text())
    steps = [json.loads(line) for line in open(d / "steps.jsonl")]
    res = [json.loads(line) for line in open(d / "transcripts.jsonl")]
    turns = [json.loads(line) for line in open(d / "turns.jsonl")]
    T = m["makespan_s"]
    bucket_ms = {}
    for f in glob.glob(str(d / "profile*" / "batch_step*.json")) + glob.glob(str(d / "profile" / "batch_step*.json")):
        bucket_ms.update({int(k): v for k, v in json.loads(Path(f).read_text()).get("bucket_ms", {}).items()})

    in_step = sum(s["t1"] - s["t0"] for s in steps)
    gaps = [b["t0"] - a["t1"] for a, b in zip(steps, steps[1:])]
    head, tail = steps[0]["t0"], T - steps[-1]["t1"]
    print(f"makespan {T:.2f}s, {len(steps)} decode steps ({1e3 * in_step / len(steps):.2f} ms avg inside step())")
    rows = [("inside step()", in_step), ("between steps (host)", sum(g for g in gaps if g < 0.05)),
            ("long gaps >50 ms (no live row / stalls)", sum(g for g in gaps if g >= 0.05)),
            ("before first step", head), ("after last step", tail)]
    print(f"\n{'part':42s} {'s':>7} {'%':>6}")
    for k, v in rows:
        print(f"{k:42s} {v:7.2f} {100 * v / T:5.1f}%")

    print(f"\n{'bucket':>6} {'steps':>6} {'wall_s':>7} {'%mk':>5} {'ms/step':>8} {'replay_ms':>9} {'host_ms':>8} "
          f"{'live/B':>6}")
    by = collections.defaultdict(list)
    for s in steps:
        by[s["bucket"]].append(s)
    per_bucket = {}
    for B in sorted(by):
        ss = by[B]
        w = sum(s["t1"] - s["t0"] for s in ss)
        ms = 1e3 * w / len(ss)
        rep = bucket_ms.get(B)
        fill = sum(s["live"] for s in ss) / len(ss) / B
        per_bucket[B] = {"steps": len(ss), "wall_s": w, "ms_per_step": ms, "replay_ms": rep, "fill": fill}
        print(f"{B:6d} {len(ss):6d} {w:7.2f} {100 * w / T:5.1f} {ms:8.2f} {rep or float('nan'):9.2f} "
              f"{(ms - rep) if rep else float('nan'):8.2f} {fill:6.2f}")

    live_row_steps = sum(s["live"] for s in steps)
    forced = sum(s["forcing"] for s in steps)
    print(f"\nrow-steps: live {live_row_steps}, of which forced (prefill) {forced} ({100 * forced / live_row_steps:.1f}%),"
          f" computed {sum(s['bucket'] for s in steps)} (padding {100 * (1 - live_row_steps / sum(s['bucket'] for s in steps)):.1f}%)")

    print("\ntimeline (1 s bins): steps, mean live rows, mean ms/step")
    bins = collections.defaultdict(list)
    for s in steps:
        bins[int(s["t0"])].append(s)
    for b in range(int(T) + 1):
        ss = bins.get(b, [])
        if ss:
            print(f"  {b:3d}s {len(ss):4d} steps  live {sum(s['live'] for s in ss) / len(ss):5.1f}  "
                  f"{1e3 * sum(s['t1'] - s['t0'] for s in ss) / len(ss):6.2f} ms  "
                  + "#" * round(sum(s["live"] for s in ss) / len(ss)))
        else:
            print(f"  {b:3d}s    0 steps")

    last = max(res, key=lambda r: r["latency_s"])
    lt = [t for t in turns if t["task_id"] == last["task_id"]]
    start = T - last["latency_s"]
    n_before = sum(1 for s in steps if s["t0"] < start)
    print(f"\ncritical path: task {last['task_id']} ({last['finish_reason']}, correct={last['correct']}), "
          f"starts at {start:.2f}s (after {n_before} steps), latency {last['latency_s']:.2f}s, "
          f"{len(lt)} turns, gen {sum(t['gen_len'] for t in lt)} tokens, tool {sum(t['tool_s'] for t in lt):.2f}s")
    lat = sorted((r["latency_s"], r["task_id"]) for r in res)[::-1][:5]
    print("longest tasks:", [(i, round(x, 1)) for x, i in lat])

    out = {"makespan_s": T, "steps": len(steps), "parts": dict(rows), "per_bucket": per_bucket,
           "forced_row_steps": forced, "live_row_steps": live_row_steps,
           "critical_task": last["task_id"], "critical_start_s": start, "critical_latency_s": last["latency_s"]}
    (d / "step_profile.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
