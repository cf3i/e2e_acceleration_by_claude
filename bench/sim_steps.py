"""Step-level simulator of the continuous runner on a measured run's turn structure.

Replays every task's turns (forced suffix tokens + generated tokens, then the measured tool time) through S
slots: each decode step advances every live slot by one token and costs `step_ms[bucket(#live)] + host_ms`.
Used to predict what slot count / bucket set / admission order would do before building it.

    python bench/sim_steps.py --run p2_s09_recurrent_launch --slots 64 96 128
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_turns(run, shared=352):
    by = {}
    for line in open(ROOT / "results" / run / "turns.jsonl"):
        t = json.loads(line)
        by.setdefault(t["task_id"], []).append(t)
    tasks = {}
    for tid, ts in by.items():
        ts.sort(key=lambda t: t["turn"])
        segs, prev_end = [], shared
        for t in ts:
            forced = max(1, t["prompt_len"] - prev_end)
            segs.append((forced + t["gen_len"], t["tool_s"]))
            prev_end = t["prompt_len"] + t["gen_len"]
        tasks[tid] = segs
    return tasks


def simulate(tasks, order, S, step_ms, host_ms):
    buckets = sorted(step_ms)
    queue = list(order)
    live = {}      # task -> [seg index, tokens left]
    sleeping = {}  # task -> (wake time, next seg index)
    t, steps, hist = 0.0, 0, {}
    while queue or live or sleeping:
        for tid, (wake, i) in list(sleeping.items()):
            if wake <= t:
                del sleeping[tid]
                live[tid] = [i, tasks[tid][i][0]]
        while queue and len(live) + len(sleeping) < S:
            tid = queue.pop(0)
            live[tid] = [0, tasks[tid][0][0]]
        if not live:
            t = min(w for w, _ in sleeping.values())
            continue
        B = next((b for b in buckets if b >= len(live)), None)
        if B is None:  # beyond the measured buckets: extrapolate from the last two
            b1, b2 = buckets[-2:]
            ms = step_ms[b2] + (len(live) - b2) * (step_ms[b2] - step_ms[b1]) / (b2 - b1)
        else:
            ms = step_ms[B]
        hist[B] = hist.get(B, 0) + 1
        t += (ms + host_ms) / 1e3
        steps += 1
        for tid in list(live):
            st = live[tid]
            st[1] -= 1
            if st[1] <= 0:
                del live[tid]
                i = st[0]
                if i + 1 < len(tasks[tid]):
                    sleeping[tid] = (t + tasks[tid][i][1], i + 1)
    return t, steps, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--profile-run", default=None, help="run whose profile/batch_step.json has bucket_ms")
    ap.add_argument("--slots", type=int, nargs="+", default=[64])
    ap.add_argument("--host-ms", type=float, default=None, help="per-step host overhead (default: calibrate)")
    args = ap.parse_args()
    tasks = load_turns(args.run)
    prof = json.load(open(ROOT / "results" / (args.profile_run or args.run) / "profile" / "batch_step.json"))
    step_ms = {int(b): v for b, v in prof["bucket_ms"].items()}
    m = json.load(open(ROOT / "results" / args.run / "metrics.json"))
    order = sorted(tasks)  # data-file order (task ids are file indices)
    chain = {tid: sum(n for n, _ in segs) for tid, segs in tasks.items()}
    print(f"{len(tasks)} tasks; longest chain {max(chain.values())} token-steps "
          f"(task {max(chain, key=chain.get)}); total {sum(chain.values())} row-steps")
    host = args.host_ms
    if host is None:  # match the measured makespan at the measured slot count
        lo, hi = 0.0, 20.0
        for _ in range(40):
            host = (lo + hi) / 2
            if simulate(tasks, order, m["engine_options"]["num_slots"], step_ms, host)[0] < m["makespan_s"]:
                lo = host
            else:
                hi = host
        print(f"calibrated host overhead {host:.2f} ms/step (measured makespan {m['makespan_s']:.2f} s, "
              f"{m['engine_stats']['decode_steps']} steps)")
    for S in args.slots:
        for name, o in [("file order", order), ("longest first (oracle)", sorted(order, key=lambda x: -chain[x]))]:
            t, steps, hist = simulate(tasks, o, S, step_ms, host)
            print(f"S={S:3d} {name:24s}: {t:6.2f} s, {steps} steps, bucket steps {dict(sorted(hist.items(), key=lambda kv: (kv[0] is None, kv[0] or 0)))}")


if __name__ == "__main__":
    main()
