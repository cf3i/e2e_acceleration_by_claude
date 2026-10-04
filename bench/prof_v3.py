"""Kernel-level profile of v3 steps on synthetic batches.

  decode: B sessions decoding with context ~ctx (graph replay)
  prefill: one step with the given chunk lengths (eager), optionally plus decode rows

    python bench/prof_v3.py --case decode --B 128 --ctx 1000
    python bench/prof_v3.py --case prefill --chunks 2048
    python bench/prof_v3.py --case prefill --chunks 256,256,256,256 --B 64
"""
import argparse
import collections
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agentopt.v3.engine import DECODE, V3Engine  # noqa: E402


def kernel_table(prof, steps, top=25):
    agg = collections.defaultdict(lambda: [0.0, 0])
    for e in prof.key_averages():
        if e.device_type.name == "CUDA" or getattr(e, "device_time_total", 0) > 0:
            t = getattr(e, "device_time_total", 0)
            if t > 0 and e.self_device_time_total > 0:
                agg[e.key][0] += e.self_device_time_total
                agg[e.key][1] += e.count
    tot = sum(v[0] for v in agg.values())
    print(f"GPU time per step: {tot / steps / 1e3:.2f} ms")
    for k, (t, n) in sorted(agg.items(), key=lambda x: -x[1][0])[:top]:
        print(f"  {t / steps / 1e3:7.3f} ms  {100 * t / tot:5.1f}%  n/step={n / steps:6.1f}  {k[:110]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="decode")
    ap.add_argument("--B", type=int, default=0)
    ap.add_argument("--ctx", type=int, default=1000)
    ap.add_argument("--chunks", default="2048")
    ap.add_argument("--steps", type=int, default=5)
    args = ap.parse_args()
    eng = V3Engine(num_slots=128, token_budget=8192)
    tok = list(range(1000, 1000 + args.ctx))
    # B decoding sessions at context ~ctx: prefill them (outside the measurement), then keep them decoding
    for i in range(args.B):
        eng.submit(i, tok, 10**6)
    while any(s.phase != DECODE for s in eng.sessions.values()):
        eng.step()
    chunks = [int(c) for c in args.chunks.split(",")] if args.case == "prefill" else []

    def one_step():
        for j, c in enumerate(chunks):
            slot = 127 - j
            eng.release(slot)
            eng.submit(slot, list(range(2000, 2000 + c)), 10**6)
        eng.step()

    for _ in range(3):
        one_step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.steps):
        one_step()
    torch.cuda.synchronize()
    print(f"wall per step: {(time.perf_counter() - t0) / args.steps * 1e3:.2f} ms  ({args.case}, B={args.B}, "
          f"ctx={args.ctx}, chunks={chunks})")
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(args.steps):
            one_step()
        torch.cuda.synchronize()
    kernel_table(prof, args.steps)


if __name__ == "__main__":
    main()
