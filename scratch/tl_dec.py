"""Timeline of one decode step (graph replay): per-kernel start/end, and how the span splits into GEMM time,
other-kernel time and gaps.   python scratch/tl_dec.py --B 1 --ctx 1000"""
import argparse, collections, sys
sys.path.insert(0, '.')
import torch
from agentopt.v3.engine import DECODE, V3Engine
ap = argparse.ArgumentParser()
ap.add_argument("--B", type=int, default=1); ap.add_argument("--ctx", type=int, default=1000)
ap.add_argument("--layer", type=int, default=20)
a = ap.parse_args()
eng = V3Engine(num_slots=128)
for i in range(a.B):
    eng.submit(i, list(range(1000, 1000 + a.ctx)), 10**6)
while any(s.phase != DECODE for s in eng.sessions.values()):
    eng.step()
for _ in range(5):
    eng.step()
torch.cuda.synchronize()
spans, cats = [], collections.defaultdict(float)
GEMM = ("_mm_fp8", "nvjet", "gemm", "cutlass")
for it in range(5):
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        eng.step(); torch.cuda.synchronize()
    ev = sorted((e for e in prof.events() if e.device_type.name == "CUDA" and e.device_time_total > 0
                 and "Memcpy" not in e.name), key=lambda e: e.time_range.start)
    st, en = ev[0].time_range.start, max(e.time_range.end for e in ev)
    spans.append(en - st)
    # attribute every us of the span to the kernel(s) running then: GEMM if any GEMM runs, else other, else gap
    marks = sorted([(e.time_range.start, 1, any(g in e.name for g in GEMM)) for e in ev] +
                   [(e.time_range.end, -1, any(g in e.name for g in GEMM)) for e in ev])
    ng = no = 0
    last = st
    for t, d, g in marks:
        dt = t - last
        cats["gemm" if ng else "other" if no else "gap"] += dt
        if g:
            ng += d
        else:
            no += d
        last = t
    if it == 4:
        i0 = [i for i, e in enumerate(ev) if "_add_rmsnorm" in e.name][2 * a.layer]
        prev_end = ev[i0 - 1].time_range.end
        for e in ev[i0:i0 + 24]:
            print(f"  start {e.time_range.start - st:9.1f}  dur {e.time_range.end - e.time_range.start:7.1f}  "
                  f"since prev end {e.time_range.start - prev_end:6.1f}  {e.name[:70]}")
            prev_end = e.time_range.end
n = len(spans)
print(f"B={a.B}: span {sum(spans) / n / 1e3:.3f} ms; " + ", ".join(f"{k} {v / n / 1e3:.3f} ms" for k, v in cats.items()))
