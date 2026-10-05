"""Critical-path attribution of one mixed step: every us of the GPU span goes to the kernels running then
(split evenly among concurrent ones), plus gaps.   python scratch/tl_mixed.py --chunks 1992 --pos0 3000 --B 56"""
import argparse, collections, sys
sys.path.insert(0, '.')
import torch
from agentopt.v3.engine import DECODE, IDLE, V3Engine
ap = argparse.ArgumentParser()
ap.add_argument("--B", type=int, default=56); ap.add_argument("--ctx", type=int, default=4000)
ap.add_argument("--chunks", default="1992"); ap.add_argument("--pos0", type=int, default=3000)
ap.add_argument("--steps", type=int, default=3)
a = ap.parse_args()
eng = V3Engine(num_slots=128)
for i in range(a.B):
    eng.submit(i, list(range(1000, 1000 + a.ctx)), 10**6)
while any(s.phase != DECODE for s in eng.sessions.values()):
    eng.step()
chunks = [int(c) for c in a.chunks.split(",")]
slots = [127 - j for j in range(len(chunks))]
def prep():
    for s in slots:
        eng.release(s)
        if a.pos0:
            eng.submit(s, list(range(3000, 3000 + a.pos0)), 0)
    while any(eng.sessions[s].phase != IDLE for s in slots if s in eng.sessions):
        eng.step()
    eng.drain()
    for s, c in zip(slots, chunks):
        h = eng.sessions[s].hist if s in eng.sessions else []
        eng.submit(s, h + list(range(2000, 2000 + c)), 0)
    torch.cuda.synchronize()
for _ in range(3):
    prep(); eng.step(); torch.cuda.synchronize()
def short(n):
    if n.startswith("nvjet") or "gemm" in n: return "cublas_gemm"
    if "FlashAttn" in n: return "fa3"
    if "flash::" in n: return "fa3_prep"
    return n[:40]
cat = collections.defaultdict(float); spans = []
for it in range(a.steps):
    prep()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        eng.step(); torch.cuda.synchronize()
    ev = [e for e in prof.events() if e.device_type.name == "CUDA" and e.device_time_total > 0 and "Memcpy" not in e.name]
    st = min(e.time_range.start for e in ev); en = max(e.time_range.end for e in ev); spans.append(en - st)
    marks = sorted([(e.time_range.start, 1, short(e.name)) for e in ev] + [(e.time_range.end, -1, short(e.name)) for e in ev])
    run = collections.Counter(); last = st
    for t, d, n in marks:
        dt = t - last
        act = [k for k, c in run.items() if c > 0]
        if not act: cat["(gap)"] += dt
        else:
            for k in act: cat[k] += dt / len(act)
        run[n] += d; last = t
n = a.steps
print(f"span {sum(spans) / n / 1e3:.2f} ms")
for k, v in sorted(cat.items(), key=lambda x: -x[1]):
    if v / n > 20: print(f"  {v / n / 1e3:7.3f} ms  {k}")
# gaps: which (prev kernel ending, next kernel starting) pairs surround idle time (last step)
ev.sort(key=lambda e: e.time_range.start)
gp = collections.defaultdict(lambda: [0.0, 0]); cur_end = ev[0].time_range.end; prev = short(ev[0].name)
for e in ev[1:]:
    if e.time_range.start > cur_end:
        k = (prev, short(e.name)); gp[k][0] += e.time_range.start - cur_end; gp[k][1] += 1
    if e.time_range.end > cur_end:
        cur_end = e.time_range.end; prev = short(e.name)
print("gaps (last step):")
for k, (v, c) in sorted(gp.items(), key=lambda x: -x[1][0])[:15]:
    print(f"  {v / 1e3:6.3f} ms  n {c:4d}  {k[0]} -> {k[1]}")
