"""Kernel timeline (graph replay of 3 calls) of ovl.py's both(): start/end of kkt, chunk, decode (us from first start).
MODE=after|once|serial   python scratch/ovlprof.py"""
import os, sys, runpy, torch
sys.argv = ["ovl.py"]; sys.path.insert(0, '.')
g = runpy.run_path("scratch/ovl.py")
mode = os.environ.get("MODE", "after")
f = {"after": lambda: g["both"](True), "once": lambda: g["both"](False),
     "serial": lambda: (g["pre"](), g["dec"]())}[mode]
s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    f()
torch.cuda.synchronize()
gr = torch.cuda.CUDAGraph()
with torch.cuda.graph(gr):
    for _ in range(3): f()
for _ in range(3): gr.replay()
torch.cuda.synchronize()
for it in range(3):
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        gr.replay(); torch.cuda.synchronize()
    ev = sorted([e for e in prof.events() if e.device_type.name == "CUDA" and "Memcpy" not in e.name and "Memset" not in e.name],
                key=lambda e: e.time_range.start)
    t0 = ev[0].time_range.start
    print(mode, " | ".join(f"{e.name[:18]} {e.time_range.start - t0:.0f}-{e.time_range.end - t0:.0f}" for e in ev))
