"""CUTLASS sm90 FP8 GEMM schedules vs cuBLAS at the gate_up / down / qkv shapes of a 2048-row mixed step,
weight copies cycled (> L2) as in the model.   CUBLASLT_WORKSPACE_SIZE=0 python scratch/cutmm/bench.py"""
import sys, os, torch
sys.path.insert(0, '.'); sys.path.insert(0, 'scratch'); sys.path.insert(0, 'scratch/cutmm')
from benchutil import bench_loop
from build import build
cm = build()
F8 = torch.float8_e4m3fn
one = torch.ones((), device='cuda')
names = [n for n in dir(cm) if n.startswith(("coop", "ping"))]
M = int(os.environ.get("M", 2048))
for shp, N, Kd in [("gate_up", 34816, 5120), ("down", 5120, 17408)]:
    ws = [(torch.randn(N, Kd, device='cuda') * .5).to(F8) for _ in range(max(2, int(1.0e9 // (N * Kd))))]
    x = torch.randn(M, Kd, device='cuda').to(F8)
    y = torch.empty(M, N, device='cuda', dtype=torch.bfloat16)
    s0 = torch.tensor(0.01, device='cuda')
    fl = 2 * M * N * Kd
    ref = (x[:256].float() @ ws[-1].float().t()) * 0.01
    tcb = bench_loop([lambda w=w: torch._scaled_mm(x, w.t(), one, s0, out_dtype=torch.bfloat16) for w in ws], iters=8, reps=3)
    yc = torch._scaled_mm(x, ws[-1].t(), one, s0, out_dtype=torch.bfloat16)
    ec = ((yc[:256].float() - ref).abs().max() / ref.abs().max()).item()
    print(f"{shp} M={M}: cublas {tcb * 1e6:7.1f} us {fl / tcb / 1e12:5.0f} TFLOPS err {ec:.1e}", flush=True)
    for n in names:
        for sw, ro in [(1, 0), (2, 0), (4, 0), (8, 1), (8, 2)]:
            f = getattr(cm, n)
            t = bench_loop([lambda w=w: f(x, w, y, 0.01, sw, ro) for w in ws], iters=8, reps=3)
            f(x, ws[-1], y, 0.01, sw, ro)
            err = ((y[:256].float() - ref).abs().max() / ref.abs().max()).item()
            print(f"  {n:24s} sw {sw} ro {ro}: {t * 1e6:7.1f} us {fl / t / 1e12:5.0f} TFLOPS err {err:.1e}", flush=True)
