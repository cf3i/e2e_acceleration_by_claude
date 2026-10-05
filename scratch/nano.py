"""Nano-batching potential: does a bandwidth-bound kernel (silu_mul_quant / add_rmsnorm_quant) hide under a
compute-bound GEMM on another stream?  t(G), t(S), t(G || S) in one CUDA graph.   python scratch/nano.py"""
import sys, torch
sys.path.insert(0, '.')
from agentopt.v3 import kernels as K
F8 = torch.float8_e4m3fn
one = torch.ones((), device='cuda')


def graph_time(streams_fns, iters=12, reps=5):
    """streams_fns: list of lists of fns, one list per stream, each list cycled iters times."""
    ss = [torch.cuda.Stream() for _ in streams_fns]
    for s, fns in zip(ss, streams_fns):
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for f in fns: f()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    cap = torch.cuda.Stream()
    with torch.cuda.graph(g, stream=cap):
        for s in ss: s.wait_stream(cap)
        for s, fns in zip(ss, streams_fns):
            with torch.cuda.stream(s):
                for i in range(iters): fns[i % len(fns)]()
        for s in ss: cap.wait_stream(s)
    g.replay(); torch.cuda.synchronize()
    best = 1e9
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); g.replay(); b.record(); b.synchronize()
        best = min(best, a.elapsed_time(b) / 1e3 / iters)
    return best


N, Kd, H = 34816, 5120, 17408
ws = [(torch.randn(N, Kd, device='cuda') * .5).to(F8) for _ in range(6)]
s0 = torch.tensor(0.01, device='cuda')
for M in (1024, 512):
    x = torch.randn(M, Kd, device='cuda').to(F8)
    gm = [lambda w=w: torch._scaled_mm(x, w.t(), one, s0, out_dtype=torch.bfloat16) for w in ws]
    gus = [torch.randn(M, N, device='cuda').to(torch.bfloat16) for _ in range(4)]
    xs = torch.ones(M, device='cuda')
    sl = [lambda gu=gu: K.silu_mul_quant(gu[:, :H], gu[:, H:], xs) for gu in gus]
    res = [torch.randn(M, 5120, device='cuda').to(torch.bfloat16) for _ in range(8)]
    w1 = torch.ones(5120, device='cuda')
    rn = [lambda r=r: K.add_rmsnorm_quant(r, w1, 1e-6, r, xs) for r in res]
    tg = graph_time([gm]); ts = graph_time([sl]); tr = graph_time([rn])
    tgs = graph_time([gm, sl]); tgr = graph_time([gm, rn])
    print(f"M {M}: gemm {tg*1e6:.1f}  silu {ts*1e6:.1f}  rmsnorm {tr*1e6:.1f} us | gemm||silu {tgs*1e6:.1f} "
          f"(serial {(tg+ts)*1e6:.1f}, hidden {(tg+ts-tgs)/ts*100:.0f}% of silu) | gemm||rmsnorm {tgr*1e6:.1f} "
          f"(serial {(tg+tr)*1e6:.1f}, hidden {(tg+tr-tgr)/tr*100:.0f}%)", flush=True)
