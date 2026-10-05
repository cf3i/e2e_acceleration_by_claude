"""Small-M N=5120 GEMMs (down K=17408, out K=6144): Triton mm_fp8 config sweep vs cuBLAS (what an epilogue /
prologue fusion of silu / gnorm into them would start from).   python scratch/mmn5120.py"""
import torch, sys, itertools
sys.path.insert(0, '.'); sys.path.insert(0, 'scratch')
from agentopt.v3 import kernels as K
from benchutil import bench_loop
F8 = torch.float8_e4m3fn
one = torch.ones((), device='cuda')
N = 5120
for name, Kd in [("down", 17408), ("out", 6144)]:
    ws = [(torch.randn(N, Kd, device='cuda') * .5).to(F8) for _ in range(max(2, int(4e8 // (N * Kd))))]
    sv = torch.full((N,), 0.01, device='cuda'); s0 = sv[0]
    for M in (1, 8, 16):
        x = torch.randn(M, Kd, device='cuda').to(F8)
        tcb = bench_loop([lambda w=w: torch._scaled_mm(x, w.t(), one, s0, out_dtype=torch.bfloat16) for w in ws], iters=12)
        rows = []
        for BN, BK, nw, st in itertools.product((16, 32, 64), (128, 256, 512), (4, 8), (3, 4, 6, 8)):
            if (BN + 16) * BK * st > 200000: continue
            cfg = dict(BM=16, BN=BN, BK=BK, num_warps=nw, STAGES=st)
            try:
                t = bench_loop([lambda w=w: K.mm_fp8(x, w, sv, cfg) for w in ws], iters=12, reps=3)
            except Exception:
                continue
            rows.append((t, cfg))
        rows.sort(key=lambda r: r[0])
        fl = N * Kd / 1e12
        print(f"{name} M={M:2d}: cublas {tcb*1e6:5.1f} us ({fl/tcb:.2f} TB/s)  triton best {rows[0][0]*1e6:5.1f} "
              f"({fl/rows[0][0]:.2f} TB/s) {rows[0][1]}  2nd {rows[1][0]*1e6:5.1f} {rows[1][1]}", flush=True)
