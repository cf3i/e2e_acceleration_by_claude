"""gdn_in b/a projections (N = 48 each) at large M: two cuBLAS calls vs one Triton mm_fp8 over the stacked [b; a]."""
import sys, torch, itertools
sys.path.insert(0, '.'); sys.path.insert(0, 'scratch')
from agentopt.v3 import kernels as K
from benchutil import bench_loop
F8 = torch.float8_e4m3fn
one = torch.ones((), device='cuda')
NL = 6
ws = [(torch.randn(96, 5120, device='cuda') * .5).to(F8) for _ in range(NL)]
sv = torch.cat([torch.full((48,), 0.01), torch.full((48,), 0.02)]).cuda()
s1, s2 = sv[0].clone(), sv[48].clone()
for M in (160, 320, 1024, 2048):
    x = torch.randn(M, 5120, device='cuda').to(F8)
    t = bench_loop([lambda w=w: (torch._scaled_mm(x, w[:48].t(), one, s1, out_dtype=torch.bfloat16),
                                 torch._scaled_mm(x, w[48:].t(), one, s2, out_dtype=torch.bfloat16)) for w in ws], iters=12)
    ref = torch.cat([torch._scaled_mm(x, ws[0][:48].t(), one, s1, out_dtype=torch.bfloat16),
                     torch._scaled_mm(x, ws[0][48:].t(), one, s2, out_dtype=torch.bfloat16)], 1).float()
    res = []
    for BM, BN, BK, nw, st in itertools.product((16, 32, 64), (32, 96 if False else 64), (128, 256), (4,), (3, 4)):
        cfg = dict(BM=BM, BN=BN, BK=BK, num_warps=nw, STAGES=st)
        try:
            y = K.mm_fp8(x, ws[0], sv, cfg)
            err = ((y.float() - ref).abs().max() / ref.abs().max()).item()
            tt = bench_loop([lambda w=w: K.mm_fp8(x, w, sv, cfg) for w in ws], iters=12)
            res.append((tt, cfg, err))
        except Exception as e:
            print('fail', cfg, str(e)[:100])
    res.sort(key=lambda r: r[0])
    print(f"M={M}: cublas b+a {t*1e6:.1f} us | triton best " + " ; ".join(f"{r[0]*1e6:.1f} {tuple(r[1].values())} err {r[2]:.1e}" for r in res[:3]), flush=True)
