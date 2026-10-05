# which part of conv1d_rows costs beyond the copy floor: shifted loads, xs scale, silu, l2 norm
import sys, torch, triton, triton.language as tl
sys.path.insert(0, '.'); sys.path.insert(0, 'scratch')
from agentopt.v3.kernels import _rbf16
from benchutil import bench_loop
T, C = 2048, 10240

@triton.jit
def _k(x, xs_ptr, w_ptr, y, T, XS, C: tl.constexpr, BT: tl.constexpr, BC: tl.constexpr, NTAP: tl.constexpr,
       XSC: tl.constexpr, SILU: tl.constexpr, NORM: tl.constexpr, EVICT: tl.constexpr):
    r = tl.program_id(0) * BT + tl.arange(0, BT)
    c = tl.program_id(1) * BC + tl.arange(0, BC)
    acc = tl.zeros((BT, BC), tl.float32)
    for k in tl.static_range(4 - NTAP, 4):
        src = r - 3 + k
        m = (src >= 0) & (src < T)
        if EVICT:
            xv = tl.load(x + src.to(tl.int64)[:, None] * XS + c[None, :], mask=m[:, None], other=0.0,
                         eviction_policy="evict_last").to(tl.float32)
        else:
            xv = tl.load(x + src.to(tl.int64)[:, None] * XS + c[None, :], mask=m[:, None], other=0.0).to(tl.float32)
        if XSC:
            xv = _rbf16(xv * tl.load(xs_ptr + src, mask=m, other=0.0)[:, None])
        acc += xv * tl.load(w_ptr + c * 4 + k)[None, :]
    if SILU:
        acc = _rbf16(acc * tl.sigmoid(acc))
    if NORM:
        y3 = tl.reshape(acc, (BT, BC // 128, 128))
        y3 = y3 * tl.rsqrt(tl.sum(y3 * y3, 2) + 1e-6)[:, :, None]
        acc = tl.reshape(y3, (BT, BC))
    tl.store(y + r.to(tl.int64)[:, None] * C + c[None, :], acc.to(tl.bfloat16), mask=(r < T)[:, None])

bufs = [torch.randn(T, 16480, device='cuda', dtype=torch.bfloat16) for _ in range(3)]
ys = [torch.empty(T, C, device='cuda', dtype=torch.bfloat16) for _ in range(3)]
xs = torch.rand(T, device='cuda') + 0.5; w = torch.randn(C, 4, device='cuda')
for BT, BC in [(16, 128), (16, 256), (32, 128)]:
    for ntap, xsc, silu, norm, ev in [(1, 0, 0, 0, 0), (4, 0, 0, 0, 0), (4, 0, 0, 0, 1), (4, 1, 0, 0, 0), (4, 1, 1, 0, 0),
                                      (4, 1, 1, 1, 0), (1, 1, 1, 1, 0)]:
        fns = [lambda b=b, y=y: _k[(T // BT, C // BC)](b, xs, w, y, T, b.stride(0), C=C, BT=BT, BC=BC, NTAP=ntap, XSC=xsc,
                                                       SILU=silu, NORM=norm, EVICT=ev, num_warps=4)
               for b, y in zip(bufs, ys)]
        t = bench_loop(fns, iters=12)
        print(f"BT{BT} BC{BC} taps {ntap} xs {xsc} silu {silu} norm {norm} evict {ev}: {t * 1e6:.1f} us")
