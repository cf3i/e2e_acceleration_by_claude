"""conv1d_rows as a row-sequential sliding window: each input row is read once (vs 4 tap loads per element)."""
import sys, torch, triton, triton.language as tl
sys.path.insert(0, '.'); sys.path.insert(0, 'scratch')
from triton.language.extra.cuda import gdc_wait
from agentopt.v3 import kernels as K
from benchutil import bench_loop
dev = 'cuda'; bf = torch.bfloat16


@triton.jit
def _xrow(x_ptr, xs_ptr, r, offs, XS, ok):
    return (tl.load(x_ptr + r.to(tl.int64) * XS + offs, mask=ok, other=0.0).to(tl.float32)
            * tl.load(xs_ptr + r, mask=ok, other=0.0)).to(tl.bfloat16).to(tl.float32)


@triton.jit
def _conv1d_rows2(x_ptr, xs_ptr, w_ptr, state_ptr, q_out, k_out, v_out, row_start, row_slot, T, XS,
                  QK: tl.constexpr, VC: tl.constexpr, RB: tl.constexpr, BLOCK_C: tl.constexpr, DK: tl.constexpr,
                  STAGES: tl.constexpr):
    gdc_wait()
    C: tl.constexpr = 2 * QK + VC
    r0 = tl.program_id(0) * RB
    c0 = tl.program_id(1) * BLOCK_C
    cc = tl.arange(0, BLOCK_C)
    offs = c0 + cc
    w0 = tl.load(w_ptr + offs * 4 + 0)
    w1 = tl.load(w_ptr + offs * 4 + 1)
    w2 = tl.load(w_ptr + offs * 4 + 2)
    w3 = tl.load(w_ptr + offs * 4 + 3)
    h0 = _xrow(x_ptr, xs_ptr, r0 - 3, offs, XS, (r0 >= 3) & (r0 - 3 < T))
    h1 = _xrow(x_ptr, xs_ptr, r0 - 2, offs, XS, (r0 >= 2) & (r0 - 2 < T))
    h2 = _xrow(x_ptr, xs_ptr, r0 - 1, offs, XS, (r0 >= 1) & (r0 - 1 < T))
    OS = QK
    if c0 < QK:
        dst = q_out + c0 + cc
    elif c0 < 2 * QK:
        dst = k_out + c0 - QK + cc
    else:
        dst = v_out + c0 - 2 * QK + cc
        OS = VC
    for i in tl.range(0, RB, num_stages=STAGES):
        r = r0 + i
        ok = r < T
        cur = _xrow(x_ptr, xs_ptr, r, offs, XS, ok)
        d = r - tl.load(row_start + r, mask=ok, other=0)
        t0 = h0
        t1 = h1
        t2 = h2
        if d < 3:
            st = state_ptr + tl.load(row_slot + r, mask=ok, other=0).to(tl.int64) * 3 * C + offs
            t0 = tl.load(st + d * C).to(tl.float32)
            if d < 2:
                t1 = tl.load(st + (d + 1) * C).to(tl.float32)
                if d < 1:
                    t2 = tl.load(st + 2 * C).to(tl.float32)
        acc = t0 * w0 + t1 * w1 + t2 * w2 + cur * w3
        y = (acc * tl.sigmoid(acc)).to(tl.bfloat16).to(tl.float32)
        if c0 < 2 * QK:
            y2 = tl.reshape(y, (BLOCK_C // DK, DK))
            y2 = y2 * tl.rsqrt(tl.sum(y2 * y2, 1) + 1e-6)[:, None]
            y = tl.reshape(y2, (BLOCK_C,))
        tl.store(dst + r.to(tl.int64) * OS, y.to(tl.bfloat16), mask=ok & (cc < BLOCK_C))
        h0 = h1
        h1 = h2
        h2 = cur


T, C = 2048, 10240
xs_ = [torch.randn(T, C, device=dev, dtype=bf) for _ in range(3)]
xs = torch.rand(T, device=dev) + .5
w = torch.randn(C, 4, device=dev)
state = torch.randn(130, 3, C, device=dev, dtype=bf)
for name, rs, sl in [("1992+56", [0] * 1992 + list(range(1992, 2048)), [5] * 1992 + list(range(10, 66))),
                     ("many seqs", [i - i % 37 for i in range(T)], [i // 37 for i in range(T)])]:
    row_start = torch.tensor(rs, dtype=torch.int32, device=dev)
    row_slot = torch.tensor(sl, dtype=torch.int32, device=dev)
    outs = K._conv_outs(T, dev)
    gb = 2 * T * C * 2 / 1e9
    f = lambda x: K._conv1d_rows_kernel[(triton.cdiv(T, 16), C // 128)](x, xs, w, state, *outs, row_start, row_slot, T,
                                                                      x.stride(0), QK=2048, VC=C - 4096, BT=16, BLOCK_C=128,
                                                                      DK=128, num_warps=4, launch_pdl=True)
    f(xs_[0]); torch.cuda.synchronize()
    ref = torch.cat([t.reshape(T, -1) for t in outs], 1).clone()
    t = bench_loop([lambda x=x: f(x) for x in xs_], iters=30)
    print(f"{name} current: {t*1e6:.1f} us {gb/t:.0f} GB/s", flush=True)
    for RB, BC, nw, st in [(32, 512, 4, 2), (32, 512, 4, 3), (32, 1024, 4, 2), (32, 1024, 8, 2), (64, 512, 4, 2),
                           (16, 512, 4, 2), (32, 256, 2, 2), (32, 512, 2, 2), (64, 1024, 4, 3), (16, 1024, 4, 2)]:
        outs2 = K._conv_outs(T, dev)
        g = lambda x: _conv1d_rows2[(triton.cdiv(T, RB), C // BC)](x, xs, w, state, *outs2, row_start, row_slot, T,
                                                                  x.stride(0), QK=2048, VC=C - 4096, RB=RB, BLOCK_C=BC,
                                                                  DK=128, STAGES=st, num_warps=nw, launch_pdl=True)
        try:
            g(xs_[0]); torch.cuda.synchronize()
            o = torch.cat([t.reshape(T, -1) for t in outs2], 1)
            t = bench_loop([lambda x=x: g(x) for x in xs_], iters=30)
            print(f"  RB {RB} BC {BC} nw {nw} st {st}: {t*1e6:.1f} us {gb/t:.0f} GB/s  diff {(o.float()-ref.float()).abs().max().item():.2g}", flush=True)
        except Exception as e:
            print(f"  RB {RB} BC {BC} nw {nw} st {st}: fail {str(e)[-600:]}", flush=True)
