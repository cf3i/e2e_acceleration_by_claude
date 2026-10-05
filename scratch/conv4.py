"""conv1d_rows ablation: which part costs the 19 us over the copy ceiling."""
import sys, torch, triton, triton.language as tl
sys.path.insert(0, '.'); sys.path.insert(0, 'scratch')
from triton.language.extra.cuda import gdc_wait
from agentopt.v3 import kernels as K
from benchutil import bench_loop
from triton.language.extra.cuda import libdevice
dev = 'cuda'; bf = torch.bfloat16


@triton.jit
def _rbf16(x):
    # round fp32 -> bf16 -> fp32, two elements per packed cvt
    return tl.inline_asm_elementwise(
        "{ .reg .b32 t; cvt.rn.bf16x2.f32 t, $3, $2; shl.b32 $0, t, 16; and.b32 $1, t, 0xffff0000; }",
        "=r,=r,r,r", [x], dtype=tl.float32, is_pure=True, pack=2)


@triton.jit
def _abl(x_ptr, xs_ptr, w_ptr, q_out, k_out, v_out, row_start, T, XS, QK: tl.constexpr, VC: tl.constexpr,
         BT: tl.constexpr, BLOCK_C: tl.constexpr, DK: tl.constexpr, NTAP: tl.constexpr, XSC: tl.constexpr,
         SILU: tl.constexpr, NORM: tl.constexpr, MASKF: tl.constexpr, FS: tl.constexpr = 0, NR: tl.constexpr = 0):
    gdc_wait()
    rows = tl.program_id(0) * BT + tl.arange(0, BT)
    rm = rows < T
    c0 = tl.program_id(1) * BLOCK_C
    cc = tl.arange(0, BLOCK_C)
    offs = c0 + cc
    start = tl.load(row_start + rows, mask=rm, other=0)
    acc = tl.zeros((BT, BLOCK_C), tl.float32)
    for k in tl.static_range(4 - NTAP, 4):
        src = rows - 3 + k
        if MASKF:
            m = rm & (src >= start)
        else:
            m = rm
        xv = tl.load(x_ptr + src.to(tl.int64)[:, None] * XS + offs[None, :], mask=m[:, None], other=0.0).to(tl.float32)
        if NR == 2:
            xv = _rbf16(xv * tl.load(xs_ptr + src, mask=m, other=0.0)[:, None])
        elif NR:
            xv = xv * tl.load(xs_ptr + src, mask=m, other=0.0)[:, None]
        elif XSC:
            xv = (xv * tl.load(xs_ptr + src, mask=m, other=0.0)[:, None]).to(tl.bfloat16).to(tl.float32)
        acc += xv * tl.load(w_ptr + offs * 4 + k)[None, :]
    if FS == 2:
        y = _rbf16(acc * tl.sigmoid(acc))
    elif FS:
        y = libdevice.fast_dividef(acc, 1.0 + tl.exp2(acc * -1.4426950408889634)).to(tl.bfloat16).to(tl.float32)
    elif SILU:
        y = (acc * tl.sigmoid(acc)).to(tl.bfloat16).to(tl.float32)
    else:
        y = acc
    r64 = rows.to(tl.int64)
    if c0 < 2 * QK:
        if NORM:
            y3 = tl.reshape(y, (BT, BLOCK_C // DK, DK))
            y3 = y3 * tl.rsqrt(tl.sum(y3 * y3, 2) + 1e-6)[:, :, None]
            y = tl.reshape(y3, (BT, BLOCK_C))
        if c0 < QK:
            dst = q_out + r64[:, None] * QK + (c0 + cc)[None, :]
        else:
            dst = k_out + r64[:, None] * QK + (c0 - QK + cc)[None, :]
    else:
        dst = v_out + r64[:, None] * VC + (c0 - 2 * QK + cc)[None, :]
    tl.store(dst, y.to(tl.bfloat16), mask=rm[:, None])


T, C = 2048, 10240
xs_ = [torch.randn(T, C, device=dev, dtype=bf) for _ in range(3)]
xs = torch.rand(T, device=dev) + .5
w = torch.randn(C, 4, device=dev)
row_start = torch.tensor([0] * 1992 + list(range(1992, 2048)), dtype=torch.int32, device=dev)
gb = 2 * T * C * 2 / 1e9
outs = K._conv_outs(T, dev)
for ntap, xsc, silu, norm, maskf, fs, nr in [(4, 1, 1, 1, 1, 0, 0), (4, 1, 1, 1, 1, 0, 2), (4, 1, 1, 1, 1, 2, 2), (4, 1, 1, 1, 1, 0, 1)]:
    for nw in (4,):
        outs = K._conv_outs(T, dev)
        f = lambda x: _abl[(triton.cdiv(T, 16), C // 128)](x, xs, w, *outs, row_start, T, x.stride(0), QK=2048, VC=C - 4096,
                                                           BT=16, BLOCK_C=128, DK=128, NTAP=ntap, XSC=xsc, SILU=silu, NORM=norm,
                                                           MASKF=maskf, FS=fs, NR=nr, num_warps=nw, launch_pdl=True)
        f(xs_[0]); torch.cuda.synchronize(); o = torch.cat([t.reshape(T, -1) for t in outs], 1).clone()
        if fs == 0 and nr == 0: ref = o
        t = bench_loop([lambda x=x: f(x) for x in xs_], iters=30)
        print(f"ndiff {(o != ref).sum().item()} fs {fs} nr {nr}: {t*1e6:.1f} us {gb/t:.0f} GB/s", flush=True)
for ntap, xsc, silu, norm, maskf in [] and [(4, 1, 1, 1, 1), (1, 1, 1, 1, 1), (4, 0, 1, 1, 1), (4, 1, 0, 1, 1), (4, 1, 1, 0, 1),
                                     (4, 1, 1, 1, 0), (4, 0, 0, 0, 0), (1, 0, 0, 0, 0), (2, 1, 1, 1, 1)]:
    for nw in (4,):
        f = lambda x: _abl[(triton.cdiv(T, 16), C // 128)](x, xs, w, *outs, row_start, T, x.stride(0), QK=2048, VC=C - 4096,
                                                           BT=16, BLOCK_C=128, DK=128, NTAP=ntap, XSC=xsc, SILU=silu, NORM=norm,
                                                           MASKF=maskf, num_warps=nw, launch_pdl=True)
        t = bench_loop([lambda x=x: f(x) for x in xs_], iters=30)
        print(f"ntap {ntap} xs {xsc} silu {silu} norm {norm} maskf {maskf} nw {nw}: {t*1e6:.1f} us {gb/t:.0f} GB/s", flush=True)
