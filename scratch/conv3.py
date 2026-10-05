"""conv1d_rows: copy ceiling for the same tiling, and grid-order (channel-fastest) variant."""
import sys, torch, triton, triton.language as tl
sys.path.insert(0, '.'); sys.path.insert(0, 'scratch')
from triton.language.extra.cuda import gdc_wait
from agentopt.v3 import kernels as K
from agentopt.v3.kernels import _conv_taps, _conv_out
from benchutil import bench_loop
dev = 'cuda'; bf = torch.bfloat16


@triton.jit
def _copy(x_ptr, q_out, k_out, v_out, T, XS, QK: tl.constexpr, VC: tl.constexpr, BT: tl.constexpr,
          BLOCK_C: tl.constexpr, SWAP: tl.constexpr):
    gdc_wait()
    if SWAP:
        rb = tl.program_id(1); cb = tl.program_id(0)
    else:
        rb = tl.program_id(0); cb = tl.program_id(1)
    rows = rb * BT + tl.arange(0, BT)
    rm = rows < T
    c0 = cb * BLOCK_C
    cc = tl.arange(0, BLOCK_C)
    y = tl.load(x_ptr + rows.to(tl.int64)[:, None] * XS + (c0 + cc)[None, :], mask=rm[:, None])
    if c0 < QK:
        tl.store(q_out + rows.to(tl.int64)[:, None] * QK + (c0 + cc)[None, :], y, mask=rm[:, None])
    elif c0 < 2 * QK:
        tl.store(k_out + rows.to(tl.int64)[:, None] * QK + (c0 - QK + cc)[None, :], y, mask=rm[:, None])
    else:
        tl.store(v_out + rows.to(tl.int64)[:, None] * VC + (c0 - 2 * QK + cc)[None, :], y, mask=rm[:, None])


@triton.jit
def _conv_sw(x_ptr, xs_ptr, w_ptr, state_ptr, q_out, k_out, v_out, row_start, row_slot, T, XS,
             QK: tl.constexpr, VC: tl.constexpr, BT: tl.constexpr, BLOCK_C: tl.constexpr, DK: tl.constexpr,
             SWAP: tl.constexpr):
    gdc_wait()
    C: tl.constexpr = 2 * QK + VC
    if SWAP:
        rb = tl.program_id(1); cb = tl.program_id(0)
    else:
        rb = tl.program_id(0); cb = tl.program_id(1)
    rows = rb * BT + tl.arange(0, BT)
    rm = rows < T
    c0 = cb * BLOCK_C
    offs = c0 + tl.arange(0, BLOCK_C)
    start = tl.load(row_start + rows, mask=rm, other=0)
    slot = tl.load(row_slot + rows, mask=rm, other=0).to(tl.int64)
    if tl.min(tl.where(rm, rows - start, 3)) < 3:
        acc = _conv_taps(x_ptr, xs_ptr, w_ptr, state_ptr, rows, rm, offs, start, slot, XS, C, BT, BLOCK_C, True)
    else:
        acc = _conv_taps(x_ptr, xs_ptr, w_ptr, state_ptr, rows, rm, offs, start, slot, XS, C, BT, BLOCK_C, False)
    _conv_out(acc, rows, rm, c0, q_out, k_out, v_out, QK, VC, BT, BLOCK_C, DK)


T, C = 2048, 10240
XS = int(sys.argv[1]) if len(sys.argv) > 1 else C
xbuf = [torch.randn(T, XS, device=dev, dtype=bf) for _ in range(3)]
xs_ = [b[:, :C] for b in xbuf]
xs = torch.rand(T, device=dev) + .5
w = torch.randn(C, 4, device=dev)
state = torch.randn(130, 3, C, device=dev, dtype=bf)
row_start = torch.tensor([0] * 1992 + list(range(1992, 2048)), dtype=torch.int32, device=dev)
row_slot = torch.tensor([5] * 1992 + list(range(10, 66)), dtype=torch.int32, device=dev)
gb = 2 * T * C * 2 / 1e9
for BT, BC, nw in [(16, 128, 4), (8, 128, 2), (16, 256, 4), (32, 128, 4), (8, 256, 4), (4, 512, 4)]:
    for sw in (0, 1):
        outs = K._conv_outs(T, dev)
        g = lambda x: _copy[(C // BC, triton.cdiv(T, BT)) if sw else (triton.cdiv(T, BT), C // BC)](
            x, *outs, T, x.stride(0), QK=2048, VC=C - 4096, BT=BT, BLOCK_C=BC, SWAP=sw, num_warps=nw, launch_pdl=True)
        tc = bench_loop([lambda x=x: g(x) for x in xs_], iters=30)
        f = lambda x: _conv_sw[(C // BC, triton.cdiv(T, BT)) if sw else (triton.cdiv(T, BT), C // BC)](
            x, xs, w, state, *outs, row_start, row_slot, T, x.stride(0), QK=2048, VC=C - 4096, BT=BT, BLOCK_C=BC,
            DK=128, SWAP=sw, num_warps=nw, launch_pdl=True)
        t = bench_loop([lambda x=x: f(x) for x in xs_], iters=30)
        print(f"BT {BT} BC {BC} nw {nw} swap {sw}: copy {tc*1e6:.1f} us {gb/tc:.0f} GB/s | conv {t*1e6:.1f} us {gb/t:.0f} GB/s", flush=True)
