# conv1d_rows with a scalar fast-path test (first/last row of the block share a sequence start >= 3 rows back)
import sys, torch, triton, triton.language as tl
sys.path.insert(0, '.'); sys.path.insert(0, 'scratch')
from agentopt.v3 import kernels as K
from agentopt.v3.kernels import _conv_taps, _conv_out
from benchutil import bench_loop
T, C = 2048, 10240

@triton.jit
def _conv_rows_s(x_ptr, xs_ptr, w_ptr, state_ptr, q_out, k_out, v_out, row_start, row_slot, T, XS,
                 QK: tl.constexpr, VC: tl.constexpr, BT: tl.constexpr, BLOCK_C: tl.constexpr, DK: tl.constexpr):
    C: tl.constexpr = 2 * QK + VC
    r0 = tl.program_id(0) * BT
    rows = r0 + tl.arange(0, BT)
    rm = rows < T
    c0 = tl.program_id(1) * BLOCK_C
    offs = c0 + tl.arange(0, BLOCK_C)
    s0 = tl.load(row_start + r0)
    s1 = tl.load(row_start + tl.minimum(r0 + BT, T) - 1)
    if (s0 == s1) & (r0 - s0 >= 3):
        acc = _conv_taps(x_ptr, xs_ptr, w_ptr, state_ptr, rows, rm, offs, s0, 0, XS, C, BT, BLOCK_C, False)
    else:
        start = tl.load(row_start + rows, mask=rm, other=0)
        slot = tl.load(row_slot + rows, mask=rm, other=0).to(tl.int64)
        acc = _conv_taps(x_ptr, xs_ptr, w_ptr, state_ptr, rows, rm, offs, start, slot, XS, C, BT, BLOCK_C, True)
    _conv_out(acc, rows, rm, c0, q_out, k_out, v_out, QK, VC, BT, BLOCK_C, DK)


def conv_s(x, xs, w, state, rs, rsl, BT=16, BC=128, nw=4):
    q, k, v = K._conv_outs(x.shape[0], x.device)
    _conv_rows_s[(triton.cdiv(x.shape[0], BT), C // BC)](x, xs, w, state, q, k, v, rs, rsl, x.shape[0], x.stride(0),
                                                         QK=2048, VC=C - 4096, BT=BT, BLOCK_C=BC, DK=128, num_warps=nw)
    return q, k, v

bufs = [torch.randn(T, 16480, device='cuda', dtype=torch.bfloat16) for _ in range(3)]
xs = torch.rand(T, device='cuda') + 0.5; w = torch.randn(C, 4, device='cuda')
state = torch.randn(130, 3, C, device='cuda', dtype=torch.bfloat16)
for lens in [(1992,), (1024, 968), (1, 2, 3, 500, 1486), (5, 17, 2000)]:
    rs, rsl, p = [], [], 0
    for i, n in enumerate(lens):
        rs += [p] * n; rsl += [100 + i] * n; p += n
    rs += list(range(p, T)); rsl += list(range(10, 10 + T - p))
    rs = torch.tensor(rs, dtype=torch.int32, device='cuda'); rsl = torch.tensor(rsl, dtype=torch.int32, device='cuda')
    n = len(lens) + T - p
    ss = torch.zeros(n, dtype=torch.int32, device='cuda')
    ref = K.conv1d_rows(bufs[0][:, :C], xs, w, state, rs, rsl, ss, ss, ss, n, update_state=False)
    tr = bench_loop([lambda b=b: K.conv1d_rows(b[:, :C], xs, w, state, rs, rsl, ss, ss, ss, n, update_state=False)
                     for b in bufs], iters=12)
    line = f"{lens}: ref {tr * 1e6:.1f} us |"
    for BT, BC, nw in [(16, 128, 4), (16, 256, 4), (32, 128, 4), (32, 256, 4), (8, 256, 2)]:
        out = conv_s(bufs[0][:, :C], xs, w, state, rs, rsl, BT, BC, nw)
        ex = all(torch.equal(a, b) for a, b in zip(ref, out))
        t = bench_loop([lambda b=b: conv_s(b[:, :C], xs, w, state, rs, rsl, BT, BC, nw) for b in bufs], iters=12)
        line += f" {BT}/{BC}/{nw} {t * 1e6:.1f}{'' if ex else ' INEXACT'}"
    print(line)
