# row-streaming conv1d_rows: each program walks R rows of a BC channel block, last 3 inputs in registers
import sys, torch, triton, triton.language as tl
sys.path.insert(0, '.'); sys.path.insert(0, 'scratch')
from agentopt.v3 import kernels as K
from agentopt.v3.kernels import _rbf16
from benchutil import bench_loop
T, C = 2048, 10240


@triton.jit
def _xrow(x_ptr, xs_ptr, r, offs, XS):
    ok = r >= 0
    return _rbf16(tl.load(x_ptr + r.to(tl.int64) * XS + offs, mask=ok & (offs >= 0), other=0.0).to(tl.float32)
                  * tl.load(xs_ptr + r, mask=ok, other=0.0))


@triton.jit
def _tap(p, state_ptr, slot, src, start, offs, C):
    use = src < start
    sv = tl.load(state_ptr + (slot * 3 + src - start + 3) * C + offs, mask=use & (offs >= 0), other=0.0).to(tl.float32)
    return tl.where(use, sv, p)


@triton.jit
def _conv_rows2(x_ptr, xs_ptr, w_ptr, state_ptr, q_out, k_out, v_out, row_start, row_slot, T, XS,
                QK: tl.constexpr, VC: tl.constexpr, R: tl.constexpr, BC: tl.constexpr, DK: tl.constexpr):
    C: tl.constexpr = 2 * QK + VC
    r0 = tl.program_id(0) * R
    c0 = tl.program_id(1) * BC
    offs = c0 + tl.arange(0, BC)
    w0 = tl.load(w_ptr + offs * 4)
    w1 = tl.load(w_ptr + offs * 4 + 1)
    w2 = tl.load(w_ptr + offs * 4 + 2)
    w3 = tl.load(w_ptr + offs * 4 + 3)
    p0 = _xrow(x_ptr, xs_ptr, r0 - 3, offs, XS)
    p1 = _xrow(x_ptr, xs_ptr, r0 - 2, offs, XS)
    p2 = _xrow(x_ptr, xs_ptr, r0 - 1, offs, XS)
    for i in tl.static_range(R):
        r = r0 + i
        if r < T:
            cur = _xrow(x_ptr, xs_ptr, r, offs, XS)
            start = tl.load(row_start + r)
            a0 = p0
            a1 = p1
            a2 = p2
            if r - start < 3:
                slot = tl.load(row_slot + r).to(tl.int64)
                a0 = _tap(p0, state_ptr, slot, r - 3, start, offs, C)
                a1 = _tap(p1, state_ptr, slot, r - 2, start, offs, C)
                a2 = _tap(p2, state_ptr, slot, r - 1, start, offs, C)
            acc = tl.zeros((BC,), tl.float32)
            acc += a0 * w0
            acc += a1 * w1
            acc += a2 * w2
            acc += cur * w3
            y = _rbf16(acc * tl.sigmoid(acc))
            r64 = r.to(tl.int64)
            cc = tl.arange(0, BC)
            if c0 < 2 * QK:
                y2 = tl.reshape(y, (BC // DK, DK))
                y2 = y2 * tl.rsqrt(tl.sum(y2 * y2, 1) + 1e-6)[:, None]
                y = tl.reshape(y2, (BC,))
                if c0 < QK:
                    tl.store(q_out + r64 * QK + c0 + cc, y.to(tl.bfloat16))
                else:
                    tl.store(k_out + r64 * QK + c0 - QK + cc, y.to(tl.bfloat16))
            else:
                tl.store(v_out + r64 * VC + c0 - 2 * QK + cc, y.to(tl.bfloat16))
            p0 = p1
            p1 = p2
            p2 = cur


def conv2(x, xs, w, state, rs, rsl, R, BC, nw):
    q, k, v = K._conv_outs(x.shape[0], x.device)
    _conv_rows2[(triton.cdiv(x.shape[0], R), C // BC)](x, xs, w, state, q, k, v, rs, rsl, x.shape[0], x.stride(0),
                                                       QK=2048, VC=C - 4096, R=R, BC=BC, DK=128, num_warps=nw)
    return q, k, v


if __name__ == "__main__":
    bufs = [torch.randn(T, 16480, device='cuda', dtype=torch.bfloat16) for _ in range(3)]
    xs = torch.rand(T, device='cuda') + 0.5; w = torch.randn(C, 4, device='cuda')
    state = torch.randn(130, 3, C, device='cuda', dtype=torch.bfloat16)
    GB = 2 * T * C * 2
    for lens in [(1992,), (1024, 968), (1, 2, 3, 500, 1486)]:
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
        print(f"{lens}: ref {tr * 1e6:.1f} us")
        for R, BC, nw in [(16, 1024, 4), (8, 1024, 4), (32, 1024, 4), (16, 512, 4), (16, 2048, 8), (16, 1024, 8), (64, 1024, 4)]:
            out = conv2(bufs[0][:, :C], xs, w, state, rs, rsl, R, BC, nw)
            ex = all(torch.equal(a, b) for a, b in zip(ref, out))
            t = bench_loop([lambda b=b: conv2(b[:, :C], xs, w, state, rs, rsl, R, BC, nw) for b in bufs], iters=12)
            print(f"  R{R} BC{BC} w{nw}: {t * 1e6:.1f} us {GB / t / 1e12:.2f} TB/s exact {ex}")
