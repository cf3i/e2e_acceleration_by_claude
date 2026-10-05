"""Triton kernels of the v3 forward (flat-token batches: decode rows and prefill chunks mixed in one step).

Conventions
- A step processes T token rows. Row-major [T, ...] activations.
- FP8 linears: `torch._scaled_mm(xq, Wq^T, 1, w_scale)` returns y / xs (the per-token activation scale is NOT
  applied inside the GEMM, so the stock cuBLASLt tensorwise kernel can be used). Every consumer of a GEMM output
  takes the producer's per-row scale `xs` and applies it on load (`* xs[row]`, then bf16 rounding, which is what a
  bf16 linear output would be).
- Activation quantization is per token (one fp32 scale per row, amax / 448), fused into the producer kernel.
- Recurrent / conv / KV state lives in pools indexed by slot (recurrent, conv) or by paged block (KV).
"""
import os

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import gdc_wait

FP8 = torch.float8_e4m3fn

# Programmatic dependent launch: every kernel below starts with gdc_wait() and is launched with launch_pdl, so the
# GPU may launch it while its predecessor is still running: its programs get resident early and wait in
# griddepcontrol.wait for the predecessor's completion and memory flush. Hides the launch latency between
# back-to-back kernels. No early griddepcontrol.launch_dependents: measured slower at large batch (dependents
# resident too early), the implicit trigger at exit is enough. AGENTOPT_PDL=0 turns it off.
PDL = os.environ.get("AGENTOPT_PDL", "1") != "0"


@triton.jit
def _rbf16(x):
    """x rounded to bf16, returned as fp32 (bit-exact .to(bf16).to(fp32)); one packed cvt per element pair, where
    the round trip compiles to one scalar cvt per element."""
    return tl.inline_asm_elementwise(
        "{ .reg .b32 t; cvt.rn.bf16x2.f32 t, $3, $2; shl.b32 $0, t, 16; and.b32 $1, t, 0xffff0000; }",
        "=r,=r,r,r", [x], dtype=tl.float32, is_pure=True, pack=2)


# ---------------------------------------------------------------- FP8 GEMM for small M (merged weights)
# y[:, n] = (x @ W^T)[:, n] * s[n]: W is several linears sharing an input stacked along N (each keeps its own
# per-tensor scale, repeated in s), so one launch streams all of them. Small M is weight-read bound: narrow N tiles
# give enough programs to fill the GPU, full-K loop, no split. For large M the per-linear cuBLAS calls are faster.
@triton.jit
def _mm_fp8_kernel(x_ptr, w_ptr, s_ptr, y_ptr, M, N, K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                   BK: tl.constexpr, STAGES: tl.constexpr, xs_ptr=None, cw_ptr=None, cst_ptr=None, row_slot=None,
                   CONV: tl.constexpr = 0):
    gdc_wait()
    m = tl.program_id(1) * BM + tl.arange(0, BM)
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    mm = m < M
    nm = n < N
    acc = tl.zeros((BM, BN), tl.float32)
    if CONV > 0:  # the conv window loads are issued before the K loop (latency hidden behind it)
        isconv = tl.program_id(0) * BN < CONV
        slot = tl.load(row_slot + m, mask=mm, other=0).to(tl.int64)
        st = cst_ptr + slot[:, None] * 3 * CONV + n[None, :]
        cm = mm[:, None] & isconv
        x0 = tl.load(st, mask=cm)
        x1 = tl.load(st + CONV, mask=cm)
        x2 = tl.load(st + 2 * CONV, mask=cm)
    for k0 in tl.range(0, K, BK, num_stages=STAGES):
        k = k0 + tl.arange(0, BK)
        x = tl.load(x_ptr + m.to(tl.int64)[:, None] * K + k[None, :], mask=mm[:, None], other=0.0)
        w = tl.load(w_ptr + n.to(tl.int64)[:, None] * K + k[None, :], mask=nm[:, None], other=0.0)
        # fp8 wgmma accumulates with reduced precision: promote to fp32 every 128 products (as cuBLAS does)
        acc = tl.dot(x, tl.trans(w), acc, max_num_imprecise_acc=128)
    s = tl.load(s_ptr + n, mask=nm, other=0.0)
    y = (acc * s[None, :]).to(tl.bfloat16)
    if CONV > 0:
        if isconv:
            y = _conv_step(y, m, mm, n, xs_ptr, cw_ptr, st, x0, x1, x2, CONV)
    tl.store(y_ptr + m.to(tl.int64)[:, None] * N + n[None, :], y, mask=mm[:, None] & nm[None, :])


@triton.jit
def _conv_step(x, m, mm, n, xs_ptr, cw_ptr, st, x0, x1, x2, C: tl.constexpr):
    """Decode-step causal conv1d of GEMM output columns n < C (one token per row = per sequence; the channels of a
    sequence's conv window st (x0..x2 = its values) belong to this program alone, so it updates the window in place):
    SiLU(conv) without the q / k l2 norm (gdn_decode applies it). Same math as _conv1d_seq_kernel."""
    xt = _rbf16(x.to(tl.float32) * tl.load(xs_ptr + m, mask=mm, other=0.0)[:, None])
    x0 = x0.to(tl.float32)
    x1 = x1.to(tl.float32)
    x2 = x2.to(tl.float32)
    w0 = tl.load(cw_ptr + n * 4 + 0)[None, :]
    w1 = tl.load(cw_ptr + n * 4 + 1)[None, :]
    w2 = tl.load(cw_ptr + n * 4 + 2)[None, :]
    w3 = tl.load(cw_ptr + n * 4 + 3)[None, :]
    acc = x0 * w0 + x1 * w1 + x2 * w2 + xt * w3
    tl.store(st, x1.to(tl.bfloat16), mask=mm[:, None])
    tl.store(st + C, x2.to(tl.bfloat16), mask=mm[:, None])
    tl.store(st + 2 * C, xt.to(tl.bfloat16), mask=mm[:, None])
    return _rbf16(acc * tl.sigmoid(acc)).to(tl.bfloat16)


def mm_fp8_cfg(M, N):
    """Tile config by M (sweep: scratch/mmtune.py, K = 5120, timed in CUDA graphs without L2 flushes); None: use
    cuBLAS."""
    wide = N > 24576
    if M <= 16:
        return dict(BM=16, BN=64, BK=128, num_warps=4, STAGES=3) if wide else \
            dict(BM=16, BN=32, BK=256, num_warps=4, STAGES=3)
    if M <= 32 and wide:
        return dict(BM=32, BN=64, BK=128, num_warps=4, STAGES=3)
    if M <= 64:
        return dict(BM=64, BN=32, BK=128, num_warps=4, STAGES=5) if wide else \
            dict(BM=64, BN=64, BK=128, num_warps=4, STAGES=5)
    if M <= 128 and not wide:
        return dict(BM=128, BN=128, BK=128, num_warps=8, STAGES=5)
    return None


def mm_fp8(x, w, s, cfg, conv=None):
    """x [M, K] fp8, w [N, K] fp8, s [N] fp32 -> [M, N] bf16. conv = (xs, conv_w, conv_state, row_slot): columns
    [0, C) of a decode step's gdn_in projection are replaced by the conv1d output (see _conv_step)."""
    M, K = x.shape
    N = w.shape[0]
    y = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)
    c = dict(cfg)
    BM, BN = c["BM"], c["BN"]
    if conv is not None:
        xs, cw, cst, row_slot = conv
        assert cst.shape[-1] % BN == 0
        c.update(xs_ptr=xs, cw_ptr=cw, cst_ptr=cst, row_slot=row_slot, CONV=cst.shape[-1])
    _mm_fp8_kernel[(triton.cdiv(N, BN), triton.cdiv(M, BM))](x, w, s, y, M, N, K=K, **c, launch_pdl=PDL)
    return y


# ---------------------------------------------------------------- residual add + RMSNorm(1 + w) + quant
@triton.jit
def _add_rmsnorm_kernel(res_ptr, d_ptr, ds_ptr, w_ptr, q_ptr, s_ptr, out_ptr, N, eps,
                        HAS_DELTA: tl.constexpr, QUANT: tl.constexpr, BLOCK: tl.constexpr):
    gdc_wait()
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK)
    m = offs < N
    x = tl.load(res_ptr + row * N + offs, mask=m, other=0.0).to(tl.float32)
    if HAS_DELTA:
        d = tl.load(d_ptr + row * N + offs, mask=m, other=0.0).to(tl.float32)
        d = _rbf16(d * tl.load(ds_ptr + row))
        x = (x + d).to(tl.bfloat16)
        tl.store(res_ptr + row * N + offs, x, mask=m)
        x = x.to(tl.float32)
    var = tl.sum(x * x, 0) / N
    w = tl.load(w_ptr + offs, mask=m, other=0.0)
    y = _rbf16(x * tl.rsqrt(var + eps) * w)
    if QUANT:
        s = tl.maximum(tl.max(tl.abs(y), 0), 1e-12) / 448.0
        tl.store(q_ptr + row * N + offs, tl.clamp(y / s, -448.0, 448.0).to(tl.float8e4nv), mask=m)
        tl.store(s_ptr + row, s)
    else:
        tl.store(out_ptr + row * N + offs, y.to(tl.bfloat16), mask=m)


def add_rmsnorm_quant(res, w1, eps, delta=None, dscale=None, quant=True):
    """res [T, N] bf16 (updated in place: res += delta * dscale[:, None] when delta is given), w1 = 1 + weight
    (fp32). Returns (q fp8 [T, N], s fp32 [T]) or, quant=False, the bf16 normalized rows."""
    T, N = res.shape
    if quant:
        q = torch.empty((T, N), dtype=FP8, device=res.device)
        s = torch.empty((T,), dtype=torch.float32, device=res.device)
        out = None
    else:
        q = s = None
        out = torch.empty_like(res)
    _add_rmsnorm_kernel[(T,)](res, delta, dscale, w1, q, s, out, N, eps, HAS_DELTA=delta is not None,
                              QUANT=quant, BLOCK=triton.next_power_of_2(N), num_warps=16, launch_pdl=PDL)
    return (q, s) if quant else out


# ---------------------------------------------------------------- SiLU(gate) * up + quant (MLP)
@triton.jit
def _silu_mul_quant_kernel(g_ptr, u_ptr, xs_ptr, q_ptr, s_ptr, N, GS, BLOCK: tl.constexpr):
    gdc_wait()
    row = tl.program_id(0).to(tl.int64)
    xs = tl.load(xs_ptr + row)
    amax = tl.zeros((BLOCK,), tl.float32)
    for off in range(0, N, BLOCK):  # pass 1: row amax
        offs = off + tl.arange(0, BLOCK)
        m = offs < N
        g = _rbf16(tl.load(g_ptr + row * GS + offs, mask=m, other=0.0).to(tl.float32) * xs)
        u = _rbf16(tl.load(u_ptr + row * GS + offs, mask=m, other=0.0).to(tl.float32) * xs)
        h = _rbf16(g * tl.sigmoid(g) * u)
        amax = tl.maximum(amax, tl.abs(h))
    s = tl.maximum(tl.max(amax, 0), 1e-12) / 448.0
    for off in range(0, N, BLOCK):  # pass 2: quantize (the row is in L2)
        offs = off + tl.arange(0, BLOCK)
        m = offs < N
        g = _rbf16(tl.load(g_ptr + row * GS + offs, mask=m, other=0.0).to(tl.float32) * xs)
        u = _rbf16(tl.load(u_ptr + row * GS + offs, mask=m, other=0.0).to(tl.float32) * xs)
        h = _rbf16(g * tl.sigmoid(g) * u)
        tl.store(q_ptr + row * N + offs, tl.clamp(h / s, -448.0, 448.0).to(tl.float8e4nv), mask=m)
    tl.store(s_ptr + row, s)


@triton.jit
def _silu_mul_chunk(g_ptr, u_ptr, base, off, xs, CH: tl.constexpr):
    offs = off + tl.arange(0, CH)
    g = _rbf16(tl.load(g_ptr + base + offs).to(tl.float32) * xs)
    u = _rbf16(tl.load(u_ptr + base + offs).to(tl.float32) * xs)
    return _rbf16(g * tl.sigmoid(g) * u)


@triton.jit
def _silu_mul_quant1_kernel(g_ptr, u_ptr, xs_ptr, q_ptr, s_ptr, GS, N: tl.constexpr, CH: tl.constexpr,
                            TAIL: tl.constexpr):
    """Single pass, the whole row (N = 2 CH + TAIL) in registers: the row is read once and the program has all
    its loads in flight at once (the two-pass loop is latency-bound for small T and re-reads rows for large T)."""
    gdc_wait()
    row = tl.program_id(0).to(tl.int64)
    xs = tl.load(xs_ptr + row)
    gb = row * GS
    h0 = _silu_mul_chunk(g_ptr, u_ptr, gb, 0, xs, CH)
    h1 = _silu_mul_chunk(g_ptr, u_ptr, gb, CH, xs, CH)
    h2 = _silu_mul_chunk(g_ptr, u_ptr, gb, 2 * CH, xs, TAIL)
    base = row * N
    amax = tl.maximum(tl.maximum(tl.max(tl.abs(h0), 0), tl.max(tl.abs(h1), 0)), tl.max(tl.abs(h2), 0))
    s = tl.maximum(amax, 1e-12) / 448.0
    tl.store(q_ptr + base + tl.arange(0, CH), tl.clamp(h0 / s, -448.0, 448.0).to(tl.float8e4nv))
    tl.store(q_ptr + base + CH + tl.arange(0, CH), tl.clamp(h1 / s, -448.0, 448.0).to(tl.float8e4nv))
    tl.store(q_ptr + base + 2 * CH + tl.arange(0, TAIL), tl.clamp(h2 / s, -448.0, 448.0).to(tl.float8e4nv))
    tl.store(s_ptr + row, s)


def silu_mul_quant(gate, up, xs):
    """gate, up [T, N] with unit column stride and a common row stride (column slices of one GEMM output ok)."""
    T, N = gate.shape
    assert up.stride(0) == gate.stride(0) and gate.stride(1) == up.stride(1) == 1
    q = torch.empty((T, N), dtype=FP8, device=gate.device)
    s = torch.empty((T,), dtype=torch.float32, device=gate.device)
    CH = 8192
    tail = N - 2 * CH
    if 0 < tail <= CH and tail & (tail - 1) == 0:
        _silu_mul_quant1_kernel[(T,)](gate, up, xs, q, s, gate.stride(0), N=N, CH=CH, TAIL=tail, num_warps=16,
                                      launch_pdl=PDL)
    else:
        _silu_mul_quant_kernel[(T,)](gate, up, xs, q, s, N, gate.stride(0), BLOCK=4096, num_warps=8, launch_pdl=PDL)
    return q, s


# ---------------------------------------------------------------- Gated-DeltaNet: causal conv1d
# Output: SiLU(conv) split into contiguous q [T, HK*DK], k [T, HK*DK], v [T, HV*DV]; q and k l2-normalized per
# head (what fla's use_qk_l2norm_in_kernel computes), so the chunked prefill and the decode step share inputs.
@triton.jit
def _conv_out(acc, rows, rm, c0, q_out, k_out, v_out, QK: tl.constexpr, VC: tl.constexpr, BT: tl.constexpr,
              BLOCK_C: tl.constexpr, DK: tl.constexpr):
    y = _rbf16(acc * tl.sigmoid(acc))
    cc = tl.arange(0, BLOCK_C)
    r64 = rows.to(tl.int64)
    if c0 < 2 * QK:
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


@triton.jit
def _conv1d_seq_kernel(x_ptr, xs_ptr, w_ptr, state_ptr, q_out, k_out, v_out, seq_slot, seq_start, seq_len, XS,
                       QK: tl.constexpr, VC: tl.constexpr, BLOCK_C: tl.constexpr, DK: tl.constexpr):
    """One program per (sequence, channel block): walks the sequence's new tokens in order with the last 3 inputs
    in registers (initialized from the slot's conv state) and writes the new state. For decode (1 token/seq)."""
    gdc_wait()
    C: tl.constexpr = 2 * QK + VC
    s = tl.program_id(0)
    n = tl.load(seq_len + s)
    if n > 0:
        slot = tl.load(seq_slot + s).to(tl.int64)
        t0 = tl.load(seq_start + s)
        c0 = tl.program_id(1) * BLOCK_C
        offs = c0 + tl.arange(0, BLOCK_C)
        w0 = tl.load(w_ptr + offs * 4 + 0)
        w1 = tl.load(w_ptr + offs * 4 + 1)
        w2 = tl.load(w_ptr + offs * 4 + 2)
        w3 = tl.load(w_ptr + offs * 4 + 3)
        st = state_ptr + slot * 3 * C
        x0 = tl.load(st + offs).to(tl.float32)
        x1 = tl.load(st + C + offs).to(tl.float32)
        x2 = tl.load(st + 2 * C + offs).to(tl.float32)
        for t in range(n):
            row = t0 + t
            xt = _rbf16(tl.load(x_ptr + row.to(tl.int64) * XS + offs).to(tl.float32) * tl.load(xs_ptr + row))
            acc = x0 * w0 + x1 * w1 + x2 * w2 + xt * w3
            _conv_out(acc[None, :], tl.full((1,), 0, tl.int32) + row, tl.full((1,), 1, tl.int1), c0,
                      q_out, k_out, v_out, QK, VC, 1, BLOCK_C, DK)
            x0 = x1
            x1 = x2
            x2 = xt
        tl.store(st + offs, x0.to(tl.bfloat16))
        tl.store(st + C + offs, x1.to(tl.bfloat16))
        tl.store(st + 2 * C + offs, x2.to(tl.bfloat16))


@triton.jit
def _conv_taps(x_ptr, xs_ptr, w_ptr, state_ptr, rows, rm, offs, start, slot, XS, C: tl.constexpr, BT: tl.constexpr,
               BLOCK_C: tl.constexpr, STATE: tl.constexpr):
    acc = tl.zeros((BT, BLOCK_C), tl.float32)
    for k in tl.static_range(4):
        src = rows - 3 + k
        fx = src >= start
        xv = tl.load(x_ptr + src.to(tl.int64)[:, None] * XS + offs[None, :], mask=(rm & fx)[:, None],
                     other=0.0).to(tl.float32)
        xv = _rbf16(xv * tl.load(xs_ptr + src, mask=rm & fx, other=0.0)[:, None])
        if STATE:
            j = src - start + 3
            sv = tl.load(state_ptr + (slot * 3 + j)[:, None] * C + offs[None, :],
                         mask=(rm & (~fx) & (j >= 0))[:, None], other=0.0).to(tl.float32)
            xv = tl.where(fx[:, None], xv, sv)
        wk = tl.load(w_ptr + offs * 4 + k)
        acc += xv * wk[None, :]
    return acc


@triton.jit
def _conv1d_rows_kernel(x_ptr, xs_ptr, w_ptr, state_ptr, q_out, k_out, v_out, row_start, row_slot, T, XS,
                        QK: tl.constexpr, VC: tl.constexpr, BT: tl.constexpr, BLOCK_C: tl.constexpr,
                        DK: tl.constexpr):
    """Token-parallel conv: one program per (row block, channel block). Tap inputs before the row's sequence start
    come from the slot's conv state. Does not update the state (see _conv1d_state_kernel)."""
    gdc_wait()
    C: tl.constexpr = 2 * QK + VC
    r0 = tl.program_id(0) * BT
    rows = r0 + tl.arange(0, BT)
    rm = rows < T
    c0 = tl.program_id(1) * BLOCK_C
    offs = c0 + tl.arange(0, BLOCK_C)
    # the state gather is expensive: only blocks holding one of the first 3 tokens of a sequence take that path.
    # A sequence's rows are contiguous, so first and last row sharing a start = one sequence (two scalar loads
    # instead of a block reduction: -10%)
    s0 = tl.load(row_start + r0)
    s1 = tl.load(row_start + tl.minimum(r0 + BT, T) - 1)
    if (s0 == s1) & (r0 - s0 >= 3):
        acc = _conv_taps(x_ptr, xs_ptr, w_ptr, state_ptr, rows, rm, offs, s0, 0, XS, C, BT, BLOCK_C, False)
    else:
        start = tl.load(row_start + rows, mask=rm, other=0)
        slot = tl.load(row_slot + rows, mask=rm, other=0).to(tl.int64)
        acc = _conv_taps(x_ptr, xs_ptr, w_ptr, state_ptr, rows, rm, offs, start, slot, XS, C, BT, BLOCK_C, True)
    _conv_out(acc, rows, rm, c0, q_out, k_out, v_out, QK, VC, BT, BLOCK_C, DK)


@triton.jit
def _conv_window_val(x_ptr, xs_ptr, st, t0, n, j, offs, C, XS):
    p = n - 3 + j
    xv = (tl.load(x_ptr + (t0 + p) * XS + offs, mask=p >= 0, other=0.0).to(tl.float32)
          * tl.load(xs_ptr + t0 + p, mask=p >= 0, other=0.0)).to(tl.bfloat16)
    sv = tl.load(st + (n + j) * C + offs, mask=p < 0, other=0.0)
    return tl.where(p >= 0, xv, sv)


@triton.jit
def _conv1d_state_kernel(x_ptr, xs_ptr, state_ptr, seq_slot, seq_start, seq_len, C, XS, BLOCK_C: tl.constexpr):
    """New conv window of each sequence = its last 3 inputs (older ones from the old window if it fed < 3)."""
    gdc_wait()
    s = tl.program_id(0)
    n = tl.load(seq_len + s)
    if n > 0:
        slot = tl.load(seq_slot + s).to(tl.int64)
        t0 = tl.load(seq_start + s).to(tl.int64)
        offs = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
        st = state_ptr + slot * 3 * C
        v0 = _conv_window_val(x_ptr, xs_ptr, st, t0, n, 0, offs, C, XS)
        v1 = _conv_window_val(x_ptr, xs_ptr, st, t0, n, 1, offs, C, XS)
        v2 = _conv_window_val(x_ptr, xs_ptr, st, t0, n, 2, offs, C, XS)
        tl.store(st + offs, v0)
        tl.store(st + C + offs, v1)
        tl.store(st + 2 * C + offs, v2)


def _conv_outs(T, dev, HK=16, HV=48, DK=128, DV=128):
    bf = torch.bfloat16
    return (torch.empty((T, HK, DK), dtype=bf, device=dev), torch.empty((T, HK, DK), dtype=bf, device=dev),
            torch.empty((T, HV, DV), dtype=bf, device=dev))


def conv1d_varlen(x, xs, w, state, seq_slot, seq_start, seq_len, n_seqs, HK=16, DK=128):
    """x [T, C] (GEMM output, scaled by xs), w [C, 4] fp32, state [S, 3, C] bf16 (updated in place).
    Returns (q, k, v). Sequential per sequence: for decode steps."""
    T, C = x.shape
    assert x.stride(1) == 1
    q, k, v = _conv_outs(T, x.device)
    BLOCK_C = 256
    _conv1d_seq_kernel[(n_seqs, C // BLOCK_C)](x, xs, w, state, q, k, v, seq_slot, seq_start, seq_len,
                                               x.stride(0), QK=HK * DK, VC=C - 2 * HK * DK, BLOCK_C=BLOCK_C, DK=DK, num_warps=2,
                                               launch_pdl=PDL)
    return q, k, v


def conv1d_rows(x, xs, w, state, row_start, row_slot, seq_slot, seq_start, seq_len, n_seqs, HK=16, DK=128,
                update_state=True):
    """Token-parallel conv1d for steps with prefill chunks (any lengths); same result as conv1d_varlen.
    update_state=False: call conv1d_state afterwards (any time after this, before the next step's conv)."""
    T, C = x.shape
    assert x.stride(1) == 1
    q, k, v = _conv_outs(T, x.device)
    BT, BLOCK_C = 16, 128
    _conv1d_rows_kernel[(triton.cdiv(T, BT), C // BLOCK_C)](x, xs, w, state, q, k, v, row_start, row_slot, T, x.stride(0),
                                                            QK=HK * DK, VC=C - 2 * HK * DK, BT=BT, BLOCK_C=BLOCK_C,
                                                            DK=DK, num_warps=4, launch_pdl=PDL)
    if update_state:
        conv1d_state(x, xs, state, seq_slot, seq_start, seq_len, n_seqs)
    return q, k, v


def conv1d_state(x, xs, state, seq_slot, seq_start, seq_len, n_seqs):
    C = x.shape[1]
    _conv1d_state_kernel[(n_seqs, triton.cdiv(C, 1024))](x, xs, state, seq_slot, seq_start, seq_len, C, x.stride(0),
                                                         BLOCK_C=1024, num_warps=4, launch_pdl=PDL)


# ---------------------------------------------------------------- Gated-DeltaNet: gates
@triton.jit
def _gdn_gates_kernel(a_ptr, b_ptr, xs_ptr, alog_ptr, dtb_ptr, g_ptr, beta_ptr, AS, HV: tl.constexpr,
                      BH: tl.constexpr):
    gdc_wait()
    r = tl.program_id(0).to(tl.int64)
    h = tl.arange(0, BH)
    m = h < HV
    xs = tl.load(xs_ptr + r)
    a = _rbf16(tl.load(a_ptr + r * AS + h, mask=m, other=0.0).to(tl.float32) * xs)
    b = _rbf16(tl.load(b_ptr + r * AS + h, mask=m, other=0.0).to(tl.float32) * xs)
    x = a + tl.load(dtb_ptr + h, mask=m, other=0.0)
    sp = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
    tl.store(g_ptr + r * HV + h, -tl.exp(tl.load(alog_ptr + h, mask=m, other=0.0)) * sp, mask=m)
    tl.store(beta_ptr + r * HV + h, tl.sigmoid(b), mask=m)


def gdn_gates(a, b, xs, A_log, dt_bias, HV=48):
    """g = -exp(A_log) * softplus(a + dt_bias), beta = sigmoid(b) (fp32 [T, HV]); a, b are GEMM outputs / xs."""
    T = a.shape[0]
    assert a.stride(0) == b.stride(0) and a.stride(1) == b.stride(1) == 1
    g = torch.empty((T, HV), dtype=torch.float32, device=a.device)
    beta = torch.empty_like(g)
    _gdn_gates_kernel[(T,)](a, b, xs, A_log, dt_bias, g, beta, a.stride(0), HV=HV, BH=triton.next_power_of_2(HV), num_warps=1,
                            launch_pdl=PDL)
    return g, beta


# ---------------------------------------------------------------- Gated-DeltaNet: fused decode step
@triton.jit
def _gdn_decode_kernel(q_ptr, k_ptr, v_ptr, a_ptr, b_ptr, xs_ptr, alog_ptr, dtb_ptr, state_ptr, o_ptr, row_slot,
                       row0_ptr, AS, QS, VS, HV: tl.constexpr, HK: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr,
                       BV: tl.constexpr, HPP: tl.constexpr, STAGES: tl.constexpr, NORM: tl.constexpr, PF: tl.constexpr):
    """One program per (V block, group of HPP value heads, decode row): gates + one gated delta-rule step on the
    slot's [DK, DV] states (bf16 storage, fp32 math), written back in place; the head loop is software-pipelined
    (the next head's state load overlaps this head's math). q, k are l2-normalized already, or NORM: here (as
    _conv_out does, then rounded to bf16 as if stored). The V block is the
    fastest grid axis, so the programs sharing a state's rows run together (DRAM locality: +3-5% bandwidth at
    BV 64, +50% at BV 32 vs row-fastest)."""
    gdc_wait()
    r = tl.load(row0_ptr).to(tl.int64) + tl.program_id(2)
    vb = tl.program_id(0)
    slot = tl.load(row_slot + r).to(tl.int64)
    if slot >= 0:
        dk = tl.arange(0, DK)
        dv = vb * BV + tl.arange(0, BV)
        xs = tl.load(xs_ptr + r)
        hv0 = tl.program_id(1) * HPP
        if PF:
            vn = tl.load(v_ptr + r * VS + hv0 * DV + dv)
            an = tl.load(a_ptr + r * AS + hv0)
            bn = tl.load(b_ptr + r * AS + hv0)
        for i in tl.range(0, HPP, num_stages=STAGES):
            hv = hv0 + i
            kh = hv // (HV // HK)
            sp_ = state_ptr + ((slot * HV + hv) * DK + dk[:, None]) * DV + dv[None, :]
            S = tl.load(sp_, eviction_policy="evict_first").to(tl.float32)
            q = tl.load(q_ptr + r * QS + kh * DK + dk).to(tl.float32)
            k = tl.load(k_ptr + r * QS + kh * DK + dk).to(tl.float32)
            if NORM:
                q = _rbf16(q * tl.rsqrt(tl.sum(q * q, 0) + 1e-6))
                k = _rbf16(k * tl.rsqrt(tl.sum(k * k, 0) + 1e-6))
            q = q * (1.0 / tl.sqrt(DK * 1.0))
            if PF:
                # v, a, b are < 4 bytes per thread, too small for the cp.async pipeline: a plain load would stall
                # every head on its latency, so the next head's are loaded into registers one head early
                v, a, b = vn.to(tl.float32), an.to(tl.float32), bn.to(tl.float32)
                hn = hv0 + tl.minimum(i + 1, HPP - 1)
                vn = tl.load(v_ptr + r * VS + hn * DV + dv)
                an = tl.load(a_ptr + r * AS + hn)
                bn = tl.load(b_ptr + r * AS + hn)
            else:
                v = tl.load(v_ptr + r * VS + hv * DV + dv).to(tl.float32)
                a = tl.load(a_ptr + r * AS + hv).to(tl.float32)
                b = tl.load(b_ptr + r * AS + hv).to(tl.float32)
            a = _rbf16(a * xs)
            b = _rbf16(b * xs)
            x = a + tl.load(dtb_ptr + hv)
            sp = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
            g = -tl.exp(tl.load(alog_ptr + hv)) * sp
            beta = tl.sigmoid(b)
            S = S * tl.exp(g)
            kv = tl.sum(S * k[:, None], 0)
            delta = (v - kv) * beta
            S = S + k[:, None] * delta[None, :]
            o = tl.sum(S * q[:, None], 0)
            tl.store(sp_, S.to(tl.bfloat16), cache_modifier=".cs")
            tl.store(o_ptr + (r * HV + hv) * DV + dv, o.to(tl.bfloat16))


# v, a, b of the next head loaded one head early (PF, bit-exact): 56 rows 69.6 -> 66.9 us, 96 rows 114 -> 110,
# GDN branch of a mixed step (prefill || decode, scratch/ovl.py) 167 -> 161 us (scratch/decpf.py, ncu: 27% of the stall
# samples were the store waiting on the plain v load)
GDN_DECODE_PF = int(os.environ.get("AGENTOPT_GDN_DECODE_PF", "1"))
GDN_DECODE_CFG = None  # override for benchmarks; default: by row count (bench/micro_gdn.py)


def _gdn_decode_cfg(n_rows):
    if GDN_DECODE_CFG:
        return GDN_DECODE_CFG
    # Tuned inside the model (scratch/gdn_dec_inmodel.sh, prof_mixed.py), not in isolation: under the power cap
    # (lower SM clocks) the software-pipelined head loop wins (128 rows 7.31 -> 6.87 ms, 56 decode rows of a mixed
    # step 4.05 -> 3.41 ms per step), while isolated microbenchmarks show a tie.
    if n_rows <= 8:
        return {"BV": 32, "num_warps": 1, "HPP": 1, "STAGES": 1}
    if n_rows < 24:
        return {"BV": 64, "num_warps": 2, "HPP": 1, "STAGES": 1}
    if n_rows < 48:
        return {"BV": 64, "num_warps": 2, "HPP": 2, "STAGES": 2}
    return {"BV": 64, "num_warps": 2, "HPP": 4, "STAGES": 3}


def gdn_decode(q, k, v, a, b, xs, A_log, dt_bias, state, row_slot, row0, n_rows, o, HV=48, HK=16, DK=128, DV=128,
               norm=False):
    """Rows row0 + [0, n_rows) with row_slot >= 0 are decode rows (row0: device int32 scalar).
    o [T, HV, DV] is written for those rows. q, k [T, HK * DK] / v [T, HV * DV]: unit column stride (column slices
    of one buffer ok); norm: q, k not l2-normalized yet."""
    c = _gdn_decode_cfg(n_rows)
    assert a.stride(0) == b.stride(0) and a.stride(1) == b.stride(1) == 1
    q, k, v = q.reshape(q.shape[0], -1), k.reshape(k.shape[0], -1), v.reshape(v.shape[0], -1)
    assert q.stride(0) == k.stride(0) and q.stride(1) == k.stride(1) == v.stride(1) == 1
    _gdn_decode_kernel[(DV // c["BV"], HV // c["HPP"], n_rows)](
        q, k, v, a, b, xs, A_log, dt_bias, state, o, row_slot, row0, a.stride(0), q.stride(0), v.stride(0), HV=HV,
        HK=HK, DK=DK, DV=DV, BV=c["BV"], HPP=c["HPP"], STAGES=c["STAGES"], NORM=norm, PF=GDN_DECODE_PF and c["HPP"] > 1,
        num_warps=c["num_warps"],
        launch_pdl=PDL)
    return o


# ---------------------------------------------------------------- Gated-DeltaNet: chunked prefill
# fla's chunked algorithm (WY representation) with fla's fused kkt + solve_tril kernel for A = (I + tril(beta K K^T
# exp(g_i - g_j)))^-1; the rest (fla's recompute_w_u + chunk_fwd_h + chunk_fwd_o) is our own fused kernel, so that
# w / u are never materialized, the per-chunk states stay in registers, the initial / final state is read / written
# in place in the slot pool (bf16) and padded sequences (length 0) cost nothing. All sizes are fixed by the caller
# (padded cu_seqlens / chunk_indices), so the whole step can be captured in a CUDA graph.
# A = (I + X)^-1, X = strictly-lower(beta_i k_i.k_j exp2(g_i - g_j)) per 64-token chunk and value head (fla's
# kkt + solve_tril, same math): k k^T in 16-row blocks, gate cumsum fused, 16 x 16 diagonal blocks inverted in fp32
# by forward substitution, block merges in tf32 like fla. One program per chunk and HPP value heads sharing one K head
# (GDN_KKT_CFG). Same error vs an fp64 inverse as fla, also on correlated keys (scratch/kkt_adv.py).
@triton.jit
def _fs_row(x, ai, i, o_c):
    # row i of (I + x)^-1 - I by forward substitution, given rows < i (fp32: the Neumann-product form
    # (I - x)(I + x^2)(I + x^4)(I + x^8) has intermediates up to ~C(15, 8) times the result for correlated keys and
    # lost all accuracy in tf32 on long prompts)
    a = tl.sum(tl.where((o_c == i)[:, None], -x, 0.0), 0)
    a = tl.where(o_c < i, a, 0.0)
    a = a + tl.sum(a[:, None] * ai, 0)
    return tl.where((o_c == i)[:, None], a[None, :], ai)


@triton.jit
def _gate_cum(g_raw, a_raw, b_raw, xs_ptr, alog_ptr, dtb_ptr, AS, gscale, bos, o_t, m_t, hv, HV: tl.constexpr,
              GATES: tl.constexpr):
    """Chunk-local cumsum of the log2 gates of value head hv (as fla's chunk_local_cumsum), and beta (GATES: the gates
    as _gdn_gates_kernel from the GEMM outputs a, b (/ xs): g = -exp(A_log) softplus(a + dt_bias), beta = sigmoid(b))."""
    pg = (bos + o_t).to(tl.int64) * HV + hv
    if GATES:
        rr = (bos + o_t).to(tl.int64)
        xsv = tl.load(xs_ptr + rr, mask=m_t, other=0.0)
        ga = _rbf16(tl.load(a_raw + rr * AS + hv, mask=m_t, other=0.0).to(tl.float32) * xsv)
        gb = _rbf16(tl.load(b_raw + rr * AS + hv, mask=m_t, other=0.0).to(tl.float32) * xsv)
        gx = ga + tl.load(dtb_ptr + hv)
        gr = -tl.exp(tl.load(alog_ptr + hv)) * tl.where(gx > 20.0, gx, tl.log(1.0 + tl.exp(gx)))
        bt = tl.sigmoid(gb)
        gr = tl.where(m_t, gr, 0.0)
    else:
        gr = tl.load(g_raw + pg, mask=m_t, other=0.0)
        bt = gr
    return tl.cumsum(gr, axis=0) * gscale, bt


@triton.jit
def _gdn_aq(q, k, Aq, g_raw, a_raw, b_raw, xs_ptr, alog_ptr, dtb_ptr, AS, gscale, bos, T, i_t, i_h,
            H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, BT: tl.constexpr, GATES: tl.constexpr):
    """Intra-chunk attention of the chunk kernel, tril(q k^T exp2(g_i - g_j)) (bf16), for the HV / H value heads
    sharing K head i_h: no h dependence, so off the chunk kernel's serial chain. Run by extra programs of the kkt
    launch (own gate cumsum, same code: no wait for the kkt programs). Same dots as the chunk kernel computed it
    (mma.sync / wgmma, any warp count: bit-identical)."""
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    o_k1 = tl.arange(0, 64)
    o_k2 = 64 + o_k1
    o_a = tl.arange(0, BT)
    q += (bos * H + i_h).to(tl.int64) * K
    k += (bos * H + i_h).to(tl.int64) * K
    b_q = tl.load(q + o_t[:, None] * (H * K) + o_k1[None, :], mask=m_t[:, None], other=0.0)
    b_k = tl.load(k + o_k1[:, None] + o_t[None, :] * (H * K), mask=m_t[None, :], other=0.0)
    b_A = tl.dot(b_q, b_k)
    b_q = tl.load(q + o_t[:, None] * (H * K) + o_k2[None, :], mask=m_t[:, None], other=0.0)
    b_k = tl.load(k + o_k2[:, None] + o_t[None, :] * (H * K), mask=m_t[None, :], other=0.0)
    b_A += tl.dot(b_q, b_k)
    m = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)
    for j in tl.static_range(HV // H):
        hv = i_h * (HV // H) + j
        b_g, _ = _gate_cum(g_raw, a_raw, b_raw, xs_ptr, alog_ptr, dtb_ptr, AS, gscale, bos, o_t, m_t, hv, HV, GATES)
        a = tl.where(m, b_A * tl.exp2(b_g[:, None] - b_g[None, :]), 0).to(tl.bfloat16)
        tl.store(Aq + ((bos + o_t) * HV + hv).to(tl.int64)[:, None] * BT + o_a[None, :], a, mask=m_t[:, None])


@triton.jit
def _gx(kk, ga, gb, ba, ma, mb, STRICT: tl.constexpr, o_c):
    m = ma[:, None] & mb[None, :]
    if STRICT:
        m = m & (o_c[:, None] > o_c[None, :])
    return tl.where(m, kk * tl.exp2(ga[:, None] - gb[None, :]), 0.0) * ba[:, None]


@triton.jit
def _gdn_kkt_kernel(k, g_raw, g, beta, A, cu_seqlens, chunk_indices, gscale, H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr,
               BT: tl.constexpr, HPP: tl.constexpr, PREC: tl.constexpr, a_raw=None, b_raw=None, xs_ptr=None,
               alog_ptr=None, dtb_ptr=None, AS=0, GATES: tl.constexpr = 0, q=None, Aq=None, AQ: tl.constexpr = 0):
    gdc_wait()
    i_t, i_hg = tl.program_id(0), tl.program_id(1)
    i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
    i_t = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
    bos = tl.load(cu_seqlens + i_n).to(tl.int32)
    T = tl.load(cu_seqlens + i_n + 1).to(tl.int32) - bos
    if i_t * BT >= T:
        return
    if AQ:
        # programs [0, H): the chunk kernel's intra-chunk attention (first: the longer programs)
        if i_hg < H:
            _gdn_aq(q, k, Aq, g_raw, a_raw, b_raw, xs_ptr, alog_ptr, dtb_ptr, AS, gscale, bos, T, i_t, i_hg,
                    H, HV, K, BT, GATES)
            return
        i_hg -= H
    BC: tl.constexpr = 16
    o_c = tl.arange(0, BC)
    t0 = i_t * BT + o_c
    t1 = t0 + BC
    t2 = t1 + BC
    t3 = t2 + BC
    m0 = t0 < T
    m1 = t1 < T
    m2 = t2 < T
    m3 = t3 < T
    G: tl.constexpr = HV // H
    tl.static_assert(G % HPP == 0)
    hv0 = i_hg * HPP
    kh = hv0 // G
    o_k = tl.arange(0, K)
    k0 = tl.load(k + ((bos + t0) * H + kh).to(tl.int64)[:, None] * K + o_k[None, :], mask=m0[:, None], other=0.0)
    k1 = tl.load(k + ((bos + t1) * H + kh).to(tl.int64)[:, None] * K + o_k[None, :], mask=m1[:, None], other=0.0)
    k2 = tl.load(k + ((bos + t2) * H + kh).to(tl.int64)[:, None] * K + o_k[None, :], mask=m2[:, None], other=0.0)
    k3 = tl.load(k + ((bos + t3) * H + kh).to(tl.int64)[:, None] * K + o_k[None, :], mask=m3[:, None], other=0.0)
    kk00 = tl.dot(k0, tl.trans(k0))
    kk10 = tl.dot(k1, tl.trans(k0))
    kk11 = tl.dot(k1, tl.trans(k1))
    kk20 = tl.dot(k2, tl.trans(k0))
    kk21 = tl.dot(k2, tl.trans(k1))
    kk22 = tl.dot(k2, tl.trans(k2))
    kk30 = tl.dot(k3, tl.trans(k0))
    kk31 = tl.dot(k3, tl.trans(k1))
    kk32 = tl.dot(k3, tl.trans(k2))
    kk33 = tl.dot(k3, tl.trans(k3))
    eye = (o_c[:, None] == o_c[None, :]).to(tl.float32)
    z = tl.zeros((BC, BC), A.dtype.element_ty)
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    for j in tl.static_range(HPP):
        # gate cumsum stored for the chunk kernel and re-read in 16-row pieces
        gcum, bt = _gate_cum(g_raw, a_raw, b_raw, xs_ptr, alog_ptr, dtb_ptr, AS, gscale, bos, o_t, m_t, hv0 + j, HV, GATES)
        pg = (bos + o_t).to(tl.int64) * HV + hv0 + j
        if GATES:
            tl.store(beta + pg, bt, mask=m_t)
        tl.store(g + pg, gcum, mask=m_t)
    tl.debug_barrier()
    for j in tl.static_range(HPP):
        hv = hv0 + j
        g0 = tl.load(g + (bos + t0).to(tl.int64) * HV + hv, mask=m0, other=0.0)
        g1 = tl.load(g + (bos + t1).to(tl.int64) * HV + hv, mask=m1, other=0.0)
        g2 = tl.load(g + (bos + t2).to(tl.int64) * HV + hv, mask=m2, other=0.0)
        g3 = tl.load(g + (bos + t3).to(tl.int64) * HV + hv, mask=m3, other=0.0)
        b0 = tl.load(beta + (bos + t0).to(tl.int64) * HV + hv, mask=m0, other=0.0)
        b1 = tl.load(beta + (bos + t1).to(tl.int64) * HV + hv, mask=m1, other=0.0)
        b2 = tl.load(beta + (bos + t2).to(tl.int64) * HV + hv, mask=m2, other=0.0)
        b3 = tl.load(beta + (bos + t3).to(tl.int64) * HV + hv, mask=m3, other=0.0)
        x00 = _gx(kk00, g0, g0, b0, m0, m0, True, o_c)
        x11 = _gx(kk11, g1, g1, b1, m1, m1, True, o_c)
        x22 = _gx(kk22, g2, g2, b2, m2, m2, True, o_c)
        x33 = _gx(kk33, g3, g3, b3, m3, m3, True, o_c)
        i00 = -x00
        i11 = -x11
        i22 = -x22
        i33 = -x33
        for i in range(2, BC):
            i00 = _fs_row(x00, i00, i, o_c)
            i11 = _fs_row(x11, i11, i, o_c)
            i22 = _fs_row(x22, i22, i, o_c)
            i33 = _fs_row(x33, i33, i, o_c)
        i00 += eye
        i11 += eye
        i22 += eye
        i33 += eye
        x10 = _gx(kk10, g1, g0, b1, m1, m0, False, o_c)
        x32 = _gx(kk32, g3, g2, b3, m3, m2, False, o_c)
        x20 = _gx(kk20, g2, g0, b2, m2, m0, False, o_c)
        x21 = _gx(kk21, g2, g1, b2, m2, m1, False, o_c)
        x30 = _gx(kk30, g3, g0, b3, m3, m0, False, o_c)
        x31 = _gx(kk31, g3, g1, b3, m3, m1, False, o_c)
        i10 = -tl.dot(tl.dot(i11, x10, input_precision=PREC), i00, input_precision=PREC)
        i32 = -tl.dot(tl.dot(i33, x32, input_precision=PREC), i22, input_precision=PREC)
        # lower-left 32 x 32: -inv(B) L inv(A)
        m20 = tl.dot(x20, i00, input_precision=PREC) + tl.dot(x21, i10, input_precision=PREC)
        m21 = tl.dot(x21, i11, input_precision=PREC)
        m30 = tl.dot(x30, i00, input_precision=PREC) + tl.dot(x31, i10, input_precision=PREC)
        m31 = tl.dot(x31, i11, input_precision=PREC)
        i20 = -tl.dot(i22, m20, input_precision=PREC)
        i21 = -tl.dot(i22, m21, input_precision=PREC)
        i30 = -(tl.dot(i32, m20, input_precision=PREC) + tl.dot(i33, m30, input_precision=PREC))
        i31 = -(tl.dot(i32, m21, input_precision=PREC) + tl.dot(i33, m31, input_precision=PREC))
        pa = A + hv * BT + o_c[None, :]
        r0 = pa + ((bos + t0) * HV).to(tl.int64)[:, None] * BT
        r1 = pa + ((bos + t1) * HV).to(tl.int64)[:, None] * BT
        r2 = pa + ((bos + t2) * HV).to(tl.int64)[:, None] * BT
        r3 = pa + ((bos + t3) * HV).to(tl.int64)[:, None] * BT
        dt = A.dtype.element_ty
        tl.store(r0, i00.to(dt), mask=m0[:, None]); tl.store(r0 + 16, z, mask=m0[:, None])
        tl.store(r0 + 32, z, mask=m0[:, None]); tl.store(r0 + 48, z, mask=m0[:, None])
        tl.store(r1, i10.to(dt), mask=m1[:, None]); tl.store(r1 + 16, i11.to(dt), mask=m1[:, None])
        tl.store(r1 + 32, z, mask=m1[:, None]); tl.store(r1 + 48, z, mask=m1[:, None])
        tl.store(r2, i20.to(dt), mask=m2[:, None]); tl.store(r2 + 16, i21.to(dt), mask=m2[:, None])
        tl.store(r2 + 32, i22.to(dt), mask=m2[:, None]); tl.store(r2 + 48, z, mask=m2[:, None])
        tl.store(r3, i30.to(dt), mask=m3[:, None]); tl.store(r3 + 16, i31.to(dt), mask=m3[:, None])
        tl.store(r3 + 32, i32.to(dt), mask=m3[:, None]); tl.store(r3 + 48, i33.to(dt), mask=m3[:, None])




@triton.jit
def _gdn_chunk_kernel(q, k, v, A, beta, g, o, state, seq_slot, cu_seqlens, scale, Aq,
                      H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr,
                      BV: tl.constexpr, AQ: tl.constexpr):
    """Inter-chunk recurrence fused with the chunk outputs: per chunk, v_new = u - w h with fla's u = A (beta v),
    w = A (beta exp(g) k) computed as v_new = A (beta (v - exp(g) k h)) (one 64 x 64 dot instead of two, no w / u
    in HBM; bf16 MMA inputs as in fla), o = scale (q h exp(g) + tril(q k^T exp(g_i - g_j)) v_new), then
    h <- h exp(g_last) + k^T (v_new exp(g_last - g)). The per-chunk states never leave registers."""
    gdc_wait()
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // HV, i_nh % HV
    bos = tl.load(cu_seqlens + i_n).to(tl.int32)
    T = tl.load(cu_seqlens + i_n + 1).to(tl.int32) - bos
    if T > 0:
        NT = tl.cdiv(T, BT)
        slot = tl.load(seq_slot + i_n).to(tl.int64)
        o_v = i_v * BV + tl.arange(0, BV)
        o_k1 = tl.arange(0, 64)
        o_k2 = 64 + o_k1
        o_a = tl.arange(0, BT)
        st = state + (slot * HV + i_h) * K * V
        b_h1 = tl.load(st + o_k1[:, None] * V + o_v[None, :]).to(tl.float32)
        b_h2 = tl.load(st + o_k2[:, None] * V + o_v[None, :]).to(tl.float32)
        v += (bos * HV + i_h).to(tl.int64) * V
        o += (bos * HV + i_h).to(tl.int64) * V
        q += (bos * H + i_h // (HV // H)).to(tl.int64) * K
        k += (bos * H + i_h // (HV // H)).to(tl.int64) * K
        A += (bos * HV + i_h).to(tl.int64) * BT
        Aq += (bos * HV + i_h).to(tl.int64) * BT
        for i_t in range(NT):
            o_t = i_t * BT + tl.arange(0, BT)
            m_t = o_t < T
            h1 = b_h1.to(tl.bfloat16)
            h2 = b_h2.to(tl.bfloat16)
            b_g = tl.load(g + (bos + o_t).to(tl.int64) * HV + i_h, mask=m_t, other=0.0).to(tl.float32)
            b_b = tl.load(beta + (bos + o_t).to(tl.int64) * HV + i_h, mask=m_t, other=0.0).to(tl.float32)
            b_k1 = tl.load(k + o_k1[:, None] + o_t[None, :] * (H * K), mask=m_t[None, :], other=0.0)
            b_k2 = tl.load(k + o_k2[:, None] + o_t[None, :] * (H * K), mask=m_t[None, :], other=0.0)
            b_v = tl.dot(tl.trans(b_k1), h1)
            b_v += tl.dot(tl.trans(b_k2), h2)
            b_v = tl.load(v + o_t[:, None] * (HV * V) + o_v[None, :], mask=m_t[:, None], other=0.0).to(tl.float32) - b_v * tl.exp2(b_g)[:, None]
            b_Aw = tl.load(A + o_t[:, None] * (HV * BT) + o_a[None, :], mask=m_t[:, None], other=0.0)
            b_v = tl.dot(b_Aw, (b_v * b_b[:, None]).to(tl.bfloat16))
            if AQ:
                b_q = tl.load(q + o_t[:, None] * (H * K) + o_k1[None, :], mask=m_t[:, None], other=0.0)
                b_o = tl.dot(b_q, h1)
                b_q = tl.load(q + o_t[:, None] * (H * K) + o_k2[None, :], mask=m_t[:, None], other=0.0)
                b_o += tl.dot(b_q, h2)
                b_o = b_o * tl.exp2(b_g)[:, None]
                b_A = tl.load(Aq + o_t[:, None] * (HV * BT) + o_a[None, :], mask=m_t[:, None], other=0.0)
            else:
                b_q = tl.load(q + o_t[:, None] * (H * K) + o_k1[None, :], mask=m_t[:, None], other=0.0)
                b_o = tl.dot(b_q, h1)
                b_A = tl.dot(b_q, b_k1)
                b_q = tl.load(q + o_t[:, None] * (H * K) + o_k2[None, :], mask=m_t[:, None], other=0.0)
                b_o += tl.dot(b_q, h2)
                b_A += tl.dot(b_q, b_k2)
                b_o = b_o * tl.exp2(b_g)[:, None]
                b_A = b_A * tl.exp2(b_g[:, None] - b_g[None, :])
                b_A = tl.where((o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t), b_A, 0).to(tl.bfloat16)
            b_o = b_o * scale + tl.dot(b_A, b_v.to(tl.bfloat16)) * scale
            tl.store(o + o_t[:, None] * (HV * V) + o_v[None, :], b_o.to(tl.bfloat16), mask=m_t[:, None])
            b_g_last = tl.load(g + (bos + tl.minimum((i_t + 1) * BT, T) - 1).to(tl.int64) * HV + i_h).to(tl.float32)
            b_v = (b_v * tl.where(m_t, tl.exp2(b_g_last - b_g), 0)[:, None]).to(tl.bfloat16)
            b_g_last = tl.exp2(b_g_last)
            b_h1 = b_h1 * b_g_last + tl.dot(b_k1, b_v)
            b_h2 = b_h2 * b_g_last + tl.dot(b_k2, b_v)
        tl.store(st + o_k1[:, None] * V + o_v[None, :], b_h1.to(tl.bfloat16))
        tl.store(st + o_k2[:, None] * V + o_v[None, :], b_h2.to(tl.bfloat16))


# Sweep (scratch/gdnb2.py): BV 64 / 4 warps / 3 stages is best or tied for 1 x 1992, 2 x 1024, 4 x 512 and
# 300 + 1700 tokens (one long sequence: 188 vs 261 us for BV 32 / 2 warps). Only 96 programs for one sequence, but
# 4-warp dots are much faster per chunk. (BV 16/32 with num_warps >= 4 and num_stages=1 miscompile in Triton 3.6:
# out-of-bounds shared read.)
GDN_CHUNK_CFG = {"BV": 64, "num_warps": 4, "num_stages": 3}
# intra-chunk attention tril(q k^T ...) by extra programs of the kkt launch (_gdn_aq) instead of on the chunk kernel's
# warps (bit-exact): kkt + chunk 1992 tokens 149 -> 119 us, 4000 268 -> 231, 4 x 512 132 -> 115, 40 tokens 20.0 ->
# 18.5 (scratch/gdnaq.py; GATES mode 153 -> 121, scratch/gdnaq2.py, aq programs first in the grid: -4 us). As its own
# kernel after kkt: 130 us; computed by the kkt programs themselves: 170 us.
GDN_AQ = int(os.environ.get("AGENTOPT_GDN_AQ", "1"))
# (Re-checked for the fused w / u version, scratch/gdnfuse.py: 1992 tokens 189 -> 160 us, 4000: 350 -> 289 us,
# 4 x 512: 177 -> 142 us per layer including kkt_solve; num_stages 2 / 4, 8 warps and BV 32 are slower. Masking the
# upper triangle of A in-kernel instead of zero-filling it is slower: the masked load stalls the serial chain.)




# (HPP, num_warps). In gdn_prefill (scratch/kkt_small.py) one head per program and 1 warp is best at every size:
# small steps are one wave, so per-program latency decides. gdn_prefill per layer (vs fla's kkt): 2 chunks 25.5 us
# (27.2), 1992 tokens 143 us (163), 4 x 512 129 us (144).
GDN_KKT_CFG = (1, 1)


def gdn_prefill(q, k, v, g, beta, state, seq_slot, cu_seqlens, chunk_indices, n_seqs,
                scale=128 ** -0.5, o=None, after_kkt=None, gates=None):
    """Chunked gated delta rule over varlen sequences in rows [0, cu_seqlens[-1]) of q/k [T, HK, DK],
    v [T, HV, DV], g/beta fp32 [T, HV]; state [S, HV, DK, DV] bf16 indexed by seq_slot (updated in place).
    cu_seqlens [n_seqs + 1], chunk_indices [NT, 2] (int64, fixed sizes; padded
    sequences have length 0, padded chunk entries point to a padded sequence). Returns o [T, HV, DV] (rows outside
    the sequences are undefined). gates=(a, b, xs, A_log, dt_bias): g / beta computed in the kkt kernel from the GEMM
    outputs (pass g = beta = None)."""
    from fla.ops.utils.constant import RCP_LN2
    T, HK, DK = k.shape
    HV, DV = v.shape[1], v.shape[2]
    assert k.is_contiguous() and v.is_contiguous()
    gx = {}
    if gates is not None:
        a, b, xs, A_log, dt_bias = gates
        assert a.stride(0) == b.stride(0) and a.stride(1) == b.stride(1) == 1
        gc = torch.empty((T, HV), dtype=torch.float32, device=k.device)
        beta = torch.empty_like(gc)
        gx = dict(a_raw=a, b_raw=b, xs_ptr=xs, alog_ptr=A_log, dtb_ptr=dt_bias, AS=a.stride(0), GATES=1)
    else:
        assert g.is_contiguous() and beta.is_contiguous()
        gc = torch.empty_like(g)
    A = torch.empty(T, HV, 64, device=k.device, dtype=k.dtype)
    aq = GDN_AQ
    Aq = torch.empty(T, HV, 64, device=k.device, dtype=k.dtype) if aq else A
    hpp, nw = GDN_KKT_CFG
    _gdn_kkt_kernel[(chunk_indices.shape[0], (HK if aq else 0) + HV // hpp)](
        k, g, gc, beta, A, cu_seqlens, chunk_indices, RCP_LN2, H=HK, HV=HV, K=DK, BT=64, HPP=hpp, PREC="tf32",
        num_warps=nw, launch_pdl=PDL, q=q, Aq=Aq, AQ=aq, **gx)
    if after_kkt is not None:
        after_kkt()
    if o is None:
        o = torch.empty_like(v)
    c = GDN_CHUNK_CFG
    _gdn_chunk_kernel[(DV // c["BV"], n_seqs * HV)](q, k, v, A, beta, gc, o, state, seq_slot, cu_seqlens, scale, Aq,
                                                   H=HK, HV=HV, K=DK, V=DV, BT=64, BV=c["BV"], AQ=aq,
                                                   num_warps=c["num_warps"], num_stages=c["num_stages"],
                                                   launch_pdl=PDL)
    return o


# ---------------------------------------------------------------- Gated RMSNorm (per head) * SiLU(z) + quant
@triton.jit
def _gated_norm_quant_kernel(o_ptr, z_ptr, xs_ptr, w_ptr, q_ptr, s_ptr, eps, ZS,
                             H: tl.constexpr, D: tl.constexpr, BH: tl.constexpr):
    gdc_wait()
    row = tl.program_id(0).to(tl.int64)
    h = tl.arange(0, BH)
    d = tl.arange(0, D)
    m = (h < H)[:, None]
    off = row * H * D + h[:, None] * D + d[None, :]
    o = tl.load(o_ptr + off, mask=m, other=0.0).to(tl.float32)
    z = _rbf16(tl.load(z_ptr + row * ZS + h[:, None] * D + d[None, :], mask=m, other=0.0).to(tl.float32) * tl.load(xs_ptr + row))
    var = tl.sum(o * o, 1) / D
    w = tl.load(w_ptr + d)
    y = o * tl.rsqrt(var + eps)[:, None] * w[None, :] * (z * tl.sigmoid(z))
    y = _rbf16(y)
    s = tl.maximum(tl.max(tl.max(tl.where(m, tl.abs(y), 0.0), 1), 0), 1e-12) / 448.0
    tl.store(q_ptr + off, tl.clamp(y / s, -448.0, 448.0).to(tl.float8e4nv), mask=m)
    tl.store(s_ptr + row, s)


def gated_norm_quant(o, z, xs, w, eps, H=48, D=128):
    T = o.shape[0]
    assert z.stride(1) == 1
    q = torch.empty((T, H * D), dtype=FP8, device=o.device)
    s = torch.empty((T,), dtype=torch.float32, device=o.device)
    _gated_norm_quant_kernel[(T,)](o, z, xs, w, q, s, eps, z.stride(0), H=H, D=D, BH=triton.next_power_of_2(H), num_warps=8,
                                   launch_pdl=PDL)
    return q, s


# ---------------------------------------------------------------- attention: q/k norm + RoPE + KV write
@triton.jit
def _qkv_prep_kernel(qr_ptr, kr_ptr, vr_ptr, xs_ptr, pos_ptr, slot_map, qn_ptr, kn_ptr, q_out, k_cache, v_cache,
                     eps, log2_theta, QS, KS,
                     HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr, R: tl.constexpr):
    """One program per (token, head) over HQ query + HKV key + HKV value heads. q_raw rows are [HQ, 2D]
    (query | output gate per head). Query/key: RMSNorm(1 + w) over D, RoPE on the first R dims (rotate-half),
    query -> q_out, key -> paged k cache; value -> paged v cache."""
    gdc_wait()
    t = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1)
    xs = tl.load(xs_ptr + t)
    half = tl.arange(0, R // 2)
    d = tl.arange(0, D)
    pos = tl.load(pos_ptr + t).to(tl.float32)
    slot = tl.load(slot_map + t).to(tl.int64)
    if h < HQ + HKV:
        if h < HQ:
            src = qr_ptr + t * QS + h * 2 * D
            nw = qn_ptr
        else:
            src = kr_ptr + t * KS + (h - HQ) * D
            nw = kn_ptr
        x = _rbf16(tl.load(src + d).to(tl.float32) * xs)
        rstd = tl.rsqrt(tl.sum(x * x, 0) / D + eps)
        y = (x * rstd * tl.load(nw + d)).to(tl.bfloat16)
        x1 = _rbf16(tl.load(src + half).to(tl.float32) * xs)
        x2 = _rbf16(tl.load(src + R // 2 + half).to(tl.float32) * xs)
        x1 = _rbf16(x1 * rstd * tl.load(nw + half))
        x2 = _rbf16(x2 * rstd * tl.load(nw + R // 2 + half))
        inv_freq = tl.exp2(-(2.0 * half.to(tl.float32) / R) * log2_theta)
        f = pos * inv_freq
        c = tl.cos(f)
        s = tl.sin(f)
        y1 = x1 * c - x2 * s
        y2 = x2 * c + x1 * s
        if h < HQ:
            dst = q_out + (t * HQ + h) * D
        else:
            dst = k_cache + (slot * HKV + (h - HQ)) * D
        tl.store(dst + d, y, mask=d >= R)
        tl.store(dst + half, y1.to(tl.bfloat16))
        tl.store(dst + R // 2 + half, y2.to(tl.bfloat16))
    else:
        hv = h - HQ - HKV
        v = (tl.load(vr_ptr + t * KS + hv * D + d).to(tl.float32) * xs).to(tl.bfloat16)
        tl.store(v_cache + (slot * HKV + hv) * D + d, v)


def qkv_prep(q_raw, k_raw, v_raw, xs, pos, slot_map, qn_w1, kn_w1, k_cache, v_cache, eps, theta,
             HQ=24, HKV=4, D=256, R=64):
    T = q_raw.shape[0]
    assert k_raw.stride(0) == v_raw.stride(0) and q_raw.stride(1) == k_raw.stride(1) == v_raw.stride(1) == 1
    q = torch.empty((T, HQ, D), dtype=torch.bfloat16, device=q_raw.device)
    import math
    _qkv_prep_kernel[(T, HQ + 2 * HKV)](q_raw, k_raw, v_raw, xs, pos, slot_map, qn_w1, kn_w1, q, k_cache, v_cache,
                                        eps, math.log2(theta), q_raw.stride(0), k_raw.stride(0), HQ=HQ, HKV=HKV, D=D, R=R, num_warps=2,
                                        launch_pdl=PDL)
    return q


# ---------------------------------------------------------------- paged attention
@triton.jit
def _attn_decode_kernel(q_ptr, k_cache, v_cache, block_table, row_slot, pos_ptr, row0_ptr, o_ptr, po_ptr, plse_ptr,
                        scale, max_blocks, HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr, BS: tl.constexpr,
                        BLOCK_M: tl.constexpr, NSPLIT: tl.constexpr):
    """One program per (decode row, kv head, KV split): the G = HQ / HKV query heads of the row against a
    contiguous share of the sequence's paged KV (positions [0, pos]). Rows row0 + program_id(0); rows with
    row_slot < 0 are skipped. NSPLIT == 1: writes o; else the split's normalized partial output and log2-sum-exp
    (combined by _attn_decode_combine_kernel)."""
    gdc_wait()
    i = tl.program_id(0)
    r = tl.load(row0_ptr).to(tl.int64) + i
    h = tl.program_id(1)
    sp = tl.program_id(2)
    G: tl.constexpr = HQ // HKV
    slot = tl.load(row_slot + r).to(tl.int64)
    if slot < 0:
        return
    L = tl.load(pos_ptr + r) + 1
    nblk = tl.cdiv(L, BS)
    per = tl.cdiv(nblk, NSPLIT)
    j0 = sp * per
    j1 = tl.minimum(j0 + per, nblk)
    m_ = tl.arange(0, BLOCK_M)
    d = tl.arange(0, D)
    n_ = tl.arange(0, BS)
    rows = m_ < G
    q = tl.load(q_ptr + (r * HQ + h * G + m_)[:, None] * D + d[None, :], mask=rows[:, None], other=0.0)
    m_i = tl.full((BLOCK_M,), -1e30, tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, D), tl.float32)
    qk_scale = scale * 1.4426950408889634
    for j in range(j0, j1):
        blk = tl.load(block_table + slot * max_blocks + j).to(tl.int64)
        kv = ((blk * BS + n_) * HKV + h)[:, None] * D + d[None, :]
        k = tl.load(k_cache + kv)
        s = tl.dot(q, tl.trans(k)) * qk_scale
        s = tl.where((j * BS + n_)[None, :] < L, s, -1e30)
        m_new = tl.maximum(m_i, tl.max(s, 1))
        p = tl.exp2(s - m_new[:, None])
        alpha = tl.exp2(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(v_cache + kv)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
        m_i = m_new
    l_s = tl.where(l_i == 0.0, 1.0, l_i)
    o = acc / l_s[:, None]
    if NSPLIT == 1:
        tl.store(o_ptr + (r * HQ + h * G + m_)[:, None] * D + d[None, :], o.to(tl.bfloat16), mask=rows[:, None])
    else:
        hq = h * G + m_
        tl.store(po_ptr + ((i * HQ + hq) * NSPLIT + sp)[:, None] * D + d[None, :], o, mask=rows[:, None])
        lse = tl.where(l_i == 0.0, -1e30, m_i + tl.log2(l_s))
        tl.store(plse_ptr + (i * HQ + hq) * NSPLIT + sp, lse, mask=rows)


@triton.jit
def _attn_decode_combine_kernel(po_ptr, plse_ptr, row_slot, row0_ptr, o_ptr, HQ: tl.constexpr, D: tl.constexpr,
                                NSPLIT: tl.constexpr):
    gdc_wait()
    i = tl.program_id(0)
    r = tl.load(row0_ptr).to(tl.int64) + i
    hq = tl.program_id(1)
    if tl.load(row_slot + r) < 0:
        return
    sp = tl.arange(0, NSPLIT)
    d = tl.arange(0, D)
    lse = tl.load(plse_ptr + (i * HQ + hq) * NSPLIT + sp)
    w = tl.exp2(lse - tl.max(lse, 0))
    po = tl.load(po_ptr + ((i * HQ + hq) * NSPLIT + sp)[:, None] * D + d[None, :])
    o = tl.sum(po * w[:, None], 0) / tl.sum(w, 0)
    tl.store(o_ptr + (r * HQ + hq) * D + d, o.to(tl.bfloat16))


@triton.jit
def _attn_prefill_kernel(q_ptr, k_cache, v_cache, block_table, tile_tok0, tile_ntok, tile_pos0, tile_slot,
                         o_ptr, scale, max_blocks, n_tiles,
                         HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr, BS: tl.constexpr,
                         BLOCK_M: tl.constexpr, TQ: tl.constexpr):
    """One program per (query tile, kv head). A tile = up to TQ consecutive tokens of one sequence (rows = token x
    query head of the group), causal over the sequence's paged KV. KV blocks entirely below the tile's first
    position need no mask; the last tiles of the list (longest contexts) run first."""
    gdc_wait()
    tile = n_tiles - 1 - tl.program_id(0)
    h = tl.program_id(1)
    G: tl.constexpr = HQ // HKV
    ntok = tl.load(tile_ntok + tile)
    if ntok == 0:  # padding tile
        return
    tok0 = tl.load(tile_tok0 + tile).to(tl.int64)
    pos0 = tl.load(tile_pos0 + tile)
    slot = tl.load(tile_slot + tile).to(tl.int64)
    m_ = tl.arange(0, BLOCK_M)
    tq = m_ // G
    g = m_ % G
    rows = (tq < ntok) & (m_ < TQ * G)
    qpos = pos0 + tq
    d = tl.arange(0, D)
    n_ = tl.arange(0, BS)
    qoff = ((tok0 + tq) * HQ + h * G + g)[:, None] * D + d[None, :]
    q = tl.load(q_ptr + qoff, mask=rows[:, None], other=0.0)
    m_i = tl.full((BLOCK_M,), -1e30, tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, D), tl.float32)
    qk_scale = scale * 1.4426950408889634
    bt = block_table + slot * max_blocks
    n_full = (pos0 + 1) // BS  # blocks whose keys are all <= pos0
    for j in range(0, n_full):
        blk = tl.load(bt + j).to(tl.int64)
        kv = ((blk * BS + n_) * HKV + h)[:, None] * D + d[None, :]
        k = tl.load(k_cache + kv)
        s = tl.dot(q, tl.trans(k)) * qk_scale
        m_new = tl.maximum(m_i, tl.max(s, 1))
        p = tl.exp2(s - m_new[:, None])
        alpha = tl.exp2(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(v_cache + kv)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
        m_i = m_new
    for j in range(n_full, tl.cdiv(pos0 + ntok, BS)):
        blk = tl.load(bt + j).to(tl.int64)
        kv = ((blk * BS + n_) * HKV + h)[:, None] * D + d[None, :]
        k = tl.load(k_cache + kv)
        s = tl.dot(q, tl.trans(k)) * qk_scale
        s = tl.where((j * BS + n_)[None, :] <= qpos[:, None], s, -1e30)
        m_new = tl.maximum(m_i, tl.max(s, 1))
        p = tl.exp2(s - m_new[:, None])
        alpha = tl.exp2(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(v_cache + kv)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
        m_i = m_new
    o = acc / l_i[:, None]
    tl.store(o_ptr + qoff, o.to(tl.bfloat16), mask=rows[:, None])


ATTN_DECODE_NSPLIT = None  # override for benchmarks; default: enough programs to fill the GPU
ATTN_DECODE_CFG = {"num_warps": 4, "num_stages": 2}


def attn_decode(q, k_cache, v_cache, block_table, row_slot, pos, row0, n_rows, o, BS, HQ=24, HKV=4, D=256):
    ns = ATTN_DECODE_NSPLIT or min(32, triton.next_power_of_2(max(1, 264 // (n_rows * HKV))))
    if ns > 1:
        po = torch.empty((n_rows, HQ, ns, D), dtype=torch.float32, device=q.device)
        plse = torch.empty((n_rows, HQ, ns), dtype=torch.float32, device=q.device)
    else:
        po = plse = o
    _attn_decode_kernel[(n_rows, HKV, ns)](q, k_cache, v_cache, block_table, row_slot, pos, row0, o, po, plse,
                                           D ** -0.5, block_table.shape[1], HQ=HQ, HKV=HKV, D=D, BS=BS, BLOCK_M=16,
                                           NSPLIT=ns, **ATTN_DECODE_CFG, launch_pdl=PDL)
    if ns > 1:
        _attn_decode_combine_kernel[(n_rows, HQ)](po, plse, row_slot, row0, o, HQ=HQ, D=D, NSPLIT=ns, num_warps=4,
                                                  launch_pdl=PDL)
    return o


try:  # FA3 (github.com/Dao-AILab/flash-attention hopper/, built with only hdim 256 bf16 forward kernels)
    import flash_attn_3._C  # noqa: F401
    FA3 = torch.ops.flash_attn_3
except ImportError:
    FA3 = None


def attn_prefill_fa3(q, k_cache, v_cache, page_table, seqk, cu, max_q, o, BS, HKV=4, D=256):
    """Same as attn_prefill with FA3: the sequences of cu (q rows cu[i]:cu[i+1], int32) attend causally to their
    seqk[i] cached tokens (pages page_table[i]); writes only their rows of o. max_q: bound on a sequence's rows."""
    NB = k_cache.shape[0] // BS
    FA3.fwd(q, k_cache.view(NB, BS, HKV, D), v_cache.view(NB, BS, HKV, D), None, None, None, o, cu, None, None, None,
            seqk, max_q, None, page_table, None, None, None, None, None, None, None, None, D ** -0.5, True, -1, -1, 0,
            0.0, True, None, 1, None, 0)


PREFILL_CFG = {"BLOCK_M": 128, "num_warps": 8, "num_stages": 2}
PREFILL_TQ = PREFILL_CFG["BLOCK_M"] // 6


def attn_prefill(q, k_cache, v_cache, block_table, tile_tok0, tile_ntok, tile_pos0, tile_slot, n_tiles, o, BS,
                 HQ=24, HKV=4, D=256, cfg=None):
    c = cfg or PREFILL_CFG
    _attn_prefill_kernel[(n_tiles, HKV)](q, k_cache, v_cache, block_table, tile_tok0, tile_ntok, tile_pos0,
                                         tile_slot, o, D ** -0.5, block_table.shape[1], n_tiles, HQ=HQ, HKV=HKV,
                                         D=D, BS=BS, BLOCK_M=c["BLOCK_M"], TQ=c["BLOCK_M"] // (HQ // HKV),
                                         num_warps=c["num_warps"], num_stages=c["num_stages"], launch_pdl=PDL)
    return o


# ---------------------------------------------------------------- attention output gate + quant
@triton.jit
def _attn_gate_quant_kernel(o_ptr, qr_ptr, xs_ptr, q_ptr, s_ptr, QS, H: tl.constexpr, D: tl.constexpr,
                            BH: tl.constexpr):
    gdc_wait()
    row = tl.program_id(0).to(tl.int64)
    h = tl.arange(0, BH)
    d = tl.arange(0, D)
    m = (h < H)[:, None]
    o = tl.load(o_ptr + row * H * D + h[:, None] * D + d[None, :], mask=m, other=0.0).to(tl.float32)
    gate = tl.load(qr_ptr + row * QS + h[:, None] * 2 * D + D + d[None, :], mask=m, other=0.0).to(tl.float32)
    gate = _rbf16(gate * tl.load(xs_ptr + row))
    y = _rbf16(o * tl.sigmoid(gate))
    s = tl.maximum(tl.max(tl.max(tl.where(m, tl.abs(y), 0.0), 1), 0), 1e-12) / 448.0
    tl.store(q_ptr + row * H * D + h[:, None] * D + d[None, :], tl.clamp(y / s, -448.0, 448.0).to(tl.float8e4nv),
             mask=m)
    tl.store(s_ptr + row, s)


def attn_gate_quant(o, q_raw, xs, H=24, D=256):
    T = o.shape[0]
    q = torch.empty((T, H * D), dtype=FP8, device=o.device)
    s = torch.empty((T,), dtype=torch.float32, device=o.device)
    _attn_gate_quant_kernel[(T,)](o, q_raw, xs, q, s, q_raw.stride(0), H=H, D=D, BH=triton.next_power_of_2(H), num_warps=8,
                                  launch_pdl=PDL)
    return q, s
