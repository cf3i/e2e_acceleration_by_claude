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


# ---------------------------------------------------------------- FP8 GEMM for small M (merged weights)
# y[:, n] = (x @ W^T)[:, n] * s[n]: W is several linears sharing an input stacked along N (each keeps its own
# per-tensor scale, repeated in s), so one launch streams all of them. Small M is weight-read bound: narrow N tiles
# give enough programs to fill the GPU, full-K loop, no split. For large M the per-linear cuBLAS calls are faster.
@triton.jit
def _mm_fp8_kernel(x_ptr, w_ptr, s_ptr, y_ptr, M, N, K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                   BK: tl.constexpr, STAGES: tl.constexpr):
    gdc_wait()
    m = tl.program_id(1) * BM + tl.arange(0, BM)
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    mm = m < M
    nm = n < N
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in tl.range(0, K, BK, num_stages=STAGES):
        k = k0 + tl.arange(0, BK)
        x = tl.load(x_ptr + m.to(tl.int64)[:, None] * K + k[None, :], mask=mm[:, None], other=0.0)
        w = tl.load(w_ptr + n.to(tl.int64)[:, None] * K + k[None, :], mask=nm[:, None], other=0.0)
        # fp8 wgmma accumulates with reduced precision: promote to fp32 every 128 products (as cuBLAS does)
        acc = tl.dot(x, tl.trans(w), acc, max_num_imprecise_acc=128)
    s = tl.load(s_ptr + n, mask=nm, other=0.0)
    tl.store(y_ptr + m.to(tl.int64)[:, None] * N + n[None, :], (acc * s[None, :]).to(tl.bfloat16),
             mask=mm[:, None] & nm[None, :])


def mm_fp8_cfg(M, N):
    """Tile config by M (sweep: scratch/mm_merge.py, K = 5120); None: use cuBLAS."""
    wide = N > 24576
    if M <= 16:
        return dict(BM=16, BN=64, BK=128, num_warps=4, STAGES=3) if wide else \
            dict(BM=16, BN=32, BK=256, num_warps=4, STAGES=3)
    if M <= 64:
        return dict(BM=64, BN=128, BK=128, num_warps=4, STAGES=3) if wide else \
            dict(BM=64, BN=64, BK=256, num_warps=4, STAGES=3)
    if M <= 128 and not wide:
        return dict(BM=128, BN=64, BK=128, num_warps=4, STAGES=4)
    return None


def mm_fp8(x, w, s, cfg):
    """x [M, K] fp8, w [N, K] fp8, s [N] fp32 -> [M, N] bf16."""
    M, K = x.shape
    N = w.shape[0]
    y = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)
    c = dict(cfg)
    BM, BN = c["BM"], c["BN"]
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
        d = (d * tl.load(ds_ptr + row)).to(tl.bfloat16).to(tl.float32)
        x = (x + d).to(tl.bfloat16)
        tl.store(res_ptr + row * N + offs, x, mask=m)
        x = x.to(tl.float32)
    var = tl.sum(x * x, 0) / N
    w = tl.load(w_ptr + offs, mask=m, other=0.0)
    y = (x * tl.rsqrt(var + eps) * w).to(tl.bfloat16).to(tl.float32)
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
        g = (tl.load(g_ptr + row * GS + offs, mask=m, other=0.0).to(tl.float32) * xs).to(tl.bfloat16).to(tl.float32)
        u = (tl.load(u_ptr + row * GS + offs, mask=m, other=0.0).to(tl.float32) * xs).to(tl.bfloat16).to(tl.float32)
        h = (g * tl.sigmoid(g) * u).to(tl.bfloat16).to(tl.float32)
        amax = tl.maximum(amax, tl.abs(h))
    s = tl.maximum(tl.max(amax, 0), 1e-12) / 448.0
    for off in range(0, N, BLOCK):  # pass 2: quantize (the row is in L2)
        offs = off + tl.arange(0, BLOCK)
        m = offs < N
        g = (tl.load(g_ptr + row * GS + offs, mask=m, other=0.0).to(tl.float32) * xs).to(tl.bfloat16).to(tl.float32)
        u = (tl.load(u_ptr + row * GS + offs, mask=m, other=0.0).to(tl.float32) * xs).to(tl.bfloat16).to(tl.float32)
        h = (g * tl.sigmoid(g) * u).to(tl.bfloat16).to(tl.float32)
        tl.store(q_ptr + row * N + offs, tl.clamp(h / s, -448.0, 448.0).to(tl.float8e4nv), mask=m)
    tl.store(s_ptr + row, s)


@triton.jit
def _silu_mul_chunk(g_ptr, u_ptr, base, off, xs, CH: tl.constexpr):
    offs = off + tl.arange(0, CH)
    g = (tl.load(g_ptr + base + offs).to(tl.float32) * xs).to(tl.bfloat16).to(tl.float32)
    u = (tl.load(u_ptr + base + offs).to(tl.float32) * xs).to(tl.bfloat16).to(tl.float32)
    return (g * tl.sigmoid(g) * u).to(tl.bfloat16).to(tl.float32)


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
    y = (acc * tl.sigmoid(acc)).to(tl.bfloat16).to(tl.float32)
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
            xt = (tl.load(x_ptr + row.to(tl.int64) * XS + offs).to(tl.float32)
                  * tl.load(xs_ptr + row)).to(tl.bfloat16).to(tl.float32)
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
        xv = (xv * tl.load(xs_ptr + src, mask=rm & fx, other=0.0)[:, None]).to(tl.bfloat16).to(tl.float32)
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
    rows = tl.program_id(0) * BT + tl.arange(0, BT)
    rm = rows < T
    c0 = tl.program_id(1) * BLOCK_C
    offs = c0 + tl.arange(0, BLOCK_C)
    start = tl.load(row_start + rows, mask=rm, other=0)
    slot = tl.load(row_slot + rows, mask=rm, other=0).to(tl.int64)
    # the state gather is expensive: only blocks holding one of the first 3 tokens of a sequence take that path
    if tl.min(tl.where(rm, rows - start, 3)) < 3:
        acc = _conv_taps(x_ptr, xs_ptr, w_ptr, state_ptr, rows, rm, offs, start, slot, XS, C, BT, BLOCK_C, True)
    else:
        acc = _conv_taps(x_ptr, xs_ptr, w_ptr, state_ptr, rows, rm, offs, start, slot, XS, C, BT, BLOCK_C, False)
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


def conv1d_rows(x, xs, w, state, row_start, row_slot, seq_slot, seq_start, seq_len, n_seqs, HK=16, DK=128):
    """Token-parallel conv1d for steps with prefill chunks (any lengths); same result as conv1d_varlen."""
    T, C = x.shape
    assert x.stride(1) == 1
    q, k, v = _conv_outs(T, x.device)
    BT, BLOCK_C = 16, 128
    _conv1d_rows_kernel[(triton.cdiv(T, BT), C // BLOCK_C)](x, xs, w, state, q, k, v, row_start, row_slot, T, x.stride(0),
                                                            QK=HK * DK, VC=C - 2 * HK * DK, BT=BT, BLOCK_C=BLOCK_C,
                                                            DK=DK, num_warps=4, launch_pdl=PDL)
    _conv1d_state_kernel[(n_seqs, triton.cdiv(C, 1024))](x, xs, state, seq_slot, seq_start, seq_len, C, x.stride(0),
                                                         BLOCK_C=1024, num_warps=4, launch_pdl=PDL)
    return q, k, v


# ---------------------------------------------------------------- Gated-DeltaNet: gates
@triton.jit
def _gdn_gates_kernel(a_ptr, b_ptr, xs_ptr, alog_ptr, dtb_ptr, g_ptr, beta_ptr, AS, HV: tl.constexpr,
                      BH: tl.constexpr):
    gdc_wait()
    r = tl.program_id(0).to(tl.int64)
    h = tl.arange(0, BH)
    m = h < HV
    xs = tl.load(xs_ptr + r)
    a = (tl.load(a_ptr + r * AS + h, mask=m, other=0.0).to(tl.float32) * xs).to(tl.bfloat16).to(tl.float32)
    b = (tl.load(b_ptr + r * AS + h, mask=m, other=0.0).to(tl.float32) * xs).to(tl.bfloat16).to(tl.float32)
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
                       row0_ptr, AS, HV: tl.constexpr, HK: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr,
                       BV: tl.constexpr, HPP: tl.constexpr, STAGES: tl.constexpr):
    """One program per (decode row, group of HPP value heads, V block): gates + one gated delta-rule step on the
    slot's [DK, DV] states (bf16 storage, fp32 math), written back in place; the head loop is software-pipelined
    (the next head's state load overlaps this head's math). q, k are l2-normalized already."""
    gdc_wait()
    r = tl.load(row0_ptr).to(tl.int64) + tl.program_id(0)
    vb = tl.program_id(2)
    slot = tl.load(row_slot + r).to(tl.int64)
    if slot >= 0:
        dk = tl.arange(0, DK)
        dv = vb * BV + tl.arange(0, BV)
        xs = tl.load(xs_ptr + r)
        for i in tl.range(0, HPP, num_stages=STAGES):
            hv = tl.program_id(1) * HPP + i
            kh = hv // (HV // HK)
            sp_ = state_ptr + ((slot * HV + hv) * DK + dk[:, None]) * DV + dv[None, :]
            S = tl.load(sp_, eviction_policy="evict_first").to(tl.float32)
            q = tl.load(q_ptr + (r * HK + kh) * DK + dk).to(tl.float32) * (1.0 / tl.sqrt(DK * 1.0))
            k = tl.load(k_ptr + (r * HK + kh) * DK + dk).to(tl.float32)
            v = tl.load(v_ptr + (r * HV + hv) * DV + dv).to(tl.float32)
            a = (tl.load(a_ptr + r * AS + hv).to(tl.float32) * xs).to(tl.bfloat16).to(tl.float32)
            b = (tl.load(b_ptr + r * AS + hv).to(tl.float32) * xs).to(tl.bfloat16).to(tl.float32)
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


GDN_DECODE_CFG = None  # override for benchmarks; default: by row count (bench/micro_gdn.py)


def _gdn_decode_cfg(n_rows):
    if GDN_DECODE_CFG:
        return GDN_DECODE_CFG
    if n_rows <= 8:
        return {"BV": 32, "num_warps": 1, "HPP": 1, "STAGES": 1}
    return {"BV": 64, "num_warps": 2, "HPP": 1, "STAGES": 1}


def gdn_decode(q, k, v, a, b, xs, A_log, dt_bias, state, row_slot, row0, n_rows, o, HV=48, HK=16, DK=128, DV=128):
    """Rows row0 + [0, n_rows) with row_slot >= 0 are decode rows (row0: device int32 scalar).
    o [T, HV, DV] is written for those rows."""
    c = _gdn_decode_cfg(n_rows)
    assert a.stride(0) == b.stride(0) and a.stride(1) == b.stride(1) == 1
    _gdn_decode_kernel[(n_rows, HV // c["HPP"], DV // c["BV"])](
        q, k, v, a, b, xs, A_log, dt_bias, state, o, row_slot, row0, a.stride(0), HV=HV, HK=HK, DK=DK, DV=DV, BV=c["BV"],
        HPP=c["HPP"], STAGES=c["STAGES"], num_warps=c["num_warps"], launch_pdl=PDL)
    return o


# ---------------------------------------------------------------- Gated-DeltaNet: chunked prefill
# fla's chunked algorithm (WY representation) with fla's kernels for the intra-chunk parts; the inter-chunk state
# recurrence and the outputs are our own fused kernel, so that the per-chunk states stay in registers, the initial /
# final state is read / written in place in the slot pool (bf16) and padded sequences (length 0) cost nothing. All sizes are fixed by the caller (padded
# cu_seqlens / chunk_indices), so the whole step can be captured in a CUDA graph.
@triton.jit
def _gdn_chunk_kernel(q, k, v, w, g, o, state, seq_slot, cu_seqlens, scale,
                      H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr,
                      BV: tl.constexpr):
    """Inter-chunk recurrence fused with the chunk outputs (same arithmetic as fla's chunk_fwd_h + chunk_fwd_o,
    bf16 states and v_new as MMA inputs): per chunk, v_new = u - w h, o = scale (q h * exp(g) + tril(q k^T * exp(g_i - g_j)) v_new), then
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
        st = state + (slot * HV + i_h) * K * V
        b_h1 = tl.load(st + o_k1[:, None] * V + o_v[None, :]).to(tl.float32)
        b_h2 = tl.load(st + o_k2[:, None] * V + o_v[None, :]).to(tl.float32)
        v += (bos * HV + i_h).to(tl.int64) * V
        o += (bos * HV + i_h).to(tl.int64) * V
        q += (bos * H + i_h // (HV // H)).to(tl.int64) * K
        k += (bos * H + i_h // (HV // H)).to(tl.int64) * K
        w += (bos * HV + i_h).to(tl.int64) * K
        for i_t in range(NT):
            o_t = i_t * BT + tl.arange(0, BT)
            m_t = o_t < T
            h1 = b_h1.to(tl.bfloat16)
            h2 = b_h2.to(tl.bfloat16)
            b_w = tl.load(w + o_t[:, None] * (HV * K) + o_k1[None, :], mask=m_t[:, None], other=0.0)
            b_v = tl.dot(b_w, h1)
            b_w = tl.load(w + o_t[:, None] * (HV * K) + o_k2[None, :], mask=m_t[:, None], other=0.0)
            b_v += tl.dot(b_w, h2)
            b_v = tl.load(v + o_t[:, None] * (HV * V) + o_v[None, :], mask=m_t[:, None], other=0.0) - b_v
            b_q = tl.load(q + o_t[:, None] * (H * K) + o_k1[None, :], mask=m_t[:, None], other=0.0)
            b_k1 = tl.load(k + o_k1[:, None] + o_t[None, :] * (H * K), mask=m_t[None, :], other=0.0)
            b_o = tl.dot(b_q, h1)
            b_A = tl.dot(b_q, b_k1)
            b_q = tl.load(q + o_t[:, None] * (H * K) + o_k2[None, :], mask=m_t[:, None], other=0.0)
            b_k2 = tl.load(k + o_k2[:, None] + o_t[None, :] * (H * K), mask=m_t[None, :], other=0.0)
            b_o += tl.dot(b_q, h2)
            b_A += tl.dot(b_q, b_k2)
            b_g = tl.load(g + (bos + o_t).to(tl.int64) * HV + i_h, mask=m_t, other=0.0).to(tl.float32)
            b_o = b_o * tl.exp2(b_g)[:, None]
            b_A = b_A * tl.exp2(b_g[:, None] - b_g[None, :])
            b_A = tl.where((o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t), b_A, 0)
            b_o = b_o * scale + tl.dot(b_A.to(tl.bfloat16), b_v.to(tl.bfloat16)) * scale
            tl.store(o + o_t[:, None] * (HV * V) + o_v[None, :], b_o.to(tl.bfloat16), mask=m_t[:, None])
            b_g_last = tl.load(g + (bos + tl.minimum((i_t + 1) * BT, T) - 1).to(tl.int64) * HV + i_h).to(tl.float32)
            b_v = (b_v * tl.where(m_t, tl.exp2(b_g_last - b_g), 0)[:, None]).to(tl.bfloat16)
            b_g_last = tl.exp2(b_g_last)
            b_h1 = b_h1 * b_g_last + tl.dot(b_k1, b_v)
            b_h2 = b_h2 * b_g_last + tl.dot(b_k2, b_v)
        tl.store(st + o_k1[:, None] * V + o_v[None, :], b_h1.to(tl.bfloat16))
        tl.store(st + o_k2[:, None] * V + o_v[None, :], b_h2.to(tl.bfloat16))


# (BV 16/32 with num_warps >= 4 and num_stages=1 miscompile in Triton 3.6: out-of-bounds shared read)
GDN_CHUNK_CFG = {"BV": 32, "num_warps": 2, "num_stages": 2}




def gdn_prefill(q, k, v, g, beta, state, seq_slot, cu_seqlens, chunk_indices, n_seqs,
                scale=128 ** -0.5):
    """Chunked gated delta rule over varlen sequences in rows [0, cu_seqlens[-1]) of q/k [T, HK, DK],
    v [T, HV, DV], g/beta fp32 [T, HV]; state [S, HV, DK, DV] bf16 indexed by seq_slot (updated in place).
    cu_seqlens [n_seqs + 1], chunk_indices [NT, 2] (int64, fixed sizes; padded
    sequences have length 0, padded chunk entries point to a padded sequence). Returns o [T, HV, DV] (rows outside
    the sequences are undefined)."""
    from fla.ops.gated_delta_rule.chunk_fwd import chunk_gated_delta_rule_fwd_intra
    from fla.ops.utils import chunk_local_cumsum
    from fla.ops.utils.constant import RCP_LN2
    T, HK, DK = k.shape
    HV, DV = v.shape[1], v.shape[2]
    q, k, v, g, beta = q[None], k[None], v[None], g[None], beta[None]
    gc = chunk_local_cumsum(g, chunk_size=64, scale=RCP_LN2, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices)
    w, u, _ = chunk_gated_delta_rule_fwd_intra(k=k, v=v, g=gc, beta=beta, cu_seqlens=cu_seqlens,
                                               chunk_indices=chunk_indices, chunk_size=64)
    o = torch.empty_like(v)
    c = GDN_CHUNK_CFG
    _gdn_chunk_kernel[(DV // c["BV"], n_seqs * HV)](q, k, u, w, gc, o, state, seq_slot, cu_seqlens, scale,
                                                   H=HK, HV=HV, K=DK, V=DV, BT=64, BV=c["BV"],
                                                   num_warps=c["num_warps"], num_stages=c["num_stages"],
                                                   launch_pdl=PDL)
    return o[0]


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
    z = (tl.load(z_ptr + row * ZS + h[:, None] * D + d[None, :], mask=m, other=0.0).to(tl.float32) * tl.load(xs_ptr + row)).to(tl.bfloat16).to(tl.float32)
    var = tl.sum(o * o, 1) / D
    w = tl.load(w_ptr + d)
    y = o * tl.rsqrt(var + eps)[:, None] * w[None, :] * (z * tl.sigmoid(z))
    y = y.to(tl.bfloat16).to(tl.float32)
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
        x = (tl.load(src + d).to(tl.float32) * xs).to(tl.bfloat16).to(tl.float32)
        rstd = tl.rsqrt(tl.sum(x * x, 0) / D + eps)
        y = (x * rstd * tl.load(nw + d)).to(tl.bfloat16)
        x1 = (tl.load(src + half).to(tl.float32) * xs).to(tl.bfloat16).to(tl.float32)
        x2 = (tl.load(src + R // 2 + half).to(tl.float32) * xs).to(tl.bfloat16).to(tl.float32)
        x1 = (x1 * rstd * tl.load(nw + half)).to(tl.bfloat16).to(tl.float32)
        x2 = (x2 * rstd * tl.load(nw + R // 2 + half)).to(tl.bfloat16).to(tl.float32)
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


def attn_decode(q, k_cache, v_cache, block_table, row_slot, pos, row0, n_rows, o, BS, HQ=24, HKV=4, D=256):
    ns = ATTN_DECODE_NSPLIT or min(32, triton.next_power_of_2(max(1, 264 // (n_rows * HKV))))
    if ns > 1:
        po = torch.empty((n_rows, HQ, ns, D), dtype=torch.float32, device=q.device)
        plse = torch.empty((n_rows, HQ, ns), dtype=torch.float32, device=q.device)
    else:
        po = plse = o
    _attn_decode_kernel[(n_rows, HKV, ns)](q, k_cache, v_cache, block_table, row_slot, pos, row0, o, po, plse,
                                           D ** -0.5, block_table.shape[1], HQ=HQ, HKV=HKV, D=D, BS=BS, BLOCK_M=16,
                                           NSPLIT=ns, num_warps=4, num_stages=2, launch_pdl=PDL)
    if ns > 1:
        _attn_decode_combine_kernel[(n_rows, HQ)](po, plse, row_slot, row0, o, HQ=HQ, D=D, NSPLIT=ns, num_warps=4,
                                                  launch_pdl=PDL)
    return o


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
    gate = (gate * tl.load(xs_ptr + row)).to(tl.bfloat16).to(tl.float32)
    y = (o * tl.sigmoid(gate)).to(tl.bfloat16).to(tl.float32)
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
