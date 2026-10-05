"""Unit tests: every v3 Triton kernel against a plain-torch reference (HF semantics), random inputs.

    python bench/test_v3_kernels.py
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agentopt.v3 import kernels as K  # noqa: E402

torch.manual_seed(0)
dev = "cuda"
bf = torch.bfloat16
fails = []


def check(name, got, ref, atol, rtol=0.0):
    got, ref = got.float(), ref.float()
    err = (got - ref).abs()
    tol = atol + rtol * ref.abs()
    bad = (err > tol).float().mean().item()
    ok = bad < 1e-3 and torch.isfinite(got).all()
    print(f"{'OK ' if ok else 'BAD'} {name:28s} max err {err.max().item():.3e}  frac>tol {bad:.2e}", flush=True)
    if not ok:
        fails.append(name)


def deq(q, s):
    return q.float() * s[:, None]


def ref_quant(y):
    y = y.to(bf).float()
    s = y.abs().amax(-1).clamp_min(1e-12) / 448
    return y, s


# ---- add + rmsnorm + quant
T, N = 37, 5120
res = torch.randn(T, N, device=dev, dtype=bf)
d = torch.randn(T, N, device=dev, dtype=bf)
ds = torch.rand(T, device=dev) + 0.5
w = torch.randn(N, device=dev) * 0.1
r_res = (res.float() + (d.float() * ds[:, None]).to(bf).float()).to(bf)
x = r_res.float()
r_y = (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * (1 + w)).to(bf).float()
res2 = res.clone()
q, s = K.add_rmsnorm_quant(res2, (1 + w).contiguous(), 1e-6, d, ds)
check("add_rmsnorm residual", res2, r_res, 0)
check("add_rmsnorm quant", deq(q, s), r_y, 0, 0.07)
out = K.add_rmsnorm_quant(res.clone(), (1 + w).contiguous(), 1e-6, d, ds, quant=False)
check("add_rmsnorm noquant", out, r_y, 1e-2, 1e-2)

# ---- silu * up + quant
N = 17408
g = torch.randn(T, N, device=dev, dtype=bf)
u = torch.randn(T, N, device=dev, dtype=bf)
gs = (g.float() * ds[:, None]).to(bf).float()
us = (u.float() * ds[:, None]).to(bf).float()
r_h = (F.silu(gs) * us).to(bf).float()
q, s = K.silu_mul_quant(g, u, ds)
check("silu_mul_quant", deq(q, s), r_h, 1e-3, 0.07)
gu = torch.cat([g, u], 1)  # column slices of one buffer (merged gate/up GEMM output)
q2, s2 = K.silu_mul_quant(gu[:, :N], gu[:, N:], ds)
check("silu_mul_quant strided", deq(q2, s2), deq(q, s), 0)

# ---- small-M merged FP8 GEMM (per-column scales) vs torch._scaled_mm per part
for M, parts in [(1, (10240, 6144, 48, 48)), (13, (10240, 6144, 48, 48)), (64, (10240, 6144, 48, 48)),
                 (100, (10240, 6144, 48, 48)), (1, (17408, 17408)), (24, (17408, 17408)), (48, (17408, 17408))]:
    Kd = 5120
    ws_ = [torch.randn(n, Kd, device=dev).to(K.FP8) for n in parts]
    sc_ = [torch.rand((), device=dev) for _ in ws_]
    xq = torch.randn(M, Kd, device=dev).to(K.FP8)
    one = torch.ones((), device=dev)
    ref = torch.cat([torch._scaled_mm(xq, w.t(), one, sc, out_dtype=bf) for w, sc in zip(ws_, sc_)], 1)
    W = torch.cat(ws_)
    sv = torch.cat([sc.expand(w.shape[0]) for w, sc in zip(ws_, sc_)]).contiguous()
    y = K.mm_fp8(xq, W, sv, K.mm_fp8_cfg(M, W.shape[0]))
    check(f"mm_fp8 M={M} N={W.shape[0]}", y, ref, 1e-2, 1e-2)

# ---- conv1d varlen with slot states
def conv_cat(qkv3):
    """(q, k, v) of the conv kernels -> [T, C]"""
    return torch.cat([t.reshape(t.shape[0], -1) for t in qkv3], 1)


def ref_conv_post(y):
    """silu output [T, C] -> bf16, q/k l2-normalized per 128-head (fp32), like the kernels"""
    y = y.to(bf).float()
    qk = y[:, :4096].reshape(-1, 32, 128)
    qk = qk * torch.rsqrt((qk * qk).sum(-1, keepdim=True) + 1e-6)
    return torch.cat([qk.reshape(-1, 4096), y[:, 4096:]], 1)


C, S = 10240, 6
lens = [1, 1, 5, 0, 3, 1]
starts = [0, 1, 2, 7, 7, 10]
T = 11
x = torch.randn(T, C, device=dev, dtype=bf)
xs = torch.rand(T, device=dev) + 0.5
wc = torch.randn(C, 4, device=dev) * 0.5
state = torch.randn(S + 1, 3, C, device=dev, dtype=bf)
slots = [3, 0, 5, 6, 1, 2]
ref_state = state.clone()
ref_out = torch.zeros(T, C, device=dev)
xr = (x.float() * xs[:, None]).to(bf).float()
for i, (sl, st, n) in enumerate(zip(slots, starts, lens)):
    if n == 0:
        continue
    win = torch.cat([state[sl].float(), xr[st:st + n]], 0)  # [3 + n, C]
    for t in range(n):
        acc = (win[t:t + 4] * wc.t()).sum(0)
        ref_out[st + t] = F.silu(acc)
    ref_state[sl] = win[-3:].to(bf)
st2 = state.clone()
t32 = lambda v: torch.tensor(v, device=dev, dtype=torch.int32)  # noqa: E731
out = conv_cat(K.conv1d_varlen(x, xs, wc, st2, t32(slots), t32(starts), t32(lens), len(slots)))
ref_out = ref_conv_post(ref_out)
rows = [r for st, n in zip(starts, lens) for r in range(st, st + n)]
check("conv1d out", out[rows], ref_out[rows], 2e-2, 1e-2)
check("conv1d state", st2, ref_state, 0)
st3 = state.clone()
rstart, rslot = [0] * T, [S] * T
for sl, st, n in zip(slots, starts, lens):
    for r in range(st, st + n):
        rstart[r], rslot[r] = st, sl
out = conv_cat(K.conv1d_rows(x, xs, wc, st3, t32(rstart), t32(rslot), t32(slots), t32(starts), t32(lens),
                             len(slots)))
check("conv1d_rows out", out[rows], ref_out[rows], 2e-2, 1e-2)
check("conv1d_rows state", st3, ref_state, 0)
# longer chunks (token-parallel path across row blocks)
lens2, starts2, slots2 = [1, 40, 2, 0, 77], [0, 1, 41, 43, 43], [4, 0, 5, 6, 2]
T2 = 120
x2 = torch.randn(T2, C, device=dev, dtype=bf)
xs2 = torch.rand(T2, device=dev) + 0.5
ref_state2, ref_out2 = state.clone(), torch.zeros(T2, C, device=dev)
xr2 = (x2.float() * xs2[:, None]).to(bf).float()
rstart, rslot = [0] * T2, [S] * T2
for sl, st, n in zip(slots2, starts2, lens2):
    if n == 0:
        continue
    win = torch.cat([state[sl].float(), xr2[st:st + n]], 0)
    for t in range(n):
        ref_out2[st + t] = F.silu((win[t:t + 4] * wc.t()).sum(0))
        rstart[st + t], rslot[st + t] = st, sl
    ref_state2[sl] = win[-3:].to(bf)
st4 = state.clone()
out = conv_cat(K.conv1d_rows(x2, xs2, wc, st4, t32(rstart), t32(rslot), t32(slots2), t32(starts2), t32(lens2),
                             len(slots2)))
ref_out2 = ref_conv_post(ref_out2)
rows2 = [r for st, n in zip(starts2, lens2) for r in range(st, st + n)]
check("conv1d_rows long out", out[rows2], ref_out2[rows2], 2e-2, 1e-2)
check("conv1d_rows long state", st4, ref_state2, 0)

# ---- gated delta rule decode step
HV, HK, DK, DV = 48, 16, 128, 128
T = 7
qkv = torch.randn(T, 2 * HK * DK + HV * DV, device=dev, dtype=bf)
qkv[:, :2 * HK * DK] = ref_conv_post(qkv.float())[:, :2 * HK * DK].to(bf)
gq, gk, gv = (qkv[:, :HK * DK].reshape(T, HK, DK).contiguous(), qkv[:, HK * DK:2 * HK * DK].reshape(T, HK, DK).contiguous(),
              qkv[:, 2 * HK * DK:].reshape(T, HV, DV).contiguous())
a = torch.randn(T, HV, device=dev, dtype=bf)
b = torch.randn(T, HV, device=dev, dtype=bf)
A_log = torch.rand(HV, device=dev) * 2
dtb = torch.randn(HV, device=dev)
rec = (torch.randn(S + 1, HV, DK, DV, device=dev) * 0.1).to(bf)
row_slot = t32([-1, -1, 2, -1, 0, 4, -1])  # rows 0, 1: prefill rows (row0 = 2), then decode rows and padding
o = torch.zeros(T, HV, DV, device=dev, dtype=bf)
rec2 = rec.clone()
K.gdn_decode(gq, gk, gv, a, b, xs[:T], A_log, dtb, rec2, row_slot, t32([2]), T - 2, o)
ref_rec = rec.clone()
for r, sl in enumerate(row_slot.tolist()):
    if sl < 0:
        continue
    q_ = qkv[r, :HK * DK].float().view(HK, DK).repeat_interleave(3, 0)
    k_ = qkv[r, HK * DK:2 * HK * DK].float().view(HK, DK).repeat_interleave(3, 0)
    v_ = qkv[r, 2 * HK * DK:].float().view(HV, DV)
    q_ = q_ / DK ** 0.5
    aa = (a[r].float() * xs[r]).to(bf).float()
    bb = (b[r].float() * xs[r]).to(bf).float()
    gg = -A_log.exp() * F.softplus(aa + dtb)
    beta = bb.sigmoid()
    Sx = rec[sl].float() * gg.exp()[:, None, None]
    kv = (Sx * k_[:, :, None]).sum(1)
    delta = (v_ - kv) * beta[:, None]
    Sx = Sx + k_[:, :, None] * delta[:, None, :]
    ref_rec[sl] = Sx.to(bf)
    check(f"gdn_decode o row{r}", o[r], (Sx * q_[:, :, None]).sum(1), 2e-2, 2e-2)
check("gdn_decode state", rec2, ref_rec, 1e-2, 1e-2)

# ---- gdn gates
a2 = torch.randn(9, HV, device=dev, dtype=bf) * 3
b2 = torch.randn(9, HV, device=dev, dtype=bf)
gg, gb = K.gdn_gates(a2, b2, xs[:9], A_log, dtb)
check("gdn_gates g", gg, -A_log.exp() * F.softplus((a2.float() * xs[:9, None]).to(bf).float() + dtb), 1e-5, 1e-5)
check("gdn_gates beta", gb, ((b2.float() * xs[:9, None]).to(bf).float()).sigmoid(), 1e-6, 1e-6)

# ---- chunked prefill (varlen, slot states in place, padded sequences / chunk entries) vs fla
from fla.ops.gated_delta_rule import chunk_gated_delta_rule  # noqa: E402
lens_p, slots_p = [100, 1, 0, 64, 300, 0], [3, 0, 6, 5, 1, 6]  # zero-length = padding (scratch slot 6)
cu = [0]
for n in lens_p:
    cu.append(cu[-1] + n)
Tp = cu[-1] + 13  # trailing padding rows
qn_ = ref_conv_post(torch.randn(Tp, 10240, device=dev))
pq = qn_[:, :2048].to(bf).reshape(Tp, HK, DK).contiguous()
pk = qn_[:, 2048:4096].to(bf).reshape(Tp, HK, DK).contiguous()
pv = torch.randn(Tp, HV, DV, device=dev, dtype=bf)
ab_raw = torch.randn(Tp, 2 * HV + 5, device=dev, dtype=bf)  # strided like the GEMM outputs
xs_p = torch.rand(Tp, device=dev) + 0.5
pg, pb = K.gdn_gates(ab_raw[:, HV:2 * HV], ab_raw[:, :HV], xs_p, A_log, dtb)
rec_p = (torch.randn(S + 1, HV, DK, DV, device=dev) * 0.1).to(bf)
ci = [(i, t) for i, n in enumerate(lens_p) for t in range((n + 63) // 64)]
ci += [(2, 0)] * 5  # padded chunk entries -> zero-length sequence
ci_t = torch.tensor(ci, dtype=torch.long, device=dev)
cu_t = torch.tensor(cu, dtype=torch.long, device=dev)
rec_p2 = rec_p.clone()
po = K.gdn_prefill(pq, pk, pv, pg, pb, rec_p2, t32(slots_p), cu_t, ci_t, len(lens_p))
real = [i for i, n in enumerate(lens_p) if n]
cu_r = torch.tensor([0] + list(torch.tensor([lens_p[i] for i in real]).cumsum(0)), dtype=torch.long, device=dev)
ro, rht = chunk_gated_delta_rule(pq[None, :cu[-1]], pk[None, :cu[-1]], pv[None, :cu[-1]], pg[None, :cu[-1]],
                                 pb[None, :cu[-1]], initial_state=rec_p[[slots_p[i] for i in real]].float(),
                                 output_final_state=True, cu_seqlens=cu_r, scale=DK ** -0.5)
check("gdn_prefill o", po[:cu[-1]], ro[0], 2e-2, 2e-2)
ref_recp = rec_p.clone()
ref_recp[[slots_p[i] for i in real]] = rht.to(bf)
check("gdn_prefill state", rec_p2, ref_recp, 1e-2, 1e-2)
rec_p3 = rec_p.clone()
po3 = K.gdn_prefill(pq, pk, pv, None, None, rec_p3, t32(slots_p), cu_t, ci_t, len(lens_p),
                    gates=(ab_raw[:, HV:2 * HV], ab_raw[:, :HV], xs_p, A_log, dtb))
check("gdn_prefill gates in kkt (bit-exact)", po3[:cu[-1]], po[:cu[-1]], 0.0)
check("gdn_prefill gates in kkt state", rec_p3, rec_p2, 0.0)
# correlated keys + slow decay (real chunks; random keys are nearly orthogonal and hide unstable inverses)
pk_c = F.normalize(torch.randn(1, HK, DK, device=dev) + 0.3 * torch.randn(Tp, HK, DK, device=dev), dim=-1).to(bf)
pg_c = -0.005 * torch.rand(Tp, HV, device=dev)
pb_c = 0.5 + 0.5 * torch.rand(Tp, HV, device=dev)
rec_c = rec_p.clone()
po = K.gdn_prefill(pq, pk_c, pv, pg_c, pb_c, rec_c, t32(slots_p), cu_t, ci_t, len(lens_p))
ro, rht = chunk_gated_delta_rule(pq[None, :cu[-1]], pk_c[None, :cu[-1]], pv[None, :cu[-1]], pg_c[None, :cu[-1]],
                                 pb_c[None, :cu[-1]], initial_state=rec_p[[slots_p[i] for i in real]].float(),
                                 output_final_state=True, cu_seqlens=cu_r, scale=DK ** -0.5)
check("gdn_prefill o corr. keys", po[:cu[-1]], ro[0], 2e-2, 2e-2)

# ---- gated rmsnorm * silu(z) + quant
T = 7
o = torch.randn(T, HV, DV, device=dev, dtype=bf)
z = torch.randn(T, HV * DV, device=dev, dtype=bf)
gw = torch.rand(DV, device=dev) + 0.5
zz = (z.float() * xs[:T, None]).to(bf).float().view(T, HV, DV)
of = o.float()
r_y = (of * torch.rsqrt(of.pow(2).mean(-1, keepdim=True) + 1e-6) * gw * F.silu(zz)).reshape(T, -1)
q, s = K.gated_norm_quant(o, z, xs[:T], gw, 1e-6)
check("gated_norm_quant", deq(q, s), r_y, 1e-3, 0.07)

# ---- q/k norm + rope + paged KV write
HQ, HKV, D, R = 24, 4, 256, 64
T, BS, NB = 6, 64, 8
qr = torch.randn(T, HQ * 2 * D, device=dev, dtype=bf)
kr = torch.randn(T, HKV * D, device=dev, dtype=bf)
vr = torch.randn(T, HKV * D, device=dev, dtype=bf)
pos = t32([0, 5, 100, 1000, 7, 31000])
slot_map = t32([64, 65, 130, 200, 3, 511])
qn = torch.randn(D, device=dev) * 0.1
kn = torch.randn(D, device=dev) * 0.1
kc = torch.zeros(NB * BS, HKV, D, device=dev, dtype=bf)
vc = torch.zeros_like(kc)
qo = K.qkv_prep(qr, kr, vr, xs[:T], pos, slot_map, (1 + qn).contiguous(), (1 + kn).contiguous(), kc, vc, 1e-6, 1e7)
theta = 1e7
inv = 1.0 / theta ** (torch.arange(0, R, 2, device=dev).float() / R)
fr = pos.float()[:, None] * inv[None]
emb = torch.cat([fr, fr], -1)
cos, sin = emb.cos(), emb.sin()


def ref_nr(x, w):
    x = (x.float() * xs[:T, None, None]).to(bf).float()
    y = (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * (1 + w)).to(bf).float()
    rot, pas = y[..., :R], y[..., R:]
    x1, x2 = rot[..., :R // 2], rot[..., R // 2:]
    rh = torch.cat([-x2, x1], -1)
    return torch.cat([rot * cos[:, None] + rh * sin[:, None], pas], -1)


check("qkv_prep q", qo, ref_nr(qr.view(T, HQ, 2 * D)[..., :D], qn), 3e-2, 2e-2)
check("qkv_prep k", kc[slot_map.long()], ref_nr(kr.view(T, HKV, D), kn), 3e-2, 2e-2)
check("qkv_prep v", vc[slot_map.long()], (vr.float() * xs[:T, None]).to(bf).view(T, HKV, D), 0)

# ---- paged attention: decode rows + prefill tiles vs dense causal reference
torch.manual_seed(1)
NB, BS = 64, 64
kc = torch.randn(NB * BS, HKV, D, device=dev, dtype=bf)
vc = torch.randn(NB * BS, HKV, D, device=dev, dtype=bf)
S1 = 5
bt = torch.zeros(S1, 16, dtype=torch.int32, device=dev)
perm = torch.randperm(NB - 1, device=dev) + 1
for sl in range(S1):
    bt[sl] = perm[sl * 12:sl * 12 + 16] if sl * 12 + 16 <= NB - 1 else perm[:16]
bt[4] = perm[40:56]


def dense_kv(sl, L):
    idx = (bt[sl].long()[:, None] * BS + torch.arange(BS, device=dev)).flatten()[:L]
    return kc[idx].float(), vc[idx].float()


def ref_attn(qv, sl, qpos):  # qv [HQ, D] at position qpos
    k_, v_ = dense_kv(sl, qpos + 1)
    k_ = k_.repeat_interleave(HQ // HKV, 1)
    v_ = v_.repeat_interleave(HQ // HKV, 1)
    sc = torch.einsum("hd,lhd->hl", qv.float(), k_) / D ** 0.5
    return torch.einsum("hl,lhd->hd", sc.softmax(-1), v_)


# decode: 3 rows (seq slots 1, 3, 0) with lengths 1, 70, 700, plus a padding row (ctx 0)
ctx = [1, 70, 700, 0]
sl_ = [1, 3, 0, 4]
qd = torch.randn(4, HQ, D, device=dev, dtype=bf)
od = torch.zeros(4, HQ, D, device=dev, dtype=bf)
K.attn_decode(qd, kc, vc, bt, t32(sl_[:3] + [-1]), t32([c - 1 for c in ctx]), t32([0]), 4, od, BS)
for r in range(3):
    check(f"attn_decode row{r} ctx{ctx[r]}", od[r], ref_attn(qd[r], sl_[r], ctx[r] - 1), 2e-2, 2e-2)
# prefill: seq slot 2, 150 new tokens at positions 300..449; seq slot 4, 7 tokens at 0..6
segs = [(2, 300, 150), (4, 0, 7)]
Tp = sum(n for _, _, n in segs)
qp = torch.randn(Tp, HQ, D, device=dev, dtype=bf)
op = torch.zeros_like(qp)
tiles, r0 = [], 0
for sl, p0, n in segs:
    for t in range(0, n, K.PREFILL_TQ):
        tiles.append((r0 + t, min(K.PREFILL_TQ, n - t), p0 + t, sl))
    r0 += n
tt = torch.tensor(tiles, device=dev, dtype=torch.int32).t().contiguous()
K.attn_prefill(qp, kc, vc, bt, tt[0], tt[1], tt[2], tt[3], len(tiles), op, BS)
r0 = 0
for sl, p0, n in segs:
    for t in [0, 1, n // 2, n - 1]:
        check(f"attn_prefill slot{sl} t{t}", op[r0 + t], ref_attn(qp[r0 + t], sl, p0 + t), 2e-2, 2e-2)
    r0 += n
if K.FA3 is not None:  # same segments through FA3 (+ a padding sequence: 0 rows, 0 cached tokens)
    of = torch.zeros_like(qp)
    K.attn_prefill_fa3(qp, kc, vc, bt[t32([sl for sl, _, _ in segs] + [0]).long()],
                       t32([p0 + n for _, p0, n in segs] + [0]), t32([0, segs[0][2], Tp, Tp]), Tp, of, BS)
    r0 = 0
    for sl, p0, n in segs:
        for t in [0, 1, n // 2, n - 1]:
            check(f"attn_prefill_fa3 slot{sl} t{t}", of[r0 + t], ref_attn(qp[r0 + t], sl, p0 + t), 2e-2, 2e-2)
        r0 += n

# ---- attention output gate + quant
T = 5
o = torch.randn(T, HQ, D, device=dev, dtype=bf)
qr = torch.randn(T, HQ * 2 * D, device=dev, dtype=bf)
gate = (qr.view(T, HQ, 2 * D)[..., D:].float() * xs[:T, None, None]).to(bf).float()
r_y = (o.float() * gate.sigmoid()).reshape(T, -1)
q, s = K.attn_gate_quant(o, qr, xs[:T])
check("attn_gate_quant", deq(q, s), r_y, 1e-3, 0.07)

print("ALL OK" if not fails else f"FAILED: {fails}")
sys.exit(1 if fails else 0)
