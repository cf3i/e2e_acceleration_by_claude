"""Producer / consumer GDN chunk kernel: producers run the h chain only (v_new, h update) and publish h_i (bf16) and
v_new per chunk; consumers compute o_i = scale (e^g q h_i + tril(q k^T e^{g_i-g_j}) v_new) concurrently.
Roles by atomic ticket (start order), so no reliance on CTA dispatch order.   python scratch/gdnpc.py"""
import os, sys, torch, triton, triton.language as tl
sys.path.insert(0, '.'); sys.path.insert(0, 'scratch')
from triton.language.extra.cuda import gdc_wait
from agentopt.v3 import kernels as K
from benchutil import bench_loop
from gdnb2_setup import setup
from fla.ops.utils.constant import RCP_LN2


@triton.jit
def _chunk_pc(q, k, v, A, beta, g, o, state, seq_slot, cu_seqlens, scale, hbuf, flags, counter,
              H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr,
              BV: tl.constexpr, NV: tl.constexpr, NP, LAG: tl.constexpr = 1, SPIN: tl.constexpr = 0):
    gdc_wait()
    t = tl.atomic_add(counter, 1)
    if t == 2 * NP - 1:
        tl.atomic_xchg(counter, 0)
    role = t >= NP
    t = tl.where(role, t - NP, t)
    i_v, i_nh = t % NV, t // NV
    i_n, i_h = i_nh // HV, i_nh % HV
    bos = tl.load(cu_seqlens + i_n).to(tl.int32)
    T = tl.load(cu_seqlens + i_n + 1).to(tl.int32) - bos
    if T > 0:
        NT = tl.cdiv(T, BT)
        flag = flags + t
        o_v = i_v * BV + tl.arange(0, BV)
        o_k1 = tl.arange(0, 64)
        o_k2 = 64 + o_k1
        o_a = tl.arange(0, BT)
        v += (bos * HV + i_h).to(tl.int64) * V
        o += (bos * HV + i_h).to(tl.int64) * V
        q += (bos * H + i_h // (HV // H)).to(tl.int64) * K
        k += (bos * H + i_h // (HV // H)).to(tl.int64) * K
        A += (bos * HV + i_h).to(tl.int64) * BT
        hb = hbuf + ((bos // BT + i_n) * HV + i_h).to(tl.int64) * (K * V)
        HS: tl.constexpr = HV * K * V
        if role == 0:
            slot = tl.load(seq_slot + i_n).to(tl.int64)
            st = state + (slot * HV + i_h) * K * V
            b_h1 = tl.load(st + o_k1[:, None] * V + o_v[None, :]).to(tl.float32)
            b_h2 = tl.load(st + o_k2[:, None] * V + o_v[None, :]).to(tl.float32)
            for i_t in range(NT):
                o_t = i_t * BT + tl.arange(0, BT)
                m_t = o_t < T
                h1 = b_h1.to(tl.bfloat16)
                h2 = b_h2.to(tl.bfloat16)
                if LAG:
                    if i_t > 0:
                        tl.debug_barrier()
                        tl.atomic_xchg(flag, i_t, sem="release")
                tl.store(hb + i_t * HS + o_k1[:, None] * V + o_v[None, :], h1)
                tl.store(hb + i_t * HS + o_k2[:, None] * V + o_v[None, :], h2)
                b_g = tl.load(g + (bos + o_t).to(tl.int64) * HV + i_h, mask=m_t, other=0.0).to(tl.float32)
                b_b = tl.load(beta + (bos + o_t).to(tl.int64) * HV + i_h, mask=m_t, other=0.0).to(tl.float32)
                b_k1 = tl.load(k + o_k1[:, None] + o_t[None, :] * (H * K), mask=m_t[None, :], other=0.0)
                b_k2 = tl.load(k + o_k2[:, None] + o_t[None, :] * (H * K), mask=m_t[None, :], other=0.0)
                b_v = tl.dot(tl.trans(b_k1), h1)
                b_v += tl.dot(tl.trans(b_k2), h2)
                b_v = tl.load(v + o_t[:, None] * (HV * V) + o_v[None, :], mask=m_t[:, None], other=0.0).to(tl.float32) - b_v * tl.exp2(b_g)[:, None]
                b_Aw = tl.load(A + o_t[:, None] * (HV * BT) + o_a[None, :], mask=m_t[:, None], other=0.0)
                b_v = tl.dot(b_Aw, (b_v * b_b[:, None]).to(tl.bfloat16))
                tl.store(o + o_t[:, None] * (HV * V) + o_v[None, :], b_v.to(tl.bfloat16), mask=m_t[:, None])
                if not LAG:
                    tl.debug_barrier()
                    tl.atomic_xchg(flag, i_t + 1, sem="release")
                b_g_last = tl.load(g + (bos + tl.minimum((i_t + 1) * BT, T) - 1).to(tl.int64) * HV + i_h).to(tl.float32)
                b_v = (b_v * tl.where(m_t, tl.exp2(b_g_last - b_g), 0)[:, None]).to(tl.bfloat16)
                b_g_last = tl.exp2(b_g_last)
                b_h1 = b_h1 * b_g_last + tl.dot(b_k1, b_v)
                b_h2 = b_h2 * b_g_last + tl.dot(b_k2, b_v)
            if LAG:
                tl.debug_barrier()
                tl.atomic_xchg(flag, NT, sem="release")
            tl.store(st + o_k1[:, None] * V + o_v[None, :], b_h1.to(tl.bfloat16))
            tl.store(st + o_k2[:, None] * V + o_v[None, :], b_h2.to(tl.bfloat16))
        else:
            for i_t in range(NT):
                o_t = i_t * BT + tl.arange(0, BT)
                m_t = o_t < T
                b_g = tl.load(g + (bos + o_t).to(tl.int64) * HV + i_h, mask=m_t, other=0.0).to(tl.float32)
                b_k1 = tl.load(k + o_k1[:, None] + o_t[None, :] * (H * K), mask=m_t[None, :], other=0.0)
                b_k2 = tl.load(k + o_k2[:, None] + o_t[None, :] * (H * K), mask=m_t[None, :], other=0.0)
                b_q1 = tl.load(q + o_t[:, None] * (H * K) + o_k1[None, :], mask=m_t[:, None], other=0.0)
                b_q2 = tl.load(q + o_t[:, None] * (H * K) + o_k2[None, :], mask=m_t[:, None], other=0.0)
                b_A = tl.dot(b_q1, b_k1)
                b_A += tl.dot(b_q2, b_k2)
                b_A = b_A * tl.exp2(b_g[:, None] - b_g[None, :])
                b_A = tl.where((o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t), b_A, 0)
                while tl.atomic_add(flag, 0, sem="acquire") <= i_t:
                    pass
                tl.debug_barrier()
                h1 = tl.load(hb + i_t * HS + o_k1[:, None] * V + o_v[None, :], cache_modifier=".cg")
                h2 = tl.load(hb + i_t * HS + o_k2[:, None] * V + o_v[None, :], cache_modifier=".cg")
                b_vn = tl.load(o + o_t[:, None] * (HV * V) + o_v[None, :], mask=m_t[:, None], other=0.0, cache_modifier=".cg")
                b_o = tl.dot(b_q1, h1)
                b_o += tl.dot(b_q2, h2)
                b_o = b_o * tl.exp2(b_g)[:, None]
                b_o = b_o * scale + tl.dot(b_A.to(tl.bfloat16), b_vn) * scale
                tl.store(o + o_t[:, None] * (HV * V) + o_v[None, :], b_o.to(tl.bfloat16), mask=m_t[:, None])
            tl.atomic_xchg(flag, 0)


def chunk_pc(q, k, v, A, beta, gc, o, state, slots, cu, ns, cfg, hbuf, flags, counter, scale=128 ** -0.5):
    BV, nw, nst, lag = cfg
    NV = 128 // BV
    NP = NV * ns * 48
    _chunk_pc[(2 * NP,)](q, k, v, A, beta, gc, o, state, slots, cu, scale, hbuf, flags, counter,
                         H=16, HV=48, K=128, V=128, BT=64, BV=BV, NV=NV, NP=NP, LAG=lag, num_warps=nw, num_stages=nst,
                         launch_pdl=True)


for lens in [(1992,), (512, 512, 512, 512), (100, 30)]:
    q, k, v, g, b, st, slots, cu, ci, ns = setup(lens)
    T = sum(lens)
    gc = torch.empty_like(g); A = torch.empty(T, 48, 64, device='cuda', dtype=torch.bfloat16)
    K._gdn_kkt_kernel[(ci.shape[0], 48)](k, g, gc, b, A, cu, ci, RCP_LN2, H=16, HV=48, K=128, BT=64, HPP=1, PREC="tf32", num_warps=1)
    c = K.GDN_CHUNK_CFG
    o_ref = torch.empty_like(v); st_ref = st.clone()
    ref = lambda o_, st_: K._gdn_chunk_kernel[(128 // c["BV"], ns * 48)](q, k, v, A, b, gc, o_, st_, slots, cu, 128 ** -0.5,
              H=16, HV=48, K=128, V=128, BT=64, BV=c["BV"], num_warps=c["num_warps"], num_stages=c["num_stages"], launch_pdl=True)
    ref(o_ref, st_ref); torch.cuda.synchronize()
    st_w = st.clone()
    t_ref = bench_loop([lambda: ref(torch.empty_like(v), st_w)], iters=10)
    hbuf = torch.empty(T // 64 + ns + 1, 48, 128, 128, device='cuda', dtype=torch.bfloat16)
    flags = torch.zeros(2 * 48 * ns, dtype=torch.int32, device='cuda')
    counter = torch.zeros(1, dtype=torch.int32, device='cuda')
    for cfg in [(64, 4, 3, 0), (64, 4, 3, 1), (64, 4, 2, 1)]:
        try:
            o1 = torch.empty_like(v); st1 = st.clone()
            chunk_pc(q, k, v, A, b, gc, o1, st1, slots, cu, ns, cfg, hbuf, flags, counter); torch.cuda.synchronize()
            ex = torch.equal(o1[:T], o_ref[:T]) and torch.equal(st1, st_ref)
            d = (o1[:T].float() - o_ref[:T].float()).abs().max().item()
            t = bench_loop([lambda: chunk_pc(q, k, v, A, b, gc, torch.empty_like(v), st_w, slots, cu, ns, cfg, hbuf, flags, counter)], iters=10)
            print(f"{lens} ref {t_ref * 1e6:.1f} us | pc {cfg}: {t * 1e6:.1f} us  exact {ex} maxdiff {d:.3g}", flush=True)
        except Exception as e:
            print(f"{lens} pc {cfg}: fail {str(e)[:300]}", flush=True)
