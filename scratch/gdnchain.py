"""How long is the serial h chain alone in the chunk kernel? OUT=1: current kernel; OUT=0: chain only (no q dots,
no o).   python scratch/gdnchain.py"""
import os, sys, torch, triton, triton.language as tl
sys.path.insert(0, '.'); sys.path.insert(0, 'scratch')
from triton.language.extra.cuda import gdc_wait
from agentopt.v3 import kernels as K
from benchutil import bench_loop
from gdnb2_setup import setup
from fla.ops.utils.constant import RCP_LN2


@triton.jit
def _chunk(q, k, v, A, beta, g, o, state, seq_slot, cu_seqlens, scale,
           H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr,
           BV: tl.constexpr, OUT: tl.constexpr):
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
            if OUT:
                b_q = tl.load(q + o_t[:, None] * (H * K) + o_k1[None, :], mask=m_t[:, None], other=0.0)
                b_o = tl.dot(b_q, h1)
                b_A = tl.dot(b_q, b_k1)
                b_q = tl.load(q + o_t[:, None] * (H * K) + o_k2[None, :], mask=m_t[:, None], other=0.0)
                b_o += tl.dot(b_q, h2)
                b_A += tl.dot(b_q, b_k2)
                b_o = b_o * tl.exp2(b_g)[:, None]
                b_A = b_A * tl.exp2(b_g[:, None] - b_g[None, :])
                b_A = tl.where((o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t), b_A, 0)
                b_o = b_o * scale + tl.dot(b_A.to(tl.bfloat16), b_v.to(tl.bfloat16)) * scale
                tl.store(o + o_t[:, None] * (HV * V) + o_v[None, :], b_o.to(tl.bfloat16), mask=m_t[:, None])
            elif OUT == 0:
                # keep v_new alive (what a split design would store)
                tl.store(o + o_t[:, None] * (HV * V) + o_v[None, :], b_v.to(tl.bfloat16), mask=m_t[:, None])
            b_g_last = tl.load(g + (bos + tl.minimum((i_t + 1) * BT, T) - 1).to(tl.int64) * HV + i_h).to(tl.float32)
            b_v = (b_v * tl.where(m_t, tl.exp2(b_g_last - b_g), 0)[:, None]).to(tl.bfloat16)
            b_g_last = tl.exp2(b_g_last)
            b_h1 = b_h1 * b_g_last + tl.dot(b_k1, b_v)
            b_h2 = b_h2 * b_g_last + tl.dot(b_k2, b_v)
        tl.store(st + o_k1[:, None] * V + o_v[None, :], b_h1.to(tl.bfloat16))
        tl.store(st + o_k2[:, None] * V + o_v[None, :], b_h2.to(tl.bfloat16))


for lens in [(1992,), (512, 512, 512, 512)]:
    q, k, v, g, b, st, slots, cu, ci, ns = setup(lens)
    T = sum(lens)
    gc = torch.empty_like(g); A = torch.empty(T + 0, 48, 64, device='cuda', dtype=torch.bfloat16)
    K._gdn_kkt_kernel[(ci.shape[0], 48)](k, g, gc, b, A, cu, ci, RCP_LN2, H=16, HV=48, K=128, BT=64, HPP=1, PREC="tf32", num_warps=1)
    o = torch.empty_like(v)
    for OUT in (1, 0):
        for BV, nw, ns_ in [(64, 4, 3), (32, 4, 3), (32, 4, 2), (64, 8, 3)]:
            try:
                f = lambda: _chunk[(128 // BV, ns * 48)](q, k, v, A, b, gc, o, st, slots, cu, 0.088, H=16, HV=48, K=128, V=128, BT=64,
                                                         BV=BV, OUT=OUT, num_warps=nw, num_stages=ns_, launch_pdl=True)
                t = bench_loop([f], iters=10)
                print(f"lens {lens} OUT {OUT} BV {BV} nw {nw} ns {ns_}: {t * 1e6:.1f} us", flush=True)
            except Exception as e:
                print(f"lens {lens} OUT {OUT} BV {BV} nw {nw}: fail {str(e)[:100]}", flush=True)
