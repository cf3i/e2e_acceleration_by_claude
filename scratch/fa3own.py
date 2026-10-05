"""Our FA3 build (flash_attn_3, hdim 256 bf16 paged) vs K.attn_prefill on prefill-chunk shapes; correctness too."""
import sys, torch
sys.path.insert(0, '.'); sys.path.insert(0, 'scratch')
from flash_attn_interface import flash_attn_with_kvcache
from agentopt.v3 import kernels as K
from benchutil import bench_loop
dev, bf = "cuda", torch.bfloat16
HQ, HKV, D, BS = 24, 4, 256, 64


def tiles_for(lens, pos0s, tq):
    t0, nt, p0, sl = [], [], [], []
    off = 0
    for s, (n, p) in enumerate(zip(lens, pos0s)):
        for i in range(0, n, tq):
            t0.append(off + i); nt.append(min(tq, n - i)); p0.append(p + i); sl.append(s)
        off += n
    f = lambda x: torch.tensor(x, dtype=torch.int32, device=dev)
    return f(t0), f(nt), f(p0), f(sl)


for lens, pos0s in (([1992], [3000]), ([1992], [0]), ([1024, 1024], [2000, 2000]), ([512] * 4, [1000] * 4),
                    ([1992], [6000]), ([100] * 3 + [40], [500, 2000, 4000, 100])):
    T = sum(lens); MB = 512; NB = len(lens) * MB + 1
    kc = torch.randn(NB * BS, HKV, D, device=dev, dtype=bf); vc = torch.randn_like(kc)
    bt = (torch.randperm(NB - 1, device=dev, dtype=torch.int32) + 1).reshape(len(lens), MB).contiguous()
    q = torch.randn(T, HQ, D, device=dev, dtype=bf) * 3
    flops = sum(4 * HQ * D * (n * p + n * (n + 1) / 2) for n, p in zip(lens, pos0s))
    o0 = torch.empty_like(q)
    t = tiles_for(lens, pos0s, K.PREFILL_TQ)
    f0 = lambda: K.attn_prefill(q, kc, vc, bt, *t, t[0].shape[0], o0, BS)
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int32, device=dev)
    sk = torch.tensor([p + n for n, p in zip(lens, pos0s)], dtype=torch.int32, device=dev)
    k4, v4 = kc.view(NB, BS, HKV, D), vc.view(NB, BS, HKV, D)
    res = {}
    for pg in (None, True, False):
        f1 = lambda: res.__setitem__(pg, flash_attn_with_kvcache(q, k4, v4, cache_seqlens=sk, page_table=bt, cu_seqlens_q=cu,
                                     max_seqlen_q=T, softmax_scale=D ** -0.5, causal=True, num_splits=1, pack_gqa=pg))
        f1(); torch.cuda.synchronize()
        s1 = bench_loop([f1], iters=10)
        f0(); torch.cuda.synchronize()
        err = (res[pg].float() - o0.float()).abs().max().item()
        print(f"FA3 pack_gqa={pg} lens={lens} pos0={pos0s}: {s1*1e6:.1f} us {flops/s1/1e12:.0f} TFLOPS  err vs ours {err:.2e}", flush=True)
    s0 = bench_loop([f0], iters=10)
    print(f"ours: {s0*1e6:.1f} us {flops/s0/1e12:.0f} TFLOPS", flush=True)
