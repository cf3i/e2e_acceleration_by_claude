"""Microbenchmark of the v3 attention kernels (paged KV, Qwen3.5 shapes) vs FLOP / byte rooflines.

    python bench/micro_attn.py
"""
import sys
from pathlib import Path

import torch
import triton

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agentopt.v3 import kernels as K  # noqa: E402

dev, bf = "cuda", torch.bfloat16
HQ, HKV, D, BS = 24, 4, 256, 64
NB = 1024
kc = torch.randn(NB * BS, HKV, D, device=dev, dtype=bf)
vc = torch.randn_like(kc)
bt = torch.randperm(NB - 1, device=dev, dtype=torch.int32)[:512].add(1).reshape(1, 512).contiguous()


def prefill_case(T, pos0, cfg):
    tq = cfg["BLOCK_M"] // 6
    q = torch.randn(T, HQ, D, device=dev, dtype=bf)
    o = torch.empty_like(q)
    i = torch.arange(0, T, tq, device=dev, dtype=torch.int32)
    tiles = [i, torch.clamp(T - i, max=tq), pos0 + i, torch.zeros_like(i)]
    fn = lambda: K.attn_prefill(q, kc, vc, bt, *tiles, len(i), o, BS, cfg=cfg)  # noqa: E731
    ms = triton.testing.do_bench(fn, warmup=5, rep=50)
    flops = 4 * HQ * D * sum(pos0 + t + 1 for t in range(T))
    return ms, flops / ms / 1e9


for T, pos0 in []:
    for bm in (64, 128):
        for nw in (4, 8):
            for ns in (1, 2, 3):
                cfg = {"BLOCK_M": bm, "num_warps": nw, "num_stages": ns}
                try:
                    ms, tf = prefill_case(T, pos0, cfg)
                    print(f"prefill T={T} pos0={pos0} {cfg}: {ms * 1e3:8.1f} us  {tf:6.0f} TFLOPS", flush=True)
                except Exception as e:  # noqa: BLE001
                    print(f"prefill T={T} pos0={pos0} {cfg}: fail {type(e).__name__}", flush=True)


def decode_case(B, ctx, ns):
    K.ATTN_DECODE_NSPLIT = ns
    btb = torch.stack([torch.randperm(NB - 1, device=dev, dtype=torch.int32)[:512] + 1 for _ in range(B)])
    q = torch.randn(B, HQ, D, device=dev, dtype=bf)
    o = torch.empty_like(q)
    rs = torch.arange(B, device=dev, dtype=torch.int32)
    pos = torch.full((B,), ctx - 1, device=dev, dtype=torch.int32)
    row0 = torch.zeros(1, device=dev, dtype=torch.int32)
    us = triton.testing.do_bench_cudagraph(lambda: K.attn_decode(q, kc, vc, btb, rs, pos, row0, B, o, BS), rep=30) * 1e3
    return us, B * ctx * HKV * D * 2 * 2 / 3.2e12 * 1e6


for B, ctx in [(1, 1000), (1, 8000), (8, 1000), (8, 8000), (32, 2000), (128, 1000)]:
    res = {ns: decode_case(B, ctx, ns)[0] for ns in (1, 2, 4, 8, 16, 32, 64)}
    K.ATTN_DECODE_NSPLIT = None
    auto = decode_case(B, ctx, None)
    print(f"decode B={B} ctx={ctx}: ideal {auto[1]:.1f} us, auto {auto[0]:.1f} us; "
          + " ".join(f"ns{k}={v:.1f}" for k, v in res.items()), flush=True)
