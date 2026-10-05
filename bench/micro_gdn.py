"""Microbenchmark of the GDN decode step kernel vs the state-traffic roofline (CUDA-graph timing).

    python bench/micro_gdn.py
"""
import sys
from pathlib import Path

import torch
import triton

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agentopt.v3 import kernels as K  # noqa: E402

dev, bf = "cuda", torch.bfloat16
HV, HK, DK, DV, S = 48, 16, 128, 128, 128
rec = (torch.randn(S + 1, HV, DK, DV, device=dev) * 0.1).to(bf)
A_log = torch.rand(HV, device=dev)
dtb = torch.randn(HV, device=dev)
for B in [int(x) for x in sys.argv[1:]] or (1, 8, 32, 128):
    q = torch.randn(B, HK, DK, device=dev, dtype=bf)
    k = torch.randn(B, HK, DK, device=dev, dtype=bf)
    v = torch.randn(B, HV, DV, device=dev, dtype=bf)
    a = torch.randn(B, HV, device=dev, dtype=bf)
    b = torch.randn(B, HV, device=dev, dtype=bf)
    xs = torch.ones(B, device=dev)
    rs = torch.randperm(S, device=dev)[:B].int()
    row0 = torch.zeros(1, device=dev, dtype=torch.int32)
    o = torch.empty(B, HV, DV, device=dev, dtype=bf)
    ideal = B * HV * DK * DV * 2 * 2 / 3.2e12 * 1e6
    res = []
    cfgs = [(bv, nw, hpp, st) for bv in (32, 64, 128) for nw in (1, 2, 4, 8) for hpp, st in
            ((1, 1), (2, 2), (4, 2), (4, 3), (8, 3))]
    for bv, nw, hpp, st in cfgs:
            K.GDN_DECODE_CFG = dict(BV=bv, num_warps=nw, HPP=hpp, STAGES=st)
            try:
                us = triton.testing.do_bench_cudagraph(
                    lambda: K.gdn_decode(q, k, v, a, b, xs, A_log, dtb, rec, rs, row0, B, o), rep=30) * 1e3
                res.append((us, bv, nw, hpp, st))
            except Exception as e:  # noqa: BLE001
                print("fail", bv, nw, hpp, st, type(e).__name__)
    res.sort()
    print(f"B={B} ideal {ideal:.1f} us; best: " + ", ".join(f"BV{bv}/w{nw}/h{h}/s{st} {us:.1f}" for us, bv, nw, h, st in res[:6])
          + f"; worst {res[-1][0]:.1f}", flush=True)
