"""Launch-config sweep of fla's decode kernel (fused_recurrent_gated_delta_rule_fwd_kernel), in place (ht = h0).

One decode step of one Gated-DeltaNet layer of Qwen3.5-27B (H=16 key heads, HV=48 value heads, K=V=128) for a
batch of B rows, with fp32 or bf16 state storage. Bytes moved are dominated by the state (read + write).

    python bench/micro_recurrent.py --out results/p2_s09_recurrent_launch/micro_recurrent.json
"""
import argparse
import json
from pathlib import Path

import torch
import triton
from fla.ops.gated_delta_rule.fused_recurrent import fused_recurrent_gated_delta_rule_fwd_kernel

H, HV, K, V = 16, 48, 128, 128


def launch(q, k, v, g, beta, h, o, BV, warps, stages):
    B = q.shape[0]
    fused_recurrent_gated_delta_rule_fwd_kernel[(triton.cdiv(V, BV), B * HV)](
        q=q, k=k, v=v, g=g, gk=None, gv=None, beta=beta, A_log=None, dt_bias=None, o=o, h0=h, ht=h,
        cu_seqlens=None, scale=K ** -0.5, T=1, H=H, HV=HV, K=K, V=V, BK=K, BV=BV, IS_BETA_HEADWISE=True,
        USE_QK_L2NORM_IN_KERNEL=True, APPLY_BETA_SIGMOID=False, ALLOW_NEG_EIGVAL=False, STATE_V_FIRST=False,
        num_warps=warps, num_stages=stages)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.manual_seed(0)
    res = []
    for B in (8, 32, 64):
        q = torch.randn(B, 1, H, K, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(B, 1, H, K, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(B, 1, HV, V, device="cuda", dtype=torch.bfloat16)
        g = -torch.rand(B, 1, HV, device="cuda", dtype=torch.float32)
        beta = torch.rand(B, 1, HV, device="cuda", dtype=torch.bfloat16)
        o = torch.empty_like(v)
        ref_o = None
        for dt in (torch.float32, torch.bfloat16):
            h0 = torch.randn(B, HV, K, V, device="cuda", dtype=torch.float32) * 0.1
            for BV in (8, 16, 32, 64):
                for warps in (1, 2, 4, 8):
                    for stages in (1, 3):
                        h = h0.to(dt).clone()
                        try:
                            launch(q, k, v, g, beta, h, o, BV, warps, stages)
                        except Exception as e:  # e.g. out of shared memory / registers
                            res.append({"B": B, "dtype": str(dt), "BV": BV, "warps": warps, "stages": stages,
                                        "error": type(e).__name__})
                            continue
                        torch.cuda.synchronize()
                        if ref_o is None:
                            ref_o = o.float().clone()
                        err = (o.float() - ref_o).abs().max().item()
                        gr = torch.cuda.CUDAGraph()  # time 20 back-to-back launches as one graph (no host)
                        with torch.cuda.graph(gr):
                            for _ in range(20):
                                launch(q, k, v, g, beta, h, o, BV, warps, stages)
                        for _ in range(3):
                            gr.replay()
                        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        e0.record()
                        for _ in range(10):
                            gr.replay()
                        e1.record()
                        torch.cuda.synchronize()
                        us = e0.elapsed_time(e1) / 200 * 1e3
                        gbs = 2 * h.numel() * h.element_size() / us / 1e3
                        res.append({"B": B, "dtype": str(dt), "BV": BV, "warps": warps, "stages": stages,
                                    "us": us, "state_GBps": gbs, "max_abs_o_diff_vs_first": err})
                        print(f"B={B:3d} {str(dt):15s} BV={BV:2d} warps={warps} stages={stages}: {us:7.1f} us "
                              f"({gbs:6.0f} GB/s)  o diff {err:.2e}", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))
    for B in (8, 32, 64):
        for dt in ("torch.float32", "torch.bfloat16"):
            rs = [r for r in res if r["B"] == B and r["dtype"] == dt and "us" in r]
            base = next(r for r in rs if r["BV"] == 8 and r["warps"] == 1 and r["stages"] == 3)
            best = min(rs, key=lambda r: r["us"])
            print(f"B={B} {dt}: default {base['us']:.1f} us -> best {best['us']:.1f} us "
                  f"(BV={best['BV']}, warps={best['warps']}, stages={best['stages']}, {best['state_GBps']:.0f} GB/s)")


if __name__ == "__main__":
    main()
