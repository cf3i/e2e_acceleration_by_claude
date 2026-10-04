"""Speed-of-light yardsticks on this GPU (phase 3, step 0).

1. HBM bandwidth actually reachable: Triton streaming read, torch copy (read+write), sizes far above L2.
2. Every FP8 linear shape of the model (and the merged shapes B2 would create) through the current library
   kernel (`torch._scaled_mm`, per-tensor, cuBLASLt) at decode-sized M: time and achieved bandwidth on the
   weight bytes. Weights rotate over enough copies that L2 cannot hold them (decode streams every weight once).
3. bf16 lm_head GEMM likewise.
4. Cost of a kernel boundary inside a CUDA graph (tiny dependent kernels), i.e. what fusion / PDL can win.

    python bench/micro_bw.py --stage p3_s00_rebaseline
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench.run_bench import env_info  # noqa: E402

H, I = 5120, 17408
SHAPES = {  # name: (out_features N, in_features K, instances per step)
    "mlp.gate/up": (I, H, 128), "mlp.down": (H, I, 64),
    "gdn.in_proj_qkv": (10240, H, 48), "gdn.in_proj_z": (6144, H, 48), "gdn.out_proj": (H, 6144, 48),
    "attn.q_proj": (12288, H, 16), "attn.k/v_proj": (1024, H, 32), "attn.o_proj": (H, 6144, 16),
    # merged shapes (B2): gate+up, GDN qkv+z (+b, a: 2*48 rows, padded to 16), attention q+k+v
    "merged.gate_up": (2 * I, H, 64), "merged.gdn_in": (10240 + 6144 + 96, H, 48), "merged.attn_qkv": (14336, H, 16),
}
MS = [1, 8, 16, 32, 64, 128, 256]


@triton.jit
def _read_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    step = tl.num_programs(0) * BLOCK
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for off in range(pid * BLOCK, n, step):
        idx = off + tl.arange(0, BLOCK)
        acc += tl.load(x_ptr + idx, mask=idx < n, other=0.0).to(tl.float32)
    tl.store(out_ptr + pid, tl.sum(acc))


def bench(fn, iters=50, warmup=5):
    for _ in range(warmup):
        fn()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1e-3  # seconds


def hbm():
    rows = []
    sm = torch.cuda.get_device_properties(0).multi_processor_count
    for gb in (0.5, 2, 8):
        n = int(gb * 2**30) // 2
        x = torch.empty(n, dtype=torch.bfloat16, device="cuda").normal_()
        out = torch.empty(sm * 8, dtype=torch.float32, device="cuda")
        best = 0
        for BLOCK in (1024, 4096, 8192):
            for progs in (sm * 2, sm * 4, sm * 8):
                t = bench(lambda: _read_kernel[(progs,)](x, out, n, BLOCK=BLOCK, num_warps=8), iters=20)
                best = max(best, n * 2 / t)
        y = torch.empty_like(x)
        tc = bench(lambda: y.copy_(x), iters=20)
        rows.append({"GB": gb, "read_GBps": best / 1e9, "copy_GBps": 2 * n * 2 / tc / 1e9})
        print(f"HBM {gb:4.1f} GB: triton read {best / 1e9:7.0f} GB/s, torch copy (r+w) {2 * n * 2 / tc / 1e9:7.0f} GB/s",
              flush=True)
        del x, y
    return rows


def gemms(read_bw):
    rows = []
    one = torch.ones((), device="cuda")
    for name, (N, K, per_step) in SHAPES.items():
        copies = max(2, int(2 * 2**30 // (N * K)))  # >= 2 GB of distinct weights: no L2 reuse between calls
        ws = [torch.randn(N, K, device="cuda").to(torch.float8_e4m3fn) for _ in range(copies)]
        for M in MS:
            a = torch.randn(M, K, device="cuda").to(torch.float8_e4m3fn)
            it = iter(range(10**9))
            t = bench(lambda: torch._scaled_mm(a, ws[next(it) % copies].t(), one, one, out_dtype=torch.bfloat16),
                      iters=copies * max(1, 40 // copies))
            byt = N * K + M * K + 2 * M * N
            rows.append({"name": name, "N": N, "K": K, "M": M, "per_step": per_step, "us": t * 1e6,
                         "GBps": byt / t / 1e9, "frac_of_read_bw": byt / t / read_bw})
        print(f"{name:18s} N={N:6d} K={K:6d} " + "  ".join(
            f"M{r['M']}:{r['us']:6.1f}us/{100 * r['frac_of_read_bw']:3.0f}%" for r in rows[-len(MS):]), flush=True)
        del ws
    # bf16 lm_head
    w = torch.randn(248320, H, device="cuda", dtype=torch.bfloat16)
    for M in MS:
        x = torch.randn(M, H, device="cuda", dtype=torch.bfloat16)
        t = bench(lambda: x @ w.t(), iters=20)
        byt = w.numel() * 2 + M * 248320 * 2
        rows.append({"name": "lm_head.bf16", "N": 248320, "K": H, "M": M, "per_step": 1, "us": t * 1e6,
                     "GBps": byt / t / 1e9, "frac_of_read_bw": byt / t / read_bw})
    print("lm_head.bf16       " + "  ".join(
        f"M{r['M']}:{r['us']:6.1f}us/{100 * r['frac_of_read_bw']:3.0f}%" for r in rows[-len(MS):]), flush=True)
    return rows


def step_sol(rows, read_bw):
    """Per-step GEMM time with the current (unmerged) shapes vs SOL, per M."""
    out = {}
    for M in MS:
        cur = sum(r["us"] * r["per_step"] for r in rows if r["M"] == M and not r["name"].startswith("merged"))
        merged_names = {"mlp.down", "gdn.out_proj", "attn.o_proj", "merged.gate_up", "merged.gdn_in",
                        "merged.attn_qkv", "lm_head.bf16"}
        mer = sum(r["us"] * r["per_step"] for r in rows if r["M"] == M and r["name"] in merged_names)
        sol = sum((r["N"] * r["K"] * (2 if r["name"] == "lm_head.bf16" else 1)) * r["per_step"]
                  for r in rows if r["M"] == M and not r["name"].startswith("merged")) / read_bw * 1e6
        out[M] = {"gemm_us_current_shapes": cur, "gemm_us_merged_shapes": mer, "weights_sol_us": sol}
        print(f"M={M:4d}: all GEMMs of a step {cur / 1e3:6.2f} ms (merged shapes {mer / 1e3:6.2f} ms), "
              f"weight-read SOL {sol / 1e3:5.2f} ms -> {100 * sol / cur:4.1f}% of SOL", flush=True)
    return out


def graph_boundary():
    x = torch.zeros(1024, device="cuda")
    res = {}
    for n in (100, 2000):
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            for _ in range(3):
                x.add_(1)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(n):
                x.add_(1)
        t = bench(g.replay, iters=50)
        res[n] = t / n * 1e6
    print(f"tiny dependent kernel inside a CUDA graph: {res[2000]:.2f} us each "
          f"(2000/graph), {res[100]:.2f} us (100/graph)", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    args = ap.parse_args()
    torch.manual_seed(0)
    h = hbm()
    read_bw = max(r["read_GBps"] for r in h) * 1e9
    g = gemms(read_bw)
    s = step_sol(g, read_bw)
    b = graph_boundary()
    out = ROOT / "results" / args.stage
    out.mkdir(parents=True, exist_ok=True)
    (out / "micro_bw.json").write_text(json.dumps(
        {"hbm": h, "read_bw_GBps": read_bw / 1e9, "gemm": g, "step_gemm": s, "graph_kernel_us": b,
         "env": env_info()}, indent=2))


if __name__ == "__main__":
    main()
