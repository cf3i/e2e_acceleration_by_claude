"""Bandwidth floor of the gdn_decode state traffic: a kernel with the same grid / tiles that only reads each
[DK, BV] state tile and writes it back (x 1.0), vs gdn_decode.   ND=128 python scratch/decfloor.py"""
import os, sys, runpy, torch, triton, triton.language as tl
sys.argv = ["ovl.py"]; sys.path.insert(0, '.'); sys.path.insert(0, 'scratch'); os.environ["NOBENCH"] = "1"
g = runpy.run_path("scratch/ovl.py")
from benchutil import bench_loop
from agentopt.v3 import kernels as K


@triton.jit
def _copy(state_ptr, row_slot, row0_ptr, HV: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr, BV: tl.constexpr,
          HPP: tl.constexpr, STAGES: tl.constexpr):
    r = tl.load(row0_ptr).to(tl.int64) + tl.program_id(2)
    slot = tl.load(row_slot + r).to(tl.int64)
    dk = tl.arange(0, DK)
    dv = tl.program_id(0) * BV + tl.arange(0, BV)
    if slot >= 0:
        for i in tl.range(0, HPP, num_stages=STAGES):
            hv = tl.program_id(1) * HPP + i
            sp_ = state_ptr + ((slot * HV + hv) * DK + dk[:, None]) * DV + dv[None, :]
            S = tl.load(sp_, eviction_policy="evict_first").to(tl.float32)
            tl.store(sp_, (S * 1.0001).to(tl.bfloat16), cache_modifier=".cs")


ND = g["ND"]
pool, row_slot, row0 = g["pool"], g["row_slot"], g["row0"]
nbytes = ND * 48 * 128 * 128 * 2 * 2
td = bench_loop([g["dec"]], iters=10)
print(f"ND {ND}: gdn_decode {K._gdn_decode_cfg(ND)} {td * 1e6:.1f} us = {nbytes / td / 1e12:.2f} TB/s", flush=True)
for BV, nw, HPP, ST in [(64, 2, 4, 3), (64, 4, 4, 3), (128, 4, 4, 3), (64, 2, 1, 1), (32, 1, 4, 3), (128, 8, 2, 2)]:
    f = lambda: _copy[(128 // BV, 48 // HPP, ND)](pool, row_slot, row0, HV=48, DK=128, DV=128, BV=BV, HPP=HPP, STAGES=ST,
                                                  num_warps=nw)
    t = bench_loop([f], iters=10)
    print(f"  copy BV {BV} nw {nw} HPP {HPP} ST {ST}: {t * 1e6:.1f} us = {nbytes / t / 1e12:.2f} TB/s", flush=True)
