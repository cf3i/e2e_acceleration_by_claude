"""GDN branch of a mixed step in isolation: prefill (kkt -> chunk, side stream) || gdn_decode (main, after kkt), as
model._gdn does for big buckets.   CHUNK=BV,nw,ns DEC=BV,nw,HPP,STAGES python scratch/ovl.py"""
import os, sys, torch
sys.path.insert(0, '.'); sys.path.insert(0, 'scratch')
from agentopt.v3 import kernels as K
from benchutil import bench_loop
from gdnb2_setup import setup
HV, HK, DK, DV = 48, 16, 128, 128
P, ND = int(os.environ.get("P", 1992)), int(os.environ.get("ND", 56))
T = P + ND
q, k, v, g, b, st, slots, cu, ci, ns = setup((P,))
dev = 'cuda'
# decode rows after the prefill rows, as in a flat mixed batch
qd = torch.nn.functional.normalize(torch.randn(T, HK, DK, device=dev), dim=-1).to(torch.bfloat16)
kd = torch.nn.functional.normalize(torch.randn(T, HK, DK, device=dev), dim=-1).to(torch.bfloat16)
vd = torch.randn(T, HV, DV, device=dev, dtype=torch.bfloat16)
ab = torch.randn(T, 2 * HV + 5, device=dev, dtype=torch.bfloat16)
xs = torch.rand(T, device=dev) + 0.5
A_log = torch.log(torch.rand(HV, device=dev) * 15 + 1); dtb = torch.randn(HV, device=dev) * 0.5
gates = (ab[:P, HV:2 * HV], ab[:P, :HV], xs[:P], A_log, dtb)
pool = (torch.randn(130, HV, DK, DV, device=dev) * .1).to(torch.bfloat16)
row_slot = torch.full((T,), -1, dtype=torch.int32, device=dev); row_slot[P:] = torch.arange(ND, dtype=torch.int32, device=dev)
row0 = torch.tensor(P, dtype=torch.int32, device=dev)
od = torch.empty(T, HV, DV, device=dev, dtype=torch.bfloat16)
if "CHUNK" in os.environ:
    bv, nw, nst = map(int, os.environ["CHUNK"].split(","))
    K.GDN_CHUNK_CFG = {"BV": bv, "num_warps": nw, "num_stages": nst}
if "DEC" in os.environ:
    bv, nw, hpp, stg = map(int, os.environ["DEC"].split(","))
    K.GDN_DECODE_CFG = {"BV": bv, "num_warps": nw, "HPP": hpp, "STAGES": stg}
side = torch.cuda.Stream(priority=-1)
def pre(after=None):
    K.gdn_prefill(q, k, v, None, None, st, slots, cu, ci, ns, gates=gates, after_kkt=after)
def dec():
    K.gdn_decode(qd, kd, vd, ab[:, HV:2 * HV], ab[:, :HV], xs, A_log, dtb, pool, row_slot, row0, ND, od)
def both(wait_kkt=True):
    main = torch.cuda.current_stream()
    side.wait_stream(main)
    with torch.cuda.stream(side):
        pre((lambda: main.wait_stream(side)) if wait_kkt else None)
    dec()
    main.wait_stream(side)
NOBENCH = os.environ.get("NOBENCH")
if not NOBENCH:
  tp = bench_loop([pre], iters=10); td = bench_loop([dec], iters=10)
  tb = bench_loop([both], iters=10); tb0 = bench_loop([lambda: both(False)], iters=10)
  print(f"P {P} ND {ND} chunk {K.GDN_CHUNK_CFG} dec {K._gdn_decode_cfg(ND)}: prefill {tp * 1e6:.1f} us, decode {td * 1e6:.1f}, "
        f"both (dec after kkt) {tb * 1e6:.1f}, both (dec at once) {tb0 * 1e6:.1f}", flush=True)
