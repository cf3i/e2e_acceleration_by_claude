# is _gdn_aq_kernel bit-identical with 1 / 2 / 8 warps (mma.sync) vs 4 (wgmma)? and its time alone
import sys, torch
sys.path.insert(0, '.'); sys.path.insert(0, 'scratch')
from agentopt.v3 import kernels as K
from benchutil import bench_loop
from gdnb2_setup import setup
from fla.ops.utils.constant import RCP_LN2
q, k, v, g, b, st, slots, cu, ci, ns = setup((1992,))
T = 1992
gc = torch.empty_like(g); A = torch.empty(T, 48, 64, device='cuda', dtype=torch.bfloat16)
K._gdn_kkt_kernel[(ci.shape[0], 48)](k, g, gc, b, A, cu, ci, RCP_LN2, H=16, HV=48, K=128, BT=64, HPP=1, PREC="tf32", num_warps=1)
tk = bench_loop([lambda: K._gdn_kkt_kernel[(ci.shape[0], 48)](k, g, gc, b, A, cu, ci, RCP_LN2, H=16, HV=48, K=128, BT=64, HPP=1, PREC="tf32", num_warps=1)], iters=10)
print(f"kkt alone {tk * 1e6:.1f} us")
ref = None
for nw in (4, 1, 2, 8):
    Aq = torch.zeros(T, 48, 64, device='cuda', dtype=torch.bfloat16)
    f = lambda: K._gdn_aq_kernel[(ci.shape[0], 16)](q, k, gc, Aq, cu, ci, H=16, HV=48, K=128, BT=64, num_warps=nw)
    f()
    if ref is None: ref = Aq.clone()
    t = bench_loop([f], iters=10)
    print(f"aq nw {nw}: {t * 1e6:.1f} us exact {torch.equal(ref, Aq)} maxdiff {(ref.float() - Aq.float()).abs().max().item():.3g}")
