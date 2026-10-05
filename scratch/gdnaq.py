# gdn_prefill (kkt + chunk) with the intra-chunk q k^T by its own kernel (GDN_AQ=1) vs on the chunk kernel (0)
import sys, torch
sys.path.insert(0, '.'); sys.path.insert(0, 'scratch')
from agentopt.v3 import kernels as K
from benchutil import bench_loop
from gdnb2_setup import setup
for lens in [(1992,), (1024, 968), (512, 512, 512, 512), (300, 1700), (4000,), (40,), (100, 30), (64, 1, 63, 65)]:
    q, k, v, g, b, st, slots, cu, ci, ns = setup(lens)
    T = sum(lens)
    res, ts = {}, {}
    for aq in (0, 1):
        K.GDN_AQ = aq
        s = st.clone()
        res[aq] = (K.gdn_prefill(q, k, v, g, b, s, slots, cu, ci, ns)[:T].clone(), s)
    for aq in (0, 1):
        K.GDN_AQ = aq
        s = st.clone()
        ts[aq] = bench_loop([lambda: K.gdn_prefill(q, k, v, g, b, s, slots, cu, ci, ns)], iters=10)
    ex = torch.equal(res[0][0], res[1][0]) and torch.equal(res[0][1], res[1][1])
    print(f"{lens}: aq0 {ts[0] * 1e6:.1f} us, aq1 {ts[1] * 1e6:.1f} us, exact {ex}", flush=True)
