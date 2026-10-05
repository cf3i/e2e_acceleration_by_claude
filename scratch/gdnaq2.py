# gdn_prefill in GATES mode (as in the model): GDN_AQ 0 / 1
import sys, torch
sys.path.insert(0, '.'); sys.path.insert(0, 'scratch')
from agentopt.v3 import kernels as K
from benchutil import bench_loop
from gdnb2_setup import setup
HV = 48
for lens in [(1992,), (512, 512, 512, 456), (4000,), (40,), (300, 30)]:
    q, k, v, g, b, st, slots, cu, ci, ns = setup(lens)
    T = q.shape[0]
    ab = torch.randn(T, 2 * HV + 5, device='cuda', dtype=torch.bfloat16)
    xs = torch.rand(T, device='cuda') + 0.5
    A_log = torch.log(torch.rand(HV, device='cuda') * 15 + 1); dtb = torch.randn(HV, device='cuda') * 0.5
    gates = (ab[:, HV:2 * HV], ab[:, :HV], xs, A_log, dtb)
    res, ts = {}, {}
    for aq in (0, 1):
        K.GDN_AQ = aq
        s = st.clone()
        res[aq] = (K.gdn_prefill(q, k, v, None, None, s, slots, cu, ci, ns, gates=gates)[:sum(lens)].clone(), s)
    for aq in (0, 1):
        K.GDN_AQ = aq
        s = st.clone()
        ts[aq] = bench_loop([lambda: K.gdn_prefill(q, k, v, None, None, s, slots, cu, ci, ns, gates=gates)], iters=10)
    ex = torch.equal(res[0][0], res[1][0]) and torch.equal(res[0][1], res[1][1])
    print(f"{lens}: aq0 {ts[0] * 1e6:.1f} us, aq1 {ts[1] * 1e6:.1f}, exact {ex}", flush=True)
