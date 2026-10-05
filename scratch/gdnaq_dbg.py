import sys, torch
sys.path.insert(0, '.'); sys.path.insert(0, 'scratch')
from agentopt.v3 import kernels as K
from gdnb2_setup import setup
q, k, v, g, b, st, slots, cu, ci, ns = setup((1992,))
T = 1992
outs = []
for aq in (0, 0, 2, 2):
    K.GDN_AQ = aq
    s = st.clone()
    outs.append((K.gdn_prefill(q, k, v, g, b, s, slots, cu, ci, ns)[:T].clone(), s))
print("aq0 deterministic", torch.equal(outs[0][0], outs[1][0]), "aq2 det", torch.equal(outs[2][0], outs[3][0]))
d = (outs[0][0].float() - outs[2][0].float()).abs()
print("o diff frac", (d > 0).float().mean().item(), "max", d.max().item(), "state equal", torch.equal(outs[0][1], outs[2][1]))
idx = (d > 0).nonzero()
print("rows with diffs (first 20):", sorted(set(idx[:, 0].tolist()))[:20], "pos-in-chunk hist", torch.bincount(idx[:, 0] % 64, minlength=64).tolist())
