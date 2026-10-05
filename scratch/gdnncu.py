# one launch of the production chunk kernel (1992 tokens) for ncu:  ncu -k regex:_gdn_chunk python scratch/gdnncu.py
import sys, torch
sys.path.insert(0, '.'); sys.path.insert(0, 'scratch')
from agentopt.v3 import kernels as K
from gdnb2_setup import setup
q, k, v, g, b, st, slots, cu, ci, ns = setup((1992,))
for _ in range(3):
    K.gdn_prefill(q, k, v, g, b, st, slots, cu, ci, ns)
torch.cuda.synchronize()
