"""gdn_decode with / without the register prefetch of v, a, b: bit-exactness and time.   python scratch/decpf.py"""
import sys, runpy, os, torch
sys.argv = ["ovl.py"]; sys.path.insert(0, '.'); sys.path.insert(0, 'scratch'); os.environ["NOBENCH"] = "1"
g = runpy.run_path("scratch/ovl.py")
from agentopt.v3 import kernels as K
from benchutil import bench_loop
pool0 = g["pool"].clone()
res = {}
for nd in [int(x) for x in os.environ.get("NDS", "8,20,40,56").split(",")]:
    g["row_slot"].fill_(-1); g["row_slot"][g["P"]:g["P"] + nd] = torch.arange(nd, dtype=torch.int32, device='cuda')
    for norm in (False, True):
        for pf in (0, 1):
            K.GDN_DECODE_PF = pf
            g["pool"].copy_(pool0); g["od"].zero_()
            K.gdn_decode(g["qd"], g["kd"], g["vd"], g["ab"][:, 48:96], g["ab"][:, :48], g["xs"], g["A_log"], g["dtb"], g["pool"],
                         g["row_slot"], g["row0"], nd, g["od"], norm=norm)
            res[pf] = (g["od"].clone(), g["pool"].clone())
        same = torch.equal(res[0][0], res[1][0]) and torch.equal(res[0][1], res[1][1])
        print(f"rows {nd} norm {norm}: bit-exact {same}", flush=True)
    for pf in (0, 1):
        K.GDN_DECODE_PF = pf
        t = bench_loop([lambda: K.gdn_decode(g["qd"], g["kd"], g["vd"], g["ab"][:, 48:96], g["ab"][:, :48], g["xs"], g["A_log"],
                                             g["dtb"], g["pool"], g["row_slot"], g["row0"], nd, g["od"])], iters=10)
        print(f"rows {nd} cfg {K._gdn_decode_cfg(nd)} PF {pf}: {t * 1e6:.1f} us", flush=True)
