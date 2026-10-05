"""smem / regs of the chunk and decode kernels as compiled in ovl.py.   python scratch/smem.py"""
import sys, runpy
sys.argv = ["ovl.py"]
sys.path.insert(0, '.')
from agentopt.v3 import kernels as K
import os; os.environ.setdefault("P", "1992")
runpy.run_path("scratch/ovl.py")
for kn in ("_gdn_chunk_kernel", "_gdn_decode_kernel", "_gdn_kkt_kernel"):
    kern = getattr(K, kn)
    for dev, cache in kern.device_caches.items():
        for key, c in cache[0].items():
            print(kn, "smem", c.metadata.shared, "regs", c.n_regs, "spills", c.n_spills, "warps", c.metadata.num_warps)
