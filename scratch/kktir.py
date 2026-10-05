"""Dump the kkt kernel's TTGIR / PTX / SASS (model config).   python scratch/kktir.py"""
import sys, runpy, os
sys.argv = ["ovl.py"]; sys.path.insert(0, '.'); os.environ["NOBENCH"] = "1"
g = runpy.run_path("scratch/ovl.py")
g["pre"]()
from agentopt.v3 import kernels as K
D = "/mnt/data/u/feic/tmp_ab/"
for dev, cache in K._gdn_kkt_kernel.device_caches.items():
    for key, c in cache[0].items():
        for ext in ("ttgir", "ptx", "sass"):
            if ext in c.asm:
                open(D + "kkt." + ext, "w").write(c.asm[ext])
