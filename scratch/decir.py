"""Dump the decode kernel's layouts / reduction code.   python scratch/decir.py"""
import sys, runpy, os
sys.argv = ["ovl.py"]; sys.path.insert(0, '.'); os.environ["NOBENCH"] = "1"
g = runpy.run_path("scratch/ovl.py")
g["dec"]()
from agentopt.v3 import kernels as K
for dev, cache in K._gdn_decode_kernel.device_caches.items():
    for key, c in cache[0].items():
        open("/mnt/data/u/feic/tmp_ab/dec.ttgir", "w").write(c.asm["ttgir"]); open("/mnt/data/u/feic/tmp_ab/dec.sass", "w").write(c.asm.get("sass", ""))
        open("/mnt/data/u/feic/tmp_ab/dec.ptx", "w").write(c.asm["ptx"])
