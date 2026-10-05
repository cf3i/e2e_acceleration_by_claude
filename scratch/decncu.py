"""ncu target: gdn_decode (FN=dec) or gdn_prefill (FN=pre) of ovl.py alone.   ncu -k regex:_gdn_decode -s 2 -c 1 python scratch/decncu.py"""
import sys, runpy, torch
sys.argv = ["ovl.py"]; sys.path.insert(0, '.')
import os; os.environ["NOBENCH"] = "1"
g = runpy.run_path("scratch/ovl.py")
fn = g[os.environ.get("FN", "dec")]
for _ in range(4): fn()
torch.cuda.synchronize()
