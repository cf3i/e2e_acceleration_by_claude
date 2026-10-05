"""kkt kernel time alone (GATES + AQ as in the model), 1992 tokens.   python scratch/kktt.py"""
import sys, runpy, os, torch
sys.argv = ["ovl.py"]; sys.path.insert(0, '.'); sys.path.insert(0, 'scratch'); os.environ["NOBENCH"] = "1"
g = runpy.run_path("scratch/ovl.py")
from benchutil import bench_loop
from agentopt.v3 import kernels as K
# time kkt alone: prefill with the chunk launch made a no-op
ck = K._gdn_chunk_kernel
class Nop:
    def __getitem__(self, grid): return lambda *a, **k: None
if "KKT" in os.environ:
    K.GDN_KKT_CFG = tuple(map(int, os.environ["KKT"].split(",")))
tp = bench_loop([g["pre"]], iters=10)
K._gdn_chunk_kernel = Nop()
tk = bench_loop([g["pre"]], iters=10)
print(f"cfg {K.GDN_KKT_CFG} KKT_FS {os.environ.get('KKT_FS', '1')} AQ {K.GDN_AQ}: prefill {tp * 1e6:.1f} us, kkt alone {tk * 1e6:.1f} us", flush=True)
