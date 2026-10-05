# conv1d_rows vs a Triton strided copy (true bandwidth floor) and other tile configs
import sys, torch, triton, triton.language as tl
sys.path.insert(0, '.'); sys.path.insert(0, 'scratch')
from agentopt.v3 import kernels as K
from benchutil import bench_loop
T, C = 2048, 10240

@triton.jit
def _cp(x, y, T, XS, C: tl.constexpr, BT: tl.constexpr, BC: tl.constexpr):
    r = tl.program_id(0) * BT + tl.arange(0, BT)
    c = tl.program_id(1) * BC + tl.arange(0, BC)
    m = (r < T)[:, None]
    v = tl.load(x + r[:, None].to(tl.int64) * XS + c[None, :], mask=m)
    tl.store(y + r[:, None].to(tl.int64) * C + c[None, :], v, mask=m)

bufs = [torch.randn(T, 16480, device='cuda', dtype=torch.bfloat16) for _ in range(3)]
ys = [torch.empty(T, C, device='cuda', dtype=torch.bfloat16) for _ in range(3)]
GB = 2 * T * C * 2
for BT, BC, nw in [(16, 128, 4), (32, 256, 4), (64, 128, 4), (16, 512, 4), (8, 1024, 4), (32, 512, 8), (64, 256, 8)]:
    fns = [lambda b=b, y=y: _cp[(triton.cdiv(T, BT), C // BC)](b, y, T, b.stride(0), C=C, BT=BT, BC=BC, num_warps=nw)
           for b, y in zip(bufs, ys)]
    t = bench_loop(fns, iters=12)
    print(f"triton copy BT{BT} BC{BC} w{nw}: {t * 1e6:.1f} us {GB / t / 1e12:.2f} TB/s")
xs = torch.rand(T, device='cuda') + 0.5; w = torch.randn(C, 4, device='cuda')
state = torch.randn(130, 3, C, device='cuda', dtype=torch.bfloat16)
rs = torch.tensor([0] * 1992 + list(range(1992, 2048)), dtype=torch.int32, device='cuda')
rsl = torch.tensor([100] * 1992 + list(range(10, 66)), dtype=torch.int32, device='cuda')
for BT, BC, nw in [(16, 128, 4), (32, 128, 4), (16, 256, 4), (32, 256, 8), (64, 128, 8), (8, 512, 4), (16, 512, 8)]:
    def f(b):
        q, k, v = K._conv_outs(T, b.device)
        K._conv1d_rows_kernel[(triton.cdiv(T, BT), C // BC)](b, xs, w, state, q, k, v, rs, rsl, T, b.stride(0),
            QK=2048, VC=C - 4096, BT=BT, BLOCK_C=BC, DK=128, num_warps=nw)
    t = bench_loop([lambda b=b: f(b) for b in bufs], iters=12)
    print(f"conv_rows BT{BT} BC{BC} w{nw}: {t * 1e6:.1f} us {GB / t / 1e12:.2f} TB/s")
