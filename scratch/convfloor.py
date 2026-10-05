import sys, torch
sys.path.insert(0, '.'); sys.path.insert(0, 'scratch')
from agentopt.v3 import kernels as K
from benchutil import bench_loop
T, C = 2048, 10240
bufs = [torch.randn(T, 16480, device='cuda', dtype=torch.bfloat16) for _ in range(3)]
xs = torch.rand(T, device='cuda') + 0.5; w = torch.randn(C, 4, device='cuda')
state = torch.randn(130, 3, C, device='cuda', dtype=torch.bfloat16)
for lens in [(1992, 56), (1024, 968, 56)]:
    rs, rsl, ss, sst, sl = [], [], [], [], []
    p = 0
    for i, n in enumerate(lens):
        if i == len(lens) - 1:  # decode rows: one per seq
            for j in range(n):
                rs.append(p); rsl.append(10 + j); ss.append(10 + j); sst.append(p); sl.append(1); p += 1
        else:
            rs += [p] * n; rsl += [100 + i] * n; ss.append(100 + i); sst.append(p); sl.append(n); p += n
    t = lambda a: torch.tensor(a, dtype=torch.int32, device='cuda')
    rs, rsl, ss, sst, sl = map(t, (rs, rsl, ss, sst, sl))
    fns = [lambda b=b: K.conv1d_rows(b[:, :C], xs, w, state, rs, rsl, ss, sst, sl, ss.shape[0]) for b in bufs]
    tc = bench_loop(fns, iters=12)
    ys = [torch.empty(T, C, device='cuda', dtype=torch.bfloat16) for _ in range(3)]
    cp = [lambda b=b, y=y: y.copy_(b[:, :C]) for b, y in zip(bufs, ys)]
    tp = bench_loop(cp, iters=12)
    print(f"{lens}: conv1d_rows (+state) {tc * 1e6:.1f} us, strided copy of the same bytes {tp * 1e6:.1f} us "
          f"({2 * T * C * 2 / tp / 1e12:.2f} TB/s)")
