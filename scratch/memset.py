"""Which _scaled_mm shapes put a Memset in front of the cuBLAS kernel (in a CUDA graph)?"""
import torch
torch.manual_seed(0)
one = torch.ones((), device="cuda")
shapes = [(5120, 12352 + 0), (17408 * 2, 5120), (5120, 17408), (5120, 6144), (5120, 4096)]
for M in (2048, 512, 128, 56):
    for N, Kd in [(34816, 5120), (5120, 17408), (5120, 6144), (16480, 5120), (5120, 4096)]:
        x = torch.randn(M, Kd, device="cuda").to(torch.float8_e4m3fn)
        w = torch.randn(N, Kd, device="cuda").to(torch.float8_e4m3fn)
        s = torch.ones((), device="cuda")
        f = lambda: torch._scaled_mm(x, w.t(), one, s, out_dtype=torch.bfloat16)
        f(); torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as p:
            f(); torch.cuda.synchronize()
        names = [e.name[:50] for e in p.events() if e.device_type.name == "CUDA"]
        print(M, N, Kd, names)
