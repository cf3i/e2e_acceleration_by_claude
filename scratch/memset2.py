import torch, os
one = torch.ones((), device="cuda")
for M in (2048, 512):
    for N, Kd in [(34816, 5120), (5120, 17408), (16480, 5120), (5120, 4096)]:
        xs = [torch.randn(M, Kd, device="cuda").to(torch.float8_e4m3fn) for _ in range(2)]
        ws = [torch.randn(N, Kd, device="cuda").to(torch.float8_e4m3fn) for _ in range(2)]
        s = torch.ones((), device="cuda")
        f = lambda i: torch._scaled_mm(xs[i], ws[i].t(), one, s, out_dtype=torch.bfloat16)
        f(0); f(1); torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for r in range(10):
                f(r % 2)
        g.replay(); torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record(); [g.replay() for _ in range(5)]; e1.record(); torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as p:
            f(0); torch.cuda.synchronize()
        names = [e.name[:44] for e in p.events() if e.device_type.name == "CUDA"]
        print(os.environ.get("CUBLASLT_WORKSPACE_SIZE"), M, N, Kd, f"{e0.elapsed_time(e1) / 50 * 1e3:.1f} us", names)
