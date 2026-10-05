import torch, os
one = torch.ones((), device="cuda")
def run(M, N, Kd, fp8=True):
    dt = torch.float8_e4m3fn if fp8 else torch.bfloat16
    xs = [torch.randn(M, Kd, device="cuda").to(dt) for _ in range(2)]
    ws = [torch.randn(N, Kd, device="cuda").to(dt) for _ in range(4)]
    s = torch.ones((), device="cuda")
    f = (lambda i: torch._scaled_mm(xs[i % 2], ws[i].t(), one, s, out_dtype=torch.bfloat16)) if fp8 else (lambda i: xs[i % 2] @ ws[i].t())
    [f(i) for i in range(4)]; torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for r in range(20):
            f(r % 4)
    g.replay(); torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record(); [g.replay() for _ in range(5)]; e1.record(); torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as p:
        f(0); torch.cuda.synchronize()
    names = [e.name[:40] for e in p.events() if e.device_type.name == "CUDA"]
    print(os.environ.get("CUBLASLT_WORKSPACE_SIZE"), M, N, Kd, fp8, f"{e0.elapsed_time(e1) / 100 * 1e3:.1f} us", names, flush=True)
for M in (1, 8, 32, 64, 128, 256, 1024):
    for N, Kd in [(5120, 17408), (5120, 6144), (5120, 4096), (34816, 5120)]:
        run(M, N, Kd)
for M in (1, 56, 128):
    run(M, 248320, 5120, False)
