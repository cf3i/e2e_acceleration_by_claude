import torch, sys
torch.manual_seed(0); one = torch.ones((), device="cuda")
out = []
for M in (2048, 1024, 512, 256):
    for N, Kd in [(34816, 5120), (5120, 17408), (16480, 5120), (5120, 4096)]:
        x = torch.randn(M, Kd, device="cuda").to(torch.float8_e4m3fn); w = torch.randn(N, Kd, device="cuda").to(torch.float8_e4m3fn)
        out.append(torch._scaled_mm(x, w.t(), one, torch.full((), 0.01, device="cuda"), out_dtype=torch.bfloat16).cpu())
torch.save(out, sys.argv[1])
