"""Fused conv epilogue (mm_fp8 conv=...) + gdn_decode(norm=True) vs mm_fp8 + conv1d_varlen + gdn_decode."""
import sys, torch
sys.path.insert(0, '.')
from agentopt.v3 import kernels as K
torch.manual_seed(0)
dev = 'cuda'; F8 = torch.float8_e4m3fn
HK, HV, DK, DV, C, H = 16, 48, 128, 128, 10240, 5120
N = C + 6144 + 96
S = 10
for M in (1, 5, 16, 33, 64, 100):
    w = (torch.randn(N, H, device=dev) * .3).to(F8); sv = torch.rand(N, device=dev) * 0.02 + 0.01
    x = (torch.randn(M, H, device=dev) * 4).to(F8); xs = torch.rand(M, device=dev) * 0.1 + 0.01
    cw = torch.randn(C, 4, device=dev) * .5
    st0 = (torch.randn(S + 1, 3, C, device=dev)).to(torch.bfloat16)
    slots = torch.randperm(S, device=dev)[:M].to(torch.int32) if M <= S else torch.arange(M, device=dev, dtype=torch.int32) % (S + 1)
    if M > S:  # distinct slots needed: bigger pool
        st0 = torch.randn(M + 1, 3, C, device=dev).to(torch.bfloat16); slots = torch.randperm(M, device=dev).to(torch.int32)
    cfg = K.mm_fp8_cfg(M, N)
    # reference
    y = K.mm_fp8(x, w, sv, cfg)
    st1 = st0.clone()
    ar = torch.arange(M, device=dev, dtype=torch.int32)
    q, k, v = K.conv1d_varlen(y[:, :C], xs, cw, st1, slots, ar, torch.ones_like(ar), M)
    # fused
    st2 = st0.clone()
    y2 = K.mm_fp8(x, w, sv, cfg, (xs, cw, st2, slots))
    print(f"M={M}: z/b/a equal {torch.equal(y[:, C:], y2[:, C:])}, state equal {torch.equal(st1, st2)}, "
          f"v equal {torch.equal(v.reshape(M, -1), y2[:, 2 * HK * DK:C])}", end=' ')
    # unnormalized q vs normalized q: normalize y2 q in torch the kernel's way
    qq = y2[:, :HK * DK].float().reshape(M, HK, DK)
    qn = (qq * torch.rsqrt((qq * qq).sum(-1, keepdim=True) + 1e-6)).to(torch.bfloat16)
    print(f"q(norm in torch) mismatches {(qn != q).sum().item()} / {q.numel()}", end=' ')
    # full decode compare
    rec = torch.randn(S + 1 if M <= S else M + 1, HV, DK, DV, device=dev).to(torch.bfloat16)
    a = y[:, C + 6144 + 48:]; b = y[:, C + 6144:C + 6144 + 48]
    A_log = torch.randn(HV, device=dev); dtb = torch.randn(HV, device=dev)
    r1, r2 = rec.clone(), rec.clone()
    o1 = torch.empty(M, HV, DV, device=dev, dtype=torch.bfloat16); o2 = torch.empty_like(o1)
    row0 = torch.zeros(1, device=dev, dtype=torch.int32)
    K.gdn_decode(q, k, v, a, b, xs, A_log, dtb, r1, slots, row0, M, o1)
    K.gdn_decode(y2[:, :HK * DK], y2[:, HK * DK:2 * HK * DK], y2[:, 2 * HK * DK:C], a, b, xs, A_log, dtb, r2, slots, row0, M, o2, norm=True)
    print(f"| o equal {torch.equal(o1, o2)} (max diff {(o1.float() - o2.float()).abs().max().item():.2e}), rec equal {torch.equal(r1, r2)}")
