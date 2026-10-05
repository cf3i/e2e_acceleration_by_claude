"""kkt inverse accuracy on correlated keys (real keys in a chunk are far from random): ours vs fla vs fp64."""
import sys, torch
sys.path.insert(0, '.'); sys.path.insert(0, 'scratch')
from fla.ops.utils.constant import RCP_LN2
from fla.ops.utils import chunk_local_cumsum
from fla.ops.gated_delta_rule.chunk_fwd import chunk_gated_delta_rule_fwd_kkt_solve_kernel
from agentopt.v3 import kernels as K
dev = 'cuda'; bf = torch.bfloat16
HK, HV, DK = 16, 48, 128
torch.manual_seed(0)


def ref64(k, gc, beta):
    T = k.shape[0]
    out = torch.zeros(T, HV, 64, dtype=torch.float64, device=dev)
    kk = k.double()
    for c0 in range(0, T, 64):
        n = min(64, T - c0)
        kc = kk[c0:c0 + n].repeat_interleave(HV // HK, dim=1).transpose(0, 1)
        gg = gc[c0:c0 + n].double().t(); bb = beta[c0:c0 + n].double().t()
        x = torch.tril((kc @ kc.transpose(1, 2)) * torch.exp2(gg[:, :, None] - gg[:, None, :]) * bb[:, :, None], -1)
        out[c0:c0 + n, :, :n] = torch.linalg.inv(torch.eye(n, dtype=torch.float64, device=dev) + x).transpose(0, 1)
    return out


T = 1024
cu = torch.tensor([0, T], device=dev); ci = torch.tensor([(0, t) for t in range(T // 64)], device=dev)
for eps in (10.0, 1.0, 0.3, 0.1, 0.03):
    for decay in (0.5, 0.05, 0.005):
        base = torch.randn(1, HK, DK, device=dev)
        k = torch.nn.functional.normalize(base + eps * torch.randn(T, HK, DK, device=dev) / DK ** .5 * DK ** .5 / 10, dim=-1).to(bf)
        g = -decay * torch.rand(T, HV, device=dev)
        beta = 0.5 + 0.5 * torch.rand(T, HV, device=dev)
        gc = chunk_local_cumsum(g[None], chunk_size=64, scale=RCP_LN2, cu_seqlens=cu, chunk_indices=ci)[0]
        r = ref64(k, gc, beta)
        Af = torch.zeros(1, T, HV, 64, device=dev, dtype=bf)
        chunk_gated_delta_rule_fwd_kkt_solve_kernel[(ci.shape[0], HV)](k=k[None], g=gc[None], beta=beta[None], A=Af, cu_seqlens=cu,
                                                                     chunk_indices=ci, T=T, H=HK, HV=HV, K=DK, BT=64, BC=16)
        gc2 = torch.empty_like(g); A2 = torch.empty(T, HV, 64, device=dev, dtype=bf)
        K._gdn_kkt_kernel[(ci.shape[0], HV)](k, g, gc2, beta, A2, cu, ci, RCP_LN2, H=HK, HV=HV, K=DK, BT=64, HPP=1, PREC="tf32", num_warps=1)
        sc = r.abs().amax().item()
        ef = (Af[0].double() - r).abs().amax().item(); e2 = (A2.double() - r).abs().amax().item()
        print(f"eps {eps:5} decay {decay:6}: max|A| {sc:9.3g}  fla err {ef:9.3g}  ours err {e2:9.3g}", flush=True)
