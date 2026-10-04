"""Decode-time Gated-DeltaNet update that writes the new recurrent state in place.

HF's decode path calls fla's `fused_recurrent_gated_delta_rule(initial_state=cache_state, output_final_state=True)`,
which reads the state and writes the new state into a freshly allocated tensor; the cache then copies it back
(`update_recurrent_state` -> copy_). At batch 32 the fp32 states are 4.8 GB, so that copy is another full read +
write of them every step (p2_s03 profile: `triton_poi_fused_copy_` 3.2 ms of a 29.8 ms step).

The fla kernel itself loads each (head, V-block) tile of h0 at the start and stores the same tile of ht at the
end, one program per tile, so passing the cache buffer as both h0 and ht is safe. This module launches that same
fla kernel (as fla 0.5's `fused_recurrent_gated_delta_rule_fwd` does) with ht = h0, and tells the slot cache to
skip its copy. No new kernel: only where the existing one writes.
"""
import torch
import triton
from fla.ops.gated_delta_rule.fused_recurrent import fused_recurrent_gated_delta_rule_fwd_kernel


@torch.compiler.disable
def recurrent_inplace(q, k, v, g=None, beta=None, initial_state=None, output_final_state=False,
                      use_qk_l2norm_in_kernel=False, scale=None, **kw):
    assert initial_state is not None and initial_state.dtype == torch.float32 and initial_state.is_contiguous()
    q, k, v, g, beta = (x.contiguous() for x in (q, k, v, g, beta))
    B, T, H, K = k.shape
    HV, V = v.shape[2], v.shape[-1]
    BK, BV = triton.next_power_of_2(K), min(8, triton.next_power_of_2(V))
    o = torch.empty_like(v)
    fused_recurrent_gated_delta_rule_fwd_kernel[(triton.cdiv(V, BV), B * HV)](
        q=q, k=k, v=v, g=g, gk=None, gv=None, beta=beta, A_log=None, dt_bias=None, o=o,
        h0=initial_state, ht=initial_state, cu_seqlens=None, scale=K ** -0.5 if scale is None else scale,
        T=T, H=H, HV=HV, K=K, V=V, BK=BK, BV=BV, IS_BETA_HEADWISE=beta.ndim != v.ndim,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel, APPLY_BETA_SIGMOID=False, ALLOW_NEG_EIGVAL=False,
        STATE_V_FIRST=False, num_warps=1, num_stages=3,
    )
    return o, initial_state


def install(model, cache):
    for layer in model.model.layers:
        if layer.layer_type == "linear_attention":
            layer.linear_attn.recurrent_gated_delta_rule = recurrent_inplace
    cache.recurrent_inplace = True
