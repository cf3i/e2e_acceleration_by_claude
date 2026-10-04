"""Explicit control over HF Qwen3.5 linear-attention fast paths.

HF silently switches Gated-DeltaNet layers to `flash-linear-attention` / `causal-conv1d` kernels when those
packages are importable. Engines choose explicitly so results do not depend on what happens to be installed.
Must be called before the model is constructed (layers bind the functions in __init__).
"""
import transformers.models.qwen3_5.modeling_qwen3_5 as qm


def set_linear_attn_fast_path(enabled: bool):
    if enabled:
        from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
        from fla.modules import FusedRMSNormGated
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
    else:
        causal_conv1d_fn = causal_conv1d_update = FusedRMSNormGated = None
        chunk_gated_delta_rule = fused_recurrent_gated_delta_rule = None
    qm.causal_conv1d_fn, qm.causal_conv1d_update = causal_conv1d_fn, causal_conv1d_update
    qm.chunk_gated_delta_rule = chunk_gated_delta_rule
    qm.fused_recurrent_gated_delta_rule = fused_recurrent_gated_delta_rule
    qm.FusedRMSNormGated = FusedRMSNormGated
    qm.is_fast_path_available = enabled
