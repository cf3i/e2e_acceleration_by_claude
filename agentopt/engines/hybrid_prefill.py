"""Let HF's Gated-DeltaNet prefill continue from a slot's saved state (prefill "extend").

HF's prefill path always starts from zero: `causal_conv1d_fn` on the new tokens only (zero left context) and
`chunk_gated_delta_rule(initial_state=None)`. To prefill only the new suffix of a sequence whose prefix is
already in a slot, both must continue the slot's history:
  - conv: prepend the previous K-1 inputs (the slot's conv window) and drop the K-1 extra outputs;
  - delta rule: pass the slot's recurrent state as `initial_state` (the fla kernel supports it).
Done by wrapping the two per-layer function attributes, so the HF forward itself is unchanged. The decode path
(`causal_conv1d_update` / fused recurrent rule, compiled + graph-captured) does not go through these wrappers.
"""


def install(model, cache):
    for layer in model.model.layers:
        if layer.layer_type != "linear_attention":
            continue
        la = layer.linear_attn
        la.causal_conv1d_fn = _wrap_conv(la.causal_conv1d_fn, cache)
        la.chunk_gated_delta_rule = _wrap_chunk(la.chunk_gated_delta_rule, cache, la.layer_idx)


def _wrap_conv(conv_fn, cache):
    import torch

    def conv(x, weight, bias=None, activation=None, seq_idx=None):
        if not cache.extending:
            return conv_fn(x=x, weight=weight, bias=bias, activation=activation, seq_idx=seq_idx)
        k = weight.shape[-1]
        # cache.prev_conv was saved by SlotCache.update_conv_state, which HF calls right before this.
        x = torch.cat([cache.prev_conv[None, :, 1:].to(x.dtype), x], -1)
        return conv_fn(x=x, weight=weight, bias=bias, activation=activation, seq_idx=seq_idx)[..., k - 1:]
    return conv


def _wrap_chunk(chunk_fn, cache, layer_idx):
    def chunk(q, k, v, g=None, beta=None, initial_state=None, **kw):
        if cache.extending:
            initial_state = cache.layers[layer_idx].rec_full[cache.slot:cache.slot + 1]
        return chunk_fn(q, k, v, g=g, beta=beta, initial_state=initial_state, **kw)
    return chunk
