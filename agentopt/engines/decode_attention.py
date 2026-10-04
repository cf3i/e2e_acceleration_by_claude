"""GQA-folded single-token attention for decode over a masked static KV cache.

With an explicit mask, HF's SDPA path first expands K/V from 4 to 24 heads (`repeat_kv`: a 6x copy of the whole
cache, every layer, every token) and then runs the memory-efficient kernel, which for one query token gets one
thread block per head and walks all `max_cache_len` slots serially (~245 us/layer at 4096 slots).

For q_len == 1 this computes the same attention with plain tensor ops: the 6 query heads that share a KV head are
folded into the query axis, so scores = (b, 4, 6, d) @ (b, 4, d, L) and out = softmax(scores) @ (b, 4, L, d) —
two batched GEMMs that read each K/V slot once (no copy) and that cuBLAS parallelizes along L. Softmax in fp32
like HF's eager path. Prefill (q_len > 1) is delegated to HF's SDPA unchanged.
"""
import torch
from transformers import AttentionInterface
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.masking_utils import AttentionMaskInterface, sdpa_mask

NAME = "sdpa_gqa_decode"


def gqa_decode_attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
    b, hq, q_len, d = query.shape
    if q_len != 1 or attention_mask is None or attention_mask.dtype != torch.bool:
        return sdpa_attention_forward(module, query, key, value, attention_mask, scaling=scaling,
                                      dropout=dropout, **kwargs)
    hk = key.shape[1]
    scale = scaling if scaling is not None else d ** -0.5
    q = query.reshape(b, hk, hq // hk, d)  # query head h uses KV head h // (hq // hk), as in repeat_kv
    scores = torch.matmul(q, key.transpose(-1, -2)).float() * scale  # (b, hk, g, L)
    scores = scores.masked_fill(~attention_mask, float("-inf"))  # mask (b, 1, 1, L) broadcasts
    probs = torch.softmax(scores, dim=-1).to(value.dtype)
    out = torch.matmul(probs, value)  # (b, hk, g, d)
    return out.reshape(b, hq, 1, d).transpose(1, 2).contiguous(), None  # (b, q_len, hq, d) like SDPA


def install(model):
    AttentionInterface.register(NAME, gqa_decode_attention)
    AttentionMaskInterface.register(NAME, sdpa_mask)  # prefill masks are built exactly as for "sdpa"
    model.set_attn_implementation(NAME)
    assert model.config._attn_implementation == NAME, model.config._attn_implementation
