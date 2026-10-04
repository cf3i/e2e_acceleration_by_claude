"""FP8 linear layers via torchao (dynamic per-row activation scale x per-row FP8 weight, `torch._scaled_mm`).

At batch 32 a decode step is ~70% GEMM and the GEMMs are bound by reading the bf16 weights (54 GB/step). E4M3
weights halve that read; activations are quantized per token on the fly. lm_head stays bf16 (logit precision
decides greedy ties; it is 2% of the weights).
"""
import torch


def fp8_quantize(model):
    from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, PerRow, quantize_

    def keep(mod, fqn):
        return (isinstance(mod, torch.nn.Linear) and not fqn.endswith("lm_head")
                and mod.in_features % 16 == 0 and mod.out_features % 16 == 0)

    n_before = sum(p.numel() * p.element_size() for p in model.parameters())
    quantize_(model, Float8DynamicActivationFloat8WeightConfig(granularity=PerRow()), filter_fn=keep)
    torch.cuda.empty_cache()
    return n_before
