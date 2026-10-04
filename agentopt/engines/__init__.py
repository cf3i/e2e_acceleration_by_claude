# Engine presets: each phase-1 stage is the previous preset plus one change, so every
# stage stays reproducible by name (`--engine <preset>`).
PRESETS = {
    "naive": {},                                     # stage00_baseline
    "p1_s01": {"cudnn_sdpa": False},                 # avoid per-shape cuDNN SDPA plan building
    "p1_s02": {"cudnn_sdpa": False,                  # + fla / causal-conv1d kernels for Gated-DeltaNet layers
               "linear_attn_fast_path": True},
}


def make_engine(name: str, **kw):
    if name in PRESETS:
        from .naive import NaiveEngine
        return NaiveEngine(**{**PRESETS[name], **kw})
    raise ValueError(f"unknown engine {name}")
