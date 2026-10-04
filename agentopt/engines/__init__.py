# Engine presets: each phase-1 stage is the previous preset plus one change, so every
# stage stays reproducible by name (`--engine <preset>`).
_S02 = {"cudnn_sdpa": False, "linear_attn_fast_path": True}
PRESETS = {
    "naive": ("naive", {}),                                    # stage00_baseline
    "p1_s01": ("naive", {"cudnn_sdpa": False}),                # avoid per-shape cuDNN SDPA plan building
    "p1_s02": ("naive", _S02),                                 # + fla / causal-conv1d kernels for Gated-DeltaNet
    "p1_s03": ("static", _S02),                                # + preallocated StaticCache, in-place decode buffers
    "p1_s04": ("static", {**_S02, "cuda_graph": True}),        # + CUDA-graph-captured decode step
    "p1_s05": ("static", {**_S02, "cuda_graph": True,          # + overlap graph launch with the previous step
                          "async_readback": True}),
}


def make_engine(name: str, **kw):
    if name not in PRESETS:
        raise ValueError(f"unknown engine {name}")
    cls, opts = PRESETS[name]
    if cls == "naive":
        from .naive import NaiveEngine as Engine
    else:
        from .static import StaticEngine as Engine
    return Engine(**{**opts, **kw})
