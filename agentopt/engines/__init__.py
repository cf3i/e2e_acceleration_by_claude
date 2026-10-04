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
    "p1_s06": ("static", {**_S02, "cuda_graph": True,          # + torch.compile (inductor) the decode forward
                          "async_readback": True, "compile_decode": True}),
    "p1_s07": ("static", {**_S02, "cuda_graph": True,          # + GQA-folded GEMM attention for decode
                          "async_readback": True, "compile_decode": True, "gqa_decode_attn": True}),
}
# Phase 2 (continuous batching): "batch" engines are driven by the `continuous` runner.
_P1 = {**_S02, "compile_decode": True, "gqa_decode_attn": True}
PRESETS.update({
    "p2_s01": ("batch", {**_P1, "num_slots": 32}),                # continuous batching, 32 slots
    "p2_s02": ("batch", {**_P1, "num_slots": 32,                  # + cross-turn / shared-prefix state reuse
                         "prefix_reuse": True}),
    "p2_s03": ("batch", {**_P1, "num_slots": 32,                  # + short suffixes fed through the decode graph
                         "prefix_reuse": True, "force_max": 128}),
    "p2_s05": ("batch", {**_P1, "num_slots": 32,                  # + decode writes recurrent states in place
                         "prefix_reuse": True, "force_max": 128, "inplace_state": True}),
})


def make_engine(name: str, **kw):
    if name not in PRESETS:
        raise ValueError(f"unknown engine {name}")
    cls, opts = PRESETS[name]
    if cls == "naive":
        from .naive import NaiveEngine as Engine
    elif cls == "batch":
        from .batch import BatchEngine as Engine
    else:
        from .static import StaticEngine as Engine
    return Engine(**{**opts, **kw})
