"""Build scratch/cutmm/cutmm.cu as a torch extension (CUTLASS from the flash-attention checkout)."""
import os
from torch.utils.cpp_extension import load
CUT = "/mnt/data/u/feic/pjs/src/flash-attention/csrc/cutlass"
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0a")
D = os.path.dirname(os.path.abspath(__file__))
def build(name="cutmm", src="cutmm.cu"):
    bd = f"/mnt/data/u/feic/tmp_ab/build_{name}"; os.makedirs(bd, exist_ok=True)
    return load(name, [os.path.join(D, src)], build_directory=bd, verbose=True,
                extra_include_paths=[f"{CUT}/include", f"{CUT}/tools/util/include"],
                extra_cuda_cflags=["-O3", "-std=c++17", "--expt-relaxed-constexpr", "-DNDEBUG",
                                   "-gencode=arch=compute_90a,code=sm_90a", "--use_fast_math"])
if __name__ == "__main__":
    build()
