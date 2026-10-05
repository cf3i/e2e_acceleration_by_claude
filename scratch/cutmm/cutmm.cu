// CUTLASS sm90 FP8 GEMM y[M,N] bf16 = alpha * x[M,K] w[N,K]^T, a few schedules (speed/precision vs cuBLAS).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/util/packed_stride.hpp"
using namespace cute;

template <class Tile, class Cluster, class KSched, class ESched>
struct Cfg {
  using EA = cutlass::float_e4m3_t;
  using Fusion = cutlass::epilogue::fusion::LinearCombination<cutlass::bfloat16_t, float, void, float>;
  using Epi = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, Tile, Cluster, cutlass::epilogue::collective::EpilogueTileAuto,
      float, float, void, cutlass::layout::RowMajor, 8, cutlass::bfloat16_t, cutlass::layout::RowMajor, 8,
      ESched, Fusion>::CollectiveOp;
  using Main = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, EA, cutlass::layout::RowMajor, 16, EA,
      cutlass::layout::ColumnMajor, 16, float, Tile, Cluster,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename Epi::SharedStorage))>,
      KSched>::CollectiveOp;
  using Kernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, Main, Epi,
                                                     cutlass::gemm::PersistentScheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};

template <class C>
void run(torch::Tensor x, torch::Tensor w, torch::Tensor y, double alpha, int swizzle, int raster) {
  using Gemm = typename C::Gemm;
  int M = x.size(0), K = x.size(1), N = w.size(0);
  auto sA = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideA{}, {M, K, 1});
  auto sB = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideB{}, {N, K, 1});
  auto sD = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideD{}, {M, N, 1});
  typename Gemm::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K, 1},
      {(typename C::EA*)x.data_ptr(), sA, (typename C::EA*)w.data_ptr(), sB},
      {{}, nullptr, sD, (cutlass::bfloat16_t*)y.data_ptr(), sD}};
  args.epilogue.thread.alpha = (float)alpha;
  args.epilogue.thread.beta = 0.f;
  args.scheduler.max_swizzle_size = swizzle;
  using RO = cutlass::gemm::kernel::detail::RasterOrderOptions;
  args.scheduler.raster_order = raster == 0 ? RO::Heuristic : raster == 1 ? RO::AlongM : RO::AlongN;
  Gemm gemm;
  TORCH_CHECK(gemm.can_implement(args) == cutlass::Status::kSuccess, "cannot implement");
  size_t ws = Gemm::get_workspace_size(args);
  auto wsb = torch::empty({(long)ws + 1}, x.options().dtype(torch::kUInt8));
  auto st = at::cuda::getCurrentCUDAStream();
  TORCH_CHECK(gemm.initialize(args, wsb.data_ptr(), st) == cutlass::Status::kSuccess, "init");
  TORCH_CHECK(gemm.run(st) == cutlass::Status::kSuccess, "run");
}

using namespace cutlass::gemm;
using EC = cutlass::epilogue::TmaWarpSpecializedCooperative;
using EP = cutlass::epilogue::TmaWarpSpecialized;
#define DEF(name, ...) void name(torch::Tensor x, torch::Tensor w, torch::Tensor y, double a, int64_t sw, int64_t ro) { run<Cfg<__VA_ARGS__>>(x, w, y, a, sw, ro); }
DEF(coop_128x128_c12, Shape<_128, _128, _128>, Shape<_1, _2, _1>, KernelTmaWarpSpecializedCooperative, EC)
DEF(coop_128x128_c21, Shape<_128, _128, _128>, Shape<_2, _1, _1>, KernelTmaWarpSpecializedCooperative, EC)
DEF(coop_128x128_c11, Shape<_128, _128, _128>, Shape<_1, _1, _1>, KernelTmaWarpSpecializedCooperative, EC)
DEF(coop_128x192_c21, Shape<_128, _192, _128>, Shape<_2, _1, _1>, KernelTmaWarpSpecializedCooperative, EC)
DEF(coop_256x96_c12, Shape<_256, _96, _128>, Shape<_1, _2, _1>, KernelTmaWarpSpecializedCooperative, EC)
DEF(ping_64x192_c21, Shape<_64, _192, _128>, Shape<_2, _1, _1>, KernelTmaWarpSpecializedPingpong, EP)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("coop_128x128_c12", &coop_128x128_c12);
  m.def("coop_128x128_c21", &coop_128x128_c21);
  m.def("coop_128x128_c11", &coop_128x128_c11);
  m.def("coop_128x192_c21", &coop_128x192_c21);
  m.def("coop_256x96_c12", &coop_256x96_c12);
  m.def("ping_64x192_c21", &ping_64x192_c21);
}
