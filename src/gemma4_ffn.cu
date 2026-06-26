#include "gemma4_ffn.cuh"

#include <cute/layout.hpp>
#include <cutlass/array.h>
#include <cutlass/epilogue/thread/activation.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/epilogue/thread/scale_type.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

#include "device/dual_gemm.h"


namespace gemma4_ffn_decode_device {

// Loads one BF16 pack with a build-selected cache policy for FFN load ablations.
__device__ inline int4 ffn_load_int4(const __nv_bfloat16 *ptr) {
  const int4 *address = reinterpret_cast<const int4 *>(ptr);
#if !defined(GEMMA4_FFN_VECTOR_LOAD_POLICY) || GEMMA4_FFN_VECTOR_LOAD_POLICY == 0
  return *address;
#elif GEMMA4_FFN_VECTOR_LOAD_POLICY == 1
  return __ldg(address);
#elif GEMMA4_FFN_VECTOR_LOAD_POLICY == 2
  unsigned int x;
  unsigned int y;
  unsigned int z;
  unsigned int w;
  asm volatile("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(x), "=r"(y), "=r"(z), "=r"(w)
               : "l"(address));
  int4 value;
  value.x = static_cast<int>(x);
  value.y = static_cast<int>(y);
  value.z = static_cast<int>(z);
  value.w = static_cast<int>(w);
  return value;
#elif GEMMA4_FFN_VECTOR_LOAD_POLICY == 3
  unsigned int x;
  unsigned int y;
  unsigned int z;
  unsigned int w;
  asm volatile("ld.global.ca.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(x), "=r"(y), "=r"(z), "=r"(w)
               : "l"(address));
  int4 value;
  value.x = static_cast<int>(x);
  value.y = static_cast<int>(y);
  value.z = static_cast<int>(z);
  value.w = static_cast<int>(w);
  return value;
#elif GEMMA4_FFN_VECTOR_LOAD_POLICY == 4
  unsigned int x;
  unsigned int y;
  unsigned int z;
  unsigned int w;
  asm volatile("ld.global.cs.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(x), "=r"(y), "=r"(z), "=r"(w)
               : "l"(address));
  int4 value;
  value.x = static_cast<int>(x);
  value.y = static_cast<int>(y);
  value.z = static_cast<int>(z);
  value.w = static_cast<int>(w);
  return value;
#else
#error Unsupported GEMMA4_FFN_VECTOR_LOAD_POLICY
#endif
}

// Swizzles one row of hidden-width BF16 packs for decode-friendly layout.
__device__ inline void swizzle_hidden_packs(
    __nv_bfloat16 *__restrict__ dst,
    const __nv_bfloat16 *__restrict__ src,
    int row_count,
    int row,
    int block_idx,
    int grid_x,
    int block_dim,
    int thread_idx) {
  constexpr int hidden_size = kHiddenPacks * kBf16Packed128Elements;
  const auto row_layout = cute::make_layout(
      cute::make_shape(row_count, hidden_size),
      cute::make_stride(
          static_cast<int64_t>(hidden_size),
          static_cast<int64_t>(1)));
  const auto hidden_layout = cute::make_layout(
      cute::make_shape(kHiddenPacks),
      cute::make_stride(kBf16Packed128Elements));
  const int pack_stride = grid_x * block_dim;
  for (int pack = block_idx * block_dim + thread_idx;
       pack < kHiddenPacks; pack += pack_stride) {
    const int src_col = hidden_layout(pack);
    const int dst_col = hidden_layout(hidden_pack_swizzle_index(pack));
    const int64_t src_offset = row_layout(row, src_col);
    const int64_t dst_offset = row_layout(row, dst_col);
    const FfnBf16Pack pack_value = FfnBf16Pack{
        ffn_load_int4(src + src_offset)};
    *reinterpret_cast<int4 *>(dst + dst_offset) = pack_value.bits();
  }
}

// RMS-normalizes a swizzled down-projection row, then adds the residual.
template <int Threads>
__device__ inline void rmsnorm_residual_from_swizzled_down(
    __nv_bfloat16 *__restrict__ residual_out,
    __nv_bfloat16 *__restrict__ normed_out,
    const __nv_bfloat16 *__restrict__ down_swizzled,
    const __nv_bfloat16 *__restrict__ residual,
    const __nv_bfloat16 *__restrict__ rms_weight,
    float eps,
    float *__restrict__ warp_sums,
    float &scale,
    int row_count,
    int row,
    int thread_idx) {
  const int hidden_pack = thread_idx;
  constexpr int hidden_size = kHiddenPacks * kBf16Packed128Elements;
  const auto row_layout = cute::make_layout(
      cute::make_shape(row_count, hidden_size),
      cute::make_stride(
          static_cast<int64_t>(hidden_size),
          static_cast<int64_t>(1)));
  const auto hidden_layout = cute::make_layout(
      cute::make_shape(kHiddenPacks),
      cute::make_stride(kBf16Packed128Elements));
  const int natural_col = hidden_layout(hidden_pack);
  const int swizzled_col = hidden_layout(hidden_pack_swizzle_index(hidden_pack));
  const int64_t down_offset = row_layout(row, swizzled_col);
  const int64_t natural_offset = row_layout(row, natural_col);

  const FfnBf16Pack down_pack = FfnBf16Pack{
      ffn_load_int4(down_swizzled + down_offset)};
  float sum_sq = 0.0f;
  gemma4_bf16_pack_accumulate_square(down_pack, sum_sq);

  const float total =
      gemma4_block_reduce_sum<Threads>(sum_sq, warp_sums, thread_idx);
  if (thread_idx == 0) {
    scale = rsqrtf(total / static_cast<float>(GEMMA4_HIDDEN_SIZE) + eps);
  }
  __syncthreads();

  const FfnBf16Pack gamma_pack = FfnBf16Pack{
      ffn_load_int4(rms_weight + natural_col)};
  const FfnBf16Pack normed_pack =
      gemma4_bf16_pack_apply_scale_weight(down_pack, gamma_pack, scale);
  const FfnBf16Pack residual_pack = FfnBf16Pack{
      ffn_load_int4(residual + natural_offset)};
  const FfnBf16Pack residual_out_pack =
      gemma4_bf16_pack_add(residual_pack, normed_pack);
  *reinterpret_cast<int4 *>(normed_out + natural_offset) = normed_pack.bits();
  *reinterpret_cast<int4 *>(residual_out + natural_offset) = residual_out_pack.bits();
}

// Restores swizzled hidden packs to natural hidden-column order.
__device__ inline void unswizzle_hidden_packs(
    __nv_bfloat16 *__restrict__ dst,
    const __nv_bfloat16 *__restrict__ src,
    int row_count,
    int row,
    int block_dim,
    int thread_idx) {
  constexpr int hidden_size = kHiddenPacks * kBf16Packed128Elements;
  const auto row_layout = cute::make_layout(
      cute::make_shape(row_count, hidden_size),
      cute::make_stride(
          static_cast<int64_t>(hidden_size),
          static_cast<int64_t>(1)));
  const auto hidden_layout = cute::make_layout(
      cute::make_shape(kHiddenPacks),
      cute::make_stride(kBf16Packed128Elements));

  for (int hidden_pack = thread_idx; hidden_pack < kHiddenPacks;
       hidden_pack += block_dim) {
    const int natural_col = hidden_layout(hidden_pack);
    const int swizzled_col =
        hidden_layout(hidden_pack_swizzle_index(hidden_pack));
    const int64_t src_offset = row_layout(row, swizzled_col);
    const int64_t dst_offset = row_layout(row, natural_col);
    const FfnBf16Pack pack = FfnBf16Pack{
        ffn_load_int4(src + src_offset)};
    *reinterpret_cast<int4 *>(dst + dst_offset) = pack.bits();
  }
}

// Interleaves gate/up rows while applying the hidden-pack decode swizzle.
__device__ inline void swizzle_gate_up_interleaved(
    __nv_bfloat16 *__restrict__ dst,
    const __nv_bfloat16 *__restrict__ src,
    int row_count,
    int dst_row,
    int block_idx,
    int grid_x,
    int block_dim,
    int thread_idx) {
  const int src_row =
      (dst_row & 1) == 0 ? dst_row / 2
                         : GEMMA4_INTERMEDIATE_SIZE + dst_row / 2;
  constexpr int hidden_size = kHiddenPacks * kBf16Packed128Elements;
  const auto row_layout = cute::make_layout(
      cute::make_shape(row_count, hidden_size),
      cute::make_stride(
          static_cast<int64_t>(hidden_size),
          static_cast<int64_t>(1)));
  const auto hidden_layout = cute::make_layout(
      cute::make_shape(kHiddenPacks),
      cute::make_stride(kBf16Packed128Elements));
  const int pack_stride = grid_x * block_dim;
  for (int pack = block_idx * block_dim + thread_idx;
       pack < kHiddenPacks; pack += pack_stride) {
    const int src_col = hidden_layout(pack);
    const int dst_col = hidden_layout(hidden_pack_swizzle_index(pack));
    const int64_t src_offset = row_layout(src_row, src_col);
    const int64_t dst_offset = row_layout(dst_row, dst_col);
    const FfnBf16Pack pack_value = FfnBf16Pack{
        ffn_load_int4(src + src_offset)};
    *reinterpret_cast<int4 *>(dst + dst_offset) = pack_value.bits();
  }
}

}  // namespace gemma4_ffn_decode_device

namespace {

namespace ffn_dev = gemma4_ffn_decode_device;

constexpr int kFfnThreads = GEMMA4_FFN_DECODE_THREADS;
constexpr int kFfnWarps = kFfnThreads / 32;
constexpr int kHiddenPacks = ffn_dev::kHiddenPacks;
static_assert(kFfnThreads == kHiddenPacks,
              "FFN decode maps one CTA thread to one hidden pack");
constexpr int kSwizzleThreads = 96;
constexpr int kActualSwizzleBlocksPerRow =
    div_up(kHiddenPacks, kSwizzleThreads);
#ifndef GEMMA4_FFN_GATE_UP_STAGE_ROWS64
static constexpr int GEMMA4_FFN_GATE_UP_STAGE_ROWS64 = 3;
#endif
#ifndef GEMMA4_FFN_GATE_UP_STAGE_ROWS128
static constexpr int GEMMA4_FFN_GATE_UP_STAGE_ROWS128 = 5;
#endif
#ifndef GEMMA4_FFN_GATE_UP_STAGE_DEFAULT
static constexpr int GEMMA4_FFN_GATE_UP_STAGE_DEFAULT = 3;
#endif
#ifndef GEMMA4_FFN_DOWN_STAGE_ROWS64
static constexpr int GEMMA4_FFN_DOWN_STAGE_ROWS64 = 10;
#endif
#ifndef GEMMA4_FFN_DOWN_STAGE_ROWS128
static constexpr int GEMMA4_FFN_DOWN_STAGE_ROWS128 = 6;
#endif
#ifndef GEMMA4_FFN_DOWN_STAGE_ROWS256
static constexpr int GEMMA4_FFN_DOWN_STAGE_ROWS256 = 3;
#endif
#ifndef GEMMA4_FFN_DOWN_STAGE_ROWS512
static constexpr int GEMMA4_FFN_DOWN_STAGE_ROWS512 = 3;
#endif
#ifndef GEMMA4_FFN_DOWN_STAGE_DEFAULT
static constexpr int GEMMA4_FFN_DOWN_STAGE_DEFAULT = 5;
#endif


template <typename ElementOutput_,
          int Count,
          typename ElementAccumulator_,
          typename ElementCompute_,
          cutlass::FloatRoundStyle Round = cutlass::FloatRoundStyle::round_to_nearest>

class GateGeluTanhUpMul {
 public:
  using FragmentOutput = cutlass::Array<ElementOutput_, Count>;
  using FragmentAccumulator = cutlass::Array<ElementAccumulator_, Count>;
  using FragmentCompute = cutlass::Array<ElementCompute_, Count>;

  struct Params {};

  CUTLASS_HOST_DEVICE explicit GateGeluTanhUpMul(Params const &) {}

  CUTLASS_HOST_DEVICE FragmentOutput operator()(FragmentAccumulator const &gate,
                            FragmentAccumulator const &up) const {
    cutlass::NumericArrayConverter<ElementCompute_, ElementAccumulator_, Count, Round> to_compute;
    cutlass::NumericArrayConverter<ElementOutput_, ElementCompute_, Count, Round> to_output;

    const FragmentCompute gate_compute = to_compute(gate);
    const FragmentCompute up_compute = to_compute(up);
    cutlass::epilogue::thread::GELU_taylor<FragmentCompute> gelu;
    cutlass::multiplies<FragmentCompute> multiply;
    return to_output(multiply(gelu(gate_compute), up_compute));
  }
};

// Launch wrapper for swizzling hidden packs into decode weight layout.
__global__ void swizzle_hidden_packs_kernel(
    __nv_bfloat16 *__restrict__ dst,
    const __nv_bfloat16 *__restrict__ src,
    int rows) {
  ffn_dev::swizzle_hidden_packs(
      dst, src, rows, int(blockIdx.y), int(blockIdx.x),
      int(gridDim.x), int(blockDim.x), int(threadIdx.x));
}

// RMS-normalizes a swizzled down-projection row, then adds the residual.
__global__ __launch_bounds__(kFfnThreads, 1) void
rmsnorm_residual_from_swizzled_down_kernel(
    __nv_bfloat16 *__restrict__ residual_out,
    __nv_bfloat16 *__restrict__ normed_out,
    const __nv_bfloat16 *__restrict__ down_swizzled,
    const __nv_bfloat16 *__restrict__ residual,
    const __nv_bfloat16 *__restrict__ rms_weight,
    float eps) {
  __shared__ float s_rms_warp_sums[kFfnWarps];
  __shared__ float s_scale;

  ffn_dev::rmsnorm_residual_from_swizzled_down<kFfnThreads>(
      residual_out, normed_out, down_swizzled, residual, rms_weight, eps,
      s_rms_warp_sums, s_scale, int(gridDim.x), int(blockIdx.x),
      int(threadIdx.x));
}

// Reorders the swizzled hidden packs produced by the prefill down GEMM back to
// the natural hidden layout expected by the standalone post-FFN RMSNorm.
__global__ __launch_bounds__(kFfnThreads, 1) void
unswizzle_hidden_packs_kernel(
    __nv_bfloat16 *__restrict__ dst,
    const __nv_bfloat16 *__restrict__ src) {
  ffn_dev::unswizzle_hidden_packs(
      dst, src, int(gridDim.x), int(blockIdx.x), int(blockDim.x),
      int(threadIdx.x));
}

// Launch wrapper for gate/up row interleaving plus hidden-pack swizzling.
__global__ void swizzle_gate_up_interleaved_kernel(
    __nv_bfloat16 *__restrict__ dst,
    const __nv_bfloat16 *__restrict__ src) {
  ffn_dev::swizzle_gate_up_interleaved(
      dst, src, int(gridDim.y), int(blockIdx.y), int(blockIdx.x),
      int(gridDim.x), int(blockDim.x), int(threadIdx.x));
}

template <int ThreadblockM,
          int ThreadblockN,
          int ThreadblockK,
          int WarpM,
          int WarpN,
          int Stages>
cudaError_t launch_cutlass_bf16_gemm(
    const __nv_bfloat16 *__restrict__ a,
    const __nv_bfloat16 *__restrict__ b,
    __nv_bfloat16 *__restrict__ c,
    int m,
    int k,
    int n,
    int ldb,
    cudaStream_t stream) {
  using Element = cutlass::bfloat16_t;
  using Gemm = cutlass::gemm::device::Gemm<
      Element,
      cutlass::layout::RowMajor,
      Element,
      cutlass::layout::RowMajor,
      Element,
      cutlass::layout::RowMajor,
      float,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<ThreadblockM, ThreadblockN, ThreadblockK>,
      cutlass::gemm::GemmShape<WarpM, WarpN, ThreadblockK>,
      cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<Element, 8, float, float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
      Stages>;

  const typename Gemm::Arguments args(
      {m, n, k},
      {reinterpret_cast<const Element *>(a), k},
      {reinterpret_cast<const Element *>(b), ldb},
      {reinterpret_cast<Element *>(c), n},
      {reinterpret_cast<Element *>(c), n},
      {1.0f, 0.0f});

  Gemm gemm;
  cutlass::Status status = gemm.can_implement(args);
  if (status != cutlass::Status::kSuccess) {
    return cudaErrorInvalidValue;
  }
  status = gemm(args, nullptr, stream);
  if (status != cutlass::Status::kSuccess) {
    return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
}

// Runs one measured CUTLASS DualGemm tile for gate/up plus GeGLU.
template <int ThreadblockM,
          int ThreadblockN,
          int ThreadblockK,
          int WarpM,
          int WarpN,
          int Stages>
cudaError_t run_gate_up_geglu_decode_layout_dual_gemm_config(
    __nv_bfloat16 *__restrict__ act,
    const __nv_bfloat16 *__restrict__ x_swizzled,
    const __nv_bfloat16 *__restrict__ w_gate_up_decode,
    int rows,
    cudaStream_t stream) {
  using Element = cutlass::bfloat16_t;
  using RowMajor = cutlass::layout::RowMajor;
  using ColumnMajor = cutlass::layout::ColumnMajor;
  using TensorOp = cutlass::arch::OpClassTensorOp;
  using Sm80 = cutlass::arch::Sm80;
  using ThreadblockSwizzle =
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>;
  using ThreadblockShape =
      cutlass::gemm::GemmShape<ThreadblockM, ThreadblockN, ThreadblockK>;
  using WarpShape = cutlass::gemm::GemmShape<WarpM, WarpN, ThreadblockK>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
  static constexpr bool kStoreGate = false;
  static constexpr bool kStoreUp = false;
  static constexpr bool kSplitKSerial = false;
  using OutputOp0 = cutlass::epilogue::thread::LinearCombination<
      Element, 8, float, float,
      cutlass::epilogue::thread::ScaleType::Nothing>;
  using OutputOp1 = OutputOp0;
  using OutputOp2 = GateGeluTanhUpMul<Element, 8, Element, float>;
  using DualGemm = cutlass::gemm::device::DualGemm<
      Element,
      RowMajor,
      Element,
      ColumnMajor,
      ColumnMajor,
      Element,
      RowMajor,
      float,
      TensorOp,
      Sm80,
      ThreadblockShape,
      WarpShape,
      InstructionShape,
      OutputOp0,
      OutputOp1,
      OutputOp2,
      ThreadblockSwizzle,
      Stages,
      kStoreGate,
      kStoreUp,
      kSplitKSerial>;

  const Element *w_gate = reinterpret_cast<const Element *>(w_gate_up_decode);
  const Element *w_up = w_gate + GEMMA4_HIDDEN_SIZE;
  const Element *act_const = reinterpret_cast<const Element *>(act);
  typename DualGemm::TensorRefD null_ref{};

  // Decode layout interleaves gate/up rows, so adjacent output columns are
  // separated by two hidden rows instead of one.
  const typename DualGemm::Arguments args(
      cutlass::gemm::DualGemmMode::kGemm,
      {rows, GEMMA4_INTERMEDIATE_SIZE, GEMMA4_HIDDEN_SIZE},
      {reinterpret_cast<const Element *>(x_swizzled), GEMMA4_HIDDEN_SIZE},
      {w_gate, 2 * GEMMA4_HIDDEN_SIZE},
      {act_const, GEMMA4_INTERMEDIATE_SIZE},
      null_ref,
      {w_up, 2 * GEMMA4_HIDDEN_SIZE},
      {act_const, GEMMA4_INTERMEDIATE_SIZE},
      null_ref,
      {reinterpret_cast<Element *>(act), GEMMA4_INTERMEDIATE_SIZE},
      typename OutputOp0::Params(),
      typename OutputOp1::Params(),
      typename OutputOp2::Params());

  DualGemm gemm;
  cutlass::Status status = gemm.can_implement(args);
  if (status != cutlass::Status::kSuccess) {
    return cudaErrorInvalidValue;
  }
  status = gemm(args, nullptr, stream);
  if (status != cutlass::Status::kSuccess) {
    return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
}

// Selects the measured gate/up DualGemm shape for the requested row count.
cudaError_t run_gate_up_geglu_decode_layout_dual_gemm(
    __nv_bfloat16 *__restrict__ act,
    const __nv_bfloat16 *__restrict__ x_swizzled,
    const __nv_bfloat16 *__restrict__ w_gate_up_decode,
    int rows,
    cudaStream_t stream) {
  // Row thresholds mirror the measured 12B FFN gate/up DualGemm sweep.
  switch (rows <= 64 ? 64 : rows <= 128 ? 128 : 0) {
    case 64:
      return run_gate_up_geglu_decode_layout_dual_gemm_config<
          64, 64, 32, 64, 32, GEMMA4_FFN_GATE_UP_STAGE_ROWS64>(
          act, x_swizzled, w_gate_up_decode, rows, stream);
    case 128:
      return run_gate_up_geglu_decode_layout_dual_gemm_config<
          128, 64, 32, 64, 32, GEMMA4_FFN_GATE_UP_STAGE_ROWS128>(
          act, x_swizzled, w_gate_up_decode, rows, stream);
    default:
      return run_gate_up_geglu_decode_layout_dual_gemm_config<
          256, 64, 32, 64, 32, GEMMA4_FFN_GATE_UP_STAGE_DEFAULT>(
          act, x_swizzled, w_gate_up_decode, rows, stream);
  }
}

// Launches the measured FFN-down GEMM for decode-swizzled weight layout.
cudaError_t launch_down_decode_layout_gemm(
    const __nv_bfloat16 *__restrict__ act,
    const __nv_bfloat16 *__restrict__ w_down_decode,
    __nv_bfloat16 *__restrict__ down,
    int rows,
    cudaStream_t stream) {
  // Row thresholds mirror the measured 12B FFN-down decode-layout sweep.
  // Template args: ThreadblockM, ThreadblockN, ThreadblockK, WarpM, WarpN, Stages.
  switch (rows <= 64 ? 64
                     : rows <= 128 ? 128
                                   : rows <= 256 ? 256
                                                 : rows <= 512 ? 512 : 0) {
    case 64:
      return launch_cutlass_bf16_gemm<
          64, 64, 32, 32, 32, GEMMA4_FFN_DOWN_STAGE_ROWS64>(
          act, w_down_decode, down, rows, GEMMA4_INTERMEDIATE_SIZE,
          GEMMA4_HIDDEN_SIZE, GEMMA4_HIDDEN_SIZE, stream);
    case 128:
      return launch_cutlass_bf16_gemm<
          64, 128, 32, 32, 64, GEMMA4_FFN_DOWN_STAGE_ROWS128>(
          act, w_down_decode, down, rows, GEMMA4_INTERMEDIATE_SIZE,
          GEMMA4_HIDDEN_SIZE, GEMMA4_HIDDEN_SIZE, stream);
    case 256:
      return launch_cutlass_bf16_gemm<
          128, 128, 64, 64, 64, GEMMA4_FFN_DOWN_STAGE_ROWS256>(
          act, w_down_decode, down, rows, GEMMA4_INTERMEDIATE_SIZE,
          GEMMA4_HIDDEN_SIZE, GEMMA4_HIDDEN_SIZE, stream);
    case 512:
      return launch_cutlass_bf16_gemm<
          256, 128, 32, 64, 64, GEMMA4_FFN_DOWN_STAGE_ROWS512>(
          act, w_down_decode, down, rows, GEMMA4_INTERMEDIATE_SIZE,
          GEMMA4_HIDDEN_SIZE, GEMMA4_HIDDEN_SIZE, stream);
    default:
      return launch_cutlass_bf16_gemm<
          128, 128, 32, 64, 64, GEMMA4_FFN_DOWN_STAGE_DEFAULT>(
          act, w_down_decode, down, rows, GEMMA4_INTERMEDIATE_SIZE,
          GEMMA4_HIDDEN_SIZE, GEMMA4_HIDDEN_SIZE, stream);
  }
}

// Runs one-token decode FFN with swizzled CUTLASS GEMMs and post-FFN norm.
cudaError_t gemma4_ffn_decode_fused_bf16_impl(
    __nv_bfloat16 *__restrict__ residual_out,
    __nv_bfloat16 *__restrict__ normed_out,
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ residual,
    const __nv_bfloat16 *__restrict__ rms_weight,
    const __nv_bfloat16 *__restrict__ w_gate_up_decode,
    const __nv_bfloat16 *__restrict__ w_down_decode,
    Gemma4FfnDecodeScratch *__restrict__ scratch,
    float eps,
    cudaStream_t stream) {
  const dim3 block_dim(kSwizzleThreads);
  const dim3 swizzle_grid_dim(kActualSwizzleBlocksPerRow, 1);
  swizzle_hidden_packs_kernel<<<swizzle_grid_dim, block_dim, 0, stream>>>(
      scratch->down, x, 1);
  cudaError_t status = cudaGetLastError();
  GEMMA4_RETURN_IF_CUDA_ERROR(status);

  status = run_gate_up_geglu_decode_layout_dual_gemm(
      scratch->act, scratch->down, w_gate_up_decode, 1, stream);
  GEMMA4_RETURN_IF_CUDA_ERROR(status);

  status = launch_down_decode_layout_gemm(
      scratch->act, w_down_decode, scratch->down, 1, stream);
  GEMMA4_RETURN_IF_CUDA_ERROR(status);

  rmsnorm_residual_from_swizzled_down_kernel<<<1, kFfnThreads, 0, stream>>>(
      residual_out, normed_out, scratch->down, residual, rms_weight, eps);
  return cudaGetLastError();
}

// Runs multi-row FFN through the decode-swizzled weight layout.
cudaError_t gemma4_ffn_prefill_bf16_impl(const Gemma4FfnBf16Args &args) {
  if (args.prefill_scratch.capacity_rows < args.rows) {
    return cudaErrorInvalidValue;
  }

  const dim3 block_dim(kSwizzleThreads);
  const dim3 swizzle_grid_dim(kActualSwizzleBlocksPerRow, args.rows);
  swizzle_hidden_packs_kernel<<<swizzle_grid_dim, block_dim, 0, args.stream>>>(
      args.prefill_scratch.down, args.x, args.rows);
  cudaError_t status = cudaGetLastError();
  GEMMA4_RETURN_IF_CUDA_ERROR(status);

  status = run_gate_up_geglu_decode_layout_dual_gemm(
      args.prefill_scratch.act, args.prefill_scratch.down,
      args.w_gate_up_decode, args.rows, args.stream);
  GEMMA4_RETURN_IF_CUDA_ERROR(status);

  status = launch_down_decode_layout_gemm(
      args.prefill_scratch.act, args.w_down_decode,
      args.prefill_scratch.down, args.rows, args.stream);
  GEMMA4_RETURN_IF_CUDA_ERROR(status);

  rmsnorm_residual_from_swizzled_down_kernel<<<
      args.rows, kFfnThreads, 0, args.stream>>>(
      args.residual_out, args.normed_out, args.prefill_scratch.down,
      args.residual, args.rms_weight, args.eps);
  return cudaGetLastError();
}

}  // namespace

cudaError_t gemma4_ffn_decode_swizzle_weights_bf16(
    __nv_bfloat16 *__restrict__ w_gate_up_swizzled,
    const __nv_bfloat16 *__restrict__ w_gate_up_col_major,
    __nv_bfloat16 *__restrict__ w_down_swizzled,
    const __nv_bfloat16 *__restrict__ w_down_row_major,
    cudaStream_t stream) {
  const dim3 block_dim(kSwizzleThreads);
  const dim3 gate_up_grid_dim(kActualSwizzleBlocksPerRow,
                              GEMMA4_PACKED_FFN_SIZE);
  swizzle_gate_up_interleaved_kernel<<<
      gate_up_grid_dim, block_dim, 0, stream>>>(
      w_gate_up_swizzled, w_gate_up_col_major);
  GEMMA4_RETURN_IF_CUDA_ERROR(cudaGetLastError());

  const dim3 down_grid_dim(kActualSwizzleBlocksPerRow,
                           GEMMA4_INTERMEDIATE_SIZE);
  swizzle_hidden_packs_kernel<<<down_grid_dim, block_dim, 0, stream>>>(
      w_down_swizzled, w_down_row_major, GEMMA4_INTERMEDIATE_SIZE);
  return cudaGetLastError();
}

size_t gemma4_ffn_prefill_scratch_elements(int rows) {
  if (rows <= 0) {
    return 0;
  }

  const auto act_layout =
      cute::make_layout(cute::make_shape(rows, GEMMA4_INTERMEDIATE_SIZE));
  const auto down_layout =
      cute::make_layout(cute::make_shape(rows, GEMMA4_HIDDEN_SIZE));
  return cute::size(act_layout) + cute::size(down_layout);
}

Gemma4FfnPrefillScratch gemma4_ffn_prefill_scratch_from_buffer(
    __nv_bfloat16 *buffer,
    int rows) {
  Gemma4FfnPrefillScratch scratch = {};
  scratch.capacity_rows = rows;
  if (buffer == nullptr || rows <= 0) {
    return scratch;
  }

  const auto act_layout =
      cute::make_layout(cute::make_shape(rows, GEMMA4_INTERMEDIATE_SIZE));
  const size_t act_elements = cute::size(act_layout);
  scratch.act = buffer;
  scratch.down = scratch.act + act_elements;
  return scratch;
}

cudaError_t gemma4_ffn_bf16(const Gemma4FfnBf16Args &args) {
  if (args.rows <= 0) {
    return args.rows == 0 ? cudaSuccess : cudaErrorInvalidValue;
  }
  if (args.rows == 1) {
    return gemma4_ffn_decode_fused_bf16_impl(
        args.residual_out, args.normed_out, args.x, args.residual,
        args.rms_weight, args.w_gate_up_decode, args.w_down_decode,
        args.decode_scratch, args.eps, args.stream);
  }
  return gemma4_ffn_prefill_bf16_impl(args);
}

// Runs the prefill GeGLU MLP and leaves the down-projection in natural order.
cudaError_t gemma4_ffn_prefill_mlp_bf16(
    __nv_bfloat16 *__restrict__ out,
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ w_gate_up_decode,
    const __nv_bfloat16 *__restrict__ w_down_decode,
    Gemma4FfnPrefillScratch scratch,
    int rows,
    cudaStream_t stream) {
  if (rows < 0 || scratch.capacity_rows < rows) {
    return cudaErrorInvalidValue;
  }
  if (rows == 0) {
    return cudaSuccess;
  }

  const dim3 block_dim(kSwizzleThreads);
  const dim3 swizzle_grid_dim(kActualSwizzleBlocksPerRow, rows);
  swizzle_hidden_packs_kernel<<<swizzle_grid_dim, block_dim, 0, stream>>>(
      scratch.down, x, rows);
  cudaError_t status = cudaGetLastError();
  GEMMA4_RETURN_IF_CUDA_ERROR(status);

  status = run_gate_up_geglu_decode_layout_dual_gemm(
      scratch.act, scratch.down, w_gate_up_decode, rows, stream);
  GEMMA4_RETURN_IF_CUDA_ERROR(status);

  status = launch_down_decode_layout_gemm(
      scratch.act, w_down_decode, scratch.down, rows, stream);
  GEMMA4_RETURN_IF_CUDA_ERROR(status);

  unswizzle_hidden_packs_kernel<<<rows, kFfnThreads, 0, stream>>>(
      out, scratch.down);
  return cudaGetLastError();
}

// Runs one-token decode FFN using pre-swizzled decode weights.
cudaError_t gemma4_ffn_decode_fused_bf16(
    __nv_bfloat16 *__restrict__ residual_out,
    __nv_bfloat16 *__restrict__ normed_out,
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ residual,
    const __nv_bfloat16 *__restrict__ rms_weight,
    const __nv_bfloat16 *__restrict__ w_gate_up_decode,
    const __nv_bfloat16 *__restrict__ w_down_decode,
    Gemma4FfnDecodeScratch *__restrict__ scratch,
    float eps,
    cudaStream_t stream) {
  return gemma4_ffn_decode_fused_bf16_impl(
      residual_out, normed_out, x, residual, rms_weight,
      w_gate_up_decode, w_down_decode, scratch, eps, stream);
}
