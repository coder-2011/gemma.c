#include "gemma4_ffn.cuh"

#ifndef GEMMA4_WEIGHT_LOAD_POLICY
#define GEMMA4_WEIGHT_LOAD_POLICY 0
#endif

#include "gemma4_cuda_utils.cuh"
#include "gemma4_ffn_decode_device.cuh"
#include "gemma4_rmsnorm.cuh"

#include <cutlass/array.h>
#include <cutlass/epilogue/thread/activation.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/epilogue/thread/scale_type.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

#include "device/dual_gemm.h"

#include <math.h>

namespace {

namespace ffn_dev = gemma4_ffn_decode_device;

constexpr int kFfnThreads = GEMMA4_FFN_DECODE_THREADS;
constexpr int kFfnWarps = kFfnThreads / WARP_SIZE;
constexpr int kIntermediateTile = ffn_dev::kIntermediateTile;
constexpr int kIntermediateTiles = ffn_dev::kIntermediateTiles;
constexpr int kHiddenPacks = ffn_dev::kHiddenPacks;
constexpr int kActTile = ffn_dev::kActTile;
constexpr int kReductionPolicy = ffn_dev::kReductionPolicy;
constexpr int kPartialGroups = ffn_dev::kPartialGroups;
constexpr int kSwizzleThreads = 96;
constexpr int kActualSwizzleBlocksPerRow =
    div_up(kHiddenPacks, kSwizzleThreads);
constexpr int kAccumBlocks = ffn_dev::kAccumBlocks;

static_assert((kFfnThreads % WARP_SIZE) == 0, "FFN decode thread count must be a whole number of warps");
static_assert(kSwizzleThreads > 0 && kSwizzleThreads <= 1024 && (kSwizzleThreads % WARP_SIZE) == 0, "FFN swizzle threads must be a valid warp-multiple block size");

using FfnBf16Pack = ffn_dev::FfnBf16Pack;
using ffn_dev::atomic_add_accum_pack;
using ffn_dev::hidden_pack_swizzle_index;
using ffn_dev::store_partial_pack;

template <typename ElementOutput_,
          int Count,
          typename ElementAccumulator_,
          typename ElementCompute_,
          cutlass::FloatRoundStyle Round =
          cutlass::FloatRoundStyle::round_to_nearest>
class GateGeluTanhUpMul {
 public:
  using ElementOutput = ElementOutput_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementCompute = ElementCompute_;
  using FragmentOutput = cutlass::Array<ElementOutput, Count>;
  using FragmentAccumulator = cutlass::Array<ElementAccumulator, Count>;
  using FragmentCompute = cutlass::Array<ElementCompute, Count>;

  struct Params {};

  CUTLASS_HOST_DEVICE
  explicit GateGeluTanhUpMul(Params const &) {}

  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(FragmentAccumulator const &gate,
                            FragmentAccumulator const &up) const {
    cutlass::NumericArrayConverter<ElementCompute, ElementAccumulator, Count,
                                   Round>
        to_compute;
    cutlass::NumericArrayConverter<ElementOutput, ElementCompute, Count,
                                   Round>
        to_output;
    FragmentCompute gate_compute = to_compute(gate);
    FragmentCompute up_compute = to_compute(up);
    cutlass::epilogue::thread::GELU_taylor<FragmentCompute> gelu;
    cutlass::multiplies<FragmentCompute> multiply;
    return to_output(multiply(gelu(gate_compute), up_compute));
  }
};

__global__ __launch_bounds__(kFfnThreads, 1) void
gemma4_ffn_decode_accumulate_bf16_kernel(
    const floatX *__restrict__ x,
    const floatX *__restrict__ w_gate_up_col_major,
    const floatX *__restrict__ w_down_row_major,
    Gemma4FfnDecodeScratch *__restrict__ scratch) {
  __shared__ float s_matmul_warp_sums[2][kActTile][kFfnWarps];
  __shared__ float s_act[kActTile];

  float partial[kBf16Packed128Elements] = {};
  const int hidden_pack = threadIdx.x;
  const int swizzled_hidden_col =
      hidden_pack_swizzle_index(hidden_pack) * kBf16Packed128Elements;

  if constexpr (kAccumBlocks == kIntermediateTiles) {
    const int intermediate_begin =
        static_cast<int>(blockIdx.x) * kIntermediateTile;
    ffn_dev::accumulate_intermediate_tile<kFfnThreads, false>(
        x, w_gate_up_col_major, w_down_row_major, intermediate_begin,
        swizzled_hidden_col, true, partial, s_matmul_warp_sums, s_act);
  } else if constexpr (kAccumBlocks * 2 > kIntermediateTiles) {
    const int tile0 = static_cast<int>(blockIdx.x);
    ffn_dev::accumulate_intermediate_tile<kFfnThreads, false>(
        x, w_gate_up_col_major, w_down_row_major, tile0 * kIntermediateTile,
        swizzled_hidden_col, true, partial, s_matmul_warp_sums, s_act);

    const int tile1 = tile0 + kAccumBlocks;
    if (tile1 < kIntermediateTiles) {
      ffn_dev::accumulate_intermediate_tile<kFfnThreads, false>(
          x, w_gate_up_col_major, w_down_row_major, tile1 * kIntermediateTile,
          swizzled_hidden_col, true, partial, s_matmul_warp_sums, s_act);
    }
  } else {
    for (int tile = static_cast<int>(blockIdx.x); tile < kIntermediateTiles;
         tile += kAccumBlocks) {
      const int intermediate_begin = tile * kIntermediateTile;
      ffn_dev::accumulate_intermediate_tile<kFfnThreads, false>(
          x, w_gate_up_col_major, w_down_row_major, intermediate_begin,
          swizzled_hidden_col, true, partial, s_matmul_warp_sums, s_act);
    }
  }

  atomic_add_accum_pack(scratch, hidden_pack, partial);
}

__global__ __launch_bounds__(kFfnThreads, 1) void
gemma4_ffn_decode_accumulate_partials_bf16_kernel(
    const floatX *__restrict__ x,
    const floatX *__restrict__ w_gate_up_col_major,
    const floatX *__restrict__ w_down_row_major,
    Gemma4FfnDecodeScratch *__restrict__ scratch) {
  __shared__ float s_matmul_warp_sums[2][kActTile][kFfnWarps];
  __shared__ float s_act[kActTile];

  float partial[kBf16Packed128Elements] = {};
  const int hidden_pack = threadIdx.x;
  const int swizzled_hidden_col =
      hidden_pack_swizzle_index(hidden_pack) * kBf16Packed128Elements;
  const int group = static_cast<int>(blockIdx.x);

  for (int tile = group; tile < kIntermediateTiles; tile += kPartialGroups) {
    ffn_dev::accumulate_intermediate_tile<kFfnThreads, false>(
        x, w_gate_up_col_major, w_down_row_major, tile * kIntermediateTile,
        swizzled_hidden_col, true, partial, s_matmul_warp_sums, s_act);
  }

  store_partial_pack(scratch, group, hidden_pack, partial);
}

__global__ __launch_bounds__(kFfnThreads, 1) void
gemma4_ffn_decode_finalize_bf16_kernel(
    floatX *__restrict__ residual_out,
    floatX *__restrict__ normed_out,
    const floatX *__restrict__ residual,
    const floatX *__restrict__ rms_weight,
    const Gemma4FfnDecodeScratch *__restrict__ scratch,
    float eps) {
  __shared__ float s_rms_warp_sums[kFfnWarps];
  __shared__ float s_scale;

  ffn_dev::finalize_rmsnorm_residual<
      kFfnThreads, false, kReductionPolicy == 1>(
      residual_out, normed_out, residual, rms_weight, scratch, eps,
      s_rms_warp_sums, s_scale, int(threadIdx.x));
}

bool ffn_decode_args_valid(const floatX *residual_out,
                           const floatX *normed_out,
                           const floatX *x,
                           const floatX *residual,
                           const floatX *rms_weight,
                           const floatX *w_gate_up_col_major,
                           const floatX *w_down_row_major,
                           const Gemma4FfnDecodeScratch *scratch) {
  return residual_out != nullptr && normed_out != nullptr && x != nullptr &&
         residual != nullptr && rms_weight != nullptr &&
         w_gate_up_col_major != nullptr && w_down_row_major != nullptr &&
         scratch != nullptr && is_aligned_16(residual_out) &&
         is_aligned_16(normed_out) && is_aligned_16(x) &&
         is_aligned_16(residual) && is_aligned_16(rms_weight) &&
         is_aligned_16(w_gate_up_col_major) &&
         is_aligned_16(w_down_row_major) && is_aligned_128(scratch);
}

__global__ void swizzle_hidden_packs_kernel(
    floatX *__restrict__ dst,
    const floatX *__restrict__ src,
    int rows) {
  const int row = static_cast<int>(blockIdx.y);
  if (row >= rows) {
    return;
  }

  const int64_t row_offset = static_cast<int64_t>(row) * GEMMA4_HIDDEN_SIZE;
  const int pack_stride = static_cast<int>(gridDim.x) * blockDim.x;
  for (int pack = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
       pack < kHiddenPacks; pack += pack_stride) {
    const int src_col = pack * kBf16Packed128Elements;
    const int dst_col =
        hidden_pack_swizzle_index(pack) * kBf16Packed128Elements;
    const FfnBf16Pack pack_value = load128g(src + row_offset + src_col);
    store128(dst + row_offset + dst_col, pack_value);
  }
}

// RMS-normalizes a swizzled down-projection row, then adds the residual.
__global__ __launch_bounds__(kFfnThreads, 1) void
rmsnorm_residual_from_swizzled_down_kernel(
    floatX *__restrict__ residual_out,
    floatX *__restrict__ normed_out,
    const floatX *__restrict__ down_swizzled,
    const floatX *__restrict__ residual,
    const floatX *__restrict__ rms_weight,
    float eps) {
  __shared__ float s_rms_warp_sums[kFfnWarps];
  __shared__ float s_scale;

  const int row = static_cast<int>(blockIdx.x);
  const int hidden_pack = static_cast<int>(threadIdx.x);
  const int natural_col = hidden_pack * kBf16Packed128Elements;
  const int swizzled_col =
      hidden_pack_swizzle_index(hidden_pack) * kBf16Packed128Elements;
  const int64_t row_offset = static_cast<int64_t>(row) * GEMMA4_HIDDEN_SIZE;

  const FfnBf16Pack down_pack =
      load128g(down_swizzled + row_offset + swizzled_col);
  float sum_sq = 0.0f;
  gemma4_bf16_pack_accumulate_square(down_pack, sum_sq);

  const float total =
      gemma4_block_reduce_sum<kFfnThreads>(
          sum_sq, s_rms_warp_sums, threadIdx.x);
  if (threadIdx.x == 0) {
    s_scale = rsqrtf(total / static_cast<float>(GEMMA4_HIDDEN_SIZE) + eps);
  }
  __syncthreads();

  const FfnBf16Pack gamma_pack =
      load128g(rms_weight + natural_col);
  const FfnBf16Pack normed_pack =
      gemma4_bf16_pack_apply_rmsnorm(down_pack, gamma_pack, s_scale);
  const FfnBf16Pack residual_pack =
      load128g(residual + row_offset + natural_col);
  const FfnBf16Pack residual_out_pack =
      gemma4_bf16_pack_add(residual_pack, normed_pack);
  store128wb(normed_out + row_offset + natural_col, normed_pack);
  store128(residual_out + row_offset + natural_col, residual_out_pack);
}

// Reorders the swizzled hidden packs produced by the prefill down GEMM back to
// the natural hidden layout expected by the standalone post-FFN RMSNorm.
__global__ __launch_bounds__(kFfnThreads, 1) void
unswizzle_hidden_packs_kernel(
    floatX *__restrict__ dst,
    const floatX *__restrict__ src) {
  const int row = static_cast<int>(blockIdx.x);
  const int64_t row_offset = static_cast<int64_t>(row) * GEMMA4_HIDDEN_SIZE;

  for (int hidden_pack = static_cast<int>(threadIdx.x);
       hidden_pack < kHiddenPacks; hidden_pack += blockDim.x) {
    const int natural_col = hidden_pack * kBf16Packed128Elements;
    const int swizzled_col =
        hidden_pack_swizzle_index(hidden_pack) * kBf16Packed128Elements;
    const FfnBf16Pack pack = load128g(src + row_offset + swizzled_col);
    store128(dst + row_offset + natural_col, pack);
  }
}

__global__ void swizzle_gate_up_interleaved_kernel(
    floatX *__restrict__ dst,
    const floatX *__restrict__ src) {
  const int dst_row = static_cast<int>(blockIdx.y);
  if (dst_row >= GEMMA4_PACKED_FFN_SIZE) {
    return;
  }

  const int src_row =
      (dst_row & 1) == 0 ? dst_row / 2
                         : GEMMA4_INTERMEDIATE_SIZE + dst_row / 2;
  const int64_t src_row_offset =
      static_cast<int64_t>(src_row) * GEMMA4_HIDDEN_SIZE;
  const int64_t dst_row_offset =
      static_cast<int64_t>(dst_row) * GEMMA4_HIDDEN_SIZE;
  const int pack_stride = static_cast<int>(gridDim.x) * blockDim.x;
  for (int pack = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
       pack < kHiddenPacks; pack += pack_stride) {
    const int src_col = pack * kBf16Packed128Elements;
    const int dst_col =
        hidden_pack_swizzle_index(pack) * kBf16Packed128Elements;
    const FfnBf16Pack pack_value =
        load128g(src + src_row_offset + src_col);
    store128(dst + dst_row_offset + dst_col, pack_value);
  }
}

template <typename LayoutB,
          int ThreadblockM,
          int ThreadblockN,
          int ThreadblockK,
          int WarpM,
          int WarpN,
          int Stages>
cudaError_t launch_cutlass_bf16_gemm(
    const floatX *__restrict__ a,
    const floatX *__restrict__ b,
    floatX *__restrict__ c,
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
      LayoutB,
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

  typename Gemm::Arguments args(
      {m, n, k},
      {reinterpret_cast<const Element *>(a), k},
      {reinterpret_cast<const Element *>(b), ldb},
      {reinterpret_cast<Element *>(c), n},
      {reinterpret_cast<Element *>(c), n},
      {1.0f, 0.0f});

  Gemm gemm;
  const cutlass::Status status = gemm(args, nullptr, stream);
  if (status != cutlass::Status::kSuccess) {
    return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
}

// Runs one measured CUTLASS DualGemm tile for prefill gate/up plus GeGLU.
template <int ThreadblockM,
          int ThreadblockN,
          int ThreadblockK,
          int WarpM,
          int WarpN,
          int Stages>
cudaError_t run_prefill_gate_up_geglu_decode_layout_dual_gemm_config(
    floatX *__restrict__ act,
    const floatX *__restrict__ x_swizzled,
    const floatX *__restrict__ w_gate_up_decode,
    int rows,
    cudaStream_t stream) {
  using Element = cutlass::bfloat16_t;
  using ThreadblockShape =
      cutlass::gemm::GemmShape<ThreadblockM, ThreadblockN, ThreadblockK>;
  using WarpShape = cutlass::gemm::GemmShape<WarpM, WarpN, ThreadblockK>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
  using OutputOp0 = cutlass::epilogue::thread::LinearCombination<
      Element, 8, float, float,
      cutlass::epilogue::thread::ScaleType::Nothing>;
  using OutputOp1 = OutputOp0;
  using OutputOp2 = GateGeluTanhUpMul<Element, 8, Element, float>;
  using DualGemm = cutlass::gemm::device::DualGemm<
      Element,
      cutlass::layout::RowMajor,
      Element,
      cutlass::layout::ColumnMajor,
      cutlass::layout::ColumnMajor,
      Element,
      cutlass::layout::RowMajor,
      float,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80,
      ThreadblockShape,
      WarpShape,
      InstructionShape,
      OutputOp0,
      OutputOp1,
      OutputOp2,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
      Stages,
      false,
      false,
      false>;

  const Element *w_gate =
      reinterpret_cast<const Element *>(w_gate_up_decode);
  const Element *w_up = w_gate + GEMMA4_HIDDEN_SIZE;
  const Element *act_const = reinterpret_cast<const Element *>(act);
  typename DualGemm::TensorRefD null_ref{};

  // Decode layout interleaves gate/up rows, so adjacent output columns are
  // separated by two hidden rows instead of one.
  typename DualGemm::Arguments args(
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

cudaError_t run_prefill_gate_up_geglu_decode_layout_dual_gemm(
    floatX *__restrict__ act,
    const floatX *__restrict__ x_swizzled,
    const floatX *__restrict__ w_gate_up_decode,
    int rows,
    cudaStream_t stream) {
  // Row thresholds mirror the measured 12B FFN gate/up DualGemm sweep.
  if (rows <= 64) {
    return run_prefill_gate_up_geglu_decode_layout_dual_gemm_config<
        64, 64, 32, 64, 32, 3>(
        act, x_swizzled, w_gate_up_decode, rows, stream);
  }
  if (rows <= 128) {
    return run_prefill_gate_up_geglu_decode_layout_dual_gemm_config<
        128, 64, 32, 64, 32, 5>(
        act, x_swizzled, w_gate_up_decode, rows, stream);
  }
  return run_prefill_gate_up_geglu_decode_layout_dual_gemm_config<
      256, 64, 32, 64, 32, 3>(
      act, x_swizzled, w_gate_up_decode, rows, stream);
}

cudaError_t launch_prefill_down_gemm(
    const floatX *__restrict__ act,
    const floatX *__restrict__ w_down_row_major,
    floatX *__restrict__ down,
    int rows,
    cudaStream_t stream) {
  // Row thresholds mirror the measured 12B FFN-down prefill sweep.
  if (rows <= 64) {
    return launch_cutlass_bf16_gemm<cutlass::layout::RowMajor, 64, 64, 32,
                                    32, 32, 10>(
        act, w_down_row_major, down, rows, GEMMA4_INTERMEDIATE_SIZE,
        GEMMA4_HIDDEN_SIZE, GEMMA4_HIDDEN_SIZE, stream);
  }
  if (rows <= 128) {
    return launch_cutlass_bf16_gemm<cutlass::layout::RowMajor, 64, 128, 32,
                                    32, 64, 6>(
        act, w_down_row_major, down, rows, GEMMA4_INTERMEDIATE_SIZE,
        GEMMA4_HIDDEN_SIZE, GEMMA4_HIDDEN_SIZE, stream);
  }
  if (rows <= 256) {
    return launch_cutlass_bf16_gemm<cutlass::layout::RowMajor, 128, 128, 64,
                                    64, 64, 3>(
        act, w_down_row_major, down, rows, GEMMA4_INTERMEDIATE_SIZE,
        GEMMA4_HIDDEN_SIZE, GEMMA4_HIDDEN_SIZE, stream);
  }
  if (rows <= 512) {
    return launch_cutlass_bf16_gemm<cutlass::layout::RowMajor, 256, 128, 32,
                                    64, 64, 3>(
        act, w_down_row_major, down, rows, GEMMA4_INTERMEDIATE_SIZE,
        GEMMA4_HIDDEN_SIZE, GEMMA4_HIDDEN_SIZE, stream);
  }
  return launch_cutlass_bf16_gemm<cutlass::layout::RowMajor, 128, 128, 32,
                                  64, 64, 5>(
      act, w_down_row_major, down, rows, GEMMA4_INTERMEDIATE_SIZE,
      GEMMA4_HIDDEN_SIZE, GEMMA4_HIDDEN_SIZE, stream);
}

bool ffn_common_args_valid(const Gemma4FfnBf16Args &args) {
  return args.rows >= 0 && args.residual_out != nullptr &&
         args.normed_out != nullptr && args.x != nullptr &&
         args.residual != nullptr && args.rms_weight != nullptr &&
         is_aligned_16(args.residual_out) && is_aligned_16(args.normed_out) &&
         is_aligned_16(args.x) && is_aligned_16(args.residual) &&
         is_aligned_16(args.rms_weight);
}

bool ffn_prefill_decode_layout_args_valid(const Gemma4FfnBf16Args &args) {
  return ffn_common_args_valid(args) && args.rows > 1 &&
         args.w_gate_up_decode != nullptr &&
         args.w_down_decode != nullptr &&
         args.prefill_scratch.act != nullptr &&
         args.prefill_scratch.down != nullptr &&
         args.prefill_scratch.capacity_rows >= args.rows &&
         is_aligned_16(args.w_gate_up_decode) &&
         is_aligned_16(args.w_down_decode) &&
         is_aligned_16(args.prefill_scratch.act) &&
         is_aligned_16(args.prefill_scratch.down);
}

// Checks the standalone prefill MLP path without requiring residual/norm args.
bool ffn_prefill_mlp_args_valid(
    const floatX *__restrict__ out,
    const floatX *__restrict__ x,
    const floatX *__restrict__ w_gate_up_decode,
    const floatX *__restrict__ w_down_decode,
    Gemma4FfnPrefillScratch scratch,
    int rows) {
  return rows >= 0 && out != nullptr && x != nullptr &&
         w_gate_up_decode != nullptr && w_down_decode != nullptr &&
         scratch.act != nullptr && scratch.down != nullptr &&
         scratch.capacity_rows >= rows && is_aligned_16(out) &&
         is_aligned_16(x) && is_aligned_16(w_gate_up_decode) &&
         is_aligned_16(w_down_decode) && is_aligned_16(scratch.act) &&
         is_aligned_16(scratch.down);
}

cudaError_t gemma4_ffn_decode_fused_bf16_impl(
    floatX *__restrict__ residual_out,
    floatX *__restrict__ normed_out,
    const floatX *__restrict__ x,
    const floatX *__restrict__ residual,
    const floatX *__restrict__ rms_weight,
    const floatX *__restrict__ w_gate_up_col_major,
    const floatX *__restrict__ w_down_row_major,
    Gemma4FfnDecodeScratch *__restrict__ scratch,
    float eps,
    cudaStream_t stream) {
  if (!ffn_decode_args_valid(residual_out, normed_out, x, residual,
                             rms_weight, w_gate_up_col_major,
                             w_down_row_major, scratch)) {
    return cudaErrorInvalidValue;
  }

  cudaError_t status = cudaMemsetAsync(scratch, 0, sizeof(*scratch), stream);
  GEMMA4_RETURN_IF_CUDA_ERROR(status);

  if constexpr (kReductionPolicy == 1) {
    gemma4_ffn_decode_accumulate_partials_bf16_kernel<<<
        kPartialGroups, kFfnThreads, 0, stream>>>(
        x, w_gate_up_col_major, w_down_row_major, scratch);
  } else {
    gemma4_ffn_decode_accumulate_bf16_kernel<<<
        kAccumBlocks, kFfnThreads, 0, stream>>>(
        x, w_gate_up_col_major, w_down_row_major, scratch);
  }
  GEMMA4_RETURN_IF_CUDA_ERROR(cudaGetLastError());

  gemma4_ffn_decode_finalize_bf16_kernel<<<1, kFfnThreads, 0, stream>>>(
      residual_out, normed_out, residual, rms_weight, scratch, eps);
  return cudaGetLastError();
}

cudaError_t gemma4_ffn_prefill_bf16_impl(const Gemma4FfnBf16Args &args) {
  if (!ffn_prefill_decode_layout_args_valid(args)) {
    return cudaErrorInvalidValue;
  }

  const dim3 block_dim(kSwizzleThreads);
  const dim3 swizzle_grid_dim(kActualSwizzleBlocksPerRow, args.rows);
  swizzle_hidden_packs_kernel<<<swizzle_grid_dim, block_dim, 0, args.stream>>>(
      args.prefill_scratch.down, args.x, args.rows);
  cudaError_t status = cudaGetLastError();
  GEMMA4_RETURN_IF_CUDA_ERROR(status);

  status = run_prefill_gate_up_geglu_decode_layout_dual_gemm(
      args.prefill_scratch.act, args.prefill_scratch.down,
      args.w_gate_up_decode, args.rows, args.stream);
  GEMMA4_RETURN_IF_CUDA_ERROR(status);

  status = launch_prefill_down_gemm(
      args.prefill_scratch.act, args.w_down_decode,
      args.prefill_scratch.down, args.rows, args.stream);
  GEMMA4_RETURN_IF_CUDA_ERROR(status);

  rmsnorm_residual_from_swizzled_down_kernel<<<
      args.rows, kFfnThreads, 0, args.stream>>>(
      args.residual_out, args.normed_out, args.prefill_scratch.down,
      args.residual, args.rms_weight, args.eps);
  return cudaGetLastError();
}

// Runs the prefill GeGLU MLP and leaves the down-projection in natural order.
cudaError_t gemma4_ffn_prefill_mlp_bf16_impl(
    floatX *__restrict__ out,
    const floatX *__restrict__ x,
    const floatX *__restrict__ w_gate_up_decode,
    const floatX *__restrict__ w_down_decode,
    Gemma4FfnPrefillScratch scratch,
    int rows,
    cudaStream_t stream) {
  if (rows == 0) {
    return cudaSuccess;
  }
  if (!ffn_prefill_mlp_args_valid(
          out, x, w_gate_up_decode, w_down_decode, scratch, rows)) {
    return cudaErrorInvalidValue;
  }

  const dim3 block_dim(kSwizzleThreads);
  const dim3 swizzle_grid_dim(kActualSwizzleBlocksPerRow, rows);
  swizzle_hidden_packs_kernel<<<swizzle_grid_dim, block_dim, 0, stream>>>(
      scratch.down, x, rows);
  cudaError_t status = cudaGetLastError();
  GEMMA4_RETURN_IF_CUDA_ERROR(status);

  status = run_prefill_gate_up_geglu_decode_layout_dual_gemm(
      scratch.act, scratch.down, w_gate_up_decode, rows, stream);
  GEMMA4_RETURN_IF_CUDA_ERROR(status);

  status = launch_prefill_down_gemm(
      scratch.act, w_down_decode, scratch.down, rows, stream);
  GEMMA4_RETURN_IF_CUDA_ERROR(status);

  unswizzle_hidden_packs_kernel<<<rows, kFfnThreads, 0, stream>>>(
      out, scratch.down);
  return cudaGetLastError();
}

}  // namespace

cudaError_t gemma4_ffn_decode_swizzle_weights_bf16(
    floatX *__restrict__ w_gate_up_swizzled,
    const floatX *__restrict__ w_gate_up_col_major,
    floatX *__restrict__ w_down_swizzled,
    const floatX *__restrict__ w_down_row_major,
    cudaStream_t stream) {
  if (w_gate_up_swizzled == nullptr || w_gate_up_col_major == nullptr ||
      w_down_swizzled == nullptr || w_down_row_major == nullptr ||
      !is_aligned_16(w_gate_up_swizzled) ||
      !is_aligned_16(w_gate_up_col_major) ||
      !is_aligned_16(w_down_swizzled) ||
      !is_aligned_16(w_down_row_major)) {
    return cudaErrorInvalidValue;
  }

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

cudaError_t gemma4_ffn_decode_configure_scratch_l2(
    Gemma4FfnDecodeScratch *scratch,
    cudaStream_t stream) {
  (void)stream;
  if (scratch == nullptr || !is_aligned_128(scratch)) {
    return cudaErrorInvalidValue;
  }
  return cudaSuccess;
}

size_t gemma4_ffn_prefill_scratch_elements(int rows) {
  constexpr size_t per_row =
      static_cast<size_t>(GEMMA4_INTERMEDIATE_SIZE) +
      static_cast<size_t>(GEMMA4_HIDDEN_SIZE);
  return rows <= 0 ? 0 : static_cast<size_t>(rows) * per_row;
}

Gemma4FfnPrefillScratch gemma4_ffn_prefill_scratch_from_buffer(
    floatX *buffer,
    int rows) {
  Gemma4FfnPrefillScratch scratch = {};
  scratch.capacity_rows = rows;
  if (buffer == nullptr || rows <= 0) {
    return scratch;
  }

  const size_t act_elements =
      static_cast<size_t>(rows) * GEMMA4_INTERMEDIATE_SIZE;
  scratch.act = buffer;
  scratch.down = scratch.act + act_elements;
  return scratch;
}

cudaError_t gemma4_ffn_bf16(const Gemma4FfnBf16Args &args) {
  if (args.rows <= 0) {
    return args.rows == 0 ? cudaSuccess : cudaErrorInvalidValue;
  }
  if (!ffn_common_args_valid(args)) {
    return cudaErrorInvalidValue;
  }
  if (args.rows == 1) {
    return gemma4_ffn_decode_fused_bf16_impl(
        args.residual_out, args.normed_out, args.x, args.residual,
        args.rms_weight, args.w_gate_up_decode, args.w_down_decode,
        args.decode_scratch, args.eps, args.stream);
  }
  return gemma4_ffn_prefill_bf16_impl(args);
}

cudaError_t gemma4_ffn_prefill_mlp_bf16(
    floatX *__restrict__ out,
    const floatX *__restrict__ x,
    const floatX *__restrict__ w_gate_up_decode,
    const floatX *__restrict__ w_down_decode,
    Gemma4FfnPrefillScratch scratch,
    int rows,
    cudaStream_t stream) {
  return gemma4_ffn_prefill_mlp_bf16_impl(
      out, x, w_gate_up_decode, w_down_decode, scratch, rows, stream);
}

cudaError_t gemma4_ffn_decode_fused_bf16(
    floatX *__restrict__ residual_out,
    floatX *__restrict__ normed_out,
    const floatX *__restrict__ x,
    const floatX *__restrict__ residual,
    const floatX *__restrict__ rms_weight,
    const floatX *__restrict__ w_gate_up_col_major,
    const floatX *__restrict__ w_down_row_major,
    Gemma4FfnDecodeScratch *__restrict__ scratch,
    float eps,
    cudaStream_t stream) {
  return gemma4_ffn_decode_fused_bf16_impl(
      residual_out, normed_out, x, residual, rms_weight,
      w_gate_up_col_major, w_down_row_major, scratch, eps, stream);
}
