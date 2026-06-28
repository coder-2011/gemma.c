#include "gemma4_ffn.cuh"
#include "gemma4_matmul_kernels.cuh"

#include <cooperative_groups.h>
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
    const FfnBf16Pack pack_value =
        FfnBf16Pack{*reinterpret_cast<const int4 *>(src + src_offset)};
    *reinterpret_cast<int4 *>(dst + dst_offset) = pack_value.bits();
  }
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
    const FfnBf16Pack pack =
        FfnBf16Pack{*reinterpret_cast<const int4 *>(src + src_offset)};
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
    const FfnBf16Pack pack_value =
        FfnBf16Pack{*reinterpret_cast<const int4 *>(src + src_offset)};
    *reinterpret_cast<int4 *>(dst + dst_offset) = pack_value.bits();
  }
}

}  // namespace gemma4_ffn_decode_device

namespace {

namespace ffn_dev = gemma4_ffn_decode_device;
namespace matmul_dev = gemma4_matmul_kernel_impl;

constexpr int kFfnThreads = GEMMA4_FFN_DECODE_THREADS;
constexpr int kFfnWarps = kFfnThreads / 32;
constexpr int kHiddenPacks = ffn_dev::kHiddenPacks;
static_assert(kFfnThreads == kHiddenPacks,
              "FFN decode maps one CTA thread to one hidden pack");
constexpr int kSwizzleThreads = 96;
constexpr int kActualSwizzleBlocksPerRow =
    div_up(kHiddenPacks, kSwizzleThreads);
constexpr int kDirectTile = 2;
constexpr int kDirectIntermediateTiles =
    GEMMA4_INTERMEDIATE_SIZE / kDirectTile;
static_assert((GEMMA4_INTERMEDIATE_SIZE % kDirectTile) == 0,
              "direct FFN decode intermediate tile must divide F");

// Applies GeGLU in the DualGemm epilogue after gate/up fragments are materialized.
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

// Computes the tanh GELU used by Gemma's GeGLU FFN activation.
__device__ inline float gelu_tanh(float x) {
  constexpr float kSqrtTwoOverPi = 0.7978845608028654f;
  constexpr float kGeluCubic = 0.044715f;
  const float x2 = x * x;
  const float inner = kSqrtTwoOverPi * (x + kGeluCubic * x * x2);
  return 0.5f * x * (1.0f + tanhf(inner));
}

// Accumulates one BF16 down-projection pack scaled by an on-the-fly GeGLU value.
__device__ inline void accumulate_scaled_pack(
    float scale,
    const ffn_dev::FfnBf16Pack &pack,
    float (&values)[kBf16Packed128Elements]) {
  const __nv_bfloat162 *pairs =
      reinterpret_cast<const __nv_bfloat162 *>(pack.payload);
#pragma unroll
  for (int p = 0; p < kBf16Packed128Pairs; ++p) {
    const float2 packed = __bfloat1622float2(pairs[p]);
    values[2 * p] = fmaf(scale, packed.x, values[2 * p]);
    values[2 * p + 1] = fmaf(scale, packed.y, values[2 * p + 1]);
  }
}

// Reduces only live hidden-pack lanes while allowing larger caller CTAs.
__device__ inline float reduce_ffn_hidden_pack_sum(
    float value,
    float *__restrict__ warp_sums,
    int thread_idx) {
  const int lane = thread_idx & (warpSize - 1);
  const int warp = thread_idx / warpSize;

  value = warp_reduce_sum(value);
  if (lane == 0 && warp < kFfnWarps) {
    warp_sums[warp] = value;
  }
  __syncthreads();

  value = thread_idx < kFfnWarps ? warp_sums[lane] : 0.0f;
  if (warp == 0) {
    value = warp_reduce_sum(value);
  }
  return value;
}

// Computes one intermediate tile and immediately folds it into the MLP output.
__device__ inline void accumulate_direct_intermediate_tile(
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ w_gate_up_decode,
    const __nv_bfloat16 *__restrict__ w_down_decode,
    int intermediate_begin,
    int swizzled_hidden_col,
    float (&partial)[kBf16Packed128Elements],
    float (&warp_sums)[2][kDirectTile][kFfnWarps],
    float (&activation)[kDirectTile],
    bool active_hidden_pack) {
  float gate[kDirectTile] = {};
  float up[kDirectTile] = {};

  matmul_dev::gemma4_ffn_gate_up_tile_bf16_device(
      x, w_gate_up_decode, intermediate_begin, int(threadIdx.x),
      &warp_sums[0][0][0], gate, up);

  if (threadIdx.x == 0) {
#pragma unroll
    for (int t = 0; t < kDirectTile; ++t) {
      activation[t] = gelu_tanh(gate[t]) * up[t];
    }
  }
  __syncthreads();

  const __nv_bfloat16 *down_row =
      w_down_decode +
      static_cast<int64_t>(intermediate_begin) * GEMMA4_HIDDEN_SIZE +
      swizzled_hidden_col;
#pragma unroll
  for (int t = 0; t < kDirectTile; ++t) {
    if (active_hidden_pack) {
      const ffn_dev::FfnBf16Pack down_pack =
          ffn_dev::FfnBf16Pack{
              *reinterpret_cast<const int4 *>(
                  down_row + static_cast<int64_t>(t) * GEMMA4_HIDDEN_SIZE)};
      accumulate_scaled_pack(activation[t], down_pack, partial);
    }
  }
}

// Launch wrapper for swizzling hidden packs into decode weight layout.
__global__ void swizzle_hidden_packs_kernel(
    __nv_bfloat16 *__restrict__ dst,
    const __nv_bfloat16 *__restrict__ src,
    int rows) {
  ffn_dev::swizzle_hidden_packs(
      dst, src, rows, int(blockIdx.y), int(blockIdx.x),
      int(gridDim.x), int(blockDim.x), int(threadIdx.x));
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

}  // namespace

// Runs direct decode FFN math and post norm inside the caller's cooperative grid.
extern "C" __device__ void gemma4_ffn_decode_fused_bf16_device(
    __nv_bfloat16 *__restrict__ residual_out,
    __nv_bfloat16 *__restrict__ normed_out,
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ residual,
    const __nv_bfloat16 *__restrict__ rms_weight,
    const __nv_bfloat16 *__restrict__ w_gate_up_decode,
    const __nv_bfloat16 *__restrict__ w_down_decode,
    Gemma4FfnDecodeScratch *__restrict__ scratch,
    const __nv_bfloat16 *__restrict__ layer_scalar,
    float eps) {
  cooperative_groups::grid_group grid = cooperative_groups::this_grid();
  const int thread_idx = int(threadIdx.x);
  const bool active_hidden_pack = thread_idx < kFfnThreads;
  const int global_thread =
      int(blockIdx.x) * int(blockDim.x) + thread_idx;
  const int global_stride = int(gridDim.x) * int(blockDim.x);
  constexpr int accum_values =
      GEMMA4_FFN_DECODE_BF16_PACK_ELEMENTS * GEMMA4_FFN_DECODE_HIDDEN_PACKS;
  float *accum = &scratch->accum[0][0];
  for (int i = global_thread; i < accum_values; i += global_stride) {
    accum[i] = 0.0f;
  }
  grid.sync();

  __shared__ float s_warp_sums[2][kDirectTile][kFfnWarps];
  __shared__ float s_activation[kDirectTile];
  float partial[kBf16Packed128Elements] = {};
  const int swizzled_hidden_col =
      active_hidden_pack
          ? ffn_dev::hidden_pack_swizzle_index(thread_idx) *
                kBf16Packed128Elements
          : 0;

  for (int tile = int(blockIdx.x); tile < kDirectIntermediateTiles;
       tile += int(gridDim.x)) {
    accumulate_direct_intermediate_tile(
        x, w_gate_up_decode, w_down_decode, tile * kDirectTile,
        swizzled_hidden_col, partial, s_warp_sums, s_activation,
        active_hidden_pack);
  }

  if (active_hidden_pack) {
#pragma unroll
    for (int i = 0; i < kBf16Packed128Elements; ++i) {
      atomicAdd(&scratch->accum[i][thread_idx], partial[i]);
    }
  }
  grid.sync();

  // CTA 0 owns post-FFN RMSNorm/residual after all CTAs finish accumulation.
  if (blockIdx.x == 0) {
    __shared__ float s_post_warp_sums[kFfnWarps];
    __shared__ float s_rms_scale;
    ffn_dev::FfnBf16Pack mlp_pack;
    float sum_sq = 0.0f;

    if (active_hidden_pack) {
      __nv_bfloat162 *pairs =
          reinterpret_cast<__nv_bfloat162 *>(mlp_pack.payload);
#pragma unroll
      for (int p = 0; p < kBf16Packed128Pairs; ++p) {
        const float x_value = scratch->accum[2 * p][thread_idx];
        const float y_value = scratch->accum[2 * p + 1][thread_idx];
        pairs[p] = __floats2bfloat162_rn(x_value, y_value);
      }
      gemma4_bf16_pack_accumulate_square(mlp_pack, sum_sq);
    }

    const float total =
        reduce_ffn_hidden_pack_sum(sum_sq, s_post_warp_sums, thread_idx);
    if (thread_idx == 0) {
      s_rms_scale = rsqrtf(total / float(GEMMA4_HIDDEN_SIZE) + eps);
    }
    __syncthreads();

    if (active_hidden_pack) {
      const int offset = thread_idx * kBf16Packed128Elements;
      const ffn_dev::FfnBf16Pack weight_pack =
          ffn_dev::FfnBf16Pack{
              *reinterpret_cast<const int4 *>(rms_weight + offset)};
      const ffn_dev::FfnBf16Pack normed_pack =
          gemma4_bf16_pack_apply_scale_weight(
              mlp_pack, weight_pack, s_rms_scale);
      const ffn_dev::FfnBf16Pack residual_pack =
          ffn_dev::FfnBf16Pack{
              *reinterpret_cast<const int4 *>(residual + offset)};
      const ffn_dev::FfnBf16Pack residual_out_pack =
          gemma4_bf16_pack_add(residual_pack, normed_pack);
      const float output_scale =
          layer_scalar == nullptr
              ? 1.0f
              : __bfloat162float(__ldg(layer_scalar));
      const ffn_dev::FfnBf16Pack scaled_residual_out_pack =
          output_scale == 1.0f
              ? residual_out_pack
              : gemma4_bf16_pack_apply_scale(residual_out_pack, output_scale);
      *reinterpret_cast<int4 *>(normed_out + offset) = normed_pack.bits();
      *reinterpret_cast<int4 *>(residual_out + offset) =
          scaled_residual_out_pack.bits();
    }
  }
}

namespace {

// Runs direct decode FFN math and post norm in one cooperative launch.
__global__ __launch_bounds__(kFfnThreads, 1) void direct_decode_ffn_kernel(
    __nv_bfloat16 *__restrict__ residual_out,
    __nv_bfloat16 *__restrict__ normed_out,
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ residual,
    const __nv_bfloat16 *__restrict__ rms_weight,
    const __nv_bfloat16 *__restrict__ w_gate_up_decode,
    const __nv_bfloat16 *__restrict__ w_down_decode,
    Gemma4FfnDecodeScratch *__restrict__ scratch,
    const __nv_bfloat16 *__restrict__ layer_scalar,
    float eps) {
  gemma4_ffn_decode_fused_bf16_device(
      residual_out, normed_out, x, residual, rms_weight, w_gate_up_decode,
      w_down_decode, scratch, layer_scalar, eps);
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
          64, 64, 32, 64, 32, 3>(
          act, x_swizzled, w_gate_up_decode, rows, stream);
    case 128:
      return run_gate_up_geglu_decode_layout_dual_gemm_config<
          128, 64, 32, 64, 32, 5>(
          act, x_swizzled, w_gate_up_decode, rows, stream);
    default:
      return run_gate_up_geglu_decode_layout_dual_gemm_config<
          256, 64, 32, 64, 32, 3>(
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
      return launch_cutlass_bf16_gemm<64, 64, 32, 32, 32, 10>(
          act, w_down_decode, down, rows, GEMMA4_INTERMEDIATE_SIZE,
          GEMMA4_HIDDEN_SIZE, GEMMA4_HIDDEN_SIZE, stream);
    case 128:
      return launch_cutlass_bf16_gemm<64, 128, 32, 32, 64, 6>(
          act, w_down_decode, down, rows, GEMMA4_INTERMEDIATE_SIZE,
          GEMMA4_HIDDEN_SIZE, GEMMA4_HIDDEN_SIZE, stream);
    case 256:
      return launch_cutlass_bf16_gemm<128, 128, 64, 64, 64, 3>(
          act, w_down_decode, down, rows, GEMMA4_INTERMEDIATE_SIZE,
          GEMMA4_HIDDEN_SIZE, GEMMA4_HIDDEN_SIZE, stream);
    case 512:
      return launch_cutlass_bf16_gemm<256, 128, 32, 64, 64, 3>(
          act, w_down_decode, down, rows, GEMMA4_INTERMEDIATE_SIZE,
          GEMMA4_HIDDEN_SIZE, GEMMA4_HIDDEN_SIZE, stream);
    default:
      return launch_cutlass_bf16_gemm<128, 128, 32, 64, 64, 5>(
          act, w_down_decode, down, rows, GEMMA4_INTERMEDIATE_SIZE,
          GEMMA4_HIDDEN_SIZE, GEMMA4_HIDDEN_SIZE, stream);
  }
}

// Launches the one-token direct FFN cooperative kernel.
cudaError_t launch_direct_decode_ffn(
    __nv_bfloat16 *__restrict__ residual_out,
    __nv_bfloat16 *__restrict__ normed_out,
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ residual,
    const __nv_bfloat16 *__restrict__ rms_weight,
    const __nv_bfloat16 *__restrict__ w_gate_up_decode,
    const __nv_bfloat16 *__restrict__ w_down_decode,
    Gemma4FfnDecodeScratch *__restrict__ scratch,
    const __nv_bfloat16 *__restrict__ layer_scalar,
    float eps,
    cudaStream_t stream) {
  if (residual_out == nullptr || normed_out == nullptr || x == nullptr ||
      residual == nullptr || rms_weight == nullptr ||
      w_gate_up_decode == nullptr || w_down_decode == nullptr ||
      scratch == nullptr) {
    return cudaErrorInvalidValue;
  }

  static int cached_device = -1;
  static int cached_active_blocks = 0;
  int device = 0;
  cudaError_t status = cudaGetDevice(&device);
  GEMMA4_RETURN_IF_CUDA_ERROR(status);

  if (cached_device != device || cached_active_blocks <= 0) {
    cudaDeviceProp prop = {};
    status = cudaGetDeviceProperties(&prop, device);
    GEMMA4_RETURN_IF_CUDA_ERROR(status);
    if (!prop.cooperativeLaunch) {
      return cudaErrorNotSupported;
    }

    int active_blocks_per_sm = 0;
    status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks_per_sm, direct_decode_ffn_kernel, kFfnThreads, 0);
    GEMMA4_RETURN_IF_CUDA_ERROR(status);
    if (active_blocks_per_sm <= 0) {
      return cudaErrorInvalidValue;
    }
    cached_device = device;
    cached_active_blocks = active_blocks_per_sm * prop.multiProcessorCount;
  }

  __nv_bfloat16 *residual_out_arg = residual_out;
  __nv_bfloat16 *normed_out_arg = normed_out;
  const __nv_bfloat16 *x_arg = x;
  const __nv_bfloat16 *residual_arg = residual;
  const __nv_bfloat16 *rms_weight_arg = rms_weight;
  const __nv_bfloat16 *w_gate_up_arg = w_gate_up_decode;
  const __nv_bfloat16 *w_down_arg = w_down_decode;
  Gemma4FfnDecodeScratch *scratch_arg = scratch;
  const __nv_bfloat16 *layer_scalar_arg = layer_scalar;
  void *kernel_args[] = {
      &residual_out_arg,
      &normed_out_arg,
      &x_arg,
      &residual_arg,
      &rms_weight_arg,
      &w_gate_up_arg,
      &w_down_arg,
      &scratch_arg,
      &layer_scalar_arg,
      &eps};
  status = cudaLaunchCooperativeKernel(
      reinterpret_cast<void *>(direct_decode_ffn_kernel), cached_active_blocks,
      kFfnThreads, kernel_args, 0, stream);
  if (status != cudaSuccess) {
    return status;
  }
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
    const __nv_bfloat16 *__restrict__ layer_scalar,
    float eps,
    cudaStream_t stream) {
  return launch_direct_decode_ffn(
      residual_out, normed_out, x, residual, rms_weight,
      w_gate_up_decode, w_down_decode, scratch, layer_scalar, eps, stream);
}
