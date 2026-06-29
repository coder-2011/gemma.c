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

constexpr int kFfnThreads = GEMMA4_FFN_DECODE_THREADS;
constexpr int kFfnWarps = kFfnThreads / 32;
constexpr int kHiddenPacks = ffn_dev::kHiddenPacks;
static_assert(kFfnThreads == kHiddenPacks, "FFN decode maps one CTA thread to one hidden pack");
constexpr int kSwizzleThreads = 96;
constexpr int kActualSwizzleBlocksPerRow = div_up(kHiddenPacks, kSwizzleThreads);
constexpr int kDirectIntermediateTile = 2;
constexpr int kDirectActTile = 2;
constexpr int kDirectIntermediateTiles =
    GEMMA4_INTERMEDIATE_SIZE / kDirectIntermediateTile;
constexpr int kDirectAccumBlocks = kDirectIntermediateTiles - kHiddenPacks;
static_assert((GEMMA4_INTERMEDIATE_SIZE % kDirectIntermediateTile) == 0,
              "direct FFN decode intermediate tile must divide F");
static_assert(kDirectIntermediateTile == kDirectActTile,
              "direct FFN decode computes one activation group per tile");
static_assert(kDirectAccumBlocks > 0,
              "direct FFN decode accumulate grid must be positive");

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

// Reduces the gate/up dot products for one tiny intermediate tile across a CTA.
__device__ inline void reduce_gate_up_tile(
    float (&gate)[kDirectActTile],
    float (&up)[kDirectActTile],
    float (&warp_sums)[2][kDirectActTile][kFfnWarps]) {
  const int lane = int(threadIdx.x) & (warpSize - 1);
  const int warp = int(threadIdx.x) / warpSize;

#pragma unroll
  for (int t = 0; t < kDirectActTile; ++t) {
    gate[t] = warp_reduce_sum(gate[t]);
    up[t] = warp_reduce_sum(up[t]);
    if (lane == 0) {
      warp_sums[0][t][warp] = gate[t];
      warp_sums[1][t][warp] = up[t];
    }
  }
  __syncthreads();

#pragma unroll
  for (int t = 0; t < kDirectActTile; ++t) {
    gate[t] = threadIdx.x < kFfnWarps ? warp_sums[0][t][lane] : 0.0f;
    up[t] = threadIdx.x < kFfnWarps ? warp_sums[1][t][lane] : 0.0f;
    if (warp == 0) {
      gate[t] = warp_reduce_sum(gate[t]);
      up[t] = warp_reduce_sum(up[t]);
    }
  }
}

// Computes gate/up dot products for a swizzled hidden pack of X and W_upgate.
__device__ inline void dot_gate_up_pack(
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ w_gate_up_decode,
    int gate_col0,
    int pack_idx,
    float (&gate)[kDirectActTile],
    float (&up)[kDirectActTile]) {
  const int x_col = pack_idx * kBf16Packed128Elements;
  const int weight_col =
      ffn_dev::hidden_pack_swizzle_index(pack_idx) * kBf16Packed128Elements;
  const ffn_dev::FfnBf16Pack x_pack =
      ffn_dev::FfnBf16Pack{*reinterpret_cast<const int4 *>(x + x_col)};
  const __nv_bfloat16 *gate_ptr =
      w_gate_up_decode +
      static_cast<int64_t>(2 * gate_col0) * GEMMA4_HIDDEN_SIZE + weight_col;
  const __nv_bfloat16 *up_ptr = gate_ptr + GEMMA4_HIDDEN_SIZE;

#pragma unroll
  for (int t = 0; t < kDirectActTile; ++t) {
    const int64_t row_offset =
        static_cast<int64_t>(2 * t) * GEMMA4_HIDDEN_SIZE;
    const ffn_dev::FfnBf16Pack gate_pack =
        ffn_dev::FfnBf16Pack{
            *reinterpret_cast<const int4 *>(gate_ptr + row_offset)};
    const ffn_dev::FfnBf16Pack up_pack =
        ffn_dev::FfnBf16Pack{
            *reinterpret_cast<const int4 *>(up_ptr + row_offset)};
    gemma4_bf16_pack_accumulate_dot(x_pack, gate_pack, gate[t]);
    gemma4_bf16_pack_accumulate_dot(x_pack, up_pack, up[t]);
  }
}

// Computes one intermediate tile and immediately folds it into the MLP output.
__device__ inline void accumulate_direct_intermediate_tile(
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ w_gate_up_decode,
    const __nv_bfloat16 *__restrict__ w_down_decode,
    int intermediate_begin,
    int swizzled_hidden_col,
    float (&partial)[kBf16Packed128Elements],
    float (&warp_sums)[2][kDirectActTile][kFfnWarps],
    float (&activation)[kDirectActTile]) {
  float gate[kDirectActTile] = {};
  float up[kDirectActTile] = {};

  dot_gate_up_pack(
      x, w_gate_up_decode, intermediate_begin, int(threadIdx.x), gate, up);
  reduce_gate_up_tile(gate, up, warp_sums);

  if (threadIdx.x == 0) {
#pragma unroll
    for (int t = 0; t < kDirectActTile; ++t) {
      activation[t] = gelu_tanh(gate[t]) * up[t];
    }
  }
  __syncthreads();

  const __nv_bfloat16 *down_row =
      w_down_decode +
      static_cast<int64_t>(intermediate_begin) * GEMMA4_HIDDEN_SIZE +
      swizzled_hidden_col;
#pragma unroll
  for (int t = 0; t < kDirectActTile; ++t) {
    const ffn_dev::FfnBf16Pack down_pack =
        ffn_dev::FfnBf16Pack{
            *reinterpret_cast<const int4 *>(
                down_row + static_cast<int64_t>(t) * GEMMA4_HIDDEN_SIZE)};
    accumulate_scaled_pack(activation[t], down_pack, partial);
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

// Accumulates the decode MLP output without materializing the GeGLU vector.
__global__ __launch_bounds__(kFfnThreads, 1) void
direct_decode_mlp_accumulate_kernel(
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ w_gate_up_decode,
    const __nv_bfloat16 *__restrict__ w_down_decode,
    Gemma4FfnDecodeScratch *__restrict__ scratch) {
  __shared__ float s_warp_sums[2][kDirectActTile][kFfnWarps];
  __shared__ float s_activation[kDirectActTile];

  float partial[kBf16Packed128Elements] = {};
  const int hidden_pack = int(threadIdx.x);
  const int swizzled_hidden_col =
      ffn_dev::hidden_pack_swizzle_index(hidden_pack) *
      kBf16Packed128Elements;

  const int tile0 = int(blockIdx.x);
  accumulate_direct_intermediate_tile(
      x, w_gate_up_decode, w_down_decode, tile0 * kDirectIntermediateTile,
      swizzled_hidden_col, partial, s_warp_sums, s_activation);

  const int tile1 = tile0 + kDirectAccumBlocks;
  if (tile1 < kDirectIntermediateTiles) {
    accumulate_direct_intermediate_tile(
        x, w_gate_up_decode, w_down_decode, tile1 * kDirectIntermediateTile,
        swizzled_hidden_col, partial, s_warp_sums, s_activation);
  }

#pragma unroll
  for (int i = 0; i < kBf16Packed128Elements; ++i) {
    atomicAdd(&scratch->accum[i][hidden_pack], partial[i]);
  }
}

// Applies post-FFN RMSNorm/residual directly from the accumulated MLP row.
__global__ __launch_bounds__(kFfnThreads, 1) void direct_decode_post_kernel(
    __nv_bfloat16 *__restrict__ residual_out,
    __nv_bfloat16 *__restrict__ normed_out,
    const __nv_bfloat16 *__restrict__ residual,
    const __nv_bfloat16 *__restrict__ rms_weight,
    const Gemma4FfnDecodeScratch *__restrict__ scratch,
    const __nv_bfloat16 *__restrict__ layer_scalar,
    float eps) {
  __shared__ float s_warp_sums[kFfnWarps];
  __shared__ float s_rms_scale;
  const int hidden_pack = int(threadIdx.x);
  const int offset = hidden_pack * kBf16Packed128Elements;
  ffn_dev::FfnBf16Pack mlp_pack;
  __nv_bfloat162 *pairs =
      reinterpret_cast<__nv_bfloat162 *>(mlp_pack.payload);

#pragma unroll
  for (int p = 0; p < kBf16Packed128Pairs; ++p) {
    const float x = scratch->accum[2 * p][hidden_pack];
    const float y = scratch->accum[2 * p + 1][hidden_pack];
    pairs[p] = __floats2bfloat162_rn(x, y);
  }

  float sum_sq = 0.0f;
  gemma4_bf16_pack_accumulate_square(mlp_pack, sum_sq);

  const float total =
      gemma4_block_reduce_sum<kFfnThreads>(sum_sq, s_warp_sums, hidden_pack);
  if (hidden_pack == 0) {
    s_rms_scale = rsqrtf(total / float(GEMMA4_HIDDEN_SIZE) + eps);
  }
  __syncthreads();

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
      layer_scalar == nullptr ? 1.0f : __bfloat162float(__ldg(layer_scalar));
  const ffn_dev::FfnBf16Pack scaled_residual_out_pack =
      output_scale == 1.0f
          ? residual_out_pack
          : gemma4_bf16_pack_apply_scale(residual_out_pack, output_scale);
  *reinterpret_cast<int4 *>(normed_out + offset) = normed_pack.bits();
  *reinterpret_cast<int4 *>(residual_out + offset) =
      scaled_residual_out_pack.bits();
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
    const __nv_bfloat16 *__restrict__ layer_scalar,
    float eps,
    cudaStream_t stream) {
  if (scratch == nullptr) {
    return cudaErrorInvalidValue;
  }

  cudaError_t status =
      cudaMemsetAsync(scratch->accum, 0, sizeof(scratch->accum), stream);
  GEMMA4_RETURN_IF_CUDA_ERROR(status);

  direct_decode_mlp_accumulate_kernel<<<
      kDirectAccumBlocks, kFfnThreads, 0, stream>>>(
      x, w_gate_up_decode, w_down_decode, scratch);
  status = cudaGetLastError();
  GEMMA4_RETURN_IF_CUDA_ERROR(status);

  direct_decode_post_kernel<<<1, kFfnThreads, 0, stream>>>(
      residual_out, normed_out, residual, rms_weight, scratch,
      layer_scalar, eps);
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
  return gemma4_ffn_decode_fused_bf16_impl(
      residual_out, normed_out, x, residual, rms_weight,
      w_gate_up_decode, w_down_decode, scratch, layer_scalar, eps, stream);
}
