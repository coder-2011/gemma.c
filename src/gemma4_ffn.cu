#ifndef GEMMA4_WEIGHT_LOAD_POLICY
#define GEMMA4_WEIGHT_LOAD_POLICY 0
#endif

#include "gemma4_ffn.cuh"

#include <cutlass/array.h>
#include <cutlass/epilogue/thread/activation.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/epilogue/thread/scale_type.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

#include "device/dual_gemm.h"


namespace {

namespace ffn_dev = gemma4_ffn_decode_device;

constexpr int kFfnThreads = GEMMA4_FFN_DECODE_THREADS;
constexpr int kFfnWarps = kFfnThreads / WARP_SIZE;
constexpr int kHiddenPacks = ffn_dev::kHiddenPacks;
constexpr int kActTile = ffn_dev::kActTile;
constexpr int kReductionPolicy = ffn_dev::kReductionPolicy;
constexpr int kPartialGroups = ffn_dev::kPartialGroups;
constexpr int kSwizzleThreads = 96;
constexpr int kActualSwizzleBlocksPerRow =
    div_up(kHiddenPacks, kSwizzleThreads);
constexpr int kAccumBlocks = ffn_dev::kAccumBlocks;


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

    FragmentCompute gate_compute = to_compute(gate);
    FragmentCompute up_compute = to_compute(up);
    cutlass::epilogue::thread::GELU_taylor<FragmentCompute> gelu;
    cutlass::multiplies<FragmentCompute> multiply;
    return to_output(multiply(gelu(gate_compute), up_compute));
  }
};

// Launch wrapper for the standalone decode FFN accumulation phase.
__global__ __launch_bounds__(kFfnThreads, 1) void
gemma4_ffn_decode_accumulate_bf16_kernel(
    const floatX *__restrict__ x,
    const floatX *__restrict__ w_gate_up_col_major,
    const floatX *__restrict__ w_down_row_major,
    Gemma4FfnDecodeScratch *__restrict__ scratch) {
  __shared__ float s_matmul_warp_sums[2][kActTile][kFfnWarps];
  __shared__ float s_act[kActTile];

  ffn_dev::decode_accumulate<kFfnThreads>(
      x, w_gate_up_col_major, w_down_row_major, scratch, int(blockIdx.x),
      int(threadIdx.x), s_matmul_warp_sums, s_act);
}

// Launch wrapper for the optional partial-group decode FFN accumulation phase.
__global__ __launch_bounds__(kFfnThreads, 1) void
gemma4_ffn_decode_accumulate_partials_bf16_kernel(
    const floatX *__restrict__ x,
    const floatX *__restrict__ w_gate_up_col_major,
    const floatX *__restrict__ w_down_row_major,
    Gemma4FfnDecodeScratch *__restrict__ scratch) {
  __shared__ float s_matmul_warp_sums[2][kActTile][kFfnWarps];
  __shared__ float s_act[kActTile];

  ffn_dev::decode_accumulate_partials<kFfnThreads>(
      x, w_gate_up_col_major, w_down_row_major, scratch, int(blockIdx.x),
      int(threadIdx.x), s_matmul_warp_sums, s_act);
}

// Launch wrapper for post-FFN RMSNorm and residual addition.
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

  ffn_dev::decode_finalize<kFfnThreads>(
      residual_out, normed_out, residual, rms_weight, scratch, eps,
      s_rms_warp_sums, s_scale, int(threadIdx.x));
}

// Launch wrapper for swizzling hidden packs into decode weight layout.
__global__ void swizzle_hidden_packs_kernel(
    floatX *__restrict__ dst,
    const floatX *__restrict__ src,
    int rows) {
  (void)rows;
  ffn_dev::swizzle_hidden_packs(
      dst, src, int(blockIdx.y), int(blockIdx.x), int(gridDim.x),
      int(blockDim.x), int(threadIdx.x));
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

  ffn_dev::rmsnorm_residual_from_swizzled_down<kFfnThreads>(
      residual_out, normed_out, down_swizzled, residual, rms_weight, eps,
      s_rms_warp_sums, s_scale, int(blockIdx.x), int(threadIdx.x));
}

// Reorders the swizzled hidden packs produced by the prefill down GEMM back to
// the natural hidden layout expected by the standalone post-FFN RMSNorm.
__global__ __launch_bounds__(kFfnThreads, 1) void
unswizzle_hidden_packs_kernel(
    floatX *__restrict__ dst,
    const floatX *__restrict__ src) {
  ffn_dev::unswizzle_hidden_packs(
      dst, src, int(blockIdx.x), int(blockDim.x), int(threadIdx.x));
}

// Launch wrapper for gate/up row interleaving plus hidden-pack swizzling.
__global__ void swizzle_gate_up_interleaved_kernel(
    floatX *__restrict__ dst,
    const floatX *__restrict__ src) {
  ffn_dev::swizzle_gate_up_interleaved(
      dst, src, int(blockIdx.y), int(blockIdx.x), int(gridDim.x),
      int(blockDim.x), int(threadIdx.x));
}

template <int ThreadblockM,
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

  typename Gemm::Arguments args(
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
  switch (rows <= 64 ? 64 : rows <= 128 ? 128 : 0) {
    case 64:
      return run_prefill_gate_up_geglu_decode_layout_dual_gemm_config<64, 64, 32, 64, 32, 3>(act, x_swizzled, w_gate_up_decode, rows, stream);
    case 128:
      return run_prefill_gate_up_geglu_decode_layout_dual_gemm_config<128, 64, 32, 64, 32, 5>(act, x_swizzled, w_gate_up_decode, rows, stream);
    default:
      return run_prefill_gate_up_geglu_decode_layout_dual_gemm_config<256, 64, 32, 64, 32, 3>(act, x_swizzled, w_gate_up_decode, rows, stream);
  }
}

cudaError_t launch_prefill_down_gemm(
    const floatX *__restrict__ act,
    const floatX *__restrict__ w_down_row_major,
    floatX *__restrict__ down,
    int rows,
    cudaStream_t stream) {
  // Row thresholds mirror the measured 12B FFN-down prefill sweep.
  // Template args: ThreadblockM, ThreadblockN, ThreadblockK, WarpM, WarpN, Stages.
  switch (rows <= 64 ? 64 : rows <= 128 ? 128 : rows <= 256 ? 256 : rows <= 512 ? 512 : 0) {
    case 64:
      return launch_cutlass_bf16_gemm<64, 64, 32, 32, 32, 10>(act, w_down_row_major, down, rows, GEMMA4_INTERMEDIATE_SIZE, GEMMA4_HIDDEN_SIZE, GEMMA4_HIDDEN_SIZE, stream);
    case 128:
      return launch_cutlass_bf16_gemm<64, 128, 32, 32, 64, 6>(act, w_down_row_major, down, rows, GEMMA4_INTERMEDIATE_SIZE, GEMMA4_HIDDEN_SIZE, GEMMA4_HIDDEN_SIZE, stream);
    case 256:
      return launch_cutlass_bf16_gemm<128, 128, 64, 64, 64, 3>(act, w_down_row_major, down, rows, GEMMA4_INTERMEDIATE_SIZE, GEMMA4_HIDDEN_SIZE, GEMMA4_HIDDEN_SIZE, stream);
    case 512:
      return launch_cutlass_bf16_gemm<256, 128, 32, 64, 64, 3>(act, w_down_row_major, down, rows, GEMMA4_INTERMEDIATE_SIZE, GEMMA4_HIDDEN_SIZE, GEMMA4_HIDDEN_SIZE, stream);
    default:
      return launch_cutlass_bf16_gemm<128, 128, 32, 64, 64, 5>(act, w_down_row_major, down, rows, GEMMA4_INTERMEDIATE_SIZE, GEMMA4_HIDDEN_SIZE, GEMMA4_HIDDEN_SIZE, stream);
  }
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
  if (args.prefill_scratch.capacity_rows < args.rows) {
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
