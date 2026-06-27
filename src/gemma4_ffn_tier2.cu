#include "gemma4_ffn_tier2.cuh"

#include "gemma4.h"
#include "gemma4_cuda_utils.cuh"
#include "gemma4_ffn.cuh"

namespace {

namespace ffn_dev = gemma4_ffn_decode_device;

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kOutputCols = 256;
constexpr int kFtile = 2 * kWarps;
constexpr int kHiddenPacks = GEMMA4_HIDDEN_SIZE / kBf16Packed128Elements;
constexpr int kPostThreads = 128;
constexpr int kPostWarps = kPostThreads / 32;

static_assert(GEMMA4_HIDDEN_SIZE % kOutputCols == 0,
              "Tier-2 FFN output tile must divide hidden width");
static_assert(GEMMA4_INTERMEDIATE_SIZE % kFtile == 0,
              "Tier-2 FFN intermediate tile must divide FFN width");
static_assert(kPostThreads % 32 == 0,
              "Tier-2 post kernel uses full warps for reductions");

// Applies Gemma's tanh GELU approximation in float for the generated A1 value.
__device__ inline float gelu_tanh(float x) {
  constexpr float kSqrtTwoOverPi = 0.7978845608028654f;
  constexpr float kGeluCubic = 0.044715f;
  const float x_sq = x * x;
  const float inner = kSqrtTwoOverPi * (x + kGeluCubic * x * x_sq);
  return 0.5f * x * (1.0f + tanhf(inner));
}

// Computes one gate/up dot pair for one intermediate channel in a single warp.
__device__ inline void dot_gate_up_channel(
    const ffn_dev::FfnBf16Pack *__restrict__ s_x,
    const __nv_bfloat16 *__restrict__ w_gate_up_decode,
    int f,
    int lane,
    float &gate,
    float &up) {
  gate = 0.0f;
  up = 0.0f;
  const int gate_row = 2 * f;
  const int up_row = gate_row + 1;
  for (int pack = lane; pack < kHiddenPacks; pack += warpSize) {
    const int swizzled_col =
        ffn_dev::hidden_pack_swizzle_index(pack) * kBf16Packed128Elements;
    const ffn_dev::FfnBf16Pack x_pack = s_x[pack];
    const ffn_dev::FfnBf16Pack gate_pack =
        ffn_dev::FfnBf16Pack{*reinterpret_cast<const int4 *>(
            w_gate_up_decode +
            static_cast<int64_t>(gate_row) * GEMMA4_HIDDEN_SIZE +
            swizzled_col)};
    const ffn_dev::FfnBf16Pack up_pack =
        ffn_dev::FfnBf16Pack{*reinterpret_cast<const int4 *>(
            w_gate_up_decode +
            static_cast<int64_t>(up_row) * GEMMA4_HIDDEN_SIZE +
            swizzled_col)};
    gemma4_bf16_pack_accumulate_dot(x_pack, gate_pack, gate);
    gemma4_bf16_pack_accumulate_dot(x_pack, up_pack, up);
  }
  gate = warp_reduce_sum(gate);
  up = warp_reduce_sum(up);
}

// Maps a natural hidden column to the decode-swizzled weight column.
__device__ inline int swizzled_hidden_col(int natural_col) {
  const int pack = natural_col / kBf16Packed128Elements;
  const int pack_col = natural_col % kBf16Packed128Elements;
  return ffn_dev::hidden_pack_swizzle_index(pack) *
         kBf16Packed128Elements + pack_col;
}

// Computes one output-column group while reusing each generated GeGLU tile.
__global__ __launch_bounds__(kThreads, 1) void
gemma4_ffn_tier2_decode_mlp_bf16_kernel(
    __nv_bfloat16 *__restrict__ out,
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ w_gate_up_decode,
    const __nv_bfloat16 *__restrict__ w_down_decode) {
  __shared__ ffn_dev::FfnBf16Pack s_x[kHiddenPacks];
  __shared__ float s_act[kFtile];

  const int thread_idx = int(threadIdx.x);
  const int lane = thread_idx & (warpSize - 1);
  const int warp = thread_idx / warpSize;
  const int col = int(blockIdx.x) * kOutputCols + thread_idx;
  const bool has_output_col = thread_idx < kOutputCols;
  const int down_col = has_output_col ? swizzled_hidden_col(col) : 0;
  float acc = 0.0f;

  for (int pack = thread_idx; pack < kHiddenPacks; pack += kThreads) {
    s_x[pack] = ffn_dev::FfnBf16Pack{
        *reinterpret_cast<const int4 *>(
            x + pack * kBf16Packed128Elements)};
  }
  __syncthreads();

  for (int f0 = 0; f0 < GEMMA4_INTERMEDIATE_SIZE; f0 += kFtile) {
#pragma unroll
    for (int phase = 0; phase < 2; ++phase) {
      const int f_local = phase * kWarps + warp;
      float gate = 0.0f;
      float up = 0.0f;
      dot_gate_up_channel(s_x, w_gate_up_decode, f0 + f_local, lane,
                          gate, up);
      if (lane == 0) {
        s_act[f_local] = gelu_tanh(gate) * up;
      }
    }
    __syncthreads();

    if (has_output_col) {
#pragma unroll
      for (int f_local = 0; f_local < kFtile; ++f_local) {
        const __nv_bfloat16 w = __ldg(
            w_down_decode +
            static_cast<int64_t>(f0 + f_local) * GEMMA4_HIDDEN_SIZE +
            down_col);
        acc = fmaf(s_act[f_local], __bfloat162float(w), acc);
      }
    }
    __syncthreads();
  }

  if (has_output_col) {
    out[col] = __float2bfloat16_rn(acc);
  }
}

// RMS-normalizes the natural-order MLP row and adds the residual in one CTA.
__global__ __launch_bounds__(kPostThreads, 1) void
gemma4_ffn_tier2_post_bf16_kernel(
    __nv_bfloat16 *__restrict__ residual_out,
    __nv_bfloat16 *__restrict__ normed_out,
    const __nv_bfloat16 *__restrict__ mlp,
    const __nv_bfloat16 *__restrict__ residual,
    const __nv_bfloat16 *__restrict__ rms_weight,
    float eps) {
  __shared__ float s_warp_sums[kPostWarps];
  __shared__ float s_scale;

  const int thread_idx = int(threadIdx.x);
  float sum_sq = 0.0f;
  for (int pack = thread_idx; pack < kHiddenPacks; pack += kPostThreads) {
    const int offset = pack * kBf16Packed128Elements;
    const ffn_dev::FfnBf16Pack mlp_pack =
        ffn_dev::FfnBf16Pack{*reinterpret_cast<const int4 *>(mlp + offset)};
    gemma4_bf16_pack_accumulate_square(mlp_pack, sum_sq);
  }

  const float total =
      gemma4_block_reduce_sum<kPostThreads>(sum_sq, s_warp_sums, thread_idx);
  if (thread_idx == 0) {
    s_scale = rsqrtf(total / float(GEMMA4_HIDDEN_SIZE) + eps);
  }
  __syncthreads();

  for (int pack = thread_idx; pack < kHiddenPacks; pack += kPostThreads) {
    const int offset = pack * kBf16Packed128Elements;
    const ffn_dev::FfnBf16Pack mlp_pack =
        ffn_dev::FfnBf16Pack{*reinterpret_cast<const int4 *>(mlp + offset)};
    const ffn_dev::FfnBf16Pack weight_pack =
        ffn_dev::FfnBf16Pack{*reinterpret_cast<const int4 *>(
            rms_weight + offset)};
    const ffn_dev::FfnBf16Pack normed_pack =
        gemma4_bf16_pack_apply_scale_weight(mlp_pack, weight_pack, s_scale);
    const ffn_dev::FfnBf16Pack residual_pack =
        ffn_dev::FfnBf16Pack{*reinterpret_cast<const int4 *>(
            residual + offset)};
    const ffn_dev::FfnBf16Pack residual_out_pack =
        gemma4_bf16_pack_add(residual_pack, normed_pack);
    *reinterpret_cast<int4 *>(normed_out + offset) = normed_pack.bits();
    *reinterpret_cast<int4 *>(residual_out + offset) =
        residual_out_pack.bits();
  }
}

}  // namespace

// Launches the fixed-shape Tier-2 decode FFN MLP prototype.
cudaError_t gemma4_ffn_tier2_decode_mlp_bf16(
    __nv_bfloat16 *__restrict__ out,
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ w_gate_up_decode,
    const __nv_bfloat16 *__restrict__ w_down_decode,
    cudaStream_t stream) {
  if (out == nullptr || x == nullptr || w_gate_up_decode == nullptr ||
      w_down_decode == nullptr) {
    return cudaErrorInvalidValue;
  }

  const dim3 grid_dim(GEMMA4_HIDDEN_SIZE / kOutputCols);
  const dim3 block_dim(kThreads);
  gemma4_ffn_tier2_decode_mlp_bf16_kernel<<<
      grid_dim, block_dim, 0, stream>>>(
      out, x, w_gate_up_decode, w_down_decode);
  return cudaGetLastError();
}

// Launches Tier-2 MLP followed by fused post-FFN RMSNorm/residual add.
cudaError_t gemma4_ffn_tier2_decode_full_bf16(
    __nv_bfloat16 *__restrict__ residual_out,
    __nv_bfloat16 *__restrict__ normed_out,
    __nv_bfloat16 *__restrict__ mlp_scratch,
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ residual,
    const __nv_bfloat16 *__restrict__ rms_weight,
    const __nv_bfloat16 *__restrict__ w_gate_up_decode,
    const __nv_bfloat16 *__restrict__ w_down_decode,
    float eps,
    cudaStream_t stream) {
  if (residual_out == nullptr || normed_out == nullptr ||
      mlp_scratch == nullptr || residual == nullptr || rms_weight == nullptr) {
    return cudaErrorInvalidValue;
  }

  cudaError_t status = gemma4_ffn_tier2_decode_mlp_bf16(
      mlp_scratch, x, w_gate_up_decode, w_down_decode, stream);
  GEMMA4_RETURN_IF_CUDA_ERROR(status);

  gemma4_ffn_tier2_post_bf16_kernel<<<1, kPostThreads, 0, stream>>>(
      residual_out, normed_out, mlp_scratch, residual, rms_weight, eps);
  return cudaGetLastError();
}
