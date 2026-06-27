#include "gemma4_ffn_tier2.cuh"

namespace {

namespace ffn_dev = gemma4_ffn_decode_device;

constexpr int kPostThreads = 128;
constexpr int kPostWarps = kPostThreads / 32;
constexpr int kHiddenPacks = ffn_dev::kHiddenPacks;

static_assert(kPostThreads % 32 == 0,
              "Tier-2 post kernel uses full warps for reductions");

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

// Runs the Tier-2 MLP through the measured decode-layout GEMM chain.
cudaError_t gemma4_ffn_tier2_decode_mlp_bf16(
    __nv_bfloat16 *__restrict__ out,
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ w_gate_up_decode,
    const __nv_bfloat16 *__restrict__ w_down_decode,
    Gemma4FfnPrefillScratch scratch,
    cudaStream_t stream) {
  if (out == nullptr || x == nullptr || w_gate_up_decode == nullptr ||
      w_down_decode == nullptr) {
    return cudaErrorInvalidValue;
  }
  return gemma4_ffn_prefill_mlp_bf16(
      out, x, w_gate_up_decode, w_down_decode, scratch, 1, stream);
}

// Runs the Tier-2 MLP, then applies post-FFN RMSNorm/residual in natural order.
cudaError_t gemma4_ffn_tier2_decode_full_bf16(
    __nv_bfloat16 *__restrict__ residual_out,
    __nv_bfloat16 *__restrict__ normed_out,
    __nv_bfloat16 *__restrict__ mlp_out,
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ residual,
    const __nv_bfloat16 *__restrict__ rms_weight,
    const __nv_bfloat16 *__restrict__ w_gate_up_decode,
    const __nv_bfloat16 *__restrict__ w_down_decode,
    Gemma4FfnPrefillScratch scratch,
    float eps,
    cudaStream_t stream) {
  if (residual_out == nullptr || normed_out == nullptr ||
      mlp_out == nullptr || residual == nullptr || rms_weight == nullptr) {
    return cudaErrorInvalidValue;
  }

  cudaError_t status = gemma4_ffn_tier2_decode_mlp_bf16(
      mlp_out, x, w_gate_up_decode, w_down_decode, scratch, stream);
  GEMMA4_RETURN_IF_CUDA_ERROR(status);

  gemma4_ffn_tier2_post_bf16_kernel<<<1, kPostThreads, 0, stream>>>(
      residual_out, normed_out, mlp_out, residual, rms_weight, eps);
  return cudaGetLastError();
}
