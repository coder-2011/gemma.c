// BSD-3-Clause FlashAttention-derived Gemma 4 sliding-window attention path.
//
// This translation unit intentionally keeps the project-facing implementation
// in one file while reusing the upstream FlashAttention-2 SM80 device templates
// cloned under experiments/flash-attention. The cloned project carries the full
// upstream BSD-3-Clause license in experiments/flash-attention/LICENSE.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

#include "gemma4.h"

#define FLASH_NAMESPACE gemma4_flash_attn_sm80
#include "flash.h"
#include "flash_fwd_launch_template.h"

namespace gemma4_flash_attention {

using Gemma4Fa2Element = cutlass::bfloat16_t;
using Gemma4Fa2KernelTraits =
    Flash_fwd_kernel_traits<
        GEMMA4_SLIDING_HEAD_DIM, 64, 64, 4, false, false, Gemma4Fa2Element>;
using Gemma4Fa2FwdParams = gemma4_flash_attn_sm80::Flash_fwd_params;
using Gemma4SelectedKernel =
    decltype(&gemma4_flash_attn_sm80::flash_fwd_kernel<
             Gemma4Fa2KernelTraits,
             /*Is_dropout=*/false,
             /*Is_causal=*/false,
             /*Is_local=*/true,
             /*Has_alibi=*/false,
             /*Is_even_MN=*/false,
             /*Is_even_K=*/true,
             /*Is_softcap=*/false,
             /*Return_softmax=*/false>);

template <typename T>
constexpr T round_up(T value, T multiple) {
  return ((value + multiple - 1) / multiple) * multiple;
}

Gemma4Fa2FwdParams make_params(
    __nv_bfloat16 *__restrict__ d_out,
    float *__restrict__ d_softmax_lse,
    const __nv_bfloat16 *__restrict__ d_q,
    const __nv_bfloat16 *__restrict__ d_k,
    const __nv_bfloat16 *__restrict__ d_v,
    int batch_size,
    int seqlen_q,
    int seqlen_k,
    int window_left,
    float softmax_scale) {
  Gemma4Fa2FwdParams params{};
  params.q_ptr = const_cast<__nv_bfloat16 *>(d_q);
  params.k_ptr = const_cast<__nv_bfloat16 *>(d_k);
  params.v_ptr = const_cast<__nv_bfloat16 *>(d_v);
  params.o_ptr = d_out;
  params.softmax_lse_ptr = d_softmax_lse;

  params.b = batch_size;
  params.h = GEMMA4_NUM_QUERY_HEADS;
  params.h_k = GEMMA4_SLIDING_KV_HEADS;
  params.h_h_k_ratio = GEMMA4_NUM_QUERY_HEADS / GEMMA4_SLIDING_KV_HEADS;
  params.seqlen_q = seqlen_q;
  params.seqlen_k = seqlen_k;
  params.seqlen_q_rounded = round_up(seqlen_q, 128);
  params.seqlen_k_rounded = round_up(seqlen_k, 128);
  params.d = GEMMA4_SLIDING_HEAD_DIM;
  params.d_rounded = GEMMA4_SLIDING_HEAD_DIM;
  params.total_q = batch_size * seqlen_q;

  params.q_head_stride = GEMMA4_SLIDING_HEAD_DIM;
  params.k_head_stride = GEMMA4_SLIDING_HEAD_DIM;
  params.v_head_stride = GEMMA4_SLIDING_HEAD_DIM;
  params.o_head_stride = GEMMA4_SLIDING_HEAD_DIM;

  params.q_row_stride = GEMMA4_NUM_QUERY_HEADS * GEMMA4_SLIDING_HEAD_DIM;
  params.k_row_stride = GEMMA4_SLIDING_KV_HEADS * GEMMA4_SLIDING_HEAD_DIM;
  params.v_row_stride = GEMMA4_SLIDING_KV_HEADS * GEMMA4_SLIDING_HEAD_DIM;
  params.o_row_stride = GEMMA4_NUM_QUERY_HEADS * GEMMA4_SLIDING_HEAD_DIM;

  params.q_batch_stride = int64_t(seqlen_q) * params.q_row_stride;
  params.k_batch_stride = int64_t(seqlen_k) * params.k_row_stride;
  params.v_batch_stride = int64_t(seqlen_k) * params.v_row_stride;
  params.o_batch_stride = int64_t(seqlen_q) * params.o_row_stride;

  params.scale_softmax = softmax_scale;
  params.scale_softmax_log2 = softmax_scale * float(M_LOG2E);
  params.p_dropout = 1.0f;
  params.p_dropout_in_uint8_t = 255;
  params.rp_dropout = 1.0f;
  params.scale_softmax_rp_dropout = softmax_scale;
  params.window_size_left = window_left;
  params.window_size_right = 0;
  params.page_block_size = 1;
  params.is_bf16 = true;
  params.is_causal = false;
  params.is_seqlens_k_cumulative = true;
  return params;
}

cudaError_t launch_sliding(Gemma4Fa2FwdParams &params, cudaStream_t stream) {
  gemma4_flash_attn_sm80::run_flash_fwd<
      Gemma4Fa2KernelTraits,
      /*Is_dropout=*/false,
      /*Is_causal=*/false>(params, stream);
  return cudaGetLastError();
}

cudaError_t selected_kernel_attributes(long long *out, int len) {
  if (out == nullptr || len < 16) return cudaErrorInvalidValue;
  auto kernel = &gemma4_flash_attn_sm80::flash_fwd_kernel<
      Gemma4Fa2KernelTraits,
      /*Is_dropout=*/false,
      /*Is_causal=*/false,
      /*Is_local=*/true,
      /*Has_alibi=*/false,
      /*Is_even_MN=*/false,
      /*Is_even_K=*/true,
      /*Is_softcap=*/false,
      /*Return_softmax=*/false>;
  if (Gemma4Fa2KernelTraits::kSmemSize >= 48 * 1024) {
    cudaError_t status = cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
        Gemma4Fa2KernelTraits::kSmemSize);
    if (status != cudaSuccess) return status;
  }
  cudaFuncAttributes attr{};
  cudaError_t status = cudaFuncGetAttributes(&attr, kernel);
  if (status != cudaSuccess) return status;
  out[0] = static_cast<long long>(attr.sharedSizeBytes);
  out[1] = static_cast<long long>(attr.constSizeBytes);
  out[2] = static_cast<long long>(attr.localSizeBytes);
  out[3] = attr.maxThreadsPerBlock;
  out[4] = attr.numRegs;
  out[5] = attr.ptxVersion;
  out[6] = attr.binaryVersion;
  out[7] = attr.cacheModeCA;
  out[8] = attr.maxDynamicSharedSizeBytes;
  out[9] = attr.preferredShmemCarveout;
  out[10] = attr.clusterDimMustBeSet;
  out[11] = attr.requiredClusterWidth;
  out[12] = attr.requiredClusterHeight;
  out[13] = attr.requiredClusterDepth;
  out[14] = attr.clusterSchedulingPolicyPreference;
  out[15] = attr.nonPortableClusterSizeAllowed;
  return cudaSuccess;
}

}  // namespace gemma4_flash_attention

extern "C" cudaError_t gemma4_flash_attention_sliding_fwd_bf16(
    __nv_bfloat16 *__restrict__ d_out,
    float *__restrict__ d_softmax_lse,
    const __nv_bfloat16 *__restrict__ d_q,
    const __nv_bfloat16 *__restrict__ d_k,
    const __nv_bfloat16 *__restrict__ d_v,
    int batch_size,
    int seqlen_q,
    int seqlen_k,
    int window_left,
    float softmax_scale,
    cudaStream_t stream) {
  if (d_out == nullptr || d_softmax_lse == nullptr || d_q == nullptr ||
      d_k == nullptr || d_v == nullptr) {
    return cudaErrorInvalidValue;
  }
  if (batch_size <= 0 || seqlen_q <= 0 || seqlen_k <= 0) {
    return cudaErrorInvalidValue;
  }
  if (window_left < 0) {
    return cudaErrorInvalidValue;
  }

  gemma4_flash_attention::Gemma4Fa2FwdParams params =
      gemma4_flash_attention::make_params(d_out, d_softmax_lse, d_q, d_k, d_v,
                                          batch_size, seqlen_q, seqlen_k,
                                          window_left, softmax_scale);
  return gemma4_flash_attention::launch_sliding(params, stream);
}

extern "C" size_t gemma4_flash_attention_sliding_smem_bytes() {
  return gemma4_flash_attention::Gemma4Fa2KernelTraits::kSmemSize;
}

extern "C" int gemma4_flash_attention_sliding_threads_per_block() {
  return gemma4_flash_attention::Gemma4Fa2KernelTraits::kNThreads;
}

extern "C" cudaError_t gemma4_flash_attention_sliding_kernel_attributes(
    long long *out,
    int len) {
  return gemma4_flash_attention::selected_kernel_attributes(out, len);
}
