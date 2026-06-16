// BSD-3-Clause FlashAttention reference wrapper for Gemma 4 attention.
//
// This file is only an experiment harness. It calls the upstream FlashAttention
// BF16 forward path from experiments/flash-attention so the project kernel can
// be compared against the same FA2 source path.

#include <cuda_bf16.h>
#include <cuda/cmath>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

#include <cutlass/numeric_types.h>

#include "flash.h"
#include "flash_fwd_launch_template.h"
#include "gemma4.h"
#include "kernel_traits.h"

namespace gemma4_flash_attention_reference_detail {

using ReferenceElement = cutlass::bfloat16_t;
using ReferenceSlidingSmem96 =
    Flash_fwd_kernel_traits<
        GEMMA4_SLIDING_HEAD_DIM, 64, 64, 4, false, false, ReferenceElement>;
using ReferenceSlidingSmem128 =
    Flash_fwd_kernel_traits<
        GEMMA4_SLIDING_HEAD_DIM, 128, 64, 8, false, false, ReferenceElement>;
using ReferenceGlobal =
    Flash_fwd_kernel_traits<
        GEMMA4_GLOBAL_HEAD_DIM, 32, 32, 2, false, false, ReferenceElement>;

bool upstream_uses_128k_trait() {
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess) return false;
  int max_smem_per_sm = 0;
  int max_smem_per_block = 0;
  if (cudaDeviceGetAttribute(&max_smem_per_sm,
                             cudaDevAttrMaxSharedMemoryPerMultiprocessor,
                             device) != cudaSuccess) {
    return false;
  }
  if (cudaDeviceGetAttribute(&max_smem_per_block,
                             cudaDevAttrMaxSharedMemoryPerBlockOptin,
                             device) != cudaSuccess) {
    return false;
  }
  constexpr int kHeadDim = GEMMA4_SLIDING_HEAD_DIM;
  return max_smem_per_block >= 2 * kHeadDim * (128 + 2 * 64) &&
         max_smem_per_sm < 4 * kHeadDim * (64 + 2 * 64);
}

void set_reference_params(
    flash::Flash_fwd_params &params,
    __nv_bfloat16 *__restrict__ d_out,
    float *__restrict__ d_softmax_lse,
    const __nv_bfloat16 *__restrict__ d_q,
    const __nv_bfloat16 *__restrict__ d_k,
    const __nv_bfloat16 *__restrict__ d_v,
    int batch_size,
    int seqlen_q,
    int seqlen_k,
    int q_heads,
    int kv_heads,
    int head_dim,
    int window_left,
    float softmax_scale) {
  params = {};
  params.q_ptr = const_cast<__nv_bfloat16 *>(d_q);
  params.k_ptr = const_cast<__nv_bfloat16 *>(d_k);
  params.v_ptr = const_cast<__nv_bfloat16 *>(d_v);
  params.o_ptr = d_out;
  params.softmax_lse_ptr = d_softmax_lse;

  params.b = batch_size;
  params.h = q_heads;
  params.h_k = kv_heads;
  params.h_h_k_ratio = q_heads / kv_heads;
  params.seqlen_q = seqlen_q;
  params.seqlen_k = seqlen_k;
  params.seqlen_q_rounded = cuda::round_up(seqlen_q, 128);
  params.seqlen_k_rounded = cuda::round_up(seqlen_k, 128);
  params.d = head_dim;
  params.d_rounded = head_dim;
  params.total_q = batch_size * seqlen_q;

  params.q_head_stride = head_dim;
  params.k_head_stride = head_dim;
  params.v_head_stride = head_dim;
  params.o_head_stride = head_dim;

  params.q_row_stride = q_heads * head_dim;
  params.k_row_stride = kv_heads * head_dim;
  params.v_row_stride = kv_heads * head_dim;
  params.o_row_stride = q_heads * head_dim;

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
}

template <typename KernelTraits,
          bool IsCausal,
          bool IsLocal>
cudaError_t selected_direct_reference_kernel_attributes(long long *out, int len) {
  if (out == nullptr || len < 16) return cudaErrorInvalidValue;
  auto kernel = &flash::flash_fwd_kernel<
      KernelTraits,
      /*Is_dropout=*/false,
      IsCausal,
      IsLocal,
      /*Has_alibi=*/false,
      /*Is_even_MN=*/false,
      /*Is_even_K=*/true,
      /*Is_softcap=*/false,
      /*Return_softmax=*/false>;
  if (KernelTraits::kSmemSize >= 48 * 1024) {
    cudaError_t status = cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
        KernelTraits::kSmemSize);
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

template <typename KernelTraits,
          bool IsCausal,
          bool IsLocal>
cudaError_t launch_direct_reference(flash::Flash_fwd_params &params,
                                    cudaStream_t stream) {
  auto kernel = &flash::flash_fwd_kernel<
      KernelTraits,
      /*Is_dropout=*/false,
      IsCausal,
      IsLocal,
      /*Has_alibi=*/false,
      /*Is_even_MN=*/false,
      /*Is_even_K=*/true,
      /*Is_softcap=*/false,
      /*Return_softmax=*/false>;
  if (KernelTraits::kSmemSize >= 48 * 1024) {
    cudaError_t status = cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
        KernelTraits::kSmemSize);
    if (status != cudaSuccess) return status;
  }
  const dim3 grid_dim(cute::ceil_div(params.seqlen_q, KernelTraits::kBlockM),
                      params.b,
                      params.h);
  constexpr dim3 block_dim(KernelTraits::kNThreads);
  kernel<<<grid_dim, block_dim, KernelTraits::kSmemSize, stream>>>(params);
  return cudaGetLastError();
}

}  // namespace gemma4_flash_attention_reference_detail

extern "C" cudaError_t gemma4_flash_attention_reference_sliding_fwd_bf16(
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
  if (batch_size <= 0 || seqlen_q <= 0 || seqlen_k <= 0 || window_left < 0) {
    return cudaErrorInvalidValue;
  }

  flash::Flash_fwd_params params;
  gemma4_flash_attention_reference_detail::set_reference_params(
      params, d_out, d_softmax_lse, d_q, d_k, d_v, batch_size, seqlen_q,
      seqlen_k, GEMMA4_NUM_QUERY_HEADS, GEMMA4_SLIDING_KV_HEADS,
      GEMMA4_SLIDING_HEAD_DIM, window_left, softmax_scale);
  flash::run_mha_fwd_<
      gemma4_flash_attention_reference_detail::ReferenceElement,
      GEMMA4_SLIDING_HEAD_DIM,
      false>(params, stream);
  return cudaGetLastError();
}

extern "C" size_t gemma4_flash_attention_reference_sliding_smem_bytes() {
  return gemma4_flash_attention_reference_detail::upstream_uses_128k_trait()
             ? gemma4_flash_attention_reference_detail::ReferenceSlidingSmem128::kSmemSize
             : gemma4_flash_attention_reference_detail::ReferenceSlidingSmem96::kSmemSize;
}

extern "C" int gemma4_flash_attention_reference_sliding_threads_per_block() {
  return gemma4_flash_attention_reference_detail::upstream_uses_128k_trait()
             ? gemma4_flash_attention_reference_detail::ReferenceSlidingSmem128::kNThreads
             : gemma4_flash_attention_reference_detail::ReferenceSlidingSmem96::kNThreads;
}

extern "C" cudaError_t gemma4_flash_attention_reference_sliding_kernel_attributes(
    long long *out,
    int len) {
  return gemma4_flash_attention_reference_detail::
      selected_direct_reference_kernel_attributes<
          gemma4_flash_attention_reference_detail::ReferenceSlidingSmem96,
          false,
          true>(out, len);
}

extern "C" cudaError_t gemma4_flash_attention_reference_global_fwd_bf16(
    __nv_bfloat16 *__restrict__ d_out,
    float *__restrict__ d_softmax_lse,
    const __nv_bfloat16 *__restrict__ d_q,
    const __nv_bfloat16 *__restrict__ d_k,
    const __nv_bfloat16 *__restrict__ d_v,
    int batch_size,
    int seqlen_q,
    int seqlen_k,
    float softmax_scale,
    cudaStream_t stream) {
  if (d_out == nullptr || d_softmax_lse == nullptr || d_q == nullptr ||
      d_k == nullptr || d_v == nullptr) {
    return cudaErrorInvalidValue;
  }
  if (batch_size <= 0 || seqlen_q <= 0 || seqlen_k <= 0) {
    return cudaErrorInvalidValue;
  }

  flash::Flash_fwd_params params;
  gemma4_flash_attention_reference_detail::set_reference_params(
      params, d_out, d_softmax_lse, d_q, d_k, d_v, batch_size, seqlen_q,
      seqlen_k, GEMMA4_NUM_QUERY_HEADS, GEMMA4_GLOBAL_KV_HEADS,
      GEMMA4_GLOBAL_HEAD_DIM, seqlen_k, softmax_scale);
  params.is_causal = true;
  params.window_size_left = -1;
  params.window_size_right = 0;
  return gemma4_flash_attention_reference_detail::launch_direct_reference<
      gemma4_flash_attention_reference_detail::ReferenceGlobal,
      true,
      false>(params, stream);
}

extern "C" size_t gemma4_flash_attention_reference_global_smem_bytes() {
  return gemma4_flash_attention_reference_detail::ReferenceGlobal::kSmemSize;
}

extern "C" int gemma4_flash_attention_reference_global_threads_per_block() {
  return gemma4_flash_attention_reference_detail::ReferenceGlobal::kNThreads;
}

extern "C" cudaError_t gemma4_flash_attention_reference_global_kernel_attributes(
    long long *out,
    int len) {
  return gemma4_flash_attention_reference_detail::
      selected_direct_reference_kernel_attributes<
          gemma4_flash_attention_reference_detail::ReferenceGlobal,
          true,
          false>(out, len);
}
