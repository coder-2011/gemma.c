#pragma once

#include "gemma4.h"
#include "gemma4_cuda_utils.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stddef.h>

static constexpr int GEMMA4_FFN_DECODE_BF16_PACK_ELEMENTS = 8;
static constexpr int GEMMA4_FFN_DECODE_HIDDEN_PACKS =
    GEMMA4_HIDDEN_SIZE / GEMMA4_FFN_DECODE_BF16_PACK_ELEMENTS;

#ifndef GEMMA4_FFN_DECODE_SWIZZLE_X
static constexpr int GEMMA4_FFN_DECODE_SWIZZLE_X = 1;
#endif

#ifndef GEMMA4_FFN_DECODE_THREADS
static constexpr int GEMMA4_FFN_DECODE_THREADS =
    GEMMA4_FFN_DECODE_HIDDEN_PACKS;
#endif

struct alignas(128) Gemma4FfnDecodeScratch {
  // Direct decode accumulates the final MLP row without a GeGLU activation spill.
  float accum[GEMMA4_FFN_DECODE_BF16_PACK_ELEMENTS]
             [GEMMA4_FFN_DECODE_HIDDEN_PACKS];
};

struct Gemma4FfnPrefillScratch {
  __nv_bfloat16 *act = nullptr;
  // Reused as swizzled hidden input before gate/up and swizzled down output
  // before post-FFN RMSNorm + residual add.
  __nv_bfloat16 *down = nullptr;
  int capacity_rows = 0;
};

size_t gemma4_ffn_prefill_scratch_elements(int rows);

Gemma4FfnPrefillScratch gemma4_ffn_prefill_scratch_from_buffer(
    __nv_bfloat16 *buffer,
    int rows);

// Runs prefill GeGLU FFN only and writes the natural-order down-projection row.
cudaError_t gemma4_ffn_prefill_mlp_bf16(
    __nv_bfloat16 *__restrict__ out,
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ w_gate_up_decode,
    const __nv_bfloat16 *__restrict__ w_down_decode,
    Gemma4FfnPrefillScratch scratch,
    int rows,
    cudaStream_t stream);

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
    cudaStream_t stream);

cudaError_t gemma4_ffn_decode_swizzle_weights_bf16(
    __nv_bfloat16 *__restrict__ w_gate_up_swizzled,
    const __nv_bfloat16 *__restrict__ w_gate_up_col_major,
    __nv_bfloat16 *__restrict__ w_down_swizzled,
    const __nv_bfloat16 *__restrict__ w_down_row_major,
    cudaStream_t stream);

namespace gemma4_ffn_decode_device {

constexpr int kHiddenPacks = GEMMA4_HIDDEN_SIZE / kBf16Packed128Elements;
constexpr int kSwizzleX = GEMMA4_FFN_DECODE_SWIZZLE_X;

static_assert(kHiddenPacks == GEMMA4_FFN_DECODE_HIDDEN_PACKS,
              "FFN scratch hidden-pack shape must match bf16 pack width");
static_assert(kSwizzleX == 0 || kSwizzleX == 1,
              "FFN hidden-pack swizzle must be 0 or 1");

using FfnBf16Pack = Bf16Packed128;

// Maps natural hidden packs into the swizzled decode weight layout.
__host__ __device__ inline int hidden_pack_swizzle_index(int chunk) {
  if constexpr (kSwizzleX) {
    constexpr int kSwizzleChunks = 8;
    const unsigned u = static_cast<unsigned>(chunk);
    const unsigned col = u & (kSwizzleChunks - 1);
    const unsigned row = (u >> 3) & (kSwizzleChunks - 1);
    return static_cast<int>((u & ~(kSwizzleChunks - 1u)) | (col ^ row));
  }
  return chunk;
}

}  // namespace gemma4_ffn_decode_device
