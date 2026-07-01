#pragma once

#include "gemma4.h"
#include "gemma4_cuda_utils.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stddef.h>

// GEMMA4_FFN_FOLDED_PRE_NORM=1 folds the pre-FFN RMSNorm into the gate/up
// GEMV: the FFN reads the raw residual row, multiplies gamma inside the dot,
// and applies the scalar s2 after the block reduction (default FFN path only).
#ifndef GEMMA4_FFN_FOLDED_PRE_NORM
#define GEMMA4_FFN_FOLDED_PRE_NORM 0
#endif

static constexpr int GEMMA4_FFN_DECODE_BF16_PACK_ELEMENTS = kBf16Packed128Elements;
static constexpr int GEMMA4_FFN_DECODE_HIDDEN_PACKS =
    GEMMA4_HIDDEN_SIZE / GEMMA4_FFN_DECODE_BF16_PACK_ELEMENTS;

struct alignas(128) Gemma4FfnDecodeScratch {
  // Tiled decode accumulates the final MLP row without a GeGLU activation spill.
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

// Returns BF16 scratch elements needed by the prefill FFN MLP temporaries.
size_t gemma4_ffn_prefill_scratch_elements(int rows);

// Splits caller-owned BF16 scratch into prefill FFN temporary spans.
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

// Runs the decode FFN tail inside a caller-owned cooperative CUDA grid.
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
    float eps,
    const __nv_bfloat16 *__restrict__ pre_norm_weight,
    const float *__restrict__ pre_ffn_scale);

namespace gemma4_ffn_decode_device {

constexpr int kHiddenPacks = GEMMA4_HIDDEN_SIZE / kBf16Packed128Elements;

static_assert(kHiddenPacks == GEMMA4_FFN_DECODE_HIDDEN_PACKS,
              "FFN scratch hidden-pack shape must match bf16 pack width");

// Maps natural hidden packs into the swizzled decode weight layout.
__host__ __device__ inline int hidden_pack_swizzle_index(int chunk) {
  constexpr int kSwizzleChunks = 8;
  const unsigned u = static_cast<unsigned>(chunk);
  const unsigned col = u & (kSwizzleChunks - 1);
  const unsigned row = (u >> 3) & (kSwizzleChunks - 1);
  return static_cast<int>((u & ~(kSwizzleChunks - 1u)) | (col ^ row));
}

}  // namespace gemma4_ffn_decode_device
