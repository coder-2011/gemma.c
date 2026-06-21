#ifndef GEMMA4_FFN_CUH
#define GEMMA4_FFN_CUH

#include "gemma4.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stddef.h>

#define GEMMA4_FFN_DECODE_BF16_PACK_ELEMENTS 8
#define GEMMA4_FFN_DECODE_FLOAT_PACK_ELEMENTS 4
#define GEMMA4_FFN_DECODE_HIDDEN_PACKS \
    (GEMMA4_HIDDEN_SIZE / GEMMA4_FFN_DECODE_BF16_PACK_ELEMENTS)

#ifndef GEMMA4_FFN_DECODE_REDUCTION_POLICY
#define GEMMA4_FFN_DECODE_REDUCTION_POLICY 0
#endif

#ifndef GEMMA4_FFN_DECODE_PARTIAL_GROUPS
#define GEMMA4_FFN_DECODE_PARTIAL_GROUPS \
    (2 * GEMMA4_FFN_DECODE_HIDDEN_PACKS)
#endif

struct alignas(128) Gemma4FfnDecodeScratch {
#if GEMMA4_FFN_DECODE_REDUCTION_POLICY == 1
  float partials[GEMMA4_FFN_DECODE_PARTIAL_GROUPS]
                [GEMMA4_FFN_DECODE_BF16_PACK_ELEMENTS]
                [GEMMA4_FFN_DECODE_HIDDEN_PACKS];
#endif
  float accum[GEMMA4_FFN_DECODE_BF16_PACK_ELEMENTS]
             [GEMMA4_FFN_DECODE_HIDDEN_PACKS];
};

struct Gemma4FfnPrefillScratch {
  __nv_bfloat16 *act = nullptr;
  // Reused as swizzled hidden input before gate/up and swizzled down output
  // before residual add + RMSNorm.
  __nv_bfloat16 *down = nullptr;
  int capacity_rows = 0;
};

struct Gemma4FfnBf16Args {
  __nv_bfloat16 *residual_out = nullptr;
  __nv_bfloat16 *normed_out = nullptr;
  const __nv_bfloat16 *x = nullptr;
  const __nv_bfloat16 *residual = nullptr;
  const __nv_bfloat16 *rms_weight = nullptr;

  Gemma4FfnPrefillScratch prefill_scratch = {};

  const __nv_bfloat16 *w_gate_up_decode = nullptr;
  const __nv_bfloat16 *w_down_decode = nullptr;
  Gemma4FfnDecodeScratch *decode_scratch = nullptr;

  int rows = 0;
  float eps = GEMMA4_RMS_NORM_EPS;
  cudaStream_t stream = nullptr;
};

cudaError_t gemma4_ffn_decode_configure_scratch_l2(
    Gemma4FfnDecodeScratch *scratch,
    cudaStream_t stream);

size_t gemma4_ffn_prefill_scratch_elements(int rows);

Gemma4FfnPrefillScratch gemma4_ffn_prefill_scratch_from_buffer(
    __nv_bfloat16 *buffer,
    int rows);

cudaError_t gemma4_ffn_bf16(const Gemma4FfnBf16Args &args);

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
    const __nv_bfloat16 *__restrict__ w_gate_up_col_major,
    const __nv_bfloat16 *__restrict__ w_down_row_major,
    Gemma4FfnDecodeScratch *__restrict__ scratch,
    float eps,
    cudaStream_t stream);

cudaError_t gemma4_ffn_decode_swizzle_weights_bf16(
    __nv_bfloat16 *__restrict__ w_gate_up_swizzled,
    const __nv_bfloat16 *__restrict__ w_gate_up_col_major,
    __nv_bfloat16 *__restrict__ w_down_swizzled,
    const __nv_bfloat16 *__restrict__ w_down_row_major,
    cudaStream_t stream);

#endif
