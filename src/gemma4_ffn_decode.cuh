#ifndef GEMMA4_FFN_DECODE_CUH
#define GEMMA4_FFN_DECODE_CUH

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
#define GEMMA4_FFN_DECODE_PARTIAL_GROUPS 1344
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
  // Temporary [rows, 43008] packed gate/up projection output.
  __nv_bfloat16 *gate_up = nullptr;
  // Temporary [rows, 21504] GeGLU activation.
  __nv_bfloat16 *act = nullptr;
  // Temporary [rows, 5376] down projection output.
  __nv_bfloat16 *down = nullptr;
  int capacity_rows = 0;
};

struct Gemma4FfnBf16Args {
  __nv_bfloat16 *residual_out = nullptr;
  __nv_bfloat16 *normed_out = nullptr;
  const __nv_bfloat16 *x = nullptr;
  const __nv_bfloat16 *residual = nullptr;
  const __nv_bfloat16 *rms_weight = nullptr;

  // Canonical prefill layouts:
  // - gate/up: [5376, 43008] column-major, gate columns first then up.
  // - down: [21504, 5376] row-major.
  const __nv_bfloat16 *w_gate_up_prefill_col_major = nullptr;
  const __nv_bfloat16 *w_down_prefill_row_major = nullptr;
  Gemma4FfnPrefillScratch prefill_scratch = {};

  // Decode-prepared layouts produced by gemma4_ffn_decode_swizzle_weights_bf16().
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

// Dense FFN for decode or prefill. rows == 1 uses the existing fused decode
// path; rows > 1 uses CUTLASS prefill GEMMs plus standalone GeGLU and
// residual+RMSNorm kernels.
cudaError_t gemma4_ffn_bf16(const Gemma4FfnBf16Args &args);

// Decode-only fused dense FFN for one token. Kept for callers that still invoke
// the old decode-specific entry point; new code should call gemma4_ffn_bf16().
//
// Weight layouts expected by this fused path:
// - w_gate_up_col_major: decode-prepared [43008, 5376] rows where gate/up rows
//   are interleaved by intermediate column and the hidden dimension is
//   pre-swizzled in 128-bit bf16 packs by
//   gemma4_ffn_decode_swizzle_weights_bf16().
// - w_down_row_major: decode-prepared [21504, 5376] row-major, with the hidden
//   dimension pre-swizzled in 128-bit bf16 packs by the same helper.
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
    // Source layout: [5376, 43008] column-major, gate columns first then up.
    const __nv_bfloat16 *__restrict__ w_gate_up_col_major,
    __nv_bfloat16 *__restrict__ w_down_swizzled,
    // Source layout: [21504, 5376] row-major.
    const __nv_bfloat16 *__restrict__ w_down_row_major,
    cudaStream_t stream);

#endif
