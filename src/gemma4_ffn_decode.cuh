#ifndef GEMMA4_FFN_DECODE_CUH
#define GEMMA4_FFN_DECODE_CUH

#include "gemma4.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

struct alignas(128) Gemma4FfnDecodeScratch {
  float accum[GEMMA4_HIDDEN_SIZE];
  int lock;
  int padding[31];
};

cudaError_t gemma4_ffn_decode_configure_scratch_l2(
    Gemma4FfnDecodeScratch *scratch,
    cudaStream_t stream);

// Decode-only fused dense FFN for one token.
//
// Weight layouts:
// - w_gate_up_col_major: [5376, 43008] column-major, with gate columns first
//   and up columns second.
// - w_down_row_major: [21504, 5376] row-major. This differs from the generic
//   projection path so a CTA can update coalesced full-width output columns.
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

#endif
