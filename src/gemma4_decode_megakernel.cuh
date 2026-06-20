#ifndef GEMMA4_DECODE_MEGAKERNEL_CUH
#define GEMMA4_DECODE_MEGAKERNEL_CUH

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>

typedef struct {
  __nv_bfloat16 *state = nullptr;
  __nv_bfloat16 *next_hidden = nullptr;
  int32_t *next_token = nullptr;
  const __nv_bfloat16 *final_norm_weight = nullptr;
  const __nv_bfloat16 *lm_head_col_major = nullptr;
} Gemma4DecodeMegakernelSpineArgs;

typedef struct {
  __nv_bfloat16 *residual_out = nullptr;
  __nv_bfloat16 *normed_out = nullptr;
  __nv_bfloat16 *next_hidden = nullptr;
  int32_t *next_token = nullptr;
  const __nv_bfloat16 *ffn_x = nullptr;
  const __nv_bfloat16 *ffn_residual = nullptr;
  const __nv_bfloat16 *ffn_norm_weight = nullptr;
  const __nv_bfloat16 *ffn_gate_up_col_major = nullptr;
  const __nv_bfloat16 *ffn_down_row_major = nullptr;
  const __nv_bfloat16 *final_norm_weight = nullptr;
  const __nv_bfloat16 *lm_head_col_major = nullptr;
  float eps = 1.0e-6f;
} Gemma4DecodeMegakernelFfnTailArgs;

size_t gemma4_decode_megakernel_spine_scratch_bytes(void);

size_t gemma4_decode_megakernel_ffn_tail_scratch_bytes(void);

// Runs the first cooperative decode megakernel spine tail: final RMSNorm,
// greedy LM-head projection, and tied embedding gather.
cudaError_t gemma4_decode_megakernel_spine_bf16(
    const Gemma4DecodeMegakernelSpineArgs &args,
    void *__restrict__ scratch,
    size_t scratch_bytes,
    cudaStream_t stream);

// Runs one cooperative FFN phase followed by the final sampling tail. This
// phase does not include attention.
cudaError_t gemma4_decode_megakernel_ffn_tail_bf16(
    const Gemma4DecodeMegakernelFfnTailArgs &args,
    void *__restrict__ scratch,
    size_t scratch_bytes,
    cudaStream_t stream);

#endif
