#pragma once

#include "gemma4_ffn.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

// Runs the Tier-2 decode FFN MLP and writes one natural-order row.
cudaError_t gemma4_ffn_tier2_decode_mlp_bf16(
    __nv_bfloat16 *__restrict__ out,
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ w_gate_up_decode,
    const __nv_bfloat16 *__restrict__ w_down_decode,
    Gemma4FfnPrefillScratch scratch,
    cudaStream_t stream);

// Runs Tier-2 decode FFN plus post-FFN RMSNorm/residual add.
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
    cudaStream_t stream);
