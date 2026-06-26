#pragma once

#include "gemma4_cuda_utils.cuh"

namespace gemma4_rope {

// Stores one split-half RoPE pair after FP32 rotation and BF16 rounding.
__device__ void store_rotated_pair_bf16(
    __nv_bfloat16 *__restrict__ head,
    const float *__restrict__ cos_row,
    const float *__restrict__ sin_row,
    int rotary_half,
    int i,
    float lo,
    float hi);

}  // namespace gemma4_rope
