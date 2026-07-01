#include "gemma4_rope.cuh"

#include "gemma4_cuda_utils.cuh"

namespace gemma4_rope {

// Stores one already-loaded split-half RoPE pair after FP32 rotation.
__device__ void store_rotated_pair_bf16(
    __nv_bfloat16 *__restrict__ head,
    const float *__restrict__ cos_row,
    const float *__restrict__ sin_row,
    int rotary_half,
    int i,
    float lo,
    float hi) {
  const float c = loadg(cos_row + i);
  const float s = loadg(sin_row + i);
  head[i] = __float2bfloat16_rn(fmaf(-hi, s, lo * c));
  head[rotary_half + i] = __float2bfloat16_rn(fmaf(lo, s, hi * c));
}

}  // namespace gemma4_rope
