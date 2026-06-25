#pragma once

#include "gemma4.h"
#include "gemma4_cuda_utils.cuh"

#include <cuda_bf16.h>
#include <math.h>

namespace gemma4_rmsnorm_device {

// Applies learned RMSNorm to one hidden row using caller-owned shared storage.
template <int Threads>
__device__ inline void rmsnorm_hidden_row_bf16(
    __nv_bfloat16 *__restrict__ out,
    const __nv_bfloat16 *__restrict__ in,
    const __nv_bfloat16 *__restrict__ weight,
    float eps,
    Bf16Packed128 *__restrict__ cached_input,
    float *__restrict__ warp_sums,
    float &scale,
    int thread_idx) {
  constexpr int packs = GEMMA4_HIDDEN_SIZE / kBf16Packed128Elements;
  static_assert((GEMMA4_HIDDEN_SIZE % kBf16Packed128Elements) == 0,
                "hidden RMSNorm width must divide bf16 pack width");

  float sum_sq = 0.0f;
  for (int pack = thread_idx; pack < packs; pack += Threads) {
    const int offset = pack * kBf16Packed128Elements;
    const Bf16Packed128 values = load128g(in + offset);
    // Keep the row pack on-chip for the post-reduction normalization pass.
    cached_input[pack] = values;
    gemma4_bf16_pack_accumulate_square(values, sum_sq);
  }

  const float total =
      gemma4_block_reduce_sum<Threads>(sum_sq, warp_sums, thread_idx);
  if (thread_idx == 0) {
    scale = rsqrtf(total / float(GEMMA4_HIDDEN_SIZE) + eps);
  }
  __syncthreads();

  for (int pack = thread_idx; pack < packs; pack += Threads) {
    const int offset = pack * kBf16Packed128Elements;
    const Bf16Packed128 values = cached_input[pack];
    const Bf16Packed128 gamma = load128g(weight + offset);
    const Bf16Packed128 result =
        gemma4_bf16_pack_apply_rmsnorm(values, gamma, scale);
    store128(out + offset, result);
  }
}

}  // namespace gemma4_rmsnorm_device
