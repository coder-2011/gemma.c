#pragma once

#include "gemma4_cuda_utils.cuh"

#include <cmath>
#include <stdint.h>
#include <vector>

namespace gemma4_rope {

// Fills compact row-major cos/sin tables shared by standalone and fused RoPE paths.
inline void fill_tables(
    std::vector<float> &cos_table,
    std::vector<float> &sin_table,
    int32_t max_seq_len,
    int32_t rotary_half,
    float theta,
    int32_t exponent_dim) {
  cos_table.resize(static_cast<size_t>(max_seq_len) * rotary_half);
  sin_table.resize(cos_table.size());
  for (int32_t pos = 0; pos < max_seq_len; ++pos) {
    for (int32_t i = 0; i < rotary_half; ++i) {
      const float exponent = -2.0f * static_cast<float>(i) /
                             static_cast<float>(exponent_dim);
      const float angle = static_cast<float>(pos) * std::pow(theta, exponent);
      const size_t offset = static_cast<size_t>(pos) * rotary_half + i;
      cos_table[offset] = std::cos(angle);
      sin_table[offset] = std::sin(angle);
    }
  }
}

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
