#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "gemma4_cuda_utils.cuh"

static constexpr int GEMMA4_ROPE_QK_LOAD_CS = 1;

namespace gemma4_rope {

using RopePack = Bf16Packed128;
static constexpr int kRopePairsPerPack = kBf16Packed128Elements;

__device__ __forceinline__ float4 load_float4g(
    const float *__restrict__ address) {
  return __ldg(reinterpret_cast<const float4 *>(address));
}

__device__ __forceinline__ void rotate_values(float lo,
                                              float hi,
                                              float c,
                                              float s,
                                              float &out_lo,
                                              float &out_hi) {
  out_lo = fmaf(-hi, s, lo * c);
  out_hi = fmaf(lo, s, hi * c);
}

template <int Elem>
__device__ __forceinline__ void rotate_pack_element(RopePack &out_lo,
                                                    RopePack &out_hi,
                                                    const RopePack &lo,
                                                    const RopePack &hi,
                                                    float c,
                                                    float s) {
  float rotated_lo;
  float rotated_hi;
  rotate_values(__bfloat162float(lo[Elem]), __bfloat162float(hi[Elem]), c, s,
                rotated_lo, rotated_hi);
  out_lo[Elem] = __float2bfloat16_rn(rotated_lo);
  out_hi[Elem] = __float2bfloat16_rn(rotated_hi);
}

__device__ __forceinline__ void store_rotated_pair_bf16(
    floatX *__restrict__ head,
    const float *__restrict__ cos_row,
    const float *__restrict__ sin_row,
    int rotary_half,
    int i,
    float lo,
    float hi) {
  float rotated_lo;
  float rotated_hi;
  rotate_values(lo, hi, loadg(cos_row + i), loadg(sin_row + i),
                rotated_lo, rotated_hi);
  head[i] = __float2bfloat16_rn(rotated_lo);
  head[rotary_half + i] = __float2bfloat16_rn(rotated_hi);
}

__device__ __forceinline__ void rotate_pair_bf16(
    floatX *__restrict__ head,
    const float *__restrict__ cos_row,
    const float *__restrict__ sin_row,
    int rotary_half,
    int i) {
  const float lo = __bfloat162float(head[i]);
  const float hi = __bfloat162float(head[rotary_half + i]);
  store_rotated_pair_bf16(head, cos_row, sin_row, rotary_half, i, lo, hi);
}

__device__ __forceinline__ void rotate_pack_bf16(
    floatX *__restrict__ head,
    const float *__restrict__ cos_row,
    const float *__restrict__ sin_row,
    int rotary_half,
    int pack) {
  const int i = pack * kRopePairsPerPack;
  RopePack lo;
  RopePack hi;
  if constexpr (GEMMA4_ROPE_QK_LOAD_CS) {
    lo = load128cs(head + i);
    hi = load128cs(head + rotary_half + i);
  } else {
    lo = load128(head + i);
    hi = load128(head + rotary_half + i);
  }
  const float4 c0 = load_float4g(cos_row + i);
  const float4 c1 = load_float4g(cos_row + i + 4);
  const float4 s0 = load_float4g(sin_row + i);
  const float4 s1 = load_float4g(sin_row + i + 4);
  RopePack out_lo;
  RopePack out_hi;

  rotate_pack_element<0>(out_lo, out_hi, lo, hi, c0.x, s0.x);
  rotate_pack_element<1>(out_lo, out_hi, lo, hi, c0.y, s0.y);
  rotate_pack_element<2>(out_lo, out_hi, lo, hi, c0.z, s0.z);
  rotate_pack_element<3>(out_lo, out_hi, lo, hi, c0.w, s0.w);
  rotate_pack_element<4>(out_lo, out_hi, lo, hi, c1.x, s1.x);
  rotate_pack_element<5>(out_lo, out_hi, lo, hi, c1.y, s1.y);
  rotate_pack_element<6>(out_lo, out_hi, lo, hi, c1.z, s1.z);
  rotate_pack_element<7>(out_lo, out_hi, lo, hi, c1.w, s1.w);

  store128(head + i, out_lo);
  store128(head + rotary_half + i, out_hi);
}

// Supports both full RoPE and partial/p-RoPE. The caller owns rotary_half, so
// unrotated trailing dimensions remain untouched.
__device__ __forceinline__ void rotate_head_bf16(
    floatX *__restrict__ head,
    const float *__restrict__ cos_row,
    const float *__restrict__ sin_row,
    int rotary_half,
    int lane,
    int thread_count) {
  const int full_packs = rotary_half / kRopePairsPerPack;
  for (int pack = lane; pack < full_packs; pack += thread_count) {
    rotate_pack_bf16(head, cos_row, sin_row, rotary_half, pack);
  }

  const int tail_start = full_packs * kRopePairsPerPack;
  for (int i = tail_start + lane; i < rotary_half; i += thread_count) {
    rotate_pair_bf16(head, cos_row, sin_row, rotary_half, i);
  }
}

}  // namespace gemma4_rope
