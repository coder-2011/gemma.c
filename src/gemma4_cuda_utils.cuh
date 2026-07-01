#pragma once

#include "gemma4.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#define GEMMA4_RETURN_IF_CUDA_ERROR(expr)            \
  do {                                               \
    const cudaError_t gemma4_cuda_status__ = (expr); \
    if (gemma4_cuda_status__ != cudaSuccess) {       \
      return gemma4_cuda_status__;                   \
    }                                                \
  } while (0)

// Rounds integer division up for launch and layout dimensions.
template <typename T, typename U>
__host__ __device__ constexpr auto div_up(T n, U d) {
  return (n + d - 1) / d;
}

// Checks that a host-side pointer satisfies a power-of-two alignment.
template <size_t Alignment>
inline bool is_aligned_to(const void *ptr) {
  return (reinterpret_cast<uintptr_t>(ptr) & (Alignment - 1)) == 0;
}

// Checks the 16-byte alignment needed by Packed128 loads and stores.
inline bool is_aligned_16(const void *ptr) {
  return is_aligned_to<16>(ptr);
}

template <typename ElementType>
struct alignas(16) Packed128 {
  static constexpr int size = sizeof(int4) / sizeof(ElementType);

  Packed128() = default;

  // Reinterprets a caller-loaded 128-bit register as typed elements.
  __device__ explicit Packed128(int4 bits) {
    static_assert(sizeof(bits) == sizeof(payload), "Packed128 size mismatch");
    memcpy(payload, &bits, sizeof(bits));
  }

  __device__ ElementType &operator[](int index) { return payload[index]; }

  __device__ const ElementType &operator[](int index) const {
    return payload[index];
  }

  __device__ int4 bits() const {
    int4 result;
    static_assert(sizeof(result) == sizeof(payload), "Packed128 size mismatch");
    memcpy(&result, payload, sizeof(result));
    return result;
  }

  ElementType payload[size];
};

// Loads through CUDA's read-only path when a value is not written by this kernel.
template <typename ElementType>
__device__ inline ElementType loadg(const ElementType *address) {
  return __ldg(address);
}

using Bf16Packed128 = Packed128<__nv_bfloat16>;
static constexpr int kBf16Packed128Elements = Bf16Packed128::size;
static constexpr int kBf16Packed128Pairs = kBf16Packed128Elements / 2;

__device__ inline void gemma4_bf16_pack_accumulate_square(
    const Bf16Packed128 &values,
    float &sum_sq) {
  const __nv_bfloat162 *pairs =
      reinterpret_cast<const __nv_bfloat162 *>(values.payload);
#pragma unroll
  for (int p = 0; p < kBf16Packed128Pairs; ++p) {
    const float2 value = __bfloat1622float2(pairs[p]);
    sum_sq = fmaf(value.x, value.x, sum_sq);
    sum_sq = fmaf(value.y, value.y, sum_sq);
  }
}

__device__ inline void gemma4_bf16_pack_accumulate_dot(
    const Bf16Packed128 &x_pack,
    const Bf16Packed128 &w_pack,
    float &sum) {
  const __nv_bfloat162 *x_pairs =
      reinterpret_cast<const __nv_bfloat162 *>(x_pack.payload);
  const __nv_bfloat162 *w_pairs =
      reinterpret_cast<const __nv_bfloat162 *>(w_pack.payload);
#pragma unroll
  for (int p = 0; p < kBf16Packed128Pairs; ++p) {
    const float2 xv = __bfloat1622float2(x_pairs[p]);
    const float2 wv = __bfloat1622float2(w_pairs[p]);
    sum = fmaf(xv.x, wv.x, sum);
    sum = fmaf(xv.y, wv.y, sum);
  }
}

__device__ inline Bf16Packed128 gemma4_bf16_pack_add(
    const Bf16Packed128 &a,
    const Bf16Packed128 &b) {
  const __nv_bfloat162 *a_pairs =
      reinterpret_cast<const __nv_bfloat162 *>(a.payload);
  const __nv_bfloat162 *b_pairs =
      reinterpret_cast<const __nv_bfloat162 *>(b.payload);
  Bf16Packed128 result;
  __nv_bfloat162 *out_pairs =
      reinterpret_cast<__nv_bfloat162 *>(result.payload);
#pragma unroll
  for (int p = 0; p < kBf16Packed128Pairs; ++p) {
    const float2 av = __bfloat1622float2(a_pairs[p]);
    const float2 bv = __bfloat1622float2(b_pairs[p]);
    out_pairs[p] = __floats2bfloat162_rn(av.x + bv.x, av.y + bv.y);
  }
  return result;
}

// Applies a row-wide scalar to one BF16 pack.
__device__ inline Bf16Packed128 gemma4_bf16_pack_apply_scale(
    const Bf16Packed128 &values,
    float scale) {
  const __nv_bfloat162 *value_pairs =
      reinterpret_cast<const __nv_bfloat162 *>(values.payload);
  Bf16Packed128 result;
  __nv_bfloat162 *out_pairs =
      reinterpret_cast<__nv_bfloat162 *>(result.payload);
#pragma unroll
  for (int p = 0; p < kBf16Packed128Pairs; ++p) {
    const float2 value = __bfloat1622float2(value_pairs[p]);
    out_pairs[p] = __floats2bfloat162_rn(value.x * scale,
                                         value.y * scale);
  }
  return result;
}

// Applies a row-wide scalar and per-channel weight to one BF16 pack.
__device__ inline Bf16Packed128 gemma4_bf16_pack_apply_scale_weight(
    const Bf16Packed128 &values,
    const Bf16Packed128 &weight,
    float scale) {
  const __nv_bfloat162 *value_pairs =
      reinterpret_cast<const __nv_bfloat162 *>(values.payload);
  const __nv_bfloat162 *weight_pairs =
      reinterpret_cast<const __nv_bfloat162 *>(weight.payload);
  Bf16Packed128 result;
  __nv_bfloat162 *out_pairs =
      reinterpret_cast<__nv_bfloat162 *>(result.payload);
#pragma unroll
  for (int p = 0; p < kBf16Packed128Pairs; ++p) {
    const float2 value = __bfloat1622float2(value_pairs[p]);
    const float2 gamma = __bfloat1622float2(weight_pairs[p]);
    out_pairs[p] = __floats2bfloat162_rn(value.x * scale * gamma.x,
                                         value.y * scale * gamma.y);
  }
  return result;
}

// Reduces one scalar across the active warp.
__device__ inline float warp_reduce_sum(float value) {
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    value += __shfl_xor_sync(0xffffffffu, value, offset);
  }
  return value;
}

// Reduces fixed-size per-thread arrays so lane 0 owns each final sum.
template <int Count>
__device__ inline void warp_reduce_sum_to_lane0(
    float (&values)[Count]) {
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
#pragma unroll
    for (int i = 0; i < Count; ++i) {
      values[i] += __shfl_down_sync(0xffffffffu, values[i], offset);
    }
  }
}
