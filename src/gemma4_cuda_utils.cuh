#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <stdint.h>
#include <string.h>

#ifndef GEMMA4_WEIGHT_LOAD_POLICY
#define GEMMA4_WEIGHT_LOAD_POLICY 1
#endif

static constexpr int WARP_SIZE = 32;

using floatX = __nv_bfloat16;

template <typename T, typename U>
__host__ __device__ constexpr auto div_up(T n, U d)
    -> decltype((n + d - 1) / d) {
  return (n + d - 1) / d;
}

inline bool is_aligned_16(const void *ptr) {
  return (reinterpret_cast<uintptr_t>(ptr) & 0xfu) == 0;
}

template <typename ElementType>
struct alignas(16) Packed128 {
  static constexpr int size = sizeof(int4) / sizeof(ElementType);

  Packed128() = default;

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

template <typename ElementType>
__device__ inline Packed128<ElementType>
load128(const ElementType *address) {
  return Packed128<ElementType>{*reinterpret_cast<const int4 *>(address)};
}

template <typename ElementType>
__device__ inline Packed128<ElementType>
load128g(const ElementType *address) {
  return Packed128<ElementType>{__ldg(reinterpret_cast<const int4 *>(address))};
}

template <typename ElementType>
__device__ inline Packed128<ElementType>
load128cs(const ElementType *address) {
  return Packed128<ElementType>{__ldcs(reinterpret_cast<const int4 *>(address))};
}

template <typename ElementType>
__device__ inline Packed128<ElementType>
load128cg(const ElementType *address) {
  int4 bits;
  asm volatile("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(bits.x), "=r"(bits.y), "=r"(bits.z), "=r"(bits.w)
               : "l"(address));
  return Packed128<ElementType>{bits};
}

template <typename ElementType>
__device__ inline Packed128<ElementType>
load128weight(const ElementType *address) {
  static_assert(GEMMA4_WEIGHT_LOAD_POLICY >= 0 &&
                    GEMMA4_WEIGHT_LOAD_POLICY <= 2,
                "GEMMA4_WEIGHT_LOAD_POLICY must be 0=cs, 1=cg, or 2=ldg");
  if constexpr (GEMMA4_WEIGHT_LOAD_POLICY == 0) {
    return load128cs(address);
  } else if constexpr (GEMMA4_WEIGHT_LOAD_POLICY == 1) {
    return load128cg(address);
  } else {
    return load128g(address);
  }
}

template <typename ElementType>
__device__ inline ElementType loadg(const ElementType *address) {
  return __ldg(address);
}

template <typename ElementType>
__device__ inline void store128(
    ElementType *address, Packed128<ElementType> value) {
  *reinterpret_cast<int4 *>(address) = value.bits();
}

template <typename ElementType>
__device__ inline void store128cs(
    ElementType *address, Packed128<ElementType> value) {
  __stcs(reinterpret_cast<int4 *>(address), value.bits());
}

using Bf16Packed128 = Packed128<__nv_bfloat16>;
static constexpr int kBf16Packed128Elements = Bf16Packed128::size;
static constexpr int kBf16Packed128Pairs = kBf16Packed128Elements / 2;
static_assert(sizeof(Bf16Packed128) == sizeof(int4) &&
                  alignof(Bf16Packed128) >= alignof(int4),
              "Packed128 bf16 must map to one aligned int4 load");
static_assert((kBf16Packed128Elements % 2) == 0,
              "Packed128 bf16 width must contain whole bf16 pairs");
static_assert(alignof(Bf16Packed128) >= alignof(__nv_bfloat162),
              "Packed128 bf16 payload must be aligned for bf16x2 access");

__device__ inline const __nv_bfloat162 *
gemma4_bf16_pack_pairs(const Bf16Packed128 &pack) {
  return reinterpret_cast<const __nv_bfloat162 *>(pack.payload);
}

__device__ inline __nv_bfloat162 *
gemma4_bf16_pack_pairs(Bf16Packed128 &pack) {
  return reinterpret_cast<__nv_bfloat162 *>(pack.payload);
}

__device__ inline void gemma4_bf16_pack_accumulate_square(
    const Bf16Packed128 &values,
    float &sum_sq) {
  const __nv_bfloat162 *pairs = gemma4_bf16_pack_pairs(values);
#pragma unroll
  for (int p = 0; p < kBf16Packed128Pairs; ++p) {
    float2 value = __bfloat1622float2(pairs[p]);
    sum_sq = fmaf(value.x, value.x, sum_sq);
    sum_sq = fmaf(value.y, value.y, sum_sq);
  }
}

__device__ inline void gemma4_bf16_pack_accumulate_dot(
    const Bf16Packed128 &x_pack,
    const Bf16Packed128 &w_pack,
    float &sum) {
  const __nv_bfloat162 *x_pairs = gemma4_bf16_pack_pairs(x_pack);
  const __nv_bfloat162 *w_pairs = gemma4_bf16_pack_pairs(w_pack);
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
  const __nv_bfloat162 *a_pairs = gemma4_bf16_pack_pairs(a);
  const __nv_bfloat162 *b_pairs = gemma4_bf16_pack_pairs(b);
  Bf16Packed128 result;
  __nv_bfloat162 *out_pairs = gemma4_bf16_pack_pairs(result);
#pragma unroll
  for (int p = 0; p < kBf16Packed128Pairs; ++p) {
    float2 av = __bfloat1622float2(a_pairs[p]);
    float2 bv = __bfloat1622float2(b_pairs[p]);
    out_pairs[p] = __floats2bfloat162_rn(av.x + bv.x, av.y + bv.y);
  }
  return result;
}

__device__ inline Bf16Packed128 gemma4_bf16_pack_apply_rmsnorm(
    const Bf16Packed128 &values,
    const Bf16Packed128 &gamma,
    float scale) {
  const __nv_bfloat162 *value_pairs = gemma4_bf16_pack_pairs(values);
  const __nv_bfloat162 *gamma_pairs = gemma4_bf16_pack_pairs(gamma);
  Bf16Packed128 result;
  __nv_bfloat162 *out_pairs = gemma4_bf16_pack_pairs(result);
#pragma unroll
  for (int p = 0; p < kBf16Packed128Pairs; ++p) {
    float2 value = __bfloat1622float2(value_pairs[p]);
    float2 weight = __bfloat1622float2(gamma_pairs[p]);
    out_pairs[p] = __floats2bfloat162_rn(value.x * scale * weight.x,
                                         value.y * scale * weight.y);
  }
  return result;
}

__device__ inline Bf16Packed128 gemma4_bf16_pack_apply_scale(
    const Bf16Packed128 &values,
    float scale) {
  const __nv_bfloat162 *value_pairs = gemma4_bf16_pack_pairs(values);
  Bf16Packed128 result;
  __nv_bfloat162 *out_pairs = gemma4_bf16_pack_pairs(result);
#pragma unroll
  for (int p = 0; p < kBf16Packed128Pairs; ++p) {
    float2 value = __bfloat1622float2(value_pairs[p]);
    out_pairs[p] = __floats2bfloat162_rn(value.x * scale,
                                         value.y * scale);
  }
  return result;
}

__device__ inline float warp_reduce_sum(float value) {
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    value += __shfl_xor_sync(0xffffffffu, value, offset);
  }
  return value;
}

template <int Count>
__device__ inline void warp_reduce_sum_to_lane0(
    float (&values)[Count]) {
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
#pragma unroll
    for (int i = 0; i < Count; ++i) {
      values[i] += __shfl_down_sync(0xffffffffu, values[i], offset);
    }
  }
}

#endif
