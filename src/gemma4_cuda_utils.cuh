#ifndef GEMMA4_CUDA_UTILS_CUH
#define GEMMA4_CUDA_UTILS_CUH

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <stdint.h>
#include <string.h>

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
__device__ __forceinline__ Packed128<ElementType>
load128(const ElementType *address) {
  return Packed128<ElementType>{*reinterpret_cast<const int4 *>(address)};
}

template <typename ElementType>
__device__ __forceinline__ Packed128<ElementType>
load128cs(const ElementType *address) {
  return Packed128<ElementType>{__ldcs(reinterpret_cast<const int4 *>(address))};
}

template <typename ElementType>
__device__ __forceinline__ void store128(
    ElementType *address, Packed128<ElementType> value) {
  *reinterpret_cast<int4 *>(address) = value.bits();
}

template <typename ElementType>
__device__ __forceinline__ void store128cs(
    ElementType *address, Packed128<ElementType> value) {
  __stcs(reinterpret_cast<int4 *>(address), value.bits());
}

__device__ __forceinline__ float warp_reduce_sum(float value) {
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    value += __shfl_xor_sync(0xffffffffu, value, offset);
  }
  return value;
}

#endif
