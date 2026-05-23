#ifndef GEMMA4_MATMUL_DEVICE_CUH
#define GEMMA4_MATMUL_DEVICE_CUH

#include "gemma4_cuda_utils.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace gemma4_matmul_device {

__device__ inline int pack_offset(int pack_idx) {
  return pack_idx * kBf16Packed128Elements;
}

template <int K>
__device__ inline int weight_offset(int col, int element_idx) {
  return col * K + element_idx;
}

template <int BlockCount, int SwizzleTileBlocks>
__device__ inline int swizzle_col_block(int block_idx) {
  if constexpr (SwizzleTileBlocks <= 1) {
    return block_idx;
  } else {
    static_assert((BlockCount % SwizzleTileBlocks) == 0,
                  "swizzled decode GEMV block count must divide tile size");
    constexpr int tiles = BlockCount / SwizzleTileBlocks;
    return (block_idx % SwizzleTileBlocks) * tiles +
           block_idx / SwizzleTileBlocks;
  }
}

__device__ inline Bf16Packed128
load_activation_pack(const __nv_bfloat16 *__restrict__ x, int element_idx) {
  return load128g(x + element_idx);
}

template <int K>
__device__ inline Bf16Packed128
load_weight_pack(
    const __nv_bfloat16 *__restrict__ w_col_major, int col, int element_idx) {
  return load128cs(w_col_major + weight_offset<K>(col, element_idx));
}

template <int ColsPerBlock>
__device__ inline void store_cols(
    __nv_bfloat16 *__restrict__ dst, const float (&sums)[ColsPerBlock]) {
  if constexpr (ColsPerBlock == kBf16Packed128Elements) {
    Bf16Packed128 out;
#pragma unroll
    for (int col = 0; col < ColsPerBlock; ++col) {
      out[col] = __float2bfloat16_rn(sums[col]);
    }
    store128(dst, out);
  } else {
#pragma unroll
    for (int col = 0; col < ColsPerBlock; ++col) {
      dst[col] = __float2bfloat16_rn(sums[col]);
    }
  }
}

template <int K, int ColsPerBlock, int Threads>
__device__ inline void dot_cols(
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ w_col_major, int col0, int thread_idx,
    float (&sums)[ColsPerBlock]) {
  static_assert((K % kBf16Packed128Elements) == 0,
                "decode GEMV K must be divisible by Packed128 bf16 width");
  static_assert((Threads % WARP_SIZE) == 0,
                "decode thread count must be a whole number of warps");

  constexpr int packs_per_col = K / kBf16Packed128Elements;

#pragma unroll
  for (int pack_idx = thread_idx; pack_idx < packs_per_col; pack_idx += Threads) {
    const int element_idx = pack_offset(pack_idx);
    const Bf16Packed128 x_pack = load_activation_pack(x, element_idx);
#pragma unroll
    for (int col = 0; col < ColsPerBlock; ++col) {
      const Bf16Packed128 w_pack =
          load_weight_pack<K>(w_col_major, col0 + col, element_idx);
      gemma4_bf16_pack_accumulate_dot(x_pack, w_pack, sums[col]);
    }
  }
}

template <int K, int ColsPerBlock, int Threads>
__device__ inline void dot_cols_pair(
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ w_a_col_major,
    const __nv_bfloat16 *__restrict__ w_b_col_major,
    int col0_a,
    int col0_b,
    int thread_idx,
    float (&a_sums)[ColsPerBlock],
    float (&b_sums)[ColsPerBlock]) {
  static_assert((K % kBf16Packed128Elements) == 0,
                "decode GEMV K must be divisible by Packed128 bf16 width");
  static_assert((Threads % WARP_SIZE) == 0,
                "decode thread count must be a whole number of warps");

  constexpr int packs_per_col = K / kBf16Packed128Elements;

#pragma unroll
  for (int pack_idx = thread_idx; pack_idx < packs_per_col; pack_idx += Threads) {
    const int element_idx = pack_offset(pack_idx);
    const Bf16Packed128 x_pack = load_activation_pack(x, element_idx);
#pragma unroll
    for (int col = 0; col < ColsPerBlock; ++col) {
      const Bf16Packed128 a_pack =
          load_weight_pack<K>(w_a_col_major, col0_a + col, element_idx);
      const Bf16Packed128 b_pack =
          load_weight_pack<K>(w_b_col_major, col0_b + col, element_idx);
      gemma4_bf16_pack_accumulate_dot(x_pack, a_pack, a_sums[col]);
      gemma4_bf16_pack_accumulate_dot(x_pack, b_pack, b_sums[col]);
    }
  }
}

template <bool SwizzleX>
__device__ inline int shared_pack_index(int chunk) {
  if constexpr (SwizzleX) {
    constexpr int kSwizzleChunks = 8;  // 8 x 128-bit chunks = one 128-byte row.
    const unsigned u = static_cast<unsigned>(chunk);
    const unsigned col = u & (kSwizzleChunks - 1);
    const unsigned row = (u >> 3) & (kSwizzleChunks - 1);
    return static_cast<int>((u & ~(kSwizzleChunks - 1u)) | (col ^ row));
  }
  return chunk;
}

template <int K, int ColsPerBlock, int Threads, bool SwizzleX>
__device__ inline void dot_cols_pair_shared_x(
    const Bf16Packed128 *__restrict__ s_x,
    const __nv_bfloat16 *__restrict__ w_a_col_major,
    const __nv_bfloat16 *__restrict__ w_b_col_major,
    int col0_a,
    int col0_b,
    int thread_idx,
    float (&a_sums)[ColsPerBlock],
    float (&b_sums)[ColsPerBlock]) {
  static_assert((K % kBf16Packed128Elements) == 0,
                "decode GEMV K must be divisible by Packed128 bf16 width");
  static_assert((Threads % WARP_SIZE) == 0,
                "decode thread count must be a whole number of warps");

  constexpr int packs_per_col = K / kBf16Packed128Elements;

#pragma unroll
  for (int pack_idx = thread_idx; pack_idx < packs_per_col; pack_idx += Threads) {
    const int element_idx = pack_offset(pack_idx);
    const Bf16Packed128 x_pack = s_x[shared_pack_index<SwizzleX>(pack_idx)];
#pragma unroll
    for (int col = 0; col < ColsPerBlock; ++col) {
      const Bf16Packed128 a_pack =
          load_weight_pack<K>(w_a_col_major, col0_a + col, element_idx);
      const Bf16Packed128 b_pack =
          load_weight_pack<K>(w_b_col_major, col0_b + col, element_idx);
      gemma4_bf16_pack_accumulate_dot(x_pack, a_pack, a_sums[col]);
      gemma4_bf16_pack_accumulate_dot(x_pack, b_pack, b_sums[col]);
    }
  }
}

template <int ColsPerBlock, int Threads>
__device__ inline void reduce_cols(
    int thread_idx,
    float (&warp_sums)[ColsPerBlock][Threads / WARP_SIZE],
    float (&sums)[ColsPerBlock]) {
  constexpr int warps = Threads / WARP_SIZE;
  const int lane = thread_idx & (WARP_SIZE - 1);
  const int warp = thread_idx / WARP_SIZE;

  warp_reduce_sum_to_lane0(sums);

  if (lane == 0) {
#pragma unroll
    for (int col = 0; col < ColsPerBlock; ++col) {
      warp_sums[col][warp] = sums[col];
    }
  }
  __syncthreads();

#pragma unroll
  for (int col = 0; col < ColsPerBlock; ++col) {
    sums[col] = thread_idx < warps ? warp_sums[col][lane] : 0.0f;
  }

  if (warp == 0) {
    warp_reduce_sum_to_lane0(sums);
  }
}

template <int ColsPerBlock, int Threads>
__device__ inline void reduce_cols_pair(
    int thread_idx,
    float (&a_warp_sums)[ColsPerBlock][Threads / WARP_SIZE],
    float (&b_warp_sums)[ColsPerBlock][Threads / WARP_SIZE],
    float (&a_sums)[ColsPerBlock],
    float (&b_sums)[ColsPerBlock]) {
  constexpr int warps = Threads / WARP_SIZE;
  const int lane = thread_idx & (WARP_SIZE - 1);
  const int warp = thread_idx / WARP_SIZE;

  warp_reduce_sum_to_lane0(a_sums);
  warp_reduce_sum_to_lane0(b_sums);

  if (lane == 0) {
#pragma unroll
    for (int col = 0; col < ColsPerBlock; ++col) {
      a_warp_sums[col][warp] = a_sums[col];
      b_warp_sums[col][warp] = b_sums[col];
    }
  }
  __syncthreads();

#pragma unroll
  for (int col = 0; col < ColsPerBlock; ++col) {
    a_sums[col] = thread_idx < warps ? a_warp_sums[col][lane] : 0.0f;
    b_sums[col] = thread_idx < warps ? b_warp_sums[col][lane] : 0.0f;
  }

  if (warp == 0) {
    warp_reduce_sum_to_lane0(a_sums);
    warp_reduce_sum_to_lane0(b_sums);
  }
}

template <int K, int ColsPerBlock, int Threads>
__device__ inline void dot_cols_reduce(
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ w_col_major,
    int col0,
    int thread_idx,
    float (&warp_sums)[ColsPerBlock][Threads / WARP_SIZE],
    float (&sums)[ColsPerBlock]) {
  dot_cols<K, ColsPerBlock, Threads>(x, w_col_major, col0, thread_idx, sums);
  reduce_cols<ColsPerBlock, Threads>(thread_idx, warp_sums, sums);
}

template <int K, int ColsPerBlock, int Threads>
__device__ inline void dot_cols_pair_reduce(
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ w_a_col_major,
    const __nv_bfloat16 *__restrict__ w_b_col_major,
    int col0_a,
    int col0_b,
    int thread_idx,
    float (&a_warp_sums)[ColsPerBlock][Threads / WARP_SIZE],
    float (&b_warp_sums)[ColsPerBlock][Threads / WARP_SIZE],
    float (&a_sums)[ColsPerBlock],
    float (&b_sums)[ColsPerBlock]) {
  dot_cols_pair<K, ColsPerBlock, Threads>(
      x, w_a_col_major, w_b_col_major, col0_a, col0_b, thread_idx,
      a_sums, b_sums);
  reduce_cols_pair<ColsPerBlock, Threads>(
      thread_idx, a_warp_sums, b_warp_sums, a_sums, b_sums);
}

template <int K, int ColsPerBlock, int Threads, bool SwizzleX>
__device__ inline void dot_cols_pair_shared_x_reduce(
    const Bf16Packed128 *__restrict__ s_x,
    const __nv_bfloat16 *__restrict__ w_a_col_major,
    const __nv_bfloat16 *__restrict__ w_b_col_major,
    int col0_a,
    int col0_b,
    int thread_idx,
    float (&a_warp_sums)[ColsPerBlock][Threads / WARP_SIZE],
    float (&b_warp_sums)[ColsPerBlock][Threads / WARP_SIZE],
    float (&a_sums)[ColsPerBlock],
    float (&b_sums)[ColsPerBlock]) {
  dot_cols_pair_shared_x<K, ColsPerBlock, Threads, SwizzleX>(
      s_x, w_a_col_major, w_b_col_major, col0_a, col0_b, thread_idx,
      a_sums, b_sums);
  reduce_cols_pair<ColsPerBlock, Threads>(
      thread_idx, a_warp_sums, b_warp_sums, a_sums, b_sums);
}

template <int K, int ColsPerBlock, int Threads, bool SwizzleX>
__device__ inline void dot_cols_pair_shared_x_reduce_to_smem(
    const Bf16Packed128 *__restrict__ s_x,
    const __nv_bfloat16 *__restrict__ w_a_col_major,
    const __nv_bfloat16 *__restrict__ w_b_col_major,
    int col0_a,
    int col0_b,
    int thread_idx,
    float (&a_warp_sums)[ColsPerBlock][Threads / WARP_SIZE],
    float (&b_warp_sums)[ColsPerBlock][Threads / WARP_SIZE],
    float *__restrict__ s_a_out,
    float *__restrict__ s_b_out) {
  float a_sums[ColsPerBlock] = {};
  float b_sums[ColsPerBlock] = {};
  dot_cols_pair_shared_x_reduce<K, ColsPerBlock, Threads, SwizzleX>(
      s_x, w_a_col_major, w_b_col_major, col0_a, col0_b, thread_idx,
      a_warp_sums, b_warp_sums, a_sums, b_sums);

  if (thread_idx == 0) {
#pragma unroll
    for (int col = 0; col < ColsPerBlock; ++col) {
      s_a_out[col] = a_sums[col];
      s_b_out[col] = b_sums[col];
    }
  }
}

template <int K,
          int N,
          int ColsPerBlock,
          int Threads,
          int SwizzleTileBlocks>
__device__ inline void decode_gemv_cols_device(
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ w_col_major,
    __nv_bfloat16 *__restrict__ y,
    int physical_block_idx,
    float (&warp_sums)[ColsPerBlock][Threads / WARP_SIZE]) {
  static_assert((N % ColsPerBlock) == 0,
                "decode GEMV N must be divisible by columns per block");

  constexpr int blocks = N / ColsPerBlock;
  const int logical_block =
      swizzle_col_block<blocks, SwizzleTileBlocks>(physical_block_idx);
  const int col0 = logical_block * ColsPerBlock;

  float sums[ColsPerBlock] = {};
  dot_cols_reduce<K, ColsPerBlock, Threads>(
      x, w_col_major, col0, threadIdx.x, warp_sums, sums);

  if (threadIdx.x == 0) {
    store_cols<ColsPerBlock>(y + col0, sums);
  }
}

template <int K,
          int N,
          int ColsPerBlock,
          int Threads,
          int SwizzleTileBlocks>
__device__ inline int decode_gemv_cols_smem_device(
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ w_col_major,
    __nv_bfloat16 *__restrict__ s_y_tile,
    int physical_block_idx,
    float (&warp_sums)[ColsPerBlock][Threads / WARP_SIZE]) {
  static_assert((N % ColsPerBlock) == 0,
                "decode GEMV N must be divisible by columns per block");

  constexpr int blocks = N / ColsPerBlock;
  const int logical_block =
      swizzle_col_block<blocks, SwizzleTileBlocks>(physical_block_idx);
  const int col0 = logical_block * ColsPerBlock;

  float sums[ColsPerBlock] = {};
  dot_cols_reduce<K, ColsPerBlock, Threads>(
      x, w_col_major, col0, threadIdx.x, warp_sums, sums);

  if (threadIdx.x == 0) {
    store_cols<ColsPerBlock>(s_y_tile, sums);
  }
  return col0;
}

}  // namespace gemma4_matmul_device

#endif
