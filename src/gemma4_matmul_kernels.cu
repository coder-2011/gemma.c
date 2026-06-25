#include "gemma4_matmul_kernels.cuh"
#include "gemma4.h"

// Concrete Gemma projection kernels and host dispatch.

#ifndef GEMMA4_WEIGHT_LOAD_POLICY
#define GEMMA4_WEIGHT_LOAD_POLICY 0
#endif

#include "gemma4_cuda_utils.cuh"

#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/numeric_types.h>

namespace gemma4_matmul_kernel_impl {

constexpr int kFinalLogitsDecodeThreads = 1024;
constexpr int kFinalLogitsDecodeColsPerBlock = 8;
constexpr int kFinalLogitsDecodeMinBlocksPerSm = 1;

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
  return load128weight(w_col_major + weight_offset<K>(col, element_idx));
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

#if GEMMA4_DECODE_GEMV_BUFFER_STAGES <= 1
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
#else
  constexpr int kStages = GEMMA4_DECODE_GEMV_BUFFER_STAGES;
  static_assert(kStages > 1 && kStages <= 4,
                "decode GEMV register buffering supports 2-4 stages");
  Bf16Packed128 x_stage[kStages];
  Bf16Packed128 w_stage[kStages][ColsPerBlock];

  int pack_idx = thread_idx;
#pragma unroll
  for (int stage = 0; stage < kStages; ++stage) {
    const int stage_pack_idx = thread_idx + stage * Threads;
    if (stage_pack_idx < packs_per_col) {
      const int element_idx = pack_offset(stage_pack_idx);
      x_stage[stage] = load_activation_pack(x, element_idx);
#pragma unroll
      for (int col = 0; col < ColsPerBlock; ++col) {
        w_stage[stage][col] =
            load_weight_pack<K>(w_col_major, col0 + col, element_idx);
      }
    }
  }

  for (int iter = 0; pack_idx < packs_per_col; ++iter, pack_idx += Threads) {
    const int stage = iter % kStages;
    Bf16Packed128 x_pack = x_stage[stage];
    Bf16Packed128 w_pack[ColsPerBlock];
#pragma unroll
    for (int col = 0; col < ColsPerBlock; ++col) {
      w_pack[col] = w_stage[stage][col];
    }

    const int next_pack_idx = pack_idx + kStages * Threads;
    if (next_pack_idx < packs_per_col) {
      const int element_idx = pack_offset(next_pack_idx);
      x_stage[stage] = load_activation_pack(x, element_idx);
#pragma unroll
      for (int col = 0; col < ColsPerBlock; ++col) {
        w_stage[stage][col] =
            load_weight_pack<K>(w_col_major, col0 + col, element_idx);
      }
    }

#pragma unroll
    for (int col = 0; col < ColsPerBlock; ++col) {
      gemma4_bf16_pack_accumulate_dot(x_pack, w_pack[col], sums[col]);
    }
  }
#endif
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
    const int swizzled_pack_idx = shared_pack_index<SwizzleX>(pack_idx);
    const int swizzled_element_idx = pack_offset(swizzled_pack_idx);
    const Bf16Packed128 x_pack = s_x[swizzled_pack_idx];
#pragma unroll
    for (int col = 0; col < ColsPerBlock; ++col) {
      const Bf16Packed128 a_pack =
          load_weight_pack<K>(
              w_a_col_major, col0_a + col, swizzled_element_idx);
      const Bf16Packed128 b_pack =
          load_weight_pack<K>(
              w_b_col_major, col0_b + col, swizzled_element_idx);
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

template <int K,
          int N,
          int ColsPerBlock,
          int Threads,
          int SwizzleTileBlocks,
          bool StoreOutput = true>
__device__ inline void decode_gemv_cols_device(
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ w_col_major,
    __nv_bfloat16 *__restrict__ y,
    int physical_block_idx,
    float (&warp_sums)[ColsPerBlock][Threads / WARP_SIZE],
    float (&sums)[ColsPerBlock]) {
  static_assert((N % ColsPerBlock) == 0,
                "decode GEMV N must be divisible by columns per block");

  constexpr int blocks = N / ColsPerBlock;
  const int logical_block =
      swizzle_col_block<blocks, SwizzleTileBlocks>(physical_block_idx);
  const int col0 = logical_block * ColsPerBlock;

  dot_cols_reduce<K, ColsPerBlock, Threads>(
      x, w_col_major, col0, threadIdx.x, warp_sums, sums);

  if constexpr (StoreOutput) {
    if (threadIdx.x == 0) {
      store_cols<ColsPerBlock>(y + col0, sums);
    }
  }
}

template <int K,
          int N,
          int ColsPerBlock,
          int Threads,
          int SwizzleTileBlocks,
          bool StoreOutput = true>
__device__ inline void decode_gemv_cols_device(
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ w_col_major,
    __nv_bfloat16 *__restrict__ y,
    int physical_block_idx,
    float (&warp_sums)[ColsPerBlock][Threads / WARP_SIZE]) {
  float sums[ColsPerBlock] = {};
  decode_gemv_cols_device<K, N, ColsPerBlock, Threads, SwizzleTileBlocks,
                          StoreOutput>(
      x, w_col_major, y, physical_block_idx, warp_sums, sums);
}

}  // namespace gemma4_matmul_kernel_impl

namespace {

constexpr int kDefaultThreads = 512;
constexpr int kDefaultColsPerBlock = 8;
constexpr int kDefaultMinBlocksPerSm = 2;
constexpr int kInterleaveSwizzleBlocks = 16;
constexpr int kFfnDownThreads = 960;
constexpr int kFfnDownColsPerBlock = 8;
constexpr int kFfnDownMinBlocksPerSm = 1;
constexpr int kGlobalOThreads = 512;
constexpr int kGlobalOColsPerBlock = 8;
constexpr int kGlobalOMinBlocksPerSm = 1;
constexpr int kFinalLogitsThreads =
    gemma4_matmul_kernel_impl::kFinalLogitsDecodeThreads;
constexpr int kFinalLogitsColsPerBlock =
    gemma4_matmul_kernel_impl::kFinalLogitsDecodeColsPerBlock;
constexpr int kFinalLogitsMinBlocksPerSm =
    gemma4_matmul_kernel_impl::kFinalLogitsDecodeMinBlocksPerSm;

static_assert((kDefaultThreads % WARP_SIZE) == 0,
              "decode thread count must be a whole number of warps");

template <int K,
          int N,
          int ColsPerBlock,
          int Threads,
          int MinBlocksPerSM,
          int SwizzleTileBlocks>
__global__ __launch_bounds__(Threads, MinBlocksPerSM) void
gemma4_decode_gemv_cols_kernel(const __nv_bfloat16 *__restrict__ x,
                               const __nv_bfloat16 *__restrict__ w_col_major,
                               __nv_bfloat16 *__restrict__ y) {
  static_assert((N % ColsPerBlock) == 0,
                "decode GEMV N must be divisible by columns per block");

  constexpr int warps = Threads / WARP_SIZE;
  __shared__ float warp_sums[ColsPerBlock][warps];

  float sums[ColsPerBlock] = {};
  gemma4_matmul_kernel_impl::decode_gemv_cols_device<
      K, N, ColsPerBlock, Threads, SwizzleTileBlocks>(
      x, w_col_major, y, blockIdx.x, warp_sums, sums);
}

template <int ThreadblockM,
          int ThreadblockN,
          int ThreadblockK,
          int WarpM,
          int WarpN,
          int Stages>
// Launches one concrete CUTLASS BF16 Tensor Core GEMM configuration.
cudaError_t launch_prefill_cutlass_gemm(
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ w_col_major,
    __nv_bfloat16 *__restrict__ y,
    int rows,
    int k,
    int n,
    cudaStream_t stream) {
  using Element = cutlass::bfloat16_t;
  using Gemm = cutlass::gemm::device::Gemm<
      Element,
      cutlass::layout::RowMajor,
      Element,
      cutlass::layout::ColumnMajor,
      Element,
      cutlass::layout::RowMajor,
      float,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<ThreadblockM, ThreadblockN, ThreadblockK>,
      cutlass::gemm::GemmShape<WarpM, WarpN, ThreadblockK>,
      cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<Element, 8, float, float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
      Stages>;

  typename Gemm::Arguments args(
      {rows, n, k},
      {reinterpret_cast<const Element *>(x), k},
      {reinterpret_cast<const Element *>(w_col_major), k},
      {reinterpret_cast<Element *>(y), n},
      {reinterpret_cast<Element *>(y), n},
      {1.0f, 0.0f});

  Gemm gemm;
  cutlass::Status status = gemm.can_implement(args);
  if (status != cutlass::Status::kSuccess) {
    return cudaErrorInvalidValue;
  }
  status = gemm(args, nullptr, stream);
  if (status != cutlass::Status::kSuccess) {
    return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
}

// Launches the measured CUTLASS 64x64x32, 10-stage BF16 GEMM.
cudaError_t launch_prefill_cutlass_64x64_s10(
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ w_col_major,
    __nv_bfloat16 *__restrict__ y,
    int rows,
    int k,
    int n,
    cudaStream_t stream) {
  return launch_prefill_cutlass_gemm<64, 64, 32, 32, 32, 10>(
      x, w_col_major, y, rows, k, n, stream);
}

// Launches the measured CUTLASS 64x128x32, 6-stage BF16 GEMM.
cudaError_t launch_prefill_cutlass_64x128_s6(
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ w_col_major,
    __nv_bfloat16 *__restrict__ y,
    int rows,
    int k,
    int n,
    cudaStream_t stream) {
  return launch_prefill_cutlass_gemm<64, 128, 32, 32, 64, 6>(
      x, w_col_major, y, rows, k, n, stream);
}

// Launches the measured CUTLASS 128x128x32, 5-stage BF16 GEMM.
cudaError_t launch_prefill_cutlass_128x128_s5(
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ w_col_major,
    __nv_bfloat16 *__restrict__ y,
    int rows,
    int k,
    int n,
    cudaStream_t stream) {
  return launch_prefill_cutlass_gemm<128, 128, 32, 64, 64, 5>(
      x, w_col_major, y, rows, k, n, stream);
}

// Launches the measured CUTLASS 128x256x32, 3-stage BF16 GEMM.
cudaError_t launch_prefill_cutlass_128x256(
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ w_col_major,
    __nv_bfloat16 *__restrict__ y,
    int rows,
    int k,
    int n,
    cudaStream_t stream) {
  return launch_prefill_cutlass_gemm<128, 256, 32, 64, 64, 3>(
      x, w_col_major, y, rows, k, n, stream);
}

// Launches the measured CUTLASS 256x128x32, 3-stage BF16 GEMM.
cudaError_t launch_prefill_cutlass_256x128(
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ w_col_major,
    __nv_bfloat16 *__restrict__ y,
    int rows,
    int k,
    int n,
    cudaStream_t stream) {
  return launch_prefill_cutlass_gemm<256, 128, 32, 64, 64, 3>(
      x, w_col_major, y, rows, k, n, stream);
}

template <int K,
          int N,
          int ColsPerBlock,
          int Threads,
          int MinBlocksPerSM,
          int SwizzleTileBlocks>
cudaError_t launch_decode_gemv(const __nv_bfloat16 *__restrict__ x,
                               const __nv_bfloat16 *__restrict__ w_col_major,
                               __nv_bfloat16 *__restrict__ y,
                               cudaStream_t stream) {
  if (!x || !w_col_major || !y || !is_aligned_16(x) ||
      !is_aligned_16(w_col_major) || !is_aligned_16(y)) {
    return cudaErrorInvalidValue;
  }

  constexpr int blocks = N / ColsPerBlock;
  gemma4_decode_gemv_cols_kernel<K,
                                 N,
                                 ColsPerBlock,
                                 Threads,
                                 MinBlocksPerSM,
                                 SwizzleTileBlocks>
      <<<blocks, Threads, 0, stream>>>(x, w_col_major, y);
  return cudaGetLastError();
}

template <int SwizzleTileBlocks>
cudaError_t projection_decode_impl(
    Gemma4Projection projection,
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ w_col_major,
    __nv_bfloat16 *__restrict__ y,
    cudaStream_t stream) {
  switch (projection) {
  case GEMMA4_PROJECTION_FFN_GATE_UP:
    return launch_decode_gemv<GEMMA4_HIDDEN_SIZE, GEMMA4_PACKED_FFN_SIZE,
                              kDefaultColsPerBlock, kDefaultThreads,
                              kDefaultMinBlocksPerSm, SwizzleTileBlocks>(
        x, w_col_major, y, stream);
  case GEMMA4_PROJECTION_FFN_DOWN:
    return launch_decode_gemv<
        GEMMA4_INTERMEDIATE_SIZE, GEMMA4_HIDDEN_SIZE,
        kFfnDownColsPerBlock, kFfnDownThreads,
        kFfnDownMinBlocksPerSm, SwizzleTileBlocks>(
        x, w_col_major, y, stream);
  case GEMMA4_PROJECTION_SLIDING_QKV:
    return launch_decode_gemv<GEMMA4_HIDDEN_SIZE, GEMMA4_SLIDING_QKV_SIZE,
                              kDefaultColsPerBlock, kDefaultThreads,
                              kDefaultMinBlocksPerSm, SwizzleTileBlocks>(
        x, w_col_major, y, stream);
  case GEMMA4_PROJECTION_SLIDING_O:
    return launch_decode_gemv<GEMMA4_SLIDING_ATTENTION_OUT_SIZE,
                              GEMMA4_HIDDEN_SIZE, kDefaultColsPerBlock,
                              kDefaultThreads, kDefaultMinBlocksPerSm,
                              SwizzleTileBlocks>(x, w_col_major, y, stream);
  case GEMMA4_PROJECTION_GLOBAL_Q:
    return launch_decode_gemv<GEMMA4_HIDDEN_SIZE, GEMMA4_GLOBAL_Q_PROJ_SIZE,
                              kDefaultColsPerBlock, kDefaultThreads,
                              kDefaultMinBlocksPerSm, SwizzleTileBlocks>(
        x, w_col_major, y, stream);
  case GEMMA4_PROJECTION_GLOBAL_K:
    return launch_decode_gemv<GEMMA4_HIDDEN_SIZE, GEMMA4_GLOBAL_K_PROJ_SIZE,
                              kDefaultColsPerBlock, kDefaultThreads,
                              kDefaultMinBlocksPerSm, SwizzleTileBlocks>(
        x, w_col_major, y, stream);
  case GEMMA4_PROJECTION_GLOBAL_O:
    return launch_decode_gemv<
        GEMMA4_GLOBAL_ATTENTION_OUT_SIZE, GEMMA4_HIDDEN_SIZE,
        kGlobalOColsPerBlock, kGlobalOThreads,
        kGlobalOMinBlocksPerSm, SwizzleTileBlocks>(x, w_col_major, y, stream);
  case GEMMA4_PROJECTION_FINAL_LOGITS:
    return launch_decode_gemv<
        GEMMA4_HIDDEN_SIZE, GEMMA4_VOCAB_SIZE,
        kFinalLogitsColsPerBlock, kFinalLogitsThreads,
        kFinalLogitsMinBlocksPerSm, SwizzleTileBlocks>(
        x, w_col_major, y, stream);
  }
  return cudaErrorInvalidValue;
}

}  // namespace

cudaError_t gemma4_projection_decode(Gemma4Projection projection,
                                     const __nv_bfloat16 *__restrict__ x,
                                     const __nv_bfloat16 *__restrict__ w_col_major,
                                     __nv_bfloat16 *__restrict__ y,
                                     cudaStream_t stream) {
  return projection_decode_impl<1>(projection, x, w_col_major, y, stream);
}

cudaError_t gemma4_projection_decode_swizzled(
    Gemma4Projection projection,
    Gemma4DecodeSwizzle swizzle,
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ w_col_major,
    __nv_bfloat16 *__restrict__ y,
    cudaStream_t stream) {
  switch (swizzle) {
  case GEMMA4_DECODE_SWIZZLE_IDENTITY:
    return projection_decode_impl<1>(projection, x, w_col_major, y, stream);
  case GEMMA4_DECODE_SWIZZLE_INTERLEAVE_16:
    return projection_decode_impl<kInterleaveSwizzleBlocks>(
        projection, x, w_col_major, y, stream);
  }
  return cudaErrorInvalidValue;
}

// Runs the public generic BF16 prefill projection GEMM.
cudaError_t gemma4_prefill_gemm_bf16(
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ w_col_major,
    __nv_bfloat16 *__restrict__ y,
    int rows,
    int k,
    int n,
    cudaStream_t stream) {
  if (rows < 0 || k <= 0 || n <= 0 || x == nullptr ||
      w_col_major == nullptr || y == nullptr) {
    return cudaErrorInvalidValue;
  }
  if (rows == 0) {
    return cudaSuccess;
  }
  // These exact 12B prefill shapes were measured on RTX A6000 with CUDA events.
  if (k == GEMMA4_HIDDEN_SIZE && n == GEMMA4_GLOBAL_K_PROJ_SIZE) {
    if (rows <= 512) {
      return launch_prefill_cutlass_64x64_s10(
          x, w_col_major, y, rows, k, n, stream);
    }
    return launch_prefill_cutlass_64x128_s6(
        x, w_col_major, y, rows, k, n, stream);
  }
  if (k == GEMMA4_HIDDEN_SIZE && n == GEMMA4_SLIDING_KV_PROJ_SIZE) {
    if (rows <= 128) {
      return launch_prefill_cutlass_64x64_s10(
          x, w_col_major, y, rows, k, n, stream);
    }
    if (rows <= 256) {
      return launch_prefill_cutlass_64x128_s6(
          x, w_col_major, y, rows, k, n, stream);
    }
    if (rows <= 512) {
      return launch_prefill_cutlass_128x128_s5(
          x, w_col_major, y, rows, k, n, stream);
    }
    return launch_prefill_cutlass_128x256(
        x, w_col_major, y, rows, k, n, stream);
  }
  if (k == GEMMA4_HIDDEN_SIZE && n == GEMMA4_SLIDING_Q_PROJ_SIZE) {
    if (rows <= 64) {
      return launch_prefill_cutlass_64x64_s10(
          x, w_col_major, y, rows, k, n, stream);
    }
    if (rows <= 128) {
      return launch_prefill_cutlass_64x128_s6(
          x, w_col_major, y, rows, k, n, stream);
    }
    if (rows <= 256) {
      return launch_prefill_cutlass_128x128_s5(
          x, w_col_major, y, rows, k, n, stream);
    }
    if (rows <= 512) {
      return launch_prefill_cutlass_256x128(
          x, w_col_major, y, rows, k, n, stream);
    }
    return launch_prefill_cutlass_128x256(
        x, w_col_major, y, rows, k, n, stream);
  }
  if (k == GEMMA4_HIDDEN_SIZE && n == GEMMA4_GLOBAL_Q_PROJ_SIZE) {
    if (rows <= 64) {
      return launch_prefill_cutlass_64x128_s6(
          x, w_col_major, y, rows, k, n, stream);
    }
    if (rows <= 128) {
      return launch_prefill_cutlass_128x128_s5(
          x, w_col_major, y, rows, k, n, stream);
    }
    if (rows <= 256) {
      return launch_prefill_cutlass_256x128(
          x, w_col_major, y, rows, k, n, stream);
    }
    if (rows <= 512) {
      return launch_prefill_cutlass_128x256(
          x, w_col_major, y, rows, k, n, stream);
    }
    return launch_prefill_cutlass_128x128_s5(
        x, w_col_major, y, rows, k, n, stream);
  }
  if (k == GEMMA4_SLIDING_ATTENTION_OUT_SIZE && n == GEMMA4_HIDDEN_SIZE) {
    if (rows <= 64) {
      return launch_prefill_cutlass_64x64_s10(
          x, w_col_major, y, rows, k, n, stream);
    }
    if (rows <= 128) {
      return launch_prefill_cutlass_64x128_s6(
          x, w_col_major, y, rows, k, n, stream);
    }
    if (rows <= 256) {
      return launch_prefill_cutlass_128x128_s5(
          x, w_col_major, y, rows, k, n, stream);
    }
    if (rows <= 512) {
      return launch_prefill_cutlass_256x128(
          x, w_col_major, y, rows, k, n, stream);
    }
    return launch_prefill_cutlass_128x128_s5(
        x, w_col_major, y, rows, k, n, stream);
  }
  if (k == GEMMA4_GLOBAL_ATTENTION_OUT_SIZE && n == GEMMA4_HIDDEN_SIZE) {
    if (rows <= 64) {
      return launch_prefill_cutlass_64x64_s10(
          x, w_col_major, y, rows, k, n, stream);
    }
    if (rows <= 128) {
      return launch_prefill_cutlass_64x128_s6(
          x, w_col_major, y, rows, k, n, stream);
    }
    if (rows <= 256) {
      return launch_prefill_cutlass_128x128_s5(
          x, w_col_major, y, rows, k, n, stream);
    }
    if (rows <= 512) {
      return launch_prefill_cutlass_128x256(
          x, w_col_major, y, rows, k, n, stream);
    }
    return launch_prefill_cutlass_128x128_s5(
        x, w_col_major, y, rows, k, n, stream);
  }
  if (rows <= 128) {
    return launch_prefill_cutlass_gemm<64, 128, 64, 32, 64, 3>(
        x, w_col_major, y, rows, k, n, stream);
  }
  return launch_prefill_cutlass_gemm<128, 128, 64, 64, 64, 3>(
      x, w_col_major, y, rows, k, n, stream);
}
