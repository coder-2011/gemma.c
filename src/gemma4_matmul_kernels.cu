#include "gemma4_matmul_kernels.cuh"
#include "gemma4.h"

// Concrete Gemma projection kernels and host dispatch.

#include "gemma4_cuda_utils.cuh"

#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/numeric_types.h>


namespace gemma4_matmul_kernel_impl {

constexpr int kFfnGateUpTileCols = 2;
constexpr int kFfnGateUpThreads = GEMMA4_HIDDEN_SIZE / kBf16Packed128Elements;
constexpr int kFfnGateUpWarps = kFfnGateUpThreads / 32;
static_assert((kFfnGateUpThreads % 32) == 0,
              "FFN gate/up device helper requires whole warps");

template <int K, int ColsPerBlock, int Threads>
__device__ inline void dot_cols(
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ w_col_major, int col0, int thread_idx,
    float (&sums)[ColsPerBlock]) {
  static_assert((Threads % 32) == 0,
                "decode thread count must be a whole number of warps");

  constexpr int packs_per_col = K / kBf16Packed128Elements;

#if GEMMA4_DECODE_GEMV_BUFFER_STAGES <= 1
#pragma unroll
  for (int pack_idx = thread_idx; pack_idx < packs_per_col; pack_idx += Threads) {
    const int element_idx = pack_idx * kBf16Packed128Elements;
    const Bf16Packed128 x_pack =
        Bf16Packed128{*reinterpret_cast<const int4 *>(x + element_idx)};
#pragma unroll
    for (int col = 0; col < ColsPerBlock; ++col) {
      const int weight_idx = (col0 + col) * K + element_idx;
      const Bf16Packed128 w_pack = Bf16Packed128{
          *reinterpret_cast<const int4 *>(w_col_major + weight_idx)};
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
      const int element_idx = stage_pack_idx * kBf16Packed128Elements;
      x_stage[stage] =
          Bf16Packed128{*reinterpret_cast<const int4 *>(x + element_idx)};
#pragma unroll
      for (int col = 0; col < ColsPerBlock; ++col) {
        const int weight_idx = (col0 + col) * K + element_idx;
        w_stage[stage][col] = Bf16Packed128{
            *reinterpret_cast<const int4 *>(w_col_major + weight_idx)};
      }
    }
  }

  for (int iter = 0; pack_idx < packs_per_col; ++iter, pack_idx += Threads) {
    const int stage = iter % kStages;
    const Bf16Packed128 x_pack = x_stage[stage];
    Bf16Packed128 w_pack[ColsPerBlock];
#pragma unroll
    for (int col = 0; col < ColsPerBlock; ++col) {
      w_pack[col] = w_stage[stage][col];
    }

    const int next_pack_idx = pack_idx + kStages * Threads;
    if (next_pack_idx < packs_per_col) {
      const int element_idx = next_pack_idx * kBf16Packed128Elements;
      x_stage[stage] =
          Bf16Packed128{*reinterpret_cast<const int4 *>(x + element_idx)};
#pragma unroll
      for (int col = 0; col < ColsPerBlock; ++col) {
        const int weight_idx = (col0 + col) * K + element_idx;
        w_stage[stage][col] = Bf16Packed128{
            *reinterpret_cast<const int4 *>(w_col_major + weight_idx)};
      }
    }

#pragma unroll
    for (int col = 0; col < ColsPerBlock; ++col) {
      gemma4_bf16_pack_accumulate_dot(x_pack, w_pack[col], sums[col]);
    }
  }
#endif
}

__device__ inline int shared_pack_index(int chunk) {
  constexpr int kSwizzleChunks = 8;  // 8 x 128-bit chunks = one 128-byte row.
  const unsigned u = static_cast<unsigned>(chunk);
  const unsigned col = u & (kSwizzleChunks - 1);
  const unsigned row = (u >> 3) & (kSwizzleChunks - 1);
  return static_cast<int>((u & ~(kSwizzleChunks - 1u)) | (col ^ row));
}

// Computes one gate/up column pair with warp-local reduction only. The caller
// assigns one interleaved gate/up pair per warp, so no block barrier is needed
// between the dot loop and the reduction; consecutive lanes read consecutive
// (swizzle-permuted) 16-byte packs, which stays 128-byte coalesced because the
// swizzle permutes packs only inside each 8-pack block.
extern "C" __device__ void gemma4_ffn_gate_up_warp_col_bf16_device(
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ w_interleaved_row_major,
    int col,
    int lane,
    float *__restrict__ gate,
    float *__restrict__ up) {
  constexpr int kWarpThreads = 32;
  constexpr int packs_per_col =
      GEMMA4_HIDDEN_SIZE / kBf16Packed128Elements;
  static_assert((packs_per_col % kWarpThreads) == 0,
                "warp-col gate/up requires whole warp iterations");

  const __nv_bfloat16 *gate_row =
      w_interleaved_row_major +
      static_cast<int64_t>(2 * col) * GEMMA4_HIDDEN_SIZE;
  const __nv_bfloat16 *up_row = gate_row + GEMMA4_HIDDEN_SIZE;

  float gate_sum = 0.0f;
  float up_sum = 0.0f;
#pragma unroll
  for (int iter = 0; iter < packs_per_col / kWarpThreads; ++iter) {
    const int pack_idx = lane + iter * kWarpThreads;
    const int x_col = pack_idx * kBf16Packed128Elements;
    const int weight_col =
        shared_pack_index(pack_idx) * kBf16Packed128Elements;
    const Bf16Packed128 x_pack =
        Bf16Packed128{*reinterpret_cast<const int4 *>(x + x_col)};
    const Bf16Packed128 gate_pack =
        Bf16Packed128{*reinterpret_cast<const int4 *>(gate_row + weight_col)};
    const Bf16Packed128 up_pack =
        Bf16Packed128{*reinterpret_cast<const int4 *>(up_row + weight_col)};
    gemma4_bf16_pack_accumulate_dot(x_pack, gate_pack, gate_sum);
    gemma4_bf16_pack_accumulate_dot(x_pack, up_pack, up_sum);
  }

  *gate = warp_reduce_sum(gate_sum);
  *up = warp_reduce_sum(up_sum);
}

// Computes one Gemma 4 FFN gate/up decode tile for a caller-owned CTA.
extern "C" __device__ void gemma4_ffn_gate_up_tile_bf16_device(
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ w_interleaved_row_major,
    int col0,
    int thread_idx,
    float *__restrict__ warp_sums,
    float *__restrict__ gate,
    float *__restrict__ up) {
  constexpr int packs_per_col =
      GEMMA4_HIDDEN_SIZE / kBf16Packed128Elements;

  for (int pack_idx = thread_idx; pack_idx < packs_per_col;
       pack_idx += kFfnGateUpThreads) {
    const int x_col = pack_idx * kBf16Packed128Elements;
    const int weight_col =
        shared_pack_index(pack_idx) * kBf16Packed128Elements;
    const Bf16Packed128 x_pack =
        Bf16Packed128{*reinterpret_cast<const int4 *>(x + x_col)};
    const __nv_bfloat16 *gate_ptr =
        w_interleaved_row_major +
        static_cast<int64_t>(2 * col0) * GEMMA4_HIDDEN_SIZE + weight_col;
    const __nv_bfloat16 *up_ptr = gate_ptr + GEMMA4_HIDDEN_SIZE;

#pragma unroll
    for (int t = 0; t < kFfnGateUpTileCols; ++t) {
      const int64_t row_offset =
          static_cast<int64_t>(2 * t) * GEMMA4_HIDDEN_SIZE;
      const Bf16Packed128 gate_pack =
          Bf16Packed128{
              *reinterpret_cast<const int4 *>(gate_ptr + row_offset)};
      const Bf16Packed128 up_pack =
          Bf16Packed128{
              *reinterpret_cast<const int4 *>(up_ptr + row_offset)};
      gemma4_bf16_pack_accumulate_dot(x_pack, gate_pack, gate[t]);
      gemma4_bf16_pack_accumulate_dot(x_pack, up_pack, up[t]);
    }
  }

  const int lane = thread_idx & (warpSize - 1);
  const int warp = thread_idx / warpSize;

#pragma unroll
  for (int t = 0; t < kFfnGateUpTileCols; ++t) {
    gate[t] = warp_reduce_sum(gate[t]);
    up[t] = warp_reduce_sum(up[t]);
    if (lane == 0 && warp < kFfnGateUpWarps) {
      warp_sums[t * kFfnGateUpWarps + warp] = gate[t];
      warp_sums[(kFfnGateUpTileCols + t) * kFfnGateUpWarps + warp] = up[t];
    }
  }
  __syncthreads();

#pragma unroll
  for (int t = 0; t < kFfnGateUpTileCols; ++t) {
    gate[t] = thread_idx < kFfnGateUpWarps
                  ? warp_sums[t * kFfnGateUpWarps + lane]
                  : 0.0f;
    up[t] = thread_idx < kFfnGateUpWarps
                ? warp_sums[(kFfnGateUpTileCols + t) * kFfnGateUpWarps + lane]
                : 0.0f;
    if (warp == 0) {
      gate[t] = warp_reduce_sum(gate[t]);
      up[t] = warp_reduce_sum(up[t]);
    }
  }
}

template <int ColsPerBlock, int Threads>
__device__ inline void reduce_cols(
    int thread_idx,
    float (&warp_sums)[ColsPerBlock][Threads / 32],
    float (&sums)[ColsPerBlock]) {
  constexpr int warps = Threads / 32;
  const int lane = thread_idx & (warpSize - 1);
  const int warp = thread_idx / warpSize;

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
    float (&warp_sums)[ColsPerBlock][Threads / 32],
    float (&sums)[ColsPerBlock]) {
  constexpr int blocks = N / ColsPerBlock;
  int logical_block = physical_block_idx;
  if constexpr (SwizzleTileBlocks > 1) {
    constexpr int tiles = blocks / SwizzleTileBlocks;
    logical_block = (physical_block_idx % SwizzleTileBlocks) * tiles +
                    physical_block_idx / SwizzleTileBlocks;
  }
  const int col0 = logical_block * ColsPerBlock;

  dot_cols<K, ColsPerBlock, Threads>(
      x, w_col_major, col0, threadIdx.x, sums);
  reduce_cols<ColsPerBlock, Threads>(threadIdx.x, warp_sums, sums);

  if (threadIdx.x == 0) {
    if constexpr (ColsPerBlock == kBf16Packed128Elements) {
      Bf16Packed128 out;
#pragma unroll
      for (int col = 0; col < ColsPerBlock; ++col) {
        out[col] = __float2bfloat16_rn(sums[col]);
      }
      *reinterpret_cast<int4 *>(y + col0) = out.bits();
    } else {
#pragma unroll
      for (int col = 0; col < ColsPerBlock; ++col) {
        y[col0 + col] = __float2bfloat16_rn(sums[col]);
      }
    }
  }
}

}  // namespace gemma4_matmul_kernel_impl

namespace {

constexpr int kDefaultThreads = 512;
constexpr int kDefaultColsPerBlock = 8;
constexpr int kDefaultMinBlocksPerSm = 2;
constexpr int kFfnDownThreads = 960;
constexpr int kFfnDownColsPerBlock = 8;
constexpr int kFfnDownMinBlocksPerSm = 1;
constexpr int kGlobalOThreads = 512;
constexpr int kGlobalOColsPerBlock = 8;
constexpr int kGlobalOMinBlocksPerSm = 1;
constexpr int kFinalLogitsThreads = 1024;
constexpr int kFinalLogitsColsPerBlock = 8;
constexpr int kFinalLogitsMinBlocksPerSm = 1;

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
  constexpr int warps = Threads / 32;
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

  const typename Gemm::Arguments args(
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
  case GEMMA4_PROJECTION_SLIDING_Q:
    return launch_decode_gemv<GEMMA4_HIDDEN_SIZE, GEMMA4_SLIDING_Q_PROJ_SIZE,
                              kDefaultColsPerBlock, kDefaultThreads,
                              kDefaultMinBlocksPerSm, SwizzleTileBlocks>(
        x, w_col_major, y, stream);
  case GEMMA4_PROJECTION_SLIDING_KV:
    return launch_decode_gemv<GEMMA4_HIDDEN_SIZE, GEMMA4_SLIDING_KV_PROJ_SIZE,
                              kDefaultColsPerBlock, kDefaultThreads,
                              kDefaultMinBlocksPerSm, SwizzleTileBlocks>(
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
  case GEMMA4_PROJECTION_GLOBAL_QK:
    return launch_decode_gemv<GEMMA4_HIDDEN_SIZE, GEMMA4_GLOBAL_QK_PROJ_SIZE,
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
