#include "gemma4_ffn_decode.cuh"
#include "gemma4_cuda_utils.cuh"

#include <algorithm>

namespace {

constexpr int kFfnThreads = 1024;
constexpr int kFfnWarps = kFfnThreads / WARP_SIZE;
constexpr int kIntermediateTile = 256;
constexpr int kIntermediateTiles =
    GEMMA4_INTERMEDIATE_SIZE / kIntermediateTile;
constexpr int kColsPerThread =
    (GEMMA4_HIDDEN_SIZE + kFfnThreads - 1) / kFfnThreads;
static_assert((GEMMA4_INTERMEDIATE_SIZE % kIntermediateTile) == 0,
              "FFN intermediate width must divide the decode tile width");
static_assert((kFfnThreads % WARP_SIZE) == 0,
              "FFN decode thread count must be a whole number of warps");

__device__ inline float bf16_to_float(const floatX value) {
  return __bfloat162float(value);
}

__device__ inline float gelu_tanh(float x) {
  constexpr float kSqrtTwoOverPi = 0.7978845608028654f;
  constexpr float kGeluCubic = 0.044715f;
  const float x2 = x * x;
  const float inner = kSqrtTwoOverPi * (x + kGeluCubic * x * x2);
  return 0.5f * x * (1.0f + tanhf(inner));
}

__device__ inline float block_reduce_sum(float value,
                                         float *__restrict__ warp_sums) {
  const int lane = threadIdx.x & (WARP_SIZE - 1);
  const int warp = threadIdx.x / WARP_SIZE;

  value = warp_reduce_sum(value);
  if (lane == 0) {
    warp_sums[warp] = value;
  }
  __syncthreads();

  value = threadIdx.x < kFfnWarps ? warp_sums[lane] : 0.0f;
  if (warp == 0) {
    value = warp_reduce_sum(value);
  }
  return value;
}

__device__ inline void block_reduce_pair(float &a,
                                         float &b,
                                         float *__restrict__ a_warp_sums,
                                         float *__restrict__ b_warp_sums) {
  const int lane = threadIdx.x & (WARP_SIZE - 1);
  const int warp = threadIdx.x / WARP_SIZE;

  a = warp_reduce_sum(a);
  b = warp_reduce_sum(b);
  if (lane == 0) {
    a_warp_sums[warp] = a;
    b_warp_sums[warp] = b;
  }
  __syncthreads();

  a = threadIdx.x < kFfnWarps ? a_warp_sums[lane] : 0.0f;
  b = threadIdx.x < kFfnWarps ? b_warp_sums[lane] : 0.0f;
  if (warp == 0) {
    a = warp_reduce_sum(a);
    b = warp_reduce_sum(b);
  }
}

__device__ inline void wait_for_reduce_turn(const int *__restrict__ lock,
                                            int turn) {
  if (threadIdx.x == 0) {
    int state = -1;
    do {
      asm volatile("ld.global.acquire.gpu.b32 %0, [%1];\n"
                   : "=r"(state)
                   : "l"(lock));
    } while (state != turn);
  }
  __syncthreads();
}

__device__ inline void release_reduce_turn(int *__restrict__ lock) {
  __syncthreads();
  if (threadIdx.x == 0) {
    constexpr int one = 1;
    asm volatile("fence.acq_rel.gpu;\n");
    asm volatile("red.relaxed.gpu.global.add.s32 [%0], %1;\n"
                 :
                 : "l"(lock), "r"(one));
  }
}

__global__ __launch_bounds__(kFfnThreads, 1) void
gemma4_ffn_decode_fused_bf16_kernel(
    floatX *__restrict__ residual_out,
    floatX *__restrict__ normed_out,
    const floatX *__restrict__ x,
    const floatX *__restrict__ residual,
    const floatX *__restrict__ rms_weight,
    const floatX *__restrict__ w_gate_up_col_major,
    const floatX *__restrict__ w_down_row_major,
    Gemma4FfnDecodeScratch *__restrict__ scratch,
    float eps) {
  __shared__ floatX s_x[GEMMA4_HIDDEN_SIZE];
  __shared__ float s_reduce_warp_sums[2][kFfnWarps];
  __shared__ float s_act;
  __shared__ float s_scale;

  for (int i = threadIdx.x; i < GEMMA4_HIDDEN_SIZE; i += kFfnThreads) {
    s_x[i] = loadg(x + i);
  }
  __syncthreads();

  float partial[kColsPerThread] = {};
  const int intermediate_begin =
      static_cast<int>(blockIdx.x) * kIntermediateTile;

  for (int local_col = 0; local_col < kIntermediateTile; ++local_col) {
    const int intermediate_col = intermediate_begin + local_col;
    const floatX *gate_col =
        w_gate_up_col_major +
        static_cast<int64_t>(intermediate_col) * GEMMA4_HIDDEN_SIZE;
    const floatX *up_col =
        w_gate_up_col_major +
        static_cast<int64_t>(GEMMA4_INTERMEDIATE_SIZE + intermediate_col) *
            GEMMA4_HIDDEN_SIZE;

    float gate = 0.0f;
    float up = 0.0f;
    for (int k = threadIdx.x; k < GEMMA4_HIDDEN_SIZE; k += kFfnThreads) {
      const float xv = bf16_to_float(s_x[k]);
      gate = fmaf(xv, bf16_to_float(loadg(gate_col + k)), gate);
      up = fmaf(xv, bf16_to_float(loadg(up_col + k)), up);
    }

    block_reduce_pair(gate, up, s_reduce_warp_sums[0],
                      s_reduce_warp_sums[1]);
    if (threadIdx.x == 0) {
      s_act = gate * gelu_tanh(up);
    }
    __syncthreads();
    const float act = s_act;
    const floatX *down_row =
        w_down_row_major +
        static_cast<int64_t>(intermediate_col) * GEMMA4_HIDDEN_SIZE;

#pragma unroll
    for (int slot = 0; slot < kColsPerThread; ++slot) {
      const int col = threadIdx.x + slot * kFfnThreads;
      if (col < GEMMA4_HIDDEN_SIZE) {
        partial[slot] =
            fmaf(act, bf16_to_float(loadg(down_row + col)), partial[slot]);
      }
    }
    __syncthreads();
  }

  float merged[kColsPerThread] = {};
  const bool first = blockIdx.x == 0;
  const bool last = blockIdx.x == kIntermediateTiles - 1;
  wait_for_reduce_turn(&scratch->lock, static_cast<int>(blockIdx.x));

#pragma unroll
  for (int slot = 0; slot < kColsPerThread; ++slot) {
    const int col = threadIdx.x + slot * kFfnThreads;
    if (col < GEMMA4_HIDDEN_SIZE) {
      float value = partial[slot];
      if (!first) {
        value += scratch->accum[col];
      }
      merged[slot] = value;
      if (!last) {
        scratch->accum[col] = value;
      }
    }
  }

  if (!last) {
    release_reduce_turn(&scratch->lock);
    return;
  }

  float sum_sq = 0.0f;
#pragma unroll
  for (int slot = 0; slot < kColsPerThread; ++slot) {
    const int col = threadIdx.x + slot * kFfnThreads;
    if (col < GEMMA4_HIDDEN_SIZE) {
      const float value = merged[slot] + bf16_to_float(loadg(residual + col));
      sum_sq = fmaf(value, value, sum_sq);
      residual_out[col] = __float2bfloat16_rn(value);
    }
  }

  const float total = block_reduce_sum(sum_sq, s_reduce_warp_sums[0]);
  if (threadIdx.x == 0) {
    s_scale =
        rsqrtf(total / static_cast<float>(GEMMA4_HIDDEN_SIZE) + eps);
  }
  __syncthreads();

#pragma unroll
  for (int slot = 0; slot < kColsPerThread; ++slot) {
    const int col = threadIdx.x + slot * kFfnThreads;
    if (col < GEMMA4_HIDDEN_SIZE) {
      const float value = merged[slot] + bf16_to_float(loadg(residual + col));
      const float gamma = bf16_to_float(loadg(rms_weight + col));
      normed_out[col] = __float2bfloat16_rn(value * s_scale * gamma);
    }
  }
}

bool ffn_decode_args_valid(const floatX *residual_out,
                           const floatX *normed_out,
                           const floatX *x,
                           const floatX *residual,
                           const floatX *rms_weight,
                           const floatX *w_gate_up_col_major,
                           const floatX *w_down_row_major,
                           const Gemma4FfnDecodeScratch *scratch) {
  return residual_out != nullptr && normed_out != nullptr && x != nullptr &&
         residual != nullptr && rms_weight != nullptr &&
         w_gate_up_col_major != nullptr && w_down_row_major != nullptr &&
         scratch != nullptr && is_aligned_16(residual_out) &&
         is_aligned_16(normed_out) && is_aligned_16(x) &&
         is_aligned_16(residual) && is_aligned_16(rms_weight) &&
         is_aligned_16(w_gate_up_col_major) &&
         is_aligned_16(w_down_row_major) && is_aligned_16(scratch);
}

cudaError_t check_resident_reduction_supported() {
  int device = 0;
  cudaError_t status = cudaGetDevice(&device);
  if (status != cudaSuccess) {
    return status;
  }

  int sm_count = 0;
  status = cudaDeviceGetAttribute(
      &sm_count, cudaDevAttrMultiProcessorCount, device);
  if (status != cudaSuccess) {
    return status;
  }

  int active_blocks_per_sm = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks_per_sm, gemma4_ffn_decode_fused_bf16_kernel,
      kFfnThreads, 0);
  if (status != cudaSuccess) {
    return status;
  }

  if (sm_count * active_blocks_per_sm < kIntermediateTiles) {
    return cudaErrorNotSupported;
  }
  return cudaSuccess;
}

}  // namespace

cudaError_t gemma4_ffn_decode_configure_scratch_l2(
    Gemma4FfnDecodeScratch *scratch,
    cudaStream_t stream) {
  if (scratch == nullptr || !is_aligned_16(scratch)) {
    return cudaErrorInvalidValue;
  }

  int device = 0;
  cudaError_t status = cudaGetDevice(&device);
  if (status != cudaSuccess) {
    return status;
  }

  cudaDeviceProp prop = {};
  status = cudaGetDeviceProperties(&prop, device);
  if (status != cudaSuccess) {
    return status;
  }
  if (prop.persistingL2CacheMaxSize == 0 ||
      prop.accessPolicyMaxWindowSize == 0) {
    return cudaSuccess;
  }

  const size_t requested_set_aside =
      std::min(static_cast<size_t>(prop.l2CacheSize * 3 / 4),
               static_cast<size_t>(prop.persistingL2CacheMaxSize));
  status = cudaDeviceSetLimit(
      cudaLimitPersistingL2CacheSize, requested_set_aside);
  if (status != cudaSuccess) {
    return status;
  }

  cudaStreamAttrValue attr = {};
  attr.accessPolicyWindow.base_ptr = scratch;
  attr.accessPolicyWindow.num_bytes =
      std::min(sizeof(Gemma4FfnDecodeScratch),
               static_cast<size_t>(prop.accessPolicyMaxWindowSize));
  attr.accessPolicyWindow.hitRatio = 1.0;
  attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
  attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
  return cudaStreamSetAttribute(
      stream, cudaStreamAttributeAccessPolicyWindow, &attr);
}

cudaError_t gemma4_ffn_decode_fused_bf16(
    floatX *__restrict__ residual_out,
    floatX *__restrict__ normed_out,
    const floatX *__restrict__ x,
    const floatX *__restrict__ residual,
    const floatX *__restrict__ rms_weight,
    const floatX *__restrict__ w_gate_up_col_major,
    const floatX *__restrict__ w_down_row_major,
    Gemma4FfnDecodeScratch *__restrict__ scratch,
    float eps,
    cudaStream_t stream) {
  if (!ffn_decode_args_valid(residual_out, normed_out, x, residual,
                             rms_weight, w_gate_up_col_major,
                             w_down_row_major, scratch)) {
    return cudaErrorInvalidValue;
  }

  cudaError_t status = check_resident_reduction_supported();
  if (status != cudaSuccess) {
    return status;
  }

  status = cudaMemsetAsync(scratch, 0, sizeof(Gemma4FfnDecodeScratch), stream);
  if (status != cudaSuccess) {
    return status;
  }

  gemma4_ffn_decode_fused_bf16_kernel<<<
      kIntermediateTiles, kFfnThreads, 0, stream>>>(
      residual_out, normed_out, x, residual, rms_weight,
      w_gate_up_col_major, w_down_row_major, scratch, eps);
  return cudaGetLastError();
}
