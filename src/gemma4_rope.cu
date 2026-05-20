#include "gemma4_rope.cuh"
#include "gemma4_cuda_utils.cuh"
#include "gemma4.h"

#include <stdint.h>

namespace {

constexpr int kRopeThreads = 128;
constexpr int kGemma4GlobalRotaryDim = GEMMA4_GLOBAL_HEAD_DIM / 4;

__device__ __forceinline__ void rotate_head_bf16(floatX *head,
                                                 const float *cos_row,
                                                 const float *sin_row,
                                                 int rotary_half) {
  for (int i = threadIdx.x; i < rotary_half; i += blockDim.x) {
    float x1 = __bfloat162float(head[i]);
    float x2 = __bfloat162float(head[rotary_half + i]);
    float c = cos_row[i];
    float s = sin_row[i];
    head[i] = __float2bfloat16_rn(fmaf(-x2, s, x1 * c));
    head[rotary_half + i] = __float2bfloat16_rn(fmaf(x1, s, x2 * c));
  }
}

__global__ void gemma4_rope_bf16_kernel(floatX *q,
                                        int64_t q_row_stride,
                                        floatX *k,
                                        int64_t k_row_stride,
                                        const float *cos,
                                        int64_t cos_row_stride,
                                        const float *sin,
                                        int64_t sin_row_stride,
                                        int seq_len,
                                        int cos_batch_size,
                                        int q_heads,
                                        int kv_heads,
                                        int head_dim,
                                        int rotary_half) {
  int row = blockIdx.x;
  int head = blockIdx.y;
  int batch = row / seq_len;
  int seq = row - batch * seq_len;
  int table_row = cos_batch_size == 1 ? seq : batch * seq_len + seq;

  const float *cos_row = cos + static_cast<int64_t>(table_row) * cos_row_stride;
  const float *sin_row = sin + static_cast<int64_t>(table_row) * sin_row_stride;

  if (head < q_heads) {
    floatX *q_head = q + static_cast<int64_t>(row) * q_row_stride +
                     static_cast<int64_t>(head) * head_dim;
    rotate_head_bf16(q_head, cos_row, sin_row, rotary_half);
  }
  if (head < kv_heads) {
    floatX *k_head = k + static_cast<int64_t>(row) * k_row_stride +
                     static_cast<int64_t>(head) * head_dim;
    rotate_head_bf16(k_head, cos_row, sin_row, rotary_half);
  }
}

__global__ void gemma4_rope_forward_bf16_kernel(floatX *q,
                                                floatX *k,
                                                const float *cos,
                                                int64_t cos_row_stride,
                                                const float *sin,
                                                int64_t sin_row_stride,
                                                int seq_len,
                                                int cos_batch_size,
                                                int q_heads,
                                                int kv_heads,
                                                int head_dim,
                                                int rotary_half) {
  int row = blockIdx.x;
  int head = blockIdx.y;
  int batch = row / seq_len;
  int seq = row - batch * seq_len;
  int table_row = cos_batch_size == 1 ? seq : batch * seq_len + seq;

  const float *cos_row = cos + static_cast<int64_t>(table_row) * cos_row_stride;
  const float *sin_row = sin + static_cast<int64_t>(table_row) * sin_row_stride;

  if (head < q_heads) {
    floatX *q_head =
        q + (static_cast<int64_t>(batch) * q_heads + head) * seq_len *
                head_dim +
        static_cast<int64_t>(seq) * head_dim;
    rotate_head_bf16(q_head, cos_row, sin_row, rotary_half);
  }
  if (head < kv_heads) {
    floatX *k_head =
        k + (static_cast<int64_t>(batch) * kv_heads + head) * seq_len *
                head_dim +
        static_cast<int64_t>(seq) * head_dim;
    rotate_head_bf16(k_head, cos_row, sin_row, rotary_half);
  }
}

bool rope_args_valid(const floatX *q,
                     int64_t q_row_stride,
                     const floatX *k,
                     int64_t k_row_stride,
                     const float *cos,
                     int64_t cos_row_stride,
                     const float *sin,
                     int64_t sin_row_stride,
                     int seq_len,
                     int batch_size,
                     int cos_batch_size,
                     int q_heads,
                     int kv_heads,
                     int head_dim,
                     int rotary_dim) {
  if (seq_len < 0 || batch_size < 0 || q_heads < 0 || kv_heads < 0 ||
      head_dim <= 0 || rotary_dim <= 0) {
    return false;
  }
  if (seq_len == 0 || batch_size == 0) {
    return true;
  }
  if (q == nullptr || k == nullptr || cos == nullptr || sin == nullptr) {
    return false;
  }
  if (cos_batch_size != 1 && cos_batch_size != batch_size) {
    return false;
  }
  if ((rotary_dim % 2) != 0 || rotary_dim > head_dim) {
    return false;
  }
  int rotary_half = rotary_dim / 2;
  if (q_row_stride < static_cast<int64_t>(q_heads) * head_dim ||
      k_row_stride < static_cast<int64_t>(kv_heads) * head_dim ||
      cos_row_stride < rotary_half || sin_row_stride < rotary_half) {
    return false;
  }
  return is_aligned_16(q) && is_aligned_16(k);
}

bool rope_forward_args_valid(const floatX *q,
                             const floatX *k,
                             const float *cos,
                             const float *sin,
                             int seq_len,
                             int batch_size,
                             int cos_batch_size,
                             int q_heads,
                             int kv_heads,
                             int head_dim,
                             int rotary_dim) {
  if (seq_len < 0 || batch_size < 0 || q_heads < 0 || kv_heads < 0 ||
      head_dim <= 0 || rotary_dim <= 0) {
    return false;
  }
  if (seq_len == 0 || batch_size == 0) {
    return true;
  }
  if (q == nullptr || k == nullptr || cos == nullptr || sin == nullptr) {
    return false;
  }
  if (cos_batch_size != 1 && cos_batch_size != batch_size) {
    return false;
  }
  if ((rotary_dim % 2) != 0 || rotary_dim > head_dim) {
    return false;
  }
  return is_aligned_16(q) && is_aligned_16(k);
}

}  // namespace

cudaError_t gemma4_rope_bf16(floatX *q,
                             int64_t q_row_stride,
                             floatX *k,
                             int64_t k_row_stride,
                             const float *cos,
                             int64_t cos_row_stride,
                             const float *sin,
                             int64_t sin_row_stride,
                             int seq_len,
                             int batch_size,
                             int cos_batch_size,
                             int q_heads,
                             int kv_heads,
                             int head_dim,
                             int rotary_dim,
                             cudaStream_t stream) {
  if (!rope_args_valid(q, q_row_stride, k, k_row_stride, cos, cos_row_stride,
                       sin, sin_row_stride, seq_len, batch_size,
                       cos_batch_size, q_heads, kv_heads, head_dim,
                       rotary_dim)) {
    return cudaErrorInvalidValue;
  }
  if (seq_len == 0 || batch_size == 0) {
    return cudaSuccess;
  }

  int rows = batch_size * seq_len;
  int heads = q_heads > kv_heads ? q_heads : kv_heads;
  if (heads == 0) {
    return cudaSuccess;
  }

  dim3 grid(rows, heads);
  gemma4_rope_bf16_kernel<<<grid, kRopeThreads, 0, stream>>>(
      q, q_row_stride, k, k_row_stride, cos, cos_row_stride, sin,
      sin_row_stride, seq_len, cos_batch_size, q_heads, kv_heads, head_dim,
      rotary_dim / 2);
  return cudaGetLastError();
}

cudaError_t gemma4_rope_forward_bf16(floatX *q,
                                     floatX *k,
                                     const float *cos,
                                     const float *sin,
                                     int seq_len,
                                     int batch_size,
                                     int cos_batch_size,
                                     int q_heads,
                                     int kv_heads,
                                     int head_dim,
                                     int rotary_dim,
                                     cudaStream_t stream) {
  if (!rope_forward_args_valid(q, k, cos, sin, seq_len, batch_size,
                               cos_batch_size, q_heads, kv_heads, head_dim,
                               rotary_dim)) {
    return cudaErrorInvalidValue;
  }
  if (seq_len == 0 || batch_size == 0) {
    return cudaSuccess;
  }

  int rows = batch_size * seq_len;
  int heads = q_heads > kv_heads ? q_heads : kv_heads;
  if (heads == 0) {
    return cudaSuccess;
  }

  dim3 grid(rows, heads);
  int rotary_half = rotary_dim / 2;
  gemma4_rope_forward_bf16_kernel<<<grid, kRopeThreads, 0, stream>>>(
      q, k, cos, head_dim, sin, head_dim, seq_len, cos_batch_size,
      q_heads, kv_heads, head_dim, rotary_half);
  return cudaGetLastError();
}

cudaError_t gemma4_sliding_rope_bf16(floatX *q,
                                     floatX *k,
                                     const float *cos,
                                     const float *sin,
                                     int seq_len,
                                     int batch_size,
                                     int cos_batch_size,
                                     cudaStream_t stream) {
  return gemma4_rope_bf16(
      q, GEMMA4_SLIDING_Q_PROJ_SIZE, k, GEMMA4_SLIDING_KV_PROJ_SIZE, cos,
      GEMMA4_SLIDING_HEAD_DIM, sin, GEMMA4_SLIDING_HEAD_DIM, seq_len,
      batch_size, cos_batch_size, GEMMA4_NUM_QUERY_HEADS,
      GEMMA4_SLIDING_KV_HEADS, GEMMA4_SLIDING_HEAD_DIM,
      GEMMA4_SLIDING_HEAD_DIM, stream);
}

cudaError_t gemma4_sliding_rope_forward_bf16(floatX *q,
                                             floatX *k,
                                             const float *cos,
                                             const float *sin,
                                             int seq_len,
                                             int batch_size,
                                             int cos_batch_size,
                                             cudaStream_t stream) {
  return gemma4_rope_forward_bf16(
      q, k, cos, sin, seq_len, batch_size, cos_batch_size,
      GEMMA4_NUM_QUERY_HEADS, GEMMA4_SLIDING_KV_HEADS,
      GEMMA4_SLIDING_HEAD_DIM, GEMMA4_SLIDING_HEAD_DIM, stream);
}

cudaError_t gemma4_global_rope_bf16(floatX *q,
                                    floatX *k,
                                    const float *cos,
                                    const float *sin,
                                    int seq_len,
                                    int batch_size,
                                    int cos_batch_size,
                                    cudaStream_t stream) {
  return gemma4_rope_bf16(
      q, GEMMA4_GLOBAL_Q_PROJ_SIZE, k, GEMMA4_GLOBAL_K_PROJ_SIZE, cos,
      GEMMA4_GLOBAL_HEAD_DIM, sin, GEMMA4_GLOBAL_HEAD_DIM, seq_len, batch_size,
      cos_batch_size, GEMMA4_NUM_QUERY_HEADS, GEMMA4_GLOBAL_KV_HEADS,
      GEMMA4_GLOBAL_HEAD_DIM, kGemma4GlobalRotaryDim, stream);
}

cudaError_t gemma4_global_rope_forward_bf16(floatX *q,
                                            floatX *k,
                                            const float *cos,
                                            const float *sin,
                                            int seq_len,
                                            int batch_size,
                                            int cos_batch_size,
                                            cudaStream_t stream) {
  return gemma4_rope_forward_bf16(
      q, k, cos, sin, seq_len, batch_size, cos_batch_size,
      GEMMA4_NUM_QUERY_HEADS, GEMMA4_GLOBAL_KV_HEADS, GEMMA4_GLOBAL_HEAD_DIM,
      kGemma4GlobalRotaryDim, stream);
}
