#include "gemma4_rope.cuh"
#include "gemma4_cuda_utils.cuh"

#include <stdint.h>

#ifndef GEMMA4_ROPE_HEAD_FAST_GRID
#define GEMMA4_ROPE_HEAD_FAST_GRID 0
#endif

namespace {

constexpr int kRopeThreads = WARP_SIZE;

__global__ __launch_bounds__(kRopeThreads) void
gemma4_rope_bf16_kernel(floatX *__restrict__ q,
                        int64_t q_row_stride,
                        floatX *__restrict__ k,
                        int64_t k_row_stride,
                        const float *__restrict__ cos,
                        int64_t cos_row_stride,
                        const float *__restrict__ sin,
                        int64_t sin_row_stride,
                        int seq_len,
                        int cos_batch_size,
                        int q_heads,
                        int kv_heads,
                        int head_dim,
                        int rotary_half) {
#if GEMMA4_ROPE_HEAD_FAST_GRID
  int head = blockIdx.x;
  int row = blockIdx.y;
#else
  int row = blockIdx.x;
  int head = blockIdx.y;
#endif
  int batch = row / seq_len;
  int seq = row - batch * seq_len;
  int table_row = cos_batch_size == 1 ? seq : batch * seq_len + seq;
  int64_t table_offset = table_row;

  const float *cos_row = cos + table_offset * cos_row_stride;
  const float *sin_row = sin + table_offset * sin_row_stride;

  if (head < q_heads) {
    int64_t q_offset = static_cast<int64_t>(row) * q_row_stride + head * head_dim;
    floatX *q_head = q + q_offset;
    gemma4_rope::rotate_head_bf16(q_head, cos_row, sin_row, rotary_half,
                                  threadIdx.x, blockDim.x);
  }
  if (head < kv_heads) {
    int64_t k_offset = static_cast<int64_t>(row) * k_row_stride + head * head_dim;
    floatX *k_head = k + k_offset;
    gemma4_rope::rotate_head_bf16(k_head, cos_row, sin_row, rotary_half,
                                  threadIdx.x, blockDim.x);
  }
}

__global__ __launch_bounds__(kRopeThreads) void
gemma4_rope_forward_bf16_kernel(floatX *__restrict__ q,
                                floatX *__restrict__ k,
                                const float *__restrict__ cos,
                                int64_t cos_row_stride,
                                const float *__restrict__ sin,
                                int64_t sin_row_stride,
                                int seq_len,
                                int cos_batch_size,
                                int q_heads,
                                int kv_heads,
                                int head_dim,
                                int rotary_half) {
#if GEMMA4_ROPE_HEAD_FAST_GRID
  int head = blockIdx.x;
  int row = blockIdx.y;
#else
  int row = blockIdx.x;
  int head = blockIdx.y;
#endif
  int batch = row / seq_len;
  int seq = row - batch * seq_len;
  int table_row = cos_batch_size == 1 ? seq : batch * seq_len + seq;
  int64_t table_offset = table_row;

  const float *cos_row = cos + table_offset * cos_row_stride;
  const float *sin_row = sin + table_offset * sin_row_stride;

  if (head < q_heads) {
    int64_t batch_head = static_cast<int64_t>(batch) * q_heads + head;
    int64_t q_offset = (batch_head * seq_len + seq) * head_dim;
    floatX *q_head = q + q_offset;
    gemma4_rope::rotate_head_bf16(q_head, cos_row, sin_row, rotary_half,
                                  threadIdx.x, blockDim.x);
  }
  if (head < kv_heads) {
    int64_t batch_head = static_cast<int64_t>(batch) * kv_heads + head;
    int64_t k_offset = (batch_head * seq_len + seq) * head_dim;
    floatX *k_head = k + k_offset;
    gemma4_rope::rotate_head_bf16(k_head, cos_row, sin_row, rotary_half,
                                  threadIdx.x, blockDim.x);
  }
}

}  // namespace

cudaError_t gemma4_rope_bf16(floatX *__restrict__ q,
                             int64_t q_row_stride,
                             floatX *__restrict__ k,
                             int64_t k_row_stride,
                             const float *__restrict__ cos,
                             int64_t cos_row_stride,
                             const float *__restrict__ sin,
                             int64_t sin_row_stride,
                             int seq_len,
                             int batch_size,
                             int cos_batch_size,
                             int q_heads,
                             int kv_heads,
                             int head_dim,
                             int rotary_dim,
                             cudaStream_t stream) {
  if (seq_len < 0 || batch_size < 0 || q_heads < 0 || kv_heads < 0 ||
      head_dim <= 0 || rotary_dim <= 0 || rotary_dim > head_dim ||
      (rotary_dim % 2) != 0 ||
      (cos_batch_size != 1 && cos_batch_size != batch_size)) {
    return cudaErrorInvalidValue;
  }
  if (seq_len == 0 || batch_size == 0 || (q_heads == 0 && kv_heads == 0)) {
    return cudaSuccess;
  }
  if (q == nullptr || k == nullptr || cos == nullptr || sin == nullptr) {
    return cudaErrorInvalidValue;
  }

  int rotary_half = rotary_dim / 2;
  if (q_row_stride < static_cast<int64_t>(q_heads) * head_dim ||
      k_row_stride < static_cast<int64_t>(kv_heads) * head_dim ||
      cos_row_stride < rotary_half || sin_row_stride < rotary_half) {
    return cudaErrorInvalidValue;
  }
  int rows = batch_size * seq_len;
  int heads = q_heads > kv_heads ? q_heads : kv_heads;

#if GEMMA4_ROPE_HEAD_FAST_GRID
  dim3 grid(heads, rows);
#else
  dim3 grid(rows, heads);
#endif
  gemma4_rope_bf16_kernel<<<grid, kRopeThreads, 0, stream>>>(
      q, q_row_stride, k, k_row_stride, cos, cos_row_stride, sin,
      sin_row_stride, seq_len, cos_batch_size, q_heads, kv_heads, head_dim,
      rotary_half);
  return cudaGetLastError();
}

cudaError_t gemma4_rope_forward_bf16(floatX *__restrict__ q,
                                     floatX *__restrict__ k,
                                     const float *__restrict__ cos,
                                     const float *__restrict__ sin,
                                     int seq_len,
                                     int batch_size,
                                     int cos_batch_size,
                                     int q_heads,
                                     int kv_heads,
                                     int head_dim,
                                     int rotary_dim,
                                     cudaStream_t stream) {
  if (seq_len < 0 || batch_size < 0 || q_heads < 0 || kv_heads < 0 ||
      head_dim <= 0 || rotary_dim <= 0 || rotary_dim > head_dim ||
      (rotary_dim % 2) != 0 ||
      (cos_batch_size != 1 && cos_batch_size != batch_size)) {
    return cudaErrorInvalidValue;
  }
  if (seq_len == 0 || batch_size == 0 || (q_heads == 0 && kv_heads == 0)) {
    return cudaSuccess;
  }
  if (q == nullptr || k == nullptr || cos == nullptr || sin == nullptr) {
    return cudaErrorInvalidValue;
  }

  int rows = batch_size * seq_len;
  int heads = q_heads > kv_heads ? q_heads : kv_heads;

#if GEMMA4_ROPE_HEAD_FAST_GRID
  dim3 grid(heads, rows);
#else
  dim3 grid(rows, heads);
#endif
  int rotary_half = rotary_dim / 2;
  gemma4_rope_forward_bf16_kernel<<<grid, kRopeThreads, 0, stream>>>(
      q, k, cos, rotary_half, sin, rotary_half, seq_len, cos_batch_size,
      q_heads, kv_heads, head_dim, rotary_half);
  return cudaGetLastError();
}
