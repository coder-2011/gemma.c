#include "gemma4_rope.cuh"
#include "gemma4_cuda_utils.cuh"

#include <stdint.h>

namespace {

constexpr int kRopeThreads = 128;

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
  int64_t table_offset = table_row;

  const float *cos_row = cos + table_offset * cos_row_stride;
  const float *sin_row = sin + table_offset * sin_row_stride;

  if (head < q_heads) {
    int64_t q_offset = static_cast<int64_t>(row) * q_row_stride + head * head_dim;
    floatX *q_head = q + q_offset;
    rotate_head_bf16(q_head, cos_row, sin_row, rotary_half);
  }
  if (head < kv_heads) {
    int64_t k_offset = static_cast<int64_t>(row) * k_row_stride + head * head_dim;
    floatX *k_head = k + k_offset;
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
  int64_t table_offset = table_row;

  const float *cos_row = cos + table_offset * cos_row_stride;
  const float *sin_row = sin + table_offset * sin_row_stride;

  if (head < q_heads) {
    int64_t batch_head = static_cast<int64_t>(batch) * q_heads + head;
    int64_t q_offset = (batch_head * seq_len + seq) * head_dim;
    floatX *q_head = q + q_offset;
    rotate_head_bf16(q_head, cos_row, sin_row, rotary_half);
  }
  if (head < kv_heads) {
    int64_t batch_head = static_cast<int64_t>(batch) * kv_heads + head;
    int64_t k_offset = (batch_head * seq_len + seq) * head_dim;
    floatX *k_head = k + k_offset;
    rotate_head_bf16(k_head, cos_row, sin_row, rotary_half);
  }
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
  if (seq_len < 0 || batch_size < 0 || q_heads < 0 || kv_heads < 0 || head_dim <= 0 || rotary_dim <= 0 || rotary_dim > head_dim || (rotary_dim % 2) != 0 || (cos_batch_size != 1 && cos_batch_size != batch_size)) return cudaErrorInvalidValue;
  if (seq_len == 0 || batch_size == 0 || (q_heads == 0 && kv_heads == 0)) return cudaSuccess;
  if (q == nullptr || k == nullptr || cos == nullptr || sin == nullptr) return cudaErrorInvalidValue;

  int rotary_half = rotary_dim / 2;
  if (q_row_stride < static_cast<int64_t>(q_heads) * head_dim || k_row_stride < static_cast<int64_t>(kv_heads) * head_dim || cos_row_stride < rotary_half || sin_row_stride < rotary_half) return cudaErrorInvalidValue;
  int rows = batch_size * seq_len;
  int heads = q_heads > kv_heads ? q_heads : kv_heads;

  dim3 grid(rows, heads);
  gemma4_rope_bf16_kernel<<<grid, kRopeThreads, 0, stream>>>(q, q_row_stride, k, k_row_stride, cos, cos_row_stride, sin, sin_row_stride, seq_len, cos_batch_size, q_heads, kv_heads, head_dim, rotary_half);
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
  if (seq_len < 0 || batch_size < 0 || q_heads < 0 || kv_heads < 0 || head_dim <= 0 || rotary_dim <= 0 || rotary_dim > head_dim || (rotary_dim % 2) != 0 || (cos_batch_size != 1 && cos_batch_size != batch_size)) return cudaErrorInvalidValue;
  if (seq_len == 0 || batch_size == 0 || (q_heads == 0 && kv_heads == 0)) return cudaSuccess;
  if (q == nullptr || k == nullptr || cos == nullptr || sin == nullptr) return cudaErrorInvalidValue;

  int rows = batch_size * seq_len;
  int heads = q_heads > kv_heads ? q_heads : kv_heads;

  dim3 grid(rows, heads);
  int rotary_half = rotary_dim / 2;
  gemma4_rope_forward_bf16_kernel<<<grid, kRopeThreads, 0, stream>>>(q, k, cos, head_dim, sin, head_dim, seq_len, cos_batch_size, q_heads, kv_heads, head_dim, rotary_half);
  return cudaGetLastError();
}
