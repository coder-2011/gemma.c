#include "gemma4_rope.cuh"
#include "gemma4_cuda_utils.cuh"

#include <stdint.h>

#ifndef GEMMA4_ROPE_HEAD_FAST_GRID
#define GEMMA4_ROPE_HEAD_FAST_GRID 0
#endif

#ifndef GEMMA4_ROPE_QK_LOAD_CS
#define GEMMA4_ROPE_QK_LOAD_CS 1
#endif

namespace {

constexpr int kRopeThreads = WARP_SIZE;
using RopePack = Bf16Packed128;
static constexpr int kRopePairsPerPack = kBf16Packed128Elements;

__device__ inline float4 load_float4g(const float *__restrict__ address) {
  return __ldg(reinterpret_cast<const float4 *>(address));
}

template <int Elem>
__device__ inline void rotate_pack_element(RopePack &out_lo,
                                           RopePack &out_hi,
                                           const RopePack &lo,
                                           const RopePack &hi,
                                           float c,
                                           float s) {
  float x1 = __bfloat162float(lo[Elem]);
  float x2 = __bfloat162float(hi[Elem]);
  out_lo[Elem] = __float2bfloat16_rn(fmaf(-x2, s, x1 * c));
  out_hi[Elem] = __float2bfloat16_rn(fmaf(x1, s, x2 * c));
}

__device__ inline void rotate_pair_bf16(floatX *__restrict__ head,
                                        const float *__restrict__ cos_row,
                                        const float *__restrict__ sin_row,
                                        int rotary_half,
                                        int i) {
  float x1 = __bfloat162float(head[i]);
  float x2 = __bfloat162float(head[rotary_half + i]);
  float c = loadg(cos_row + i);
  float s = loadg(sin_row + i);
  head[i] = __float2bfloat16_rn(fmaf(-x2, s, x1 * c));
  head[rotary_half + i] = __float2bfloat16_rn(fmaf(x1, s, x2 * c));
}

__device__ inline void rotate_pack_bf16(floatX *__restrict__ head,
                                        const float *__restrict__ cos_row,
                                        const float *__restrict__ sin_row,
                                        int rotary_half,
                                        int pack) {
  int i = pack * kRopePairsPerPack;
#if GEMMA4_ROPE_QK_LOAD_CS
  RopePack lo = load128cs(head + i);
  RopePack hi = load128cs(head + rotary_half + i);
#else
  RopePack lo = load128(head + i);
  RopePack hi = load128(head + rotary_half + i);
#endif
  float4 c0 = load_float4g(cos_row + i);
  float4 c1 = load_float4g(cos_row + i + 4);
  float4 s0 = load_float4g(sin_row + i);
  float4 s1 = load_float4g(sin_row + i + 4);
  RopePack out_lo;
  RopePack out_hi;

  rotate_pack_element<0>(out_lo, out_hi, lo, hi, c0.x, s0.x);
  rotate_pack_element<1>(out_lo, out_hi, lo, hi, c0.y, s0.y);
  rotate_pack_element<2>(out_lo, out_hi, lo, hi, c0.z, s0.z);
  rotate_pack_element<3>(out_lo, out_hi, lo, hi, c0.w, s0.w);
  rotate_pack_element<4>(out_lo, out_hi, lo, hi, c1.x, s1.x);
  rotate_pack_element<5>(out_lo, out_hi, lo, hi, c1.y, s1.y);
  rotate_pack_element<6>(out_lo, out_hi, lo, hi, c1.z, s1.z);
  rotate_pack_element<7>(out_lo, out_hi, lo, hi, c1.w, s1.w);

  store128(head + i, out_lo);
  store128(head + rotary_half + i, out_hi);
}

__device__ inline void rotate_head_bf16(floatX *__restrict__ head,
                                        const float *__restrict__ cos_row,
                                        const float *__restrict__ sin_row,
                                        int rotary_half) {
  int full_packs = rotary_half / kRopePairsPerPack;
  for (int pack = threadIdx.x; pack < full_packs; pack += blockDim.x) {
    rotate_pack_bf16(head, cos_row, sin_row, rotary_half, pack);
  }

  int tail_start = full_packs * kRopePairsPerPack;
  for (int i = tail_start + threadIdx.x; i < rotary_half; i += blockDim.x) {
    rotate_pair_bf16(head, cos_row, sin_row, rotary_half, i);
  }
}

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
    rotate_head_bf16(q_head, cos_row, sin_row, rotary_half);
  }
  if (head < kv_heads) {
    int64_t k_offset = static_cast<int64_t>(row) * k_row_stride + head * head_dim;
    floatX *k_head = k + k_offset;
    rotate_head_bf16(k_head, cos_row, sin_row, rotary_half);
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
