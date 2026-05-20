#ifndef GEMMA4_ROPE_CUH
#define GEMMA4_ROPE_CUH

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <stdint.h>

// In-place RoPE for physical [batch, seq, heads, head_dim] Q/K buffers.
// cos_row_stride/sin_row_stride may be compact rotary_dim / 2 or full head_dim.
cudaError_t gemma4_rope_bf16(__nv_bfloat16 *q,
                             int64_t q_row_stride,
                             __nv_bfloat16 *k,
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
                             cudaStream_t stream);

// In-place RoPE matching the Python forward signature layout:
// q: [batch, q_heads, seq, head_dim]
// k: [batch, kv_heads, seq, head_dim]
// cos/sin: [1 or batch, seq, head_dim]
cudaError_t gemma4_rope_forward_bf16(__nv_bfloat16 *q,
                                     __nv_bfloat16 *k,
                                     const float *cos,
                                     const float *sin,
                                     int seq_len,
                                     int batch_size,
                                     int cos_batch_size,
                                     int q_heads,
                                     int kv_heads,
                                     int head_dim,
                                     int rotary_dim,
                                     cudaStream_t stream);

#endif
