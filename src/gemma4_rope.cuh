#ifndef GEMMA4_ROPE_CUH
#define GEMMA4_ROPE_CUH

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <stdint.h>

extern "C" {

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

cudaError_t gemma4_sliding_rope_bf16(__nv_bfloat16 *q,
                                     __nv_bfloat16 *k,
                                     const float *cos,
                                     const float *sin,
                                     int seq_len,
                                     int batch_size,
                                     int cos_batch_size,
                                     cudaStream_t stream);

cudaError_t gemma4_global_rope_bf16(__nv_bfloat16 *q,
                                    __nv_bfloat16 *k,
                                    const float *cos,
                                    const float *sin,
                                    int seq_len,
                                    int batch_size,
                                    int cos_batch_size,
                                    cudaStream_t stream);

}

#endif
