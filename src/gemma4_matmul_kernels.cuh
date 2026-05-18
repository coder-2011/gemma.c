#ifndef GEMMA4_MATMUL_KERNELS_CUH
#define GEMMA4_MATMUL_KERNELS_CUH

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

extern "C" {

cublasStatus_t gemma4_ffn_gate_up_prefill(cublasHandle_t handle,
                                           const half *x,
                                           const half *w_col_major, half *y,
                                           int m, cudaStream_t stream);

cudaError_t gemma4_ffn_gate_up_decode(const half *x,
                                       const half *w_col_major, half *y,
                                       cudaStream_t stream);
cudaError_t gemma4_ffn_down_decode(const half *x, const half *w_col_major,
                                    half *y, cudaStream_t stream);
cudaError_t gemma4_sliding_qkv_decode(const half *x, const half *w_col_major,
                                       half *y, cudaStream_t stream);
cudaError_t gemma4_sliding_o_decode(const half *x, const half *w_col_major,
                                     half *y, cudaStream_t stream);
cudaError_t gemma4_global_q_decode(const half *x, const half *w_col_major,
                                    half *y, cudaStream_t stream);
cudaError_t gemma4_global_k_decode(const half *x, const half *w_col_major,
                                    half *y, cudaStream_t stream);
cudaError_t gemma4_global_o_decode(const half *x, const half *w_col_major,
                                    half *y, cudaStream_t stream);
cudaError_t gemma4_final_logits_decode(const half *x,
                                        const half *w_col_major, half *y,
                                        cudaStream_t stream);

}

#endif
