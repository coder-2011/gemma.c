#ifndef GEMMA4_MATMUL_KERNELS_CUH
#define GEMMA4_MATMUL_KERNELS_CUH

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

extern "C" {

cublasStatus_t gemma4_ffn_gate_up_prefill(cublasHandle_t handle,
                                           const __nv_bfloat16 *x,
                                           const __nv_bfloat16 *w_col_major,
                                           __nv_bfloat16 *y,
                                           int m, cudaStream_t stream);
cublasStatus_t gemma4_ffn_down_prefill(cublasHandle_t handle,
                                        const __nv_bfloat16 *x,
                                        const __nv_bfloat16 *w_col_major,
                                        __nv_bfloat16 *y,
                                        int m, cudaStream_t stream);
cublasStatus_t gemma4_sliding_qkv_prefill(cublasHandle_t handle,
                                           const __nv_bfloat16 *x,
                                           const __nv_bfloat16 *w_col_major,
                                           __nv_bfloat16 *y,
                                           int m, cudaStream_t stream);
cublasStatus_t gemma4_sliding_o_prefill(cublasHandle_t handle,
                                         const __nv_bfloat16 *x,
                                         const __nv_bfloat16 *w_col_major,
                                         __nv_bfloat16 *y,
                                         int m, cudaStream_t stream);
cublasStatus_t gemma4_global_q_prefill(cublasHandle_t handle,
                                        const __nv_bfloat16 *x,
                                        const __nv_bfloat16 *w_col_major,
                                        __nv_bfloat16 *y,
                                        int m, cudaStream_t stream);
cublasStatus_t gemma4_global_k_prefill(cublasHandle_t handle,
                                        const __nv_bfloat16 *x,
                                        const __nv_bfloat16 *w_col_major,
                                        __nv_bfloat16 *y,
                                        int m, cudaStream_t stream);
cublasStatus_t gemma4_global_o_prefill(cublasHandle_t handle,
                                        const __nv_bfloat16 *x,
                                        const __nv_bfloat16 *w_col_major,
                                        __nv_bfloat16 *y,
                                        int m, cudaStream_t stream);
cublasStatus_t gemma4_final_logits_prefill(cublasHandle_t handle,
                                            const __nv_bfloat16 *x,
                                            const __nv_bfloat16 *w_col_major,
                                            __nv_bfloat16 *y,
                                            int m, cudaStream_t stream);

cudaError_t gemma4_ffn_gate_up_decode(const __nv_bfloat16 *x,
                                       const __nv_bfloat16 *w_col_major,
                                       __nv_bfloat16 *y,
                                       cudaStream_t stream);
cudaError_t gemma4_ffn_down_decode(const __nv_bfloat16 *x,
                                    const __nv_bfloat16 *w_col_major,
                                    __nv_bfloat16 *y, cudaStream_t stream);
cudaError_t gemma4_sliding_qkv_decode(const __nv_bfloat16 *x,
                                       const __nv_bfloat16 *w_col_major,
                                       __nv_bfloat16 *y,
                                       cudaStream_t stream);
cudaError_t gemma4_sliding_o_decode(const __nv_bfloat16 *x,
                                     const __nv_bfloat16 *w_col_major,
                                     __nv_bfloat16 *y, cudaStream_t stream);
cudaError_t gemma4_global_q_decode(const __nv_bfloat16 *x,
                                    const __nv_bfloat16 *w_col_major,
                                    __nv_bfloat16 *y, cudaStream_t stream);
cudaError_t gemma4_global_k_decode(const __nv_bfloat16 *x,
                                    const __nv_bfloat16 *w_col_major,
                                    __nv_bfloat16 *y, cudaStream_t stream);
cudaError_t gemma4_global_o_decode(const __nv_bfloat16 *x,
                                    const __nv_bfloat16 *w_col_major,
                                    __nv_bfloat16 *y, cudaStream_t stream);
cudaError_t gemma4_final_logits_decode(const __nv_bfloat16 *x,
                                        const __nv_bfloat16 *w_col_major,
                                        __nv_bfloat16 *y,
                                        cudaStream_t stream);

}

#endif
