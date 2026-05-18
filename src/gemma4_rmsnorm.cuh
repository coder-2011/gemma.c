#ifndef GEMMA4_RMSNORM_CUH
#define GEMMA4_RMSNORM_CUH

#include <cuda_bf16.h>
#include <cuda_runtime.h>

extern "C" {

cudaError_t gemma4_rmsnorm_bf16(__nv_bfloat16 *out,
                                float *rstd,
                                const __nv_bfloat16 *inp,
                                const __nv_bfloat16 *weight,
                                int rows,
                                int width,
                                float eps,
                                cudaStream_t stream);

cudaError_t gemma4_residual_add_bf16(__nv_bfloat16 *out,
                                     const __nv_bfloat16 *inp1,
                                     const __nv_bfloat16 *inp2,
                                     int count,
                                     cudaStream_t stream);

cudaError_t gemma4_residual_add_rmsnorm_bf16(__nv_bfloat16 *residual,
                                             __nv_bfloat16 *normed,
                                             float *rstd,
                                             const __nv_bfloat16 *inp1,
                                             const __nv_bfloat16 *inp2,
                                             const __nv_bfloat16 *weight,
                                             int rows,
                                             int width,
                                             float eps,
                                             cudaStream_t stream);

}

#endif
