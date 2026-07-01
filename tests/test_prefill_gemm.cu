#include "gemma4_matmul_kernels.cuh"
#include "gemma4.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

#define CHECK_CUDA(expr)                                                        \
  do {                                                                          \
    if (const cudaError_t status = (expr); status != cudaSuccess) {              \
      std::fprintf(stderr, "%s:%d: CUDA error for %s: %s\n", __FILE__,          \
                   __LINE__, #expr, cudaGetErrorString(status));                \
      std::exit(1);                                                             \
    }                                                                           \
  } while (0)

// Returns the raw CUDA pointer owned by a Thrust device vector.
template <typename T>
T *raw_ptr(thrust::device_vector<T> &v) {
  return thrust::raw_pointer_cast(v.data());
}

// Returns the raw const CUDA pointer owned by a Thrust device vector.
template <typename T>
const T *raw_ptr(const thrust::device_vector<T> &v) {
  return thrust::raw_pointer_cast(v.data());
}

template <typename T>
void copy_to_device(thrust::device_vector<T> &dst, const std::vector<T> &src) {
  CHECK_CUDA(cudaMemcpy(raw_ptr(dst), src.data(), src.size() * sizeof(T),
                        cudaMemcpyHostToDevice));
}

template <typename T>
void copy_to_host(std::vector<T> &dst, const thrust::device_vector<T> &src) {
  CHECK_CUDA(cudaMemcpy(dst.data(), raw_ptr(src), dst.size() * sizeof(T),
                        cudaMemcpyDeviceToHost));
}

// Converts BF16 to FP32 for reference math and tolerance checks.
float bf16_to_float(__nv_bfloat16 value) {
  return __bfloat162float(value);
}

// Installs two nonzero weights per output column and computes the reference.
void fill_sparse_case(std::vector<__nv_bfloat16> &x,
                      std::vector<__nv_bfloat16> &w,
                      std::vector<__nv_bfloat16> &expected,
                      int rows,
                      int k,
                      int n) {
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < k; ++col) {
      const int index = row * 37 + col;
      const int centered = ((index * 13 + 7) % 97) - 48;
      x[static_cast<size_t>(row) * k + col] =
          __float2bfloat16_rn(static_cast<float>(centered) / 96.0f);
    }
  }

  for (int col = 0; col < n; ++col) {
    const int k0 = (col * 17 + 3) % k;
    const int k1 = (k0 + k / 2 + 1) % k;
    const int index0 = col * 11;
    const int index1 = col * 11 + 1;
    const int centered0 = ((index0 * 19 + 5) % 83) - 41;
    const int centered1 = ((index1 * 19 + 5) % 83) - 41;
    w[static_cast<size_t>(col) * k + k0] =
        __float2bfloat16_rn(static_cast<float>(centered0) / 128.0f);
    w[static_cast<size_t>(col) * k + k1] =
        __float2bfloat16_rn(static_cast<float>(centered1) / 128.0f);
  }

  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < n; ++col) {
      const int k0 = (col * 17 + 3) % k;
      const int k1 = (k0 + k / 2 + 1) % k;
      const float x0 = bf16_to_float(x[static_cast<size_t>(row) * k + k0]);
      const float x1 = bf16_to_float(x[static_cast<size_t>(row) * k + k1]);
      const float w0 = bf16_to_float(w[static_cast<size_t>(col) * k + k0]);
      const float w1 = bf16_to_float(w[static_cast<size_t>(col) * k + k1]);
      const float value = x0 * w0 + x1 * w1;
      expected[static_cast<size_t>(row) * n + col] =
          __float2bfloat16_rn(value);
    }
  }
}

// Checks one prefill GEMM shape against the sparse CPU reference.
void run_sparse_case(int rows, int k, int n, const char *label) {
  std::vector<__nv_bfloat16> x(static_cast<size_t>(rows) * k);
  std::vector<__nv_bfloat16> w(static_cast<size_t>(n) * k);
  std::vector<__nv_bfloat16> expected(static_cast<size_t>(rows) * n);
  std::vector<__nv_bfloat16> actual(expected.size());

  fill_sparse_case(x, w, expected, rows, k, n);

  thrust::device_vector<__nv_bfloat16> d_x(x.size());
  thrust::device_vector<__nv_bfloat16> d_w(w.size());
  thrust::device_vector<__nv_bfloat16> d_y(actual.size());
  copy_to_device(d_x, x);
  copy_to_device(d_w, w);

  CHECK_CUDA(gemma4_prefill_gemm_bf16(
      raw_ptr(d_x), raw_ptr(d_w), raw_ptr(d_y), rows, k, n, 0));
  copy_to_host(actual, d_y);

  float max_abs = 0.0f;
  int max_index = 0;
  for (int i = 0; i < static_cast<int>(actual.size()); ++i) {
    const float diff =
        std::fabs(bf16_to_float(actual[i]) - bf16_to_float(expected[i]));
    if (diff > max_abs) {
      max_abs = diff;
      max_index = i;
    }
  }
  if (max_abs > 0.00390625f) {
    std::fprintf(stderr,
                 "%s max_abs=%g index=%d actual=%g expected=%g\n",
                 label, max_abs, max_index, bf16_to_float(actual[max_index]),
                 bf16_to_float(expected[max_index]));
    std::exit(1);
  }
}

// Verifies basic argument validation for unsupported tensor-core shapes.
void run_invalid_shape_case() {
  thrust::device_vector<__nv_bfloat16> d_x(64);
  thrust::device_vector<__nv_bfloat16> d_w(64);
  thrust::device_vector<__nv_bfloat16> d_y(64);
  const cudaError_t status =
      gemma4_prefill_gemm_bf16(raw_ptr(d_x), raw_ptr(d_w), raw_ptr(d_y), 1, 63, 32, 0);
  if (status != cudaErrorInvalidValue) {
    std::fprintf(stderr, "invalid prefill GEMM shape returned %d\n", status);
    std::exit(1);
  }
}

}  // namespace

// Runs focused CUTLASS prefill GEMM correctness checks.
int main() {
  run_sparse_case(
      3, GEMMA4_HIDDEN_SIZE, GEMMA4_GLOBAL_K_PROJ_SIZE, "global-k prefill");
  run_sparse_case(129, 64, 32, "large-row prefill");
  run_invalid_shape_case();
  std::printf("prefill GEMM tests passed\n");
  return 0;
}
