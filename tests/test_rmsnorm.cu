#include "gemma4_rmsnorm.cuh"
#include "gemma4.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

void check_cuda(cudaError_t status, const char *expr, const char *file, int line) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "%s:%d: CUDA error for %s: %s\n", file, line, expr,
                 cudaGetErrorString(status));
    std::exit(1);
  }
}

#define CHECK_CUDA(expr) check_cuda((expr), #expr, __FILE__, __LINE__)

float bf16_to_float(__nv_bfloat16 value) { return __bfloat162float(value); }

__nv_bfloat16 make_input_value(int index) {
  int centered = ((index * 37 + 17) % 257) - 128;
  return __float2bfloat16(static_cast<float>(centered) / 64.0f);
}

__nv_bfloat16 make_residual_value(int index) {
  int centered = ((index * 19 + 23) % 211) - 105;
  return __float2bfloat16(static_cast<float>(centered) / 80.0f);
}

__nv_bfloat16 make_weight_value(int channel) {
  float value = 0.75f + static_cast<float>((channel * 13 + 5) % 97) / 128.0f;
  return __float2bfloat16(value);
}

void reference_rmsnorm(std::vector<__nv_bfloat16> &out,
                       std::vector<float> &rstd,
                       const std::vector<__nv_bfloat16> &inp,
                       const std::vector<__nv_bfloat16> *weight,
                       int rows,
                       int width,
                       float eps) {
  for (int row = 0; row < rows; ++row) {
    double sum_sq = 0.0;
    for (int c = 0; c < width; ++c) {
      float value = bf16_to_float(inp[static_cast<size_t>(row) * width + c]);
      sum_sq += static_cast<double>(value) * value;
    }
    float scale = 1.0f / std::sqrt(static_cast<float>(sum_sq / width) + eps);
    rstd[row] = scale;
    for (int c = 0; c < width; ++c) {
      float value = bf16_to_float(inp[static_cast<size_t>(row) * width + c]);
      float gamma = weight != nullptr ? bf16_to_float((*weight)[c]) : 1.0f;
      out[static_cast<size_t>(row) * width + c] =
          __float2bfloat16(value * scale * gamma);
    }
  }
}

void compare_bf16(const std::vector<__nv_bfloat16> &actual,
                  const std::vector<__nv_bfloat16> &expected,
                  float tolerance,
                  const char *label) {
  float max_abs = 0.0f;
  int max_index = 0;
  for (int i = 0; i < static_cast<int>(actual.size()); ++i) {
    float diff = std::fabs(bf16_to_float(actual[i]) - bf16_to_float(expected[i]));
    if (diff > max_abs) {
      max_abs = diff;
      max_index = i;
    }
  }
  if (max_abs > tolerance) {
    std::fprintf(stderr, "%s max_abs=%g at index=%d exceeds tolerance=%g\n",
                 label, max_abs, max_index, tolerance);
    std::exit(1);
  }
}

void compare_float(const std::vector<float> &actual,
                   const std::vector<float> &expected,
                   float tolerance,
                   const char *label) {
  float max_abs = 0.0f;
  int max_index = 0;
  for (int i = 0; i < static_cast<int>(actual.size()); ++i) {
    float diff = std::fabs(actual[i] - expected[i]);
    if (diff > max_abs) {
      max_abs = diff;
      max_index = i;
    }
  }
  if (max_abs > tolerance) {
    std::fprintf(stderr, "%s max_abs=%g at index=%d exceeds tolerance=%g\n",
                 label, max_abs, max_index, tolerance);
    std::exit(1);
  }
}

void run_rmsnorm_case(int rows, int width) {
  const int elems = rows * width;
  std::vector<__nv_bfloat16> inp(elems);
  std::vector<__nv_bfloat16> weight(width);
  std::vector<__nv_bfloat16> actual(elems);
  std::vector<__nv_bfloat16> expected(elems);
  std::vector<float> actual_rstd(rows);
  std::vector<float> expected_rstd(rows);

  for (int i = 0; i < elems; ++i) {
    inp[i] = make_input_value(static_cast<int>(i));
  }
  for (int c = 0; c < width; ++c) {
    weight[c] = make_weight_value(c);
  }
  reference_rmsnorm(expected, expected_rstd, inp, &weight, rows, width,
                    GEMMA4_RMS_NORM_EPS);

  __nv_bfloat16 *d_inp = nullptr;
  __nv_bfloat16 *d_weight = nullptr;
  __nv_bfloat16 *d_out = nullptr;
  float *d_rstd = nullptr;
  CHECK_CUDA(cudaMalloc(&d_inp, elems * sizeof(__nv_bfloat16)));
  CHECK_CUDA(cudaMalloc(&d_weight, static_cast<size_t>(width) * sizeof(__nv_bfloat16)));
  CHECK_CUDA(cudaMalloc(&d_out, elems * sizeof(__nv_bfloat16)));
  CHECK_CUDA(cudaMalloc(&d_rstd, static_cast<size_t>(rows) * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(d_inp, inp.data(), elems * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_weight, weight.data(),
                        static_cast<size_t>(width) * sizeof(__nv_bfloat16),
                        cudaMemcpyHostToDevice));

  CHECK_CUDA(gemma4_rmsnorm_bf16(d_out, d_rstd, d_inp, d_weight, rows,
                                 width, GEMMA4_RMS_NORM_EPS, 0));
  CHECK_CUDA(cudaMemcpy(actual.data(), d_out, elems * sizeof(__nv_bfloat16),
                        cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(actual_rstd.data(), d_rstd,
                        static_cast<size_t>(rows) * sizeof(float),
                        cudaMemcpyDeviceToHost));

  compare_bf16(actual, expected, 0.03125f, "rmsnorm output");
  compare_float(actual_rstd, expected_rstd, 2.0e-4f, "rmsnorm rstd");

  CHECK_CUDA(cudaFree(d_inp));
  CHECK_CUDA(cudaFree(d_weight));
  CHECK_CUDA(cudaFree(d_out));
  CHECK_CUDA(cudaFree(d_rstd));
}

void run_scale_free_rmsnorm_case(int rows, int width, bool use_wrapper) {
  const int elems = rows * width;
  std::vector<__nv_bfloat16> inp(elems);
  std::vector<__nv_bfloat16> actual(elems);
  std::vector<__nv_bfloat16> expected(elems);
  std::vector<float> actual_rstd(rows);
  std::vector<float> expected_rstd(rows);

  for (int i = 0; i < elems; ++i) {
    inp[i] = make_input_value(static_cast<int>(i));
  }
  reference_rmsnorm(expected, expected_rstd, inp, nullptr, rows, width,
                    GEMMA4_RMS_NORM_EPS);

  __nv_bfloat16 *d_inp = nullptr;
  __nv_bfloat16 *d_out = nullptr;
  float *d_rstd = nullptr;
  CHECK_CUDA(cudaMalloc(&d_inp, elems * sizeof(__nv_bfloat16)));
  CHECK_CUDA(cudaMalloc(&d_out, elems * sizeof(__nv_bfloat16)));
  CHECK_CUDA(cudaMalloc(&d_rstd, static_cast<size_t>(rows) * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(d_inp, inp.data(), elems * sizeof(__nv_bfloat16),
                        cudaMemcpyHostToDevice));

  if (use_wrapper) {
    CHECK_CUDA(gemma4_rmsnorm_scale_free_bf16(
        d_out, d_rstd, d_inp, rows, width, GEMMA4_RMS_NORM_EPS, 0));
  } else {
    CHECK_CUDA(gemma4_rmsnorm_bf16(
        d_out, d_rstd, d_inp, nullptr, rows, width,
        GEMMA4_RMS_NORM_EPS, 0));
  }
  CHECK_CUDA(cudaMemcpy(actual.data(), d_out, elems * sizeof(__nv_bfloat16),
                        cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(actual_rstd.data(), d_rstd,
                        static_cast<size_t>(rows) * sizeof(float),
                        cudaMemcpyDeviceToHost));

  compare_bf16(actual, expected, 0.03125f, "scale-free rmsnorm output");
  compare_float(actual_rstd, expected_rstd, 2.0e-4f,
                "scale-free rmsnorm rstd");

  CHECK_CUDA(cudaFree(d_inp));
  CHECK_CUDA(cudaFree(d_out));
  CHECK_CUDA(cudaFree(d_rstd));
}

void run_residual_add_case(int rows, int width) {
  const int elems = rows * width;
  std::vector<__nv_bfloat16> inp1(elems);
  std::vector<__nv_bfloat16> inp2(elems);
  std::vector<__nv_bfloat16> actual(elems);
  std::vector<__nv_bfloat16> expected(elems);

  for (int i = 0; i < elems; ++i) {
    inp1[i] = make_input_value(static_cast<int>(i));
    inp2[i] = make_residual_value(static_cast<int>(i));
    expected[i] = __float2bfloat16(bf16_to_float(inp1[i]) + bf16_to_float(inp2[i]));
  }

  __nv_bfloat16 *d_inp1 = nullptr;
  __nv_bfloat16 *d_inp2 = nullptr;
  __nv_bfloat16 *d_out = nullptr;
  CHECK_CUDA(cudaMalloc(&d_inp1, elems * sizeof(__nv_bfloat16)));
  CHECK_CUDA(cudaMalloc(&d_inp2, elems * sizeof(__nv_bfloat16)));
  CHECK_CUDA(cudaMalloc(&d_out, elems * sizeof(__nv_bfloat16)));
  CHECK_CUDA(cudaMemcpy(d_inp1, inp1.data(), elems * sizeof(__nv_bfloat16),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_inp2, inp2.data(), elems * sizeof(__nv_bfloat16),
                        cudaMemcpyHostToDevice));

  CHECK_CUDA(gemma4_residual_add_bf16(d_out, d_inp1, d_inp2, static_cast<int>(elems), 0));
  CHECK_CUDA(cudaMemcpy(actual.data(), d_out, elems * sizeof(__nv_bfloat16),
                        cudaMemcpyDeviceToHost));
  compare_bf16(actual, expected, 0.0f, "residual add output");

  CHECK_CUDA(cudaFree(d_inp1));
  CHECK_CUDA(cudaFree(d_inp2));
  CHECK_CUDA(cudaFree(d_out));
}

void run_fused_case(int rows, int width) {
  const int elems = rows * width;
  std::vector<__nv_bfloat16> inp1(elems);
  std::vector<__nv_bfloat16> inp2(elems);
  std::vector<__nv_bfloat16> weight(width);
  std::vector<__nv_bfloat16> actual_residual(elems);
  std::vector<__nv_bfloat16> expected_residual(elems);
  std::vector<__nv_bfloat16> actual_normed(elems);
  std::vector<__nv_bfloat16> expected_normed(elems);
  std::vector<float> actual_rstd(rows);
  std::vector<float> expected_rstd(rows);

  for (int i = 0; i < elems; ++i) {
    inp1[i] = make_input_value(static_cast<int>(i));
    inp2[i] = make_residual_value(static_cast<int>(i));
    expected_residual[i] =
        __float2bfloat16(bf16_to_float(inp1[i]) + bf16_to_float(inp2[i]));
  }
  for (int c = 0; c < width; ++c) {
    weight[c] = make_weight_value(c);
  }
  reference_rmsnorm(expected_normed, expected_rstd, expected_residual, &weight,
                    rows, width, GEMMA4_RMS_NORM_EPS);

  __nv_bfloat16 *d_inp1 = nullptr;
  __nv_bfloat16 *d_inp2 = nullptr;
  __nv_bfloat16 *d_weight = nullptr;
  __nv_bfloat16 *d_residual = nullptr;
  __nv_bfloat16 *d_normed = nullptr;
  float *d_rstd = nullptr;
  CHECK_CUDA(cudaMalloc(&d_inp1, elems * sizeof(__nv_bfloat16)));
  CHECK_CUDA(cudaMalloc(&d_inp2, elems * sizeof(__nv_bfloat16)));
  CHECK_CUDA(cudaMalloc(&d_weight, static_cast<size_t>(width) * sizeof(__nv_bfloat16)));
  CHECK_CUDA(cudaMalloc(&d_residual, elems * sizeof(__nv_bfloat16)));
  CHECK_CUDA(cudaMalloc(&d_normed, elems * sizeof(__nv_bfloat16)));
  CHECK_CUDA(cudaMalloc(&d_rstd, static_cast<size_t>(rows) * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(d_inp1, inp1.data(), elems * sizeof(__nv_bfloat16),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_inp2, inp2.data(), elems * sizeof(__nv_bfloat16),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_weight, weight.data(),
                        static_cast<size_t>(width) * sizeof(__nv_bfloat16),
                        cudaMemcpyHostToDevice));

  CHECK_CUDA(gemma4_residual_add_rmsnorm_bf16(
      d_residual, d_normed, d_rstd, d_inp1, d_inp2, d_weight, rows, width,
      GEMMA4_RMS_NORM_EPS, 0));
  CHECK_CUDA(cudaMemcpy(actual_residual.data(), d_residual,
                        elems * sizeof(__nv_bfloat16),
                        cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(actual_normed.data(), d_normed,
                        elems * sizeof(__nv_bfloat16),
                        cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(actual_rstd.data(), d_rstd,
                        static_cast<size_t>(rows) * sizeof(float),
                        cudaMemcpyDeviceToHost));

  compare_bf16(actual_residual, expected_residual, 0.0f,
               "fused residual output");
  compare_bf16(actual_normed, expected_normed, 0.03125f,
               "fused normed output");
  compare_float(actual_rstd, expected_rstd, 2.0e-4f, "fused rstd");

  CHECK_CUDA(cudaFree(d_inp1));
  CHECK_CUDA(cudaFree(d_inp2));
  CHECK_CUDA(cudaFree(d_weight));
  CHECK_CUDA(cudaFree(d_residual));
  CHECK_CUDA(cudaFree(d_normed));
  CHECK_CUDA(cudaFree(d_rstd));
}

void run_scale_free_fused_case(int rows, int width) {
  const int elems = rows * width;
  std::vector<__nv_bfloat16> inp1(elems);
  std::vector<__nv_bfloat16> inp2(elems);
  std::vector<__nv_bfloat16> actual_residual(elems);
  std::vector<__nv_bfloat16> expected_residual(elems);
  std::vector<__nv_bfloat16> actual_normed(elems);
  std::vector<__nv_bfloat16> expected_normed(elems);
  std::vector<float> actual_rstd(rows);
  std::vector<float> expected_rstd(rows);

  for (int i = 0; i < elems; ++i) {
    inp1[i] = make_input_value(static_cast<int>(i));
    inp2[i] = make_residual_value(static_cast<int>(i));
    expected_residual[i] =
        __float2bfloat16(bf16_to_float(inp1[i]) + bf16_to_float(inp2[i]));
  }
  reference_rmsnorm(expected_normed, expected_rstd, expected_residual, nullptr,
                    rows, width, GEMMA4_RMS_NORM_EPS);

  __nv_bfloat16 *d_inp1 = nullptr;
  __nv_bfloat16 *d_inp2 = nullptr;
  __nv_bfloat16 *d_residual = nullptr;
  __nv_bfloat16 *d_normed = nullptr;
  float *d_rstd = nullptr;
  CHECK_CUDA(cudaMalloc(&d_inp1, elems * sizeof(__nv_bfloat16)));
  CHECK_CUDA(cudaMalloc(&d_inp2, elems * sizeof(__nv_bfloat16)));
  CHECK_CUDA(cudaMalloc(&d_residual, elems * sizeof(__nv_bfloat16)));
  CHECK_CUDA(cudaMalloc(&d_normed, elems * sizeof(__nv_bfloat16)));
  CHECK_CUDA(cudaMalloc(&d_rstd, static_cast<size_t>(rows) * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(d_inp1, inp1.data(), elems * sizeof(__nv_bfloat16),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_inp2, inp2.data(), elems * sizeof(__nv_bfloat16),
                        cudaMemcpyHostToDevice));

  CHECK_CUDA(gemma4_residual_add_rmsnorm_bf16(
      d_residual, d_normed, d_rstd, d_inp1, d_inp2, nullptr, rows, width,
      GEMMA4_RMS_NORM_EPS, 0));
  CHECK_CUDA(cudaMemcpy(actual_residual.data(), d_residual,
                        elems * sizeof(__nv_bfloat16),
                        cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(actual_normed.data(), d_normed,
                        elems * sizeof(__nv_bfloat16),
                        cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(actual_rstd.data(), d_rstd,
                        static_cast<size_t>(rows) * sizeof(float),
                        cudaMemcpyDeviceToHost));

  compare_bf16(actual_residual, expected_residual, 0.0f,
               "scale-free fused residual output");
  compare_bf16(actual_normed, expected_normed, 0.03125f,
               "scale-free fused normed output");
  compare_float(actual_rstd, expected_rstd, 2.0e-4f,
                "scale-free fused rstd");

  CHECK_CUDA(cudaFree(d_inp1));
  CHECK_CUDA(cudaFree(d_inp2));
  CHECK_CUDA(cudaFree(d_residual));
  CHECK_CUDA(cudaFree(d_normed));
  CHECK_CUDA(cudaFree(d_rstd));
}

}  // namespace

int main() {
  run_rmsnorm_case(1, 256);
  run_rmsnorm_case(17, GEMMA4_HIDDEN_SIZE);
  run_scale_free_rmsnorm_case(1, 256, false);
  run_scale_free_rmsnorm_case(7, 512, true);
  run_scale_free_rmsnorm_case(1, GEMMA4_HIDDEN_SIZE, false);
  run_residual_add_case(9, GEMMA4_HIDDEN_SIZE);
  run_fused_case(1, 512);
  run_fused_case(19, GEMMA4_HIDDEN_SIZE);
  run_scale_free_fused_case(5, 512);
  run_scale_free_fused_case(1, GEMMA4_HIDDEN_SIZE);

  cudaError_t invalid = gemma4_rmsnorm_bf16(
      nullptr, nullptr, nullptr, nullptr, 1, GEMMA4_HIDDEN_SIZE + 1,
      GEMMA4_RMS_NORM_EPS, 0);
  if (invalid != cudaErrorInvalidValue) {
    std::fprintf(stderr, "expected cudaErrorInvalidValue for invalid RMSNorm args\n");
    return 1;
  }

  std::printf("rmsnorm tests passed\n");
  return 0;
}
