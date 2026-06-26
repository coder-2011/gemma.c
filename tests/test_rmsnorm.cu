#include "gemma4_rmsnorm.cuh"
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

void check_cuda(cudaError_t status, const char *expr, const char *file, int line) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "%s:%d: CUDA error for %s: %s\n", file, line, expr,
                 cudaGetErrorString(status));
    std::exit(1);
  }
}

#define CHECK_CUDA(expr) check_cuda((expr), #expr, __FILE__, __LINE__)

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

enum class RmsnormMode {
  LearnedWeight,
  ScaleFreeWrapper,
};

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

void fill_values(std::vector<__nv_bfloat16> &values,
                 __nv_bfloat16 (*make_value)(int)) {
  for (int i = 0; i < static_cast<int>(values.size()); ++i) {
    values[i] = make_value(i);
  }
}

void reference_rmsnorm(std::vector<__nv_bfloat16> &out,
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
    for (int c = 0; c < width; ++c) {
      size_t index = static_cast<size_t>(row) * width + c;
      float gamma = weight != nullptr ? bf16_to_float((*weight)[c]) : 1.0f;
      out[index] = __float2bfloat16(bf16_to_float(inp[index]) * scale * gamma);
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
    float diff =
        std::fabs(bf16_to_float(actual[i]) - bf16_to_float(expected[i]));
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

void run_rmsnorm_case(int rows, int width, RmsnormMode mode) {
  const int elems = rows * width;
  const bool has_weight = mode == RmsnormMode::LearnedWeight;

  std::vector<__nv_bfloat16> inp(elems);
  std::vector<__nv_bfloat16> weight(width);
  std::vector<__nv_bfloat16> actual(elems);
  std::vector<__nv_bfloat16> expected(elems);

  fill_values(inp, make_input_value);
  fill_values(weight, make_weight_value);
  reference_rmsnorm(
      expected, inp, has_weight ? &weight : nullptr, rows, width,
      GEMMA4_RMS_NORM_EPS);

  thrust::device_vector<__nv_bfloat16> d_inp(elems);
  thrust::device_vector<__nv_bfloat16> d_weight(has_weight ? width : 0);
  thrust::device_vector<__nv_bfloat16> d_out(elems);
  copy_to_device(d_inp, inp);
  if (has_weight) {
    copy_to_device(d_weight, weight);
  }

  if (mode == RmsnormMode::ScaleFreeWrapper) {
    CHECK_CUDA(gemma4_rmsnorm_scale_free_bf16(
        raw_ptr(d_out), raw_ptr(d_inp), rows, width, GEMMA4_RMS_NORM_EPS, 0));
  } else {
    CHECK_CUDA(gemma4_rmsnorm_bf16(
        raw_ptr(d_out), raw_ptr(d_inp), raw_ptr(d_weight), rows, width,
        GEMMA4_RMS_NORM_EPS, 0));
  }

  copy_to_host(actual, d_out);

  const char *label = has_weight ? "rmsnorm" : "scale-free rmsnorm";
  compare_bf16(actual, expected, 0.03125f, label);
}

void fill_residual_inputs(std::vector<__nv_bfloat16> &inp1,
                          std::vector<__nv_bfloat16> &inp2,
                          std::vector<__nv_bfloat16> &expected) {
  for (int i = 0; i < static_cast<int>(inp1.size()); ++i) {
    inp1[i] = make_input_value(i);
    inp2[i] = make_residual_value(i);
    expected[i] = __float2bfloat16(bf16_to_float(inp1[i]) +
                                   bf16_to_float(inp2[i]));
  }
}

void run_residual_add_case(int rows, int width) {
  const int elems = rows * width;
  std::vector<__nv_bfloat16> inp1(elems);
  std::vector<__nv_bfloat16> inp2(elems);
  std::vector<__nv_bfloat16> actual(elems);
  std::vector<__nv_bfloat16> expected(elems);
  fill_residual_inputs(inp1, inp2, expected);

  thrust::device_vector<__nv_bfloat16> d_inp1(elems);
  thrust::device_vector<__nv_bfloat16> d_inp2(elems);
  thrust::device_vector<__nv_bfloat16> d_out(elems);
  copy_to_device(d_inp1, inp1);
  copy_to_device(d_inp2, inp2);

  CHECK_CUDA(gemma4_residual_add_bf16(
      raw_ptr(d_out), raw_ptr(d_inp1), raw_ptr(d_inp2), elems, 0));
  copy_to_host(actual, d_out);
  compare_bf16(actual, expected, 0.0f, "residual add output");
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

  fill_residual_inputs(inp1, inp2, expected_residual);
  fill_values(weight, make_weight_value);
  reference_rmsnorm(
      expected_normed, expected_residual, &weight, rows, width,
      GEMMA4_RMS_NORM_EPS);

  thrust::device_vector<__nv_bfloat16> d_inp1(elems);
  thrust::device_vector<__nv_bfloat16> d_inp2(elems);
  thrust::device_vector<__nv_bfloat16> d_weight(width);
  thrust::device_vector<__nv_bfloat16> d_residual(elems);
  thrust::device_vector<__nv_bfloat16> d_normed(elems);
  copy_to_device(d_inp1, inp1);
  copy_to_device(d_inp2, inp2);
  copy_to_device(d_weight, weight);

  CHECK_CUDA(gemma4_residual_add_rmsnorm_bf16(
      raw_ptr(d_residual), raw_ptr(d_normed), raw_ptr(d_inp1), raw_ptr(d_inp2),
      raw_ptr(d_weight), rows, width, GEMMA4_RMS_NORM_EPS, 0));

  copy_to_host(actual_residual, d_residual);
  copy_to_host(actual_normed, d_normed);

  compare_bf16(actual_residual, expected_residual, 0.0f, "fused");
  compare_bf16(actual_normed, expected_normed, 0.03125f, "fused");
}

}  // namespace

int main() {
  run_rmsnorm_case(1, 256, RmsnormMode::LearnedWeight);
  run_rmsnorm_case(17, GEMMA4_HIDDEN_SIZE, RmsnormMode::LearnedWeight);
  run_rmsnorm_case(7, 512, RmsnormMode::ScaleFreeWrapper);
  run_residual_add_case(9, GEMMA4_HIDDEN_SIZE);
  run_fused_case(1, GEMMA4_HIDDEN_SIZE);
  run_fused_case(19, GEMMA4_HIDDEN_SIZE);

  std::printf("rmsnorm tests passed\n");
  return 0;
}
