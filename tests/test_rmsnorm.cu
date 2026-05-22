#include "gemma4_rmsnorm.cuh"
#include "gemma4.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

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

template <typename T>
class DeviceBuffer {
 public:
  explicit DeviceBuffer(size_t count) : count_(count) {
    if (count_ > 0) {
      CHECK_CUDA(cudaMalloc(&ptr_, count_ * sizeof(T)));
    }
  }

  ~DeviceBuffer() {
    if (ptr_ != nullptr) {
      cudaFree(ptr_);
    }
  }

  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;

  T *get() { return ptr_; }
  const T *get() const { return ptr_; }

  void copy_from(const std::vector<T> &src) {
    CHECK_CUDA(cudaMemcpy(ptr_, src.data(), count_ * sizeof(T),
                          cudaMemcpyHostToDevice));
  }

  void copy_to(std::vector<T> &dst) const {
    CHECK_CUDA(cudaMemcpy(dst.data(), ptr_, count_ * sizeof(T),
                          cudaMemcpyDeviceToHost));
  }

 private:
  T *ptr_ = nullptr;
  size_t count_ = 0;
};

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

bool uses_learned_weight(RmsnormMode mode) {
  return mode == RmsnormMode::LearnedWeight;
}

const char *rmsnorm_label(RmsnormMode mode) {
  return uses_learned_weight(mode) ? "rmsnorm" : "scale-free rmsnorm";
}

void run_rmsnorm_case(int rows, int width, RmsnormMode mode) {
  const int elems = rows * width;
  const bool has_weight = uses_learned_weight(mode);

  std::vector<__nv_bfloat16> inp(elems);
  std::vector<__nv_bfloat16> weight(width);
  std::vector<__nv_bfloat16> actual(elems);
  std::vector<__nv_bfloat16> expected(elems);
  std::vector<float> expected_rstd(rows);

  fill_values(inp, make_input_value);
  fill_values(weight, make_weight_value);
  reference_rmsnorm(expected, expected_rstd, inp, has_weight ? &weight : nullptr,
                    rows, width, GEMMA4_RMS_NORM_EPS);

  DeviceBuffer<__nv_bfloat16> d_inp(elems);
  DeviceBuffer<__nv_bfloat16> d_weight(has_weight ? width : 0);
  DeviceBuffer<__nv_bfloat16> d_out(elems);
  d_inp.copy_from(inp);
  if (has_weight) {
    d_weight.copy_from(weight);
  }

  if (mode == RmsnormMode::ScaleFreeWrapper) {
    CHECK_CUDA(gemma4_rmsnorm_scale_free_bf16(
        d_out.get(), d_inp.get(), rows, width, GEMMA4_RMS_NORM_EPS, 0));
  } else {
    CHECK_CUDA(gemma4_rmsnorm_bf16(
        d_out.get(), d_inp.get(), d_weight.get(), rows, width,
        GEMMA4_RMS_NORM_EPS, 0));
  }

  d_out.copy_to(actual);

  const char *label = rmsnorm_label(mode);
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

  DeviceBuffer<__nv_bfloat16> d_inp1(elems);
  DeviceBuffer<__nv_bfloat16> d_inp2(elems);
  DeviceBuffer<__nv_bfloat16> d_out(elems);
  d_inp1.copy_from(inp1);
  d_inp2.copy_from(inp2);

  CHECK_CUDA(gemma4_residual_add_bf16(
      d_out.get(), d_inp1.get(), d_inp2.get(), elems, 0));
  d_out.copy_to(actual);
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
  std::vector<float> expected_rstd(rows);

  fill_residual_inputs(inp1, inp2, expected_residual);
  fill_values(weight, make_weight_value);
  reference_rmsnorm(expected_normed, expected_rstd, expected_residual, &weight,
                    rows, width,
                    GEMMA4_RMS_NORM_EPS);

  DeviceBuffer<__nv_bfloat16> d_inp1(elems);
  DeviceBuffer<__nv_bfloat16> d_inp2(elems);
  DeviceBuffer<__nv_bfloat16> d_weight(width);
  DeviceBuffer<__nv_bfloat16> d_residual(elems);
  DeviceBuffer<__nv_bfloat16> d_normed(elems);
  d_inp1.copy_from(inp1);
  d_inp2.copy_from(inp2);
  d_weight.copy_from(weight);

  CHECK_CUDA(gemma4_residual_add_rmsnorm_bf16(
      d_residual.get(), d_normed.get(), d_inp1.get(), d_inp2.get(),
      d_weight.get(), rows, width, GEMMA4_RMS_NORM_EPS, 0));

  d_residual.copy_to(actual_residual);
  d_normed.copy_to(actual_normed);

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

  cudaError_t invalid = gemma4_rmsnorm_bf16(
      nullptr, nullptr, nullptr, 1, GEMMA4_HIDDEN_SIZE + 1,
      GEMMA4_RMS_NORM_EPS, 0);
  if (invalid != cudaErrorInvalidValue) {
    std::fprintf(stderr, "expected cudaErrorInvalidValue for invalid RMSNorm args\n");
    return 1;
  }

  std::printf("rmsnorm tests passed\n");
  return 0;
}
