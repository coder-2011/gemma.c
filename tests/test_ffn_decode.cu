#include "gemma4_ffn.cuh"
#include "gemma4_rmsnorm.cuh"
#include "gemma4.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

#include <algorithm>
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

float bf16_to_float(__nv_bfloat16 value) { return __bfloat162float(value); }

float gelu_tanh_reference(float x) {
  constexpr float kSqrtTwoOverPi = 0.7978845608028654f;
  constexpr float kGeluCubic = 0.044715f;
  return 0.5f * x *
         (1.0f + std::tanh(kSqrtTwoOverPi *
                           (x + kGeluCubic * x * x * x)));
}

__nv_bfloat16 make_x_value(int index) {
  const int centered = ((index * 17 + 5) % 127) - 63;
  return __float2bfloat16(static_cast<float>(centered) / 96.0f);
}

__nv_bfloat16 make_residual_value(int index) {
  const int centered = ((index * 29 + 11) % 181) - 90;
  return __float2bfloat16(static_cast<float>(centered) / 128.0f);
}

__nv_bfloat16 make_down_value(int tile, int col) {
  const int centered = ((tile * 13 + col * 5 + 17) % 97) - 48;
  return __float2bfloat16(static_cast<float>(centered) / 512.0f);
}

void compare_bf16(const std::vector<__nv_bfloat16> &actual,
                  const std::vector<__nv_bfloat16> &expected,
                  float tolerance,
                  const char *label) {
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
  if (max_abs > tolerance) {
    std::fprintf(stderr,
                 "%s max_abs=%g index=%d actual=%g expected=%g "
                 "tolerance=%g\n",
                 label, max_abs, max_index, bf16_to_float(actual[max_index]),
                 bf16_to_float(expected[max_index]), tolerance);
    std::exit(1);
  }
}

void run_sparse_case() {
  constexpr int tile_width = 256;
  constexpr int tiles = GEMMA4_INTERMEDIATE_SIZE / tile_width;
  const size_t gate_up_count =
      static_cast<size_t>(GEMMA4_HIDDEN_SIZE) * GEMMA4_PACKED_FFN_SIZE;
  const size_t down_count =
      static_cast<size_t>(GEMMA4_INTERMEDIATE_SIZE) * GEMMA4_HIDDEN_SIZE;

  std::vector<__nv_bfloat16> x(GEMMA4_HIDDEN_SIZE);
  std::vector<__nv_bfloat16> residual(GEMMA4_HIDDEN_SIZE);
  std::vector<__nv_bfloat16> gamma(GEMMA4_HIDDEN_SIZE);
  std::vector<__nv_bfloat16> actual_residual(GEMMA4_HIDDEN_SIZE);
  std::vector<__nv_bfloat16> actual_normed(GEMMA4_HIDDEN_SIZE);
  std::vector<__nv_bfloat16> expected_residual(GEMMA4_HIDDEN_SIZE);
  std::vector<__nv_bfloat16> expected_scaled_residual(GEMMA4_HIDDEN_SIZE);
  std::vector<__nv_bfloat16> expected_normed(GEMMA4_HIDDEN_SIZE);
  std::vector<float> ffn_out(GEMMA4_HIDDEN_SIZE, 0.0f);

  for (int i = 0; i < GEMMA4_HIDDEN_SIZE; ++i) {
    x[i] = make_x_value(i);
    residual[i] = make_residual_value(i);
    const float gamma_value =
        0.85f + static_cast<float>((i * 7 + 3) % 41) / 128.0f;
    gamma[i] = __float2bfloat16(gamma_value);
  }

  thrust::device_vector<__nv_bfloat16> d_x(GEMMA4_HIDDEN_SIZE);
  thrust::device_vector<__nv_bfloat16> d_residual(GEMMA4_HIDDEN_SIZE);
  thrust::device_vector<__nv_bfloat16> d_gamma(GEMMA4_HIDDEN_SIZE);
  thrust::device_vector<__nv_bfloat16> d_residual_out(GEMMA4_HIDDEN_SIZE);
  thrust::device_vector<__nv_bfloat16> d_normed_out(GEMMA4_HIDDEN_SIZE);
  thrust::device_vector<__nv_bfloat16> d_gate_up_src(gate_up_count);
  thrust::device_vector<__nv_bfloat16> d_gate_up(gate_up_count);
  thrust::device_vector<__nv_bfloat16> d_down_src(down_count);
  thrust::device_vector<__nv_bfloat16> d_down(down_count);
  thrust::device_vector<unsigned char> d_scratch(
      sizeof(Gemma4FfnDecodeScratch));
  thrust::device_vector<__nv_bfloat16> d_layer_scalar(1);

  copy_to_device(d_x, x);
  copy_to_device(d_residual, residual);
  copy_to_device(d_gamma, gamma);
  d_layer_scalar[0] = __float2bfloat16(0.5f);
  CHECK_CUDA(cudaMemset(raw_ptr(d_gate_up_src), 0,
                        gate_up_count * sizeof(__nv_bfloat16)));
  CHECK_CUDA(cudaMemset(raw_ptr(d_down_src), 0,
                        down_count * sizeof(__nv_bfloat16)));

  std::vector<__nv_bfloat16> gate_col(GEMMA4_HIDDEN_SIZE);
  std::vector<__nv_bfloat16> up_col(GEMMA4_HIDDEN_SIZE);
  std::vector<__nv_bfloat16> down_row(GEMMA4_HIDDEN_SIZE);

  for (int tile = 0; tile < tiles; ++tile) {
    const int intermediate_col =
        tile * tile_width + ((tile * 19 + 7) % tile_width);
    const int x_index = (tile * 97 + 13) % GEMMA4_HIDDEN_SIZE;
    std::fill(gate_col.begin(), gate_col.end(), __float2bfloat16(0.0f));
    std::fill(up_col.begin(), up_col.end(), __float2bfloat16(0.0f));

    const __nv_bfloat16 gate_weight =
        __float2bfloat16(0.35f + static_cast<float>(tile % 5) * 0.03125f);
    const __nv_bfloat16 up_weight =
        __float2bfloat16(-0.25f + static_cast<float>(tile % 7) * 0.0234375f);
    gate_col[x_index] = gate_weight;
    up_col[x_index] = up_weight;

    CHECK_CUDA(cudaMemcpy(
        raw_ptr(d_gate_up_src) +
            static_cast<size_t>(intermediate_col) * GEMMA4_HIDDEN_SIZE,
        gate_col.data(), gate_col.size() * sizeof(__nv_bfloat16),
        cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(
        raw_ptr(d_gate_up_src) +
            static_cast<size_t>(GEMMA4_INTERMEDIATE_SIZE + intermediate_col) *
                GEMMA4_HIDDEN_SIZE,
        up_col.data(), up_col.size() * sizeof(__nv_bfloat16),
        cudaMemcpyHostToDevice));

    for (int col = 0; col < GEMMA4_HIDDEN_SIZE; ++col) {
      down_row[col] = make_down_value(tile, col);
    }
    CHECK_CUDA(cudaMemcpy(
        raw_ptr(d_down_src) +
            static_cast<size_t>(intermediate_col) * GEMMA4_HIDDEN_SIZE,
        down_row.data(), down_row.size() * sizeof(__nv_bfloat16),
        cudaMemcpyHostToDevice));

    const float xv = bf16_to_float(x[x_index]);
    const float gate = xv * bf16_to_float(gate_weight);
    const float up = xv * bf16_to_float(up_weight);
    const float act = gelu_tanh_reference(gate) * up;
    for (int col = 0; col < GEMMA4_HIDDEN_SIZE; ++col) {
      ffn_out[col] = fmaf(act, bf16_to_float(down_row[col]), ffn_out[col]);
    }
  }

  double sum_sq = 0.0;
  std::vector<float> ffn_float(GEMMA4_HIDDEN_SIZE);
  for (int col = 0; col < GEMMA4_HIDDEN_SIZE; ++col) {
    const __nv_bfloat16 ffn_bf16 = __float2bfloat16_rn(ffn_out[col]);
    const float value = bf16_to_float(ffn_bf16);
    ffn_float[col] = value;
    sum_sq += static_cast<double>(value) * value;
  }
  const float scale =
      1.0f / std::sqrt(static_cast<float>(sum_sq / GEMMA4_HIDDEN_SIZE) +
                       GEMMA4_RMS_NORM_EPS);
  for (int col = 0; col < GEMMA4_HIDDEN_SIZE; ++col) {
    const float normed = ffn_float[col] * scale * bf16_to_float(gamma[col]);
    expected_normed[col] = __float2bfloat16_rn(normed);
    expected_residual[col] = __float2bfloat16_rn(
        bf16_to_float(residual[col]) + bf16_to_float(expected_normed[col]));
  }

  CHECK_CUDA(gemma4_ffn_decode_swizzle_weights_bf16(
      raw_ptr(d_gate_up), raw_ptr(d_gate_up_src), raw_ptr(d_down), raw_ptr(d_down_src),
      0));
  CHECK_CUDA(gemma4_ffn_decode_fused_bf16(
      raw_ptr(d_residual_out), raw_ptr(d_normed_out), raw_ptr(d_x), raw_ptr(d_residual),
      raw_ptr(d_gamma), raw_ptr(d_gate_up), raw_ptr(d_down),
      reinterpret_cast<Gemma4FfnDecodeScratch *>(raw_ptr(d_scratch)),
      nullptr, GEMMA4_RMS_NORM_EPS, 0));
  copy_to_host(actual_residual, d_residual_out);
  copy_to_host(actual_normed, d_normed_out);

  compare_bf16(actual_residual, expected_residual, 0.03125f,
               "fused FFN residual");
  compare_bf16(actual_normed, expected_normed, 0.03125f,
               "fused FFN normed");

  for (int col = 0; col < GEMMA4_HIDDEN_SIZE; ++col) {
    expected_scaled_residual[col] =
        __float2bfloat16_rn(bf16_to_float(expected_residual[col]) * 0.5f);
  }
  CHECK_CUDA(gemma4_ffn_decode_fused_bf16(
      raw_ptr(d_residual_out), raw_ptr(d_normed_out), raw_ptr(d_x), raw_ptr(d_residual),
      raw_ptr(d_gamma), raw_ptr(d_gate_up), raw_ptr(d_down),
      reinterpret_cast<Gemma4FfnDecodeScratch *>(raw_ptr(d_scratch)),
      raw_ptr(d_layer_scalar), GEMMA4_RMS_NORM_EPS, 0));
  copy_to_host(actual_residual, d_residual_out);
  compare_bf16(actual_residual, expected_scaled_residual, 0.03125f,
               "fused FFN scaled residual");

  constexpr int prefill_rows = 3;
  std::vector<__nv_bfloat16> x_prefill(
      static_cast<size_t>(prefill_rows) * GEMMA4_HIDDEN_SIZE);
  std::vector<__nv_bfloat16> residual_prefill(x_prefill.size());
  std::vector<__nv_bfloat16> actual_prefill_residual(x_prefill.size());
  std::vector<__nv_bfloat16> actual_prefill_normed(x_prefill.size());
  std::vector<__nv_bfloat16> expected_prefill_residual(x_prefill.size());
  std::vector<__nv_bfloat16> expected_prefill_normed(x_prefill.size());

  for (int row = 0; row < prefill_rows; ++row) {
    for (int col = 0; col < GEMMA4_HIDDEN_SIZE; ++col) {
      const int offset = row * GEMMA4_HIDDEN_SIZE + col;
      x_prefill[offset] = make_x_value(offset + row * 31);
      residual_prefill[offset] = make_residual_value(offset + row * 43);
    }
  }

  for (int row = 0; row < prefill_rows; ++row) {
    std::fill(ffn_out.begin(), ffn_out.end(), 0.0f);
    for (int tile = 0; tile < tiles; ++tile) {
      const int x_index = (tile * 97 + 13) % GEMMA4_HIDDEN_SIZE;
      const __nv_bfloat16 gate_weight =
          __float2bfloat16(0.35f + static_cast<float>(tile % 5) * 0.03125f);
      const __nv_bfloat16 up_weight =
          __float2bfloat16(-0.25f + static_cast<float>(tile % 7) * 0.0234375f);
      const float xv = bf16_to_float(
          x_prefill[row * GEMMA4_HIDDEN_SIZE + x_index]);
      const float gate = xv * bf16_to_float(gate_weight);
      const float up = xv * bf16_to_float(up_weight);
      const float act = gelu_tanh_reference(gate) * up;
      for (int col = 0; col < GEMMA4_HIDDEN_SIZE; ++col) {
        ffn_out[col] = fmaf(act, bf16_to_float(make_down_value(tile, col)),
                            ffn_out[col]);
      }
    }

    sum_sq = 0.0;
    for (int col = 0; col < GEMMA4_HIDDEN_SIZE; ++col) {
      const __nv_bfloat16 ffn_bf16 = __float2bfloat16_rn(ffn_out[col]);
      const float value = bf16_to_float(ffn_bf16);
      ffn_float[col] = value;
      sum_sq += static_cast<double>(value) * value;
    }
    const float row_scale =
        1.0f / std::sqrt(static_cast<float>(sum_sq / GEMMA4_HIDDEN_SIZE) +
                         GEMMA4_RMS_NORM_EPS);
    for (int col = 0; col < GEMMA4_HIDDEN_SIZE; ++col) {
      const int offset = row * GEMMA4_HIDDEN_SIZE + col;
      const float normed =
          ffn_float[col] * row_scale * bf16_to_float(gamma[col]);
      expected_prefill_normed[offset] = __float2bfloat16_rn(normed);
      expected_prefill_residual[offset] = __float2bfloat16_rn(
          bf16_to_float(residual_prefill[offset]) +
          bf16_to_float(expected_prefill_normed[offset]));
    }
  }

  thrust::device_vector<__nv_bfloat16> d_x_prefill(x_prefill.size());
  thrust::device_vector<__nv_bfloat16> d_residual_prefill(residual_prefill.size());
  thrust::device_vector<__nv_bfloat16> d_prefill_mlp_out(x_prefill.size());
  thrust::device_vector<__nv_bfloat16> d_prefill_residual_out(x_prefill.size());
  thrust::device_vector<__nv_bfloat16> d_prefill_normed_out(x_prefill.size());
  thrust::device_vector<__nv_bfloat16> d_prefill_scratch(
      gemma4_ffn_prefill_scratch_elements(prefill_rows));
  copy_to_device(d_x_prefill, x_prefill);
  copy_to_device(d_residual_prefill, residual_prefill);

  Gemma4FfnPrefillScratch prefill_scratch =
      gemma4_ffn_prefill_scratch_from_buffer(
          raw_ptr(d_prefill_scratch), prefill_rows);
  CHECK_CUDA(gemma4_ffn_prefill_mlp_bf16(
      raw_ptr(d_prefill_mlp_out), raw_ptr(d_x_prefill), raw_ptr(d_gate_up),
      raw_ptr(d_down), prefill_scratch, prefill_rows, 0));
  CHECK_CUDA(gemma4_rmsnorm_bf16(
      raw_ptr(d_prefill_normed_out), raw_ptr(d_prefill_mlp_out),
      raw_ptr(d_gamma), prefill_rows, GEMMA4_HIDDEN_SIZE,
      GEMMA4_RMS_NORM_EPS, 0));
  CHECK_CUDA(gemma4_residual_add_bf16(
      raw_ptr(d_prefill_residual_out), raw_ptr(d_residual_prefill),
      raw_ptr(d_prefill_normed_out), int(x_prefill.size()), 0));
  copy_to_host(actual_prefill_residual, d_prefill_residual_out);
  copy_to_host(actual_prefill_normed, d_prefill_normed_out);

  compare_bf16(actual_prefill_residual, expected_prefill_residual, 0.03125f,
               "prefill FFN residual");
  compare_bf16(actual_prefill_normed, expected_prefill_normed, 0.03125f,
               "prefill FFN normed");
}

}  // namespace

int main() {
  run_sparse_case();

  std::printf("ffn decode tests passed\n");
  return 0;
}
