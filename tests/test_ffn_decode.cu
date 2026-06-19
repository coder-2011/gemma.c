#include "gemma4_ffn.cuh"
#include "gemma4.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

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
    CHECK_CUDA(cudaMemcpy(ptr_, src.data(), src.size() * sizeof(T),
                          cudaMemcpyHostToDevice));
  }

  void copy_to(std::vector<T> &dst) const {
    CHECK_CUDA(cudaMemcpy(dst.data(), ptr_, dst.size() * sizeof(T),
                          cudaMemcpyDeviceToHost));
  }

 private:
  T *ptr_ = nullptr;
  size_t count_ = 0;
};

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

__nv_bfloat16 make_gamma_value(int index) {
  const float value = 0.85f + static_cast<float>((index * 7 + 3) % 41) / 128.0f;
  return __float2bfloat16(value);
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
  std::vector<__nv_bfloat16> expected_normed(GEMMA4_HIDDEN_SIZE);
  std::vector<float> ffn_out(GEMMA4_HIDDEN_SIZE, 0.0f);

  for (int i = 0; i < GEMMA4_HIDDEN_SIZE; ++i) {
    x[i] = make_x_value(i);
    residual[i] = make_residual_value(i);
    gamma[i] = make_gamma_value(i);
  }

  DeviceBuffer<__nv_bfloat16> d_x(GEMMA4_HIDDEN_SIZE);
  DeviceBuffer<__nv_bfloat16> d_residual(GEMMA4_HIDDEN_SIZE);
  DeviceBuffer<__nv_bfloat16> d_gamma(GEMMA4_HIDDEN_SIZE);
  DeviceBuffer<__nv_bfloat16> d_residual_out(GEMMA4_HIDDEN_SIZE);
  DeviceBuffer<__nv_bfloat16> d_normed_out(GEMMA4_HIDDEN_SIZE);
  DeviceBuffer<__nv_bfloat16> d_gate_up_src(gate_up_count);
  DeviceBuffer<__nv_bfloat16> d_gate_up(gate_up_count);
  DeviceBuffer<__nv_bfloat16> d_down_src(down_count);
  DeviceBuffer<__nv_bfloat16> d_down(down_count);
  DeviceBuffer<Gemma4FfnDecodeScratch> d_scratch(1);

  d_x.copy_from(x);
  d_residual.copy_from(residual);
  d_gamma.copy_from(gamma);
  CHECK_CUDA(cudaMemset(d_gate_up_src.get(), 0,
                        gate_up_count * sizeof(__nv_bfloat16)));
  CHECK_CUDA(cudaMemset(d_down_src.get(), 0,
                        down_count * sizeof(__nv_bfloat16)));

  std::vector<__nv_bfloat16> gate_col(GEMMA4_HIDDEN_SIZE);
  std::vector<__nv_bfloat16> up_col(GEMMA4_HIDDEN_SIZE);
  std::vector<__nv_bfloat16> down_row(GEMMA4_HIDDEN_SIZE);

  // Keep the reference sparse: one nonzero intermediate per tile makes the CPU
  // expected value cheap while still forcing the CUDA path to walk the full
  // Gemma FFN shapes and weight swizzle.
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
        d_gate_up_src.get() +
            static_cast<size_t>(intermediate_col) * GEMMA4_HIDDEN_SIZE,
        gate_col.data(), gate_col.size() * sizeof(__nv_bfloat16),
        cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(
        d_gate_up_src.get() +
            static_cast<size_t>(GEMMA4_INTERMEDIATE_SIZE + intermediate_col) *
                GEMMA4_HIDDEN_SIZE,
        up_col.data(), up_col.size() * sizeof(__nv_bfloat16),
        cudaMemcpyHostToDevice));

    for (int col = 0; col < GEMMA4_HIDDEN_SIZE; ++col) {
      down_row[col] = make_down_value(tile, col);
    }
    CHECK_CUDA(cudaMemcpy(
        d_down_src.get() +
            static_cast<size_t>(intermediate_col) * GEMMA4_HIDDEN_SIZE,
        down_row.data(), down_row.size() * sizeof(__nv_bfloat16),
        cudaMemcpyHostToDevice));

    const float xv = bf16_to_float(x[x_index]);
    const float gate = xv * bf16_to_float(gate_weight);
    const float up = xv * bf16_to_float(up_weight);
    const float act = gate * gelu_tanh_reference(up);
    for (int col = 0; col < GEMMA4_HIDDEN_SIZE; ++col) {
      ffn_out[col] = fmaf(act, bf16_to_float(down_row[col]), ffn_out[col]);
    }
  }

  double sum_sq = 0.0;
  std::vector<float> residual_float(GEMMA4_HIDDEN_SIZE);
  for (int col = 0; col < GEMMA4_HIDDEN_SIZE; ++col) {
    const float value = ffn_out[col] + bf16_to_float(residual[col]);
    residual_float[col] = value;
    expected_residual[col] = __float2bfloat16_rn(value);
    sum_sq += static_cast<double>(value) * value;
  }
  const float scale =
      1.0f / std::sqrt(static_cast<float>(sum_sq / GEMMA4_HIDDEN_SIZE) +
                       GEMMA4_RMS_NORM_EPS);
  for (int col = 0; col < GEMMA4_HIDDEN_SIZE; ++col) {
    const float value = residual_float[col] * scale * bf16_to_float(gamma[col]);
    expected_normed[col] = __float2bfloat16_rn(value);
  }

  CHECK_CUDA(gemma4_ffn_decode_swizzle_weights_bf16(
      d_gate_up.get(), d_gate_up_src.get(), d_down.get(), d_down_src.get(),
      0));
  CHECK_CUDA(gemma4_ffn_decode_configure_scratch_l2(d_scratch.get(), 0));
  // rows == 1 should take the fused decode branch through the generalized
  // gemma4_ffn_bf16() entry point.
  Gemma4FfnBf16Args decode_args = {};
  decode_args.residual_out = d_residual_out.get();
  decode_args.normed_out = d_normed_out.get();
  decode_args.x = d_x.get();
  decode_args.residual = d_residual.get();
  decode_args.rms_weight = d_gamma.get();
  decode_args.w_gate_up_decode = d_gate_up.get();
  decode_args.w_down_decode = d_down.get();
  decode_args.decode_scratch = d_scratch.get();
  decode_args.rows = 1;
  decode_args.eps = GEMMA4_RMS_NORM_EPS;
  CHECK_CUDA(gemma4_ffn_bf16(decode_args));
  d_residual_out.copy_to(actual_residual);
  d_normed_out.copy_to(actual_normed);

  compare_bf16(actual_residual, expected_residual, 0.03125f,
               "fused FFN residual");
  compare_bf16(actual_normed, expected_normed, 0.03125f,
               "fused FFN normed");

  constexpr int prefill_rows = 3;
  // rows > 1 should take the CUTLASS prefill branch using canonical,
  // unswizzled weights and caller-provided scratch.
  std::vector<__nv_bfloat16> x_prefill(prefill_rows * GEMMA4_HIDDEN_SIZE);
  std::vector<__nv_bfloat16> residual_prefill(prefill_rows *
                                              GEMMA4_HIDDEN_SIZE);
  std::vector<__nv_bfloat16> actual_prefill_residual(prefill_rows *
                                                     GEMMA4_HIDDEN_SIZE);
  std::vector<__nv_bfloat16> actual_prefill_normed(prefill_rows *
                                                   GEMMA4_HIDDEN_SIZE);
  std::vector<__nv_bfloat16> expected_prefill_residual(prefill_rows *
                                                       GEMMA4_HIDDEN_SIZE);
  std::vector<__nv_bfloat16> expected_prefill_normed(prefill_rows *
                                                     GEMMA4_HIDDEN_SIZE);
  for (int row = 0; row < prefill_rows; ++row) {
    for (int col = 0; col < GEMMA4_HIDDEN_SIZE; ++col) {
      x_prefill[row * GEMMA4_HIDDEN_SIZE + col] =
          make_x_value(row * 1009 + col);
      residual_prefill[row * GEMMA4_HIDDEN_SIZE + col] =
          make_residual_value(row * 787 + col);
    }
  }

  for (int row = 0; row < prefill_rows; ++row) {
    std::vector<float> row_down(GEMMA4_HIDDEN_SIZE, 0.0f);
    for (int tile = 0; tile < tiles; ++tile) {
      const int intermediate_col =
          tile * tile_width + ((tile * 19 + 7) % tile_width);
      const int x_index = (tile * 97 + 13) % GEMMA4_HIDDEN_SIZE;
      const __nv_bfloat16 gate_weight =
          __float2bfloat16(0.35f + static_cast<float>(tile % 5) * 0.03125f);
      const __nv_bfloat16 up_weight =
          __float2bfloat16(-0.25f + static_cast<float>(tile % 7) * 0.0234375f);
      const float xv =
          bf16_to_float(x_prefill[row * GEMMA4_HIDDEN_SIZE + x_index]);
      const float gate = bf16_to_float(
          __float2bfloat16_rn(xv * bf16_to_float(gate_weight)));
      const float up = bf16_to_float(
          __float2bfloat16_rn(xv * bf16_to_float(up_weight)));
      const float act =
          bf16_to_float(__float2bfloat16_rn(gate * gelu_tanh_reference(up)));
      for (int col = 0; col < GEMMA4_HIDDEN_SIZE; ++col) {
        row_down[col] =
            fmaf(act, bf16_to_float(make_down_value(tile, col)),
                 row_down[col]);
      }
      (void)intermediate_col;
    }

    double row_sum_sq = 0.0;
    std::vector<float> row_residual(GEMMA4_HIDDEN_SIZE);
    for (int col = 0; col < GEMMA4_HIDDEN_SIZE; ++col) {
      const int idx = row * GEMMA4_HIDDEN_SIZE + col;
      const float down = bf16_to_float(__float2bfloat16_rn(row_down[col]));
      const float value = bf16_to_float(__float2bfloat16_rn(
          down + bf16_to_float(residual_prefill[idx])));
      row_residual[col] = value;
      expected_prefill_residual[idx] = __float2bfloat16_rn(value);
      row_sum_sq += static_cast<double>(value) * value;
    }
    const float row_scale =
        1.0f / std::sqrt(static_cast<float>(row_sum_sq /
                                            GEMMA4_HIDDEN_SIZE) +
                         GEMMA4_RMS_NORM_EPS);
    for (int col = 0; col < GEMMA4_HIDDEN_SIZE; ++col) {
      const int idx = row * GEMMA4_HIDDEN_SIZE + col;
      const float value =
          row_residual[col] * row_scale * bf16_to_float(gamma[col]);
      expected_prefill_normed[idx] = __float2bfloat16_rn(value);
    }
  }

  DeviceBuffer<__nv_bfloat16> d_x_prefill(prefill_rows *
                                          GEMMA4_HIDDEN_SIZE);
  DeviceBuffer<__nv_bfloat16> d_residual_prefill(prefill_rows *
                                                 GEMMA4_HIDDEN_SIZE);
  DeviceBuffer<__nv_bfloat16> d_prefill_residual_out(prefill_rows *
                                                     GEMMA4_HIDDEN_SIZE);
  DeviceBuffer<__nv_bfloat16> d_prefill_normed_out(prefill_rows *
                                                   GEMMA4_HIDDEN_SIZE);
  DeviceBuffer<__nv_bfloat16> d_prefill_scratch(
      gemma4_ffn_prefill_scratch_elements(prefill_rows));
  d_x_prefill.copy_from(x_prefill);
  d_residual_prefill.copy_from(residual_prefill);

  Gemma4FfnBf16Args prefill_args = {};
  prefill_args.residual_out = d_prefill_residual_out.get();
  prefill_args.normed_out = d_prefill_normed_out.get();
  prefill_args.x = d_x_prefill.get();
  prefill_args.residual = d_residual_prefill.get();
  prefill_args.rms_weight = d_gamma.get();
  prefill_args.w_gate_up_prefill_col_major = d_gate_up_src.get();
  prefill_args.w_down_prefill_row_major = d_down_src.get();
  prefill_args.prefill_scratch =
      gemma4_ffn_prefill_scratch_from_buffer(d_prefill_scratch.get(),
                                             prefill_rows);
  prefill_args.rows = prefill_rows;
  prefill_args.eps = GEMMA4_RMS_NORM_EPS;
  CHECK_CUDA(gemma4_ffn_bf16(prefill_args));
  d_prefill_residual_out.copy_to(actual_prefill_residual);
  d_prefill_normed_out.copy_to(actual_prefill_normed);

  compare_bf16(actual_prefill_residual, expected_prefill_residual, 0.0625f,
               "prefill FFN residual");
  compare_bf16(actual_prefill_normed, expected_prefill_normed, 0.0625f,
               "prefill FFN normed");

  DeviceBuffer<unsigned char> d_scratch_bytes(
      sizeof(Gemma4FfnDecodeScratch) + 128);
  auto *misaligned_scratch = reinterpret_cast<Gemma4FfnDecodeScratch *>(
      d_scratch_bytes.get() + 16);
  // Alignment failures should be caught on the host side, before launching a
  // kernel that would issue vectorized loads/stores from an invalid address.
  Gemma4FfnBf16Args invalid_scratch_args = decode_args;
  invalid_scratch_args.decode_scratch = misaligned_scratch;
  cudaError_t invalid_scratch = gemma4_ffn_bf16(invalid_scratch_args);
  if (invalid_scratch != cudaErrorInvalidValue) {
    std::fprintf(stderr,
                 "expected cudaErrorInvalidValue for misaligned scratch\n");
    std::exit(1);
  }
}

}  // namespace

int main() {
  run_sparse_case();

  Gemma4FfnBf16Args invalid_args = {};
  invalid_args.rows = 1;
  cudaError_t invalid = gemma4_ffn_bf16(invalid_args);
  if (invalid != cudaErrorInvalidValue) {
    std::fprintf(stderr, "expected cudaErrorInvalidValue for invalid args\n");
    return 1;
  }

  std::printf("ffn tests passed\n");
  return 0;
}
