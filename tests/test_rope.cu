#include "gemma4_rope.cuh"
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

__nv_bfloat16 make_value(int index) {
  int centered = ((index * 29 + 7) % 251) - 125;
  return __float2bfloat16(static_cast<float>(centered) / 71.0f);
}

void fill_rope_table(std::vector<float> &cos,
                     std::vector<float> &sin,
                     int seq_len,
                     int cos_batch_size,
                     int table_width) {
  for (int b = 0; b < cos_batch_size; ++b) {
    for (int pos = 0; pos < seq_len; ++pos) {
      for (int i = 0; i < table_width; ++i) {
        float angle = static_cast<float>((pos + 1) * (i + 3)) * 0.0037f +
                      static_cast<float>(b) * 0.019f;
        int offset = (b * seq_len + pos) * table_width + i;
        cos[offset] = std::cos(angle);
        sin[offset] = std::sin(angle);
      }
    }
  }
}

void reference_rope(std::vector<__nv_bfloat16> &values,
                    const std::vector<float> &cos,
                    const std::vector<float> &sin,
                    int batch_size,
                    int seq_len,
                    int cos_batch_size,
                    int cos_row_stride,
                    int heads,
                    int head_dim,
                    int rotary_dim) {
  int row_stride = heads * head_dim;
  int rotary_half = rotary_dim / 2;
  for (int b = 0; b < batch_size; ++b) {
    for (int pos = 0; pos < seq_len; ++pos) {
      int row = b * seq_len + pos;
      int table_batch = cos_batch_size == 1 ? 0 : b;
      const float *cos_row = cos.data() + (table_batch * seq_len + pos) * cos_row_stride;
      const float *sin_row = sin.data() + (table_batch * seq_len + pos) * cos_row_stride;
      for (int h = 0; h < heads; ++h) {
        int base = row * row_stride + h * head_dim;
        for (int i = 0; i < rotary_half; ++i) {
          float x1 = bf16_to_float(values[base + i]);
          float x2 = bf16_to_float(values[base + rotary_half + i]);
          float c = cos_row[i];
          float s = sin_row[i];
          values[base + i] = __float2bfloat16_rn(fmaf(-x2, s, x1 * c));
          values[base + rotary_half + i] =
              __float2bfloat16_rn(fmaf(x1, s, x2 * c));
        }
      }
    }
  }
}

void reference_rope_forward_layout(std::vector<__nv_bfloat16> &values,
                                   const std::vector<float> &cos,
                                   const std::vector<float> &sin,
                                   int batch_size,
                                   int seq_len,
                                   int cos_batch_size,
                                   int cos_row_stride,
                                   int heads,
                                   int head_dim,
                                   int rotary_dim) {
  int rotary_half = rotary_dim / 2;
  for (int b = 0; b < batch_size; ++b) {
    for (int h = 0; h < heads; ++h) {
      for (int pos = 0; pos < seq_len; ++pos) {
        int table_batch = cos_batch_size == 1 ? 0 : b;
        const float *cos_row =
            cos.data() + (table_batch * seq_len + pos) * cos_row_stride;
        const float *sin_row =
            sin.data() + (table_batch * seq_len + pos) * cos_row_stride;
        int base = (static_cast<int64_t>(b) * heads + h) * seq_len *
                   head_dim + pos * head_dim;
        for (int i = 0; i < rotary_half; ++i) {
          float x1 = bf16_to_float(values[base + i]);
          float x2 = bf16_to_float(values[base + rotary_half + i]);
          float c = cos_row[i];
          float s = sin_row[i];
          values[base + i] = __float2bfloat16_rn(fmaf(-x2, s, x1 * c));
          values[base + rotary_half + i] =
              __float2bfloat16_rn(fmaf(x1, s, x2 * c));
        }
      }
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
    std::fprintf(stderr,
                 "%s max_abs=%g at index=%d actual=%g expected=%g "
                 "exceeds tolerance=%g\n",
                 label, max_abs, max_index, bf16_to_float(actual[max_index]),
                 bf16_to_float(expected[max_index]), tolerance);
    std::exit(1);
  }
}

void run_case(int batch_size,
              int seq_len,
              int cos_batch_size,
              int q_heads,
              int kv_heads,
              int head_dim,
              int rotary_dim) {
  int q_count = batch_size * seq_len * q_heads * head_dim;
  int k_count = batch_size * seq_len * kv_heads * head_dim;
  int rotary_half = rotary_dim / 2;
  int table_width = rotary_half;
  int table_count = cos_batch_size * seq_len * table_width;

  std::vector<__nv_bfloat16> q(q_count);
  std::vector<__nv_bfloat16> k(k_count);
  std::vector<__nv_bfloat16> expected_q(q_count);
  std::vector<__nv_bfloat16> expected_k(k_count);
  std::vector<float> cos(table_count);
  std::vector<float> sin(table_count);

  for (int i = 0; i < q_count; ++i) {
    q[i] = make_value(i);
  }
  for (int i = 0; i < k_count; ++i) {
    k[i] = make_value(i + 1009);
  }
  expected_q = q;
  expected_k = k;
  fill_rope_table(cos, sin, seq_len, cos_batch_size, table_width);
  reference_rope(expected_q, cos, sin, batch_size, seq_len, cos_batch_size,
                 table_width, q_heads, head_dim, rotary_dim);
  reference_rope(expected_k, cos, sin, batch_size, seq_len, cos_batch_size,
                 table_width, kv_heads, head_dim, rotary_dim);

  __nv_bfloat16 *d_q = nullptr;
  __nv_bfloat16 *d_k = nullptr;
  float *d_cos = nullptr;
  float *d_sin = nullptr;
  CHECK_CUDA(cudaMalloc(&d_q, static_cast<size_t>(q_count) * sizeof(q[0])));
  CHECK_CUDA(cudaMalloc(&d_k, static_cast<size_t>(k_count) * sizeof(k[0])));
  CHECK_CUDA(cudaMalloc(&d_cos, static_cast<size_t>(table_count) * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_sin, static_cast<size_t>(table_count) * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(d_q, q.data(),
                        static_cast<size_t>(q_count) * sizeof(q[0]),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_k, k.data(),
                        static_cast<size_t>(k_count) * sizeof(k[0]),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_cos, cos.data(),
                        static_cast<size_t>(table_count) * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_sin, sin.data(),
                        static_cast<size_t>(table_count) * sizeof(float),
                        cudaMemcpyHostToDevice));

  CHECK_CUDA(gemma4_rope_bf16(
      d_q, static_cast<int64_t>(q_heads) * head_dim, d_k,
      static_cast<int64_t>(kv_heads) * head_dim, d_cos, rotary_half, d_sin,
      rotary_half, seq_len, batch_size, cos_batch_size, q_heads, kv_heads,
      head_dim, rotary_dim, 0));
  CHECK_CUDA(cudaMemcpy(q.data(), d_q,
                        static_cast<size_t>(q_count) * sizeof(q[0]),
                        cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(k.data(), d_k,
                        static_cast<size_t>(k_count) * sizeof(k[0]),
                        cudaMemcpyDeviceToHost));

  compare_bf16(q, expected_q, 0.0078125f, "rope q");
  compare_bf16(k, expected_k, 0.0078125f, "rope k");

  CHECK_CUDA(cudaFree(d_q));
  CHECK_CUDA(cudaFree(d_k));
  CHECK_CUDA(cudaFree(d_cos));
  CHECK_CUDA(cudaFree(d_sin));
}

void run_forward_case(int batch_size,
                      int seq_len,
                      int cos_batch_size,
                      int q_heads,
                      int kv_heads,
                      int head_dim,
                      int rotary_dim) {
  int q_count = batch_size * q_heads * seq_len * head_dim;
  int k_count = batch_size * kv_heads * seq_len * head_dim;
  int table_width = rotary_dim / 2;
  int table_count = cos_batch_size * seq_len * table_width;

  std::vector<__nv_bfloat16> q(q_count);
  std::vector<__nv_bfloat16> k(k_count);
  std::vector<__nv_bfloat16> expected_q(q_count);
  std::vector<__nv_bfloat16> expected_k(k_count);
  std::vector<float> cos(table_count);
  std::vector<float> sin(table_count);

  for (int i = 0; i < q_count; ++i) {
    q[i] = make_value(i + 11);
  }
  for (int i = 0; i < k_count; ++i) {
    k[i] = make_value(i + 2113);
  }
  expected_q = q;
  expected_k = k;
  fill_rope_table(cos, sin, seq_len, cos_batch_size, table_width);
  reference_rope_forward_layout(expected_q, cos, sin, batch_size, seq_len,
                                cos_batch_size, table_width, q_heads, head_dim,
                                rotary_dim);
  reference_rope_forward_layout(expected_k, cos, sin, batch_size, seq_len,
                                cos_batch_size, table_width, kv_heads, head_dim,
                                rotary_dim);

  __nv_bfloat16 *d_q = nullptr;
  __nv_bfloat16 *d_k = nullptr;
  float *d_cos = nullptr;
  float *d_sin = nullptr;
  CHECK_CUDA(cudaMalloc(&d_q, static_cast<size_t>(q_count) * sizeof(q[0])));
  CHECK_CUDA(cudaMalloc(&d_k, static_cast<size_t>(k_count) * sizeof(k[0])));
  CHECK_CUDA(cudaMalloc(&d_cos, static_cast<size_t>(table_count) * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_sin, static_cast<size_t>(table_count) * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(d_q, q.data(),
                        static_cast<size_t>(q_count) * sizeof(q[0]),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_k, k.data(),
                        static_cast<size_t>(k_count) * sizeof(k[0]),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_cos, cos.data(),
                        static_cast<size_t>(table_count) * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_sin, sin.data(),
                        static_cast<size_t>(table_count) * sizeof(float),
                        cudaMemcpyHostToDevice));

  CHECK_CUDA(gemma4_rope_forward_bf16(
      d_q, d_k, d_cos, d_sin, seq_len, batch_size, cos_batch_size, q_heads,
      kv_heads, head_dim, rotary_dim, 0));
  CHECK_CUDA(cudaMemcpy(q.data(), d_q,
                        static_cast<size_t>(q_count) * sizeof(q[0]),
                        cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(k.data(), d_k,
                        static_cast<size_t>(k_count) * sizeof(k[0]),
                        cudaMemcpyDeviceToHost));

  compare_bf16(q, expected_q, 0.0078125f, "forward rope q");
  compare_bf16(k, expected_k, 0.0078125f, "forward rope k");

  CHECK_CUDA(cudaFree(d_q));
  CHECK_CUDA(cudaFree(d_k));
  CHECK_CUDA(cudaFree(d_cos));
  CHECK_CUDA(cudaFree(d_sin));
}

}  // namespace

int main() {
  run_case(2, 7, 1, GEMMA4_NUM_QUERY_HEADS, GEMMA4_SLIDING_KV_HEADS,
           GEMMA4_SLIDING_HEAD_DIM, GEMMA4_SLIDING_HEAD_DIM);
  run_case(2, 5, 2, GEMMA4_NUM_QUERY_HEADS, GEMMA4_GLOBAL_KV_HEADS,
           GEMMA4_GLOBAL_HEAD_DIM, 128);
  run_forward_case(2, 7, 1, GEMMA4_NUM_QUERY_HEADS,
                   GEMMA4_SLIDING_KV_HEADS, GEMMA4_SLIDING_HEAD_DIM,
                   GEMMA4_SLIDING_HEAD_DIM);
  run_forward_case(2, 5, 2, GEMMA4_NUM_QUERY_HEADS, GEMMA4_GLOBAL_KV_HEADS,
                   GEMMA4_GLOBAL_HEAD_DIM, 128);
  cudaError_t invalid = gemma4_rope_bf16(
      nullptr, 0, nullptr, 0, nullptr, 0, nullptr, 0, 1, 1, 1, 1, 1,
      GEMMA4_SLIDING_HEAD_DIM, 127, 0);
  if (invalid != cudaErrorInvalidValue) {
    std::fprintf(stderr, "expected cudaErrorInvalidValue for invalid RoPE args\n");
    return 1;
  }

  std::printf("rope tests passed\n");
  return 0;
}
