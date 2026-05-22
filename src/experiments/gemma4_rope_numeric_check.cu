#include "gemma4_rope.cuh"
#include "gemma4.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

namespace {

struct Shape {
  const char *name;
  int q_heads;
  int kv_heads;
  int head_dim;
  int rotary_dim;
};

struct Metrics {
  uint64_t count = 0;
  uint64_t gpu_eq_fp32 = 0;
  uint64_t gpu_eq_double = 0;
  uint64_t bf16_eq_double = 0;
  uint64_t nope_exact = 0;
  uint64_t nope_count = 0;
  double gpu_abs_sum = 0.0;
  double bf16_abs_sum = 0.0;
  double gpu_ulp_sum = 0.0;
  double bf16_ulp_sum = 0.0;
  int gpu_max_ulp = 0;
  int bf16_max_ulp = 0;
  double gpu_max_abs = 0.0;
  double bf16_max_abs = 0.0;
};

void check_cuda(cudaError_t status, const char *expr, const char *file, int line) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "%s:%d: CUDA error for %s: %s\n", file, line, expr,
                 cudaGetErrorString(status));
    std::exit(1);
  }
}

#define CUDA_CHECK(expr) check_cuda((expr), #expr, __FILE__, __LINE__)

uint16_t bf16_bits(__nv_bfloat16 value) {
  uint16_t bits;
  static_assert(sizeof(bits) == sizeof(value), "unexpected bf16 size");
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

__nv_bfloat16 bf16_from_bits(uint16_t bits) {
  __nv_bfloat16 value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

float bf16_to_float(uint16_t bits) {
  return __bfloat162float(bf16_from_bits(bits));
}

uint16_t round_bf16_bits(float value) {
  return bf16_bits(__float2bfloat16_rn(value));
}

int ordered_bf16(uint16_t bits) {
  return (bits & 0x8000u) ? static_cast<int>((~bits) & 0xffffu)
                          : static_cast<int>(bits | 0x8000u);
}

int bf16_ulp_diff(uint16_t a, uint16_t b) {
  return std::abs(ordered_bf16(a) - ordered_bf16(b));
}

uint16_t fp32_fma_rope_bits(float x1, float x2, float c, float s, bool high_half) {
  float product = high_half ? x2 * c : x1 * c;
  float value = high_half ? std::fma(x1, s, product)
                          : std::fma(-x2, s, product);
  return round_bf16_bits(value);
}

uint16_t double_rope_bits(float x1, float x2, float c, float s, bool high_half) {
  double xd1 = static_cast<double>(x1);
  double xd2 = static_cast<double>(x2);
  double cd = static_cast<double>(c);
  double sd = static_cast<double>(s);
  double value = high_half ? xd2 * cd + xd1 * sd : xd1 * cd - xd2 * sd;
  return round_bf16_bits(static_cast<float>(value));
}

float round_bf16_float(float value) {
  return __bfloat162float(__float2bfloat16_rn(value));
}

uint16_t bf16_intermediate_rope_bits(
    float x1, float x2, float c, float s, bool high_half) {
  float product_a = high_half ? round_bf16_float(x2 * c)
                              : round_bf16_float(x1 * c);
  float product_b = high_half ? round_bf16_float(x1 * s)
                              : round_bf16_float(-x2 * s);
  return round_bf16_bits(round_bf16_float(product_a + product_b));
}

void update_metric(Metrics &metrics,
                   uint16_t gpu,
                   uint16_t fp32_ref,
                   uint16_t double_ref,
                   uint16_t bf16_ref) {
  float gpu_value = bf16_to_float(gpu);
  float double_value = bf16_to_float(double_ref);
  float bf16_value = bf16_to_float(bf16_ref);
  int gpu_ulp = bf16_ulp_diff(gpu, double_ref);
  int bf16_ulp = bf16_ulp_diff(bf16_ref, double_ref);
  double gpu_abs = std::abs(static_cast<double>(gpu_value) - double_value);
  double bf16_abs = std::abs(static_cast<double>(bf16_value) - double_value);

  metrics.count += 1;
  metrics.gpu_eq_fp32 += gpu == fp32_ref;
  metrics.gpu_eq_double += gpu == double_ref;
  metrics.bf16_eq_double += bf16_ref == double_ref;
  metrics.gpu_ulp_sum += gpu_ulp;
  metrics.bf16_ulp_sum += bf16_ulp;
  metrics.gpu_abs_sum += gpu_abs;
  metrics.bf16_abs_sum += bf16_abs;
  metrics.gpu_max_ulp = std::max(metrics.gpu_max_ulp, gpu_ulp);
  metrics.bf16_max_ulp = std::max(metrics.bf16_max_ulp, bf16_ulp);
  metrics.gpu_max_abs = std::max(metrics.gpu_max_abs, gpu_abs);
  metrics.bf16_max_abs = std::max(metrics.bf16_max_abs, bf16_abs);
}

void compare_tensor(Metrics &metrics,
                    const std::vector<__nv_bfloat16> &actual,
                    const std::vector<__nv_bfloat16> &original,
                    const std::vector<float> &cos,
                    const std::vector<float> &sin,
                    int batch_size,
                    int seq_len,
                    int cos_batch_size,
                    int heads,
                    int head_dim,
                    int rotary_dim) {
  int rotary_half = rotary_dim / 2;
  int row_stride = heads * head_dim;
  for (int batch = 0; batch < batch_size; ++batch) {
    for (int seq = 0; seq < seq_len; ++seq) {
      int row = batch * seq_len + seq;
      int table_batch = cos_batch_size == 1 ? 0 : batch;
      int table_base = (table_batch * seq_len + seq) * rotary_half;
      for (int head = 0; head < heads; ++head) {
        int head_base = row * row_stride + head * head_dim;
        for (int i = 0; i < rotary_half; ++i) {
          float x1 = __bfloat162float(original[head_base + i]);
          float x2 = __bfloat162float(original[head_base + rotary_half + i]);
          float c = cos[table_base + i];
          float s = sin[table_base + i];

          uint16_t gpu_lo = bf16_bits(actual[head_base + i]);
          uint16_t gpu_hi = bf16_bits(actual[head_base + rotary_half + i]);
          update_metric(metrics, gpu_lo, fp32_fma_rope_bits(x1, x2, c, s, false),
                        double_rope_bits(x1, x2, c, s, false),
                        bf16_intermediate_rope_bits(x1, x2, c, s, false));
          update_metric(metrics, gpu_hi, fp32_fma_rope_bits(x1, x2, c, s, true),
                        double_rope_bits(x1, x2, c, s, true),
                        bf16_intermediate_rope_bits(x1, x2, c, s, true));
        }
        for (int i = rotary_dim; i < head_dim; ++i) {
          metrics.nope_count += 1;
          metrics.nope_exact +=
              bf16_bits(actual[head_base + i]) == bf16_bits(original[head_base + i]);
        }
      }
    }
  }
}

Metrics run_shape(const Shape &shape,
                  int batch_size,
                  int seq_len,
                  int cos_batch_size,
                  int trials,
                  uint64_t seed) {
  Metrics metrics;
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<float> value_dist(-16.0f, 16.0f);
  std::uniform_real_distribution<float> angle_dist(-4096.0f, 4096.0f);

  size_t q_elems = static_cast<size_t>(batch_size) * seq_len * shape.q_heads *
                   shape.head_dim;
  size_t k_elems = static_cast<size_t>(batch_size) * seq_len * shape.kv_heads *
                   shape.head_dim;
  int rotary_half = shape.rotary_dim / 2;
  size_t table_elems =
      static_cast<size_t>(cos_batch_size) * seq_len * rotary_half;

  __nv_bfloat16 *d_q = nullptr;
  __nv_bfloat16 *d_k = nullptr;
  float *d_cos = nullptr;
  float *d_sin = nullptr;
  CUDA_CHECK(cudaMalloc(&d_q, q_elems * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&d_k, k_elems * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&d_cos, table_elems * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_sin, table_elems * sizeof(float)));

  std::vector<__nv_bfloat16> q(q_elems);
  std::vector<__nv_bfloat16> k(k_elems);
  std::vector<__nv_bfloat16> q_original(q_elems);
  std::vector<__nv_bfloat16> k_original(k_elems);
  std::vector<float> cos(table_elems);
  std::vector<float> sin(table_elems);

  for (int trial = 0; trial < trials; ++trial) {
    for (size_t i = 0; i < q_elems; ++i) {
      q[i] = __float2bfloat16_rn(value_dist(rng));
    }
    for (size_t i = 0; i < k_elems; ++i) {
      k[i] = __float2bfloat16_rn(value_dist(rng));
    }
    for (size_t i = 0; i < table_elems; ++i) {
      float angle = angle_dist(rng);
      cos[i] = std::cos(angle);
      sin[i] = std::sin(angle);
    }
    q_original = q;
    k_original = k;

    CUDA_CHECK(cudaMemcpy(d_q, q.data(), q_elems * sizeof(__nv_bfloat16),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k, k.data(), k_elems * sizeof(__nv_bfloat16),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cos, cos.data(), table_elems * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sin, sin.data(), table_elems * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(gemma4_rope_bf16(
        d_q, static_cast<int64_t>(shape.q_heads) * shape.head_dim, d_k,
        static_cast<int64_t>(shape.kv_heads) * shape.head_dim, d_cos,
        rotary_half, d_sin, rotary_half, seq_len, batch_size, cos_batch_size,
        shape.q_heads, shape.kv_heads, shape.head_dim, shape.rotary_dim, 0));
    CUDA_CHECK(cudaMemcpy(q.data(), d_q, q_elems * sizeof(__nv_bfloat16),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(k.data(), d_k, k_elems * sizeof(__nv_bfloat16),
                          cudaMemcpyDeviceToHost));

    compare_tensor(metrics, q, q_original, cos, sin, batch_size, seq_len,
                   cos_batch_size, shape.q_heads, shape.head_dim,
                   shape.rotary_dim);
    compare_tensor(metrics, k, k_original, cos, sin, batch_size, seq_len,
                   cos_batch_size, shape.kv_heads, shape.head_dim,
                   shape.rotary_dim);
  }

  CUDA_CHECK(cudaFree(d_q));
  CUDA_CHECK(cudaFree(d_k));
  CUDA_CHECK(cudaFree(d_cos));
  CUDA_CHECK(cudaFree(d_sin));
  return metrics;
}

void print_metrics(const Shape &shape, const Metrics &metrics) {
  double n = static_cast<double>(metrics.count);
  std::printf(
      "%s,count=%llu,gpu_eq_fp32=%.8f,gpu_eq_double=%.8f,"
      "bf16_intermediate_eq_double=%.8f,gpu_mean_ulp=%.8f,"
      "bf16_intermediate_mean_ulp=%.8f,gpu_max_ulp=%d,"
      "bf16_intermediate_max_ulp=%d,gpu_mean_abs=%.9g,"
      "bf16_intermediate_mean_abs=%.9g,gpu_max_abs=%.9g,"
      "bf16_intermediate_max_abs=%.9g,nope_exact=%.8f\n",
      shape.name, static_cast<unsigned long long>(metrics.count),
      metrics.gpu_eq_fp32 / n, metrics.gpu_eq_double / n,
      metrics.bf16_eq_double / n, metrics.gpu_ulp_sum / n,
      metrics.bf16_ulp_sum / n, metrics.gpu_max_ulp, metrics.bf16_max_ulp,
      metrics.gpu_abs_sum / n, metrics.bf16_abs_sum / n, metrics.gpu_max_abs,
      metrics.bf16_max_abs,
      metrics.nope_count == 0
          ? 1.0
          : static_cast<double>(metrics.nope_exact) / metrics.nope_count);
}

}  // namespace

int main(int argc, char **argv) {
  int trials = argc > 1 ? std::atoi(argv[1]) : 24;
  int seq_len = argc > 2 ? std::atoi(argv[2]) : 37;
  int batch_size = argc > 3 ? std::atoi(argv[3]) : 2;
  uint64_t seed = argc > 4 ? std::strtoull(argv[4], nullptr, 0) : 0x20260521ull;
  if (trials <= 0 || seq_len <= 0 || batch_size <= 0) {
    std::fprintf(stderr, "usage: %s [trials=24] [seq=37] [batch=2] [seed]\n",
                 argv[0]);
    return 1;
  }

  const Shape shapes[] = {
      {"sliding", GEMMA4_NUM_QUERY_HEADS, GEMMA4_SLIDING_KV_HEADS,
       GEMMA4_SLIDING_HEAD_DIM, GEMMA4_SLIDING_HEAD_DIM},
      {"global", GEMMA4_NUM_QUERY_HEADS, GEMMA4_GLOBAL_KV_HEADS,
       GEMMA4_GLOBAL_HEAD_DIM, GEMMA4_GLOBAL_HEAD_DIM / 4},
  };

  std::printf("trials=%d,seq=%d,batch=%d,seed=0x%llx\n", trials, seq_len,
              batch_size, static_cast<unsigned long long>(seed));
  for (const Shape &shape : shapes) {
    Metrics metrics =
        run_shape(shape, batch_size, seq_len, 1, trials, seed ^ shape.head_dim);
    print_metrics(shape, metrics);
  }
  return 0;
}
