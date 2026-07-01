#include "gemma4_kv_cache.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

#include <cmath>
#include <cstdint>
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

// Convert BF16 test values back to float for comparisons.
float bf16_to_float(__nv_bfloat16 value) { return __bfloat162float(value); }

// Generate deterministic nontrivial BF16 values without storing fixtures.
__nv_bfloat16 make_value(int seed) {
  int centered = ((seed * 37 + 17) % 257) - 128;
  return __float2bfloat16_rn(static_cast<float>(centered) / 64.0f);
}

// Return total BF16 slots in the paged K/V cache layout.
int64_t cache_elements(const Gemma4KvCacheConfig &config) {
  return int64_t(config.num_layers) * config.num_pages * config.page_size *
         config.num_heads * config.head_dim;
}

// Copy one host vector into an equally sized device buffer.
template <typename T>
void copy_to_device(thrust::device_vector<T> &dst, const std::vector<T> &src) {
  CHECK_CUDA(cudaMemcpy(raw_ptr(dst), src.data(), src.size() * sizeof(T),
                        cudaMemcpyHostToDevice));
}

// Copy one full device buffer back into a host vector.
template <typename T>
std::vector<T> copy_to_host(const thrust::device_vector<T> &src) {
  std::vector<T> dst(src.size());
  CHECK_CUDA(cudaMemcpy(dst.data(), raw_ptr(src), dst.size() * sizeof(T),
                        cudaMemcpyDeviceToHost));
  return dst;
}

// Compare BF16 vectors with an absolute tolerance.
void compare_bf16(const std::vector<__nv_bfloat16> &actual,
                  const std::vector<__nv_bfloat16> &expected,
                  float tolerance,
                  const char *label) {
  float max_abs = 0.0f;
  int max_index = 0;
  for (int i = 0; i < static_cast<int>(actual.size()); ++i) {
    float diff = std::fabs(bf16_to_float(actual[i]) -
                           bf16_to_float(expected[i]));
    if (diff > max_abs) {
      max_abs = diff;
      max_index = i;
    }
  }
  if (max_abs > tolerance) {
    std::fprintf(stderr,
                 "%s max_abs=%g index=%d actual=%g expected=%g tolerance=%g\n",
                 label, max_abs, max_index, bf16_to_float(actual[max_index]),
                 bf16_to_float(expected[max_index]), tolerance);
    std::exit(1);
  }
}

// Exercise out-of-range batch rows for the vector writer.
void run_invalid_page_write_case() {
  Gemma4KvCacheConfig config = {1, 2, 4, 2, 1, 2, 16, 0};
  std::vector<int32_t> page_table(config.max_pages_per_seq, -1);
  std::vector<int32_t> token_batch = {1};
  std::vector<int32_t> token_position = {0};

  std::vector<__nv_bfloat16> k(config.num_heads * config.head_dim);
  std::vector<__nv_bfloat16> v(k.size());
  for (int i = 0; i < static_cast<int>(k.size()); ++i) {
    k[i] = make_value(43000 + i);
    v[i] = make_value(44000 + i);
  }

  thrust::device_vector<__nv_bfloat16> d_cache_k(cache_elements(config));
  thrust::device_vector<__nv_bfloat16> d_cache_v(cache_elements(config));
  thrust::device_vector<__nv_bfloat16> d_k(k.size());
  thrust::device_vector<__nv_bfloat16> d_v(v.size());
  thrust::device_vector<int32_t> d_page_table(page_table.size());
  thrust::device_vector<int32_t> d_token_batch(token_batch.size());
  thrust::device_vector<int32_t> d_token_position(token_position.size());
  CHECK_CUDA(cudaMemset(raw_ptr(d_cache_k), 0,
                        cache_elements(config) * sizeof(__nv_bfloat16)));
  CHECK_CUDA(cudaMemset(raw_ptr(d_cache_v), 0,
                        cache_elements(config) * sizeof(__nv_bfloat16)));
  copy_to_device(d_k, k);
  copy_to_device(d_v, v);
  copy_to_device(d_page_table, page_table);
  copy_to_device(d_token_batch, token_batch);
  copy_to_device(d_token_position, token_position);

  CHECK_CUDA(gemma4_kv_cache_write_bf16(
      raw_ptr(d_cache_k), raw_ptr(d_cache_v), config, raw_ptr(d_page_table),
      raw_ptr(d_token_batch), raw_ptr(d_token_position), 1, 0, raw_ptr(d_k), raw_ptr(d_v), 0));
  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<__nv_bfloat16> zero(cache_elements(config),
                                  __float2bfloat16_rn(0.0f));
  compare_bf16(copy_to_host(d_cache_k), zero, 0.0f, "invalid page cache K");
  compare_bf16(copy_to_host(d_cache_v), zero, 0.0f, "invalid page cache V");
}

}  // namespace

int main() {
  run_invalid_page_write_case();
  std::puts("kv cache tests passed");
  return 0;
}
