#include "gemma4_bench_utils.cuh"
#include "gemma4_ffn_decode.cuh"
#include "gemma4.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>

namespace {

constexpr int kHidden = GEMMA4_HIDDEN_SIZE;
constexpr int kIntermediate = GEMMA4_INTERMEDIATE_SIZE;

__device__ uint32_t mix_u32(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}

__global__ void fill_random_bf16_kernel(__nv_bfloat16 *ptr,
                                        size_t count,
                                        uint64_t seed,
                                        float scale) {
  const size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
  if (i >= count) {
    return;
  }

  uint32_t x = uint32_t(i) ^ uint32_t(i >> 32) ^ uint32_t(seed) ^
               uint32_t(seed >> 32);
  x = mix_u32(x);
  const float u = float(x >> 8) * (1.0f / 16777216.0f);
  ptr[i] = __float2bfloat16_rn((u * 2.0f - 1.0f) * scale);
}

__global__ void flush_cache_kernel(const uint32_t *__restrict__ in,
                                   uint32_t *__restrict__ out,
                                   size_t count) {
  uint32_t acc = 0;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
       i += size_t(blockDim.x) * gridDim.x) {
    acc ^= in[i] + uint32_t(i);
  }
  out[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}

template <typename T>
class DeviceBuffer {
 public:
  explicit DeviceBuffer(size_t count) {
    CUDA_CHECK(cudaMalloc(&ptr_, count * sizeof(T)));
  }

  ~DeviceBuffer() {
    if (ptr_ != nullptr) {
      cudaFree(ptr_);
    }
  }

  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;

  operator T *() { return ptr_; }
  operator const T *() const { return ptr_; }

 private:
  T *ptr_ = nullptr;
};

void fill_random_bf16(__nv_bfloat16 *ptr,
                      size_t count,
                      uint64_t seed,
                      float scale,
                      cudaStream_t stream) {
  constexpr int threads = 256;
  const int blocks = int((count + threads - 1) / threads);
  fill_random_bf16_kernel<<<blocks, threads, 0, stream>>>(
      ptr, count, seed, scale);
  CUDA_CHECK(cudaGetLastError());
}

void flush_cache(const uint32_t *in,
                 uint32_t *out,
                 size_t count,
                 cudaStream_t stream) {
  constexpr int threads = 256;
  constexpr int blocks = 4096;
  flush_cache_kernel<<<blocks, threads, 0, stream>>>(in, out, count);
  CUDA_CHECK(cudaGetLastError());
}

uint64_t make_seed() {
  if (const char *env = std::getenv("GEMMA4_FFN_LOAD_BENCH_SEED")) {
    return std::strtoull(env, nullptr, 0);
  }

  std::random_device rd;
  uint64_t seed = uint64_t(rd()) << 32;
  seed ^= uint64_t(rd());
  seed ^= uint64_t(std::chrono::high_resolution_clock::now()
                       .time_since_epoch()
                       .count());
  return seed;
}

}  // namespace

int main(int argc, char **argv) {
  const int iters = argc > 1 ? std::atoi(argv[1]) : 30;
  const int warmup = argc > 2 ? std::atoi(argv[2]) : 8;
  const int trials = argc > 3 ? std::atoi(argv[3]) : 3;
  if (iters <= 0 || warmup < 0 || trials <= 0) {
    std::fprintf(stderr, "usage: %s [iters=30] [warmup=8] [trials=3]\n",
                 argv[0]);
    return 1;
  }

  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  const uint64_t seed = make_seed();
  const size_t x_elems = kHidden;
  const size_t gate_up_elems =
      static_cast<size_t>(kHidden) * GEMMA4_PACKED_FFN_SIZE;
  const size_t down_elems = static_cast<size_t>(kIntermediate) * kHidden;

  DeviceBuffer<__nv_bfloat16> d_x(x_elems);
  DeviceBuffer<__nv_bfloat16> d_residual(x_elems);
  DeviceBuffer<__nv_bfloat16> d_rms_weight(x_elems);
  DeviceBuffer<__nv_bfloat16> d_residual_out(x_elems);
  DeviceBuffer<__nv_bfloat16> d_normed_out(x_elems);
  DeviceBuffer<__nv_bfloat16> d_gate_up_src(gate_up_elems);
  DeviceBuffer<__nv_bfloat16> d_gate_up(gate_up_elems);
  DeviceBuffer<__nv_bfloat16> d_down_src(down_elems);
  DeviceBuffer<__nv_bfloat16> d_down(down_elems);
  DeviceBuffer<Gemma4FfnDecodeScratch> d_scratch(1);

  fill_random_bf16(d_x, x_elems, seed ^ 0x1001u, 0.05f, stream);
  fill_random_bf16(d_residual, x_elems, seed ^ 0x2002u, 0.05f, stream);
  fill_random_bf16(d_rms_weight, x_elems, seed ^ 0x3003u, 1.0f, stream);
  fill_random_bf16(d_gate_up_src, gate_up_elems, seed ^ 0x4004u, 0.01f,
                   stream);
  fill_random_bf16(d_down_src, down_elems, seed ^ 0x5005u, 0.01f, stream);
  CUDA_CHECK(gemma4_ffn_decode_swizzle_weights_bf16(
      d_gate_up, d_gate_up_src, d_down, d_down_src, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(gemma4_ffn_decode_configure_scratch_l2(d_scratch, stream));

  int device = 0;
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDevice(&device));
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  const size_t flush_bytes =
      std::max<size_t>(256ull * 1024ull * 1024ull,
                       static_cast<size_t>(prop.l2CacheSize) * 4ull);
  const size_t flush_count = flush_bytes / sizeof(uint32_t);
  DeviceBuffer<uint32_t> d_flush_in(flush_count);
  DeviceBuffer<uint32_t> d_flush_out(size_t(4096) * 256);
  CUDA_CHECK(cudaMemsetAsync(d_flush_in, 0x5a, flush_bytes, stream));
  CUDA_CHECK(cudaMemsetAsync(d_flush_out, 0,
                             size_t(4096) * 256 * sizeof(uint32_t), stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  auto run_custom = [&]() {
    CUDA_CHECK(gemma4_ffn_decode_fused_bf16(
        d_residual_out, d_normed_out, d_x, d_residual, d_rms_weight,
        d_gate_up, d_down, d_scratch, GEMMA4_RMS_NORM_EPS, stream));
  };
  auto run_clear = [&]() {
    auto *scratch = static_cast<Gemma4FfnDecodeScratch *>(d_scratch);
    CUDA_CHECK(cudaMemsetAsync(scratch, 0, sizeof(*scratch), stream));
  };
  auto run_cold_custom = [&]() {
    flush_cache(d_flush_in, d_flush_out, flush_count, stream);
    run_custom();
  };
  auto run_cold_clear = [&]() {
    flush_cache(d_flush_in, d_flush_out, flush_count, stream);
    run_clear();
  };

  run_custom();
  CUDA_CHECK(cudaStreamSynchronize(stream));

  const TimingStats custom = time_ms_graph(run_custom, stream, warmup,
                                           iters, trials);
  const TimingStats clear = time_ms_graph(run_clear, stream, warmup,
                                          iters, trials);
  const TimingStats cold_custom = time_ms(run_cold_custom, stream, warmup,
                                          iters, trials);
  const TimingStats cold_clear = time_ms(run_cold_clear, stream, warmup,
                                         iters, trials);
  const TimingStats flush = time_ms(
      [&]() { flush_cache(d_flush_in, d_flush_out, flush_count, stream); },
      stream, warmup, iters, trials);

  const float warm_minus_clear = std::max(custom.best_ms - clear.best_ms, 0.0f);
  const float cold_minus_flush =
      std::max(cold_custom.best_ms - flush.best_ms, 0.0f);
  const float cold_clear_minus_flush =
      std::max(cold_clear.best_ms - flush.best_ms, 0.0f);
  const float cold_minus_flush_clear =
      std::max(cold_minus_flush - cold_clear_minus_flush, 0.0f);

  std::printf("device=%s\n", prop.name);
  std::printf("seed=0x%llx,iters=%d,warmup=%d,trials=%d\n",
              static_cast<unsigned long long>(seed), iters, warmup, trials);
  std::printf("flush_bytes=%zu\n", flush_bytes);
  std::printf("metric,best_ms,avg_ms\n");
  std::printf("custom_graph,%.6f,%.6f\n", custom.best_ms, custom.avg_ms);
  std::printf("scratch_clear_graph,%.6f,%.6f\n", clear.best_ms, clear.avg_ms);
  std::printf("warm_minus_clear,%.6f,%.6f\n", warm_minus_clear,
              std::max(custom.avg_ms - clear.avg_ms, 0.0f));
  std::printf("cold_custom,%.6f,%.6f\n", cold_custom.best_ms,
              cold_custom.avg_ms);
  std::printf("cold_clear,%.6f,%.6f\n", cold_clear.best_ms,
              cold_clear.avg_ms);
  std::printf("flush_only,%.6f,%.6f\n", flush.best_ms, flush.avg_ms);
  std::printf("cold_minus_flush_clear,%.6f,%.6f\n",
              cold_minus_flush_clear,
              std::max(cold_custom.avg_ms - flush.avg_ms -
                           std::max(cold_clear.avg_ms - flush.avg_ms, 0.0f),
                       0.0f));

  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}
