#include "gemma4_rmsnorm.cuh"
#include "gemma4_bench_utils.cuh"
#include "gemma4.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace {

__device__ uint32_t mix_u32_device(uint32_t x) {
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
  size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
  if (i >= count) {
    return;
  }

  uint32_t x = uint32_t(i) ^ uint32_t(i >> 32) ^ uint32_t(seed) ^
               uint32_t(seed >> 32);
  x = mix_u32_device(x);
  float u = float(x >> 8) * (1.0f / 16777216.0f);
  ptr[i] = __float2bfloat16_rn((u * 2.0f - 1.0f) * scale);
}

uint64_t make_seed() {
  if (const char *env = std::getenv("GEMMA4_RMSNORM_BENCH_SEED")) {
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

void fill_random_bf16(__nv_bfloat16 *ptr,
                      size_t count,
                      uint64_t seed,
                      float scale,
                      cudaStream_t stream) {
  constexpr int threads = 256;
  int blocks = int((count + threads - 1) / threads);
  fill_random_bf16_kernel<<<blocks, threads, 0, stream>>>(
      ptr, count, seed, scale);
  CUDA_CHECK(cudaGetLastError());
}

std::vector<int> row_counts_up_to(int max_rows) {
  std::vector<int> counts;
  for (int rows : {4, 16, 64, 256, 1024, 4096, 8192}) {
    if (rows <= max_rows) {
      counts.push_back(rows);
    }
  }
  if (counts.empty() || counts.back() != max_rows) {
    counts.push_back(max_rows);
  }
  return counts;
}

double gib_per_second(double bytes, float ms) {
  double gib = bytes / (1024.0 * 1024.0 * 1024.0);
  return gib / (static_cast<double>(ms) / 1000.0);
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

}  // namespace

int main(int argc, char **argv) {
  const int iters = argc > 1 ? std::atoi(argv[1]) : 200;
  const int warmup = argc > 2 ? std::atoi(argv[2]) : 20;
  const int trials = argc > 3 ? std::atoi(argv[3]) : 5;
  const int max_rows = argc > 4 ? std::atoi(argv[4]) : 1024;

  if (iters <= 0 || warmup < 0 || trials <= 0 || max_rows <= 1) {
    std::fprintf(stderr,
                 "usage: %s [iters=200] [warmup=20] [trials=5] "
                 "[max_rows=1024]\n",
                 argv[0]);
    return 1;
  }

  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  const int width = GEMMA4_HIDDEN_SIZE;
  const size_t max_elems = static_cast<size_t>(max_rows) * width;
  const uint64_t seed = make_seed();

  DeviceBuffer<__nv_bfloat16> d_inp1(max_elems);
  DeviceBuffer<__nv_bfloat16> d_inp2(max_elems);
  DeviceBuffer<__nv_bfloat16> d_weight(width);
  DeviceBuffer<__nv_bfloat16> d_residual(max_elems);
  DeviceBuffer<__nv_bfloat16> d_normed(max_elems);
  DeviceBuffer<float> d_rstd(max_rows);

  fill_random_bf16(d_inp1, max_elems, seed ^ 0x1001u, 1.0f, stream);
  fill_random_bf16(d_inp2, max_elems, seed ^ 0x2002u, 1.0f, stream);
  fill_random_bf16(d_weight, width, seed ^ 0x3003u, 0.5f, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  int device = 0;
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDevice(&device));
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

  std::printf("device=%s\n", prop.name);
  std::printf("shape=width%d,max_rows=%d,seed=0x%llx\n", width, max_rows,
              static_cast<unsigned long long>(seed));
  std::printf("iters=%d,warmup_graph_launches=%d,trials=%d\n", iters, warmup,
              trials);
  std::printf("rows,fused_graph_best_ms,fused_graph_avg_ms,"
              "fused_graph_gib_s\n");

  for (int rows : row_counts_up_to(max_rows)) {
    const double bytes =
        double(rows) * width * sizeof(__nv_bfloat16) * 5.0 +
        double(rows) * sizeof(float);
    auto run_fused = [&]() {
      CUDA_CHECK(gemma4_residual_add_rmsnorm_bf16(
          d_residual, d_normed, d_rstd, d_inp1, d_inp2, d_weight, rows, width,
          GEMMA4_RMS_NORM_EPS, stream));
    };

    run_fused();
    CUDA_CHECK(cudaStreamSynchronize(stream));

    TimingStats graph_stats =
        time_ms_graph(run_fused, stream, warmup, iters, trials);
    std::printf("%d,%.6f,%.6f,%.3f\n", rows, graph_stats.best_ms,
                graph_stats.avg_ms,
                gib_per_second(bytes, graph_stats.best_ms));
  }

  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}
