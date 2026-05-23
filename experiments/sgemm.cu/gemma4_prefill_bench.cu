#include "gemma4.h"
#include "sgemm.cuh"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace {

constexpr int kFillThreads = 256;
constexpr int kDefaultWarmup = 20;
constexpr int kDefaultIters = 100;

struct Gemma4PrefillOp {
  const char *name;
  int k;
  int n;
};

constexpr Gemma4PrefillOp kPrefillOps[] = {
    {"ffn_gate_up", GEMMA4_HIDDEN_SIZE, GEMMA4_PACKED_FFN_SIZE},
    {"ffn_down", GEMMA4_INTERMEDIATE_SIZE, GEMMA4_HIDDEN_SIZE},
    {"sliding_qkv", GEMMA4_HIDDEN_SIZE, GEMMA4_SLIDING_QKV_SIZE},
    {"sliding_o", GEMMA4_SLIDING_ATTENTION_OUT_SIZE, GEMMA4_HIDDEN_SIZE},
    {"global_q", GEMMA4_HIDDEN_SIZE, GEMMA4_GLOBAL_Q_PROJ_SIZE},
    {"global_k", GEMMA4_HIDDEN_SIZE, GEMMA4_GLOBAL_K_PROJ_SIZE},
    {"global_o", GEMMA4_GLOBAL_ATTENTION_OUT_SIZE, GEMMA4_HIDDEN_SIZE},
    {"final_logits", GEMMA4_HIDDEN_SIZE, GEMMA4_VOCAB_SIZE},
};

inline void cuda_check(cudaError_t status, const char *expr, const char *file, int line) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "%s:%d: CUDA error for %s: %s\n", file, line, expr,
                 cudaGetErrorString(status));
    std::exit(1);
  }
}

inline void cublas_check(cublasStatus_t status, const char *expr, const char *file, int line) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::fprintf(stderr, "%s:%d: cuBLAS error for %s: %d\n", file, line, expr,
                 int(status));
    std::exit(1);
  }
}

#define CUDA_CHECK(expr) cuda_check((expr), #expr, __FILE__, __LINE__)
#define CUBLAS_CHECK(expr) cublas_check((expr), #expr, __FILE__, __LINE__)

template <typename T>
class DeviceBuffer {
 public:
  explicit DeviceBuffer(size_t count) : count_(count) {
    if (count_ > 0) {
      CUDA_CHECK(cudaMalloc(&ptr_, count_ * sizeof(T)));
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
  size_t count() const { return count_; }

 private:
  T *ptr_ = nullptr;
  size_t count_ = 0;
};

__device__ uint32_t mix_u32(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}

__global__ void fill_float_kernel(float *ptr, size_t count, uint64_t seed, float scale) {
  const size_t idx = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
  if (idx >= count) {
    return;
  }
  uint32_t x = uint32_t(idx) ^ uint32_t(idx >> 32) ^ uint32_t(seed) ^
               uint32_t(seed >> 32);
  x = mix_u32(x);
  const float u = float(x >> 8) * (1.0f / 16777216.0f);
  ptr[idx] = (u * 2.0f - 1.0f) * scale;
}

void fill_float(float *ptr, size_t count, uint64_t seed, float scale) {
  const int blocks = int((count + kFillThreads - 1) / kFillThreads);
  fill_float_kernel<<<blocks, kFillThreads>>>(ptr, count, seed, scale);
  CUDA_CHECK(cudaGetLastError());
}

void cublas_prefill(cublasHandle_t handle, const float *x, const float *w_row_major,
                    float *y, int m, int k, int n) {
  const float alpha = 1.0f;
  const float beta = 0.0f;
  CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha,
                           w_row_major, n, x, k, &beta, y, n));
}

void custom_prefill(const float *x, const float *w_row_major, float *y, int m, int k,
                    int n) {
  const float alpha = 1.0f;
  const float beta = 0.0f;
  sgemm(m, n, k, &alpha, const_cast<float *>(x), k, const_cast<float *>(w_row_major),
        n, &beta, y, n);
  CUDA_CHECK(cudaGetLastError());
}

template <typename Fn>
float time_cuda(Fn fn, int warmup, int iters) {
  for (int i = 0; i < warmup; ++i) {
    fn();
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i) {
    fn();
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return ms / float(iters);
}

float max_abs_diff(const std::vector<float> &actual, const std::vector<float> &expected) {
  float max_abs = 0.0f;
  for (size_t i = 0; i < actual.size(); ++i) {
    max_abs = std::max(max_abs, std::fabs(actual[i] - expected[i]));
  }
  return max_abs;
}

const Gemma4PrefillOp *find_op(const char *name) {
  for (const Gemma4PrefillOp &op : kPrefillOps) {
    if (std::strcmp(op.name, name) == 0) {
      return &op;
    }
  }
  return nullptr;
}

std::vector<int> default_m_sweep() {
  return {16, 64, 256, 1024};
}

std::vector<int> parse_m_sweep(const char *arg) {
  std::vector<int> values;
  const char *p = arg;
  while (*p != '\0') {
    char *end = nullptr;
    const long value = std::strtol(p, &end, 10);
    if (end == p || value <= 0 || value > std::numeric_limits<int>::max()) {
      std::fprintf(stderr, "invalid M sweep value near '%s'\n", p);
      std::exit(1);
    }
    values.push_back(int(value));
    p = *end == ',' ? end + 1 : end;
  }
  return values;
}

void print_usage(const char *argv0) {
  std::fprintf(stderr,
               "usage: %s [op|all] [iters] [warmup] [m_csv]\n"
               "example: %s global_k 100 20 16,64,256,1024\n",
               argv0, argv0);
}

void run_case(const Gemma4PrefillOp &op, int m, int iters, int warmup,
              cublasHandle_t handle) {
  DeviceBuffer<float> x(size_t(m) * op.k);
  DeviceBuffer<float> w(size_t(op.k) * op.n);
  DeviceBuffer<float> custom_y(size_t(m) * op.n);
  DeviceBuffer<float> cublas_y(size_t(m) * op.n);
  std::vector<float> h_custom(size_t(m) * op.n);
  std::vector<float> h_cublas(size_t(m) * op.n);

  const uint64_t seed = 0x20260520ull ^ (uint64_t(op.k) << 32) ^ uint64_t(op.n) ^
                        uint64_t(m);
  fill_float(x.get(), x.count(), seed, 0.2f);
  fill_float(w.get(), w.count(), seed + 0x9e3779b97f4a7c15ull, 0.15f);
  CUDA_CHECK(cudaMemset(custom_y.get(), 0, custom_y.count() * sizeof(float)));
  CUDA_CHECK(cudaMemset(cublas_y.get(), 0, cublas_y.count() * sizeof(float)));

  custom_prefill(x.get(), w.get(), custom_y.get(), m, op.k, op.n);
  cublas_prefill(handle, x.get(), w.get(), cublas_y.get(), m, op.k, op.n);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(h_custom.data(), custom_y.get(), h_custom.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_cublas.data(), cublas_y.get(), h_cublas.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));
  const float diff = max_abs_diff(h_custom, h_cublas);

  const float cublas_ms =
      time_cuda([&]() { cublas_prefill(handle, x.get(), w.get(), cublas_y.get(),
                                       m, op.k, op.n); },
                warmup, iters);
  const float custom_ms =
      time_cuda([&]() { custom_prefill(x.get(), w.get(), custom_y.get(), m, op.k,
                                       op.n); },
                warmup, iters);
  const double flops = 2.0 * double(m) * double(op.k) * double(op.n);
  std::printf("%-14s M=%4d K=%5d N=%6d cublas_ms=%8.4f custom_ms=%8.4f "
              "cublas_tflops=%7.2f custom_tflops=%7.2f speedup=%6.3fx "
              "max_abs=%g\n",
              op.name, m, op.k, op.n, cublas_ms, custom_ms,
              flops / (double(cublas_ms) * 1.0e9),
              flops / (double(custom_ms) * 1.0e9), double(cublas_ms) / custom_ms,
              diff);
}

}  // namespace

int main(int argc, char **argv) {
  const char *selected_op = argc > 1 ? argv[1] : "all";
  const int iters = argc > 2 ? std::atoi(argv[2]) : kDefaultIters;
  const int warmup = argc > 3 ? std::atoi(argv[3]) : kDefaultWarmup;
  const std::vector<int> m_sweep = argc > 4 ? parse_m_sweep(argv[4]) : default_m_sweep();

  if (iters <= 0 || warmup < 0 ||
      (std::strcmp(selected_op, "all") != 0 && find_op(selected_op) == nullptr)) {
    print_usage(argv[0]);
    return 1;
  }

  cublasHandle_t handle = nullptr;
  CUBLAS_CHECK(cublasCreate(&handle));

  std::printf("Gemma 4 SGEMM prefill benchmark, iters=%d warmup=%d\n", iters,
              warmup);
  std::printf("This SGEMM source is FP32, so this is a shape/tile baseline, not the "
              "final Gemma BF16 datatype path.\n");

  for (const Gemma4PrefillOp &op : kPrefillOps) {
    if (std::strcmp(selected_op, "all") != 0 && std::strcmp(selected_op, op.name) != 0) {
      continue;
    }
    for (int m : m_sweep) {
      run_case(op, m, iters, warmup, handle);
    }
  }

  CUBLAS_CHECK(cublasDestroy(handle));
  return 0;
}
