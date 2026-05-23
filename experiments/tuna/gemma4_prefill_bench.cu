#include "gemma4.h"

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace {

using nvcuda::wmma::accumulator;
using nvcuda::wmma::col_major;
using nvcuda::wmma::fragment;
using nvcuda::wmma::load_matrix_sync;
using nvcuda::wmma::matrix_a;
using nvcuda::wmma::matrix_b;
using nvcuda::wmma::mem_row_major;
using nvcuda::wmma::mma_sync;
using nvcuda::wmma::row_major;
using nvcuda::wmma::store_matrix_sync;

constexpr int kWmmaM = 16;
constexpr int kWmmaN = 16;
constexpr int kWmmaK = 16;
constexpr int kThreadsPerWarp = 32;
constexpr int kFillThreads = 256;
constexpr int kDefaultWarmup = 20;
constexpr int kDefaultIters = 100;
constexpr float kCorrectnessTolerance = 0.125f;
constexpr size_t kCublasLtWorkspaceBytes = 32ull * 1024ull * 1024ull;
constexpr int kMaxCublasLtHeuristics = 64;

enum class CublasBaseline {
  GemmEx,
  Lt,
};

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

struct TunaTileConfig {
  const char *name;
  int warp_tiles_m;
  int warp_tiles_n;
  bool use_smem;
};

constexpr TunaTileConfig kTileConfigs[] = {
    {"wmma_16x16", 1, 1, false},
    {"wmma_16x32", 1, 2, false},
    {"wmma_16x64", 1, 4, false},
    {"wmma_32x64", 2, 4, false},
    {"wmma_64x64", 4, 4, false},
    {"smem_16x64", 1, 4, true},
    {"smem_16x128", 1, 8, true},
    {"smem_32x64", 2, 4, true},
    {"smem_32x128", 2, 8, true},
    {"smem_64x64", 4, 4, true},
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

cublasGemmAlgo_t selected_cublas_algo() {
  const char *env = std::getenv("GEMMA4_PREFILL_CUBLAS_ALGO");
  if (env == nullptr || std::strcmp(env, "default_tensor") == 0) {
    return CUBLAS_GEMM_DEFAULT_TENSOR_OP;
  }
  if (std::strcmp(env, "default") == 0) {
    return CUBLAS_GEMM_DEFAULT;
  }
  if (std::strncmp(env, "algo", 4) == 0) {
    const int algo = std::atoi(env + 4);
    if (algo >= 0 && algo <= 15) {
      return static_cast<cublasGemmAlgo_t>(CUBLAS_GEMM_ALGO0_TENSOR_OP + algo);
    }
  }
  std::fprintf(stderr,
               "unsupported GEMMA4_PREFILL_CUBLAS_ALGO=%s "
               "(use default, default_tensor, or algo0..algo15)\n",
               env);
  std::exit(1);
}

const char *selected_cublas_algo_name() {
  const char *env = std::getenv("GEMMA4_PREFILL_CUBLAS_ALGO");
  return env == nullptr ? "default_tensor" : env;
}

CublasBaseline selected_cublas_baseline() {
  const char *env = std::getenv("GEMMA4_PREFILL_CUBLAS_BACKEND");
  if (env == nullptr || std::strcmp(env, "gemmex") == 0) {
    return CublasBaseline::GemmEx;
  }
  if (std::strcmp(env, "lt") == 0) {
    return CublasBaseline::Lt;
  }
  std::fprintf(stderr,
               "unsupported GEMMA4_PREFILL_CUBLAS_BACKEND=%s "
               "(use gemmex or lt)\n",
               env);
  std::exit(1);
}

const char *selected_cublas_baseline_name() {
  return selected_cublas_baseline() == CublasBaseline::Lt ? "lt" : "gemmex";
}

int selected_graph_repeats() {
  const char *env = std::getenv("GEMMA4_PREFILL_GRAPH_REPEATS");
  if (env == nullptr) {
    return 0;
  }
  const int repeats = std::atoi(env);
  if (repeats < 0) {
    std::fprintf(stderr, "GEMMA4_PREFILL_GRAPH_REPEATS must be >= 0\n");
    std::exit(1);
  }
  return repeats;
}

int selected_cublaslt_heuristics() {
  const char *env = std::getenv("GEMMA4_PREFILL_CUBLASLT_HEURISTICS");
  if (env == nullptr) {
    return 1;
  }
  const int count = std::atoi(env);
  if (count <= 0 || count > kMaxCublasLtHeuristics) {
    std::fprintf(stderr, "GEMMA4_PREFILL_CUBLASLT_HEURISTICS must be in [1,%d]\n",
                 kMaxCublasLtHeuristics);
    std::exit(1);
  }
  return count;
}

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

__global__ void fill_bf16_kernel(__nv_bfloat16 *ptr, size_t count, uint64_t seed,
                                 float scale) {
  const size_t idx = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
  if (idx >= count) {
    return;
  }
  uint32_t x = uint32_t(idx) ^ uint32_t(idx >> 32) ^ uint32_t(seed) ^
               uint32_t(seed >> 32);
  x = mix_u32(x);
  const float u = float(x >> 8) * (1.0f / 16777216.0f);
  ptr[idx] = __float2bfloat16_rn((u * 2.0f - 1.0f) * scale);
}

void fill_bf16(__nv_bfloat16 *ptr, size_t count, uint64_t seed, float scale,
               cudaStream_t stream) {
  const int blocks = int((count + kFillThreads - 1) / kFillThreads);
  fill_bf16_kernel<<<blocks, kFillThreads, 0, stream>>>(ptr, count, seed, scale);
  CUDA_CHECK(cudaGetLastError());
}

template <int WarpTilesM, int WarpTilesN>
__global__ __launch_bounds__(WarpTilesM * WarpTilesN * kThreadsPerWarp, 2)
void gemma4_tuna_prefill_bf16_kernel(const __nv_bfloat16 *__restrict__ x,
                                     const __nv_bfloat16 *__restrict__ w_col_major,
                                     __nv_bfloat16 *__restrict__ y, int m, int k,
                                     int n) {
  constexpr int warps_per_block = WarpTilesM * WarpTilesN;
  constexpr int block_m = WarpTilesM * kWmmaM;
  constexpr int block_n = WarpTilesN * kWmmaN;

  const int warp = threadIdx.x / kThreadsPerWarp;
  const int warp_m = warp / WarpTilesN;
  const int warp_n = warp - warp_m * WarpTilesN;
  const int tile_m = blockIdx.y * block_m + warp_m * kWmmaM;
  const int tile_n = blockIdx.x * block_n + warp_n * kWmmaN;
  const int lane = threadIdx.x & (kThreadsPerWarp - 1);
  __shared__ float out_tiles[warps_per_block][kWmmaM * kWmmaN];

  fragment<matrix_a, kWmmaM, kWmmaN, kWmmaK, __nv_bfloat16, row_major> a_frag;
  fragment<matrix_b, kWmmaM, kWmmaN, kWmmaK, __nv_bfloat16, col_major> b_frag;
  fragment<accumulator, kWmmaM, kWmmaN, kWmmaK, float> acc_frag;
  nvcuda::wmma::fill_fragment(acc_frag, 0.0f);

  for (int k0 = 0; k0 < k; k0 += kWmmaK) {
    load_matrix_sync(a_frag, x + tile_m * k + k0, k);
    load_matrix_sync(b_frag, w_col_major + tile_n * k + k0, k);
    mma_sync(acc_frag, a_frag, b_frag, acc_frag);
  }
  float *out_tile = out_tiles[warp];
  store_matrix_sync(out_tile, acc_frag, kWmmaN, mem_row_major);
  __syncwarp();
  for (int i = lane; i < kWmmaM * kWmmaN; i += kThreadsPerWarp) {
    const int row = i / kWmmaN;
    const int col = i - row * kWmmaN;
    y[(tile_m + row) * n + tile_n + col] = __float2bfloat16_rn(out_tile[i]);
  }
}

template <int WarpTilesM, int WarpTilesN>
__global__ __launch_bounds__(WarpTilesM * WarpTilesN * kThreadsPerWarp, 2)
void gemma4_tuna_prefill_bf16_smem_kernel(const __nv_bfloat16 *__restrict__ x,
                                          const __nv_bfloat16 *__restrict__ w_col_major,
                                          __nv_bfloat16 *__restrict__ y, int m,
                                          int k, int n) {
  constexpr int warps_per_block = WarpTilesM * WarpTilesN;
  constexpr int block_m = WarpTilesM * kWmmaM;
  constexpr int block_n = WarpTilesN * kWmmaN;
  constexpr int threads = warps_per_block * kThreadsPerWarp;

  const int warp = threadIdx.x / kThreadsPerWarp;
  const int warp_m = warp / WarpTilesN;
  const int warp_n = warp - warp_m * WarpTilesN;
  const int tile_m = blockIdx.y * block_m + warp_m * kWmmaM;
  const int tile_n = blockIdx.x * block_n + warp_n * kWmmaN;
  const int block_row = blockIdx.y * block_m;
  const int block_col = blockIdx.x * block_n;
  const int lane = threadIdx.x & (kThreadsPerWarp - 1);

  __shared__ __nv_bfloat16 smem_a[block_m * kWmmaK];
  __shared__ __nv_bfloat16 smem_b[block_n * kWmmaK];
  __shared__ float out_tiles[warps_per_block][kWmmaM * kWmmaN];

  fragment<matrix_a, kWmmaM, kWmmaN, kWmmaK, __nv_bfloat16, row_major> a_frag;
  fragment<matrix_b, kWmmaM, kWmmaN, kWmmaK, __nv_bfloat16, col_major> b_frag;
  fragment<accumulator, kWmmaM, kWmmaN, kWmmaK, float> acc_frag;
  nvcuda::wmma::fill_fragment(acc_frag, 0.0f);

  for (int k0 = 0; k0 < k; k0 += kWmmaK) {
    for (int idx = threadIdx.x; idx < block_m * kWmmaK; idx += threads) {
      const int row = idx / kWmmaK;
      const int col = idx - row * kWmmaK;
      smem_a[idx] = x[(block_row + row) * k + k0 + col];
    }
    for (int idx = threadIdx.x; idx < block_n * kWmmaK; idx += threads) {
      const int col = idx / kWmmaK;
      const int row = idx - col * kWmmaK;
      smem_b[idx] = w_col_major[(block_col + col) * k + k0 + row];
    }
    __syncthreads();

    load_matrix_sync(a_frag, smem_a + warp_m * kWmmaM * kWmmaK, kWmmaK);
    load_matrix_sync(b_frag, smem_b + warp_n * kWmmaN * kWmmaK, kWmmaK);
    mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    __syncthreads();
  }

  float *out_tile = out_tiles[warp];
  store_matrix_sync(out_tile, acc_frag, kWmmaN, mem_row_major);
  __syncwarp();
  for (int i = lane; i < kWmmaM * kWmmaN; i += kThreadsPerWarp) {
    const int row = i / kWmmaN;
    const int col = i - row * kWmmaN;
    y[(tile_m + row) * n + tile_n + col] = __float2bfloat16_rn(out_tile[i]);
  }
}

template <int WarpTilesM, int WarpTilesN>
cudaError_t launch_tuna_prefill(const __nv_bfloat16 *x, const __nv_bfloat16 *w,
                                __nv_bfloat16 *y, int m, int k, int n,
                                cudaStream_t stream) {
  constexpr int block_m = WarpTilesM * kWmmaM;
  constexpr int block_n = WarpTilesN * kWmmaN;
  constexpr int threads = WarpTilesM * WarpTilesN * kThreadsPerWarp;
  const dim3 block_dim(threads);
  const dim3 grid_dim((n + block_n - 1) / block_n, (m + block_m - 1) / block_m);
  gemma4_tuna_prefill_bf16_kernel<WarpTilesM, WarpTilesN>
      <<<grid_dim, block_dim, 0, stream>>>(x, w, y, m, k, n);
  return cudaGetLastError();
}

template <int WarpTilesM, int WarpTilesN>
cudaError_t launch_tuna_prefill_smem(const __nv_bfloat16 *x, const __nv_bfloat16 *w,
                                     __nv_bfloat16 *y, int m, int k, int n,
                                     cudaStream_t stream) {
  constexpr int block_m = WarpTilesM * kWmmaM;
  constexpr int block_n = WarpTilesN * kWmmaN;
  constexpr int threads = WarpTilesM * WarpTilesN * kThreadsPerWarp;
  const dim3 block_dim(threads);
  const dim3 grid_dim((n + block_n - 1) / block_n, (m + block_m - 1) / block_m);
  gemma4_tuna_prefill_bf16_smem_kernel<WarpTilesM, WarpTilesN>
      <<<grid_dim, block_dim, 0, stream>>>(x, w, y, m, k, n);
  return cudaGetLastError();
}

cudaError_t launch_tuna_prefill(const TunaTileConfig &config, const __nv_bfloat16 *x,
                                const __nv_bfloat16 *w, __nv_bfloat16 *y, int m,
                                int k, int n, cudaStream_t stream) {
  if (config.use_smem) {
    if (config.warp_tiles_m == 1 && config.warp_tiles_n == 4) {
      return launch_tuna_prefill_smem<1, 4>(x, w, y, m, k, n, stream);
    }
    if (config.warp_tiles_m == 1 && config.warp_tiles_n == 8) {
      return launch_tuna_prefill_smem<1, 8>(x, w, y, m, k, n, stream);
    }
    if (config.warp_tiles_m == 2 && config.warp_tiles_n == 4) {
      return launch_tuna_prefill_smem<2, 4>(x, w, y, m, k, n, stream);
    }
    if (config.warp_tiles_m == 2 && config.warp_tiles_n == 8) {
      return launch_tuna_prefill_smem<2, 8>(x, w, y, m, k, n, stream);
    }
    if (config.warp_tiles_m == 4 && config.warp_tiles_n == 4) {
      return launch_tuna_prefill_smem<4, 4>(x, w, y, m, k, n, stream);
    }
    return cudaErrorInvalidValue;
  }
  if (config.warp_tiles_m == 1 && config.warp_tiles_n == 1) {
    return launch_tuna_prefill<1, 1>(x, w, y, m, k, n, stream);
  }
  if (config.warp_tiles_m == 1 && config.warp_tiles_n == 2) {
    return launch_tuna_prefill<1, 2>(x, w, y, m, k, n, stream);
  }
  if (config.warp_tiles_m == 1 && config.warp_tiles_n == 4) {
    return launch_tuna_prefill<1, 4>(x, w, y, m, k, n, stream);
  }
  if (config.warp_tiles_m == 1 && config.warp_tiles_n == 8) {
    return launch_tuna_prefill<1, 8>(x, w, y, m, k, n, stream);
  }
  if (config.warp_tiles_m == 1 && config.warp_tiles_n == 16) {
    return launch_tuna_prefill<1, 16>(x, w, y, m, k, n, stream);
  }
  if (config.warp_tiles_m == 2 && config.warp_tiles_n == 4) {
    return launch_tuna_prefill<2, 4>(x, w, y, m, k, n, stream);
  }
  if (config.warp_tiles_m == 2 && config.warp_tiles_n == 8) {
    return launch_tuna_prefill<2, 8>(x, w, y, m, k, n, stream);
  }
  if (config.warp_tiles_m == 4 && config.warp_tiles_n == 4) {
    return launch_tuna_prefill<4, 4>(x, w, y, m, k, n, stream);
  }
  return cudaErrorInvalidValue;
}

void cublas_prefill(cublasHandle_t handle, const __nv_bfloat16 *x,
                    const __nv_bfloat16 *w, __nv_bfloat16 *y, int m, int k, int n) {
  const float alpha = 1.0f;
  const float beta = 0.0f;
  CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha, w,
                            CUDA_R_16BF, k, x, CUDA_R_16BF, k, &beta, y,
                            CUDA_R_16BF, n, CUBLAS_COMPUTE_32F,
                            selected_cublas_algo()));
}

void set_row_major(cublasLtMatrixLayout_t layout) {
  const cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
  CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
      layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
}

class CublasLtPrefillPlan {
 public:
  CublasLtPrefillPlan(cublasLtHandle_t handle, int m, int k, int n,
                      size_t workspace_bytes) {
    CUBLAS_CHECK(cublasLtMatmulDescCreate(&operation_, CUBLAS_COMPUTE_32F,
                                          CUDA_R_32F));
    const cublasOperation_t transa = CUBLAS_OP_N;
    const cublasOperation_t transb = CUBLAS_OP_T;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        operation_, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        operation_, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&a_, CUDA_R_16BF, m, k, k));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&b_, CUDA_R_16BF, n, k, k));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&c_, CUDA_R_16BF, m, n, n));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&d_, CUDA_R_16BF, m, n, n));
    set_row_major(a_);
    set_row_major(b_);
    set_row_major(c_);
    set_row_major(d_);

    CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference_));
    CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
        preference_, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &workspace_bytes, sizeof(workspace_bytes)));
    const int requested_algos = selected_cublaslt_heuristics();
    std::vector<cublasLtMatmulHeuristicResult_t> heuristic_candidates(
        requested_algos);
    CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(
        handle, operation_, a_, b_, c_, d_, preference_, requested_algos,
        heuristic_candidates.data(), &returned_algos_));
    if (returned_algos_ <= 0) {
      std::fprintf(stderr, "cuBLASLt found no supported heuristic for M=%d K=%d N=%d\n",
                   m, k, n);
      std::exit(1);
    }
    for (int i = 0; i < returned_algos_; ++i) {
      if (heuristic_candidates[i].state == CUBLAS_STATUS_SUCCESS) {
        heuristics_.push_back(heuristic_candidates[i]);
      }
    }
    if (heuristics_.empty()) {
      std::fprintf(stderr,
                   "cuBLASLt returned no successful heuristic for M=%d K=%d N=%d\n",
                   m, k, n);
      std::exit(1);
    }
  }

  ~CublasLtPrefillPlan() {
    if (preference_ != nullptr) {
      cublasLtMatmulPreferenceDestroy(preference_);
    }
    if (d_ != nullptr) {
      cublasLtMatrixLayoutDestroy(d_);
    }
    if (c_ != nullptr) {
      cublasLtMatrixLayoutDestroy(c_);
    }
    if (b_ != nullptr) {
      cublasLtMatrixLayoutDestroy(b_);
    }
    if (a_ != nullptr) {
      cublasLtMatrixLayoutDestroy(a_);
    }
    if (operation_ != nullptr) {
      cublasLtMatmulDescDestroy(operation_);
    }
  }

  CublasLtPrefillPlan(const CublasLtPrefillPlan &) = delete;
  CublasLtPrefillPlan &operator=(const CublasLtPrefillPlan &) = delete;

  cublasLtMatmulDesc_t operation() const { return operation_; }
  cublasLtMatrixLayout_t a() const { return a_; }
  cublasLtMatrixLayout_t b() const { return b_; }
  cublasLtMatrixLayout_t c() const { return c_; }
  cublasLtMatrixLayout_t d() const { return d_; }
  const cublasLtMatmulAlgo_t *algo(int index) const {
    return &heuristics_[index].algo;
  }
  int algo_count() const { return int(heuristics_.size()); }

 private:
  cublasLtMatmulDesc_t operation_ = nullptr;
  cublasLtMatrixLayout_t a_ = nullptr;
  cublasLtMatrixLayout_t b_ = nullptr;
  cublasLtMatrixLayout_t c_ = nullptr;
  cublasLtMatrixLayout_t d_ = nullptr;
  cublasLtMatmulPreference_t preference_ = nullptr;
  std::vector<cublasLtMatmulHeuristicResult_t> heuristics_;
  int returned_algos_ = 0;
};

void cublaslt_prefill(cublasLtHandle_t handle, const CublasLtPrefillPlan &plan,
                      const __nv_bfloat16 *x, const __nv_bfloat16 *w,
                      __nv_bfloat16 *y, void *workspace, size_t workspace_bytes,
                      int algo_index, cudaStream_t stream) {
  const float alpha = 1.0f;
  const float beta = 0.0f;
  CUBLAS_CHECK(cublasLtMatmul(handle, plan.operation(), &alpha, x, plan.a(), w,
                              plan.b(), &beta, y, plan.c(), y, plan.d(),
                              plan.algo(algo_index), workspace, workspace_bytes,
                              stream));
}

float elapsed_ms(cudaEvent_t start, cudaEvent_t stop) {
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  return ms;
}

template <typename Fn>
float time_cuda(Fn fn, int warmup, int iters, cudaStream_t stream) {
  for (int i = 0; i < warmup; ++i) {
    fn();
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start, stream));
  for (int i = 0; i < iters; ++i) {
    fn();
  }
  CUDA_CHECK(cudaEventRecord(stop, stream));
  CUDA_CHECK(cudaEventSynchronize(stop));
  const float ms = elapsed_ms(start, stop) / float(iters);
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return ms;
}

template <typename Fn>
float time_cuda_graph_stream(Fn fn, cudaStream_t stream, int warmup, int iters,
                             int graph_repeats) {
  if (graph_repeats <= 1) {
    return time_cuda(fn, warmup, iters, stream);
  }

  for (int i = 0; i < warmup; ++i) {
    fn();
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));

  cudaGraph_t graph = nullptr;
  cudaGraphExec_t graph_exec = nullptr;
  CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  for (int i = 0; i < graph_repeats; ++i) {
    fn();
  }
  CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
  CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));

  for (int i = 0; i < warmup; ++i) {
    CUDA_CHECK(cudaGraphLaunch(graph_exec, stream));
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start, stream));
  for (int i = 0; i < iters; ++i) {
    CUDA_CHECK(cudaGraphLaunch(graph_exec, stream));
  }
  CUDA_CHECK(cudaEventRecord(stop, stream));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaGraphExecDestroy(graph_exec));
  CUDA_CHECK(cudaGraphDestroy(graph));
  return ms / float(iters * graph_repeats);
}

float max_abs_diff(const std::vector<__nv_bfloat16> &actual,
                   const std::vector<__nv_bfloat16> &expected) {
  float max_abs = 0.0f;
  for (size_t i = 0; i < actual.size(); ++i) {
    const float diff =
        std::fabs(__bfloat162float(actual[i]) - __bfloat162float(expected[i]));
    max_abs = std::max(max_abs, diff);
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

const TunaTileConfig *find_config(const char *name) {
  for (const TunaTileConfig &config : kTileConfigs) {
    if (std::strcmp(config.name, name) == 0) {
      return &config;
    }
  }
  return nullptr;
}

std::vector<int> default_m_sweep() {
  return {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048};
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

int round_up_to(int value, int multiple) {
  return ((value + multiple - 1) / multiple) * multiple;
}

int custom_m_for_config(int m, const TunaTileConfig &config) {
  return round_up_to(m, config.warp_tiles_m * kWmmaM);
}

void print_usage(const char *argv0) {
  std::fprintf(stderr,
               "usage: %s [op|all] [iters] [warmup] [m_csv] [config|all]\n"
               "example: %s ffn_gate_up 100 20 1,16,64,256,1024 wmma_64x64\n",
               argv0, argv0);
}

void run_case(const Gemma4PrefillOp &op, int m, int iters, int warmup,
              const TunaTileConfig *selected_config, cudaStream_t stream,
              cublasHandle_t handle, cublasLtHandle_t lt_handle) {
  int custom_m = m;
  for (const TunaTileConfig &config : kTileConfigs) {
    custom_m = std::max(custom_m, custom_m_for_config(m, config));
  }
  DeviceBuffer<__nv_bfloat16> x(size_t(custom_m) * op.k);
  DeviceBuffer<__nv_bfloat16> w(size_t(op.n) * op.k);
  DeviceBuffer<__nv_bfloat16> cublas_y(size_t(m) * op.n);
  DeviceBuffer<__nv_bfloat16> tuna_y(size_t(custom_m) * op.n);
  DeviceBuffer<unsigned char> lt_workspace(
      selected_cublas_baseline() == CublasBaseline::Lt ? kCublasLtWorkspaceBytes : 0);
  std::vector<__nv_bfloat16> h_cublas(size_t(m) * op.n);
  std::vector<__nv_bfloat16> h_tuna(size_t(custom_m) * op.n);

  const uint64_t seed = 0x20260520ull ^ (uint64_t(op.k) << 32) ^ uint64_t(op.n) ^
                        uint64_t(m);
  fill_bf16(x.get(), x.count(), seed, 0.2f, stream);
  fill_bf16(w.get(), w.count(), seed + 0x9e3779b97f4a7c15ull, 0.15f, stream);
  CUDA_CHECK(cudaMemsetAsync(cublas_y.get(), 0, cublas_y.count() * sizeof(__nv_bfloat16),
                             stream));
  CUDA_CHECK(cudaMemsetAsync(tuna_y.get(), 0, tuna_y.count() * sizeof(__nv_bfloat16),
                             stream));

  const TunaTileConfig &correctness_config =
      selected_config != nullptr ? *selected_config : kTileConfigs[0];
  std::unique_ptr<CublasLtPrefillPlan> lt_plan;
  if (selected_cublas_baseline() == CublasBaseline::Lt) {
    lt_plan = std::make_unique<CublasLtPrefillPlan>(
        lt_handle, m, op.k, op.n, kCublasLtWorkspaceBytes);
  }
  auto run_cublas_baseline = [&](int lt_algo_index, cudaStream_t cublas_stream) {
    if (selected_cublas_baseline() == CublasBaseline::Lt) {
      cublaslt_prefill(lt_handle, *lt_plan, x.get(), w.get(), cublas_y.get(),
                       lt_workspace.get(), lt_workspace.count(), lt_algo_index,
                       cublas_stream);
    } else {
      cublas_prefill(handle, x.get(), w.get(), cublas_y.get(), m, op.k, op.n);
    }
  };
  auto run_default_cublas_baseline = [&]() { run_cublas_baseline(0, stream); };

  run_default_cublas_baseline();
  const int correctness_m = custom_m_for_config(m, correctness_config);
  CUDA_CHECK(launch_tuna_prefill(correctness_config, x.get(), w.get(), tuna_y.get(),
                                 correctness_m, op.k, op.n, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaMemcpy(h_cublas.data(), cublas_y.get(),
                        h_cublas.size() * sizeof(__nv_bfloat16),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_tuna.data(), tuna_y.get(), h_tuna.size() * sizeof(__nv_bfloat16),
                        cudaMemcpyDeviceToHost));
  std::vector<__nv_bfloat16> h_tuna_active(size_t(m) * op.n);
  for (int row = 0; row < m; ++row) {
    std::memcpy(h_tuna_active.data() + size_t(row) * op.n,
                h_tuna.data() + size_t(row) * op.n, size_t(op.n) * sizeof(__nv_bfloat16));
  }
  const float diff = max_abs_diff(h_tuna_active, h_cublas);
  if (diff > kCorrectnessTolerance) {
    std::fprintf(stderr,
                 "%s M=%d config=%s failed correctness: max_abs=%g tolerance=%g\n",
                 op.name, m, correctness_config.name, diff, kCorrectnessTolerance);
    std::exit(1);
  }

  float cublas_ms = 0.0f;
  int cublaslt_algo_index = 0;
  const int graph_repeats = selected_graph_repeats();
  const int algo_count =
      selected_cublas_baseline() == CublasBaseline::Lt ? lt_plan->algo_count() : 1;
  cublas_ms = std::numeric_limits<float>::infinity();
  for (int algo_index = 0; algo_index < algo_count; ++algo_index) {
    auto run_cublas_candidate = [&]() { run_cublas_baseline(algo_index, stream); };
    const float candidate_ms = time_cuda_graph_stream(run_cublas_candidate, stream,
                                                      warmup, iters, graph_repeats);
    if (candidate_ms < cublas_ms) {
      cublas_ms = candidate_ms;
      cublaslt_algo_index = algo_index;
    }
  }

  const double flops = 2.0 * double(m) * double(op.k) * double(op.n);
  std::printf("%-14s M=%4d K=%5d N=%6d cublas_ms=%8.4f cublas_tflops=%7.2f "
              "cublaslt_algo=%d max_abs=%g\n",
              op.name, m, op.k, op.n, cublas_ms, flops / (double(cublas_ms) * 1.0e9),
              cublaslt_algo_index, diff);

  for (const TunaTileConfig &config : kTileConfigs) {
    if (selected_config != nullptr && std::strcmp(config.name, selected_config->name) != 0) {
      continue;
    }
    const int config_m = custom_m_for_config(m, config);
    CUDA_CHECK(launch_tuna_prefill(config, x.get(), w.get(), tuna_y.get(),
                                   config_m, op.k, op.n, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpy(h_tuna.data(), tuna_y.get(), h_tuna.size() * sizeof(__nv_bfloat16),
                          cudaMemcpyDeviceToHost));
    for (int row = 0; row < m; ++row) {
      std::memcpy(h_tuna_active.data() + size_t(row) * op.n,
                  h_tuna.data() + size_t(row) * op.n,
                  size_t(op.n) * sizeof(__nv_bfloat16));
    }
    const float config_diff = max_abs_diff(h_tuna_active, h_cublas);
    if (config_diff > kCorrectnessTolerance) {
      std::fprintf(stderr,
                   "%s M=%d config=%s failed correctness: max_abs=%g tolerance=%g\n",
                   op.name, m, config.name, config_diff, kCorrectnessTolerance);
      std::exit(1);
    }
    const float tuna_ms = time_cuda_graph_stream(
        [&]() {
          CUDA_CHECK(launch_tuna_prefill(config, x.get(), w.get(), tuna_y.get(),
                                         config_m, op.k, op.n, stream));
        },
        stream, warmup, iters, graph_repeats);
    std::printf("  %-10s custom_ms=%8.4f custom_tflops=%7.2f speedup=%6.3fx\n",
                config.name, tuna_ms, flops / (double(tuna_ms) * 1.0e9),
                double(cublas_ms) / double(tuna_ms));
  }
}

}  // namespace

int main(int argc, char **argv) {
  const char *selected_op = argc > 1 ? argv[1] : "all";
  const int iters = argc > 2 ? std::atoi(argv[2]) : kDefaultIters;
  const int warmup = argc > 3 ? std::atoi(argv[3]) : kDefaultWarmup;
  const std::vector<int> m_sweep = argc > 4 ? parse_m_sweep(argv[4]) : default_m_sweep();
  const char *selected_config_name = argc > 5 ? argv[5] : "all";
  const TunaTileConfig *selected_config =
      std::strcmp(selected_config_name, "all") == 0 ? nullptr : find_config(selected_config_name);

  if (iters <= 0 || warmup < 0) {
    print_usage(argv[0]);
    return 1;
  }
  if (std::strcmp(selected_op, "all") != 0 && find_op(selected_op) == nullptr) {
    print_usage(argv[0]);
    return 1;
  }
  if (std::strcmp(selected_config_name, "all") != 0 && selected_config == nullptr) {
    print_usage(argv[0]);
    return 1;
  }

  cudaStream_t stream = nullptr;
  cublasHandle_t handle = nullptr;
  cublasLtHandle_t lt_handle = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CUBLAS_CHECK(cublasCreate(&handle));
  CUBLAS_CHECK(cublasLtCreate(&lt_handle));
  CUBLAS_CHECK(cublasSetStream(handle, stream));
  CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

  std::printf("Gemma 4 Tuna BF16 prefill benchmark, iters=%d warmup=%d "
              "cublas_backend=%s cublas_algo=%s cublaslt_heuristics=%d "
              "graph_repeats=%d\n",
              iters, warmup, selected_cublas_baseline_name(),
              selected_cublas_algo_name(), selected_cublaslt_heuristics(),
              selected_graph_repeats());
  std::printf("Weights use Gemma decode layout [N,K], so the custom path computes "
              "Y[M,N] = X[M,K] * W[N,K]^T.\n");

  for (const Gemma4PrefillOp &op : kPrefillOps) {
    if (std::strcmp(selected_op, "all") != 0 && std::strcmp(selected_op, op.name) != 0) {
      continue;
    }
    for (int m : m_sweep) {
      run_case(op, m, iters, warmup, selected_config, stream, handle, lt_handle);
    }
  }

  CUBLAS_CHECK(cublasLtDestroy(lt_handle));
  CUBLAS_CHECK(cublasDestroy(handle));
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}
