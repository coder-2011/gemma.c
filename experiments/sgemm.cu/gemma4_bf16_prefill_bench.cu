#include "gemma4.h"

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_universal.h>
#include <cutlass/gemm/threadblock/threadblock_swizzle_streamk.h>
#include <cutlass/numeric_types.h>
#include <mma.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
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
constexpr int kWarpSize = 32;
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

struct SgemmBf16Config {
  const char *name;
  int warp_tiles_m;
  int warp_tiles_n;
  bool use_smem;
  int smem_a_stage;
  bool wide_warp;
  int split_k;
};

constexpr SgemmBf16Config kConfigs[] = {
    {"bf16_cutlass_64x64", 1, 1, false, 0, false, 1},
    {"bf16_cutlass_64x64_s10", 1, 1, false, 0, false, 1},
    {"bf16_cutlass_64x64x64_s5", 1, 1, false, 0, false, 1},
    {"bf16_cutlass_64x128", 1, 1, false, 0, false, 1},
    {"bf16_cutlass_64x128x64", 1, 1, false, 0, false, 1},
    {"bf16_cutlass_64x128_s2", 1, 1, false, 0, false, 1},
    {"bf16_cutlass_64x128_s4", 1, 1, false, 0, false, 1},
    {"bf16_cutlass_64x128_s6", 1, 1, false, 0, false, 1},
    {"bf16_cutlass_64x256_s4", 1, 1, false, 0, false, 1},
    {"bf16_cutlass_128x64", 1, 1, false, 0, false, 1},
    {"bf16_cutlass_128x64x64", 1, 1, false, 0, false, 1},
    {"bf16_cutlass_128x64_s6", 1, 1, false, 0, false, 1},
    {"bf16_cutlass_128x128", 1, 1, false, 0, false, 1},
    {"bf16_cutlass_128x128x64", 1, 1, false, 0, false, 1},
    {"bf16_cutlass_128x128_s5", 1, 1, false, 0, false, 1},
    {"bf16_cutlass_128x256", 1, 1, false, 0, false, 1},
    {"bf16_cutlass_256x64", 1, 1, false, 0, false, 1},
    {"bf16_cutlass_256x64_s4", 1, 1, false, 0, false, 1},
    {"bf16_cutlass_256x128", 1, 1, false, 0, false, 1},
    {"bf16_streamk_64x64x64", 1, 1, false, 0, false, 1},
    {"bf16_streamk_64x128x64", 1, 1, false, 0, false, 1},
    {"bf16_streamk_128x128x64", 1, 1, false, 0, false, 1},
    {"bf16_streamk_s2_64x64x64", 1, 1, false, 0, false, 1},
    {"bf16_streamk_s2_64x128x64", 1, 1, false, 0, false, 1},
    {"bf16_streamk_s2_128x128x64", 1, 1, false, 0, false, 1},
    {"bf16_streamk_s4_64x64x64", 1, 1, false, 0, false, 1},
    {"bf16_streamk_s4_64x128x64", 1, 1, false, 0, false, 1},
    {"bf16_streamk_s4_128x128x64", 1, 1, false, 0, false, 1},
    {"bf16_auto_ffn_down", 1, 2, false, 0, false, 1},
    {"bf16_16x16", 1, 1, false, 0, false, 1},
    {"bf16_16x32", 1, 2, false, 0, false, 1},
    {"bf16_16x64", 1, 4, false, 0, false, 1},
    {"bf16_16x128", 1, 8, false, 0, false, 1},
    {"bf16_16x256", 1, 16, false, 0, false, 1},
    {"bf16_32x32", 2, 2, false, 0, false, 1},
    {"bf16_32x64", 2, 4, false, 0, false, 1},
    {"bf16_32x128", 2, 8, false, 0, false, 1},
    {"bf16_32x256", 2, 16, false, 0, false, 1},
    {"bf16_64x64", 4, 4, false, 0, false, 1},
    {"bf16_64x128", 4, 8, false, 0, false, 1},
    {"bf16_128x64", 8, 4, false, 0, false, 1},
    {"bf16_256x32", 16, 2, false, 0, false, 1},
    {"bf16_smem64_32x64", 2, 4, true, 0, false, 1},
    {"bf16_smem64_64x64", 4, 4, true, 0, false, 1},
    {"bf16_smemA64_16x32", 1, 2, false, 64, false, 1},
    {"bf16_smemA64_16x64", 1, 4, false, 64, false, 1},
    {"bf16_smemA64_16x128", 1, 8, false, 64, false, 1},
    {"bf16_smemA128_16x32", 1, 2, false, 128, false, 1},
    {"bf16_smemA128_16x64", 1, 4, false, 128, false, 1},
    {"bf16_smemA128_16x128", 1, 8, false, 128, false, 1},
    {"bf16_warp_16x32", 1, 2, false, 0, true, 1},
    {"bf16_warp_16x64", 1, 4, false, 0, true, 1},
    {"bf16_warp_16x128", 1, 8, false, 0, true, 1},
    {"bf16_splitk4_16x32", 1, 2, false, 0, false, 4},
    {"bf16_splitk4_32x64", 2, 4, false, 0, false, 4},
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

void fill_bf16(__nv_bfloat16 *ptr, size_t count, uint64_t seed, float scale) {
  const int blocks = int((count + kFillThreads - 1) / kFillThreads);
  fill_bf16_kernel<<<blocks, kFillThreads>>>(ptr, count, seed, scale);
  CUDA_CHECK(cudaGetLastError());
}

template <int WarpTilesM, int WarpTilesN>
__global__ __launch_bounds__(WarpTilesM * WarpTilesN * kWarpSize, 1)
void gemma4_sgemm_bf16_kernel(const __nv_bfloat16 *__restrict__ x,
                              const __nv_bfloat16 *__restrict__ w_col_major,
                              __nv_bfloat16 *__restrict__ y, int m, int k, int n) {
  constexpr int warps_per_block = WarpTilesM * WarpTilesN;
  constexpr int block_m = WarpTilesM * kWmmaM;
  constexpr int block_n = WarpTilesN * kWmmaN;

  const int warp = threadIdx.x / kWarpSize;
  const int warp_m = warp / WarpTilesN;
  const int warp_n = warp - warp_m * WarpTilesN;
  const int tile_m = blockIdx.y * block_m + warp_m * kWmmaM;
  const int tile_n = blockIdx.x * block_n + warp_n * kWmmaN;
  const int lane = threadIdx.x & (kWarpSize - 1);

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
  for (int i = lane; i < kWmmaM * kWmmaN; i += kWarpSize) {
    const int row = i / kWmmaN;
    const int col = i - row * kWmmaN;
    y[(tile_m + row) * n + tile_n + col] = __float2bfloat16_rn(out_tile[i]);
  }
}

template <int WarpTilesM, int WarpTilesN>
cudaError_t launch_bf16(const __nv_bfloat16 *x, const __nv_bfloat16 *w,
                        __nv_bfloat16 *y, int m, int k, int n) {
  constexpr int block_m = WarpTilesM * kWmmaM;
  constexpr int block_n = WarpTilesN * kWmmaN;
  constexpr int threads = WarpTilesM * WarpTilesN * kWarpSize;
  const dim3 grid_dim((n + block_n - 1) / block_n, (m + block_m - 1) / block_m);
  gemma4_sgemm_bf16_kernel<WarpTilesM, WarpTilesN>
      <<<grid_dim, threads>>>(x, w, y, m, k, n);
  return cudaGetLastError();
}

template <int WarpTilesM, int WarpTilesN>
__global__ __launch_bounds__(WarpTilesM * WarpTilesN * kWarpSize, 1)
void gemma4_sgemm_bf16_smem_kernel(const __nv_bfloat16 *__restrict__ x,
                                   const __nv_bfloat16 *__restrict__ w_col_major,
                                   __nv_bfloat16 *__restrict__ y, int m, int k,
                                   int n) {
  constexpr int warps_per_block = WarpTilesM * WarpTilesN;
  constexpr int block_m = WarpTilesM * kWmmaM;
  constexpr int block_n = WarpTilesN * kWmmaN;
  constexpr int threads = warps_per_block * kWarpSize;
  constexpr int kStage = 64;

  const int warp = threadIdx.x / kWarpSize;
  const int warp_m = warp / WarpTilesN;
  const int warp_n = warp - warp_m * WarpTilesN;
  const int block_row = blockIdx.y * block_m;
  const int block_col = blockIdx.x * block_n;
  const int tile_m = block_row + warp_m * kWmmaM;
  const int tile_n = block_col + warp_n * kWmmaN;
  const int lane = threadIdx.x & (kWarpSize - 1);

  __shared__ __nv_bfloat16 s_a[block_m * kStage];
  __shared__ __nv_bfloat16 s_b[block_n * kStage];
  __shared__ float out_tiles[warps_per_block][kWmmaM * kWmmaN];

  fragment<matrix_a, kWmmaM, kWmmaN, kWmmaK, __nv_bfloat16, row_major> a_frag;
  fragment<matrix_b, kWmmaM, kWmmaN, kWmmaK, __nv_bfloat16, col_major> b_frag;
  fragment<accumulator, kWmmaM, kWmmaN, kWmmaK, float> acc_frag;
  nvcuda::wmma::fill_fragment(acc_frag, 0.0f);

  for (int k_base = 0; k_base < k; k_base += kStage) {
    for (int idx = threadIdx.x; idx < block_m * kStage; idx += threads) {
      const int row = idx / kStage;
      const int kk = idx - row * kStage;
      s_a[idx] = x[(block_row + row) * k + k_base + kk];
    }
    for (int idx = threadIdx.x; idx < block_n * kStage; idx += threads) {
      const int col = idx / kStage;
      const int kk = idx - col * kStage;
      s_b[idx] = w_col_major[(block_col + col) * k + k_base + kk];
    }
    __syncthreads();

    for (int kk = 0; kk < kStage; kk += kWmmaK) {
      load_matrix_sync(a_frag, s_a + warp_m * kWmmaM * kStage + kk, kStage);
      load_matrix_sync(b_frag, s_b + warp_n * kWmmaN * kStage + kk, kStage);
      mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }
    __syncthreads();
  }

  float *out_tile = out_tiles[warp];
  store_matrix_sync(out_tile, acc_frag, kWmmaN, mem_row_major);
  __syncwarp();
  for (int i = lane; i < kWmmaM * kWmmaN; i += kWarpSize) {
    const int row = i / kWmmaN;
    const int col = i - row * kWmmaN;
    y[(tile_m + row) * n + tile_n + col] = __float2bfloat16_rn(out_tile[i]);
  }
}

template <int WarpTilesM, int WarpTilesN>
cudaError_t launch_bf16_smem(const __nv_bfloat16 *x, const __nv_bfloat16 *w,
                             __nv_bfloat16 *y, int m, int k, int n) {
  constexpr int block_m = WarpTilesM * kWmmaM;
  constexpr int block_n = WarpTilesN * kWmmaN;
  constexpr int threads = WarpTilesM * WarpTilesN * kWarpSize;
  const dim3 grid_dim((n + block_n - 1) / block_n, (m + block_m - 1) / block_m);
  gemma4_sgemm_bf16_smem_kernel<WarpTilesM, WarpTilesN>
      <<<grid_dim, threads>>>(x, w, y, m, k, n);
  return cudaGetLastError();
}

template <int WarpTilesN, int KStage>
__global__ __launch_bounds__(WarpTilesN * kWarpSize, 1)
void gemma4_sgemm_bf16_smem_a_kernel(const __nv_bfloat16 *__restrict__ x,
                                     const __nv_bfloat16 *__restrict__ w_col_major,
                                     __nv_bfloat16 *__restrict__ y, int m, int k,
                                     int n) {
  constexpr int block_m = kWmmaM;
  constexpr int block_n = WarpTilesN * kWmmaN;
  constexpr int threads = WarpTilesN * kWarpSize;

  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int tile_m = blockIdx.y * block_m;
  const int tile_n = blockIdx.x * block_n + warp * kWmmaN;

  __shared__ __nv_bfloat16 s_a[block_m * KStage];
  __shared__ float out_tiles[WarpTilesN][kWmmaM * kWmmaN];

  fragment<matrix_a, kWmmaM, kWmmaN, kWmmaK, __nv_bfloat16, row_major> a_frag;
  fragment<matrix_b, kWmmaM, kWmmaN, kWmmaK, __nv_bfloat16, col_major> b_frag;
  fragment<accumulator, kWmmaM, kWmmaN, kWmmaK, float> acc_frag;
  nvcuda::wmma::fill_fragment(acc_frag, 0.0f);

  for (int k_base = 0; k_base < k; k_base += KStage) {
    for (int idx = threadIdx.x; idx < block_m * KStage; idx += threads) {
      const int row = idx / KStage;
      const int kk = idx - row * KStage;
      s_a[idx] = x[(tile_m + row) * k + k_base + kk];
    }
    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < KStage; kk += kWmmaK) {
      load_matrix_sync(a_frag, s_a + kk, KStage);
      load_matrix_sync(b_frag, w_col_major + tile_n * k + k_base + kk, k);
      mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }
    __syncthreads();
  }

  float *out_tile = out_tiles[warp];
  store_matrix_sync(out_tile, acc_frag, kWmmaN, mem_row_major);
  __syncwarp();
  for (int i = lane; i < kWmmaM * kWmmaN; i += kWarpSize) {
    const int row = i / kWmmaN;
    const int col = i - row * kWmmaN;
    y[(tile_m + row) * n + tile_n + col] = __float2bfloat16_rn(out_tile[i]);
  }
}

template <int WarpTilesN, int KStage>
cudaError_t launch_bf16_smem_a(const __nv_bfloat16 *x, const __nv_bfloat16 *w,
                               __nv_bfloat16 *y, int m, int k, int n) {
  constexpr int block_m = kWmmaM;
  constexpr int block_n = WarpTilesN * kWmmaN;
  constexpr int threads = WarpTilesN * kWarpSize;
  static_assert((KStage % kWmmaK) == 0, "A-stage depth must align to WMMA K");
  const dim3 grid_dim((n + block_n - 1) / block_n, (m + block_m - 1) / block_m);
  gemma4_sgemm_bf16_smem_a_kernel<WarpTilesN, KStage>
      <<<grid_dim, threads>>>(x, w, y, m, k, n);
  return cudaGetLastError();
}

template <int WarpCols, int WarpsPerBlock>
__global__ __launch_bounds__(WarpsPerBlock * kWarpSize, 1)
void gemma4_sgemm_bf16_wide_warp_kernel(
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ w_col_major,
    __nv_bfloat16 *__restrict__ y,
    int m,
    int k,
    int n) {
  constexpr int block_n = WarpCols * kWmmaN;
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int col_tiles = n / block_n;
  const int tile_idx = blockIdx.x * WarpsPerBlock + warp;
  const int tile_count = (m / kWmmaM) * col_tiles;
  if (tile_idx >= tile_count) {
    return;
  }
  const int row_tile = tile_idx / col_tiles;
  const int col_tile = tile_idx - row_tile * col_tiles;
  const int tile_m = row_tile * kWmmaM;
  const int tile_n = col_tile * block_n;

  __shared__ float out_tiles[WarpsPerBlock][WarpCols][kWmmaM * kWmmaN];

  fragment<matrix_a, kWmmaM, kWmmaN, kWmmaK, __nv_bfloat16, row_major> a_frag;
  fragment<matrix_b, kWmmaM, kWmmaN, kWmmaK, __nv_bfloat16, col_major>
      b_frag[WarpCols];
  fragment<accumulator, kWmmaM, kWmmaN, kWmmaK, float> acc_frag[WarpCols];
#pragma unroll
  for (int col = 0; col < WarpCols; ++col) {
    nvcuda::wmma::fill_fragment(acc_frag[col], 0.0f);
  }

  for (int k0 = 0; k0 < k; k0 += kWmmaK) {
    load_matrix_sync(a_frag, x + tile_m * k + k0, k);
#pragma unroll
    for (int col = 0; col < WarpCols; ++col) {
      load_matrix_sync(
          b_frag[col], w_col_major + (tile_n + col * kWmmaN) * k + k0, k);
      mma_sync(acc_frag[col], a_frag, b_frag[col], acc_frag[col]);
    }
  }

#pragma unroll
  for (int col_tile_offset = 0; col_tile_offset < WarpCols; ++col_tile_offset) {
    float *out_tile = out_tiles[warp][col_tile_offset];
    store_matrix_sync(out_tile, acc_frag[col_tile_offset], kWmmaN, mem_row_major);
    __syncwarp();
    for (int i = lane; i < kWmmaM * kWmmaN; i += kWarpSize) {
      const int row = i / kWmmaN;
      const int col = i - row * kWmmaN;
      y[(tile_m + row) * n + tile_n + col_tile_offset * kWmmaN + col] =
          __float2bfloat16_rn(out_tile[i]);
    }
  }
}

template <int WarpCols>
cudaError_t launch_bf16_wide_warp(const __nv_bfloat16 *x,
                                  const __nv_bfloat16 *w,
                                  __nv_bfloat16 *y,
                                  int m,
                                  int k,
                                  int n) {
  constexpr int kWarpsPerBlock = 4;
  constexpr int block_n = WarpCols * kWmmaN;
  const int row_tiles = m / kWmmaM;
  const int col_tiles = n / block_n;
  const int tile_count = row_tiles * col_tiles;
  const dim3 grid_dim((tile_count + kWarpsPerBlock - 1) / kWarpsPerBlock);
  gemma4_sgemm_bf16_wide_warp_kernel<WarpCols, kWarpsPerBlock>
      <<<grid_dim, kWarpsPerBlock * kWarpSize>>>(x, w, y, m, k, n);
  return cudaGetLastError();
}

template <int WarpTilesM, int WarpTilesN, int SplitK>
__global__ __launch_bounds__(WarpTilesM * WarpTilesN * kWarpSize, 1)
void gemma4_sgemm_bf16_splitk_kernel(
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ w_col_major,
    float *__restrict__ partials,
    int m,
    int k,
    int n) {
  constexpr int warps_per_block = WarpTilesM * WarpTilesN;
  constexpr int block_m = WarpTilesM * kWmmaM;
  constexpr int block_n = WarpTilesN * kWmmaN;
  constexpr int split_k = GEMMA4_INTERMEDIATE_SIZE / SplitK;
  static_assert((GEMMA4_INTERMEDIATE_SIZE % SplitK) == 0,
                "ffn_down split-K must divide K exactly");
  static_assert((split_k % kWmmaK) == 0, "split-K chunk must align to WMMA K");

  const int warp = threadIdx.x / kWarpSize;
  const int warp_m = warp / WarpTilesN;
  const int warp_n = warp - warp_m * WarpTilesN;
  const int tile_m = blockIdx.y * block_m + warp_m * kWmmaM;
  const int tile_n = blockIdx.x * block_n + warp_n * kWmmaN;
  const int split = blockIdx.z;
  const int k_begin = split * split_k;
  const int k_end = k_begin + split_k;
  const int lane = threadIdx.x & (kWarpSize - 1);

  __shared__ float out_tiles[warps_per_block][kWmmaM * kWmmaN];

  fragment<matrix_a, kWmmaM, kWmmaN, kWmmaK, __nv_bfloat16, row_major> a_frag;
  fragment<matrix_b, kWmmaM, kWmmaN, kWmmaK, __nv_bfloat16, col_major> b_frag;
  fragment<accumulator, kWmmaM, kWmmaN, kWmmaK, float> acc_frag;
  nvcuda::wmma::fill_fragment(acc_frag, 0.0f);

  for (int k0 = k_begin; k0 < k_end; k0 += kWmmaK) {
    load_matrix_sync(a_frag, x + tile_m * k + k0, k);
    load_matrix_sync(b_frag, w_col_major + tile_n * k + k0, k);
    mma_sync(acc_frag, a_frag, b_frag, acc_frag);
  }

  float *out_tile = out_tiles[warp];
  store_matrix_sync(out_tile, acc_frag, kWmmaN, mem_row_major);
  __syncwarp();
  float *partial = partials + size_t(split) * m * n;
  for (int i = lane; i < kWmmaM * kWmmaN; i += kWarpSize) {
    const int row = i / kWmmaN;
    const int col = i - row * kWmmaN;
    partial[(tile_m + row) * n + tile_n + col] = out_tile[i];
  }
}

template <int SplitK>
__global__ void gemma4_sgemm_bf16_splitk_reduce_kernel(
    __nv_bfloat16 *__restrict__ y,
    const float *__restrict__ partials,
    int elements) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= elements) {
    return;
  }
  float sum = 0.0f;
#pragma unroll
  for (int split = 0; split < SplitK; ++split) {
    sum += partials[size_t(split) * elements + idx];
  }
  y[idx] = __float2bfloat16_rn(sum);
}

template <int WarpTilesM, int WarpTilesN, int SplitK>
cudaError_t launch_bf16_splitk(const __nv_bfloat16 *x,
                               const __nv_bfloat16 *w,
                               __nv_bfloat16 *y,
                               float *partials,
                               int m,
                               int k,
                               int n) {
  constexpr int block_m = WarpTilesM * kWmmaM;
  constexpr int block_n = WarpTilesN * kWmmaN;
  constexpr int threads = WarpTilesM * WarpTilesN * kWarpSize;
  if (k != GEMMA4_INTERMEDIATE_SIZE || n != GEMMA4_HIDDEN_SIZE || partials == nullptr) {
    return cudaErrorInvalidValue;
  }
  const dim3 grid_dim((n + block_n - 1) / block_n, (m + block_m - 1) / block_m,
                      SplitK);
  gemma4_sgemm_bf16_splitk_kernel<WarpTilesM, WarpTilesN, SplitK>
      <<<grid_dim, threads>>>(x, w, partials, m, k, n);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    return status;
  }
  const int elements = m * n;
  constexpr int reduce_threads = 256;
  const int reduce_blocks = (elements + reduce_threads - 1) / reduce_threads;
  gemma4_sgemm_bf16_splitk_reduce_kernel<SplitK>
      <<<reduce_blocks, reduce_threads>>>(y, partials, elements);
  return cudaGetLastError();
}

template <int ThreadblockM, int ThreadblockN, int ThreadblockK,
          int WarpM, int WarpN, int Stages>
cudaError_t launch_cutlass_gemm(const __nv_bfloat16 *x, const __nv_bfloat16 *w,
                                __nv_bfloat16 *y, int m, int k, int n,
                                cudaStream_t stream = nullptr) {
  using Element = cutlass::bfloat16_t;
  using Gemm = cutlass::gemm::device::Gemm<
      Element, cutlass::layout::RowMajor,
      Element, cutlass::layout::ColumnMajor,
      Element, cutlass::layout::RowMajor,
      float,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<ThreadblockM, ThreadblockN, ThreadblockK>,
      cutlass::gemm::GemmShape<WarpM, WarpN, ThreadblockK>,
      cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<Element, 8, float, float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
      Stages>;

  typename Gemm::Arguments args(
      {m, n, k},
      {reinterpret_cast<const Element *>(x), k},
      {reinterpret_cast<const Element *>(w), k},
      {reinterpret_cast<Element *>(y), n},
      {reinterpret_cast<Element *>(y), n},
      {1.0f, 0.0f});
  Gemm gemm;
  const cutlass::Status status = gemm(args, nullptr, stream);
  if (status != cutlass::Status::kSuccess) {
    return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
}

template <int ThreadblockM, int ThreadblockN, int ThreadblockK,
          int WarpM, int WarpN, int Stages>
cudaError_t launch_cutlass_streamk_gemm(const __nv_bfloat16 *x,
                                        const __nv_bfloat16 *w,
                                        __nv_bfloat16 *y, int m, int k, int n,
                                        int split_k_factor,
                                        cudaStream_t stream = nullptr) {
  using Element = cutlass::bfloat16_t;
  using Gemm = cutlass::gemm::device::GemmUniversal<
      Element, cutlass::layout::RowMajor,
      Element, cutlass::layout::ColumnMajor,
      Element, cutlass::layout::RowMajor,
      float,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<ThreadblockM, ThreadblockN, ThreadblockK>,
      cutlass::gemm::GemmShape<WarpM, WarpN, ThreadblockK>,
      cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<Element, 8, float, float>,
      cutlass::gemm::threadblock::ThreadblockSwizzleStreamK,
      Stages,
      8,
      8>;

  typename Gemm::Arguments args(
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k},
      split_k_factor,
      {1.0f, 0.0f},
      reinterpret_cast<const Element *>(x),
      reinterpret_cast<const Element *>(w),
      reinterpret_cast<Element *>(y),
      reinterpret_cast<Element *>(y),
      int64_t(m) * k,
      int64_t(n) * k,
      int64_t(m) * n,
      int64_t(m) * n,
      k,
      k,
      n,
      n,
      -1);

  Gemm gemm;
  const cutlass::Status can_implement = gemm.can_implement(args);
  if (can_implement != cutlass::Status::kSuccess) {
    return cudaErrorInvalidValue;
  }

  const size_t workspace_size = Gemm::get_workspace_size(args);
  static void *d_workspace = nullptr;
  static size_t workspace_capacity = 0;
  if (workspace_size > workspace_capacity) {
    if (d_workspace != nullptr) {
      CUDA_CHECK(cudaFree(d_workspace));
    }
    CUDA_CHECK(cudaMalloc(&d_workspace, workspace_size));
    workspace_capacity = workspace_size;
  }

  const cutlass::Status initialized = gemm.initialize(args, d_workspace, stream);
  if (initialized != cutlass::Status::kSuccess) {
    return cudaErrorInvalidValue;
  }
  const cutlass::Status launched = gemm(stream);
  if (launched != cutlass::Status::kSuccess) {
    return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
}

cudaError_t launch_bf16(const SgemmBf16Config &config, const __nv_bfloat16 *x,
                        const __nv_bfloat16 *w, __nv_bfloat16 *y, int m, int k,
                        int n, float *splitk_partials = nullptr,
                        cudaStream_t stream = nullptr) {
  if (std::strcmp(config.name, "bf16_cutlass_64x64") == 0) {
    return launch_cutlass_gemm<64, 64, 32, 32, 32, 3>(x, w, y, m, k, n, stream);
  }
  if (std::strcmp(config.name, "bf16_cutlass_64x64_s10") == 0) {
    return launch_cutlass_gemm<64, 64, 32, 32, 32, 10>(x, w, y, m, k, n,
                                                       stream);
  }
  if (std::strcmp(config.name, "bf16_cutlass_64x64x64_s5") == 0) {
    return launch_cutlass_gemm<64, 64, 64, 32, 32, 5>(x, w, y, m, k, n,
                                                      stream);
  }
  if (std::strcmp(config.name, "bf16_cutlass_64x128") == 0) {
    return launch_cutlass_gemm<64, 128, 32, 32, 64, 3>(x, w, y, m, k, n, stream);
  }
  if (std::strcmp(config.name, "bf16_cutlass_64x128x64") == 0) {
    return launch_cutlass_gemm<64, 128, 64, 32, 64, 3>(x, w, y, m, k, n,
                                                      stream);
  }
  if (std::strcmp(config.name, "bf16_cutlass_64x128_s2") == 0) {
    return launch_cutlass_gemm<64, 128, 32, 32, 64, 2>(x, w, y, m, k, n,
                                                      stream);
  }
  if (std::strcmp(config.name, "bf16_cutlass_64x128_s4") == 0) {
    return launch_cutlass_gemm<64, 128, 32, 32, 64, 4>(x, w, y, m, k, n,
                                                      stream);
  }
  if (std::strcmp(config.name, "bf16_cutlass_64x128_s6") == 0) {
    return launch_cutlass_gemm<64, 128, 32, 32, 64, 6>(x, w, y, m, k, n,
                                                      stream);
  }
  if (std::strcmp(config.name, "bf16_cutlass_64x256_s4") == 0) {
    return launch_cutlass_gemm<64, 256, 32, 64, 64, 4>(x, w, y, m, k, n,
                                                      stream);
  }
  if (std::strcmp(config.name, "bf16_cutlass_128x64") == 0) {
    return launch_cutlass_gemm<128, 64, 32, 64, 32, 3>(x, w, y, m, k, n,
                                                      stream);
  }
  if (std::strcmp(config.name, "bf16_cutlass_128x64x64") == 0) {
    return launch_cutlass_gemm<128, 64, 64, 64, 32, 3>(x, w, y, m, k, n,
                                                      stream);
  }
  if (std::strcmp(config.name, "bf16_cutlass_128x64_s6") == 0) {
    return launch_cutlass_gemm<128, 64, 32, 64, 32, 6>(x, w, y, m, k, n,
                                                      stream);
  }
  if (std::strcmp(config.name, "bf16_cutlass_128x128") == 0) {
    return launch_cutlass_gemm<128, 128, 32, 64, 64, 3>(x, w, y, m, k, n,
                                                       stream);
  }
  if (std::strcmp(config.name, "bf16_cutlass_128x128x64") == 0) {
    return launch_cutlass_gemm<128, 128, 64, 64, 64, 3>(x, w, y, m, k, n,
                                                       stream);
  }
  if (std::strcmp(config.name, "bf16_cutlass_128x128_s5") == 0) {
    return launch_cutlass_gemm<128, 128, 32, 64, 64, 5>(x, w, y, m, k, n,
                                                       stream);
  }
  if (std::strcmp(config.name, "bf16_cutlass_128x256") == 0) {
    return launch_cutlass_gemm<128, 256, 32, 64, 64, 3>(x, w, y, m, k, n,
                                                       stream);
  }
  if (std::strcmp(config.name, "bf16_cutlass_256x64") == 0) {
    return launch_cutlass_gemm<256, 64, 32, 64, 64, 3>(x, w, y, m, k, n,
                                                      stream);
  }
  if (std::strcmp(config.name, "bf16_cutlass_256x64_s4") == 0) {
    return launch_cutlass_gemm<256, 64, 32, 64, 64, 4>(x, w, y, m, k, n,
                                                      stream);
  }
  if (std::strcmp(config.name, "bf16_cutlass_256x128") == 0) {
    return launch_cutlass_gemm<256, 128, 32, 64, 64, 3>(x, w, y, m, k, n,
                                                       stream);
  }
  if (std::strcmp(config.name, "bf16_streamk_64x64x64") == 0) {
    return launch_cutlass_streamk_gemm<64, 64, 64, 32, 32, 3>(x, w, y, m, k, n,
                                                            1, stream);
  }
  if (std::strcmp(config.name, "bf16_streamk_64x128x64") == 0) {
    return launch_cutlass_streamk_gemm<64, 128, 64, 32, 64, 3>(x, w, y, m, k, n,
                                                             1, stream);
  }
  if (std::strcmp(config.name, "bf16_streamk_128x128x64") == 0) {
    return launch_cutlass_streamk_gemm<128, 128, 64, 64, 64, 3>(x, w, y, m, k,
                                                              n, 1, stream);
  }
  if (std::strcmp(config.name, "bf16_streamk_s2_64x64x64") == 0) {
    return launch_cutlass_streamk_gemm<64, 64, 64, 32, 32, 3>(x, w, y, m, k, n,
                                                            2, stream);
  }
  if (std::strcmp(config.name, "bf16_streamk_s2_64x128x64") == 0) {
    return launch_cutlass_streamk_gemm<64, 128, 64, 32, 64, 3>(x, w, y, m, k, n,
                                                             2, stream);
  }
  if (std::strcmp(config.name, "bf16_streamk_s2_128x128x64") == 0) {
    return launch_cutlass_streamk_gemm<128, 128, 64, 64, 64, 3>(x, w, y, m, k,
                                                              n, 2, stream);
  }
  if (std::strcmp(config.name, "bf16_streamk_s4_64x64x64") == 0) {
    return launch_cutlass_streamk_gemm<64, 64, 64, 32, 32, 3>(x, w, y, m, k, n,
                                                            4, stream);
  }
  if (std::strcmp(config.name, "bf16_streamk_s4_64x128x64") == 0) {
    return launch_cutlass_streamk_gemm<64, 128, 64, 32, 64, 3>(x, w, y, m, k, n,
                                                             4, stream);
  }
  if (std::strcmp(config.name, "bf16_streamk_s4_128x128x64") == 0) {
    return launch_cutlass_streamk_gemm<128, 128, 64, 64, 64, 3>(x, w, y, m, k,
                                                              n, 4, stream);
  }
  if (std::strcmp(config.name, "bf16_auto_ffn_down") == 0) {
    if (k != GEMMA4_INTERMEDIATE_SIZE || n != GEMMA4_HIDDEN_SIZE) {
      return cudaErrorInvalidValue;
    }
    if (m <= 128) {
      return launch_cutlass_gemm<64, 128, 64, 32, 64, 3>(x, w, y, m, k, n,
                                                        stream);
    }
    return launch_cutlass_gemm<128, 128, 64, 64, 64, 3>(x, w, y, m, k, n,
                                                       stream);
  }
  if (stream != nullptr) {
    return cudaErrorInvalidValue;
  }
  if (config.split_k == 4) {
    if (config.warp_tiles_m == 1 && config.warp_tiles_n == 2) {
      return launch_bf16_splitk<1, 2, 4>(x, w, y, splitk_partials, m, k, n);
    }
    if (config.warp_tiles_m == 2 && config.warp_tiles_n == 4) {
      return launch_bf16_splitk<2, 4, 4>(x, w, y, splitk_partials, m, k, n);
    }
    return cudaErrorInvalidValue;
  }
  if (config.wide_warp) {
    if (config.warp_tiles_n == 2) {
      return launch_bf16_wide_warp<2>(x, w, y, m, k, n);
    }
    if (config.warp_tiles_n == 4) {
      return launch_bf16_wide_warp<4>(x, w, y, m, k, n);
    }
    if (config.warp_tiles_n == 8) {
      return launch_bf16_wide_warp<8>(x, w, y, m, k, n);
    }
    return cudaErrorInvalidValue;
  }
  if (config.use_smem) {
    if (config.warp_tiles_m == 2 && config.warp_tiles_n == 4) {
      return launch_bf16_smem<2, 4>(x, w, y, m, k, n);
    }
    if (config.warp_tiles_m == 4 && config.warp_tiles_n == 4) {
      return launch_bf16_smem<4, 4>(x, w, y, m, k, n);
    }
    return cudaErrorInvalidValue;
  }
  if (config.smem_a_stage == 64) {
    if (config.warp_tiles_m == 1 && config.warp_tiles_n == 2) {
      return launch_bf16_smem_a<2, 64>(x, w, y, m, k, n);
    }
    if (config.warp_tiles_m == 1 && config.warp_tiles_n == 4) {
      return launch_bf16_smem_a<4, 64>(x, w, y, m, k, n);
    }
    if (config.warp_tiles_m == 1 && config.warp_tiles_n == 8) {
      return launch_bf16_smem_a<8, 64>(x, w, y, m, k, n);
    }
    return cudaErrorInvalidValue;
  }
  if (config.smem_a_stage == 128) {
    if (config.warp_tiles_m == 1 && config.warp_tiles_n == 2) {
      return launch_bf16_smem_a<2, 128>(x, w, y, m, k, n);
    }
    if (config.warp_tiles_m == 1 && config.warp_tiles_n == 4) {
      return launch_bf16_smem_a<4, 128>(x, w, y, m, k, n);
    }
    if (config.warp_tiles_m == 1 && config.warp_tiles_n == 8) {
      return launch_bf16_smem_a<8, 128>(x, w, y, m, k, n);
    }
    return cudaErrorInvalidValue;
  }
  if (config.warp_tiles_m == 1 && config.warp_tiles_n == 2) {
    return launch_bf16<1, 2>(x, w, y, m, k, n);
  }
  if (config.warp_tiles_m == 1 && config.warp_tiles_n == 1) {
    return launch_bf16<1, 1>(x, w, y, m, k, n);
  }
  if (config.warp_tiles_m == 1 && config.warp_tiles_n == 4) {
    return launch_bf16<1, 4>(x, w, y, m, k, n);
  }
  if (config.warp_tiles_m == 1 && config.warp_tiles_n == 8) {
    return launch_bf16<1, 8>(x, w, y, m, k, n);
  }
  if (config.warp_tiles_m == 1 && config.warp_tiles_n == 16) {
    return launch_bf16<1, 16>(x, w, y, m, k, n);
  }
  if (config.warp_tiles_m == 2 && config.warp_tiles_n == 2) {
    return launch_bf16<2, 2>(x, w, y, m, k, n);
  }
  if (config.warp_tiles_m == 2 && config.warp_tiles_n == 4) {
    return launch_bf16<2, 4>(x, w, y, m, k, n);
  }
  if (config.warp_tiles_m == 2 && config.warp_tiles_n == 8) {
    return launch_bf16<2, 8>(x, w, y, m, k, n);
  }
  if (config.warp_tiles_m == 2 && config.warp_tiles_n == 16) {
    return launch_bf16<2, 16>(x, w, y, m, k, n);
  }
  if (config.warp_tiles_m == 4 && config.warp_tiles_n == 4) {
    return launch_bf16<4, 4>(x, w, y, m, k, n);
  }
  if (config.warp_tiles_m == 4 && config.warp_tiles_n == 8) {
    return launch_bf16<4, 8>(x, w, y, m, k, n);
  }
  if (config.warp_tiles_m == 8 && config.warp_tiles_n == 4) {
    return launch_bf16<8, 4>(x, w, y, m, k, n);
  }
  if (config.warp_tiles_m == 16 && config.warp_tiles_n == 2) {
    return launch_bf16<16, 2>(x, w, y, m, k, n);
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
        heuristic_candidates.data(),
        &returned_algos_));
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
                      int algo_index, cudaStream_t stream = 0) {
  const float alpha = 1.0f;
  const float beta = 0.0f;
  CUBLAS_CHECK(cublasLtMatmul(handle, plan.operation(), &alpha, x, plan.a(), w,
                              plan.b(), &beta, y, plan.c(), y, plan.d(),
                              plan.algo(algo_index), workspace, workspace_bytes,
                              stream));
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

template <typename Fn>
float time_cuda_graph(Fn fn, int warmup, int iters, int graph_repeats) {
  if (graph_repeats <= 1) {
    return time_cuda(fn, warmup, iters);
  }

  for (int i = 0; i < warmup; ++i) {
    fn();
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaGraph_t graph = nullptr;
  cudaGraphExec_t graph_exec = nullptr;
  CUDA_CHECK(cudaStreamBeginCapture(0, cudaStreamCaptureModeThreadLocal));
  for (int i = 0; i < graph_repeats; ++i) {
    fn();
  }
  CUDA_CHECK(cudaStreamEndCapture(0, &graph));
  CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));

  for (int i = 0; i < warmup; ++i) {
    CUDA_CHECK(cudaGraphLaunch(graph_exec, 0));
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i) {
    CUDA_CHECK(cudaGraphLaunch(graph_exec, 0));
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaGraphExecDestroy(graph_exec));
  CUDA_CHECK(cudaGraphDestroy(graph));
  return ms / float(iters * graph_repeats);
}

template <typename Fn>
float time_cuda_graph_stream(Fn fn, cudaStream_t stream, int warmup, int iters,
                             int graph_repeats) {
  if (graph_repeats <= 1) {
    return time_cuda(fn, warmup, iters);
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

const SgemmBf16Config *find_config(const char *name) {
  for (const SgemmBf16Config &config : kConfigs) {
    if (std::strcmp(config.name, name) == 0) {
      return &config;
    }
  }
  return nullptr;
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

int custom_m_for_config(int m, const SgemmBf16Config &config) {
  if (std::strncmp(config.name, "bf16_cutlass_", 13) == 0) {
    return m;
  }
  if (std::strncmp(config.name, "bf16_streamk_", 13) == 0) {
    return m;
  }
  if (std::strcmp(config.name, "bf16_auto_ffn_down") == 0) {
    return m;
  }
  return round_up_to(m, config.warp_tiles_m * kWmmaM);
}

void print_usage(const char *argv0) {
  std::fprintf(stderr,
               "usage: %s [op|all] [iters] [warmup] [m_csv] [config|all]\n"
               "example: %s global_k 100 20 16,64,256,1024 bf16_64x64\n",
               argv0, argv0);
}

void run_case(const Gemma4PrefillOp &op, int m, int iters, int warmup,
              const SgemmBf16Config *selected_config, cublasHandle_t handle,
              cublasLtHandle_t lt_handle) {
  int custom_m = m;
  int max_split_k = 1;
  for (const SgemmBf16Config &config : kConfigs) {
    if (selected_config != nullptr && std::strcmp(config.name, selected_config->name) != 0) {
      continue;
    }
    if (config.split_k > 1 &&
        (op.k != GEMMA4_INTERMEDIATE_SIZE || op.n != GEMMA4_HIDDEN_SIZE)) {
      continue;
    }
    custom_m = std::max(custom_m, custom_m_for_config(m, config));
    max_split_k = std::max(max_split_k, config.split_k);
  }

  DeviceBuffer<__nv_bfloat16> x(size_t(custom_m) * op.k);
  DeviceBuffer<__nv_bfloat16> w(size_t(op.n) * op.k);
  DeviceBuffer<__nv_bfloat16> cublas_y(size_t(m) * op.n);
  DeviceBuffer<__nv_bfloat16> custom_y(size_t(custom_m) * op.n);
  const bool needs_splitk =
      max_split_k > 1 && op.k == GEMMA4_INTERMEDIATE_SIZE && op.n == GEMMA4_HIDDEN_SIZE;
  DeviceBuffer<float> splitk_partials(
      needs_splitk ? size_t(max_split_k) * custom_m * op.n : 0);
  DeviceBuffer<unsigned char> lt_workspace(
      selected_cublas_baseline() == CublasBaseline::Lt ? kCublasLtWorkspaceBytes : 0);
  std::vector<__nv_bfloat16> h_cublas(size_t(m) * op.n);
  std::vector<__nv_bfloat16> h_custom(size_t(custom_m) * op.n);

  const uint64_t seed = 0x20260520ull ^ (uint64_t(op.k) << 32) ^ uint64_t(op.n) ^
                        uint64_t(m);
  fill_bf16(x.get(), x.count(), seed, 0.2f);
  fill_bf16(w.get(), w.count(), seed + 0x9e3779b97f4a7c15ull, 0.15f);
  CUDA_CHECK(cudaMemset(cublas_y.get(), 0, cublas_y.count() * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMemset(custom_y.get(), 0, custom_y.count() * sizeof(__nv_bfloat16)));

  const SgemmBf16Config &correctness_config =
      selected_config != nullptr ? *selected_config : kConfigs[0];
  const int correctness_m = custom_m_for_config(m, correctness_config);
  std::unique_ptr<CublasLtPrefillPlan> lt_plan;
  if (selected_cublas_baseline() == CublasBaseline::Lt) {
    lt_plan = std::make_unique<CublasLtPrefillPlan>(
        lt_handle, m, op.k, op.n, kCublasLtWorkspaceBytes);
  }
  auto run_cublas_baseline = [&](int lt_algo_index, cudaStream_t stream) {
    if (selected_cublas_baseline() == CublasBaseline::Lt) {
      cublaslt_prefill(lt_handle, *lt_plan, x.get(), w.get(), cublas_y.get(),
                       lt_workspace.get(), lt_workspace.count(), lt_algo_index,
                       stream);
    } else {
      cublas_prefill(handle, x.get(), w.get(), cublas_y.get(), m, op.k, op.n);
    }
  };
  auto run_default_cublas_baseline = [&]() { run_cublas_baseline(0, 0); };
  const int graph_repeats = selected_graph_repeats();

  run_default_cublas_baseline();
  CUDA_CHECK(launch_bf16(correctness_config, x.get(), w.get(), custom_y.get(),
                         correctness_m, op.k, op.n, splitk_partials.get()));
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(h_cublas.data(), cublas_y.get(),
                        h_cublas.size() * sizeof(__nv_bfloat16),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_custom.data(), custom_y.get(),
                        h_custom.size() * sizeof(__nv_bfloat16),
                        cudaMemcpyDeviceToHost));
  std::vector<__nv_bfloat16> h_active(size_t(m) * op.n);
  for (int row = 0; row < m; ++row) {
    std::memcpy(h_active.data() + size_t(row) * op.n,
                h_custom.data() + size_t(row) * op.n,
                size_t(op.n) * sizeof(__nv_bfloat16));
  }
  const float diff = max_abs_diff(h_active, h_cublas);
  if (diff > kCorrectnessTolerance) {
    std::fprintf(stderr,
                 "%s M=%d config=%s failed correctness: max_abs=%g tolerance=%g\n",
                 op.name, m, correctness_config.name, diff, kCorrectnessTolerance);
    std::exit(1);
  }

  float cublas_ms = 0.0f;
  int cublaslt_algo_index = 0;
  if (graph_repeats > 1) {
    cudaStream_t graph_stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&graph_stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasSetStream(handle, graph_stream));
    const int algo_count =
        selected_cublas_baseline() == CublasBaseline::Lt ? lt_plan->algo_count() : 1;
    cublas_ms = std::numeric_limits<float>::infinity();
    for (int algo_index = 0; algo_index < algo_count; ++algo_index) {
      auto run_cublas_graph_baseline = [&]() {
        run_cublas_baseline(algo_index, graph_stream);
      };
      const float candidate_ms =
          time_cuda_graph_stream(run_cublas_graph_baseline, graph_stream, warmup,
                                 iters, graph_repeats);
      if (candidate_ms < cublas_ms) {
        cublas_ms = candidate_ms;
        cublaslt_algo_index = algo_index;
      }
    }
    CUBLAS_CHECK(cublasSetStream(handle, 0));
    CUDA_CHECK(cudaStreamDestroy(graph_stream));
  } else {
    const int algo_count =
        selected_cublas_baseline() == CublasBaseline::Lt ? lt_plan->algo_count() : 1;
    cublas_ms = std::numeric_limits<float>::infinity();
    for (int algo_index = 0; algo_index < algo_count; ++algo_index) {
      auto run_cublas_candidate = [&]() { run_cublas_baseline(algo_index, 0); };
      const float candidate_ms = time_cuda(run_cublas_candidate, warmup, iters);
      if (candidate_ms < cublas_ms) {
        cublas_ms = candidate_ms;
        cublaslt_algo_index = algo_index;
      }
    }
  }

  const double flops = 2.0 * double(m) * double(op.k) * double(op.n);
  std::printf("%-14s M=%4d K=%5d N=%6d cublas_ms=%8.4f "
              "cublas_tflops=%7.2f cublaslt_algo=%d max_abs=%g\n",
              op.name, m, op.k, op.n, cublas_ms,
              flops / (double(cublas_ms) * 1.0e9), cublaslt_algo_index, diff);

  for (const SgemmBf16Config &config : kConfigs) {
    if (selected_config != nullptr && std::strcmp(config.name, selected_config->name) != 0) {
      continue;
    }
    if (config.split_k > 1 &&
        (op.k != GEMMA4_INTERMEDIATE_SIZE || op.n != GEMMA4_HIDDEN_SIZE)) {
      continue;
    }
    const int config_m = custom_m_for_config(m, config);
    auto run_custom = [&](cudaStream_t stream) {
      CUDA_CHECK(launch_bf16(config, x.get(), w.get(), custom_y.get(), config_m,
                             op.k, op.n, splitk_partials.get(), stream));
    };
    float custom_ms = 0.0f;
    if (graph_repeats > 1) {
      cudaStream_t custom_stream = nullptr;
      CUDA_CHECK(cudaStreamCreateWithFlags(&custom_stream, cudaStreamNonBlocking));
      auto run_custom_graph = [&]() { run_custom(custom_stream); };
      custom_ms = time_cuda_graph_stream(run_custom_graph, custom_stream, warmup,
                                         iters, graph_repeats);
      CUDA_CHECK(cudaStreamDestroy(custom_stream));
    } else {
      auto run_custom_default = [&]() { run_custom(nullptr); };
      custom_ms = time_cuda(run_custom_default, warmup, iters);
    }
    std::printf("  %-12s custom_ms=%8.4f custom_tflops=%7.2f speedup=%6.3fx\n",
                config.name, custom_ms, flops / (double(custom_ms) * 1.0e9),
                double(cublas_ms) / custom_ms);
  }
}

}  // namespace

int main(int argc, char **argv) {
  const char *selected_op = argc > 1 ? argv[1] : "all";
  const int iters = argc > 2 ? std::atoi(argv[2]) : kDefaultIters;
  const int warmup = argc > 3 ? std::atoi(argv[3]) : kDefaultWarmup;
  const std::vector<int> m_sweep =
      argc > 4 ? parse_m_sweep(argv[4]) : std::vector<int>{16, 64, 256, 1024};
  const char *selected_config_name = argc > 5 ? argv[5] : "all";
  const SgemmBf16Config *selected_config =
      std::strcmp(selected_config_name, "all") == 0 ? nullptr : find_config(selected_config_name);

  if (iters <= 0 || warmup < 0 ||
      (std::strcmp(selected_op, "all") != 0 && find_op(selected_op) == nullptr) ||
      (std::strcmp(selected_config_name, "all") != 0 && selected_config == nullptr)) {
    print_usage(argv[0]);
    return 1;
  }

  cublasHandle_t handle = nullptr;
  cublasLtHandle_t lt_handle = nullptr;
  CUBLAS_CHECK(cublasCreate(&handle));
  CUBLAS_CHECK(cublasLtCreate(&lt_handle));
  CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

  std::printf("Gemma 4 SGEMM BF16 prefill benchmark, iters=%d warmup=%d "
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
      run_case(op, m, iters, warmup, selected_config, handle, lt_handle);
    }
  }

  CUBLAS_CHECK(cublasLtDestroy(lt_handle));
  CUBLAS_CHECK(cublasDestroy(handle));
  return 0;
}
