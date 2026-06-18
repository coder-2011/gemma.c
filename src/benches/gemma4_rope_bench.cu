#include "gemma4_rope.cuh"
#include "gemma4_bench_utils.cuh"
#include "gemma4.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#if __has_include(<cudnn.h>)
#include <cudnn.h>
#define GEMMA4_HAS_CUDNN 1
#else
#define GEMMA4_HAS_CUDNN 0
#endif

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct RopeShape {
  const char *name;
  int q_heads;
  int kv_heads;
  int head_dim;
  int rotary_dim;
};

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

__global__ void fill_unit_rope_table_kernel(float *cos,
                                            float *sin,
                                            size_t count,
                                            int seq_len,
                                            int table_width,
                                            uint64_t seed) {
  size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
  if (i >= count) {
    return;
  }

  int dim = int(i % table_width);
  int pos = int((i / table_width) % seq_len);
  int batch = int(i / (static_cast<size_t>(seq_len) * table_width));
  uint32_t mixed =
      mix_u32_device(uint32_t(dim * 131 + pos * 17 + batch * 8191) ^
                     uint32_t(seed));
  float jitter = float(mixed & 0xffffu) * (1.0f / 65536.0f);
  float angle = 0.0007f * float((pos + 1) * (dim + 1)) +
                0.013f * float(batch) + 0.002f * jitter;
  sincosf(angle, &sin[i], &cos[i]);
}

__global__ void float_to_bf16_kernel(const float *src,
                                     __nv_bfloat16 *dst,
                                     size_t count) {
  size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
  if (i < count) {
    dst[i] = __float2bfloat16_rn(src[i]);
  }
}

uint64_t make_seed() {
  if (const char *env = std::getenv("GEMMA4_ROPE_BENCH_SEED")) {
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
  fill_random_bf16_kernel<<<blocks, threads, 0, stream>>>(ptr, count, seed, scale);
  CUDA_CHECK(cudaGetLastError());
}

void fill_unit_rope_table(float *cos,
                          float *sin,
                          int cos_batch_size,
                          int seq_len,
                          int table_width,
                          uint64_t seed,
                          cudaStream_t stream) {
  constexpr int threads = 256;
  size_t count = static_cast<size_t>(cos_batch_size) * seq_len * table_width;
  int blocks = int((count + threads - 1) / threads);
  fill_unit_rope_table_kernel<<<blocks, threads, 0, stream>>>(
      cos, sin, count, seq_len, table_width, seed);
  CUDA_CHECK(cudaGetLastError());
}

void float_to_bf16(const float *src,
                   __nv_bfloat16 *dst,
                   size_t count,
                   cudaStream_t stream) {
  constexpr int threads = 256;
  int blocks = int((count + threads - 1) / threads);
  float_to_bf16_kernel<<<blocks, threads, 0, stream>>>(src, dst, count);
  CUDA_CHECK(cudaGetLastError());
}

std::vector<int> seq_counts_up_to(int max_seq) {
  std::vector<int> counts;
  for (int seq : {1, 4, 16, 64, 256, 1024, 4096}) {
    if (seq <= max_seq) {
      counts.push_back(seq);
    }
  }
  if (counts.empty() || counts.back() != max_seq) {
    counts.push_back(max_seq);
  }
  return counts;
}

double gib_per_second(double bytes, float ms) {
  if (ms <= 0.0f) {
    return 0.0;
  }
  double gib = bytes / (1024.0 * 1024.0 * 1024.0);
  return gib / (static_cast<double>(ms) / 1000.0);
}

double ideal_rope_bytes(int batch_size,
                        int seq_len,
                        int q_heads,
                        int kv_heads,
                        int rotary_dim) {
  int heads = q_heads + kv_heads;
  int rotary_half = rotary_dim / 2;
  double bf16_load_store =
      double(batch_size) * seq_len * heads * rotary_dim *
      sizeof(__nv_bfloat16) * 2.0;
  double table_load =
      double(batch_size) * seq_len * heads * rotary_half * sizeof(float) * 2.0;
  return bf16_load_store + table_load;
}

#if GEMMA4_HAS_CUDNN

void check_cudnn(cudnnStatus_t status, const char *expr) {
  if (status != CUDNN_STATUS_SUCCESS) {
    throw std::runtime_error(std::string("cuDNN error for ") + expr + ": " +
                             cudnnGetErrorString(status));
  }
}

void set_tensor_desc(cudnnTensorDescriptor_t desc,
                     cudnnDataType_t data_type,
                     int n,
                     int h,
                     int s,
                     int d,
                     int n_stride,
                     int h_stride,
                     int s_stride,
                     int d_stride) {
  int dims[4] = {n, h, s, d};
  int strides[4] = {n_stride, h_stride, s_stride, d_stride};
  check_cudnn(cudnnSetTensorNdDescriptor(desc, data_type, 4, dims, strides),
              "cudnnSetTensorNdDescriptor");
}

struct TensorDesc {
  cudnnTensorDescriptor_t desc = nullptr;

  TensorDesc() {
    check_cudnn(cudnnCreateTensorDescriptor(&desc),
                "cudnnCreateTensorDescriptor");
  }

  ~TensorDesc() {
    if (desc != nullptr) {
      cudnnDestroyTensorDescriptor(desc);
    }
  }

  TensorDesc(const TensorDesc &) = delete;
  TensorDesc &operator=(const TensorDesc &) = delete;
};

struct OpDesc {
  cudnnOpTensorDescriptor_t desc = nullptr;

  explicit OpDesc(cudnnOpTensorOp_t op) {
    check_cudnn(cudnnCreateOpTensorDescriptor(&desc),
                "cudnnCreateOpTensorDescriptor");
    check_cudnn(cudnnSetOpTensorDescriptor(desc, op, CUDNN_DATA_FLOAT,
                                           CUDNN_PROPAGATE_NAN),
                "cudnnSetOpTensorDescriptor");
  }

  ~OpDesc() {
    if (desc != nullptr) {
      cudnnDestroyOpTensorDescriptor(desc);
    }
  }

  OpDesc(const OpDesc &) = delete;
  OpDesc &operator=(const OpDesc &) = delete;
};

struct CudnnRope {
  cudnnHandle_t handle = nullptr;
  TensorDesc q_half_desc;
  TensorDesc k_half_desc;
  TensorDesc q_temp_desc;
  TensorDesc k_temp_desc;
  TensorDesc table_desc;
  OpDesc mul_desc{CUDNN_OP_TENSOR_MUL};
  OpDesc add_desc{CUDNN_OP_TENSOR_ADD};
  __nv_bfloat16 *tmp1 = nullptr;
  __nv_bfloat16 *tmp2 = nullptr;
  __nv_bfloat16 *tmp3 = nullptr;
  __nv_bfloat16 *tmp4 = nullptr;
  int q_heads = 0;
  int kv_heads = 0;
  int rotary_half = 0;

  CudnnRope(int batch_size,
            int seq_len,
            int cos_batch_size,
            int q_heads_,
            int kv_heads_,
            int head_dim,
            int rotary_dim,
            cudaStream_t stream)
      : q_heads(q_heads_), kv_heads(kv_heads_), rotary_half(rotary_dim / 2) {
    check_cudnn(cudnnCreate(&handle), "cudnnCreate");
    check_cudnn(cudnnSetStream(handle, stream), "cudnnSetStream");

    set_tensor_desc(q_half_desc.desc, CUDNN_DATA_BFLOAT16, batch_size,
                    q_heads, seq_len, rotary_half,
                    q_heads * seq_len * head_dim, seq_len * head_dim,
                    head_dim, 1);
    set_tensor_desc(k_half_desc.desc, CUDNN_DATA_BFLOAT16, batch_size,
                    kv_heads, seq_len, rotary_half,
                    kv_heads * seq_len * head_dim, seq_len * head_dim,
                    head_dim, 1);
    set_tensor_desc(q_temp_desc.desc, CUDNN_DATA_BFLOAT16, batch_size, q_heads,
                    seq_len, rotary_half, q_heads * seq_len * rotary_half,
                    seq_len * rotary_half, rotary_half, 1);
    set_tensor_desc(k_temp_desc.desc, CUDNN_DATA_BFLOAT16, batch_size, kv_heads,
                    seq_len, rotary_half, kv_heads * seq_len * rotary_half,
                    seq_len * rotary_half, rotary_half, 1);
    set_tensor_desc(table_desc.desc, CUDNN_DATA_BFLOAT16, cos_batch_size, 1,
                    seq_len, rotary_half, seq_len * rotary_half,
                    seq_len * rotary_half, rotary_half, 1);

    size_t max_temp_elems =
        static_cast<size_t>(batch_size) * std::max(q_heads, kv_heads) *
        seq_len * rotary_half;
    CUDA_CHECK(cudaMalloc(&tmp1, max_temp_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&tmp2, max_temp_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&tmp3, max_temp_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&tmp4, max_temp_elems * sizeof(__nv_bfloat16)));
  }

  ~CudnnRope() {
    if (tmp1 != nullptr) {
      cudaFree(tmp1);
    }
    if (tmp2 != nullptr) {
      cudaFree(tmp2);
    }
    if (tmp3 != nullptr) {
      cudaFree(tmp3);
    }
    if (tmp4 != nullptr) {
      cudaFree(tmp4);
    }
    if (handle != nullptr) {
      cudnnDestroy(handle);
    }
  }

  CudnnRope(const CudnnRope &) = delete;
  CudnnRope &operator=(const CudnnRope &) = delete;

  void run_branch(cudnnTensorDescriptor_t half_desc,
                  cudnnTensorDescriptor_t temp_desc,
                  const __nv_bfloat16 *in,
                  const __nv_bfloat16 *cos,
                  const __nv_bfloat16 *sin,
                  __nv_bfloat16 *out) {
    const __nv_bfloat16 *first = in;
    const __nv_bfloat16 *second = in + rotary_half;
    __nv_bfloat16 *out_first = out;
    __nv_bfloat16 *out_second = out + rotary_half;

    const float one = 1.0f;
    const float neg_one = -1.0f;
    const float zero = 0.0f;

    check_cudnn(cudnnOpTensor(handle, mul_desc.desc, &one, half_desc, first,
                              &one, table_desc.desc, cos, &zero, temp_desc,
                              tmp1),
                "cudnnOpTensor first*cos");
    check_cudnn(cudnnOpTensor(handle, mul_desc.desc, &one, half_desc, second,
                              &one, table_desc.desc, sin, &zero, temp_desc,
                              tmp2),
                "cudnnOpTensor second*sin");
    check_cudnn(cudnnOpTensor(handle, add_desc.desc, &one, temp_desc, tmp1,
                              &neg_one, temp_desc, tmp2, &zero, half_desc,
                              out_first),
                "cudnnOpTensor out_first");

    check_cudnn(cudnnOpTensor(handle, mul_desc.desc, &one, half_desc, second,
                              &one, table_desc.desc, cos, &zero, temp_desc,
                              tmp3),
                "cudnnOpTensor second*cos");
    check_cudnn(cudnnOpTensor(handle, mul_desc.desc, &one, half_desc, first,
                              &one, table_desc.desc, sin, &zero, temp_desc,
                              tmp4),
                "cudnnOpTensor first*sin");
    check_cudnn(cudnnOpTensor(handle, add_desc.desc, &one, temp_desc, tmp3,
                              &one, temp_desc, tmp4, &zero, half_desc,
                              out_second),
                "cudnnOpTensor out_second");
  }

  void run(const __nv_bfloat16 *q_in,
           const __nv_bfloat16 *k_in,
           const __nv_bfloat16 *cos_in,
           const __nv_bfloat16 *sin_in,
           __nv_bfloat16 *q_out,
           __nv_bfloat16 *k_out) {
    run_branch(q_half_desc.desc, q_temp_desc.desc, q_in, cos_in, sin_in, q_out);
    run_branch(k_half_desc.desc, k_temp_desc.desc, k_in, cos_in, sin_in, k_out);
  }
};

#endif

}  // namespace

int main(int argc, char **argv) {
  const int iters = argc > 1 ? std::atoi(argv[1]) : 200;
  const int warmup = argc > 2 ? std::atoi(argv[2]) : 30;
  const int trials = argc > 3 ? std::atoi(argv[3]) : 5;
  const int max_seq = argc > 4 ? std::atoi(argv[4]) : 1024;
  const int batch_size = argc > 5 ? std::atoi(argv[5]) : 1;
  const int cos_batch_size = argc > 6 ? std::atoi(argv[6]) : 1;
  const bool run_cudnn_compare = argc > 7 ? std::atoi(argv[7]) != 0 : true;

  if (iters <= 0 || warmup < 0 || trials <= 0 || max_seq <= 0 ||
      batch_size <= 0 ||
      (cos_batch_size != 1 && cos_batch_size != batch_size)) {
    std::fprintf(
        stderr,
        "usage: %s [iters=200] [warmup=30] [trials=5] [max_seq=1024] "
        "[batch=1] [cos_batch=1|batch] [run_cudnn=1]\n",
        argv[0]);
    return 1;
  }

  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  int device = 0;
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDevice(&device));
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

  const uint64_t seed = make_seed();
  const RopeShape shapes[] = {
      {"sliding", GEMMA4_NUM_QUERY_HEADS, GEMMA4_SLIDING_KV_HEADS,
       GEMMA4_SLIDING_HEAD_DIM, GEMMA4_SLIDING_HEAD_DIM},
      {"global", GEMMA4_NUM_QUERY_HEADS, GEMMA4_GLOBAL_KV_HEADS,
       GEMMA4_GLOBAL_HEAD_DIM, GEMMA4_GLOBAL_HEAD_DIM / 4},
  };

  std::printf("device=%s\n", prop.name);
  std::printf("seed=0x%llx\n", static_cast<unsigned long long>(seed));
  std::printf("iters=%d,warmup_iters=%d,trials=%d,max_seq=%d,batch=%d,"
              "cos_batch=%d,run_cudnn=%d\n",
              iters, warmup, trials, max_seq, batch_size, cos_batch_size,
              run_cudnn_compare ? 1 : 0);
  std::printf("cudnn_op_tensor=%s\n",
#if GEMMA4_HAS_CUDNN
              run_cudnn_compare ? "compiled" : "skipped"
#else
              "not_compiled"
#endif
  );
  std::printf("case,seq,q_heads,kv_heads,head_dim,rotary_dim,"
              "physical_ms,physical_gib_s,physical_graph_ms,"
              "physical_graph_gib_s,forward_ms,forward_gib_s,"
              "forward_graph_ms,forward_graph_gib_s,"
              "cudnn_ms,cudnn_gib_s,cudnn_graph_ms,cudnn_graph_gib_s,"
              "cudnn_q_max_abs,cudnn_k_max_abs\n");
  std::fflush(stdout);

  for (const RopeShape &shape : shapes) {
    for (int seq_len : seq_counts_up_to(max_seq)) {
      const size_t q_elems = static_cast<size_t>(batch_size) * shape.q_heads *
                             seq_len * shape.head_dim;
      const size_t k_elems = static_cast<size_t>(batch_size) * shape.kv_heads *
                             seq_len * shape.head_dim;
      const size_t table_elems =
          static_cast<size_t>(cos_batch_size) * seq_len *
          (shape.rotary_dim / 2);
      const double ideal_bytes =
          ideal_rope_bytes(batch_size, seq_len, shape.q_heads, shape.kv_heads,
                           shape.rotary_dim);

      __nv_bfloat16 *d_q_in = nullptr;
      __nv_bfloat16 *d_k_in = nullptr;
      __nv_bfloat16 *d_q_work = nullptr;
      __nv_bfloat16 *d_k_work = nullptr;
      __nv_bfloat16 *d_q_cudnn = nullptr;
      __nv_bfloat16 *d_k_cudnn = nullptr;
      float *d_cos = nullptr;
      float *d_sin = nullptr;
      __nv_bfloat16 *d_cos_bf16 = nullptr;
      __nv_bfloat16 *d_sin_bf16 = nullptr;

      CUDA_CHECK(cudaMalloc(&d_q_in, q_elems * sizeof(__nv_bfloat16)));
      CUDA_CHECK(cudaMalloc(&d_k_in, k_elems * sizeof(__nv_bfloat16)));
      CUDA_CHECK(cudaMalloc(&d_q_work, q_elems * sizeof(__nv_bfloat16)));
      CUDA_CHECK(cudaMalloc(&d_k_work, k_elems * sizeof(__nv_bfloat16)));
      CUDA_CHECK(cudaMalloc(&d_q_cudnn, q_elems * sizeof(__nv_bfloat16)));
      CUDA_CHECK(cudaMalloc(&d_k_cudnn, k_elems * sizeof(__nv_bfloat16)));
      CUDA_CHECK(cudaMalloc(&d_cos, table_elems * sizeof(float)));
      CUDA_CHECK(cudaMalloc(&d_sin, table_elems * sizeof(float)));
      CUDA_CHECK(cudaMalloc(&d_cos_bf16, table_elems * sizeof(__nv_bfloat16)));
      CUDA_CHECK(cudaMalloc(&d_sin_bf16, table_elems * sizeof(__nv_bfloat16)));

      fill_random_bf16(d_q_in, q_elems, seed ^ 0x1111u, 1.0f, stream);
      fill_random_bf16(d_k_in, k_elems, seed ^ 0x2222u, 1.0f, stream);
      fill_unit_rope_table(d_cos, d_sin, cos_batch_size, seq_len,
                           shape.rotary_dim / 2, seed ^ 0x3333u, stream);
      float_to_bf16(d_cos, d_cos_bf16, table_elems, stream);
      float_to_bf16(d_sin, d_sin_bf16, table_elems, stream);
      CUDA_CHECK(cudaMemcpyAsync(d_q_work, d_q_in,
                                 q_elems * sizeof(__nv_bfloat16),
                                 cudaMemcpyDeviceToDevice, stream));
      CUDA_CHECK(cudaMemcpyAsync(d_k_work, d_k_in,
                                 k_elems * sizeof(__nv_bfloat16),
                                 cudaMemcpyDeviceToDevice, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));

      auto run_physical = [&]() {
        CUDA_CHECK(gemma4_rope_bf16(
            d_q_work, static_cast<int64_t>(shape.q_heads) * shape.head_dim,
            d_k_work, static_cast<int64_t>(shape.kv_heads) * shape.head_dim,
            d_cos, shape.rotary_dim / 2, d_sin, shape.rotary_dim / 2,
            seq_len, batch_size, cos_batch_size, shape.q_heads,
            shape.kv_heads, shape.head_dim, shape.rotary_dim, stream));
      };

      run_physical();
      CUDA_CHECK(cudaStreamSynchronize(stream));

      TimingStats physical_stats =
          time_ms(run_physical, stream, warmup, iters, trials);
      float physical_graph_ms = -1.0f;
      double physical_graph_gib_s = 0.0;
      try {
        TimingStats graph_stats =
            time_ms_graph(run_physical, stream, warmup, iters, trials);
        physical_graph_ms = graph_stats.best_ms;
        physical_graph_gib_s = gib_per_second(ideal_bytes,
                                              physical_graph_ms);
      } catch (const std::exception &e) {
        std::fprintf(stderr,
                     "physical RoPE CUDA graph timing unavailable for %s "
                     "seq=%d: %s\n",
                     shape.name, seq_len, e.what());
      }

      CUDA_CHECK(cudaMemcpyAsync(d_q_work, d_q_in,
                                 q_elems * sizeof(__nv_bfloat16),
                                 cudaMemcpyDeviceToDevice, stream));
      CUDA_CHECK(cudaMemcpyAsync(d_k_work, d_k_in,
                                 k_elems * sizeof(__nv_bfloat16),
                                 cudaMemcpyDeviceToDevice, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));

      auto run_forward = [&]() {
        CUDA_CHECK(gemma4_rope_forward_bf16(
            d_q_work, d_k_work, d_cos, d_sin, seq_len, batch_size,
            cos_batch_size, shape.q_heads, shape.kv_heads, shape.head_dim,
            shape.rotary_dim, stream));
      };

      run_forward();
      CUDA_CHECK(cudaStreamSynchronize(stream));

      TimingStats forward_stats =
          time_ms(run_forward, stream, warmup, iters, trials);
      float forward_graph_ms = -1.0f;
      double forward_graph_gib_s = 0.0;
      try {
        TimingStats graph_stats =
            time_ms_graph(run_forward, stream, warmup, iters, trials);
        forward_graph_ms = graph_stats.best_ms;
        forward_graph_gib_s = gib_per_second(ideal_bytes, forward_graph_ms);
      } catch (const std::exception &e) {
        std::fprintf(stderr,
                     "forward RoPE CUDA graph timing unavailable for %s seq=%d: %s\n",
                     shape.name, seq_len, e.what());
      }

      float cudnn_ms = -1.0f;
      double cudnn_gib_s = 0.0;
      float cudnn_graph_ms = -1.0f;
      double cudnn_graph_gib_s = 0.0;
      float cudnn_q_max_abs = -1.0f;
      float cudnn_k_max_abs = -1.0f;

#if GEMMA4_HAS_CUDNN
      if (run_cudnn_compare) {
      try {
        CudnnRope cudnn(batch_size, seq_len, cos_batch_size, shape.q_heads,
                        shape.kv_heads, shape.head_dim, shape.rotary_dim,
                        stream);
        auto run_cudnn = [&]() {
          cudnn.run(d_q_in, d_k_in, d_cos_bf16, d_sin_bf16, d_q_cudnn, d_k_cudnn);
        };

        CUDA_CHECK(cudaMemcpyAsync(d_q_work, d_q_in,
                                   q_elems * sizeof(__nv_bfloat16),
                                   cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_k_work, d_k_in,
                                   k_elems * sizeof(__nv_bfloat16),
                                   cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_q_cudnn, d_q_in,
                                   q_elems * sizeof(__nv_bfloat16),
                                   cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_k_cudnn, d_k_in,
                                   k_elems * sizeof(__nv_bfloat16),
                                   cudaMemcpyDeviceToDevice, stream));
        run_forward();
        run_cudnn();
        CUDA_CHECK(cudaStreamSynchronize(stream));

        DiffStats q_diff =
            diff_stats_bf16(d_q_work, d_q_cudnn, static_cast<int>(q_elems));
        DiffStats k_diff =
            diff_stats_bf16(d_k_work, d_k_cudnn, static_cast<int>(k_elems));
        cudnn_q_max_abs = q_diff.max_abs;
        cudnn_k_max_abs = k_diff.max_abs;

        TimingStats cudnn_stats =
            time_ms(run_cudnn, stream, warmup, iters, trials);
        cudnn_ms = cudnn_stats.best_ms;
        cudnn_gib_s = gib_per_second(ideal_bytes, cudnn_ms);

        try {
          TimingStats cudnn_graph_stats =
              time_ms_graph(run_cudnn, stream, warmup, iters, trials);
          cudnn_graph_ms = cudnn_graph_stats.best_ms;
          cudnn_graph_gib_s = gib_per_second(ideal_bytes, cudnn_graph_ms);
        } catch (const std::exception &e) {
          std::fprintf(stderr,
                       "cuDNN RoPE CUDA graph timing unavailable for %s seq=%d: %s\n",
                       shape.name, seq_len, e.what());
        }
      } catch (const std::exception &e) {
        std::fprintf(stderr, "cuDNN RoPE unavailable for %s seq=%d: %s\n",
                     shape.name, seq_len, e.what());
      }
      }
#endif

      std::printf("%s,%d,%d,%d,%d,%d,%.6f,%.3f,%.6f,%.3f,"
                  "%.6f,%.3f,%.6f,%.3f,%.6f,%.3f,%.6f,%.3f,%.6g,%.6g\n",
                  shape.name, seq_len, shape.q_heads, shape.kv_heads,
                  shape.head_dim, shape.rotary_dim, physical_stats.best_ms,
                  gib_per_second(ideal_bytes, physical_stats.best_ms),
                  physical_graph_ms, physical_graph_gib_s,
                  forward_stats.best_ms,
                  gib_per_second(ideal_bytes, forward_stats.best_ms),
                  forward_graph_ms, forward_graph_gib_s, cudnn_ms, cudnn_gib_s,
                  cudnn_graph_ms, cudnn_graph_gib_s, cudnn_q_max_abs,
                  cudnn_k_max_abs);
      std::fflush(stdout);

      CUDA_CHECK(cudaFree(d_q_in));
      CUDA_CHECK(cudaFree(d_k_in));
      CUDA_CHECK(cudaFree(d_q_work));
      CUDA_CHECK(cudaFree(d_k_work));
      CUDA_CHECK(cudaFree(d_q_cudnn));
      CUDA_CHECK(cudaFree(d_k_cudnn));
      CUDA_CHECK(cudaFree(d_cos));
      CUDA_CHECK(cudaFree(d_sin));
      CUDA_CHECK(cudaFree(d_cos_bf16));
      CUDA_CHECK(cudaFree(d_sin_bf16));
    }
  }

  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}
