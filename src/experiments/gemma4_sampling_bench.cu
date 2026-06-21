#include "gemma4_sampling.cuh"
#include "gemma4_cuda_utils.cuh"
#include "gemma4_embedding_gather.cuh"
#include "gemma4_matmul_kernels.cuh"
#include "gemma4_sampling_device.cuh"
#include "gemma4.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>

namespace {

namespace sampling_dev = gemma4_sampling_device;

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
  explicit DeviceBuffer(size_t count) {
    if (count > 0) {
      CHECK_CUDA(cudaMalloc(&ptr_, count * sizeof(T)));
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

 private:
  T *ptr_ = nullptr;
};

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

void fill_random_bf16(__nv_bfloat16 *ptr,
                      size_t count,
                      uint64_t seed,
                      float scale,
                      cudaStream_t stream) {
  constexpr int threads = 256;
  const int blocks = int((count + threads - 1) / threads);
  fill_random_bf16_kernel<<<blocks, threads, 0, stream>>>(
      ptr, count, seed, scale);
  CHECK_CUDA(cudaGetLastError());
}

__device__ inline bool better_candidate(float logit,
                                        int32_t token_id,
                                        float best_logit,
                                        int32_t best_token_id) {
  return logit > best_logit ||
         (logit == best_logit && token_id < best_token_id);
}

// Samples from materialized final logits with the same full-vocab Gumbel-Max contract.
__global__ __launch_bounds__(1024) void materialized_logits_gumbel_embed_kernel(
    __nv_bfloat16 *__restrict__ d_next_hidden,
    int32_t *__restrict__ d_next_token,
    const __nv_bfloat16 *__restrict__ d_lm_head_col_major,
    const __nv_bfloat16 *__restrict__ d_logits,
    Gemma4GumbelSamplingParams params) {
  __shared__ float s_scores[1024];
  __shared__ int32_t s_token_ids[1024];

  const int thread_idx = threadIdx.x;
  const float inv_temperature = 1.0f / params.temperature;
  float best_score = sampling_dev::kNegativeInfinity;
  int32_t best_token_id = GEMMA4_VOCAB_SIZE;

  for (int token_id = thread_idx; token_id < GEMMA4_VOCAB_SIZE;
       token_id += blockDim.x) {
    const float raw_logit = __bfloat162float(d_logits[token_id]);
    const float score =
        sampling_dev::transformed_lm_head_score(raw_logit, inv_temperature) +
        sampling_dev::gumbel_noise(params.seed, params.step, token_id);
    if (better_candidate(score, token_id, best_score, best_token_id)) {
      best_score = score;
      best_token_id = token_id;
    }
  }

  s_scores[thread_idx] = best_score;
  s_token_ids[thread_idx] = best_token_id;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (thread_idx < stride) {
      const float other_score = s_scores[thread_idx + stride];
      const int32_t other_token_id = s_token_ids[thread_idx + stride];
      if (better_candidate(other_score, other_token_id, s_scores[thread_idx],
                           s_token_ids[thread_idx])) {
        s_scores[thread_idx] = other_score;
        s_token_ids[thread_idx] = other_token_id;
      }
    }
    __syncthreads();
  }

  const int32_t selected_token_id = s_token_ids[0];
  if (thread_idx == 0) {
    *d_next_token = selected_token_id;
  }
  gemma4_embedding_gather::copy_embedding_row_bf16(
      d_next_hidden, d_lm_head_col_major, selected_token_id, thread_idx,
      blockDim.x);
}

template <typename Fn>
std::vector<float> time_samples(Fn &&fn,
                                cudaStream_t stream,
                                int warmup,
                                int iters,
                                int samples) {
  for (int i = 0; i < warmup; ++i) {
    fn();
  }
  CHECK_CUDA(cudaStreamSynchronize(stream));

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  std::vector<float> values;
  values.reserve(samples);
  for (int sample = 0; sample < samples; ++sample) {
    CHECK_CUDA(cudaEventRecord(start, stream));
    for (int i = 0; i < iters; ++i) {
      fn();
    }
    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float total_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));
    values.push_back(total_ms / float(iters));
  }

  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  return values;
}

float percentile(std::vector<float> sorted, float p) {
  std::sort(sorted.begin(), sorted.end());
  const float scaled = p * float(sorted.size() - 1);
  const size_t lo = size_t(scaled);
  const size_t hi = std::min(lo + 1, sorted.size() - 1);
  const float t = scaled - float(lo);
  return sorted[lo] * (1.0f - t) + sorted[hi] * t;
}

void print_stats(const char *variant,
                 int batch_size,
                 int top_k,
                 float top_p,
                 float temperature,
                 const std::vector<float> &samples) {
  const float sum = std::accumulate(samples.begin(), samples.end(), 0.0f);
  const float mean = sum / float(samples.size());
  auto sorted = samples;
  std::sort(sorted.begin(), sorted.end());
  std::printf("variant=%s batch_size=%d top_k=%d top_p=%.3f temp=%.3f "
              "min_us=%.3f median_us=%.3f mean_us=%.3f p95_us=%.3f "
              "max_us=%.3f samples_us=[",
              variant, batch_size, top_k, top_p, temperature,
              sorted.front() * 1000.0f, percentile(samples, 0.50f) * 1000.0f,
              mean * 1000.0f, percentile(samples, 0.95f) * 1000.0f,
              sorted.back() * 1000.0f);
  for (size_t i = 0; i < samples.size(); ++i) {
    std::printf("%s%.3f", i == 0 ? "" : ",", samples[i] * 1000.0f);
  }
  std::printf("]\n");
}

}  // namespace

int main(int argc, char **argv) {
  const int warmup = argc > 1 ? std::atoi(argv[1]) : 25;
  const int iters = argc > 2 ? std::atoi(argv[2]) : 100;
  const int samples = argc > 3 ? std::atoi(argv[3]) : 21;
  if (warmup < 0 || iters <= 0 || samples <= 0) {
    std::fprintf(stderr,
                 "usage: %s [warmup=25] [iters=100] [samples=21]\n",
                 argv[0]);
    return 1;
  }

  cudaStream_t stream = nullptr;
  CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  cudaDeviceProp prop = {};
  CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
  std::printf("env=gpu=\"%s\" global_mem_bytes=%zu\n",
              prop.name, static_cast<size_t>(prop.totalGlobalMem));
  std::printf("contract=sampling_decode timing=cuda_events "
              "stream=nonblocking cache=warm-ish launch_overhead=included "
              "warmup=%d iters=%d samples=%d dtype=bf16 "
              "vocab=%d hidden=%d\n",
              warmup, iters, samples, GEMMA4_VOCAB_SIZE, GEMMA4_HIDDEN_SIZE);

  const size_t lm_head_elems =
      static_cast<size_t>(GEMMA4_VOCAB_SIZE) * GEMMA4_HIDDEN_SIZE;
  DeviceBuffer<__nv_bfloat16> d_lm_head(lm_head_elems);
  DeviceBuffer<__nv_bfloat16> d_final_hidden(GEMMA4_HIDDEN_SIZE);
  DeviceBuffer<__nv_bfloat16> d_fused_hidden(GEMMA4_HIDDEN_SIZE);
  DeviceBuffer<__nv_bfloat16> d_materialized_logits(GEMMA4_VOCAB_SIZE);
  DeviceBuffer<__nv_bfloat16> d_materialized_hidden(GEMMA4_HIDDEN_SIZE);
  DeviceBuffer<int32_t> d_fused_token(1);
  DeviceBuffer<int32_t> d_materialized_token(1);
  DeviceBuffer<unsigned char> d_fused_scratch(
      gemma4_gumbel_sample_next_scratch_bytes());

  fill_random_bf16(d_lm_head.get(), lm_head_elems, 0x5678u, 0.05f, stream);
  fill_random_bf16(d_final_hidden.get(), GEMMA4_HIDDEN_SIZE, 0x2468u, 0.05f,
                   stream);
  CHECK_CUDA(cudaStreamSynchronize(stream));

  Gemma4GumbelSamplingParams fused_params = {1.0f, 0x12345678u, 5u};
  auto fused_gumbel = [&]() {
    CHECK_CUDA(gemma4_gumbel_sample_next_decode_bf16(
        d_fused_hidden.get(), d_fused_token.get(), d_fused_scratch.get(),
        gemma4_gumbel_sample_next_scratch_bytes(), d_final_hidden.get(),
        d_lm_head.get(), fused_params, stream));
  };
  auto materialized_gumbel = [&]() {
    CHECK_CUDA(gemma4_projection_decode(
        GEMMA4_PROJECTION_FINAL_LOGITS, d_final_hidden.get(),
        d_lm_head.get(), d_materialized_logits.get(), stream));
    materialized_logits_gumbel_embed_kernel<<<1, 1024, 0, stream>>>(
        d_materialized_hidden.get(), d_materialized_token.get(),
        d_lm_head.get(), d_materialized_logits.get(), fused_params);
    CHECK_CUDA(cudaGetLastError());
  };
  fused_gumbel();
  materialized_gumbel();
  CHECK_CUDA(cudaStreamSynchronize(stream));
  int32_t fused_token = -1;
  int32_t materialized_token = -2;
  CHECK_CUDA(cudaMemcpy(&fused_token, d_fused_token.get(), sizeof(fused_token),
                        cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(&materialized_token, d_materialized_token.get(),
                        sizeof(materialized_token), cudaMemcpyDeviceToHost));
  if (fused_token != materialized_token) {
    std::fprintf(stderr,
                 "fused/materialized mismatch fused=%d materialized=%d\n",
                 fused_token, materialized_token);
    return 1;
  }
  print_stats("fused_lm_head_gumbel_full_vocab", 1, GEMMA4_VOCAB_SIZE,
              1.0f, fused_params.temperature,
              time_samples(fused_gumbel, stream, warmup, iters, samples));
  print_stats("materialized_lm_head_gumbel_full_vocab", 1, GEMMA4_VOCAB_SIZE,
              1.0f, fused_params.temperature,
              time_samples(materialized_gumbel, stream, warmup, iters,
                           samples));

  CHECK_CUDA(cudaStreamDestroy(stream));
  return 0;
}
