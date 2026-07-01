#include "gemma4_cuda_utils.cuh"

#include <cuda_runtime.h>

#include <cmath>
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

__global__ void warp_reduce_sum_real_kernel(const float *inp, float *out,
                                            int warps_per_block) {
  int lane = threadIdx.x & (warpSize - 1);
  int warp = threadIdx.x / warpSize;
  int global_warp = blockIdx.x * warps_per_block + warp;
  int index = global_warp * warpSize + lane;

  float value = inp[index];
  out[index] = warp_reduce_sum(value);
}

}  // namespace

int main() {
  constexpr int blocks = 2;
  constexpr int warps_per_block = 3;
  constexpr int total_warps = blocks * warps_per_block;
  int device = 0;
  cudaDeviceProp prop{};
  CHECK_CUDA(cudaGetDevice(&device));
  CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
  const int total_values = total_warps * prop.warpSize;
  const int threads = warps_per_block * prop.warpSize;

  std::vector<float> inp(total_values);
  std::vector<float> out(total_values);
  std::vector<float> expected(total_warps);

  for (int i = 0; i < total_values; ++i) {
    const int centered = ((i * 37 + 11) % 113) - 56;
    const float base = static_cast<float>(centered) / 7.0f;
    inp[i] = base + 0.001f * static_cast<float>((i % 5) - 2);
  }

  for (int warp = 0; warp < total_warps; ++warp) {
    float sum = 0.0f;
    for (int lane = 0; lane < prop.warpSize; ++lane) {
      sum += inp[warp * prop.warpSize + lane];
    }
    expected[warp] = sum;
  }

  float *d_inp = nullptr;
  float *d_out = nullptr;
  CHECK_CUDA(cudaMalloc(&d_inp, inp.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_out, out.size() * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(d_inp, inp.data(), inp.size() * sizeof(float), cudaMemcpyHostToDevice));

  warp_reduce_sum_real_kernel<<<blocks, threads>>>(d_inp, d_out, warps_per_block);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaMemcpy(out.data(), d_out, out.size() * sizeof(float), cudaMemcpyDeviceToHost));

  for (int warp = 0; warp < total_warps; ++warp) {
    for (int lane = 0; lane < prop.warpSize; ++lane) {
      int index = warp * prop.warpSize + lane;
      float diff = std::fabs(out[index] - expected[warp]);
      if (diff > 1.0e-5f) {
        std::fprintf(stderr,
                     "warp_reduce_sum mismatch warp=%d lane=%d actual=%0.9g "
                     "expected=%0.9g diff=%0.9g\n",
                     warp, lane, out[index], expected[warp], diff);
        return 1;
      }
    }
  }

  CHECK_CUDA(cudaFree(d_inp));
  CHECK_CUDA(cudaFree(d_out));

  std::printf("cuda utils tests passed\n");
  return 0;
}
