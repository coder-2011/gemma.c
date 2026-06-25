#include "gemma4.h"
#include "gemma4_bench_utils.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cutlass/array.h>
#include <cutlass/epilogue/thread/activation.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/epilogue/thread/scale_type.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

#include "device/dual_gemm.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kFillThreads = 256;
constexpr float kCorrectnessTolerance = 1.0f;

enum DualGemmConfigId {
  kCurrent128x64x32S3 = 0,
  kDual128x64S4,
  kDual128x64S5,
  kDual128x64S6,
  kDual128x64S10,
  kDual128x64x64S3,
  kDual64x64W64x32S3,
  kDual128x32S3,
  kDual256x64S3,
  kDual64x64S10,
  kDual64x128S6,
  kDual128x128S5,
  kDual128x256S3,
  kDual256x128S3,
  kDual128x128x64S3,
};

struct DualGemmConfig {
  const char *name;
  DualGemmConfigId id;
};

constexpr DualGemmConfig kDualConfigs[] = {
    {"current_128x64x32_s3", kCurrent128x64x32S3},
    {"dual_128x64_s4", kDual128x64S4},
    {"dual_128x64_s5", kDual128x64S5},
    {"dual_128x64_s6", kDual128x64S6},
    {"dual_128x64_s10", kDual128x64S10},
    {"dual_128x64x64_s3", kDual128x64x64S3},
    {"dual_64x64_w64x32_s3", kDual64x64W64x32S3},
    {"dual_128x32_s3", kDual128x32S3},
    {"dual_256x64_s3", kDual256x64S3},
    {"dual_64x64_s10", kDual64x64S10},
    {"dual_64x128_s6", kDual64x128S6},
    {"dual_128x128_s5", kDual128x128S5},
    {"dual_128x256_s3", kDual128x256S3},
    {"dual_256x128_s3", kDual256x128S3},
    {"dual_128x128x64_s3", kDual128x128x64S3},
};

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

 private:
  T *ptr_ = nullptr;
  size_t count_ = 0;
};

// Mixes integer indices into deterministic pseudo-random bits.
__device__ uint32_t mix_u32(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}

// Fills BF16 buffers on device so initialization stays outside timed regions.
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

// Launches the deterministic BF16 fill kernel.
void fill_random_bf16(__nv_bfloat16 *ptr,
                      size_t count,
                      uint64_t seed,
                      float scale,
                      cudaStream_t stream) {
  const int blocks = int((count + kFillThreads - 1) / kFillThreads);
  fill_random_bf16_kernel<<<blocks, kFillThreads, 0, stream>>>(
      ptr, count, seed, scale);
  CUDA_CHECK(cudaGetLastError());
}

// Parses comma-separated row counts such as "16,64,128".
std::vector<int> parse_rows(const char *text) {
  std::vector<int> rows;
  std::stringstream stream(text);
  std::string item;
  while (std::getline(stream, item, ',')) {
    const int value = std::atoi(item.c_str());
    if (value <= 0) {
      throw std::runtime_error("row counts must be positive");
    }
    rows.push_back(value);
  }
  if (rows.empty()) {
    throw std::runtime_error("at least one row count is required");
  }
  return rows;
}

// Looks up one benchmark config by name.
const DualGemmConfig *find_config(const char *name) {
  for (const DualGemmConfig &config : kDualConfigs) {
    if (std::strcmp(config.name, name) == 0) {
      return &config;
    }
  }
  return nullptr;
}

template <typename ElementOutput_,
          int Count,
          typename ElementAccumulator_,
          typename ElementCompute_,
          cutlass::FloatRoundStyle Round =
              cutlass::FloatRoundStyle::round_to_nearest>
class GateGeluTanhUpMul {
 public:
  using ElementOutput = ElementOutput_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementCompute = ElementCompute_;
  using FragmentOutput = cutlass::Array<ElementOutput, Count>;
  using FragmentAccumulator = cutlass::Array<ElementAccumulator, Count>;
  using FragmentCompute = cutlass::Array<ElementCompute, Count>;

  struct Params {};

  CUTLASS_HOST_DEVICE
  explicit GateGeluTanhUpMul(Params const &) {}

  // Applies Gemma's GeGLU activation to the two GEMM accumulator fragments.
  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(FragmentAccumulator const &gate,
                            FragmentAccumulator const &up) const {
    cutlass::NumericArrayConverter<ElementCompute, ElementAccumulator, Count,
                                   Round>
        to_compute;
    cutlass::NumericArrayConverter<ElementOutput, ElementCompute, Count,
                                   Round>
        to_output;
    FragmentCompute gate_compute = to_compute(gate);
    FragmentCompute up_compute = to_compute(up);
    cutlass::epilogue::thread::GELU_taylor<FragmentCompute> gelu;
    cutlass::multiplies<FragmentCompute> multiply;
    return to_output(multiply(gelu(gate_compute), up_compute));
  }
};

template <int ThreadblockM,
          int ThreadblockN,
          int ThreadblockK,
          int WarpM,
          int WarpN,
          int Stages>
cudaError_t launch_dual_gemm(__nv_bfloat16 *__restrict__ act,
                             const __nv_bfloat16 *__restrict__ x_swizzled,
                             const __nv_bfloat16 *__restrict__ w_gate_up,
                             int rows,
                             cudaStream_t stream) {
  using Element = cutlass::bfloat16_t;
  using ThreadblockShape =
      cutlass::gemm::GemmShape<ThreadblockM, ThreadblockN, ThreadblockK>;
  using WarpShape = cutlass::gemm::GemmShape<WarpM, WarpN, ThreadblockK>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
  using OutputOp0 = cutlass::epilogue::thread::LinearCombination<
      Element, 8, float, float,
      cutlass::epilogue::thread::ScaleType::Nothing>;
  using OutputOp1 = OutputOp0;
  using OutputOp2 = GateGeluTanhUpMul<Element, 8, Element, float>;
  using DualGemm = cutlass::gemm::device::DualGemm<
      Element,
      cutlass::layout::RowMajor,
      Element,
      cutlass::layout::ColumnMajor,
      cutlass::layout::ColumnMajor,
      Element,
      cutlass::layout::RowMajor,
      float,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80,
      ThreadblockShape,
      WarpShape,
      InstructionShape,
      OutputOp0,
      OutputOp1,
      OutputOp2,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
      Stages,
      false,
      false,
      false>;

  const Element *w_gate = reinterpret_cast<const Element *>(w_gate_up);
  const Element *w_up = w_gate + GEMMA4_HIDDEN_SIZE;
  const Element *act_const = reinterpret_cast<const Element *>(act);
  typename DualGemm::TensorRefD null_ref{};

  typename DualGemm::Arguments args(
      cutlass::gemm::DualGemmMode::kGemm,
      {rows, GEMMA4_INTERMEDIATE_SIZE, GEMMA4_HIDDEN_SIZE},
      {reinterpret_cast<const Element *>(x_swizzled), GEMMA4_HIDDEN_SIZE},
      {w_gate, 2 * GEMMA4_HIDDEN_SIZE},
      {act_const, GEMMA4_INTERMEDIATE_SIZE},
      null_ref,
      {w_up, 2 * GEMMA4_HIDDEN_SIZE},
      {act_const, GEMMA4_INTERMEDIATE_SIZE},
      null_ref,
      {reinterpret_cast<Element *>(act), GEMMA4_INTERMEDIATE_SIZE},
      typename OutputOp0::Params(),
      typename OutputOp1::Params(),
      typename OutputOp2::Params());

  DualGemm gemm;
  cutlass::Status status = gemm.can_implement(args);
  if (status != cutlass::Status::kSuccess) {
    return cudaErrorInvalidValue;
  }
  status = gemm(args, nullptr, stream);
  if (status != cutlass::Status::kSuccess) {
    return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
}

// Dispatches one named DualGemm candidate.
cudaError_t launch_config(const DualGemmConfig &config,
                          __nv_bfloat16 *__restrict__ act,
                          const __nv_bfloat16 *__restrict__ x_swizzled,
                          const __nv_bfloat16 *__restrict__ w_gate_up,
                          int rows,
                          cudaStream_t stream) {
  switch (config.id) {
  case kCurrent128x64x32S3:
    return launch_dual_gemm<128, 64, 32, 64, 32, 3>(
        act, x_swizzled, w_gate_up, rows, stream);
  case kDual128x64S4:
    return launch_dual_gemm<128, 64, 32, 64, 32, 4>(
        act, x_swizzled, w_gate_up, rows, stream);
  case kDual128x64S5:
    return launch_dual_gemm<128, 64, 32, 64, 32, 5>(
        act, x_swizzled, w_gate_up, rows, stream);
  case kDual128x64S6:
    return launch_dual_gemm<128, 64, 32, 64, 32, 6>(
        act, x_swizzled, w_gate_up, rows, stream);
  case kDual128x64S10:
    return launch_dual_gemm<128, 64, 32, 64, 32, 10>(
        act, x_swizzled, w_gate_up, rows, stream);
  case kDual128x64x64S3:
    return launch_dual_gemm<128, 64, 64, 64, 32, 3>(
        act, x_swizzled, w_gate_up, rows, stream);
  case kDual64x64W64x32S3:
    return launch_dual_gemm<64, 64, 32, 64, 32, 3>(
        act, x_swizzled, w_gate_up, rows, stream);
  case kDual128x32S3:
    return launch_dual_gemm<128, 32, 32, 64, 32, 3>(
        act, x_swizzled, w_gate_up, rows, stream);
  case kDual256x64S3:
    return launch_dual_gemm<256, 64, 32, 64, 32, 3>(
        act, x_swizzled, w_gate_up, rows, stream);
  case kDual64x64S10:
    return launch_dual_gemm<64, 64, 32, 32, 32, 10>(
        act, x_swizzled, w_gate_up, rows, stream);
  case kDual64x128S6:
    return launch_dual_gemm<64, 128, 32, 32, 64, 6>(
        act, x_swizzled, w_gate_up, rows, stream);
  case kDual128x128S5:
    return launch_dual_gemm<128, 128, 32, 64, 64, 5>(
        act, x_swizzled, w_gate_up, rows, stream);
  case kDual128x256S3:
    return launch_dual_gemm<128, 256, 32, 64, 64, 3>(
        act, x_swizzled, w_gate_up, rows, stream);
  case kDual256x128S3:
    return launch_dual_gemm<256, 128, 32, 64, 64, 3>(
        act, x_swizzled, w_gate_up, rows, stream);
  case kDual128x128x64S3:
    return launch_dual_gemm<128, 128, 64, 64, 64, 3>(
        act, x_swizzled, w_gate_up, rows, stream);
  }
  return cudaErrorInvalidValue;
}

}  // namespace

int main(int argc, char **argv) {
  const int iters = argc > 1 ? std::atoi(argv[1]) : 20;
  const int warmup = argc > 2 ? std::atoi(argv[2]) : 10;
  const int trials = argc > 3 ? std::atoi(argv[3]) : 3;
  const char *row_text = argc > 4 ? argv[4] : "16,64,96,128,256,512,1024";
  const char *selected_name = argc > 5 ? argv[5] : "all";
  if (iters <= 0 || warmup < 0 || trials <= 0) {
    std::fprintf(stderr,
                 "usage: %s [iters=20] [warmup=10] [trials=3] "
                 "[rows=16,64,96,128,256,512,1024] [config=all]\n",
                 argv[0]);
    return 1;
  }

  const std::vector<int> rows = parse_rows(row_text);
  const int max_rows = *std::max_element(rows.begin(), rows.end());
  const DualGemmConfig *selected_config =
      std::strcmp(selected_name, "all") == 0 ? nullptr : find_config(selected_name);
  if (std::strcmp(selected_name, "all") != 0 && selected_config == nullptr) {
    std::fprintf(stderr, "unknown config: %s\n", selected_name);
    return 1;
  }

  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  Gemma4BenchmarkContract contract;
  contract.benchmark = "ffn_dual_gemm_bench";
  contract.measurement = "ffn_gate_up_dual_gemm_only";
  contract.timing = "cuda_events_same_stream_repeated_launches";
  contract.cache = "warm_repeated_buffers";
  contract.launch_overhead = "excluded_from_gpu_elapsed_time";
  contract.correctness = "candidate_vs_current_dualgemm";
  contract.warmup = warmup;
  contract.iters = iters;
  contract.samples = trials;
  contract.graph_inner_iters = 0;
  contract.notes = "12B FFN gate/up DualGemm tile sweep";
  gemma4_bench_print_environment(contract.benchmark);
  gemma4_bench_print_contract(contract);

  const size_t x_elems = size_t(max_rows) * GEMMA4_HIDDEN_SIZE;
  const size_t w_elems =
      size_t(GEMMA4_PACKED_FFN_SIZE) * GEMMA4_HIDDEN_SIZE;
  const size_t out_elems = size_t(max_rows) * GEMMA4_INTERMEDIATE_SIZE;
  DeviceBuffer<__nv_bfloat16> d_x(x_elems);
  DeviceBuffer<__nv_bfloat16> d_w(w_elems);
  DeviceBuffer<__nv_bfloat16> d_ref(out_elems);
  DeviceBuffer<__nv_bfloat16> d_out(out_elems);

  fill_random_bf16(d_x.get(), x_elems, 0x20260623u, 1.0f, stream);
  fill_random_bf16(d_w.get(), w_elems, 0x20260624u, 0.25f, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  std::printf("config,rows,best_ms,avg_ms,max_abs,mean_abs,max_rel,status\n");
  for (const DualGemmConfig &config : kDualConfigs) {
    if (selected_config != nullptr && config.id != selected_config->id) {
      continue;
    }
    for (int row_count : rows) {
      CUDA_CHECK(cudaMemsetAsync(
          d_ref.get(), 0, size_t(row_count) * GEMMA4_INTERMEDIATE_SIZE *
                              sizeof(__nv_bfloat16),
          stream));
      CUDA_CHECK(launch_config(kDualConfigs[0], d_ref.get(), d_x.get(),
                               d_w.get(), row_count, stream));
      CUDA_CHECK(cudaMemsetAsync(
          d_out.get(), 0, size_t(row_count) * GEMMA4_INTERMEDIATE_SIZE *
                              sizeof(__nv_bfloat16),
          stream));
      cudaError_t status =
          launch_config(config, d_out.get(), d_x.get(), d_w.get(), row_count,
                        stream);
      if (status != cudaSuccess) {
        std::printf("%s,%d,0,0,0,0,0,invalid\n", config.name, row_count);
        continue;
      }
      CUDA_CHECK(cudaStreamSynchronize(stream));

      const int count = row_count * GEMMA4_INTERMEDIATE_SIZE;
      const DiffStats diff = diff_stats_bf16(d_out.get(), d_ref.get(), count);
      if (diff.max_abs > kCorrectnessTolerance) {
        std::printf("%s,%d,0,0,%.6g,%.6g,%.6g,wrong\n", config.name,
                    row_count, diff.max_abs, diff.mean_abs, diff.max_rel);
        continue;
      }

      auto run_candidate = [&]() {
        CUDA_CHECK(launch_config(config, d_out.get(), d_x.get(), d_w.get(),
                                 row_count, stream));
      };
      const TimingStats stats =
          time_ms(run_candidate, stream, warmup, iters, trials);
      gemma4_bench_warn_if_tiny(config.name, stats.best_ms);
      std::printf("%s,%d,%.6f,%.6f,%.6g,%.6g,%.6g,ok\n",
                  config.name, row_count, stats.best_ms, stats.avg_ms,
                  diff.max_abs, diff.mean_abs, diff.max_rel);
    }
  }

  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}
