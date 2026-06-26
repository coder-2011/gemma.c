#include "gemma4_prefill_megakernel.cuh"
#include "gemma4.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

// Aborts on the first CUDA error so the failing call site stays visible.
void check_cuda(cudaError_t status, const char *expr, const char *file, int line) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "%s:%d: CUDA error for %s: %s\n", file, line, expr,
                 cudaGetErrorString(status));
    std::exit(1);
  }
}

#define CHECK_CUDA(expr) check_cuda((expr), #expr, __FILE__, __LINE__)

// Owns one device allocation used by this integrated CUDA test.
template <typename T>
class DeviceBuffer {
 public:
  explicit DeviceBuffer(size_t count) : count_(count) {
    if (count_ > 0) {
      CHECK_CUDA(cudaMalloc(&ptr_, count_ * sizeof(T)));
    }
  }

  ~DeviceBuffer() {
    if (ptr_ != nullptr) {
      cudaFree(ptr_);
    }
  }

  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;

  T *get() const { return ptr_; }

  // Copies host data into this device allocation.
  void copy_from(const std::vector<T> &src) const {
    if (!src.empty()) {
      CHECK_CUDA(cudaMemcpy(ptr_, src.data(), src.size() * sizeof(T),
                            cudaMemcpyHostToDevice));
    }
  }

  // Copies this full device allocation back to host memory.
  std::vector<T> copy_to_host() const {
    std::vector<T> dst(count_);
    if (!dst.empty()) {
      CHECK_CUDA(cudaMemcpy(dst.data(), ptr_, dst.size() * sizeof(T),
                            cudaMemcpyDeviceToHost));
    }
    return dst;
  }

  // Clears this allocation without staging a host vector.
  void zero() const {
    if (count_ > 0) {
      CHECK_CUDA(cudaMemset(ptr_, 0, count_ * sizeof(T)));
    }
  }

 private:
  T *ptr_ = nullptr;
  size_t count_ = 0;
};

// Converts BF16 values to FP32 for tolerance checks.
float bf16_to_float(__nv_bfloat16 value) {
  return __bfloat162float(value);
}

// Writes one BF16 scalar into a large device tensor.
void write_device_bf16(
    const DeviceBuffer<__nv_bfloat16> &dst,
    int64_t index,
    float value) {
  __nv_bfloat16 bf16 = __float2bfloat16_rn(value);
  CHECK_CUDA(cudaMemcpy(dst.get() + index, &bf16, sizeof(bf16),
                        cudaMemcpyHostToDevice));
}

// Compares BF16 tensors after converting to FP32.
void compare_bf16(
    const std::vector<__nv_bfloat16> &actual,
    const std::vector<__nv_bfloat16> &expected,
    float tolerance,
    const char *label) {
  float max_abs = 0.0f;
  int max_index = 0;
  for (int i = 0; i < static_cast<int>(actual.size()); ++i) {
    const float diff =
        std::fabs(bf16_to_float(actual[i]) - bf16_to_float(expected[i]));
    if (diff > max_abs) {
      max_abs = diff;
      max_index = i;
    }
  }
  if (max_abs > tolerance) {
    std::fprintf(stderr,
                 "%s max_abs=%g index=%d actual=%g expected=%g\n",
                 label, max_abs, max_index, bf16_to_float(actual[max_index]),
                 bf16_to_float(expected[max_index]));
    std::exit(1);
  }
}

// Owns the real Gemma layer-weight struct plus the device buffers it points to.
struct LayerBuffers {
  explicit LayerBuffers(bool global_)
      : global(global_),
        q_width(global ? GEMMA4_GLOBAL_Q_PROJ_SIZE
                       : GEMMA4_SLIDING_Q_PROJ_SIZE),
        kv_width(global ? GEMMA4_GLOBAL_K_PROJ_SIZE
                        : GEMMA4_SLIDING_KV_PROJ_SIZE),
        head_dim(global ? GEMMA4_GLOBAL_HEAD_DIM
                        : GEMMA4_SLIDING_HEAD_DIM),
        input_norm(GEMMA4_HIDDEN_SIZE),
        post_attention_norm(GEMMA4_HIDDEN_SIZE),
        pre_feedforward_norm(GEMMA4_HIDDEN_SIZE),
        post_feedforward_norm(GEMMA4_HIDDEN_SIZE),
        layer_scalar(1),
        q_norm(head_dim),
        k_norm(head_dim),
        q_proj(static_cast<size_t>(q_width) * GEMMA4_HIDDEN_SIZE),
        k_proj(static_cast<size_t>(kv_width) * GEMMA4_HIDDEN_SIZE),
        v_proj(global ? 0 : static_cast<size_t>(kv_width) * GEMMA4_HIDDEN_SIZE),
        o_proj(static_cast<size_t>(GEMMA4_HIDDEN_SIZE) * q_width),
        ffn_gate_up(static_cast<size_t>(GEMMA4_PACKED_FFN_SIZE) *
                    GEMMA4_HIDDEN_SIZE),
        ffn_down(static_cast<size_t>(GEMMA4_INTERMEDIATE_SIZE) *
                 GEMMA4_HIDDEN_SIZE) {}

  // Initializes weights to make the runner output a scaled hidden tensor.
  void initialize() const {
    std::vector<__nv_bfloat16> hidden_ones(
        GEMMA4_HIDDEN_SIZE, __float2bfloat16_rn(1.0f));
    std::vector<__nv_bfloat16> head_ones(
        head_dim, __float2bfloat16_rn(1.0f));
    std::vector<__nv_bfloat16> scalar(1, __float2bfloat16_rn(0.5f));

    input_norm.copy_from(hidden_ones);
    post_attention_norm.copy_from(hidden_ones);
    pre_feedforward_norm.copy_from(hidden_ones);
    post_feedforward_norm.copy_from(hidden_ones);
    layer_scalar.copy_from(scalar);
    q_norm.copy_from(head_ones);
    k_norm.copy_from(head_ones);

    q_proj.zero();
    k_proj.zero();
    v_proj.zero();
    o_proj.zero();
    ffn_gate_up.zero();
    ffn_down.zero();

    write_device_bf16(k_proj, 0, 1.0f);
    if (!global) {
      write_device_bf16(v_proj, 0, -0.75f);
    }
  }

  // Returns the production layer-weight view consumed by the runner.
  Gemma4TextLayerWeightsDevice view() const {
    Gemma4TextLayerWeightsDevice weights = {};
    weights.input_norm_weight = input_norm.get();
    weights.post_attention_norm_weight = post_attention_norm.get();
    weights.pre_feedforward_norm_weight = pre_feedforward_norm.get();
    weights.post_feedforward_norm_weight = post_feedforward_norm.get();
    weights.layer_scalar = layer_scalar.get();
    weights.q_norm_weight = q_norm.get();
    weights.k_norm_weight = k_norm.get();
    weights.q_proj_col_major = q_proj.get();
    weights.k_proj_col_major = k_proj.get();
    weights.v_proj_col_major = global ? nullptr : v_proj.get();
    weights.o_proj_col_major = o_proj.get();
    weights.ffn_gate_up_decode = ffn_gate_up.get();
    weights.ffn_down_decode = ffn_down.get();
    return weights;
  }

  bool global;
  int32_t q_width;
  int32_t kv_width;
  int32_t head_dim;
  DeviceBuffer<__nv_bfloat16> input_norm;
  DeviceBuffer<__nv_bfloat16> post_attention_norm;
  DeviceBuffer<__nv_bfloat16> pre_feedforward_norm;
  DeviceBuffer<__nv_bfloat16> post_feedforward_norm;
  DeviceBuffer<__nv_bfloat16> layer_scalar;
  DeviceBuffer<__nv_bfloat16> q_norm;
  DeviceBuffer<__nv_bfloat16> k_norm;
  DeviceBuffer<__nv_bfloat16> q_proj;
  DeviceBuffer<__nv_bfloat16> k_proj;
  DeviceBuffer<__nv_bfloat16> v_proj;
  DeviceBuffer<__nv_bfloat16> o_proj;
  DeviceBuffer<__nv_bfloat16> ffn_gate_up;
  DeviceBuffer<__nv_bfloat16> ffn_down;
};

// Runs one sliding or global prefill layer and checks scalar/output/cache wiring.
void run_prefill_layer_case(bool global) {
  constexpr int batch_size = 1;
  constexpr int seq_len = 2;
  constexpr int rows = batch_size * seq_len;
  const int32_t layer_index = global ? 5 : 0;
  const int rotary_half =
      global ? GEMMA4_GLOBAL_HEAD_DIM / 8 : GEMMA4_SLIDING_HEAD_DIM / 2;

  std::vector<__nv_bfloat16> hidden(
      static_cast<size_t>(rows) * GEMMA4_HIDDEN_SIZE);
  for (int i = 0; i < static_cast<int>(hidden.size()); ++i) {
    const int centered = ((i * 17 + 3) % 251) - 125;
    hidden[i] = __float2bfloat16_rn(static_cast<float>(centered) / 128.0f);
  }
  std::vector<float> cos(seq_len * rotary_half, 1.0f);
  std::vector<float> sin(cos.size(), 0.0f);

  LayerBuffers layer(global);
  layer.initialize();
  Gemma4TextLayerWeightsDevice weights = layer.view();

  DeviceBuffer<__nv_bfloat16> d_hidden(hidden.size());
  DeviceBuffer<__nv_bfloat16> d_out(hidden.size());
  DeviceBuffer<float> d_cos(cos.size());
  DeviceBuffer<float> d_sin(sin.size());
  d_hidden.copy_from(hidden);
  d_cos.copy_from(cos);
  d_sin.copy_from(sin);

  DeviceBuffer<__nv_bfloat16> d_scratch(
      gemma4_prefill_megakernel_layer_scratch_elements(global, rows));
  Gemma4PrefillMegakernelLayerScratch scratch =
      gemma4_prefill_megakernel_layer_scratch_from_buffer(
          d_scratch.get(), global, rows);

  Gemma4KvCacheConfig cache_config =
      gemma4_kv_cache_make_config(global, 1, 4, 1);
  const size_t cache_size =
      static_cast<size_t>(cache_config.num_layers) * cache_config.num_pages *
      cache_config.page_size * cache_config.num_heads * cache_config.head_dim;
  DeviceBuffer<__nv_bfloat16> d_cache_k(cache_size);
  DeviceBuffer<__nv_bfloat16> d_cache_v(cache_size);
  const size_t page_table_size =
      static_cast<size_t>(batch_size) * cache_config.max_pages_per_seq;
  DeviceBuffer<int32_t> d_page_table(page_table_size);
  DeviceBuffer<int32_t> d_token_batch(rows);
  DeviceBuffer<int32_t> d_token_position(rows);
  std::vector<int32_t> page_table(page_table_size, 0);
  std::vector<int32_t> token_batch(rows, 0);
  std::vector<int32_t> token_position = {0, 1};
  d_cache_k.zero();
  d_cache_v.zero();
  d_page_table.copy_from(page_table);
  d_token_batch.copy_from(token_batch);
  d_token_position.copy_from(token_position);

  Gemma4PrefillMegakernelLayerArgs args = {};
  args.out = d_out.get();
  args.hidden = d_hidden.get();
  args.weights = &weights;
  args.layer_index = layer_index;
  args.batch_size = batch_size;
  args.seq_len = seq_len;
  args.cos = d_cos.get();
  args.sin = d_sin.get();
  args.softmax_scale = 1.0f;
  args.cache_k = d_cache_k.get();
  args.cache_v = d_cache_v.get();
  args.cache_config = cache_config;
  args.page_table = d_page_table.get();
  args.token_batch = d_token_batch.get();
  args.token_position = d_token_position.get();

  CHECK_CUDA(gemma4_prefill_megakernel_layer_bf16(args, scratch));
  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<__nv_bfloat16> expected = hidden;
  for (size_t i = 0; i < expected.size(); ++i) {
    expected[i] = __float2bfloat16_rn(bf16_to_float(expected[i]) * 0.5f);
  }

  compare_bf16(d_out.copy_to_host(), expected, 0.0f,
               global ? "global prefill scaled output"
                      : "sliding prefill scaled output");

  const int32_t cache_layer = gemma4_kv_cache_layer_index(layer_index, global);
  const int64_t first_cache_value =
      gemma4_kv_cache_offset(cache_config, cache_layer, 0, 0, 0, 0);
  if (bf16_to_float(d_cache_k.copy_to_host()[first_cache_value]) == 0.0f ||
      bf16_to_float(d_cache_v.copy_to_host()[first_cache_value]) == 0.0f) {
    std::fprintf(stderr, "prefill KV cache write missed page zero offset zero\n");
    std::exit(1);
  }

  Gemma4PrefillMegakernelLayerArgs bad_cache_args = args;
  bad_cache_args.batch_size = 2;
  bad_cache_args.seq_len = 1;
  cudaError_t bad_status =
      gemma4_prefill_megakernel_layer_bf16(bad_cache_args, scratch);
  if (bad_status != cudaErrorInvalidValue) {
    std::fprintf(stderr, "expected invalid prefill cache batch capacity\n");
    std::exit(1);
  }
}

}  // namespace

// Runs integrated prefill layer coverage for local and global layer wiring.
int main() {
  run_prefill_layer_case(false);
  run_prefill_layer_case(true);
  std::printf("prefill megakernel tests passed\n");
  return 0;
}
