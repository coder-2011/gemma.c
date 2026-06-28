#include "gemma4_checkpoint.cuh"

#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cute/layout.hpp>
#include <cute/swizzle.hpp>

#include <string>
#include <vector>

namespace {

constexpr int kBf16PackWidth = 8;

struct MappedFile {
  void *mapping = nullptr;
  size_t bytes = 0;

  // Releases a temporary mapping unless ownership moved to the checkpoint.
  ~MappedFile() {
    if (mapping != nullptr) munmap(mapping, bytes);
  }
};

// Transposes checkpoint down_proj into the decode FFN row-major stream layout.
void pack_down_decode(std::vector<__nv_bfloat16> &dst, const __nv_bfloat16 *down) {
  dst.resize(static_cast<size_t>(GEMMA4_INTERMEDIATE_SIZE) * GEMMA4_HIDDEN_SIZE);

  const auto src_layout = cute::make_layout(
      cute::make_shape(GEMMA4_HIDDEN_SIZE, GEMMA4_INTERMEDIATE_SIZE),
      cute::make_stride(GEMMA4_INTERMEDIATE_SIZE, 1));
  const auto dst_layout = cute::make_layout(
      cute::make_shape(GEMMA4_INTERMEDIATE_SIZE, GEMMA4_HIDDEN_SIZE),
      cute::make_stride(GEMMA4_HIDDEN_SIZE, 1));
  const auto hidden_layout = cute::make_layout(
      cute::make_shape(GEMMA4_HIDDEN_SIZE / kBf16PackWidth, kBf16PackWidth),
      cute::make_stride(kBf16PackWidth, 1));
  const cute::Swizzle<3, 0, 3> pack_swizzle;

  for (int hidden = 0; hidden < GEMMA4_HIDDEN_SIZE; ++hidden) {
    const int pack = hidden / kBf16PackWidth;
    const int lane = hidden % kBf16PackWidth;
    const int swizzled_pack = pack_swizzle(pack);
    const int dst_col = hidden_layout(swizzled_pack, lane);
    for (int row = 0; row < GEMMA4_INTERMEDIATE_SIZE; ++row) {
      dst[dst_layout(row, dst_col)] = down[src_layout(hidden, row)];
    }
  }
}

// Allocates and copies the native FFN decode buffers for one layer.
cudaError_t load_layer_ffn_decode(
    Gemma4TextLayerWeightsDevice &dst, const Gemma4CheckpointLayerHost &src,
    std::vector<__nv_bfloat16> &gate_up_stage,
    std::vector<__nv_bfloat16> &down_stage) {
  constexpr int hidden_packs = GEMMA4_HIDDEN_SIZE / kBf16PackWidth;
  gate_up_stage.resize(static_cast<size_t>(GEMMA4_PACKED_FFN_SIZE) * GEMMA4_HIDDEN_SIZE);
  const auto ffn_layout = cute::make_layout(
      cute::make_shape(GEMMA4_INTERMEDIATE_SIZE, GEMMA4_HIDDEN_SIZE),
      cute::make_stride(GEMMA4_HIDDEN_SIZE, 1));
  const auto stage_layout = cute::make_layout(
      cute::make_shape(GEMMA4_PACKED_FFN_SIZE, GEMMA4_HIDDEN_SIZE),
      cute::make_stride(GEMMA4_HIDDEN_SIZE, 1));
  const auto hidden_layout = cute::make_layout(
      cute::make_shape(hidden_packs, kBf16PackWidth),
      cute::make_stride(kBf16PackWidth, 1));
  const cute::Swizzle<3, 0, 3> pack_swizzle;

  for (int row = 0; row < GEMMA4_INTERMEDIATE_SIZE; ++row) {
    __nv_bfloat16 *gate_dst = gate_up_stage.data() + stage_layout(2 * row, 0);
    __nv_bfloat16 *up_dst = gate_up_stage.data() + stage_layout(2 * row + 1, 0);
    const __nv_bfloat16 *gate_src = src.gate_proj_col_major + ffn_layout(row, 0);
    const __nv_bfloat16 *up_src = src.up_proj_col_major + ffn_layout(row, 0);

    for (int pack = 0; pack < hidden_packs; ++pack) {
      const int swizzled_pack = pack_swizzle(pack);
      const int src_col = hidden_layout(pack, 0);
      const int dst_col = hidden_layout(swizzled_pack, 0);
      memcpy(gate_dst + dst_col, gate_src + src_col,
             kBf16PackWidth * sizeof(__nv_bfloat16));
      memcpy(up_dst + dst_col, up_src + src_col,
             kBf16PackWidth * sizeof(__nv_bfloat16));
    }
  }

  size_t bytes = gate_up_stage.size() * sizeof(__nv_bfloat16);
  cudaError_t status = cudaMalloc(&dst.ffn_gate_up_decode, bytes);
  if (status != cudaSuccess) return status;
  status = cudaMemcpy(dst.ffn_gate_up_decode, gate_up_stage.data(), bytes, cudaMemcpyHostToDevice);
  if (status != cudaSuccess) {
    cudaFree(dst.ffn_gate_up_decode);
    dst.ffn_gate_up_decode = nullptr;
    return status;
  }

  pack_down_decode(down_stage, src.down_proj_checkpoint);
  bytes = down_stage.size() * sizeof(__nv_bfloat16);
  status = cudaMalloc(&dst.ffn_down_decode, bytes);
  if (status == cudaSuccess) {
    status = cudaMemcpy(dst.ffn_down_decode, down_stage.data(), bytes, cudaMemcpyHostToDevice);
  }
  if (status != cudaSuccess) {
    cudaFree(dst.ffn_down_decode);
    dst.ffn_down_decode = nullptr;
    cudaFree(dst.ffn_gate_up_decode);
    dst.ffn_gate_up_decode = nullptr;
  }
  return status;
}

// Packs Q/K/(V) projection matrices into the contiguous decode layout.
cudaError_t load_layer_qkv_decode(
    Gemma4TextLayerWeightsDevice &dst,
    const Gemma4CheckpointLayerHost &src,
    const Gemma4AttentionSpec &spec,
    std::vector<__nv_bfloat16> &qkv_stage) {
  const int q_width = spec.q_heads * spec.head_dim;
  const int kv_width = spec.kv_heads * spec.head_dim;
  const int v_width = spec.global ? 0 : kv_width;
  const auto q_layout = cute::make_layout(
      cute::make_shape(q_width, GEMMA4_HIDDEN_SIZE),
      cute::make_stride(GEMMA4_HIDDEN_SIZE, 1));
  const auto kv_layout = cute::make_layout(
      cute::make_shape(kv_width, GEMMA4_HIDDEN_SIZE),
      cute::make_stride(GEMMA4_HIDDEN_SIZE, 1));
  const auto v_layout = cute::make_layout(
      cute::make_shape(v_width, GEMMA4_HIDDEN_SIZE),
      cute::make_stride(GEMMA4_HIDDEN_SIZE, 1));
  const auto qkv_layout = cute::make_layout(
      cute::make_shape(q_width + kv_width + v_width, GEMMA4_HIDDEN_SIZE),
      cute::make_stride(GEMMA4_HIDDEN_SIZE, 1));
  const size_t q_elements = cute::size(q_layout);
  const size_t kv_elements = cute::size(kv_layout);
  const size_t v_elements = cute::size(v_layout);
  qkv_stage.resize(q_elements + kv_elements + v_elements);

  __nv_bfloat16 *q_dst = qkv_stage.data() + qkv_layout(0, 0);
  __nv_bfloat16 *k_dst = qkv_stage.data() + qkv_layout(q_width, 0);
  __nv_bfloat16 *v_dst = qkv_stage.data() + qkv_layout(q_width + kv_width, 0);
  memcpy(q_dst, src.q_proj_col_major, q_elements * sizeof(__nv_bfloat16));
  memcpy(k_dst, src.k_proj_col_major, kv_elements * sizeof(__nv_bfloat16));

  // Global layers have K=V semantics, so only sliding layers carry a V projection.
  if (v_elements != 0) {
    memcpy(v_dst, src.v_proj_col_major, v_elements * sizeof(__nv_bfloat16));
  }

  const size_t bytes = qkv_stage.size() * sizeof(__nv_bfloat16);
  cudaError_t status = cudaMalloc(&dst.qkv_proj_col_major, bytes);
  if (status == cudaSuccess) {
    status = cudaMemcpy(dst.qkv_proj_col_major, qkv_stage.data(), bytes, cudaMemcpyHostToDevice);
  }
  if (status != cudaSuccess) {
    cudaFree(dst.qkv_proj_col_major);
    dst.qkv_proj_col_major = nullptr;
    return status;
  }

  dst.q_proj_col_major = dst.qkv_proj_col_major;
  dst.k_proj_col_major = dst.qkv_proj_col_major + qkv_layout(q_width, 0);
  dst.v_proj_col_major =
      spec.global ? nullptr : dst.qkv_proj_col_major + qkv_layout(q_width + kv_width, 0);
  return cudaSuccess;
}

}  // namespace

bool gemma4_checkpoint_open_text_bf16(
    Gemma4CheckpointHost *checkpoint,
    const char *path) {
  if (checkpoint == nullptr || path == nullptr) {
    return false;
  }
  *checkpoint = Gemma4CheckpointHost();

  MappedFile file;
  const int fd = open(path, O_RDONLY);
  if (fd < 0) {
    return false;
  }

  struct stat st = {};
  if (fstat(fd, &st) != 0 || st.st_size < 8) {
    close(fd);
    return false;
  }
  file.bytes = static_cast<size_t>(st.st_size);
  void *mapping = mmap(nullptr, file.bytes, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd);
  if (mapping == MAP_FAILED) {
    return false;
  }
  file.mapping = mapping;

  const char *bytes = static_cast<const char *>(file.mapping);
  uint64_t header_bytes = 0;
  memcpy(&header_bytes, bytes, sizeof(header_bytes));
  if (header_bytes > static_cast<uint64_t>(file.bytes - 8)) {
    return false;
  }
  const size_t data_start = 8 + static_cast<size_t>(header_bytes);

  const std::string header(bytes + 8, bytes + 8 + header_bytes);
  const char *data_base = bytes + data_start;
  const size_t data_bytes = file.bytes - data_start;

  struct TensorRequest {
    std::string name;
    size_t elements = 0;
    const __nv_bfloat16 **out = nullptr;
  };
  std::vector<TensorRequest> requests;
  requests.reserve(2 + GEMMA4_NUM_LAYERS * 14);
  requests.push_back({"model.language_model.embed_tokens.weight", static_cast<size_t>(GEMMA4_VOCAB_SIZE) * GEMMA4_HIDDEN_SIZE, &checkpoint->token_embedding});
  requests.push_back({"model.language_model.norm.weight", GEMMA4_HIDDEN_SIZE, &checkpoint->final_norm_weight});

  for (int layer = 0; layer < GEMMA4_NUM_LAYERS; ++layer) {
    Gemma4CheckpointLayerHost &dst = checkpoint->layers[layer];
    const Gemma4AttentionSpec spec = gemma4_attention_spec(layer);
    const size_t q_size = static_cast<size_t>(spec.q_heads) * spec.head_dim;
    const size_t kv_size = static_cast<size_t>(spec.kv_heads) * spec.head_dim;
    const std::string prefix = "model.language_model.layers." + std::to_string(layer) + ".";
    requests.insert(requests.end(), {
        {prefix + "input_layernorm.weight", GEMMA4_HIDDEN_SIZE, &dst.input_norm_weight},
        {prefix + "post_attention_layernorm.weight", GEMMA4_HIDDEN_SIZE, &dst.post_attention_norm_weight},
        {prefix + "pre_feedforward_layernorm.weight", GEMMA4_HIDDEN_SIZE, &dst.pre_feedforward_norm_weight},
        {prefix + "post_feedforward_layernorm.weight", GEMMA4_HIDDEN_SIZE, &dst.post_feedforward_norm_weight},
        {prefix + "layer_scalar", 1, &dst.layer_scalar},
        {prefix + "self_attn.q_norm.weight", static_cast<size_t>(spec.head_dim), &dst.q_norm_weight},
        {prefix + "self_attn.k_norm.weight", static_cast<size_t>(spec.head_dim), &dst.k_norm_weight},
        {prefix + "self_attn.q_proj.weight", q_size * GEMMA4_HIDDEN_SIZE, &dst.q_proj_col_major},
        {prefix + "self_attn.k_proj.weight", kv_size * GEMMA4_HIDDEN_SIZE, &dst.k_proj_col_major},
        {prefix + "self_attn.o_proj.weight", GEMMA4_HIDDEN_SIZE * static_cast<size_t>(q_size), &dst.o_proj_col_major},
        {prefix + "mlp.gate_proj.weight", static_cast<size_t>(GEMMA4_INTERMEDIATE_SIZE) * GEMMA4_HIDDEN_SIZE, &dst.gate_proj_col_major},
        {prefix + "mlp.up_proj.weight", static_cast<size_t>(GEMMA4_INTERMEDIATE_SIZE) * GEMMA4_HIDDEN_SIZE, &dst.up_proj_col_major},
        {prefix + "mlp.down_proj.weight", static_cast<size_t>(GEMMA4_HIDDEN_SIZE) * GEMMA4_INTERMEDIATE_SIZE, &dst.down_proj_checkpoint},
    });
    if (!spec.global) {
      requests.push_back({
          prefix + "self_attn.v_proj.weight",
          kv_size * GEMMA4_HIDDEN_SIZE,
          &dst.v_proj_col_major,
      });
    }
  }

  constexpr char kOffsets[] = "\"data_offsets\":[";

  for (const TensorRequest &request : requests) {
    const std::string key = "\"" + request.name + "\":";
    const size_t object = header.find(key);
    if (object == std::string::npos) {
      return false;
    }

    const size_t object_end = header.find('}', object + key.size());
    const size_t offsets = header.find(kOffsets, object + key.size());
    if (object_end == std::string::npos ||
        offsets == std::string::npos ||
        offsets > object_end) {
      return false;
    }

    unsigned long long first = 0;
    unsigned long long second = 0;
    const char *offset_text = header.c_str() + offsets + sizeof(kOffsets) - 1;
    if (sscanf(offset_text, "%llu , %llu", &first, &second) != 2 ||
        second < first) {
      return false;
    }

    const size_t expected_bytes = request.elements * sizeof(__nv_bfloat16);
    if (second - first != expected_bytes ||
        second > static_cast<unsigned long long>(data_bytes)) {
      return false;
    }
    *request.out = reinterpret_cast<const __nv_bfloat16 *>(data_base + first);
  }

  checkpoint->mapping = file.mapping;
  checkpoint->mapping_bytes = file.bytes;
  file.mapping = nullptr;
  return true;
}

cudaError_t gemma4_load_text_weights_device_bf16(
    Gemma4TextWeightsDevice *weights, const char *path, std::string *error) {
  if (weights == nullptr || path == nullptr) {
    if (error != nullptr) *error = "weights and path are required";
    return cudaErrorInvalidValue;
  }
  *weights = Gemma4TextWeightsDevice();

  Gemma4CheckpointHost checkpoint;
  if (!gemma4_checkpoint_open_text_bf16(&checkpoint, path)) {
    if (error != nullptr) *error = "failed to load checkpoint";
    return cudaErrorInvalidValue;
  }

  size_t bytes = static_cast<size_t>(GEMMA4_VOCAB_SIZE) * GEMMA4_HIDDEN_SIZE *
                 sizeof(__nv_bfloat16);
  cudaError_t status = cudaMalloc(&weights->token_embedding, bytes);
  if (status == cudaSuccess) {
    status = cudaMemcpy(weights->token_embedding, checkpoint.token_embedding, bytes,
                        cudaMemcpyHostToDevice);
  }
  if (status == cudaSuccess) {
    bytes = GEMMA4_HIDDEN_SIZE * sizeof(__nv_bfloat16);
    status = cudaMalloc(&weights->final_norm_weight, bytes);
  }
  if (status == cudaSuccess) {
    status = cudaMemcpy(weights->final_norm_weight, checkpoint.final_norm_weight, bytes,
                        cudaMemcpyHostToDevice);
  }

  std::vector<__nv_bfloat16> gate_up_stage;
  std::vector<__nv_bfloat16> down_stage;
  std::vector<__nv_bfloat16> qkv_stage;
  for (int layer = 0; status == cudaSuccess && layer < GEMMA4_NUM_LAYERS; ++layer) {
    Gemma4TextLayerWeightsDevice &dst = weights->layers[layer];
    const Gemma4CheckpointLayerHost &src = checkpoint.layers[layer];
    const Gemma4AttentionSpec spec = gemma4_attention_spec(layer);
    const size_t q_size = static_cast<size_t>(spec.q_heads) * spec.head_dim;

    struct DeviceCopy {
      __nv_bfloat16 **dst;
      const __nv_bfloat16 *src;
      size_t elements;
    };
    const DeviceCopy layer_copies[] = {
        {&dst.input_norm_weight, src.input_norm_weight, GEMMA4_HIDDEN_SIZE},
        {&dst.post_attention_norm_weight, src.post_attention_norm_weight, GEMMA4_HIDDEN_SIZE},
        {&dst.pre_feedforward_norm_weight, src.pre_feedforward_norm_weight, GEMMA4_HIDDEN_SIZE},
        {&dst.post_feedforward_norm_weight, src.post_feedforward_norm_weight, GEMMA4_HIDDEN_SIZE},
        {&dst.layer_scalar, src.layer_scalar, 1},
        {&dst.q_norm_weight, src.q_norm_weight, static_cast<size_t>(spec.head_dim)},
        {&dst.k_norm_weight, src.k_norm_weight, static_cast<size_t>(spec.head_dim)},
        {&dst.o_proj_col_major, src.o_proj_col_major, GEMMA4_HIDDEN_SIZE * q_size},
    };
    for (const DeviceCopy &copy : layer_copies) {
      bytes = copy.elements * sizeof(__nv_bfloat16);
      status = cudaMalloc(copy.dst, bytes);
      if (status != cudaSuccess) break;
      status = cudaMemcpy(*copy.dst, copy.src, bytes, cudaMemcpyHostToDevice);
      if (status != cudaSuccess) {
        cudaFree(*copy.dst);
        *copy.dst = nullptr;
        break;
      }
    }
    if (status == cudaSuccess) {
      status = load_layer_qkv_decode(dst, src, spec, qkv_stage);
    }
    if (status == cudaSuccess) {
      status = load_layer_ffn_decode(dst, src, gate_up_stage, down_stage);
    }
  }

  munmap(checkpoint.mapping, checkpoint.mapping_bytes);
  checkpoint = Gemma4CheckpointHost();
  if (status != cudaSuccess) {
    gemma4_text_weights_device_free(weights);
  }
  return status;
}
