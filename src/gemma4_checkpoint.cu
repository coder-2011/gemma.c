#include "gemma4_checkpoint.cuh"

#include <fcntl.h>
#include <stdint.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

struct TensorMeta {
  std::string dtype;
  std::vector<int64_t> shape;
  size_t begin = 0;
  size_t end = 0;
};

using TensorMap = std::unordered_map<std::string, TensorMeta>;

// Writes a diagnostic only when the caller asked for one.
void set_error(std::string *error, const std::string &message) {
  if (error != nullptr) {
    *error = message;
  }
}

// Treats spaces, tabs, and line breaks as JSON whitespace.
bool is_json_space(char c) {
  return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

// Advances past insignificant JSON whitespace.
void skip_ws(const std::string &text, size_t &pos) {
  while (pos < text.size() && is_json_space(text[pos])) {
    ++pos;
  }
}

// Parses a JSON string, including escaped characters used in tensor names.
bool parse_string(const std::string &text, size_t &pos, std::string &out) {
  skip_ws(text, pos);
  if (pos >= text.size() || text[pos] != '"') {
    return false;
  }
  ++pos;
  out.clear();
  while (pos < text.size()) {
    const char c = text[pos++];
    if (c == '"') {
      return true;
    }
    if (c == '\\') {
      if (pos >= text.size()) {
        return false;
      }
      out.push_back(text[pos++]);
    } else {
      out.push_back(c);
    }
  }
  return false;
}

// Parses a non-negative integer from the safetensors metadata.
bool parse_uint(const std::string &text, size_t &pos, size_t &out) {
  skip_ws(text, pos);
  if (pos >= text.size() || text[pos] < '0' || text[pos] > '9') {
    return false;
  }
  size_t value = 0;
  while (pos < text.size() && text[pos] >= '0' && text[pos] <= '9') {
    value = value * 10 + static_cast<size_t>(text[pos] - '0');
    ++pos;
  }
  out = value;
  return true;
}

// Parses an array of non-negative integer dimensions or byte offsets.
bool parse_int_array(
    const std::string &text,
    size_t &pos,
    std::vector<int64_t> &out) {
  skip_ws(text, pos);
  if (pos >= text.size() || text[pos] != '[') {
    return false;
  }
  ++pos;
  out.clear();
  skip_ws(text, pos);
  if (pos < text.size() && text[pos] == ']') {
    ++pos;
    return true;
  }
  while (pos < text.size()) {
    size_t value = 0;
    if (!parse_uint(text, pos, value)) {
      return false;
    }
    out.push_back(static_cast<int64_t>(value));
    skip_ws(text, pos);
    if (pos < text.size() && text[pos] == ']') {
      ++pos;
      return true;
    }
    if (pos >= text.size() || text[pos] != ',') {
      return false;
    }
    ++pos;
  }
  return false;
}

// Skips an arbitrary JSON value that is not needed for tensor loading.
bool skip_value(const std::string &text, size_t &pos) {
  skip_ws(text, pos);
  if (pos >= text.size()) {
    return false;
  }
  if (text[pos] == '"') {
    std::string ignored;
    return parse_string(text, pos, ignored);
  }
  if (text[pos] != '{' && text[pos] != '[') {
    while (pos < text.size() && text[pos] != ',' &&
           text[pos] != '}' && text[pos] != ']') {
      ++pos;
    }
    return true;
  }

  const char open = text[pos++];
  const char close = open == '{' ? '}' : ']';
  int depth = 1;
  while (pos < text.size() && depth > 0) {
    if (text[pos] == '"') {
      std::string ignored;
      if (!parse_string(text, pos, ignored)) {
        return false;
      }
    } else if (text[pos] == open) {
      ++depth;
      ++pos;
    } else if (text[pos] == close) {
      --depth;
      ++pos;
    } else {
      ++pos;
    }
  }
  return depth == 0;
}

// Parses one tensor object from the safetensors JSON header.
bool parse_tensor_object(
    const std::string &text,
    size_t &pos,
    TensorMeta &meta) {
  skip_ws(text, pos);
  if (pos >= text.size() || text[pos] != '{') {
    return false;
  }
  ++pos;
  while (pos < text.size()) {
    skip_ws(text, pos);
    if (pos < text.size() && text[pos] == '}') {
      ++pos;
      return !meta.dtype.empty() && !meta.shape.empty() && meta.end > meta.begin;
    }

    std::string field;
    if (!parse_string(text, pos, field)) {
      return false;
    }
    skip_ws(text, pos);
    if (pos >= text.size() || text[pos] != ':') {
      return false;
    }
    ++pos;

    if (field == "dtype") {
      if (!parse_string(text, pos, meta.dtype)) {
        return false;
      }
    } else if (field == "shape") {
      if (!parse_int_array(text, pos, meta.shape)) {
        return false;
      }
    } else if (field == "data_offsets") {
      std::vector<int64_t> offsets;
      if (!parse_int_array(text, pos, offsets) || offsets.size() != 2) {
        return false;
      }
      meta.begin = static_cast<size_t>(offsets[0]);
      meta.end = static_cast<size_t>(offsets[1]);
    } else if (!skip_value(text, pos)) {
      return false;
    }

    skip_ws(text, pos);
    if (pos < text.size() && text[pos] == ',') {
      ++pos;
    }
  }
  return false;
}

// Parses the top-level safetensors header into a tensor-name map.
bool parse_header(const std::string &header, TensorMap &tensors) {
  size_t pos = 0;
  skip_ws(header, pos);
  if (pos >= header.size() || header[pos] != '{') {
    return false;
  }
  ++pos;
  while (pos < header.size()) {
    skip_ws(header, pos);
    if (pos < header.size() && header[pos] == '}') {
      ++pos;
      return true;
    }

    std::string name;
    if (!parse_string(header, pos, name)) {
      return false;
    }
    skip_ws(header, pos);
    if (pos >= header.size() || header[pos] != ':') {
      return false;
    }
    ++pos;

    if (name == "__metadata__") {
      if (!skip_value(header, pos)) {
        return false;
      }
    } else {
      TensorMeta meta;
      if (!parse_tensor_object(header, pos, meta)) {
        return false;
      }
      tensors.emplace(std::move(name), std::move(meta));
    }

    skip_ws(header, pos);
    if (pos < header.size() && header[pos] == ',') {
      ++pos;
    }
  }
  return false;
}

// Computes the number of elements in a validated tensor shape.
size_t element_count(const std::vector<int64_t> &shape) {
  size_t count = 1;
  for (int64_t dim : shape) {
    count *= static_cast<size_t>(dim);
  }
  return count;
}

// Builds a canonical Hugging Face language-model tensor name.
std::string layer_name(int layer, const char *suffix) {
  return "model.language_model.layers." + std::to_string(layer) + "." + suffix;
}

// Validates a tensor and returns its mmap-backed BF16 data pointer.
bool require_tensor(
    const TensorMap &tensors,
    const char *name,
    const std::vector<int64_t> &shape,
    const char *data_base,
    size_t data_bytes,
    const __nv_bfloat16 **out,
    std::string *error) {
  const auto it = tensors.find(name);
  if (it == tensors.end()) {
    set_error(error, std::string("missing tensor: ") + name);
    return false;
  }

  const TensorMeta &meta = it->second;
  const size_t bytes = element_count(shape) * sizeof(__nv_bfloat16);
  if (meta.dtype != "BF16" || meta.shape != shape ||
      meta.end < meta.begin || meta.end - meta.begin != bytes ||
      meta.end > data_bytes) {
    set_error(error, std::string("bad tensor metadata: ") + name);
    return false;
  }

  *out = reinterpret_cast<const __nv_bfloat16 *>(data_base + meta.begin);
  return true;
}

// Copies one BF16 tensor from the mapped file into a new device allocation.
cudaError_t alloc_copy(
    __nv_bfloat16 **dst,
    const __nv_bfloat16 *src,
    size_t elements) {
  *dst = nullptr;
  cudaError_t status = cudaMalloc(dst, elements * sizeof(__nv_bfloat16));
  if (status != cudaSuccess) {
    return status;
  }
  status = cudaMemcpy(*dst, src, elements * sizeof(__nv_bfloat16),
                      cudaMemcpyHostToDevice);
  if (status != cudaSuccess) {
    cudaFree(*dst);
    *dst = nullptr;
  }
  return status;
}

// Reuses the FFN decode hidden-pack swizzle while packing checkpoint weights.
int hidden_pack_swizzle_index(int chunk) {
  constexpr int kSwizzleChunks = 8;
  const unsigned u = static_cast<unsigned>(chunk);
  const unsigned col = u & (kSwizzleChunks - 1);
  const unsigned row = (u >> 3) & (kSwizzleChunks - 1);
  return static_cast<int>((u & ~(kSwizzleChunks - 1u)) | (col ^ row));
}

// Builds the interleaved, hidden-pack-swizzled gate/up buffer for decode FFN.
void pack_gate_up_decode(
    std::vector<__nv_bfloat16> &dst,
    const __nv_bfloat16 *gate,
    const __nv_bfloat16 *up) {
  constexpr int hidden_packs = GEMMA4_HIDDEN_SIZE / 8;
  dst.resize(static_cast<size_t>(GEMMA4_PACKED_FFN_SIZE) *
             GEMMA4_HIDDEN_SIZE);

  for (int row = 0; row < GEMMA4_INTERMEDIATE_SIZE; ++row) {
    __nv_bfloat16 *gate_dst =
        dst.data() + static_cast<size_t>(2 * row) * GEMMA4_HIDDEN_SIZE;
    __nv_bfloat16 *up_dst = gate_dst + GEMMA4_HIDDEN_SIZE;
    const __nv_bfloat16 *gate_src =
        gate + static_cast<size_t>(row) * GEMMA4_HIDDEN_SIZE;
    const __nv_bfloat16 *up_src =
        up + static_cast<size_t>(row) * GEMMA4_HIDDEN_SIZE;

    for (int pack = 0; pack < hidden_packs; ++pack) {
      const int src_col = pack * 8;
      const int dst_col = hidden_pack_swizzle_index(pack) * 8;
      memcpy(gate_dst + dst_col, gate_src + src_col,
             8 * sizeof(__nv_bfloat16));
      memcpy(up_dst + dst_col, up_src + src_col,
             8 * sizeof(__nv_bfloat16));
    }
  }
}

// Transposes checkpoint down_proj into the decode FFN row-major stream layout.
void pack_down_decode(
    std::vector<__nv_bfloat16> &dst,
    const __nv_bfloat16 *down) {
  dst.resize(static_cast<size_t>(GEMMA4_INTERMEDIATE_SIZE) *
             GEMMA4_HIDDEN_SIZE);

  for (int hidden = 0; hidden < GEMMA4_HIDDEN_SIZE; ++hidden) {
    const int pack = hidden / 8;
    const int lane = hidden & 7;
    const int dst_col = hidden_pack_swizzle_index(pack) * 8 + lane;
    const __nv_bfloat16 *src =
        down + static_cast<size_t>(hidden) * GEMMA4_INTERMEDIATE_SIZE;
    for (int row = 0; row < GEMMA4_INTERMEDIATE_SIZE; ++row) {
      dst[static_cast<size_t>(row) * GEMMA4_HIDDEN_SIZE + dst_col] = src[row];
    }
  }
}

// Allocates and copies the native FFN decode buffers for one layer.
cudaError_t load_layer_ffn_decode(
    Gemma4TextLayerWeightsDevice &dst,
    const Gemma4CheckpointLayerHost &src,
    std::vector<__nv_bfloat16> &gate_up_stage,
    std::vector<__nv_bfloat16> &down_stage) {
  pack_gate_up_decode(gate_up_stage, src.gate_proj_col_major,
                      src.up_proj_col_major);
  cudaError_t status = alloc_copy(
      &dst.ffn_gate_up_decode, gate_up_stage.data(), gate_up_stage.size());
  if (status != cudaSuccess) {
    return status;
  }

  pack_down_decode(down_stage, src.down_proj_checkpoint);
  status = alloc_copy(&dst.ffn_down_decode, down_stage.data(),
                      down_stage.size());
  if (status != cudaSuccess) {
    cudaFree(dst.ffn_gate_up_decode);
    dst.ffn_gate_up_decode = nullptr;
  }
  return status;
}

// Copies all direct-layout per-layer tensors into device memory.
cudaError_t load_layer_direct(
    Gemma4TextLayerWeightsDevice &dst,
    const Gemma4CheckpointLayerHost &src,
    int layer) {
  const bool global =
      layer == GEMMA4_NUM_LAYERS - 1 ||
      (layer + 1) % GEMMA4_GLOBAL_LAYER_PERIOD == 0;
  const size_t head_dim =
      global ? GEMMA4_GLOBAL_HEAD_DIM : GEMMA4_SLIDING_HEAD_DIM;
  const size_t q_size =
      global ? GEMMA4_GLOBAL_Q_PROJ_SIZE : GEMMA4_SLIDING_Q_PROJ_SIZE;
  const size_t k_size =
      global ? GEMMA4_GLOBAL_K_PROJ_SIZE : GEMMA4_SLIDING_KV_PROJ_SIZE;
  const size_t v_size = global ? 0 : GEMMA4_SLIDING_KV_PROJ_SIZE;
  const size_t o_k =
      global ? GEMMA4_GLOBAL_ATTENTION_OUT_SIZE
             : GEMMA4_SLIDING_ATTENTION_OUT_SIZE;

  cudaError_t status = alloc_copy(
      &dst.input_norm_weight, src.input_norm_weight, GEMMA4_HIDDEN_SIZE);
  if (status != cudaSuccess) return status;
  status = alloc_copy(&dst.post_attention_norm_weight,
                      src.post_attention_norm_weight, GEMMA4_HIDDEN_SIZE);
  if (status != cudaSuccess) return status;
  status = alloc_copy(&dst.pre_feedforward_norm_weight,
                      src.pre_feedforward_norm_weight, GEMMA4_HIDDEN_SIZE);
  if (status != cudaSuccess) return status;
  status = alloc_copy(&dst.post_feedforward_norm_weight,
                      src.post_feedforward_norm_weight, GEMMA4_HIDDEN_SIZE);
  if (status != cudaSuccess) return status;
  status = alloc_copy(&dst.layer_scalar, src.layer_scalar, 1);
  if (status != cudaSuccess) return status;
  status = alloc_copy(&dst.q_norm_weight, src.q_norm_weight, head_dim);
  if (status != cudaSuccess) return status;
  status = alloc_copy(&dst.k_norm_weight, src.k_norm_weight, head_dim);
  if (status != cudaSuccess) return status;
  status = alloc_copy(&dst.q_proj_col_major, src.q_proj_col_major,
                      q_size * GEMMA4_HIDDEN_SIZE);
  if (status != cudaSuccess) return status;
  status = alloc_copy(&dst.k_proj_col_major, src.k_proj_col_major,
                      k_size * GEMMA4_HIDDEN_SIZE);
  if (status != cudaSuccess) return status;
  if (v_size > 0) {
    status = alloc_copy(&dst.v_proj_col_major, src.v_proj_col_major,
                        v_size * GEMMA4_HIDDEN_SIZE);
    if (status != cudaSuccess) return status;
  }
  return alloc_copy(&dst.o_proj_col_major, src.o_proj_col_major,
                    GEMMA4_HIDDEN_SIZE * o_k);
}

// Validates and records the host views for one transformer layer.
bool fill_layer_host(
    Gemma4CheckpointLayerHost &layer,
    const TensorMap &tensors,
    int layer_idx,
    const char *data_base,
    size_t data_bytes,
    std::string *error) {
  const bool global =
      layer_idx == GEMMA4_NUM_LAYERS - 1 ||
      (layer_idx + 1) % GEMMA4_GLOBAL_LAYER_PERIOD == 0;
  const int q_size =
      global ? GEMMA4_GLOBAL_Q_PROJ_SIZE : GEMMA4_SLIDING_Q_PROJ_SIZE;
  const int k_size =
      global ? GEMMA4_GLOBAL_K_PROJ_SIZE : GEMMA4_SLIDING_KV_PROJ_SIZE;
  const int v_size = GEMMA4_SLIDING_KV_PROJ_SIZE;
  const int o_k =
      global ? GEMMA4_GLOBAL_ATTENTION_OUT_SIZE
             : GEMMA4_SLIDING_ATTENTION_OUT_SIZE;
  const int head_dim =
      global ? GEMMA4_GLOBAL_HEAD_DIM : GEMMA4_SLIDING_HEAD_DIM;

  return require_tensor(
             tensors, layer_name(layer_idx, "input_layernorm.weight").c_str(),
             {GEMMA4_HIDDEN_SIZE}, data_base, data_bytes,
             &layer.input_norm_weight, error) &&
         require_tensor(
             tensors,
             layer_name(layer_idx, "post_attention_layernorm.weight").c_str(),
             {GEMMA4_HIDDEN_SIZE}, data_base, data_bytes,
             &layer.post_attention_norm_weight, error) &&
         require_tensor(
             tensors,
             layer_name(layer_idx, "pre_feedforward_layernorm.weight").c_str(),
             {GEMMA4_HIDDEN_SIZE}, data_base, data_bytes,
             &layer.pre_feedforward_norm_weight, error) &&
         require_tensor(
             tensors,
             layer_name(layer_idx, "post_feedforward_layernorm.weight").c_str(),
             {GEMMA4_HIDDEN_SIZE}, data_base, data_bytes,
             &layer.post_feedforward_norm_weight, error) &&
         require_tensor(
             tensors, layer_name(layer_idx, "layer_scalar").c_str(), {1},
             data_base, data_bytes, &layer.layer_scalar, error) &&
         require_tensor(
             tensors, layer_name(layer_idx, "self_attn.q_norm.weight").c_str(),
             {head_dim}, data_base, data_bytes, &layer.q_norm_weight, error) &&
         require_tensor(
             tensors, layer_name(layer_idx, "self_attn.k_norm.weight").c_str(),
             {head_dim}, data_base, data_bytes, &layer.k_norm_weight, error) &&
         require_tensor(
             tensors, layer_name(layer_idx, "self_attn.q_proj.weight").c_str(),
             {q_size, GEMMA4_HIDDEN_SIZE}, data_base, data_bytes,
             &layer.q_proj_col_major, error) &&
         require_tensor(
             tensors, layer_name(layer_idx, "self_attn.k_proj.weight").c_str(),
             {k_size, GEMMA4_HIDDEN_SIZE}, data_base, data_bytes,
             &layer.k_proj_col_major, error) &&
         (global || require_tensor(
             tensors, layer_name(layer_idx, "self_attn.v_proj.weight").c_str(),
             {v_size, GEMMA4_HIDDEN_SIZE}, data_base, data_bytes,
             &layer.v_proj_col_major, error)) &&
         require_tensor(
             tensors, layer_name(layer_idx, "self_attn.o_proj.weight").c_str(),
             {GEMMA4_HIDDEN_SIZE, o_k}, data_base, data_bytes,
             &layer.o_proj_col_major, error) &&
         require_tensor(
             tensors, layer_name(layer_idx, "mlp.gate_proj.weight").c_str(),
             {GEMMA4_INTERMEDIATE_SIZE, GEMMA4_HIDDEN_SIZE}, data_base,
             data_bytes, &layer.gate_proj_col_major, error) &&
         require_tensor(
             tensors, layer_name(layer_idx, "mlp.up_proj.weight").c_str(),
             {GEMMA4_INTERMEDIATE_SIZE, GEMMA4_HIDDEN_SIZE}, data_base,
             data_bytes, &layer.up_proj_col_major, error) &&
         require_tensor(
             tensors, layer_name(layer_idx, "mlp.down_proj.weight").c_str(),
             {GEMMA4_HIDDEN_SIZE, GEMMA4_INTERMEDIATE_SIZE}, data_base,
             data_bytes, &layer.down_proj_checkpoint, error);
}

}  // namespace

bool gemma4_checkpoint_open_text_bf16(
    Gemma4CheckpointHost *checkpoint,
    const char *path,
    std::string *error) {
  if (checkpoint == nullptr || path == nullptr) {
    set_error(error, "checkpoint and path are required");
    return false;
  }
  *checkpoint = Gemma4CheckpointHost();

  const int fd = open(path, O_RDONLY);
  if (fd < 0) {
    set_error(error, std::string("open failed: ") + path);
    return false;
  }

  struct stat st = {};
  if (fstat(fd, &st) != 0 || st.st_size < 8) {
    close(fd);
    set_error(error, "checkpoint file is too small");
    return false;
  }

  void *mapping = mmap(nullptr, static_cast<size_t>(st.st_size), PROT_READ,
                       MAP_PRIVATE, fd, 0);
  close(fd);
  if (mapping == MAP_FAILED) {
    set_error(error, "mmap failed");
    return false;
  }

  const char *bytes = static_cast<const char *>(mapping);
  uint64_t header_bytes = 0;
  memcpy(&header_bytes, bytes, sizeof(header_bytes));
  const size_t data_start = 8 + static_cast<size_t>(header_bytes);
  if (data_start > static_cast<size_t>(st.st_size)) {
    munmap(mapping, static_cast<size_t>(st.st_size));
    set_error(error, "bad safetensors header size");
    return false;
  }

  TensorMap tensors;
  const std::string header(bytes + 8, bytes + 8 + header_bytes);
  if (!parse_header(header, tensors)) {
    munmap(mapping, static_cast<size_t>(st.st_size));
    set_error(error, "could not parse safetensors header");
    return false;
  }

  const char *data_base = bytes + data_start;
  const size_t data_bytes = static_cast<size_t>(st.st_size) - data_start;
  if (!require_tensor(tensors, "model.language_model.embed_tokens.weight",
                      {GEMMA4_VOCAB_SIZE, GEMMA4_HIDDEN_SIZE}, data_base,
                      data_bytes, &checkpoint->token_embedding, error) ||
      !require_tensor(tensors, "model.language_model.norm.weight",
                      {GEMMA4_HIDDEN_SIZE}, data_base, data_bytes,
                      &checkpoint->final_norm_weight, error)) {
    munmap(mapping, static_cast<size_t>(st.st_size));
    return false;
  }

  for (int layer = 0; layer < GEMMA4_NUM_LAYERS; ++layer) {
    if (!fill_layer_host(checkpoint->layers[layer], tensors, layer,
                         data_base, data_bytes, error)) {
      munmap(mapping, static_cast<size_t>(st.st_size));
      return false;
    }
  }

  checkpoint->mapping = mapping;
  checkpoint->mapping_bytes = static_cast<size_t>(st.st_size);
  return true;
}

void gemma4_checkpoint_close(Gemma4CheckpointHost *checkpoint) {
  if (checkpoint == nullptr) {
    return;
  }
  if (checkpoint->mapping != nullptr) {
    munmap(checkpoint->mapping, checkpoint->mapping_bytes);
  }
  *checkpoint = Gemma4CheckpointHost();
}

void gemma4_text_weights_device_free(Gemma4TextWeightsDevice *weights) {
  if (weights == nullptr) {
    return;
  }
  cudaFree(weights->token_embedding);
  cudaFree(weights->final_norm_weight);
  for (int layer = 0; layer < GEMMA4_NUM_LAYERS; ++layer) {
    Gemma4TextLayerWeightsDevice &w = weights->layers[layer];
    cudaFree(w.input_norm_weight);
    cudaFree(w.post_attention_norm_weight);
    cudaFree(w.pre_feedforward_norm_weight);
    cudaFree(w.post_feedforward_norm_weight);
    cudaFree(w.layer_scalar);
    cudaFree(w.q_norm_weight);
    cudaFree(w.k_norm_weight);
    cudaFree(w.q_proj_col_major);
    cudaFree(w.k_proj_col_major);
    cudaFree(w.v_proj_col_major);
    cudaFree(w.o_proj_col_major);
    cudaFree(w.ffn_gate_up_decode);
    cudaFree(w.ffn_down_decode);
  }
  *weights = Gemma4TextWeightsDevice();
}

cudaError_t gemma4_load_text_weights_device_bf16(
    Gemma4TextWeightsDevice *weights,
    const char *path,
    std::string *error) {
  if (weights == nullptr || path == nullptr) {
    set_error(error, "weights and path are required");
    return cudaErrorInvalidValue;
  }
  *weights = Gemma4TextWeightsDevice();

  Gemma4CheckpointHost checkpoint;
  if (!gemma4_checkpoint_open_text_bf16(&checkpoint, path, error)) {
    return cudaErrorInvalidValue;
  }

  cudaError_t status = alloc_copy(
      &weights->token_embedding, checkpoint.token_embedding,
      static_cast<size_t>(GEMMA4_VOCAB_SIZE) * GEMMA4_HIDDEN_SIZE);
  if (status == cudaSuccess) {
    status = alloc_copy(&weights->final_norm_weight,
                        checkpoint.final_norm_weight, GEMMA4_HIDDEN_SIZE);
  }

  std::vector<__nv_bfloat16> gate_up_stage;
  std::vector<__nv_bfloat16> down_stage;
  for (int layer = 0; status == cudaSuccess && layer < GEMMA4_NUM_LAYERS;
       ++layer) {
    status = load_layer_direct(weights->layers[layer],
                               checkpoint.layers[layer], layer);
    if (status == cudaSuccess) {
      status = load_layer_ffn_decode(weights->layers[layer],
                                     checkpoint.layers[layer],
                                     gate_up_stage, down_stage);
    }
  }

  gemma4_checkpoint_close(&checkpoint);
  if (status != cudaSuccess) {
    gemma4_text_weights_device_free(weights);
  }
  return status;
}
