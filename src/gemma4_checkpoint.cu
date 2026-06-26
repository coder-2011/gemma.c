#include "gemma4_checkpoint.cuh"

#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <string>
#include <vector>

namespace {

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

  for (int hidden = 0; hidden < GEMMA4_HIDDEN_SIZE; ++hidden) {
    const int pack = hidden / 8;
    const int lane = hidden & 7;
    const int swizzled_pack = (pack & ~7) | ((pack & 7) ^ ((pack >> 3) & 7));
    const int dst_col = swizzled_pack * 8 + lane;
    const __nv_bfloat16 *src = down + static_cast<size_t>(hidden) * GEMMA4_INTERMEDIATE_SIZE;
    for (int row = 0; row < GEMMA4_INTERMEDIATE_SIZE; ++row) {
      dst[static_cast<size_t>(row) * GEMMA4_HIDDEN_SIZE + dst_col] = src[row];
    }
  }
}

// Allocates and copies the native FFN decode buffers for one layer.
cudaError_t load_layer_ffn_decode(
    Gemma4TextLayerWeightsDevice &dst, const Gemma4CheckpointLayerHost &src,
    std::vector<__nv_bfloat16> &gate_up_stage,
    std::vector<__nv_bfloat16> &down_stage) {
  constexpr int hidden_packs = GEMMA4_HIDDEN_SIZE / 8;
  gate_up_stage.resize(static_cast<size_t>(GEMMA4_PACKED_FFN_SIZE) * GEMMA4_HIDDEN_SIZE);

  for (int row = 0; row < GEMMA4_INTERMEDIATE_SIZE; ++row) {
    __nv_bfloat16 *gate_dst = gate_up_stage.data() + static_cast<size_t>(2 * row) * GEMMA4_HIDDEN_SIZE;
    __nv_bfloat16 *up_dst = gate_dst + GEMMA4_HIDDEN_SIZE;
    const __nv_bfloat16 *gate_src = src.gate_proj_col_major + static_cast<size_t>(row) * GEMMA4_HIDDEN_SIZE;
    const __nv_bfloat16 *up_src = src.up_proj_col_major + static_cast<size_t>(row) * GEMMA4_HIDDEN_SIZE;

    for (int pack = 0; pack < hidden_packs; ++pack) {
      const int src_col = pack * 8;
      const int swizzled_pack = (pack & ~7) | ((pack & 7) ^ ((pack >> 3) & 7));
      const int dst_col = swizzled_pack * 8;
      memcpy(gate_dst + dst_col, gate_src + src_col, 8 * sizeof(__nv_bfloat16));
      memcpy(up_dst + dst_col, up_src + src_col, 8 * sizeof(__nv_bfloat16));
    }
  }

  dst.ffn_gate_up_decode = nullptr;
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
  dst.ffn_down_decode = nullptr;
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

}  // namespace

void gemma4_checkpoint_open_text_bf16(Gemma4CheckpointHost *checkpoint, const char *path) {
  MappedFile file;
  const int fd = open(path, O_RDONLY);
  struct stat st = {};
  fstat(fd, &st);
  file.bytes = static_cast<size_t>(st.st_size);
  file.mapping = mmap(nullptr, file.bytes, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd);
  *checkpoint = Gemma4CheckpointHost();

  const char *bytes = static_cast<const char *>(file.mapping);
  uint64_t header_bytes = 0;
  memcpy(&header_bytes, bytes, sizeof(header_bytes));
  const size_t data_start = 8 + static_cast<size_t>(header_bytes);


  const std::string header(bytes + 8, bytes + 8 + header_bytes);
  const char *data_base = bytes + data_start;

  struct TensorRequest {
    std::string name;
    size_t elements = 0;
    const __nv_bfloat16 **out = nullptr;
    bool required = true;
  };
  std::vector<TensorRequest> requests;
  requests.reserve(2 + GEMMA4_NUM_LAYERS * 14);
  requests.push_back({"model.language_model.embed_tokens.weight", static_cast<size_t>(GEMMA4_VOCAB_SIZE) * GEMMA4_HIDDEN_SIZE, &checkpoint->token_embedding});
  requests.push_back({"model.language_model.norm.weight", GEMMA4_HIDDEN_SIZE, &checkpoint->final_norm_weight});

  for (int layer = 0; layer < GEMMA4_NUM_LAYERS; ++layer) {
    Gemma4CheckpointLayerHost &dst = checkpoint->layers[layer];
    const Gemma4AttentionSpec spec = gemma4_attention_spec(layer);
    const int64_t q_size = int64_t(spec.q_heads) * spec.head_dim;
    const int64_t kv_size = int64_t(spec.kv_heads) * spec.head_dim;
    const std::string prefix = "model.language_model.layers." + std::to_string(layer) + ".";
    requests.insert(requests.end(), {
        {prefix + "input_layernorm.weight", GEMMA4_HIDDEN_SIZE, &dst.input_norm_weight},
        {prefix + "post_attention_layernorm.weight", GEMMA4_HIDDEN_SIZE, &dst.post_attention_norm_weight},
        {prefix + "pre_feedforward_layernorm.weight", GEMMA4_HIDDEN_SIZE, &dst.pre_feedforward_norm_weight},
        {prefix + "post_feedforward_layernorm.weight", GEMMA4_HIDDEN_SIZE, &dst.post_feedforward_norm_weight},
        {prefix + "layer_scalar", 1, &dst.layer_scalar},
        {prefix + "self_attn.q_norm.weight", static_cast<size_t>(spec.head_dim), &dst.q_norm_weight},
        {prefix + "self_attn.k_norm.weight", static_cast<size_t>(spec.head_dim), &dst.k_norm_weight},
        {prefix + "self_attn.q_proj.weight", static_cast<size_t>(q_size) * GEMMA4_HIDDEN_SIZE, &dst.q_proj_col_major},
        {prefix + "self_attn.k_proj.weight", static_cast<size_t>(kv_size) * GEMMA4_HIDDEN_SIZE, &dst.k_proj_col_major},
        {prefix + "self_attn.v_proj.weight", static_cast<size_t>(spec.global ? 0 : kv_size) * GEMMA4_HIDDEN_SIZE, &dst.v_proj_col_major, !spec.global},
        {prefix + "self_attn.o_proj.weight", GEMMA4_HIDDEN_SIZE * static_cast<size_t>(q_size), &dst.o_proj_col_major},
        {prefix + "mlp.gate_proj.weight", static_cast<size_t>(GEMMA4_INTERMEDIATE_SIZE) * GEMMA4_HIDDEN_SIZE, &dst.gate_proj_col_major},
        {prefix + "mlp.up_proj.weight", static_cast<size_t>(GEMMA4_INTERMEDIATE_SIZE) * GEMMA4_HIDDEN_SIZE, &dst.up_proj_col_major},
        {prefix + "mlp.down_proj.weight", static_cast<size_t>(GEMMA4_HIDDEN_SIZE) * GEMMA4_INTERMEDIATE_SIZE, &dst.down_proj_checkpoint},
    });
  }

  constexpr char kOffsets[] = "\"data_offsets\":[";

  for (const TensorRequest &request : requests) {
    if (!request.required) continue;

    const std::string key = "\"" + request.name + "\":";
    const size_t object = header.find(key);
    const size_t offsets = header.find(kOffsets, object + key.size());
    unsigned long long first = 0;
    sscanf(header.c_str() + offsets + sizeof(kOffsets) - 1, "%llu", &first);
    *request.out = reinterpret_cast<const __nv_bfloat16 *>(data_base + first);
  }

  checkpoint->mapping = file.mapping;
  checkpoint->mapping_bytes = file.bytes;
  file.mapping = nullptr;
}

cudaError_t gemma4_load_text_weights_device_bf16(
    Gemma4TextWeightsDevice *weights, const char *path, std::string *error) {
  if (weights == nullptr || path == nullptr) {
    if (error != nullptr) *error = "weights and path are required";
    return cudaErrorInvalidValue;
  }
  *weights = Gemma4TextWeightsDevice();

  Gemma4CheckpointHost checkpoint;
  gemma4_checkpoint_open_text_bf16(&checkpoint, path);

  struct DeviceCopy {
    __nv_bfloat16 **dst = nullptr;
    const __nv_bfloat16 *src = nullptr;
    size_t elements = 0;
  };
  cudaError_t status = cudaSuccess;
  for (const DeviceCopy &copy : {
      DeviceCopy{&weights->token_embedding, checkpoint.token_embedding,
                 static_cast<size_t>(GEMMA4_VOCAB_SIZE) * GEMMA4_HIDDEN_SIZE},
      DeviceCopy{&weights->final_norm_weight, checkpoint.final_norm_weight,
                 GEMMA4_HIDDEN_SIZE},
  }) {
    if (copy.src == nullptr) continue;

    *copy.dst = nullptr;
    const size_t bytes = copy.elements * sizeof(__nv_bfloat16);
    status = cudaMalloc(copy.dst, bytes);
    if (status != cudaSuccess) break;
    status = cudaMemcpy(*copy.dst, copy.src, bytes, cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
      cudaFree(*copy.dst);
      *copy.dst = nullptr;
      break;
    }
  }
  std::vector<__nv_bfloat16> gate_up_stage;
  std::vector<__nv_bfloat16> down_stage;
  for (int layer = 0; status == cudaSuccess && layer < GEMMA4_NUM_LAYERS; ++layer) {
    Gemma4TextLayerWeightsDevice &dst = weights->layers[layer];
    const Gemma4CheckpointLayerHost &src = checkpoint.layers[layer];
    const Gemma4AttentionSpec spec = gemma4_attention_spec(layer);
    const int64_t q_size = int64_t(spec.q_heads) * spec.head_dim;
    const int64_t kv_size = int64_t(spec.kv_heads) * spec.head_dim;

    for (const DeviceCopy &copy : {
        DeviceCopy{&dst.input_norm_weight, src.input_norm_weight, GEMMA4_HIDDEN_SIZE},
        DeviceCopy{&dst.post_attention_norm_weight, src.post_attention_norm_weight, GEMMA4_HIDDEN_SIZE},
        DeviceCopy{&dst.pre_feedforward_norm_weight, src.pre_feedforward_norm_weight, GEMMA4_HIDDEN_SIZE},
        DeviceCopy{&dst.post_feedforward_norm_weight, src.post_feedforward_norm_weight, GEMMA4_HIDDEN_SIZE},
        DeviceCopy{&dst.layer_scalar, src.layer_scalar, 1},
        DeviceCopy{&dst.q_norm_weight, src.q_norm_weight, static_cast<size_t>(spec.head_dim)},
        DeviceCopy{&dst.k_norm_weight, src.k_norm_weight, static_cast<size_t>(spec.head_dim)},
        DeviceCopy{&dst.q_proj_col_major, src.q_proj_col_major, static_cast<size_t>(q_size) * GEMMA4_HIDDEN_SIZE},
        DeviceCopy{&dst.k_proj_col_major, src.k_proj_col_major, static_cast<size_t>(kv_size) * GEMMA4_HIDDEN_SIZE},
        DeviceCopy{&dst.v_proj_col_major, src.v_proj_col_major,
                   static_cast<size_t>(spec.global ? 0 : kv_size) * GEMMA4_HIDDEN_SIZE},
        DeviceCopy{&dst.o_proj_col_major, src.o_proj_col_major, GEMMA4_HIDDEN_SIZE * static_cast<size_t>(q_size)},
    }) {
      if (copy.src == nullptr) continue;

      *copy.dst = nullptr;
      const size_t bytes = copy.elements * sizeof(__nv_bfloat16);
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
      status = load_layer_ffn_decode(weights->layers[layer], checkpoint.layers[layer], gate_up_stage, down_stage);
    }
  }

  munmap(checkpoint.mapping, checkpoint.mapping_bytes);
  checkpoint = Gemma4CheckpointHost();
  if (status != cudaSuccess) {
    for (__nv_bfloat16 *ptr : {weights->token_embedding, weights->final_norm_weight}) cudaFree(ptr);
    for (int layer = 0; layer < GEMMA4_NUM_LAYERS; ++layer) {
      Gemma4TextLayerWeightsDevice &w = weights->layers[layer];
      for (__nv_bfloat16 *ptr : {
          w.input_norm_weight, w.post_attention_norm_weight,
          w.pre_feedforward_norm_weight, w.post_feedforward_norm_weight,
          w.layer_scalar, w.q_norm_weight, w.k_norm_weight, w.q_proj_col_major,
          w.k_proj_col_major, w.v_proj_col_major, w.o_proj_col_major,
          w.ffn_gate_up_decode, w.ffn_down_decode}) {
        cudaFree(ptr);
      }
    }
    *weights = Gemma4TextWeightsDevice();
  }
  return status;
}
