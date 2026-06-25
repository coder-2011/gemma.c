#pragma once

#include <stdint.h>

#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

struct Gemma4MergeInfo {
  int32_t rank = 0;
  int32_t token_id = 0;
};

// Loads and runs the Gemma 4 checkpoint BPE tokenizer used by the prompt path.
class Gemma4Tokenizer {
 public:
  // Loads vocab, merges, and special-token metadata from a tokenizer.json file.
  bool load(const std::string &path, std::string *error);

  // Encodes one prompt string into token IDs using Gemma 4's checkpoint BPE.
  bool encode(const std::string &text, std::vector<int32_t> *ids) const;

  // Decodes generated token IDs, matching tokenizers' default special-token skip.
  bool decode(const std::vector<int32_t> &ids, std::string *text) const;

 private:
  std::vector<std::string> id_to_token_;
  std::unordered_map<std::string_view, int32_t> vocab_;
  std::unordered_map<uint64_t, Gemma4MergeInfo> merge_ranks_;
  std::unordered_map<std::string_view, int32_t> special_token_ids_;
  std::vector<unsigned char> skip_decode_;
};
