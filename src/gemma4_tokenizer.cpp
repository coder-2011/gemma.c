#include "gemma4_tokenizer.cuh"

#include <algorithm>
#include <fstream>
#include <limits>
#include <queue>
#include <sstream>
#include <string_view>

namespace {

constexpr const char *kSpaceMarker = "\xE2\x96\x81";

// Writes a diagnostic only when the caller asked for one.
inline void set_error(std::string *error, const std::string &message) {
  if (error != nullptr) {
    *error = message;
  }
}

// Advances past insignificant JSON whitespace.
void skip_ws(const std::string &text, size_t *pos) {
  const size_t next = text.find_first_not_of(" \t\n\r", *pos);
  *pos = next == std::string::npos ? text.size() : next;
}

// Converts one hexadecimal digit to its integer value.
bool hex_digit(char c, uint32_t *out) {
    if (c >= '0' && c <= '9') { *out = c - '0';                    return true; }
    c = std::tolower((unsigned char) c);
    if (c >= 'a' && c <= 'f') { *out = c - 'a' + 10;              return true; }
    return false;
}

// Appends one Unicode scalar as UTF-8.
bool append_utf8(uint32_t codepoint, std::string *out) {
  if (codepoint <= 0x7F) {
    out->push_back(static_cast<char>(codepoint));
    return true;
  }
  if (codepoint <= 0x7FF) {
    out->push_back(static_cast<char>(0xC0 | (codepoint >> 6)));
    out->push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
    return true;
  }
  if (codepoint >= 0xD800 && codepoint <= 0xDFFF) {
    return false;
  }
  if (codepoint <= 0xFFFF) {
    out->push_back(static_cast<char>(0xE0 | (codepoint >> 12)));
    out->push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
    out->push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
    return true;
  }
  if (codepoint <= 0x10FFFF) {
    out->push_back(static_cast<char>(0xF0 | (codepoint >> 18)));
    out->push_back(static_cast<char>(0x80 | ((codepoint >> 12) & 0x3F)));
    out->push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
    out->push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
    return true;
  }
  return false;
}

// Parses four JSON unicode-escape hex digits.
bool parse_hex4(const std::string &text, size_t *pos, uint32_t *out) {
  uint32_t value = 0;
  for (int i = 0; i < 4; ++i) {
    uint32_t digit = 0;
    if (*pos >= text.size() || !hex_digit(text[*pos], &digit)) {
      return false;
    }
    value = value * 16 + digit;
    ++*pos;
  }
  *out = value;
  return true;
}

// Parses a JSON string and unescapes the forms used in tokenizer files.
bool parse_json_string(const std::string &text, size_t *pos, std::string *out) {
  skip_ws(text, pos);
  if (*pos >= text.size() || text[*pos] != '"') {
    return false;
  }
  ++*pos;
  out->clear();
  while (*pos < text.size()) {
    const char c = text[(*pos)++];
    if (c == '"') {
      return true;
    }
    if (c != '\\') {
      out->push_back(c);
      continue;
    }
    if (*pos >= text.size()) {
      return false;
    }
    const char esc = text[(*pos)++];
    if (esc == '"' || esc == '\\' || esc == '/') {
      out->push_back(esc);
    } else if (esc == 'b') {
      out->push_back('\b');
    } else if (esc == 'f') {
      out->push_back('\f');
    } else if (esc == 'n') {
      out->push_back('\n');
    } else if (esc == 'r') {
      out->push_back('\r');
    } else if (esc == 't') {
      out->push_back('\t');
    } else if (esc == 'u') {
      uint32_t codepoint = 0;
      if (!parse_hex4(text, pos, &codepoint)) {
        return false;
      }
      if (codepoint >= 0xD800 && codepoint <= 0xDBFF) {
        if (*pos + 1 >= text.size() || text[*pos] != '\\' ||
            text[*pos + 1] != 'u') {
          return false;
        }
        *pos += 2;
        uint32_t low = 0;
        if (!parse_hex4(text, pos, &low) || low < 0xDC00 || low > 0xDFFF) {
          return false;
        }
        codepoint = 0x10000 + ((codepoint - 0xD800) << 10) + (low - 0xDC00);
      }
      if (!append_utf8(codepoint, out)) {
        return false;
      }
    } else {
      return false;
    }
  }
  return false;
}

// Parses a non-negative JSON integer into an int32 token ID or merge rank.
bool parse_json_i32(const std::string &text, size_t *pos, int32_t *out) {
  skip_ws(text, pos);
  if (*pos >= text.size() || text[*pos] < '0' || text[*pos] > '9') {
    return false;
  }
  int64_t value = 0;
  while (*pos < text.size() && text[*pos] >= '0' && text[*pos] <= '9') {
    value = value * 10 + static_cast<int64_t>(text[*pos] - '0');
    if (value > std::numeric_limits<int32_t>::max()) {
      return false;
    }
    ++*pos;
  }
  *out = static_cast<int32_t>(value);
  return true;
}

// Parses one JSON boolean.
bool parse_json_bool(const std::string &text, size_t *pos, bool *out) {
  skip_ws(text, pos);
  if (text.compare(*pos, 4, "true") == 0) {
    *pos += 4;
    *out = true;
    return true;
  }
  if (text.compare(*pos, 5, "false") == 0) {
    *pos += 5;
    *out = false;
    return true;
  }
  return false;
}

// Consumes one expected JSON punctuation character.
bool expect_char(const std::string &text, size_t *pos, char expected) {
  skip_ws(text, pos);
  if (*pos >= text.size() || text[*pos] != expected) {
    return false;
  }
  ++*pos;
  return true;
}

// Consumes one expected JSON literal such as null.
bool expect_literal(
    const std::string &text,
    size_t *pos,
    const char *literal) {
  skip_ws(text, pos);
  const size_t len = std::char_traits<char>::length(literal);
  if (text.compare(*pos, len, literal) != 0) {
    return false;
  }
  *pos += len;
  return true;
}

// Consumes one known field name and its colon separator.
bool expect_field(
    const std::string &text,
    size_t *pos,
    const char *field) {
  std::string actual;
  if (!parse_json_string(text, pos, &actual)) {
    return false;
  }
  if (actual != field) {
    return false;
  }
  return expect_char(text, pos, ':');
}

// Consumes a string field and checks its exact value.
bool expect_string_field(
    const std::string &text,
    size_t *pos,
    const char *field,
    const std::string &expected) {
  std::string actual;
  if (!expect_field(text, pos, field) ||
      !parse_json_string(text, pos, &actual)) {
    return false;
  }
  return actual == expected;
}

// Consumes a boolean field and checks its exact value.
bool expect_bool_field(
    const std::string &text,
    size_t *pos,
    const char *field,
    bool expected) {
  bool actual = false;
  if (!expect_field(text, pos, field) ||
      !parse_json_bool(text, pos, &actual)) {
    return false;
  }
  return actual == expected;
}

// Consumes an integer field and checks its exact value.
bool expect_i32_field(
    const std::string &text,
    size_t *pos,
    const char *field,
    int32_t expected) {
  int32_t actual = 0;
  if (!expect_field(text, pos, field) ||
      !parse_json_i32(text, pos, &actual)) {
    return false;
  }
  return actual == expected;
}

// Consumes a null field in the fixed tokenizer schema.
bool expect_null_field(
    const std::string &text,
    size_t *pos,
    const char *field) {
  return expect_field(text, pos, field) &&
         expect_literal(text, pos, "null");
}

// Consumes a tokenizer JSON pattern object with one String field.
bool expect_string_pattern(
    const std::string &text,
    size_t *pos,
    const std::string &expected) {
  return expect_char(text, pos, '{') &&
         expect_string_field(text, pos, "String", expected) &&
         expect_char(text, pos, '}');
}

// Validates and consumes the Gemma normalizer schema.
bool validate_normalizer_schema(const std::string &text, size_t *pos) {
  return expect_char(text, pos, '{') &&
         expect_string_field(text, pos, "type", "Replace") &&
         expect_char(text, pos, ',') &&
         expect_field(text, pos, "pattern") &&
         expect_string_pattern(text, pos, " ") &&
         expect_char(text, pos, ',') &&
         expect_string_field(text, pos, "content", kSpaceMarker) &&
         expect_char(text, pos, '}');
}

// Validates and consumes the Gemma pre-tokenizer schema.
bool validate_pre_tokenizer_schema(const std::string &text, size_t *pos) {
  return expect_char(text, pos, '{') &&
         expect_string_field(text, pos, "type", "Split") &&
         expect_char(text, pos, ',') &&
         expect_field(text, pos, "pattern") &&
         expect_string_pattern(text, pos, " ") &&
         expect_char(text, pos, ',') &&
         expect_string_field(text, pos, "behavior", "MergedWithPrevious") &&
         expect_char(text, pos, ',') &&
         expect_bool_field(text, pos, "invert", false) &&
         expect_char(text, pos, '}');
}

// Validates and consumes one TemplateProcessing Sequence entry.
bool validate_template_sequence(
    const std::string &text,
    size_t *pos,
    const char *id,
    int32_t type_id) {
  return expect_char(text, pos, '{') &&
         expect_field(text, pos, "Sequence") &&
         expect_char(text, pos, '{') &&
         expect_string_field(text, pos, "id", id) &&
         expect_char(text, pos, ',') &&
         expect_i32_field(text, pos, "type_id", type_id) &&
         expect_char(text, pos, '}') &&
         expect_char(text, pos, '}');
}

// Validates and consumes the Gemma post-processor schema.
bool validate_post_processor_schema(const std::string &text, size_t *pos) {
  return expect_char(text, pos, '{') &&
         expect_string_field(text, pos, "type", "TemplateProcessing") &&
         expect_char(text, pos, ',') &&
         expect_field(text, pos, "single") &&
         expect_char(text, pos, '[') &&
         validate_template_sequence(text, pos, "A", 0) &&
         expect_char(text, pos, ']') &&
         expect_char(text, pos, ',') &&
         expect_field(text, pos, "pair") &&
         expect_char(text, pos, '[') &&
         validate_template_sequence(text, pos, "A", 0) &&
         expect_char(text, pos, ',') &&
         validate_template_sequence(text, pos, "B", 1) &&
         expect_char(text, pos, ']') &&
         expect_char(text, pos, ',') &&
         expect_field(text, pos, "special_tokens") &&
         expect_char(text, pos, '{') &&
         expect_char(text, pos, '}') &&
         expect_char(text, pos, '}');
}

// Packs two token IDs into the merge-rank lookup key.
uint64_t merge_key(int32_t left_id, int32_t right_id) {
  return (uint64_t(uint32_t(left_id)) << 32) | uint32_t(right_id);
}

struct BpeSymbol {
  size_t n = 0;
  int32_t token_id = -1;
  int prev = -1;
  int next = -1;
};

struct MergeCandidate {
  int32_t rank = 0;
  int left = 0;
  int right = 0;
  int32_t token_id = -1;
  size_t size = 0;
};

struct MergeCandidateGreater {
  // Orders BPE candidates by rank so the next checkpoint merge is on top.
  bool operator()(
      const MergeCandidate &a,
      const MergeCandidate &b) const {
    if (a.rank != b.rank) {
      return a.rank > b.rank;
    }
    return a.left > b.left;
  }
};

// Counts direct children in the object or array at pos for exact reserve sizes.
size_t count_json_children(const std::string &text, size_t pos) {
  skip_ws(text, &pos);
  if (pos >= text.size() || (text[pos] != '{' && text[pos] != '[')) {
    return 0;
  }

  const char close = text[pos] == '{' ? '}' : ']';
  int depth = 0;
  size_t count = 0;
  bool escaped = false;
  bool in_string = false;
  bool saw_value = false;

  for (; pos < text.size(); ++pos) {
    const char c = text[pos];
    if (in_string) {
      if (escaped) {
        escaped = false;
      } else if (c == '\\') {
        escaped = true;
      } else if (c == '"') {
        in_string = false;
      }
      continue;
    }

    if (c == '"') {
      in_string = true;
      saw_value = saw_value || depth == 1;
    } else if (c == '{' || c == '[') {
      if (depth++ == 1) {
        saw_value = true;
      }
    } else if (c == '}' || c == ']') {
      if (depth == 1 && c == close) {
        return saw_value ? count + 1 : 0;
      }
      --depth;
    } else if (depth == 1 && c == ',') {
      ++count;
      saw_value = false;
    } else if (depth == 1 && c != ':' && c > ' ') {
      saw_value = true;
    }
  }
  return 0;
}

// Parses the BPE vocabulary object from tokenizer.json.
bool parse_vocab(
    const std::string &text,
    size_t *pos,
    std::unordered_map<std::string, int32_t> *vocab,
    std::vector<std::string> *id_to_token) {
  skip_ws(text, pos);
  if (*pos >= text.size() || text[*pos] != '{') {
    return false;
  }
  const size_t vocab_size = count_json_children(text, *pos);
  ++*pos;
  vocab->clear();
  vocab->reserve(vocab_size);
  id_to_token->clear();
  id_to_token->reserve(vocab_size);

  while (*pos < text.size()) {
    skip_ws(text, pos);
    if (*pos < text.size() && text[*pos] == '}') {
      ++*pos;
      return !vocab->empty();
    }
    std::string token;
    int32_t id = 0;
    if (!parse_json_string(text, pos, &token)) {
      return false;
    }
    skip_ws(text, pos);
    if (*pos >= text.size() || text[*pos] != ':') {
      return false;
    }
    ++*pos;
    if (!parse_json_i32(text, pos, &id)) {
      return false;
    }
    if (static_cast<size_t>(id) >= id_to_token->size()) {
      id_to_token->resize(static_cast<size_t>(id) + 1);
    }
    (*id_to_token)[id] = token;
    vocab->emplace(std::move(token), id);

    skip_ws(text, pos);
    if (*pos < text.size() && text[*pos] == ',') {
      ++*pos;
    }
  }
  return false;
}

// Adds one live adjacent token pair to the merge heap when the checkpoint ranks it.
void push_merge_candidate(
    const std::unordered_map<uint64_t, Gemma4MergeInfo> &merge_ranks,
    const std::vector<BpeSymbol> &symbols,
    int left,
    int right,
    std::priority_queue<
        MergeCandidate,
        std::vector<MergeCandidate>,
        MergeCandidateGreater> *heap) {
  if (left < 0 || right < 0) {
    return;
  }
  const BpeSymbol &left_sym = symbols[left];
  const BpeSymbol &right_sym = symbols[right];
  if (left_sym.n == 0 || right_sym.n == 0) {
    return;
  }
  const auto it =
      merge_ranks.find(merge_key(left_sym.token_id, right_sym.token_id));
  if (it != merge_ranks.end()) {
    heap->push({
        it->second.rank,
        left,
        right,
        it->second.token_id,
        left_sym.n + right_sym.n,
    });
  }
}

// Parses the BPE merge list into rank order.
bool parse_merges(
    const std::string &text,
    size_t *pos,
    const std::unordered_map<std::string, int32_t> &vocab,
    std::unordered_map<uint64_t, Gemma4MergeInfo> *merge_ranks) {
  skip_ws(text, pos);
  if (*pos >= text.size() || text[*pos] != '[') {
    return false;
  }
  const size_t merge_count = count_json_children(text, *pos);
  ++*pos;
  int32_t rank = 0;
  merge_ranks->clear();
  merge_ranks->reserve(merge_count);
  // Reuse one merge buffer during load instead of allocating per merge pair.
  size_t max_token_len = 0;
  for (const auto &token : vocab) {
    max_token_len = std::max(max_token_len, token.first.size());
  }
  std::string merged;
  merged.reserve(max_token_len);

  while (*pos < text.size()) {
    skip_ws(text, pos);
    if (*pos < text.size() && text[*pos] == ']') {
      ++*pos;
      return !merge_ranks->empty();
    }
    if (*pos >= text.size() || text[*pos] != '[') {
      return false;
    }
    ++*pos;

    std::string left;
    std::string right;
    if (!parse_json_string(text, pos, &left)) {
      return false;
    }
    skip_ws(text, pos);
    if (*pos >= text.size() || text[*pos] != ',') {
      return false;
    }
    ++*pos;
    if (!parse_json_string(text, pos, &right)) {
      return false;
    }
    skip_ws(text, pos);
    if (*pos >= text.size() || text[*pos] != ']') {
      return false;
    }
    ++*pos;

    const auto left_id = vocab.find(left);
    const auto right_id = vocab.find(right);
    merged.clear();
    merged.append(left);
    merged.append(right);
    const auto token_id = vocab.find(merged);
    if (left_id == vocab.end() || right_id == vocab.end() ||
        token_id == vocab.end()) {
      return false;
    }
    merge_ranks->emplace(
        merge_key(left_id->second, right_id->second),
        Gemma4MergeInfo{rank++, token_id->second});
    skip_ws(text, pos);
    if (*pos < text.size() && text[*pos] == ',') {
      ++*pos;
    }
  }
  return false;
}

// Parses one added-token object in the known Gemma tokenizer schema.
bool parse_added_token_schema(
    const std::string &text,
    size_t *pos,
    std::string *content,
    int32_t *id) {
  return expect_char(text, pos, '{') &&
         expect_field(text, pos, "id") &&
         parse_json_i32(text, pos, id) &&
         expect_char(text, pos, ',') &&
         expect_field(text, pos, "content") &&
         parse_json_string(text, pos, content) &&
         expect_char(text, pos, ',') &&
         expect_bool_field(text, pos, "single_word", false) &&
         expect_char(text, pos, ',') &&
         expect_bool_field(text, pos, "lstrip", false) &&
         expect_char(text, pos, ',') &&
         expect_bool_field(text, pos, "rstrip", false) &&
         expect_char(text, pos, ',') &&
         expect_bool_field(text, pos, "normalized", false) &&
         expect_char(text, pos, ',') &&
         expect_bool_field(text, pos, "special", true) &&
         expect_char(text, pos, '}');
}

// Parses special tokens so encode can preserve them and decode can skip them.
bool parse_added_tokens(
    const std::string &text,
    size_t *pos,
    std::vector<std::pair<std::string, int32_t>> *special_tokens) {
  if (!expect_char(text, pos, '[')) {
    return false;
  }
  special_tokens->clear();

  while (*pos < text.size()) {
    skip_ws(text, pos);
    if (*pos < text.size() && text[*pos] == ']') {
      ++*pos;
      return true;
    }
    std::string content;
    int32_t id = 0;
    if (!parse_added_token_schema(text, pos, &content, &id)) {
      return false;
    }
    special_tokens->push_back({std::move(content), id});
    skip_ws(text, pos);
    if (*pos < text.size() && text[*pos] == ',') {
      ++*pos;
    }
  }
  return false;
}

// Parses the model object payloads the runtime actually needs.
bool parse_model_schema(
    const std::string &text,
    size_t *pos,
    std::unordered_map<std::string, int32_t> *vocab,
    std::unordered_map<uint64_t, Gemma4MergeInfo> *merge_ranks,
    std::vector<std::string> *id_to_token) {
  if (!expect_char(text, pos, '{')) {
    return false;
  }

  constexpr const char *kVocabField = "\"vocab\"";
  const size_t vocab_key = text.find(kVocabField, *pos);
  if (vocab_key == std::string::npos) {
    return false;
  }
  *pos = vocab_key + std::char_traits<char>::length(kVocabField);

  return expect_char(text, pos, ':') &&
         parse_vocab(text, pos, vocab, id_to_token) &&
         expect_char(text, pos, ',') &&
         expect_field(text, pos, "merges") &&
         parse_merges(text, pos, *vocab, merge_ranks) &&
         expect_char(text, pos, '}');
}

// Parses tokenizer.json in the exact Gemma 4 layout exported by tokenizers.
bool parse_tokenizer_schema(
    const std::string &text,
    std::unordered_map<std::string, int32_t> *vocab,
    std::unordered_map<uint64_t, Gemma4MergeInfo> *merge_ranks,
    std::vector<std::string> *id_to_token,
    std::vector<std::pair<std::string, int32_t>> *special_tokens) {
  size_t pos = 0;
  const bool ok =
      expect_char(text, &pos, '{') &&
      expect_string_field(text, &pos, "version", "1.0") &&
      expect_char(text, &pos, ',') &&
      expect_null_field(text, &pos, "truncation") &&
      expect_char(text, &pos, ',') &&
      expect_null_field(text, &pos, "padding") &&
      expect_char(text, &pos, ',') &&
      expect_field(text, &pos, "added_tokens") &&
      parse_added_tokens(text, &pos, special_tokens) &&
      expect_char(text, &pos, ',') &&
      expect_field(text, &pos, "normalizer") &&
      validate_normalizer_schema(text, &pos) &&
      expect_char(text, &pos, ',') &&
      expect_field(text, &pos, "pre_tokenizer") &&
      validate_pre_tokenizer_schema(text, &pos) &&
      expect_char(text, &pos, ',') &&
      expect_field(text, &pos, "post_processor") &&
      validate_post_processor_schema(text, &pos) &&
      expect_char(text, &pos, ',');
  if (!ok) {
    return false;
  }

  constexpr const char *kModelField = "\"model\"";
  pos = text.find(kModelField, pos);
  if (pos == std::string::npos) {
    return false;
  }
  pos += std::char_traits<char>::length(kModelField);

  const bool model_ok =
      expect_char(text, &pos, ':') &&
      parse_model_schema(text, &pos, vocab, merge_ranks, id_to_token) &&
      expect_char(text, &pos, '}');
  skip_ws(text, &pos);
  return model_ok && pos == text.size();
}

// Returns the UTF-8 byte length for the next valid codepoint, or 1 on bad input.
size_t utf8_len(const std::string &text, size_t pos) {
  const unsigned char c = static_cast<unsigned char>(text[pos]);
  if (c < 0x80) {
    return 1;
  }
  if ((c >> 5) == 0x6 && pos + 1 < text.size()) {
    return 2;
  }
  if ((c >> 4) == 0xE && pos + 2 < text.size()) {
    return 3;
  }
  if ((c >> 3) == 0x1E && pos + 3 < text.size()) {
    return 4;
  }
  return 1;
}

// Builds the tokenizer.json byte-fallback token for one UTF-8 source byte.
std::string byte_fallback_token(unsigned char byte) {
  constexpr char kHex[] = "0123456789ABCDEF";
  std::string token = "<0x00>";
  token[3] = kHex[byte >> 4];
  token[4] = kHex[byte & 0xF];
  return token;
}

// Appends one BPE symbol while keeping the adjacent-symbol chain linked.
void append_bpe_symbol(
    std::vector<BpeSymbol> *symbols,
    size_t n,
    int32_t token_id) {
  const int index = static_cast<int>(symbols->size());
  const int prev = index - 1;
  if (prev >= 0) {
    (*symbols)[prev].next = index;
  }
  symbols->push_back({n, token_id, prev, -1});
}

// Encodes normalized text with the same symbol-chain BPE shape as llama.cpp.
bool tokenize_normalized_text(
    const std::string &text,
    const std::unordered_map<std::string_view, int32_t> &vocab,
    const std::unordered_map<uint64_t, Gemma4MergeInfo> &merge_ranks,
    std::vector<int32_t> *ids) {
  std::vector<BpeSymbol> symbols;
  symbols.reserve(text.size());

  for (size_t pos = 0; pos < text.size();) {
    const size_t n = std::min(utf8_len(text, pos), text.size() - pos);
    const std::string_view piece(text.data() + pos, n);
    const auto token = vocab.find(piece);
    if (token != vocab.end()) {
      append_bpe_symbol(&symbols, n, token->second);
      pos += n;
      continue;
    }
    for (size_t byte = 0; byte < n; ++byte) {
      const unsigned char value = static_cast<unsigned char>(text[pos + byte]);
      const std::string fallback = byte_fallback_token(value);
      const auto fallback_token = vocab.find(fallback);
      if (fallback_token == vocab.end()) {
        return false;
      }
      append_bpe_symbol(&symbols, 1, fallback_token->second);
    }
    pos += n;
  }

  std::priority_queue<
      MergeCandidate,
      std::vector<MergeCandidate>,
      MergeCandidateGreater> heap;

  for (int i = 1; i < static_cast<int>(symbols.size()); ++i) {
    push_merge_candidate(merge_ranks, symbols, i - 1, i, &heap);
  }

  while (!heap.empty()) {
    const MergeCandidate candidate = heap.top();
    heap.pop();
    BpeSymbol &left = symbols[candidate.left];
    BpeSymbol &right = symbols[candidate.right];
    if (left.n == 0 || right.n == 0 || left.next != candidate.right ||
        left.n + right.n != candidate.size) {
      continue;
    }

    left.n += right.n;
    left.token_id = candidate.token_id;
    right.n = 0;

    left.next = right.next;
    if (right.next >= 0) {
      symbols[right.next].prev = candidate.left;
    }
    push_merge_candidate(
        merge_ranks, symbols, left.prev, candidate.left, &heap);
    push_merge_candidate(
        merge_ranks, symbols, candidate.left, left.next, &heap);
  }

  for (int i = symbols.empty() ? -1 : 0; i != -1; i = symbols[i].next) {
    ids->push_back(symbols[i].token_id);
  }
  return true;
}

// Normalizes Gemma spaces by replacing ASCII space with U+2581.
void normalize_text(std::string_view text, std::string *out) {
  out->clear();
  out->reserve(text.size() * 3);
  for (char c : text) {
    if (c == ' ') {
      *out += kSpaceMarker;
    } else {
      out->push_back(c);
    }
  }
}

// Encodes one non-special text span.
bool encode_text_piece(
    std::string_view text,
    const std::unordered_map<std::string_view, int32_t> &vocab,
    const std::unordered_map<uint64_t, Gemma4MergeInfo> &merge_ranks,
    std::string *normalized,
    std::vector<int32_t> *ids) {
  normalize_text(text, normalized);
  return tokenize_normalized_text(*normalized, vocab, merge_ranks, ids);
}

// Applies the tokenizer Replace decoder for Gemma's space marker.
std::string decode_replace_space_marker(const std::string &token) {
  std::string out;
  for (size_t pos = 0; pos < token.size();) {
    if (token.compare(pos, 3, kSpaceMarker) == 0) {
      out.push_back(' ');
      pos += 3;
    } else {
      out.push_back(token[pos++]);
    }
  }
  return out;
}

}  // namespace

// Loads vocab, merges, and special-token metadata from a tokenizer.json file.
bool Gemma4Tokenizer::load(const std::string &path, std::string *error) {
  std::string json;
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    set_error(error, "failed to read tokenizer.json");
    return false;
  }
  std::ostringstream ss;
  ss << in.rdbuf();
  json = ss.str();

  std::unordered_map<std::string, int32_t> vocab;
  std::unordered_map<uint64_t, Gemma4MergeInfo> merge_ranks;
  std::vector<std::string> id_to_token;
  std::vector<std::pair<std::string, int32_t>> special_tokens;

  if (!parse_tokenizer_schema(
          json, &vocab, &merge_ranks, &id_to_token, &special_tokens)) {
    set_error(error, "tokenizer.json does not match Gemma 4 tokenizer schema");
    return false;
  }
  std::vector<unsigned char> skip_decode(id_to_token.size(), 0);
  for (const auto &entry : special_tokens) {
    if (entry.second >= 0 &&
        static_cast<size_t>(entry.second) < skip_decode.size()) {
      skip_decode[entry.second] = 1;
    }
  }

  vocab_.clear();
  special_token_ids_.clear();
  id_to_token_ = std::move(id_to_token);

  vocab_.reserve(vocab.size());
  for (const auto &entry : vocab) {
    const int32_t id = entry.second;
    if (id < 0 || static_cast<size_t>(id) >= id_to_token_.size()) {
      set_error(error, "tokenizer vocab ID is out of range");
      return false;
    }
    vocab_.emplace(std::string_view(id_to_token_[id]), id);
  }

  special_token_ids_.reserve(special_tokens.size());
  for (const auto &entry : special_tokens) {
    const int32_t id = entry.second;
    if (id < 0 || static_cast<size_t>(id) >= id_to_token_.size() ||
        id_to_token_[id] != entry.first) {
      set_error(error, "tokenizer special token metadata is inconsistent");
      return false;
    }
    special_token_ids_.emplace(std::string_view(id_to_token_[id]), id);
  }

  merge_ranks_ = std::move(merge_ranks);
  skip_decode_ = std::move(skip_decode);

  return true;
}

// Encodes one prompt string into token IDs using Gemma 4's checkpoint BPE.
bool Gemma4Tokenizer::encode(
    const std::string &text,
    std::vector<int32_t> *ids) const {
  ids->clear();
  size_t span_start = 0;
  std::string normalized;
  for (size_t pos = 0; pos < text.size();) {
    if (text[pos] != '<') {
      ++pos;
      continue;
    }
    const size_t end = text.find('>', pos + 1);
    if (end == std::string::npos) {
      break;
    }
    const std::string_view token(text.data() + pos, end - pos + 1);
    const auto special = special_token_ids_.find(token);
    if (special == special_token_ids_.end()) {
      ++pos;
      continue;
    }
    if (span_start < pos) {
      const std::string_view span(text.data() + span_start, pos - span_start);
      if (!encode_text_piece(
              span, vocab_, merge_ranks_, &normalized, ids)) {
        return false;
      }
    }
    ids->push_back(special->second);
    pos = end + 1;
    span_start = pos;
  }
  if (span_start < text.size()) {
    const std::string_view span(text.data() + span_start,
                                text.size() - span_start);
    if (!encode_text_piece(span, vocab_, merge_ranks_, &normalized, ids)) {
      return false;
    }
  }
  return true;
}

// Decodes generated token IDs, matching tokenizers' default special-token skip.
bool Gemma4Tokenizer::decode(
    const std::vector<int32_t> &ids,
    std::string *text) const {
  text->clear();
  for (int32_t id : ids) {
    if (id < 0 || static_cast<size_t>(id) >= id_to_token_.size()) {
      return false;
    }
    if (skip_decode_[id]) {
      continue;
    }
    const std::string token = decode_replace_space_marker(id_to_token_[id]);
    *text += token;
  }
  return true;
}
