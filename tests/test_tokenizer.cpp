#include "gemma4_tokenizer.cuh"

#include <stdio.h>
#include <stdlib.h>

#include <sstream>
#include <string>
#include <vector>

// Quotes one shell argument for the uv tokenizer reference command.
std::string shell_quote(const std::string &text) {
  std::string out = "'";
  for (char c : text) {
    if (c == '\'') {
      out += "'\\''";
    } else {
      out.push_back(c);
    }
  }
  out.push_back('\'');
  return out;
}

// Parses the reference tokenizer's whitespace-separated token ID output.
std::vector<int32_t> parse_id_line(const std::string &line) {
  std::vector<int32_t> ids;
  std::stringstream ss(line);
  int32_t id = 0;
  while (ss >> id) {
    ids.push_back(id);
  }
  return ids;
}

// Runs the tokenizers package through uv and returns checkpoint tokenizer IDs.
std::vector<int32_t> uv_tokenizer_ids(
    const std::string &tokenizer_path,
    const std::string &text) {
  const std::string command =
      "uv run python tests/tokenizer_reference_ids.py " +
      shell_quote(tokenizer_path) + " " + shell_quote(text);
  FILE *pipe = popen(command.c_str(), "r");
  if (pipe == nullptr) {
    fprintf(stderr, "failed to run uv tokenizer reference\n");
    exit(1);
  }

  char buffer[4096];
  std::string output;
  while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
    output += buffer;
  }
  if (pclose(pipe) != 0) {
    fprintf(stderr, "uv tokenizer reference failed\n");
    exit(1);
  }
  return parse_id_line(output);
}

// Compares an encoded prompt with expected checkpoint token IDs.
void expect_ids(
    const Gemma4Tokenizer &tokenizer,
    const std::string &text,
    const std::vector<int32_t> &expected) {
  std::vector<int32_t> actual;
  if (!tokenizer.encode(text, &actual)) {
    fprintf(stderr, "encode failed for input size %zu\n", text.size());
    exit(1);
  }
  if (actual != expected) {
    fprintf(stderr, "encoded IDs mismatch for input size %zu\n", text.size());
    exit(1);
  }
}

// Compares the custom tokenizer with the uv-run tokenizers reference.
void expect_uv_equivalent(
    const Gemma4Tokenizer &tokenizer,
    const std::string &tokenizer_path,
    const std::string &text) {
  std::vector<int32_t> actual;
  if (!tokenizer.encode(text, &actual)) {
    fprintf(stderr, "encode failed for uv case input size %zu\n", text.size());
    exit(1);
  }

  const std::vector<int32_t> expected = uv_tokenizer_ids(tokenizer_path, text);
  if (actual != expected) {
    fprintf(stderr, "uv tokenizer mismatch for input size %zu\n", text.size());
    exit(1);
  }
}

// Compares decoded checkpoint token IDs with expected text.
void expect_text(
    const Gemma4Tokenizer &tokenizer,
    const std::vector<int32_t> &ids,
    const std::string &expected) {
  std::string actual;
  if (!tokenizer.decode(ids, &actual)) {
    fprintf(stderr, "decode failed\n");
    exit(1);
  }
  if (actual != expected) {
    fprintf(stderr, "decoded text mismatch for %zu IDs\n", ids.size());
    exit(1);
  }
}

// Validates the local Gemma tokenizer against stable checkpoint-tokenizer cases.
int main(int argc, char **argv) {
  const char *path = argc > 1 ? argv[1] :
      "models/gemma-4-12B/tokenizer.json";

  Gemma4Tokenizer tokenizer;
  std::string error;
  if (!tokenizer.load(path, &error)) {
    fprintf(stderr, "tokenizer load failed: %s\n", error.c_str());
    return 1;
  }

  const std::string grinning_face = "\xF0\x9F\x98\x80";

  expect_ids(tokenizer, "Hello world", {9259, 1902});
  expect_ids(tokenizer, " Hello", {26352});
  expect_ids(tokenizer, "<bos>Hello<eos>", {2, 9259, 1});
  expect_ids(tokenizer, "", {});
  expect_ids(tokenizer, "   ", {139});
  expect_ids(tokenizer, grinning_face, {242398});
  expect_ids(tokenizer, "<pad><unk><mask>", {0, 3, 4});
  expect_ids(tokenizer, "<|video|><unused99>",
             {258884, 236820, 63634, 236819, 236819, 236813});

  expect_uv_equivalent(tokenizer, path, "alpha<bos>beta<eos>gamma");
  expect_uv_equivalent(tokenizer, path, "<bos><eos><pad><mask>");
  expect_uv_equivalent(tokenizer, path, "left<unused99><|image|>right");

  expect_uv_equivalent(
      tokenizer,
      path,
      "Tabs\tnewlines\nspaces   and NBSP\xC2\xA0" "end");
  expect_uv_equivalent(
      tokenizer,
      path,
      "<bos>prefix <|image|> mix 中文 عربى देवनागरी 😀🏳️‍🌈 <eos>");
  expect_uv_equivalent(
      tokenizer,
      path,
      "JSON-ish: {\"key\": [1, 2, 3]} <> <unused99> café vs café -- end");

  expect_text(tokenizer, {}, "");
  expect_text(tokenizer, {2, 9259, 1}, "Hello");
  expect_text(tokenizer, {26352}, " Hello");
  expect_text(tokenizer, {0, 3, 4}, "");
  expect_text(tokenizer, {242398}, grinning_face);
  expect_text(tokenizer, {464, 368, 410}, "\xE2\x82\xAC");

  puts("tokenizer tests passed");
  return 0;
}
