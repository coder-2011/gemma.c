#include "gemma4_tokenizer.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

struct Config {
  const char *tokenizer_path = nullptr;
  const char *data_path = nullptr;
  int warmup = 1;
  int iters = 1;
  int samples = 5;
  int sleep_ms = 0;
};

// Prints the positional command format for this custom-tokenizer runner.
void print_usage(const char *program) {
  std::fprintf(stderr,
               "usage: %s TOKENIZER_JSON DATA_FILE [warmup iters samples "
               "sleep_ms]\n",
               program);
}

// Parses a bounded integer argument so bad benchmark counts fail loudly.
int parse_count(const char *text, const char *name, int min_value) {
  char *end = nullptr;
  const long value = std::strtol(text, &end, 10);
  if (end == text || *end != '\0' || value < min_value) {
    throw std::runtime_error(std::string("invalid ") + name);
  }
  return static_cast<int>(value);
}

// Converts positional arguments into the minimal benchmark configuration.
Config parse_args(int argc, char **argv) {
  if (argc < 3 || argc > 7) {
    print_usage(argv[0]);
    std::exit(2);
  }

  Config config;
  config.tokenizer_path = argv[1];
  config.data_path = argv[2];
  if (argc > 3) {
    config.warmup = parse_count(argv[3], "warmup", 0);
  }
  if (argc > 4) {
    config.iters = parse_count(argv[4], "iters", 1);
  }
  if (argc > 5) {
    config.samples = parse_count(argv[5], "samples", 1);
  }
  if (argc > 6) {
    config.sleep_ms = parse_count(argv[6], "sleep_ms", 0);
  }
  return config;
}

// Reads the benchmark corpus before timing so disk I/O is not measured.
std::string read_file(const char *path) {
  FILE *file = std::fopen(path, "rb");
  if (file == nullptr) {
    throw std::runtime_error("failed to open data file");
  }
  if (std::fseek(file, 0, SEEK_END) != 0) {
    std::fclose(file);
    throw std::runtime_error("failed to seek data file");
  }
  const long size = std::ftell(file);
  if (size < 0) {
    std::fclose(file);
    throw std::runtime_error("failed to size data file");
  }
  std::rewind(file);

  std::string text(static_cast<size_t>(size), '\0');
  const size_t read = std::fread(text.data(), 1, text.size(), file);
  std::fclose(file);
  if (read != text.size()) {
    throw std::runtime_error("failed to read data file");
  }
  return text;
}

// Splits the corpus into non-empty lines to mimic repeated prompt tokenization.
std::vector<std::string> split_nonempty_lines(const std::string &text) {
  std::vector<std::string> lines;
  size_t start = 0;
  for (size_t pos = 0; pos <= text.size(); ++pos) {
    if (pos < text.size() && text[pos] != '\n') {
      continue;
    }
    size_t end = pos;
    if (end > start && text[end - 1] == '\r') {
      --end;
    }
    if (end > start) {
      lines.emplace_back(text.data() + start, end - start);
    }
    start = pos + 1;
  }
  return lines;
}

// Counts corpus bytes once so throughput math is outside the timed loop.
uint64_t total_bytes(const std::vector<std::string> &texts) {
  uint64_t bytes = 0;
  for (const std::string &text : texts) {
    bytes += text.size();
  }
  return bytes;
}

// Encodes the full dataset repeatedly and consumes token counts.
uint64_t encode_dataset(
    const Gemma4Tokenizer &tokenizer,
    const std::vector<std::string> &texts,
    int iters) {
  uint64_t tokens = 0;
  std::vector<int32_t> ids;
  for (int iter = 0; iter < iters; ++iter) {
    for (const std::string &text : texts) {
      ids.clear();
      if (!tokenizer.encode(text, &ids)) {
        throw std::runtime_error("tokenizer encode failed");
      }
      tokens += ids.size();
    }
  }
  return tokens;
}

// Returns the median from an already collected set of sample timings.
double median_ms(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  const size_t mid = values.size() / 2;
  if (values.size() % 2 == 1) {
    return values[mid];
  }
  return 0.5 * (values[mid - 1] + values[mid]);
}

}  // namespace

// Benchmarks the custom tokenizer with initialization excluded from timing.
int main(int argc, char **argv) {
  try {
    const Config config = parse_args(argc, argv);

    Gemma4Tokenizer tokenizer;
    std::string error;
    if (!tokenizer.load(config.tokenizer_path, &error)) {
      throw std::runtime_error("tokenizer load failed: " + error);
    }

    const std::string data = read_file(config.data_path);
    const std::vector<std::string> texts = split_nonempty_lines(data);
    if (texts.empty()) {
      throw std::runtime_error("data file has no non-empty lines");
    }

    uint64_t checksum = 0;
    for (int i = 0; i < config.warmup; ++i) {
      checksum += encode_dataset(tokenizer, texts, 1);
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(config.sleep_ms));

    std::vector<double> samples;
    samples.reserve(config.samples);
    uint64_t sample_tokens = 0;
    for (int i = 0; i < config.samples; ++i) {
      const auto start = std::chrono::steady_clock::now();
      sample_tokens = encode_dataset(tokenizer, texts, config.iters);
      const auto stop = std::chrono::steady_clock::now();
      checksum += sample_tokens;
      const std::chrono::duration<double, std::milli> elapsed = stop - start;
      samples.push_back(elapsed.count());
    }

    const double median = median_ms(samples);
    const double min_ms = *std::min_element(samples.begin(), samples.end());
    const double max_ms = *std::max_element(samples.begin(), samples.end());
    const double mean_ms =
        std::accumulate(samples.begin(), samples.end(), 0.0) /
        double(samples.size());
    double variance = 0.0;
    for (double sample : samples) {
      const double delta = sample - mean_ms;
      variance += delta * delta;
    }
    const double stddev_ms =
        samples.size() > 1 ? std::sqrt(variance / double(samples.size() - 1))
                           : 0.0;
    const double seconds = median / 1000.0;
    const double docs = double(texts.size()) * double(config.iters);
    const double mib = double(total_bytes(texts)) * double(config.iters) /
                       (1024.0 * 1024.0);

    std::printf("benchmark_contract name=tokenizer_bench "
                "measurement=host_visible_tokenizer_end_to_end "
                "timing=steady_clock_cpu cache=process_warm_filesystem_excluded "
                "launch_overhead=not_applicable aggregation=raw_samples "
                "correctness=encode_success_and_token_count_consumed "
                "warmup=%d iters=%d samples=%d sleep_ms=%d\n",
                config.warmup, config.iters, config.samples, config.sleep_ms);
    if (config.sleep_ms > 0) {
      std::printf("benchmark_warning name=tokenizer_bench sleep_ms=%d "
                  "reason=pre_timing_sleep_models_bursty_cpu_work\n",
                  config.sleep_ms);
    }
    std::printf("tokenizer_bench impl=custom_cpp warmup=%d iters=%d "
                "samples=%d docs_per_iter=%zu median_ms=%.3f min_ms=%.3f "
                "max_ms=%.3f mean_ms=%.3f stddev_ms=%.3f docs_per_s=%.2f "
                "mib_per_s=%.2f tokens_per_s=%.2f tokens_per_sample=%llu "
                "checksum=%llu samples_ms=[",
                config.warmup, config.iters, config.samples, texts.size(),
                median, min_ms, max_ms, mean_ms, stddev_ms, docs / seconds,
                mib / seconds, double(sample_tokens) / seconds,
                static_cast<unsigned long long>(sample_tokens),
                static_cast<unsigned long long>(checksum));
    for (size_t i = 0; i < samples.size(); ++i) {
      std::printf("%s%.3f", i == 0 ? "" : ",", samples[i]);
    }
    std::printf("]\n");
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "tokenizer bench failed: %s\n", e.what());
    return 1;
  }
}
