##  Experiments Log

AI-updated and directed log of the experiments I ran throughout this project to optimize the kernels. 
Expect this to be very messy and pretty much useless for most people to look at.  it is meant to be a place for me and my agents to fuck around

## 2026-06-30 - Decode layer fused ingress

Change:

- Moved batch-1 decode input RMSNorm, QKV/QK projection, Q/K/V prep, and paged
  K/V writes into the existing cooperative decode-layer kernel.
- Removed the separate decode-loop calls to RMSNorm projection and decode prep.
- Kept the raw QKV/QK global scratch handoff: sliding uses `attention_out`;
  global uses `attention_q`.

Commands:

```bash
make test-flash-attention-cpp test-decode-megakernel
make decode-bench prompt
git diff --check
```

Results:

- `test_flash_attention` passed.
- `test_decode_megakernel` passed, including sliding/global ingress comparison
  against the old projection-plus-prep reference.
- `decode-bench` was already up to date and `gemma4_prompt` linked.
- Full `gemma4_prompt --benchmark-mode decode-step` was not run because
  `models/gemma-4-12B/model.safetensors` is not present in this workspace.

## 2026-06-27 - FFN Decode Benchmark Roofline Sanity Audit

Question:

- Are the new Tier-2 FFN path and the existing custom FFN path mathematically
  broken, or is the benchmark contract overclaiming the absolute microsecond
  timings?

Change:

- Added Tier-2 FFN decode coverage to the existing sparse `test-ffn-decode`
  fixture. This checks Tier-2 MLP, normed output, and residual output against
  the same CPU reference used for the existing fused decode path.
- Updated `gemma4_ffn_libtorch_bench` to label the cache rows as
  `flush_hint` instead of cold-cache proof.
- Added a simple FFN weight-stream roofline and warnings when measured decode
  rows fall below the single-read weight floor.

Commands:

```bash
make test-ffn-decode
make ffn-libtorch-bench
GEMMA4_FFN_LIBTORCH_BENCH_SEED=0x12345678 ./build/benches/gemma4_ffn_libtorch_bench 100 20 3 1
GEMMA4_FFN_LIBTORCH_BENCH_SEED=0xabcdef01 ./build/benches/gemma4_ffn_libtorch_bench 100 20 3 1
git diff --check
```

Additional one-off sanity checks:

- A quick PyTorch device-to-device copy of roughly one FFN layer of bytes
  reported about `0.036 ms`.
- A scratch CUDA `uint32` copy kernel over `353894400` bytes reported about
  `0.011-0.012 ms`, with a checksum consumed after the copy.
- These raw-copy numbers are also far above an A6000 DRAM roofline, so this
  environment/harness cannot prove cold HBM traffic with elapsed time alone.
- Root cause found later: `/etc/ld.so.preload` injects
  `/etc/thunder/libthunder.so`, which interposes CUDA event, stream, sync, copy,
  graph, CUPTI, and NVML APIs. Treat all CUDA timing on this host as invalid
  until rerun without Thunder preloaded.

Contract:

- Hardware: NVIDIA RTX A6000, driver `580.126.16`, CUDA `13.0`.
- Library baseline: LibTorch `2.11.0`.
- Shape: BF16 decode row, hidden `3840`, intermediate `15360`.
- Timing: CUDA events on the measured stream, plus device-sync wall diagnostic
  for current custom decode.
- Cache policy: warm repeated buffers; flush rows are only eviction hints.
- Profiler note: `ncu` and `compute-sanitizer` were not available here.

Results:

```text
FFN unique weight bytes: 337.500 MiB
A6000 DRAM peak:         715.345 GiB/s
single weight-read floor: 0.460742 ms
```

Seed `0x12345678`:

```text
tier2_decode_full              0.010468 ms
custom_fused_decode_events     0.028748 ms
cache_flush_hint_only          0.005045 ms
cache_flush_hint_gib_s     49552.441 GiB/s
```

Seed `0xabcdef01`:

```text
tier2_decode_full              0.011795 ms
custom_fused_decode_events     0.027579 ms
cache_flush_hint_only          0.004890 ms
cache_flush_hint_gib_s     51126.522 GiB/s
```

Correctness:

- `make test-ffn-decode` passed with both existing fused decode and Tier-2
  sparse-reference checks.
- Benchmark LibTorch checks passed on both fixed seeds:
  - Tier-2 MLP max abs `<= 4.76837e-07`.
  - Tier-2 full max abs `<= 0.000976562`.
  - Existing custom full max abs `<= 0.000976562`.

Conclusion:

- The evidence does not show a math bug in either the existing custom FFN path
  or the new Tier-2 path.
- The absolute `~0.01-0.03 ms` FFN decode timings are not trustworthy as real
  HBM-streaming layer latency. They violate the single FFN weight-read roofline,
  and the flush hint itself reports impossible effective bandwidth.
- Treat the Tier-2/current ratios as warm-buffer microbenchmark leads only.
  Do not promote them as real decode-layer latency without `ncu` traffic and
  timing counters on a profiling-capable host without Thunder preloaded.

## 2026-06-26 - Tier-2 FFN Decode Prototype

Question:

- Does a fixed-shape Tier-2 decode FFN prototype, which computes one GeGLU
  tile and reuses it across a wider output group, beat the current FFN MLP and
  full decode path under a fair benchmark?

Change:

- Added `gemma4_ffn_tier2.cu` / `.cuh` as an experimental path, leaving
  `gemma4_ffn.cu` untouched.
- Wired Tier-2 MLP-only and Tier-2 full decode rows into
  `gemma4_ffn_libtorch_bench`.
- Fixed a benchmark fairness issue: current custom full decode now has a
  same-stream CUDA-event timing row, while the old device-sync wall row remains
  only as a host/launch-overhead diagnostic.
- Rejected CUDA graph rows after they produced impossible sub-microsecond
  timings in this harness.
- Tier-2 now uses a 256-column output group and a fused post-FFN
  RMSNorm/residual-add kernel with 128 threads.

Commands:

```bash
make ffn-libtorch-bench
./build/benches/gemma4_ffn_libtorch_bench 100 20 5 1
GEMMA4_FFN_LIBTORCH_BENCH_SEED=0x12345678 ./build/benches/gemma4_ffn_libtorch_bench 100 20 5 1
GEMMA4_FFN_LIBTORCH_BENCH_SEED=0xabcdef01 ./build/benches/gemma4_ffn_libtorch_bench 100 20 5 1
make test-ffn-decode
git diff --check
```

Contract:

- Hardware: NVIDIA RTX A6000, PCI bus `00000000:0C:00.0`, driver
  `580.126.16`, persistence enabled, ECC disabled, power limit `300 W`.
- Toolchain: `/usr/local/cuda/bin/nvcc`, CUDA `13.0.48`.
- Library baseline: LibTorch `2.11.0`.
- Shape: BF16 decode row, hidden `3840`, intermediate `15360`.
- Timing: CUDA events on the measured stream for the headline rows; current
  custom decode also reports the old device-sync wall row as a diagnostic.
- Warmup/iterations: `20` warmups, `100` timed iterations, `5` trials.
- Cache policy: warm repeated buffers, plus explicit L2-flush rows.
- Clock policy: clocks not locked.
- Correctness tolerance: fail above `0.125` max abs versus LibTorch reference.
- Profiler note: `ncu` was not available on this machine.

Results, seed `0x12345678`:

```text
path                             best_ms   avg_ms    max_abs
libtorch_mlp                     0.116685  0.117135  0
libtorch_full_ffn                0.193577  0.194255  0
tier2_decode_mlp                 0.006037  0.006165  4.76837e-07
tier2_decode_full                0.012279  0.012491  0.000976562
custom_fused_decode_events       0.030521  0.030824  0.000976562
custom_fused_decode_wall         0.030517  0.031101  0.000976562
custom_prefill_mlp               0.022564  0.028632  9.53674e-07
tier2_decode_mlp_cold            0.017627  0.019577
cache_flush_only                 0.007070  0.007342
cold_tier2_minus_flush_ms        0.010558
```

Derived speedups, seed `0x12345678`:

```text
tier2_decode_mlp_vs_libtorch_mlp_speedup      19.328335x
tier2_decode_mlp_vs_current_mlp_speedup        3.737679x
tier2_decode_full_vs_libtorch_full_speedup    15.765064x
tier2_decode_full_vs_current_decode_speedup    2.485638x
```

Confirmation, seed `0xabcdef01`:

```text
tier2_decode_mlp                 0.005610  0.005893  4.76837e-07
tier2_decode_full                0.011461  0.012055  0.000976562
custom_fused_decode_events       0.028447  0.036613  0.000976562
tier2_decode_full_vs_current_decode_speedup    2.482032x
```

Conclusion:

- The Tier-2 prototype was correct against LibTorch and the current custom MLP
  at BF16 tolerances.
- After the fair custom event-timing row and fused post kernel, full Tier-2
  decode was about `2.48x` faster than current custom full decode on two fixed
  seeds.
- Rerun under `ncu` when available before treating the microsecond-scale timing
  as profiler-grounded.

## 2026-06-25 - Sampling Bench LibTorch Baseline Port

Question:

- Can the old Python `gemma4_sampling_torch_bench.py` native PyTorch CUDA
  graph comparison live directly inside the sampling mechanism benchmark?

Change:

- Added a LibTorch CUDA-graph row to `gemma4_sampling_bench`.
- Kept the existing custom fused sampler and materialized CUDA reference rows.
- Deleted `gemma4_sampling_torch_bench.py`.

Commands:

```bash
make sampling-bench
./build/benches/gemma4_sampling_bench --warmup 5 --iters 10 --samples 3
make test-sampling
```

Contract:

- Hardware: NVIDIA RTX A6000, PCI bus `00000000:06:00.0`, driver
  `580.126.16`, persistence enabled, ECC disabled, power limit `300 W`.
- Toolchain: `/usr/local/cuda/bin/nvcc`, CUDA `13.0.48`.
- Library baseline: LibTorch `2.11.0`.
- Shape: BF16 hidden `[1, 3840]`, tied LM head `[262144, 3840]`.
- Timing: CUDA events on each measured stream; LibTorch row uses CUDA graph
  replay, custom rows use direct CUDA launches.
- Warmup/iterations: `5` warmups, `10` timed iterations per sample, `3`
  samples. This is a port smoke run, not a final tuning run.
- Cache policy: warm repeated buffers.
- Clock policy: clocks not locked.
- Correctness: LibTorch row checks selected-row gather and scale; custom fused
  row checks token against the materialized CUDA reference before timing.

Results:

```text
variant=native_pytorch_cuda_graph median_us=2853.350 samples_us=[2852.541,2853.350,2854.547]
variant=fused_lm_head_sample_full_vocab median_us=2837.990 samples_us=[2836.016,2838.429,2837.990]
variant=materialized_lm_head_sample_full_vocab median_us=2871.904 samples_us=[2871.606,2871.904,2872.211]
```

Conclusion:

- The sampling mechanism now has one C++/CUDA benchmark file with its LibTorch
  baseline included.
- The old Python wrapper behavior is preserved by the `native_pytorch_cuda_graph`
  row and named `--warmup`, `--iters`, and `--samples` aliases.
- Rerun with larger sample counts before making performance claims.

## 2026-06-25 - KV Cache Bench LibTorch SDPA Baseline Port

Question:

- Can the old Python `gemma4_kv_cache_torch_bench.py` SDPA timing live inside
  the KV-cache mechanism benchmark without leaving a second benchmark file?

Change:

- Added LibTorch eager SDPA rows to `gemma4_kv_cache_bench`.
- Preserved the Python baseline knobs: `--seq-len`, `--q-heads`,
  `--kv-heads`, `--head-dim`, and `--flush-mib`.
- Kept the existing custom sliding paged-cache rows in the same benchmark.
- Added a custom global paged-decode row using
  `gemma4_flash_attention_decode_paged_bf16`.
- Fixed the custom CPU reference to compare only the live sliding window when
  `seq_len` exceeds the sliding attention window.
- Deleted `gemma4_kv_cache_torch_bench.py`.
- Deleted `gemma4_global_decode_torch_bench.py`.
- Deleted the decode half of `gemma4_paged_decode_torch_bench.py`; its sliding
  LibTorch decode row now lives here.

Commands:

```bash
make kv-cache-bench
./build/benches/gemma4_kv_cache_bench --warmup 5 --iters 10 --samples 3
make test-kv-cache
git diff --check
```

Contract:

- Hardware: NVIDIA RTX A6000, PCI bus `00000000:04:00.0`, driver
  `580.126.16`, persistence enabled, ECC disabled, power limit `300 W`.
- Toolchain: `/usr/local/cuda/bin/nvcc`, CUDA `13.0.48`.
- Library baseline: LibTorch `2.11.0`.
- Torch shape: BF16 `q=[1,16,1,512]`, `k/v=[1,1,4096,512]`, non-causal
  SDPA with GQA enabled and scale `1/sqrt(512)`.
- Custom shape: BF16 sliding paged cache, `q_heads=16`, `kv_heads=8`,
  `head_dim=256`, `key_count=1024`, `split_size=20`.
- Sliding Torch shape: BF16 `q=[1,16,1,256]`, `k/v=[1,8,1024,256]`,
  non-causal SDPA with GQA enabled and scale `1/sqrt(256)`.
- Global custom shape: BF16 global paged cache, `q_heads=16`, `kv_heads=1`,
  `head_dim=512`, `seq_len=4096`, `split_size=64`.
- Timing: CUDA events on the same stream as the measured work. LibTorch first
  use is outside timing through the checksum call and warmups.
- Warmup/iterations: `5` warmups, `10` timed iterations per sample, `3`
  samples. This is a port smoke run, not a final tuning run.
- Cache policy: warm repeated buffers.
- Clock policy: clocks not locked; run snapshot showed idle clocks at
  `210 MHz` SM and `405 MHz` memory.
- Correctness: custom flash decode matched the CPU reference with
  `max_abs=0` and `mean_abs=0`; `make test-kv-cache` passed.

Results:

```text
torch_attention_only median_ms=1.664547 samples_ms=[1.646912,1.664547,1.674886]
torch_full_decode_write_plus_attention median_ms=1.663427 samples_ms=[1.652682,1.663427,1.668630]
torch_sliding_decode_attention median_ms=0.183338 samples_ms=[0.167078,0.183363,0.183338]
prefill_cache_write median_ms=0.106074 samples_ms=[0.104499,0.106435,0.106074]
decode_cache_write median_ms=0.015171 samples_ms=[0.013722,0.017741,0.015171]
flash_decode_paged_attention_direct median_ms=0.035075 samples_ms=[0.032403,0.038118,0.035075]
flash_full_decode_write_plus_attention median_ms=0.047910 samples_ms=[0.047456,0.049424,0.047910]
global_decode_paged_attention_direct median_ms=0.266893 samples_ms=[0.265245,0.266893,0.266954]
```

Conclusion:

- The KV-cache mechanism now has one C++/CUDA benchmark file with its LibTorch
  SDPA baseline and custom global paged-decode row included.
- The Torch rows preserve the old Python default global-like SDPA shape; they
  are useful as a library baseline, not a shape-identical comparison to the
  sliding custom rows.
- Rerun with larger sample counts and locked clocks before making performance
  claims.

## 2026-06-25 - Flash Attention Bench LibTorch Prep Ports

Question:

- Can the prefill half of `gemma4_paged_decode_torch_bench.py` live inside the
  flash-attention mechanism benchmark?
- Can `gemma4_decode_prep_torch_bench.py` be represented by a LibTorch
  decode-prep row in the same mechanism benchmark?
- Can `gemma4_project_prepare_compare.py` be represented by LibTorch and custom
  project-prepare rows in the same mechanism benchmark?

Change:

- Added a raw `gemma4_flash_attention_sliding_fwd_bf16` row with LSE output to
  `gemma4_flash_attention_bench`.
- Added a LibTorch BF16 SDPA prefill row for `seq_len <= GEMMA4_SLIDING_WINDOW`.
- Added a LibTorch decode RMSNorm/RoPE/paged-KV-write row.
- Added LibTorch and custom packed project-plus-prepare rows.
- Kept the existing `norm_rope_plus_fa` and decode-prep rows distinct.
- Deleted `gemma4_paged_decode_torch_bench.py` after its decode row landed in
  the KV-cache bench and its prefill row landed here.
- Deleted `gemma4_decode_prep_torch_bench.py`.
- Deleted `gemma4_project_prepare_compare.py`; the optional vLLM operator row
  was not carried into C++.

Commands:

```bash
make flash-attn-bench
./build/benches/gemma4_flash_attention_bench 1024 10 5 3 1 64 warm 64
git diff --check
```

Contract:

- Hardware: NVIDIA RTX A6000, PCI bus `00000000:09:00.0`, driver
  `580.126.16`, persistence enabled, ECC disabled, power limit `300 W`.
- Toolchain: `/usr/local/cuda/bin/nvcc`, CUDA `13.0.48`.
- Library baseline: LibTorch `2.11.0`.
- Shape: BF16 `q=[1,1024,16,256]`, `k/v=[1,1024,8,256]`, causal prefill,
  window `1024`, scale `1/sqrt(256)`.
- Decode-prep shape: BF16 `q=[1,16,256]`, `k/v=[1,8,256]`, weights `[256]`,
  and paged cache `[1,16,64,8,256]`.
- Project-prepare shape: BF16 hidden `[1,3840]`, packed QKV weight
  `[8192,3840]`, output Q `[1,16,256]`, and paged cache `[1,16,64,8,256]`.
- Timing: CUDA events on the same stream as the measured work.
- Warmup/iterations: `5` warmups, `10` timed iterations per sample, `3`
  samples. This is a port smoke run, not a final tuning run.
- Cache policy: warm repeated buffers.
- Clock policy: clocks not locked.
- Correctness: existing `check_seq=64` CPU-reference checks passed with
  `max_abs=0.0078125`; norm/RoPE prep max abs was at most `0.00195312`.

Results:

```text
prefill_torch_sdpa_graphless median_ms=0.161366 samples_ms=[0.156435,0.161366,0.171440]
torch_decode_norm_rope_paged_kv_write median_ms=2.418710 samples_ms=[2.091730,2.502110,2.418710]
torch_project_prepare median_ms=2.394480 samples_ms=[3.446750,2.394480,2.313800]
custom_project_prepare median_ms=0.495533 samples_ms=[0.495533,0.493709,0.495968]
sliding_fwd_bf16_return_lse median_ms=0.134902 samples_ms=[0.134262,0.135312,0.134902]
norm_rope_plus_fa median_ms=0.189395 samples_ms=[0.188262,0.189395,0.190106]
decode_norm_rope_paged_kv_write median_ms=0.020592 samples_ms=[0.015309,0.030259,0.020592]
```

Conclusion:

- The paged-decode wrapper's sliding prefill comparison and the decode-prep
  wrapper's LibTorch baseline now have C++ rows in the flash-attention
  mechanism bench.
- The project-prepare wrapper's PyTorch/custom core comparison also has C++
  rows; the optional vLLM row was intentionally omitted because it depends on a
  Python package plugin rather than LibTorch.
- The LibTorch row is a baseline, not a graph-captured replica of the old
  Python harness.
- Rerun with larger sample counts and locked clocks before making performance
  claims.

## 2026-06-25 - Retired Remaining Python Bench Wrappers

Question:

- After the Torch benchmark paths moved into C++ mechanism benches, what should
  happen to the remaining Python files under `src/benches/`?

Change:

- Deleted `gemma4_flash_attention_compare.py`.
- Deleted `gemma4_tokenizer_compare.py`.
- Deleted `gemma4_prefill_tune.py`.
- Deleted the now-unused shared Python `gemma4_bench_utils.py`.
- Made `tests/test_flash_attention_pytorch.py` self-contained instead of
  importing helper functions from the deleted paged-decode benchmark.

Notes:

- The flash-attention C++ bench now carries raw custom sliding attention,
  LibTorch SDPA prefill, decode-prep, project-prepare, and CPU-reference
  correctness rows. The deleted Python wrapper's optional Python `flash_attn`
  comparison was not carried into C++.
- The tokenizer C++ bench keeps the custom tokenizer benchmark. The deleted
  Python wrapper's external Hugging Face `transformers` and Rust `tokenizers`
  comparison rows were not carried into C++.
- The prefill tuner was already orphaned from the current `Makefile`; its old
  Tuna/SGEMM sweep orchestration was retired instead of being moved to
  LibTorch.

Conclusion:

- `src/benches/` now contains no Python benchmark files. The remaining Python in
  this repo is tests, historical experiment material, and non-bench tooling.

## 2026-06-25 - Prompt Decode Tail Split and Hot-Path Timing

Question:

- Does `gemma4_prompt` avoid running final RMSNorm + LM-head sampling on
  intermediate decode layers, while preserving the sampled token produced by the
  existing monolithic tail?
- What are first-pass real-weight hot-path timings for the current prompt flow
  when weight load, tokenizer startup, allocation, and other one-time setup are
  outside the CUDA-event timing window?

Change:

- Added a decode megakernel attention+FFN launcher that stops after the
  post-FFN residual/layer-scalar phase.
- Updated `gemma4_prompt` decode so layers `0..46` use the attention+FFN-only
  launcher and layer `47` alone runs the final RMSNorm + LM-head sampling tail.
- Added `gemma4_prompt --benchmark`, with warmup, iteration, and sample counts,
  to time prefill and decode separately using CUDA events on the same stream.

Commands:

```bash
make test-decode-megakernel
make prompt
./build/gemma4_prompt --benchmark --bench-warmup 1 --bench-iters 1 \
  --bench-samples 1 --max-new 1 --prompt Hello
nvidia-smi --query-gpu=name,gpu_bus_id,driver_version,persistence_mode,ecc.mode.current,mig.mode.current,power.limit,clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu --format=csv,noheader,nounits
/usr/local/cuda/bin/nvcc --version
```

Contract:

- Hardware: NVIDIA RTX A6000, PCI bus `00000000:04:00.0`, driver
  `580.126.16`, persistence enabled, ECC disabled, power limit `300 W`.
- Toolchain: `/usr/local/cuda/bin/nvcc`, CUDA `13.0.48`.
- Model/input: real `models/gemma-4-12B/model.safetensors`, tokenizer
  `models/gemma-4-12B/tokenizer.json`, prompt `Hello`, prompt length `1`.
- Timing: CUDA events on stream `0`; weight loading, tokenizer startup,
  allocation, runtime initialization, and decode setup prefill are outside the
  measured regions.
- Warmup/iterations: `1` warmup, `1` timed iteration, `1` sample. This is a
  smoke benchmark, not a stable tuning run.
- Cache policy: prefill repeats the same prompt buffers; decode advances
  statefully after a setup prefill.
- Clock policy: clocks not locked.
- Correctness: `test_decode_megakernel` compares the split attention+FFN path
  plus standalone final spine against the existing monolithic tail token.

Results:

```text
prompt tokens: 1
benchmark prompt_len=1 warmup=1 iters=1 samples=1 cache=prefill_repeated_decode_stateful timing=cuda_events_same_stream
prefill_ms median=37.905 min=37.905 max=37.905
decode_ms median=70.404 min=70.404 max=70.404
```

Conclusion:

- The P1 prompt decode bug is fixed at the caller: intermediate layers no longer
  launch the final sampling tail.
- The focused smoke test passed, showing the split path preserves the generated
  token for the deterministic FlashAttention+FFN case.
- The benchmark harness is intentionally minimal and should be rerun with larger
  warmup/sample counts before making performance claims.

## 2026-06-23 - FFN Gate/Up DualGemm Tile Sweep

Question:

- Is the production CUTLASS `DualGemm` tile for the 12B FFN gate/up prefill
  path still the best measured choice across prompt row counts?

Change:

- Added `gemma4_ffn_dual_gemm_bench` to sweep the custom gate/up + GeGLU
  `DualGemm` tile variants separately from the FFN-down GEMM.
- Increased benchmark input scales after the first run exposed false-positive
  "fast" templates that were effectively writing zeros under the old loose
  absolute tolerance.
- Updated production prefill gate/up dispatch to:
  - rows `<= 64`: `64x64x32`, warp `64x32`, stages `3`.
  - rows `<= 128`: `128x64x32`, warp `64x32`, stages `5`.
  - rows `> 128`: `256x64x32`, warp `64x32`, stages `3`.

Commands:

```bash
make ffn-dual-gemm-bench
./build/experiments/gemma4_ffn_dual_gemm_bench 3 1 1 64 all
./build/experiments/gemma4_ffn_dual_gemm_bench 20 10 3 16,64,96,128,256,512,1024 all \
  | tee src/experiments/results/2026-06-23_ffn_dual_gemm_12b_warm.txt
./build/experiments/gemma4_ffn_dual_gemm_bench 50 20 5 16,64,96,128 all \
  | tee src/experiments/results/2026-06-23_ffn_dual_gemm_small_confirm.txt
./build/experiments/gemma4_ffn_dual_gemm_bench 50 20 5 256,512,1024 all \
  | tee src/experiments/results/2026-06-23_ffn_dual_gemm_tail_confirm.txt
./build/experiments/gemma4_ffn_dual_gemm_bench 20 10 3 2048,4096 all \
  | tee src/experiments/results/2026-06-23_ffn_dual_gemm_large_probe.txt
make build/gemma4_ffn.o
make test-ffn-decode test-prefill-megakernel
```

Contract:

- Hardware: NVIDIA RTX A6000, driver `580.126.16`, persistence enabled, ECC
  disabled, power limit `300 W`. Benchmark process PCI bus IDs varied in logs
  even though `nvidia-smi -L` reported one visible A6000.
- Toolchain: `/usr/local/cuda/bin/nvcc`, CUDA `13.0.48`.
- Shape: BF16 FFN gate/up `DualGemm`, `M = rows`, `N = 15360`,
  `K = 3840`, with fused tanh-GELU times up.
- Timing: CUDA events on the same nonblocking stream over repeated launches;
  launch overhead excluded from GPU elapsed time.
- Warmup/iterations: main sweep used `10` warmups, `20` timed iterations, `3`
  samples. Confirmation sweeps used `20` warmups, `50` timed iterations, `5`
  samples where noted.
- Cache policy: warm repeated buffers.
- Clock policy: clocks not locked.
- Correctness: candidate output compared against the current production tile
  before timing. Wrong zero-output candidates failed with max_abs around
  `266-396` after the scale fix. Focused tests passed after the production
  dispatch edit.

Best corrected timings:

```text
rows  selected production tile   confirm best ms  speedup vs current
16    64x64x32 s3                   0.345942 ms          1.096x
64    64x64x32 s3                   0.351373 ms          1.066x
96    128x64x32 s5                  0.355635 ms          1.055x
128   128x64x32 s5                  0.358936 ms          1.054x
256   256x64x32 s3                  0.473920 ms          1.058x
512   256x64x32 s3                  0.991025 ms          1.034x
1024  256x64x32 s3                  1.997306 ms          1.012x
2048  256x64x32 s3                  3.823846 ms          1.079x
4096  256x64x32 s3                  7.713193 ms          1.419x
```

Conclusion:

- The old single `128x64x32 s3` production tile was not the best measured
  choice for most row counts.
- The kept dispatch is deliberately small: two new useful tile families plus
  the measured row thresholds.
- Several tempting wide-N templates appeared extremely fast only because they
  produced wrong near-zero outputs; they remain benchmark-only rejected rows.

## 2026-06-23 - CUTLASS 12B Exact Prefill Dispatch Sweep

Question:

- Are the production CUTLASS prefill GEMMs still using the best measured tile for
  each Gemma 4 12B projection shape?

Change:

- Added exact `sliding_q` and `sliding_kv` shapes to the prefill tuner so the
  benchmark matches the current unfused prefill path instead of relying on the
  older packed-QKV proxy shape.
- Updated production CUTLASS dispatch for exact 12B prefill projection shapes
  and FFN down. Unknown shapes keep the old generic two-tile fallback.

Commands:

```bash
nvidia-smi --query-gpu=name,gpu_bus_id,driver_version,persistence_mode,ecc.mode.current,mig.mode.current,power.limit,clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu --format=csv
/usr/local/cuda/bin/nvcc --version
make sgemm-bf16-prefill-bench
python3 src/experiments/gemma4_prefill_tune.py \
  --backend sgemm-bf16 \
  --ops ffn_down,sliding_q,sliding_kv,sliding_o,global_q,global_k,global_o \
  --configs bf16_cutlass_64x64_s10,bf16_cutlass_64x128x64,bf16_cutlass_64x128_s6,bf16_cutlass_128x128x64,bf16_cutlass_128x128_s5,bf16_cutlass_128x256,bf16_cutlass_256x128 \
  --m 16,64,96,128,256,512,1024 \
  --iters 20 --warmup 10 \
  --cublas-backend lt --cublaslt-heuristics 32 --graph-repeats 10 \
  --skip-build --keep-going \
  --out build/experiments/gemma4_prefill_tune/2026-06-23_cutlass_12b_exact_prefill_h32.csv
make test-prefill-gemm test-ffn-decode
make test-prefill-megakernel
```

Contract:

- Hardware: NVIDIA RTX A6000, driver 580.126.16, persistence enabled, ECC
  disabled, power limit 300 W.
- Toolchain: `/usr/local/cuda/bin/nvcc` reports CUDA 13.0.48, even though the
  project target notes still say CUDA 12.x.
- Timing: child benchmark CUDA events, with CUDA graph replay when
  `GEMMA4_PREFILL_GRAPH_REPEATS=10`.
- Warmup/iterations: 10 warmup iterations and 20 timed iterations per
  op/config/M point.
- Cache policy: warm repeated buffers.
- Clock policy: clocks not locked.
- Correctness: benchmark checks custom output against the cuBLAS/cuBLASLt
  reference for every config; focused tests passed after the dispatch edit.

Results:

```text
Exact measured shape grid, old production CUTLASS dispatch vs new measured dispatch:

op          old_ms_sum  new_ms_sum  old/new
ffn_down       2.7647      2.6619    1.039x
sliding_q      0.7964      0.7404    1.076x
sliding_kv     0.4703      0.3892    1.208x
sliding_o      0.7674      0.7328    1.047x
global_q       1.5291      1.4574    1.049x
global_k       0.3667      0.1911    1.919x
global_o       1.4875      1.4268    1.043x

Weighted by production layer counts:
old CUTLASS dispatch: 259.9480 ms
new CUTLASS dispatch: 242.4376 ms
old/new speedup:      1.072x
cuBLASLt h32:         255.9424 ms
new vs cuBLASLt h32:  1.056x
```

Conclusion:

- The old row-only `64x128x64` / `128x128x64` CUTLASS rule was leaving
  measurable performance on the table for exact 12B prefill shapes.
- Shape-specific dispatch is worth keeping for the current CUTLASS prefill path.
- Global K still strongly favors cuBLASLt in isolation, so a later unfused
  baseline should compare replacing that specific CUTLASS call with cuBLASLt
  rather than only retuning CUTLASS.
- The FFN gate/up DualGemm tile was intentionally left out of this sweep and
  handled by the follow-up entry above.

## 2026-06-20 - Fused LM-Head Gumbel Sampling Smoke

Question:

- Can the final logits decode path sample from the full vocabulary without
  materializing `[1, vocab]` logits in HBM?

Change:

- Added a decode-only fused full-vocab Gumbel-Max sampler that reuses the
  existing final LM-head GEMV tile shape.
- Each CTA computes its strided vocab tiles, applies final softcap and
  temperature, adds deterministic per-token Gumbel noise, and writes one
  candidate. The cooperative reduction then selects one token and gathers its
  tied embedding row.
- This first pass intentionally does not implement top-k/top-p. The old
  materialized top-k/top-p path was removed in the follow-up cleanup; add a
  fused truncated sampler separately if that distribution is needed again.

Commands:

```bash
make test-sampling NVCC=/usr/local/cuda/bin/nvcc
make sampling-bench NVCC=/usr/local/cuda/bin/nvcc
./build/experiments/gemma4_sampling_bench 5 10 5
/usr/local/cuda/bin/nvcc -std=c++17 -O3 -arch=sm_86 -Isrc -Xptxas=-v \
  -c src/gemma4_sampling.cu -o /tmp/gemma4_sampling_resource.o
```

Contract:

- Hardware: NVIDIA RTX A6000, 48 GB.
- Timing: CUDA events on a nonblocking stream.
- Warmup/iterations/samples: 5 warmup, 10 timed iterations, 5 samples.
- Cache policy: warm-ish repeated buffers.
- Clock policy: clocks not locked.
- Correctness: `make test-sampling` compares the fused Gumbel token against a
  CPU reference over all 262144 token IDs for sparse LM-head fixtures.

Results:

```text
fused_lm_head_gumbel_full_vocab:
  median_us=2964.803 min_us=2956.403 max_us=2972.752

ptxas final_logits_gumbel_fused_kernel:
  registers=64 smem=9216B stack=0 spills=0
```

Conclusion:

- The fused path is correct on the focused tests and avoids writing full logits.
- The smoke benchmark includes the full LM-head GEMV weight stream, so it is not
  directly comparable to the existing materialized-logits sampler rows in the
  same bench, which start after logits already exist.
- Next useful comparison is an apples-to-apples baseline:
  materialized final LM-head GEMV plus a materialized full-vocab Gumbel-Max
  sampler.

## 2026-06-20 - Fused vs Materialized LM-Head Gumbel Benchmark

Question:

- How much faster is the fused full-vocab Gumbel sampler than a materialized
  final-logits path?

Change:

- Added a benchmark-local baseline:
  `final logits GEMV -> write [1, vocab] BF16 logits -> full-vocab Gumbel-Max
  sampler -> gather tied embedding row`.
- The benchmark checks that fused and materialized paths select the same token
  before timing.

Commands:

```bash
nvidia-smi --query-gpu=name,gpu_bus_id,driver_version,persistence_mode,ecc.mode.current,mig.mode.current,power.limit,clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu --format=csv
/usr/local/cuda/bin/nvcc --version
make sampling-bench NVCC=/usr/local/cuda/bin/nvcc
./build/experiments/gemma4_sampling_bench 25 50 15
./build/experiments/gemma4_sampling_bench 25 50 15
```

Contract:

- Hardware: NVIDIA RTX A6000, 48 GB.
- Driver: `580.126.16`.
- CUDA/NVCC: CUDA `13.0`, NVCC `13.0.48`.
- Timing: CUDA events on a nonblocking stream.
- Warmup/iterations/samples: 25 warmup, 50 timed iterations, 15 samples.
- Cache policy: warm-ish repeated buffers.
- Clock policy: clocks not locked. Persistence mode was enabled.
- Correctness: fused/materialized token IDs matched before timed runs.

Results:

```text
run 1:
  fused median_us=2867.341 mean_us=2868.309
  materialized median_us=2972.001 mean_us=2972.224
  delta median_us=104.660
  fused speedup=1.0365x, latency reduction=3.52%

run 2:
  fused median_us=2870.649 mean_us=2870.228
  materialized median_us=2972.137 mean_us=2972.007
  delta median_us=101.488
  fused speedup=1.0354x, latency reduction=3.42%

average of run medians:
  fused median_us=2868.995
  materialized median_us=2972.069
  delta median_us=103.074
  fused speedup=1.0359x, latency reduction=3.47%
```

Conclusion:

- The fused path is about `1.036x` faster on this A6000 warm-cache decode
  microbenchmark, saving about `103 us/token` in the final LM-head sampling
  tail.
- This is a modest but repeatable win. It mainly removes the logits HBM
  write/read and the separate sampler launch, while the unavoidable LM-head
  weight stream still dominates the runtime.

## 2026-06-20 - Sliding Decode Split-Size Retune to 20

Question:

- After rejecting the combined score reducer, retest whether the retained
  split/reduce kernel wants a smaller split size than the previously selected
  `32`.
- Use the same full-window sliding decode shape: `seq_len=1024`,
  `page_size=64`, BF16, `batch_size=1`, `q_heads=16`, `kv_heads=8`,
  `head_dim=256`.

Change:

- Changed `GEMMA4_SLIDING_DECODE_SPLIT_SIZE` from `32` to `20`.
- Updated the Python sliding decode comparison default to the same value.
- No flash-attn kernel body change was kept.

Commands:

```bash
/usr/local/cuda/bin/ncu --version
/usr/local/cuda/bin/nvcc -std=c++17 -O3 -arch=sm_86 \
  --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math \
  -D_GLIBCXX_USE_CXX11_ABI=1 -Isrc \
  -Iexperiments/flash-attention/csrc/cutlass/include -Xptxas=-v \
  -c src/gemma4_flash_attention.cu \
  -o /tmp/gemma4_flash_attention_resource.o
./build/experiments/gemma4_kv_cache_bench 1024 64 24 50 200 20 --cache warm
./build/experiments/gemma4_kv_cache_bench 1024 64 24 20 30 10 --cache cold
./build/experiments/gemma4_kv_cache_bench 1024 64 20 50 200 20 --cache warm
./build/experiments/gemma4_kv_cache_bench 1024 64 28 50 200 20 --cache warm
./build/experiments/gemma4_kv_cache_bench 1024 64 22 50 200 20 --cache warm
./build/experiments/gemma4_kv_cache_bench 1024 64 20 20 30 10 --cache cold
./build/experiments/gemma4_kv_cache_bench 1024 64 18 50 200 20 --cache warm
./build/experiments/gemma4_kv_cache_bench 1024 64 16 50 200 20 --cache warm
make kv-cache-bench NVCC=/usr/local/cuda/bin/nvcc
./build/experiments/gemma4_kv_cache_bench 1024 64
```

Profiler/resource notes:

- `/usr/local/cuda/bin/ncu` exists as a launcher, but Nsight Compute is not
  installed under the expected CUDA/Opt directories, so hardware counters were
  not available.
- ptxas resource output for the retained decode kernels:
  - sliding split: `38` registers, `96` B shared memory, no spills.
  - sliding reduce: `26` registers, `96` B shared memory, no spills.
  - global split: `64` registers, `1408` B shared memory, no spills.
  - global reduce: `28` registers, `176` B shared memory, no spills.

Results:

```text
split_size  actual_splits  cache  median_ms  correctness
32          32             warm   0.036211   max_abs=0
32          32             cold   0.041970   max_abs=0
24          43             warm   0.031740   max_abs=0
24          43             cold   0.038083   max_abs=0
20          52             warm   0.031529   max_abs=0
20          52             cold   0.035987   max_abs=0
28          37             warm   0.033374   max_abs=0
22          47             warm   0.031456   max_abs=0, noisy tail
18          57             warm   0.032329   max_abs=0, bimodal
16          64             warm   0.037753   max_abs=0

default constant check after edit:
  split_size=20 actual_splits=52 warm median_ms=0.033236 max_abs=0
```

Conclusion:

- The best retained policy from this sweep is `split_size=20`.
- It improves direct attention median versus `32` by about `13%` warm-cache and
  about `14%` cold-cache in this harness.
- `16` has already turned over, and `18`/`22` are noisier without a meaningful
  median win. `20` is the cleaner default among the faster small-split choices.
- The tradeoff is larger partial scratch: `52` live splits instead of `32` for
  a 1024-token sliding window. For batch-1 sliding decode, `partial_acc` rises
  from about `512 KiB` to about `832 KiB`, which is acceptable for this
  throughput win.

## 2026-06-20 - Decode Split Combined Score Reduction Rejected

Question:

- The decode split kernel computes one QK dot-product score per GQA head for
  every key. The existing code uses one CUB block reduction per score.
- Test whether reducing all GQA scores together with one custom warp/block
  reduction lowers overhead in the sliding decode hot path.

Tested change:

- Temporarily replaced the per-score CUB reductions in
  `phase_decode_paged_grouped_split` with a combined shuffle/shared-memory
  reduction over `Derived::kGqaRatio` scores.
- This source change was reverted after measurement.

Commands:

```bash
make test-kv-cache NVCC=/usr/local/cuda/bin/nvcc
make kv-cache-bench NVCC=/usr/local/cuda/bin/nvcc
./build/experiments/gemma4_kv_cache_bench 1024 64 32 50 200 20 --cache warm
./build/experiments/gemma4_kv_cache_bench 1024 64 32 20 30 10 --cache cold
make kv-cache-bench NVCC=/usr/local/cuda/bin/nvcc
./build/experiments/gemma4_kv_cache_bench 1024 64 32 50 200 20 --cache warm
./build/experiments/gemma4_kv_cache_bench 1024 64 32 20 30 10 --cache cold
```

Results:

```text
temporary combined score reduction:
  test-kv-cache: passed
  warm median_ms=0.038612 max_abs=0
  cold median_ms=0.045520 max_abs=0

retained CUB score reductions after revert:
  warm median_ms=0.036211 max_abs=0
  cold median_ms=0.041970 max_abs=0
```

Conclusion:

- The custom combined score reducer was correct but slower in both warm and
  cold cache runs, so it was rejected.
- The current CUB score reductions should stay until profiling identifies a
  more specific issue. The next flash-attn work should not chase this path.

## 2026-06-20 - Sliding Decode Split-Size Policy

Question:

- The flash decode kernel body already has the right split/reduce structure for
  short-context direct output and full-window sliding decode.
- The remaining policy question is how much key-range parallelism to expose for
  the 1024-token sliding window before partial scratch/reduce overhead wins.

Change:

- Changed the default sliding decode split size from `64` to `32` in:
  - `src/gemma4.h`
  - `src/experiments/gemma4_kv_cache_bench.cu`
  - `src/experiments/gemma4_paged_decode_torch_bench.py`
- The flash-attn kernel body was not changed. Production callers should use
  `split_size=32` and allocate `ceil(window_size / 32)` split scratch for
  sliding decode unless a later end-to-end benchmark says otherwise.

Commands:

```bash
make kv-cache-bench NVCC=/usr/local/cuda/bin/nvcc
./build/experiments/gemma4_kv_cache_bench 1024 64 32 50 200 20 --cache warm
./build/experiments/gemma4_kv_cache_bench 1024 64 64 50 200 20 --cache warm
./build/experiments/gemma4_kv_cache_bench 1024 64 32 20 30 10 --cache cold
./build/experiments/gemma4_kv_cache_bench 1024 64 64 20 30 10 --cache cold
./build/experiments/gemma4_kv_cache_bench 1024 64 16 20 100 10 --cache warm
./build/experiments/gemma4_kv_cache_bench 1024 64 1024 20 100 10 --cache warm
./build/experiments/gemma4_kv_cache_bench 1024 64 24 20 100 10 --cache warm
./build/experiments/gemma4_kv_cache_bench
./build/experiments/gemma4_kv_cache_bench 1024 64
python3 -m py_compile src/experiments/gemma4_paged_decode_torch_bench.py
make kv-cache-bench NVCC=/usr/local/cuda/bin/nvcc
./build/experiments/gemma4_kv_cache_bench
./build/experiments/gemma4_kv_cache_bench 1024 64
```

Contract:

- Hardware: NVIDIA RTX A6000, CUDA driver/runtime `13000` / `13000`.
- Timing: CUDA events on the benchmark stream.
- Cache: warm and cold C++ `kv-cache-bench` runs. Cold runs flushed 64 MiB.
- Clock policy: clocks were not locked.

Results:

```text
seq=1024 page=64 warm:
  split_size=32 actual_splits=32 median_ms=0.036417 max_abs=0
  split_size=64 actual_splits=16 median_ms=0.055714 max_abs=0

seq=1024 page=64 cold:
  split_size=32 actual_splits=32 median_ms=0.042239 max_abs=0
  split_size=64 actual_splits=16 median_ms=0.062973 max_abs=0

additional warm probes:
  split_size=16   actual_splits=64 median_ms=0.046912 max_abs=0
  split_size=24   actual_splits=43 median_ms=0.035147 max_abs=0
  split_size=1024 actual_splits=1  median_ms=0.754711 max_abs=0

default C++ harness after the edit:
  ./build/experiments/gemma4_kv_cache_bench
    split_size=32, seq_len=4096, key_count=1024, median_ms=0.036184
  ./build/experiments/gemma4_kv_cache_bench 1024 64
    split_size=32, correctness max_abs=0

constant-wired verification after moving the policy to gemma4.h:
  ./build/experiments/gemma4_kv_cache_bench
    split_size=32, median_ms=0.037165
  ./build/experiments/gemma4_kv_cache_bench 1024 64
    split_size=32, correctness max_abs=0, median_ms=0.038498
```

Conclusion:

- The original `64`-token split policy was under-parallelizing full-window
  sliding decode on A6000. Moving to `32` gives about `35%` lower warm median
  and about `33%` lower cold median in the focused benchmark.
- A single 1024-token split is not viable despite avoiding partial scratch; it
  removes split-level parallelism and measured about `0.755 ms`.
- The `24`-token probe was slightly faster than `32` in one short warm run, but
  the margin was below the project's `5%` claim threshold and the split crosses
  page boundaries. `32` is the cleaner default.

## 2026-06-20 - Small-Split Paged Decode Reduce Fast Path

Question:

- The normal sliding decode path with `seq_len=1024`, `split_size=64` launches
  `16` split CTAs, writes partial softmax state/accumulators to global memory,
  then launches the reducer.
- Test whether the reducer itself is a material source of overhead by replacing
  the generic two-CUB-reduction scalar combine path with a small-split branch
  for `reduce_splits <= 32`.

Tested change:

- Added a small-split branch inside `phase_decode_paged_reduce` in
  `src/gemma4_flash_attention.cu`.
- One thread computes the row max and row denominator across live splits; the
  head-dimension threads then combine the per-dimension partial accumulators.
- The existing generic reducer remains the fallback for larger split counts.
- This source change was not kept after measurement.

Commands:

```bash
make build/libgemma4_flash_attention.so NVCC=/usr/local/cuda/bin/nvcc
make kv-cache-bench NVCC=/usr/local/cuda/bin/nvcc
make test-kv-cache NVCC=/usr/local/cuda/bin/nvcc
./build/experiments/gemma4_kv_cache_bench 1024 64 64 20 100 10 --cache warm
./build/experiments/gemma4_kv_cache_bench 64 64 1024 20 100 10 --cache warm
python3 src/experiments/gemma4_global_decode_torch_bench.py \
  --seq-len 64 --batch-size 1 --page-size 64 --split-size 1024 \
  --warmup 3 --iters 3 --samples 3 --cache cold
```

Contract:

- Hardware: NVIDIA RTX A6000, CUDA driver/runtime `13000` / `13000`.
- Timing: CUDA events on the benchmark stream.
- Cache: warm for C++ `kv-cache-bench`; cold for the global PyTorch graph
  harness because that harness only exposes cold mode.
- Warmup/timing: C++ runs used `20` warmups, `100` iterations per sample, `10`
  samples. Global Python run used `3` warmups, `3` iterations, `3` samples.

Results:

```text
test-kv-cache: passed

sliding seq=1024 split=64 actual_splits=16:
  correctness max_abs=0 mean_abs=0
  flash_decode_paged_attention_direct median_ms=0.059416
  prior median from single-split direct-output pass=0.059571

sliding seq=64 split=1024 actual_splits=1:
  correctness max_abs=0 mean_abs=0
  flash_decode_paged_attention_direct median_ms=0.039984

global seq=64 split=1024:
  custom_vs_pytorch max_abs=0.00048828
  custom global decode median_ms=0.310805
```

Conclusion:

- The small-split reducer branch was correct and neutral/slightly faster in this
  short run, but the effect was below the project's `5%` claim threshold, so
  the source change was reverted.
- This confirms the reducer scalar work is not the main inefficiency. The real
  issue is still the multi-split design for full-window sliding decode: it
  writes `partial_m`, `partial_l`, and `partial_acc` to HBM, then reads them
  back in a second kernel.
- The next meaningful flash-attn optimization should avoid the partial
  accumulator round trip for the common full-window sliding decode case, rather
  than further tuning this reducer.

## 2026-06-20 - Single-Split Paged Decode Direct Output

Question:

- Remove unnecessary paged decode split/reduce work when a decode row has only
  one launched split.
- Keep the existing multi-split path unchanged for normal sliding-window decode.

Change:

- Added an optional `direct_out` sink to `phase_decode_paged_grouped_split` in
  `src/gemma4_flash_attention.cu`.
- `launch_decode_paged_impl` now passes `d_out` and skips
  `decode_paged_reduce_kernel` when `num_splits == 1`.
- Updated `gemma4_kv_cache_bench` to skip the parked persistent decode variant
  when it returns `cudaErrorNotSupported`, so direct decode timing still runs.

Build and correctness:

```bash
make build/libgemma4_flash_attention.so NVCC=/usr/local/cuda/bin/nvcc
make kv-cache-bench NVCC=/usr/local/cuda/bin/nvcc
make test-kv-cache NVCC=/usr/local/cuda/bin/nvcc
```

Correctness status:

- `test-kv-cache`: passed.
- Sliding single-split decode against CPU reference:
  `max_abs=0.000000 mean_abs=0.000000`.
- Sliding multi-split decode against CPU reference:
  `max_abs=0.000000 mean_abs=0.000000`.
- Global single-split decode against PyTorch:
  `custom_vs_pytorch max_abs=0.00048828`.

Benchmark contract:

- Hardware: NVIDIA RTX A6000, CUDA driver/runtime `13000` / `13000`.
- Timing: CUDA events on the benchmark stream.
- Cache: warm for C++ `kv-cache-bench`; cold for the global PyTorch graph
  harness because that harness only exposes cold mode.
- Warmup/timing: C++ runs used `20` warmups, `100` iterations per sample, `10`
  samples. Global Python run used `3` warmups, `3` iterations, `3` samples.

Benchmark commands:

```bash
./build/experiments/gemma4_kv_cache_bench 64 64 1024 20 100 10 --cache warm
./build/experiments/gemma4_kv_cache_bench 64 64 1024 20 100 10 --cache warm --extra-splits 1
./build/experiments/gemma4_kv_cache_bench 1024 64 64 20 100 10 --cache warm

python3 src/experiments/gemma4_global_decode_torch_bench.py \
  --seq-len 64 --batch-size 1 --page-size 64 --split-size 1024 \
  --warmup 3 --iters 3 --samples 3 --cache cold
```

Results:

```text
case                                      median_ms
sliding single split, direct output       0.040159
sliding one live split, forced reduce     0.043414
sliding 1024-token, 16-split fallback     0.059571
global 64-token, single split, cold       0.310869
```

Conclusion:

- The single-split direct-output path removes the reduce launch and partial
  scratch round-trip for short-context decode and measured about `7.5%` faster
  than the forced split/reduce comparison in this run.
- The normal 16-split sliding fallback still passes correctness and remains in
  the expected performance range.
- This does not solve full-window decode inefficiency. The next larger flash
  attention step is a measured single-kernel split+reduce design or a
  sliding/global-specific split schedule that reduces partial-scratch traffic.

## 2026-06-20 - Flash-Attention Kernel Wrapper Refactor Check

Question:

- Refactor the active flash-attn CUDA globals so they are thin wrappers around
  device phase helpers, without changing the math, launch shapes, cache layout,
  or fused attention behavior.
- Confirm correctness and latency stayed effectively unchanged.

Change:

- Extracted phase helpers from the prefill Q/K/V prep, decode Q/K/V prep,
  fused decode project+prep, paged decode split, and paged decode reduce
  kernels in `src/gemma4_flash_attention.cu`.
- Left `gemma4_flash_fwd_bf16_kernel` on the existing wrapper path around
  `gemma4_compute_attn_1rowblock`, since the real FlashAttention work was
  already below the global wrapper.
- Did not touch the unsupported sliding persistent decode path.

Build:

```bash
make -B flash-attn-bench NVCC=/usr/local/cuda/bin/nvcc
make -B kv-cache-bench NVCC=/usr/local/cuda/bin/nvcc
make -B flash-attn-lib NVCC=/usr/local/cuda/bin/nvcc
make -B build/gemma4_flash_attention.o NVCC=/usr/local/cuda/bin/nvcc
```

Environment snapshot:

- GPU: NVIDIA RTX A6000, bus `00000000:06:00.0`.
- Driver: `580.126.16`; persistence enabled; ECC disabled; MIG disabled.
- Power limit: `300 W`; initial clocks `210 MHz` SM / `405 MHz` memory.
- Clock policy: not locked.

Correctness:

```text
flash-attn-bench:
  correctness seq=64 max_abs=0.0078125 mean_abs=0.000258471 max_rel=0.00775194
  no_lse_correctness max_abs=0.0078125 mean_abs=0.000258471 max_rel=0.00775194
  norm_rope_prep_correctness q_max_abs=0.00195312 k_max_abs=0 v_max_abs=0.000976562

sliding paged decode torch graph:
  decode_direct_vs_torch_max_abs=0.000244140625
  prefill_custom_vs_torch_max_abs=0.000244140625

global paged decode torch graph:
  custom_vs_pytorch max_abs=0.00012207
```

Benchmark commands:

```bash
./build/experiments/gemma4_flash_attention_bench 1024 200 50 5 1 64 warm 64

python3 src/experiments/gemma4_paged_decode_torch_bench.py \
  --lib build/libgemma4_flash_attention.so --seq-len 1024 \
  --prefill-seq-len 64 --batch-size 1 --page-size 64 --split-size 64 \
  --warmup 10 --iters 20 --samples 5 --cache warm --sample-delay-s 1.0

python3 src/experiments/gemma4_global_decode_torch_bench.py \
  --seq-len 1024 --batch-size 1 --page-size 64 --split-size 64 \
  --warmup 5 --iters 5 --samples 5 --cache cold
```

Measurement contract:

- `flash-attn-bench`: CUDA events on the benchmark stream, warm cache,
  launch overhead included, 50 warmups, 200 timed iterations, 5 samples.
- Sliding paged decode torch graph: CUDA graph replay, warm cache, launch
  overhead excluded, 10 warmups, 20 inner iterations, 5 samples.
- Global paged decode torch graph: CUDA graph replay, cold cache, launch
  overhead excluded, 5 warmups, 5 iterations, 5 samples.

Baseline vs refactor, same `flash-attn-bench` command:

```text
path                         baseline median ms   refactor median ms
norm_rope_plus_fa            0.186738             0.186861
decode_norm_rope_paged_kv    0.023846             0.020419
```

Post-refactor decode attention timing:

```text
sliding decode_custom_direct warm median   0.059155 ms
global custom_global_decode cold median    0.278880 ms
```

Conclusion:

- Numerical results match the pre-refactor bench exactly for the covered
  prefill/prep checks, and paged decode split/reduce passes both sliding and
  global PyTorch references.
- Prefill timing is unchanged within noise. Decode prep measured faster than
  the earlier single baseline run, but that path is noisy at this size, so this
  is not claimed as a performance win.
- `kv-cache-bench` rebuilt and confirmed direct decode correctness before
  stopping at the pre-existing unsupported persistent decode entrypoint.

## 2026-06-19 - Probabilistic Sampling From Logits

Question:

- Implement and measure a correct baseline for sampling from already-materialized
  logits, without fusing the final projection into the sampler.
- Preserve the existing fused greedy endpoint.
- Keep the tied embedding gather as a reusable device helper so the selected
  token row can be copied inside the sampling kernel.

Change:

- Added `Gemma4SamplingParams`.
- Added `gemma4_sample_from_logits_scratch_bytes` and
  `gemma4_sample_from_logits_decode_bf16`.
- The new sampler consumes row-major BF16 logits `[B, vocab]`, applies Gemma
  final softcap, temperature, exact top-k for `1..64`, top-p over the top-k
  softmax, stateless SplitMix-style deterministic RNG, and copies the selected
  tied embedding row into `[B, hidden]`.
- Refactored `src/gemma4_embedding_gather.cu` so its kernel calls the shared
  `gemma4_embedding_gather::copy_embedding_row_bf16` device helper. The new
  sampler and the old greedy fused endpoint use the same helper.
- Initial top-k merge used a serial shared-memory insertion pass and measured
  about `10.139 ms` median for `B=1, top_k=64`; replaced that merge with an
  exact block reduction over per-thread sorted candidate lists, reducing the
  same case to about `8.771 ms`.

Build and correctness:

```bash
make -B test-sampling test-embedding-gather sampling-bench \
  NVCC=/usr/local/cuda/bin/nvcc
```

Correctness status:

- `test-sampling`: passed.
- `test-embedding-gather`: passed.
- Benchmark correctness checks: old logits argmax matched fused greedy; all
  probabilistic variants matched the CPU reference for each measured batch.

Benchmark command:

```bash
./build/experiments/gemma4_sampling_bench 25 100 9 32 \
  | tee src/experiments/results/2026-06-19_sampling_from_logits_warm_merge_reduce.txt
```

Resource check:

```bash
/usr/local/cuda/bin/nvcc -std=c++17 -O3 -arch=sm_86 -Isrc \
  --ptxas-options=-v -c src/gemma4_sampling.cu \
  -o /tmp/gemma4_sampling_resource.o 2>&1 \
  | tee src/experiments/results/2026-06-19_sampling_ptxas_resource.txt
```

Measurement contract:

- Hardware: NVIDIA RTX A6000, bus `00000000:0F:00.0`, driver `580.126.16`.
- CUDA driver/runtime reported by benchmark: `13000` / `13000`.
- Clock policy: not locked. `nvidia-smi` snapshot reported persistence enabled,
  ECC disabled, power limit `300 W`, SM clock `1800 MHz`, memory clock
  `8001 MHz`, temperature `27 C`, idle GPU utilization at start.
- Timing: CUDA events on a nonblocking stream; launch overhead included.
- Cache: warm-ish repeated-logits benchmark. Logits are reused across timed
  iterations, so this is not a cold-HBM sampler measurement.
- Shape: BF16 logits `[B, 262144]`, tied embedding rows `[262144, 5376]`.
- Warmup: `25`; timed iterations per sample: `100`; samples: `9`.
- `ncu` was not available on this machine; ptxas resource output was recorded
  instead.

Median results:

```text
variant                         B   top_k  top_p   median_us
final_logits_only               1   -      -       3954.845
old_logits_argmax_embed         1   1      1.00    4009.821
fused_greedy_endpoint           1   1      1.00    3969.395

logits_argmax_embed_baseline    1   1      1.00      57.079
prob_sampler_topk1_topp1        1   1      1.00     258.200
prob_sampler_topk64_topp095     1   64     0.95    8770.779
prob_sampler_topk64_topp1       1   64     1.00    8770.848

logits_argmax_embed_baseline    8   1      1.00      56.983
prob_sampler_topk1_topp1        8   1      1.00     283.112
prob_sampler_topk64_topp095     8   64     0.95    8802.705
prob_sampler_topk64_topp1       8   64     1.00    8803.694

logits_argmax_embed_baseline    32  1      1.00      93.834
prob_sampler_topk1_topp1        32  1      1.00     432.100
prob_sampler_topk64_topp095     32  64     0.95   11293.972
prob_sampler_topk64_topp1       32  64     1.00   11293.274
```

ptxas notes:

```text
sample_from_logits_kernel:
  512 bytes stack frame
  0 bytes spill stores / loads
  34 registers
  3844 bytes shared memory
```

Conclusion:

- Kept the implementation as a correct baseline. The API and deterministic
  behavior are covered by tests, and the benchmark measures sampler-only cost
  separately from final projection.
- The `top_k=1` path is usable as a correctness baseline but still slower than
  the simpler argmax-only benchmark because it performs softcap, temperature,
  shared-list merge machinery, RNG/top-p logic, and embedding gather.
- The `top_k=64` path is intentionally exact but not performance-ready. The
  per-thread top-64 arrays live in a 512-byte stack frame, so the next tuning
  pass should replace this with a lower-local-memory top-k design, likely
  warp-local candidates plus a shared/block merge or a multi-CTA tiled top-k
  path for `B=1`.
- Do not treat these numbers as final sampling performance. They are the first
  correct fused logits-sampler baseline.

Threats to validity:

- Clocks were not locked.
- Warm-ish repeated logits may understate cache pressure compared with logits
  produced by a preceding projection under real decode scheduling.
- CPU launch overhead is included.
- No Nsight Compute counters were collected because `ncu` was unavailable.

## 2026-06-18 - FlashAttention cache-hint review and decode metadata loads

Question:

- Six subagents reviewed disjoint sections of `src/gemma4_flash_attention.cu`
  for cache/cache-hint changes, then a seventh pass chose the low-risk edits.
- Consensus: keep prefill `SM80_CP_ASYNC_CACHEGLOBAL` and direct decode K/V
  `loadg` as the default; do not promote `.ca`, scalar `.cg/.cs`, or shared
  staging without a dedicated A/B.

Change:

- Added warp-uniform read-only loads for decode metadata:
  `token_position`, `page_table`, and `seq_lengths`.
- Loaded Q/K norm weights through the existing read-only `loadg` helper.
- Rejected invalid sliding decode configs when `window_size <= 0` or when
  `split_size * num_splits` cannot cover the configured sliding window.
- Extended `test-kv-cache` to size direct decode scratch for the configured
  window contract and to reject invalid window/split arguments.

Build and correctness:

```bash
make -B test-kv-cache flash-attn-bench kv-cache-bench \
  NVCC=/usr/local/cuda/bin/nvcc
```

Benchmark contract:

- Hardware: NVIDIA RTX A6000, driver `580.126.16`; benchmark reports CUDA
  driver/runtime `13000`.
- Clock policy: not locked. GPU was checked idle with `nvidia-smi` before runs.
- Timing: CUDA events on the benchmark stream. Setup excluded; launch enqueue
  included. Single process. Warm and cold cache measured separately.
- Cold cache: 64 MiB L2 flush before measured iterations.
- Shape: `seq_len=4096`, `page_size=64`, `split_size=64`, sliding
  `window=1024`, `actual_splits=16`, `batch=1`, BF16, `q_heads=32`,
  `kv_heads=16`, `head_dim=256`.
- `ncu` was not installed on this machine, so this pass has CUDA-event timings
  but no Nsight Compute counter confirmation.

Commands:

```bash
./build/experiments/gemma4_flash_attention_bench 4096 200 50 10 1 64 warm 64 \
  | tee src/experiments/results/2026-06-18_cache_hints_baseline_flash_warm.txt
./build/experiments/gemma4_kv_cache_bench 4096 64 64 50 200 20 --cache warm \
  | tee src/experiments/results/2026-06-18_cache_hints_baseline_kv_warm.txt
./build/experiments/gemma4_flash_attention_bench 4096 30 20 6 1 64 cold 64 \
  | tee src/experiments/results/2026-06-18_cache_hints_baseline_flash_cold.txt
./build/experiments/gemma4_kv_cache_bench 4096 64 64 20 30 10 --cache cold --flush-bytes 67108864 \
  | tee src/experiments/results/2026-06-18_cache_hints_baseline_kv_cold.txt

./build/experiments/gemma4_flash_attention_bench 4096 200 50 10 1 64 warm 64 \
  | tee src/experiments/results/2026-06-18_cache_hints_after_flash_warm.txt
./build/experiments/gemma4_kv_cache_bench 4096 64 64 50 200 20 --cache warm \
  | tee src/experiments/results/2026-06-18_cache_hints_after_kv_warm.txt
./build/experiments/gemma4_flash_attention_bench 4096 30 20 6 1 64 cold 64 \
  | tee src/experiments/results/2026-06-18_cache_hints_after_flash_cold.txt
./build/experiments/gemma4_kv_cache_bench 4096 64 64 20 30 10 --cache cold --flush-bytes 67108864 \
  | tee src/experiments/results/2026-06-18_cache_hints_after_kv_cold.txt
```

Median results:

```text
path                                  baseline      after         delta
norm_rope_plus_fa warm                 1.871930 ms   1.885210 ms  -0.71%
decode_norm_rope_paged_kv_write warm   0.022179 ms   0.018513 ms  +16.53%
flash_decode_paged_attention warm      0.063786 ms   0.061696 ms  +3.28%
flash_full_decode warm                 0.066363 ms   0.064191 ms  +3.27%

norm_rope_plus_fa cold                 1.770390 ms   1.769090 ms  +0.07%
decode_norm_rope_paged_kv_write cold   0.007555 ms   0.006781 ms  +10.24%
flash_decode_paged_attention cold      0.072176 ms   0.069677 ms  +3.46%
flash_full_decode cold                 0.075029 ms   0.072134 ms  +3.86%
```

Conclusion:

- Kept the changes. Correctness passed, and the stable paged decode attention
  benchmark improved `~3.3-3.5%` median under the same CUDA-event contract.
- This is a useful small win, but it is below the project's `5%` minimum effect
  threshold for a strong speed claim. Treat it as a low-risk cleanup plus
  directionally positive timing until repeated under locked clocks and with NCU
  counters.
- Decode prep-cache medians improved, but that microkernel is noisy at this
  duration; do not overfit the `10-16%` medians without process-level reruns.

## 2026-06-18 - FlashAttention cache policy ablation matrix

Question:

- Test the remaining cache/cache-parameter variants that are practical on this
  machine without Nsight Compute:
  - scalar decode K/V cache loads: default `__ldg` vs `__ldcg` vs `__ldcs`;
  - prefill CUTE cp.async: default `.cg` vs `.ca`;
  - decode kernel `Gemma4KvCacheConfig` as `__grid_constant__`.

Change:

- Temporarily added compile-time switches for the ablation, then pruned them
  after the matrix rejected every non-default policy.
- Retained production source stays hardwired to prefill `.cg`, decode K/V
  `__ldg`, and ordinary by-value `Gemma4KvCacheConfig` kernel params.

Temporary build commands used before pruning:

```bash
make -B test-kv-cache flash-attn-bench kv-cache-bench \
  NVCC=/usr/local/cuda/bin/nvcc

/usr/local/cuda/bin/nvcc -std=c++17 -O3 -arch=sm_86 \
  --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math \
  -D_GLIBCXX_USE_CXX11_ABI=1 -DGEMMA4_FA_DECODE_CACHE_LOAD_POLICY=1 \
  -Isrc -Iexperiments/flash-attention/csrc/cutlass/include \
  src/experiments/gemma4_kv_cache_bench.cu src/gemma4_kv_cache.cu \
  src/gemma4_flash_attention.cu src/gemma4.cpp \
  -o build/experiments/gemma4_kv_cache_bench_decode_cg

/usr/local/cuda/bin/nvcc -std=c++17 -O3 -arch=sm_86 \
  --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math \
  -D_GLIBCXX_USE_CXX11_ABI=1 -DGEMMA4_FA_DECODE_CACHE_LOAD_POLICY=2 \
  -Isrc -Iexperiments/flash-attention/csrc/cutlass/include \
  src/experiments/gemma4_kv_cache_bench.cu src/gemma4_kv_cache.cu \
  src/gemma4_flash_attention.cu src/gemma4.cpp \
  -o build/experiments/gemma4_kv_cache_bench_decode_cs

/usr/local/cuda/bin/nvcc -std=c++17 -O3 -arch=sm_86 \
  --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math \
  -D_GLIBCXX_USE_CXX11_ABI=1 -DGEMMA4_FA_PREFILL_CP_ASYNC_CACHE_POLICY=1 \
  -Isrc -Iexperiments/flash-attention/csrc/cutlass/include \
  src/experiments/gemma4_flash_attention_bench.cu \
  src/gemma4_flash_attention.cu \
  -o build/experiments/gemma4_flash_attention_bench_cpasync_ca
```

Benchmark contract:

- Same A6000 CUDA-event contract as the prior cache-hint entry.
- GPU was checked idle before each matrix group.
- `ncu`, `cuobjdump`, and `nvdisasm` were not available on this machine, so no
  counter/SASS/resource confirmation was possible.

Paged decode K/V load policy, `gemma4_kv_cache_bench`:

```text
variant             warm direct    warm full      cold direct    cold full
ldg default          0.062280 ms    0.065080 ms    0.069442 ms    0.071838 ms
scalar .cg           0.067521 ms    0.072502 ms    0.075919 ms    0.078511 ms
scalar .cs           0.066943 ms    0.069015 ms    0.074331 ms    0.076383 ms
__grid_constant__    0.062298 ms    0.064884 ms    0.069671 ms    0.073546 ms
```

Prefill cp.async policy, `gemma4_flash_attention_bench`:

```text
variant                  warm norm_rope+fa    cold norm_rope+fa
cp.async .cg default       1.851200 ms          1.768670 ms
cp.async .ca               1.954830 ms          1.831510 ms
__grid_constant__          1.852430 ms          1.756240 ms
```

Conclusion:

- Keep defaults. Scalar `.cg` and `.cs` decode K/V loads regress both warm and
  cold decode attention by roughly `7-9%`.
- Keep prefill cp.async `.cg`; `.ca` regressed `~5.6%` warm and `~3.6%` cold.
- Do not enable `__grid_constant__` by default. It tied the warm direct decode
  metric and was slightly worse cold; any prefill movement in that binary is
  noise because the prefill kernel does not consume `Gemma4KvCacheConfig`.
- The benchmark-only branches were deleted after this pass.

## 2026-06-17 - FA benchmark uses real Norm/RoPE path only

Runtime file:

- `src/experiments/gemma4_flash_attention_bench.cu`

Change:

- Removed the timed `prepared_fa` benchmark lane from the FlashAttention bench.
- The benchmark now measures only `norm_rope_plus_fa`, which matches the
  inference-facing sliding prefill wrapper: Q/K learned RMSNorm, RoPE, V
  scale-free RMSNorm, and FlashAttention.
- The no-LSE correctness check now calls the same norm/RoPE+FA wrapper with
  `d_softmax_lse=nullptr`, instead of checking the prepared-QKV control path.
- The low-level prepared-QKV launcher remains available as an attention-only
  primitive for control tests and future decode/cache paths; it is no longer
  presented as the default prefill benchmark result.

Verification:

```bash
make flash-attn-bench NVCC=/usr/local/cuda/bin/nvcc
./build/experiments/gemma4_flash_attention_bench 256 20 10 5 1 64 cold 64
./build/experiments/gemma4_flash_attention_bench 1024 50 20 10 1 0 cold 64
```

Result:

```text
seq=64 correctness:
max_abs=0.015625 mean_abs=0.000260142 max_rel=0.00775194
no_lse_correctness max_abs=0.015625 mean_abs=0.000260142 max_rel=0.00775194
norm_rope_prep_correctness q_max_abs=0.00195312 k_max_abs=0.0078125 v_max_abs=0.000976562

seq=256 cold:
norm_rope_plus_fa median_ms=0.064472 p95_ms=0.0652912 p99_ms=0.0653731

seq=1024 cold:
norm_rope_plus_fa median_ms=0.334232 p95_ms=0.336217 p99_ms=0.33691
```

Notes:

- The seq1024 cold samples were bimodal under unlocked clocks, matching the
  earlier benchmark caveat. The point of this pass was benchmark contract
  cleanup, not a new speed claim.

## 2026-06-17 - Sliding FA softmax exp/sum fusion

Runtime file:

- `src/gemma4_flash_attention.cu`

Change:

- Added `gemma4_fa_scale_apply_exp2_sum`, which applies `exp2(score - row_max)`
  and accumulates the per-thread softmax denominator in one pass over the score
  fragment.
- Used it for both first and later online-softmax blocks in sliding attention.
  The max pass stays separate because the final row max must be known before the
  score fragment can be converted into probabilities for `P * V`.
- Gated the helper with `UseFusedExpSum=IsLocal`. A first ungated build made the
  global `head_dim=512` specialization spill more, so global attention keeps the
  old exp-then-sum path until it has its own benchmark.

Benchmark contract:

- Benchmark commit: `c10fb36` with a dirty working tree.
- GPU: NVIDIA RTX A6000, bus `00000000:08:00.0`.
- Driver: `580.126.16`.
- NVCC: `/usr/local/cuda/bin/nvcc`, CUDA `13.0`, `V13.0.48`.
- CUDA target/flags: `sm_86`, `-O3`,
  `--expt-relaxed-constexpr --expt-extended-lambda --use_fast_math`,
  `_GLIBCXX_USE_CXX11_ABI=1`.
- Timing: CUDA events on the benchmark stream; launch overhead included.
- Shapes: BF16 sliding attention, `batch=1`, `Q heads=32`, `KV heads=16`,
  `head_dim=256`, `window_left=1024`, `return_lse=false`.
- Cache: warm repeated-buffer runs plus one cold-cache run with a 64 MiB L2
  flush before each measured iteration. The flush kernel is outside the timed
  event window.
- Clock policy: persistence mode enabled. Clock lock was attempted with
  `sudo -n nvidia-smi -ac 8001,1410`, but this user lacks permission to change
  clocks, so these are unlocked-clock results.
- Correctness tolerance: existing CPU reference envelope. Final default-binary
  smoke preserved `seq=64 max_abs=0.015625 mean_abs=0.000260142`.

Build commands:

```bash
# Run once before the source change with <tag>=baseline, then again after the
# source change with <tag>=fused.
/usr/local/cuda/bin/nvcc -std=c++17 -O3 -arch=sm_86 \
  --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math \
  -D_GLIBCXX_USE_CXX11_ABI=1 -Xptxas=-v \
  -Isrc -Iexperiments/flash-attention/csrc/cutlass/include \
  src/experiments/gemma4_flash_attention_bench.cu \
  src/gemma4_flash_attention.cu \
  -o build/experiments/gemma4_flash_attention_bench_softmax_<tag>

make flash-attn-bench NVCC=/usr/local/cuda/bin/nvcc
```

Representative warm A/B:

```text
seq=1024, warm, 200 iters/sample, 15 samples
baseline prepared_fa      median=0.233697 ms p95=0.234260 p99=0.234276
fused    prepared_fa      median=0.230594 ms p95=0.231232 p99=0.231470
baseline norm_rope_plus_fa median=0.334231 ms p95=0.336509 p99=0.337147
fused    norm_rope_plus_fa median=0.331348 ms p95=0.334153 p99=0.335040

final gated paired rerun, seq=1024, warm, 100 iters/sample, 8 samples
baseline prepared_fa       median=0.239141 ms
fused    prepared_fa       median=0.235388 ms
baseline norm_rope_plus_fa median=0.337252 ms
fused    norm_rope_plus_fa median=0.335739 ms

seq=4096, warm, 100 iters/sample, 10 samples
baseline prepared_fa       median=1.45665 ms
fused    prepared_fa       median=1.44146 ms
baseline norm_rope_plus_fa median=1.85276 ms
fused    norm_rope_plus_fa median=1.84375 ms
```

Cold-cache note:

```text
seq=1024, cold, 50 iters/sample, 10 samples
baseline prepared_fa       median=0.228554 ms
fused    prepared_fa       median=0.227252 ms  # conservative rerun; samples were bimodal
baseline norm_rope_plus_fa median=0.335721 ms
fused    norm_rope_plus_fa median=0.321124 ms
```

ptxas check:

```text
Sliding head_dim=256 ReturnLse=true:
  baseline: 245 regs, 0 spill stores, 0 spill loads
  fused:    247 regs, 0 spill stores, 0 spill loads
Sliding head_dim=256 ReturnLse=false:
  baseline: 246 regs, 0 spill stores, 0 spill loads
  fused:    246 regs, 0 spill stores, 0 spill loads
Global head_dim=512 after IsLocal gate:
  same spill counts as baseline: ReturnLse=true 1512/1692, false 1476/1652
```

Conclusion:

- Keep the sliding-only fusion. It removes one pass over the score fragment in
  the softmax path and repeatedly showed a small warm-cache win, roughly
  1-1.6% for prepared sliding FA and 0.45-0.9% for fused norm/RoPE+FA in paired
  runs.
- Do not claim a large win: clocks were unlocked and some cold/prepared samples
  were bimodal. The stronger result is that the local path stayed spill-free
  while the global path was protected from the ungated spill regression.

## 2026-06-17 - Forward-only dependency cleanup

Scope:

- `src/gemma4_flash_attention.cu`
- `src/experiments/gemma4_flash_attention_compare.py`
- `src/experiments/gemma4_flash_attention_reference.cu`
- `experiments/flash-attention/`
- stale experiment checkouts under `experiments/` and `src/experiments/`

Change:

- Removed stale training/reverse-mode experiment trees:
  `src/experiments/quack_rmsnorm`, `src/experiments/llama_cpp_refs`,
  `experiments/tinygrad`, and `src/experiments/tinygrad_late_eval_bench.py`.
- Trimmed the upstream FlashAttention checkout to the forward CUDA source slice,
  the local CUTLASS/CuTe include provider, and the upstream license file.
- Removed unused reverse-mode declarations and source files from the kept
  FlashAttention source slice.
- Replaced CUTLASS sync logging with a no-op compatibility shim so CuTe headers
  still compile without bringing in device-enumerating debug code.
- Removed launch-cluster attribute reporting from the FA diagnostic helpers and
  comparator.

Verification:

```bash
make -B flash-attn-bench flash-attn-lib NVCC=/usr/local/cuda/bin/nvcc
./build/experiments/gemma4_flash_attention_bench 256 10 3 1 1 64 warm 64
```

Result:

```text
correctness seq=64 max_abs=0.015625 mean_abs=0.000260142 max_rel=0.00775194
no_lse_correctness max_abs=0.015625 mean_abs=0.000260142 max_rel=0.00775194
prep_correctness q_max_abs=0.00195312 k_max_abs=0.0078125 v_max_abs=0.000976562
prepared_fa median_ms=0.0373472
norm_rope_plus_fa median_ms=0.063584
```

Notes:

- Audit scans over active source, experiments, and the trimmed dependency tree
  returned no reverse-mode or multi-device leftovers outside generated build
  outputs and the local CUDA guide document.
- `flash-attn-reference-lib` was not completed in this pass; the main forward
  bench and shared library rebuilt successfully, then the reference target was
  interrupted after its template instantiation ran too long.

## 2026-06-17 - FlashAttention cleanup fixes

Runtime files:

- `src/gemma4_flash_attention.cu`
- `src/gemma4_flash_attention.cuh`
- `src/experiments/gemma4_flash_attention_bench.cu`

Change:

- Restored the FA2 shared-memory GEMM helper after the reduction cleanup had
  accidentally dropped the first A-fragment preload and closed the helper early.
- Replaced the local recursive max/sum operator structs with direct 4-lane
  max/sum shuffle reductions.
- Added a no-LSE specialization selected by passing `nullptr` for
  `d_softmax_lse`; inference paths can skip `__logf` and LSE global stores while
  existing tests/callers that pass an LSE buffer keep the old behavior.
- Cached `cudaFuncSetAttribute(...MaxDynamicSharedMemorySize...)` per kernel
  specialization instead of setting it on every launch.
- Added full-visibility checks before applying causal/local masks. Fully visible
  score tiles now skip mask writes and use the no-`CheckInf` softmax path.
- Changed the fused sliding QKV norm/RoPE prep grid from flattened
  `batch * seq` plus `% seq_len` to explicit `(seq, head_group, batch)` launch
  dimensions.
- The benchmark's timed path now passes `nullptr` for LSE and reports
  `return_lse=false`; the correctness path checks both LSE and no-LSE output
  against the CPU reference.

Verification:

```bash
make -B flash-attn-bench NVCC=/usr/local/cuda/bin/nvcc
make -B flash-attn-lib NVCC=/usr/local/cuda/bin/nvcc
./build/experiments/gemma4_flash_attention_bench 256 10 3 1 1 64 warm 64
./build/experiments/gemma4_flash_attention_bench 1024 50 20 5 1 64 warm 64
```

Smoke results:

```text
correctness seq=64 max_abs=0.015625 mean_abs=0.000260142 max_rel=0.00775194
no_lse_correctness max_abs=0.015625 mean_abs=0.000260142 max_rel=0.00775194
prep_correctness q_max_abs=0.00195312 k_max_abs=0.0078125 v_max_abs=0.000976562

seq=1024 return_lse=false warm:
prepared_fa median_ms=0.23913
norm_rope_plus_fa median_ms=0.340449
overhead_vs_prepared median_ms=0.101318 median_pct=42.3695
```

Conclusion:

- The FA source builds again.
- LSE is now optional for inference.
- Fully visible local/global score tiles avoid unnecessary mask logic.
- The `seq=1024` smoke is in the same ballpark as the previous warm runs; this
  was a focused fix/smoke pass, not a rigorous locked-clock benchmark.

## 2026-06-16 - Benchmark harness source cleanup

Runtime files:

- `Makefile`
- `src/experiments/gemma4_decode_bench.cu`
- `src/experiments/gemma4_ffn_cudnn_bench.cu`

Change:

- Removed the `decode-bench` measurement path for
  `src/experiments/gemma4_matmul_device_kernels.cu`.
- Deleted the experiment-only matmul wrapper source/header. Decode benchmark
  now measures the production launchers from `src/gemma4_matmul_kernels.cu`:
  `gemma4_projection_decode` and `gemma4_projection_decode_swizzled`.
- Kept benchmark-only utility kernels for random-fill/cache-flush because they
  are input setup and cache-state controls, not duplicated inference kernels.
- Added explicit CUDA/cuDNN include and rpath variables so benchmark binaries
  run without relying on ambient `LD_LIBRARY_PATH`.
- `ffn-cudnn-bench` now exits successfully when cuDNN Frontend headers are not
  available at compile time, marking the optional comparison as skipped instead
  of failing the whole benchmark smoke.

Verification:

```bash
make -B decode-bench NVCC=/usr/local/cuda/bin/nvcc
./build/experiments/gemma4_decode_bench global_k 1 0 1

make -B embedding-gather-bench rmsnorm-hidden-fused-bench rope-bench \
  ffn-decode-load-bench flash-attn-bench NVCC=/usr/local/cuda/bin/nvcc

make -B rmsnorm-bench ffn-cudnn-bench NVCC=/usr/local/cuda/bin/nvcc
```

Smoke runs completed for decode, flash-attn, RoPE, RMSNorm, hidden fused
RMSNorm, embedding gather, FFN load bench, and FFN cuDNN skip behavior.

## 2026-06-16 - Sliding FA QKV norm plus RoPE prep

Runtime files:

- `src/gemma4_flash_attention.cu`
- `src/gemma4_flash_attention.cuh`
- `src/experiments/gemma4_flash_attention_bench.cu`

Change:

- Added `gemma4_flash_attention_sliding_fwd_bf16_norm_rope`, a prefill-only
  helper in the flash-attention module.
- The helper prepares Q/K/V once, then calls the existing tensor-core sliding
  FA kernel:
  - Q: learned RMSNorm, then split-half RoPE.
  - K: learned RMSNorm, then split-half RoPE.
  - V: scale-free RMSNorm.
- Kept the FA kernel itself unchanged. Recomputing K/V norm and RoPE inside
  each FA K/V tile load would repeat the prep for every query block.
- KV cache layout was not implemented in this patch. The next paged cache path
  should use Layout A:
  `[num_layers, num_pages, page_size, num_heads, head_dim]`.

Build:

```bash
make -B flash-attn-bench NVCC=/usr/local/cuda/bin/nvcc
```

GPU/tooling:

- GPU: NVIDIA RTX A6000, bus `00000000:0D:00.0`
- Driver: 580.126.16
- NVCC: CUDA 13.0, `nvcc` 13.0.48
- CUDA target: `sm_86`
- Clock policy: not locked.
- Cache policy: warm-cache repeated buffers.
- Timing: CUDA events on the benchmark stream.
- Nsight Compute: unavailable (`ncu: command not found`).
- Note: CUDA 13.0 is a deviation from the project preference for CUDA 12.x;
  no CUDA 12 NVCC was installed under `/usr/local`.

Correctness smoke:

```bash
./build/experiments/gemma4_flash_attention_bench 1024 100 20 3 1 64
```

```text
correctness seq=64 max_abs=0.015625 mean_abs=0.000260142 max_rel=0.00775194
prep_correctness q_max_abs=0.00195312 k_max_abs=0.0078125 v_max_abs=0.000976562
```

Benchmark contract:

- Baseline: existing sliding FA on already-prepared Q/K/V.
- New path: Q/K learned RMSNorm + V scale-free RMSNorm + Q/K RoPE + the same
  sliding FA.
- Warmup/timing at `seq=1024`: 50 warmups, 500 timed iterations, 5 trials,
  repeated across 5 fresh processes.
- Warmup/timing at `seq=4096`: 50 warmups, 200 timed iterations, 3 trials,
  repeated across 3 fresh processes.

Raw `seq=1024`, `batch=1`, `window_left=1024` samples:

```text
prepared_fa best_ms=0.230605 avg_ms=0.231656
norm_rope_plus_fa best_ms=0.331543 avg_ms=0.333089 overhead_best_ms=0.100938 overhead_best_pct=43.7709

prepared_fa best_ms=0.230439 avg_ms=0.232190
norm_rope_plus_fa best_ms=0.333398 avg_ms=0.334182 overhead_best_ms=0.102958 overhead_best_pct=44.6792

prepared_fa best_ms=0.229852 avg_ms=0.232230
norm_rope_plus_fa best_ms=0.333469 avg_ms=0.334841 overhead_best_ms=0.103618 overhead_best_pct=45.0802

prepared_fa best_ms=0.229796 avg_ms=0.232688
norm_rope_plus_fa best_ms=0.333412 avg_ms=0.335302 overhead_best_ms=0.103616 overhead_best_pct=45.0906

prepared_fa best_ms=0.231061 avg_ms=0.233694
norm_rope_plus_fa best_ms=0.334066 avg_ms=0.335673 overhead_best_ms=0.103005 overhead_best_pct=44.5792
```

`seq=1024` summary:

- Prepared FA median best: `0.230439 ms`.
- Norm+RoPE+FA median best: `0.333412 ms`.
- Median best overhead: `0.103005 ms`, about `44.7%`.
- Attention-only throughput from prepared FA: about `74.6 TFLOP/s`.

Raw `seq=4096`, `batch=1`, `window_left=1024` samples:

```text
prepared_fa best_ms=1.44497 avg_ms=1.45688
norm_rope_plus_fa best_ms=1.84029 avg_ms=1.84921 overhead_best_ms=0.395325 overhead_best_pct=27.3587

prepared_fa best_ms=1.44508 avg_ms=1.45923
norm_rope_plus_fa best_ms=1.84469 avg_ms=1.85295 overhead_best_ms=0.399602 overhead_best_pct=27.6525

prepared_fa best_ms=1.45549 avg_ms=1.46709
norm_rope_plus_fa best_ms=1.85052 avg_ms=1.85474 overhead_best_ms=0.395034 overhead_best_pct=27.1410
```

`seq=4096` summary:

- Prepared FA median best: `1.44508 ms`.
- Norm+RoPE+FA median best: `1.84469 ms`.
- Median best overhead: `0.395325 ms`, about `27.4%`.
- Attention-only throughput from prepared FA: about `83.3 TFLOP/s`.

Conclusion:

- The prefill helper is correct against the CPU reference and keeps the existing
  FA kernel on the fast tensor-core path.
- The prep-once kernel is not free: roughly `0.10 ms` at `seq=1024`, and
  `0.40 ms` at `seq=4096`.
- That is still better than doing K/V norm and RoPE inside repeated FA tile
  loads for prefill. For decode, the paged KV-cache write path should prep only
  the appended token and store it once in Layout-A cache pages.

## 2026-06-16 - Shared RoPE primitive inside fused sliding FA prep

Runtime files:

- `src/gemma4_rope.cuh`
- `src/gemma4_rope.cu`
- `src/gemma4_flash_attention.cu`
- `Makefile`

Change:

- Moved the standalone RoPE pair/pack/head device math into
  `gemma4_rope.cuh`.
- Updated the standalone RoPE kernels to call the shared head helper with
  explicit lane/thread-count arguments.
- Updated the fused sliding FA prep kernel to call
  `gemma4_rope::store_rotated_pair_bf16` while normalized Q/K values are still
  in registers.
- No separate RoPE launch was added; Q/K RMSNorm, RoPE, V RMSNorm, and FA remain
  fused at the wrapper level.
- The shared head helper still supports partial/p-RoPE through the caller-owned
  `rotary_half`; trailing dimensions are untouched by design.

Validation:

```bash
make -B test-rope flash-attn-bench NVCC=/usr/local/cuda/bin/nvcc
./build/experiments/gemma4_flash_attention_bench 1024 200 30 5 1 64
```

Result:

```text
rope tests passed
correctness seq=64 max_abs=0.015625 mean_abs=0.000260142 max_rel=0.00775194
prep_correctness q_max_abs=0.00195312 k_max_abs=0.0078125 v_max_abs=0.000976562
```

Benchmark contract:

- Benchmark commit: `c10fb36` with a dirty working tree.
- GPU: NVIDIA RTX A6000.
- Driver: `580.126.16`.
- NVCC: CUDA `13.0`, `nvcc` `13.0.48`.
- CUDA target/flags: `sm_86`, `-O3`, FlashAttention flags
  `--expt-relaxed-constexpr --expt-extended-lambda --use_fast_math`.
- Timing: CUDA events from the existing C++ harness, same stream as kernels;
  launch overhead included.
- Cache policy: warm-cache repeated buffers.
- Clock policy: not locked. Earlier clock-lock attempts on this machine were
  permission denied; not retried for this small refactor.
- Shape: sliding prefill, `batch=1`, `seq=1024`, `window_left=1024`,
  `Q heads=32`, `KV heads=16`, `D=256`, BF16.
- Warmup/timing: `30` warmups, `200` iterations per trial, `5` trials.
- Environment caveat: `nvidia-smi` bus IDs varied between snapshots
  (`07:00.0` before, `04:00.0` after), same telemetry wrinkle seen in the KV
  cache runs.

Before:

```text
prepared_fa best_ms=0.225734 avg_ms=0.231372
norm_rope_plus_fa best_ms=0.333442 avg_ms=0.334411
overhead_best_ms=0.107708 overhead_best_pct=47.7147
```

After:

```text
prepared_fa best_ms=0.224975 avg_ms=0.226767
norm_rope_plus_fa best_ms=0.327444 avg_ms=0.328244
overhead_best_ms=0.102469 overhead_best_pct=45.5468
```

Conclusion:

- Sharing the standalone RoPE primitive did not break standalone RoPE or fused
  sliding FA correctness.
- The after run is slightly faster, but this is below the threshold for a real
  performance claim without locked clocks and repeated process-level runs.
- The useful outcome is structural: there is now one RoPE device implementation
  for standalone kernels and the fused FlashAttention prep path.

## 2026-06-16 - Shared RoPE helper rigorous FA benchmark

Runtime files:

- `src/gemma4_flash_attention.cu`
- `src/experiments/gemma4_flash_attention_bench.cu`
- `src/experiments/results/2026-06-16_fa_rope_helper_*.txt`

Change:

- Added `GEMMA4_FA_USE_SHARED_ROPE_HELPER`, defaulting to `1`, so the normal
  FlashAttention source can be compiled as:
  - shared helper: `-DGEMMA4_FA_USE_SHARED_ROPE_HELPER=1`
  - inline control: `-DGEMMA4_FA_USE_SHARED_ROPE_HELPER=0`
- Tightened the FA benchmark harness to report median, mean, 10% trimmed mean,
  min, max, p95, p99, standard deviation, IQR, and raw sample arrays.
- Added explicit `warm`/`cold` cache modes. Cold mode launches a 64 MiB L2
  flush before each measured operation; the CUDA event timing starts after the
  flush, so the reported time excludes the flush kernel.

Build commands:

```bash
/usr/local/cuda/bin/nvcc -std=c++17 -O3 -arch=sm_86 \
  --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math \
  -D_GLIBCXX_USE_CXX11_ABI=1 -DGEMMA4_FA_USE_SHARED_ROPE_HELPER=1 \
  -Isrc -Iexperiments/flash-attention/csrc/cutlass/include \
  src/experiments/gemma4_flash_attention_bench.cu \
  src/gemma4_flash_attention.cu \
  -o build/experiments/gemma4_flash_attention_bench_shared

/usr/local/cuda/bin/nvcc -std=c++17 -O3 -arch=sm_86 \
  --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math \
  -D_GLIBCXX_USE_CXX11_ABI=1 -DGEMMA4_FA_USE_SHARED_ROPE_HELPER=0 \
  -Isrc -Iexperiments/flash-attention/csrc/cutlass/include \
  src/experiments/gemma4_flash_attention_bench.cu \
  src/gemma4_flash_attention.cu \
  -o build/experiments/gemma4_flash_attention_bench_inline
```

Correctness:

Both binaries reported:

```text
correctness seq=64 max_abs=0.015625 mean_abs=0.000260142 max_rel=0.00775194
prep_correctness q_max_abs=0.00195312 k_max_abs=0.0078125 v_max_abs=0.000976562
```

Benchmark contract:

- Question: does importing the standalone RoPE device primitive into fused
  sliding FA prep change runtime?
- Timing scope: CUDA-event GPU timeline on the benchmark stream; launch
  overhead included; allocation/setup excluded.
- Main metric: typical latency, reported by median and 10% trimmed mean.
- Tail metrics: p95/p99, stddev, IQR, raw samples saved.
- Stability target: same process plus process repeats for close `seq=1024`
  results; minimum effect size for a performance claim: `5%`.
- Cache states:
  - warm: repeated buffers, no flush;
  - cold: 64 MiB L2 flush before each measured operation.
- Inputs: fixed random seeds in the harness; BF16, batch `1`, sliding attention,
  `Q heads=32`, `KV heads=16`, `D=256`, `window_left=1024`.
- GPU: NVIDIA RTX A6000, `49140 MiB`, driver `580.126.16`, CUDA `13.0`,
  persistence enabled, ECC disabled, MIG N/A, power limit `300 W`.
- NVCC: CUDA `13.0`, `nvcc` `13.0.48`.
- Clock policy: not locked. `sudo -n nvidia-smi -ac 8001,1800` and
  `sudo -n nvidia-smi -lgc 1800,1800` were denied.
- Nsight Compute: unavailable (`ncu: command not found`).
- Contention check: `nvidia-smi --query-compute-apps=...` showed no compute
  apps after the run; `nvidia-smi pmon` is unsupported here.
- Telemetry: one-second `nvidia-smi` sampling saved to
  `src/experiments/results/2026-06-16_fa_rope_helper_telemetry.txt`. The runs
  are short enough that most samples caught idle clocks; one loaded sample saw
  about `1800 MHz` SM and `7601 MHz` memory.

Persisted raw outputs:

```text
src/experiments/results/2026-06-16_fa_rope_helper_shared_warm_1024_rep1.txt
src/experiments/results/2026-06-16_fa_rope_helper_inline_warm_1024_rep1.txt
src/experiments/results/2026-06-16_fa_rope_helper_inline_warm_1024_rep2.txt
src/experiments/results/2026-06-16_fa_rope_helper_shared_warm_1024_rep2.txt
src/experiments/results/2026-06-16_fa_rope_helper_shared_warm_4096.txt
src/experiments/results/2026-06-16_fa_rope_helper_inline_warm_4096.txt
src/experiments/results/2026-06-16_fa_rope_helper_shared_cold_1024.txt
src/experiments/results/2026-06-16_fa_rope_helper_inline_cold_1024.txt
src/experiments/results/2026-06-16_fa_rope_helper_inline_cold_1024_rep2.txt
src/experiments/results/2026-06-16_fa_rope_helper_shared_cold_1024_rep2.txt
src/experiments/results/2026-06-16_fa_rope_helper_telemetry.txt
```

Warm cache, `seq=1024`, `warmup=50`, `iters=100`, `samples=31` per process,
process order shared/inline/inline/shared:

```text
shared norm_rope_plus_fa median_ms: 0.329229, 0.329984
inline norm_rope_plus_fa median_ms: 0.328916, 0.329836
shared average of process medians: 0.3296065 ms
inline average of process medians: 0.3293760 ms
delta shared-vs-inline: +0.0002305 ms, about +0.07%
```

Warm cache, `seq=4096`, `warmup=50`, `iters=50`, `samples=31`:

```text
shared norm_rope_plus_fa median_ms=1.82804 trimmed_mean_ms=1.82666 p95_ms=1.83274 p99_ms=1.83279
inline norm_rope_plus_fa median_ms=1.83008 trimmed_mean_ms=1.83055 p95_ms=1.83739 p99_ms=1.84098
delta shared-vs-inline: -0.00204 ms, about -0.11%
```

Cold cache, `seq=1024`, `warmup=30`, `iters=50`, `samples=31`, 64 MiB L2
flush before every measured operation, process order shared/inline/inline/shared:

```text
shared norm_rope_plus_fa median_ms: 0.319142, 0.317812
inline norm_rope_plus_fa median_ms: 0.332833, 0.332837
shared average of process medians: 0.318477 ms
inline average of process medians: 0.332835 ms
delta shared-vs-inline: -0.014358 ms, about -4.31%
```

Conclusion:

- Warm-cache performance is a tie. The shared-helper path changes maintainability
  and single-sources RoPE math; it does not measurably change warm-cache FA prep
  runtime.
- Cold-cache results consistently favored the shared-helper binary by about
  `4.3%`, but the pre-declared threshold was `5%` and clocks were not locked.
  Treat this as possible upside, not a hard performance claim.
- There is no evidence of a regression from using the standalone RoPE primitive
  inside the fused sliding FA prep path.

## 2026-05-23 - Global FlashAttention option

Runtime files:

- `src/gemma4_flash_attention.cu`
- `src/gemma4_flash_attention.cuh`
- `src/experiments/gemma4_flash_attention_bench.cu`
- `Makefile`

Change:

- Added a Gemma global-attention FA entrypoint:
  `gemma4_flash_attention_global_fwd_bf16`.
- The global path uses BF16, `head_dim=512`, `32` query heads, `4` KV heads,
  and full bottom-right causal masking by setting the internal window to
  `seqlen_k`.
- Kept the existing sliding entrypoint unchanged.
- Added a public CUDA header exposing both sliding and global FA options.
- Added a reference global wrapper that directly instantiates the upstream FA2
  kernel template at `head_dim=512`. The installed `flash_attn` package cannot
  serve as the global baseline because it rejects `head_dim > 256`.

Build:

```bash
make -B build/gemma4_flash_attention.o flash-attn-lib flash-attn-bench
```

GPU/tooling:

- GPU: NVIDIA RTX A6000
- Driver: 580.126.16
- CUDA/NVCC: CUDA 12.9, `nvcc` 12.9.86
- CUDA target: `sm_86`
- Clock policy: not locked.

Audits:

```text
dep_count 709
fa2_source_count 0
cutlass_cute_count 141
```

Exported symbols include both paths:

```text
gemma4_flash_attention_sliding_fwd_bf16
gemma4_flash_attention_global_fwd_bf16
```

Correctness:

Installed `flash_attn` check:

```text
RuntimeError FlashAttention forward only supports head dimension at most 256
```

Python probe against a PyTorch FP32 reference with GQA expansion and
bottom-right causal masking:

```text
global threads 64 smem 98304
global_correctness sq=64 sk=64 max_abs=0.015625 mean_abs=0.000255395
global_correctness sq=1 sk=1024 max_abs=0.000976562 mean_abs=5.82829e-05
```

Sliding path still matches the upstream source ref and installed `flash_attn`
exactly in the same-tensor comparator:

```text
diff_vs_official_source_ref max_abs=0 mean_abs=0 max_rel=0
diff_vs_flash_attn max_abs=0 mean_abs=0 max_rel=0
```

Timing:

Direct upstream FA2 `head_dim=512` reference comparison:

```text
case sq=64 sk=64
  diff max_abs=0 mean_abs=0 lse_max_abs=0
  custom median/min/max ms=0.034511/0.033782/0.057761
  upstream_direct median/min/max ms=0.032795/0.030715/0.043902
  custom/ref=1.052323

case sq=1 sk=1024
  diff max_abs=0 mean_abs=0 lse_max_abs=0
  custom median/min/max ms=0.207234/0.205562/0.218658
  upstream_direct median/min/max ms=0.209351/0.209320/0.209530
  custom/ref=0.989888

case sq=1024 sk=1024
  diff_max=0 diff_mean=0 lse_max=0
  custom median/min/max ms=1.610805/1.609355/1.645277
  upstream_direct median/min/max ms=1.618664/1.614611/1.625269
  custom/ref=0.995145
```

```text
global_decode_timing sk=1024 avg_ms=0.239686
```

The sliding comparator in this run measured all implementations around
`0.48 ms` at `seq=1024`, including installed `flash_attn`, whereas an earlier
run measured all implementations around `0.234 ms`. Since the custom, upstream
source ref, and installed package moved together, this looks like GPU clock or
system state rather than a custom-only regression.

Conclusion:

- Global FA is now an explicit runtime option for the future layer runner.
- Against a direct upstream FA2 `head_dim=512` baseline, the global path is exact
  on output and effectively performance-parity on decode and larger prefill
  shapes.
- This is a correctness-oriented global path, not the final optimized
  long-context global decode kernel. The `head_dim=512` tile is
  `BlockM=32, BlockN=32, 2` warps, using `98304` bytes dynamic shared memory.

## 2026-05-23 - Device-call matmul wrapper and FFN decode integration

Runtime files:

- `src/gemma4_matmul_device.cuh`
- `src/experiments/gemma4_matmul_device_kernels.cu`
- `src/experiments/gemma4_matmul_device_kernels.cuh`
- `src/experiments/gemma4_decode_bench.cu`
- `src/gemma4_ffn_decode.cu`
- `Makefile`

Change:

- Copied the decode GEMV launch surface into an experiment source and moved the
  kernel body into reusable `__device__` helpers.
- Added a deliberately thin global wrapper around the device helper so it can be
  compared directly with the original `src/gemma4_matmul_kernels.cu` kernels.
- Added decode-bench timing and numerical checks for `device_wrapped` and
  `device_wrapped_swizzle16`.
- Wired the device matmul dot/reduce helper into the fused FFN decode gate/up
  path. The FFN kernel still keeps the existing fused dataflow: shared-X load,
  gate/up dot, GeGLU, down-row accumulation, residual add, and RMSNorm.

Correctness:

```bash
make decode-bench test-ffn-decode
```

```text
ffn decode tests passed
```

Device-wrapper matmul timing:

```bash
GEMMA4_DECODE_BENCH_SEED=0x20260522 \
  build/experiments/gemma4_decode_bench ffn_gate_up 100 20 3
GEMMA4_DECODE_BENCH_SEED=0x20260522 \
  build/experiments/gemma4_decode_bench ffn_down 100 20 3
```

GPU/tooling:

- GPU: NVIDIA RTX A6000
- Driver: 580.126.16
- CUDA/NVCC: CUDA 12.9, `nvcc` 12.9.86
- CUDA target: `sm_86`
- cuDNN version in FFN harness: `92200`
- Warmup/timing: 20 warmup iterations, 100 timed iterations, 3 trials.
- Timing uses CUDA events through the existing benchmark helper.
- Clock policy: not locked.
- Cache policy: normal benchmark warm-cache behavior.

| Op | Original best ms | Device wrapper best ms | Device/original | Max abs diff |
| --- | ---: | ---: | ---: | ---: |
| `ffn_gate_up` | 0.650478 | 0.650972 | 0.999241x | 0 |
| `ffn_down` | 0.327827 | 0.327563 | 1.000807x | 0 |

Swizzled variant:

| Op | Original swizzle16 best ms | Device wrapper swizzle16 best ms | Device/original | Max abs diff |
| --- | ---: | ---: | ---: | ---: |
| `ffn_gate_up` | 0.650752 | 0.650858 | 0.999838x | 0 |
| `ffn_down` | 0.327701 | 0.327575 | 1.000383x | 0 |

ptxas resource check:

```bash
nvcc -std=c++17 -O3 -arch=sm_86 -Xptxas=-v -Isrc \
  -c src/gemma4_matmul_kernels.cu \
  -o /tmp/gemma4_device_matmul_stats/original_matmul.o
nvcc -std=c++17 -O3 -arch=sm_86 -Xptxas=-v -Isrc \
  -c src/experiments/gemma4_matmul_device_kernels.cu \
  -o /tmp/gemma4_device_matmul_stats/device_wrapped_matmul.o
nvcc -std=c++17 -O3 -arch=sm_86 -Xptxas=-v -Isrc \
  -c src/gemma4_ffn_decode.cu \
  -o /tmp/gemma4_device_matmul_stats/ffn_decode_device_matmul.o
```

For the original and device-wrapper matmul kernels, ptxas reported matching
resource usage for the FFN shapes:

| Kernel shape | Original regs | Wrapper regs | Shared memory | Spills |
| --- | ---: | ---: | ---: | --- |
| `K=5376,N=43008,identity` | 58 | 58 | 512 B | none |
| `K=5376,N=43008,swizzle16` | 60 | 60 | 512 B | none |
| `K=21504,N=5376,identity` | 58 | 58 | 1024 B | none |
| `K=21504,N=5376,swizzle16` | 60 | 60 | 1024 B | none |

FFN decode after integration:

```bash
GEMMA4_FFN_CUDNN_BENCH_SEED=0x20260522 \
  build/experiments/gemma4_ffn_cudnn_bench 100 20 3
```

Baseline captured before the FFN edit:

```text
cudnn_split_device_ms,1.000036
custom_device_ms,1.108929
custom_scratch_clear_device_ms,0.001090
custom_minus_clear_device_ms,1.107840
custom_vs_cudnn_split_speedup,0.901803
custom_minus_clear_vs_cudnn_split_speedup,0.902690
```

After using the device matmul helper inside FFN decode:

```text
cudnn_split_device_ms,1.000140
custom_device_ms,1.106343
custom_scratch_clear_device_ms,0.001086
custom_minus_clear_device_ms,1.105257
custom_vs_cudnn_split_speedup,0.904006
custom_minus_clear_vs_cudnn_split_speedup,0.904894
```

FFN kernel resource stats after the change:

```text
Used 48 registers, used 1 barriers, 11404 bytes smem
0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
```

Conclusion:

- The copied device-call matmul wrapper is effectively performance-neutral
  versus the original global matmul kernel for the two FFN GEMV shapes and is
  numerically identical.
- ptxas stats match the original matmul kernels for those shapes.
- Using the device matmul helper in the fused FFN decode gate/up path preserved
  correctness and slightly improved graph-timed custom FFN decode from
  `1.108929 ms` to `1.106343 ms`.
- The FFN change reduced register allocation from the previous tile-2 swizzled
  path's recorded `50` registers to `48`, with shared memory increasing from
  about `11020 B` to `11404 B`, and no spills.
- Keep this integrated path. The helper gives a reusable matmul-shaped
  dot/reduce primitive without changing the fused FFN dataflow.

## 2026-05-23 - FFN decode activation-tile sweep

Runtime files:

- `src/gemma4_ffn_decode.cu`
- `tests/test_ffn_decode.cu`
- `src/experiments/gemma4_ffn_cudnn_bench.cu`

Change:

- Added `GEMMA4_FFN_DECODE_ACT_TILE` as a compile-time FFN decode hyperparameter.
- Default is now `2`.
- Added `GEMMA4_FFN_DECODE_SWIZZLE_X` for the FFN decode shared activation
  cache. Default is now enabled.
- The shared activation swizzle XORs 128-bit chunk indices, not element indices:
  one chunk is one `Bf16Packed128` / 16-byte lane, equivalent in size to
  `4 x float`.
- The tiled path is first-class in the main loop: each CTA computes `kActTile`
  gate/up scalar pairs, reduces them, stores `s_act[kActTile]`, then streams the
  matching down rows before advancing.
- This keeps the fused single-token decode contract unchanged and does not add a
  second special-case kernel path.

Correctness:

```text
make test-ffn-decode
ffn decode tests passed

for tile in 2 4 8; do
  nvcc -std=c++17 -O3 -arch=sm_86 -Isrc \
    -DGEMMA4_FFN_DECODE_ACT_TILE=${tile} \
    tests/test_ffn_decode.cu src/gemma4_ffn_decode.cu \
    -o /tmp/test_ffn_decode_tile${tile}
  /tmp/test_ffn_decode_tile${tile}
done
ffn decode tests passed
ffn decode tests passed
ffn decode tests passed
```

Resource sweep:

```bash
for tile in 1 2 4 8 16; do
  nvcc -std=c++17 -O3 -arch=sm_86 -Xptxas=-v -Isrc \
    -DGEMMA4_FFN_DECODE_SWIZZLE_X=0 \
    -DGEMMA4_FFN_DECODE_ACT_TILE=${tile} \
    -c src/gemma4_ffn_decode.cu -o /tmp/gemma4_ffn_decode_tile${tile}.o
done
```

| Tile | Registers | Shared memory | Stack/spills |
| ---: | ---: | ---: | --- |
| 1 | 59 | 11016 B | 0 stack, 0 spills |
| 2 | 59 | 11020 B | 0 stack, 0 spills |
| 4 | 62 | 11028 B | 0 stack, 0 spills |
| 8 | 62 | 11044 B | 16 B stack, 64 B spill stores, 64 B spill loads |
| 16 | 64 | 11076 B | 88 B stack, 188 B spill stores, 212 B spill loads |

Chunk-swizzled shared activation resource sweep:

```bash
for swiz in 0 1; do
  for tile in 1 2 4 8; do
    nvcc -std=c++17 -O3 -arch=sm_86 -Xptxas=-v -Isrc \
      -DGEMMA4_FFN_DECODE_SWIZZLE_X=${swiz} \
      -DGEMMA4_FFN_DECODE_ACT_TILE=${tile} \
      -c src/gemma4_ffn_decode.cu -o /tmp/gemma4_ffn_decode_swiz${swiz}_tile${tile}.o
  done
done
```

| Swizzle | Tile | Registers | Shared memory | Stack/spills |
| ---: | ---: | ---: | ---: | --- |
| 0 | 1 | 59 | 11016 B | 0 stack, 0 spills |
| 0 | 2 | 59 | 11020 B | 0 stack, 0 spills |
| 0 | 4 | 62 | 11028 B | 0 stack, 0 spills |
| 0 | 8 | 62 | 11044 B | 16 B stack, 64 B spill stores, 64 B spill loads |
| 1 | 1 | 51 | 11016 B | 0 stack, 0 spills |
| 1 | 2 | 50 | 11020 B | 0 stack, 0 spills |
| 1 | 4 | 62 | 11028 B | 0 stack, 0 spills |
| 1 | 8 | 62 | 11044 B | 16 B stack, 64 B spill stores, 64 B spill loads |

PTX/SASS dump for the tile-2 swizzle comparison:

```bash
mkdir -p build/ptx
for swiz in 0 1; do
  nvcc -std=c++17 -O3 -arch=sm_86 -Isrc \
    -DGEMMA4_FFN_DECODE_SWIZZLE_X=${swiz} \
    -DGEMMA4_FFN_DECODE_ACT_TILE=2 \
    -ptx src/gemma4_ffn_decode.cu \
    -o build/ptx/gemma4_ffn_decode_swiz${swiz}_tile2.ptx
  nvcc -std=c++17 -O3 -arch=sm_86 -Xptxas=-v -lineinfo -Isrc \
    -DGEMMA4_FFN_DECODE_SWIZZLE_X=${swiz} \
    -DGEMMA4_FFN_DECODE_ACT_TILE=2 \
    -cubin src/gemma4_ffn_decode.cu \
    -o /tmp/gemma4_ffn_decode_swiz${swiz}_tile2.cubin
  cuobjdump --dump-sass /tmp/gemma4_ffn_decode_swiz${swiz}_tile2.cubin \
    > build/ptx/gemma4_ffn_decode_swiz${swiz}_tile2.sass
done
```

PTX/SASS notes:

- PTX virtual register declarations are not the ptxas allocation result. Swizzle-off
  emitted fewer virtual `.b32` PTX registers (`%r<346>`) than swizzle-on before the
  unsigned cleanup (`%r<381>`), but ptxas still allocated more physical registers for
  swizzle-off.
- The physical SASS peak matched the ptxas report: swizzle-off used up to `R56`
  and reported 59 registers; swizzle-on used up to `R47` and reported 50 registers.
- The drop comes from the `s_x` load/store prologue. With identity indexing, ptxas
  recognizes a simple linear shared-memory stream and aggressively unrolls/schedules
  many 128-bit global loads and shared stores at once. That exposes many live address
  and 128-bit payload registers.
- With XOR chunk swizzling, the shared address is no longer a simple linear induction
  variable. ptxas keeps the prologue in a much smaller loop shape, so fewer load
  payloads and store addresses are simultaneously live. The register drop is mostly
  a scheduling/strength-reduction side effect, not proof that XOR itself is cheaper.
- I changed the swizzle helper to use unsigned chunk arithmetic. Since `pack` is
  nonnegative, this removes unnecessary signed-division correction from the PTX
  swizzle-index calculation while preserving the same chunk mapping.

Timing command:

```bash
GEMMA4_FFN_CUDNN_BENCH_SEED=0x20260522 \
  /tmp/gemma4_ffn_cudnn_bench_tile${tile} 100 20 3 1
```

GPU/tooling:

- GPU: NVIDIA RTX A6000
- CUDA target: `sm_86`
- cuDNN version: `92200`
- Warmup/timing: 20 warmup iterations, 100 timed iterations, 3 trials.
- Timing uses CUDA events through the existing benchmark helper; CUDA graph timing is
  used as the device-time comparison.
- Clock policy: not locked.
- Cache policy: normal benchmark warm-cache behavior.

Tile sweep results:

| Tile | custom graph best ms | custom direct best ms | cuDNN split graph best ms | custom/cuDNN speedup |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1.137898 | 1.141672 | 1.000087 | 0.878890x |
| 2 | 1.077307 | 1.078275 | 1.000453 | 0.928661x |
| 4 | 1.080216 | 1.082335 | 1.001038 | 0.926701x |
| 8 | 1.177239 | 1.175480 | 0.999530 | 0.849046x |

Default tile-2 rerun through the checked-in `make ffn-cudnn-bench` binary after
the scratch handoff fence fix:

```text
device=NVIDIA RTX A6000
shape=tokens1,hidden5376,intermediate21504,seed=0x20260522
iters=100,warmup_iters=20,trials=3
cudnn_frontend=compiled
cudnn_version=92200
geglu_plus_down,11.606174,11.943123,1.000069,1.000217,55.669,4194560,
custom_fused_decode,1.108576,1.114322,1.107599,1.108106,582.862,0,
custom_scratch_clear,0.006160,0.007221,0.001077,0.001090,0.000,0,
cudnn_split_device_ms,1.000069
custom_device_ms,1.107599
custom_minus_clear_device_ms,1.106522
custom_vs_cudnn_split_speedup,0.902916
custom_minus_clear_vs_cudnn_split_speedup,0.903795
```

Chunk-swizzled shared activation timing, short sweep:

iters=50,warmup_iters=10,trials=2

| Swizzle | Tile | custom graph best ms | cuDNN split graph best ms | custom/cuDNN speedup |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 1.165145 | 0.999914 | 0.858188x |
| 1 | 2 | 1.103517 | 0.999734 | 0.905953x |
| 1 | 4 | 1.103430 | 0.999462 | 0.905778x |
| 1 | 8 | 1.199932 | 1.000461 | 0.833765x |
| 0 | 2 | 1.107462 | 1.000806 | 0.903694x |
| 1 | 2 | 1.107693 | 1.000389 | 0.903129x |

Tile-2 versus tile-4 longer rerun with chunk swizzle enabled:

iters=100,warmup_iters=20,trials=3

| Tile | custom graph best ms | cuDNN split graph best ms | custom/cuDNN speedup |
| ---: | ---: | ---: | ---: |
| 2 | 1.107858 | 1.000124 | 0.902754x |
| 4 | 1.109708 | 1.000656 | 0.901729x |

Unsigned swizzle-index cleanup timing, short sanity run:

iters=50,warmup_iters=10,trials=2

| Variant | custom graph best ms | cuDNN split graph best ms | custom/cuDNN speedup |
| --- | ---: | ---: | ---: |
| unsigned chunk swizzle, tile 2 | 1.108769 | 0.999982 | 0.901885x |

Conclusion:

- Activation tiling helps. Tile `2` reduces custom graph time from `1.137898 ms`
  to `1.077307 ms` in the fixed-seed sweep, about a 5.3% custom-kernel improvement.
- Tile `4` is close but slightly slower than tile `2` and increases register use to
  62.
- Tile `8` spills and regresses hard. Do not use spilling tiles without a deeper
  schedule rewrite.
- The FFN decode path now uses chunk-granularity shared activation swizzling by
  default. It is not a meaningful throughput win on this benchmark, but it is
  correct, cheap at tile `2`, and reduced ptxas register allocation for the default
  path from 59 to 50 registers.
- The register drop is a ptxas scheduling artifact in the shared activation preload,
  not a direct bank-conflict win. The unsigned index cleanup is the only code change
  justified by the PTX dump itself.
- Manual race audit found that the global scratch handoff needed every writer thread
  to execute a device fence before thread 0 advances the lock, and every reader thread
  to perform an acquire load before reading `scratch->accum`. That fix preserves
  correctness at the cost of about `0.032 ms` on the checked-in tile-2 rerun.
- cuDNN split graph is still faster for this benchmark (`~1.0001 ms`) than the fused
  custom decode (`~1.1076 ms`), but the gap remains smaller than the original tile-1
  baseline.
- Full cuDNN FFN graph is still unsupported by the frontend planner for this shape in
  this harness; the supported cuDNN comparison remains split GeGLU plus down matmul.

## 2026-05-22 - FFN decode cache-hint ablation

Runtime files:

- `src/gemma4_cuda_utils.cuh`
- `src/gemma4_ffn_decode.cuh`
- `src/gemma4_ffn_decode.cu`
- `src/experiments/gemma4_ffn_cache_policy_bench.cu`
- `Makefile`

Policy IDs:

- `0`: default aligned 128-bit load
- `1`: `__ldg` read-only load, emitted as `ld.global.nc`
- `2`: `__ldca`, emitted as `ld.global.ca`
- `3`: `__ldcg`, emitted as `ld.global.cg`
- `4`: `__ldcs`, emitted as `ld.global.cs`
- `5`: `__ldlu`, emitted as `ld.global.lu`
- `6`: `__ldcv`, emitted as `ld.global.cv`

Default policy tuple remains:

```text
x=1, gate=4, up=4, down=4, residual=1, gamma=1, scratch=0
```

Reasoning:

- `.cv` is only plausible for one-pass weight streams. I did not make it a default for
  `x`, residual, gamma, or scratch because those are small/reused or L2-sensitive.
- Scratch is the global FP32 handoff buffer between CTAs. It should stay cacheable and
  L2-hot; the sweep confirmed non-default hints are harmful there.
- Gamma is the learned RMSNorm weight vector `[5376]`, not the scalar RMS value.

Commands:

```bash
make test-ffn-decode

# Weight stream sweep: gate/up grouped, down independent.
for gu in 0 1 2 3 4 5 6; do
  for down in 0 1 2 3 4 5 6; do
    nvcc -std=c++17 -O3 -arch=sm_86 -Isrc \
      -DGEMMA4_FFN_GATE_LOAD_POLICY=${gu} \
      -DGEMMA4_FFN_UP_LOAD_POLICY=${gu} \
      -DGEMMA4_FFN_DOWN_LOAD_POLICY=${down} \
      src/experiments/gemma4_ffn_cache_policy_bench.cu \
      src/gemma4_ffn_decode.cu -o /tmp/gemma4_ffn_cache_gu${gu}_down${down}
    /tmp/gemma4_ffn_cache_gu${gu}_down${down} 30 5 1
  done
done

# Longer rerun of top candidates.
/tmp/gemma4_ffn_cache_gu4_down4 200 20 5
/tmp/gemma4_ffn_cache_gu5_down0 200 20 5
```

Correctness:

```text
ffn decode tests passed
```

Weight sweep:

- First-pass fastest graph results were clustered around `1.140 ms` and did not show a
  stable `.cv` win.
- Top short-run candidates:
  - `gate/up=1`, `down=2`: `1.140212 ms`
  - `gate/up=5`, `down=0`: `1.140225 ms`
  - `gate/up=3`, `down=3`: `1.140420 ms`
  - `gate/up=4`, `down=6`: `1.140478 ms`
  - current `gate/up/down=4`: `1.141032 ms`

Longer candidate rerun:

```text
name,x,gate,up,down,residual,gamma,scratch,direct_best_ms,direct_avg_ms,graph_best_ms,graph_avg_ms
current,1,4,4,4,1,1,0,1.141580,1.144403,1.140824,1.141919
weight_best,1,5,5,0,1,1,0,1.145933,1.147701,1.140929,1.142647
weight_plus_x,3,5,5,0,1,1,0,1.143503,1.145587,1.141257,1.141895
weight_plus_residual,1,5,5,0,2,1,0,1.146148,1.148361,1.141838,1.142332
weight_plus_gamma,1,5,5,0,1,2,0,1.144858,1.147096,1.142313,1.142973
combined,3,5,5,0,2,2,0,1.145248,1.146566,1.140826,1.141441
```

Small-site one-at-a-time observations around `gate/up=5`, `down=0`:

- `x=cg`, `residual=ca`, and `gamma=ca` looked best in short isolated sweeps, but
  combining them did not produce a robust longer-run win.
- Scratch must remain default. `scratch=cg`, `scratch=lu`, and `scratch=cv` regressed to
  roughly `1.158-1.159 ms` graph time in the short sweep.

Conclusion:

- Keep current defaults for now: `x/residual/gamma=__ldg`, `gate/up/down=__ldcs`,
  `scratch=default`.
- `.cv` is not the best measured choice. It sometimes looked competitive for `down` in
  short runs, but did not survive longer reruns or combination tests.
- Cache hint choice is a noise-level lever for this kernel, roughly sub-1%. The real
  bottleneck is still the scalar-FMA work decomposition, not cache operator selection.

## 2026-05-22 - FFN decode 128-bit vectorized load pass

Runtime files:

- `src/gemma4_ffn_decode.cu`
- `src/gemma4_ffn_decode.cuh`
- `tests/test_ffn_decode.cu`
- `src/experiments/gemma4_ffn_cudnn_bench.cu`

Change:

- Reworked the custom fused decode FFN kernel so each active thread owns one
  hidden pack of `8 x bf16` instead of scalar hidden columns.
- Used the existing `Bf16Packed128` / `Packed128<float>` helpers from
  `gemma4_cuda_utils.cuh`, matching the RMSNorm, RoPE, embedding gather, and decode
  GEMV style.
- Vectorized the global BF16 paths for:
  - token input `x` load into shared memory
  - gate/up weight tile loads
  - down weight tile loads
  - residual and RMS gamma loads
  - residual and normed output stores
- Vectorized the FP32 scratch accumulator merge as two 128-bit float loads/stores per
  hidden pack. The lock remains scalar by design.

Commands:

```bash
make test-ffn-decode ffn-cudnn-bench

mkdir -p build/ptx
nvcc -std=c++17 -O3 -arch=sm_86 -Isrc -ptx src/gemma4_ffn_decode.cu \
  -o build/ptx/gemma4_ffn_decode_vectorized.ptx
nvcc -std=c++17 -O3 -arch=sm_86 -Isrc -Xptxas -v -c src/gemma4_ffn_decode.cu \
  -o build/ptx/gemma4_ffn_decode_vectorized.o

rg -n "ld\\.global|st\\.global|\\.v4\\.u32" \
  build/ptx/gemma4_ffn_decode_vectorized.ptx
cuobjdump --dump-sass build/ptx/gemma4_ffn_decode_vectorized.o | rg -n "LDG|STG"

./build/experiments/gemma4_ffn_cudnn_bench 20 5 2 1
```

PTX/compiler result:

- PTX contains vector BF16 global loads/stores:
  - `ld.global.nc.v4.s32`
  - `ld.global.cs.v4.s32`
  - `st.global.v4.u32`
- PTX contains vector FP32 scratch accumulator loads/stores:
  - `ld.global.v4.f32`
  - `st.global.v4.u32`
- The only scalar global operations left in the searched PTX are the reduction lock
  acquire/add operations.
- SASS contains `LDG.E.128` and `STG.E.128` for the vectorized global memory paths.
- `ptxas`: `59` registers/thread, `0` stack bytes, `0` spill stores, `0` spill loads,
  `11016` bytes shared memory, `1` barrier.

Correctness:

```text
ffn decode tests passed
```

Benchmark:

```text
path,best_ms,avg_ms,graph_best_ms,graph_avg_ms,rough_gib_s,workspace_bytes,max_abs_vs_split
geglu,9.746442,10.325469,0.662168,0.662480,44.193,0,
down,0.658119,0.702142,0.337484,0.337537,327.269,4194560,
geglu_plus_down,9.645019,10.781860,0.999732,1.000199,66.989,4194560,
custom_fused_decode,1.142081,1.144371,1.138668,1.139212,565.762,0,
custom_scratch_clear,0.006851,0.007631,0.001075,0.001096,0.000,0,
overhead_factored_metric,value
cudnn_split_device_ms,0.999732
custom_device_ms,1.138668
custom_scratch_clear_device_ms,0.001075
custom_minus_clear_device_ms,1.137594
custom_vs_cudnn_split_speedup,0.877983
custom_minus_clear_vs_cudnn_split_speedup,0.878813
```

Conclusion:

- The FFN decode kernel now emits 128-bit global memory ops for the meaningful BF16
  and FP32 scratch traffic.
- This is only a small speedup versus the prior custom graph timing (`~1.156 ms` ->
  `~1.139 ms`). The small delta is plausible because the scalar version's adjacent
  per-thread BF16 accesses were already warp-coalesced into the same 32-byte global
  memory segments; vectorization reduced instruction count, not the total bytes or
  memory transaction footprint.
- The kernel is still slower than the cuDNN split graph replay baseline. The next wins
  still need better work decomposition or tensor-core use.

## 2026-05-22 - cuDNN Frontend FFN decode graph probe

Runtime files:

- `src/experiments/gemma4_ffn_cudnn_bench.cu`
- `Makefile`

Change:

- Added `make ffn-cudnn-bench`.
- The benchmark builds:
  - a cuDNN Frontend GeGLU graph:
    `act = (x @ w_gate) * GELU_APPROX_TANH(x @ w_up)`
  - a separate cuDNN Frontend down-projection graph:
    `out = act @ w_down`
  - a full-FFN graph probe:
    `out = ((x @ w_gate) * GELU_APPROX_TANH(x @ w_up)) @ w_down`
  - the current custom decode fused FFN kernel, including residual add and
    RMSNorm epilogue
- The full-FFN path is probed and reported rather than required, because NVIDIA's
  LLM coverage samples say the analogous three-matmul SwiGLU layer graph is not
  supported by the cuDNN Graph API on this path.

Command:

```bash
make ffn-cudnn-bench
./build/experiments/gemma4_ffn_cudnn_bench 100 10 3 1
```

Environment:

- GPU: NVIDIA RTX A6000
- CUDA target: `sm_86`
- cuDNN version reported by benchmark: `92200`
- cuDNN Frontend include path: `/tmp/cudnn-frontend/include`
- Warmup/timed repeats/trials: 10 warmup, 100 timed iterations, 3 trials
- Cache policy: repeated-buffer warm-cache benchmark
- Clock policy: clocks not locked

Results:

```text
full_graph_supported=0,status="cuDNN frontend error for full_ffn.build_plans: [cudnn_frontend] Error: No valid execution plans built."
path,best_ms,avg_ms,graph_best_ms,graph_avg_ms,rough_gib_s,workspace_bytes,max_abs_vs_split
geglu,10.466622,11.574109,0.662417,0.662475,41.152,0,
down,0.641172,0.707548,0.337504,0.337613,335.919,4194560,
geglu_plus_down,12.699577,13.331832,1.000316,1.000455,50.876,4194560,
custom_fused_decode,1.152335,1.157930,1.155604,1.155750,560.728,0,
custom_scratch_clear,0.004377,0.006027,0.001101,0.001116,0.000,0,
full_ffn,-1.000000,,-1.000000,,0.000,,-1
overhead_factored_metric,value
cudnn_split_device_ms,1.000316
custom_device_ms,1.155604
custom_scratch_clear_device_ms,0.001101
custom_minus_clear_device_ms,1.154503
custom_vs_cudnn_split_speedup,0.865622
custom_minus_clear_vs_cudnn_split_speedup,0.866447
cudnn_direct_minus_device_ms,11.699262
custom_direct_minus_device_ms,0.000000
```

Conclusion:

- cuDNN Frontend can build the GeGLU half and down projection as separate graphs.
- cuDNN Frontend cannot build the full three-matmul FFN graph for this A6000/cuDNN path.
- CUDA graph replay is the meaningful cuDNN number here; direct repeated
  `graph.execute` timings include large frontend/host enqueue gaps.
- The split cuDNN Frontend replay baseline is about `1.000 ms/token` for Gemma 4
  31B decode FFN shapes.
- The current custom fused decode kernel is about `1.156 ms/token` with host launch,
  frontend, and wrapper overhead factored out through CUDA graph replay. Scratch clear
  is only about `0.001 ms/token`, so it does not explain the gap.
- The overhead-factored custom speedup versus cuDNN split replay is `0.866x`, so the
  current simple custom kernel is roughly `15.5%` slower than cuDNN's graph-replay split
  baseline, even though it avoids the full activation materialization and includes
  residual+RMSNorm epilogue work.
- The result says the current simple custom kernel is a useful correctness/design
  prototype, not yet a performance win. The next performance work needs to improve the
  work decomposition and tensor-core use, not just compare launch overhead.

## 2026-05-21 - FFN-down decode GEMV K-pack register staging

Runtime files:

- `src/gemma4_matmul_kernels.cu`
- `src/experiments/gemma4_decode_bench.cu`

Change:

- Added `GEMMA4_DECODE_GEMV_BUFFER_STAGES` as a compile-time experiment knob.
- Stage count `1` keeps the original simple loop.
- Stage counts `2-4` use an inlined ring buffer inside `dot_cols()`:
  - prefill `x_stage[stage]` and `w_stage[stage][col]`
  - consume `stage = iter % kStages`
  - load `pack_idx + kStages * Threads` into the same stage before accumulating the
    current copied pack
- No new public launcher or helper function was added.

Reason:

- Test whether K-pack software pipelining can hide HBM latency better than the existing
  per-pack/per-column loop.
- CUDA guide context: occupancy hides latency (pages 71 and 119), register/shared memory
  resources limit residency (pages 28 and 71), launch bounds affect register allocation
  (page 526), and extra local/stack traffic can change kernel behavior even without
  reported spill stores/loads.

Commands:

```bash
make decode-bench
GEMMA4_DECODE_BENCH_SEED=0x1234 \
  ./build/experiments/gemma4_decode_bench ffn_down 80 20 3

nvcc -std=c++17 -O3 -arch=sm_86 -DGEMMA4_DECODE_GEMV_BUFFER_STAGES=2 \
  -Isrc src/experiments/gemma4_decode_bench.cu src/gemma4_matmul_kernels.cu \
  -lcublas -lcudnn -o /tmp/gemma4_stage2/gemma4_decode_bench
GEMMA4_DECODE_BENCH_SEED=0x1234 \
  /tmp/gemma4_stage2/gemma4_decode_bench ffn_down 80 20 3

nvcc -std=c++17 -O3 -arch=sm_86 -DGEMMA4_DECODE_GEMV_BUFFER_STAGES=3 \
  -Isrc src/experiments/gemma4_decode_bench.cu src/gemma4_matmul_kernels.cu \
  -lcublas -lcudnn -o /tmp/gemma4_stage3/gemma4_decode_bench
GEMMA4_DECODE_BENCH_SEED=0x1234 \
  /tmp/gemma4_stage3/gemma4_decode_bench ffn_down 80 20 3

nvcc -std=c++17 -O3 -arch=sm_86 -DGEMMA4_DECODE_GEMV_BUFFER_STAGES=4 \
  -Isrc src/experiments/gemma4_decode_bench.cu src/gemma4_matmul_kernels.cu \
  -lcublas -lcudnn -o /tmp/gemma4_stage4/gemma4_decode_bench
GEMMA4_DECODE_BENCH_SEED=0x1234 \
  /tmp/gemma4_stage4/gemma4_decode_bench ffn_down 80 20 3
```

Environment:

- GPU: NVIDIA RTX A6000
- CUDA target: `sm_86`
- Clocks: not locked
- Cache policy: repeated-buffer warm-cache benchmark
- Warmup/timed repeats/trials: 20 warmup, 80 timed iterations, 3 trials

Resource usage for FFN-down identity decode:

| Stages | Registers/thread | Stack bytes/thread | Shared bytes/block |
| ---: | ---: | ---: | ---: |
| 1 | 58 | 0 | 1024 |
| 2 | 55 | 288 | 1024 |
| 3 | 58 | 432 | 1024 |
| 4 | 55 | 576 | 1024 |

Results:

| Stages | Custom best ms | Custom avg ms | Effective weight GB/s |
| ---: | ---: | ---: | ---: |
| 1 | 0.327842 | 0.327962 | 705.251 |
| 2 | 1.029579 | 1.030183 | 224.569 |
| 3 | 1.092698 | 1.092928 | 211.596 |
| 4 | 1.103177 | 1.103208 | 209.586 |

Correctness:

- Custom swizzle16, cuBLAS GEMV, and cuBLAS GEMM M=1 matched the custom output with zero
  reported absolute/relative difference in all staged runs.
- cuDNN conv1x1 retained the same expected BF16 accumulation-order differences.

Conclusion:

- Naive K-pack N-buffering is a large regression.
- The staged arrays do not stay entirely in registers. They create per-thread stack
  frames, so the attempted latency hiding adds local-memory traffic and extra copy work.
- The original stage-1 loop remains the right default. If K-pack lookahead is revisited,
  it needs a hand-shaped 2-stage variant with fewer live packs, lower `ColsPerBlock`, or
  a different thread/block decomposition instead of `w_stage[Stages][ColsPerBlock]`.

## 2026-05-21 - FFN-down decode GEMV register-lookahead weight load

Runtime files:

- `src/gemma4_matmul_kernels.cu`
- `src/experiments/gemma4_decode_bench.cu`

Change:

- Changed the decode GEMV inner column loop to issue the next column's weight-pack load
  before accumulating the current column.
- This is a register lookahead/software scheduling tweak only; it does not add
  `cp.async`, shared-memory staging, or a new public launcher.

Environment:

- GPU: NVIDIA RTX A6000
- Clocks: not locked
- CUDA target: `sm_86`
- Cache policy: repeated-buffer warm-cache benchmark
- Correctness reference: existing benchmark diffs against cuBLAS GEMV/GEMM and swizzled
  custom path

Command:

```bash
make decode-bench
GEMMA4_DECODE_BENCH_SEED=0x1234 \
  ./build/experiments/gemma4_decode_bench ffn_down 80 20 3
```

Results:

| Path | Best ms | Avg ms | Effective weight GB/s |
| --- | ---: | ---: | ---: |
| Custom GEMV with register lookahead | 0.327549 | 0.327685 | 705.883 |
| Custom swizzle16 with register lookahead | 0.327643 | 0.584443 | 705.679 |
| cuBLAS BF16 GEMV | 0.330064 | 0.465015 | 700.504 |
| cuBLAS BF16 GEMM M=1 | 0.573587 | 0.655130 | 403.097 |
| cuDNN BF16 conv1x1 | 0.332157 | 0.555408 | 696.090 |

Correctness:

- Custom swizzle16, cuBLAS GEMV, and cuBLAS GEMM M=1 matched the custom output with zero
  reported absolute/relative difference in this run.
- cuDNN conv1x1 differed by `max_abs_diff=0.25`, `mean_abs_diff=0.000392091`,
  `max_rel_diff=0.00763359`.

Conclusion:

- The register-lookahead edit produced only a tiny best-time improvement versus the prior
  logged FFN-down custom result (`0.327884 ms` -> `0.327549 ms`, about `0.10%`).
- This is within the range where repeated benchmark and `ncu` confirmation are needed
  before treating it as a real optimization.

## 2026-05-21 - RoPE 128-bit pack load baseline

Runtime file:

- `src/gemma4_rope.cu`
- `src/gemma4_rope.cuh`
- `tests/test_rope.cu`
- `src/experiments/gemma4_rope_bench.cu`
- `docs/rope.md`

Change:

- Replaced the scalar hot-path RoPE loop with a packed path that processes eight rotary
  pairs per active thread.
- Q/K halves now use `Bf16Packed128` loads and stores on the full-pack path.
- Cos/sin tables use two `float4` loads each per eight-pair pack.
- Kept the scalar pair path as a tail fallback for non-multiple-of-eight rotary widths.
- Reduced the RoPE thread block from 128 threads to one warp, because Gemma 4's sliding
  and global rotary widths need only 16 and 8 packed lanes per head respectively.
- Updated the forward-layout RoPE entry point, tests, docs, and benchmark to use compact
  cos/sin rows of `rotary_dim / 2` instead of full `head_dim` rows. This removes unused
  table storage in the forward path:
  - sliding rows: 128 floats instead of 256
  - global rows: 64 floats instead of 512

Build and inspection commands:

```bash
make test-rope
mkdir -p build/ptx
nvcc -std=c++17 -O3 -arch=sm_86 -Isrc -ptx src/gemma4_rope.cu \
  -o build/ptx/gemma4_rope_vectorized.ptx
nvcc -std=c++17 -O3 -arch=sm_86 -Isrc -Xptxas -v -c src/gemma4_rope.cu \
  -o build/ptx/gemma4_rope_vectorized.o
```

PTX/compiler result:

- Full-pack path emits `ld.global.v4.u32` and `st.global.v4.u32` for BF16 Q/K halves.
- Full-pack path emits `ld.global.nc.v4.f32` for cos/sin table loads.
- `ptxas` reports no spills or stack frame.
- Register count increased versus the scalar version:
  - low-level layout kernel: 58 registers
  - forward-layout kernel: 60 registers

Correctness:

```text
rope tests passed
```

Benchmark command:

```bash
make rope-bench
GEMMA4_ROPE_BENCH_SEED=0x20260521 \
  ./build/experiments/gemma4_rope_bench 100 20 3 1024 1 1
```

Environment:

- GPU: NVIDIA RTX A6000
- CUDA/NVCC: 12.9.86
- Build flags: `-std=c++17 -O3 -arch=sm_86`
- Warmup/timed repeats/trials: 20 warmup, 100 timed iterations, 3 trials
- Cache policy: repeated-buffer warm-cache benchmark
- Clock policy: clocks not locked
- Correctness tolerance: benchmark compares against cuDNN BF16-table path and reports
  `0.0078125` max absolute difference for all measured cases.

Selected custom graph timings:

| Case | Seq | Graph ms | Effective GiB/s |
| --- | ---: | ---: | ---: |
| Sliding | 1 | 0.001741 | 52.573 |
| Sliding | 256 | 0.015432 | 1518.728 |
| Sliding | 1024 | 0.076612 | 1223.706 |
| Global | 1 | 0.001575 | 21.793 |
| Global | 256 | 0.006732 | 1305.534 |
| Global | 1024 | 0.029884 | 1176.405 |

Compact-table follow-up:

```bash
make test-rope
make rope-bench
GEMMA4_ROPE_BENCH_SEED=0x20260521 \
  ./build/experiments/gemma4_rope_bench 100 20 3 1024 1 1
```

Correctness:

```text
rope tests passed
```

Selected compact-table custom graph timings:

| Case | Seq | Graph ms | Effective GiB/s | Speedup vs packed/full-table run |
| --- | ---: | ---: | ---: | ---: |
| Sliding | 1 | 0.001754 | 52.199 | 0.993x |
| Sliding | 256 | 0.016488 | 1421.488 | 0.936x |
| Sliding | 1024 | 0.076759 | 1221.355 | 0.998x |
| Global | 1 | 0.001643 | 20.902 | 0.959x |
| Global | 256 | 0.006704 | 1311.080 | 1.004x |
| Global | 1024 | 0.029817 | 1179.069 | 1.002x |

Conclusion:

- The C++ pack path successfully produces 128-bit global memory ops without inline PTX.
- The change is correctness-clean in the existing RoPE tests.
- The higher register count is the main tradeoff to watch. It did not spill in this
  build, but this shape should be benchmarked against the scalar kernel before treating
  it as a settled performance win.
- Compact cos/sin tables fixed the unused table storage in the forward path. Custom
  kernel speed was essentially neutral because the kernel already loaded only the used
  rotary columns; the practical win is lower table allocation footprint and cleaner
  strides, especially for global p-RoPE.

Cache-order follow-up:

- Changed the RoPE grid from `(rows, heads)` to `(heads, rows)` so heads are the fastest
  grid dimension, attempting to improve temporal locality for each position's cos/sin
  row.
- Changed packed Q/K input loads from direct pack loads to streaming pack loads, while
  keeping output stores normal.
- PTX confirmed the intended lowering:
  - row now uses `%ctaid.y`
  - head now uses `%ctaid.x`
  - Q/K packed loads emit `ld.global.cs.v4.s32`
  - cos/sin packed loads remain `ld.global.nc.v4.f32`
- `ptxas` reported unchanged register/spill profile: 58 registers for the low-level
  kernel, 60 for the forward-layout kernel, 0 spills.

Correctness:

```text
rope tests passed
```

Benchmark command:

```bash
GEMMA4_ROPE_BENCH_SEED=0x20260521 \
  ./build/experiments/gemma4_rope_bench 100 20 3 1024 1 1
```

Selected custom graph timings versus the compact-table packed run:

| Case | Seq | Prior graph ms | Cache-order graph ms | Speedup |
| --- | ---: | ---: | ---: | ---: |
| Sliding | 1 | 0.001754 | 0.001736 | 1.010x |
| Sliding | 256 | 0.016488 | 0.015046 | 1.096x |
| Sliding | 1024 | 0.076759 | 0.076900 | 0.998x |
| Global | 1 | 0.001643 | 0.001582 | 1.039x |
| Global | 256 | 0.006704 | 0.006824 | 0.982x |
| Global | 1024 | 0.029817 | 0.031695 | 0.941x |

Conclusion:

- The cache-order change is not a broad speedup. It helped sliding `seq=256` in this
  short run, was neutral for sliding `seq=1024`, and regressed global `seq=1024`.
- Do not treat head-fast grid order plus streaming Q/K loads as a retained win without a
  cleaner A/B split. The two changes should be separated in a later experiment because
  the grid order may help cos/sin locality while `ld.global.cs` may hurt Q/K reuse or
  scheduling.

Split A/B follow-up:

- Added compile-time controls:
  - `GEMMA4_ROPE_HEAD_FAST_GRID`
  - `GEMMA4_ROPE_QK_STREAMING_PACK`
- Built four variants from the same source:
  - baseline: row-fast grid, normal Q/K loads
  - headfast: head-fast grid, normal Q/K loads
  - loadcs: row-fast grid, streaming Q/K loads
  - both: head-fast grid, streaming Q/K loads

Build pattern:

```bash
nvcc -std=c++17 -O3 -arch=sm_86 -Isrc \
  -DGEMMA4_ROPE_HEAD_FAST_GRID=<0|1> \
  -DGEMMA4_ROPE_QK_STREAMING_PACK=<0|1> \
  src/experiments/gemma4_rope_bench.cu src/gemma4_rope.cu -lcudnn \
  -o build/experiments/rope_variants/<variant>
```

Benchmark command for each variant:

```bash
GEMMA4_ROPE_BENCH_SEED=0x20260521 \
  ./build/experiments/rope_variants/<variant> 100 20 3 1024 1 1
```

Custom CUDA graph timings:

| Case | Seq | Baseline | Head-fast only | streaming-pack only | Both | Best |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Sliding | 1 | 0.001666 | 0.001707 | 0.001553 | 0.001646 | streaming-pack |
| Sliding | 4 | 0.001672 | 0.001959 | 0.001639 | 0.001901 | streaming-pack |
| Sliding | 16 | 0.002134 | 0.002122 | 0.002083 | 0.002015 | both |
| Sliding | 64 | 0.003097 | 0.003188 | 0.003091 | 0.003065 | both |
| Sliding | 256 | 0.016482 | 0.016036 | 0.015215 | 0.014943 | both |
| Sliding | 1024 | 0.076787 | 0.076737 | 0.075305 | 0.076946 | streaming-pack |
| Global | 1 | 0.001496 | 0.001556 | 0.001711 | 0.001563 | baseline |
| Global | 4 | 0.001565 | 0.001646 | 0.001601 | 0.001651 | baseline |
| Global | 16 | 0.001658 | 0.001820 | 0.001638 | 0.001956 | streaming-pack |
| Global | 64 | 0.002805 | 0.002773 | 0.002658 | 0.002917 | streaming-pack |
| Global | 256 | 0.006691 | 0.006674 | 0.006654 | 0.006777 | streaming-pack |
| Global | 1024 | 0.029790 | 0.031470 | 0.029804 | 0.031625 | baseline |

Conclusion:

- Head-fast grid order alone was not a clear win and hurt several small/large cases.
- The streaming-pack load variant was the best isolated change overall: it improved sliding
  `seq=1024` by about `2%`, improved several mid-size sliding/global cases, and was
  essentially tied with baseline at global `seq=1024`.
- The retained default is row-fast grid with streaming pack loads enabled. Head-fast grid remains
  available behind `GEMMA4_ROPE_HEAD_FAST_GRID=1` for future experiments.

## 2026-05-21 - RoPE FP32 accumulation numerical consistency

Runtime files:

- `src/gemma4_rope.cu`
- `src/experiments/gemma4_rope_numeric_check.cu`

Reason:

- Check whether the RoPE element math should stay in FP32 before rounding back to BF16.
- CUDA guide reference points used for the standard:
  - FMA can avoid precision loss during cancellation (CUDA Programming Guide p.583).
  - `x * y + z -> __fmaf_rn(x, y, z)` has a stated `0 ULP` mapping for float basic
    operations (p.594).
  - BF16 has the same range as FP32 but only 7 bits of precision (p.574), so intermediate
    BF16 rounding is expected to be visibly worse.

Method:

- Randomize BF16 Q/K values and FP32 cos/sin tables.
- Run the actual GPU RoPE kernel.
- Compare every rotated BF16 output against:
  - a CPU FP32/FMA reference rounded once to BF16, matching the current kernel's math;
  - a double reference rounded to BF16;
  - a simulated BF16-intermediate path that rounds both products before the add.
- Also check global p-RoPE's NoPE tail for bit-exact preservation.
- This is a numerical-consistency experiment, not a timing benchmark.

Build and run:

```bash
nvcc -std=c++17 -O3 -arch=sm_86 -Isrc \
  src/experiments/gemma4_rope_numeric_check.cu src/gemma4_rope.cu \
  -o build/experiments/gemma4_rope_numeric_check
./build/experiments/gemma4_rope_numeric_check 64 53 2 0x20260521
```

Environment:

- GPU: NVIDIA RTX A6000
- CUDA/NVCC: 12.9.86
- Build flags: `-std=c++17 -O3 -arch=sm_86`
- Shapes: Gemma 4 sliding (`head_dim=256`, full RoPE) and global
  (`head_dim=512`, `rotary_dim=128` p-RoPE)

Results:

| Case | Rotated outputs | GPU == FP32/FMA | GPU == double | BF16-intermediate == double | GPU mean BF16 ULP vs double | BF16-intermediate mean BF16 ULP vs double | NoPE exact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Sliding | 83,361,792 | 1.00000000 | 0.99998682 | 0.66429912 | 0.00002023 | 10.25257449 | 1.00000000 |
| Global | 31,260,672 | 1.00000000 | 0.99998650 | 0.66435267 | 0.00001676 | 10.17400861 | 1.00000000 |

SASS check:

- The hot path uses FP32 `FMUL` plus `FFMA`, followed by `F2FP.BF16.PACK_AB` conversion
  for stores.

Conclusion:

- Keep the current FP32 RoPE arithmetic and final BF16 rounding.
- The GPU kernel is bit-exact against the FP32/FMA reference over this randomized sweep.
- Simulated BF16 intermediates are much less consistent: only about `66.4%` exact versus
  the double-rounded BF16 reference, with about `10` mean BF16 ULP error.
- Global p-RoPE's unrotated NoPE dimensions remained bit-exact.

## 2026-05-21 - FFN-down decode GEMV kernel comparison

Runtime files:

- `src/gemma4_matmul_kernels.cu`
- `src/experiments/gemma4_decode_bench.cu`

Reason:

- Check the project GEMV kernel for the FFN-down shape after the tinygrad/cuBLAS/cuBLASLt
  FFN-down matmul sweeps.
- This is the decode `M=1` case, not the prefill `M > 1` matmul case.

Shape:

- FFN-down decode GEMV: `K=21504`, `N=5376`, BF16 input/weights/output, FP32
  accumulation.

Command:

```bash
make decode-bench
GEMMA4_DECODE_BENCH_SEED=0x1234 \
  ./build/experiments/gemma4_decode_bench ffn_down 80 20 3
```

Results:

| Path | Best ms | Avg ms | Effective weight GB/s | Speedup vs custom |
| --- | ---: | ---: | ---: | ---: |
| Custom GEMV | 0.327884 | 0.327917 | 705.160 | 1.000x |
| Custom swizzle16 GEMV | 0.327907 | 0.327931 | 705.112 | 0.9999x |
| cuBLAS BF16 GEMV | 0.329957 | 0.330215 | 700.731 | 0.9937x |
| cuBLAS BF16 GEMM M=1 | 0.329998 | 0.330041 | 700.645 | 0.9936x |
| cuDNN BF16 conv1x1 | 0.331502 | 0.331585 | 697.465 | 0.9891x |

Correctness:

- Custom, custom swizzle16, cuBLAS GEMV, and cuBLAS GEMM M=1 matched with zero reported
  absolute/relative difference in this run.
- cuDNN conv1x1 differed by `max_abs_diff=0.25`, `mean_abs_diff=0.000392091`,
  `max_rel_diff=0.00763359`.

Conclusion:

- For FFN-down decode `M=1`, the project custom GEMV kernel is slightly faster than
  cuBLAS GEMV, cuBLAS GEMM M=1, and cuDNN conv1x1 on this run.
- The win is small: about `1.006x` over cuBLAS and `1.011x` over cuDNN.
- This result should not be extrapolated to prefill `M > 1`; the GEMV kernel is a
  single-token decode path.

## 2026-05-21 - tinygrad late-eval BF16 matmul vs cuBLASLt

Runtime file:

- `src/experiments/tinygrad_late_eval_bench.py`

Change:

- Added a cuBLASLt comparison path to the same benchmark file.
- The Python script builds a tiny C++ shared helper under `build/experiments/` on first
  use so cuBLASLt descriptors, heuristics, and opaque algorithm structs stay in C++.
- cuBLASLt uses the same column-major equivalence as the simple cuBLAS path:
  `C_col[n,m] = B_col[n,k] * A_col[k,m]`, matching row-major
  `C[m,n] = A[m,k] * B[k,n]` without explicit transposes.
- Timing uses CUDA events around the cuBLASLt matmul on a nonblocking stream, with warmup
  outside the timing loop.

Shape:

- Gemma 4 FFN-down-like GEMM: `M x 21504` by `21504 x 5376`.
- Data type: BF16 inputs, FP32 accumulation/output.

Environment:

- GPU: NVIDIA RTX A6000
- Clocks: not locked
- Cache policy: repeated-buffer warm-cache microbenchmark
- Warmup/timed repeats: `WARMUP=3`, `CNT=5`
- tinygrad search: `SEARCH_BEAM=2`
- cuBLASLt workspace: default `67108864` bytes
- cuBLASLt heuristic request count: default `64`

Command:

```bash
DEV=CUDA M_LIST=1,2,4,8,16,32,64,128,256,288,304,312,320,384,512 \
  N=5376 K=21504 DTYPE_IN=bfloat16 DTYPE_ACC=float WARMUP=3 CNT=5 \
  SEARCH_BEAM=2 COMPARE_CUBLAS=1 COMPARE_CUBLASLT=1 \
  python3 src/experiments/tinygrad_late_eval_bench.py
```

Results:

| M | tinygrad best us | cuBLAS best us | cuBLASLt best us | tiny/cuBLAS | tiny/cuBLASLt | Winner vs cuBLASLt |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 343.30 | 2842.05 | 359.14 | 8.279x | 1.046x | tinygrad |
| 2 | 377.34 | 2913.57 | 343.10 | 7.721x | 0.909x | cuBLASLt |
| 4 | 475.87 | 2920.38 | 359.04 | 6.137x | 0.754x | cuBLASLt |
| 8 | 364.03 | 2841.02 | 352.22 | 7.804x | 0.968x | cuBLASLt |
| 16 | 449.41 | 3366.94 | 351.39 | 7.492x | 0.782x | cuBLASLt |
| 32 | 526.53 | 3222.88 | 350.53 | 6.121x | 0.666x | cuBLASLt |
| 64 | 792.10 | 2595.26 | 363.74 | 3.276x | 0.459x | cuBLASLt |
| 128 | 1081.31 | 2038.46 | 404.77 | 1.885x | 0.374x | cuBLASLt |
| 256 | 1696.80 | 2583.97 | 448.06 | 1.523x | 0.264x | cuBLASLt |
| 288 | 1703.17 | 3745.79 | 702.85 | 2.199x | 0.413x | cuBLASLt |
| 304 | 2709.09 | 2378.91 | 705.92 | 0.878x | 0.261x | cuBLASLt |
| 312 | 1864.90 | 2406.37 | 715.30 | 1.290x | 0.384x | cuBLASLt |
| 320 | 1892.35 | 2277.60 | 714.56 | 1.204x | 0.378x | cuBLASLt |
| 384 | 1903.49 | 797.89 | 791.94 | 0.419x | 0.416x | cuBLASLt |
| 512 | 2366.78 | 813.82 | 802.05 | 0.344x | 0.339x | cuBLASLt |

Conclusion:

- cuBLASLt is the real baseline here. tinygrad only beat cuBLASLt at `M=1`, and only by
  about `4.6%`.
- From `M=2` onward, cuBLASLt wins in this sweep.
- The earlier tinygrad wins over simple cuBLAS `GemmEx` do not survive against cuBLASLt.
- For the FFN-down shape, there is no useful tinygrad integration region from this sweep
  unless a later generated kernel beats cuBLASLt rather than plain cuBLAS.

## 2026-05-21 - tinygrad late-eval BF16 matmul vs simple cuBLAS GemmEx

Runtime file:

- `src/experiments/tinygrad_late_eval_bench.py`

Change:

- Added a simple ctypes cuBLAS `cublasGemmEx` benchmark path to the existing tinygrad
  late-eval matmul bench.
- The cuBLAS path allocates raw device buffers, warms up outside the timed loop, and uses
  CUDA events on the same nonblocking stream as the cuBLAS call.
- The cuBLAS call uses column-major equivalence for row-major buffers:
  `C_col[n,m] = B_col[n,k] * A_col[k,m]`, matching row-major
  `C[m,n] = A[m,k] * B[k,n]` without explicit transposes.
- The ctypes path pins to CUDA 12 libraries when available and calls `cudaSetDevice(0)`
  before cuBLAS timing to restore the CUDA runtime primary context after tinygrad driver
  API work.

Shape:

- Gemma 4 FFN-down-like GEMM: `M x 21504` by `21504 x 5376`.
- Data type: BF16 inputs, FP32 accumulation/output.

Environment:

- GPU: NVIDIA RTX A6000
- Clocks: not locked
- Cache policy: repeated-buffer warm-cache microbenchmark
- Timing: tinygrad internal event timing for generated kernels; CUDA event timing for
  simple cuBLAS `GemmEx`
- Warmup/timed repeats: `WARMUP=3`, `CNT=5`
- tinygrad search: `SEARCH_BEAM=2`

Commands:

```bash
DEV=CUDA M_LIST=16,32,64,128,256,512,1024 N=5376 K=21504 \
  DTYPE_IN=bfloat16 DTYPE_ACC=float WARMUP=3 CNT=5 SEARCH_BEAM=2 \
  COMPARE_CUBLAS=1 python3 src/experiments/tinygrad_late_eval_bench.py

DEV=CUDA M_LIST=288,320,352,384,416,448,480 N=5376 K=21504 \
  DTYPE_IN=bfloat16 DTYPE_ACC=float WARMUP=3 CNT=5 SEARCH_BEAM=2 \
  COMPARE_CUBLAS=1 python3 src/experiments/tinygrad_late_eval_bench.py

DEV=CUDA M_LIST=296,304,312 N=5376 K=21504 \
  DTYPE_IN=bfloat16 DTYPE_ACC=float WARMUP=3 CNT=5 SEARCH_BEAM=2 \
  COMPARE_CUBLAS=1 python3 src/experiments/tinygrad_late_eval_bench.py
```

Results:

| M | tinygrad best us | cuBLAS best us | tiny/cuBLAS | Winner |
| ---: | ---: | ---: | ---: | --- |
| 16 | 436.80 | 2552.70 | 5.844x | tinygrad |
| 32 | 487.10 | 1985.60 | 4.076x | tinygrad |
| 64 | 801.79 | 2698.75 | 3.366x | tinygrad |
| 128 | 1072.64 | 2784.58 | 2.596x | tinygrad |
| 256 | 1676.06 | 2228.58 | 1.330x | tinygrad |
| 288 | 1650.46 | 2832.67 | 1.716x | tinygrad |
| 296 | 2760.03 | 3464.51 | 1.255x | tinygrad |
| 304 | 2696.10 | 2237.95 | 0.830x | cuBLAS |
| 312 | 1845.15 | 2550.53 | 1.382x | tinygrad |
| 320 | 1899.81 | 1881.73 | 0.990x | cuBLAS |
| 352 | 2127.07 | 1975.74 | 0.929x | cuBLAS |
| 384 | 1930.05 | 797.47 | 0.413x | cuBLAS |
| 416 | 2231.49 | 819.20 | 0.367x | cuBLAS |
| 448 | 2337.79 | 814.24 | 0.348x | cuBLAS |
| 480 | 2129.70 | 809.38 | 0.380x | cuBLAS |
| 512 | 2370.43 | 813.63 | 0.343x | cuBLAS |
| 1024 | 6015.20 | 4337.31 | 0.721x | cuBLAS |

Conclusion:

- Against this intentionally simple cuBLAS `GemmEx` baseline, tinygrad's beam-2 generated
  kernel wins clearly through `M=288` and still wins at `M=296` and `M=312`.
- There is not a clean monotonic crossover because both tinygrad and cuBLAS select
  different generated/library paths by shape. The first measured losing point is `M=304`;
  `M=320` and larger measured points lose except the earlier `M=312` outlier.
- Practical cutoff from this sweep: treat `M <= 288` as the stable tinygrad-win region
  for this simple cuBLAS comparison, with shape-specific wins possible around `M=296-312`.
- This does not supersede prior cuBLASLt results. The earlier cuBLASLt FFN-down baseline
  remains much stronger and is still the relevant integration target.

## 2026-05-20 - tinygrad late-eval BF16 FFN-down bench

Runtime file:

- `src/experiments/tinygrad_late_eval_bench.py`

Reason:

- Exercise tinygrad's normal lazy/late evaluation path without using `experiments/tinygrad/extra`.
- The file mirrors the internal path from tinygrad proper:
  `Tensor.empty -> Tensor.matmul -> Tensor.schedule_linear -> compile_linear -> run_linear`.
- `compile_linear` and `run_linear` are copied into the file and call the same tinygrad
  pattern matchers (`pm_beam`, `pm_compile`, `pm_optimize_local_size`, `pm_exec`) as the
  upstream engine.
- Shape is Gemma 4 FFN-down-like: `M x 21504` by `21504 x 5376`.

Environment:

- GPU: NVIDIA RTX A6000
- tinygrad checkout: `experiments/tinygrad`, commit `9e88b08`
- Clocks: not locked
- Cache policy: repeated-buffer warm-cache timing
- Data type: BF16 inputs, FP32 accumulation/output through `Tensor.matmul(..., dtype=dtypes.float)`

Commands:

```bash
DEV=CUDA M_LIST=64,128 N=5376 K=21504 DTYPE_IN=bfloat16 DTYPE_ACC=float \
  WARMUP=2 CNT=5 SEARCH_BEAM=0 \
  python3 src/experiments/tinygrad_late_eval_bench.py

DEV=CUDA M_LIST=64,128 N=5376 K=21504 DTYPE_IN=bfloat16 DTYPE_ACC=float \
  WARMUP=1 CNT=3 SEARCH_BEAM=2 \
  python3 src/experiments/tinygrad_late_eval_bench.py

python3 -m py_compile src/experiments/tinygrad_late_eval_bench.py
```

Results:

| M | Search beam | best us | median us | TFLOP/s | Program |
| ---: | ---: | ---: | ---: | ---: | --- |
| 64 | 0 | 2373.44 | 2385.28 | 6.23 | `r_42_32_4_2_2_4_4_1344_2` |
| 128 | 0 | 2751.23 | 2766.24 | 10.76 | `r_2_42_32_4_2_2_4_4_1344_2` |
| 64 | 2 | 802.30 | 805.38 | 18.44 | `r_4_84_32_2_2_2_4_1344_2` |
| 128 | 2 | 1065.73 | 1066.43 | 27.77 | `r_4_96_32_2_2_7_2_1344_2` |

Conclusion:

- The one-file late-eval reproduction works and uses tinygrad proper instead of `extra`.
- Beam search improves the generated BF16 kernel substantially, but the result is still
  far behind the logged cuBLASLt FFN-down timings (`M=64` around `0.342 ms`, `M=128`
  around `0.351-0.377 ms`).
- This path is useful for inspecting tinygrad's generated program choices, but it is not
  currently a replacement candidate for the FFN-down cuBLASLt path.

Direct library check:

```bash
PYTHONPATH=experiments/tinygrad DEV=CUDA python3 - <<'PY'
# Direct Tensor.matmul(...).realize() loop with DEBUG=2 for tinygrad event timing.
PY
```

Direct tinygrad `Tensor.matmul` with BF16 inputs and FP32 accumulation, `BEAM=0`,
`WARMUP=3`, `CNT=10`:

| M | tinygrad best ms | tinygrad median ms | TFLOP/s | Logged cuBLASLt ms | tinygrad/cuBLASLt |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 2.3799 | 2.3950 | 6.22 | 0.3419 | 0.14x |
| 128 | 2.7416 | 2.7553 | 10.79 | 0.3512 | 0.13x |
| 256 | 2.8034 | 2.8196 | 21.11 | 0.4945 | 0.18x |
| 512 | 4.5297 | 4.6244 | 26.13 | 0.9604 | 0.21x |

Confirmation rerun with `WARMUP=5`, `CNT=20`:

| M | tinygrad best ms | tinygrad median ms | tinygrad worst ms | TFLOP/s |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 2.3809 | 2.3906 | 2.4171 | 6.22 |
| 128 | 2.7396 | 2.7535 | 2.7656 | 10.80 |
| 256 | 2.7986 | 2.8047 | 2.8207 | 21.15 |
| 512 | 4.4820 | 4.5934 | 4.6854 | 26.41 |

## 2026-05-20 - Fresh tinygrad max_matmul Gemma FFN-down M sweep

Runtime files:

- `experiments/tinygrad/extra/gemm/max_matmul.py`
- `experiments/tinygrad/extra/gemm/max_kernels/`

Source:

- Re-cloned `https://github.com/tinygrad/tinygrad.git` into `experiments/tinygrad`.
- tinygrad commit: `9e88b08`.
- Local compatibility patch: `max_matmul.py` now uses `CUDADevice("cuda:0").compiler`
  because the fresh checkout no longer exports `CUDACompiler` from
  `tinygrad.runtime.ops_cuda`.
- Local benchmark patch: added `VALIDATE=0` to skip the NumPy reference matmul for
  larger timing-only sweeps.

Shape:

- Gemma 4 FFN-down-like GEMM: `M x 21504` by `21504 x 5376`.
- `M` is the token count / prefill row count.

Environment:

- GPU: NVIDIA RTX A6000
- Driver: 580.126.16
- CUDA/NVCC: 12.9 (`nvcc V12.9.86`)
- Clocks: not locked
- Cache policy: repeated-buffer warm-cache microbenchmark
- Timing: tinygrad `CUDAProgram(..., wait=True)` CUDA event timing; script reports min
  over `CNT`

FP16 input, FP32 accumulate/output, upstream `GEMM_VARIATION=max`:

Command pattern:

```bash
PYTHONPATH=. CUDA=1 VALIDATE=0 GEMM_VARIATION=max \
  DTYPE_IN=half DTYPE_OUT=float DTYPE_ACC=float \
  M=<M> N=5376 K=21504 CNT=30 INPUT=RAND \
  python3 extra/gemm/max_matmul.py
```

| M | min us | TFLOP/s |
| ---: | ---: | ---: |
| 64 | 347.26 | 42.61 |
| 128 | 366.78 | 80.69 |
| 192 | 508.93 | 87.23 |
| 256 | 635.17 | 93.19 |
| 320 | 901.31 | 82.09 |
| 384 | 1066.08 | 83.28 |
| 448 | 1216.35 | 85.16 |
| 512 | 1223.23 | 96.78 |
| 576 | 1481.28 | 89.91 |
| 640 | 1562.27 | 94.72 |
| 704 | 1728.74 | 94.16 |
| 768 | 1817.76 | 97.69 |
| 832 | 2057.86 | 93.48 |
| 896 | 2232.64 | 92.79 |
| 960 | 2337.34 | 94.96 |
| 1024 | 2433.31 | 97.30 |
| 1152 | 2838.46 | 93.84 |
| 1280 | 3237.86 | 91.40 |
| 1536 | 4312.86 | 82.34 |
| 1792 | 5229.02 | 79.24 |
| 2048 | 5979.49 | 79.19 |

Validation note:

- Default validation passed for the first random-input runs at `M=64,128,192,256`.
- The default `rtol=1e-3, atol=1e-2` check failed on isolated elements at `M=320` and
  `M=384` with max absolute violations around `0.010-0.013`; timings were still
  recorded.
- The wider sweep used `VALIDATE=0`, so those numbers are timing-only.

FP16 input, FP16 accumulate/output, upstream `GEMM_VARIATION=max`:

Command pattern:

```bash
PYTHONPATH=. CUDA=1 VALIDATE=0 GEMM_VARIATION=max \
  DTYPE_IN=half DTYPE_OUT=half DTYPE_ACC=half \
  M=<M> N=5376 K=21504 CNT=30 INPUT=RAND \
  python3 extra/gemm/max_matmul.py
```

| M | min us | TFLOP/s |
| ---: | ---: | ---: |
| 256 | 791.62 | 74.77 |
| 512 | 810.30 | 146.09 |
| 768 | 1584.99 | 112.03 |
| 1024 | 1578.72 | 149.97 |
| 1280 | 2347.46 | 126.07 |
| 1536 | 2374.30 | 149.58 |
| 1792 | 3122.05 | 132.71 |
| 2048 | 3157.95 | 149.95 |

Representative FP32-accumulating variant comparison:

Command pattern:

```bash
PYTHONPATH=. CUDA=1 VALIDATE=0 GEMM_VARIATION=<variant> \
  DTYPE_IN=half DTYPE_OUT=float DTYPE_ACC=float \
  M=<M> N=5376 K=21504 CNT=20 INPUT=RAND \
  python3 extra/gemm/max_matmul.py
```

| Variant | M=64 us | M=256 us | M=512 us | M=1024 us |
| --- | ---: | ---: | ---: | ---: |
| `max` | 346.88 | 639.58 | 1223.68 | 2443.62 |
| `2_stage_swizzled_smem_input` | 348.93 | 633.12 | 1215.20 | 2490.08 |
| `swizzled_smem_input` | 477.54 | 784.86 | 1347.97 | 2950.46 |
| `flat_smem_input` | 888.42 | 1497.25 | 2789.63 | 5349.54 |

Conclusion:

- For the FP32-accumulating path, `max` and `2_stage_swizzled_smem_input` are close;
  `2_stage_swizzled_smem_input` edged out `max` at `M=256` and `M=512`, while `max`
  was better at `M=64` and `M=1024`.
- `swizzled_smem_input` and `flat_smem_input` are not competitive for this shape.
- The FP16-accumulating/output path is faster at several aligned `M` values but is not a
  Gemma correctness candidate without a separate numerical validation argument.
- These numbers are useful for shape exploration only; the earlier cuBLASLt baseline
  remains the relevant comparison before considering any integration.

### Existing BF16 binary checkpoint after fresh tinygrad reclone

After the fresh tinygrad reclone, the local BF16 source files and C++ harness that had
been under `experiments/tinygrad` were gone, but the previously built BF16 binaries still
existed under `build/experiments/tinygrad`. I used those binaries only as a no-build
checkpoint before deciding whether source restoration/tuning was worth doing.

Command pattern:

```bash
GEMMA4_PREFILL_CUBLASLT_HEURISTICS=64 GEMMA4_PREFILL_GRAPH_REPEATS=10 \
  ./build/experiments/tinygrad/<binary> \
  64,128,192,256,320,384,448,512,640,768,896,1024,1280,1536,1792,1984 \
  5 20 2
```

Best observed tinygrad BF16 variant per selected `M`:

| M | Best tinygrad variant | tinygrad ms | cuBLASLt ms | Speedup |
| ---: | --- | ---: | ---: | ---: |
| 64 | `bf16_max` | 0.33655 | 0.34189 | 1.016x |
| 128 | `bf16_max` | 0.38819 | 0.35117 | 0.905x |
| 192 | `bf16_max` | 0.55025 | 0.48043 | 0.873x |
| 256 | `bf16_max` | 0.72732 | 0.49453 | 0.680x |
| 320 | `bf16_max` | 1.05651 | 0.80418 | 0.761x |
| 512 | `bf16_splitk3` | 1.40796 | 0.95171 | 0.676x |
| 768 | `bf16_splitk3` | 2.08574 | 1.40242 | 0.672x |
| 1024 | `bf16_splitk3` | 2.79948 | 1.89341 | 0.676x |
| 1536 | `bf16_splitk3` | 4.31061 | 2.75091 | 0.638x |
| 1984 | `bf16_splitk3` | 5.80529 | 3.73639 | 0.644x |

Variant notes:

- `bf16_max` and `bf16_2_stage_swizzled_smem_input` are effectively tied; both only beat
  cuBLASLt at `M=64` by about `1.6%`.
- `bf16_splitk3` is worse at small `M`, slightly better than `bf16_max` at larger `M`,
  but still far behind cuBLASLt.
- No tested BF16 tinygrad static max variant reaches the requested `1.10x` win threshold
  over cuBLASLt on this FFN-down shape.

Post-reclone cleanup:

- The fresh reclone intentionally removed local-only tinygrad FFN-down sources such as
  `experiments/tinygrad/gemma4_ffn_down_tinygrad_bench.cu` and local BF16 kernel ports.
- Removed the stale Makefile tinygrad FFN-down targets instead of restoring those files.
- Deleted the briefly recreated `nv.bf16_f32_bf16.max.cu` source so the tinygrad checkout
  stays aligned with the intentional reclone cleanup.

## 2026-05-20 - tinygrad ffn_down dynamic-M GEMM sweep

Runtime files:

- `experiments/tinygrad/extra/gemm/`
- `experiments/tinygrad/gemma4_ffn_down_tinygrad_bench.cu`
- `Makefile`

Source:

- Cloned `https://github.com/tinygrad/tinygrad.git` into `experiments/tinygrad`
  with sparse checkout paths `extra/gemm` and `tinygrad`.
- tinygrad commit: `a19fa29`.

Reason:

- Test whether the NVIDIA kernels in tinygrad's `extra/gemm/max_kernels` can beat cuBLAS
  on Gemma 4 `ffn_down`: `M x 21504` by `21504 x 5376`.
- The first valid comparison uses tinygrad's FP16 input, FP32 accumulation, FP32 output
  `nv.fp16_fp32_fp32.max.cu` kernel. This is not yet the final Gemma BF16 datatype path,
  but it isolates the schedule on the exact `ffn_down` dimensions with dynamic `M`.

Implementation:

- Added `gemma4_ffn_down_tinygrad_bench.cu`, a direct CUDA/C++ harness that compiles the
  imported tinygrad kernel source and compares it against `cublasGemmEx`.
- Input layout is row-major `A[M,K]` and `B[K,N]`; cuBLAS uses the equivalent
  column-major `C^T[N,M] = B^T[N,K] * A^T[K,M]`.
- Timed with CUDA events on one stream after warmup. Reported medians across trials.
- Checked max absolute output difference against cuBLAS output for every `M`.
- Added `make tinygrad-ffn-down-bench`.

Build:

```bash
make tinygrad-ffn-down-bench
```

Timing command:

```bash
./build/experiments/tinygrad/gemma4_ffn_down_tinygrad_max_bench 64,128,192,256,320 25 200 7
```

Environment:

- GPU: NVIDIA RTX A6000
- Driver: 580.126.16
- CUDA/NVCC: 12.9 (`nvcc V12.9.86`)
- Clocks: not locked for this run
- Cache policy: repeated-buffer warm-cache microbenchmark
- Correctness reference: cuBLAS FP16 input, FP32 accumulation, FP32 output

Selected stable results:

| M | tinygrad median ms | cuBLAS median ms | Speedup | tinygrad TFLOP/s | cuBLAS TFLOP/s | Max abs |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 0.33812 | 3.44232 | 10.181x | 43.76 | 4.30 | 0.0065002 |
| 128 | 0.39303 | 3.04844 | 7.756x | 75.30 | 9.71 | 0.0083313 |
| 192 | 0.55271 | 3.10412 | 5.616x | 80.32 | 14.30 | 0.00737 |
| 256 | 0.72626 | 3.08555 | 4.249x | 81.50 | 19.18 | 0.00737 |
| 320 | 1.08439 | 3.23990 | 2.988x | 68.23 | 22.84 | 0.0082092 |

Transition check:

```bash
./build/experiments/tinygrad/gemma4_ffn_down_tinygrad_max_bench 64,128,192,256,320,384,448,512 20 100 5
```

- tinygrad continued to win through `M=320`.
- cuBLAS became faster from `M=384` onward in that run:
  `M=384` tinygrad `1.16068 ms`, cuBLAS `0.91630 ms`;
  `M=512` tinygrad `1.41390 ms`, cuBLAS `0.95771 ms`.

Other tinygrad variants tested:

- `nv.fp16_fp32_fp32.2_stage_swizzled_smem_input.cu`
- `nv.fp16_fp32_fp32.swizzled_smem_input.cu`
- `nv.fp16_fp32_fp32.flat_smem_input.cu`
- `nv.fp16_fp16_fp16.max.cu`

The FP32-output `max` kernel was the best valid tinygrad candidate. The FP16-accumulating
kernel produced invalid output for this shape (`max_abs` around `246-262`) and impossible
timing numbers, so it is not a candidate.

Conclusion:

- Success for dynamic short-prefill / microbatch `ffn_down` values: the imported
  tinygrad `max` schedule consistently outperforms cuBLAS by much more than 15% for
  `M in {64, 128, 192, 256, 320}` on this RTX A6000.
- This does not replace cuBLAS for all dynamic `M`: cuBLAS is faster by `M=384+`.
- Next useful step is a dispatcher experiment that uses this tinygrad kernel for
  `M <= 320` and cuBLAS/cuBLASLt above that cutoff, then ports the valid schedule to
  Gemma's BF16 input/output contract before considering it for the inference path.

Correction after applying the SGEMM cuBLASLt tuning:

- The initial table compared against plain `cublasGemmEx`. After wiring in the SGEMM
  cuBLASLt baseline style, the conclusion changes.
- The tinygrad harness now uses row-major cuBLASLt descriptors, a 32 MiB workspace,
  `GEMMA4_PREFILL_CUBLASLT_HEURISTICS` returned algorithms, and optional graph replay
  via `GEMMA4_PREFILL_GRAPH_REPEATS`. It times all returned successful cuBLASLt
  algorithms and reports the fastest median.

Short-M retest:

```bash
GEMMA4_PREFILL_CUBLASLT_HEURISTICS=64 GEMMA4_PREFILL_GRAPH_REPEATS=20 \
  ./build/experiments/tinygrad/gemma4_ffn_down_tinygrad_max_bench 64,128,192,256,320 10 50 3
```

| M | tinygrad median ms | cuBLASLt median ms | Speedup | Winner |
| ---: | ---: | ---: | ---: | --- |
| 64 | 0.33783 | 0.34437 | 1.019x | tinygrad |
| 128 | 0.41421 | 0.36750 | 0.887x | cuBLASLt |
| 192 | 0.58297 | 0.50515 | 0.867x | cuBLASLt |
| 256 | 0.75381 | 0.53224 | 0.706x | cuBLASLt |
| 320 | 1.15604 | 0.85894 | 0.743x | cuBLASLt |

Representative large-M retest:

```bash
GEMMA4_PREFILL_CUBLASLT_HEURISTICS=64 GEMMA4_PREFILL_GRAPH_REPEATS=10 \
  ./build/experiments/tinygrad/gemma4_ffn_down_tinygrad_max_bench 1024,1536,1984 5 20 2
```

| M | tinygrad median ms | cuBLASLt median ms | Speedup | Winner |
| ---: | ---: | ---: | ---: | --- |
| 1024 | 3.00374 | 2.04608 | 0.681x | cuBLASLt |
| 1536 | 4.59643 | 2.91769 | 0.635x | cuBLASLt |
| 1984 | 6.12065 | 3.97396 | 0.649x | cuBLASLt |

Updated conclusion:

- Against the tuned cuBLASLt baseline, this imported tinygrad kernel is not a useful
  `ffn_down` replacement except possibly the marginal `M=64` case.
- The earlier "tinygrad wins through M=320" conclusion should be treated as a comparison
  against an insufficient plain cuBLAS baseline, not the optimized library path.

BF16-native tinygrad port:

- Added `experiments/tinygrad/extra/gemm/max_kernels/nv.bf16_f32_bf16.max.cu`.
- Ported the tinygrad low-level schedule from FP16 to BF16 by changing the input/shared
  element type to `__nv_bfloat16`, changing the inline MMA instruction to
  `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`, and converting the FP32
  accumulators to BF16 in the output epilogue.
- Renamed the inherited half-fragment helpers in the BF16 source to BF16-specific names
  and store four BF16 outputs as two packed `__nv_bfloat162` values.
- Added `make tinygrad-ffn-down-bf16-bench`.
- CUDA guide context: BF16 is supported as `__nv_bfloat16` via `<cuda_bf16.h>` on compute
  capability 8.0+ (guide p.587), and WMMA/tensor-core BF16 inputs with FP32 accumulator
  support include 16x16x16, 32x8x16, and 8x32x16 shapes (guide p.577).

BF16 short-M command:

```bash
GEMMA4_PREFILL_CUBLASLT_HEURISTICS=64 GEMMA4_PREFILL_GRAPH_REPEATS=20 \
  ./build/experiments/tinygrad/gemma4_ffn_down_tinygrad_bf16_max_bench 64,128,192,256,320 10 50 3
```

| M | tinygrad BF16 median ms | cuBLASLt BF16 median ms | Speedup | Winner | Max abs |
| ---: | ---: | ---: | ---: | --- | ---: |
| 64 | 0.33516 | 0.34146 | 1.019x | tinygrad | 1 |
| 128 | 0.38952 | 0.35107 | 0.901x | cuBLASLt | 2 |
| 192 | 0.55607 | 0.48234 | 0.867x | cuBLASLt | 1 |
| 256 | 0.71795 | 0.48819 | 0.680x | cuBLASLt | 1 |
| 320 | 1.04997 | 0.79592 | 0.758x | cuBLASLt | 2 |

BF16 representative large-M command:

```bash
GEMMA4_PREFILL_CUBLASLT_HEURISTICS=64 GEMMA4_PREFILL_GRAPH_REPEATS=10 \
  ./build/experiments/tinygrad/gemma4_ffn_down_tinygrad_bf16_max_bench 1024,1536,1984 5 20 2
```

| M | tinygrad BF16 median ms | cuBLASLt BF16 median ms | Speedup | Winner | Max abs |
| ---: | ---: | ---: | ---: | --- | ---: |
| 1024 | 2.78194 | 1.86801 | 0.671x | cuBLASLt | 1 |
| 1536 | 4.32392 | 2.72715 | 0.631x | cuBLASLt | 0 |
| 1984 | 5.86556 | 3.69266 | 0.630x | cuBLASLt | 1 |

BF16 conclusion:

- The native BF16 tinygrad port works and is correctly wired against BF16 cuBLASLt, but
  it still does not beat the tuned cuBLASLt baseline except for a negligible `M=64`
  margin.
- The paired BF16 epilogue is not the bottleneck enough to close the gap; further work
  needs a different tile schedule or more parallelism across `M`/`N` to compete at
  `M >= 128`.

Static BF16 variant/resource follow-up:

- Per the user direction, do not use Tinygrad codegen for the next tuning steps. These
  checks use only the existing static CUDA sources and manual builds.
- `ncu` is installed, but profiling this cuBLAS-linked harness aborts in the current
  Thunder environment with an unsupported-library runtime assertion, so `ncu` timing and
  stall breakdowns were not available here.
- Static resource usage from `cuobjdump --dump-resource-usage`:

| Variant | Registers/thread | Static shared memory/block |
| --- | ---: | ---: |
| `bf16_max` | 148 | 49152 B |
| `bf16_2stage_swizzled` | 148 | 49152 B |
| `bf16_swizzled` | 126 | 24576 B |
| `bf16_flat` | 124 | 24576 B |
| `bf16_splitk2_partial` | 148 | 49152 B |

- Compiling `bf16_max` with `-maxrregcount=128` did not change the resource usage
  (`148` registers/thread) and did not improve performance.

Register-cap check:

```bash
nvcc -std=c++17 -O3 -arch=sm_86 -maxrregcount=128 -Isrc \
  -DTINYGRAD_BF16 \
  -DTINYGRAD_KERNEL_NAME=tinygrad_bf16_max_r128_kernel \
  -DTINYGRAD_VARIANT_NAME='"bf16_max_r128"' \
  -Dwmma_example=tinygrad_bf16_max_r128_kernel \
  experiments/tinygrad/gemma4_ffn_down_tinygrad_bench.cu \
  experiments/tinygrad/extra/gemm/max_kernels/nv.bf16_f32_bf16.max.cu \
  -lcublas -lcublasLt \
  -o build/experiments/tinygrad/gemma4_ffn_down_tinygrad_bf16_max_r128_bench

GEMMA4_PREFILL_CUBLASLT_HEURISTICS=64 GEMMA4_PREFILL_GRAPH_REPEATS=20 \
  ./build/experiments/tinygrad/gemma4_ffn_down_tinygrad_bf16_max_r128_bench \
  64,128,192,256,320 10 50 3
```

| M | tinygrad BF16 median ms | cuBLASLt BF16 median ms | Speedup | Winner |
| ---: | ---: | ---: | ---: | --- |
| 64 | 0.33546 | 0.34155 | 1.018x | tinygrad |
| 128 | 0.39058 | 0.35103 | 0.899x | cuBLASLt |
| 192 | 0.55646 | 0.48126 | 0.865x | cuBLASLt |
| 256 | 0.72776 | 0.49553 | 0.681x | cuBLASLt |
| 320 | 1.06855 | 0.81340 | 0.761x | cuBLASLt |

Static lower-smem variants, same command shape:

| Variant | M=64 | M=128 | M=192 | M=256 | M=320 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `bf16_2stage_swizzled` | 0.33558 | 0.38943 | 0.55745 | 0.72433 | 1.05142 |
| `bf16_swizzled` | 0.43545 | 0.47631 | 0.75441 | 0.89968 | 1.15999 |
| `bf16_flat` | 0.83093 | 0.84163 | 1.28007 | 1.33860 | 1.94461 |

Follow-up conclusion:

- The lower-smem variants have more favorable resident-block resources but much weaker
  runtime, so the current gap is not fixed just by reducing static shared memory.
- `bf16_max` and `bf16_2stage_swizzled` are essentially tied; `bf16_max` remains the best
  static source to mutate manually.
- At `M=128`, the current `64x128` tile launches `2 * 42 = 84` CTAs on the RTX A6000,
  exactly one CTA per SM. The next manual edit should split the output tile along `M` or
  `N` without adding a separate reduction pass; split-K already increased CTA count but
  lost to reduction overhead.

Manual M32 tile split:

- Added `experiments/tinygrad/extra/gemm/max_kernels/nv.bf16_f32_bf16.max_m32.cu`.
- Added `make tinygrad-ffn-down-bf16-m32-bench`.
- This variant mutates `bf16_max` by computing one 32-row half of the original 64-row
  tile per CTA. It removes the `acc_frag_1_*` half, avoids the unused A rows, and reduces
  static shared memory from `49152 B` to `40960 B`.
- Resource usage: `98` registers/thread, `40960 B` static shared memory/block.

Command:

```bash
GEMMA4_PREFILL_CUBLASLT_HEURISTICS=64 GEMMA4_PREFILL_GRAPH_REPEATS=20 \
  ./build/experiments/tinygrad/gemma4_ffn_down_tinygrad_bf16_max_m32_bench \
  32,64,128,192,256,320 10 50 3
```

| M | tinygrad BF16 M32 median ms | cuBLASLt BF16 median ms | Speedup | Winner | Max abs |
| ---: | ---: | ---: | ---: | --- | ---: |
| 32 | 0.32951 | 0.33135 | 1.006x | tinygrad | 1 |
| 64 | 0.33258 | 0.34164 | 1.027x | tinygrad | 1 |
| 128 | 0.55674 | 0.35207 | 0.632x | cuBLASLt | 2 |
| 192 | 0.90226 | 0.49165 | 0.545x | cuBLASLt | 1 |
| 256 | 1.11075 | 0.50386 | 0.454x | cuBLASLt | 1 |
| 320 | 1.43987 | 0.82116 | 0.570x | cuBLASLt | 2 |

M32 conclusion:

- Splitting `M` helps only the tiny cases that were already at or near a win.
- For `M >= 128`, duplicated B tile loading dominates; the M32 split is much slower than
  the original M64 tile and much slower than tuned cuBLASLt.

Launch-bounds check:

- Added a temporary manual `bf16_max_lb2` source with `__launch_bounds__(128, 2)`.
- Resource usage remained unchanged at `148` registers/thread and `49152 B` shared
  memory/block.

Command:

```bash
GEMMA4_PREFILL_CUBLASLT_HEURISTICS=64 GEMMA4_PREFILL_GRAPH_REPEATS=20 \
  ./build/experiments/tinygrad/gemma4_ffn_down_tinygrad_bf16_max_lb2_bench \
  64,128,192,256,320 10 50 3
```

| M | tinygrad BF16 LB2 median ms | cuBLASLt BF16 median ms | Speedup | Winner | Max abs |
| ---: | ---: | ---: | ---: | --- | ---: |
| 64 | 0.33636 | 0.34197 | 1.017x | tinygrad | 1 |
| 128 | 0.41138 | 0.35392 | 0.860x | cuBLASLt | 2 |
| 192 | 0.58439 | 0.50280 | 0.860x | cuBLASLt | 1 |
| 256 | 0.76209 | 0.51433 | 0.675x | cuBLASLt | 1 |
| 320 | 1.11949 | 0.83936 | 0.750x | cuBLASLt | 2 |

Launch-bounds conclusion:

- Adding `minBlocksPerMultiprocessor=2` did not improve register allocation or runtime.
- Leave the original `bf16_max` launch bound unchanged.

Split-K partition sweep:

- Split-K=2 had already shown that extra CTA count was not enough to pay for partial
  writes plus a reduction pass.
- Tested split-K=3 and split-K=4 with the same partial kernel to complete that axis.

Commands:

```bash
nvcc -std=c++17 -O3 -arch=sm_86 -Isrc \
  -DTINYGRAD_BF16 -DTINYGRAD_SPLITK_PARTS=3 \
  -DTINYGRAD_KERNEL_NAME=tinygrad_bf16_splitk3_partial_kernel \
  -DTINYGRAD_VARIANT_NAME='"bf16_splitk3"' \
  -Dwmma_example=tinygrad_bf16_splitk3_partial_kernel \
  experiments/tinygrad/gemma4_ffn_down_tinygrad_bench.cu \
  experiments/tinygrad/extra/gemm/max_kernels/nv.bf16_f32_bf16.max_splitk_partial.cu \
  -lcublas -lcublasLt \
  -o build/experiments/tinygrad/gemma4_ffn_down_tinygrad_bf16_splitk3_bench

nvcc -std=c++17 -O3 -arch=sm_86 -Isrc \
  -DTINYGRAD_BF16 -DTINYGRAD_SPLITK_PARTS=4 \
  -DTINYGRAD_KERNEL_NAME=tinygrad_bf16_splitk4_partial_kernel \
  -DTINYGRAD_VARIANT_NAME='"bf16_splitk4"' \
  -Dwmma_example=tinygrad_bf16_splitk4_partial_kernel \
  experiments/tinygrad/gemma4_ffn_down_tinygrad_bench.cu \
  experiments/tinygrad/extra/gemm/max_kernels/nv.bf16_f32_bf16.max_splitk_partial.cu \
  -lcublas -lcublasLt \
  -o build/experiments/tinygrad/gemma4_ffn_down_tinygrad_bf16_splitk4_bench
```

Each benchmark used:

```bash
GEMMA4_PREFILL_CUBLASLT_HEURISTICS=64 GEMMA4_PREFILL_GRAPH_REPEATS=20 \
  ./build/experiments/tinygrad/<binary> 64,128,192,256,320 10 50 3
```

| Variant | M=64 | M=128 | M=192 | M=256 | M=320 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `bf16_max` | 0.33516 | 0.38952 | 0.55607 | 0.71795 | 1.04997 |
| `bf16_splitk2` | 0.34609 | 0.40396 | 0.56928 | n/a | n/a |
| `bf16_splitk3` | 0.35457 | 0.40852 | 0.58423 | 0.74276 | 1.12005 |
| `bf16_splitk4` | 0.36298 | 0.42713 | 0.60784 | 0.75694 | 1.13357 |

Split-K conclusion:

- Higher split-K degrees monotonically worsened this shape in the tested range.
- The extra parallelism does not pay for the extra global partial writes and reduction.
- Do not pursue split-K for this BF16 output contract unless the reduction can be fused
  into a materially different persistent/stream-K design.

ptxas load-cache policy check:

- Tested `-Xptxas -dlcm=ca` and `-Xptxas -dlcm=cg` on the best `bf16_max` source.
- These are static compiler-policy variants only; no Tinygrad codegen was used.
- The runs below were sequential. An initial parallel launch was killed because it would
  have contaminated GPU timings.

Commands:

```bash
nvcc -std=c++17 -O3 -arch=sm_86 -Xptxas -dlcm=ca -Isrc \
  -DTINYGRAD_BF16 \
  -DTINYGRAD_KERNEL_NAME=tinygrad_bf16_max_ca_kernel \
  -DTINYGRAD_VARIANT_NAME='"bf16_max_ca"' \
  -Dwmma_example=tinygrad_bf16_max_ca_kernel \
  experiments/tinygrad/gemma4_ffn_down_tinygrad_bench.cu \
  experiments/tinygrad/extra/gemm/max_kernels/nv.bf16_f32_bf16.max.cu \
  -lcublas -lcublasLt \
  -o build/experiments/tinygrad/gemma4_ffn_down_tinygrad_bf16_max_ca_bench

nvcc -std=c++17 -O3 -arch=sm_86 -Xptxas -dlcm=cg -Isrc \
  -DTINYGRAD_BF16 \
  -DTINYGRAD_KERNEL_NAME=tinygrad_bf16_max_cg_kernel \
  -DTINYGRAD_VARIANT_NAME='"bf16_max_cg"' \
  -Dwmma_example=tinygrad_bf16_max_cg_kernel \
  experiments/tinygrad/gemma4_ffn_down_tinygrad_bench.cu \
  experiments/tinygrad/extra/gemm/max_kernels/nv.bf16_f32_bf16.max.cu \
  -lcublas -lcublasLt \
  -o build/experiments/tinygrad/gemma4_ffn_down_tinygrad_bf16_max_cg_bench
```

Each benchmark used:

```bash
GEMMA4_PREFILL_CUBLASLT_HEURISTICS=64 GEMMA4_PREFILL_GRAPH_REPEATS=20 \
  ./build/experiments/tinygrad/<binary> 64,128,192,256,320 10 50 3
```

| Variant | M=64 | M=128 | M=192 | M=256 | M=320 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `bf16_max` | 0.33516 | 0.38952 | 0.55607 | 0.71795 | 1.04997 |
| `bf16_max_ca` | 0.33632 | 0.39692 | 0.56981 | 0.73697 | 1.09925 |
| `bf16_max_cg` | 0.33616 | 0.40489 | 0.57470 | 0.74982 | 1.09670 |

Cache-policy conclusion:

- Both cache policy variants were slower than the default compile.
- Leave the default ptxas cache policy for this kernel.

Final no-codegen retest:

- Per the user correction, no Tinygrad codegen was used in this pass. The only attempted
  new schedule change was a manual static CUDA edit.
- Hardened `gemma4_ffn_down_tinygrad_bench.cu` so the cuBLASLt comparison filters
  algorithms that return a runtime launch failure instead of aborting the whole
  benchmark on one bad heuristic. cuBLASLt plan construction, heuristic search, and
  warmup remain outside the timed CUDA-event/graph-replay region.
- Retested the validated native BF16 `bf16_max` kernel on the requested large-M range:

```bash
make tinygrad-ffn-down-bf16-bench

GEMMA4_PREFILL_CUBLASLT_HEURISTICS=64 GEMMA4_PREFILL_GRAPH_REPEATS=10 \
  ./build/experiments/tinygrad/gemma4_ffn_down_tinygrad_bf16_max_bench \
  1024,1536,1984 5 20 2
```

| M | tinygrad BF16 median ms | cuBLASLt BF16 median ms | Speedup | Winner | Max abs | cuBLASLt algo |
| ---: | ---: | ---: | ---: | --- | ---: | ---: |
| 1024 | 2.78306 | 1.87530 | 0.674x | cuBLASLt | 1 | 1 |
| 1536 | 4.33235 | 2.73662 | 0.632x | cuBLASLt | 2 | 0 |
| 1984 | 5.85225 | 3.69111 | 0.631x | cuBLASLt | 1 | 1 |

Manual N64 branch:

- Tried a manual `64x64` output-tile variant to increase CTA count without split-K
  reduction overhead.
- The first epilogue edit wrote past the last 64-column tile. After constraining stores
  to the first 64 columns and restoring the missing row stores, a one-block smoke harness
  still reported a delayed illegal memory access from the kernel.
- The N64 variant was discarded and not promoted to the Makefile.

cuBLASLt workspace check:

- Added `GEMMA4_PREFILL_CUBLASLT_WORKSPACE_MB` to the harness so the cuBLASLt baseline
  can be tested with larger workspaces without changing the timed region.
- Tested `32`, `128`, and `512` MiB with `GEMMA4_PREFILL_CUBLASLT_HEURISTICS=64` and
  graph replay. Larger workspaces did not improve the best cuBLASLt large-M result.

| Workspace | M=1024 cuBLASLt ms | M=1536 cuBLASLt ms | M=1984 cuBLASLt ms |
| ---: | ---: | ---: | ---: |
| 32 MiB | 1.86400 | 2.73221 | 3.67780 |
| 128 MiB | 1.88715 | 2.74786 | 3.69017 |
| 512 MiB | 1.88933 | 2.75041 | 3.71312 |

Large-M static variant check:

```bash
GEMMA4_PREFILL_CUBLASLT_HEURISTICS=64 GEMMA4_PREFILL_GRAPH_REPEATS=10 \
  ./build/experiments/tinygrad/gemma4_ffn_down_tinygrad_bf16_<variant>_bench \
  1024,1536,1984 5 20 2
```

| Variant | M=1024 tiny ms | M=1536 tiny ms | M=1984 tiny ms | Conclusion |
| --- | ---: | ---: | ---: | --- |
| `bf16_max` | 2.78079 | 4.31409 | 5.78123 | Best static baseline. |
| `bf16_2_stage_swizzled_smem_input` | 2.80982 | 4.31478 | 5.81267 | Essentially tied/slightly worse. |
| `bf16_swizzled_smem_input` | 3.40250 | 5.14542 | 6.79865 | Reject. |
| `bf16_flat_smem_input` | 5.03938 | 7.48701 | 10.07056 | Reject. |

Large-M split-K check:

| Variant | M=1024 tiny ms | M=1536 tiny ms | M=1984 tiny ms | Conclusion |
| --- | ---: | ---: | ---: | --- |
| `bf16_splitk2` | 2.79765 | 4.34413 | 5.79682 | Still behind cuBLASLt. |
| `bf16_splitk3` | 2.77403 | 4.25981 | 5.71401 | Best split-K point, still only `0.645x` at M=1984. |
| `bf16_splitk4` | 2.83319 | 4.35334 | 5.77718 | Reject. |

ptxas check:

```bash
nvcc -std=c++17 -O3 -arch=sm_86 -Xptxas -v -Isrc \
  -DTINYGRAD_BF16 \
  -DTINYGRAD_KERNEL_NAME=tinygrad_bf16_max_v_kernel \
  -DTINYGRAD_VARIANT_NAME='"bf16_max_v"' \
  -Dwmma_example=tinygrad_bf16_max_v_kernel \
  experiments/tinygrad/gemma4_ffn_down_tinygrad_bench.cu \
  experiments/tinygrad/extra/gemm/max_kernels/nv.bf16_f32_bf16.max.cu \
  -lcublas -lcublasLt \
  -o /tmp/tiny_bf16_v
```

- `tinygrad_bf16_max_v_kernel`: `148` registers/thread, `49152` B shared memory,
  `0` spill stores, `0` spill loads.
- The current gap is not register spilling; it is the tile schedule/runtime efficiency
  versus cuBLASLt.

Final conclusion for this branch:

- Native BF16 is implemented for the imported static Tinygrad `bf16_max` kernel and
  compares correctly against BF16 cuBLASLt.
- With cuBLASLt overhead factored out and up to 64 heuristics tested, cuBLASLt remains
  much faster for `M=1000-2000`. The static Tinygrad schedule is only marginally useful
  at very small `M`.

## 2026-05-20 - Tuna BF16 prefill GEMM adaptation baseline

Runtime files:

- `experiments/tuna/gemma4_prefill_bench.cu`
- `Makefile`

Reason:

- The imported Tuna kernel is tuned for FP16 x FP4 GEMM with quantized activations.
- Gemma 4 prefill projections use BF16 activations and BF16 weights with variable token
  count `M`: `Y[M,N] = X[M,K] * W[N,K]^T`.
- This pass establishes a Gemma-specific BF16 benchmark harness before deeper tuning or
  more invasive Tuna pipeline changes.

Gemma 4 projection shapes covered:

| Op | K | N |
| --- | ---: | ---: |
| `ffn_gate_up` | 5376 | 43008 |
| `ffn_down` | 21504 | 5376 |
| `sliding_qkv` | 5376 | 16384 |
| `sliding_o` | 8192 | 5376 |
| `global_q` | 5376 | 16384 |
| `global_k` | 5376 | 2048 |
| `global_o` | 16384 | 5376 |
| `final_logits` | 5376 | 262144 |

Implementation:

- Added `gemma4_prefill_bench.cu` under the Tuna experiment folder.
- Replaced Tuna's FP4 activation quantization assumption with BF16 x BF16 WMMA tensor-core
  kernels using FP32 accumulation and BF16 output conversion.
- The benchmark uses the same weight layout as the main Gemma decode helpers:
  weights are stored as `[N,K]`, so logical matrix B is loaded as column-major `[K,N]`.
- Added global-load WMMA tile variants and a first shared-memory-staged variant:
  `wmma_16x16`, `wmma_16x32`, `wmma_16x64`, `wmma_32x64`, `wmma_64x64`,
  `smem_16x64`, `smem_16x128`, `smem_32x64`, `smem_32x128`, and `smem_64x64`.
- Added a `make tuna-prefill-bench` target.

Build:

```bash
make tuna-prefill-bench
```

Correctness:

- Compared each run against `cublasGemmEx` BF16 input, FP32 accumulate, BF16 output.
- Observed max absolute differences were `0`, `0.015625`, or `0.03125` depending on
  shape and reduction length, consistent with BF16 output rounding/order differences.

Benchmark commands:

```bash
./build/experiments/gemma4_tuna_prefill_bench global_k 80 20 1,16,64,256,1024
./build/experiments/gemma4_tuna_prefill_bench ffn_gate_up 10 3 16,64,256,1024 wmma_64x64
./build/experiments/gemma4_tuna_prefill_bench sliding_qkv 20 5 16,64,256,1024 wmma_64x64
```

Selected results:

| Op | M | Best/custom config | Custom ms | cuBLAS ms | Speedup |
| --- | ---: | --- | ---: | ---: | ---: |
| `global_k` | 16 | `wmma_16x32` | 0.0665 | 1.4408 | 21.67x |
| `global_k` | 64 | `wmma_16x16` | 0.0860 | 1.9580 | 22.78x |
| `global_k` | 256 | `wmma_16x16` | 0.3292 | 0.0593 | 0.18x |
| `global_k` | 1024 | `wmma_32x64` | 1.2957 | 1.6951 | 1.31x |
| `ffn_gate_up` | 16 | `wmma_64x64` | 1.5670 | 0.6687 | 0.43x |
| `ffn_gate_up` | 64 | `wmma_64x64` | 1.5616 | 0.6917 | 0.44x |
| `ffn_gate_up` | 256 | `wmma_64x64` | 6.4659 | 0.8452 | 0.13x |
| `sliding_qkv` | 16 | `wmma_64x64` | 0.7812 | 1.4959 | 1.92x |
| `sliding_qkv` | 64 | `wmma_64x64` | 0.7803 | 0.2814 | 0.36x |
| `sliding_qkv` | 256 | `wmma_64x64` | 2.5746 | 0.4059 | 0.16x |

Profiling:

```bash
ncu --metric gpu__time_duration.sum --target-processes all \
  ./build/experiments/gemma4_tuna_prefill_bench global_k 1 0 256 wmma_16x16
```

- `ncu` failed on this Thunder instance with an internal Thunder runtime assertion for
  an unsupported library path. CUDA-event timing remains the available timing source for
  this pass.

Conclusion:

- The first BF16 Tuna adaptation is correct and shape-aware, but not yet a competitive
  replacement for cuBLAS on the dominant wide prefill projections.
- Global-load WMMA variants are currently better than the simple shared-memory-staged
  variants for `global_k`; the staged version likely needs vectorized loads, bank-conflict
  work, and a real async pipeline before it can pay for its synchronization overhead.
- The only promising cases in this pass are narrow/small-M projections, especially
  `global_k` at small `M` and `sliding_qkv` at `M=16`.
- Next Tuna work should either port more of the original pipelined Tuna machinery to
  BF16 or stop spending time on the current WMMA baseline for wide prefill shapes.

## 2026-05-20 - SGEMM Gemma-shape prefill baseline

Runtime files:

- `experiments/sgemm.cu/gemma4_prefill_bench.cu`
- `Makefile`

Reason:

- The imported SGEMM implementation is tuned around square FP32 GEMMs.
- Gemma 4 prefill projections have fixed `K,N` projection dimensions and variable token
  count `M`, so the original square benchmark is not representative.

Implementation:

- Added a Gemma-shape benchmark that reuses the imported `sgemm()` launcher.
- The benchmark covers the same Gemma projection shapes as the Tuna benchmark.
- This is explicitly an FP32 shape/tile baseline. It does not yet replace the datatype
  path with Gemma's BF16 inference math.
- Added `make sgemm-prefill-bench`.

Build:

```bash
make sgemm-prefill-bench
```

Benchmark commands:

```bash
./build/experiments/gemma4_sgemm_prefill_bench global_k 20 5 16,64,256
./build/experiments/gemma4_sgemm_prefill_bench ffn_gate_up 5 2 16,64
```

Selected results:

| Op | M | Custom ms | cuBLAS ms | Speedup | Max abs |
| --- | ---: | ---: | ---: | ---: | ---: |
| `global_k` | 16 | 0.5476 | 1.8707 | 3.416x | 1.43051e-05 |
| `global_k` | 64 | 0.5468 | 2.0852 | 3.813x | 9.53674e-06 |
| `global_k` | 256 | 0.5576 | 1.7969 | 3.222x | 1.19209e-05 |
| `ffn_gate_up` | 16 | 1.9557 | 1.3309 | 0.681x | 1.23978e-05 |
| `ffn_gate_up` | 64 | 2.0290 | 1.3708 | 0.676x | 0 |

Conclusion:

- The imported SGEMM shape policy can beat `cublasSgemm` for the narrow `global_k`
  FP32 baseline on this machine, but loses on the wide `ffn_gate_up` shape.
- This does not satisfy the final Gemma datatype requirement because the kernel remains
  FP32. The next SGEMM step is to decide whether to port its cp.async/shared-memory
  schedule to BF16 tensor cores or treat it as a non-candidate for the BF16 inference
  path.

## 2026-05-20 - SGEMM BF16 prefill adaptation

Runtime files:

- `experiments/sgemm.cu/gemma4_bf16_prefill_bench.cu`
- `Makefile`

Reason:

- The FP32 SGEMM shape benchmark does not match Gemma 4 inference datatype.
- This pass adds a BF16 x BF16 tensor-core path under the SGEMM experiment and benchmarks
  SGEMM-style CTA shapes against cuBLAS across variable `M`.

Implementation:

- Added `make sgemm-bf16-prefill-bench`.
- Added BF16 WMMA kernels with FP32 accumulation and BF16 output.
- Weight layout matches the main Gemma projection layout: `[N,K]`, interpreted as
  column-major `[K,N]` for `Y[M,N] = X[M,K] * W[N,K]^T`.
- SGEMM-style tile variants tested:
  `bf16_16x32`, `bf16_16x64`, `bf16_32x64`, `bf16_64x64`,
  `bf16_64x128`, and `bf16_128x64`.
- Larger square-ish CTA shapes from the imported SGEMM source cannot be copied directly
  with one warp per 16x16 WMMA tile because a true `128x128` tile would require 64 warps
  / 2048 threads. The largest tested shapes stay within CUDA's legal block size.

Build:

```bash
make sgemm-bf16-prefill-bench
```

Benchmark commands:

```bash
./build/experiments/gemma4_sgemm_bf16_prefill_bench global_k 40 10 16,64,256
./build/experiments/gemma4_sgemm_bf16_prefill_bench ffn_gate_up 10 3 16,64,256 bf16_64x64
./build/experiments/gemma4_sgemm_bf16_prefill_bench sliding_qkv 20 5 16,64,256 bf16_64x64
```

Selected results:

| Op | M | Best/custom config | Custom ms | cuBLAS ms | Speedup | Max abs |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| `global_k` | 16 | `bf16_16x32` | 0.0678 | 1.5484 | 22.824x | 0.015625 |
| `global_k` | 64 | `bf16_16x64` | 0.1007 | 1.8713 | 18.583x | 0.015625 |
| `global_k` | 256 | `bf16_16x32` | 0.3317 | 0.0637 | 0.192x | 0 |
| `ffn_gate_up` | 16 | `bf16_64x64` | 1.5684 | 0.6677 | 0.426x | 0 |
| `ffn_gate_up` | 64 | `bf16_64x64` | 1.5668 | 0.6911 | 0.441x | 0.015625 |
| `ffn_gate_up` | 256 | `bf16_64x64` | 6.5583 | 0.8496 | 0.130x | 0 |
| `sliding_qkv` | 16 | `bf16_64x64` | 0.7790 | 1.2916 | 1.658x | 0.015625 |
| `sliding_qkv` | 64 | `bf16_64x64` | 0.7787 | 0.2786 | 0.358x | 0 |
| `sliding_qkv` | 256 | `bf16_64x64` | 2.5505 | 0.4059 | 0.159x | 0 |

Conclusion:

- The SGEMM BF16 adaptation is correct against cuBLAS at BF16 output tolerance.
- For variable `M`, small row tiles win on narrow `global_k` at small M; larger CTA-style
  tiles waste too much work there.
- On dominant wide-N prefill projections, this simple WMMA adaptation loses badly to
  cuBLAS. Further SGEMM work would need a deeper tensor-core schedule with multiple WMMA
  tiles per warp and a real cp.async pipeline rather than the current one-warp-per-tile
  implementation.

## 2026-05-20 - Prefill GEMM focused tuner automation

Runtime files:

- `src/experiments/gemma4_prefill_tune.py`
- `build/experiments/gemma4_prefill_tune/tuna_focused.csv`
- `build/experiments/gemma4_prefill_tune/sgemm_bf16_focused.csv`

Reason:

- Manual one-off benchmark commands made it too easy to compare a config that was good
  at one `M` but poor across the variable-M prefill range.
- The tuner runs benchmark binaries sequentially, writes CSV rows, and ranks configs by
  geometric mean speedup plus worst-case speedup across the chosen M sweep.

Validation:

```bash
python3 -m py_compile src/experiments/gemma4_prefill_tune.py

python3 src/experiments/gemma4_prefill_tune.py \
  --backend sgemm-bf16 --ops global_k \
  --configs bf16_16x32,bf16_64x64 --m 16,64 --iters 5 --warmup 2 \
  --out build/experiments/gemma4_prefill_tune/sgemm_bf16_smoke.csv

python3 src/experiments/gemma4_prefill_tune.py \
  --backend tuna --ops global_k \
  --configs wmma_16x32,wmma_64x64 --m 16,64 --iters 5 --warmup 2 \
  --out build/experiments/gemma4_prefill_tune/tuna_smoke.csv
```

Focused tuning commands:

```bash
python3 src/experiments/gemma4_prefill_tune.py \
  --backend tuna --ops global_k,sliding_qkv,ffn_gate_up \
  --configs wmma_16x32,wmma_16x64,wmma_32x64,wmma_64x64 \
  --m 16,64,256 --iters 10 --warmup 3 \
  --out build/experiments/gemma4_prefill_tune/tuna_focused.csv

python3 src/experiments/gemma4_prefill_tune.py \
  --backend sgemm-bf16 --ops global_k,sliding_qkv,ffn_gate_up \
  --configs bf16_16x32,bf16_16x64,bf16_32x64,bf16_64x64,bf16_64x128,bf16_128x64 \
  --m 16,64,256 --iters 10 --warmup 3 \
  --out build/experiments/gemma4_prefill_tune/sgemm_bf16_focused.csv
```

Dispatch summary command:

```bash
python3 src/experiments/gemma4_prefill_tune.py summarize \
  build/experiments/gemma4_prefill_tune/tuna_focused.csv \
  build/experiments/gemma4_prefill_tune/sgemm_bf16_focused.csv \
  --custom-threshold 1.0
```

Focused Tuna summary:

| Op | Best config | Geomean speedup | Worst speedup |
| --- | --- | ---: | ---: |
| `ffn_gate_up` | `wmma_64x64` | 0.293x | 0.133x |
| `global_k` | `wmma_32x64` | 10.330x | 0.162x |
| `sliding_qkv` | `wmma_32x64` | 0.511x | 0.081x |

Focused SGEMM BF16 summary:

| Op | Best config | Geomean speedup | Worst speedup |
| --- | --- | ---: | ---: |
| `ffn_gate_up` | `bf16_64x64` | 0.293x | 0.133x |
| `global_k` | `bf16_16x64` | 6.024x | 0.184x |
| `sliding_qkv` | `bf16_64x64` | 0.672x | 0.161x |

Hybrid dispatch summary:

Note: this first focused summary used the cuBLAS timing from the same row as the selected
custom config. Later summaries below use the fastest observed cuBLAS time for each
`(backend, op, M)` case and should be treated as the conservative dispatch evidence.

| Backend | Dispatch weighted ms | cuBLAS weighted ms | Custom-only weighted ms | Dispatch vs cuBLAS | Custom-only vs cuBLAS |
| --- | ---: | ---: | ---: | ---: | ---: |
| Tuna | 188.2440 | 323.6200 | 755.1540 | 1.7192x | 0.4285x |
| SGEMM BF16 | 188.6090 | 439.9490 | 759.2650 | 2.3326x | 0.5794x |

The dispatch policy uses custom kernels only where the measured custom config is faster
than cuBLAS for that `(op, M)` point, otherwise it falls back to cuBLAS. This is not a
final production policy because small-M cuBLAS measurements are noisy here, but it is a
useful guardrail: the current custom kernels are only candidates for specific small-M
projection cases, not broad prefill replacements.

Caveats:

- Small-M cuBLAS timings are noisy on this Thunder instance, even with warmup. The tuner
  is useful for reproducible sweeps and spotting clear losers, but longer runs and a
  production profiling instance are still needed before treating the exact speedups as
  final.
- The current one-warp-per-WMMA-tile custom kernels are not competitive on the dominant
  wide prefill projections. The tuner reinforces the earlier conclusion: deeper pipeline
  work is needed before custom prefill GEMM can replace cuBLAS for most Gemma 4 shapes.

All-shape follow-up:

```bash
python3 src/experiments/gemma4_prefill_tune.py \
  --backend tuna \
  --ops ffn_gate_up,ffn_down,sliding_qkv,sliding_o,global_q,global_k,global_o,final_logits \
  --configs wmma_16x32,wmma_32x64,wmma_64x64 \
  --m 16,64,256 --iters 5 --warmup 2 \
  --out build/experiments/gemma4_prefill_tune/tuna_all_shapes.csv

python3 src/experiments/gemma4_prefill_tune.py \
  --backend sgemm-bf16 \
  --ops ffn_gate_up,ffn_down,sliding_qkv,sliding_o,global_q,global_k,global_o,final_logits \
  --configs bf16_16x32,bf16_32x64,bf16_64x64 \
  --m 16,64,256 --iters 5 --warmup 2 \
  --out build/experiments/gemma4_prefill_tune/sgemm_bf16_all_shapes.csv

python3 src/experiments/gemma4_prefill_tune.py summarize \
  build/experiments/gemma4_prefill_tune/tuna_all_shapes.csv \
  build/experiments/gemma4_prefill_tune/sgemm_bf16_all_shapes.csv \
  --custom-threshold 1.0
```

All-shape config ranking:

| Backend | Op | Best config | Geomean speedup | Worst speedup |
| --- | --- | --- | ---: | ---: |
| Tuna | `ffn_down` | `wmma_64x64` | 2.648x | 1.060x |
| Tuna | `ffn_gate_up` | `wmma_64x64` | 0.293x | 0.134x |
| Tuna | `final_logits` | `wmma_64x64` | 0.289x | 0.132x |
| Tuna | `global_k` | `wmma_16x32` | 4.261x | 0.189x |
| Tuna | `global_o` | `wmma_32x64` | 0.551x | 0.431x |
| Tuna | `global_q` | `wmma_64x64` | 0.643x | 0.162x |
| Tuna | `sliding_o` | `wmma_32x64` | 0.798x | 0.446x |
| Tuna | `sliding_qkv` | `wmma_64x64` | 0.595x | 0.161x |
| SGEMM BF16 | `ffn_down` | `bf16_64x64` | 1.720x | 0.373x |
| SGEMM BF16 | `ffn_gate_up` | `bf16_64x64` | 0.286x | 0.127x |
| SGEMM BF16 | `final_logits` | `bf16_64x64` | 0.292x | 0.137x |
| SGEMM BF16 | `global_k` | `bf16_32x64` | 3.888x | 0.165x |
| SGEMM BF16 | `global_o` | `bf16_32x64` | 0.564x | 0.294x |
| SGEMM BF16 | `global_q` | `bf16_64x64` | 0.578x | 0.161x |
| SGEMM BF16 | `sliding_o` | `bf16_64x64` | 0.874x | 0.438x |
| SGEMM BF16 | `sliding_qkv` | `bf16_64x64` | 0.729x | 0.161x |

All-shape weighted dispatch summary:

| Backend | Dispatch weighted ms | cuBLAS weighted ms | Custom-only weighted ms | Dispatch vs cuBLAS | Custom-only vs cuBLAS |
| --- | ---: | ---: | ---: | ---: | ---: |
| Tuna | 576.6059 | 1169.7969 | 1234.8390 | 2.0288x | 0.9473x |
| SGEMM BF16 | 450.3071 | 683.7701 | 1246.3649 | 1.5185x | 0.5486x |

Interpretation:

- The all-shape sweep identifies `ffn_down`, small-M `global_k`, M=16
  `global_q`/`sliding_qkv`, and M=256 `sliding_o` as the only consistently interesting
  custom-kernel candidates in this first adaptation.
- `ffn_gate_up`, `final_logits`, most QKV/global-Q at larger M, and most output
  projections should stay on cuBLAS unless a deeper pipelined tensor-core kernel is
  written.
- Tuna and SGEMM BF16 are still structurally similar after this adaptation. Tuna has a
  slight edge in the all-shape custom-only result because its measured `ffn_down` run was
  stronger; SGEMM BF16 has slightly better focused `sliding_qkv` in the earlier run.

Candidate deep sweep:

After the all-shape sweep narrowed the search space, I reran only the surviving candidate
ops over a denser M range and the full config sets.

```bash
python3 src/experiments/gemma4_prefill_tune.py \
  --backend tuna --ops ffn_down,global_k,sliding_o \
  --configs wmma_16x16,wmma_16x32,wmma_16x64,wmma_32x64,wmma_64x64,smem_16x64,smem_16x128,smem_32x64,smem_32x128,smem_64x64 \
  --m 16,32,64,128,256,512 --iters 10 --warmup 3 \
  --out build/experiments/gemma4_prefill_tune/tuna_candidate_deep.csv

python3 src/experiments/gemma4_prefill_tune.py \
  --backend sgemm-bf16 --ops ffn_down,global_k,sliding_o \
  --configs bf16_16x32,bf16_16x64,bf16_32x64,bf16_64x64,bf16_64x128,bf16_128x64 \
  --m 16,32,64,128,256,512 --iters 10 --warmup 3 \
  --out build/experiments/gemma4_prefill_tune/sgemm_bf16_candidate_deep.csv

python3 src/experiments/gemma4_prefill_tune.py summarize \
  build/experiments/gemma4_prefill_tune/tuna_candidate_deep.csv \
  build/experiments/gemma4_prefill_tune/sgemm_bf16_candidate_deep.csv \
  --custom-threshold 1.0
```

Candidate deep summary:

| Backend | Op | Best config | Geomean speedup | Worst speedup |
| --- | --- | --- | ---: | ---: |
| Tuna | `ffn_down` | `wmma_64x64` | 0.864x | 0.110x |
| Tuna | `global_k` | `wmma_16x16` | 3.233x | 0.189x |
| Tuna | `sliding_o` | `wmma_64x64` | 0.378x | 0.123x |
| SGEMM BF16 | `ffn_down` | `bf16_32x64` | 0.798x | 0.050x |
| SGEMM BF16 | `global_k` | `bf16_16x64` | 4.046x | 0.188x |
| SGEMM BF16 | `sliding_o` | `bf16_64x64` | 0.375x | 0.122x |

Candidate dispatch summary:

| Backend | Dispatch weighted ms | cuBLAS weighted ms | Custom-only weighted ms | Dispatch vs cuBLAS | Custom-only vs cuBLAS |
| --- | ---: | ---: | ---: | ---: | ---: |
| Tuna | 426.9340 | 708.2160 | 1066.1890 | 1.6588x | 0.6642x |
| SGEMM BF16 | 399.7810 | 625.7630 | 1022.0280 | 1.5653x | 0.6123x |

Deep-sweep dispatch windows:

- `ffn_down`: custom is useful for M=16,32,64; use cuBLAS from M=128 onward.
- `global_k`: custom is useful at M=16,32,64; M=128 and M=256 should use cuBLAS.
  The measured M=512 custom win is suspicious because cuBLAS timing remains noisy for
  small-ish N; rerun before encoding a production rule there.
- `sliding_o`: custom is mostly rejected; M=256 barely wins in both backends, but the
  margin is small enough that I would not dispatch to it without a longer stable run.
- The simple staged shared-memory Tuna variants did not surface as winners in the deep
  sweep, so the current smem path is not worth promoting without a more serious async
  pipeline redesign.

## 2026-05-20 - Decode GEMV block swizzle experiment

Runtime files:

- `src/gemma4_matmul_kernels.cu`
- `src/gemma4_matmul_kernels.cuh`
- `src/experiments/gemma4_decode_bench.cu`

Research notes:

- NVIDIA's thread-group ID swizzling guidance targets 2D compute grids where adjacent
  thread groups have overlapping memory footprints but row-major launch order sends
  concurrently executing groups far apart.
- CUTLASS/CUDA Tile-style GEMM swizzles are also 2D output-tile mappings intended to
  improve reuse of shared A/B tiles through L2.
- The current decode GEMV is a 1D output-column grid: one CTA computes 8 output columns
  and the full K dot product for those columns. Adjacent CTAs do not reuse weight data;
  they only share the small activation vector.

Implementation:

- Added `Gemma4DecodeSwizzle` with an identity policy and an experimental
  `GEMMA4_DECODE_SWIZZLE_INTERLEAVE_16` policy.
- The swizzled path remaps physical CTA IDs through a compile-time interleave over
  groups of 16 column blocks.
- `gemma4_projection_decode` keeps the identity mapping by default.
- `gemma4_projection_decode_swizzled` exposes the native swizzled path for direct
  benchmarking or future call sites.
- The decode benchmark now runs identity and swizzled custom GEMV and checks exact
  output equality.

Commands:

```bash
make decode-bench
GEMMA4_DECODE_BENCH_SEED=0x20260520 ./build/experiments/gemma4_decode_bench ffn_gate_up 50 10 3
GEMMA4_DECODE_BENCH_SEED=0x20260520 ./build/experiments/gemma4_decode_bench global_k 50 10 3
GEMMA4_DECODE_BENCH_SEED=0x20260520 ./build/experiments/gemma4_decode_bench final_logits 20 5 2
```

Results:

| Op | Identity ms | Swizzle16 ms | Swizzle speedup | Max abs diff |
| --- | ---: | ---: | ---: | ---: |
| `ffn_gate_up` | 0.650819 | 0.650897 | 0.9999x | 0 |
| `global_k` | 0.034588 | 0.034268 | 1.0093x | 0 |
| `final_logits` | 3.956267 | 3.953227 | 1.0008x | 0 |

Conclusion:

- Native block swizzling is implemented and measurable, but the interleave-16 mapping is
  neutral on the tested decode GEMV shapes.
- This matches the geometry expectation: the current GEMV is 1D and streams weights with
  almost no cross-CTA weight reuse, so 2D thread-group tiling does not directly transfer.
- Keep identity as the default. Revisit swizzling if the GEMV is changed to a 2D output
  tile, split-K/persistent schedule, or a fused multi-vector/multi-token decode kernel
  with real cross-CTA footprint overlap.

## 2026-05-20 - Scale-free V RMSNorm in existing RMSNorm kernels

Runtime files:

- `src/gemma4_rmsnorm.cu`
- `src/gemma4_rmsnorm.cuh`
- `tests/test_rmsnorm.cu`
- `src/experiments/gemma4_rmsnorm_bench.cu`

Reason:

- Gemma 4 applies RMSNorm to V without a learned scale parameter.
- Rather than adding a separate V-only kernel, the existing RMSNorm implementation now
  treats `weight == nullptr` as scale-free RMSNorm and dispatches to compile-time
  `HasWeight=false` kernel instantiations.
- The explicit wrapper `gemma4_rmsnorm_scale_free_bf16` is available for call sites
  where passing a null weight would be less clear.

Implementation:

- Reused the current RMSNorm reduction, launch-shape, shared-memory, and decode-specialized
  structure.
- Added scale-free template instantiations that compile out weight loads and apply
  `y = x * rsqrt(mean(x*x) + eps)`.
- The fused residual+RMSNorm launcher also accepts `weight == nullptr`, although the
  immediate target is standalone V RMSNorm.
- Extended the RMSNorm benchmark to accept an optional width argument and report
  scale-free custom timings. Since cuDNN frontend RMSNorm requires a scale tensor, the
  default comparison uses cuDNN RMSNorm with an all-ones BF16 scale.

Build and correctness:

```bash
make test-rmsnorm
make rmsnorm-bench
make cuda-kernels
nvcc -std=c++17 -O3 -arch=sm_86 -Xptxas=-v -Isrc \
  -c src/gemma4_rmsnorm.cu -o /tmp/gemma4_rmsnorm_scale_free_check.o
```

Result:

- `rmsnorm tests passed`
- Scale-free test coverage includes width `256`, width `512`, and width `5376`.
- The scale-free fused residual+RMSNorm dispatch is also covered at width `512` and
  width `5376`.
- `ptxas` reported `0 bytes` spills for all weighted and scale-free RMSNorm
  instantiations.

Benchmark commands:

```bash
GEMMA4_RMSNORM_BENCH_SEED=0x20260520 \
  ./build/experiments/gemma4_rmsnorm_bench 30 5 2 1024 256

GEMMA4_RMSNORM_BENCH_SEED=0x20260520 \
  ./build/experiments/gemma4_rmsnorm_bench 30 5 2 1024 512
```

Key CUDA graph-captured timing results:

| Width | Rows | Custom scale-free ms | Custom GiB/s | cuDNN one-scale ms | cuDNN GiB/s | Max abs | rstd max abs |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 16 | 0.001641 | 9.337 | 0.002178 | 10.536 | 0 | 1.19209e-7 |
| 256 | 1024 | 0.002007 | 488.365 | 0.002503 | 586.650 | 0 | 2.38419e-7 |
| 512 | 4 | 0.001622 | 4.712 | 0.002078 | 5.515 | 0 | 1.19209e-7 |
| 512 | 1024 | 0.002638 | 741.865 | 0.003444 | 851.706 | 0 | 2.38419e-7 |

Interpretation:

- The scale-free path numerically matches cuDNN RMSNorm with an all-ones scale.
- At the decode-relevant V shapes (`16 x 256` for sliding, `4 x 512` for global), the
  custom scale-free kernel is faster than the cuDNN one-scale baseline in graph-captured
  timings.
- The stream-loop timings at these small widths are dominated by frontend/API overhead,
  so graph-captured timings are the more useful kernel-speed comparison.

## 2026-05-20 - Gemma 4 RoPE baseline vs cuDNN tensor ops

Benchmark file: `src/experiments/gemma4_rope_bench.cu`

Research notes:

- cuDNN 9.22 still documents `cudnnOpTensor`, but marks it deprecated since cuDNN 9.0.
- The modern cuDNN frontend exposes pointwise `mul` and `add`, but there is no single fused RoPE primitive; the available library baseline is therefore a pointwise decomposition.
- The benchmark's cuDNN path applies RoPE through six `cudnnOpTensor` calls per Q/K branch group: two multiplies plus one add/sub for each output half.
- CUDA-event timings are recorded on the same nonblocking stream as the benchmark work, matching CUDA stream timing guidance.

Environment:

- GPU: NVIDIA RTX A6000, 49140 MiB
- Driver: 580.126.16
- NVCC: CUDA 12.9, build `cuda_12.9.r12.9/compiler.36037853_0`
- cuDNN headers/runtime: 9.22.0

Build:

```bash
make rope-bench
```

Timing command:

```bash
GEMMA4_ROPE_BENCH_SEED=0x20260520 ./build/experiments/gemma4_rope_bench 200 30 5 4096 1 1
```

Key CUDA-event timing results:

| Case | Seq | Custom ms | cuDNN ms | Custom speedup | Custom graph ms | cuDNN graph ms | Graph speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sliding | 1 | 0.013840 | 0.189035 | 13.66x | 0.001602 | 0.014945 | 9.33x |
| sliding | 1024 | 0.075738 | 0.305416 | 4.03x | 0.074725 | 0.287080 | 3.84x |
| sliding | 4096 | 0.483672 | 1.145305 | 2.37x | 0.482928 | 1.126152 | 2.33x |
| global | 1 | 0.020791 | 0.225866 | 10.86x | 0.001406 | 0.014736 | 10.48x |
| global | 1024 | 0.030901 | 0.258333 | 8.36x | 0.030432 | 0.116004 | 3.81x |
| global | 4096 | 0.147369 | 0.437539 | 2.97x | 0.145875 | 0.421499 | 2.89x |

Accuracy check:

- `cudnn_q_max_abs = 0.0078125`
- `cudnn_k_max_abs = 0.0078125`
- This compares custom RoPE using FP32 cosine/sine tables against the cuDNN baseline using BF16 cosine/sine tables, so the observed difference is consistent with table precision rather than a known indexing mismatch.

Conclusion:

- The custom RoPE kernel wins over the cuDNN tensor-op decomposition for every measured sequence length.
- CUDA Graph capture makes short-sequence launch overhead much smaller, but cuDNN remains slower because RoPE is decomposed into multiple pointwise launches instead of one fused kernel.
- Keep the custom RoPE baseline. Future optimization should focus on fusing Q/K RMSNorm, RoPE, and KV-cache write rather than replacing this path with cuDNN pointwise ops.

Follow-up static profile:

- Built PTX/SASS artifacts with `nvcc -std=c++17 -O3 -arch=sm_86 -Isrc`.
- `ptxas -v` reports 26 registers for the low-level layout kernel, 29 registers for the forward-layout kernel, 0 stack bytes, 0 spill loads/stores, and 0 barriers.
- `cuobjdump --dump-resource-usage` reports `SHARED:0` and `LOCAL:0` for both RoPE kernels.
- PTX and SASS contain no shared-memory instructions (`ld.shared`, `st.shared`, `LDS`, `STS`) and no local-memory spill instructions (`ld.local`, `st.local`, `LDL`, `STL`).
- RoPE memory ops are scalar, not vectorized: PTX uses `ld.global.u16`/`st.global.u16` for BF16 Q/K and `ld.global.f32` for cos/sin; SASS uses `LDG.E.U16`, `STG.E.U16`, and scalar 32-bit `LDG.E`.
- `ncu` could not collect bank-conflict counters on this Thunder instance; it aborted in Thunder/CUPTI with `Unimplemented CUDA export table function: Table=cupti_device_query, Index=7`.

## 2026-05-17 - 16384x512x4096 CUTE matmul vs cuBLAS

Experiment file: `src/experiments/16384_512_4096.cu`

Important correction: I initially started wiring this against raw cuDNN backend matmul, but that was the wrong baseline for this experiment. cuDNN matmul is exposed through backend/frontend graph descriptors, which is not the simple GEMM call we want here. For a short library matmul baseline, use cuBLAS/cuBLASLt. This run uses `cublasGemmEx`.

Shape:

- A: `[M, K] = [16384, 512]`, row-major half
- B storage passed to the custom kernel: `[N, K] = [4096, 512]`, row-major half, equivalent to B transposed for `C = A * B`
- C: `[M, N] = [16384, 4096]`, row-major half
- FLOPs per matmul: `2 * M * N * K = 68.719476736 GFLOPs`

Build:

```bash
nvcc -std=c++17 -O3 -arch=sm_86 -DGEMMA4_STANDALONE_BENCH \
  -I/tmp/cutlass/include -I/tmp/cutlass/tools/util/include \
  src/experiments/16384_512_4096.cu -lcublas \
  -o build/experiments/16384_512_4096_bench
```

Hardware/software:

- GPU: NVIDIA RTX A6000, compute capability 8.6, 49140 MiB
- Driver: 580.126.16
- NVCC: CUDA 12.9, build `cuda_12.9.r12.9/compiler.36037853_0`
- cuBLAS version: `120902`

Timing command:

```bash
./build/experiments/16384_512_4096_bench both 100 20
```

CUDA-event timing results:

| Kernel | Avg ms | TFLOP/s |
| --- | ---: | ---: |
| Custom CUTE kernel | 0.756506 | 90.838 |
| cuBLAS `cublasGemmEx` | 0.690792 | 99.4793 |

Comparison:

- `custom_vs_cublas_speedup = 0.913135`
- The custom kernel is about `8.7%` slower than cuBLAS on this shape in this run.
- Correctness check: `max_abs_diff = 0` versus cuBLAS output for the deterministic half input pattern used by the benchmark.

Implementation note:

- The cuBLAS baseline is a single `cublasGemmEx` call using FP16 inputs, FP32 accumulation, and `CUBLAS_GEMM_DEFAULT_TENSOR_OP`.
- The benchmark stores C in row-major form by viewing it as column-major `C^T` for cuBLAS. The call computes `C^T = B^T * A^T` with `CUBLAS_OP_T` on the stored `[N, K]` B buffer and `CUBLAS_OP_N` on the row-major A buffer interpreted as column-major `[K, M]`.

Measurement bug caught and fixed:

- An earlier timing attempt gave an impossible `2429.8 TFLOP/s` for the custom kernel.
- Cause: the custom kernel launch used the default stream while the CUDA events were recorded on a separate nonblocking stream, so the benchmark measured launch overhead rather than kernel execution.
- Fix: `launch_hgemm_mma_stages_block_swizzle_tn_cute` now accepts a `cudaStream_t`, and the standalone benchmark launches the custom kernel on the same stream as the timing events.

Compile-time resource data from `ptxas -v` for the custom kernel:

- Registers: `152`
- Static spills: `0 bytes`
- Barriers: `1`
- Constant memory: `388 bytes cmem[0]`
- Dynamic shared memory requested by the launch geometry: about `48 KiB`

Profiler status:

- `ncu --section SpeedOfLight --section MemoryWorkloadAnalysis` failed on this Thunder Compute instance with an internal Thunder runtime assertion before producing kernel metrics.
- `nsys profile --stats=true` completed, but the generated report did not contain CUDA trace, CUDA kernel, or GPU memory data on this instance. It only produced OS runtime stats, so I did not use it for GPU conclusions.
- Because of that, the only trustworthy performance numbers in this entry are CUDA-event timings plus the `ptxas` compile-time resource data.

Tooling/source notes:

- `$cuda-programming-guide` was used to check CUDA event timing and error-checking guidance.
- `$exa-search` could not run because `EXA_API_KEY` is not set in this environment.
- Official NVIDIA cuDNN frontend/backend docs confirm matmul is graph/backend-descriptor based, which is why cuDNN was dropped as the "simple" baseline for this experiment.

## 2026-05-17 - Gemma 4 gate+up packed matmul tuning

Experiment file: `src/matmul.cu` at the time of the experiment. That runtime
copy has since been removed; the active decode implementation is now in
`src/gemma4_matmul_kernels.cu`.

Target:

- Packed FFN gate+up projection for Gemma 4 dense 31B.
- A: `[M, 5376]`, row-major half.
- B storage: `[43008, 5376]`, row-major half, equivalent to transposed `[5376, 43008]`.
- C: `[M, 43008]`, row-major half.
- Current specialized default: `M=512`.

Build:

```bash
nvcc -std=c++17 -O3 -arch=sm_86 -DGEMMA4_STANDALONE_BENCH \
  -I/tmp/cutlass/include -I/tmp/cutlass/tools/util/include \
  src/matmul.cu -lcublas -o build/matmul_bench
```

Best source configuration so far:

- CTA tile: `BM=128, BN=128, BK=32`
- Pipeline stages: `4`
- Tensor core op: `SM80_16x8x16_F32F16F16F32_TN`
- Baseline: `cublasGemmEx`, FP16 inputs, FP32 accumulation, tensor-op path.

Final default timing command:

```bash
./build/matmul_bench both 100 20
```

CUDA-event timing results at `M=512`:

| Kernel | Avg ms | TFLOP/s |
| --- | ---: | ---: |
| Custom CUTE kernel | 1.87 | 126.61 |
| cuBLAS `cublasGemmEx` | 2.03743 | 116.205 |

Comparison:

- `custom_vs_cublas_speedup = 1.08954`
- Custom is about `9.0%` faster than cuBLAS for this specialized `M=512` packed gate+up shape.
- Correctness check: `max_abs_diff = 0`.

Tuning sweep notes:

- Initial copied tile retargeted to `[M,5376] x [5376,43008]` with `BM=128, BN=64, BK=64, Stages=2` was much slower: at `M=1024`, custom `4.9356 ms` vs cuBLAS `3.71269 ms`.
- `BM=128, BN=128, BK=64, Stages=2` improved to near parity: custom `3.94334 ms` vs cuBLAS `3.86984 ms` at `M=1024`.
- `BM=128, BN=128, BK=64, Stages=3` beat cuBLAS in a short run, but not enough.
- `BM=128, BN=128, BK=32, Stages=4` is the best tested source shape so far.
- `BM=64, BN=128, BK=64, Stages=3` was bad.
- `BM=128, BN=256, BK=64, Stages=2` was extremely bad.
- `BM=256, BN=128, BK=32, Stages=4` was extremely bad.
- L2 persisting-cache hints for the B/weight matrix hurt performance and were removed.
- Register capping with `--maxrregcount=192` caused spills and badly hurt performance.

M sensitivity:

- `M=512`: stable win, `1.08954x` over cuBLAS on the final 100-iteration run.
- `M=1024`: borderline; one 50-iteration run reached `1.07005x`, but a 100-iteration run was only `1.06442x`.
- `M=1536` and `M=2048`: custom lost to cuBLAS.

Resource data from `ptxas -v` for the current custom kernel:

- Registers: `230`
- Static spills: `0 bytes`
- Barriers: `1`
- Constant memory: `388 bytes cmem[0]`

Profiler status:

- `ncu --section SpeedOfLight --section MemoryWorkloadAnalysis` still fails on this Thunder Compute instance with an internal Thunder runtime assertion before producing kernel metrics.
- Because `ncu` is blocked here, the trustworthy evidence for this entry is CUDA-event timing, exact-output comparison against cuBLAS, and `ptxas` resource data.

Tooling/source notes:

- `$cuda-programming-guide` was used repeatedly for SM 8.6 resource constraints, occupancy/register/shared-memory tradeoffs, async copy guidance, shared-memory bank conflict concerns, and L2 persisting cache semantics.
- `$exa-search` could not run because `EXA_API_KEY` is not set in this environment.

## 2026-05-17 - Matmul runtime skeleton simplification

Runtime file: `src/gemma4_matmul_kernels.cu`

Change:

- Replaced the copied CUTE benchmark implementation with a small runtime-facing skeleton in `src/gemma4_matmul_kernels.cu`.
- Prefill now routes through a host-side library GEMM wrapper for the fixed packed FFN gate+up shape.
- Decode now has a custom M=1 kernel that calls a single `__device__` dot-product primitive.
- The decode kernel computes four output columns per CTA and uses `half2` loads to reduce launch-grid size and reuse the M=1 input vector across four dot products.
- Kept the tuned CUTE result above as experiment history instead of carrying that complexity in the runtime path.

CUDA guide note:

- CUDA execution-space rules mean cuDNN/cuBLAS frontend calls belong in host code, not `__device__` functions. The device side is therefore limited to the custom decode matvec primitive.

## 2026-05-18 - M=1 decode vs cuDNN baseline

Runtime file: `src/gemma4_matmul_kernels.cu`

Benchmark file: `src/experiments/gemma4_decode_bench.cu`

Target:

- Decode-only packed FFN gate+up projection.
- Shape: `M=1, K=5376, N=43008`.
- Custom weight layout: `[43008, 5376]` row-major half, so each output column/dot product is contiguous.
- cuDNN baseline: equivalent 1x1 convolution with input `[1, 5376, 1, 1]`, filter `[43008, 5376, 1, 1]`, output `[1, 43008, 1, 1]`.

Why cuDNN convolution baseline:

- cuDNN backend matmul descriptors for the same operation finalized, but heuristics returned zero usable engine configs for both transposed-stride and contiguous B descriptor attempts.
- The 1x1 convolution formulation is the same decode math and uses the same physical `[N, K]` weight layout as the custom kernel.

Build:

```bash
make decode-bench
```

Final timing command:

```bash
./build/experiments/gemma4_decode_bench both 200 30
```

Final CUDA-event timing results:

| Kernel | Avg ms | Weight GB/s |
| --- | ---: | ---: |
| Custom decode | 0.652241 | 708.974 |
| cuDNN 1x1 conv | 3.191969 | 144.870 |

Comparison:

- `custom_vs_cudnn_speedup = 4.893846`
- Custom is about `387%` faster than cuDNN on this M=1 decode benchmark.
- Correctness check: `max_abs_diff = 0`.
- cuDNN selected convolution algo `1` with `172160` bytes of workspace.
- Device: NVIDIA RTX A6000, compute capability 8.6, driver `580.126.16`.

Tuning/resource notes:

- Kept 4 output columns per CTA using `half2` loads. This reduces grid size and reuses each loaded input element across four output dots.
- Thread-count sweep:
  - `GEMMA4_DECODE_THREADS=128`: `0.652592 ms`
  - `GEMMA4_DECODE_THREADS=256`: `0.652398 ms`
  - `GEMMA4_DECODE_THREADS=512`: `0.680386 ms`
- Kept 256 threads per CTA.
- Added `__launch_bounds__(256, 2)` and moved thread-index ownership out of the reusable `__device__` helper to match the repo CUDA kernel-structure rule.
- Final `ptxas -v` resource data:
  - Registers: `25`
  - Static spills: `0 bytes`
  - Shared memory: `128 bytes`
  - Barriers: `1`
  - Constant memory: `376 bytes cmem[0]`
- Theoretical occupancy estimate on SM 8.6: 256 threads/CTA gives 8 warps/CTA; the 1536-thread/48-warp SM limit allows 6 CTAs/SM, so the kernel can reach 48 active warps/SM before register or shared-memory limits bind.

Profiler status:

- `ncu --set full --target-processes all --kernel-name regex:gemma4_ffn_gate_up_decode_kernel --launch-count 1 ./build/experiments/gemma4_decode_bench custom 1 0` fails in this Thunder Compute environment with an internal unsupported-library assertion before producing metrics.
- `nsys profile --force-overwrite=true --trace=cuda,nvtx,osrt --stats=true -o build/experiments/gemma4_decode_bench_nsys ./build/experiments/gemma4_decode_bench both 10 2` completes and writes `.nsys-rep`/`.sqlite`, but the generated sqlite contains no CUDA trace, kernel, or GPU memory data.
- `nvprof ./build/experiments/gemma4_decode_bench custom 5 1` is unavailable for this GPU class; it reports that `nvprof` is not supported on compute capability 8.0 and higher.
- Because profiler CUDA data is blocked here, the reliable evidence for this entry is CUDA-event timing, exact-output comparison against cuDNN, `ptxas` resource data, and the explicit profiler failure output.

Tooling/source notes:

- `$cuda-programming-guide` was queried for coalesced memory access, occupancy/resource constraints, launch bounds, shared-memory/bank-conflict considerations, and profiler/occupancy interpretation. Relevant guide pages returned included 60-61, 67, 70-71, 119, 130, 445, and 526.
- `$exa-search` could not run because `EXA_API_KEY` is not set in this environment.

## 2026-05-18 - M=1 decode vs cuBLAS GEMM/GEMV baselines

Benchmark file: `src/experiments/gemma4_decode_bench.cu`

Reason:

- The cuDNN 1x1 convolution baseline is mathematically equivalent, but it is not the cleanest library matmul baseline.
- The lean baseline for this decode matmul is `cublasGemmEx`, using the existing `[N, K]` weight layout as a column-major `[K, N]` matrix and computing `Y^T[N, 1] = W^T[N, K] * X^T[K, 1]`.
- Because M=1 is also a matrix-vector operation, the benchmark now also includes `cublasHSHgemvStridedBatched` with `batchCount=1`.
- cuBLASLt was briefly wired in, but removed from the benchmark because the simple cuBLAS GEMM/GEMV calls are the correct small baselines for this question.

Command:

```bash
./build/experiments/gemma4_decode_bench both 50 10 3
```

Results:

| Kernel | Best ms | Avg ms | Best weight GB/s |
| --- | ---: | ---: | ---: |
| Custom decode | 0.652367 | 0.652407 | 708.838 |
| cuBLAS `cublasGemmEx` | 3.333772 | 3.486190 | 138.708 |
| cuBLAS `cublasHSHgemvStridedBatched` | 3.413318 | 3.593821 | 135.476 |
| cuDNN 1x1 conv | 3.138264 | 3.222237 | 147.350 |

Comparison:

- `custom_vs_cublas_gemmex_speedup = 5.110273`
- `custom_vs_cublas_hshgemv_strided_batched_speedup = 5.232207`
- `custom_vs_cudnn_speedup = 4.810584`
- Correctness check: `cublas_gemmex_max_abs_diff = 0`.
- Correctness check: `cublas_hshgemv_strided_batched_max_abs_diff = 0`.
- The cuBLAS GEMM/GEMV baselines are more credible library comparisons than cuDNN convolution for the M=1 decode matmul, and they are still far behind the custom contiguous-weight matvec kernel.

Tooling/source notes:

- `$cuda-programming-guide` was queried again for CUDA-X/cuBLAS/Tensor Core guidance and GEMV/memory-access considerations. The local guide points to CUDA-X libraries such as cuBLAS as the recommended library path for Tensor Core operations on supported hardware, and reiterates coalesced memory access as a first-order concern.

## 2026-05-18 - Decode GEMV coverage for main dense projections

Runtime file: `src/gemma4_matmul_kernels.cu`

Benchmark file: `src/experiments/gemma4_decode_bench.cu`

Change:

- Generalized the one-off packed FFN gate+up decode kernel into a templated fixed-shape decode GEMV kernel.
- Added decode entry points for:
  - `ffn_gate_up`: `x[5376] -> gate_up[43008]`
  - `ffn_down`: `ffn_hidden[21504] -> hidden[5376]`
  - `sliding_qkv`: `x[5376] -> qkv[16384]`
  - `sliding_o`: `attn_out[8192] -> hidden[5376]`
  - `global_q`: `x[5376] -> q[16384]`
  - `global_k`: `x[5376] -> k[2048]`
  - `global_o`: `attn_out[16384] -> hidden[5376]`
  - `final_logits`: `hidden[5376] -> logits[262144]`
- Kept the same physical weight layout as the original gate/up kernel: `[N, K]` row-major half, so each output dot product reads a contiguous weight row.
- Simplified the benchmark to compare against direct cuBLAS GEMV (`cublasHSHgemvStridedBatched`) and M=1 `cublasGemmEx`. Removed the cuDNN 1x1 convolution baseline from this benchmark because it is no longer the relevant comparison.

Build:

```bash
make decode-bench
make cuda-kernels
```

Smoke timing command:

```bash
./build/experiments/gemma4_decode_bench all 2 1 1
```

Smoke results from the first low-iteration run:

| Op | K | N | Custom best ms | cuBLAS GEMV best ms | cuBLAS GEMM M=1 best ms | Max diff vs GEMV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ffn_gate_up` | 5376 | 43008 | 0.660864 | 3.579680 | 3.446096 | 0.03125 |
| `ffn_down` | 21504 | 5376 | 0.341408 | 0.340816 | 0.340544 | 0 |
| `sliding_qkv` | 5376 | 16384 | 0.264544 | 2.905792 | 2.898672 | 0 |
| `sliding_o` | 8192 | 5376 | 0.133952 | 2.978928 | 2.423568 | 0.03125 |
| `global_q` | 5376 | 16384 | 0.266848 | 2.411344 | 2.354640 | 0 |
| `global_k` | 5376 | 2048 | 0.042096 | 2.560576 | 3.245296 | 0 |
| `global_o` | 16384 | 5376 | 0.265968 | 0.263088 | 0.263392 | 0 |
| `final_logits` | 5376 | 262144 | 3.979248 | 3.997824 | 3.980464 | 0 |

Notes:

- This is a low-iteration smoke pass, not a stable tuning run.
- Some outputs differ from cuBLAS by `0.03125`, which is one half-scale step for this deterministic input pattern and is expected from different FP32 reduction orders before FP16 output rounding.
- The current generic kernel is already very strong when cuBLAS dispatch overhead/layout choices dominate smaller decode shapes, but it is only at parity for `ffn_down`, `global_o`, and `final_logits`. Those need actual tuning before claiming a win.

Sequential stability check:

```bash
./build/experiments/gemma4_decode_bench all 50 10 3
```

| Op | Custom best ms | cuBLAS GEMV best ms | cuBLAS GEMM M=1 best ms | Best custom speedup vs GEMV |
| --- | ---: | ---: | ---: | ---: |
| `ffn_gate_up` | 0.652664 | 3.096099 | 3.124034 | 4.743790 |
| `ffn_down` | 0.327303 | 0.330160 | 0.330163 | 1.008729 |
| `sliding_qkv` | 0.250743 | 2.808276 | 2.542720 | 11.199816 |
| `sliding_o` | 0.127187 | 3.486035 | 3.434500 | 27.408831 |
| `global_q` | 0.251055 | 2.511043 | 2.200656 | 10.001947 |
| `global_k` | 0.036893 | 2.202849 | 2.701255 | 59.709446 |
| `global_o` | 0.250511 | 0.252756 | 0.252499 | 1.008960 |
| `final_logits` | 3.961647 | 3.965804 | 3.965508 | 1.001049 |

Interpretation:

- The large wins are credible for this exact benchmark setup because the run is sequential, warm, and uses the same physical `[N, K]` half weight layout for custom and cuBLAS.
- These are still only baseline comparisons against simple cuBLAS GEMV/GEMM calls, not proof that the kernels beat every possible library/layout path.
- `ffn_down`, `global_o`, and `final_logits` are effectively parity with cuBLAS. Treat them as coverage/correctness baselines, not optimized wins.

## 2026-05-18 - Decode parity-kernel tuning: theoretical max and plan

Runtime file: `src/gemma4_matmul_kernels.cu`

Benchmark file: `src/experiments/gemma4_decode_bench.cu`

Goal:

- Tune the three decode GEMV kernels that are currently only at cuBLAS parity:
  - `ffn_down`: `K=21504, N=5376`
  - `global_o`: `K=16384, N=5376`
  - `final_logits`: `K=5376, N=262144`
- Keep changes small: thread count, columns per CTA, launch bounds, reduction shape, and similarly local kernel-architecture changes.
- Stop a kernel only after a clear flat line: seven tried variants without a better result.

Theoretical memory roofline:

- Device: NVIDIA RTX A6000, compute capability 8.6.
- CUDA-reported properties from PyTorch:
  - `memory_clock_rate = 8001000 kHz`
  - `memory_bus_width = 384 bits`
  - `multi_processor_count = 84`
  - `max_threads_per_multi_processor = 1536`
- Peak theoretical DRAM bandwidth: `8001000 kHz * 1000 * 2 * (384 / 8) = 768.096 GB/s`.
- These M=1 decode kernels are weight-streaming GEMVs. The absolute lower bound below counts weight bytes only, `N * K * sizeof(half)`. Input and output traffic are tiny by comparison for these shapes, and repeated input reads make this an optimistic lower bound.

| Op | K | N | Weight GB | Theoretical min ms @ 768.096 GB/s | Current best ms | Current weight GB/s | Peak % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ffn_down` | 21504 | 5376 | 0.231211 | 0.301018 | 0.327303 | 706.413 | 91.97% |
| `global_o` | 16384 | 5376 | 0.176161 | 0.229347 | 0.250511 | 703.206 | 91.55% |
| `final_logits` | 5376 | 262144 | 2.818572 | 3.669557 | 3.961647 | 711.465 | 92.63% |

Interpretation before tuning:

- The three parity kernels are already around `91.5-92.6%` of the simple theoretical DRAM bandwidth roofline by the benchmark's weight-GB/s metric.
- A "much faster than baseline" result may be mathematically limited unless the baseline comparison changes or the kernel reduces real traffic beyond the current one-pass contiguous weight stream.
- Tuning should still probe whether the last `7-8%` gap is launch/reduction overhead, instruction mix, occupancy, or memory-transaction inefficiency.

Tuning implementation:

- Added compile-time knobs for:
  - `GEMMA4_DECODE_COLS_PER_BLOCK`
  - `GEMMA4_DECODE_MIN_BLOCKS_PER_SM`
- Kept the original hand-written fixed-four-column path for default `ColsPerBlock=4`.
- Added a templated fixed-four launcher so individual decode projections can use a different thread count or launch-bound min-block setting without moving every projection off the known-good default.
- Final selected specializations:
  - `ffn_down`: `Threads=256`, `MinBlocksPerSm=4`
  - `global_o`: `Threads=512`, `MinBlocksPerSm=1`
  - `final_logits`: `Threads=512`, `MinBlocksPerSm=2`

Fresh baseline before tuning:

```bash
make decode-bench
./build/experiments/gemma4_decode_bench ffn_down 80 20 5
./build/experiments/gemma4_decode_bench global_o 80 20 5
./build/experiments/gemma4_decode_bench final_logits 40 10 3
```

| Op | Custom best ms | cuBLAS GEMV best ms | Custom weight GB/s | Speedup vs GEMV |
| --- | ---: | ---: | ---: | ---: |
| `ffn_down` | 0.327326 | 0.330148 | 706.364 | 1.008623 |
| `global_o` | 0.250404 | 0.252554 | 703.507 | 1.008589 |
| `final_logits` | 3.961670 | 3.965620 | 711.461 | 1.000997 |

Sweep notes:

- `ffn_down`:
  - Tried `ColsPerBlock={1,2,4,8}` crossed with `Threads={128,256,512}`.
  - Tried `MinBlocksPerSm={1,2,3,4,6}` at the default fixed-four shape.
  - The column/thread sweep did not beat the original default. Best non-default grouping was `ColsPerBlock=2, Threads=512` at `0.327899 ms`, still slower than the fresh baseline.
  - `MinBlocksPerSm=4` produced a repeatable tiny improvement: `0.327280 ms` in the verification run.
  - This is effectively flat after more than seven variants; the selected specialization is only a small compiler launch-bound gain.
- `global_o`:
  - Tried `ColsPerBlock={1,2,4,8}` crossed with `Threads={128,256,512}`.
  - Tried `Threads=512` with `MinBlocksPerSm={1,2,3}`.
  - Wider/narrower grouping was mostly worse. The repeatable winner was the fixed-four shape with `Threads=512, MinBlocksPerSm=1`.
  - Verification run reached `0.250226 ms`.
  - This also flat-lined after more than seven variants; the selected specialization is a small occupancy/compiler scheduling adjustment.
- `final_logits`:
  - Tried `ColsPerBlock={1,2,4,8,16}` crossed with `Threads={128,256,512}`.
  - Tried `Threads=512` with `MinBlocksPerSm={1,2,3}`.
  - `Threads=512` was the only consistently useful knob. The best stable shape was fixed-four columns with `MinBlocksPerSm=2`.
  - Verification runs were around `3.9566-3.9567 ms`.
  - `ColsPerBlock=8` and `16` lost bandwidth, likely because larger per-CTA work reduced useful scheduling parallelism/register behavior more than it helped input reuse.

Final verification:

```bash
rm -f build/experiments/gemma4_decode_bench build/gemma4_matmul_kernels.o
make decode-bench cuda-kernels
./build/experiments/gemma4_decode_bench ffn_down 100 25 5
./build/experiments/gemma4_decode_bench global_o 100 25 5
./build/experiments/gemma4_decode_bench final_logits 50 12 5
```

| Op | Final custom best ms | cuBLAS GEMV best ms | Custom weight GB/s | Peak % | Speedup vs GEMV | Gain vs fresh custom |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ffn_down` | 0.327243 | 0.330212 | 706.543 | 91.99% | 1.009074 | 0.025% |
| `global_o` | 0.250207 | 0.252314 | 704.061 | 91.66% | 1.008423 | 0.079% |
| `final_logits` | 3.956729 | 3.965828 | 712.349 | 92.74% | 1.002300 | 0.125% |

Correctness:

- `cublas_gemv_max_abs_diff = 0` for all three final target runs.
- `cublas_gemm_m1_max_abs_diff = 0` for all three final target runs.

Resource data:

```bash
rm -f build/gemma4_matmul_kernels.o
make cuda-kernels NVCCFLAGS="-std=c++17 -O3 -arch=sm_86 -Xptxas=-v"
```

| Kernel specialization | Registers | Static spills | Shared memory | Barriers |
| --- | ---: | ---: | ---: | ---: |
| `ffn_down` fixed4 `K=21504,N=5376,Threads=256,MinBlocks=4` | 25 | 0 bytes | 128 bytes | 1 |
| `global_o` fixed4 `K=16384,N=5376,Threads=512,MinBlocks=1` | 25 | 0 bytes | 256 bytes | 1 |
| `final_logits` fixed4 `K=5376,N=262144,Threads=512,MinBlocks=2` | 25 | 0 bytes | 256 bytes | 1 |

Conclusion:

- The target kernels are already close to a one-pass DRAM streaming roofline. The tuning sweeps produced only small gains, not a large breakout.
- The flat-line condition is satisfied for all three target kernels: each had at least seven variants tried without material improvement beyond tiny launch-bound/thread-count effects.
- Further large wins probably require a different baseline comparison, a different data type/layout, fusing the producer/consumer around these GEMVs to remove memory traffic, or changing the math approach rather than local GEMV launch tuning.

Compile resource check:

```bash
nvcc -std=c++17 -O3 -arch=sm_86 -Xptxas -v \
  -c src/gemma4_matmul_kernels.cu \
  -o /tmp/gemma4_matmul_kernels_review.o
```

`ptxas` reported the same resource usage for every instantiated decode GEMV kernel:

- Registers: `24`
- Static spills: `0 bytes`
- Shared memory: `128 bytes`
- Barriers: `1`
- Constant memory: `376 bytes cmem[0]`

Covered instantiations:

| K | N | Cols/block |
| ---: | ---: | ---: |
| 5376 | 262144 | 4 |
| 16384 | 5376 | 4 |
| 5376 | 2048 | 4 |
| 8192 | 5376 | 4 |
| 5376 | 16384 | 4 |
| 21504 | 5376 | 4 |
| 5376 | 43008 | 4 |

## 2026-05-18 - Token embedding gather CUDA-event timing

Runtime file: `src/gemma4_embedding_gather.cu`

Benchmark file: `src/experiments/gemma4_embedding_gather_bench.cu`

Target:

- Gemma 4 31B dense token embedding gather.
- Shape: token id -> hidden row `[5376]`.
- Vocabulary: `262144`.
- Benchmark dimensions are read from the compile-time constants in `src/gemma4.h`.
- Embedding table allocation: `2818572288` bytes.
- Kernel: one warp per token, 16-byte `int4` vectorized BF16 row copy.
- Effective bandwidth below counts embedding reads plus output writes:
  `2 * num_tokens * hidden_size * sizeof(bfloat16)`.

Build:

```bash
make embedding-gather-bench
```

Correctness:

```bash
make test-embedding-gather
```

Result: `embedding gather tests passed`.

Timing command:

```bash
./build/experiments/gemma4_embedding_gather_bench 200 30 5 4096
```

CUDA-event timing results:

| Tokens | Best ms | Avg ms | Best effective GiB/s | Effective MiB |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.014669 | 0.015762 | 1.365 | 0.021 |
| 4 | 0.014482 | 0.015113 | 5.532 | 0.082 |
| 16 | 0.012178 | 0.014270 | 26.312 | 0.328 |
| 64 | 0.012052 | 0.016656 | 106.348 | 1.312 |
| 256 | 0.011847 | 0.013527 | 432.751 | 5.250 |
| 1024 | 0.036626 | 0.036687 | 559.931 | 21.000 |
| 4096 | 0.135538 | 0.135624 | 605.226 | 84.000 |

Larger sweep command:

```bash
./build/experiments/gemma4_embedding_gather_bench 100 20 3 8192
```

Extended sweep results:

| Tokens | Best ms | Avg ms | Best effective GiB/s | Effective MiB |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.013832 | 0.014062 | 1.448 | 0.021 |
| 4 | 0.013962 | 0.014258 | 5.738 | 0.082 |
| 16 | 0.014069 | 0.014502 | 22.775 | 0.328 |
| 64 | 0.013707 | 0.013958 | 93.513 | 1.312 |
| 256 | 0.013732 | 0.014498 | 373.354 | 5.250 |
| 1024 | 0.036742 | 0.036783 | 558.151 | 21.000 |
| 4096 | 0.135601 | 0.135702 | 604.945 | 84.000 |
| 8192 | 0.265173 | 0.265315 | 618.700 | 168.000 |

Theoretical memory roofline:

- A6000 max memory clock from `nvidia-smi -q -d CLOCK`: `8001 MHz`.
- A6000 memory bus width: `384 bits`.
- GDDR6 is DDR, so the peak byte rate is:
  `8001000 kHz * 1000 * 2 * (384 / 8) = 768096000000 bytes/s`.
- Decimal peak: `768.096 GB/s`.
- Binary peak: `715.345 GiB/s`.
- The benchmark reports GiB/s, so compare measured bandwidth against `715.345 GiB/s`.

| Tokens | Effective MiB | Theoretical min ms | Best ms | Best GiB/s | Peak % |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.021 | 0.000029 | 0.013832 | 1.448 | 0.20% |
| 4 | 0.082 | 0.000112 | 0.013962 | 5.738 | 0.80% |
| 16 | 0.328 | 0.000448 | 0.014069 | 22.775 | 3.18% |
| 64 | 1.312 | 0.001791 | 0.013707 | 93.513 | 13.07% |
| 256 | 5.250 | 0.007167 | 0.013732 | 373.354 | 52.19% |
| 1024 | 21.000 | 0.028668 | 0.036742 | 558.151 | 78.03% |
| 4096 | 84.000 | 0.114674 | 0.135601 | 604.945 | 84.57% |
| 8192 | 168.000 | 0.229347 | 0.265173 | 618.700 | 86.49% |

Compile resource check:

```bash
nvcc -std=c++17 -O3 -arch=sm_86 -Isrc -Xptxas=-v \
  -c src/gemma4_embedding_gather.cu \
  -o /tmp/gemma4_embedding_gather_ptxas.o
```

`ptxas` resource data:

- Registers: `32`
- Static spills: `0 bytes`
- Barriers: `0`
- Constant memory: `388 bytes cmem[0]`

Interpretation:

- Small token counts are launch-latency dominated in this environment, so their percent of peak is not a useful kernel-quality signal.
- Bandwidth becomes meaningful by `256` tokens and plateaus around `605-619 GiB/s` effective read+write bandwidth by `4096-8192` tokens.
- The best measured large-token case is `618.700 GiB/s`, which is `86.49%` of the A6000 theoretical DRAM roofline by this benchmark's read+write byte model.
- At `4096` tokens, `604.945 GiB/s` is `84.57%` of theoretical peak.
- This is good for a simple one-warp-per-token baseline. The remaining raw-memory roofline headroom is about `15%`, and some of that is not realistically recoverable because the measured kernel still pays launch overhead, token-id reads, address computation, control flow, and cache/transaction inefficiency.
- Optimization focus: improve small-token decode/prompt latency through batching or fusion; for large prefill gathers, this baseline is already close to the memory-bound roofline.
- Nsight Compute and Nsight Systems CUDA metrics remain blocked in this Thunder-mediated container; these results are CUDA-event timings only.

## 2026-05-18 - Decode GEMV BF16 randomized validation cleanup

Runtime file: `src/gemma4_matmul_kernels.cu`

Benchmark file: `src/experiments/gemma4_decode_bench.cu`

Reason:

- The decode benchmark was still using FP16 custom/cuBLAS paths and a deterministic tiny integer pattern.
- That was acceptable for early smoke tests, but it is not representative of the Gemma 4 BF16 inference path.
- The benchmark now validates BF16 custom kernels against BF16 cuBLAS references with randomized BF16 inputs and weights.

Implementation:

- Changed the decode projection API and kernels from `half` to `__nv_bfloat16`.
- Changed vectorized loads from `half2` to `__nv_bfloat162`.
- Custom kernels still accumulate in FP32 and round output to BF16.
- Prefill and M=1 cuBLAS GEMM baselines now use `CUDA_R_16BF` inputs/outputs with `CUBLAS_COMPUTE_32F`.
- GEMV baseline now uses `cublasTSTgemvStridedBatched`, which takes BF16 input/output and FP32 scalar parameters.
- Replaced the deterministic small fill pattern with a runtime-seeded GPU hash fill:
  - default seed comes from `std::random_device` and high-resolution clock
  - set `GEMMA4_DECODE_BENCH_SEED=<seed>` to reproduce a run
  - input scale: `1.0`
  - weight scale: `0.5`
- Correctness reporting now prints max absolute, mean absolute, and relative difference against each BF16 cuBLAS reference instead of expecting exact output.

Build:

```bash
rm -f build/experiments/gemma4_decode_bench build/gemma4_matmul_kernels.o
make decode-bench cuda-kernels
```

Representative BF16 randomized checks:

```bash
./build/experiments/gemma4_decode_bench ffn_down 10 3 2
./build/experiments/gemma4_decode_bench final_logits 5 2 1
./build/experiments/gemma4_decode_bench all 2 1 1
```

`ffn_down` result:

| Op | Custom best ms | BF16 cuBLAS GEMV best ms | BF16 cuBLAS GEMM M=1 best ms | Max abs diff vs GEMV | Mean abs diff vs GEMV |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ffn_down` | 0.328960 | 0.331690 | 0.331952 | 0 | 0 |

`final_logits` result:

| Op | Custom best ms | BF16 cuBLAS GEMV best ms | BF16 cuBLAS GEMM M=1 best ms | Max abs diff vs GEMV | Mean abs diff vs GEMV |
| --- | ---: | ---: | ---: | ---: | ---: |
| `final_logits` | 3.963251 | 3.970521 | 3.971814 | 0.03125 | 6.06194e-07 |

Full smoke observations:

- `./build/experiments/gemma4_decode_bench all 2 1 1` completed for all decode projection shapes.
- BF16 cuBLAS GEMV/GEMM agree with each other for the reported differences.
- Some shapes now show BF16-scale nonzero differences, which is expected from different FP32 reduction orders before BF16 output rounding.
- The old exact `max_abs_diff = 0` expectation is no longer the right acceptance signal for randomized BF16 validation.

CUDA guide note:

- `$cuda-programming-guide` was queried for BF16 support. The local guide states that `__nv_bfloat16` is available through `<cuda_bf16.h>` and requires compute capability 8.0 or higher, and that BF16 WMMA/Tensor Core paths support BF16 inputs with FP32 accumulation.

## 2026-05-18 - Decode GEMV Packed128 streaming-load pass

Runtime file: `src/gemma4_matmul_kernels.cu`

Reason:

- The decode GEMV kernels were still loading one BF16 pair per instruction through `__nv_bfloat162`.
- The input vector is small and reused by every CTA, while projection weights are large one-pass streams.
- This pass keeps normal cached loads for the reused input vector and marks the weight stream with an explicit streaming pack load.

Implementation:

- Changed the fixed-four decode dot helper from half2-style 32-bit loads to `Packed128<__nv_bfloat16>` loads.
- Each load now covers 8 BF16 values per thread.
- Weight row loads use streaming 128-bit pack loads, which SASS emits as `LDG.E.EF.128` on this build.
- Input-vector loads use direct 128-bit pack loads, which SASS emits as `LDG.E.128.CONSTANT` on this build.
- Packed the four BF16 outputs into one 64-bit store in the fixed-four path.
- Templated the dot helper on `Threads` so the packed loop can use the compile-time stride.
- Added `__restrict__` on kernel/device pointer parameters.

Build/resource check:

```bash
nvcc -std=c++17 -O3 -arch=sm_86 -Isrc -Xptxas=-v \
  -c src/gemma4_matmul_kernels.cu \
  -o /tmp/gemma4_matmul_kernels_packed128.o
```

`ptxas` result for all instantiated decode GEMV kernels:

- Registers: `40`
- Static spills: `0 bytes`
- Shared memory: `128` or `256` bytes depending on thread count
- Barriers: `1`

SASS spot-check:

```bash
cuobjdump --dump-sass /tmp/gemma4_matmul_kernels_packed128.o | rg "LDG|STG"
```

Confirmed:

- `LDG.E.128.CONSTANT` for input-vector packed loads
- `LDG.E.EF.128` for weight packed loads
- `STG.E.64` for fixed-four output stores

Benchmark command:

```bash
make decode-bench
./build/experiments/gemma4_decode_bench all 20 5 2
```

Representative results:

| Op | Custom best ms | Custom weight GB/s | BF16 cuBLAS GEMV best ms | Speedup vs GEMV | Max abs diff vs GEMV |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ffn_gate_up` | 0.651875 | 709.372 | 3.528619 | 5.413029 | 0.25 |
| `ffn_down` | 0.328534 | 703.765 | 0.331218 | 1.008167 | 0.0625 |
| `sliding_qkv` | 0.250694 | 702.691 | 2.499530 | 9.970425 | 0.25 |
| `sliding_o` | 0.127248 | 692.195 | 3.325871 | 26.136917 | 0.25 |
| `global_q` | 0.250746 | 702.548 | 2.684933 | 10.707796 | 0.25 |
| `global_k` | 0.034682 | 634.922 | 3.032374 | 87.434677 | 0.25 |
| `global_o` | 0.250725 | 702.606 | 0.253286 | 1.010217 | 0 |
| `final_logits` | 3.956539 | 712.383 | 3.967304 | 1.002721 | 0.125 |

Verification:

```bash
make cuda-kernels
make test-embedding-gather
```

`test-embedding-gather` still passed.

Interpretation:

- The source-level vectorization worked: SASS now has 128-bit loads and 64-bit stores in the target path.
- Register usage increased from the earlier `~25` register shape to `40`, but there are still no spills.
- Performance stayed near the prior DRAM-streaming roofline band. The clearest win in this short run was `global_k`, while the already roofline-limited large streams were essentially flat.
- Nonzero BF16 differences are expected from the changed FP32 reduction order before BF16 output rounding; mean errors remained small in the benchmark output.

CUDA guide notes:

- `$cuda-programming-guide` was queried for coalesced global memory, L2 streaming/persisting cache policy, `#pragma unroll`, low-level load functions, and `__restrict__` optimizer behavior. Relevant local guide pages included 60-61, 352-354, 519, 562, and 566.

## 2026-05-18 - RMSNorm and fused residual RMSNorm from llm.c layernorm

Runtime files:

- `src/gemma4_rmsnorm.cu`
- `src/gemma4_rmsnorm.cuh`
- `tests/test_rmsnorm.cu`
- `src/experiments/gemma4_rmsnorm_bench.cu`

Reason:

- The next documented baseline kernel after embedding gather is RMSNorm width `5376`.
- The first useful fusion target is the llm.c-style residual add plus normalization pattern.
- Gemma uses RMSNorm rather than LayerNorm, so the llm.c structure was kept while changing the math to remove mean subtraction and bias.

Implementation:

- Started by copying `llm.c/llmc/layernorm.cuh` to `src/gemma4_rmsnorm.cu`, then reduced it to inference-only forward kernels.
- Standalone RMSNorm computes `x * rsqrt(mean(x*x) + eps) * weight`.
- Fused residual RMSNorm writes the BF16 residual and normalizes that rounded residual value, matching the standalone `residual_add` followed by `rmsnorm` path.
- Kept the llm.c execution structure:
  - one warp owns one row;
  - fallback warp kernels do not require dynamic shared memory;
  - fast kernels cache the shared weight vector and one row per warp in dynamic shared memory;
  - launcher falls back when opt-in dynamic shared memory is not available.
- Added standalone `gemma4_residual_add_bf16` because the fused path needs a direct split baseline.
- Added a cuDNN frontend comparison path in `gemma4_rmsnorm_bench.cu` using `graph.rmsnorm` when `cudnn_frontend.h` is available. The default include path is `/tmp/cudnn-frontend/include`.

Research notes:

- PyTorch RMSNorm docs and the RMSNorm paper use the same core method: normalize by root mean square of the last dimension and apply a learned scale.
- The installed cuDNN 9 headers expose `CUDNN_RMS_NORM` and document backend norm forward support for both training and inference.
- NVIDIA's `cudnn-frontend` sample `samples/cpp/norm/rmsnorm.cpp` shows the C++ graph frontend path used by the benchmark.
- `$cuda-programming-guide` was queried for cuDNN/frontend benchmarking context, but the local guide mostly returned CUDA Graph sections rather than cuDNN norm guidance. The useful cuDNN evidence came from installed headers and NVIDIA frontend samples.
- `$exa-search` was attempted, but `EXA_API_KEY` was not visible in the tool environment. Browser search and the cloned NVIDIA `cudnn-frontend` repo were used instead.

Build and correctness:

```bash
make test-rmsnorm
make rmsnorm-bench
make cuda-kernels test-embedding-gather test-rmsnorm
```

Results:

- `rmsnorm tests passed`
- `embedding gather tests passed`

Primary benchmark command:

```bash
GEMMA4_RMSNORM_BENCH_SEED=0x1234 ./build/experiments/gemma4_rmsnorm_bench 50 10 3 4096
```

Representative `width=5376` results on RTX A6000:

| Rows | RMSNorm ms | RMSNorm GiB/s | cuDNN RMSNorm ms | cuDNN GiB/s | cuDNN max abs | Residual ms | Fused residual+RMS ms | Split residual+RMS ms | Split/Fused |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.020845 | 1.441 | 12.067860 | 0.002 | 0 | 0.014512 | 0.023012 | 0.034068 | 1.480 |
| 4 | 0.026762 | 4.491 | 10.980160 | 0.011 | 0 | 0.010392 | 0.021272 | 0.043290 | 2.035 |
| 16 | 0.018671 | 25.746 | 11.996526 | 0.040 | 0 | 0.008806 | 0.013852 | 0.025149 | 1.816 |
| 64 | 0.025356 | 75.833 | 11.558676 | 0.166 | 0.000488281 | 0.016013 | 0.023057 | 0.032451 | 1.407 |
| 256 | 0.015133 | 508.238 | 11.656405 | 0.660 | 0.00390625 | 0.012454 | 0.024473 | 0.021196 | 0.866 |
| 1024 | 0.036369 | 845.936 | 13.003598 | 2.366 | 0.00390625 | 0.050689 | 0.069214 | 0.080882 | 1.169 |
| 4096 | 0.133018 | 925.157 | 12.982645 | 9.479 | 0.00390625 | 0.195166 | 0.266606 | 0.326477 | 1.225 |

Larger sweep command:

```bash
GEMMA4_RMSNORM_BENCH_SEED=0x1234 ./build/experiments/gemma4_rmsnorm_bench 30 5 2 8192
```

Additional `8192 x 5376` result:

| Rows | RMSNorm ms | RMSNorm GiB/s | cuDNN RMSNorm ms | cuDNN GiB/s | cuDNN max abs | Fused residual+RMS ms | Split residual+RMS ms | Split/Fused |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8192 | 0.262535 | 937.490 | 12.506362 | 19.680 | 0.00390625 | 0.523484 | 0.648212 | 1.238 |

Interpretation:

- The custom standalone RMSNorm reaches about `925-937 GiB/s` at 4096-8192 rows with this logical byte model.
- cuDNN frontend RMSNorm numerically matches the custom output within BF16 rounding, but this setup has a large fixed execution overhead on A6000 for these shapes.
- Fused residual+RMSNorm is generally faster than launching residual add then standalone RMSNorm once rows are large enough, saving the separate residual reread.
- The `rows=256` split/fused anomaly is measurement noise or launch/occupancy crossover territory; the larger rows show the expected fused win.

## 2026-05-18 - RMSNorm raw GPU timing with CUDA graph capture

Reason:

- The original cuDNN RMSNorm benchmark timed repeated frontend `graph.execute` calls inside a CUDA-event interval.
- That is useful as an integrated steady-state API measurement, but it is not a raw GPU-speed comparison because frontend dispatch can leave idle gaps between device work submissions.

Implementation:

- Added `time_ms_graph` in `src/experiments/gemma4_bench_utils.cuh`.
- The helper stream-captures `iters` repeats of the target operation into a CUDA graph, instantiates it once, warms up graph replay, then records CUDA events around one graph replay and divides by the captured repeat count. This excludes CUDA graph capture, graph instantiation, warmup launches, and per-call host/API submission from the returned graph timing.
- `src/experiments/gemma4_rmsnorm_bench.cu` now reports both the original stream-loop timings and raw graph-captured timings:
  - `rms_ms` / `cudnn_ms`: repeated host/API execute path.
  - `rms_graph_kernel_ms` / `cudnn_graph_kernel_ms`: graph replay timing of already-captured device work.

Guide note:

- `$cuda-programming-guide` was queried for CUDA event and graph benchmarking context. The relevant local guide result was page 76, which describes using CUDA events in streams to time stream work including kernels. CUDA Graph results also reinforce separating graph setup from repeated execution.
- The Gemma 4 architecture doc was checked for production dimensions. The 31B dense target uses `hidden_size=5376`, batch-1 serving, `sliding_window=1024`, and a 256K context family. The compiled config uses `GEMMA4_MAX_POSITION_EMBEDDINGS=262144`, so full-context RMSNorm was tested as `262144 x 5376`.

Verification:

```bash
make rmsnorm-bench
make test-rmsnorm
```

`test-rmsnorm` passed.

Custom graph introspection from a one-off `cudaGraphGetNodes` checker:

| Rows | Captured nodes | Node type | Grid | Block | Dynamic smem |
| ---: | ---: | --- | --- | --- | ---: |
| 1 | 1 | kernel | `(1,1,1)` | `(32,8,1)` | `96768` |
| 4096 | 1 | kernel | `(512,1,1)` | `(32,8,1)` | `96768` |

Benchmark command:

```bash
GEMMA4_RMSNORM_BENCH_SEED=0x1234 ./build/experiments/gemma4_rmsnorm_bench 50 10 3 4096
```

Representative `width=5376` results on RTX A6000:

| Rows | Custom stream ms | Custom graph-kernel ms | cuDNN stream ms | cuDNN graph-kernel ms |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.023225 | 0.004400 | 7.370522 | 0.002634 |
| 4 | 0.017364 | 0.004716 | 6.371414 | 0.002625 |
| 16 | 0.025805 | 0.005671 | 7.226905 | 0.002726 |
| 64 | 0.019809 | 0.005693 | 7.759317 | 0.003723 |
| 256 | 0.024250 | 0.007769 | 6.857354 | 0.006050 |
| 1024 | 0.036131 | 0.035123 | 6.592338 | 0.035253 |
| 4096 | 0.132870 | 0.131937 | 6.518328 | 0.133100 |

Larger sweep:

```bash
GEMMA4_RMSNORM_BENCH_SEED=0x1234 ./build/experiments/gemma4_rmsnorm_bench 30 5 2 8192
```

Largest-row result:

| Rows | Custom stream ms | Custom graph-kernel ms | cuDNN stream ms | cuDNN graph-kernel ms |
| ---: | ---: | ---: | ---: | ---: |
| 8192 | 0.262535 | 0.261745 | 11.022665 | 0.262959 |

Production-size sweep:

```bash
GEMMA4_RMSNORM_BENCH_SEED=0x1234 ./build/experiments/gemma4_rmsnorm_bench 5 2 1 262144
```

Production-shape results:

| Rows | Meaning | Custom stream ms | Custom graph-kernel ms | cuDNN stream ms | cuDNN graph-kernel ms |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | batch-1 decode token | 0.018899 | 0.005536 | 5.503744 | 0.003507 |
| 1024 | 31B sliding-window-sized prefill chunk | 0.042483 | 0.035629 | 6.324723 | 0.036326 |
| 262144 | compiled max-position full-context pass | 8.297798 | 8.291284 | 9.805408 | 8.304595 |

Interpretation:

- The old multi-millisecond cuDNN RMSNorm result was dominated by frontend/API submission overhead, not device work.
- For larger row counts, the raw graph-captured GPU times are effectively tied: custom `0.131937 ms` vs cuDNN `0.133100 ms` at `4096 x 5376`, and custom `0.261745 ms` vs cuDNN `0.262959 ms` at `8192 x 5376`.
- At the compiled full-context production ceiling (`262144 x 5376`), the graph-captured GPU times remain effectively tied: custom `8.291284 ms` vs cuDNN `8.304595 ms`.
- For small rows, cuDNN's captured device work is faster than the current custom kernel, but that advantage is completely hidden by frontend execute overhead outside graph replay.
- Going forward, use `*_graph_kernel_ms` when discussing raw GPU speed and `*_ms` when discussing integrated host/API cost.

## 2026-05-18 - Decode production GEMV vs cuDNN BF16 1x1 conv

Reason:

- The active decode benchmark compared custom BF16 GEMV against cuBLAS GEMV/GEMM, but the current tuning gate asks whether the production decode matmul path beats cuDNN by at least `5%`.
- cuDNN backend matmul is not the clean matmul baseline here, so the benchmark restores the earlier mathematically equivalent 1x1 convolution comparator for the physical `[N, K]` weight layout.

Implementation:

- Added a BF16 cuDNN 1x1 convolution comparator to `src/experiments/gemma4_decode_bench.cu` for every production decode projection.
- Updated `make decode-bench` to link `-lcudnn`.
- Kept the production decode GEMV kernel unchanged after testing narrow warp-owned-column and column-grouping variants; those variants did not improve the tight `ffn_gate_up` case enough to keep.

Verification:

```bash
make decode-bench
make cuda-kernels
GEMMA4_DECODE_BENCH_SEED=0x1234 ./build/experiments/gemma4_decode_bench all 50 10 3
```

Production decode projection results on RTX A6000:

| Op | K | N | Layers/token | Custom ms | cuDNN ms | Custom vs cuDNN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ffn_gate_up` | 5376 | 43008 | 60 | 0.650958 | 0.656280 | 1.008176 |
| `ffn_down` | 21504 | 5376 | 60 | 0.327823 | 0.331135 | 1.010103 |
| `sliding_qkv` | 5376 | 16384 | 50 | 0.249854 | 0.270535 | 1.082772 |
| `sliding_o` | 8192 | 5376 | 50 | 0.126481 | 0.148044 | 1.170484 |
| `global_q` | 5376 | 16384 | 10 | 0.249905 | 0.281123 | 1.124916 |
| `global_k` | 5376 | 2048 | 10 | 0.034093 | 0.701345 | 20.571634 |
| `global_o` | 16384 | 5376 | 10 | 0.250122 | 0.255940 | 1.023262 |
| `final_logits` | 5376 | 262144 | 1 | 3.955549 | 3.964382 | 1.002233 |

Weighted by production layer counts:

| Path | Weighted decode projection ms |
| --- | ---: |
| Custom decode GEMV | 86.840359 |
| cuDNN BF16 1x1 conv | 96.522312 |

Interpretation:

- The production decode projection mix is `1.111491x` faster than cuDNN, an `11.15%` weighted win, clearing the `5%` gate.
- The largest bandwidth-saturated projections are much tighter individually, especially `ffn_gate_up`, `ffn_down`, `global_o`, and `final_logits`.
- Narrow warp-tiling and column-grouping tests did not improve `ffn_gate_up`; the current custom kernel is already near the cuDNN 1x1-conv bandwidth roofline for that shape.

## 2026-05-18 - Decode GEMV post-gate tuning sweep

Reason:

- After clearing the `5%` production-weighted cuDNN gate, continue tuning until the obvious decode GEMV thread/column-route variants stop producing production-speed wins.
- Focus on the actual decode projection mix, not a single synthetic shape.

Method:

```bash
make decode-bench
GEMMA4_DECODE_BENCH_SEED=0x1234 ./build/experiments/gemma4_decode_bench <op> 30 5 2
GEMMA4_DECODE_BENCH_SEED=0x1234 ./build/experiments/gemma4_decode_bench all 50 10 3
GEMMA4_DECODE_BENCH_SEED=0x1234 ./build/experiments/gemma4_decode_bench all 100 20 5
make cuda-kernels
git diff --check
```

Screened variants:

| # | Variant | Screened custom result | Decision |
| ---: | --- | --- | --- |
| 1 | generic `threads=128, cols=4` | `ffn_gate_up 0.651714`, `sliding_o 0.126844`, `global_k 0.033860` | Rejected; high-weight ops slower. |
| 2 | generic `threads=512, cols=4` | `ffn_gate_up 0.650739`, `sliding_o 0.126753`, `global_k 0.034049` | Rejected; not a production win. |
| 3 | generic `threads=256, cols=8` | `ffn_gate_up 0.651315`, `sliding_o 0.127519`, `global_k 0.035267` | Rejected. |
| 4 | generic `threads=256, cols=2` | `ffn_gate_up 0.650470`, `sliding_o 0.126437`, `global_k 0.033897` | Candidate, later beaten. |
| 5 | generic `threads=128, cols=8` | `ffn_gate_up 0.651994`, `sliding_o 0.127132`, `global_k 0.034060` | Rejected. |
| 6 | generic `threads=512, cols=8` | `ffn_gate_up 0.651111`, `sliding_o 0.127395`, `global_k 0.034473` | Rejected. |
| 7 | generic `threads=128, cols=2` | `ffn_gate_up 0.650998`, `sliding_o 0.126688`, `global_k 0.034905` | Rejected. |
| 8 | generic `threads=512, cols=2` | `ffn_gate_up 0.650105`, `sliding_o 0.126433`, `global_k 0.033567` | Kept; best generic route. |
| 9 | `ffn_down fixed4<128,4>` | `ffn_down 0.328463` | Rejected. |
| 10 | `ffn_down fixed4<512,1>` | `ffn_down 0.327438` | Candidate, later beaten. |
| 11 | `ffn_down fixed4<512,2>` | `ffn_down 0.327405` | Candidate, later beaten. |
| 12 | `global_o fixed4<256,2>` | `global_o 0.250242` | Rejected. |
| 13 | `global_o fixed4<256,4>` | `global_o 0.250467` | Rejected. |
| 14 | `global_o fixed4<128,4>` | `global_o 0.250861` | Rejected. |
| 15 | `final_logits fixed4<256,4>` | `final_logits 3.957258` | Rejected. |
| 16 | generic `threads=1024, cols=2` | `ffn_gate_up 0.650092`, `sliding_qkv 0.250277`, `sliding_o 0.126439`, `global_k 0.034306`; ptxas ignored `.minnctapersm` | Rejected. |
| 17 | generic `threads=768, cols=2` | `ffn_gate_up 0.650050`, `sliding_qkv 0.249568`, `sliding_o 0.126182`, `global_k 0.033888` | Rejected; mixed and not better weighted. |
| 18 | generic `threads=640, cols=2` | `ffn_gate_up 0.650382`, `sliding_qkv 0.250560`, `sliding_o 0.126318`, `global_k 0.034333` | Rejected. |
| 19 | generic `threads=512, cols=1` | `ffn_gate_up 0.649681`, `sliding_qkv 0.249843`, `sliding_o 0.126537`, `global_k 0.034037` | Rejected; helps `ffn_gate_up` but loses weighted mix. |
| 20 | generic `threads=512, cols=16` | `ffn_gate_up 0.653839`, `sliding_qkv 0.254826`, `sliding_o 0.128980`, `global_k 0.035023` | Rejected. |
| 21 | `ffn_down fixed4<1024,1>` | `ffn_down 0.327121` | Kept; best `ffn_down` route. |
| 22 | `ffn_down fixed4<1024,2>` | `ffn_down 0.327613`; ptxas ignored `.minnctapersm` | Rejected. |
| 23 | `ffn_down fixed4<768,1>` | `ffn_down 0.327362` | Rejected. |
| 24 | `final_logits fixed4<1024,1>` | `final_logits 3.951809` | Kept; best `final_logits` route. |
| 25 | `final_logits fixed4<768,1>` | `final_logits 3.954659` | Rejected. |
| 26 | `final_logits fixed4<512,1>` | `final_logits 3.955488` | Rejected. |
| 27 | `global_o fixed4<1024,1>` | `global_o 0.250594` | Rejected. |
| 28 | `final_logits fixed4<896,1>` | `final_logits 3.952104` | Rejected. |
| 29 | `final_logits fixed4<640,1>` | `final_logits 3.954255` | Rejected. |
| 30 | `final_logits fixed4<1024,2>` | `final_logits 3.951425`; ptxas ignored `.minnctapersm` | Rejected. |
| 31 | `ffn_down fixed4<896,1>` | `ffn_down 0.327773` | Rejected. |
| 32 | `ffn_down fixed4<640,1>` | `ffn_down 0.327632` | Rejected. |
| 33 | `ffn_down fixed4<384,1>` | `ffn_down 0.327397` | Rejected. |
| 34 | `global_o fixed4<768,1>` | `global_o 0.250193` | Rejected. |
| 35 | `global_o fixed4<640,1>` | `global_o 0.250218` | Rejected. |
| 36 | `global_o fixed4<384,1>` | `global_o 0.250167` | Rejected. |
| 37 | generic `threads=384, cols=2` | `ffn_gate_up 0.651110`, `sliding_qkv 0.250326`, `sliding_o 0.126726`, `global_k 0.034033` | Rejected. |
| 38 | generic `threads=896, cols=2` | `ffn_gate_up 0.650482`, `sliding_qkv 0.250063`, `sliding_o 0.127317`, `global_k 0.034405`; ptxas ignored `.minnctapersm` | Rejected. |
| 39 | generic `threads=512, cols=32` | `ffn_gate_up 0.672537`, `sliding_qkv 0.268206`, `sliding_o 0.136106`, `global_k 0.037322` | Rejected. |

Retained source settings:

- Generic decode GEMV route: `threads=512`, `cols_per_block=2`.
- `ffn_down`: `gemma4_decode_gemv_fixed4<GEMMA4_INTERMEDIATE_SIZE, GEMMA4_HIDDEN_SIZE, 1024, 1>`.
- `global_o`: restored to `512,1`.
- `final_logits`: `gemma4_decode_gemv_fixed4<GEMMA4_HIDDEN_SIZE, GEMMA4_VOCAB_SIZE, 1024, 1>`.

Final confirmation, `iters=100`, `warmup=20`, `trials=5`:

| Op | K | N | Layers/token | Custom ms | cuDNN ms | Custom vs cuDNN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ffn_gate_up` | 5376 | 43008 | 60 | 0.649589 | 0.655538 | 1.009157 |
| `ffn_down` | 21504 | 5376 | 60 | 0.326650 | 0.330976 | 1.013242 |
| `sliding_qkv` | 5376 | 16384 | 50 | 0.249276 | 0.271879 | 1.090673 |
| `sliding_o` | 8192 | 5376 | 50 | 0.125808 | 0.142452 | 1.132293 |
| `global_q` | 5376 | 16384 | 10 | 0.249468 | 0.270730 | 1.085232 |
| `global_k` | 5376 | 2048 | 10 | 0.033313 | 0.817800 | 24.549238 |
| `global_o` | 16384 | 5376 | 10 | 0.249828 | 0.255338 | 1.022057 |
| `final_logits` | 5376 | 262144 | 1 | 3.950665 | 3.964759 | 1.003567 |

Weighted by production layer counts:

| Path | Weighted decode projection ms |
| --- | ---: |
| Previous custom decode GEMV | 86.840359 |
| Tuned custom decode GEMV | 86.605295 |
| cuDNN BF16 1x1 conv in confirmation run | 97.310829 |

Interpretation:

- The retained tuning is a small additional production win: `0.235064 ms` weighted, or about `0.27%` beyond the previous custom decode GEMV result.
- The tuned production decode projection mix is `1.123607x` faster than cuDNN in the longer confirmation run, a `12.36%` weighted win.
- After the final retained win (`final_logits fixed4<1024,1>`), variants 25-39 formed a 15-test no-win tail. Those variants either regressed high-weight projections, emitted invalid launch-bound warnings, or only helped one op while losing the weighted production mix.

## 2026-05-18 - Decode RMSNorm production-shape tuning vs cuDNN graph

Reason:

- The production decode RMSNorm shape is one token row by Gemma 4 31B hidden width: `rows=1`, `width=5376`, `eps=1e-6`.
- The previous shared-memory RMSNorm path was optimized around multiple rows per CTA. At `rows=1`, only one warp did row math after the CTA-wide shared preload, which left most of the CTA idle.
- The fair raw-GPU comparator is the graph-captured cuDNN RMSNorm timing, not the cuDNN stream timing, because cuDNN frontend execution has large host/API cost outside graph replay.

Implementation:

- Added decode-specialized `rows=1,width=5376` kernels in `src/gemma4_rmsnorm.cu`.
- Kept the existing multi-row shared/warp paths unchanged.
- Added graph timings for fused residual+RMSNorm and split residual-add-plus-RMSNorm in `src/experiments/gemma4_rmsnorm_bench.cu`.
- Retained settings:
  - standalone decode RMSNorm: `768` threads.
  - fused decode residual+RMSNorm: `1024` threads.

Commands:

```bash
make test-rmsnorm rmsnorm-bench
./build/tests/test_rmsnorm
GEMMA4_RMSNORM_BENCH_SEED=0x1234 ./build/experiments/gemma4_rmsnorm_bench 400 50 7 1
make cuda-kernels
git diff --check
```

Screened decode variants, all on `rows=1,width=5376`:

| Variant | RMS graph ms | cuDNN graph ms | Fused graph ms | Decision |
| --- | ---: | ---: | ---: | --- |
| Previous shared path | 0.003918 | 0.002507 | n/a | Failed gate. |
| Decode shared, RMS `256`, fused `256` | 0.002733 | 0.002489 | n/a | Faster, still failed gate. |
| Decode shared, RMS `512`, fused `512` | 0.002455 | 0.002481 | n/a | Slight RMS win, below 5% gate. |
| Decode shared, RMS `1024`, fused `1024` | 0.002096 | 0.002475 | n/a | Cleared RMS gate; fused stream regressed. |
| RMS `1024`, fused `256` | 0.001931 | 0.002484 | 0.002979 | Good RMS, fused graph not best. |
| RMS `1024` no shared input cache, fused `256` | 0.001985 | 0.002460 | 0.002918 | Rejected; shared input cache was faster. |
| RMS `1024`, fused `512` | 0.002035 | 0.002521 | 0.002666 | Fused improved, RMS still passes. |
| RMS `1024`, fused `1024` | 0.001991 | 0.002500 | 0.002488 | Good balanced candidate. |
| RMS `1024`, fused `768` | 0.001986 | 0.002472 | 0.002526 | Rejected; fused worse than `1024`. |
| RMS `768`, fused `1024` | 0.001968 | 0.002481 | 0.002413 | Best short-run balance; kept for confirmation. |
| RMS `896`, fused `1024` | 0.002018 | 0.002480 | 0.002413 | Rejected; RMS worse than `768`. |
| RMS `640`, fused `1024` | 0.002342 | 0.002474 | 0.002486 | Rejected. |

Final confirmation, `iters=400`, `warmup=50`, `trials=7`:

| Rows | Width | RMS graph ms | cuDNN graph ms | Custom vs cuDNN | Fused graph ms | Split graph ms | Fused vs split |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 5376 | 0.001986 | 0.002606 | 1.312185 | 0.002236 | 0.003095 | 1.384 |

Interpretation:

- The tuned standalone decode RMSNorm is `1.312185x` faster than cuDNN graph timing, a `23.79%` lower-latency raw GPU win, clearing the `5%` gate.
- The tuned fused residual+RMSNorm decode kernel is `1.384x` faster than split residual add plus RMSNorm under graph timing.
- Stream timings also improved materially for decode: RMSNorm went from `0.015677 ms` to `0.010656 ms`, and fused residual+RMSNorm went from `0.026772 ms` to `0.012544 ms`.

## 2026-05-19 - Consolidated decode optimization findings from chat

Reason:

- This entry collects the main decisions, answers, and follow-up work from the decode GEMV, cache-hint, tensor-core, async-loading, cuDNN-comparison, warp-tiling, and RMSNorm discussion.
- The goal is to make the chat conclusions durable without requiring future agents to recover them from conversation context.

Production decode shape context:

- Target remains Gemma 4 dense inference, with the 31B dense path as the main optimization target.
- Steady-state decode is `M=1`, or one token row at a time.
- Hidden width is `5376`; production RMSNorm uses `rows=1,width=5376,eps=1e-6`.
- The dense text path has `60` layers: `50` sliding layers and `10` global layers.
- Decode projection shapes used for weighted production comparisons:

| Op | K | N | Layers/token |
| --- | ---: | ---: | ---: |
| `ffn_gate_up` | 5376 | 43008 | 60 |
| `ffn_down` | 21504 | 5376 | 60 |
| `sliding_qkv` | 5376 | 16384 | 50 |
| `sliding_o` | 8192 | 5376 | 50 |
| `global_q` | 5376 | 16384 | 10 |
| `global_k` | 5376 | 2048 | 10 |
| `global_o` | 16384 | 5376 | 10 |
| `final_logits` | 5376 | 262144 | 1 |

Benchmarking and fairness conclusions:

- For raw GPU speed, prefer CUDA graph replay timings such as `*_graph_kernel_ms`, or Nsight Compute kernel duration, over host-side stream/API timings.
- Stream-loop timing can include frontend/API costs, CPU launch overhead, or stream starvation effects. It is still useful for end-to-end host-visible behavior, but it is not the cleanest raw-kernel comparator.
- The decode GEMV benchmark currently uses CUDA events around stream loops. It excludes allocations, setup, descriptor construction, random fill, and correctness checks from the timed region, but it is not CUDA graph replay.
- The cuDNN decode GEMV comparator intentionally skips handle, descriptor, plan, and workspace setup in the timed region. The timed section measures the GPU work that cuDNN enqueues between CUDA events.
- The available cuDNN decode comparator is BF16 1x1 convolution. The cuDNN backend matmul path was not a clean usable comparator for these shapes.
- cuDNN frontend overhead made RMSNorm stream timing misleading. Graph replay timing is the fair raw-GPU comparison for RMSNorm.
- If we need stricter API-free decode GEMV comparison, add graph replay timing to the decode GEMV benchmark as was done for RMSNorm.

Decode GEMV implementation conclusions:

- The custom decode GEMV kernels do not use tensor cores.
- For strict `M=1` GEMV, the work is dominated by global memory traffic, vector loads, and reductions. It does not naturally map to tensor-core matrix tiles without batching or grouping multiple rows/tokens.
- Prefill dense GEMMs should use cuBLAS or cuBLASLt. Those library GEMMs can use tensor cores automatically for BF16 tensor-op math when the library selects a tensor-core algorithm.
- cuBLAS and cuDNN are host APIs. They cannot be called from inside a device-side fused kernel.
- `Packed128` is already implemented and retained. Decode GEMV uses 128-bit BF16 packed loads instead of the older 32-bit half2-style BF16 pair load pattern.
- The useful retained cache hint is streaming global load for weights, because weights are streamed through the decode GEMV.
- The input vector load remains normally cached, because the same input row is reused across output columns.
- Broad L2 persisting-cache windows or persisting hints for the B/weight matrix were tried and removed because they hurt performance.
- There is no broad persisting L2 policy currently retained for decode GEMV.
- Output writes are vectorized on the current eight-column path as one 128-bit store for eight BF16 outputs.

Warp tiling and reduction conclusions:

- Current production decode GEMV is still CTA-cooperative over reductions and output columns.
- Local thread-count and column-count tuning produced the retained production settings, but most later variants after the retained win did not improve the weighted production mix.
- Warp-owned output-column mappings remain plausible benchmark candidates, not proven wins.
- The warp-tiling TODO should benchmark:
  - `1` warp -> `1` output column.
  - `1` warp -> `2` output columns.
  - `2` warps -> `1` output column for larger `K`.
  - The current CTA-reduction baseline.
- The point of those variants is to attack reduction overhead in `global_k`, `sliding_o`, `qkv`, and other decode GEMV shapes without assuming the mapping is better before measurement.

Shared-memory and banking conclusions:

- The current decode GEMV path does not stage bulk weight or activation tiles through shared memory.
- Decode GEMV uses only small shared-memory reductions for warp or block partial sums, so shared-memory bank conflicts and shared-memory data-mover pressure are not a primary current bottleneck.
- RMSNorm has shared caching in the multi-row path and a decode-specialized cache for the single-row path, but no evidence currently points to shared-memory banking as the limiting factor.
- Because decode GEMV mostly streams from global memory and reduces, moving more data into shared memory is not automatically beneficial. It should be justified by reuse and measured with Nsight Compute.

Async load and double-buffering conclusions:

- The current decode GEMV and RMSNorm kernels do not use `cp.async` or explicit double buffering.
- `cp.async` is most useful when staging global memory tiles into shared memory and reusing those tiles enough to hide latency.
- For the current strict `M=1` GEMV, weights are streamed once and not staged through shared memory, so async copy is not obviously faster.
- Async loading may become useful in a future tiled kernel, batched decode kernel, or attention-style kernel with real shared-memory tile reuse.
- More promising near-term fusion work is producer/consumer fusion: merge adjacent memory-bound operations so an intermediate does not have to be written to global memory and then read back by the next kernel.

Tensor-core conclusions:

- Tensor cores are not used by the custom decode GEMV kernels today.
- A tensor-core decode path would require reformulating the work into matrix tiles, most likely by batching multiple tokens, grouping independent rows, or accepting a different latency/throughput tradeoff.
- For single-token low-latency decode, tensor-core use is not automatically a win because the matrix has `M=1` and reductions/global memory movement dominate.
- Prefill is different: prefill has real matrix sizes, so cuBLAS/cuBLASLt should be the tensor-core path there.

Current decode GEMV source routing after the 2026-05-19 8-column pass:

- Generic decode GEMV route after the later eight-column store pass: `threads=512`, `cols_per_block=8`.
- `ffn_down`: fixed8 specialization with `1024` threads and `1` eight-column group per block.
- `global_o`: fixed8 specialization with `512` threads and `1` eight-column group per block.
- `final_logits`: fixed8 specialization with `1024` threads and `1` eight-column group per block.
- Earlier long confirmation weighted production decode projection result before the 8-column output-store pass:

| Path | Weighted decode projection ms |
| --- | ---: |
| Tuned custom decode GEMV | 86.605295 |
| cuDNN BF16 1x1 conv | 97.310829 |

Interpretation:

- The retained tuned custom decode GEMV mix is `1.123607x` faster than the cuDNN BF16 1x1 convolution comparator, a `12.36%` weighted win.
- The final retained GEMV win was small beyond the earlier Packed128/custom baseline, and the later 15-test no-win tail suggests further local tuning has diminishing returns without a larger structural change.

Retained decode RMSNorm tuning:

- The old multi-row shared RMSNorm path was the wrong shape for decode. At `rows=1`, most of the CTA was idle after a CTA-wide preload.
- The decode-specialized RMSNorm path now cooperates across the single row with a block-wide reduction.
- Retained settings:
  - Standalone decode RMSNorm: `768` threads.
  - Fused decode residual+RMSNorm: `1024` threads.
- Final graph replay result:

| Rows | Width | RMS graph ms | cuDNN graph ms | Custom vs cuDNN | Fused graph ms | Split graph ms | Fused vs split |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 5376 | 0.001986 | 0.002606 | 1.312185 | 0.002236 | 0.003095 | 1.384 |

Interpretation:

- The tuned standalone decode RMSNorm is `23.79%` lower latency than cuDNN graph replay for this production shape.
- The fused decode residual+RMSNorm kernel is `1.384x` faster than split residual add followed by RMSNorm under graph replay timing.

Follow-up work:

- Add CUDA graph replay timing to the decode GEMV benchmark if we need the same API-free raw-GPU comparison style used for RMSNorm.
- Run Nsight Compute on the retained decode GEMV and RMSNorm kernels to verify memory throughput, cache behavior, occupancy, stalls, instruction mix, shared-memory transactions, and achieved occupancy.
- Keep the warp-tiled GEMV TODO and benchmark it directly against the retained CTA-reduction baseline before changing production routing.
- If tensor cores are revisited, test batched decode or grouped multi-token decode, not strict `M=1` decode first.
- Consider producer/consumer fusion only after the unfused path is correct and measured. Likely candidates remain residual+RMSNorm, activation handling, softcap/sampling, and attention-output direct layout.
- Avoid broad cache-hint changes without profiling. The retained useful cache behavior is normal cached input reuse plus streaming loads for weights.

## 2026-05-19 - Decode GEMV 8-column 128-bit output stores

Change:

- Updated the generic decode GEMV route from `cols_per_block=2` to `cols_per_block=8`.
- Changed the eight-column block to write its eight BF16 outputs with one 128-bit store.
- Changed the fixed decode output paths for `ffn_down`, `global_o`, and `final_logits` from the old four-column store path to the same eight-column kernel route.
- Removed the now-unused fixed4 helper/kernel code so the matmul source no longer has a 64-bit BF16 output-store path.

Build and SASS checks:

```bash
make cuda-kernels decode-bench
cuobjdump --dump-sass build/gemma4_matmul_kernels.o | rg "LDG\.E|STG\.E"
```

SASS evidence:

- Input and weight traffic still uses 128-bit loads, including `LDG.E.128.CONSTANT` and streaming `LDG.E.EF.128`.
- Output stores for the compiled decode GEMV kernels show `STG.E.128`.
- The previous filtered `STG.E.64` line is no longer present after moving the fixed output paths to eight columns.

Correctness and timing smoke test:

```bash
GEMMA4_DECODE_BENCH_SEED=0x1234 ./build/experiments/gemma4_decode_bench all 3 1 1
```

| Op | Custom ms | Custom GB/s | cuBLAS GEMV ms | cuDNN conv1x1 ms | Max abs diff vs cuBLAS GEMV |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ffn_gate_up` | 0.655093 | 705.887 | 2.593568 | 0.661675 | 0.25 |
| `ffn_down` | 0.333333 | 693.633 | 0.334560 | 0.337685 | 0 |
| `sliding_qkv` | 0.255147 | 690.429 | 1.773760 | 0.275808 | 0.25 |
| `sliding_o` | 0.132693 | 663.789 | 2.324064 | 0.203061 | 0.25 |
| `global_q` | 0.256949 | 685.586 | 2.417675 | 0.279477 | 0.25 |
| `global_k` | 0.037152 | 592.703 | 1.288171 | 0.959061 | 0.25 |
| `global_o` | 0.256256 | 687.441 | 0.255104 | 0.259637 | 0 |
| `final_logits` | 3.963936 | 711.054 | 3.970923 | 3.970005 | 0.125 |

Interpretation:

- The requested 8-column grouping and 128-bit output-store route compiles and passes the decode benchmark correctness checks against cuBLAS/cuDNN comparators.
- The main generic route remains in the same performance band as the previous 8-column smoke tests, with `ffn_gate_up` roughly tied with cuDNN 1x1 and much faster than the cuBLAS GEMV comparator in this run.
- The fixed output routes are now also using the 8-column/128-bit-store path. `ffn_down`, `global_o`, and `final_logits` are effectively tied with cuBLAS and cuDNN on the large memory-streaming shapes in this short run.

## 2026-05-19 - Decode GEMV accumulator and shuffle-reduction sweep

Runtime file: `src/gemma4_matmul_kernels.cu`

Reason:

- The decode GEMV inner loop has two reduction-like pieces:
  - per-thread dot accumulation into one `sums[col]` value per output column;
  - cross-thread warp/block reduction of those per-thread partials.
- The concern was that the per-thread accumulation may be too sequential and that the warp reduction mode might matter.
- `$cuda-programming-guide` was queried for shuffle/reduction behavior. Relevant local guide pages:
  - p. 550: `__shfl_sync`, `__shfl_down_sync`, and `__shfl_xor_sync` exchange values between non-exited warp threads without shared memory.
  - p. 551: `__shfl_xor_sync` uses bitwise-XOR lane addressing and is used in tree reduction and broadcast patterns.
  - p. 552: shuffle intrinsics do not imply a memory barrier or memory ordering.
  - p. 555: the guide shows a full-warp butterfly reduction with `__shfl_xor_sync`.
- `$exa-search` returned relevant external references including a NVFP4 GEMV writeup, NVIDIA CUTLASS efficient GEMM documentation, CUTLASS reduction-fusion example code, and GEMM optimization writeups. They did not change the measured decision below.

Helper cleanup retained:

- The BF16 pair helper chain was simplified from four helpers into two helpers:
  - `gemma4_accumulate_bf16_pairs<Pair = 0>(...)`
  - `gemma4_accumulate_bf16_pack(...)`
- This keeps pair indices compile-time constants and keeps the call site unchanged.
- Added/kept compile-time contracts for packed BF16 pair width and alignment.
- Verified with:

```bash
make cuda-kernels
make decode-bench
GEMMA4_DECODE_BENCH_SEED=0x1234 ./build/experiments/gemma4_decode_bench all 3 1 1
git diff --check
```

Quantization/order check:

- Active runtime code is still BF16 storage with FP32 arithmetic, not SFP8/Q4.
- Decode GEMV order remains:
  `BF16 input/weight -> FP32 FMA accumulation -> BF16 output store`.
- Prefill cuBLAS order remains:
  `CUDA_R_16BF` inputs/outputs with `CUBLAS_COMPUTE_32F`.
- RMSNorm order remains:
  `BF16 input -> FP32 sumsq/scale/multiply -> BF16 output`.
- Fused residual+RMSNorm intentionally rounds the residual to BF16 before normalizing it, matching standalone `residual_add_bf16` followed by `rmsnorm_bf16`.

Baseline timing before accumulator variants:

```bash
make decode-bench
GEMMA4_DECODE_BENCH_SEED=0x1234 ./build/experiments/gemma4_decode_bench <op> 80 20 3
```

| Op | Baseline custom best ms |
| --- | ---: |
| `ffn_gate_up` | 0.650572 |
| `ffn_down` | 0.327785 |
| `sliding_qkv` | 0.250094 |
| `global_k` | 0.034028 |

Accumulator variant A: four independent accumulators kept live across the K loop:

- Implemented four per-column partial accumulators across `pack_idx`, then horizontally reduced them before the warp reduction.
- This created too much register pressure.
- `ptxas -v` showed most instantiated decode kernels using `64` registers plus `88 bytes` of spill stores and `88 bytes` of spill loads.

| Op | Baseline ms | 4-way cross-K accum ms | Decision |
| --- | ---: | ---: | --- |
| `ffn_gate_up` | 0.650572 | 0.894991 | Reject; large regression. |
| `ffn_down` | 0.327785 | 0.442062 | Reject; large regression. |
| `sliding_qkv` | 0.250094 | 0.346468 | Reject; large regression. |
| `global_k` | 0.034028 | 0.046345 | Reject; large regression. |

Accumulator variant B: two independent accumulators kept live across the K loop:

- Reduced the cross-K partials from four to two.
- `ptxas -v` showed no spills, but register usage stayed around `64` registers for most kernels.
- Timing was effectively tied on the large shapes and slightly worse on `global_k`.

| Op | Baseline ms | 2-way cross-K accum ms | Decision |
| --- | ---: | ---: | --- |
| `ffn_gate_up` | 0.650572 | 0.651055 | Reject; no win. |
| `ffn_down` | 0.327785 | 0.327732 | Reject; noise-level change. |
| `sliding_qkv` | 0.250094 | 0.250204 | Reject; no win. |
| `global_k` | 0.034028 | 0.034216 | Reject; slightly slower. |

Accumulator variant C: four independent accumulators local to one 128-bit pack:

- This variant is different from variant A: it split only the eight BF16 lanes inside a single `Packed128`, then folded back into `sum` before the next `pack_idx`.
- It avoided spills and mostly reduced register use by one register, but `global_o` rose to `71` registers.
- Timing was still noise-level and inconsistent.

| Op | Baseline/reference ms | Pack-local split ms | Decision |
| --- | ---: | ---: | --- |
| `ffn_gate_up` | 0.650572 | 0.650773 | Reject; tiny loss. |
| `ffn_down` | 0.327785 | 0.327668 | Reject; tiny gain only. |
| `sliding_qkv` | 0.250094 | 0.250071 | Reject; tiny gain only. |
| `global_k` | 0.034028 | 0.034070 | Reject; tiny loss. |
| `global_o` | 0.250931 | 0.251125 | Reject; tiny loss and higher regs. |
| `final_logits` | 3.954915 | 3.953999 | Reject; tiny gain only. |

Interpretation for accumulator variants:

- The per-thread accumulation is not as obviously bad as it looks because the current kernel already has `ColsPerBlock=8`, so each thread carries eight independent column accumulators.
- For common shapes, per-thread K work is small:
  - `K=5376, Threads=512`: about one or two 128-bit packs per thread.
  - `K=21504, Threads=1024`: about two or three 128-bit packs per thread.
- Additional independent accumulators increase register pressure and did not produce a measured win.

Shuffle-reduction variants:

- Temporarily added `GEMMA4_SHUFFLE_REDUCE_MODE`:
  - `0`: `__shfl_down_sync`
  - `1`: `__shfl_xor_sync`
  - `2`: direct indexed `__shfl_sync`
- Built each mode with:

```bash
rm -f build/experiments/gemma4_decode_bench
make decode-bench NVCCFLAGS="-std=c++17 -O3 -arch=sm_86 -Xptxas=-v -DGEMMA4_SHUFFLE_REDUCE_MODE=<mode>"
GEMMA4_DECODE_BENCH_SEED=0x1234 ./build/experiments/gemma4_decode_bench all 50 10 3
```

Resource observations:

- Mode `0` (`__shfl_down_sync`): `64` registers, `0` spills for all decode GEMV instantiations.
- Mode `1` (`__shfl_xor_sync`): `64` registers, `0` spills for all decode GEMV instantiations.
- Mode `2` (direct `__shfl_sync`): mostly `64` registers and `0` spills, but `global_o` rose to `78` registers.

Timing:

| Op | `down` ms | `xor` ms | direct `shfl_sync` ms |
| --- | ---: | ---: | ---: |
| `ffn_gate_up` | 0.650925 | 0.650830 | 0.651079 |
| `ffn_down` | 0.328040 | 0.328142 | 0.327603 |
| `sliding_qkv` | 0.250172 | 0.249986 | 0.250122 |
| `sliding_o` | 0.126863 | 0.126801 | 0.127135 |
| `global_q` | 0.250284 | 0.250235 | 0.250271 |
| `global_k` | 0.034077 | 0.034140 | 0.034198 |
| `global_o` | 0.250931 | 0.250852 | 0.251219 |
| `final_logits` | 3.954915 | 3.955053 | 3.954971 |

Weighted decode projection totals:

| Reduction mode | Weighted decode projection ms |
| --- | ---: |
| `__shfl_down_sync` | 86.897 |
| `__shfl_xor_sync` | 86.885 |
| direct `__shfl_sync` | 86.895 |

Correctness:

- All shuffle modes stayed within the same comparator ranges as the baseline.
- Representative max-abs diff ranges remained:
  - `0.25` for most cuBLAS GEMV/GEMM comparisons on projection outputs with reduction-order differences.
  - `0` for `ffn_down` and `global_o` versus cuBLAS.
  - `0.125` to `0.25` versus cuDNN 1x1 conv depending on op.

Conclusion:

- Keep the current `__shfl_down_sync` reduction.
- `__shfl_xor_sync` is effectively tied, but it makes every lane carry the final reduction value while the current kernel only needs lane 0 to write the warp partial.
- Direct indexed `__shfl_sync` adds source-lane/conditional logic and raised register use for at least one instantiation.
- Do not keep accumulator-splitting variants unless a future mapping changes the register/occupancy balance.
- Better next candidates are structural: warp-owned output columns, different columns-per-CTA by shape, CUDA graph replay timing for decode GEMV, or producer/consumer fusion after the unfused path is correct.

## 2026-05-19 - Decode GEMV pack-local ILP retest

Runtime file tested: `src/gemma4_matmul_kernels.cu`

Change tested:

- Replaced recursive pair accumulation inside one `Packed128<__nv_bfloat16>` with four independent pair-local partial sums:
  - `s0 = dot(pair 0)`
  - `s1 = dot(pair 1)`
  - `s2 = dot(pair 2)`
  - `s3 = dot(pair 3)`
  - `sum += (s0 + s1) + (s2 + s3)`
- This tests instruction-level parallelism inside one 128-bit BF16 pack without changing the thread/block mapping or warp/block reductions.

Commands:

```bash
make -B decode-bench NVCCFLAGS="-std=c++17 -O3 -arch=sm_86 -Xptxas=-v"
GEMMA4_DECODE_BENCH_SEED=0x1234 ./build/experiments/gemma4_decode_bench all 80 20 5
```

Resource result:

- Baseline: `64` registers, `0` spills for decode GEMV instantiations.
- ILP variant: no spills; most instantiations used `63` registers, but `global_o` rose to `70` registers.

Timing:

| Op | Baseline best ms | ILP best ms | Decision |
| --- | ---: | ---: | --- |
| `ffn_gate_up` | 0.650792 | 0.650826 | Tie/no win. |
| `ffn_down` | 0.327714 | 0.327387 | Tiny win. |
| `sliding_qkv` | 0.250011 | 0.249998 | Tie/no win. |
| `sliding_o` | 0.126681 | 0.126778 | Tiny loss. |
| `global_q` | 0.250040 | 0.250148 | Tiny loss. |
| `global_k` | 0.033776 | 0.034112 | Loss. |
| `global_o` | 0.250428 | 0.250979 | Loss with higher registers. |
| `final_logits` | 3.954785 | 3.954022 | Tiny win. |

Weighted decode projection total:

| Variant | Weighted decode projection ms |
| --- | ---: |
| Baseline recursive pair accumulation | 86.842 |
| Pack-local ILP accumulation | 86.838 |

Conclusion:

- The weighted difference was about `0.004 ms`, which is noise-level for this benchmark.
- The ILP helper also changed FP32 accumulation order and increased register use for `global_o`.
- Reverted the ILP helper. Keep the current recursive compile-time pair accumulation until a larger mapping change creates enough K work per thread to justify a different local accumulator shape.

## 2026-05-19 - Decode GEMV bound check without Nsight Systems

Question:

- Determine whether the decode GEMV kernel is memory-bound without using `nsys`.
- Working hypothesis: the kernel is memory-bound, so pack-local ILP should not help much.

Profiler attempt:

```bash
env GEMMA4_DECODE_BENCH_SEED=0x1234 \
  ncu --target-processes all --set speedOfLight \
  --kernel-name "regex:gemma4_decode_gemv_cols_kernel" \
  --launch-count 1 \
  ./build/experiments/gemma4_decode_bench ffn_gate_up 1 0 1
```

- `ncu` is installed (`2025.2.1.0`), but the benchmark process aborted under the Thunder runtime with an unsupported-library/internal-assertion failure.
- No Nsight Compute hardware-counter result was usable in this environment.

Roofline-style check from benchmark timings:

- The custom benchmark already prints effective weight-streaming throughput:

```bash
GEMMA4_DECODE_BENCH_SEED=0x1234 ./build/experiments/gemma4_decode_bench all 80 20 3
```

| Op | Best ms | Weight GB/s | FP32 TFLOP/s from `2*K*N` |
| --- | ---: | ---: | ---: |
| `ffn_gate_up` | 0.650852 | 710.487 | 0.710 |
| `ffn_down` | 0.327798 | 705.347 | 0.705 |
| `sliding_qkv` | 0.250012 | 704.608 | 0.705 |
| `sliding_o` | 0.126658 | 695.421 | 0.695 |
| `global_q` | 0.250019 | 704.589 | 0.704 |
| `global_k` | 0.033932 | 648.955 | 0.649 |
| `global_o` | 0.250604 | 702.946 | 0.703 |
| `final_logits` | 3.954875 | 712.683 | 0.713 |

Arithmetic intensity:

- Weight bytes are `2*K*N`.
- FP32 scalar dot work is also `2*K*N` FLOP if each FMA is counted as two FLOPs.
- So the optimistic arithmetic intensity using weight traffic only is about `1.0 FLOP/byte`.
- Input `x` and output `y` traffic make the true DRAM arithmetic intensity lower unless `x` is served from cache. Since `x` is tiny relative to weights and reused across many column blocks, weight traffic remains the useful lower-bound stream.

Interpretation:

- The kernel reaches roughly `649-713 GB/s` on weight traffic alone.
- It only reaches roughly `0.65-0.71 FP32 TFLOP/s`, far below RTX A6000-class FP32 peak.
- With arithmetic intensity around `1 FLOP/byte`, a bandwidth roof near the high hundreds of GB/s predicts a compute rate in the same `~0.7 TFLOP/s` band observed here.
- This explains why adding local ILP inside one BF16 pack did not help: the kernel is already dominated by streaming the weight matrix, not by FP32 FMA issue throughput.

Conclusion:

- Treat the current decode GEMV as memory-bandwidth-bound.
- Local arithmetic ILP is unlikely to matter unless it also improves memory behavior or reduces overhead without increasing register pressure.
- Better next levers are memory/layout/mapping changes: improve weight-load efficiency, tune columns per CTA by shape, use cache/read-only policies carefully, or change the warp/block ownership model.

## 2026-05-19 - Decode GEMV prefetch and cache-policy sweep

Runtime file tested: `src/gemma4_matmul_kernels.cu`

Question:

- Try explicit data prefetching or a related cache-policy change for the memory-bound decode GEMV.
- Consider whether shared-memory double buffering is worth implementing.

CUDA guide notes:

- `cooperative_groups::memcpy_async` / asynchronous data movement is global-to-shared and is described as useful when copies can overlap with computation and staged data is consumed from shared memory.
- The lower-level LDGSTS async-copy path can prefetch future iterations into shared memory and may increase bytes in flight.
- For this decode GEMV, the dominant weight data is consumed once by the same thread that loads it. Staging that data through shared memory would add shared-memory traffic and synchronization without cross-thread reuse.

Commands:

```bash
make -B decode-bench NVCCFLAGS="-std=c++17 -O3 -arch=sm_86 -Xptxas=-v <variant macro>"
GEMMA4_DECODE_BENCH_SEED=0x1234 ./build/experiments/gemma4_decode_bench all 50 10 3
```

Variants tested:

- Baseline: current streaming 128-bit weight loads.
- `prefetch_next_iter`: inline `prefetch.global.L2` for the next K-stride `x` pack and all eight next K-stride weight packs.
- `prefetch_next_col`: inline `prefetch.global.L2` for the next output column's weight pack before loading/computing the current column.
- `normal_weight_load`: replace streaming 128-bit weight loads with normal direct 128-bit weight loads.

Resource result:

| Variant | Register result | Spills |
| --- | --- | --- |
| Baseline | `64` regs for decode GEMV instantiations | `0` |
| `prefetch_next_iter` | mostly `56` regs, but `global_o` rose to `74` regs | `0` |
| `prefetch_next_col` | mostly `64` regs, but `global_o` rose to `72` regs | `0` |
| `normal_weight_load` | `60` regs for decode GEMV instantiations | `0` |

Timing:

| Op | Baseline ms | Next-iter prefetch ms | Next-col prefetch ms | Normal load ms |
| --- | ---: | ---: | ---: | ---: |
| `ffn_gate_up` | 0.650769 | 0.652039 | 0.651096 | 0.652257 |
| `ffn_down` | 0.327995 | 0.334938 | 0.328007 | 0.328684 |
| `sliding_qkv` | 0.249914 | 0.250089 | 0.250109 | 0.250370 |
| `sliding_o` | 0.126680 | 0.128577 | 0.126916 | 0.127416 |
| `global_q` | 0.249910 | 0.250213 | 0.250189 | 0.250563 |
| `global_k` | 0.034218 | 0.034480 | 0.034090 | 0.034455 |
| `global_o` | 0.250586 | 0.252122 | 0.250829 | 0.251311 |
| `final_logits` | 3.954963 | 3.955522 | 3.956603 | 3.955877 |

Weighted decode projection total:

| Variant | Weighted decode projection ms |
| --- | ---: |
| Baseline | 86.858 |
| `prefetch_next_iter` | 87.476 |
| `prefetch_next_col` | 86.905 |
| `normal_weight_load` | 87.065 |

Conclusion:

- Do not keep explicit L2 prefetching in this kernel.
- Do not switch the weight loads from streaming 128-bit loads to normal direct 128-bit loads.
- The current access pattern already has coalesced 128-bit streaming weight loads and enough resident warps to hide latency; extra prefetch instructions mostly add overhead.
- Shared-memory double buffering is not attractive for this mapping because the weight packs are one-use data with no cross-thread reuse, and `x` is already loaded once per thread and reused across all eight columns.
- A useful async-copy/double-buffer design likely needs a different tile mapping where staged data is reused by multiple consumers.

## 2026-05-19 - Decode GEMV shared-memory double-buffer test

Runtime file tested: `src/gemma4_matmul_kernels.cu`

Question:

- Implement a real shared-memory double-buffer variant, not only L2 prefetching, and benchmark it.

CUDA guide notes:

- The CUDA guide describes pipelines as the mechanism for double- or multi-buffering producer-consumer patterns (p. 129).
- `memcpy_async` is global-to-shared and is intended to overlap data movement with computation when staged shared-memory data is consumed later (p. 244).
- The guide's pipeline examples schedule multiple async-copy stages, wait for the current batch, compute it, and schedule the next batch (pp. 295, 312-313).

Implementation shape tested:

- The impossible full-buffer design would stage two copies of all eight columns for every CTA thread:
  `2 stages * Threads * 8 columns * 16B`.
  - For `Threads=512`: about `128 KiB` before reductions.
  - For `Threads=1024`: about `256 KiB` before reductions.
  - This does not fit the sm_86 shared-memory budget for this kernel.
- The tested feasible design stages one weight column at a time:
  `2 stages * Threads * 16B`.
  - For `Threads=512`: `16,896B` total shared memory including existing warp reductions.
  - For `Threads=1024`: `33,792B` total shared memory including existing warp reductions.
- For each K-pack handled by a thread:
  - Load `x_pack` into a register.
  - Async copy column 0's 128-bit weight pack from global memory to shared stage 0.
  - Wait for stage 0.
  - For each column:
    - Async copy the next column's weight pack into the other shared stage.
    - Compute the current column from the current shared stage.
    - Wait for the next stage and swap.

Representative code shape:

```cuda
__pipeline_memcpy_async(&w_stage[0][threadIdx.x],
                        w_col_major + col0 * K + element_idx,
                        sizeof(Gemma4Bf16Pack));
__pipeline_commit();
__pipeline_wait_prior(0);

int stage = 0;
#pragma unroll
for (int col = 0; col < ColsPerBlock; ++col) {
  const int next_col = col + 1;
  const int next_stage = stage ^ 1;
  if (next_col < ColsPerBlock) {
    __pipeline_memcpy_async(&w_stage[next_stage][threadIdx.x],
                            w_col_major + (col0 + next_col) * K + element_idx,
                            sizeof(Gemma4Bf16Pack));
    __pipeline_commit();
  }

  gemma4_accumulate_bf16_pack(x_pack, w_stage[stage][threadIdx.x], sums[col]);

  if (next_col < ColsPerBlock) {
    __pipeline_wait_prior(0);
    stage = next_stage;
  }
}
```

Commands:

```bash
make -B decode-bench NVCCFLAGS="-std=c++17 -O3 -arch=sm_86 -Xptxas=-v -DGEMMA4_DECODE_DOUBLE_BUFFER_WEIGHT_COLS=1"
GEMMA4_DECODE_BENCH_SEED=0x1234 ./build/experiments/gemma4_decode_bench all 50 10 3
```

Resource result:

- Baseline: `64` registers, existing reduction shared memory only, `0` spills.
- Double buffer:
  - `60` regs for most instantiations, `55` regs for `global_o`, `0` spills.
  - Shared memory rose to `16,896B` for `512`-thread kernels and `33,792B` for `1024`-thread kernels.

Timing:

| Op | Baseline ms | Double-buffer ms | Decision |
| --- | ---: | ---: | --- |
| `ffn_gate_up` | 0.650769 | 0.654609 | Reject; slower. |
| `ffn_down` | 0.327995 | 0.329315 | Reject; slower. |
| `sliding_qkv` | 0.249914 | 0.253850 | Reject; slower. |
| `sliding_o` | 0.126680 | 0.127734 | Reject; slower. |
| `global_q` | 0.249910 | 0.254698 | Reject; slower. |
| `global_k` | 0.034218 | 0.036264 | Reject; slower. |
| `global_o` | 0.250586 | 0.251602 | Reject; slower. |
| `final_logits` | 3.954963 | 3.958124 | Reject; slower. |

Weighted decode projection total:

| Variant | Weighted decode projection ms |
| --- | ---: |
| Baseline | 86.858 |
| Shared-memory double buffer | 87.498 |

Conclusion:

- The real shared-memory double-buffer variant is correct but slower.
- It reduced register use, but that did not compensate for the extra async-copy instructions, shared-memory traffic, wait operations, and increased shared-memory footprint.
- Do not keep this implementation in the runtime path.
- Shared-memory async copy may become useful only after changing the mapping so staged data is reused by multiple threads or multiple output columns instead of being consumed once by the same thread that loaded it.

## 2026-05-19 - Decode GEMV explicit 128-bit load policy

Runtime file tested: `src/gemma4_matmul_kernels.cu`

Question:

- Keep the direct global-to-register GEMV path, but make the intended load policy explicit:
  normal 128-bit loads for reused activation packs and streaming 128-bit loads for one-use weight packs.

Implementation:

- Added inlined decode pack-load helpers:
  - activation packs use direct 128-bit loads.
  - weight packs use streaming 128-bit loads.
- Added a compile-time contract that `Gemma4Bf16Pack` maps to one aligned `int4` load.
- Added a host-side decode pointer guard for non-null, 16-byte-aligned `x`, `w_col_major`, and `y`.
- Kept the runtime path as global memory -> registers -> ALUs; no shared-memory staging.

Commands:

```bash
make cuda-kernels decode-bench
nvcc -std=c++17 -O3 -arch=sm_86 -Isrc --ptx src/gemma4_matmul_kernels.cu -o /tmp/gemma4_matmul_kernels_load_policy.ptx
nvcc -std=c++17 -O3 -arch=sm_86 -Xptxas=-v -Isrc -c src/gemma4_matmul_kernels.cu -o /tmp/gemma4_matmul_kernels_load_policy.o
nvcc -std=c++17 -O3 -arch=sm_86 -Isrc /tmp/gemma4_decode_arg_guard_test.cu src/gemma4_matmul_kernels.cu -lcublas -lcudnn -o /tmp/gemma4_decode_arg_guard_test
GEMMA4_DECODE_BENCH_SEED=0x1234 ./build/experiments/gemma4_decode_bench all 2 1 1
GEMMA4_DECODE_BENCH_SEED=0x1234 ./build/experiments/gemma4_decode_bench all 50 10 3
```

PTX/resource result:

- PTX load mix:
  - `ld.global.nc.v4.u32`: `7`
  - `ld.global.cs.v4.s32`: `56`
  - `prefetch`: `0`
  - `cp.async`: `0`
- ptxas resource use:
  - `63` registers for all decode GEMV instantiations except `global_o`.
  - `72` registers for `global_o`.
  - `0` spill stores and `0` spill loads.
  - Shared memory remains only the warp-reduction scratch: `512B` or `1024B` depending on thread count.

Correctness:

- `all 2 1 1` completed for all decode projection shapes.
- BF16 reference diffs stayed in the same expected range as prior runs; largest max absolute difference was `0.25` versus BF16 cuBLAS/cuDNN references.
- Temporary arg-guard test returned `cudaErrorInvalidValue` for null decode pointers before launching a kernel.

Timing:

| Op | Custom best ms |
| --- | ---: |
| `ffn_gate_up` | 0.651138 |
| `ffn_down` | 0.328560 |
| `sliding_qkv` | 0.250545 |
| `sliding_o` | 0.127626 |
| `global_q` | 0.250479 |
| `global_k` | 0.034163 |
| `global_o` | 0.250664 |
| `final_logits` | 3.954700 |

Weighted decode projection total:

| Variant | Weighted decode projection ms |
| --- | ---: |
| Prior direct-load baseline | 86.858 |
| Explicit load-policy helpers | 86.998 |

Conclusion:

- Keep this shape. The helper split makes the intended memory policy explicit without changing the generated load instructions.
- The tiny timing delta is noise-level against the prior direct-load baseline.
- This reinforces the current decision: use direct loads for reused activation packs, streaming loads for weight packs, and avoid shared-memory async copy until the GEMV mapping has real staged-data reuse.

## 2026-05-19 - Decode GEMV register double-buffer cleanup and bench

Runtime file tested: `src/gemma4_matmul_kernels.cu`

Question:

- Replace the existing `GEMMA4_DECODE_REGISTER_DOUBLE_BUFFER` implementation with a clearer register-prefetch variant and check whether it is correct and faster.

Research notes:

- NVIDIA's CUDA prefetching guidance says register prefetching can help when profiling shows memory stalls and bandwidth is not saturated, but it spends extra registers and must be verified with profiling/benchmarks.
- The CUDA guide occupancy sections are relevant because register count and shared-memory use limit resident warps/blocks, which are the normal latency-hiding mechanism for memory-bound kernels.
- For this decode GEMV mapping, the register double buffer only changes instruction timing: it preloads the next column's one-use weight pack before accumulating the current column's pack. It does not reduce total DRAM traffic.

Implementation:

- Removed the old `gemma4_accumulate_bf16_cols_register_buffered` helper.
- Added a separate compile-time double-buffer path:
  - `gemma4_decode_gemv_cols_dot_register_double_buffered_device(...)`
  - `gemma4_accumulate_prefetched_weight_cols(...)`
- Kept the default direct-load path separate, so the register-prefetch path is enabled only with `-DGEMMA4_DECODE_REGISTER_DOUBLE_BUFFER`.

Commands:

```bash
make -B decode-bench NVCCFLAGS="-std=c++17 -O3 -arch=sm_86 -DGEMMA4_DECODE_REGISTER_DOUBLE_BUFFER"
GEMMA4_DECODE_BENCH_SEED=0x1234 ./build/experiments/gemma4_decode_bench all 2 1 1
GEMMA4_DECODE_BENCH_SEED=0x1234 ./build/experiments/gemma4_decode_bench all 50 10 3
nvcc -std=c++17 -O3 -arch=sm_86 -DGEMMA4_DECODE_REGISTER_DOUBLE_BUFFER -Xptxas=-v -Isrc -c src/gemma4_matmul_kernels.cu -o /tmp/gemma4_matmul_register_double_buffer.o

make -B decode-bench
GEMMA4_DECODE_BENCH_SEED=0x1234 ./build/experiments/gemma4_decode_bench all 50 10 3
```

Correctness:

- The smoke run completed for all decode projection shapes.
- BF16 reference diffs stayed in the same expected range as the direct-load path; largest max absolute difference was `0.25` versus BF16 cuBLAS/cuDNN references.
- Exact-zero diffs versus cuBLAS remained for `ffn_down` and `global_o`.

Resource result for `GEMMA4_DECODE_REGISTER_DOUBLE_BUFFER`:

- `0` stack frame bytes.
- `0` spill stores and `0` spill loads for all instantiated decode GEMV kernels.
- `63` registers for all instantiations except `global_o`.
- `72` registers for `global_o`.
- Shared memory stayed at the existing warp-reduction scratch size: `512B` or `1024B` depending on thread count.

Timing, seeded `all 50 10 3`:

| Op | Direct-load best ms | Register double-buffer best ms |
| --- | ---: | ---: |
| `ffn_gate_up` | 0.651085 | 0.651015 |
| `ffn_down` | 0.328530 | 0.328137 |
| `sliding_qkv` | 0.250395 | 0.250356 |
| `sliding_o` | 0.127427 | 0.127486 |
| `global_q` | 0.250749 | 0.250852 |
| `global_k` | 0.034523 | 0.034418 |
| `global_o` | 0.251091 | 0.250967 |
| `final_logits` | 3.954532 | 3.954656 |

Weighted decode projection total:

| Variant | Weighted decode projection ms |
| --- | ---: |
| Direct-load | 86.986 |
| Register double buffer | 86.958 |

Conclusion:

- The cleaned-up register double-buffer path appears correct.
- The timing difference is noise-level: about `0.03 ms` on an `~87 ms` weighted decode projection total.
- Do not enable it by default yet. Keep it as a compile-time experiment path unless repeated profiling shows a real stall/bandwidth reason to prefer it.

## 2026-05-19 - Decode GEMV cp.async double-buffer implementation

Runtime file tested: `src/gemma4_matmul_kernels.cu`

Question:

- Replace the register-prefetch experiment path with real cp.async global-to-shared staging.

Research notes:

- The CUDA guide describes pipelines as the mechanism for staging work and coordinating double- or multi-buffer producer-consumer patterns.
- The primitives API supports `__pipeline_memcpy_async`, `__pipeline_commit`, and `__pipeline_wait_prior(N)` for global-to-shared asynchronous copies.
- The primitive requirements match this kernel's pack shape: destination is shared memory, source is global memory, and `size_and_align` must be `4`, `8`, or `16`. `Gemma4Bf16Pack` is a 16-byte aligned pack.
- The guide also warns not to read the shared-memory destination before waiting for the async copy to complete.

Implementation:

- Removed the `GEMMA4_DECODE_REGISTER_DOUBLE_BUFFER` source path.
- Added `GEMMA4_DECODE_CP_ASYNC_DOUBLE_BUFFER`.
- Each participating thread copies one 16-byte weight pack into `__shared__ Gemma4Bf16Pack weight_stages[2][Threads]`.
- The inner column loop alternates two stages:
  - issue cp.async for the next column,
  - accumulate the current column from the waited-on shared stage,
  - wait for the next stage before consuming it.
- The default direct global-to-register path remains unchanged unless the cp.async macro is defined.

Commands:

```bash
make -B decode-bench NVCCFLAGS="-std=c++17 -O3 -arch=sm_86 -DGEMMA4_DECODE_CP_ASYNC_DOUBLE_BUFFER"
GEMMA4_DECODE_BENCH_SEED=0x1234 ./build/experiments/gemma4_decode_bench all 2 1 1
GEMMA4_DECODE_BENCH_SEED=0x1234 ./build/experiments/gemma4_decode_bench all 50 10 3
nvcc -std=c++17 -O3 -arch=sm_86 -DGEMMA4_DECODE_CP_ASYNC_DOUBLE_BUFFER -Xptxas=-v -Isrc -c src/gemma4_matmul_kernels.cu -o /tmp/gemma4_matmul_cp_async.o
nvcc -std=c++17 -O3 -arch=sm_86 -DGEMMA4_DECODE_CP_ASYNC_DOUBLE_BUFFER -Isrc --ptx src/gemma4_matmul_kernels.cu -o /tmp/gemma4_matmul_cp_async.ptx

make -B decode-bench
GEMMA4_DECODE_BENCH_SEED=0x1234 ./build/experiments/gemma4_decode_bench all 50 10 3
```

PTX/resource result:

- PTX contains `cp.async.cg.shared.global`, `cp.async.commit_group`, and `cp.async.wait_group`.
- `0` stack frame bytes.
- `0` spill stores and `0` spill loads for all instantiated decode GEMV kernels.
- Registers:
  - `55` for all cp.async instantiations except `global_o`.
  - `54` for `global_o`.
- Shared memory:
  - `16896B` for 512-thread kernels.
  - `33792B` for 1024-thread kernels.

Correctness:

- The smoke run completed for all decode projection shapes.
- BF16 reference diffs stayed in the same expected range as the direct-load path; largest max absolute difference was `0.25` versus BF16 cuBLAS/cuDNN references.
- Exact-zero diffs versus cuBLAS remained for `ffn_down` and `global_o`.

Timing, seeded `all 50 10 3`:

| Op | Direct-load best ms | cp.async best ms |
| --- | ---: | ---: |
| `ffn_gate_up` | 0.651042 | 0.654468 |
| `ffn_down` | 0.328219 | 0.328632 |
| `sliding_qkv` | 0.250456 | 0.253651 |
| `sliding_o` | 0.127536 | 0.127636 |
| `global_q` | 0.250470 | 0.254575 |
| `global_k` | 0.034400 | 0.036015 |
| `global_o` | 0.250852 | 0.251440 |
| `final_logits` | 3.954719 | 3.957656 |

Weighted decode projection total:

| Variant | Weighted decode projection ms |
| --- | ---: |
| Direct-load | 86.967 |
| cp.async double buffer | 87.428 |

Conclusion:

- The cp.async path is correct and actually emits cp.async instructions, but it is slower for this GEMV mapping.
- The slowdown is expected: each weight pack is still one-use data, so shared-memory staging adds async-copy, commit/wait, and shared-load overhead without reducing DRAM traffic.
- Keep cp.async as a compile-time experiment path only. The runtime default should remain the direct 128-bit pack-load path.

## 2026-05-19 - Nsight Compute counter profiling attempt for decode GEMV

Runtime files tested:

- `src/gemma4_matmul_kernels.cu`
- Temporary CUDA-only harness: `/tmp/gemma4_decode_ncu_harness.cu`

Question:

- Collect hardware counters for the direct-load decode GEMV and the cp.async variant rather than reasoning only from source and CUDA-event timing.

Requested metrics:

```bash
gpu__time_duration.sum
l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum
lts__t_bytes.sum
dram__bytes.sum
l1tex__t_sector_hit_rate.pct
lts__t_sector_hit_rate.pct
smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct
smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct
smsp__warps_active.avg.pct_of_peak_sustained_active
```

Setup:

- Verified Nsight Compute is installed: `ncu 2025.2.1.0`.
- Verified the machine reports `NVIDIA RTX A6000`, driver `580.126.16`.
- Queried GA102 metric names with:

```bash
ncu --query-metrics --chips ga102
```

- First attempted profiling the existing `gemma4_decode_bench` with a kernel filter:

```bash
GEMMA4_DECODE_BENCH_SEED=0x1234 ncu \
  --target-processes all \
  --kernel-name regex:gemma4_decode_gemv_cols_kernel \
  --launch-count 1 \
  --metrics <requested metrics> \
  --csv \
  /tmp/gemma4_decode_bench_direct ffn_gate_up 1 0 1
```

Result:

- Thunder aborted before counters were collected:

```text
Your program encountered a fatal error likely caused by a Thunder Compute runtime issue:
  Your program uses a library that is not yet supported on Thunder.
```

Follow-up isolation:

- Built `/tmp/gemma4_decode_ncu_harness.cu`, a CUDA-only standalone harness for the same `ffn_gate_up` decode GEMV mapping:
  - no cuBLAS
  - no cuDNN
  - direct-load binary: `/tmp/gemma4_decode_ncu_direct`
  - cp.async binary: `/tmp/gemma4_decode_ncu_cp_async`
- Both harness binaries run normally outside `ncu`.

Standalone `ncu` command:

```bash
ncu \
  --target-processes all \
  --kernel-name regex:decode_kernel \
  --launch-count 1 \
  --metrics gpu__time_duration.sum,dram__bytes.sum \
  --csv \
  /tmp/gemma4_decode_ncu_direct 1
```

Result:

- Thunder still aborted under Nsight Compute injection, even without cuBLAS/cuDNN.
- Captured process state showed:

```text
abortMessage="Unimplemented CUDA export table function: Table=cupti_device_query, Index=7"
lastErrorUser="Your program uses a library that is not yet supported on Thunder."
```

Conclusion:

- Nsight Compute hardware-counter profiling is blocked on this Thunder instance by missing CUPTI support in Thunder's CUDA export table.
- This is not a benchmark-code or cuBLAS/cuDNN issue; it reproduces with a CUDA-only harness.
- No reliable hardware-counter conclusion can be drawn from this environment. To answer the bandwidth-vs-latency question properly, rerun the same `ncu` commands on a production instance or another host where CUPTI/Nsight Compute profiling is supported.

## 2026-05-19 - Decode GEMV cp.async debug with direct/streaming staging

Runtime file tested: `src/gemma4_matmul_kernels.cu`

Question:

- The cp.async double buffer should plausibly be faster, so isolate whether the slowdown comes from shared-memory staging itself or from cp.async commit/wait mechanics.

Implementation:

- Kept the same two-stage shared-memory layout used by the cp.async path:
  `Gemma4Bf16Pack weight_stages[2][Threads]`.
- Added two compile-time sibling variants:
  - `GEMMA4_DECODE_SHARED_STAGE_DIRECT_PACK`
  - `GEMMA4_DECODE_SHARED_STAGE_STREAMING_PACK`
- Both variants use the same shared-stage consume path as cp.async, but stage data with normal global loads instead of `__pipeline_memcpy_async`.
- The direct default path remains unchanged.

Commands:

```bash
make -B decode-bench NVCCFLAGS="-std=c++17 -O3 -arch=sm_86"
GEMMA4_DECODE_BENCH_SEED=0x1234 ./build/experiments/gemma4_decode_bench all 50 10 3

make -B decode-bench NVCCFLAGS="-std=c++17 -O3 -arch=sm_86 -DGEMMA4_DECODE_CP_ASYNC_DOUBLE_BUFFER"
GEMMA4_DECODE_BENCH_SEED=0x1234 ./build/experiments/gemma4_decode_bench all 50 10 3

make -B decode-bench NVCCFLAGS="-std=c++17 -O3 -arch=sm_86 -DGEMMA4_DECODE_SHARED_STAGE_DIRECT_PACK"
GEMMA4_DECODE_BENCH_SEED=0x1234 ./build/experiments/gemma4_decode_bench all 50 10 3

make -B decode-bench NVCCFLAGS="-std=c++17 -O3 -arch=sm_86 -DGEMMA4_DECODE_SHARED_STAGE_STREAMING_PACK"
GEMMA4_DECODE_BENCH_SEED=0x1234 ./build/experiments/gemma4_decode_bench all 50 10 3
```

Timing, seeded `all 50 10 3`:

| Op | Direct | cp.async | Shared direct pack | Shared streaming pack |
| --- | ---: | ---: | ---: | ---: |
| `ffn_gate_up` | 0.651158 | 0.654705 | 0.652065 | 0.651222 |
| `ffn_down` | 0.328492 | 0.329076 | 0.329271 | 0.328312 |
| `sliding_qkv` | 0.250387 | 0.253834 | 0.250493 | 0.250413 |
| `sliding_o` | 0.127457 | 0.127696 | 0.127501 | 0.127295 |
| `global_q` | 0.250526 | 0.254549 | 0.250682 | 0.250627 |
| `global_k` | 0.034491 | 0.035948 | 0.034846 | 0.034957 |
| `global_o` | 0.251147 | 0.251456 | 0.251411 | 0.251354 |
| `final_logits` | 3.954678 | 3.957649 | 3.955973 | 3.955886 |

Weighted decode projection total:

| Variant | Weighted decode projection ms |
| --- | ---: |
| Direct | 86.988 |
| cp.async | 87.481 |
| Shared direct pack | 87.105 |
| Shared streaming pack | 86.983 |

PTX/resource check:

| Variant | Registers | Shared memory | Spills | PTX evidence |
| --- | --- | --- | --- | --- |
| Direct | `63`, `72` | `512B`, `1024B` | `0/0` | `ld.global.cs=56`, `ld.global.nc=7` |
| cp.async | `54`, `55` | `16896B`, `33792B` | `0/0` | `cp.async.cg.shared.global=56`, `commit_group=56`, `wait_group=56` |
| Shared direct pack | `59`, `60` | `16896B`, `33792B` | `0/0` | `ld.global.nc=63`, no cp.async |
| Shared streaming pack | `64` | `16896B`, `33792B` | `0/0` | `ld.global.cs=56`, `ld.global.nc=7`, no cp.async |

Interpretation:

- Shared-memory staging itself is not the issue. Shared streaming pack loads are essentially tied with direct.
- The weight load policy matters: shared streaming pack loads beat shared direct pack loads, matching the direct path's streaming-weight policy.
- The cp.async-specific overhead is the likely issue in this mapping: the compiler emits one `cp.async`, one commit, and one wait group per staged weight pack group in the unrolled column loop.
- The work between issuing the next stage and waiting for it is only one 16-byte-pack BF16 dot contribution, so there is not enough independent compute to amortize commit/wait overhead.

Conclusion:

- Do not enable cp.async for this current per-thread/per-column GEMV mapping.
- If cp.async is revisited, change the tiling so each async-staged payload is reused more, or group larger batches per wait. A same-thread one-use 16-byte pack is too fine-grained for this pipeline structure.

## 2026-05-20 - RoPE CUDA baseline from Triton reference

Runtime files:

- `src/gemma4_rope.cu`
- `src/gemma4_rope.cuh`
- `tests/test_rope.cu`
- `docs/rope.md`

Reason:

- The next attention-prep step after Q/K/V projection and normalization is to apply RoPE before KV-cache write and attention.
- The user provided a Triton split-half RoPE implementation. The CUDA baseline mirrors its physical `[batch, seq, heads, head_dim]` layout while adding Gemma 4's partial global p-RoPE support.

Implementation:

- Added in-place BF16 Q/K rotation with precomputed FP32 cos/sin tables.
- The low-level entry point accepts explicit cos/sin row strides, including compact `rotary_dim / 2` rows and full `head_dim` rows.
- Added forward-layout launchers for direct `[batch, heads, seq, head_dim]` Q/K buffers, matching the Python `forward` inputs for inference.
- Sliding wrapper uses `head_dim=256`, `rotary_dim=256`.
- Global wrapper uses `head_dim=512`, `rotary_dim=128`, leaving the NoPE tail unchanged.
- Baseline mapping is one CUDA block per `(batch_seq_row, head)` pair; later fusion can combine Q/K RMSNorm, RoPE, and KV-cache write.

Verification:

```bash
make cuda-kernels test-rope
```

Result:

- `rope tests passed`

## 2026-05-20 - RoPE CUDA vs cuDNN pointwise benchmark

Runtime files:

- `src/experiments/gemma4_rope_bench.cu`
- `src/experiments/gemma4_bench_utils.cuh`
- `src/gemma4_rope.cu`

Reason:

- Benchmark the standalone RoPE CUDA kernel against a cuDNN implementation while excluding compile time, cuDNN frontend/plan overhead, and per-call host launch overhead where possible.

Implementation notes:

- cuDNN does not expose a standalone RoPE primitive in the installed headers/docs.
- A cuDNN frontend pointwise/concat graph and a pointwise-only graph were both tried first. Both failed plan creation on RTX A6000/cuDNN 9.22.
- The benchmark therefore uses deprecated-but-available `cudnnOpTensor` pointwise calls for the cuDNN path. Setup, descriptor creation, table conversion, warmup, and CUDA graph capture/instantiate are outside the measured region.
- The main comparison column is `*_graph_ms`: it captures repeated work once, instantiates outside timing, warms up, then records CUDA events around graph replay.
- cuDNN `cudnnOpTensor` required BF16 cos/sin tables, so the cuDNN diff is expected to be about one BF16 step versus the custom CUDA path's FP32 table reads.

Commands:

```bash
make rope-bench
./build/experiments/gemma4_rope_bench 100 20 3 1024
make test-rope
```

Environment:

- Device: NVIDIA RTX A6000
- Seed: `0xa5bcb03db46f9b6`
- Iterations: `100`, warmup: `20`, trials: `3`, batch: `1`, cos batch: `1`

CUDA graph replay timings:

| Case | Seq | Custom ms | cuDNN ms | Custom GiB/s | cuDNN GiB/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| sliding | 1 | 0.001553 | 0.016028 | 58.954 | 5.712 |
| sliding | 4 | 0.001557 | 0.015719 | 235.185 | 23.297 |
| sliding | 16 | 0.001793 | 0.016584 | 816.852 | 88.327 |
| sliding | 64 | 0.003277 | 0.023316 | 1788.139 | 251.308 |
| sliding | 256 | 0.015829 | 0.078103 | 1480.627 | 300.083 |
| sliding | 1024 | 0.074939 | 0.288012 | 1251.025 | 325.508 |
| global | 1 | 0.001417 | 0.014462 | 24.235 | 2.374 |
| global | 4 | 0.001425 | 0.015022 | 96.352 | 9.142 |
| global | 16 | 0.001637 | 0.015486 | 335.473 | 35.472 |
| global | 64 | 0.002676 | 0.017714 | 821.248 | 124.044 |
| global | 256 | 0.006655 | 0.032649 | 1320.726 | 269.202 |
| global | 1024 | 0.030191 | 0.116031 | 1164.472 | 302.989 |

Correctness:

- cuDNN max abs diff stayed at `0.0078125` for Q/K except one global seq-1 K case at `0.00390625`, consistent with BF16 table precision.
- `make test-rope` passed after the benchmark changes.

## 2026-05-20 - RMSNorm scale-free V launch tuning

Runtime files:

- `src/gemma4_rmsnorm.cu`
- `src/experiments/gemma4_rmsnorm_bench.cu`
- `tests/test_rmsnorm.cu`

Reason:

- Add and tune a separate scale-free BF16 RMSNorm path for V normalization without making the learned-weight RMSNorm kernels carry `has_weight` template logic.
- Compare realistic V head shapes against a cuDNN one-scale RMSNorm baseline.

Reference checks:

- NVIDIA CUDA Programming Guide query confirmed that `--use_fast_math`, `-ftz`, `-prec-div`, `-prec-sqrt`, and `-fmad` can change device math behavior and should be correctness-checked before use.
- The same guide notes that register/occupancy tuning can help or hurt, especially when it causes spills.
- Exa search found the official NVIDIA NVCC docs and CUDA Best Practices NVCC switch notes for the compiler-flag sweep.

Commands:

```bash
make -B rmsnorm-bench
GEMMA4_RMSNORM_BENCH_SEED=0x20260520 ./build/experiments/gemma4_rmsnorm_bench 80 20 3 1024 5376
GEMMA4_RMSNORM_BENCH_SEED=0x20260520 ./build/experiments/gemma4_rmsnorm_bench 80 20 3 16384 256
GEMMA4_RMSNORM_BENCH_SEED=0x20260520 ./build/experiments/gemma4_rmsnorm_bench 80 20 3 4096 512
make test-rmsnorm
make cuda-kernels
```

Environment:

- Device: NVIDIA RTX A6000
- Seed: `0x20260520`
- Main timings use CUDA graph replay columns.

Candidates:

| Candidate | Decision | Notes |
| --- | --- | --- |
| `--use_fast_math` | Rejected | Mixed timing and changes math semantics. |
| Direct BF16 scale helper | Rejected | No stable win over the shared unit-gamma helper. |
| `-ftz=true` | Rejected | Mixed timing and changes subnormal behavior. |
| `kDecodeRmsnormThreads=512` | Rejected | Helped scale-free decode row 1, but later decode thread counts were better. |
| `kDecodeRmsnormThreads=1024` | Rejected | Improved weighted row 1 to `0.001998 ms`, then was superseded by 704 threads. |
| `kDecodeRmsnormThreads=704` | Kept | Best weighted row 1, `0.001952 ms`; scale-free tied the best measured value at `0.001760 ms`. |
| `kDecodeFusedThreads=768` | Rejected | Improved fused row 1 to `0.002409 ms`, then was superseded by 672 threads. |
| `kDecodeFusedThreads=512` | Rejected | Fused row 1 regressed to `0.002491 ms`. |
| `kDecodeFusedThreads=672` | Kept | Best fused row 1, `0.002395 ms`. |
| `kDecodeFusedThreads=736` | Rejected | Close to 672, but slower at `0.002401 ms`. |
| `kRmsnormBlockSize=32` | Rejected | Helped some small rows but lost width-256 rows 64 and width-512 rows 4 versus 64. |
| `kRmsnormBlockSize=96` | Rejected | Did not beat 64 on the balanced small-row set. |
| `kRmsnormBlockSize=128` | Rejected | Good for larger prefill rows, but regressed global decode-like width-512 rows 4. |
| `kRmsnormBlockSize=512` | Rejected | Regressed most width-256 and width-512 V rows. |
| `kRmsnormBlockSize=64` | Kept | Best decode-like V points, with acceptable large-row tradeoffs. |

Final scale-free V graph replay timings:

| Width | Rows | Custom ms | cuDNN one-scale ms | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 16 | 0.001421 | 0.001892 | 1.33x |
| 256 | 64 | 0.001414 | 0.002142 | 1.51x |
| 256 | 1024 | 0.001860 | 0.002342 | 1.26x |
| 256 | 16384 | 0.026069 | 0.027506 | 1.06x |
| 512 | 4 | 0.001476 | 0.001918 | 1.30x |
| 512 | 64 | 0.001519 | 0.002438 | 1.61x |
| 512 | 1024 | 0.002185 | 0.003346 | 1.53x |
| 512 | 4096 | 0.010502 | 0.011870 | 1.13x |

Verification:

```bash
make test-rmsnorm
make cuda-kernels
git diff --check
```

Result:

- `rmsnorm tests passed`
- Final kept constants: `kDecodeRmsnormThreads=704`, `kDecodeFusedThreads=672`, `kRmsnormBlockSize=64`.

## 2026-05-20 - RMSNorm real-shape benchmark rerun

Runtime files:

- `src/gemma4_rmsnorm.cu`
- `src/experiments/gemma4_rmsnorm_bench.cu`
- `src/experiments/gemma4_bench_utils.cuh`

Reason:

- Rerun the RMSNorm benchmark suite on realistic Gemma 4 shapes after the final launch-constant tuning.
- Use only CUDA graph replay columns for cuDNN comparisons so cuDNN handle/frontend graph construction, plan build, workspace allocation, CUDA graph capture, and CUDA graph instantiation are outside the measured region.

Benchmarking rule:

- The benchmark still prints stream-loop columns, but the comparison below uses `*_graph_kernel_ms` columns only.
- The cuDNN frontend graph is built once per `(rows, width)` before timing. CUDA graph capture and instantiation happen inside `time_ms_graph` before CUDA events are recorded.

Commands:

```bash
make -B test-rmsnorm rmsnorm-bench
mkdir -p /tmp/gemma4_rmsnorm_real_20260520
GEMMA4_RMSNORM_BENCH_SEED=0x20260520 ./build/experiments/gemma4_rmsnorm_bench 100 30 5 1024 5376 > /tmp/gemma4_rmsnorm_real_20260520/hidden_w5376.csv
GEMMA4_RMSNORM_BENCH_SEED=0x20260520 ./build/experiments/gemma4_rmsnorm_bench 100 30 5 16384 256 > /tmp/gemma4_rmsnorm_real_20260520/sliding_v_w256.csv
GEMMA4_RMSNORM_BENCH_SEED=0x20260520 ./build/experiments/gemma4_rmsnorm_bench 100 30 5 4096 512 > /tmp/gemma4_rmsnorm_real_20260520/global_v_w512.csv
```

Environment:

- Device: NVIDIA RTX A6000
- Seed: `0x20260520`
- Iterations: `100`, warmup: `30`, trials: `5`
- cuDNN frontend: compiled

Real-shape mapping:

- Hidden RMSNorm: width `5376`, rows up to `1024`.
- Sliding V/head RMSNorm: width `256`, rows up to `16384` (`1024 * 16` KV heads).
- Global V/head RMSNorm: width `512`, rows up to `4096` (`1024 * 4` KV heads).

Scale-free RMSNorm graph replay:

| Shape | Rows | Custom ms | cuDNN one-scale ms | Speedup |
| --- | ---: | ---: | ---: | ---: |
| hidden w5376 | 1 | 0.001924 | 0.002529 | 1.31x |
| hidden w5376 | 4 | 0.004052 | 0.002548 | 0.63x |
| hidden w5376 | 16 | 0.004088 | 0.002608 | 0.64x |
| hidden w5376 | 64 | 0.004221 | 0.003134 | 0.74x |
| hidden w5376 | 256 | 0.007315 | 0.004972 | 0.68x |
| hidden w5376 | 1024 | 0.034361 | 0.034851 | 1.01x |
| sliding w256 | 1 | 0.001384 | 0.001965 | 1.42x |
| sliding w256 | 4 | 0.001389 | 0.001843 | 1.33x |
| sliding w256 | 16 | 0.001383 | 0.001869 | 1.35x |
| sliding w256 | 64 | 0.001403 | 0.002114 | 1.51x |
| sliding w256 | 256 | 0.001427 | 0.002149 | 1.51x |
| sliding w256 | 1024 | 0.001690 | 0.002354 | 1.39x |
| sliding w256 | 4096 | 0.002899 | 0.003645 | 1.26x |
| sliding w256 | 8192 | 0.010870 | 0.011645 | 1.07x |
| sliding w256 | 16384 | 0.026101 | 0.027444 | 1.05x |
| global w512 | 1 | 0.001490 | 0.001882 | 1.26x |
| global w512 | 4 | 0.001476 | 0.001879 | 1.27x |
| global w512 | 16 | 0.001483 | 0.001991 | 1.34x |
| global w512 | 64 | 0.001553 | 0.002437 | 1.57x |
| global w512 | 256 | 0.001551 | 0.002528 | 1.63x |
| global w512 | 1024 | 0.002083 | 0.003436 | 1.65x |
| global w512 | 4096 | 0.010461 | 0.012035 | 1.15x |

Weighted RMSNorm graph replay:

| Shape | Rows | Custom ms | cuDNN ms | Speedup |
| --- | ---: | ---: | ---: | ---: |
| hidden w5376 | 1 | 0.002172 | 0.002523 | 1.16x |
| hidden w5376 | 4 | 0.006324 | 0.002559 | 0.40x |
| hidden w5376 | 16 | 0.006284 | 0.002688 | 0.43x |
| hidden w5376 | 64 | 0.006550 | 0.003594 | 0.55x |
| hidden w5376 | 256 | 0.009953 | 0.005540 | 0.56x |
| hidden w5376 | 1024 | 0.040808 | 0.035017 | 0.86x |
| sliding w256 | 16 | 0.001680 | 0.001878 | 1.12x |
| sliding w256 | 1024 | 0.002185 | 0.002346 | 1.07x |
| sliding w256 | 16384 | 0.026403 | 0.027580 | 1.04x |
| global w512 | 4 | 0.001753 | 0.001910 | 1.09x |
| global w512 | 1024 | 0.002826 | 0.003336 | 1.18x |
| global w512 | 4096 | 0.011070 | 0.011980 | 1.08x |

Fused residual add plus weighted RMSNorm graph replay:

| Shape | Rows | Fused ms | Split CUDA ms | Split speedup | Custom residual + cuDNN RMSNorm ms | cuDNN split speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hidden w5376 | 1 | 0.002435 | 0.003458 | 1.42x | 0.004036 | 1.66x |
| hidden w5376 | 1024 | 0.075144 | 0.083685 | 1.11x | 0.085250 | 1.13x |
| sliding w256 | 16 | 0.001693 | 0.002876 | 1.70x | 0.003399 | 2.01x |
| sliding w256 | 4096 | 0.010832 | 0.012856 | 1.19x | 0.014506 | 1.34x |
| sliding w256 | 16384 | 0.050849 | 0.058269 | 1.15x | 0.066084 | 1.30x |
| global w512 | 4 | 0.001973 | 0.002967 | 1.50x | 0.003447 | 1.75x |
| global w512 | 1024 | 0.005133 | 0.004814 | 0.94x | 0.005964 | 1.16x |
| global w512 | 4096 | 0.026419 | 0.026671 | 1.01x | 0.029587 | 1.12x |

Correctness:

- `make test-rmsnorm` passed.
- Max output diffs versus cuDNN were BF16-sized or zero:
  - hidden w5376: up to `0.00390625`
  - sliding w256: up to `0.00390625`
  - global w512: up to `0.00195312`
- Max `rstd` diffs were at most `2.38419e-07`.

Notes:

- Nsight Compute was not rerun for this benchmark entry. Earlier in this environment it aborted under Thunder's unsupported CUPTI path, so these results are benchmark timings, not NCU counter profiles.

## 2026-05-20 - Hidden fused residual add RMSNorm prefill, one CTA per row

Implemented a hidden-width `5376` prefill path for fused residual add plus weighted
RMSNorm. It keeps one CUDA block per token row, does the RMS reduction inside that
block, writes the residual row, then writes the normalized row. No cross-block sync is
needed because rows are independent.

Command:

```bash
GEMMA4_RMSNORM_BENCH_SEED=0x20260520 \
  ./build/experiments/gemma4_rmsnorm_bench 100 30 5 1024 5376 \
  > /tmp/gemma4_rmsnorm_prefill_blockrow_20260520/hidden_w5376.csv
```

Fused residual add plus weighted RMSNorm graph replay, hidden width `5376`:

| Rows | Previous fused ms | One-CTA fused ms | Split CUDA ms | cuDNN split ms |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.002435 | 0.002412 | 0.003476 | 0.004045 |
| 4 | 0.008589 | 0.002443 | 0.007402 | 0.004110 |
| 16 | 0.011001 | 0.002609 | 0.007430 | 0.004306 |
| 64 | 0.012346 | 0.003585 | 0.008250 | 0.005822 |
| 256 | 0.021028 | 0.017522 | 0.021120 | 0.020122 |
| 1024 | 0.075144 | 0.066300 | 0.083404 | 0.085198 |

Correctness versus the custom residual plus cuDNN RMSNorm split path:

- output max abs diff: `0` through rows `64`, `0.000976562` at rows `256`,
  `0.00195312` at rows `1024`
- `rstd` max abs diff: at most `1.19209e-07`

Conclusion: the one-block-per-row hidden prefill kernel fixes the earlier prefill
weakness. It beats split CUDA by `1.21x` to `3.03x` and cuDNN split by `1.15x` to
`1.68x` on these graph replay timings.

## 2026-05-20 - Hidden fused prefill with residual pack kept in registers

Changed the hidden-width fused residual add plus RMSNorm prefill kernel to keep one
128-bit residual pack per thread in registers instead of writing all residual packs to
shared memory. This works cleanly because width `5376` is exactly `672` BF16 packs and
the prefill launch uses `672` threads.

Same benchmark command as above, output:
`/tmp/gemma4_rmsnorm_prefill_reg_20260520/hidden_w5376.csv`.

| Rows | Shared-residual fused ms | Register-residual fused ms | cuDNN split ms |
| ---: | ---: | ---: | ---: |
| 1 | 0.002412 | 0.002507 | 0.004096 |
| 4 | 0.002443 | 0.002252 | 0.004129 |
| 16 | 0.002609 | 0.002187 | 0.004345 |
| 64 | 0.003585 | 0.003276 | 0.005835 |
| 256 | 0.017522 | 0.017339 | 0.020076 |
| 1024 | 0.066300 | 0.065971 | 0.085200 |

Conclusion: keep the register version. It is better for prefill rows `4+`, especially
the smaller prefill sizes, and still beats cuDNN split by `1.16x` to `1.99x` for those
rows.

## 2026-05-20 - Hidden fused prefill weight prefetch check

Changed the hidden-width fused residual add plus RMSNorm prefill kernel to load the
per-pack learned weight before the row reduction, keeping that `128`-bit weight pack
live in registers until the normalized output write.

Commands:

```bash
make -B rmsnorm-bench
GEMMA4_RMSNORM_BENCH_SEED=0x20260520 \
  ./build/experiments/gemma4_rmsnorm_bench 100 30 5 1024 5376 \
  > /tmp/gemma4_rmsnorm_gamma_prefetch_20260520/hidden_w5376.csv
make test-rmsnorm
```

Environment:

- Device: NVIDIA RTX A6000
- Seed: `0x20260520`
- Iterations: `100`, warmup: `30`, trials: `5`
- cuDNN frontend: compiled

Fused residual add plus weighted RMSNorm graph replay, hidden width `5376`:

| Rows | Previous register ms | Weight-prefetch ms | Delta | Split CUDA ms | cuDNN split ms |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.002507 | 0.002459 | -1.91% | 0.003425 | 0.004092 |
| 4 | 0.002252 | 0.002307 | +2.44% | 0.007255 | 0.004069 |
| 16 | 0.002187 | 0.002330 | +6.54% | 0.008076 | 0.004305 |
| 64 | 0.003276 | 0.003605 | +10.04% | 0.008811 | 0.005825 |
| 256 | 0.017339 | 0.017248 | -0.52% | 0.021150 | 0.020125 |
| 1024 | 0.065971 | 0.065816 | -0.23% | 0.083469 | 0.085200 |

Correctness:

- `make test-rmsnorm` passed.
- Correctness versus the custom residual plus cuDNN RMSNorm split path remained unchanged:
  output max abs diff was `0` through rows `64`, `0.000976562` at rows `256`,
  and `0.00195312` at rows `1024`; `rstd` max abs diff was at most `1.19209e-07`.

Conclusion: the weight-prefetch version is not a clear win. It slightly improves rows
`1`, `256`, and `1024`, but regresses rows `4`, `16`, and `64` in this run. Keep this
only if later full-pipeline profiling shows those larger-row points matter more than the
small-prefill regression.

## 2026-05-20 - Hidden fused prefill focused tuning pass

Runtime files:

- `src/gemma4_rmsnorm.cu`
- `src/experiments/gemma4_rmsnorm_hidden_fused_bench.cu`
- `Makefile`

Reason:

- Tune only small settings/minor code choices for the hidden-width fused residual add
  plus weighted RMSNorm prefill kernel.
- Avoid the full RMSNorm benchmark's cuDNN/frontend setup and unrelated kernels during
  iteration. The focused benchmark times only
  `gemma4_residual_add_rmsnorm_bf16(..., rows, width=5376, ...)` for `rows > 1`.
  `rows=1` is intentionally excluded because production routing sends that case to the
  separate decode kernel.

Commands:

```bash
make -B rmsnorm-hidden-fused-bench \
  NVCCFLAGS="-std=c++17 -O3 -arch=sm_86 -Xptxas=-v"
GEMMA4_RMSNORM_BENCH_SEED=0x20260520 \
  ./build/experiments/gemma4_rmsnorm_hidden_fused_bench 300 20 7 1024 \
  > /tmp/gemma4_hidden_fused_tune_20260520/final_simplified.csv
make -B test-rmsnorm
```

Environment:

- Device: NVIDIA RTX A6000
- Seed: `0x20260520`
- Focused benchmark: `300` captured kernel calls, `20` graph warmup launches,
  `7` trials
- Timing column used for decisions: CUDA graph replay `fused_graph_best_ms`

Tuning sweep:

| Variant | Sum best ms rows 4-1024 | Sum avg ms rows 4-1024 | Aggregate best delta |
| --- | ---: | ---: | ---: |
| Baseline: prefetch weight, min-blocks 1, streaming input loads | 0.090220 | 0.090350 | baseline |
| Late weight load, min-blocks 1 | 0.090030 | 0.090300 | -0.21% |
| Late weight load, min-blocks 2 | 0.089957 | 0.090114 | -0.29% |
| Late weight load, min-blocks 2, cached input loads | 0.089807 | 0.090009 | -0.46% |
| Same plus alternate write variant | 0.089786 | 0.089957 | -0.48% |
| Final simplified kept code | 0.089864 | 0.090050 | -0.39% |

The alternate-store candidate moved the aggregate by only about `0.023%` versus the
previous candidate, below the requested `0.05%` stopping threshold, so tuning stopped
there. The final kept code uses the meaningful settings only:

- `__launch_bounds__(Threads, 2)` for the hidden prefill kernel
- cached input pack loads for `inp1` and `inp2`
- no early gamma/weight prefetch; load the weight at output application time
- ordinary global stores for residual and normed outputs

Final focused benchmark, graph replay:

| Rows | Final ms | Effective GiB/s |
| ---: | ---: | ---: |
| 4 | 0.002065 | 96.968 |
| 16 | 0.001980 | 404.521 |
| 64 | 0.002867 | 1117.629 |
| 256 | 0.017154 | 747.249 |
| 1024 | 0.065798 | 779.257 |

Resource check:

- Hidden prefill kernel ptxas stayed at `28` registers, `0` stack, `0` spill stores,
  `0` spill loads, and `96` bytes shared memory.

Correctness:

- `make -B test-rmsnorm` passed.

## 2026-05-20 - RMSNorm helper cleanup validation

Runtime file:

- `src/gemma4_rmsnorm.cu`

Reason:

- Remove dead scale-free fused residual add plus RMSNorm paths and tiny RMSNorm-local
  wrapper helpers.
- Keep the remaining block reduction helper explicit: callers now pass thread, lane,
  warp, and shared scratch storage instead of letting the helper read `threadIdx.x`.

Commands:

```bash
make -B test-rmsnorm
nvcc -std=c++17 -O3 -arch=sm_86 -Xptxas=-v -Isrc \
  -c src/gemma4_rmsnorm.cu -o /tmp/gemma4_rmsnorm.o
make -B rmsnorm-hidden-fused-bench
./build/experiments/gemma4_rmsnorm_hidden_fused_bench
```

Environment:

- Device: NVIDIA RTX A6000
- Focused benchmark defaults: `200` captured kernel calls, `20` graph warmup launches,
  `5` trials
- Benchmark seed printed by run: `0x6b1233712354a17b`

Focused hidden-width fused benchmark, graph replay:

| Rows | Best ms | Avg ms | Effective GiB/s |
| ---: | ---: | ---: | ---: |
| 4 | 0.002104 | 0.002119 | 95.171 |
| 16 | 0.002157 | 0.002158 | 371.451 |
| 64 | 0.002839 | 0.003056 | 1128.947 |
| 256 | 0.017169 | 0.017213 | 746.606 |
| 1024 | 0.065835 | 0.065862 | 778.819 |

Resource check:

- Hidden prefill kernel: `28` registers, `0` stack, `0` spills, `96` bytes shared memory.
- Decode kernels: `38` registers, `0` stack, `0` spills.
- Shared generic kernels: up to `44` registers, `0` stack, `0` spills.

Correctness and cleanup checks:

- `make -B test-rmsnorm` passed.
- `git diff --check -- src/gemma4_rmsnorm.cu src/gemma4_rmsnorm.cuh tests/test_rmsnorm.cu`
  passed.
- No remaining RMSNorm-local `__forceinline__` helpers or removed helper symbols.

## 2026-05-20 - RMSNorm dead fallback removal

Runtime files:

- `src/gemma4_rmsnorm.cu`
- `src/gemma4_rmsnorm.cuh`
- `tests/test_rmsnorm.cu`

Reason:

- Remove the fallback warp kernels and generic fused residual add plus RMSNorm paths
  that are not part of the Gemma 4 dense model path.
- Keep fused residual add plus learned RMSNorm hidden-width-only, because the residual
  stream is width `5376`.
- Keep standalone learned RMSNorm and scale-free RMSNorm shared kernels, because Q/K/V
  norms still need width `256` and `512` coverage.

Commands:

```bash
make -B test-rmsnorm
nvcc -std=c++17 -O3 -arch=sm_86 -Xptxas=-v -Isrc \
  -c src/gemma4_rmsnorm.cu -o /tmp/gemma4_rmsnorm.o
make -B rmsnorm-hidden-fused-bench
./build/experiments/gemma4_rmsnorm_hidden_fused_bench
make -B rmsnorm-bench
```

Environment:

- Device: NVIDIA RTX A6000
- Focused benchmark defaults: `200` captured kernel calls, `20` graph warmup launches,
  `5` trials
- Benchmark seed printed by run: `0x71383c77c5b4702c`

Focused hidden-width fused benchmark, graph replay:

| Rows | Best ms | Avg ms | Effective GiB/s |
| ---: | ---: | ---: | ---: |
| 4 | 0.002080 | 0.002083 | 96.277 |
| 16 | 0.002133 | 0.002136 | 375.631 |
| 64 | 0.002841 | 0.002947 | 1127.866 |
| 256 | 0.017191 | 0.017252 | 745.627 |
| 1024 | 0.065856 | 0.065873 | 778.564 |

Resource check:

- Kernel count in `src/gemma4_rmsnorm.cu`: `6`.
- Hidden prefill kernel: `28` registers, `0` stack, `0` spills, `96` bytes shared memory.
- Decode kernels: `38` registers, `0` stack, `0` spills.
- Remaining shared kernels: up to `44` registers, `0` stack, `0` spills.

Correctness and build checks:

- `make -B test-rmsnorm` passed.
- `make -B rmsnorm-hidden-fused-bench` passed.
- `make -B rmsnorm-bench` passed.
- `git diff --check` passed before this note was added.

## 2026-05-20 - Prefill GEMM stability rerun and dispatch manifests

Runtime files:

- `src/experiments/gemma4_prefill_tune.py`
- `build/experiments/gemma4_prefill_tune/candidate_deep_dispatch.csv`
- `build/experiments/gemma4_prefill_tune/all_shapes_dispatch.csv`
- `build/experiments/gemma4_prefill_tune/tuna_stability_globalk_slidingo.csv`
- `build/experiments/gemma4_prefill_tune/sgemm_bf16_stability_globalk_slidingo.csv`
- `build/experiments/gemma4_prefill_tune/stability_globalk_slidingo_dispatch.csv`
- `build/experiments/gemma4_prefill_tune/stability_globalk_slidingo_dispatch_threshold_1p25.csv`
- `build/experiments/gemma4_prefill_tune/tuna_cublas_algo_probe_global_k_m512.txt`
- `build/experiments/gemma4_prefill_tune/tuna_cublas_algo_probe_sliding_o_m256.txt`

Reason:

- The previous deep sweep showed suspicious or marginal custom wins for `global_k` at
  `M=512` and `sliding_o` at `M=256`.
- The tuner summary was useful interactively, but did not write the chosen
  custom-vs-cuBLAS route as a machine-readable artifact.

Implementation:

- Added `--dispatch-out` to the tuner `summarize` mode. It writes one CSV row per
  `(backend, op, M)` with route, config, chosen time, cuBLAS time, custom time, speedup,
  and layer-count weight.
- Fixed the combined summary printout to rank and display best configs per backend and
  op, instead of printing backend-ambiguous op summaries.
- Added `GEMMA4_PREFILL_CUBLAS_ALGO` support to both BF16 prefill benchmark binaries.
  Accepted values are `default`, `default_tensor`, and `algo0` through `algo15`, mapping
  to the corresponding `cublasGemmEx` algorithms. The default remains
  `CUBLAS_GEMM_DEFAULT_TENSOR_OP`.

Validation:

```bash
python3 -m py_compile src/experiments/gemma4_prefill_tune.py
rm -rf src/experiments/__pycache__

python3 src/experiments/gemma4_prefill_tune.py summarize \
  build/experiments/gemma4_prefill_tune/tuna_candidate_deep.csv \
  build/experiments/gemma4_prefill_tune/sgemm_bf16_candidate_deep.csv \
  --custom-threshold 1.0 \
  --dispatch-out build/experiments/gemma4_prefill_tune/candidate_deep_dispatch.csv

python3 src/experiments/gemma4_prefill_tune.py summarize \
  build/experiments/gemma4_prefill_tune/tuna_all_shapes.csv \
  build/experiments/gemma4_prefill_tune/sgemm_bf16_all_shapes.csv \
  --custom-threshold 1.0 \
  --dispatch-out build/experiments/gemma4_prefill_tune/all_shapes_dispatch.csv

make tuna-prefill-bench sgemm-bf16-prefill-bench
GEMMA4_PREFILL_CUBLAS_ALGO=algo7 \
  ./build/experiments/gemma4_sgemm_bf16_prefill_bench global_k 5 2 512 bf16_16x64
```

Focused stability commands:

```bash
python3 src/experiments/gemma4_prefill_tune.py \
  --backend tuna --ops global_k,sliding_o \
  --configs wmma_16x16,wmma_16x32,wmma_16x64,wmma_32x64,wmma_64x64 \
  --m 256,512 --iters 100 --warmup 25 \
  --out build/experiments/gemma4_prefill_tune/tuna_stability_globalk_slidingo.csv

python3 src/experiments/gemma4_prefill_tune.py \
  --backend sgemm-bf16 --ops global_k,sliding_o \
  --configs bf16_16x32,bf16_16x64,bf16_32x64,bf16_64x64,bf16_64x128,bf16_128x64 \
  --m 256,512 --iters 100 --warmup 25 \
  --out build/experiments/gemma4_prefill_tune/sgemm_bf16_stability_globalk_slidingo.csv

python3 src/experiments/gemma4_prefill_tune.py summarize \
  build/experiments/gemma4_prefill_tune/tuna_stability_globalk_slidingo.csv \
  build/experiments/gemma4_prefill_tune/sgemm_bf16_stability_globalk_slidingo.csv \
  --custom-threshold 1.25 \
  --dispatch-out build/experiments/gemma4_prefill_tune/stability_globalk_slidingo_dispatch_threshold_1p25.csv
```

Focused stability results:

| Backend | Op | Best config | Geomean speedup | Worst speedup |
| --- | --- | --- | ---: | ---: |
| Tuna | `global_k` | `wmma_16x16` | 0.608x | 0.182x |
| Tuna | `sliding_o` | `wmma_64x64` | 0.350x | 0.120x |
| SGEMM BF16 | `global_k` | `bf16_16x64` | 0.629x | 0.176x |
| SGEMM BF16 | `sliding_o` | `bf16_64x128` | 0.382x | 0.125x |

Conservative dispatch with threshold `1.25`:

| Backend | Dispatch weighted ms | cuBLAS weighted ms | Custom-only weighted ms | Dispatch vs cuBLAS | Custom-only vs cuBLAS |
| --- | ---: | ---: | ---: | ---: | ---: |
| Tuna | 85.9960 | 92.8600 | 198.7840 | 1.0798x | 0.4671x |
| SGEMM BF16 | 92.3180 | 100.4880 | 190.1170 | 1.0885x | 0.5286x |

Direct repeat checks:

```bash
./build/experiments/gemma4_tuna_prefill_bench global_k 200 50 512 wmma_16x64
./build/experiments/gemma4_tuna_prefill_bench global_k 200 50 512 wmma_16x64
./build/experiments/gemma4_tuna_prefill_bench global_k 200 50 256 wmma_16x64
./build/experiments/gemma4_tuna_prefill_bench sliding_o 200 50 256 wmma_64x64
./build/experiments/gemma4_tuna_prefill_bench sliding_o 200 50 512 wmma_64x64
```

Observed direct checks:

| Op | M | Config | Custom ms | cuBLAS ms | Speedup |
| --- | ---: | --- | ---: | ---: | ---: |
| `global_k` | 512 | `wmma_16x64` | 0.6590 | 3.1496 | 4.779x |
| `global_k` | 512 | `wmma_16x64` | 0.6846 | 1.7154 | 2.506x |
| `global_k` | 256 | `wmma_16x64` | 0.3422 | 0.0611 | 0.179x |
| `sliding_o` | 256 | `wmma_64x64` | 1.2435 | 1.8639 | 1.499x |
| `sliding_o` | 512 | `wmma_64x64` | 2.5940 | 0.3484 | 0.134x |

Interpretation:

- The direct repeats confirmed that cuBLAS has shape-specific timing cliffs in this
  harness: `global_k M=256` and `sliding_o M=512` are fast, while adjacent larger or
  smaller shapes can be much slower.
- These results justify keeping the custom kernels as measured candidates for narrow
  small-M or cuBLAS-cliff cases, but not as broad prefill GEMM replacements.
- `sliding_o M=256` is too marginal under the conservative `1.25` threshold and should
  stay on cuBLAS unless repeated on a production profiling setup or against a tuned
  cuBLASLt baseline.
- `global_k M=512` remains a candidate against this `cublasGemmEx` baseline, but it is
  suspicious enough that it should not be hard-coded without checking cuBLASLt or
  explicit cuBLAS algorithm selection.

Explicit cuBLAS algorithm probe:

```bash
for algo in default_tensor algo0 algo1 algo2 algo3 algo4 algo5 algo6 algo7 \
    algo8 algo9 algo10 algo11 algo12 algo13 algo14 algo15; do
  GEMMA4_PREFILL_CUBLAS_ALGO=$algo \
    ./build/experiments/gemma4_tuna_prefill_bench global_k 50 15 512 wmma_16x64 |
    awk -v algo=$algo '/global_k/{print algo ",global_k," $0} /wmma_16x64/{print algo ",global_k," $0}'
done > build/experiments/gemma4_prefill_tune/tuna_cublas_algo_probe_global_k_m512.txt

for algo in default_tensor algo0 algo1 algo2 algo3 algo4 algo5 algo6 algo7 \
    algo8 algo9 algo10 algo11 algo12 algo13 algo14 algo15; do
  GEMMA4_PREFILL_CUBLAS_ALGO=$algo \
    ./build/experiments/gemma4_tuna_prefill_bench sliding_o 50 15 256 wmma_64x64 |
    awk -v algo=$algo '/sliding_o/{print algo ",sliding_o," $0} /wmma_64x64/{print algo ",sliding_o," $0}'
done > build/experiments/gemma4_prefill_tune/tuna_cublas_algo_probe_sliding_o_m256.txt
```

Best observed explicit-algorithm checks:

| Op | M | Best cuBLAS algorithm | cuBLAS ms | Custom config | Custom ms | Speedup |
| --- | ---: | --- | ---: | --- | ---: | ---: |
| `global_k` | 512 | `algo7` | 1.3571 | `wmma_16x64` | 0.6461 | 2.100x |
| `sliding_o` | 256 | `algo14` | 1.5024 | `wmma_64x64` | 1.2408 | 1.211x |

Conclusion after the algorithm probe:

- The explicit `cublasGemmEx` tensor-op algorithms did not remove the timing cliff for
  these two cases.
- `global_k M=512` is still a plausible custom dispatch candidate against cuBLAS GEMMEx.
- `sliding_o M=256` is still a weak candidate because the best explicit cuBLAS algorithm
  narrows the margin to roughly `1.21x`; keep the conservative `1.25x` route on cuBLAS
  until cuBLASLt is tested.

## 2026-05-20 - Prefill GEMM cuBLASLt baseline

Runtime files:

- `experiments/tuna/gemma4_prefill_bench.cu`
- `experiments/sgemm.cu/gemma4_bf16_prefill_bench.cu`
- `src/experiments/gemma4_prefill_tune.py`
- `build/experiments/gemma4_prefill_tune/tuna_candidate_cublaslt.csv`
- `build/experiments/gemma4_prefill_tune/sgemm_bf16_candidate_cublaslt.csv`
- `build/experiments/gemma4_prefill_tune/candidate_cublaslt_dispatch.csv`
- `build/experiments/gemma4_prefill_tune/tuna_all_shapes_cublaslt.csv`
- `build/experiments/gemma4_prefill_tune/sgemm_bf16_all_shapes_cublaslt.csv`
- `build/experiments/gemma4_prefill_tune/all_shapes_cublaslt_dispatch.csv`

Reason:

- Earlier custom wins were measured against `cublasGemmEx` with
  `CUBLAS_GEMM_DEFAULT_TENSOR_OP`.
- Direct explicit-algorithm probing showed that GEMMEx has timing cliffs on the remaining
  candidate shapes, so cuBLASLt needed to be checked before any production dispatch rule.

Implementation:

- Added `GEMMA4_PREFILL_CUBLAS_BACKEND=gemmex|lt` to both BF16 prefill benchmark
  binaries. The default remains `gemmex`.
- The `lt` path uses row-major cuBLASLt descriptors for the actual Gemma layout:
  `Y[M,N] = X[M,K] * W[N,K]^T`, with BF16 inputs, FP32 compute, BF16 output, and a
  `32 MiB` workspace preference.
- cuBLASLt descriptor creation and heuristic selection happen outside the timed loop.
  Timed work is only warmup plus repeated `cublasLtMatmul` calls on the same benchmark
  path.
- The tuner now accepts `--cublas-backend gemmex|lt` and records `cublas_backend` and
  `cublas_algo` columns in new CSVs and dispatch manifests.

Validation:

```bash
make tuna-prefill-bench sgemm-bf16-prefill-bench

GEMMA4_PREFILL_CUBLAS_BACKEND=lt \
  ./build/experiments/gemma4_tuna_prefill_bench global_k 5 2 16 wmma_16x32

GEMMA4_PREFILL_CUBLAS_BACKEND=lt \
  ./build/experiments/gemma4_sgemm_bf16_prefill_bench global_k 5 2 16 bf16_16x32

python3 -m py_compile src/experiments/gemma4_prefill_tune.py
rm -rf src/experiments/__pycache__
```

Smoke results:

| Backend | Op | M | Library | Library ms | Custom config | Custom ms | Max abs |
| --- | --- | ---: | --- | ---: | --- | ---: | ---: |
| Tuna | `global_k` | 16 | cuBLASLt | 0.0458 | `wmma_16x32` | 0.0717 | 0.015625 |
| SGEMM BF16 | `global_k` | 16 | cuBLASLt | 0.0455 | `bf16_16x32` | 0.0717 | 0.015625 |

Candidate cuBLASLt sweeps:

```bash
python3 src/experiments/gemma4_prefill_tune.py \
  --backend tuna --ops ffn_down,global_k,sliding_o \
  --configs wmma_16x16,wmma_16x32,wmma_16x64,wmma_32x64,wmma_64x64,smem_16x64,smem_16x128,smem_32x64,smem_32x128,smem_64x64 \
  --m 16,32,64,128,256,512 --iters 20 --warmup 5 \
  --cublas-backend lt \
  --out build/experiments/gemma4_prefill_tune/tuna_candidate_cublaslt.csv

python3 src/experiments/gemma4_prefill_tune.py \
  --backend sgemm-bf16 --ops ffn_down,global_k,sliding_o \
  --configs bf16_16x32,bf16_16x64,bf16_32x64,bf16_64x64,bf16_64x128,bf16_128x64 \
  --m 16,32,64,128,256,512 --iters 20 --warmup 5 \
  --cublas-backend lt \
  --out build/experiments/gemma4_prefill_tune/sgemm_bf16_candidate_cublaslt.csv
```

Candidate cuBLASLt summary:

| Backend | Op | Best config | Geomean speedup vs cuBLASLt | Worst speedup |
| --- | --- | --- | ---: | ---: |
| Tuna | `ffn_down` | `wmma_64x64` | 0.249x | 0.110x |
| Tuna | `global_k` | `wmma_16x16` | 0.348x | 0.188x |
| Tuna | `sliding_o` | `wmma_64x64` | 0.276x | 0.122x |
| SGEMM BF16 | `ffn_down` | `bf16_64x64` | 0.254x | 0.114x |
| SGEMM BF16 | `global_k` | `bf16_16x64` | 0.329x | 0.181x |
| SGEMM BF16 | `sliding_o` | `bf16_64x64` | 0.276x | 0.120x |

All-shape cuBLASLt sweeps:

```bash
python3 src/experiments/gemma4_prefill_tune.py \
  --backend tuna \
  --ops ffn_gate_up,ffn_down,sliding_qkv,sliding_o,global_q,global_k,global_o,final_logits \
  --configs wmma_16x32,wmma_32x64,wmma_64x64 \
  --m 16,64,256 --iters 10 --warmup 3 --cublas-backend lt \
  --out build/experiments/gemma4_prefill_tune/tuna_all_shapes_cublaslt.csv

python3 src/experiments/gemma4_prefill_tune.py \
  --backend sgemm-bf16 \
  --ops ffn_gate_up,ffn_down,sliding_qkv,sliding_o,global_q,global_k,global_o,final_logits \
  --configs bf16_16x32,bf16_32x64,bf16_64x64 \
  --m 16,64,256 --iters 10 --warmup 3 --cublas-backend lt \
  --out build/experiments/gemma4_prefill_tune/sgemm_bf16_all_shapes_cublaslt.csv

python3 src/experiments/gemma4_prefill_tune.py summarize \
  build/experiments/gemma4_prefill_tune/tuna_all_shapes_cublaslt.csv \
  build/experiments/gemma4_prefill_tune/sgemm_bf16_all_shapes_cublaslt.csv \
  --custom-threshold 1.0 \
  --dispatch-out build/experiments/gemma4_prefill_tune/all_shapes_cublaslt_dispatch.csv
```

All-shape cuBLASLt dispatch:

| Backend | Dispatch weighted ms | cuBLASLt weighted ms | Custom-only weighted ms | Dispatch vs cuBLASLt | Custom-only vs cuBLASLt |
| --- | ---: | ---: | ---: | ---: | ---: |
| Tuna | 306.8089 | 306.8089 | 1238.9731 | 1.0000x | 0.2476x |
| SGEMM BF16 | 305.9398 | 305.9398 | 1236.2197 | 1.0000x | 0.2475x |

Conclusion:

- cuBLASLt removes the GEMMEx timing cliffs that made the simple custom WMMA kernels look
  useful for small-M `global_k`, `ffn_down`, and `sliding_o`.
- After checking all Gemma 4 projection shapes over `M=16,64,256`, every dispatch route
  goes to cuBLASLt. The custom-only path is roughly `4x` slower in the weighted all-shape
  sweep.
- Do not promote the current Tuna or SGEMM BF16 adaptations into the inference path for
  prefill GEMM. The useful artifact is the shape-aware benchmark/tuner plus the negative
  result: cuBLASLt is the baseline to use for the unfused prefill path.
- Further custom prefill GEMM work should only resume with a substantially different
  design, such as a real CUTLASS/Tuna-style pipelined tensor-core kernel with multiple
  MMA tiles per warp and async staging. The current one-warp-per-WMMA-tile variants have
  no remaining low-hanging parameter-tuning path against cuBLASLt.

## 2026-05-20 - SGEMM BF16 ffn_down production-shape tuning

Goal: tune only the SGEMM BF16 prefill GEMM path as hard as practical for one Gemma 4
production shape while keeping it valid across a token-count sweep. The chosen shape was
`ffn_down`, because Gemma 4 31B has 60 dense FFN down projections with
`intermediate_size=21504` and `hidden_size=5376`:

```text
Y[M,5376] = X[M,21504] * W[5376,21504]^T
```

Architecture basis:

- `gemma4_architecture.md`: Gemma 4 31B dense text config has `hidden_size=5376`,
  `intermediate_size=21504`, and `num_hidden_layers=60`.
- `src/gemma4.h`: `GEMMA4_INTERMEDIATE_SIZE=21504`,
  `GEMMA4_HIDDEN_SIZE=5376`, and `GEMMA4_PACKED_FFN_SIZE=43008`.

Files changed:

- `experiments/sgemm.cu/gemma4_bf16_prefill_bench.cu`
  - Added a broader SGEMM BF16 config space:
    `bf16_16x16`, `bf16_16x32`, `bf16_16x64`, `bf16_16x128`,
    `bf16_32x32`, `bf16_32x64`, `bf16_32x128`, `bf16_64x64`,
    `bf16_64x128`, `bf16_128x64`.
  - Added staged shared-memory variants: `bf16_smem64_32x64`,
    `bf16_smem64_64x64`.
  - Added wide-warp variants that reuse one A fragment across multiple N tiles:
    `bf16_warp_16x32`, `bf16_warp_16x64`, `bf16_warp_16x128`.
  - Added split-K variants for `ffn_down`: `bf16_splitk4_16x32`,
    `bf16_splitk4_32x64`.
- `src/experiments/gemma4_prefill_tune.py`
  - Added the new SGEMM BF16 configs to the tuner.
  - Fixed duplicated CSV field names for `cublas_backend` and `cublas_algo`.

CUDA guide check:

- Queried the local CUDA Programming Guide KB for WMMA BF16, shared-memory tiling,
  occupancy, and launch bounds. Relevant guide areas were kernel occupancy, shared-memory
  access patterns, and launch bounds. This matched the tuning directions tried here:
  tile geometry, shared-memory staging, register/occupancy pressure, and launch-bound
  constraints.

Commands:

```bash
make sgemm-bf16-prefill-bench

GEMMA4_PREFILL_CUBLAS_BACKEND=lt \
  ./build/experiments/gemma4_sgemm_bf16_prefill_bench \
  ffn_down 10 3 16,64 bf16_smem64_64x64

GEMMA4_PREFILL_CUBLAS_BACKEND=lt \
  ./build/experiments/gemma4_sgemm_bf16_prefill_bench \
  ffn_down 10 3 16,64 bf16_warp_16x64

GEMMA4_PREFILL_CUBLAS_BACKEND=lt \
  ./build/experiments/gemma4_sgemm_bf16_prefill_bench \
  ffn_down 10 3 16,64 bf16_splitk4_32x64

python3 src/experiments/gemma4_prefill_tune.py \
  --backend sgemm-bf16 --ops ffn_down \
  --configs bf16_16x32,bf16_16x64,bf16_32x64,bf16_64x64,bf16_128x64,bf16_smem64_32x64,bf16_smem64_64x64,bf16_warp_16x64,bf16_splitk4_32x64 \
  --m 8,16,32,64,128,256 --iters 10 --warmup 3 \
  --cublas-backend lt \
  --out build/experiments/gemma4_prefill_tune/sgemm_bf16_ffn_down_selected_arch.csv

python3 src/experiments/gemma4_prefill_tune.py \
  --backend sgemm-bf16 --ops ffn_down --configs bf16_32x64 \
  --m 8,16,32,64,128,256 --iters 100 --warmup 25 \
  --cublas-backend lt \
  --out build/experiments/gemma4_prefill_tune/sgemm_bf16_ffn_down_best_stability.csv

python3 src/experiments/gemma4_prefill_tune.py \
  --backend sgemm-bf16 --ops ffn_down \
  --configs bf16_16x16,bf16_16x32,bf16_16x64,bf16_16x128,bf16_32x32,bf16_32x64,bf16_32x128,bf16_64x64,bf16_64x128,bf16_128x64,bf16_smem64_32x64,bf16_smem64_64x64,bf16_warp_16x32,bf16_warp_16x64,bf16_warp_16x128,bf16_splitk4_16x32,bf16_splitk4_32x64 \
  --m 8,16,32,64,128,256 --iters 10 --warmup 3 \
  --cublas-backend lt \
  --out build/experiments/gemma4_prefill_tune/sgemm_bf16_ffn_down_extreme.csv

python3 src/experiments/gemma4_prefill_tune.py summarize \
  build/experiments/gemma4_prefill_tune/sgemm_bf16_ffn_down_extreme.csv \
  --custom-threshold 1.0 \
  --dispatch-out build/experiments/gemma4_prefill_tune/sgemm_bf16_ffn_down_extreme_dispatch.csv
```

Tried and rejected:

- Shared-memory K-stage 64 (`bf16_smem64_*`): correct, but slower than direct WMMA
  loads and slower than cuBLASLt.
- Wide-warp N reuse (`bf16_warp_*`): correct, but the extra long-lived accumulator
  fragments reduced useful parallelism and did not beat the simpler tilings.
- Split-K=4 (`bf16_splitk4_*`): correct, but the partial write plus reduction overhead
  dominated for this shape.
- `__launch_bounds__(..., 2)` on the plain WMMA kernels: produced ptxas warnings for
  1024-thread block variants and did not improve the sweep; reverted to
  `__launch_bounds__(..., 1)`.
- Nsight Compute command:

```bash
ncu --metric gpu__time_duration.sum --target-processes all \
  ./build/experiments/gemma4_sgemm_bf16_prefill_bench ffn_down 1 0 32 bf16_32x64
```

This failed in the Thunder runtime with an unsupported-library assertion, so the recorded
numbers below use CUDA event timing after warmup.

Selected-architecture sweep, `M=8,16,32,64,128,256`, `iters=10`, `warmup=3`:

| M | Best custom config | Custom ms | cuBLASLt ms | Speedup vs cuBLASLt | Max abs |
| ---: | --- | ---: | ---: | ---: | ---: |
| 8 | `bf16_16x64` | 0.3856 | 0.3388 | 0.879x | 0.03125 |
| 16 | `bf16_16x32` | 0.3868 | 0.3420 | 0.884x | 0.03125 |
| 32 | `bf16_16x64` | 0.4135 | 0.3421 | 0.827x | 0.03125 |
| 64 | `bf16_16x64` | 0.7776 | 0.3549 | 0.456x | 0.03125 |
| 128 | `bf16_128x64` | 1.5976 | 0.3768 | 0.236x | 0.0625 |
| 256 | `bf16_128x64` | 3.1981 | 0.4160 | 0.130x | 0.03125 |

Full extreme sweep, `M=8,16,32,64,128,256`, `iters=10`, `warmup=3`:

| M | Best custom config | Custom ms | cuBLASLt ms | Speedup vs cuBLASLt | Custom TFLOP/s | Max abs |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 8 | `bf16_16x32` | 0.3852 | 0.3383 | 0.878x | 4.80 | 0.03125 |
| 16 | `bf16_16x32` | 0.3853 | 0.3407 | 0.884x | 9.60 | 0.03125 |
| 32 | `bf16_32x32` | 0.4119 | 0.3407 | 0.827x | 17.96 | 0.03125 |
| 64 | `bf16_16x64` | 0.7778 | 0.3542 | 0.455x | 19.03 | 0.03125 |
| 128 | `bf16_128x64` | 1.4910 | 0.3753 | 0.252x | 19.85 | 0.0625 |
| 256 | `bf16_128x64` | 2.9172 | 0.4127 | 0.141x | 20.29 | 0.03125 |

Best full-sweep config by geomean:

| Config | Geomean speedup vs cuBLASLt | Worst speedup |
| --- | ---: | ---: |
| `bf16_32x64` | 0.344x | 0.070x |
| `bf16_32x32` | 0.326x | 0.046x |
| `bf16_64x64` | 0.318x | 0.126x |
| `bf16_32x128` | 0.308x | 0.104x |
| `bf16_16x128` | 0.296x | 0.035x |
| `bf16_16x64` | 0.286x | 0.029x |
| `bf16_splitk4_32x64` | 0.255x | 0.069x |
| `bf16_128x64` | 0.208x | 0.141x |

Stability pass for `bf16_32x64`, `M=8,16,32,64,128,256`, `iters=100`,
`warmup=25`:

| M | Custom ms | cuBLASLt ms | Speedup vs cuBLASLt | Custom TFLOP/s | Max abs |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 0.4123 | 0.3376 | 0.819x | 4.49 | 0.03125 |
| 16 | 0.4087 | 0.3400 | 0.832x | 9.05 | 0.03125 |
| 32 | 0.4028 | 0.3398 | 0.844x | 18.37 | 0.03125 |
| 64 | 1.3972 | 0.3536 | 0.253x | 10.59 | 0.03125 |
| 128 | 2.1470 | 0.3769 | 0.176x | 13.78 | 0.0625 |
| 256 | 5.7529 | 0.4681 | 0.081x | 10.29 | 0.03125 |

Dispatch result from the full extreme sweep:

- All `M` values route to cuBLASLt when the custom threshold is `1.0`.
- Weighted dispatch ms: `129.7140`.
- Weighted cuBLASLt ms: `129.7140`.
- Weighted custom-only ms: `382.1040`.
- Custom-only vs cuBLASLt: `0.3395x`.

Conclusion:

- The SGEMM BF16 kernels were tuned heavily for the production `ffn_down` shape and
  validated over a variety of token counts (`M=8,16,32,64,128,256`).
- Correctness against cuBLASLt stayed within BF16-level tolerances for the tested random
  inputs (`max_abs` generally `0.03125`, worst observed `0.0625`).
- The best custom kernels are only close for `M<=32`, where they still lose by roughly
  12-17%. For larger M, cuBLASLt is much faster.
- No variant in this one-warp-per-WMMA/direct-or-lightly-staged kernel family should be
  promoted into the production prefill path. Keep cuBLASLt as the production baseline
  for this shape. Further custom GEMM work should start from a substantially different
  architecture, such as a CUTLASS-style pipelined CTA with deeper K staging and a better
  accumulator/store strategy.

## 2026-05-20 - SGEMM BF16 dynamic ffn_down CUTLASS route

Follow-up goal: make the custom SGEMM BF16 path beat cuBLAS for the Gemma 4 `ffn_down`
shape with dynamic `M`.

Shape:

```text
Y[M,5376] = X[M,21504] * W[5376,21504]^T
```

Changes:

- Added CUTLASS-backed BF16 SGEMM configs to
  `experiments/sgemm.cu/gemma4_bf16_prefill_bench.cu`.
- Added `bf16_auto_ffn_down`, a dynamic route for this shape:
  - `M <= 128`: CUTLASS `64x128x64`, warp `32x64x64`, 3 stages.
  - `M > 128`: CUTLASS `128x128x64`, warp `64x64x64`, 3 stages.
- Kept the previous hand-WMMA variants in the benchmark for comparison.
- Added correctness fail-fast at `max_abs > 0.125` so invalid fast configs cannot be
  accidentally reported as wins.
- Added `CUTLASS_INCLUDE ?= /tmp/cutlass/include` to the benchmark build rule.

Important rejected configs:

- `bf16_smemA*` A-only staging variants: correct, but slower than direct WMMA and
  slower than CUTLASS.
- Very wide direct WMMA configs such as `bf16_16x256`, `bf16_32x256`, and
  `bf16_256x32`: correct, but slower.
- CUTLASS `256x*` and `128x128x64_s4`: produced invalid results (`max_abs` around
  `7`), so they were removed and the benchmark now fails correctness instead of timing
  such configs.

Commands:

```bash
python3 src/experiments/gemma4_prefill_tune.py \
  --backend sgemm-bf16 --ops ffn_down \
  --configs bf16_auto_ffn_down \
  --m 8,16,32,64,96,128,256 --iters 100 --warmup 20 \
  --out build/experiments/gemma4_prefill_tune/sgemm_bf16_ffn_down_auto_gemmex_win.csv

python3 src/experiments/gemma4_prefill_tune.py \
  --backend sgemm-bf16 --ops ffn_down \
  --configs bf16_auto_ffn_down,bf16_cutlass_64x128x64,bf16_cutlass_128x128x64,bf16_cutlass_64x64 \
  --m 8,16,32,64,96,128,256 --iters 50 --warmup 10 \
  --cublas-backend lt \
  --out build/experiments/gemma4_prefill_tune/sgemm_bf16_ffn_down_cutlass_cublaslt.csv

python3 src/experiments/gemma4_prefill_tune.py summarize \
  build/experiments/gemma4_prefill_tune/sgemm_bf16_ffn_down_auto_gemmex_win.csv \
  build/experiments/gemma4_prefill_tune/sgemm_bf16_ffn_down_cutlass_cublaslt.csv \
  --custom-threshold 1.05 \
  --dispatch-out build/experiments/gemma4_prefill_tune/sgemm_bf16_ffn_down_goal_dispatch.csv
```

Dynamic route versus default cuBLAS `cublasGemmEx`:

| M | cuBLAS ms | Custom ms | Speedup |
| ---: | ---: | ---: | ---: |
| 8 | 3.6761 | 0.3504 | 10.491x |
| 16 | 4.0123 | 0.3506 | 11.444x |
| 32 | 2.8335 | 0.3527 | 8.034x |
| 64 | 3.2531 | 0.3541 | 9.187x |
| 96 | 2.9703 | 0.3570 | 8.320x |
| 128 | 3.8941 | 0.3634 | 10.716x |
| 256 | 3.8283 | 0.4881 | 7.843x |

Summary: `bf16_auto_ffn_down` beats default cuBLAS `cublasGemmEx` by more than 5% for
every tested dynamic M value. Geomean speedup is `9.339x`; worst speedup is `7.843x`.

Harder cuBLASLt comparison:

| M | cuBLASLt ms | Best custom ms | Best config | Speedup |
| ---: | ---: | ---: | --- | ---: |
| 8 | 0.3374 | 0.3501 | `bf16_auto_ffn_down` | 0.964x |
| 16 | 0.3398 | 0.3511 | `bf16_auto_ffn_down` | 0.968x |
| 32 | 0.3394 | 0.3527 | `bf16_cutlass_64x128x64` | 0.962x |
| 64 | 0.3530 | 0.3547 | `bf16_cutlass_64x128x64` | 0.995x |
| 96 | 0.3718 | 0.3564 | `bf16_cutlass_64x128x64` | 1.043x |
| 128 | 0.3779 | 0.3605 | `bf16_cutlass_64x128x64` | 1.048x |
| 256 | 0.4166 | 0.4264 | `bf16_auto_ffn_down` | 0.977x |

Summary: the new CUTLASS route nearly matches cuBLASLt overall (`0.993x` geomean), and
beats cuBLASLt for `M=96` and `M=128`, but it does not meet the 5% win threshold against
cuBLASLt. The achieved 5%+ result is specifically against default cuBLAS `cublasGemmEx`.

### Correction: cuBLAS graph timing with enqueue overhead factored out

The default `cublasGemmEx` result above was not a valid production comparison: the timing
loop could include host enqueue gaps between repeated cuBLAS calls. The benchmark now
supports `GEMMA4_PREFILL_GRAPH_REPEATS`; when set above `1`, cuBLAS/cuBLASLt is captured
on a nonblocking CUDA stream with repeated GEMMs inside one CUDA graph. The reported
cuBLAS time is divided by the number of captured GEMMs, so one-time setup and per-call
host enqueue overhead are factored out of the per-GEMM result.

Code changes:

- `time_cuda_graph_stream` captures repeated cuBLAS calls into one CUDA graph on an
  explicit nonblocking stream.
- cuBLASLt timing now passes the same stream into `cublasLtMatmul`.
- CUTLASS/custom timing remains the direct CUDA-event path, which is conservative for the
  custom side in this comparison.

Commands:

```bash
GEMMA4_PREFILL_GRAPH_REPEATS=16 \
  python3 src/experiments/gemma4_prefill_tune.py \
  --backend sgemm-bf16 --ops ffn_down \
  --configs bf16_auto_ffn_down,bf16_cutlass_64x128x64,bf16_cutlass_128x128x64,bf16_cutlass_64x64 \
  --m 8,16,32,64,96,128,256 --iters 50 --warmup 10 \
  --out build/experiments/gemma4_prefill_tune/sgemm_bf16_ffn_down_graph_gemmex.csv

GEMMA4_PREFILL_GRAPH_REPEATS=16 \
  python3 src/experiments/gemma4_prefill_tune.py \
  --backend sgemm-bf16 --ops ffn_down \
  --configs bf16_auto_ffn_down,bf16_cutlass_64x128x64,bf16_cutlass_128x128x64,bf16_cutlass_64x64 \
  --m 8,16,32,64,96,128,256 --iters 50 --warmup 10 \
  --cublas-backend lt \
  --out build/experiments/gemma4_prefill_tune/sgemm_bf16_ffn_down_graph_cublaslt.csv

python3 src/experiments/gemma4_prefill_tune.py summarize \
  build/experiments/gemma4_prefill_tune/sgemm_bf16_ffn_down_graph_gemmex.csv \
  build/experiments/gemma4_prefill_tune/sgemm_bf16_ffn_down_graph_cublaslt.csv \
  --custom-threshold 1.05 \
  --dispatch-out build/experiments/gemma4_prefill_tune/sgemm_bf16_ffn_down_graph_dispatch.csv
```

Graph-timed default cuBLAS `cublasGemmEx` comparison:

| M | Best config | cuBLAS ms | Custom ms | Speedup |
| ---: | --- | ---: | ---: | ---: |
| 8 | `bf16_auto_ffn_down` | 0.3349 | 0.3489 | 0.960x |
| 16 | `bf16_cutlass_64x128x64` | 0.3378 | 0.3489 | 0.968x |
| 32 | `bf16_cutlass_64x128x64` | 0.3383 | 0.3494 | 0.968x |
| 64 | `bf16_cutlass_64x128x64` | 0.3516 | 0.3533 | 0.995x |
| 96 | `bf16_cutlass_64x128x64` | 0.3716 | 0.3565 | 1.042x |
| 128 | `bf16_cutlass_64x128x64` | 0.3777 | 0.3596 | 1.050x |
| 256 | `bf16_cutlass_128x128x64` | 0.4899 | 0.5198 | 0.942x |

Summary: after factoring out cuBLAS host/setup overhead, the custom route no longer
achieves a 5% dynamic-M win. It only reaches the threshold at `M=128`; geomean speedup
versus graph-timed default cuBLAS is `0.974x`, and worst speedup is `0.873x`.

Graph-timed cuBLASLt comparison:

- Best config: `bf16_auto_ffn_down`.
- Geomean speedup: `0.988x`.
- Worst speedup: `0.946x`.
- No `M` routes to custom at a `1.05x` threshold.

Corrected conclusion: the previous default-cuBLAS win was an overhead artifact. With
cuBLAS overhead factored out, this SGEMM BF16 path still does not beat cuBLAS/cuBLASLt by
5% across dynamic M.

## 2026-05-20 - cuBLASLt multi-heuristic and Stream-K ffn_down check

Follow-up reason:

- The earlier cuBLASLt baseline used only the first heuristic returned by
  `cublasLtMatmulAlgoGetHeuristic`.
- For a fair "fastest CUDA library path" comparison, the benchmark now asks cuBLASLt for
  multiple heuristics, graph-times each returned successful algorithm, and reports the
  fastest measured candidate.
- CUTLASS Stream-K was added as a custom architecture candidate for small-`M`, large
  `K,N` GEMMs where ordinary CTA scheduling can underfeed SMs.

Code changes:

- Added `GEMMA4_PREFILL_CUBLASLT_HEURISTICS`, defaulting to `1`, capped at `64`.
- `CublasLtPrefillPlan` now stores all successful heuristic candidates returned by
  cuBLASLt.
- cuBLASLt timing loops over those candidates, using the existing graph-timed stream path
  when `GEMMA4_PREFILL_GRAPH_REPEATS > 1`, and reports the fastest candidate.
- Added Python sweep option `--cublaslt-heuristics`; CSV labels use `h<N>` for cuBLASLt
  heuristic-count baselines.
- Added Stream-K configs:
  `bf16_streamk_64x64x64`, `bf16_streamk_64x128x64`, and
  `bf16_streamk_128x128x64`.

Commands:

```bash
GEMMA4_PREFILL_GRAPH_REPEATS=16 \
  python3 src/experiments/gemma4_prefill_tune.py \
  --backend sgemm-bf16 --ops ffn_down \
  --configs bf16_auto_ffn_down,bf16_cutlass_64x128x64,bf16_cutlass_128x128x64,bf16_cutlass_64x64 \
  --m 8,16,32,64,96,128,256 --iters 30 --warmup 8 \
  --cublas-backend lt --cublaslt-heuristics 32 \
  --out build/experiments/gemma4_prefill_tune/sgemm_bf16_ffn_down_graph_cublaslt_h32.csv

GEMMA4_PREFILL_GRAPH_REPEATS=16 \
  python3 src/experiments/gemma4_prefill_tune.py \
  --backend sgemm-bf16 --ops ffn_down \
  --configs bf16_auto_ffn_down,bf16_cutlass_64x128x64,bf16_streamk_64x64x64,bf16_streamk_64x128x64,bf16_streamk_128x128x64 \
  --m 8,16,32,64,96,128,256 --iters 30 --warmup 8 \
  --cublas-backend lt --cublaslt-heuristics 32 \
  --out build/experiments/gemma4_prefill_tune/sgemm_bf16_ffn_down_streamk_cublaslt_h32.csv

GEMMA4_PREFILL_GRAPH_REPEATS=16 \
  python3 src/experiments/gemma4_prefill_tune.py \
  --backend sgemm-bf16 --ops ffn_down \
  --configs bf16_auto_ffn_down,bf16_streamk_64x64x64,bf16_streamk_64x128x64,bf16_streamk_128x128x64 \
  --m 512,1024 --iters 20 --warmup 5 \
  --cublas-backend lt --cublaslt-heuristics 32 \
  --out build/experiments/gemma4_prefill_tune/sgemm_bf16_ffn_down_streamk_large_m_cublaslt_h32.csv
```

Best custom versus graph-timed cuBLASLt h32 over `M=8..1024`:

| M | Best custom config | cuBLASLt h32 ms | Custom ms | Speedup |
| ---: | --- | ---: | ---: | ---: |
| 8 | `bf16_streamk_64x64x64` | 0.3299 | 0.3347 | 0.986x |
| 16 | `bf16_streamk_64x64x64` | 0.3304 | 0.3368 | 0.981x |
| 32 | `bf16_streamk_64x64x64` | 0.3327 | 0.3363 | 0.989x |
| 64 | `bf16_streamk_64x64x64` | 0.3415 | 0.3405 | 1.003x |
| 96 | `bf16_cutlass_64x128x64` | 0.3737 | 0.3598 | 1.039x |
| 128 | `bf16_streamk_128x128x64` | 0.3789 | 0.3650 | 1.038x |
| 256 | `bf16_auto_ffn_down` | 0.4982 | 0.5288 | 0.942x |
| 512 | `bf16_streamk_128x128x64` | 0.9061 | 0.9421 | 0.962x |
| 1024 | `bf16_streamk_128x128x64` | 1.8709 | 1.9429 | 0.963x |

Summary:

- Multi-heuristic cuBLASLt h32 is a stronger baseline than the previous single-heuristic
  cuBLASLt run. Without Stream-K, the custom geomean over `M=8..256` drops to `0.968x`.
- Stream-K helps the best custom choice for small `M`, especially `M=8..64`, but still
  does not reach the 1.05x custom-routing threshold.
- Over `M=8..1024`, no tested `M` routes to custom at a `1.05x` threshold.
- Weighted custom-only speedup over this `M` sweep is `0.977x` versus graph-timed
  cuBLASLt h32.

Conclusion: after comparing against a stronger, overhead-factored cuBLASLt baseline,
the current CUTLASS and Stream-K custom routes do not outperform cuBLASLt for Gemma 4
`ffn_down`. Further work needs a materially different custom architecture or a narrower
deployment target than "15-20% average over a large M range."

## 2026-05-20 - Custom CUDA graph timing and larger-M tile sweep

Follow-up reason:

- The previous correction graph-timed cuBLAS/cuBLASLt but left custom kernels in the
  direct launch loop. CUDA events can include GPU idle gaps between host launches when
  kernels are short.
- This pass adds stream-aware CUTLASS launches and graph-times custom CUTLASS/Stream-K
  routes too, so both sides can be compared with host-launch gaps factored out.
- It also checks whether the large-M losses were caused by `bf16_auto_ffn_down` choosing
  the wrong existing CUTLASS tile.

Code changes:

- CUTLASS and Stream-K launchers now accept an explicit `cudaStream_t`.
- `launch_bf16` forwards the stream to CUTLASS-backed configs.
- When `GEMMA4_PREFILL_GRAPH_REPEATS > 1`, custom CUTLASS-backed configs are captured
  and timed on a nonblocking CUDA stream, just like the cuBLASLt candidates.

Commands:

```bash
GEMMA4_PREFILL_GRAPH_REPEATS=16 \
  python3 src/experiments/gemma4_prefill_tune.py \
  --backend sgemm-bf16 --ops ffn_down \
  --configs bf16_auto_ffn_down,bf16_cutlass_64x128x64,bf16_streamk_64x64x64,bf16_streamk_64x128x64,bf16_streamk_128x128x64 \
  --m 8,16,32,64,96,128,256 --iters 30 --warmup 8 \
  --cublas-backend lt --cublaslt-heuristics 32 \
  --out build/experiments/gemma4_prefill_tune/sgemm_bf16_ffn_down_both_graph_streamk_cublaslt_h32.csv

GEMMA4_PREFILL_GRAPH_REPEATS=16 \
  python3 src/experiments/gemma4_prefill_tune.py \
  --backend sgemm-bf16 --ops ffn_down \
  --configs bf16_auto_ffn_down,bf16_cutlass_64x64,bf16_cutlass_64x128,bf16_cutlass_64x128x64,bf16_cutlass_128x64,bf16_cutlass_128x64x64,bf16_cutlass_128x128,bf16_cutlass_128x128x64,bf16_cutlass_128x256,bf16_streamk_64x64x64,bf16_streamk_64x128x64,bf16_streamk_128x128x64 \
  --m 256,512 --iters 20 --warmup 5 \
  --cublas-backend lt --cublaslt-heuristics 32 \
  --out build/experiments/gemma4_prefill_tune/sgemm_bf16_ffn_down_large_m_tiles_both_graph_cublaslt_h32.csv
```

Both-sides graph timing over `M=8..256`:

| M | Best custom config | cuBLASLt h32 ms | Custom ms | Speedup |
| ---: | --- | ---: | ---: | ---: |
| 8 | `bf16_streamk_64x64x64` | 0.3299 | 0.3330 | 0.991x |
| 16 | `bf16_streamk_64x64x64` | 0.3305 | 0.3337 | 0.990x |
| 32 | `bf16_streamk_64x64x64` | 0.3328 | 0.3355 | 0.992x |
| 64 | `bf16_streamk_64x64x64` | 0.3423 | 0.3389 | 1.010x |
| 96 | `bf16_cutlass_64x128x64` | 0.3747 | 0.3620 | 1.035x |
| 128 | `bf16_streamk_128x128x64` | 0.3806 | 0.3758 | 1.013x |
| 256 | `bf16_auto_ffn_down` | 0.5087 | 0.5470 | 0.930x |

Summary for `M=8..256`: best custom geomean is `0.962x`, worst is `0.930x`, and no
case reaches the `1.05x` custom routing threshold. Graph-timing the custom side does not
reveal a hidden win.

Broader existing-tile check at larger `M`:

| M | Best custom config | cuBLASLt h32 ms | Custom ms | Speedup |
| ---: | --- | ---: | ---: | ---: |
| 256 | `bf16_cutlass_128x128` | 0.4924 | 0.5482 | 0.898x |
| 512 | `bf16_cutlass_128x256` | 0.9189 | 0.9639 | 0.953x |

Conclusion: using CUDA graphs for both custom and cuBLASLt makes the comparison cleaner
but does not change the direction. The current CUTLASS tile inventory and Stream-K route
still do not beat cuBLASLt h32 for `ffn_down`, and a broader large-M tile sweep did not
find a better existing dispatch choice.

## 2026-05-20 - Stream-K split-factor ffn_down check

Follow-up reason:

- CUTLASS example 47 exposes Stream-K's `batch_split` argument as a tile-splitting factor
  for `GemmUniversalMode::kGemm`.
- The previous Stream-K configs only measured the default split factor `1`, so this pass
  checks whether Stream-K split-K emulation helps the small/medium `M` range where custom
  was closest to cuBLASLt.

Code changes:

- `launch_cutlass_streamk_gemm` now accepts a `split_k_factor` and passes it to
  `GemmUniversal::Arguments`.
- Added split-factor configs:
  `bf16_streamk_s2_64x64x64`, `bf16_streamk_s2_64x128x64`,
  `bf16_streamk_s2_128x128x64`, `bf16_streamk_s4_64x64x64`,
  `bf16_streamk_s4_64x128x64`, and `bf16_streamk_s4_128x128x64`.

Commands:

```bash
GEMMA4_PREFILL_GRAPH_REPEATS=16 \
  python3 src/experiments/gemma4_prefill_tune.py \
  --backend sgemm-bf16 --ops ffn_down \
  --configs bf16_streamk_64x64x64,bf16_streamk_s2_64x64x64,bf16_streamk_s4_64x64x64,bf16_streamk_64x128x64,bf16_streamk_s2_64x128x64,bf16_streamk_s4_64x128x64,bf16_streamk_128x128x64,bf16_streamk_s2_128x128x64,bf16_streamk_s4_128x128x64 \
  --m 64,96,128 --iters 20 --warmup 5 \
  --cublas-backend lt --cublaslt-heuristics 32 \
  --out build/experiments/gemma4_prefill_tune/sgemm_bf16_ffn_down_streamk_split_both_graph_cublaslt_h32.csv

python3 src/experiments/gemma4_prefill_tune.py summarize \
  build/experiments/gemma4_prefill_tune/sgemm_bf16_ffn_down_both_graph_streamk_cublaslt_h32.csv \
  build/experiments/gemma4_prefill_tune/sgemm_bf16_ffn_down_streamk_split_both_graph_cublaslt_h32.csv \
  --custom-threshold 1.05 \
  --dispatch-out build/experiments/gemma4_prefill_tune/sgemm_bf16_ffn_down_both_graph_with_split_cublaslt_h32_dispatch.csv
```

Targeted split-factor sweep:

| M | Best config | cuBLASLt h32 ms | Custom ms | Speedup |
| ---: | --- | ---: | ---: | ---: |
| 64 | `bf16_streamk_64x64x64` | 0.3416 | 0.3381 | 1.010x |
| 96 | `bf16_streamk_s2_128x128x64` | 0.3721 | 0.3602 | 1.033x |
| 128 | `bf16_streamk_s2_128x128x64` | 0.3760 | 0.3611 | 1.041x |

Combined dynamic route over `M=8..256` after adding split-factor configs:

- No `M` routes to custom at the `1.05x` threshold.
- Weighted custom-only speedup is `0.9935x` versus graph-timed cuBLASLt h32.
- The best split-factor config helps `M=96` and `M=128`, but not enough for the requested
  15-20% average win, and split factors are much worse at larger `M`.

Conclusion: Stream-K split-K emulation is not the missing architecture lever for this
`ffn_down` sweep. It narrows the gap in a small band but does not outperform cuBLASLt
over a large dynamic-M range.

## 2026-05-20 - CUTLASS generated SM80 BF16 shape sweep

Follow-up reason:

- The hand-instantiated CUTLASS configs only covered a small subset of SM80 BF16 tensor-op
  GEMM shapes.
- CUTLASS's generator lists many more BF16 `nt` candidates for SM80, including
  `256x128`, `256x64`, `64x256`, and higher-stage `64x64` / `64x128` tiles.
- This pass adds a focused subset of those generated shapes instead of guessing more
  hand-written tile sizes.

Research / discovery:

```bash
python3 /tmp/cutlass/python/cutlass_library/generator.py \
  --operations gemm --architectures 80 \
  --build-dir /tmp/cutlass-gemm-list \
  --curr-build-dir /tmp/cutlass-gemm-list \
  --generator-target library \
  --kernels 'cutlass_tensorop_*bf16*gemm*' \
  --selected-kernel-list /tmp/cutlass-gemm-list/kernels.txt
```

- The generator emitted 638 SM80 BF16 tensor-op GEMM candidate names.
- The relevant layout family for this benchmark is `nt` with BF16 input/output and FP32
  accumulation, matching `Y[M,N] = X[M,K] * W[N,K]^T`.

Code changes:

- Added valid generated-shape configs:
  `bf16_cutlass_64x64_s10`, `bf16_cutlass_64x64x64_s5`,
  `bf16_cutlass_64x128_s6`, `bf16_cutlass_64x256_s4`,
  `bf16_cutlass_128x64_s6`, `bf16_cutlass_128x128_s5`,
  `bf16_cutlass_256x64`, `bf16_cutlass_256x64_s4`, and
  `bf16_cutlass_256x128`.
- Added tuner `--keep-going` so generated-kernel sweeps can skip failed configs while
  preserving measurements from valid configs.
- Rejected generated configs that failed correctness with `max_abs=6.90625`:
  `bf16_cutlass_64x256x64_s4`, `bf16_cutlass_128x256x64`,
  `bf16_cutlass_256x64x64`, and `bf16_cutlass_256x128x64`.

Commands:

```bash
GEMMA4_PREFILL_GRAPH_REPEATS=16 \
  python3 src/experiments/gemma4_prefill_tune.py \
  --backend sgemm-bf16 --ops ffn_down \
  --configs bf16_cutlass_64x64_s10,bf16_cutlass_64x64x64_s5,bf16_cutlass_64x128_s6,bf16_cutlass_64x256_s4,bf16_cutlass_128x64_s6,bf16_cutlass_128x128_s5,bf16_cutlass_128x256x64,bf16_cutlass_256x64,bf16_cutlass_256x64_s4,bf16_cutlass_256x64x64,bf16_cutlass_256x128,bf16_cutlass_256x128x64 \
  --m 64,96,128,256 --iters 16 --warmup 4 \
  --cublas-backend lt --cublaslt-heuristics 16 --keep-going \
  --out build/experiments/gemma4_prefill_tune/sgemm_bf16_ffn_down_generated_cutlass_both_graph_cublaslt_h16.csv
```

Best valid generated shapes:

| M | Best config | cuBLASLt h16 ms | Custom ms | Speedup |
| ---: | --- | ---: | ---: | ---: |
| 64 | `bf16_cutlass_64x64_s10` | 0.3419 | 0.3379 | 1.012x |
| 96 | `bf16_cutlass_64x128_s6` | 0.3723 | 0.3513 | 1.060x |
| 128 | `bf16_cutlass_64x128_s6` | 0.3771 | 0.3736 | 1.009x |
| 256 | `bf16_cutlass_128x128_s5` | 0.5035 | 0.5367 | 0.938x |

Summary:

- The generated shape sweep finds a local `1.06x` win at `M=96` against h16 cuBLASLt.
- The dynamic dispatch result over `M=64,96,128,256` is only `1.013x` because the route
  still falls back to cuBLASLt for most sizes.
- Custom-only speedup remains below parity (`0.997x`) for this focused sweep.

Conclusion: CUTLASS's generated SM80 BF16 default shapes add one narrow local win but do
not produce the requested broad 15-20% average speedup over cuBLASLt. The generated list
also confirms that some legal-looking K=64 large-tile instantiations are numerically
invalid in this harness and must remain excluded.

### Explicit M >= 256 check

Follow-up correction:

- The previous generated-shape sweep covered `M=256`, but the large-token side also needs
  explicit checks beyond `256`.
- This run checks `M=256,512,1024,2048` with the current best larger-M candidate set.

Command:

```bash
GEMMA4_PREFILL_GRAPH_REPEATS=16 \
  python3 src/experiments/gemma4_prefill_tune.py \
  --backend sgemm-bf16 --ops ffn_down \
  --configs bf16_auto_ffn_down,bf16_cutlass_128x128_s5,bf16_cutlass_128x128x64,bf16_cutlass_128x256,bf16_cutlass_256x128,bf16_streamk_128x128x64 \
  --m 256,512,1024,2048 --iters 20 --warmup 5 \
  --cublas-backend lt --cublaslt-heuristics 32 --skip-build \
  --out build/experiments/gemma4_prefill_tune/sgemm_bf16_ffn_down_m256_2048_check_cublaslt_h32.csv
```

Results using the best custom config per `M` and the best graph-timed cuBLASLt h32 time:

| M | Best custom config | cuBLASLt h32 ms | Custom ms | Speedup |
| ---: | --- | ---: | ---: | ---: |
| 256 | `bf16_cutlass_128x128_s5` | 0.4909 | 0.5275 | 0.931x |
| 512 | `bf16_cutlass_256x128` | 0.9118 | 0.9423 | 0.968x |
| 1024 | `bf16_cutlass_128x256` | 1.8790 | 1.8695 | 1.005x |
| 2048 | `bf16_cutlass_128x256` | 3.6909 | 3.7630 | 0.981x |

Summary:

- Best custom geomean over `M=256,512,1024,2048`: `0.930x`.
- Weighted custom-only speedup over this explicit large-M sweep: `0.9817x`.
- No `M >= 256` reaches the `1.05x` custom routing threshold.
- `M=1024` is near parity, but not close to the requested 15-20% win.

Conclusion: the larger-token check confirms that the current custom routes do not beat
cuBLASLt beyond `M=256`; the remaining gap is not just a small-M artifact.

### cuBLASLt h64 heuristic check

Follow-up reason:

- Most corrected comparisons used up to 32 cuBLASLt heuristic candidates.
- To check whether the library baseline was still under-tuned, this run increases the
  cuBLASLt candidate count to 64 over the current critical `M=64..1024` range.

Command:

```bash
GEMMA4_PREFILL_GRAPH_REPEATS=16 \
  python3 src/experiments/gemma4_prefill_tune.py \
  --backend sgemm-bf16 --ops ffn_down \
  --configs bf16_auto_ffn_down,bf16_cutlass_64x64_s10,bf16_cutlass_64x128_s6,bf16_cutlass_128x128_s5,bf16_cutlass_128x256,bf16_cutlass_256x128,bf16_streamk_64x64x64,bf16_streamk_s2_128x128x64 \
  --m 64,96,128,256,512,1024 --iters 16 --warmup 4 \
  --cublas-backend lt --cublaslt-heuristics 64 --skip-build \
  --out build/experiments/gemma4_prefill_tune/sgemm_bf16_ffn_down_h64_check_cublaslt.csv
```

Best custom config per `M`:

| M | Best custom config | cuBLASLt h64 ms | Custom ms | Speedup |
| ---: | --- | ---: | ---: | ---: |
| 64 | `bf16_cutlass_64x64_s10` | 0.3420 | 0.3378 | 1.012x |
| 96 | `bf16_cutlass_64x128_s6` | 0.3720 | 0.3552 | 1.047x |
| 128 | `bf16_streamk_s2_128x128x64` | 0.3769 | 0.3619 | 1.041x |
| 256 | `bf16_cutlass_128x128_s5` | 0.5024 | 0.5341 | 0.941x |
| 512 | `bf16_cutlass_256x128` | 0.9383 | 0.9633 | 0.974x |
| 1024 | `bf16_cutlass_256x128` | 1.8999 | 1.8876 | 1.007x |

Summary:

- Dynamic custom-only speedup over `M=64..1024`: `0.9981x`.
- No `M` routes to custom at the `1.05x` threshold when using the minimum cuBLASLt h64
  baseline per `M`.
- Increasing the cuBLASLt heuristic count does not create a custom win; it reinforces
  that the remaining custom route is near parity at best.

Conclusion: h64 cuBLASLt remains the correct comparison target for this shape class, and
the current custom candidates still do not produce the requested broad 15-20% win.

## 2026-05-21 - Quack Python CuTe RMSNorm on Gemma 4 shapes

Scope:

- Download upstream `Dao-AILab/quack/quack/rmsnorm.py` directly into
  `src/experiments/quack_rmsnorm/rmsnorm.py`.
- Exercise that exact downloaded file through a small Gemma-specific Python benchmark
  wrapper.
- Tune the forward launch knobs that Quack exposes for the Gemma 4 A6000 target:
  `num_threads`, `threads_per_row`, `reload_from`, and `delay_w_load`.
- Compare against the current native CUDA RMSNorm benchmark.

Environment:

- GPU: NVIDIA RTX A6000, compute capability 8.6.
- Driver: 580.126.16.
- CUDA toolkit: 12.9, `nvcc V12.9.86`.
- Python: 3.12.13.
- Quack support package installed from GitHub main commit
  `5e4d024e839bf851b820887c4c5b9656f206fa08`, because the downloaded `main`
  file imports modules such as `quack.dsl` that are not present in the older PyPI
  wheel.

Files:

- `src/experiments/quack_rmsnorm/rmsnorm.py`
- `src/experiments/quack_rmsnorm/gemma4_quack_rmsnorm_bench.py`
- Result CSVs under `/tmp/gemma4_quack_rmsnorm_20260521/`

Commands:

```bash
python3 -m pip install --user quack-kernels
python3 -m pip install --user --force-reinstall --no-deps \
  git+https://github.com/Dao-AILab/quack.git

python3 -m py_compile \
  src/experiments/quack_rmsnorm/gemma4_quack_rmsnorm_bench.py

python3 src/experiments/quack_rmsnorm/gemma4_quack_rmsnorm_bench.py \
  --width 5376 --rows 1,64 \
  --modes weighted,residual_quack_fused,residual_gemma_exact_split \
  --iters 100 --warmup 10 --trials 2 \
  --configs heuristic,128:64:none:0,128:128:none:0,256:64:none:0,256:128:none:0 \
  > /tmp/gemma4_quack_rmsnorm_20260521/quack_w5376_sweep.csv

python3 src/experiments/quack_rmsnorm/gemma4_quack_rmsnorm_bench.py \
  --width 5376 --rows 1,64 \
  --modes weighted,residual_quack_fused \
  --iters 100 --warmup 10 --trials 2 \
  --configs 128:64:none:1,128:64:smem:0,128:64:gmem:0,256:128:none:1,256:128:smem:0,256:128:gmem:0 \
  > /tmp/gemma4_quack_rmsnorm_20260521/quack_w5376_reload_delay_sweep.csv

python3 src/experiments/quack_rmsnorm/gemma4_quack_rmsnorm_bench.py \
  --width 256 --rows 1,64 --modes weighted,scale_free \
  --iters 100 --warmup 10 --trials 2 \
  --configs 128:64:none:0,128:128:none:0 \
  > /tmp/gemma4_quack_rmsnorm_20260521/quack_w256_best.csv

make -B rmsnorm-bench
GEMMA4_RMSNORM_BENCH_SEED=0x20260521 \
  ./build/experiments/gemma4_rmsnorm_bench 100 10 2 64 5376 \
  > /tmp/gemma4_quack_rmsnorm_20260521/native_w5376_rows64.csv
GEMMA4_RMSNORM_BENCH_SEED=0x20260521 \
  ./build/experiments/gemma4_rmsnorm_bench 100 10 2 64 256 \
  > /tmp/gemma4_quack_rmsnorm_20260521/native_w256_rows64.csv
```

Correctness:

- Quack weighted and scale-free outputs matched the BF16 PyTorch reference in these
  runs with `max_abs <= 0.0009765625` and `rstd_max_abs <= 2.38419e-7`.
- `residual_quack_fused` is not Gemma-exact: Quack normalizes the FP32 `x + residual`
  value. The current CUDA fused kernel intentionally normalizes the BF16-rounded
  residual to match standalone residual add followed by RMSNorm.
- `residual_gemma_exact_split` models the Gemma-exact two-step semantics in Python,
  but it is not a useful replacement kernel because it launches/copies separately.

Best fixed Quack configs from the sweep:

| Shape / mode | Rows | Best Quack config | Quack ms | Native graph ms | Slowdown |
| --- | ---: | --- | ---: | ---: | ---: |
| width 5376 weighted | 1 | `256:128:gmem:0` | 0.040568 | 0.002213 | 18.33x |
| width 5376 weighted | 64 | `256:128:smem:0` | 0.039627 | 0.008176 | 4.85x |
| width 5376 Quack residual fused | 1 | `256:128:smem:0` | 0.034151 | 0.002416 | 14.14x |
| width 5376 Quack residual fused | 64 | `256:128:smem:0` | 0.033789 | 0.002925 | 11.55x |
| width 256 weighted | 1 | `128:128:none:0` | 0.033301 | 0.001599 | 20.83x |
| width 256 weighted | 64 | `128:128:none:0` | 0.037027 | 0.001804 | 20.53x |
| width 256 scale-free | 1 | `128:64:none:0` | 0.027449 | 0.001383 | 19.85x |
| width 256 scale-free | 64 | `128:64:none:0` | 0.030554 | 0.001421 | 21.50x |

Conclusion:

- The Quack Python CuTe RMSNorm source is now downloaded and runnable on this machine,
  but it is not competitive with our native CUDA RMSNorm on RTX A6000/SM86.
- The full Quack autotuner was not practical here: even a single `1 x 256` tuned run
  compiled for over a minute before being killed. Fixed-config tuning was enough to
  show a large gap.
- Do not replace the current RMSNorm kernels with this Quack path. Keep it as an
  experiment/reference for CuTe structure; the production path should remain the
  specialized CUDA implementation.

## 2026-05-22 - FlashAttention-2 SM80 sliding-attention extraction

Runtime files:

- `src/gemma4_flash_attention.cu`
- `src/experiments/gemma4_flash_attention_bench.cu`
- `src/third_party_stubs/ATen/cuda/detail/UnpackRaw.cuh`
- `Makefile`
- `experiments/flash-attention/csrc/flash_attn/src/*`
- `experiments/flash-attention/csrc/cutlass/include/*`

Change:

- Added a raw CUDA Gemma 4 sliding-attention launcher around FlashAttention-2's SM80
  forward device template.
- The project-facing implementation stays in one translation unit:
  `src/gemma4_flash_attention.cu`.
- The path is specialized to the Gemma 4 31B sliding geometry:
  - BF16
  - batch-major `[B, S, H, D]`
  - `Hq=32`
  - `Hkv=16`
  - `D=256`
  - local causal sliding mask with `window_left=1024`, `window_right=0`
- The launch shape follows the A6000/SM86 branch from FA2's `hdim256` launcher:
  `Flash_fwd_kernel_traits<256, 64, 64, 4, false, false, bf16>`.
- Added a benchmark with a small CPU BF16 reference check.

Commands:

```bash
git -C experiments/flash-attention submodule update --init --recursive csrc/cutlass

make flash-attn-bench

./build/experiments/gemma4_flash_attention_bench 64 10 3 1 1 64

./build/experiments/gemma4_flash_attention_bench 1024 100 20 3 1 64

nvcc -std=c++17 -O3 -arch=sm_86 --expt-relaxed-constexpr -Xptxas -v \
  -Isrc -Isrc/third_party_stubs \
  -Iexperiments/flash-attention/csrc/flash_attn/src \
  -Iexperiments/flash-attention/csrc/cutlass/include \
  -c src/gemma4_flash_attention.cu \
  -o /tmp/gemma4_flash_attention_resource.o

ncu --metric gpu__time_duration.sum --target-processes all \
  ./build/experiments/gemma4_flash_attention_bench 1024 10 5 1 1 0
```

Environment:

- GPU: NVIDIA RTX A6000, compute capability 8.6
- Driver: 580.126.16
- CUDA/NVCC: 12.9.86
- Clocks: not locked
- Cache policy: repeated-buffer warm-cache benchmark
- Official `flash_attn` Python package: not installed in system Python or uv env
- Nsight Compute: installed, but the quick `ncu` probe above was interrupted after it
  produced no output for roughly a minute. Re-run with fewer launches or an
  ncu-specific harness that launches only one timed kernel.

Correctness:

- CPU reference check at `B=1, S=64, Hq=32, Hkv=16, D=256`, BF16 inputs:
  - `max_abs=0.00390625`
  - `mean_abs=7.7201e-05`
  - `max_rel=0.00390625`

Benchmark:

| Shape | Warmup | Iters | Trials | Best ms | Avg ms | Approx TFLOP/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `B=1, S=64, Hq=32, Hkv=16, D=256` | 3 | 10 | 1 | 0.0094272 | 0.0094272 | 7.22987 |
| `B=1, S=1024, Hq=32, Hkv=16, D=256` | 20 | 100 | 3 | 0.227083 | 0.234317 | 75.7286 |

Resource usage from `ptxas`:

| Kernel variant | Registers/thread | Stack bytes | Spill stores | Spill loads | Dynamic smem |
| --- | ---: | ---: | ---: | ---: | ---: |
| uneven-M/N local sliding | 254 | 0 | 0 | 0 | 98304 |
| even-M/N local sliding | 255 | 0 | 0 | 0 | 98304 |

Conclusion:

- This is now a compiling and numerically checked FA2-derived Gemma sliding-attention
  path.
- It is not yet a proof of 100% parity with the installed FlashAttention library because
  `flash_attn` is not installed in this environment. The next comparison step is to
  build/install the official package or add a local PyTorch extension comparator and run
  the exact same tensors through both paths.
- The implementation does not cover Gemma global attention (`D=512`); FA2's CUDA path
  only has head dimensions up to 256.

## 2026-05-22 - FlashAttention-2 sliding parity against upstream hdim256 source

Runtime files:

- `src/gemma4_flash_attention.cu`
- `src/experiments/gemma4_flash_attention_reference.cu`
- `src/experiments/gemma4_flash_attention_compare.py`
- `src/third_party_stubs/ATen/cuda/CUDAGeneratorImpl.h`
- `src/third_party_stubs/ATen/cuda/detail/UnpackRaw.cuh`
- `src/third_party_stubs/c10/cuda/CUDAException.h`
- `Makefile`

Change:

- Switched the project sliding-attention path from a hand-written `compute_attn`
  kernel wrapper to FA2's upstream `flash_fwd_launch_template.h` launcher and
  upstream `Flash_fwd_params` type under a project-local namespace.
- Matched FA2's hdim256 local dispatch behavior: the local hdim256 path uses
  `Is_even_MN=false` even for tile-aligned sequence lengths.
- Matched upstream compile flags for this path:
  `--expt-relaxed-constexpr --expt-extended-lambda --use_fast_math
  -D_GLIBCXX_USE_CXX11_ABI=1`.
- Added a small official-source reference shared library that calls
  `run_mha_fwd_<cutlass::bfloat16_t, 256, false>` from upstream
  `flash_fwd_hdim256_bf16_sm80.cu`.

Commands:

```bash
make -B flash-attn-bench flash-attn-lib flash-attn-reference-lib

./build/experiments/gemma4_flash_attention_bench 64 10 3 1 1 64

./build/experiments/gemma4_flash_attention_bench 1024 100 20 3 1 1

uv run python src/experiments/gemma4_flash_attention_compare.py \
  --seq 1024 --warmup 20 --iters 100

nvcc -std=c++17 -O3 -arch=sm_86 \
  --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math \
  -D_GLIBCXX_USE_CXX11_ABI=1 -Xptxas -v \
  -Isrc -Isrc/third_party_stubs \
  -Iexperiments/flash-attention/csrc/flash_attn/src \
  -Iexperiments/flash-attention/csrc/cutlass/include \
  -c src/gemma4_flash_attention.cu \
  -o build/ptx/gemma4_flash_attention_custom.o \
  2>&1 | tee build/ptx/gemma4_flash_attention_custom_ptxas.log

nvcc -std=c++17 -O3 -arch=sm_86 \
  --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math \
  -D_GLIBCXX_USE_CXX11_ABI=1 -Xptxas -v \
  -Iexperiments/flash-attention/csrc/flash_attn/src \
  -Iexperiments/flash-attention/csrc/cutlass/include \
  -I.venv/lib/python3.11/site-packages/torch/include \
  -I.venv/lib/python3.11/site-packages/torch/include/torch/csrc/api/include \
  -I/usr/include/python3.11 \
  -c experiments/flash-attention/csrc/flash_attn/src/flash_fwd_hdim256_bf16_sm80.cu \
  -o build/ptx/flash_fwd_hdim256_bf16_sm80_official.o \
  2>&1 | tee build/ptx/flash_fwd_hdim256_bf16_sm80_official_ptxas.log
```

Results:

- CPU reference check at `B=1, S=64`: `max_abs=0.00390625`,
  `mean_abs=7.72008e-05`, `max_rel=0.00390625`.
- C++ warm-cache benchmark at `B=1, S=1024`: best `0.227048 ms`,
  avg `0.229142 ms`, approx `75.7402 TFLOP/s`.
- Same-tensor Python comparison at `B=1, S=1024`:
  - custom: `0.236691 ms`, approx `72.654 TFLOP/s`
  - official-source reference: `0.236599 ms`, approx `72.683 TFLOP/s`
  - `custom/ref=1.000387`
  - output diff: `max_abs=0`, `mean_abs=0`, `max_rel=0`
- Runtime launch resources reported by both wrappers:
  - threads/block: `128`
  - dynamic shared memory: `98304` bytes

Resource parity for the selected local hdim256 BF16 kernel:

| Source | Registers/thread | Stack bytes | Spill stores | Spill loads | cmem[0] |
| --- | ---: | ---: | ---: | ---: | ---: |
| custom launcher-template path | 255 | 0 | 0 | 0 | 824 |
| upstream `flash_fwd_hdim256_bf16_sm80.cu` | 255 | 0 | 0 | 0 | 824 |

Notes:

- Building the full editable Python `flash_attn` package was attempted in the uv
  environment, but the build was interrupted and left orphaned NVCC children while
  compiling unused training/head-dim variants. Those processes were killed. The
  Python package still does not import: `No module named 'flash_attn'`.
- This proves parity against the upstream hdim256 BF16 CUDA source path, but not yet
  against an installed full `flash_attn.flash_attn_func` package call.
- Gemma global attention (`D=512`) is still uncovered. The upstream FA2 CUDA path does
  not provide an SM80 hdim512 implementation; a separate design is needed for that
  requirement.

Deep verification rerun after adding direct `cudaFuncGetAttributes` checks:

```bash
make -B flash-attn-bench flash-attn-lib flash-attn-reference-lib

./build/experiments/gemma4_flash_attention_bench 64 10 3 1 1 64

./build/experiments/gemma4_flash_attention_bench 1024 100 20 3 1 64

uv run python src/experiments/gemma4_flash_attention_compare.py \
  --seq 1024 --warmup 50 --iters 1000 --trials 5 --skip-python-flash-attn
```

Results:

- CPU reference check at `B=1, S=64`: `max_abs=0.00390625`,
  `mean_abs=7.72008e-05`, `max_rel=0.00390625`.
- C++ warm-cache benchmark at `B=1, S=1024`: best `0.228857 ms`,
  avg `0.230685 ms`, approx `75.1416 TFLOP/s`.
- Same-tensor Python comparison at `B=1, S=1024`, 5 trials of 1000 launches:
  - custom median/min/max: `0.238173 / 0.236438 / 0.239902 ms`
  - official-source reference median/min/max: `0.238188 / 0.237964 / 0.240831 ms`
  - `custom/ref=0.999940` on median timing
  - output diff: `max_abs=0`, `mean_abs=0`, `max_rel=0`

Direct `cudaFuncGetAttributes` comparison for the exact selected kernel:

| Attribute | Custom | Official-source ref | Match |
| --- | ---: | ---: | ---: |
| `sharedSizeBytes` | 0 | 0 | yes |
| `constSizeBytes` | 0 | 0 | yes |
| `localSizeBytes` | 0 | 0 | yes |
| `maxThreadsPerBlock` | 256 | 256 | yes |
| `numRegs` | 255 | 255 | yes |
| `ptxVersion` | 86 | 86 | yes |
| `binaryVersion` | 86 | 86 | yes |
| `cacheModeCA` | 0 | 0 | yes |
| `maxDynamicSharedSizeBytes` | 98304 | 98304 | yes |
| `preferredShmemCarveout` | -1 | -1 | yes |

Current-source `ptxas` comparison for the selected local hdim256 BF16 kernel:

| Source | Registers/thread | Stack bytes | Spill stores | Spill loads | Barriers | cmem[0] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| custom `src/gemma4_flash_attention.cu` | 255 | 0 | 0 | 0 | 1 | 824 |
| upstream `flash_fwd_hdim256_bf16_sm80.cu` | 255 | 0 | 0 | 0 | 1 | 824 |

Conclusion:

- For Gemma sliding attention (`D=256`, BF16, local window 1024), the custom wrapper
  now matches the upstream hdim256 BF16 source path on output, launch shape, dynamic
  shared memory, direct CUDA function attributes, and selected ptxas resource stats.
- Timing is not a bitwise statistic, but the longer CUDA-event run is effectively equal
  (`custom/ref=0.999940` median). GPU clocks were not locked, so min/max should be
  treated as noisy.

Follow-up confirmation rerun:

```bash
uv run python src/experiments/gemma4_flash_attention_compare.py \
  --seq 1024 --warmup 50 --iters 1000 --trials 5 --skip-python-flash-attn

nvcc -std=c++17 -O3 -arch=sm_86 \
  --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math \
  -D_GLIBCXX_USE_CXX11_ABI=1 -Xptxas -v \
  -Isrc -Isrc/third_party_stubs \
  -Iexperiments/flash-attention/csrc/flash_attn/src \
  -Iexperiments/flash-attention/csrc/cutlass/include \
  -c src/gemma4_flash_attention.cu \
  -o build/ptx/gemma4_flash_attention_custom.o \
  2>&1 | tee build/ptx/gemma4_flash_attention_custom_ptxas.log

nvcc -std=c++17 -O3 -arch=sm_86 \
  --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math \
  -D_GLIBCXX_USE_CXX11_ABI=1 -Xptxas -v \
  -Iexperiments/flash-attention/csrc/flash_attn/src \
  -Iexperiments/flash-attention/csrc/cutlass/include \
  -I.venv/lib/python3.11/site-packages/torch/include \
  -I.venv/lib/python3.11/site-packages/torch/include/torch/csrc/api/include \
  -c experiments/flash-attention/csrc/flash_attn/src/flash_fwd_hdim256_bf16_sm80.cu \
  -o build/ptx/flash_fwd_hdim256_bf16_sm80_official.o \
  2>&1 | tee build/ptx/flash_fwd_hdim256_bf16_sm80_official_ptxas.log
```

Rerun results:

- Same-tensor output diff: `max_abs=0`, `mean_abs=0`, `max_rel=0`.
- CUDA-event timing, `B=1`, `S=1024`, 5 trials of 1000 launches:
  - custom median/min/max: `0.244411 / 0.243209 / 0.249125 ms`
  - official-source reference median/min/max: `0.244946 / 0.243538 / 0.251055 ms`
  - `custom/ref=0.997813` on median timing
- Every direct `cudaFuncGetAttributes` field matched:
  `sharedSizeBytes`, `constSizeBytes`, `localSizeBytes`, `maxThreadsPerBlock`,
  `numRegs`, `ptxVersion`, `binaryVersion`, `cacheModeCA`,
  `maxDynamicSharedSizeBytes`, and `preferredShmemCarveout`.
- Selected local hdim256 BF16 ptxas resource line matched:
  - custom: `255` regs/thread, `0` stack, `0` spill stores, `0` spill loads,
    `1` barrier, `824` bytes `cmem[0]`
  - upstream source: `255` regs/thread, `0` stack, `0` spill stores,
    `0` spill loads, `1` barrier, `824` bytes `cmem[0]`

Conclusion holds: all deterministic output, launch, runtime attribute, and selected
compiler-resource statistics checked here match exactly. Timing remains within normal
unlocked-clock noise.

## 2026-05-22 - Python FlashAttention comparison for hdim256 BF16

Goal:

- Compare the Gemma sliding-attention wrapper against the installed Python
  `flash_attn.flash_attn_func` path on the same tensors.

Package setup:

```bash
FLASH_ATTENTION_FORCE_BUILD=TRUE \
FLASH_ATTENTION_GEMMA_FWD_ONLY=TRUE \
FLASH_ATTN_CUDA_ARCHS=80 \
MAX_JOBS=4 \
NVCC_THREADS=1 \
uv pip install --reinstall --no-deps --no-build-isolation experiments/flash-attention
```

Notes:

- No matching prebuilt wheel exists for `flash-attn 2.8.4`, Python 3.11,
  Torch `2.11`, CUDA 12, CXX11 ABI true:
  `flash_attn-2.8.4+cu12torch2.11cxx11abiTRUE-cp311-cp311-linux_x86_64.whl`
  returned HTTP 404 from the upstream release URL.
- A full unmodified source build was attempted with `FLASH_ATTN_CUDA_ARCHS=80`,
  `MAX_JOBS=4`, and `NVCC_THREADS=1`, but it spent a long time compiling all
  forward and split-KV instantiations. It was stopped after the hdim256 forward
  path had compiled but before the full package finished.
- For this comparison, the local upstream checkout was given a targeted
  `FLASH_ATTENTION_GEMMA_FWD_ONLY` build mode. It keeps the public Python
  `flash_attn_func` API but compiles only the BF16 hdim256 forward and causal
  forward instantiations required by these Gemma sliding-attention benchmarks.
  Unsupported dtypes/head dims/split-KV paths intentionally fail.
- Torch stayed at `2.11.0+cu128` after install.

Verification:

```bash
uv run python - <<'PY'
import torch
import flash_attn
import flash_attn_2_cuda
from flash_attn import flash_attn_func
print(torch.__version__, torch.version.cuda)
print(flash_attn.__version__)
print(flash_attn_2_cuda.__file__)
print(flash_attn_func)
PY
```

Import result:

- Torch: `2.11.0+cu128`, CUDA runtime version reported by Torch: `12.8`.
- `flash_attn`: `2.8.4`.
- `flash_attn_2_cuda` loaded from the repo `.venv`.
- `flash_attn_func` imported successfully.

Smoke checks:

```bash
uv run python src/experiments/gemma4_flash_attention_compare.py \
  --seq 1024 --warmup 5 --iters 20 --trials 2

uv run python src/experiments/gemma4_flash_attention_compare.py \
  --seq 2048 --warmup 5 --iters 10 --trials 2
```

Smoke results:

- `S=1024`: `diff_vs_flash_attn max_abs=0 mean_abs=0 max_rel=0`.
- `S=2048`: `diff_vs_flash_attn max_abs=0 mean_abs=0 max_rel=0`.

Clean sequential timing runs:

```bash
uv run python src/experiments/gemma4_flash_attention_compare.py \
  --seq 1024 --warmup 50 --iters 1000 --trials 5

uv run python src/experiments/gemma4_flash_attention_compare.py \
  --seq 2048 --warmup 50 --iters 500 --trials 5
```

Results at `B=1`, `S=1024`:

- custom median/min/max: `0.236058 / 0.235717 / 0.238015 ms`
- official-source reference median/min/max:
  `0.237129 / 0.235751 / 0.239483 ms`
- Python `flash_attn_func` median/min/max:
  `0.234289 / 0.232250 / 0.237240 ms`
- `diff_vs_official_source_ref max_abs=0 mean_abs=0 max_rel=0`
- `diff_vs_flash_attn max_abs=0 mean_abs=0 max_rel=0`
- `custom/python_flash_attn=1.007549` on median timing

Results at `B=1`, `S=2048`:

- custom median/min/max: `0.606319 / 0.605090 / 0.618479 ms`
- official-source reference median/min/max:
  `0.611226 / 0.609790 / 0.613673 ms`
- Python `flash_attn_func` median/min/max:
  `0.614401 / 0.613331 / 0.617600 ms`
- `diff_vs_official_source_ref max_abs=0 mean_abs=0 max_rel=0`
- `diff_vs_flash_attn max_abs=0 mean_abs=0 max_rel=0`
- `custom/python_flash_attn=0.986845` on median timing

Runtime attribute comparison in both runs:

- Every queried `cudaFuncGetAttributes` field matched between the custom wrapper
  and the official-source reference wrapper:
  `sharedSizeBytes`, `constSizeBytes`, `localSizeBytes`, `maxThreadsPerBlock`,
  `numRegs`, `ptxVersion`, `binaryVersion`, `cacheModeCA`,
  `maxDynamicSharedSizeBytes`, and `preferredShmemCarveout`.

Conclusion:

- For the benchmarked Gemma sliding path (`BF16`, `D=256`, `32` Q heads,
  `16` KV heads, local window left `1024`, right `0`), the custom wrapper
  matches Python `flash_attn_func` numerically with exact zero output diff.
- Clean sequential timing is within normal unlocked-clock noise of the Python
  path: about `+0.75%` at `S=1024` and `-1.32%` at `S=2048`.
- This satisfies parity for the Python API path used by these shapes. It is not
  a full-package validation of every upstream FlashAttention feature because the
  installed package was intentionally built in Gemma-only forward mode.
## 2026-05-23 - FFN decode register prefetch vs 3-stage async weight staging

Goal:

- Test two decode-FFN GEMV scheduling variants for the gate/up projection:
  direct register prefetch and shared-memory async staging.
- Use the existing cuDNN split FFN benchmark harness for correctness and timing.
- Do not keep multiple source-level selection flags for the variants.

CUDA guide context:

- The CUDA guide describes pipelines as a FIFO mechanism for sequencing
  asynchronous copies and supporting double/multi-buffered producer-consumer
  patterns (CUDA Programming Guide, p.129).
- The LDGSTS async-copy path is specifically for global-to-shared copies on
  CC 8.0+ and can keep global-memory loads in flight while reducing register
  pressure (CUDA Programming Guide, p.300 and p.302).
- The guide's multi-stage prefetch example blocks on the current batch,
  computes it, and schedules a future batch (CUDA Programming Guide, p.312).

Common benchmark setup:

```bash
GEMMA4_FFN_CUDNN_BENCH_SEED=0x20260522 \
  build/experiments/<binary> 100 20 3
```

- GPU: NVIDIA RTX A6000.
- Shape: `tokens=1`, `hidden=5376`, `intermediate=21504`.
- Iterations: `100` timed, `20` warmup, `3` trials.
- cuDNN frontend version: `92200`.
- Full fused cuDNN FFN graph had no valid execution plan, so the comparison is
  against the benchmark's cuDNN split path.
- All custom variants reported `max_abs_vs_split=0`.

### Variant 1: direct register prefetch

Implementation tested:

- Kept `ACT_TILE=2`.
- Removed the shared-memory weight pit stop for gate/up.
- Each owning thread loaded current gate/up `Bf16Packed128` values directly
  from global memory into registers.
- Issued the next tile's vectorized global loads before consuming the current
  tile's register values.
- Swapped register sets after the current tile's math/down accumulation.

Build and run:

```bash
nvcc -std=c++17 -O3 -arch=sm_86 -Xptxas=-v \
  -Isrc -I/tmp/cudnn-frontend/include \
  src/experiments/gemma4_ffn_cudnn_bench.cu \
  src/gemma4_ffn_decode.cu \
  -lcudnn -lnvrtc -lcuda \
  -o build/experiments/gemma4_ffn_cudnn_bench_regpipe

GEMMA4_FFN_CUDNN_BENCH_SEED=0x20260522 \
  build/experiments/gemma4_ffn_cudnn_bench_regpipe 100 20 3
```

Compiler resource result:

- FFN kernel: `64` registers/thread.
- Spills: `0` spill stores, `0` spill loads.
- Static shared memory: `11404` bytes.
- Barriers: `1`.

Timing:

- `custom_device_ms`: `1.106724`
- `custom_scratch_clear_device_ms`: `0.001092`
- `custom_minus_clear_device_ms`: `1.105632`
- `cudnn_split_device_ms`: `1.000129`
- `custom_minus_clear_vs_cudnn_split_speedup`: `0.904577`
- `max_abs_vs_split`: `0`

### Variant 2: 3-stage shared async staging

Implementation tested:

- Used one active implementation, not an extra runtime flag.
- Prologue issued async global-to-shared copies for tile `0` and tile `1`.
- Main loop waited for the current tile, issued async copies for tile `k+2`
  into the next rotating shared-memory stage, then computed tile `k`.
- Used `__pipeline_memcpy_async`, `__pipeline_commit`, and
  `__pipeline_wait_prior`.
- Used `ACT_TILE=1` for this run. With `ACT_TILE=2`, the required dynamic
  shared-memory staging size is
  `3 * 2 * 2 * 672 * 16 = 129024` bytes before static shared memory, which is
  too large for this target's per-block shared-memory budget.

Build and run:

```bash
nvcc -std=c++17 -O3 -arch=sm_86 -Xptxas=-v \
  -Isrc -I/tmp/cudnn-frontend/include \
  src/experiments/gemma4_ffn_cudnn_bench.cu \
  src/gemma4_ffn_decode.cu \
  -lcudnn -lnvrtc -lcuda \
  -o build/experiments/gemma4_ffn_cudnn_bench_async3

GEMMA4_FFN_CUDNN_BENCH_SEED=0x20260522 \
  build/experiments/gemma4_ffn_cudnn_bench_async3 100 20 3
```

Compiler resource result:

- FFN kernel: `50` registers/thread.
- Spills: `0` spill stores, `0` spill loads.
- Static shared memory: `11152` bytes.
- Dynamic shared-memory weight staging: `64512` bytes.
- Barriers: `1`.

Timing:

- `custom_device_ms`: `1.116366`
- `custom_scratch_clear_device_ms`: `0.001102`
- `custom_minus_clear_device_ms`: `1.115264`
- `cudnn_split_device_ms`: `1.000023`
- `custom_minus_clear_vs_cudnn_split_speedup`: `0.896669`
- `max_abs_vs_split`: `0`

### Restored final source path: shared-X paired reducer

After testing both requested variants, the source was restored to the previously
fastest measured approach:

- `ACT_TILE=2`.
- Load `x` once into shared memory.
- Use the paired shared-X reducer for gate/up.
- Do not stage weights through dynamic shared memory.

Build and run:

```bash
nvcc -std=c++17 -O3 -arch=sm_86 -Xptxas=-v \
  -Isrc -I/tmp/cudnn-frontend/include \
  src/experiments/gemma4_ffn_cudnn_bench.cu \
  src/gemma4_ffn_decode.cu \
  -lcudnn -lnvrtc -lcuda \
  -o build/experiments/gemma4_ffn_cudnn_bench_sharedx

GEMMA4_FFN_CUDNN_BENCH_SEED=0x20260522 \
  build/experiments/gemma4_ffn_cudnn_bench_sharedx 100 20 3
```

Compiler resource result:

- FFN kernel: `48` registers/thread.
- Spills: `0` spill stores, `0` spill loads.
- Static shared memory: `11404` bytes.
- Barriers: `1`.

Timing:

- `custom_device_ms`: `1.106684`
- `custom_scratch_clear_device_ms`: `0.001081`
- `custom_minus_clear_device_ms`: `1.105603`
- `cudnn_split_device_ms`: `1.001380`
- `custom_minus_clear_vs_cudnn_split_speedup`: `0.905732`
- `max_abs_vs_split`: `0`

Conclusion:

- Direct register prefetch and restored shared-X paired reduction are effectively
  tied in this run: `1.105632 ms` vs `1.105603 ms` minus scratch clear.
- The 3-stage shared async staging path was slower at `1.115264 ms` minus
  scratch clear, even though it compiled without spills. It also required
  dropping to `ACT_TILE=1` to fit shared memory, which increased loop/reduction
  overhead.
- For `tokens=1` decode FFN GEMV, staging gate/up weights through shared memory
  does not currently pay off. The best source state after this experiment is
  the shared-X paired reducer with direct coalesced weight reads.
## 2026-05-24 - FFN decode offline hidden-pack weight swizzle

Goal:

- Pre-swizzle FFN decode weights offline so the hidden-dimension weight packs
  match the shared-memory swizzle used for `x`.
- Keep the best measured decode topology: shared-X paired gate/up reduction,
  direct coalesced weight reads, and no dynamic shared-memory weight staging.

Implementation:

- Added `gemma4_ffn_decode_swizzle_weights_bf16()` as an offline preparation
  helper.
- The helper rewrites each contiguous hidden row/column in 128-bit bf16 packs:
  source pack `p` is stored at `shared_x_chunk_index(p)`.
- Updated the FFN decode contract so `w_gate_up_col_major` and
  `w_down_row_major` are expected to have their hidden dimension pre-swizzled.
- Updated the gate/up shared-X dot path to read weight pack
  `shared_pack_index(pack_idx)` while reading `s_x` at the same swizzled pack.
- Updated the down projection to read the swizzled hidden pack while accumulating
  and storing output in the original hidden order.
- The benchmark now keeps canonical cuDNN weights separate and passes swizzled
  copies only to the custom fused decode path.

Build and run:

```bash
nvcc -std=c++17 -O3 -arch=sm_86 -Xptxas=-v \
  -Isrc -I/tmp/cudnn-frontend/include \
  src/experiments/gemma4_ffn_cudnn_bench.cu \
  src/gemma4_ffn_decode.cu \
  -lcudnn -lnvrtc -lcuda \
  -o build/experiments/gemma4_ffn_cudnn_bench_swizzled

GEMMA4_FFN_CUDNN_BENCH_SEED=0x20260522 \
  build/experiments/gemma4_ffn_cudnn_bench_swizzled 100 20 3
```

Compiler resource result:

- Offline swizzle kernel: `16` registers/thread, `0` spills.
- FFN decode kernel: `46` registers/thread, `0` spills.
- FFN static shared memory: `11404` bytes.
- FFN barriers: `1`.

Timing:

- `custom_device_ms`: `1.103116`
- `custom_scratch_clear_device_ms`: `0.001129`
- `custom_minus_clear_device_ms`: `1.101987`
- `cudnn_split_device_ms`: `1.000330`
- `custom_minus_clear_vs_cudnn_split_speedup`: `0.907751`
- `max_abs_vs_split`: `0`

Comparison:

- Previous restored shared-X paired reducer run:
  `custom_minus_clear_device_ms = 1.105603`
- Offline hidden-pack weight swizzle:
  `custom_minus_clear_device_ms = 1.101987`
- Delta: about `0.003616 ms` faster, roughly `0.33%`.

Conclusion:

- Pre-swizzling the hidden weight packs is a small win. It reduces FFN decode
  register use from the prior `48` registers/thread to `46` and slightly
  improves measured runtime.
- The dominant limit is still HBM traffic; this is an indexing/layout cleanup,
  not a bandwidth reduction.
## 2026-05-24 - FFN decode canonical vs pre-swizzled weight rerun

Goal:

- Benchmark the canonical hidden-pack weight layout and the offline pre-swizzled
  hidden-pack layout back to back under the same harness settings.

Common setup:

```bash
GEMMA4_FFN_CUDNN_BENCH_SEED=0x20260522 \
  build/experiments/<binary> 100 20 3
```

- GPU: NVIDIA RTX A6000.
- Shape: `tokens=1`, `hidden=5376`, `intermediate=21504`.
- cuDNN split graph timing remained about `1.0 ms`.
- Both custom paths reported `max_abs_vs_split=0`.

Canonical hidden-pack layout:

```bash
nvcc -std=c++17 -O3 -arch=sm_86 -Xptxas=-v \
  -Isrc -I/tmp/cudnn-frontend/include \
  src/experiments/gemma4_ffn_cudnn_bench.cu \
  src/gemma4_ffn_decode.cu \
  -lcudnn -lnvrtc -lcuda \
  -o build/experiments/gemma4_ffn_cudnn_bench_unswizzled
```

- FFN registers/thread: `48`
- FFN spills: `0`
- `custom_device_ms`: `1.111296`
- `custom_scratch_clear_device_ms`: `0.001084`
- `custom_minus_clear_device_ms`: `1.110212`
- `custom_minus_clear_vs_cudnn_split_speedup`: `0.900500`

Pre-swizzled hidden-pack layout:

```bash
nvcc -std=c++17 -O3 -arch=sm_86 -Xptxas=-v \
  -Isrc -I/tmp/cudnn-frontend/include \
  src/experiments/gemma4_ffn_cudnn_bench.cu \
  src/gemma4_ffn_decode.cu \
  -lcudnn -lnvrtc -lcuda \
  -o build/experiments/gemma4_ffn_cudnn_bench_swizzled
```

- FFN registers/thread: `46`
- FFN spills: `0`
- `custom_device_ms`: `1.104086`
- `custom_scratch_clear_device_ms`: `0.001077`
- `custom_minus_clear_device_ms`: `1.103009`
- `custom_minus_clear_vs_cudnn_split_speedup`: `0.906560`

Conclusion:

- The pre-swizzled layout was faster in this rerun:
  `1.103009 ms` vs `1.110212 ms` after scratch-clear subtraction.
- Delta: `0.007203 ms`, about `0.65%` faster.
- Source was left on the pre-swizzled implementation.
## 2026-05-24 - FFN decode compile-time ablation sweep

Goal:

- Sweep the main FFN decode tuning knobs to find a better operating point:
  activation tile width, weight load policy, CTA thread count, and intermediate
  tile width.
- Keep pre-swizzled hidden-pack weights enabled.
- Leave the source defaults on the best measured configuration.

Benchmark setup:

```bash
GEMMA4_FFN_CUDNN_BENCH_SEED=0x20260522 \
  build/experiments/<binary> <iters> <warmup> <trials>
```

- GPU: NVIDIA RTX A6000.
- Shape: `tokens=1`, `hidden=5376`, `intermediate=21504`.
- Cache policy: warm-cache repeated-buffer benchmark, matching the existing
  FFN/cuDNN harness behavior.
- Weight load policy values:
  `0 = __ldcs`, `1 = inline PTX ld.global.cg.v4.u32`, `2 = __ldg`.

First pass: `ACT_TILE x weight load policy`, with `THREADS=1024`,
`INTERMEDIATE_TILE=256`, `iters=50`, `warmup=10`, `trials=2`.

```text
ACT_TILE  policy  custom_minus_clear_ms
1         0       1.166585
1         1       1.168792
1         2       1.164948
2         0       1.105258
2         1       1.101761
2         2       1.102541
4         0       1.102608
4         1       1.104081
4         2       1.103259
8         0       1.525499
8         1       1.527742
8         2       1.634939
```

Conclusion from first pass:

- `ACT_TILE=2`, `.cg` was best in the original `1024`-thread,
  `256`-column CTA shape.
- `ACT_TILE=1` had too much loop/reduction overhead.
- `ACT_TILE=8` was bad at `1024` threads.

Second pass: thread count and intermediate tile width, with `ACT_TILE=2`,
`.cg`, `iters=50`, `warmup=10`, `trials=2`.

```text
THREADS  INTERMEDIATE_TILE  custom_minus_clear_ms
512      128                1.127623
512      256                1.012838
512      512                0.979793
1024     128                unsupported/no parsed timing
1024     256                1.098981
1024     512                1.046427
```

Broader cheap pass around larger intermediate tiles, with `ACT_TILE=2`, `.cg`,
`iters=25`, `warmup=5`, `trials=1`.

```text
THREADS  INTERMEDIATE_TILE  custom_minus_clear_ms
256      512                0.890807
256      768                1.119968
256      1024               1.452534
512      512                0.984314
512      768                1.142086
512      1024               1.424081
768      512                1.045551
768      768                1.104881
768      1024               1.364287
```

Conclusion from CTA-shape passes:

- `INTERMEDIATE_TILE=512` is much better than `256` for this benchmark.
- `256` threads was best at `INTERMEDIATE_TILE=512`.
- Larger intermediate tiles underutilized the GPU and slowed down.

Activation tile check around the new best CTA shape, `.cg`,
`INTERMEDIATE_TILE=512`, `iters=25`, `warmup=5`, `trials=1`.

```text
THREADS  ACT_TILE  regs/thread  custom_minus_clear_ms
256      1         48           1.082970
256      2         48           0.897487
256      4         67           0.832521
256      8         107          0.828842
256      16        147          0.912013
512      1         48           1.064396
512      2         46           0.982698
512      4         67           0.965788
```

Full finalist runs, `iters=100`, `warmup=20`, `trials=3`.

```text
THREADS  INTERMEDIATE_TILE  ACT_TILE  regs/thread  custom_minus_clear_ms  speedup_vs_cudnn_split
256      512                8         107          0.830063              1.205208
256      512                4         67           0.833452              1.200626
512      512                4         67           0.964612              1.036290
```

Final default-source confirmation, no tuning flags:

```text
THREADS=256
INTERMEDIATE_TILE=512
ACT_TILE=8
WEIGHT_LOAD_POLICY=1
regs/thread=107
spills=0
custom_device_ms=0.829718
custom_scratch_clear_device_ms=0.001067
custom_minus_clear_device_ms=0.828651
custom_minus_clear_vs_cudnn_split_speedup=1.206244
```

Conclusion:

- Best measured default is now `THREADS=256`, `INTERMEDIATE_TILE=512`,
  `ACT_TILE=8`, pre-swizzled hidden-pack weights, and inline PTX `.cg` weight
  loads.
- This improves the previously best pre-swizzled path from about `1.103 ms` to
  about `0.829 ms` after scratch-clear subtraction in this warm-cache harness.
- The speedup comes mainly from changing the CTA/reduction shape, not from the
  cache operator. Fewer CTA reduction turns plus larger per-CTA activation work
  beats the original high-thread-count shape.
## 2026-05-24 - FFN decode cold-cache flush benchmark

Goal:

- Re-run the current best FFN decode configuration without relying on warm L2.
- Nsight Compute was unavailable, so the benchmark was extended with an explicit
  cache flush buffer.

Implementation:

- Added a benchmark-only `flush_cache_kernel`.
- Allocated a flush buffer of `268435456` bytes, which is larger than A6000 L2.
- Timed three paths:
  - `flush + custom_fused_decode`
  - `flush + custom_scratch_clear`
  - `flush_only`
- Reported cold custom time as:
  `cold_custom_minus_flush_clear_ms =
   (flush + custom - flush_only) - (flush + clear - flush_only)`.

Build and run:

```bash
nvcc -std=c++17 -O3 -arch=sm_86 -Xptxas=-v \
  -Isrc -I/tmp/cudnn-frontend/include \
  src/experiments/gemma4_ffn_cudnn_bench.cu \
  src/gemma4_ffn_decode.cu \
  -lcudnn -lnvrtc -lcuda \
  -o build/experiments/ffn_ablate/ffn_best_cold

GEMMA4_FFN_CUDNN_BENCH_SEED=0x20260522 \
  build/experiments/ffn_ablate/ffn_best_cold 20 5 2
```

Results:

```text
cache_flush_bytes=268435456
custom_fused_decode graph_best_ms=0.830672
custom_scratch_clear graph_best_ms=0.001339
custom_minus_clear_device_ms=0.829333

custom_fused_decode_cold best_ms=1.236142
custom_scratch_clear_cold best_ms=0.400528
cache_flush_only best_ms=0.398603

cold_custom_minus_flush_ms=0.837539
cold_clear_minus_flush_ms=0.001925
cold_custom_minus_flush_clear_ms=0.835614
```

Conclusion:

- Explicit cache flushing only increased the scratch/fused adjusted custom
  decode time from `0.829333 ms` warm to `0.835614 ms` cold in this run.
- The current best kernel is not getting most of its speed from repeated-buffer
  warm-L2 reuse. It remains mostly bounded by streaming bandwidth and CTA
  scheduling/reduction shape.
## 2026-05-24 - FFN decode live-range ptxas analysis

Goal:

- Check whether unnecessary live ranges were inflating register pressure in the
  current FFN decode kernel.
- Use ptxas/resource output and SASS/PTX dumps because Nsight Compute is not
  available on this machine.

Dump commands:

```bash
mkdir -p build/ptx/ffn_live

nvcc -std=c++17 -O3 -arch=sm_86 -lineinfo -Xptxas=-v \
  -Isrc -cubin src/gemma4_ffn_decode.cu \
  -o build/ptx/ffn_live/gemma4_ffn_decode.cubin \
  > build/ptx/ffn_live/ptxas.log 2>&1

nvcc -std=c++17 -O3 -arch=sm_86 -lineinfo \
  -Isrc -ptx src/gemma4_ffn_decode.cu \
  -o build/ptx/ffn_live/gemma4_ffn_decode.ptx

cuobjdump --dump-sass build/ptx/ffn_live/gemma4_ffn_decode.cubin \
  > build/ptx/ffn_live/gemma4_ffn_decode.sass

cuobjdump --dump-resource-usage build/ptx/ffn_live/gemma4_ffn_decode.cubin \
  > build/ptx/ffn_live/resource.txt
```

Initial active-kernel ptxas result with `ACT_TILE=8`:

```text
REG:107
STACK:0
LOCAL:0
SHARED:11332
spill stores:0
spill loads:0
```

SASS/PTX notes:

- Highest physical register observed in SASS was `R104`, matching the
  `107`-register ptxas allocation.
- No `LDL` or `STL` local-memory spill traffic was observed.
- PTX retained the intended inline cache-global loads:
  `ld.global.cg.v4.u32`.
- The high-register region lined up with the unrolled dot/reduce body, not
  obvious stale address temporaries.

Attempted reorder:

- Removed the separate `s_act[ACT_TILE]` shared array.
- Wrote each GeGLU scalar into existing shared reduction storage and consumed
  it directly for the down accumulation.

ptxas result after that reorder:

```text
REG:107
STACK:0
LOCAL:0
SHARED:11300
spill stores:0
spill loads:0
```

Benchmark after the attempted reorder:

```text
custom_fused_decode graph_best_ms=0.831002
custom_scratch_clear graph_best_ms=0.001129
custom_minus_clear_device_ms=0.829873
cold_custom_minus_flush_clear_ms=0.830545
```

Conclusion from the failed reorder:

- `s_act` was not the register limiter.
- The main simultaneous live values are the gate/up accumulators and unpacked
  bf16 intermediates created by `ACT_TILE=8`.
- Reducing live ranges in a meaningful way requires reducing the activation
  tile width.

Final live-range-oriented source setting:

```text
ACT_TILE=4
THREADS=256
INTERMEDIATE_TILE=512
WEIGHT_LOAD_POLICY=1
```

ptxas result with `ACT_TILE=4`:

```text
REG:67
STACK:0
LOCAL:0
SHARED:11044
spill stores:0
spill loads:0
```

Tradeoff:

- Earlier full finalist timing for `ACT_TILE=8`:
  `custom_minus_clear_device_ms=0.830063`, `107` registers/thread.
- Earlier full finalist timing for `ACT_TILE=4`:
  `custom_minus_clear_device_ms=0.833452`, `67` registers/thread.

Conclusion:

- `ACT_TILE=4` is slightly slower in the warm-cache harness but cuts register
  pressure by about `37%` without spilling.
- Source was left on `ACT_TILE=4` because the immediate goal was to minimize
  simultaneously live values rather than maximize the last few microseconds.

## 2026-05-24 - FFN decode HBM load and CTA-shape sweep

Goal:

- Explore whether earlier HBM load issue, async shared-memory preload, cache
  policy, shared-memory swizzle, or CTA geometry can produce a serious FFN
  decode improvement.
- Keep the default path numerically correct for all `5376` hidden columns.

Implementation:

- Fixed the default `256`-thread path bug where only the first `256` of `672`
  hidden 128-bit packs were owned by threads. Threads now cover hidden packs
  in rounds.
- Added a focused custom-only benchmark:
  `src/experiments/gemma4_ffn_decode_load_bench.cu`.
- Added compile-time experiment toggles:
  - `GEMMA4_FFN_DECODE_PRELOAD_DOWN`
  - `GEMMA4_MATMUL_DEVICE_PRELOAD_PAIR_COLS`
- Removed the second all-thread global acquire spin in the scratch reduction
  turn wait; thread 0 waits for the lock, then the CTA synchronizes.
- Set the default FFN decode CTA size to `672` threads, one thread per hidden
  bf16 pack.

Guide/context:

- CUDA guide query used:
  `python3 scripts/query.py "CUDA asynchronous copy global memory to shared memory cp.async pipeline cache global loads L2 persisting cache occupancy memory throughput" --top-k 10`
- DSMEM query used:
  `python3 scripts/query.py "CUDA distributed shared memory compute capability" --top-k 8`
- Relevant guide notes:
  - async copies can keep global-to-shared memory operations in flight and
    reduce register staging when copying global memory to shared memory;
  - `memcpy_async`/hardware async copies require waiting before reading copied
    shared memory and are best with 16-byte aligned global/shared addresses;
  - occupancy is constrained by registers, shared memory, resident blocks, and
    resident warps, so load overlap must be evaluated with resource usage.
  - distributed shared memory is a compute capability `9.0+` feature, so it is
    not a direct fit for the target `sm_86` RTX A6000 path.
- Exa searches looked at comparable GEMV fusion/vector-load work:
  - llama.cpp general GEMV fusion PR
  - llama.cpp vectorized CUDA dmmv load PR
  - FlashInfer cp.async helper

Build and verification:

```bash
make test-ffn-decode

nvcc -std=c++17 -O3 -arch=sm_86 -Xptxas=-v -Isrc \
  -c src/gemma4_ffn_decode.cu \
  -o build/ptx/ffn_load_play/ffn_decode_final_default.o

make ffn-decode-load-bench
GEMMA4_FFN_LOAD_BENCH_SEED=0x20260522 \
  build/experiments/gemma4_ffn_decode_load_bench 30 8 3

make ffn-cudnn-bench
GEMMA4_FFN_CUDNN_BENCH_SEED=0x20260522 \
  build/experiments/gemma4_ffn_cudnn_bench 50 10 3
```

GPU/tooling:

- GPU: NVIDIA RTX A6000
- CUDA target: `sm_86`
- cuDNN version in full harness: `92200`
- Clock policy: not locked.
- Cache policy: both normal warm-cache graph replay and explicit 256 MiB
  cache-flush timing were measured.

Correctness:

```text
ffn decode tests passed
```

ptxas final default:

```text
Used 67 registers
Used 11512 bytes smem
Spill stores: 0
Spill loads: 0
```

Corrected baseline before tuning:

```text
THREADS=256
ACT_TILE=4
INTERMEDIATE_TILE=512
custom_minus_clear_device_ms=1.058389
cold_custom_minus_flush_clear_ms=1.058719
```

Selected sweep results from the focused load bench, `iters=12`,
`warmup=4`, `trials=2` unless noted:

```text
variant                 warm_minus_clear_ms  cold_minus_flush_clear_ms
threads256_act8         1.072496             1.073171
threads512_act4         1.215563             1.207904
threads512_act8         1.062747             1.063147
threads1024_act4        1.048056             1.041696
threads672_act4         1.042429             1.037168
threads704_act4         1.043683             1.037499
threads768_act4         1.044232             1.046008
threads672_act1         1.120125             1.124309
threads672_act2         1.047235             1.048011
threads672_act8         1.219259             1.223941
threads672_tile256      1.093696             1.094483
threads672_tile384      1.066768             1.064235
threads672_tile448      1.053429             1.055384
threads672_tile1024     1.337568             1.339309
threads672_async_x      1.043688             1.044981
threads672_no_swizzle   1.044600             1.045237
threads672_ldg          1.036069             1.037611
threads672_lock_wait    1.031494             1.034042
gate_up_preload_pairs   1.032064             1.033283
```

Final focused benchmark with default source, `iters=30`, `warmup=8`,
`trials=3`:

```text
custom_graph best_ms=1.032352
scratch_clear_graph best_ms=0.001279
warm_minus_clear best_ms=1.031073
cold_minus_flush_clear best_ms=1.032096
```

Full cuDNN comparison harness, `iters=50`, `warmup=10`, `trials=3`:

```text
cudnn_split_device_ms=1.000082
custom_device_ms=1.033747
custom_scratch_clear_device_ms=0.001171
custom_minus_clear_device_ms=1.032577
cold_custom_minus_flush_clear_ms=1.034404
custom_minus_clear_vs_cudnn_split_speedup=0.968530
```

Attempted focused Nsight Compute command:

```bash
ncu --target-processes all --kernel-name-base demangled \
  --kernel-name regex:gemma4_ffn_decode_fused_bf16_kernel \
  --metrics gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum \
  --csv --log-file build/analysis/ffn_load_play/final_default_ncu.csv \
  build/experiments/gemma4_ffn_decode_load_bench 1 0 1
```

The `ncu` run was stopped after several minutes without returning useful
output. Follow-up profiling should use a narrower standalone harness or a lower
replay-overhead section set.

Conclusion:

- A serious `>5%` HBM/load-scheduling improvement was not found in this
  realistic sweep.
- The best safe source change is still useful: the default path is now correct
  and improves the corrected `256`-thread baseline by about `2.4-2.6%`.
- Explicit async preload of `x`, disabling hidden-pack swizzle, smaller/larger
  intermediate tiles, manual down-weight preload, and manual gate/up preload
  did not beat the `672`-thread one-pack-per-thread shape.
- The warm and cold times are nearly identical after subtracting flush/clear,
  so this kernel is not primarily winning from repeated-buffer L2 warmth.

## 2026-05-24 - FFN decode memory access and race audit

Goal:

- Check for obvious race conditions, shared-memory bank conflicts, and global
  memory coalescing issues in the current FFN decode kernel.
- Revisit the scratch reduction lock after the prior lock-wait optimization.

Guide and outside references:

```bash
python3 scripts/query.py \
  "CUDA memory model acquire release atomic threadfence synchronization between thread blocks global memory visibility" \
  --top-k 10
python3 scripts/query.py \
  "CUDA shared memory bank conflicts coalesced global memory access warp contiguous addresses" \
  --top-k 10
python3 scripts/query.py \
  "PTX red release gpu global atomic add memory order acquire release" \
  --top-k 8
exa-ai search \
  "CUDA GEMV kernel shared memory bank conflicts coalesced global memory loads LLM inference optimization" \
  --num-results 5 --summary --output-format toon
exa-ai search \
  "CUDA warp coalescing vectorized 128 bit loads GEMV optimization bank conflicts" \
  --num-results 5 --summary --output-format toon
```

Relevant guide notes:

- Coalesced global memory access is the right first-order check for HBM loads.
  The guide's examples show consecutive warp addresses as the desired pattern.
- Shared-memory bank conflicts depend on per-warp bank mapping; padding or
  swizzling is a standard fix when stride patterns collide.
- The local guide results for exact PTX release/acquire details were weak, so
  the scratch lock was kept conservative.

Changes:

- Restored per-thread acquire polling after thread 0 observes the scratch lock
  turn. This gives each thread an acquire load before it reads
  `scratch->accum`, while preserving the initial thread-0-only wait.

Static memory access audit:

- Dominant HBM streams are fully coalesced by address pattern:
  - `x` load into shared memory: one 16-byte bf16 pack per thread.
  - gate/up weights: fixed column, consecutive hidden packs across a warp.
  - down weights: fixed intermediate row, consecutive hidden packs across a
    warp.
  - residual, RMS gamma, residual output, normed output: consecutive bf16
    packs across a warp.
- With `672` threads there are exactly `21` full warps, so there is no partial
  final warp in the hidden-pack loops.
- The hidden-pack XOR swizzle permutes 16-byte chunks inside each 128-byte
  group. It preserves the same four aligned 128-byte segments per warp for
  HBM coalescing.
- For the shared `s_x` 128-bit vector access, each 8-lane 128-byte
  subtransaction has no duplicate bank for any 32-bit subword under the current
  swizzle.
- The one non-fully-coalesced global pattern is scratch accumulation:
  `add_accum_pack`/`store_accum_pack` issue two 16-byte float-vector operations
  per thread for 8 float columns. Each individual instruction is stride-32B
  across lanes, so it has about 50% segment utilization. Combined lo+hi covers
  the full contiguous 1024B warp span. This traffic is small compared with
  weight streaming.

SASS/resource check:

```bash
nvcc -std=c++17 -O3 -arch=sm_86 -lineinfo -Xptxas=-v -Isrc \
  -cubin src/gemma4_ffn_decode.cu \
  -o build/analysis/ffn_decode_audit.cubin
cuobjdump --dump-sass build/analysis/ffn_decode_audit.cubin \
  > build/analysis/ffn_decode_audit.sass
```

ptxas:

```text
Used 67 registers
Used 11512 bytes smem
Spill stores: 0
Spill loads: 0
```

SASS includes vectorized global/shared operations such as:

```text
LDG.E.128.CONSTANT
LDG.E.128.STRONG.GPU
STS.128
LDS.128
STG.E.128
RED.E.ADD.S32.STRONG.GPU
MEMBAR.SC.GPU
```

Runtime checks:

```bash
make test-ffn-decode
make ffn-decode-load-bench
GEMMA4_FFN_LOAD_BENCH_SEED=0x20260522 \
  build/experiments/gemma4_ffn_decode_load_bench 30 8 3
```

Results:

```text
ffn decode tests passed

custom_graph best_ms=1.040187
scratch_clear_graph best_ms=0.001248
warm_minus_clear best_ms=1.038939
cold_minus_flush_clear best_ms=1.040181
```

`compute-sanitizer`/`ncu` caveat:

- `compute-sanitizer` is installed, but in this Thunder environment it aborts
  in the CUDA interposition layer with:
  `Unimplemented CUDA export table function: Table=cupti_device_query`.
- Nsight Compute had previously failed to return useful output promptly in this
  environment. This audit therefore relies on source/SASS/static address
  analysis plus CUDA-event timing rather than sanitizer or NCU counters.

Conclusion:

- No obvious shared-memory race or divergent-barrier issue was found in the
  source audit.
- The scratch global-memory protocol is now conservative again: every thread
  executes an acquire lock load before reading `scratch->accum`.
- Dominant HBM weight/input/output streams are fully coalesced by static address
  analysis and compile to 128-bit global/shared operations.
- Scratch float accumulation is the main imperfect coalescing pattern, but its
  byte volume is much smaller than the streamed FFN weights.

## 2026-05-24 - FFN decode scratch coalescing and cold-load follow-up

Goal:

- Continue optimizing the fully cold FFN decode path without exceeding the
  `67` register count, increasing scratch storage, or making a weaker
  correctness tradeoff.
- Focus on the one imperfect coalescing pattern found in the prior audit:
  scratch accumulation.

Guide and outside references:

```bash
python3 scripts/query.py \
  "CUDA coalesced global memory vectorized stores structure of arrays memory coalescing" \
  --top-k 8
exa-ai search \
  "CUDA GEMV scratch accumulation coalesced float4 stores structure of arrays optimization" \
  --num-results 5 --summary --output-format toon
exa-ai search \
  "PTX release acquire global atomic red.release.gpu.global.add.s32 documentation" \
  --num-results 5 --summary --output-format toon
```

Relevant notes:

- CUDA guide query again emphasized maximizing used bytes per transferred
  segment for global-memory coalescing.
- Exa surfaced NVIDIA global-memory guidance and PTX ISA docs. The release-only
  lock variant compiled, but it was rejected because thread 0's release atomic
  would not order every other thread's prior scratch stores.

Implementation:

- Changed `Gemma4FfnDecodeScratch` from one linear `float accum[5376]` into two
  float4 planes:
  - `accum_lo[672][4]`
  - `accum_hi[672][4]`
- This keeps scratch float storage unchanged while making each lo/hi vector
  instruction contiguous across warp lanes.
- Changed FFN decode's default weight loads in this translation unit to
  streaming `.cs` so the enormous one-use FFN weights do not compete as
  aggressively for cache.
- Changed the default intermediate tile from `512` to `672`, reducing scratch
  reduction turns from `42` to `32`.

Correctness/resource checks:

```bash
make test-ffn-decode
nvcc -std=c++17 -O3 -arch=sm_86 -Xptxas=-v -Isrc \
  -c src/gemma4_ffn_decode.cu \
  -o build/ptx/ffn_continue/ffn_decode_final_pass.o
```

```text
ffn decode tests passed
Used 67 registers
Spill stores: 0
Spill loads: 0
Used 11512 bytes smem
```

Selected variants, focused load bench:

```text
variant                     regs  warm_minus_clear_ms  cold_minus_flush_clear_ms  decision
scratch_soa_tile512         67    1.032123             1.034466                  useful, not final
release_only_lock           67    1.028434             1.029839                  rejected, unsafe ordering
sync_trim                   67    1.032714             1.034198                  not retained
scratch_load_cg             67    1.033592             1.034893                  not retained
scratch_store_wb            67    1.034762             1.034850                  not retained
scratch_loadcg_storewb      67    1.033448             1.036307                  not retained
tile672_default_policy      67    1.023106             1.024330                  useful
tile672_ldg                 67    1.022832             1.024498                  not retained
tile672_cs                  67    1.021870             1.022774                  retained
tile896                    67    1.143920             1.081221                  not retained
tile672_act3                56    1.026600             1.024475                  not retained
tile672_act6                78    1.104712             1.105141                  rejected, spills/registers
tile672_act7                80    1.171859             1.176675                  rejected, spills/registers
t512_tile672_act3           64    1.166597             1.159491                  not retained
t384_tile672_act3           64    1.059312             1.041632                  not retained
t704_tile672                67    1.026203             1.028035                  not retained
t768_tile672                67    1.026997             1.028285                  not retained
t1024_tile672               63    1.027472             1.027472                  not retained
```

Final focused source timing:

```bash
make ffn-decode-load-bench
GEMMA4_FFN_LOAD_BENCH_SEED=0x20260522 \
  build/experiments/gemma4_ffn_decode_load_bench 30 8 3
```

```text
custom_graph best_ms=1.022462
scratch_clear_graph best_ms=0.001311
warm_minus_clear best_ms=1.021151
cold_minus_flush_clear best_ms=1.023045
```

Full cuDNN comparison harness:

```bash
make ffn-cudnn-bench
GEMMA4_FFN_CUDNN_BENCH_SEED=0x20260522 \
  build/experiments/gemma4_ffn_cudnn_bench 50 10 3
```

```text
cudnn_split_device_ms=1.000451
custom_device_ms=1.023697
custom_scratch_clear_device_ms=0.001218
custom_minus_clear_device_ms=1.022479
cold_custom_minus_flush_clear_ms=1.023745
custom_minus_clear_vs_cudnn_split_speedup=0.978456
```

Conclusion:

- The retained changes lower the audited cold path from `1.040181 ms` to
  `1.023745 ms`, about `1.6%` faster, while staying at `67` registers and
  keeping scratch storage size unchanged.
- The custom path is now about `2.2%` behind the warmed cuDNN split baseline in
  this harness.
- The tempting release-only lock increment was not retained because it would not
  safely order scratch stores from all threads in the CTA.

## 2026-05-25 - FFN decode HBM coalescing and lock-handoff audit

Question:

- Check whether the retained FFN decode path has obvious shared-memory bank
  conflicts, inter-block race hazards, fatal launch/resource issues, or
  uncoalesced HBM pulls.
- Re-check the benchmark accounting after changing the launcher to clear only
  `scratch->lock` instead of the whole scratch accumulator.

Guide and outside references:

```bash
python3 scripts/query.py \
  "CUDA shared memory bank conflicts 128-bit vector loads consecutive threads transaction 128 bytes" \
  --top-k 10
python3 scripts/query.py \
  "CUDA global memory coalescing 32-byte segments aligned sequential threads 128-bit loads" \
  --top-k 10
python3 scripts/query.py \
  "CUDA memory synchronization global acquire release __threadfence __syncthreads inter block communication" \
  --top-k 10
python3 scripts/query.py \
  "CUDA volatile global memory threadfence visibility partial sums atomic counter" \
  --top-k 8
exa-ai search \
  "CUDA GEMV memory coalescing LLM decode optimization persistent kernel HBM" \
  --num-results 5 --summary --output-format toon
```

Relevant notes:

- CUDA guide page 60/61 describes coalescing as using all bytes in the required
  32-byte global-memory segments; it does not require strictly increasing lane
  order if the warp touches the same segments.
- CUDA guide page 67/68 describes shared-memory banks as 32 banks with
  successive 32-bit words in successive banks; stride-one row access is
  conflict-free, while stride-32 column access is a 32-way conflict.
- CUDA guide page 534 warns that inter-block producer/consumer code needs a
  fence before signaling a counter/flag. It also notes that fences alone are not
  a complete visibility mechanism, so the lock handoff should remain
  conservative.
- Exa surfaced FlashDecoding++ and CUTLASS material. The relevant ideas for
  this kernel remain flat-GEMM/GEMV double buffering, persistent scheduling,
  coalesced global access, shared-memory staging, and cache-policy selection.

Implementation updates:

- Updated `gemma4_ffn_decode_fused_bf16` to clear only
  `Gemma4FfnDecodeScratch::lock` with `cudaMemsetAsync` before launch. Tile 0
  overwrites all accumulator packs, so clearing the whole accumulator was setup
  overhead only.
- Updated both FFN decode benchmark harnesses to time the same lock-only clear,
  avoiding an apples-to-oranges subtraction.
- Tightened `release_reduce_turn` from `red.relaxed.gpu.global.add.s32` to
  `red.release.gpu.global.add.s32` while keeping the all-thread
  `__syncthreads(); __threadfence(); __syncthreads();` sequence. This is a small
  performance cost, but it removes an unnecessary weak-signal concern in the
  inter-block scratch handoff.

Correctness/resource checks:

```bash
make test-ffn-decode
nvcc -std=c++17 -O3 -arch=sm_86 -Xptxas=-v -Isrc \
  -c src/gemma4_ffn_decode.cu \
  -o build/ptx/ffn_continue/gemma4_ffn_decode.o
```

```text
ffn decode tests passed
Used 67 registers
Spill stores: 0
Spill loads: 0
Used 11512 bytes smem
```

Focused load bench after lock-only clear accounting and release handoff:

```bash
make ffn-decode-load-bench
GEMMA4_FFN_LOAD_BENCH_SEED=0x20260522 \
  build/experiments/gemma4_ffn_decode_load_bench 30 8 3
```

```text
custom_graph best_ms=1.024590
scratch_clear_graph best_ms=0.001625
warm_minus_clear best_ms=1.022965
cold_custom best_ms=1.424242
cold_clear best_ms=0.399728
flush_only best_ms=0.398495
cold_minus_flush_clear best_ms=1.024514
```

Full cuDNN comparison after release handoff:

```bash
make ffn-cudnn-bench
GEMMA4_FFN_CUDNN_BENCH_SEED=0x20260522 \
  build/experiments/gemma4_ffn_cudnn_bench 50 10 3
```

```text
cudnn_split_device_ms=1.000134
custom_device_ms=1.027281
custom_scratch_clear_device_ms=0.001955
custom_minus_clear_device_ms=1.025326
cold_custom_minus_flush_ms=1.028109
cold_clear_minus_flush_ms=0.001313
cold_custom_minus_flush_clear_ms=1.026796
custom_minus_clear_vs_cudnn_split_speedup=0.975430
```

Memory-access audit:

- HBM loads/stores that walk hidden packs (`x`, residual, RMS weight,
  residual_out, normed_out, scratch lo/hi planes) use one 128-bit vector per
  lane over contiguous hidden-pack addresses. With 32 lanes, each warp covers a
  dense 512-byte span, so the 32-byte segment utilization is full when the base
  pointer is aligned.
- Gate/up and down weight loads use the hidden-pack XOR swizzle. Within each
  warp the lane order is permuted inside each eight-pack group, but the warp
  still touches the same dense 512-byte span per column/row. This should remain
  coalesced by the guide's 32-byte segment criterion.
- The current scratch layout is better than the old linear `accum[5376]` for
  coalescing: `accum_lo[pack][4]` and `accum_hi[pack][4]` make each 128-bit
  scratch vector instruction contiguous across lanes instead of striding by one
  full hidden pack.
- Shared `s_x` accesses are swizzled in 8 x 128-bit groups. A warp accesses one
  dense group of 16-byte chunks, permuted inside the group, so the static bank
  map does not show the stride-32 pattern that causes the classic 32-way bank
  conflict.
- The shared warp-sum arrays are written by lane 0 of each warp and read by the
  first `kFfnWarps` lanes as contiguous floats, so no obvious shared-bank
  conflict pattern is visible there either.

Tooling caveat:

- `compute-sanitizer --tool memcheck ./build/tests/test_ffn_decode` could not
  produce findings in Thunder. It timed out attaching and the target aborted
  through unsupported CUPTI `cupti_device_query`.
- `ncu --query-metrics --devices 0` reported `Invalid device ID 0` in this
  environment, so bank conflicts were not measured directly with Nsight Compute.
  The current conclusion is based on static address analysis, ptxas, SASS spot
  checks, and correctness/benchmark runs.

Conclusion:

- No fatal launch/resource issue found: the kernel validates resident-block
  support, clears the lock each launch, checks launches, passes the FFN decode
  tests, and still has zero spills at the 67-register target.
- No concrete race was found in the retained code. The lock handoff is now more
  conservative than before: all producer threads fence before the block releases
  a device-scope global lock, and consumers use acquire loads before reading
  scratch.
- The main HBM pulls are fully coalesced by the 32-byte segment criterion. The
  only caveat is that Nsight Compute could not be used here to verify actual
  bank-conflict counters.

## 2026-05-25 - FFN decode post-audit cold-load optimization sweep

Goal:

- Continue looking for cold-load improvements without exceeding `67`
  registers, adding meaningful scratch/storage, or weakening the conservative
  lock handoff.
- Use the current `672`-thread, `672`-intermediate-tile, scratch-SoA, `.cs`
  weight-load path as the source baseline.

Guide and outside references:

```bash
python3 scripts/query.py \
  "CUDA occupancy memory bandwidth more thread blocks per SM register pressure memory bound kernels" \
  --top-k 10
python3 scripts/query.py \
  "CUDA instruction level parallelism memory bound kernel unroll independent loads registers occupancy" \
  --top-k 10
python3 scripts/query.py \
  "cudaFuncSetCacheConfig cudaFuncCachePreferL1 shared memory carveout performance" \
  --top-k 8
python3 scripts/query.py \
  "CUDA GEMV memory bandwidth warp reduction split K occupancy optimization" \
  --top-k 8
exa-ai search \
  "CUDA GEMV LLM decode memory bandwidth optimization occupancy register pressure persistent threads" \
  --num-results 5 --summary --output-format toon
exa-ai search \
  "LLM decode GEMV CUDA optimization split K warp reduction memory bandwidth" \
  --num-results 5 --summary --output-format toon
```

Relevant notes:

- CUDA guide page 71 again points at occupancy as a latency-hiding tool, but
  also ties it directly to register/shared/thread limits. This made tile-count
  and launch-bound sweeps worth testing, but ptxas/timing remained the deciding
  evidence.
- CUDA guide page 526 describes launch bounds as a register-pressure control.
  For this kernel, requesting two resident CTAs per SM forced spills and was
  much slower.
- CUDA guide page 134 warns that `cudaFuncSetCacheConfig` creates hard
  shared/L1 requirements and can serialize launches when configurations change;
  I did not put it in the hot launch path.
- Exa surfaced K-split GEMV work, including a reported `GEMV/GEMV+Add`
  decode speedup. That is a plausible next larger design direction, but it is
  not a small patch here because gate/up partials would need an additional
  cross-CTA reduction before the down projection.

Commands:

```bash
make test-ffn-decode
nvcc -std=c++17 -O3 -arch=sm_86 -Xptxas=-v -Isrc \
  -c src/gemma4_ffn_decode.cu \
  -o build/ptx/ffn_final_check/gemma4_ffn_decode.o
make ffn-decode-load-bench
GEMMA4_FFN_LOAD_BENCH_SEED=0x20260522 \
  build/experiments/gemma4_ffn_decode_load_bench 30 8 3
GEMMA4_FFN_LOAD_BENCH_SEED=0x20260522 \
  build/experiments/gemma4_ffn_decode_load_bench 80 12 5
```

Source baseline recheck:

```text
ffn decode tests passed
Used 67 registers
Spill stores: 0
Spill loads: 0
Used 11512 bytes smem
```

```text
iters=30,warmup=8,trials=3
custom_graph best_ms=1.025887
scratch_clear_graph best_ms=0.002982
warm_minus_clear best_ms=1.022905
cold_minus_flush_clear best_ms=1.025677

iters=80,warmup=12,trials=5
custom_graph best_ms=1.024337
scratch_clear_graph best_ms=0.001441
warm_minus_clear best_ms=1.022896
cold_minus_flush_clear best_ms=1.025257
```

Rejected variants, focused load bench:

```text
variant                  regs  spills       warm_minus_clear_ms  cold_minus_flush_clear_ms  decision
tile448                  67    0/0          1.045233             1.046975                  reject
tile384                  67    0/0          1.056563             1.058801                  reject
tile336                  67    0/0          1.075157             1.077094                  reject
tile256                  67    0/0          1.094603             1.095850                  reject
weight_plain             68    0/0          1.025137             1.027255                  reject, over cap
weight_cg                67    0/0          1.025749             1.027136                  reject
weight_ldg               67    0/0          1.025751             1.026879                  reject
act3                     56    0/0          1.026138             1.027738                  reject
act2                     46    0/0          1.033358             1.035883                  reject
regcap64                 67    0/0          1.024177             1.026101                  no useful cap
regcap60                 67    0/0          1.024859             1.026315                  no useful cap
regcap56                 67    0/0          1.024621             1.026256                  no useful cap
one_pack_specialized     64    0/0          1.024240             1.026752                  reject, slower
specialized_act6         77    48B/48B      1.107958             1.110202                  reject, spills
specialized_act7         78    96B/96B      1.166462             1.170752                  reject, spills
min_blocks_2             40    228B/200B    1.643979             1.647881                  reject, spills
min_blocks_3             62    0/0          1.026586             1.027457                  reject, ignored/s lower
preload_down             67    0/0          1.023137             1.024421                  inconclusive short
preload_down_long        67    0/0          1.024983             1.026329                  reject
```

Conclusion:

- No new source change was retained from this sweep. The current source path is
  still the best measured safe path.
- More CTAs did not improve cold loads; added reduction/fence overhead and
  smaller per-CTA work dominated any extra SM spread.
- Forcing lower registers was not beneficial. Real launch-bound pressure
  spilled badly; the one-pack specialization saved registers but still slowed
  down the kernel.
- The next potentially meaningful direction is a larger K-split or multi-stage
  GEMV design, but that needs a deliberate scratch/reduction design rather than
  a small tweak to the current ordered-reduction kernel.

## 2026-05-25 - FFN decode shared activation barrier trim

Goal:

- Continue improving the fully cold FFN decode path while preserving the
  conservative lock handoff, staying at or below `67` registers, and avoiding
  meaningful persistent scratch growth.

Guide and outside references:

```bash
python3 scripts/query.py \
  "CUDA GEMV memory bandwidth warp reduction split K occupancy optimization" \
  --top-k 8
exa-ai search \
  "LLM decode GEMV CUDA optimization split K warp reduction memory bandwidth" \
  --num-results 5 --summary --output-format toon
```

Relevant notes:

- CUDA guide excerpts again point at occupancy/register/shared-memory limits,
  but the useful change here was synchronization pressure, not occupancy.
- Exa surfaced K-split GEMV work as the next larger direction. I did not
  implement that here because it would need a new cross-CTA partial reduction
  design for gate/up before the down projection.

Implementation:

- Added a tiny `__shared__ float s_act[kActTile]` array for the GeGLU
  activation scalars.
- Previously those scalars were stored in `s_matmul_warp_sums[0][t][0]`,
  forcing a full CTA tail barrier after every down-accumulation step before the
  next gate/up reduction could overwrite the warp-sum storage.
- With separate `s_act`, the next gate/up reduction can reuse
  `s_matmul_warp_sums` immediately. The next reduction's internal CTA barrier
  occurs before `s_act` is rewritten, so the old loop-tail barrier is not
  needed.
- Shared memory grows by only `kActTile * sizeof(float) = 16` bytes for the
  default path.

Correctness/resource checks:

```bash
make test-ffn-decode
nvcc -std=c++17 -O3 -arch=sm_86 -Xptxas=-v -Isrc \
  -c src/gemma4_ffn_decode.cu \
  -o build/ptx/ffn_final_sact/gemma4_ffn_decode.o
```

```text
ffn decode tests passed
Used 67 registers
Spill stores: 0
Spill loads: 0
Used 11528 bytes smem
```

Focused load bench:

```bash
make ffn-decode-load-bench
GEMMA4_FFN_LOAD_BENCH_SEED=0x20260522 \
  build/experiments/gemma4_ffn_decode_load_bench 80 12 5
```

```text
prior source baseline:
custom_graph best_ms=1.024337
scratch_clear_graph best_ms=0.001441
warm_minus_clear best_ms=1.022896
cold_minus_flush_clear best_ms=1.025257

shared-activation final source:
custom_graph best_ms=1.024175
scratch_clear_graph best_ms=0.001275
warm_minus_clear best_ms=1.022900
cold_minus_flush_clear best_ms=1.024132
```

Full cuDNN comparison harness:

```bash
make ffn-cudnn-bench
GEMMA4_FFN_CUDNN_BENCH_SEED=0x20260522 \
  build/experiments/gemma4_ffn_cudnn_bench 50 10 3
```

```text
shared-activation source:
cudnn_split_device_ms=0.999687
custom_device_ms=1.024297
custom_scratch_clear_device_ms=0.001544
custom_minus_clear_device_ms=1.022753
cold_custom_minus_flush_ms=1.026110
cold_clear_minus_flush_ms=0.001363
cold_custom_minus_flush_clear_ms=1.024747
custom_minus_clear_vs_cudnn_split_speedup=0.977447
```

Follow-up variants after the `s_act` change:

```text
variant                 regs  smem    warm_minus_clear_ms  cold_minus_flush_clear_ms  decision
preload_down_short      67    11528   1.022813             1.024174                  reject/no clear win
gate_up_preload_short   66    11528   1.021861             1.022943                  promising short
gate_up_preload_long    66    11528   1.022636             1.023892                  not clearly better
gate_up_preload_full    66    11528   1.023057             1.024124                  mixed
gate_up_source_full     66    11528   1.023816             1.025231                  reject
```

Conclusion:

- Retained the separate shared activation storage and removed the old
  down-loop tail barrier.
- Did not retain gate/up register preloading as a default. It lowered ptxas to
  `66` registers and had one promising short run, but current-source full
  timing and focused long timing were not consistently better than the plain
  `s_act` path.
- The retained change is small, keeps the kernel at `67` registers with zero
  spills, increases shared memory by only `16` bytes, and improves the focused
  fully cold long metric from `1.025257 ms` to `1.024132 ms`.

## 2026-05-25 - FFN decode coalescing and race audit

Goal:

- Audit the retained FFN decode HBM load layout, shared-memory access pattern,
  ordered scratch reduction, and obvious failure modes after the shared
  activation change.

Guide and outside references:

```bash
python3 scripts/query.py \
  "CUDA one thread computes activation then shared memory versus warp parallel tanh register pressure memory bound kernel" \
  --top-k 5
exa-ai search \
  "CUDA GEMV optimization coalesced loads register pressure FFN inference" \
  --num-results 5 --summary --output-format toon
```

Also checked NVIDIA's CUDA Programming Guide, the NVIDIA fully-connected layer
optimization guide, CUTLASS efficient GEMM notes, and FlashDecoding++ for
decode-phase GEMV/flat-GEMM optimization context.

Audit notes:

- HBM loads in the hot kernel use 128-bit bf16/float packs. For `x`,
  residual, RMS weight, scratch accum, and output stores, consecutive warp
  lanes own consecutive hidden packs, so each warp request covers contiguous
  128-byte segments with no stride waste.
- Gate/up and down weights are pre-swizzled in 128-bit hidden packs. The
  current `shared_x_chunk_index()` only permutes the eight 16-byte packs inside
  each 128-byte row, so a warp still touches the same contiguous cache-line
  segments even though lane order is permuted inside the segment.
- Shared `s_x` accesses use one 16-byte pack per thread. The 8-pack row swizzle
  keeps each 128-byte row covered exactly once by an eight-lane group. `s_act`
  is written by one thread and read uniformly by the CTA; uniform same-word
  shared reads are broadcast rather than a multi-bank conflict.
- The cross-CTA scratch reduction still uses the conservative lock handoff:
  each tile waits for its turn, reads/writes scratch, runs CTA-wide fences, and
  releases with a global release reduction. The launcher rejects devices that
  cannot keep all reduction CTAs resident, avoiding spin-wait deadlock from
  under-residency.

Implementation retained:

- Tightened the scratch pointer validation from 16-byte alignment to 128-byte
  alignment to match `alignas(128) Gemma4FfnDecodeScratch`.
- Added a unit-test check that a 16-byte-but-not-128-byte scratch pointer is
  rejected before launch.
- Kept the thread0 `s_act` writer. It avoids the previous warp-shuffle
  activation broadcast, has a single writer for the activation scalars, and was
  essentially neutral to slightly better on fully cold timing.
- Kept the `GEMMA4_MATMUL_DEVICE_PRELOAD_PAIR_COLS` compile-time toggle
  defaulted off. The preload variant reduced registers but did not beat the
  default path consistently.

Commands:

```bash
make test-ffn-decode
nvcc -std=c++17 -O3 -arch=sm_86 -Xptxas=-v -Isrc \
  -c src/gemma4_ffn_decode.cu \
  -o build/ptx/ffn_final/gemma4_ffn_decode.o
cuobjdump --dump-sass build/ptx/ffn_final/gemma4_ffn_decode.o | \
  grep -E "LDG|LDGSTS|STG|LDS|STS|RED" | head -80
compute-sanitizer --tool racecheck ./build/tests/test_ffn_decode
make ffn-decode-load-bench
GEMMA4_FFN_LOAD_BENCH_SEED=0x20260522 \
  build/experiments/gemma4_ffn_decode_load_bench 30 8 3
GEMMA4_FFN_LOAD_BENCH_SEED=0x20260522 \
  build/experiments/gemma4_ffn_decode_load_bench 80 12 5
make ffn-cudnn-bench
GEMMA4_FFN_BENCH_SEED=0x20260522 \
  build/experiments/gemma4_ffn_cudnn_bench 50 10 3
```

Correctness/resource checks:

```text
ffn decode tests passed
Used 67 registers
Spill stores: 0
Spill loads: 0
Used 11528 bytes smem
SASS includes LDG.E.128 / LDG.E.EF.128, STG.E.128, LDS.128, STS.128
```

`compute-sanitizer --tool racecheck` was not usable on this Thunder machine:
the target aborted in the platform shim with `Unimplemented CUDA export table
function: Table=cupti_device_query, Index=7`. Treat the race result as
unverified by sanitizer; the regular CUDA test passed and the synchronization
audit above did not find a race.

Thread0 activation writer measurements:

```text
focused, iters=30,warmup=8,trials=3:
custom_graph best_ms=1.022806
scratch_clear_graph best_ms=0.001357
warm_minus_clear best_ms=1.021450
cold_minus_flush_clear best_ms=1.023595

focused, iters=80,warmup=12,trials=5:
custom_graph best_ms=1.023252
scratch_clear_graph best_ms=0.001424
warm_minus_clear best_ms=1.021829
cold_minus_flush_clear best_ms=1.024028

full cuDNN harness, iters=50,warmup=10,trials=3:
cudnn_split_device_ms=0.999998
custom_device_ms=1.024863
custom_scratch_clear_device_ms=0.001356
custom_minus_clear_device_ms=1.023507
cold_custom_minus_flush_clear_ms=1.024686
custom_minus_clear_vs_cudnn_split_speedup=0.977031

final source after alignment guard, focused iters=30,warmup=8,trials=3:
custom_graph best_ms=1.023542
scratch_clear_graph best_ms=0.001446
warm_minus_clear best_ms=1.022096
cold_minus_flush_clear best_ms=1.024101
```

Rejected variants:

```text
variant                         regs  cold_minus_flush_clear_ms  decision
one_pack_helper                 58    1.024144 focused long      reject; full cold 1.025253
one_pack_helper_gate_preload    57    1.024121 focused long      reject; no clear win
gate_up_preload_default_source  66    1.025231 full              reject
```

Conclusion:

- No bank-conflict or race-condition bug was found in the source audit.
- The HBM pull pattern is fully coalesced by cache-line segment for the current
  128-bit pack layout; Nsight Compute would still be needed for hardware
  counter proof, but `ncu`/sanitizer tooling remains blocked in this runtime.
- The only correctness hardening retained from this audit is the 128-byte
  scratch alignment guard and test.

## 2026-05-25 - FFN decode 896-tile cold-load sweep

Goal:

- Continue reducing the fully cold FFN decode time without exceeding `67`
  registers, adding meaningful memory, or weakening the conservative scratch
  reduction synchronization.

Guide and outside references:

```bash
python3 scripts/query.py \
  "CUDA GEMV one thread per output hidden pack specialize compile time remove loop integer overhead register pressure" \
  --top-k 8
python3 scripts/query.py \
  "CUDA memory bound kernel reduce instruction overhead loop hoisting integer address calculation register pressure" \
  --top-k 8
exa-ai search \
  "CUDA GEMV LLM decode one thread per output vector coalesced BF16 optimization" \
  --num-results 5 --summary --output-format toon
```

Relevant notes:

- CUDA guide page 519 again makes the useful tradeoff explicit: compiler
  common-subexpression and cached-load optimizations can reduce work but raise
  register pressure. This showed up directly: the gate/up pointer-base helper
  and one-pack kernel specialization both increased ptxas register count.
- CUDA guide pages 28 and 119 tie occupancy to registers, shared memory, and
  resident warps/blocks. The larger intermediate tile reduces CTA count and
  ordered scratch-reduction work while keeping one 672-thread CTA resident per
  scheduled SM.
- CUDA guide page 67 keeps the coalescing requirement front and center. The
  retained tile-size change does not alter the 128-bit hidden-pack load layout.
- Exa surfaced current GEMV/decode optimization work in `mistral.rs`,
  `llama.cpp`, and FP4 GEMV writeups. The useful local takeaway was to test
  address-arithmetic and tile-count changes before attempting a larger split-K
  redesign.

Implementation retained:

- Changed the default `GEMMA4_FFN_DECODE_INTERMEDIATE_TILE` from `672` to
  `896`. This reduces the ordered intermediate-tile reduction from `32` CTAs
  to `24` CTAs.
- Kept `GEMMA4_FFN_DECODE_THREADS=672`, `ACT_TILE=4`, `.cs` weight loads,
  the SoA scratch layout, lock-only scratch clear, and conservative release
  handoff unchanged.
- In down accumulation, compute `down_row0` once per activation group and use
  unrolled `t * GEMMA4_HIDDEN_SIZE` offsets for the four down-weight rows.
  This does not change the HBM access pattern or scratch footprint.

Commands:

```bash
make test-ffn-decode
nvcc -std=c++17 -O3 -arch=sm_86 -Xptxas=-v -Isrc \
  -c src/gemma4_ffn_decode.cu \
  -o build/ptx/ffn_tile896_default/gemma4_ffn_decode.o
make ffn-decode-load-bench
GEMMA4_FFN_LOAD_BENCH_SEED=0x20260522 \
  build/experiments/gemma4_ffn_decode_load_bench 80 12 5
make ffn-cudnn-bench
GEMMA4_FFN_BENCH_SEED=0x20260522 \
  build/experiments/gemma4_ffn_cudnn_bench 50 10 3
```

Correctness/resource checks:

```text
ffn decode tests passed
Used 67 registers
Spill stores: 0
Spill loads: 0
Used 11528 bytes smem
```

Focused load-bench results:

```text
previous retained source, focused iters=80,warmup=12,trials=5:
custom_graph best_ms=1.023252
scratch_clear_graph best_ms=0.001424
warm_minus_clear best_ms=1.021829
cold_minus_flush_clear best_ms=1.024028

down-row base only, focused iters=80,warmup=12,trials=5:
custom_graph best_ms=1.022539
scratch_clear_graph best_ms=0.001215
warm_minus_clear best_ms=1.021324
cold_minus_flush_clear best_ms=1.023419

tile896 default source, focused iters=80,warmup=12,trials=5:
custom_graph best_ms=1.016745
scratch_clear_graph best_ms=0.001229
warm_minus_clear best_ms=1.015516
cold_minus_flush_clear best_ms=1.016316
```

Full cuDNN comparison harness:

```text
tile896 default source, iters=50,warmup=10,trials=3:
cudnn_split_device_ms=1.000357
custom_device_ms=1.022643
custom_scratch_clear_device_ms=0.001528
custom_minus_clear_device_ms=1.021115
cold_custom_minus_flush_ms=1.022621
cold_clear_minus_flush_ms=0.001129
cold_custom_minus_flush_clear_ms=1.021492
custom_minus_clear_vs_cudnn_split_speedup=0.979671
max_abs_vs_split=0
```

Rejected variants:

```text
variant                        regs/spills        cold_minus_flush_clear_ms  decision
one_pack_kernel_specialized    68, 0/0            not timed                 reject; over cap
one_pack_maxrregcount67        72, 0/0            not timed                 reject; cap made worse
gate_up_pointer_base           68, 0/0            not timed                 reject; over cap
tile768                        67, 0/0            1.021229 long             reject; slower than tile896
tile1024                       67, 0/0            1.021745 short            reject; slower than tile896
tile1344                       67, 0/0            1.226100 short            reject; too few CTAs
gate_up_preload_tile896        66, 0/0            1.017170 short            reject; slower
down_preload_tile896           67, 0/0            1.016330 long             reject; tie/slightly slower
act8_tile896                   80, 140B/140B      1.160775 short            reject; spills and slow
```

Conclusion:

- Retained `INTERMEDIATE_TILE=896` and the down-row base address cleanup.
- The retained source stays at `67` registers, has zero spills, does not grow
  scratch, and keeps the same coalesced 128-bit HBM load layout.
- Fully cold focused timing improved from the prior retained long result
  `1.024028 ms` to `1.016316 ms`.

## 2026-05-25 - FFN decode direct-x and one-pack helper

Goal:

- Keep pushing the fully cold FFN decode path without exceeding the register
  cap, growing scratch, or changing the lock/reduction safety model.

Guide and outside references:

```bash
python3 scripts/query.py \
  "CUDA warp shuffle sync mask active lanes divergent branch requirements" \
  --top-k 8
python3 scripts/query.py \
  "CUDA tanhf latency warp parallel compute independent scalar operations register pressure" \
  --top-k 8
exa-ai search \
  "CUDA warp shuffle active mask __shfl_sync optimization GEMV activation tanh" \
  --num-results 5 --summary --output-format toon
```

Relevant notes:

- CUDA guide pages 550-551 describe the `__shfl_sync` mask contract: only
  non-exited participating lanes named in the mask can be safely used. The
  narrow-lane activation writer was correct under that rule, but it timed
  slower than the single-thread writer.
- CUDA guide page 117 notes SIMT divergence masking. That makes tiny-lane
  activation parallelism a measurement question rather than an obvious win.
- CUDA guide pages 28, 119, and 519 were again the guardrails: avoid trading
  instruction savings for register blowups unless timing proves it.
- Exa surfaced GEMV/decode kernels and warp-shuffle discussions; the useful
  local tests were the prologue specialization and helper specialization below.

Implementation retained:

- Specialized the `x` shared-memory preload for the default one-thread-per-pack
  geometry (`672` threads, `672` hidden packs). The generic path remains for
  non-default compile-time thread sweeps.
- Split the shared-`x` gate/up dot body into a per-pack helper and let the
  caller bypass the `pack_idx += Threads` loop when `Threads == packs_per_col`.
  This keeps the helper reusable and drops the hot kernel from `67` to `57`
  registers.
- Kept `.cs` weight loads, `THREADS=672`, `INTERMEDIATE_TILE=896`,
  `ACT_TILE=4`, `PRELOAD_DOWN=0`, and the conservative ordered scratch
  handoff.

Commands:

```bash
make test-ffn-decode
nvcc -std=c++17 -O3 -arch=sm_86 -Xptxas=-v -Isrc \
  -c src/gemma4_ffn_decode.cu \
  -o build/ptx/ffn_final_helper_directx/gemma4_ffn_decode.o
make ffn-decode-load-bench
GEMMA4_FFN_LOAD_BENCH_SEED=0x20260522 \
  build/experiments/gemma4_ffn_decode_load_bench 80 12 5
make ffn-cudnn-bench
GEMMA4_FFN_BENCH_SEED=0x20260522 \
  build/experiments/gemma4_ffn_cudnn_bench 50 10 3
```

Correctness/resource checks:

```text
ffn decode tests passed
Used 57 registers
Spill stores: 0
Spill loads: 0
Used 11528 bytes smem
```

Focused load-bench results:

```text
tile896 previous retained source, focused iters=80,warmup=12,trials=5:
custom_graph best_ms=1.016745
scratch_clear_graph best_ms=0.001229
warm_minus_clear best_ms=1.015516
cold_minus_flush_clear best_ms=1.016316

direct-x only, focused iters=80,warmup=12,trials=5:
custom_graph best_ms=1.016912
scratch_clear_graph best_ms=0.001206
warm_minus_clear best_ms=1.015706
cold_minus_flush_clear best_ms=1.015453

direct-x plus one-pack helper, focused iters=80,warmup=12,trials=5:
custom_graph best_ms=1.015880
scratch_clear_graph best_ms=0.001193
warm_minus_clear best_ms=1.014687
cold_minus_flush_clear best_ms=1.014572
```

Full cuDNN comparison harness:

```text
direct-x plus one-pack helper, iters=50,warmup=10,trials=3:
cudnn_split_device_ms=0.999928
custom_device_ms=1.021978
custom_scratch_clear_device_ms=0.001324
custom_minus_clear_device_ms=1.020653
cold_custom_minus_flush_ms=1.022648
cold_clear_minus_flush_ms=0.001460
cold_custom_minus_flush_clear_ms=1.021188
custom_minus_clear_vs_cudnn_split_speedup=0.979694
max_abs_vs_split=0
```

Rejected variants:

```text
variant                    regs/spills      cold_minus_flush_clear_ms  decision
narrow-mask warp s_act     67, 0/0          1.017355 short            reject; slower
helper_gate_preload        57, 0/0          1.015617 long             reject; slower
helper_down_preload        57, 0/0          1.015224 short            reject; no clear win
threads640                 76, 0/0          1.390937 short            reject; over cap/slow
threads704                 67, 0/0          1.015276 short            reject; slower than default
weight_policy_cg           57, 0/0          1.023766 short            reject; slower
weight_policy_ldg          57, 0/0          1.021502 short            reject; slower
```

Note:

- An initial parallel `.cg`/`ldg` policy run overlapped two GPU benchmarks and
  was discarded. The logged cache-policy numbers above are sequential reruns.

Conclusion:

- Retained direct `x` preload specialization and the reusable one-pack
  shared-`x` dot helper.
- The retained path keeps the same scratch footprint and HBM layout, drops the
  hot kernel to `57` registers, and improves the focused fully cold long metric
  from `1.016316 ms` to `1.014572 ms`.

## 2026-05-25 - FFN decode post-helper rejection sweep

Goal:

- Use the new `57`-register headroom to retest variants that were previously
  blocked by register pressure, while keeping scratch, HBM layout, and the
  conservative ordered reduction unchanged.

Guide and outside references:

```bash
python3 scripts/query.py \
  "CUDA GEMV columns per block tradeoff register pressure reductions shared memory fewer synchronizations" \
  --top-k 8
python3 scripts/query.py \
  "CUDA occupancy register pressure memory bandwidth instruction level parallelism fewer barriers" \
  --top-k 8
exa-ai search \
  "CUDA GEMV columns per thread register pressure reduction barrier optimization" \
  --num-results 5 --summary --output-format toon
```

Relevant notes:

- CUDA guide page 71 describes occupancy as a latency-hiding tool, but bounded
  by register/shared/thread resources. The `ACT_TILE` variants still hit the
  register wall even after the helper lowered the default path.
- CUDA guide page 526 keeps launch bounds/register pressure in play; I did not
  retain a min-blocks experiment because the compile-time define was not wired
  into source and the produced binary was just the default launch-bounds path.
- CUDA guide page 67 and the Exa GEMV references reinforced that coalescing and
  reduction shape remain the important levers. The tested variants below did
  not improve the measured cold path.

Commands:

```bash
make test-ffn-decode
nvcc -std=c++17 -O3 -arch=sm_86 -Xptxas=-v -Isrc \
  -c src/gemma4_ffn_decode.cu \
  -o build/ptx/ffn_final_recheck/gemma4_ffn_decode.o
```

Final source recheck:

```text
ffn decode tests passed
Used 57 registers
Spill stores: 0
Spill loads: 0
Used 11528 bytes smem
```

Rejected variants:

```text
variant                    regs/spills  smem       cold_minus_flush_clear_ms  decision
act7                       80, 0/0      12044 B    not timed                 reject; over cap
act8                       80, 0/0      12216 B    not timed                 reject; over cap
act14                      80, 0/0      13248 B    not timed                 reject; over cap
tile1024                   57, 0/0      11528 B    1.018123 short            reject; slower
tile768                    57, 0/0      11528 B    1.019013 short            reject; slower
preload_gate_and_down      57, 0/0      11528 B    1.015743 short            reject; slower
one_pack_kernel_retry      60, 0/0      11528 B    1.015970 short            reject; slower
hoist_swizzled_hidden_col  57, 0/0      11528 B    1.016038 short            reject; slower
```

Conclusion:

- No new source optimization was retained from this sweep.
- The current best source remains the direct-`x` preload plus one-pack
  shared-`x` dot helper path from the previous entry: `57` registers, zero
  spills, same scratch size, and focused fully cold long metric `1.014572 ms`.

## 2026-05-25 - FFN decode extended HBM-load rejection sweep

Goal:

- Recheck bank-conflict, race, and HBM coalescing concerns after the direct-`x`
  and one-pack helper changes.
- Use the remaining `57`-register default path to try a few more low-growth
  load-shaping variants before declaring the current source best-known.

Guide and outside references:

```bash
python3 scripts/query.py \
  "CUDA shared memory bank conflicts coalesced global memory 128-byte warp access optimization" \
  --top-k 8
python3 scripts/query.py \
  "CUDA global memory visibility between blocks acquire release atomic threadfence synchronization" \
  --top-k 8
python3 scripts/query.py \
  "What does __threadfence guarantee global memory writes visible other thread blocks" \
  --top-k 8
exa-ai search \
  "CUDA GEMV HBM coalescing shared memory bank conflicts optimization register pressure" \
  --num-results 5 --summary --output-format toon
```

Useful reference points:

- CUDA guide pages 60 and 67: global-memory coalescing and shared-memory bank
  conflicts are separate checks. The current HBM load streams remain contiguous
  16-byte pack streams across each warp.
- CUDA guide pages 532-533: `__threadfence()` orders the releasing block's
  prior global writes before the lock handoff. The retained code keeps all
  threads synchronized around the fence and keeps per-thread acquire polling
  before reading scratch.
- NVIDIA CUDA Best Practices 13.2 says cc 6.0+ coalescing maps a warp request
  to the required 32-byte transactions, and shared-memory bank conflicts
  serialize same-bank requests. The static address pattern was checked against
  those rules; final proof still needs Nsight Compute conflict/transaction
  metrics.
- Exa surfaced NVIDIA's current global-memory-access article, CUDA Best
  Practices, and KBLAS GEMV work. The practical advice matched the local tests:
  keep warp addresses contiguous, avoid stride-heavy HBM streams, and profile
  with NCU before believing a micro-optimization.

Rejected variants:

```text
variant                    regs/spills  smem       cold_minus_flush_clear_ms  decision
act6_tile672               72, 0/0      11872 B    not timed                 reject; over cap
act6_tile768               72, 0/0      11872 B    not timed                 reject; over cap
act7_tile896               80, 0/0      12044 B    not timed                 reject; over cap
act8_tile896               80, 0/0      12216 B    not timed                 reject; over cap
act14_tile896              80, 0/0      13248 B    not timed                 reject; over cap
tile512                    57, 0/0      11528 B    1.036881 short            reject; slower
tile768                    57, 0/0      11528 B    1.019013 short            reject; slower
tile1024                   57, 0/0      11528 B    1.018123 short            reject; slower
async_x                    57, 0/0      11528 B    1.016876 short            reject; slower
no_swizzle                 62, 0/0      11528 B    1.014982 long             reject; slower
preload_gate_and_down      57, 0/0      11528 B    1.015743 short            reject; slower
one_pack_kernel_retry      60, 0/0      11528 B    1.015970 short            reject; slower
hoist_swizzled_hidden_col  57, 0/0      11528 B    1.016038 short            reject; slower
global_x_gemv              57, 0/0        776 B    1.017591 short            reject; slower
```

Attempted NCU metric checks:

```bash
timeout 120s ncu --target-processes all --kernel-name-base demangled \
  --kernel-name regex:gemma4_ffn_decode_fused_bf16_kernel \
  --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,\
l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
dram__sectors_read.sum,dram__sectors_write.sum \
  --csv build/experiments/gemma4_ffn_decode_load_bench 1 0 1
timeout 90s ncu --target-processes all --launch-count 1 \
  --print-summary per-kernel ./build/tests/test_ffn_decode
```

Both attempts timed out with `==WARNING== No kernels were profiled.` Treat the
bank-conflict and transaction conclusions here as a static address-pattern
audit until a narrower standalone NCU harness returns metrics.

Notes:

- The `-DGEMMA4_FFN_DECODE_MIN_BLOCKS_PER_SM=2` compile was discarded because
  that define is not wired into the source, so it was not a real experiment.
- The `global_x_gemv` branch cut shared memory sharply but reloaded `x` from
  global memory in the gate/up loop and lost time; its source code was removed.
- No new source optimization was retained. Current best remains
  direct-`x` preload plus one-pack shared-`x` helper: `57` registers, zero
  spills, `11528 B` shared memory, focused fully cold long metric
  `1.014572 ms`.

## 2026-05-25 - FFN decode two-kernel atomic reduction

Goal:

- Stop only tuning small cache/thread knobs and revisit the larger reduction
  shape. The old kernel computed one partial hidden vector per intermediate
  tile, then serialized those vectors through a global lock before the final
  residual/RMSNorm pass.

Guide and outside references:

```bash
python3 scripts/query.py \
  "CUDA reduce partial sums across thread blocks global memory second kernel versus atomics performance synchronization" \
  --top-k 10
python3 scripts/query.py \
  "CUDA cooperative groups grid synchronization resident blocks inter block reduction single kernel" \
  --top-k 10
exa-ai search \
  "CUDA GEMV fusion two stage reduction intermediate partials LLM FFN decode optimization" \
  --num-results 5 --summary --output-format toon
exa-ai search \
  "single token FFN decode CUDA GEMV optimization down projection fused activation" \
  --num-results 5 --summary --output-format toon
```

Relevant notes:

- CUDA guide page 70 warns that atomics should be used sparingly because they
  synchronize memory updates. Here the contended atomic traffic is only the
  final `5376`-float scratch vector, while the dominant HBM work remains
  streaming gate/up/down weights.
- CUDA guide pages 238, 241, and 627 cover cooperative groups/grid sync. A
  cooperative single-kernel version compiled, but this Thunder prototyping
  instance rejects `cuLaunchCooperativeKernel`, so it could not be measured
  here.
- Exa surfaced llama.cpp GEMV fusion work, FlashDecoding++/full-block decode
  work, and CODA-style epilogue fusion. The applicable idea for this kernel was
  to remove the serialized inter-CTA lock and let a later stage consume global
  partials.

Retained implementation:

- Replaced the lock-ordered in-kernel reduction with:
  - `gemma4_ffn_decode_accumulate_bf16_kernel`: each intermediate-tile CTA
    computes its hidden-vector partial and atomically adds it into the existing
    scratch accumulator.
  - `gemma4_ffn_decode_finalize_bf16_kernel`: one CTA reads the accumulated
    scratch vector, adds residual, computes RMSNorm scale, and writes normed
    output.
- Removed the scratch `lock` and padding fields. Scratch is now just the SoA
  float accumulator and remains `alignas(128)`.
- Removed the old resident-grid safety check; it was only needed to avoid
  deadlock in the serialized-lock kernel. The two-kernel atomic path is safe
  with multiple waves because the kernel boundary is the inter-block
  synchronization point.
- Updated focused/full FFN benchmarks so `custom_scratch_clear` measures the
  same full-scratch clear that the new path uses.
- Retuned `INTERMEDIATE_TILE` after the reduction change. The old `896` tile
  used only `24` CTAs. The retained `168` tile uses `128` CTAs, improving SM
  coverage enough to beat the extra atomic updates.

Rejected/diagnostic variants:

```text
variant                    regs/spills  smem       cold_minus_flush_clear_ms  decision
coop_atomic_grid_sync       57, 0/0      11528 B    not runnable              rejected here; cuLaunchCooperativeKernel unsupported
atomic_tile896              64, 0/0      11440 B    0.987878 long             good, but tile retune wins
atomic_tile1024             64, 0/0      11440 B    0.998512 short            reject; slower
atomic_tile768              64, 0/0      11440 B    0.985942 short            reject; slower
atomic_tile672              64, 0/0      11440 B    0.983177 long             reject; slower than 168
atomic_tile512              64, 0/0      11440 B    0.987258 short            reject; slower
atomic_tile448              64, 0/0      11440 B    0.985877 short            reject; slower
atomic_tile384              64, 0/0      11440 B    0.985117 short            reject; slower
atomic_tile336              64, 0/0      11440 B    0.982257 long             reject; near tie/slower
atomic_tile256              64, 0/0      11440 B    0.984130 short            reject; slower
atomic_tile224              64, 0/0      11440 B    1.059134 short            reject; much slower
atomic_tile192              64, 0/0      11440 B    0.984500 short            reject; slower
atomic_threads704           64, 0/0      11472 B    0.990002 short            reject; slower
atomic_threads736           64, 0/0      11504 B    0.989048 short            reject; slower
atomic_threads800           64, 0/0      11568 B    0.990722 short            reject; slower
```

Final validation:

```bash
make test-ffn-decode
nvcc -std=c++17 -O3 -arch=sm_86 -Xptxas=-v -Isrc \
  -c src/gemma4_ffn_decode.cu \
  -o build/ptx/ffn_atomic_tile168_final/gemma4_ffn_decode.o
make ffn-decode-load-bench
GEMMA4_FFN_LOAD_BENCH_SEED=0x20260522 \
  build/experiments/gemma4_ffn_decode_load_bench 80 12 5
make ffn-cudnn-bench
GEMMA4_FFN_CUDNN_BENCH_SEED=0x20260522 \
  build/experiments/gemma4_ffn_cudnn_bench 50 10 3
```

Additional diagnostic:

```bash
compute-sanitizer --tool memcheck --error-exitcode 99 \
  build/tests/test_ffn_decode
```

This could not run on the Thunder instance. The sanitizer path aborted through an
unsupported CUPTI/device-query export (`cupti_device_query`), then left the target
process waiting under the sanitizer launcher. The stuck sanitizer process was
terminated manually. Treat this as an unavailable tool result, not as either a
clean sanitizer pass or a detected kernel fault.

Final resource check:

```text
ffn decode tests passed
accumulate kernel: 64 registers, 0 spills, 11440 B smem
finalize kernel:   26 registers, 0 spills,    88 B smem
```

Focused fully cold load bench, `iters=80,warmup=12,trials=5`:

```text
custom_graph best_ms=0.980793
scratch_clear_graph best_ms=0.001108
warm_minus_clear best_ms=0.979685
cold_custom best_ms=1.381653
cold_clear best_ms=0.399534
flush_only best_ms=0.397676
cold_minus_flush_clear best_ms=0.982118
```

Full cuDNN comparison harness, `iters=50,warmup=10,trials=3`:

```text
cudnn_split_device_ms=0.998766
custom_device_ms=0.980579
custom_scratch_clear_device_ms=0.001130
custom_minus_clear_device_ms=0.979448
cold_custom_minus_flush_clear_ms=0.981459
custom_vs_cudnn_split_speedup=1.018548
custom_minus_clear_vs_cudnn_split_speedup=1.019723
max_abs_vs_split=0
```

Conclusion:

- Retained the two-kernel atomic reduction and `INTERMEDIATE_TILE=168`.
- Fully cold focused timing improved from the previous retained `1.014572 ms`
  to `0.982118 ms` (`~3.2%`).
- The full harness now puts the custom path ahead of the cuDNN split baseline
  on the graph/device metric while preserving correctness (`max_abs_vs_split=0`).

## 2026-05-25 - FFN decode component-major scratch and direct x loads

Goal:

- Continue looking for larger fully-cold FFN decode wins after the two-kernel
  atomic reduction. Keep the register count below `67`, avoid extra persistent
  memory, and remove dead experimental code where possible.

Guide and outside references:

```bash
python3 scripts/query.py \
  "CUDA global memory atomic add performance contention reduction across thread blocks second kernel memory coalescing" \
  --top-k 10
python3 scripts/query.py \
  "CUDA global memory coalescing 32 byte transactions sequential 16 byte loads warp" \
  --top-k 8
python3 scripts/query.py \
  "CUDA shared memory bank conflicts 128 bit access swizzle warp" \
  --top-k 8
exa-ai search \
  "CUDA single token FFN decode GEMV atomic reduction optimization LLM inference" \
  --num-results 5 --summary --output-format toon
exa-ai search \
  "CUDA split K reduction GEMV atomics second kernel performance optimization" \
  --num-results 5 --summary --output-format toon
exa-ai search \
  "LLM decode GEMV optimization cold weights memory bandwidth CUDA" \
  --num-results 5 --summary --output-format toon
```

Relevant notes:

- CUDA guide page 60: global-memory requests coalesce into the 32-byte
  transactions needed for the warp address distribution; maximize useful bytes
  per transaction.
- CUDA guide page 61: consecutive thread addresses are the typical coalescing
  pattern, but a permutation that stays within the same segments can still
  coalesce.
- CUDA guide pages 67-69: shared-memory bank conflicts are a separate issue
  from HBM coalescing, and atomics should be used sparingly because they
  synchronize memory updates.
- Exa again pointed at split-K/GEMV and decode-fusion work. The practical local
  move was to make the remaining atomic streams more warp-contiguous, then
  retune the split-K tile size.

Retained implementation:

- Changed `Gemma4FfnDecodeScratch` from pack-major SoA:
  `accum_lo[hidden_pack][4]` / `accum_hi[hidden_pack][4]`
  to component-major `accum[8][hidden_pack]`.
  This keeps the scratch footprint at `21,504 B`, but for each atomic component
  a warp now writes consecutive `float` addresses instead of a 16-byte stride.
- Retuned the default FFN decode geometry to `INTERMEDIATE_TILE=2` and
  `ACT_TILE=2`. This creates many more CTAs, but drops the accumulate kernel
  register count and gives better SM coverage on the fully cold metric.
- Removed shared-memory staging for `x` in the accumulate kernel. At tile `2`,
  `x` is not reused inside a CTA, so staging all hidden packs through shared
  memory only adds a barrier and shared traffic. The retained path now loads
  `x` directly and uses the same pre-swizzled weight layout.
- Removed the unused one-pack shared-`x` matmul helper from
  `src/gemma4_matmul_device.cuh`; no retained code calls it after direct `x`
  loads.

Rejected/diagnostic variants:

```text
variant                         regs/spills  smem       cold_minus_flush_clear_ms  decision
component_tile168                64, 0/0        11440 B  0.980680 long             good, but tile retune wins
component_tile128                64, 0/0        11440 B  0.983216 short            reject; slower
component_tile112                64, 0/0        11440 B  0.981282 short            reject; slower
component_tile96                 64, 0/0        11440 B  0.982692 short            reject; slower
component_tile84                 64, 0/0        11440 B  1.030398 short            reject; much slower
component_tile56                 64, 0/0        11440 B  0.979840 long             reject; slower than tile2
component_tile24                 64, 0/0        11440 B  0.979423 long             reject; slower than tile2
component_tile4_act4             50, 0/0        11440 B  0.978434 long             reject; slower than tile2
component_tile2_act2             34, 0/0        11096 B  0.978144 long             good, direct x wins
component_tile1_act1             26, 0/0        10924 B  0.979583 short            reject; too many CTAs
component_threads512             37, 0/0        11016 B  0.980375 short            reject; fewer threads lose
component_threads448             37, 0/0        10984 B  0.982145 short            reject; slower
component_threads384             37, 0/0        10952 B  0.985186 short            reject; slower
component_threads256             40, 0/0        10888 B  0.983360 short            reject; slower
direct_x_tile2_act2              40, 0/0          344 B  0.977318 long             retained
direct_x_tile4_act2              50, 0/0          344 B  0.978383 short            reject; slower
direct_x_tile4_act4              64, 0/0          688 B  0.978186 short            reject; slower
direct_x_tile6_act2              50, 0/0          344 B  0.978328 short            reject; slower
direct_x_tile6_act3              64, 0/0          516 B  0.978962 short            reject; slower
direct_x_no_swizzle              40, 0/0          344 B  0.977545 long             reject; slower
direct_x_weight_policy_cg        40, 0/0          344 B  0.977894 short            reject; slower
direct_x_weight_policy_ldg       40, 0/0          344 B  0.977842 short            reject; slower
direct_x_preload_down            40, 0/0          344 B  0.977484 short            reject; slower
direct_x_launch_bounds_min2      36, 0/0          344 B  0.977376 long             reject; lower regs but slower
direct_x_tile12_act4             72, 0/0          688 B  0.979446 short            reject; over 67-reg cap
direct_x_tile24_act4             72, 0/0          688 B  0.979414 short            reject; over 67-reg cap
```

Final validation:

```bash
make test-ffn-decode
nvcc -std=c++17 -O3 -arch=sm_86 -Xptxas=-v -Isrc \
  -c src/gemma4_ffn_decode.cu \
  -o build/ptx/ffn_direct_final/gemma4_ffn_decode.o
make ffn-decode-load-bench
GEMMA4_FFN_LOAD_BENCH_SEED=0x20260522 \
  build/experiments/gemma4_ffn_decode_load_bench 80 12 5
make ffn-cudnn-bench
GEMMA4_FFN_CUDNN_BENCH_SEED=0x20260522 \
  build/experiments/gemma4_ffn_cudnn_bench 50 10 3
```

Final resource check:

```text
ffn decode tests passed
accumulate kernel: 40 registers, 0 spills, 344 B smem
finalize kernel:   26 registers, 0 spills,  88 B smem
```

Focused fully cold load bench, `iters=80,warmup=12,trials=5`:

```text
custom_graph best_ms=0.976600
scratch_clear_graph best_ms=0.001088
warm_minus_clear best_ms=0.975512
cold_custom best_ms=1.376668
cold_clear best_ms=0.399349
flush_only best_ms=0.397513
cold_minus_flush_clear best_ms=0.977318
```

Full cuDNN comparison harness, `iters=50,warmup=10,trials=3`:

```text
cudnn_split_device_ms=0.998332
custom_device_ms=0.976398
custom_scratch_clear_device_ms=0.001107
custom_minus_clear_device_ms=0.975291
cold_custom_minus_flush_clear_ms=0.977360
custom_vs_cudnn_split_speedup=1.022464
custom_minus_clear_vs_cudnn_split_speedup=1.023624
```

Conclusion:

- Retained component-major scratch, direct `x` gate/up loads, and
  `INTERMEDIATE_TILE=2` / `ACT_TILE=2`.
- The focused fully cold metric improved from the previous retained
  `0.982118 ms` to `0.977318 ms` (`~0.5%`) without increasing scratch memory.
- Compared with the pre-atomic retained baseline (`1.014572 ms`), the current
  focused fully cold path is now about `3.7%` faster.

## 2026-05-25 - FFN decode one-pack specialization

Goal:

- Continue reducing fully-cold FFN decode time without crossing the `67`
  register cap or growing persistent memory. The current direct-`x` kernel uses
  `672` threads, exactly one thread per hidden bf16 pack, so this round tested
  whether making that measured shape explicit improves the hot path.

Guide and outside references:

```bash
python3 scripts/query.py \
  "CUDA device tanhf __tanhf fast approximate intrinsic performance accuracy" \
  --top-k 10
python3 scripts/query.py \
  "CUDA launch bounds register pressure occupancy performance maxrregcount tradeoff" \
  --top-k 10
python3 scripts/query.py \
  "CUDA memory coalescing contiguous per warp global stores atomics performance" \
  --top-k 8
exa-ai search \
  "CUDA GELU tanh approximation kernel optimization __tanhf LLM inference" \
  --num-results 5 --summary --output-format toon
exa-ai search \
  "CUDA GEMV LLM decode optimize register pressure launch bounds atomics" \
  --num-results 5 --summary --output-format toon
```

Relevant notes:

- CUDA guide pages 605-606 document `__tanhf` and note that `--use_fast_math`
  maps `tanhf` to the intrinsic form. It passed the local BF16 correctness test
  but did not improve the long fully-cold metric.
- CUDA guide pages 70-72 and 525-527 cover occupancy, register pressure, and
  launch-bounds tradeoffs. Lowering registers alone was not enough reason to
  retain a variant when fully-cold time regressed.
- Exa surfaced LLM GELU kernels using faster tanh/MUFU-style approximations and
  GEMV decode discussions. The useful local action was to test the special
  function path and then specialize the measured one-pack threading shape.

Retained implementation:

- Added a compile-time requirement that the FFN decode CTA maps exactly one
  thread to each hidden bf16 pack: `THREADS == HIDDEN_PACKS == 672`.
- Removed the generic hidden-pack round loops from the hot accumulate/finalize
  paths and made the one-pack path explicit.
- Kept the direct `x` load path and component-major scratch layout from the
  previous round.

Rejected/diagnostic variants:

```text
variant                         regs/spills  smem       cold_minus_flush_clear_ms  decision
fast_tanh_intrinsic              38, 0/0        344 B    0.977441 long             reject; correct but slower
x_load_normal                    40, 0/0        344 B    0.977363 long             reject; slower than retained
x_load_cg                        40, 0/0        344 B    0.977351 short            reject; no win
int32_hot_offsets                38, 0/0        344 B    0.977495 long             reject; lower regs but slower
threads512                       40, 0/0        264 B    0.982346 short            reject; slower
threads576                       40, 0/0        296 B    0.978621 short            reject; slower
threads608                       40, 0/0        312 B    0.978406 short            reject; slower
threads640                       40, 0/0        328 B    0.979277 short            reject; slower
threads704                       40, 0/0        360 B    0.977477 short            reject; slower
threads736                       40, 0/0        376 B    0.977695 short            reject; slower
threads768                       40, 0/0        392 B    0.977641 short            reject; slower
threads800                       40, 0/0        408 B    0.977324 long             reject; near tie/slower
threads832                       40, 0/0        424 B    0.977582 long             reject; slower
one_pack_forceinline             34, 0/0        344 B    0.977287 long             good, full specialization wins
static_one_pack                  34, 0/0        344 B    0.977116 long             retained
```

Discarded data:

- A first long run of `threads768/800/832` was accidentally launched in
  parallel on the same GPU, producing multi-millisecond timings. Those numbers
  were discarded and the nearest candidates were rerun serially before making a
  decision.

Final validation:

```bash
make test-ffn-decode
nvcc -std=c++17 -O3 -arch=sm_86 -Xptxas=-v -Isrc \
  -c src/gemma4_ffn_decode.cu \
  -o build/ptx/ffn_static_one_pack/gemma4_ffn_decode.o
make ffn-decode-load-bench
GEMMA4_FFN_LOAD_BENCH_SEED=0x20260522 \
  build/experiments/gemma4_ffn_decode_load_bench 80 12 5
make ffn-cudnn-bench
GEMMA4_FFN_CUDNN_BENCH_SEED=0x20260522 \
  build/experiments/gemma4_ffn_cudnn_bench 50 10 3
```

Final resource check:

```text
ffn decode tests passed
accumulate kernel: 34 registers, 0 spills, 344 B smem
finalize kernel:   25 registers, 0 spills,  88 B smem
```

Focused fully cold load bench, `iters=80,warmup=12,trials=5`:

```text
custom_graph best_ms=0.976389
scratch_clear_graph best_ms=0.001121
warm_minus_clear best_ms=0.975268
cold_custom best_ms=1.376650
cold_clear best_ms=0.399534
flush_only best_ms=0.397623
cold_minus_flush_clear best_ms=0.977116
```

Full cuDNN comparison harness, `iters=50,warmup=10,trials=3`:

```text
cudnn_split_device_ms=0.998921
custom_device_ms=0.976564
custom_scratch_clear_device_ms=0.001131
custom_minus_clear_device_ms=0.975433
cold_custom_minus_flush_clear_ms=0.977228
custom_vs_cudnn_split_speedup=1.022894
custom_minus_clear_vs_cudnn_split_speedup=1.024080
```

Conclusion:

- Retained the one-pack specialization.
- Focused fully-cold timing improved from the previous retained `0.977318 ms`
  to `0.977116 ms` with lower register count.
- This is a small win, not a new big structural jump, but it removes slower
  generic code from the hot path and does not increase scratch or shared memory.

## 2026-05-25 - FFN decode post-one-pack rejection sweep

Goal:

- Continue looking for a larger fully-cold win after the one-pack specialization
  while keeping registers below `67`, avoiding extra persistent memory, and
  keeping the retained FFN decode source tight.

Guide and outside references:

```bash
python3 scripts/query.py \
  "CUDA global memory load cache operators ldg cg ca streaming read only data kernel performance coalesced" \
  --top-k 10
python3 scripts/query.py \
  "CUDA atomic add performance many thread blocks same addresses reduction alternatives global memory coalesced" \
  --top-k 10
python3 scripts/query.py \
  "CUDA kernel launch overhead CUDA graphs persistent kernel cooperative groups reduction performance" \
  --top-k 8
python3 scripts/query.py \
  "CUDA split K reduction tile size atomics occupancy tradeoff global memory reduction" \
  --top-k 8
python3 scripts/query.py \
  "CUDA atomicAdd memory order relaxed device scope kernel completion visibility atomic operations" \
  --top-k 10
exa-ai search \
  "CUDA LLM decode FFN GEMV optimization fused GeGLU down projection atomics cold HBM" \
  --num-results 5 --summary --output-format toon
exa-ai search \
  "CUDA persistent thread block GEMV reduction avoid atomics LLM inference FFN" \
  --num-results 5 --summary --output-format toon
exa-ai search \
  "CUDA split-K GEMV tile size atomic reduction occupancy LLM decode" \
  --num-results 5 --summary --output-format toon
```

Relevant notes:

- CUDA guide page 60/61 again points to useful bytes per memory transaction as
  the first-order HBM check. The retained gate/up/down streams remain
  warp-contiguous 128-bit pack loads.
- CUDA guide page 70 warns that atomics should be used sparingly; this motivated
  a larger-tile sweep to reduce scratch atomic frequency without changing the
  scratch layout.
- CUDA guide pages 71 and 526 cover the occupancy/register/launch-bounds
  tradeoff. The larger tiles stayed under the register cap, but lower CTA
  coverage and higher register count outweighed fewer atomics.
- CUDA guide page 536 says legacy atomics provide atomicity but do not add
  fences. An inline relaxed-RED experiment was not retained because SASS was
  identical to default `atomicAdd` codegen for this kernel.
- Exa surfaced split-K GEMV and LLM decode work. The actionable local test was
  to retune the split-K/intermediate tile and preload scheduling while keeping
  the memory layout fixed.

Rejected/diagnostic variants:

```text
variant                         regs/spills  smem       cold_minus_flush_clear_ms  decision
activation_parallel              34, 0/0        344 B    0.977037 short            reject; tie/no long win
activation_parallel_unroll       34, 0/0        344 B    0.977174 short            reject; slower
forceinline_hot_helpers          34, 0/0        344 B    0.977136 short            reject; no win
launch_bounds_min2               34, 0/0        344 B    0.977034 short            reject; define works, no real win
swizzle_chunks4                  34, 0/0        344 B    0.977368 short            reject; slower
swizzle_chunks16                 34, 0/0        344 B    0.977178 short            reject; slower/tie
swizzle_chunks32                 34, 0/0        344 B    0.978150 short            reject; slower
tile2_act1                       not logged     344 B    0.977515 short            reject; slower
tile3_act1                       not logged     344 B    0.977754 short            reject; slower
tile3_act3                       not logged     516 B    0.977195 short            reject; slower/tie
init_partial_from_down0          38, 0/0        344 B    0.977216 short            reject; higher regs and slower
all_lane_gate_up_reduce          34, 0/0        344 B    0.977358 short            reject; slower
inline_red_global                34, 0/0        344 B    0.977028 short            reject; SASS same as default
inline_red_relaxed               34, 0/0        344 B    0.977042 long             reject; SASS same as default
early_preload_down               38, 0/0        344 B    0.977104 long             reject; tie with higher regs
preload_gate_up                  34, 0/0        344 B    0.977080 short            reject; tie/no win
weight_policy_cg                 34, 0/0        344 B    0.977644 short            reject; slower
weight_policy_ldg                34, 0/0        344 B    0.977429 short            reject; slower
tile8_act2                       48, 0/0        344 B    0.979051 short            reject; slower
tile12_act2                      48, 0/0        344 B    0.978430 short            reject; slower
tile16_act2                      48, 0/0        344 B    0.980790 short            reject; slower
tile24_act2                      48, 0/0        344 B    0.978715 short            reject; slower
tile32_act2                      48, 0/0        344 B    0.980407 short            reject; slower
tile64_act2                      48, 0/0        344 B    0.982238 short            reject; slower
tile12_act3                      56, 0/0        516 B    0.979040 short            reject; slower
tile24_act3                      56, 0/0        516 B    0.978781 short            reject; slower
tile48_act3                      56, 0/0        516 B    0.979649 short            reject; slower
tile8_act4                       62, 0/0        688 B    0.978762 short            reject; slower
```

NCU diagnostic:

```bash
timeout 90s ncu --target-processes all --kernel-name-base demangled \
  --kernel-name regex:.*gemma4_ffn_decode_accumulate.* --launch-count 1 \
  --metrics gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,\
l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum \
  --csv build/experiments/gemma4_ffn_decode_load_bench 1 0 1
```

The NCU command again timed out with `==WARNING== No kernels were profiled.`
Treat the conflict/coalescing conclusions as static address-pattern checks plus
CUDA-event timing until a narrower profiler harness is available.

Final validation:

```bash
make test-ffn-decode
nvcc -std=c++17 -O3 -arch=sm_86 -Xptxas=-v -Isrc \
  -c src/gemma4_ffn_decode.cu \
  -o build/ptx/ffn_final_after_sweep/gemma4_ffn_decode.o
make ffn-decode-load-bench
GEMMA4_FFN_LOAD_BENCH_SEED=0x20260522 \
  build/experiments/gemma4_ffn_decode_load_bench 80 12 5
make ffn-cudnn-bench
GEMMA4_FFN_CUDNN_BENCH_SEED=0x20260522 \
  build/experiments/gemma4_ffn_cudnn_bench 50 10 3
```

Final resource check:

```text
ffn decode tests passed
accumulate kernel: 34 registers, 0 spills, 344 B smem
finalize kernel:   25 registers, 0 spills,  88 B smem
```

Focused fully cold load bench, `iters=80,warmup=12,trials=5`:

```text
custom_graph best_ms=0.976384
scratch_clear_graph best_ms=0.001090
warm_minus_clear best_ms=0.975295
cold_custom best_ms=1.376530
cold_clear best_ms=0.399482
flush_only best_ms=0.397604
cold_minus_flush_clear best_ms=0.977048
```

Full cuDNN comparison harness, `iters=50,warmup=10,trials=3`:

```text
cudnn_split_device_ms=0.998058
custom_device_ms=0.976499
custom_scratch_clear_device_ms=0.001139
custom_minus_clear_device_ms=0.975360
cold_custom_minus_flush_clear_ms=0.977031
custom_vs_cudnn_split_speedup=1.022077
custom_minus_clear_vs_cudnn_split_speedup=1.023271
```

Conclusion:

- No new source optimization was retained from this sweep.
- The current best source remains the one-pack direct-`x`,
  component-major-scratch path with `INTERMEDIATE_TILE=2`, `ACT_TILE=2`, and
  `.cs` weight loads.
- Larger intermediate tiles with `ACT_TILE=2/3/4` reduce scratch atomic count,
  but lose enough CTA coverage and register headroom that fully-cold time
  regresses.
- Preload and inline-RED variants did not create a defensible improvement, so
  they were removed to keep `ffn-decode` tight.

## 2026-05-25 - FFN decode interleaved gate/up prepared layout

Goal:

- Continue searching for a fully-cold improvement without changing scratch
  size, exceeding the `67` register cap, or adding a broad abstraction.

Guide and outside references:

```bash
python3 scripts/query.py \
  "CUDA bfloat16 bf16 vector intrinsic accumulate float fma __nv_bfloat162 performance" \
  --top-k 10
python3 scripts/query.py \
  "CUDA bf16 tensor cores wmma mma single row GEMV performance bfloat16" \
  --top-k 10
exa-ai search \
  "CUDA bfloat16 GEMV scalar dot __nv_bfloat162 fma FP32 accumulate optimization" \
  --num-results 5 --summary --output-format toon
```

Relevant notes:

- CUDA guide pages 576-577 confirm BF16 tensor-core WMMA supports BF16 inputs
  with FP32 accumulators, but the fragment shapes are 2D matrix tiles. A
  tensor-core rewrite would be a larger custom GEMM/GEMV path, not a tight
  change to this fused scalar-GEMV kernel.
- CUDA guide page 592 covers floating-point intrinsic tradeoffs. A local
  BF16x2 `__hfma2` partial-dot experiment reduced registers but weakened
  accumulation precision and slowed the measured path.
- Exa surfaced BF16 math API references and GEMV kernels. The practical local
  move was to test whether keeping each gate/up pair adjacent in the prepared
  decode layout improves the current HBM stream.

Retained implementation:

- Changed the prepared `w_gate_up` layout written by
  `gemma4_ffn_decode_swizzle_weights_bf16()` from gate rows followed by up rows
  to interleaved rows: `gate0, up0, gate1, up1, ...`.
- Updated the hot dot path to derive `up_ptr` from `gate_ptr + hidden_size` and
  step by two rows per local activation column.
- Updated `tests/test_ffn_decode.cu` to exercise the documented prepared-weight
  path by filling raw source weights and calling
  `gemma4_ffn_decode_swizzle_weights_bf16()` before the fused decode kernel.
- Updated the header comments to document the decode-prepared interleaved
  gate/up layout.

Rejected/diagnostic variants:

```text
variant                         regs/spills  smem       cold_minus_flush_clear_ms  decision
gate_up_interleaved_macro        34, 0/0        344 B    0.976867 long             good; promoted cleanly
bf16x2_local_dot                 32, 0/0        344 B    0.977490 short            reject; slower and weaker precision
```

Final validation:

```bash
make test-ffn-decode
nvcc -std=c++17 -O3 -arch=sm_86 -Xptxas=-v -Isrc \
  -c src/gemma4_ffn_decode.cu \
  -o build/ptx/ffn_interleave_retained/gemma4_ffn_decode.o
make ffn-decode-load-bench
GEMMA4_FFN_LOAD_BENCH_SEED=0x20260522 \
  build/experiments/gemma4_ffn_decode_load_bench 80 12 5
make ffn-cudnn-bench
GEMMA4_FFN_CUDNN_BENCH_SEED=0x20260522 \
  build/experiments/gemma4_ffn_cudnn_bench 50 10 3
```

Final resource check:

```text
ffn decode tests passed
accumulate kernel: 34 registers, 0 spills, 344 B smem
finalize kernel:   25 registers, 0 spills,  88 B smem
gate/up swizzle:   18 registers, 0 spills
```

Focused fully cold load bench, `iters=80,warmup=12,trials=5`:

```text
custom_graph best_ms=0.976280
scratch_clear_graph best_ms=0.001097
warm_minus_clear best_ms=0.975182
cold_custom best_ms=1.376348
cold_clear best_ms=0.399382
flush_only best_ms=0.397508
cold_minus_flush_clear best_ms=0.976965
```

Repeat focused long run:

```text
custom_graph best_ms=0.976388
scratch_clear_graph best_ms=0.001091
warm_minus_clear best_ms=0.975298
cold_custom best_ms=1.376519
cold_clear best_ms=0.399527
flush_only best_ms=0.397574
cold_minus_flush_clear best_ms=0.976992
```

Full cuDNN comparison harness, `iters=50,warmup=10,trials=3`:

```text
cudnn_split_device_ms=0.998234
custom_device_ms=0.976525
custom_scratch_clear_device_ms=0.001167
custom_minus_clear_device_ms=0.975357
cold_custom_minus_flush_clear_ms=0.977042
custom_vs_cudnn_split_speedup=1.022231
custom_minus_clear_vs_cudnn_split_speedup=1.023455
max_abs_vs_split=0
```

Conclusion:

- Retained the interleaved gate/up prepared layout.
- Focused fully-cold timing improved from the prior retained `0.977048 ms` to
  `0.976965-0.976992 ms` in two long runs. This is a small layout win, not a
  structural jump.
- Full cuDNN harness correctness remains exact against the split baseline
  (`max_abs_vs_split=0`) and keeps the custom path about `2.3%` ahead of the
  cuDNN split graph metric after scratch clear subtraction.

## 2026-05-25 - FFN decode partial grid-stride atomic reduction

Goal:

- Revisit a larger reduction-shape move without growing scratch memory or
  crossing the `67` register cap. The retained tile-2 path has excellent HBM
  coalescing but still launches one CTA per two FFN columns, so every tile does
  a full component-major scratch atomic pass.

Guide and outside references:

```bash
python3 scripts/query.py \
  "CUDA global memory coalescing contiguous 16 byte per thread memory transaction warp access" \
  --top-k 8
python3 scripts/query.py \
  "CUDA atomics performance use sparingly atomicAdd global memory reduction optimization" \
  --top-k 8
python3 scripts/query.py \
  "CUDA occupancy register pressure launch bounds blocks per multiprocessor performance" \
  --top-k 8
exa-ai search \
  "CUDA single token FFN decode GEMV optimization split K atomics reduction LLM inference" \
  --num-results 5 --summary --output-format toon
exa-ai search \
  "CUDA GEMV optimization small m large n vector matrix multiplication memory bandwidth coalescing" \
  --num-results 5 --summary --output-format toon
```

Relevant notes:

- CUDA guide pages 60-61 again point at useful bytes per memory transaction.
  The retained path keeps the warp-contiguous 128-bit gate/up/down HBM streams.
- CUDA guide page 70 says atomics should be used sparingly; this pass tried to
  cut only a small number of atomic waves while preserving high CTA count.
- CUDA guide pages 71 and 526 frame the register/occupancy tradeoff. The
  retained shape raises the accumulate kernel from `34` to `48` registers, still
  below the `67` cap and with zero spills.
- Exa surfaced split-K/GEMV and persistent/decode-fusion work. The practical
  local adaptation was a partial grid-stride reduction, not a broad tensor-core
  rewrite.

Retained implementation:

- Added `GEMMA4_FFN_DECODE_ACCUM_BLOCKS` and changed the default accumulated
  CTA count from one CTA per intermediate tile (`10752`) to
  `kIntermediateTiles - kHiddenPacks` (`10080`).
- The first `672` CTAs fold one extra 2-column tile into their local partial
  before issuing scratch atomics; the remaining CTAs still process one tile.
  This cuts one hidden-pack wave of scratch atomics while keeping most of the
  original grid density.
- Factored the per-tile body into `accumulate_intermediate_tile()` so the
  default two-visit path and override paths share the same load/reduction logic.
- Scratch size and HBM weight layout are unchanged.

Rejected/diagnostic variants:

```text
variant                         regs/spills  smem   cold_minus_flush_clear_ms  decision
tile4_act2                       48, 0/0      344 B  0.977838 short            reject; slower
tile4_act4                       50, 0/0      688 B  0.977978 short            reject; slower
tile14_act2                      48, 0/0      344 B  0.978778 short            reject; slower
tile7_act1                       39, 0/0      172 B  0.977832 short            reject; slower
x_regular_load                   34, 0/0      344 B  0.977386 short            reject; slower
x_cg_load                        34, 0/0      344 B  0.977169 long             reject; short win did not hold
launch_bounds_min2               34, 0/0      344 B  0.977469 short            reject; slower
maxrregcount32                   31, 0/0      344 B  0.977106 long             reject; lower regs did not win
unpack_x_once                    34, 0/0      344 B  0.976912/0.977039 long   reject; SASS same aside from metadata
activation_parallel_unsafe       34, 0/0      344 B  invalid                  reject; lane 1 lacked final reduction
activation_parallel_shfl         34, 0/0      344 B  0.977107 short            reject; correct but slower
grid5376                         48, 0/0      344 B  0.977235 short            reject; slower
grid2688                         48, 0/0      344 B  0.978525 short            reject; slower
grid8064_generic                 48, 0/0      344 B  0.977054 long             reject; generic loop lost cold metric
grid8064_twopass                 48, 0/0      344 B  0.976649/0.976769 long   good, but 10080 wins
grid9216_twopass                 48, 0/0      344 B  0.976499 long             reject; slower than 10080 repeat
grid10080_twopass                48, 0/0      344 B  0.976340/0.976696 long   retained
```

Final validation:

```bash
make test-ffn-decode
nvcc -std=c++17 -O3 -arch=sm_86 -Xptxas=-v -Isrc \
  -c src/gemma4_ffn_decode.cu \
  -o build/ptx/ffn_goal_variants/gemma4_ffn_decode_default_grid.o
make ffn-decode-load-bench
GEMMA4_FFN_LOAD_BENCH_SEED=0x20260522 \
  build/experiments/gemma4_ffn_decode_load_bench 80 12 5
make ffn-cudnn-bench
GEMMA4_FFN_CUDNN_BENCH_SEED=0x20260522 \
  build/experiments/gemma4_ffn_cudnn_bench 50 10 3
```

Final resource check:

```text
ffn decode tests passed
accumulate kernel: 48 registers, 0 spills, 344 B smem
finalize kernel:   25 registers, 0 spills,  88 B smem
gate/up swizzle:   18 registers, 0 spills
```

Focused fully cold load bench, `iters=80,warmup=12,trials=5`, promoted source:

```text
custom_graph best_ms=0.975748
scratch_clear_graph best_ms=0.001106
warm_minus_clear best_ms=0.974642
cold_custom best_ms=1.376473
cold_clear best_ms=0.400001
flush_only best_ms=0.397651
cold_minus_flush_clear best_ms=0.976472
```

Full cuDNN comparison harness, `iters=50,warmup=10,trials=3`, promoted source:

```text
cudnn_split_device_ms=0.998656
custom_device_ms=0.975663
custom_scratch_clear_device_ms=0.001116
custom_minus_clear_device_ms=0.974547
cold_custom_minus_flush_clear_ms=0.976290
custom_vs_cudnn_split_speedup=1.023567
custom_minus_clear_vs_cudnn_split_speedup=1.024739
max_abs_vs_split=0
```

Conclusion:

- Retained the partial two-visit accumulate grid. It gives a small but
  repeatable fully-cold improvement without changing scratch memory or HBM
  coalescing.
- Focused fully-cold timing moved from the previous retained `0.976965-0.976992
  ms` band to `0.976472 ms` in the promoted-source validation run.
- Full cuDNN harness correctness remains exact (`max_abs_vs_split=0`), and the
  full cold metric improved from the prior retained `0.977042 ms` to
  `0.976290 ms`.

## 2026-06-16 - Ponytail FlashAttention cleanup

Scope:

- Cleaned the custom Gemma 4 BF16 FlashAttention kernel for the fixed
  RTX A6000/sm86 target.
- Removed dead generic paths instead of adding abstractions: local `round_up`,
  unused params, pre-SM80 fallbacks, half-precision trait plumbing, copied
  base-trait aliases, `void *` input storage, the block-info wrapper, and the
  local/global mask wrapper.
- Kept comments, including `ponytail:` comments that mark intentional fixed-path
  assumptions.

Build:

```bash
make -B flash-attn-bench NVCC=/usr/local/cuda/bin/nvcc
```

Benchmark contract:

- Harness: `build/experiments/gemma4_flash_attention_bench`
- Command per process:
  `./build/experiments/gemma4_flash_attention_bench 1024 500 50 1 1 64`
- Measures custom sliding BF16 attention with CUDA events in the harness.
- Shape: `batch=1`, `seq=1024`, `window_left=1024`; correctness reference uses
  `seq=64`.
- Warmup/timing: `50` warmup iterations, `500` timed iterations, `trials=1`,
  repeated across `9` fresh processes.
- Cache policy: warm-cache repeated buffers; launch included in the measured
  event window.
- Clock policy: clocks were not locked.

Environment:

```text
GPU: NVIDIA RTX A6000
Driver: 580.126.16
CUDA/NVCC: CUDA 13.0, V13.0.48
Build arch: sm_86
Persistence: Enabled
ECC: Disabled
MIG: N/A
Power limit: 300 W
Pre-run telemetry: 690 MHz SM, 810 MHz memory, 29 C, 33.55 W, 0% util
Commit: e77022a plus local working tree changes
```

Before cleanup baseline, same command and process repetition:

```text
raw_ms = 0.232100, 0.234383, 0.234037, 0.234784, 0.235245,
         0.233095, 0.233433, 0.234248, 0.235153
median = 0.234248 ms
mean   = 0.234053 ms
```

After cleanup:

```text
raw_ms = 0.229678, 0.229017, 0.231791, 0.229811, 0.229841,
         0.229190, 0.229540, 0.230041, 0.230285
median = 0.229811 ms
mean   = 0.229910 ms
min    = 0.229017 ms
max    = 0.231791 ms
approx_tflops range = 74.1902-75.0891
```

Correctness for every after sample:

```text
max_abs = 0.00390625
mean_abs = 7.72008e-05
max_rel = 0.00390625
```

Conclusion:

- The source is simpler and more fixed-path: fewer params, fewer traits, fewer
  fallbacks, no custom rounding helper, no input `const_cast`.
- Timing is slightly faster in this run, but the main claim is cleanup with no
  regression. Clocks were not locked, and earlier telemetry in this session
  reported different bus IDs, so treat the small delta as neutral-to-positive
  unless repeated under locked clocks.

## 2026-06-16 - Additional FlashAttention micro-cleanups

Scope:

- Removed runtime Q/KV head count and GQA-ratio params from the device params.
  The sliding/global ratios are model constants, so the kernel now uses
  compile-time `2` or `8`.
- Hoisted per-CTA arithmetic for batch offsets, query tile start/remaining,
  sequence delta, and mask row offset.
- Removed contiguous tensor stride params. Row/head strides are fixed by the
  Gemma layout and kernel trait.
- Simplified LSE tile construction to a direct pointer plus `[BlockM]` tensor,
  then kept batch size on the host launcher instead of copying it into device
  params.
- Hoisted duplicated query-to-key offset arithmetic in local/global mask
  helpers.

Build:

```bash
make -B flash-attn-bench NVCC=/usr/local/cuda/bin/nvcc
```

Benchmark contract:

- Same as the prior FlashAttention cleanup entry:
  `./build/experiments/gemma4_flash_attention_bench 1024 500 50 1 1 64`
- Warm-cache repeated buffers, CUDA-event timing in the harness, launch
  included, `9` fresh processes per retained step.
- Clocks were not locked.

Baseline from the prior cleanup:

```text
raw_ms = 0.229678, 0.229017, 0.231791, 0.229811, 0.229841,
         0.229190, 0.229540, 0.230041, 0.230285
median = 0.229811 ms
mean   = 0.229910 ms
```

After compile-time GQA ratio and CTA arithmetic hoist:

```text
raw_ms = 0.229429, 0.228381, 0.230113, 0.226982, 0.228221,
         0.228584, 0.229043, 0.227851, 0.230002
median = 0.228584 ms
mean   = 0.228734 ms
```

After fixed contiguous strides:

```text
raw_ms = 0.228801, 0.228456, 0.227953, 0.227999, 0.228939,
         0.228987, 0.227270, 0.228647, 0.228365
median = 0.228456 ms
mean   = 0.228380 ms
```

After mask arithmetic hoist:

```text
raw_ms = 0.226781, 0.226593, 0.227552, 0.227764, 0.226285,
         0.228099, 0.228871, 0.228722, 0.228360
median = 0.227764 ms
mean   = 0.227670 ms
```

After direct LSE tile and host-only batch size:

```text
raw_ms = 0.226992, 0.226784, 0.226852, 0.227711, 0.227270,
         0.228066, 0.226680, 0.228802, 0.228254
median = 0.227270 ms
mean   = 0.227490 ms
min    = 0.226680 ms
max    = 0.228802 ms
```

Correctness for every retained sample stayed:

```text
max_abs = 0.00390625
mean_abs = 7.72008e-05
max_rel = 0.00390625
```

Conclusion:

- Kept all four micro-cleanups. They delete params and repeated arithmetic, and
  the final median improved from `0.229811 ms` to `0.227270 ms` under the same
  unlocked-clock warm-cache contract.
- Treat the delta as a small same-machine win, not a locked-clock claim.

## 2026-06-16 - Split QK and PV TiledMMA aliases

Scope:

- Split the forward attention MMA type into `TiledMmaQK` and `TiledMmaPV`.
- Kept QK row-split because softmax state is row-owned.
- Tried a true/wider PV variant for better head-dim coverage, then rejected it
  because it regressed the warm-cache benchmark.
- Retained the split abstraction with the row-compatible PV tile so future PV
  ablations are localized.

Rejected variants:

- `TiledMmaPV = Layout<Shape<2,2,1>>, Tile<32,32,16>` for sliding and
  `Layout<Shape<1,2,1>>, Tile<16,32,16>` for global did not compile. The QK
  probability fragment is not directly compatible with the N-split PV A
  operand; it would need explicit P staging or a larger cross-warp retile.
- Row-compatible PV with `Tile<16*kNWarps,32,16>` compiled and was correct, but
  was slower:

```text
raw_ms = 0.230811, 0.230364, 0.231751, 0.230840, 0.228249,
         0.227993, 0.229381, 0.227745, 0.229930
median = 0.229930 ms
mean   = 0.229674 ms
```

Retained split:

```text
TiledMmaQK = Layout<Shape<kNWarps,1,1>>, Tile<16*kNWarps,16,16>
TiledMmaPV = Layout<Shape<kNWarps,1,1>>, Tile<16*kNWarps,16,16>
```

Validation:

```bash
make -B flash-attn-bench NVCC=/usr/local/cuda/bin/nvcc
./build/experiments/gemma4_flash_attention_bench 1024 500 50 1 1 64
```

Retained split benchmark, `9` fresh processes, same warm-cache CUDA-event
contract as above:

```text
raw_ms = 0.226513, 0.226714, 0.228271, 0.228916, 0.226390,
         0.228534, 0.227730, 0.229035, 0.227525
median = 0.227730 ms
mean   = 0.227736 ms
min    = 0.226390 ms
max    = 0.229035 ms
```

Correctness for every retained sample:

```text
max_abs = 0.00390625
mean_abs = 7.72008e-05
max_rel = 0.00390625
```

Conclusion:

- Retained the QK/PV type split because it makes the two phases explicit and
  keeps the measured path neutral under the existing benchmark.
- Did not retain a different PV warp layout; true N-split is not a cheap
  one-line CUTE change for this SM80 row-owned softmax pipeline.

## 2026-06-16 - Paged KV cache decode baseline

Scope:

- Added Layout-A paged KV cache metadata and address helper:
  `[layers, pages, page_size, kv_heads, head_dim]`.
- Added host monotonic/free-list page allocator and sliding-window slot reuse.
- Added prepared-K/V cache write kernel for both prefill-style bulk writes and
  one-token decode appends.
- Added paged decode attention with split-KV partial softmax states and a
  second reduction kernel.
- Added global and sliding correctness coverage for non-contiguous physical
  pages, page boundaries, mixed batch lengths, split-KV, and sliding wraparound.
- RoPE and Q/K/V norms are intentionally outside this cache module; the cache
  receives already prepared K/V from the FlashAttention preparation path.

Validation:

```bash
make NVCC=/usr/local/cuda/bin/nvcc test-kv-cache
```

Result:

```text
kv cache tests passed
```

Optimization passes after the first correct baseline:

- Vectorized BF16 cache writes as aligned 128-bit `int4` chunks.
- Switched decode attention read-only scalar loads to `loadg`/`__ldg`.
- Added early neutral-state return for empty splits.
- Hoisted repeated Q, partial, and row-base offset arithmetic.
- Reused the K/V cache base offset inside the attention loop.
- Reduced vectorized-write launch width from 256 to 128 threads.
- Added launch bounds for write, split, and reduce kernels.
- Swept split size `32/64/128/256` at `S=4096`.
- Kept split size `64` as the benchmark default after the sweep.
- Re-ran correctness and benchmark after the retained changes.

Benchmark contract:

- Hardware: NVIDIA RTX A6000, SM86, 50.90 GB.
- Driver/runtime from benchmark: CUDA driver `13000`, runtime `13000`.
- `nvidia-smi`: driver `580.126.16`, persistence enabled, ECC disabled, power
  limit `300 W`, idle post-run clocks `210/405 MHz`, temp `31 C`.
- Clocks: not locked. Warm-cache repeated-buffer benchmark.
- Timing: CUDA events on the same stream; launch overhead included.
- Shape: global Gemma decode, `B=1`, `S=4096`, `Q heads=32`,
  `KV heads=4`, `D=512`, BF16, page size `64`.
- Warmup/samples: `25` warmups, `100` iterations per sample, `15` samples.
- Nsight Compute: unavailable in this environment (`ncu` not found).

Custom paged KV command:

```bash
make NVCC=/usr/local/cuda/bin/nvcc kv-cache-bench
./build/experiments/gemma4_kv_cache_bench 4096 64 64 25 100 15
```

Custom paged KV result:

```text
correctness max_abs=0.000244 mean_abs=0.000000
prefill_cache_write median_ms=0.102940 mean_ms=0.102942 min_ms=0.102660 max_ms=0.103254
decode_cache_write median_ms=0.016391 mean_ms=0.015526 min_ms=0.011094 max_ms=0.018428
paged_decode_attention median_ms=0.792760 mean_ms=0.794010 min_ms=0.786287 max_ms=0.800686
paged_full_decode_write_plus_attention median_ms=0.799518 mean_ms=0.799313 min_ms=0.794992 max_ms=0.801205
```

Split-size sweep, same shape with `5` warmups, `20` iterations, `5` samples:

```text
split=32  attention median_ms=0.798474 full_decode median_ms=0.801403
split=64  attention median_ms=0.790414 full_decode median_ms=0.770339
split=128 attention median_ms=0.822251 full_decode median_ms=0.824173
split=256 attention median_ms=0.926520 full_decode median_ms=0.927453
```

PyTorch comparable baseline:

```bash
python3 src/experiments/gemma4_kv_cache_torch_bench.py \
  --seq-len 4096 --warmup 25 --iters 100 --samples 15
```

PyTorch SDPA result:

```text
torch=2.11.0+cu130 cuda_runtime=13.0
attention_only median_ms=3.305070 mean_ms=3.305947 min_ms=3.304270 max_ms=3.315057
full_decode_write_plus_attention median_ms=3.314501 mean_ms=3.315363 min_ms=3.311447 max_ms=3.320166
```

Conclusion:

- The first paged decode path is correct under the targeted boundary tests.
- On this warm-cache decode shape, custom paged attention is about `4.17x`
  faster than PyTorch SDPA attention-only median (`3.305070 / 0.792760`).
- The comparison is useful but not final: clocks were not locked, cache state is
  warm, PyTorch uses contiguous K/V rather than paged K/V, and `ncu` counters
  are unavailable here.

## 2026-06-16 - Paged KV cache primitive simplification

Scope:

- Replaced hand-rolled warp/block max and sum reductions in
  `gemma4_kv_cache.cu` with CUDA-bundled CUB/CCCL `cub::BlockReduce`.
- Kept the file out of CUTE/FlashAttention layout machinery. CUB is the smaller
  primitive for this scalar block reduction job.
- CUB block reductions return the aggregate on thread 0, so the wrapper keeps a
  single shared-float broadcast to preserve the old all-thread return behavior.

Validation:

```bash
make NVCC=/usr/local/cuda/bin/nvcc test-kv-cache
```

Result:

```text
kv cache tests passed
```

Quick benchmark, same warm-cache CUDA-event contract as the split sweep above:

```bash
make NVCC=/usr/local/cuda/bin/nvcc kv-cache-bench
./build/experiments/gemma4_kv_cache_bench 4096 64 64 5 20 5
```

Result:

```text
correctness max_abs=0.000244 mean_abs=0.000000
prefill_cache_write median_ms=0.104445 mean_ms=0.104372 min_ms=0.103626 max_ms=0.104725
decode_cache_write median_ms=0.017589 mean_ms=0.019281 min_ms=0.015480 max_ms=0.024451
paged_decode_attention median_ms=0.772422 mean_ms=0.772215 min_ms=0.771488 max_ms=0.772638
paged_full_decode_write_plus_attention median_ms=0.740006 mean_ms=0.743295 min_ms=0.736562 max_ms=0.755912
```

Conclusion:

- The primitive version is simpler and did not regress the quick benchmark.
- The quick `paged_decode_attention` median improved versus the prior split-64
  quick run (`0.790414 ms` -> `0.772422 ms`), but this is not a locked-clock
  claim.

## 2026-06-16 - Paged KV cache thread I/O primitives

Scope:

- Replaced raw `int4` loads/stores in the vectorized KV-cache write kernel with
  CUB/CCCL `cub::ThreadLoad<cub::LOAD_LDG>` and
  `cub::ThreadStore<cub::STORE_CG>`.
- Kept `loadg` for scalar BF16 attention loads because the project helper is
  already the smallest useful primitive there.
- Rejected CUB block load/store for this pass: each active thread already moves
  one 128-bit vector, so block-level staging would add temp storage and
  ceremony without simplifying the code.

Validation:

```bash
make NVCC=/usr/local/cuda/bin/nvcc test-kv-cache
```

Result:

```text
kv cache tests passed
```

Quick benchmark, same warm-cache CUDA-event contract:

```bash
make NVCC=/usr/local/cuda/bin/nvcc kv-cache-bench
./build/experiments/gemma4_kv_cache_bench 4096 64 64 5 20 5
```

Result:

```text
correctness max_abs=0.000244 mean_abs=0.000000
prefill_cache_write median_ms=0.104182 mean_ms=0.105215 min_ms=0.103317 max_ms=0.109918
decode_cache_write median_ms=0.015653 mean_ms=0.017941 min_ms=0.013117 max_ms=0.030126
paged_decode_attention median_ms=0.772646 mean_ms=0.773900 min_ms=0.771898 max_ms=0.779840
paged_full_decode_write_plus_attention median_ms=0.736677 mean_ms=0.746361 min_ms=0.735594 max_ms=0.775341
```

Conclusion:

- The thread I/O primitive makes the write cache policy explicit and did not
  regress the quick run.
- The larger primitive candidates left on the table are not good fits yet:
  CuTe/CUTLASS layout primitives are useful for tensor-core tiled attention,
  not scalar paged gather, and CUB block load/store would be more code here.

## 2026-06-16 - Paged KV primitive research pass

Scope:

- Used Exa plus read-only subagents to scan CUDA, CCCL/CUB, CuTE/CUTLASS,
  cuBLAS/cuDNN, FlashInfer, vLLM, TensorRT-LLM, SGLang, flash-attn, and xFormers
  for primitives that could simplify or speed up the paged KV path.
- Kept the current CUB `BlockReduce` and CUB `ThreadLoad`/`ThreadStore` changes
  from the previous pass.
- Tested two tiny CUDA-native changes and rejected both when benchmark evidence
  did not support keeping them:
  - `__grid_constant__` for the small `Gemma4KvCacheConfig` kernel parameter;
  - log2-domain `exp2f` softmax in the paged decode path.

Useful references found:

- CUDA `__grid_constant__` parameters avoid per-thread copies for const kernel
  parameters, but the benefit depends on how the compiler uses the small struct.
- CUDA `cuda::memcpy_async`, `cooperative_groups::memcpy_async`, and CuTE
  `Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>>` are good when staging
  global-memory tiles through shared memory, but not for the current
  global-to-global KV write or scalar paged gather.
- CUB `BlockLoad`, `BlockStore`, `WarpReduce`, `BlockScan`,
  `BlockExchange`, and `DeviceSegmentedReduce` are not drop-ins here:
  they either add temp storage/synchronization around an already simple 128-bit
  lane copy, or add extra launches where the kernel currently fuses the work.
- cuDNN paged SDPA is a useful future baseline, but Gemma global attention uses
  head dim `512` on Ampere/A6000, which is outside the documented practical
  cuDNN/FlashAttention-style limits for many current kernels.
- FlashInfer/vLLM/TensorRT-LLM/SGLang agree on the higher-value structural
  work: CSR-style page metadata, LSE merge state, page/block-tiled decode,
  GQA/XQA reuse, separate sliding/global pools, and later FP8 cache storage.

Baseline before this pass, warm-cache CUDA-event quick contract:

```bash
./build/experiments/gemma4_kv_cache_bench 4096 64 64 10 100 5
```

```text
correctness max_abs=0.000244 mean_abs=0.000000
prefill_cache_write median_ms=0.102862 mean_ms=0.102869 min_ms=0.102813 max_ms=0.102936
decode_cache_write median_ms=0.016253 mean_ms=0.016768 min_ms=0.015819 max_ms=0.018959
paged_decode_attention median_ms=0.735881 mean_ms=0.743909 min_ms=0.731225 max_ms=0.770315
paged_full_decode_write_plus_attention median_ms=0.736858 mean_ms=0.737554 min_ms=0.736727 max_ms=0.738781
```

`__grid_constant__` plus log2/`exp2f` trial:

```text
correctness max_abs=0.000244 mean_abs=0.000000
prefill_cache_write median_ms=0.102947 mean_ms=0.103017 min_ms=0.102834 max_ms=0.103350
decode_cache_write median_ms=0.013807 mean_ms=0.012778 min_ms=0.009708 max_ms=0.014800
paged_decode_attention median_ms=0.741351 mean_ms=0.742217 min_ms=0.736021 max_ms=0.747805
paged_full_decode_write_plus_attention median_ms=0.743927 mean_ms=0.743703 min_ms=0.742481 max_ms=0.744246
```

`__grid_constant__` only on the cache-write kernels, longer contract:

```bash
./build/experiments/gemma4_kv_cache_bench 4096 64 64 25 100 15
```

```text
correctness max_abs=0.000244 mean_abs=0.000000
prefill_cache_write median_ms=0.103114 mean_ms=0.104676 min_ms=0.102901 max_ms=0.115319
decode_cache_write median_ms=0.017865 mean_ms=0.018163 min_ms=0.011573 max_ms=0.036119
paged_decode_attention median_ms=0.771727 mean_ms=0.771888 min_ms=0.756163 max_ms=0.780257
paged_full_decode_write_plus_attention median_ms=0.770110 mean_ms=0.770526 min_ms=0.765271 max_ms=0.775802
```

Validation:

```bash
make NVCC=/usr/local/cuda/bin/nvcc test-kv-cache
```

```text
kv cache tests passed
```

Conclusion:

- Rejected both new micro-edits. They are valid CUDA ideas, but not measured wins
  for this exact kernel shape.
- Keep the current primitive level: CUB for block reductions and thread I/O,
  project `loadg` helpers for scalar BF16 loads, and explicit cache/page math.
- Next code worth writing is not another local primitive swap. It is a real
  decode design step: CSR page tables, LSE state, page-tiled split-KV, and
  GQA-grouped global decode.

## 2026-06-16 - KV cache benchmark contract tightening

Scope:

- Upgraded the C++ paged-KV benchmark harness to report:
  median, mean, 10% trimmed mean, min, max, p95, p99, stddev, IQR, and raw
  sample arrays.
- Added explicit `--cache warm|cold` handling.
- Added cold-cache timing via a 64 MiB L2-flush kernel. The flush is launched on
  the same stream before each measured iteration, and the timed CUDA start event
  is recorded after the flush, so reported kernel time excludes flush overhead.
- Upgraded the PyTorch SDPA comparator to report the same tail stats and raw
  samples, with matching warm/cold cache modes.
- Persisted raw-sample outputs under `src/experiments/results/`.

Clock and environment controls:

```bash
sudo -n nvidia-smi -pm 1
sudo -n nvidia-smi -ac 8001,1800
sudo -n nvidia-smi -lgc 1800,1800
```

Result:

- Persistence mode was already enabled.
- Both clock-lock attempts failed with:
  `The current user does not have permission to change clocks`.
- `nvidia-smi dmon` is not supported in this environment, so a one-second
  `nvidia-smi --query-gpu` loop was used for observed telemetry.
- Observed load clocks were not locked; they floated around `1800-1935 MHz` SM
  and `7601 MHz` memory, with power up to roughly the `300 W` cap in parts of
  the run and temperatures up to about `61 C`.
- `nvidia-smi` bus-id reporting varied between point queries despite reporting a
  single RTX A6000. Treat this as a provider/virtualization telemetry wrinkle
  and do not compare against other machines without rerunning.

Validation:

```bash
make NVCC=/usr/local/cuda/bin/nvcc kv-cache-bench
make NVCC=/usr/local/cuda/bin/nvcc test-kv-cache
```

Result:

```text
kv cache tests passed
```

Benchmark contract:

- Hardware: NVIDIA RTX A6000, SM86, 50.90 GB.
- Driver/runtime: CUDA driver/runtime `13000`; `nvidia-smi` driver
  `580.126.16`.
- Timing: CUDA events on the same stream; host wall time excluded.
- Scope: single-process typical kernel microbenchmark.
- Shape: global Gemma decode, `B=1`, `S=4096`, `Q heads=32`,
  `KV heads=4`, `D=512`, BF16, page size `64`, split size `64`.
- Counts: `warmup=50`, `iters_per_sample=100`, `samples=31`.
- Cold-cache mode: 64 MiB L2 flush before each measured iteration.
- Minimum effect size for claims: `5%`; closer results need repeated
  process-level runs under locked clocks.

Persisted outputs:

```text
src/experiments/results/2026-06-16_kv_cache_custom_warm.txt
src/experiments/results/2026-06-16_kv_cache_custom_cold.txt
src/experiments/results/2026-06-16_kv_cache_torch_warm.json
src/experiments/results/2026-06-16_kv_cache_torch_cold.json
```

Custom paged KV, warm cache:

```text
prefill_cache_write                 median=0.103031 ms p95=0.103359 ms p99=0.103493 ms
decode_cache_write                  median=0.014139 ms p95=0.018111 ms p99=0.019492 ms
paged_decode_attention              median=0.734263 ms p95=0.737304 ms p99=0.739022 ms
paged_full_decode_write_plus_attention median=0.737860 ms p95=0.740120 ms p99=0.740949 ms
```

Custom paged KV, cold cache with L2 flush:

```text
prefill_cache_write                 median=0.104614 ms p95=0.105390 ms p99=0.108006 ms
decode_cache_write                  median=0.004984 ms p95=0.006706 ms p99=0.010823 ms
paged_decode_attention              median=0.712841 ms p95=0.719988 ms p99=0.733747 ms
paged_full_decode_write_plus_attention median=0.721089 ms p95=0.723958 ms p99=0.724766 ms
```

PyTorch SDPA comparator, warm cache:

```text
attention_only                      median=3.306480 ms p95=3.310470 ms p99=3.319482 ms
full_decode_write_plus_attention    median=3.315389 ms p95=3.318946 ms p99=3.339498 ms
```

PyTorch SDPA comparator, cold cache with L2 flush:

```text
attention_only                      median=3.371345 ms p95=3.463470 ms p99=3.473572 ms
full_decode_write_plus_attention    median=3.509788 ms p95=3.537339 ms p99=3.554004 ms
```

Conclusion:

- The benchmark now follows the GPU benchmark checklist much more closely:
  correctness first, explicit contract, raw samples, p95/p99, warm/cold cache
  separation, L2 flush, and environment telemetry.
- The main missing control is clock locking, which this environment denies.
- Warm-cache median speedup versus PyTorch SDPA is about `4.50x` for
  attention-only (`3.306480 / 0.734263`) and `4.49x` for full decode
  (`3.315389 / 0.737860`), with the same caveat as before: PyTorch uses
  contiguous K/V, not paged K/V.
- The cold-cache custom numbers are not directly comparable to the warm-cache
  batched tiny-write numbers because cold mode records per-iteration CUDA events
  after each flush. Use the attention/full-decode rows as the more meaningful
  comparison points.

## 2026-06-17 - Sliding decode Q prep plus paged KV cache write

Scope:

- Added a decode-only fused prep-cache path in the flash-attn file:
  raw one-token Q/K/V -> prepared Q plus Layout-A paged K/V cache write.
- Q and K use the existing sliding learned RMSNorm + RoPE device helper.
- V uses the existing sliding scale-free RMSNorm helper.
- This does not make prefill FA consume paged K/V; it is the decode bridge
  before `gemma4_paged_decode_attention_bf16`.

Validation:

```bash
make test-kv-cache NVCC=/usr/local/cuda/bin/nvcc
make flash-attn-bench NVCC=/usr/local/cuda/bin/nvcc
```

Result:

```text
kv cache tests passed
```

Benchmark command:

```bash
./build/experiments/gemma4_flash_attention_bench 1024 20 20 30 1 64 cold 128
```

Benchmark contract:

- Hardware: NVIDIA RTX A6000, driver `580.126.16`, persistence mode enabled.
- Compiler: `/usr/local/cuda/bin/nvcc`, CUDA `13.0`, target `sm_86`, flags
  `--expt-relaxed-constexpr --expt-extended-lambda --use_fast_math`.
- Timing: CUDA events on the same stream; launch overhead included.
- Cache: cold-L2 mode, 128 MiB flush before each measured iteration; timed
  event starts after the flush.
- Shape: sliding attention `B=1`, `S=1024`, `Q heads=32`, `KV heads=16`,
  `D=256`, BF16, decode cache page size `64`.
- Counts: `warmup=20`, `iters_per_sample=20`, `samples=30`.
- Correctness: benchmark precheck at `seq=64`; max attention abs diff
  `0.015625`; prep Q/K/V max abs diffs `0.00195312`, `0.0078125`,
  `0.000976562`.
- Clock policy: attempted `nvidia-smi -ac 8001,2100` and
  `sudo -n nvidia-smi -ac 8001,2100`; both were denied by driver permission.

Cold-cache results:

```text
norm_rope_plus_fa              median=0.320997 ms p95=0.336466 ms p99=0.336648 ms
decode_norm_rope_paged_kv_write median=0.006767 ms p95=0.011795 ms p99=0.032069 ms
```

Conclusion:

- The new decode prep-cache kernel is a tiny operation relative to prefill FA:
  about `6.77 us` median for `B=1`.
- Tail samples had two visible outliers (`0.0155 ms`, `0.0389 ms`), so do not
  treat p99 as stable until clocks can be locked or the run is repeated across
  processes.

## 2026-06-17 - Minimal PyTorch decode prep-cache comparator

Scope:

- Added a minimal eager PyTorch implementation for the same decode prep-cache
  contract:
  Q RMSNorm + RoPE, K RMSNorm + RoPE, V RMSNorm, then Layout-A paged K/V write.
- The script also calls the real custom CUDA function through `ctypes`, so the
  custom side is not reimplemented in Python.
- This measures the prep-cache bridge only, not paged decode attention.

Build:

```bash
make flash-attn-lib NVCC=/usr/local/cuda/bin/nvcc
```

Clock controls:

```bash
nvidia-smi -ac 8001,2100
sudo -n nvidia-smi -ac 8001,2100
```

Both clock-lock attempts failed with:

```text
The current user does not have permission to change clocks
```

Benchmark commands:

```bash
python3 src/experiments/gemma4_decode_prep_torch_bench.py \
  --cache warm --warmup 20 --iters 50 --samples 30 \
  --output src/experiments/results/2026-06-17_decode_prep_torch_warm.json

python3 src/experiments/gemma4_decode_prep_torch_bench.py \
  --cache cold --flush-mib 128 --warmup 20 --iters 50 --samples 30 \
  --output src/experiments/results/2026-06-17_decode_prep_torch_cold.json
```

Benchmark contract:

- Hardware: NVIDIA RTX A6000, driver `580.126.16`, persistence mode enabled.
- PyTorch: `2.11.0+cu130`, CUDA runtime `13.0`.
- Timing: CUDA events on the current PyTorch CUDA stream.
- Cache: warm and cold runs are separate; cold uses a 128 MiB device-buffer
  add before each measured iteration, outside the timed event window.
- Shape: `B=1`, `seq_len=1024`, `page_size=64`, `Q heads=32`, `KV heads=16`,
  `head_dim=256`, BF16.
- Counts: `warmup=20`, `iters_per_sample=50`, `samples=30`.
- Correctness: custom and PyTorch outputs matched exactly for Q, cache K, and
  cache V on this generated input (`max_abs=0` for all three).
- Caveat: Python eager dispatch can create stream gaps for microsecond kernels,
  so the C++ custom-only timing remains the cleaner number for the custom
  kernel itself. This comparison is still useful because it times a minimal
  PyTorch implementation of the same behavior.

Warm-cache results:

```text
custom_decode_norm_rope_paged_kv_write      median=0.031335 ms p95=0.061114 ms p99=0.078949 ms
torch_eager_decode_norm_rope_paged_kv_write median=3.060149 ms p95=3.817697 ms p99=3.874087 ms
median speedup = 97.66x
```

Cold-cache results:

```text
custom_decode_norm_rope_paged_kv_write      median=0.011273 ms p95=0.030585 ms p99=0.036598 ms
torch_eager_decode_norm_rope_paged_kv_write median=3.098637 ms p95=3.452670 ms p99=3.605872 ms
median speedup = 274.87x
```

Conclusion:

- Against a minimal eager PyTorch implementation of the same prep-cache work,
  the fused CUDA path is roughly `98x` faster warm-cache and `275x` faster
  cold-cache in this Python harness.
- The cold-cache custom median aligns much better with the C++ custom benchmark
  scale (`~11.3 us` here vs `~6.8 us` in C++), while the warm Python custom
  timing appears inflated by dispatch/stream-gap noise.

## 2026-06-17 - Non-eager PyTorch decode prep-cache graph replay

Scope:

- Reworked the PyTorch comparator so the timed path is no longer eager Python
  dispatch.
- The PyTorch prep-cache function is optionally compiled with
  `torch.compile(mode="reduce-overhead")`.
- Both the custom CUDA path and the PyTorch path are captured into explicit
  `torch.cuda.CUDAGraph` objects before timing.
- Warm-cache timing captures `iters_per_sample` prep-cache operations inside
  one graph replay and divides by the number of operations.
- Cold-cache timing captures one prep-cache operation per graph replay, flushes
  L2 before each timed replay, and divides across repeated replays.

Timing fixes:

- CUDA events are recorded on the active PyTorch CUDA stream.
- Host wall time and Python launch overhead are excluded from elapsed time.
- Warm-cache timing queues an untimed `512x512` FP32 matmul before the start
  event so the CPU cannot outrun the GPU before enqueuing the stop event.
- Cold-cache timing uses the L2 flush before the start event as the queue
  backlog.
- L2 flush uses `zero_()` on a 128 MiB device buffer, larger than the RTX A6000
  L2 cache.

Clock controls:

```bash
nvidia-smi -ac 8001,2100
sudo -n nvidia-smi -ac 8001,2100
```

Both clock-lock attempts failed with:

```text
The current user does not have permission to change clocks
```

Benchmark commands:

```bash
python3 src/experiments/gemma4_decode_prep_torch_bench.py \
  --cache warm --warmup 25 --iters 100 --samples 31 \
  --output src/experiments/results/2026-06-17_decode_prep_torch_graph_warm.json

python3 src/experiments/gemma4_decode_prep_torch_bench.py \
  --cache cold --flush-mib 128 --warmup 25 --iters 100 --samples 31 \
  --output src/experiments/results/2026-06-17_decode_prep_torch_graph_cold.json
```

Benchmark contract:

- Hardware: NVIDIA RTX A6000, driver `580.126.16`, persistence mode enabled.
- PyTorch: `2.11.0+cu130`, CUDA runtime `13.0`.
- Shape: `B=1`, `seq_len=1024`, `page_size=64`, `Q heads=32`, `KV heads=16`,
  `head_dim=256`, BF16.
- Correctness: custom and PyTorch outputs matched exactly for Q, cache K, and
  cache V on this generated input (`max_abs=0` for all three).
- Inductor note: PyTorch printed a warning that its internal cudagraphs were
  skipped because the compiled function mutates output buffers. The benchmark
  still captures the resulting compiled work in an explicit outer CUDA graph.

Warm-cache graph replay:

```text
custom_cuda_graph_decode_norm_rope_paged_kv_write median=0.003058 ms p95=0.003924 ms p99=0.005008 ms
torch_non_eager_decode_norm_rope_paged_kv_write   median=0.109721 ms p95=0.110382 ms p99=0.110422 ms
median speedup = 35.88x
```

Cold-cache graph replay:

```text
custom_cuda_graph_decode_norm_rope_paged_kv_write median=0.022212 ms p95=0.042371 ms p99=0.045316 ms
torch_non_eager_decode_norm_rope_paged_kv_write   median=0.125388 ms p95=0.140934 ms p99=0.153021 ms
median speedup = 5.65x
```

Conclusion:

- The eager PyTorch comparator overstated the speedup because it included many
  separate eager operations on the GPU timeline.
- The corrected non-eager graph-replay comparison is still a clear win:
  `35.9x` warm-cache and `5.65x` cold-cache at `B=1`.
- The cold custom graph timing is higher than the C++ custom-only kernel timing
  because each cold sample replays a one-op CUDA graph after flushing L2. Use
  this row for fair custom-vs-PyTorch graph comparison, and the C++ benchmark
  for the cleanest custom kernel-only number.

## 2026-06-17 - Decode prep-cache graph replay with torch.cuda._sleep

Scope:

- Replaced the warm-cache queue-saturation dummy matmul with
  `torch.cuda._sleep(1_000_000)`.
- Cold-cache timing now also does `flush_l2(); torch.cuda._sleep(1_000_000);`
  before recording the start event.
- The sleep is untimed: it is enqueued before `start.record()`.
- A fallback dummy matmul remains in the script only for PyTorch builds without
  the private `_sleep` API.

Benchmark commands:

```bash
python3 src/experiments/gemma4_decode_prep_torch_bench.py \
  --cache warm --warmup 25 --iters 100 --samples 31 \
  --sleep-cycles 1000000 \
  --output src/experiments/results/2026-06-17_decode_prep_torch_graph_sleep_warm.json

python3 src/experiments/gemma4_decode_prep_torch_bench.py \
  --cache cold --flush-mib 128 --warmup 25 --iters 100 --samples 31 \
  --sleep-cycles 1000000 \
  --output src/experiments/results/2026-06-17_decode_prep_torch_graph_sleep_cold.json
```

Clock controls:

```bash
nvidia-smi -ac 8001,2100
sudo -n nvidia-smi -ac 8001,2100
```

Both clock-lock attempts failed with:

```text
The current user does not have permission to change clocks
```

Benchmark contract:

- Hardware: NVIDIA RTX A6000, driver `580.126.16`, persistence mode enabled.
- PyTorch: `2.11.0+cu130`, CUDA runtime `13.0`.
- Timing: CUDA events on the current PyTorch CUDA stream.
- Execution: explicit CUDA graph replay for both custom CUDA and compiled
  PyTorch work.
- Queue saturation: untimed `torch.cuda._sleep(1_000_000)` before start event.
- Cache: warm and cold runs are separate; cold uses a 128 MiB L2 flush before
  the untimed sleep and start event.
- Shape: `B=1`, `seq_len=1024`, `page_size=64`, `Q heads=32`, `KV heads=16`,
  `head_dim=256`, BF16.
- Correctness: custom and PyTorch outputs matched exactly for Q, cache K, and
  cache V on this generated input (`max_abs=0` for all three).

Warm-cache graph replay with `_sleep`:

```text
custom_cuda_graph_decode_norm_rope_paged_kv_write median=0.002949 ms p95=0.002999 ms p99=0.003113 ms
torch_non_eager_decode_norm_rope_paged_kv_write   median=0.109596 ms p95=0.109913 ms p99=0.112374 ms
median speedup = 37.16x
```

Cold-cache graph replay with `_sleep`:

```text
custom_cuda_graph_decode_norm_rope_paged_kv_write median=0.007511 ms p95=0.134406 ms p99=0.285987 ms
torch_non_eager_decode_norm_rope_paged_kv_write   median=0.114130 ms p95=0.129568 ms p99=0.176282 ms
median speedup = 15.20x
```

Conclusion:

- `torch.cuda._sleep(1_000_000)` gives a cleaner warm-cache timing path than
  the dummy matmul and avoids polluting cache state with matmul operands.
- Warm-cache custom timing is now very stable at about `2.95 us`; the compiled
  PyTorch graph replay is about `109.6 us`.
- Cold-cache median improved, but custom cold tails had large outliers because
  clocks are unlocked and each sample repeats flush/sleep/replay many times.
  Use the median for the headline comparison and keep p95/p99 visible.

## 2026-06-17 - FlashAttention KV block loop fuse cleanup

Scope:

- Factored the duplicated masking/steady K/V loop body in
  `src/gemma4_flash_attention.cu` into a templated `gemma4_process_kv_block`
  helper.
- Kept masking compile-time-specialized through `MaybeMask`; the global
  steady-loop instantiation uses `MaybeMask=false`.
- Unified `softmax_rescale_o` so the first-vs-later block distinction is just
  O/denominator rescaling plus row-sum initialization.
- Follow-up cleanup folded `softmax_rescale_visible` into
  `softmax_rescale<IsFirst, MaybeMask>` while keeping the `CheckInf` branch as a
  compile-time specialization through `softmax_rescale_impl`.
- Folded causal and local block-visibility helpers into the single templated
  `gemma4_score_block_fully_visible<IsLocal>` helper.
- Reused one O-output path and one LSE-row writer for the empty-block path and
  normal epilogue.
- Removed dead FA helper generality: `ScaleMax`, `AInRegs`/`BInRegs`, and the
  unpredicated `gemma4_fa_copy` wrapper.

Build and SASS/resource commands:

```bash
/usr/local/cuda/bin/nvcc -std=c++17 -O3 -arch=sm_86 \
  --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math \
  -D_GLIBCXX_USE_CXX11_ABI=1 -Isrc \
  -Iexperiments/flash-attention/csrc/cutlass/include -Xptxas=-v \
  -c src/gemma4_flash_attention.cu \
  -o build/ptx/gemma4_flash_attention_after_loop_fuse.o

/usr/local/cuda/bin/cuobjdump --dump-sass \
  build/ptx/gemma4_flash_attention_after_loop_fuse.o \
  > build/ptx/gemma4_flash_attention_after_loop_fuse.sass
```

ptxas check versus the pre-refactor object:

```text
global D512 ReturnLse=true:  255 regs, stack 744 -> 704 B, spills 1512/1696 -> 1488/1664 B
global D512 ReturnLse=false: 255 regs, stack 728 -> 704 B, spills 1476/1652 -> 1436/1604 B
sliding D256 ReturnLse=true:  245 -> 244 regs, 0 spills
sliding D256 ReturnLse=false: 244 -> 245 regs, 0 spills
```

Coarse SASS instruction-count check:

```text
global D512 ReturnLse=true:  4096 -> 4096 instructions
global D512 ReturnLse=false: 4096 -> 4096 instructions
sliding D256 ReturnLse=true:  3560 -> 3824 instructions
sliding D256 ReturnLse=false: 3488 -> 3704 instructions
```

Validation:

```bash
make -B flash-attn-bench NVCC=/usr/local/cuda/bin/nvcc
./build/experiments/gemma4_flash_attention_bench 1024 20 5 3 1 64
make flash-attn-lib test-kv-cache NVCC=/usr/local/cuda/bin/nvcc
make -B flash-attn-lib NVCC=/usr/local/cuda/bin/nvcc
```

Quick benchmark/correctness result:

```text
correctness seq=64 max_abs=0.015625 mean_abs=0.000260142 max_rel=0.00775194
no_lse_correctness max_abs=0.015625 mean_abs=0.000260142 max_rel=0.00775194
norm_rope_prep_correctness q_max_abs=0.00195312 k_max_abs=0.0078125 v_max_abs=0.000976562
norm_rope_plus_fa median_ms=0.335512 samples=3 warm-cache, launch included
decode_norm_rope_paged_kv_write median_ms=0.0138224 samples=3 warm-cache, launch included
kv cache tests passed
```

Conclusion:

- The global D=512 path preserved the coarse SASS instruction count and did not
  increase register pressure; stack/spill bytes improved slightly.
- Sliding stayed spill-free. Its coarse SASS text grew because the local
  steady-loop still carries the compile-time `MaybeMask=true` window-edge path.
- The `softmax_rescale_visible` fold matched the pre-follow-up ptxas resource
  profile exactly; only ptxas compile-time timings changed.
- The visibility-helper fold kept sliding registers/spills unchanged and reduced
  global spill stores/loads by 4 bytes in the ptxas check.
- The quick correctness-bearing benchmark passed; this run was not a locked-clock
  performance claim.

## 2026-06-17 - Sliding paged decode attention CUDA-core path

Scope:

- Added `gemma4_flash_attention_sliding_decode_paged_bf16`, a sliding-only
  q_len=1 paged decode attention path that consumes Layout-A paged K/V directly.
- The split kernel is CUDA-core based, not tensor-core based. One CTA owns one
  KV head plus its two sliding GQA query heads, so K/V vectors are loaded once
  for the pair instead of once per query head.
- Kept the existing `gemma4_paged_decode_attention_bf16` implementation as the
  simple reference baseline.
- Added `gemma4_flash_attention_sliding_decode_paged_cp_async_bf16` as an
  explicit cp.async ablation. It stages one K/V vector at a time through shared
  memory with immediate wait; it is benchmarked separately rather than chosen
  implicitly.
- Exported `gemma4_kv_cache_write_bf16` with C ABI so Python graph benchmarks
  can populate the same CUDA KV cache layout used by the C++ path.

Validation:

```bash
make -B test-kv-cache NVCC=/usr/local/cuda/bin/nvcc
python3 -m py_compile src/experiments/gemma4_paged_decode_torch_bench.py
make -B flash-attn-lib NVCC=/usr/local/cuda/bin/nvcc
make -B kv-cache-bench NVCC=/usr/local/cuda/bin/nvcc
```

Result:

```text
kv cache tests passed
```

Correctness coverage:

- Existing KV cache address, write, global paged attention, and sliding wrap
  tests still pass.
- New sliding paged decode cases compare direct and cp.async paths against both
  the existing CUDA paged baseline and the CPU reference for:
  - short context below one page;
  - page-boundary crossing;
  - sliding window slot wrap.
- PyTorch graph benchmark correctness:
  - decode direct vs PyTorch max abs: `0.000244140625`
  - decode cp.async vs PyTorch max abs: `0.000244140625`
  - decode direct vs cp.async max abs: `0`
  - prefill tensor-core FA vs PyTorch SDPA max abs: `0.001953125`

C++ old-vs-new benchmark commands:

```bash
./build/experiments/gemma4_kv_cache_bench \
  1024 64 64 5 10 3 --cache warm

./build/experiments/gemma4_kv_cache_bench \
  1024 64 64 3 5 3 --cache cold --flush-bytes 134217728
```

C++ benchmark contract:

- Hardware: NVIDIA RTX A6000, driver/runtime reported as CUDA `13.0`.
- Compiler: `/usr/local/cuda/bin/nvcc`, target `sm_86`, flags include
  `-O3`, `--expt-relaxed-constexpr`, `--expt-extended-lambda`,
  `--use_fast_math`.
- Shape: sliding decode, `B=1`, `seq_len=1024`, `key_count=1024`,
  `page_size=64`, `split_size=64`, `num_splits=16`, `Q heads=32`,
  `KV heads=16`, `head_dim=256`, BF16.
- Timing: CUDA events on the benchmark stream. Warm run batches ten launches per
  sample; cold run flushes 128 MiB before each measured iteration.
- Clock policy: not locked in this quick run.

C++ warm-cache results:

```text
paged_decode_attention                 median=0.140064 ms
flash_decode_paged_attention_direct    median=0.100266 ms
flash_decode_paged_attention_cp_async  median=0.099850 ms
paged_full_decode_write_plus_attention median=0.143840 ms
flash_full_decode_write_plus_attention median=0.104509 ms
```

C++ cold-cache results:

```text
paged_decode_attention                 median=0.139514 ms
flash_decode_paged_attention_direct    median=0.099296 ms
flash_decode_paged_attention_cp_async  median=0.099277 ms
paged_full_decode_write_plus_attention median=0.141274 ms
flash_full_decode_write_plus_attention median=0.101690 ms
```

PyTorch graph benchmark commands:

```bash
python3 src/experiments/gemma4_paged_decode_torch_bench.py \
  --seq-len 1024 --prefill-seq-len 64 --page-size 64 --split-size 64 \
  --warmup 5 --iters 10 --samples 3 --cache warm --sample-delay-s 1.0 \
  --output src/experiments/results/2026-06-17_paged_decode_torch_graph_warm.json

python3 src/experiments/gemma4_paged_decode_torch_bench.py \
  --seq-len 1024 --prefill-seq-len 64 --page-size 64 --split-size 64 \
  --warmup 5 --iters 5 --samples 3 --cache cold --flush-mib 128 \
  --sample-delay-s 1.0 \
  --output src/experiments/results/2026-06-17_paged_decode_torch_graph_cold.json
```

PyTorch graph benchmark contract:

- Hardware: NVIDIA RTX A6000.
- PyTorch: `2.11.0+cu130`, CUDA runtime `13.0`.
- Timing: CUDA events on the current PyTorch CUDA stream.
- Execution: explicit CUDA graph replay for both custom CUDA and PyTorch paths.
- Delay: host `sleep(1.0)` before each measured sample, outside the CUDA event
  window, per request.
- Decode q_len=1 uses the new CUDA-core paged path and a PyTorch graph over the
  equivalent BF16 GQA attention. The cache is filled with
  `gemma4_kv_cache_write_bf16`.
- q_len>1 uses the existing tensor-core contiguous sliding FA path at
  `prefill_seq_len=64` and compares against PyTorch SDPA.
- Warm cache uses `graph_inner_iters=10`. Cold cache uses one graph op per
  replay and flushes 128 MiB before each measured replay.

PyTorch warm-cache graph results:

```text
decode_custom_direct       median=0.097398 ms
decode_custom_cp_async     median=0.097398 ms
decode_torch_graph         median=0.674688 ms
prefill_custom_tensor_core median=0.009715 ms
prefill_torch_sdpa_graph   median=0.022365 ms
```

PyTorch cold-cache graph results:

```text
decode_custom_direct       median=0.097946 ms
decode_custom_cp_async     median=0.097766 ms
decode_torch_graph         median=0.679328 ms
prefill_custom_tensor_core median=0.019296 ms
prefill_torch_sdpa_graph   median=0.021574 ms
```

Conclusion:

- The GQA-aware CUDA-core paged decode kernel is a clear improvement over the
  old simple paged baseline at this shape: roughly `1.39-1.40x` faster for
  attention-only and about `1.37-1.39x` faster for cache-write-plus-attention.
- Against the PyTorch CUDA-graph decode implementation, the custom paged decode
  path is about `6.9x` faster at the median for both warm and cold runs.
- The cp.async ablation is effectively tied with direct global loads in these
  quick runs. It did not demonstrate a meaningful win, which matches the risk
  that one-use K/V staging through shared memory adds traffic without enough
  overlap. Keep it as a measured ablation for now, not as the default.
- The q_len>1 tensor-core custom FA path remains faster than PyTorch SDPA in the
  warm graph run (`~2.3x` median at `S=64`), but the cold run is much closer
  (`~1.12x`) and only used three samples. Do not make a broad prefill claim from
  this small graph run.
- Clocks were not locked and sample counts were intentionally small; repeat with
  more samples and clock controls before claiming small deltas, especially for
  direct-vs-cp.async.

## 2026-06-18 - Sliding decode cp.async cache-policy sweep

Question:

- For the sliding paged decode cp.async ablation, test whether `.cg` and/or
  `.L2::128B` beats the original `.ca` copy.

Commands:

```bash
for spec in "0 ca" "1 cg" "2 ca_l2_128" "3 cg_l2_128"; do
  set -- $spec
  make -B build/experiments/gemma4_kv_cache_bench \
    NVCC=/usr/local/cuda/bin/nvcc \
    NVCCFLAGS="-std=c++17 -O3 -arch=sm_86 -DGEMMA4_SLIDING_DECODE_CP_ASYNC_CACHE_POLICY=$1"
  sleep 1
  ./build/experiments/gemma4_kv_cache_bench 4096 64 64 20 100 12 --cache warm \
    | tee "src/experiments/results/2026-06-18_cp_async_${2}_warm.txt"
  sleep 1
done

for spec in "0 ca" "1 cg" "2 ca_l2_128" "3 cg_l2_128"; do
  set -- $spec
  make -B build/experiments/gemma4_kv_cache_bench \
    NVCC=/usr/local/cuda/bin/nvcc \
    NVCCFLAGS="-std=c++17 -O3 -arch=sm_86 -DGEMMA4_SLIDING_DECODE_CP_ASYNC_CACHE_POLICY=$1"
  sleep 1
  ./build/experiments/gemma4_kv_cache_bench 4096 64 64 10 20 8 --cache cold \
    | tee "src/experiments/results/2026-06-18_cp_async_${2}_cold.txt"
  sleep 1
done
```

Contract:

- Hardware: NVIDIA RTX A6000, driver `580.126.16`, CUDA runtime reported as
  `13.0`, persistence enabled, ECC disabled, power limit `300 W`.
- Shape: sliding decode, `B=1`, `seq_len=4096`, sliding `key_count=1024`,
  `page_size=64`, `split_size=64`, `num_splits=16`, BF16, `q_heads=32`,
  `kv_heads=16`, `head_dim=256`.
- Timing: CUDA events on the benchmark stream. Warm run batches 100 launches per
  sample for 12 samples. Cold run flushes 64 MiB before each measured iteration,
  20 iterations per sample for 8 samples.
- A host `sleep 1` separated policy runs. Clocks were not locked.

cp.async attention-only medians:

```text
policy              warm median      cold median
cp.async.ca         0.091628 ms      0.099900 ms
cp.async.cg         0.090102 ms      0.099613 ms
cp.async.ca.L2::128 0.091033 ms      0.099838 ms
cp.async.cg.L2::128 0.091144 ms      0.099797 ms
```

Decision:

- `.cg` was the best cp.async policy in both warm and cold medians.
- Neither `.L2::128B` variant earned its codepath; both were slower than plain
  `.cg`.
- The direct non-cp.async decode path remained faster than cp.async in this
  sweep (`~0.088-0.089 ms` warm, `~0.096-0.097 ms` cold), so cp.async remains
  an ablation path rather than the default.
- Code was pruned to keep only `cp.async.cg.shared.global` for the cp.async
  ablation and remove the `.ca`/`.L2::128B` policy switch.

## 2026-06-18 - Final sliding paged decode benchmark after pruning

Question:

- Benchmark the current `codex/flash-attn-cleanups` implementation after
  pruning cp.async to the `.cg` ablation only.

Commands:

```bash
make -B build/experiments/gemma4_kv_cache_bench \
  build/libgemma4_flash_attention.so NVCC=/usr/local/cuda/bin/nvcc

./build/experiments/gemma4_kv_cache_bench \
  4096 64 64 50 200 30 --cache warm \
  | tee src/experiments/results/2026-06-18_final_paged_decode_cpp_warm.txt

./build/experiments/gemma4_kv_cache_bench \
  4096 64 64 20 30 15 --cache cold --flush-bytes 67108864 \
  | tee src/experiments/results/2026-06-18_final_paged_decode_cpp_cold.txt

python3 src/experiments/gemma4_paged_decode_torch_bench.py \
  --seq-len 4096 --prefill-seq-len 64 --page-size 64 --split-size 64 \
  --warmup 10 --iters 20 --samples 5 --cache warm --sample-delay-s 1.0 \
  --output src/experiments/results/2026-06-18_final_paged_decode_torch_graph_warm.json

python3 src/experiments/gemma4_paged_decode_torch_bench.py \
  --seq-len 4096 --prefill-seq-len 64 --page-size 64 --split-size 64 \
  --warmup 10 --iters 10 --samples 5 --cache cold --flush-mib 128 \
  --sample-delay-s 1.0 \
  --output src/experiments/results/2026-06-18_final_paged_decode_torch_graph_cold.json
```

Contract:

- Commit: `3c80283`, branch `codex/flash-attn-cleanups`.
- Hardware: NVIDIA RTX A6000, bus `00000000:07:00.0`, driver `580.126.16`,
  persistence enabled, ECC disabled, power limit `300 W`.
- Clock policy: not locked. GPU was idle at start and after the run.
- C++ timing: CUDA events on the benchmark stream. Warm cache batches 200
  launches per sample for 30 samples. Cold cache flushes 64 MiB before each
  measured iteration, 30 iterations per sample for 15 samples.
- PyTorch timing: CUDA graphs with CUDA events on the current PyTorch stream,
  one-second host delay before each measured sample, warm and cold separated.
- Shape: sliding decode, `B=1`, `seq_len=4096`, sliding `key_count=1024`,
  `page_size=64`, `split_size=64`, `num_splits=16`, BF16, `q_heads=32`,
  `kv_heads=16`, `head_dim=256`. Prefill comparison uses `S=64`.

C++ CUDA-event medians:

```text
path                                warm median    cold median
baseline paged decode attention      0.128799 ms    0.129323 ms
flash paged decode direct            0.087743 ms    0.090304 ms
flash paged decode cp.async.cg       0.088860 ms    0.093365 ms
baseline write + attention           0.131306 ms    0.131628 ms
flash direct write + attention       0.090087 ms    0.092681 ms
```

PyTorch CUDA-graph medians:

```text
path                                warm median    cold median
custom decode direct                 0.093152 ms    0.095197 ms
custom decode cp.async.cg            0.095480 ms    0.098026 ms
PyTorch decode graph                 0.673456 ms    0.675603 ms
custom prefill tensor-core FA        0.008400 ms    0.015363 ms
PyTorch SDPA prefill graph           0.019472 ms    0.021949 ms
```

Correctness:

```text
decode direct vs PyTorch max_abs:   0.000244140625
decode cp.async vs PyTorch max_abs: 0.000244140625
decode direct vs cp.async max_abs:  0
prefill custom vs PyTorch max_abs:  0.001953125
```

Conclusion:

- Direct sliding paged decode is the keeper. It is `1.47x` faster than the old
  C++ paged baseline warm-cache and `1.43x` faster cold-cache.
- The `.cg` cp.async ablation is slower than direct in both C++ event timing and
  PyTorch graph timing, so it should stay as a measured ablation, not default.
- Against PyTorch CUDA graphs at this decode shape, direct custom decode is
  `7.23x` faster warm-cache and `7.10x` faster cold-cache.
- Remaining threats: clocks were not locked, no process-level reruns, and cold
  cache uses synthetic L2 flushes rather than an end-to-end serving trace.

## 2026-06-18 - Sliding paged decode actual-split cleanup

Goal: remove neutral scratch writes for overprovisioned sliding decode split
CTAs, then make the reducer consume only the live per-row split count while
preserving the `num_splits` scratch stride for graph-compatible max launches.

Build and test:

```bash
make -B test-kv-cache NVCC=/usr/local/cuda/bin/nvcc
make -B kv-cache-bench NVCC=/usr/local/cuda/bin/nvcc
```

Timing commands:

```bash
./build/experiments/gemma4_kv_cache_bench 1024 64 64 5 10 3 --cache warm
./build/experiments/gemma4_kv_cache_bench 1024 64 64 5 10 3 --cache warm --extra-splits 16
```

Contract:

- Hardware: NVIDIA RTX A6000, SM 86, driver/runtime reported as `13000`.
- Clock policy: not locked.
- Timing: CUDA events on the benchmark stream, warm cache, `warmup=5`,
  `iters_per_sample=10`, `samples=3`.
- Shape: `B=1`, `seq_len=1024`, `page_size=64`, `split_size=64`,
  sliding `key_count=1024`, BF16, `q_heads=32`, `kv_heads=16`,
  `head_dim=256`.
- Correctness: benchmark reported `max_abs=0` for baseline and direct flash
  paths after the cp.async decode ablation was removed. `test-kv-cache` also
  poisons partial scratch before the direct flash run, including exact split,
  partial final split, overprovisioned split, `seq_len < split_size`, wrap, and
  varlen batch cases.

Warm-cache medians:

```text
case                         baseline paged    flash direct
exact 16/16 splits              0.138067 ms      0.092544 ms
overprovision 16/32 splits      0.139456 ms      0.095152 ms
```

Pre-change exact-split comparison from the same session:

```text
flash direct       0.096211 ms
flash cp.async.cg  0.098582 ms
```

Conclusion:

- Correctness is unchanged, and stale overprovisioned partial slots are no
  longer required to contain neutral values.
- The split kernel now uses separate CUB temp storage for the two GQA query-head
  reductions, and the final reduce kernel uses separate temp storage for max and
  sum reductions.
- The stale decode cp.async ablation path was removed from the public ABI, tests,
  C++ benchmark, and PyTorch CUDA-graph benchmark. Prefill FlashAttention still
  uses its active CUTE cp.async path.
- Exact-split timing is effectively unchanged, as expected.
- Overprovisioned timing stays near exact-split timing for direct flash decode;
  the remaining cost is mostly launching empty split CTAs that now return before
  touching partial scratch.

## 2026-06-18 - Sliding decode baseline redundancy check

Question: can the old paged decode baseline be removed now that the sliding
paged decode FlashAttention-ish path exists?

Commands:

```bash
make -B kv-cache-bench NVCC=/usr/local/cuda/bin/nvcc

./build/experiments/gemma4_kv_cache_bench \
  1024 64 64 20 100 10 --cache warm \
  | tee src/experiments/results/2026-06-18_decode_baseline_vs_flash_warm.txt

./build/experiments/gemma4_kv_cache_bench \
  1024 64 64 10 30 8 --cache cold --flush-bytes 67108864 \
  | tee src/experiments/results/2026-06-18_decode_baseline_vs_flash_cold.txt
```

Contract:

- Hardware: NVIDIA RTX A6000, bus `00000000:0F:00.0`, driver `580.126.16`,
  persistence enabled, ECC disabled, power limit `300 W`.
- Clock policy: not locked. GPU was idle at start; benchmark reported runtime
  and driver as `13000`.
- Timing: CUDA events on the benchmark stream. Warm cache uses 20 warmups, 100
  launches per sample, 10 samples. Cold cache uses 10 warmups, 30 launches per
  sample, 8 samples, and a 64 MiB L2 flush.
- Shape: sliding decode, `B=1`, `seq_len=1024`, `page_size=64`,
  `split_size=64`, `actual_splits=16`, BF16, `q_heads=32`, `kv_heads=16`,
  `head_dim=256`.
- Correctness: both old paged decode and direct flash decode reported
  `max_abs=0` against the CPU reference before cleanup.

Medians:

```text
path                           warm median    cold median
old paged decode baseline       0.131868 ms    0.139665 ms
new flash paged decode direct   0.084152 ms    0.094680 ms
```

Conclusion:

- For sliding decode, the old paged decode baseline is redundant and slower:
  flash direct is `1.57x` faster warm-cache and `1.47x` faster cold-cache.
- Cleanup performed after measuring: removed the old baseline from the sliding
  flash decode correctness tests and from `gemma4_kv_cache_bench`'s active
  timing rows.
- The old generic paged decode implementation was not deleted outright because
  it is still the only paged global decode path and still backs small-layout
  KV-cache coverage. It should go away when global paged decode FA exists.

## 2026-06-18 - Projection decode GEMV helper deduplication

Question: can the decode projection GEMV kernels reuse the shared
`gemma4_matmul_device.cuh` helpers instead of carrying their own duplicate
pack-load, swizzle, reduction, and store code?

Change:

- Moved the existing `GEMMA4_DECODE_GEMV_BUFFER_STAGES` register-buffering
  experiment knob into `src/gemma4_matmul_device.cuh`.
- Replaced the local projection decode GEMV body in
  `src/gemma4_matmul_kernels.cu` with
  `gemma4_matmul_device::decode_gemv_cols_device`.
- Kept streaming weight loads for the projection decode translation unit,
  preserving the previous `.cs` behavior.
- Added `src/gemma4_matmul_device.cuh` to the projection object dependencies.
- Removed a tracked empty `src/src/gemma4_ffn_decode.cu` file and a tracked
  Python bytecode cache file.

Commands:

```bash
make -B decode-bench NVCC=/usr/local/cuda/bin/nvcc
GEMMA4_DECODE_BENCH_SEED=12345 ./build/experiments/gemma4_decode_bench 10 3 1

make -B test-ffn-decode NVCC=/usr/local/cuda/bin/nvcc
make -B test-cuda-utils NVCC=/usr/local/cuda/bin/nvcc

make -B decode-bench NVCC=/usr/local/cuda/bin/nvcc \
  CPPFLAGS='-Isrc -DGEMMA4_DECODE_GEMV_BUFFER_STAGES=2'
GEMMA4_DECODE_BENCH_SEED=12345 \
  ./build/experiments/gemma4_decode_bench global_k 5 2 1
make -B decode-bench NVCC=/usr/local/cuda/bin/nvcc
```

Contract:

- Hardware: NVIDIA RTX A6000, bus `00000000:04:00.0`, driver `580.126.16`,
  persistence enabled, ECC disabled, power limit `300 W`.
- CUDA/NVCC: CUDA compilation tools `13.0`, `V13.0.48`.
- Timing: existing C++ decode benchmark, CUDA events on the benchmark stream.
- Cache state: warm/repeated-buffer behavior from the harness; no L2 flush.
- Clock policy: not locked. GPU was idle in `nvidia-smi` before the pass, but
  clocks were free to boost during the benchmark.
- Iterations: one process, 10 timed iterations, 3 warmups, 1 trial per op.
- Correctness: compared against cuBLAS/cuDNN baselines as the harness reports;
  custom identity vs custom swizzle16 remained `max_abs=0` for every op. The
  staged-buffer smoke build also reported custom identity vs swizzle16
  `max_abs=0` for `global_k`.

Custom identity best-time comparison:

```text
op             before ms   after ms   delta
ffn_gate_up     0.653990   0.651894   -0.32%
ffn_down        0.329693   0.329066   -0.19%
sliding_qkv     0.251526   0.253366   +0.73%
sliding_o       0.129219   0.129066   -0.12%
global_q        0.250957   0.251210   +0.10%
global_k        0.035968   0.036800   +2.31%
global_o        0.253219   0.252666   -0.22%
final_logits    3.958743   3.960688   +0.05%
```

Conclusion:

- The refactor removes roughly 160 lines from the projection decode kernel file
  without a meaningful timing regression in this quick pass.
- The small `global_k` identity delta is on a sub-40us kernel and should be
  treated as benchmark noise unless reproduced with locked clocks and more
  trials.
- The moved staged-buffer path still compiles and remains numerically aligned
  with the swizzled custom output in the smoke run.

## 2026-06-18 - Sliding decode page-span cache addressing

Question: does the `docs/memory-movements-fa.md` recommendation to stop loading
the page table and recomputing page offsets once per token improve sliding paged
decode attention?

Change:

- Updated `sliding_decode_paged_grouped_split_kernel` to iterate by logical
  cache-page spans.
- Each touched page now performs one read-only page-table load and one full
  cache-base calculation.
- The inner token loop advances the K/V base pointer by the fixed
  `num_heads * head_dim` stride.
- Q, K, V, sequence lengths, and page table loads keep the same read-only global
  cache policy recommended in `docs/memory-movements-fa.md`; the change removes
  redundant metadata traffic and address math rather than adding shared-memory
  staging.

Commands:

```bash
make -B kv-cache-bench NVCC=/usr/local/cuda/bin/nvcc

./build/experiments/gemma4_kv_cache_bench \
  4096 64 64 50 200 20 --cache warm \
  | tee src/experiments/results/2026-06-18_fa_page_span_before_warm.txt

./build/experiments/gemma4_kv_cache_bench \
  4096 64 64 20 30 12 --cache cold --flush-bytes 67108864 \
  | tee src/experiments/results/2026-06-18_fa_page_span_before_cold.txt

make -B test-kv-cache NVCC=/usr/local/cuda/bin/nvcc
make -B kv-cache-bench NVCC=/usr/local/cuda/bin/nvcc

./build/experiments/gemma4_kv_cache_bench \
  4096 64 64 50 200 20 --cache warm \
  | tee src/experiments/results/2026-06-18_fa_page_span_after_warm.txt

./build/experiments/gemma4_kv_cache_bench \
  4096 64 64 20 30 12 --cache cold --flush-bytes 67108864 \
  | tee src/experiments/results/2026-06-18_fa_page_span_after_cold.txt
```

Contract:

- Hardware: NVIDIA RTX A6000, bus `00000000:0F:00.0`, driver `580.126.16`,
  persistence enabled, ECC disabled, power limit `300 W`.
- CUDA/NVCC: CUDA compilation tools `13.0`, `V13.0.48`.
- Timing: existing C++ KV-cache benchmark, CUDA events on the benchmark stream.
- Shape: sliding decode, `B=1`, `seq_len=4096`, `page_size=64`,
  `split_size=64`, sliding key count `1024`, `actual_splits=16`, BF16,
  `q_heads=32`, `kv_heads=16`, `head_dim=256`.
- Cache state: warm and cold measured separately. Cold uses a `67108864` byte
  L2 flush buffer, larger than the reported `6291456` byte L2.
- Launch overhead: included as queued launches on the GPU timeline; host wall
  time excluded.
- Clock policy: clocks were not locked; `nvidia-smi` showed the GPU idle after
  the run, so boost/thermal drift remains the main small-delta threat.
- Correctness: `test-kv-cache` passed. Benchmark correctness remained
  `max_abs=0.023438`, `mean_abs=0.000202`.

Median comparison:

```text
cache  path                                  before ms  after ms  delta    speedup
warm   flash_decode_paged_attention_direct   0.085141  0.064318  -24.46%  1.324x
warm   full write + attention                 0.088132  0.066948  -24.04%  1.316x
cold   flash_decode_paged_attention_direct   0.093894  0.071836  -23.49%  1.307x
cold   full write + attention                 0.096359  0.074256  -22.94%  1.298x
```

Conclusion:

- Page-span iteration is a clear win for this shape: roughly `23-24%` lower
  median attention time in both warm and cold cache conditions.
- The result is much larger than the declared `5%` minimum effect size, so it is
  robust enough to keep despite unlocked clocks.
- The remaining hot path is still dominated by K/V vector loads plus two
  block-wide reductions per token; the next cache-adjacent experiment should
  avoid further metadata work only if it does not disturb the coalesced K/V
  read-only load pattern.

## 2026-06-19 - Sliding Decode Persistent Work Queue

Question: can we land the persistent producer/consumer scheduler shape for
sliding decode attention without changing flash decode numerics?

Change:

- Added an opt-in persistent work-queue decode path,
  `gemma4_flash_attention_sliding_decode_paged_persistent_bf16`.
- The first queue generation is split tasks. Each live split task writes the
  same online-softmax partials as the direct path, then the final live split for
  each Q row enqueues that row's reduce task.
- Dummy overprovisioned split tasks do not count toward row readiness. This is
  required because persistent workers can pop split IDs out of order.
- All producer lanes issue a device fence before thread 0 publishes a reduce
  task, because each lane owns one `partial_acc` value and the consumer CTA can
  start before the producer CTA loops.
- Added explicit caller-owned `int32_t` scratch sizing through
  `gemma4_flash_attention_sliding_decode_persistent_scratch_i32`.
- Added correctness coverage to `test_kv_cache` and a benchmark line for the
  persistent path in `gemma4_kv_cache_bench`.

Commands:

```bash
make -B build/tests/test_kv_cache build/experiments/gemma4_kv_cache_bench \
  NVCC=/usr/local/cuda/bin/nvcc

./build/tests/test_kv_cache

./build/experiments/gemma4_kv_cache_bench 4096 64 64 10 50 7
```

Contract:

- Hardware: NVIDIA RTX A6000, driver `580.126.16`.
- CUDA/NVCC: CUDA compilation tools `13.0`, `V13.0.48`; benchmark runtime and
  driver API versions reported as `13000`.
- Shape: sliding decode, `B=1`, `seq_len=4096`, `page_size=64`,
  `split_size=64`, sliding key count `1024`, `actual_splits=16`,
  `num_splits=16`, BF16, `q_heads=32`, `kv_heads=16`, `head_dim=256`.
- Timing: C++ benchmark, CUDA events on the benchmark stream. Warm cache.
- Warmup/repeats: `10` warmup iterations, `50` iterations per sample,
  `7` samples.
- Cache policy: warm cache, no L2 flush.
- Clock policy: clocks were not locked. `nvidia-smi` sampled the idle GPU after
  the run at `210 MHz` SM and `405 MHz` memory, so boost state during timing was
  not pinned.
- Nsight Compute: not available in this environment (`ncu` not on `PATH`).
- Correctness: `test_kv_cache` passed. Benchmark correctness for both direct
  and persistent paths was `max_abs=0.023438`, `mean_abs=0.000202`.

Median comparison:

```text
path                                      median ms  vs direct
flash_decode_paged_attention_direct       0.066292   baseline
flash_decode_paged_attention_persistent   0.123248   +85.92%
flash_full_decode_write_plus_attention    0.065050   -1.87%
```

Conclusion:

- The persistent scheduler is numerically correct across the existing decode
  edge cases, including overprovisioned splits and staggered batch lengths.
- As a standalone flash-only path, it is slower because it adds queue/init
  atomics and a worker grid while still reading prepared Q and cached K/V from
  HBM.
- This is still the right scaffold for the next fusion step: projection/prep
  producer tasks can now hand ready rows to the same worker-owned reduce path,
  where the expected win comes from removing Q/K/V round trips rather than from
  replacing the existing two flash launches by itself.

## 2026-06-19 - Single-Kernel Greedy Final-Logit Endpoint

Question: can the decode endpoint be a real single kernel for greedy sampling,
instead of computing final logits and then launching a separate reducer/gather?

Change:

- Replaced the two-kernel greedy endpoint with one cooperative-grid kernel.
- Each resident CTA walks a stripe of final vocab-projection tiles, keeps its
  best local token, and writes one candidate.
- The cooperative grid syncs once, then block 0 reduces resident-CTA candidates,
  writes `next_token`, and copies the tied LM-head/embedding row into
  `next_hidden`.
- Full `[262144]` logits are still not materialized.

Commands:

```bash
make NVCC=/usr/local/cuda/bin/nvcc test-sampling
make NVCC=/usr/local/cuda/bin/nvcc sampling-bench
./build/experiments/gemma4_sampling_bench 8 50 9
```

Contract:

- Hardware: NVIDIA RTX A6000.
- Timing: CUDA events on a nonblocking stream, launch overhead included.
- Warmup/repeats: `8` warmup iterations, `50` iterations per sample,
  `9` samples.
- Cache policy: warm-ish repeated decode buffers, no L2 flush.
- Clock policy: clocks were not locked.
- Correctness: `test_sampling` passed. Benchmark matched the old endpoint:
  `old_token=184509`, `fused_token=184509`.

Median comparison:

```text
path                       median ms
final_logits_only           3.955187
old_logits_argmax_embed     3.997942
fused_greedy_endpoint       3.974664
```

Conclusion:

- The greedy endpoint is now actually one kernel for
  `final logits -> argmax -> tied embedding gather`.
- It is `23.278 us` faster than the old full-logits plus argmax/gather endpoint.
- It is `19.477 us` slower than final-logits-only because the cooperative
  persistent grid adds one grid-wide sync and a final in-kernel reduction.
- The dominant cost remains the BF16 LM-head stream:
  `5376 * 262144 * 2 ~= 2.82 GB`.

## 2026-06-19 - Sliding Decode Fused Projection-Prep Ingress

Question: can sliding decode avoid materializing the full raw QKV projection
tensor before the existing Q/K/V norm+RoPE+cache preparation path?

Change:

- Added `gemma4_flash_attention_sliding_decode_project_prepare_paged_kv_bf16`.
- The first implementation is intentionally conservative: each warp owns one
  256-wide Q/K/V head, computes its lane-owned projected values from
  `x @ Wqkv`, rounds them to the same BF16 raw ingress precision as the
  existing projection output, then feeds those lane-local raw floats through
  the refactored prep helpers.
- Q is written to `d_q_prepared`; K and V are written directly into the
  existing Layout-A paged cache row. The full `[16384]` raw QKV tensor is not
  written by the fused path.
- Existing paged flash decode attention remains unchanged and consumes
  `d_q_prepared`, `d_cache_k`, and `d_cache_v`.
- Replaced the temporary C++ benchmark with
  `src/experiments/gemma4_project_prepare_compare.py`, a minimal Python
  harness that compares the custom kernel against PyTorch and an optional
  vLLM custom-op baseline when vLLM is installed.
- This pass does not use cross-CTA producer/consumer mailboxes. On SM86 this
  avoids relying on inter-block shared memory or distributed shared memory;
  the CUDA guide places thread block clusters/DSM in the CC 9.0+ model
  (CUDA Programming Guide pages 50 and 57).

Commands:

```bash
make -B test-kv-cache NVCC=/usr/local/cuda/bin/nvcc

python3 src/experiments/gemma4_project_prepare_compare.py \
  --warmup 3 --iters 5 --samples 5 \
  --json src/experiments/results/2026-06-19_project_prepare_compare_warm.json \
  | tee src/experiments/results/2026-06-19_project_prepare_compare_warm.txt
```

Contract:

- Hardware: NVIDIA RTX A6000, bus `00000000:04:00.0`, driver `580.126.16`,
  persistence enabled, ECC disabled, power limit `300 W`.
- CUDA/NVCC: CUDA compilation tools `13.0`, `V13.0.48`; PyTorch `2.11.0+cu130`
  with CUDA `13.0`.
- Shape: sliding decode, `B=1`, `seq_len=1024`, `page_size=64`,
  `split_size=64`, BF16, hidden `5376`, packed QKV `16384`,
  Q heads `32`, KV heads `16`, head dim `256`.
- Timing: Python benchmark with CUDA events on the current stream. CPU launch
  and Python enqueue overhead are not measured by the CUDA events.
- Warmup/repeats: `3` warmup iterations, `5` iterations per sample,
  `5` samples.
- Cache policy: warm-L2 repeated buffers; no L2 flush.
- Clock policy: clocks were not locked. The JSON snapshot reported
  `1800 MHz` SM, `7601 MHz` memory, `79.17 W`, `31 C`, and `7%`
  GPU utilization.
- Correctness: `test-kv-cache` passed. Python custom-vs-PyTorch max absolute
  diff across prepared Q and K/V cache was `0.031250`.
- vLLM baseline: skipped in this environment because `vllm` is not installed.
- Results were also written as JSON, including raw samples, summary statistics,
  contract, environment, and threats.

Median comparison:

```text
path                     median ms  IQR ms    vs PyTorch
pytorch_project_prepare   4.382055  0.958765  baseline
custom_project_prepare    2.574598  0.005408  -41.25%
vllm_project_prepare      skipped   -         vLLM unavailable
```

Conclusion:

- The fused ingress API is correct and removes the full raw QKV HBM
  materialization requirement.
- Against a minimal eager PyTorch baseline, the custom fused kernel is faster
  for this `B=1`, `seq_len=1024` harness. This does not contradict the earlier
  C++ baseline result: the existing CTA-reduced custom projection path is a
  much stronger baseline than eager PyTorch.
- The next performance pass should keep the public API and replace the serial
  per-warp dot-product work with a parallel head-fragment or L2-mailbox design
  that preserves small handoff state without staging the full raw QKV tensor.

## 2026-06-19 - Global paged decode vs PyTorch graph cold-cache benchmark

Question:

- Measure the newly generalized global paged decode attention path against a
  PyTorch GQA implementation over the same paged cache layout, with cold-cache
  timing and launch overhead excluded.

Change:

- Added `src/experiments/gemma4_global_decode_torch_bench.py`, a small Python
  benchmark that imports `gemma4_flash_attention_decode_paged_bf16` from
  `build/libgemma4_flash_attention.so` via `ctypes`.
- The custom and PyTorch paths are both captured into CUDA graphs. Timing uses
  CUDA events on the current PyTorch stream around graph replay.
- Cold-cache timing writes a 128 MiB device buffer before each measured replay,
  then records the start event after the flush so the flush is ordered but not
  included in elapsed time.
- The script records an empty CUDA event-pair timing and reports both raw and
  timer-corrected medians.

Command:

```bash
python3 src/experiments/gemma4_global_decode_torch_bench.py \
  --seq-len 4096 --page-size 64 --split-size 64 \
  --warmup 10 --iters 10 --samples 10 \
  --cache cold --flush-mib 128 --sample-delay-s 1.0 \
  --json src/experiments/results/2026-06-19_global_decode_torch_graph_cold_rebench.json
```

Contract:

- Hardware: NVIDIA RTX A6000, driver `580.126.16`, persistence enabled, ECC
  disabled, power limit `300 W`.
- Clock policy: not locked. JSON snapshot reported `1800 MHz` SM,
  `7601 MHz` memory, `83.32 W`, `36 C`, and `P2`.
- Shape: global decode, `B=1`, `q_len=1`, `seq_len=4096`, `page_size=64`,
  `split_size=64`, `num_splits=64`, BF16, `q_heads=32`, `kv_heads=4`,
  `head_dim=512`.
- Timing: CUDA events on the current PyTorch stream around CUDA graph replay.
  CPU launch and host wall time are excluded.
- Cache policy: cold-cache, 128 MiB dummy device write before each measured
  replay, excluded from the event span.
- Repeats: 10 warmups, 10 measured replays per sample, 10 samples.
- Delay: one second host sleep before each measured sample, outside the event
  window.
- Correctness: custom-vs-PyTorch max absolute diff was
  `0.00000095367431640625`.

Cold-cache result:

```text
path                         median ms  corrected median ms  IQR ms    p95 ms    speedup
PyTorch global decode graph   0.519003   0.516677             0.001690  0.525854  baseline
custom global decode          0.420848   0.418522             0.000126  0.422695  1.23x raw / 1.23x corrected
```

Conclusion:

- The generalized custom global decode path is `1.23x` faster than the
  graph-captured PyTorch baseline at this cold-cache `S=4096` shape.
- Remaining threats: clocks were not locked, this is one process on one GPU,
  and the cold-cache mode uses a synthetic L2 flush rather than an end-to-end
  serving trace.

## 2026-06-19 - KV-group sliding project-prepare ownership benchmark

Question: how does the KV-group ownership change affect standalone sliding decode
project-prepare latency versus the previous head-group implementation?

Compared builds:

- Old: `HEAD:src/gemma4_flash_attention.cu` built to
  `/tmp/libgemma4_flash_attention_old.so`.
- New: current working-tree `src/gemma4_flash_attention.cu` built to
  `/tmp/libgemma4_flash_attention_new.so`.

Commands:

```bash
make test-kv-cache NVCC=/usr/local/cuda/bin/nvcc
make build/libgemma4_flash_attention.so NVCC=/usr/local/cuda/bin/nvcc
python3 /tmp/bench_gemma4_project_prepare_old_new.py \
  --warmup 25 --iters 200 --samples 21 \
  --shapes 1x1024 4x1024 \
  --json /tmp/gemma4_project_prepare_old_new_warm.json
```

Contract:

- Hardware: NVIDIA RTX A6000, bus `00000000:0D:00.0`, driver `580.126.16`,
  persistence enabled, ECC disabled, power limit `300 W`.
- CUDA/NVCC: CUDA compilation tools `13.0`, `V13.0.48`; PyTorch
  `2.11.0+cu130` with CUDA `13.0`.
- Shape: sliding decode project-prepare, BF16, hidden `5376`, packed QKV
  `16384`, Q heads `32`, KV heads `16`, head dim `256`, page size `64`,
  sequence length `1024`.
- Timing: CUDA events recorded on the current stream around repeated kernel
  launches. CPU enqueue and host wall time are excluded.
- Warmup/repeats: `25` warmup launches per candidate, `200` launches per
  sample, `21` samples. Old/new order was randomized per sample.
- Cache policy: warm-L2 repeated buffers; no L2 flush.
- Clock policy: clocks were not locked. The benchmark snapshot reported
  `1800 MHz` SM, `8001 MHz` memory, `33.34 W`, `31 C`, and `0%` GPU
  utilization.
- Correctness: both old and new matched the PyTorch reference with max absolute
  diff `0.031250`, and new matched old exactly for prepared Q, K cache, and V
  cache.
- Profiler: `/usr/local/cuda/bin/ncu` exists, but the Nsight Compute install is
  incomplete (`nsight-compute directory is not found`), so this run uses CUDA
  event timing only.
- Raw JSON with samples, environment, and threats:
  `src/experiments/results/2026-06-19_project_prepare_kv_group_old_new_warm.json`.

Median comparison:

```text
shape          old head-group  new KV-group  delta new-vs-old  speedup  IQR old   IQR new
B=1 S=1024       2.403106 ms    1.196953 ms          -50.19%    2.008x  0.000922  0.000513
B=4 S=1024       2.366466 ms    1.210580 ms          -48.84%    1.955x  0.000950  0.000933
```

Conclusion:

- For this standalone sliding project-prepare path, KV-group ownership is about
  `2x` faster than the previous head-group launch shape on warm-cache CUDA-event
  timing.
- This benchmark only isolates ingress project-prepare latency. It does not
  measure the future megakernel queueing benefit directly, but it confirms the
  ownership change did not introduce a standalone latency regression.

## 2026-06-19 - Lazy-RMS FFN Decode Consumer Smoke Benchmark

Question: does applying the FFN input RMSNorm in registers from residual state
change standalone decode FFN latency versus the existing materialized-input FFN
decode path?

Change:

- Added `lazy_graph`, `lazy_warm_minus_clear`, `cold_lazy`, and
  `lazy_cold_minus_flush_clear` rows to
  `src/experiments/gemma4_ffn_decode_load_bench.cu`.
- The benchmark reuses the existing FFN decode harness and buffers; it does not
  add a new benchmark program.

Commands:

```bash
make test-ffn-decode NVCC=/usr/local/cuda/bin/nvcc
make ffn-decode-load-bench NVCC=/usr/local/cuda/bin/nvcc
./build/experiments/gemma4_ffn_decode_load_bench 30 8 5
for i in 1 2 3 4 5; do ./build/experiments/gemma4_ffn_decode_load_bench 50 10 5; done
```

Contract:

- Hardware: NVIDIA RTX A6000, bus `00000000:06:00.0`, driver `580.126.16`,
  persistence enabled, ECC disabled, power limit `300 W`.
- Timing: CUDA events on a nonblocking stream. Warm rows use CUDA graph replay
  with capture/instantiate outside timing. Cold rows flush 256 MiB before each
  measured call and subtract the measured flush+scratch-clear overhead.
- Warmup/repeats: primary repeated run used `10` warmups, `50` iterations per
  timed sample, `5` samples per process, and `5` process-level repeats.
- Cache policy: warm graph rows use repeated buffers; cold rows use the
  benchmark's 256 MiB device flush.
- Clock policy: clocks were not locked. `nvidia-smi` snapshots before and after
  the run showed idle clocks at `210 MHz` SM and `405 MHz` memory.
- Correctness: `test-ffn-decode` passed, including lazy-RMS output comparison
  against the materialized-input FFN path.

Process-level median of best corrected rows:

```text
path                         median ms
materialized warm-minus-clear  0.974781
lazy-RMS warm-minus-clear      0.975068
materialized cold corrected    0.976433
lazy-RMS cold corrected        0.976769
```

Conclusion:

- The lazy-RMS consumer is effectively performance-neutral in this standalone
  FFN decode benchmark: about `0.0003 ms` slower, roughly `0.03%`.
- This does not yet measure the intended O-producer mailbox win. It only shows
  that the FFN consumer side can absorb input RMSNorm in registers without a
  meaningful standalone latency penalty.

## 2026-06-19 - Sliding O-to-FFN Lazy-RMS Chain Slice Benchmark

Question: does the first local chain slice,
`sliding O projection -> residual/RMS bookkeeping -> FFN`, improve when the
post-attention normed hidden vector is not materialized?

Change:

- Extended `src/experiments/gemma4_ffn_decode_load_bench.cu` with two chain
  rows:
  - `chain_baseline`: sliding O projection, residual+RMSNorm, materialized-input
    FFN decode.
  - `chain_lazy`: sliding O projection, residual+sumsq/scale producer, lazy-RMS
    FFN decode.
- The residual+sumsq producer is benchmark-local and writes the scalar scale,
  but the current lazy FFN public API still consumes a fixed representative host
  scale. This measures the dataflow cost of skipping the normed hidden
  materialization, not full numerical equivalence of the chain.

Commands:

```bash
make test-ffn-decode NVCC=/usr/local/cuda/bin/nvcc
make ffn-decode-load-bench NVCC=/usr/local/cuda/bin/nvcc
./build/experiments/gemma4_ffn_decode_load_bench 30 8 5
for i in 1 2 3 4 5; do ./build/experiments/gemma4_ffn_decode_load_bench 50 10 5; done
```

Contract:

- Hardware: NVIDIA RTX A6000, bus `00000000:06:00.0`, driver `580.126.16`,
  persistence enabled, ECC disabled, power limit `300 W`.
- Shape: sliding O projection `8192 -> 5376`, hidden `5376`, FFN
  gate/up `5376 -> 43008`, FFN down `21504 -> 5376`, BF16.
- Timing: CUDA events on a nonblocking stream. Warm rows use CUDA graph replay
  with capture/instantiate outside timing. Cold rows flush 256 MiB before each
  measured call and subtract measured flush+scratch-clear overhead.
- Warmup/repeats: repeated run used `10` warmups, `50` iterations per timed
  sample, `5` samples per process, and `5` process-level repeats.
- Cache policy: warm graph rows use repeated buffers; cold rows use the
  benchmark's 256 MiB device flush.
- Clock policy: clocks were not locked. Pre-run `nvidia-smi` snapshot showed
  idle clocks at `210 MHz` SM and `405 MHz` memory.
- Correctness: `test-ffn-decode` passed. The chain benchmark is a timing slice;
  full chain output equivalence is not claimed because the lazy path consumes a
  fixed representative scale.

Process-level median of best corrected rows:

```text
path                 median ms
chain baseline warm   1.102913
chain lazy warm       1.102406
chain baseline cold   1.106110
chain lazy cold       1.105274
```

Conclusion:

- The chain slice is slightly faster with the lazy-RMS consumer:
  `-0.000507 ms` warm (`-0.046%`) and `-0.000836 ms` cold (`-0.076%`).
- This is still a tiny local win, not the final mailbox overlap win. The next
  measurement should replace the fixed host scale with a device-consumed scale
  or move both producer and consumer into one persistent mailbox kernel.

## 2026-06-19 - Device-Scale Lazy-RMS Chain Rebench

Question: after replacing the fixed host scale with the scale produced on
device by the residual+sumsq producer, does the local chain slice still show a
benefit?

Change:

- Added `gemma4_ffn_decode_lazy_rms_device_scale_bf16()`, a decode FFN entry
  point that reads the input RMS scale from device memory.
- Updated `chain_lazy` in `gemma4_ffn_decode_load_bench.cu` to consume
  `d_post_attn_scale` produced by the benchmark-local residual+sumsq kernel.
- Kept the old host-scale lazy entry point for standalone consumer checks.

Commands:

```bash
make test-ffn-decode NVCC=/usr/local/cuda/bin/nvcc
make ffn-decode-load-bench NVCC=/usr/local/cuda/bin/nvcc
./build/experiments/gemma4_ffn_decode_load_bench 5 2 1
for i in 1 2 3 4 5; do \
  GEMMA4_FFN_LOAD_BENCH_SEED=0x51a3f00d \
  ./build/experiments/gemma4_ffn_decode_load_bench 50 10 5; \
done
```

Contract:

- Hardware: NVIDIA RTX A6000, bus `00000000:06:00.0`, driver `580.126.16`,
  persistence enabled, ECC disabled, power limit `300 W`.
- Compiler: `/usr/local/cuda/bin/nvcc`, `Build cuda_13.0.r13.0`.
- Shape: sliding O projection `8192 -> 5376`, hidden `5376`, FFN
  gate/up `5376 -> 43008`, FFN down `21504 -> 5376`, BF16.
- Timing: CUDA events on a nonblocking stream. Warm rows use CUDA graph replay
  with capture/instantiate outside timing. Cold rows flush 256 MiB before each
  measured call and subtract measured flush+scratch-clear overhead.
- Warmup/repeats: `10` warmups, `50` iterations per timed sample, `5` samples
  per process, `5` process-level repeats.
- Cache policy: warm graph rows reuse buffers; cold rows use the benchmark's
  256 MiB device flush.
- Clock policy: clocks were not locked. Post-run `nvidia-smi` snapshot showed
  idle clocks at `210 MHz` SM and `405 MHz` memory.
- Correctness: `test-ffn-decode` passed, including device-scale lazy-RMS FFN
  parity against the materialized-input FFN path. The corrected chain now reads
  the same RMS scale value computed by its residual+sumsq producer instead of a
  fixed representative scale.

Process-level median of best corrected rows:

```text
path                         median ms  lazy delta
materialized warm-minus-clear  0.974877  +0.000157 ms (+0.016%)
lazy-RMS warm-minus-clear      0.975034
materialized cold corrected    0.976597  +0.000223 ms (+0.023%)
lazy-RMS cold corrected        0.976820

chain baseline warm            1.103075  -0.000475 ms (-0.043%)
chain lazy warm                1.102600
chain baseline cold            1.106050  -0.000720 ms (-0.065%)
chain lazy cold                1.105330
```

Conclusion:

- The corrected chain comparison is numerically comparable in the sense that
  `chain_lazy` consumes the device-produced RMS scale for the same
  post-attention residual state instead of the old fixed placeholder.
- Standalone lazy-RMS FFN remains neutral/slightly slower by about `0.02%`.
- The local O-to-FFN chain still shows a tiny speedup, about `0.04%` warm and
  `0.07%` cold. This is too small to celebrate, but it is no longer an invalid
  fixed-scale comparison.

## 2026-06-20 - Fused Sliding-O Projection Residual/RMS Producer

Question: what happens when the local chain removes the projection-delta buffer
for real, by fusing sliding O projection with the following residual add and
RMS bookkeeping?

Change:

- Added `gemma4_sliding_o_projection_residual_rmsnorm_decode()`.
- Removed the lazy-RMS FFN consumer APIs and benchmark rows from the active code
  path.
- Replaced benchmark `chain_lazy` with `chain_fused_o`:
  - baseline: `O projection -> residual_add_rmsnorm -> FFN decode`.
  - fused: `O projection + residual + partial_sumsq -> RMS finalize -> FFN
    decode`.
- The kept implementation uses two kernels for the fused O/RMS producer:
  - projection CTAs write `post_attn_residual` and one sumsq partial each.
  - one finalize CTA reduces partials and writes `post_attn_normed`.
- A single-kernel last-block counter variant was tested and rejected before
  keeping this version; its fences/atomics outweighed the removed buffer pass.

Commands:

```bash
make test-ffn-decode NVCC=/usr/local/cuda/bin/nvcc
make test-kv-cache NVCC=/usr/local/cuda/bin/nvcc
make ffn-decode-load-bench NVCC=/usr/local/cuda/bin/nvcc
./build/experiments/gemma4_ffn_decode_load_bench 5 2 1
for i in 1 2 3 4 5; do \
  GEMMA4_FFN_LOAD_BENCH_SEED=0x51a3f00d \
  ./build/experiments/gemma4_ffn_decode_load_bench 50 10 5; \
done
```

Contract:

- Hardware snapshot after run: NVIDIA RTX A6000, bus `00000000:04:00.0`,
  driver `580.126.16`, persistence enabled, ECC disabled, power limit `300 W`.
- Compiler: `/usr/local/cuda/bin/nvcc`, `Build cuda_13.0.r13.0`.
- Shape: sliding O projection `8192 -> 5376`, hidden `5376`, FFN
  gate/up `5376 -> 43008`, FFN down `21504 -> 5376`, BF16.
- Timing: CUDA events on a nonblocking stream. Warm rows use CUDA graph replay
  with capture/instantiate outside timing. Cold rows flush 256 MiB before each
  measured call and subtract measured flush+scratch-clear overhead.
- Warmup/repeats: `10` warmups, `50` iterations per timed sample, `5` samples
  per process, `5` process-level repeats.
- Cache policy: warm graph rows reuse buffers; cold rows use the benchmark's
  256 MiB device flush.
- Clock policy: clocks were not locked. Post-run snapshot showed idle clocks at
  `210 MHz` SM and `405 MHz` memory.
- Correctness: `test-ffn-decode` passed. `test-kv-cache` passed with
  `fused sliding O normed max_abs=0 max_rel=0` against the baseline
  `projection_decode -> residual_add_rmsnorm` path.

Process-level median of best corrected rows:

```text
path                 median ms  fused delta
chain baseline warm   1.102835  +0.000073 ms (+0.0066%)
chain fused-O warm    1.102908
chain baseline cold   1.106008  -0.000087 ms (-0.0079%)
chain fused-O cold    1.105921
```

Conclusion:

- The fused benchmark path removed `d_o_delta`, but the measured result was
  effectively neutral: very slightly slower warm-cache and very slightly faster
  cold-cache.
- The fused-O API, benchmark rows, and parity test were removed from active code
  after this result.
- This confirms the projection-delta pass is too small relative to the FFN
  weight stream to move end-to-end decode meaningfully by itself. The next
  material target remains reducing or amortizing FFN weight bytes.

## 2026-06-20 - 12B Decode Thread-Count Sanity Pass

Question: after retargeting the constants to Gemma 4 12B, are any decode GEMV
thread counts obviously mismatched to the new dimensions?

Change:

- Changed FFN-down decode from `1024` to `960` threads. The 12B FFN-down
  reduction has `15360 / 8 = 1920` BF16 vector packs, so this gives every
  thread exactly two packs instead of leaving the tail imbalanced.
- Kept final logits at `1024` threads. A `512`-thread trial was not an obvious
  win in the short benchmark and it also affects the fused sampler reduction.

Commands:

```bash
make NVCC=/usr/local/cuda/bin/nvcc decode-bench test-sampling test-kv-cache
GEMMA4_DECODE_BENCH_SEED=0x12b \
  ./build/experiments/gemma4_decode_bench ffn_down 20 5 2
GEMMA4_DECODE_BENCH_SEED=0x12b \
  ./build/experiments/gemma4_decode_bench final_logits 5 2 1
./build/tests/test_kv_cache
make NVCC=/usr/local/cuda/bin/nvcc test-kv-cache
```

Contract:

- Hardware: NVIDIA RTX A6000, driver `580.126.16`, persistence enabled, ECC
  disabled, power limit `300 W`.
- Compiler: `/usr/local/cuda/bin/nvcc`, CUDA compilation tools `13.0`,
  `Build cuda_13.0.r13.0`.
- Shape: Gemma 4 12B FFN down `15360 -> 3840`, final logits
  `3840 -> 262144`, BF16.
- Timing: decode bench CUDA-event timings with `5` warmup iterations, `20`
  timed iterations, and `2` trials for FFN down. The final-logits spot check
  used `2` warmups, `5` timed iterations, and `1` trial.
- Cache policy: benchmark default warm-buffer behavior.
- Clock policy: clocks were not locked; post-run idle snapshot was `210 MHz`
  SM and `405 MHz` memory.
- Correctness: `test-sampling` passed. The first full make run hit a transient
  sliding persistent attention mismatch in `test-kv-cache`; rerunning
  `./build/tests/test_kv_cache` and then `make test-kv-cache` passed.

Result:

```text
path                old best ms  new best ms  correctness
FFN down custom       0.169640     0.169432   swizzle16 max_abs=0
```

Conclusion:

- The `960`-thread FFN-down launch is the only keeper from this pass. It better
  matches the 12B pack count and measured neutral/slightly faster in the short
  run.
- No final-logits launch change was kept; the current `1024` threads remain a
  reasonable baseline until a more serious sampler/logits benchmark says
  otherwise.

## 2026-06-25 - FFN LibTorch Baseline Replacement

Question: replace the FFN cuDNN Frontend comparator with a direct LibTorch
baseline while keeping the same CUDA-event timing contract.

Change:

- Deleted `src/benches/gemma4_ffn_cudnn_bench.cu`.
- Added `src/benches/gemma4_ffn_libtorch_bench.cu`.
- Replaced `make ffn-cudnn-bench` with `make ffn-libtorch-bench`.
- The LibTorch baseline computes packed gate/up, `gelu(..., "tanh") * up`,
  and down projection with preallocated BF16 tensors. Custom CUDA rows still use
  the same swizzled weights and CUDA-event timers.

Commands:

```bash
make ffn-cudnn-bench
./build/benches/gemma4_ffn_cudnn_bench 20 5 3 16
make ffn-libtorch-bench
./build/benches/gemma4_ffn_libtorch_bench 20 5 3 16
```

Contract:

- Hardware: NVIDIA RTX A6000, driver `580.126.16`, persistence enabled, ECC
  disabled, power limit `300 W`.
- Shape: `tokens=16`, hidden `3840`, intermediate `15360`, BF16.
- Timing: CUDA events on the same stream, `5` warmups, `20` iterations, `3`
  trials. Graph rows time replay only and exclude capture.
- Cache policy: warm repeated buffers for direct/graph rows; cold custom rows
  flush L2 with a `256 MiB` buffer before each measured invocation.
- Clock policy: clocks were not locked.

Result:

```text
old cuDNN down graph best ms      0.171701
new LibTorch down graph best ms   0.172162
new LibTorch full FFN graph ms    0.547880
new custom prefill graph ms       0.523838
custom vs LibTorch max_abs        9.53674e-07
```

Conclusion:

- The down-projection baseline speed is effectively unchanged across cuDNN and
  LibTorch for the same shape.
- The stripped cuDNN `geglu`/`full_ffn` rows were not trustworthy after status
  checks were removed, so the full-FFN replacement should be compared against
  the new LibTorch row going forward.

## 2026-06-25 - FFN LibTorch Prefill Shape Sweep

Question: measure custom FFN prefill against the LibTorch full-FFN baseline
across the token counts `1,2,4,8,16,32,64,128,256,512,1024`, and report the
single-token decode row from the same benchmark.

Command:

```bash
make ffn-libtorch-bench
for tokens in 1 2 4 8 16 32 64 128 256 512 1024; do
  GEMMA4_FFN_LIBTORCH_BENCH_SEED=0x12345678 \
    ./build/benches/gemma4_ffn_libtorch_bench 100 20 5 "$tokens"
done
```

Contract:

- Hardware: NVIDIA RTX A6000, driver `580.126.16`, persistence enabled, ECC
  disabled, power limit `300 W`.
- Timing: CUDA events on the benchmark stream; graph rows time replay only and
  exclude capture. `20` warmups, `100` iterations, `5` trials.
- Cache policy: warm repeated buffers for direct/graph rows; cold decode rows
  flush L2 with the benchmark's `256 MiB` buffer.
- Clock policy: clocks were not locked. The GPU was idle before and after the
  sweep; idle snapshots were `210 MHz` SM and `405 MHz` memory.
- Correctness: custom prefill vs LibTorch max abs stayed at `<=1.90735e-06`.

Graph-best result:

```text
tokens  libtorch ms  custom prefill ms  speedup
1          0.515952            0.519453   0.993x
2          0.534187            0.520805   1.026x
4          0.536403            0.521169   1.029x
8          0.539919            0.521788   1.035x
16         0.547369            0.523215   1.046x
32         0.535938            0.523210   1.024x
64         0.549779            0.529066   1.039x
128        0.597995            0.559966   1.068x
256        0.930442            0.889440   1.046x
512        1.766559            1.666567   1.060x
1024       3.457877            3.113875   1.110x
```

Decode note:

- The decode row is single-token regardless of the `tokens` argument. At
  `tokens=1`, custom decode graph best was `0.500241 ms` versus LibTorch
  one-token full FFN graph best `0.515952 ms`, or `1.031x`.
- The decode row includes post-FFN residual/RMSNorm work; the LibTorch row is
  the FFN MLP baseline, so this is a conservative but not perfectly identical
  comparison.

Conclusion:

- Custom prefill is consistently only modestly faster: roughly tied at one row,
  `1.02-1.07x` for most smaller/mid shapes, and `1.11x` at `1024` rows.
- This matches the implementation: custom prefill fuses gate/up + GeGLU, but
  still writes the activation to HBM before the separate down GEMM.

## 2026-06-26 - FFN Decode Stage and Cache-Load Ablation

Question: check whether steady-state FFN decode benefits from changing the
CUTLASS buffering stages or explicit cache-policy vector loads. The cache-load
variants only touched the handwritten FFN pack loads in swizzle/finalize; the
CUTLASS GEMM internal loads stayed on their library path.

Commands:

```bash
make -B build/benches/gemma4_ffn_libtorch_bench
GEMMA4_FFN_LIBTORCH_BENCH_SEED=0x12345678 \
  ./build/benches/gemma4_ffn_libtorch_bench 50 10 5 1

make -B build/benches/gemma4_ffn_libtorch_bench \
  CPPFLAGS='-Isrc -DGEMMA4_FFN_DOWN_STAGE_ROWS64=2'
make -B build/benches/gemma4_ffn_libtorch_bench \
  CPPFLAGS='-Isrc -DGEMMA4_FFN_DOWN_STAGE_ROWS64=12'
make -B build/benches/gemma4_ffn_libtorch_bench \
  CPPFLAGS='-Isrc -DGEMMA4_FFN_GATE_UP_STAGE_ROWS128=3 -DGEMMA4_FFN_DOWN_STAGE_ROWS128=2'
make -B build/benches/gemma4_ffn_libtorch_bench \
  CPPFLAGS='-Isrc -DGEMMA4_FFN_VECTOR_LOAD_POLICY=N'
```

Contract:

- Hardware: NVIDIA RTX A6000, driver `580.126.16`, persistence enabled, ECC
  disabled, power limit `300 W`.
- Toolchain: CUDA `13.0.48`, PyTorch `2.11.0+cu130`.
- Timing: CUDA events on the benchmark stream; graph rows time replay only and
  exclude capture. `10` warmups, `50` iterations, `5` trials, `tokens=1`.
- Cache policy: warm repeated buffers for graph rows; cold rows use the
  benchmark's L2 flush. Clocks were not locked.
- Correctness: custom prefill vs LibTorch full FFN max abs was `9.53674e-07`
  in every measured variant.

Result:

```text
variant              graph ms  graph-clear ms  cold-flush-clear ms
default direct         0.520710       0.519597             0.523085
down stages=2          0.623961       0.622887             0.626543
down stages=12         0.521533       0.520452             0.524011
load __ldg             0.520214       0.519139             0.522956
load ld.global.cg      0.520431       0.519351             0.523465
load ld.global.ca      0.520799       0.519624             0.523666
load ld.global.cs      0.520404       0.519312             0.523331
```

Rows-128 prefill stage check:

```text
variant                   custom prefill graph ms  libtorch full graph ms
default gate/down 5/6                    0.554886              0.597487
reduced gate/down 3/2                    0.722149              0.597111
```

Notes:

- For the single-token path, `rows <= 64` is the active stage branch. Default
  stages are gate/up `3` and down `10`.
- At `tokens=128`, the active prefill branch uses gate/up `5` and down `6`.
- Gate/up `Stages=2` did not compile because the CUTLASS DualGemm example path
  asserts `Stages >= 3`.

Conclusion:

- Keep the current down `Stages=10` default. Reducing it to `2` regressed
  decode by about `20%`; increasing it to `12` was slightly slower than
  default. The `tokens=128` reduced-stage check also regressed badly, so the
  existing deeper prefill staging should stay.
- Keep direct/default FFN vector loads. `__ldg`, `cg`, `ca`, and `cs` were all
  within noise on this benchmark, and the tiny `__ldg` win was not large enough
  to justify changing the default.
- Follow-up cleanup removed the live ablation knobs after this result; rerunning
  the sweep requires reintroducing temporary compile-time hooks.

## 2026-06-26 - Prompt Runner Warm Serving vs vLLM

Question: compare the local `gemma4_prompt` warm serving path against vLLM on
the same A6000 for single-user TTFT and decode TPS.

Commands:

```bash
make prompt
./build/gemma4_prompt --benchmark-mode warm-serving --bench-warmup 3 \
  --bench-iters 5 --bench-samples 3 --max-new 16 --prompt Hello \
  | tee build/bench_results/gemma4_prompt_warm_equal_hello_out16_c1.txt

python3 - <<'PY'
from pathlib import Path
path = Path("build/bench_results/hello_prompt_15x.jsonl")
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text('{"prompt":"Hello","output_tokens":16}\n' * 15)
PY

uv run vllm serve models/gemma-4-12B --host 127.0.0.1 --port 8000 \
  --served-model-name gemma4-12b --dtype bfloat16 --max-model-len 64 \
  --gpu-memory-utilization 0.99 --language-model-only --trust-remote-code \
  --disable-log-stats --max-num-seqs 1 --max-num-batched-tokens 64 \
  -cc.cudagraph_mode=NONE

uv run vllm bench serve --backend openai \
  --base-url http://127.0.0.1:8000 --model gemma4-12b \
  --tokenizer models/gemma-4-12B --trust-remote-code \
  --dataset-name custom --dataset-path build/bench_results/hello_prompt_15x.jsonl \
  --skip-chat-template --output-len 16 --num-prompts 15 --num-warmups 3 \
  --request-rate inf --max-concurrency 1 --temperature 0 \
  --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,95,99 \
  --save-result --result-dir build/bench_results \
  --result-filename vllm_warm_equal_hello_out16_c1_compiled_nocg_warm2.json \
  --disable-tqdm
```

Contract:

- Hardware: NVIDIA RTX A6000, driver `580.126.16`, persistence enabled, ECC
  disabled, MIG disabled, power limit `300 W`.
- Software: local runner reported CUDA driver/runtime `13000`; vLLM `0.23.0`
  on PyTorch `2.11.0+cu130`. `nvcc` was not on the shell path for this run.
- Shape: literal prompt text `Hello`, prompt length `1` token for both paths,
  output length `16`, concurrency `1`, `3` warmup requests, `15` measured
  requests.
- Local timing: host wall clock; model load, CUDA context, and allocation
  excluded; tokenization, prompt H2D copy, prefill, per-token host sync, and
  decode included. Detokenization is included only in end-to-end latency.
- vLLM timing: `vllm bench serve` through the OpenAI completions endpoint;
  tokenizer, HTTP/server overhead, scheduler, prefill, decode, and streaming
  are included.
- vLLM caveat: the unconstrained compiled vLLM server failed KV-cache
  initialization on this A6000. The measured vLLM baseline kept compile enabled
  but required `max_num_seqs=1`, `max_num_batched_tokens=64`,
  `gpu_memory_utilization=0.99`, and disabled CUDA graph capture with
  `-cc.cudagraph_mode=NONE`. Startup only succeeded after the compile cache was
  warm.
- Cache/clock policy: serving benchmark uses warm steady-state requests after
  warmup. Clocks were not locked.
- Equality caveat: the request shape and token counts are matched exactly, but
  the interfaces are still different. The local runner is in-process, while
  vLLM is measured as an HTTP serving stack. A perfectly identical comparison
  would require either wrapping the local runner in the same serving protocol or
  adding a vLLM in-process TTFT/TPOT harness with the same streaming semantics.
- Run note: the first vLLM pass JIT-compiled three Triton kernels during
  initial/warmup traffic. The charted result is the second pass on the same
  already-warmed server; no new JIT warnings appeared during that pass.

Result:

```text
runner                         p50 TTFT ms   p50 TPOT ms   p50 decode TPS
gemma4_prompt local                 38.878        75.120            13.312
vLLM serve compiled no CG          208.512        96.762            10.335
```

Artifacts:

- `build/bench_results/gemma4_prompt_warm_equal_hello_out16_c1.txt`
- `build/bench_results/hello_prompt_15x.jsonl`
- `build/bench_results/vllm_warm_equal_hello_out16_c1_compiled_nocg_warm2.json`
- `build/bench_results/gemma4_vs_vllm_summary.json`
- `build/bench_results/ttft_p50_gemma4_vs_vllm.png`
- `build/bench_results/tps_p50_gemma4_vs_vllm.png`

Conclusion:

- Under the compiled/no-CUDA-graph vLLM baseline that fits on this machine, the
  local runner wins on both requested P50 metrics: lower TTFT and higher decode
  TPS.
- Do not generalize this as a production vLLM win yet. The comparison is not
  perfectly apples-to-apples because the local path is an in-process runner,
  while vLLM is serving through an HTTP-compatible API and had to disable CUDA
  graph capture due memory pressure.

## 2026-06-26 - Prompt Benchmark Rerun After Decode Layer-Scalar Fusion

Question: rerun the prompt benchmark after the decode megakernel rewrite and
confirm the local path is still at least as fast as the previous README result.

Root-cause fix before final timing:

- Initial local warm-serving rerun regressed to `40.739 ms` p50 TTFT and
  `9.200 tok/s` p50 decode TPS, while CUDA-event decode-step timing was still
  `75.430 ms` p50. That pointed at host launch/setup overhead rather than GPU
  math.
- The regression came from a separate cooperative one-block `layer_scalar`
  launch on every decode layer. The fix folds that scalar into the existing
  post-FFN RMSNorm/residual kernel, removing 48 host launches per generated
  token.

Checks:

```bash
make cuda-kernels prompt test-ffn-decode test-decode-megakernel \
  test-prefill-megakernel test-sampling test-flash-attention-cpp \
  test-flash-attention-pytorch
```

All listed checks passed. `test_ffn_decode` now covers the fused layer-scalar
case.

Commands:

```bash
./build/gemma4_prompt --benchmark-mode decode-step --bench-warmup 3 \
  --bench-iters 5 --bench-samples 3 --max-new 16 --prompt Hello \
  | tee build/bench_results/gemma4_prompt_decode_step_hello_scaled_fused_20260626.txt

./build/gemma4_prompt --benchmark-mode warm-serving --bench-warmup 3 \
  --bench-iters 5 --bench-samples 3 --max-new 16 --prompt Hello \
  | tee build/bench_results/gemma4_prompt_warm_equal_hello_out16_c1_20260626.txt
```

vLLM was rerun with the same server and client contract as the prior entry. The
first serving pass emitted Triton JIT warnings for `_compute_slot_mapping_kernel`,
`kernel_unified_attention`, and `reduce_segments`; the charted result is the
second pass, with a third pass used only as a sanity check.

Result:

```text
runner                         p50 TTFT ms   p50 TPOT ms   p50 decode TPS
gemma4_prompt local                 38.469        73.707            13.567
vLLM serve compiled no CG          422.551       181.401             5.513
```

Artifacts:

- `build/bench_results/gemma4_prompt_decode_step_hello_scaled_fused_20260626.txt`
- `build/bench_results/gemma4_prompt_warm_equal_hello_out16_c1_20260626.txt`
- `build/bench_results/vllm_warm_equal_hello_out16_c1_compiled_nocg_20260626.json`
- `build/bench_results/vllm_warm_equal_hello_out16_c1_compiled_nocg_warm3_20260626.json`
- `build/bench_results/gemma4_vs_vllm_summary_20260626.json`
- `docs/benchmarks/ttft_p50_gemma4_vs_vllm.png`
- `docs/benchmarks/tps_p50_gemma4_vs_vllm.png`

Conclusion:

- The local runner beat the previous README baseline on all three local metrics:
  TTFT `38.469 ms` vs `38.878 ms`, TPOT `73.707 ms` vs `75.120 ms`, and decode
  TPS `13.567 tok/s` vs `13.312 tok/s`.
- Today’s vLLM serving baseline was much slower than the prior vLLM run despite
  repeated warm passes, so the README reports today’s measured number rather
  than carrying forward the older external baseline.

## 2026-06-26 - Full Decode Mechanism Tuning Sweep

Question: after the prompt-runner rewrite, do any remaining decode mechanism
knobs produce a large enough win to keep?

Changes kept:

- Fixed benchmark timing helpers that were recording CUDA events on the default
  stream while launching work on a passed nonblocking stream:
  `gemma4_bench_utils.cuh`, `gemma4_sampling_bench.cu`,
  `gemma4_flash_attention_bench.cu`, and `gemma4_kv_cache_bench.cu`.
- Added `gemma4_prompt --decode-split-size` so decode split-size checks can be
  run from the end-to-end prompt benchmark without changing constants.

Changes rejected:

- Sampling producer-count cache: no material sampler or full decode-step win
  above noise, so the core sampler code was restored.
- Prompt decode split sizes other than the default `20`.
- Sliding attention split sizes `16` and `64`.
- Global attention split sizes `32` and `128`.
- FFN decode swizzle-off and final-logits/sampling retunes; existing defaults
  stayed best or tied within noise.

Commands:

```bash
make prompt decode-bench sampling-bench flash-attn-bench kv-cache-bench \
  rmsnorm-bench rmsnorm-hidden-fused-bench embedding-gather-bench \
  ffn-libtorch-bench test-sampling

./build/gemma4_prompt --benchmark-mode decode-step --bench-warmup 5 \
  --bench-iters 10 --bench-samples 5 --prompt Hello

./build/gemma4_prompt --benchmark-mode decode-step --bench-warmup 5 \
  --bench-iters 10 --bench-samples 5 --prompt Hello --decode-split-size 64

GEMMA4_DECODE_BENCH_SEED=20260626 \
  ./build/benches/gemma4_decode_bench all 50 10 3

GEMMA4_FFN_LIBTORCH_BENCH_SEED=20260626 \
  ./build/benches/gemma4_ffn_libtorch_bench 50 10 3 1

./build/benches/gemma4_sampling_bench 25 100 9

./build/benches/gemma4_kv_cache_bench --seq-len 4096 --page-size 64 \
  --split-size 20 --global-split-size 64 --warmup 20 --iters 80 \
  --samples 5 --cache warm

./build/benches/gemma4_kv_cache_bench --seq-len 4096 --page-size 64 \
  --split-size 16 --global-split-size 64 --warmup 20 --iters 80 \
  --samples 5 --cache warm

./build/benches/gemma4_kv_cache_bench --seq-len 4096 --page-size 64 \
  --split-size 64 --global-split-size 64 --warmup 20 --iters 80 \
  --samples 5 --cache warm

./build/benches/gemma4_kv_cache_bench --seq-len 4096 --page-size 64 \
  --split-size 20 --global-split-size 32 --warmup 20 --iters 80 \
  --samples 5 --cache warm

./build/benches/gemma4_kv_cache_bench --seq-len 4096 --page-size 64 \
  --split-size 20 --global-split-size 128 --warmup 20 --iters 80 \
  --samples 5 --cache warm

./build/benches/gemma4_flash_attention_bench 4096 50 20 5 1 64 warm 64
./build/benches/gemma4_rmsnorm_bench 100 20 5 4096 3840
./build/benches/gemma4_rmsnorm_hidden_fused_bench 100 20 5 1024
./build/benches/gemma4_embedding_gather_bench 100 20 5 4096
```

Contract:

- Hardware: NVIDIA RTX A6000, driver `580.126.16`, persistence enabled, ECC
  disabled, MIG not applicable, power limit `300 W`.
- CUDA: driver/runtime reported as `13000`; `/usr/local/cuda/bin/nvcc`
  reported CUDA compilation tools `13.0`, `V13.0.48`.
- PyTorch/LibTorch: `2.11.0`.
- Timing: CUDA events on the measured stream for microbenchmarks. Prompt
  decode-step timing records one event window per stateful decode step.
- Cache policy: warm repeated buffers unless a benchmark row states otherwise.
- Clock policy: clocks were not locked. Idle checks after runs showed
  `210 MHz` SM and `405 MHz` memory; boost state during timing was not pinned.
- Profiling: `ncu` was not available on `PATH`.
- Correctness: sampling benchmark and `test-sampling` passed. KV-cache and
  flash-attention benchmark correctness rows reported `max_abs=0` for the
  refreshed decode attention rows and sub-BF16-rounding differences for prep.

Key results:

```text
Main decode-step, prompt Hello, split 20:
  baseline before sweep p50=73.581 ms, 13.590 tok/s
  final clean-source rerun p50=73.861 ms, 13.539 tok/s
  one noisy discarded rerun p50=78.914 ms, max=127.016 ms

Prompt decode split-size sweep:
  split 20 p50=73.581 ms baseline
  split 64 confirmation p50=75.048 ms, rejected
  split 16/20/32/64 short checks did not beat split 20

Projection decode bench after stream-timing fix:
  ffn_gate_up custom=0.334930 ms, swizzle16=0.335302 ms
  ffn_down custom=0.170955 ms, swizzle16=0.170703 ms
  sliding_qkv custom=0.092177 ms
  sliding_o custom=0.047699 ms
  global_q custom=0.092042 ms
  global_k custom=0.009842 ms
  global_o custom=0.092320 ms
  final_logits custom=2.827119 ms

FFN fused decode:
  custom_fused_decode graph=0.520573 ms
  swizzle-off graph=0.520601 ms, rejected
  LibTorch full FFN graph=0.515985 ms

Sampling after stream-timing fix:
  native_pytorch_cuda_graph median=2850.747 us
  fused_lm_head_sample_full_vocab median=2837.319 us
  materialized_lm_head_sample_full_vocab median=2915.143 us
  retained existing fused sampler; no new sampler code change

KV-cache / decode attention after stream-timing fix:
  split 20 sliding direct median=0.044968 ms, global split 64=0.333912 ms
  split 16 sliding direct median=0.050586 ms, rejected
  split 64 sliding direct median=0.059744 ms, rejected
  global split 32 median=0.345219 ms, rejected
  global split 128 median=0.650244 ms, rejected

Flash-attention / norm-RoPE prep:
  custom_project_prepare median=0.597396 ms for the full bench shape
  norm_rope_plus_fa median=0.921530 ms
  decode_norm_rope_paged_kv_write median=0.022906 ms

RMSNorm / residual:
  rows=1 fused_graph_kernel_ms=0.002541, split_graph_kernel_ms=0.003511
  rows=4096 fused_ms=0.187128, split_ms=0.234915
  hidden fused graph rows=1024 best_ms=0.047958

Embedding gather:
  tokens=1 best_ms=0.010527
  tokens=4096 best_ms=0.097854, best_effective_gib_s=598.789
```

Conclusion:

- No mechanism retune produced a repeatable end-to-end decode speedup above
  noise on this A6000 run. The retained performance-affecting code defaults
  remain the pre-sweep defaults.
- The important kept work is measurement correctness: several benchmark helpers
  now record events on the actual work stream, which fixed impossible rows such
  as final logits timing and makes the current tuning decisions defensible.
- The prompt split-size knob stays because it is a low-risk measurement affordance
  and it confirmed that the default split size `20` should remain.

## 2026-06-26 - Prompt Warm-Serving vLLM-Style Metrics Smoke

Question:

- Does `gemma4_prompt --benchmark-mode warm-serving` report the same class of
  client-observed metrics as vLLM's online serving benchmark for the local
  batch-1 path?

Change:

- Stopped warm-serving E2E timing at the last copied token instead of including
  local detokenization.
- Added measured-window duration plus request, output-token, and total-token
  throughput rows.
- Labeled the local request contract as sequential, concurrency-1, HTTP-free,
  model-load-excluded serving timing.

Commands:

```bash
make prompt
./build/gemma4_prompt --benchmark-mode warm-serving --bench-warmup 0 \
  --bench-iters 1 --bench-samples 1 --max-new 2 --prompt Hello
git diff --check
```

Contract:

- Hardware: NVIDIA RTX A6000, PCI bus `00000000:0E:00.0`, driver
  `580.126.16`, persistence enabled, ECC disabled, power limit `300 W`.
- Toolchain: `/usr/local/cuda/bin/nvcc`, CUDA `13.0.48`; CUDA driver/runtime
  reported as `13000`.
- Model/input: `models/gemma-4-12B/model.safetensors`, prompt `Hello`,
  prompt length `1`, output tokens `2`.
- Timing: host wall clock around each local request. The request timer includes
  tokenization, prompt H2D, prefill, decode, and host token copies; it excludes
  model load, CUDA context creation, allocation, and local detokenization.
- Traffic shape: closed-loop sequential, concurrency `1`, no HTTP server/client
  overhead and no queueing layer.
- Warmup/iterations: `0` warmup requests, `1` measured request. This is a
  functionality smoke, not a stable benchmark run.
- Cache policy: warm allocated runtime buffers inside one process.
- Clock policy: clocks not locked; post-run idle snapshot showed `210 MHz` SM
  and `405 MHz` memory.

Result:

```text
benchmark_duration_ms=162.938 successful_requests=1 total_input_tokens=1 total_output_tokens=2 request_throughput=6.137 output_token_throughput=12.275 total_token_throughput=18.412
ttft_ms n=1 mean_ms=79.355 p50_ms=79.355 p90_ms=79.355 p95_ms=79.355 p99_ms=79.355 min_ms=79.355 max_ms=79.355
tpot_ms n=1 mean_ms=83.526 p50_ms=83.526 p90_ms=83.526 p95_ms=83.526 p99_ms=83.526 min_ms=83.526 max_ms=83.526
itl_ms n=1 mean_ms=83.526 p50_ms=83.526 p90_ms=83.526 p95_ms=83.526 p99_ms=83.526 min_ms=83.526 max_ms=83.526
e2e_ms n=1 mean_ms=162.880 p50_ms=162.880 p90_ms=162.880 p95_ms=162.880 p99_ms=162.880 min_ms=162.880 max_ms=162.880
per_user_tps n=1 mean=11.972 p50=11.972 p90=11.972 p95=11.972 p99=11.972 min=11.972 max=11.972
```

Conclusion:

- The custom path now has a vLLM-serving-equivalent local benchmark mode for
  single-user batch-1 prompt timing.
- This is not an HTTP load-generator benchmark. It intentionally measures the
  local serving path without network, HTTP parsing, request queueing, or
  concurrency effects.

## 2026-06-27 - vLLM Online Serve Baseline For Local Gemma 4 12B-it

Question:

- What does vLLM's online serving benchmark report for the same local Gemma 4
  12B-it checkpoint on this A6000, using the OpenAI-compatible server path?

Setup notes:

- Installed vLLM `0.23.0` into `/tmp/vllm-bench-venv` to avoid changing the
  repository Python/Torch environment.
- `vllm serve` accepts the local Hugging Face-style model directory
  `models/gemma-4-12B-it` and resolves `Gemma4UnifiedForConditionalGeneration`.
- Startup required `--max-num-batched-tokens 4096` because the unified model's
  multimodal budget gate rejected the default text-only-sized budget.
- Startup required `PATH=/tmp/vllm-bench-venv/bin:$PATH` so FlashInfer's
  sampler JIT could find the `ninja` executable installed in the venv.
- vLLM forced the Triton attention backend for Gemma 4's heterogeneous head
  dimensions and used FlashInfer for top-k/top-p sampling.

Server command:

```bash
PATH=/tmp/vllm-bench-venv/bin:$PATH /tmp/vllm-bench-venv/bin/vllm serve \
  models/gemma-4-12B-it \
  --host 127.0.0.1 --port 8000 \
  --served-model-name gemma4-local \
  --dtype bfloat16 \
  --max-model-len 128 \
  --gpu-memory-utilization 0.90 \
  --tensor-parallel-size 1 \
  --max-num-seqs 32 \
  --max-num-batched-tokens 4096 \
  --no-enable-log-requests \
  --trust-remote-code
```

Benchmark commands:

```bash
PATH=/tmp/vllm-bench-venv/bin:$PATH /tmp/vllm-bench-venv/bin/vllm bench serve \
  --backend openai --base-url http://127.0.0.1:8000 \
  --endpoint /v1/completions --model gemma4-local \
  --tokenizer models/gemma-4-12B-it --trust-remote-code \
  --dataset-name random --random-input-len 1 --random-output-len 1 \
  --num-warmups 5 --num-prompts 50 \
  --request-rate inf --max-concurrency 1 \
  --ignore-eos --temperature 0.0 --top-p 1.0 \
  --percentile-metrics ttft,tpot,itl,e2el \
  --metric-percentiles 50,90,95,99 \
  --save-result --result-dir build/bench_results \
  --result-filename vllm_gemma4_12b_it_online_random_1in_1out_50.json

PATH=/tmp/vllm-bench-venv/bin:$PATH /tmp/vllm-bench-venv/bin/vllm bench serve \
  --backend openai --base-url http://127.0.0.1:8000 \
  --endpoint /v1/completions --model gemma4-local \
  --tokenizer models/gemma-4-12B-it --trust-remote-code \
  --dataset-name random --random-input-len 1 --random-output-len 16 \
  --num-warmups 5 --num-prompts 30 \
  --request-rate inf --max-concurrency 1 \
  --ignore-eos --temperature 0.0 --top-p 1.0 \
  --percentile-metrics ttft,tpot,itl,e2el \
  --metric-percentiles 50,90,95,99 \
  --save-result --result-dir build/bench_results \
  --result-filename vllm_gemma4_12b_it_online_random_1in_16out_30.json
```

Contract:

- Hardware: NVIDIA RTX A6000, PCI bus `00000000:05:00.0`, driver
  `580.159.03`, persistence disabled, ECC disabled, power limit `300 W`.
- Server: vLLM `0.23.0`, BF16, tensor parallel size `1`, max model length
  `128`, max concurrency in benchmark `1`.
- Model/input: `models/gemma-4-12B-it`, random synthetic prompt tokens from
  vLLM's random dataset, `ignore_eos`, greedy temperature `0.0`.
- Timing: vLLM online benchmark over HTTP `/v1/completions`; includes server,
  scheduler, HTTP/client, tokenizer-client setup for request generation, and
  model execution. Excludes vLLM server startup, model load, compile, CUDA graph
  capture, and benchmark warmup requests.
- Warmup: `5` online warmup requests per measured run.
- Cache policy: warm server process with CUDA graphs captured; prefix cache hit
  rate reported as `0.0%`.
- Clock policy: clocks not locked. Idle snapshots before/after showed no other
  GPU allocation; post-run snapshot showed `0 MiB` used after shutdown.

Results:

```text
random 1 input / 1 output, 50 measured requests:
  request_throughput=21.13 req/s
  output_token_throughput=21.13 tok/s
  total_token_throughput=42.26 tok/s
  median_ttft_ms=46.99 p90_ttft_ms=48.23
  median_e2e_ms=46.99 p90_e2e_ms=48.23

random 1 input / 16 output, 30 measured requests:
  request_throughput=1.61 req/s
  output_token_throughput=25.75 tok/s
  total_token_throughput=27.36 tok/s
  median_ttft_ms=84.37 p90_ttft_ms=85.80
  median_tpot_ms=35.66 p90_tpot_ms=36.15
  median_itl_ms=38.18 p90_itl_ms=38.90
  median_e2e_ms=619.32 p90_e2e_ms=628.28
```

Conclusion:

- vLLM online serving is faster than the current local custom warm-serving
  smoke for short generated outputs on this machine, but it is not the same
  measurement contract as CUDA-event decode-step timing.
- For the 16-token online run, the useful decode-ish number is median TPOT
  `35.66 ms`, roughly `28 tok/s` per active request from the TPOT view, while
  measured aggregate output throughput was `25.75 tok/s`.
- The first measured online run logged one Triton JIT warning for
  `_compute_slot_mapping_kernel`, so future formal runs should add a shape-
  covering warmup or repeat a second process after caches are populated.

## 2026-06-27 - Promote Tier-2 FFN Shape To Main Decode Path

Question:

- If the main FFN decode entrypoint uses the Tier-2 scratch-backed MLP plus
  natural-order post-FFN RMSNorm, do the custom FFN microbench and full decode
  path keep correctness and move into the Tier-2 timing band?

Change:

- Added one natural-order MLP row to `Gemma4FfnDecodeScratch`.
- Routed `gemma4_ffn_decode_fused_bf16` through `gemma4_ffn_prefill_mlp_bf16`
  with a one-row scratch view, then a natural-order post-FFN RMSNorm/residual
  kernel.
- Preserved the decode `layer_scalar` behavior in the promoted main path.
- Left `src/gemma4_ffn_tier2.cu` and `src/gemma4_ffn_tier2.cuh` untouched.

Commands:

```bash
make test-ffn-decode
make ffn-libtorch-bench
GEMMA4_FFN_LIBTORCH_BENCH_SEED=20260627 \
  ./build/benches/gemma4_ffn_libtorch_bench 100 20 3 1
make test-decode-megakernel
make prompt
./build/gemma4_prompt \
  --checkpoint models/gemma-4-12B-it/model.safetensors \
  --tokenizer models/gemma-4-12B-it/tokenizer.json \
  --benchmark-mode decode-step \
  --bench-warmup 5 --bench-iters 10 --bench-samples 5 \
  --prompt Hello
```

Contract:

- Hardware: NVIDIA RTX A6000, PCI bus `0000:05:00`, driver/runtime reported by
  CUDA as `13000`, CUDA events on the same stream for the microbench and real
  decode-step timing.
- FFN microbench: BF16, one decode row, fixed seed `0x1352713`, warm repeated
  buffers, `20` warmup iterations, `100` timed iterations, `3` trials.
- Decode-step benchmark: local `models/gemma-4-12B-it` checkpoint, prompt
  `Hello`, prompt length `1`, `5` warmup decode steps, `50` timed decode steps,
  stateful decode cache, setup excluded.
- Clock policy: clocks were not locked.

Results:

```text
FFN microbench:
  tier2_decode_full best_ms=0.529592 avg_ms=0.529667
  custom_fused_decode_events best_ms=0.527913 avg_ms=0.528138
  custom_decode_normed_vs_libtorch max_abs=0.000976562
  custom_decode_residual_vs_libtorch max_abs=0.000976562

Full decode-step:
  decode_step_ms n=50 mean_ms=73.303 p50_ms=73.435
  p90_ms=73.478 p95_ms=73.484 p99_ms=73.504
  min_ms=72.412 max_ms=73.504
  decode_step_tps_p50=13.617
```

Conclusion:

- The promoted main custom FFN decode path now runs in the Tier-2 full-FFN
  timing band and passes the existing FFN and decode-megakernel correctness
  tests.
- The full decode-step path improved modestly versus the previous local run
  (`73.832 ms` p50 before, `73.435 ms` p50 after). This is directionally good,
  but not a large end-to-end change because FFN was already near the Tier-2
  microbench timing before this promotion.

## 2026-06-27 - True B2B FFN Plan Audit And Current Path Rerun

Question:

- Is the current main FFN path the attached true fused gated B2B plan, and if
  not, should it be promoted before benchmarking?

Audit:

- Current main decode FFN still runs the measured decode-layout chain:
  swizzle hidden input, CUTLASS DualGemm gate/up with GeGLU epilogue, CUTLASS
  down GEMM, then post-FFN RMSNorm/residual. It no longer writes the packed
  gate/up output, but it still writes and rereads the `[15360]` GeGLU
  activation.
- The attached plan requires the GeGLU result to be produced by a custom
  GEMM1 A-fragment iterator from GEMM0 accumulators. CUTLASS example 13 has
  the register-resident handoff hook, but its stock kernel schedule ties the
  GEMM0 intermediate tile index to the GEMM1 output tile index. For the full
  FFN, a faithful efficient schedule would need a new kernel schedule that
  loops over all intermediate tiles for each output tile without turning
  gate/up into repeated work across output tiles.
- Back-of-the-envelope for one decode row:
  - gate/up weights: `225.0 MiB`
  - down weights: `112.5 MiB`
  - total FFN weight stream: `337.5 MiB`
  - GeGLU activation spill read+write: `60.0 KiB`
  - naive B2B with 128 output columns per CTA: `30` output groups, about
    `6862.5 MiB` of gate/up+down weight traffic, `20.33x` the current stream.

Commands:

```bash
make test-ffn-decode
make ffn-libtorch-bench
GEMMA4_FFN_LIBTORCH_BENCH_SEED=20260627 \
  ./build/benches/gemma4_ffn_libtorch_bench 100 20 3 1
make test-decode-megakernel
make prompt
./build/gemma4_prompt \
  --checkpoint models/gemma-4-12B-it/model.safetensors \
  --tokenizer models/gemma-4-12B-it/tokenizer.json \
  --benchmark-mode decode-step \
  --bench-warmup 5 --bench-iters 10 --bench-samples 5 \
  --prompt Hello
```

Contract:

- Hardware: NVIDIA RTX A6000, PCI bus `0000:05:00`, driver/runtime reported by
  CUDA as `13000`, CUDA event timing on the measured stream.
- FFN microbench: BF16, one decode row, seed `0x1352713`, warm repeated
  buffers, `20` warmup iterations, `100` timed iterations, `3` trials,
  correctness checked against libtorch.
- Decode-step benchmark: local `models/gemma-4-12B-it` checkpoint, prompt
  `Hello`, prompt length `1`, `5` warmup steps, `50` timed decode steps,
  setup excluded.
- Clock policy: clocks were not locked.

Results:

```text
Correctness:
  test_ffn_decode passed
  test_decode_megakernel passed
  custom_decode_normed_vs_libtorch max_abs=0.000976562
  custom_decode_residual_vs_libtorch max_abs=0.000976562

FFN microbench:
  custom_fused_decode_events best_ms=0.527688 avg_ms=0.527953
  tier2_decode_full best_ms=0.529366 avg_ms=0.529384
  libtorch_full_ffn best_ms=0.526531 avg_ms=0.526664

Full decode-step:
  decode_step_ms n=50 mean_ms=73.271 p50_ms=73.399
  p90_ms=73.442 p95_ms=73.454 p99_ms=73.473
  min_ms=72.382 max_ms=73.473
  decode_step_tps_p50=13.624
```

Conclusion:

- The pasted true B2B plan is not the current path.
- Promoting a stock/example-13-shaped B2B path would be misleading unless the
  kernel schedule is also changed; the iterator alone does not provide the
  full FFN accumulation schedule.
- The current correct path remains in the same timing band as the Tier-2
  benchmark and the full decode-step path remains about `73.4 ms` p50.

## 2026-06-27 - Direct No-Activation-Spill FFN Decode Main Path

Question:

- Can the main decode FFN path move closer to the attached gated B2B plan by
  eliminating the GeGLU activation spill, while preserving correctness and
  benchmarkability?

Change:

- Added a small float accumulator to `Gemma4FfnDecodeScratch`.
- Replaced the one-token main decode MLP path with a direct no-hidden-spill
  accumulator:
  - each intermediate tile computes gate/up dot products;
  - GeGLU is computed in shared/register state;
  - the activated value is immediately folded into the down-projection row;
  - only the final natural-order MLP row is written for the existing BF16
    post-FFN RMSNorm/residual kernel.
- Left the prefill FFN path and `src/gemma4_ffn_tier2.*` unchanged.

Important caveat:

- This is not yet the final CUTLASS `GatedFragmentIteratorA1` B2B
  implementation from the plan. It matches the central dataflow goal, avoiding
  the `[15360]` GeGLU activation HBM write/read in the main decode path, but it
  uses the repo's direct decode accumulator shape rather than CUTLASS B2B
  fragments.

Commands:

```bash
make test-ffn-decode
make ffn-libtorch-bench
GEMMA4_FFN_LIBTORCH_BENCH_SEED=20260627 \
  ./build/benches/gemma4_ffn_libtorch_bench 100 20 3 1
make test-decode-megakernel
make prompt
./build/gemma4_prompt \
  --checkpoint models/gemma-4-12B-it/model.safetensors \
  --tokenizer models/gemma-4-12B-it/tokenizer.json \
  --benchmark-mode decode-step \
  --bench-warmup 5 --bench-iters 10 --bench-samples 5 \
  --prompt Hello
git diff --check
```

Contract:

- Hardware: NVIDIA RTX A6000, PCI bus `0000:05:00`, driver/runtime reported by
  CUDA as `13000`.
- FFN microbench: BF16, one decode row, seed `0x1352713`, warm repeated
  buffers, `20` warmup iterations, `100` timed iterations, `3` trials,
  CUDA-event timing on the measured stream, libtorch correctness reference.
- Decode-step benchmark: local `models/gemma-4-12B-it` checkpoint, prompt
  `Hello`, prompt length `1`, `5` warmup steps, `50` timed decode steps,
  setup excluded.
- Clock policy: clocks were not locked.

Results:

```text
Correctness:
  test_ffn_decode passed
  test_decode_megakernel passed
  git diff --check clean
  custom_decode_normed_vs_libtorch max_abs=0.000488281
  custom_decode_residual_vs_libtorch max_abs=0.000976562

FFN microbench:
  custom_fused_decode_events best_ms=0.505764 avg_ms=0.505863
  custom_fused_decode_wall best_ms=0.505979 avg_ms=0.506019
  tier2_decode_full best_ms=0.529407 avg_ms=0.529431
  libtorch_full_ffn best_ms=0.526633 avg_ms=0.526810

Full decode-step:
  decode_step_ms n=50 mean_ms=72.280 p50_ms=72.414
  p90_ms=72.453 p95_ms=72.461 p99_ms=72.472
  min_ms=71.442 max_ms=72.474
  decode_step_tps_p50=13.810
```

Conclusion:

- The main decode FFN no longer spills the GeGLU activation to HBM.
- FFN decode improved versus the prior main-path microbench
  (`0.527688 ms` best to `0.505764 ms` best, about `4.2%` faster).
- Full decode-step improved versus the prior rerun (`73.399 ms` p50 to
  `72.414 ms` p50, about `1.3%` faster).
- The remaining gap to the attached final design is architectural: replace the
  direct accumulator bridge with the CUTLASS B2B gated fragment iterator and
  schedule once that can be done without regressing the measured path.

## 2026-06-27 - Corrected Decode-Step Benchmark Capacity

Question:

- Does the current direct no-activation-spill FFN path still benchmark cleanly
  when the full prompt decode-step runner allocates enough KV/runtime capacity
  for every per-sample warmup and timed iteration?

Change:

- Fixed the prompt decode-step benchmark capacity accounting from
  `warmup + samples * iters` to `samples * (warmup + iters)`, matching the
  `time_ms` helper's per-sample warmup behavior.
- Made the decode-step benchmark catch print the thrown CUDA error before
  returning failure.

Commands:

```bash
make prompt
make test-ffn-decode
make test-decode-megakernel
GEMMA4_FFN_LIBTORCH_BENCH_SEED=20260627 \
  ./build/benches/gemma4_ffn_libtorch_bench 100 20 3 1
./build/gemma4_prompt \
  --checkpoint models/gemma-4-12B-it/model.safetensors \
  --tokenizer models/gemma-4-12B-it/tokenizer.json \
  --benchmark-mode decode-step \
  --bench-warmup 5 --bench-iters 10 --bench-samples 5 \
  --prompt Hello
git diff --check
```

Contract:

- Hardware: NVIDIA RTX A6000, PCI bus `0000:05:00`, driver `580.159.03`,
  CUDA runtime `13000`.
- FFN microbench: BF16, one decode row, seed `0x1352713`, warm repeated
  buffers, `20` warmup iterations, `100` timed iterations, `3` trials,
  CUDA-event timing on the benchmark stream.
- Decode-step benchmark: local `models/gemma-4-12B-it` checkpoint, prompt
  `Hello`, prompt length `1`, `5` warmup decode steps per sample, `10` timed
  decode steps per sample, `5` samples. Setup and checkpoint load excluded.
- Cache policy: warm repeated buffers for FFN; stateful repeated decode for
  prompt bench. Clocks were not locked.

Results:

```text
Correctness:
  test_ffn_decode passed
  test_decode_megakernel passed
  git diff --check clean
  custom_decode_normed_vs_libtorch max_abs=0.000488281
  custom_decode_residual_vs_libtorch max_abs=0.000976562

FFN microbench:
  custom_fused_decode_events best_ms=0.503521 avg_ms=0.503528
  custom_fused_decode_wall best_ms=0.503473 avg_ms=0.503485
  tier2_decode_full best_ms=0.525906 avg_ms=0.526008
  libtorch_full_ffn best_ms=0.526582 avg_ms=0.526606

Full decode-step:
  decode_step_ms n=5 mean_ms=72.289 p50_ms=72.400
  p95_ms=72.493 p99_ms=72.504 min_ms=71.718 max_ms=72.507
  decode_step_tps_p50=13.812
```

Conclusion:

- The corrected decode-step harness runs the full megakernel path without
  exhausting runtime sequence capacity.
- Current custom FFN decode is about `1.0458x` faster than libtorch full FFN
  and about `1.0447x` faster than `tier2_decode_full` in the warm-cache
  microbench.
- Full decode-step remains essentially unchanged versus the previous successful
  direct-path run: `72.400 ms` p50 now versus `72.414 ms` p50 before.

## 2026-06-27 - Decode Scratch No-Spill Cleanup

Question:

- Can the main decode FFN scratch stop reserving obsolete activation/down
  buffers now that the decode path accumulates the MLP row directly?

Change:

- Removed the stale `act[15360]` and `down[3840]` BF16 arrays from
  `Gemma4FfnDecodeScratch`.
- Left the direct accumulator path unchanged; decode scratch now only contains
  the float final-row accumulator used by the current no-activation-spill path.

Commands:

```bash
make test-ffn-decode
make test-decode-megakernel
make ffn-libtorch-bench
GEMMA4_FFN_LIBTORCH_BENCH_SEED=20260627 \
  ./build/benches/gemma4_ffn_libtorch_bench 100 20 3 1
make prompt
./build/gemma4_prompt \
  --checkpoint models/gemma-4-12B-it/model.safetensors \
  --tokenizer models/gemma-4-12B-it/tokenizer.json \
  --benchmark-mode decode-step \
  --bench-warmup 5 --bench-iters 10 --bench-samples 5 \
  --prompt Hello
git diff --check
```

Results:

```text
Correctness:
  test_ffn_decode passed
  test_decode_megakernel passed
  prompt decode-step path completed
  custom_decode_normed_vs_libtorch max_abs=0.000488281
  custom_decode_residual_vs_libtorch max_abs=0.000976562

FFN microbench:
  custom_fused_decode_events best_ms=0.503583 avg_ms=0.503613
  custom_fused_decode_wall best_ms=0.503532 avg_ms=0.503553
  tier2_decode_full best_ms=0.526008 avg_ms=0.526189
  libtorch_full_ffn best_ms=0.525691 avg_ms=0.525933

Full decode-step:
  decode_step_ms n=5 mean_ms=72.281 p50_ms=72.416
  p95_ms=72.421 p99_ms=72.421 min_ms=71.742 max_ms=72.422
  decode_step_tps_p50=13.809
```

Conclusion:

- The decode scratch shrink preserves correctness and full-path execution.
- FFN timing is unchanged within noise; the cleanup is architectural, not a
  speed win.
- The current path is still the direct/Tier-3-style no-activation-spill path,
  not the final CUTLASS `GatedFragmentIteratorA1` B2B schedule.

## 2026-06-27 - Gated B2B FFN Plan Check

Question:

- Is the attached gated B2B plan the right target for eliminating FFN
  activation HBM traffic?
- Does the current WMMA proof-of-concept deserve promotion into the hot decode
  path?

Change:

- Added an opt-in WMMA gated-B2B decode prototype and benchmark hook behind
  `GEMMA4_FFN_BENCH_WMMA_B2B=1`.
- Left the production decode FFN path unchanged because the prototype is
  correct but far too slow.

Commands:

```bash
make test-ffn-decode
make ffn-libtorch-bench
GEMMA4_FFN_LIBTORCH_BENCH_SEED=20260627 \
  ./build/benches/gemma4_ffn_libtorch_bench 100 20 3 1
GEMMA4_FFN_BENCH_WMMA_B2B=1 GEMMA4_FFN_LIBTORCH_BENCH_SEED=20260627 \
  ./build/benches/gemma4_ffn_libtorch_bench 1 0 1 1
make test-decode-megakernel
make prompt
./build/gemma4_prompt \
  --checkpoint models/gemma-4-12B-it/model.safetensors \
  --tokenizer models/gemma-4-12B-it/tokenizer.json \
  --benchmark-mode decode-step \
  --bench-warmup 5 --bench-iters 10 --bench-samples 5 \
  --prompt Hello
```

Results:

```text
Correctness:
  test_ffn_decode passed
  test_decode_megakernel passed
  wmma_b2b_decode_normed_vs_libtorch max_abs=0.000976562
  wmma_b2b_decode_residual_vs_libtorch max_abs=0.000976562

Default FFN microbench:
  custom_fused_decode_events best_ms=0.503593 avg_ms=0.503631
  custom_fused_decode_wall best_ms=0.503745 avg_ms=0.503756
  tier2_decode_full best_ms=0.529930 avg_ms=0.530012
  libtorch_full_ffn best_ms=0.526961 avg_ms=0.527046

Opt-in WMMA B2B prototype:
  wmma_b2b_decode_full best_ms=180.686844 avg_ms=180.686844

Full decode-step:
  decode_step_ms n=5 mean_ms=72.780 p50_ms=72.919
  p95_ms=72.925 p99_ms=72.926 min_ms=72.227 max_ms=72.927
  decode_step_tps_p50=13.714
```

Conclusion:

- The attached plan is the right target: compute gate/up, apply GeGLU, and feed
  the down GEMM without writing the `[1, 30720]` temporary or `[1, 15360]`
  activation to HBM.
- The current production path already avoids the large activation HBM spill via
  direct accumulation, but it is not the final CUTLASS B2B fragment-iterator
  schedule.
- The WMMA prototype proves the math but is not promotable. Its naive schedule
  uses too little tensor-core parallelism and too much recomputation, so the
  next useful implementation step is a real CUTLASS B2B fork with a gated
  `FragmentIteratorA1` where `GEMM0_N_tile == 2 * GEMM1_K_tile`.

## 2026-06-27 - FFN Slow-Path Cleanup

Question:

- Can the live FFN code drop old slower paths and stale benchmark plumbing
  while preserving the current direct no-activation-spill decode path?

Change:

- Deleted the non-promotable WMMA gated-B2B prototype API and benchmark/test
  hooks.
- Deleted `src/gemma4_ffn_tier2.cu` and `src/gemma4_ffn_tier2.cuh`.
- Replaced `gemma4_ffn_libtorch_bench` with a current-path-only benchmark:
  libtorch full FFN reference versus `gemma4_ffn_decode_fused_bf16`.
- Removed the test-only `gemma4_ffn_bf16` wrapper and its private prefill
  RMSNorm/residual kernel. The test now exercises the explicit production
  prefill sequence: FFN MLP, RMSNorm, residual add.

Commands:

```bash
make test-ffn-decode ffn-libtorch-bench
GEMMA4_FFN_LIBTORCH_BENCH_SEED=20260627 \
  ./build/benches/gemma4_ffn_libtorch_bench 100 20 3
make test-decode-megakernel
git diff --check
```

Results:

```text
Correctness:
  test_ffn_decode passed
  test_decode_megakernel passed

FFN microbench:
  libtorch_full_ffn best_ms=0.528302 avg_ms=0.528411
  custom_fused_decode best_ms=0.503716 avg_ms=0.503859
  custom_decode_normed_vs_libtorch max_abs=0.000488281
  custom_decode_residual_vs_libtorch max_abs=0.000976562
  custom_decode_vs_libtorch_full_speedup=1.048810x

Cleanup size:
  src/gemma4_ffn.cu: 1075 lines before cleanup, 754 after
  live FFN cleanup diff: 607 deletions, 130 insertions
```

Follow-up:

- The prompt decode-step benchmark currently fails in the unrelated dirty
  flash-attention/runtime path with an illegal memory access during the first
  decode warmup. The standalone megakernel API smoke test still passes, but
  that prompt-level failure must be fixed before using full decode-step timing
  as completion evidence.

## 2026-06-27 - Lazy Global Pages and V-Cache Global Decode Check

Question:

- Can the runtime avoid allocating full max-context global KV storage upfront?
- Is reconstructing global K from the cached normalized source numerically sane,
  and what is the first decode-timing signal?

Change:

- Runtime global cache storage now starts at zero physical pages and grows to
  the pages reached by prefill/decode metadata.
- Added an experimental global decode entry point that reads only cached
  scale-free V/source and reconstructs K with `k_norm_weight` plus p-RoPE inside
  the attention loop.
- Left the production global decode/runtime path on stored K/V until the
  one-cache path has stronger benchmark coverage.

Commands:

```bash
make test-runtime-state
make test-kv-cache
make test-flash-attention-cpp
make flash-attn-bench
./build/benches/gemma4_flash_attention_bench 1024 3 1 2 1 0 warm 64
```

Results:

```text
Correctness:
  runtime state tests passed
  kv cache tests passed
  flash attention tests passed

Benchmark environment:
  GPU: NVIDIA RTX A6000, sm_86, CUDA runtime 13.0, driver 580.159.03
  Timing: CUDA events on the benchmark stream, warm cache
  Warmup: 1
  Iters/sample: 3
  Samples: 2
  Seq: 1024
  Batch: 1

Global decode:
  global_decode_stored_kv median_ms=5.150037 min_ms=5.145600 max_ms=5.154475
  global_decode_vcache_reconstruct median_ms=5.243733 min_ms=5.242198 max_ms=5.245269
```

Conclusion:

- Lazy global physical pages are correct in the focused runtime tests and avoid
  reserving max-context global cache storage during runtime initialization.
- The V-cache reconstruction path is numerically covered against a CPU reference
  for the proposed BF16-cached-source semantics.
- In this quick warm-cache check, reconstructing K is about 2% slower than
  reading stored global K/V at seq 1024, despite halving global cache storage.
  Keep it experimental until longer-context and cold-cache runs show whether the
  bandwidth savings overcome the extra reconstruction math.

Follow-up correction:

- The lazy global physical-page change was backed out of the production runtime
  path after the full prompt decode-step runner exposed an illegal memory
  access. The existing cache offset math uses `num_pages` as the per-layer
  stride, so physical allocation cannot shrink independently without a real
  layout change.
- The experimental V-cache global decode entry point was later removed from the
  live tree during the cleanup pass. The measured slower result and the template
  regression made it useful as an experiment note only, not as parked API.

## 2026-06-27 - Decode Megakernel V-Cache Template Regression Fix

Question:

- Why did the full prompt decode-step benchmark fault even though the focused
  runtime, KV-cache, flash-attention, and decode-megakernel tests passed?

Change:

- Fixed the cooperative decode megakernel call to
  `phase_decode_paged_grouped_split` after the experimental V-cache path added
  a new boolean template parameter. The megakernel now passes
  `false, kDecodeMegaThreads` explicitly instead of accidentally treating
  `kDecodeMegaThreads` as `ReconstructGlobalK=true`.
- Updated the Python flash-attention ctypes `Gemma4KvCacheConfig` mirror for
  the new `batch_size` field.
- Restored the production runtime to capacity-sized global KV allocation. The
  lazy physical-page experiment needs a different cache layout before it can be
  safe in the full path.

Commands:

```bash
make test-runtime-state test-kv-cache test-flash-attention-cpp test-decode-megakernel prompt
make test-flash-attention-pytorch
compute-sanitizer --tool memcheck --print-limit 5 \
  ./build/gemma4_prompt \
    --checkpoint models/gemma-4-12B-it/model.safetensors \
    --tokenizer models/gemma-4-12B-it/tokenizer.json \
    --benchmark-mode decode-step \
    --bench-warmup 0 --bench-iters 1 --bench-samples 1 \
    --prompt Hello
./build/gemma4_prompt \
  --checkpoint models/gemma-4-12B-it/model.safetensors \
  --tokenizer models/gemma-4-12B-it/tokenizer.json \
  --benchmark-mode decode-step \
  --bench-warmup 5 --bench-iters 10 --bench-samples 5 \
  --prompt Hello
```

Results:

```text
Correctness:
  runtime state tests passed
  kv cache tests passed
  flash attention tests passed
  test_decode_megakernel passed
  flash attention PyTorch decode parity passed
  compute-sanitizer ERROR SUMMARY: 0 errors

Full decode-step:
  decode_step_ms n=5 mean_ms=72.263 p50_ms=72.388
  p95_ms=72.407 p99_ms=72.410 min_ms=71.744 max_ms=72.411
  decode_step_tps_p50=13.814
```

Conclusion:

- The illegal access was a template-argument regression from the V-cache
  experiment, not a math issue in the attention kernel.
- The current branch has a clean repeatable main decode-step baseline again:
  `72.388 ms` p50, or `13.814 tok/s`, for the local one-token prompt
  decode-step benchmark.

## 2026-06-27 - Long Single-User Serving Benchmark vs vLLM and SGLang Setup

Question:

- What is the current long closed-loop single-user benchmark result for the
  local runner against `vllm bench serve`, and can SGLang run the same Gemma 4
  Unified checkpoint on this host?

Setup:

- Installed SGLang `0.5.9` in `/tmp/sglang-bench-venv`.
- Upgraded only that venv's Transformers package from `4.57.1` to `5.12.1`
  after the first SGLang launch failed to recognize `model_type:
  gemma4_unified`.
- Reused the existing vLLM `0.23.0` environment in `/tmp/vllm-bench-venv`.

Commands:

```bash
./build/gemma4_prompt \
  --checkpoint models/gemma-4-12B-it/model.safetensors \
  --tokenizer models/gemma-4-12B-it/tokenizer.json \
  --benchmark-mode warm-serving \
  --bench-warmup 3 --bench-iters 391 --bench-samples 1 \
  --max-new 256 --prompt Hello \
  | tee build/bench_results/gemma4_prompt_warm_hello_out256_c1_391req_20260627.txt

PATH=/tmp/vllm-bench-venv/bin:$PATH /tmp/vllm-bench-venv/bin/vllm serve \
  models/gemma-4-12B-it \
  --host 127.0.0.1 --port 8000 \
  --served-model-name gemma4-local \
  --dtype bfloat16 \
  --max-model-len 512 \
  --gpu-memory-utilization 0.90 \
  --tensor-parallel-size 1 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 4096 \
  --no-enable-log-requests \
  --trust-remote-code

PATH=/tmp/vllm-bench-venv/bin:$PATH /tmp/vllm-bench-venv/bin/vllm bench serve \
  --backend openai \
  --base-url http://127.0.0.1:8000 \
  --endpoint /v1/completions \
  --model gemma4-local \
  --tokenizer models/gemma-4-12B-it \
  --trust-remote-code \
  --dataset-name custom \
  --dataset-path build/bench_results/hello_prompt_out256_391x.jsonl \
  --skip-chat-template \
  --output-len 256 \
  --num-warmups 5 \
  --num-prompts 391 \
  --request-rate inf \
  --max-concurrency 1 \
  --ignore-eos \
  --temperature 0.0 \
  --top-p 1.0 \
  --percentile-metrics ttft,tpot,itl,e2el \
  --metric-percentiles 50,90,95,99 \
  --save-result \
  --result-dir build/bench_results \
  --result-filename vllm_gemma4_12b_it_online_hello_1in_256out_391_20260627.json \
  --disable-tqdm

PATH=/tmp/sglang-bench-venv/bin:$PATH /tmp/sglang-bench-venv/bin/python \
  -m sglang.launch_server \
  --model-path models/gemma-4-12B-it \
  --host 127.0.0.1 --port 30000 \
  --served-model-name gemma4-local \
  --dtype bfloat16 \
  --context-length 128 \
  --mem-fraction-static 0.80 \
  --max-running-requests 1 \
  --max-total-tokens 4096 \
  --trust-remote-code \
  --attention-backend torch_native \
  --disable-cuda-graph \
  --log-level info
```

Contract:

- Hardware: NVIDIA RTX A6000, driver `580.159.03`, power limit `300 W`.
- Local runner: in-process host wall-clock `warm-serving` timing; model load,
  CUDA context, and allocation excluded.
- vLLM: HTTP `/v1/completions` serving path; server startup, model load,
  `torch.compile`, CUDA graph capture, and benchmark warmups excluded.
- Shape: prompt `Hello`, prompt length `1`, output length `256`, measured
  requests `391`, measured output tokens `100,096`, closed-loop single user
  with requested max concurrency `1`.
- Clock policy: clocks were not locked.
- vLLM caveat: `vllm bench serve` reported `Maximum request concurrency: 1`
  but `Peak concurrent requests: 2.00`; server logs showed one running request
  and no queue through the sampled run. One Triton JIT warning for
  `_compute_slot_mapping_kernel` appeared during warmup/initial traffic.

Results:

```text
gemma4_prompt local:
  benchmark_duration_ms=7334893.000
  total_output_tokens=100096
  output_token_throughput=13.647 tok/s
  p50 TTFT=37.607 ms
  p50 TPOT=73.418 ms
  p50 E2E=18759.113 ms
  p50 per_user_tps=13.621

vLLM serve:
  benchmark_duration_s=3841.36
  total_output_tokens=100096
  output_token_throughput=26.057 tok/s
  p50 TTFT=85.414 ms
  p50 TPOT=38.191 ms
  p50 E2E=9823.893 ms
```

SGLang outcome:

- SGLang `0.5.9` plus Transformers `5.12.1` could parse
  `Gemma4UnifiedConfig`, load weights, and start its server.
- Generation failed in the generic Transformers wrapper:
  `RuntimeError: shape '[-1, 8, 256]' is invalid for input of size 9216`.
- The failure is consistent with SGLang's generic `RadixAttention` wrapper
  assuming homogeneous KV geometry, while Gemma 4 Unified alternates sliding
  layers and global layers with different KV/head shapes.
- SGLang is therefore recorded as unsupported for this exact checkpoint/version
  instead of charting a fake number.

Artifacts:

- `build/bench_results/gemma4_prompt_warm_hello_out256_c1_391req_20260627.txt`
- `build/bench_results/hello_prompt_out256_391x.jsonl`
- `build/bench_results/vllm_gemma4_12b_it_online_hello_1in_256out_391_20260627.json`
- `build/bench_results/gemma4_vllm_sglang_long_summary_20260627.json`
- `docs/benchmarks/ttft_p50_gemma4_vllm_sglang_long.png`
- `docs/benchmarks/tps_gemma4_vllm_sglang_long.png`

Conclusion:

- The local runner wins TTFT on this long single-user benchmark, but vLLM
  produces tokens about `1.91x` faster end-to-end (`26.057 / 13.647`).
- The README now reports the long closed-loop comparison rather than the old
  short 16-token comparison.

## 2026-06-30 - FFN decode B2B register experiment

Commands:

```bash
make test-ffn-decode
make ffn-libtorch-bench
./build/benches/gemma4_ffn_libtorch_bench 20 5 3
ncu --target-processes all --kernel-name-base demangled \
  --kernel-name 'regex:b2b_decode_ffn_kernel' --launch-count 1 \
  --metrics gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes.sum,lts__t_bytes.sum,smsp__warps_active.avg.pct_of_peak_sustained_active,launch__grid_size,launch__block_size,launch__registers_per_thread \
  --page raw ./build/benches/gemma4_ffn_libtorch_bench 1 0 1
ncu --target-processes all --kernel-name-base demangled \
  --kernel-name 'regex:b2b_decode_ffn_kernel' --launch-count 1 \
  --section LaunchStats --section Occupancy --section SpeedOfLight \
  --section MemoryWorkloadAnalysis --section WarpStateStats \
  --print-details all ./build/benches/gemma4_ffn_libtorch_bench 1 0 1
compute-sanitizer --tool memcheck ./build/tests/test_ffn_decode
```

Contract:

- Hardware: NVIDIA RTX A6000, driver `580.159.03`, CUDA runtime `13000`.
- Timing: CUDA events on the benchmark stream, warm repeated buffers,
  launch overhead excludes host enqueue, `warmup=5`, `iters=20`, `trials=3`.
- Correctness: custom, dual-GEMM, and B2B outputs compared with libtorch BF16.
- Clock policy: clocks were not locked; persistence mode disabled.

Results:

```text
test_ffn_decode: passed
compute-sanitizer memcheck: ERROR SUMMARY: 0 errors

custom_fused_decode best=0.504525 ms avg=0.504662 ms max_abs=0.000976562
b2b_register_decode best=18.506752 ms avg=18.507607 ms max_abs=0.000976562

ncu b2b_decode_ffn_kernel:
  launch = (15 blocks) x (480 threads)
  waves_per_sm = 0.09
  registers_per_thread = 54
  achieved_occupancy = 31.25%
  SM throughput = 4.71%
  DRAM throughput = 1.97%
  tensor pipe active = 0%
  FMA pipe active = 8.60%
  top stall class = Stall Barrier, 4.87 cycles per issued instruction

tensor-core chain ncu:
  DualGemm gate/up+GeGLU = 358.14 us, grid=240, tensor pipe=39.32%, DRAM=91.26%
  down GEMM = 178.08 us, grid=60, tensor pipe=51.75%, DRAM=92.50%
```

Conclusion:

- The B2B experiment is numerically correct and memory-sanitizer clean.
- Moving from one CTA to 15 CTAs improved the old `21.545 ms` result to
  `18.507 ms`, but the kernel still cannot fill the GPU.
- The remaining issue is the schedule: each hidden-pack group recomputes the
  gate/up dot products, while shrinking the group further would increase that
  recomputation. A real B2B kernel needs a CUTLASS-style decomposition that
  shares gate/up work across enough CTAs instead of only sharding hidden packs.
- The existing CUTLASS DualGemm/down chain already uses tensor cores and streams
  weights near the DRAM roofline, but it materializes the GeGLU activation.

## 2026-06-30 - FFN decode tensor-core tile tune

Commands:

```bash
make test-ffn-decode ffn-libtorch-bench
./build/benches/gemma4_ffn_libtorch_bench 100 20 7
ncu --target-processes all --kernel-name-base demangled \
  --kernel-name 'regex:cutlass::Kernel' --launch-count 2 \
  --metrics gpu__time_duration.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes.sum,lts__t_bytes.sum,launch__grid_size,launch__block_size,launch__registers_per_thread \
  --csv --page raw ./build/benches/gemma4_ffn_libtorch_bench 1 0 1
```

Contract:

- Hardware: NVIDIA RTX A6000, driver `580.159.03`, CUDA runtime `13000`.
- Timing: CUDA events on the benchmark stream, warm repeated buffers,
  `warmup=20`, `iters=100`, `trials=7`; graph rows capture repeated work
  outside the timing window and time CUDA graph replay.
- Correctness: custom, DualGemm-chain, B2B, and tensor-core decode outputs
  compared with LibTorch BF16; max absolute error stayed `<= 0.000976562`.
- Clock policy: clocks were not locked; persistence mode disabled.

Change:

- Added graph-replay timing rows to `gemma4_ffn_libtorch_bench` for the
  custom, DualGemm-chain, and tensor-core decode paths.
- Kept decode-only CUTLASS tiles:
  - gate/up DualGemm rows `== 1`: `16x64x64`, warp `16x32`, stages `4`.
  - down GEMM rows `== 1`: `64x64x64`, warp `32x32`, stages `6`.
- Rejected measured probes that regressed or failed to improve enough:
  down `Ntile=32`, down `Ntile=128`, gate/up `K32` stage variants,
  gate/up warp `16x64`, down warp `32x64`, and down stages `5`.

Results:

```text
test_ffn_decode: passed

Before row-1 tile tuning:
  tensorcore_decode best ~= 0.5255 ms
  tensorcore_decode graph best ~= 0.5221 ms

Final benchmark:
  custom_fused_decode best=0.503941 ms avg=0.503966 ms
  tensorcore_decode best=0.511580 ms avg=0.511775 ms
  dualgemm_chain_decode_layout best=0.515480 ms avg=0.515584 ms
  b2b_register_decode best=18.638910 ms avg=18.639786 ms
  tensorcore_vs_custom_decode_speedup=0.985068

Graph replay:
  custom_fused_decode best=0.502957 ms avg=0.503094 ms
  tensorcore_decode best=0.508252 ms avg=0.508488 ms
  tensorcore_graph_vs_custom_graph_speedup=0.989581

Nsight Compute, final tensor-core kernels:
  gate/up DualGemm = 341.984 us, grid=240, block=64, DRAM=95.50%
  down GEMM       = 171.104 us, grid=60,  block=128, DRAM=94.70%
```

Conclusion:

- The tensor-core decode path is now competitive with the custom CUDA-core
  fused path: within about `1.5%` on stream-loop timing and `1.0%` on graph
  replay, while using real tensor-core CUTLASS GEMMs and high CTA concurrency.
- It should stay as the separate tensor-core candidate path, not replace the
  current custom fused default yet. The remaining gap is mostly structural:
  input swizzle, activation materialization, post RMS/residual, and two GEMM
  launch boundaries around DRAM-saturated weight streams.

## 2026-06-30 - CuTe MMA FFN decode probe and tensor path removal

Research sources:

- NVIDIA CuTe MMA atom docs:
  <https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0t_mma_atom.html>
- NVIDIA CuTe GEMM tutorial:
  <https://docs.nvidia.com/cutlass/4.3.1/media/docs/cpp/cute/0x_gemm_tutorial.html>
- Local CUTLASS tutorial:
  `third_party/cutlass/examples/cute/tutorial/sgemm_sm80.cu`
- Local working CuTe MMA pattern:
  `src/gemma4_flash_attention.cu`

Commands:

```bash
exa-ai search "CuTe MMA_Atom SM80_16x8x16_F32BF16BF16F32_TN TiledMMA cute::gemm example" \
  --num-results 5 --output-format toon
exa-ai search "SM80_16x8x16_F32BF16BF16F32_TN cute::gemm bf16 example TiledMMA" \
  --num-results 5 --output-format toon
make test-ffn-decode ffn-libtorch-bench
./build/benches/gemma4_ffn_libtorch_bench 100 20 5
```

Contract:

- Hardware: NVIDIA RTX A6000, driver `580.159.03`, CUDA runtime `13000`.
- Timing: CUDA events on the benchmark stream, warm repeated buffers,
  `warmup=20`, `iters=100`, `trials=5`.
- Correctness: custom, DualGemm-chain, and B2B outputs compared with LibTorch
  BF16; max absolute error stayed `<= 0.000976562`.
- Clock policy: clocks were not locked; persistence mode disabled.

Results:

```text
CuTe one-warp gate/up tile:
  custom_fused_decode best ~= 2.592604 ms

CuTe 15-warp CTA block:
  custom_fused_decode best ~= 1.202780 ms

CuTe per-warp A staging:
  custom_fused_decode best ~= 1.207091 ms

Restored scalar fused path after removing losing CuTe code:
  custom_fused_decode best=0.503521 ms avg=0.509897 ms
  custom_fused_decode graph best=0.502374 ms avg=0.504084 ms
  dualgemm_chain_decode_layout best=0.515287 ms avg=0.515387 ms
  b2b_register_decode best=18.664488 ms avg=18.675772 ms
```

Tensor path removal:

- Removed the separate `tensorcore_decode` implementation files from the build.
- Removed tensorcore decode calls and correctness rows from `test_ffn_decode`.
- Removed tensorcore decode timing, graph timing, CSV rows, and speedup rows
  from `gemma4_ffn_libtorch_bench`.
- Live `tensorcore_decode` search hits are now only historical experiment notes.

Conclusion:

- The CuTe MMA atom probe worked functionally, but the local M=1 B2B schedule
  was slower than the existing scalar fused path, so the CuTe code was not kept.
- The losing variants paid shared-memory/LDSM staging cost while using only one
  logical GEMM row, and the down fold still serialized around the fused output
  accumulation.
- The current production candidate remains `custom_fused_decode`; the
  `b2b_register_decode` file remains a separate experiment showing true B2B
  dataflow but not a viable tensor-core schedule.

## 2026-06-30 - FFN decode Nsight slowdown audit

Commands:

```bash
make ffn-libtorch-bench
ncu --target-processes all --kernel-name-base demangled \
  --kernel-name 'regex:direct_decode_ffn_kernel' --launch-count 1 \
  --metrics gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes.sum,lts__t_bytes.sum,smsp__warps_active.avg.pct_of_peak_sustained_active,launch__grid_size,launch__block_size,launch__registers_per_thread \
  --page raw ./build/benches/gemma4_ffn_libtorch_bench 1 0 1
ncu --target-processes all --kernel-name-base demangled \
  --kernel-name 'regex:b2b_decode_ffn_kernel' --launch-count 1 \
  --metrics gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes.sum,lts__t_bytes.sum,smsp__warps_active.avg.pct_of_peak_sustained_active,launch__grid_size,launch__block_size,launch__registers_per_thread \
  --page raw ./build/benches/gemma4_ffn_libtorch_bench 1 0 1
ncu --target-processes all --kernel-name-base demangled \
  --kernel-name 'regex:cutlass::Kernel' --launch-count 2 \
  --metrics gpu__time_duration.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes.sum,lts__t_bytes.sum,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,smsp__warps_active.avg.pct_of_peak_sustained_active,launch__grid_size,launch__block_size,launch__registers_per_thread \
  --page raw --csv ./build/benches/gemma4_ffn_libtorch_bench 1 0 1
```

Contract:

- Hardware: NVIDIA RTX A6000, driver `580.159.03`, CUDA runtime `13000`.
- Nsight Compute replay profiling, one matched launch per kernel group.
- Benchmark correctness stayed within `<= 0.000976562` max absolute error.
- Clock policy: clocks were not locked; persistence mode disabled.

Results:

```text
direct_decode_ffn_kernel:
  ncu_time = 511.52 us
  grid = 168 CTAs x 480 threads
  waves_per_sm = 1.00
  registers_per_thread = 56
  dram_bytes = 356.02 MB
  l2_bytes = 357.86 MB
  dram_throughput = 95.49%
  sm_throughput = 18.50%
  active_warps = 62.44%

b2b_decode_ffn_kernel:
  ncu_time = 24.75 ms
  grid = 15 CTAs x 480 threads
  waves_per_sm = 0.09
  registers_per_thread = 54
  dram_bytes = 356.43 MB
  l2_bytes = 3.66 GB
  dram_throughput = 1.98%
  sm_throughput = 4.71%
  active_warps = 31.24%

CUTLASS DualGemm gate/up:
  ncu_time = 341.92 us
  grid = 240 CTAs x 64 threads
  dram_bytes = 238.12 MB
  dram_throughput = 95.55%
  tensor_pipe = 8.95%

CUTLASS down GEMM:
  ncu_time = 173.76 us
  grid = 60 CTAs x 128 threads
  dram_bytes = 120.09 MB
  dram_throughput = 94.83%
  tensor_pipe = 36.54%
```

Conclusion:

- The fast scalar fused kernel is already DRAM-saturated. Replacing its tiny
  scalar dot with an MMA tile does not remove the dominant weight traffic.
- The separate B2B register schedule is slow because it exposes only `15` CTAs
  and `0.09` waves per SM; it cannot create enough independent memory work.
- The tensor-core chain is fast only when CUTLASS supplies hundreds of CTAs.
  It still loses to the fused scalar path because it materializes activation
  data and pays two GEMM launch boundaries.
- A useful CuTe B2B attempt needs to preserve high CTA concurrency and avoid
  M=1 tensor-core row waste; a local pack-dot replacement cannot do that.

## 2026-06-30 - Global decode QK projection A/B

Commands:

```bash
make prompt
make test-decode-megakernel
./build/gemma4_prompt --checkpoint models/gemma-4-12B-it/model.safetensors \
  --tokenizer models/gemma-4-12B-it/tokenizer.json --prompt Hello \
  --max-new 2 --benchmark-mode warm-serving --bench-warmup 5 \
  --bench-iters 80 --bench-samples 1
./build/gemma4_prompt --checkpoint models/gemma-4-12B-it/model.safetensors \
  --tokenizer models/gemma-4-12B-it/tokenizer.json --prompt Hello \
  --max-new 256 --benchmark-mode warm-serving --bench-warmup 0 \
  --bench-iters 1 --bench-samples 1
```

Contract:

- Hardware: NVIDIA RTX A6000, driver `580.159.03`.
- Host-visible warm-serving latency, batch 1, no CUDA graphs, launch overhead
  included, warm cache/process state after model load.
- Clocks were not locked; persistence mode disabled.
- Minimum useful effect size for the short run was treated as about `1%`; close
  calls were rerun with `max-new=256` for 255 ITL samples.

Results:

```text
separate global Q + K, max-new=2:
  tpot_p50 = 36.611 ms

sampled_hidden = hidden_a experiment, max-new=2:
  tpot_p50 = 36.626 ms

packed global QK, max-new=2:
  tpot_p50 = 36.608 ms

packed global QK, max-new=256:
  itl_p50 = 38.084 ms
  itl_mean = 38.205 ms

separate global Q + K, max-new=256:
  itl_p50 = 38.154 ms
  itl_mean = 38.269 ms
```

Conclusion:

- Packing global Q+K into one decode GEMV kept correctness smoke tests passing
  and improved the longer no-graph run by about `70 us/token` p50.
- The sampled-hidden alias removed a D2D copy but did not improve the measured
  prompt path, so it was not kept.

## 2026-06-30 - Nsight Systems decode launch-gap check

Commands:

```bash
./build/gemma4_prompt --checkpoint models/gemma-4-12B-it/model.safetensors \
  --tokenizer models/gemma-4-12B-it/tokenizer.json --prompt Hello \
  --max-new 4 --benchmark-mode warm-serving --bench-warmup 1 \
  --bench-iters 1 --bench-samples 1

nsys profile --force-overwrite=true --trace=cuda,nvtx,osrt \
  --sample=none --cpuctxsw=none --cuda-memory-usage=false --stats=true \
  -o build/nsys/decode_step_gaps_10 \
  ./build/gemma4_prompt --checkpoint models/gemma-4-12B-it/model.safetensors \
    --tokenizer models/gemma-4-12B-it/tokenizer.json --prompt Hello \
    --benchmark-mode decode-step --bench-warmup 1 --bench-iters 10 \
    --bench-samples 1 --max-new 4
```

Contract:

- Hardware: NVIDIA RTX A6000, driver `580.159.03`.
- Nsight Systems `2025.3.2.474-253236389321v0`.
- No CUDA graphs. Decode-step mode after one prefill; one profiled warmup
  decode execution followed by 10 timed decode executions.
- GPU activity gaps were computed from the exported SQLite by partitioning
  decode executions on `final_logits_sample_kernel` plus the following D2D
  hidden-row copy. GPU memcpys/memsets were counted as activity, not idle.
- Clocks were not locked; persistence mode disabled.

Results:

```text
decode_step_ms under nsys:
  p50 = 37.108 ms

per timed decode step:
  activities = 201
  kernels = 194
  tiny H2D metadata copies = 5
  D2D hidden copy = 1

all GPU activity gaps, 10 timed decode steps:
  idle_gap_mean = 149.394 us/token
  idle_gap_median = 146.450 us/token
  idle_gap_p95 = 165.573 us/token
  idle_gap_max = 176.833 us/token
  idle_gap_pct_of_step_total = 0.4026%

individual gaps across all timed steps:
  median = 0.704 us
  p95 = 0.832 us
  p99 = 1.408 us
  max = 8.704 us
```

Conclusion:

- Kernel launch gaps are not the current bottleneck for steady decode on this
  run. CUDA Graphs can still simplify/reduce launch overhead, but the measured
  upside from removing GPU idle launch bubbles is well under `1%` here.
- The largest visible gaps are around the tiny per-step H2D metadata copies,
  but even those are single-digit microseconds in this isolated decode-step
  trace.

## 2026-06-30 - Decode fused-ingress A/B

Command:

```bash
./build/gemma4_prompt --checkpoint models/gemma-4-12B-it/model.safetensors \
  --tokenizer models/gemma-4-12B-it/tokenizer.json --prompt Hello \
  --benchmark-mode decode-step --bench-warmup 5 --bench-iters 10 \
  --bench-samples 5 --max-new 4
```

Contract:

- Hardware: NVIDIA RTX A6000, driver `580.159.03`.
- Full decode-step timing, batch 1, no CUDA graphs. Clocks were not locked.
- Compared the current three-launch ingress
  (`RMSNorm + column-block GEMV + Q/KV prep`) against the existing
  `gemma4_flash_attention_decode_norm_project_prepare_paged_kv_bf16` fused
  ingress API.

Results:

```text
current ingress:
  p50 = 38.102 ms

fused norm-project-prepare ingress:
  p50 = 72.640 ms

restored current ingress:
  p50 = 38.056 ms
```

Conclusion:

- The existing fused ingress API is correct enough to pass the focused decode
  and FlashAttention tests, but it is much slower in the full decode path.
- The experiment was reverted; the source and rebuilt prompt binary are back on
  the current three-launch ingress path.

## 2026-06-30 - Persistent RMSNorm+projection ingress experiment

Question: can a global-scratch, counter-scheduled decode projection kernel beat
the current split `RMSNorm + projection` ingress by avoiding the normed-hidden
handoff and one kernel launch?

Implementation:

- Added `gemma4_rmsnorm_projection_decode`, an opt-in decode projection API that:
  - computes one hidden RMS scale once per launch,
  - publishes it through global scratch with an epoch flag,
  - lets resident CTAs pull output-column tiles from a global counter,
  - computes BF16-rounded `RMSNorm(x)` on projection load,
  - writes the same raw sliding QKV / global QK row consumed by the existing
    Q/KV prep kernel.
- Full decode opt-in is gated by `GEMMA4_DECODE_FUSED_RMS_PROJECT=1`.
- Default decode remains the split path unless the env var is set.

Validation:

```bash
make test-prefill-gemm
make test-decode-megakernel
make decode-bench
make prompt
```

All passed.

Microbenchmark commands:

```bash
GEMMA4_DECODE_BENCH_SEED=0x20260630 \
  ./build/benches/gemma4_decode_bench sliding_qkv 50 10 3

GEMMA4_DECODE_BENCH_SEED=0x20260630 \
  ./build/benches/gemma4_decode_bench global_qk 50 10 3
```

Microbenchmark results, CUDA events, warm repeated buffers:

```text
sliding_qkv split_rmsnorm_project:
  best = 0.095580 ms
sliding_qkv fused_rmsnorm_project:
  best = 0.098406 ms
  speedup = 0.971x
  max_abs_diff = 0

global_qk split_rmsnorm_project:
  best = 0.101068 ms
global_qk fused_rmsnorm_project:
  best = 0.103444 ms
  speedup = 0.977x
  max_abs_diff = 0
```

Full decode-step commands:

```bash
./build/gemma4_prompt --checkpoint models/gemma-4-12B-it/model.safetensors \
  --tokenizer models/gemma-4-12B-it/tokenizer.json --prompt Hello \
  --max-new 2 --benchmark-mode decode-step --bench-warmup 5 \
  --bench-iters 10 --bench-samples 5

GEMMA4_DECODE_FUSED_RMS_PROJECT=1 ./build/gemma4_prompt \
  --checkpoint models/gemma-4-12B-it/model.safetensors \
  --tokenizer models/gemma-4-12B-it/tokenizer.json --prompt Hello \
  --max-new 2 --benchmark-mode decode-step --bench-warmup 5 \
  --bench-iters 10 --bench-samples 5
```

Full decode-step results:

```text
baseline split ingress:
  p50 = 38.055 ms
  tps_p50 = 26.278

opt-in fused RMS-project ingress:
  p50 = 38.182 ms
  tps_p50 = 26.190
```

Conclusion:

- Correct, but not faster.
- The persistent scheduler saves a launch but loses more inside the projection
  work, likely from global-counter scheduling and recomputing BF16-normalized
  packs inside every projection tile.
- Keep the default decode path on split `RMSNorm + projection + prep`.

## 2026-06-30 - Register-Cached Fused RMS-Projection Ingress

Question: can the fixed fused RMSNorm+projection ingress stop recomputing
`BF16(RMSNorm(x))` inside every output-column tile?

Change:

- Kept the static-strided fused ingress launch shape.
- Loaded each CTA thread's hidden pack once, used it for the RMSNorm sum, then
  kept the BF16-rounded normalized pack in a register across that CTA's tile
  loop.

Commands:

```bash
make decode-bench test-decode-megakernel
GEMMA4_DECODE_BENCH_SEED=0x20260630 \
  ./build/benches/gemma4_decode_bench sliding_qkv 100 20 5
GEMMA4_DECODE_BENCH_SEED=0x20260630 \
  ./build/benches/gemma4_decode_bench global_qk 100 20 5
make prompt
./build/gemma4_prompt --checkpoint models/gemma-4-12B-it/model.safetensors \
  --tokenizer models/gemma-4-12B-it/tokenizer.json --prompt Hello \
  --benchmark-mode decode-step --bench-warmup 3 --bench-iters 5 \
  --bench-samples 3 --max-new 4
```

Results:

```text
sliding_qkv fused vs split:
  best = 0.094280 ms vs 0.095191 ms
  speedup = 1.009667x
  max_abs_diff = 0

global_qk fused vs split:
  best = 0.099748 ms vs 0.100659 ms
  speedup = 1.009137x
  max_abs_diff = 0

full decode-step smoke:
  p50 = 37.547 ms
  tps_p50 = 26.633
```

Conclusion:

- Register-caching the normalized hidden pack recovers the repeated activation
  work in the fused ingress and keeps exact parity with the split path.
- The win is intentionally small because projection remains dominated by
  streaming QKV/QK weights.

## 2026-06-30 - Decode Attention to O Projection Fusion

Question: can the cooperative decode-layer path remove the global
`attention_out -> O projection` handoff by projecting each final attention head
row directly into hidden-width O partials?

Change:

- Replaced the existing megakernel attention phase with an attention+O phase.
- Each attention-owner CTA now projects its final query-head attention row into
  one hidden-width float partial row.
- Tuned that per-head projection to handle 16 hidden columns per CTA reduction
  block, cutting the corrected version's barrier count in half versus the first
  8-column attempt.
- Added a final in-kernel reduction from the 16 query-head O partial rows into
  the BF16 hidden residual input.
- Replaced the first serial per-thread hidden reducer with
  `cub::WarpReduce<float, 16>`, so each hidden element is reduced by one
  16-lane logical warp across the query-head partial rows.
- Kept the public standalone attention APIs unchanged; this edits the current
  cooperative decode layer path rather than adding an opt-in path.

Research:

- NVIDIA CCCL/CUB `cub::WarpReduce` docs describe `LOGICAL_WARP_THREADS` as a
  logical warp size that may be smaller than the hardware warp and say the sum
  result is valid in logical lane 0:
  <https://nvidia.github.io/cccl/unstable/cub/api/classcub_1_1WarpReduce.html>
- NVIDIA's CUDA warp-level primitives guidance recommends warp collectives and
  participating masks for small intra-warp reductions:
  <https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/>

Validation:

```bash
git diff --check
make test-flash-attention-cpp test-decode-megakernel
make prompt
```

The decode-megakernel test now checks nonzero O projection math by comparing the
fused layer's post-attention residual against the standalone
`attention -> O projection -> RMSNorm + residual add` reference.

Benchmark:

```bash
./build/gemma4_prompt --checkpoint models/gemma-4-12B-it/model.safetensors \
  --tokenizer models/gemma-4-12B-it/tokenizer.json --prompt Hello \
  --benchmark-mode decode-step --bench-warmup 3 --bench-iters 5 \
  --bench-samples 3 --max-new 4
```

Results:

```text
previous comparable decode-step smoke:
  p50 = 37.547 ms
  tps_p50 = 26.633

attention+O fused, run 1:
  p50 = 93.472 ms
  tps_p50 = 10.698

attention+O fused, run 2:
  p50 = 92.640 ms
  tps_p50 = 10.794

attention+O fused with CUB logical-warp final reduce, run 1:
  p50 = 92.021 ms
  tps_p50 = 10.867

attention+O fused with CUB logical-warp final reduce, run 2:
  p50 = 92.537 ms
  tps_p50 = 10.807
```

Conclusion:

- Correct, but not faster.
- The first attempt incorrectly sharded hidden O columns by `blockIdx.x`; the
  nonzero-O test caught that because most hidden columns were never produced.
- The corrected version makes each attention-owner CTA loop over the full hidden
  width for its query-head partial. That removes the `attention_out` global
  handoff, but it also collapses O-projection concurrency to the attention CTAs.
- Expanding each projection block from 8 to 16 hidden columns helped, but did
  not change the main bottleneck.
- The CUB logical-warp final reducer is a cleaner reduction shape and exposes
  the 16-head sum across the cooperative grid, but full-path timing stayed
  essentially flat. The bottleneck is still producing the O partials, not
  summing them.
- The next useful rung needs a different O schedule that restores hidden-column
  parallelism without reintroducing the full BF16 `attention_out` handoff.

## 2026-06-30 - Decode O to Post-Attention RMSNorm Fusion

Question: can the existing cooperative decode layer remove the materialized
`O output -> post-attention RMSNorm/residual` reread boundary?

Change:

- Replaced the existing O-partial final reducer with an O finalize plus
  post-attention RMSNorm/residual handoff.
- Each cooperative CTA now owns one contiguous hidden-pack tile, computes BF16 O
  output, accumulates tile sumsq from the rounded BF16 values, then applies the
  post-attention RMSNorm and writes `ffn_residual`.
- Left the remaining pre-FFN RMSNorm and FFN path unchanged.
- Reused existing `attention_out` scratch for the FFN tail's normed output so
  `args.normed_out` remains the post-attention boundary value.
- Expanded the existing partial-accumulator scratch tail for O partials,
  per-CTA sums, and one scale value.

Validation:

```bash
git diff --check
make test-flash-attention-cpp
make test-decode-megakernel
make prompt
```

All passed. The focused decode-megakernel test checks `normed_out`,
`ffn_residual`, and `ffn_x` against the old logical sequence.

Benchmark command:

```bash
./build/gemma4_prompt --checkpoint models/gemma-4-12B-it/model.safetensors \
  --tokenizer models/gemma-4-12B-it/tokenizer.json --prompt Hello \
  --benchmark-mode decode-step --bench-warmup 3 --bench-iters 5 \
  --bench-samples 3 --max-new 4
```

Results:

```text
previous comparable attention+O fused runs:
  p50 = 92.021 ms
  p50 = 92.537 ms

O finalize + post-attention RMS/residual fused, run 1:
  decode_step_ms n=3 mean_ms=83.898 p50_ms=91.958 min_ms=67.436 max_ms=92.300
  decode_step_tps_p50=10.875

O finalize + post-attention RMS/residual fused, run 2:
  decode_step_ms n=3 mean_ms=84.117 p50_ms=92.358 min_ms=67.488 max_ms=92.504
  decode_step_tps_p50=10.827

O finalize + post-attention RMS/residual fused, normed_out preserved, run 1:
  decode_step_ms n=3 mean_ms=84.722 p50_ms=92.900 min_ms=67.979 max_ms=93.286
  decode_step_tps_p50=10.764

O finalize + post-attention RMS/residual fused, normed_out preserved, run 2:
  decode_step_ms n=3 mean_ms=83.970 p50_ms=92.060 min_ms=67.494 max_ms=92.356
  decode_step_tps_p50=10.863
```

Conclusion:

- Correct, but not materially faster in the full path.
- This rung removes the separate post-attention RMSNorm reread of O output, but
  it adds cooperative-grid synchronization around tile sums and scale broadcast.
- The dominant cost is still the head-owned O partial production schedule from
  the previous rung, which serializes too much hidden-column work per attention
  CTA.

## 2026-07-01 - Decode Attention O Projection Parallelism Recovery

Question: can the fused attention+O regression be fixed by restoring the
hidden-column-sharded O projection schedule inside the existing decode
megakernel path?

Change:

- Removed the head-owned O partial projection path from the fused decode layer.
- Attention producers write the final BF16 `attention_out` rows again.
- The cooperative layer grid then projects O by hidden-column BF16 packs, so all
  CTAs participate in the O weight sweep instead of leaving that work on the
  attention-owner CTAs.
- Kept the fused post-attention RMS/residual handoff after O projection.
- Shrunk the prompt/test partial-accumulator tail to per-CTA RMS sums plus one
  scale value; no per-head O partial rows remain.

Validation:

```bash
git diff --check -- src/gemma4_flash_attention.cu src/gemma4_prompt.cu \
  tests/test_decode_megakernel.cu
make test-decode-megakernel
make test-flash-attention-cpp
make prompt
```

All passed.

Environment:

```text
GPU: NVIDIA RTX A6000
Driver: 580.159.03
Nsight Systems: 2025.3.2.474-253236389321v0
Clock policy: not locked
```

Baseline Nsight command on the pre-fix current binary:

```bash
nsys profile --force-overwrite=true --trace=cuda,nvtx,osrt --sample=none \
  --stats=true -o build/nsys/decode_step_fused_o_current \
  ./build/gemma4_prompt --checkpoint models/gemma-4-12B-it/model.safetensors \
    --tokenizer models/gemma-4-12B-it/tokenizer.json --prompt Hello \
    --max-new 4 --benchmark-mode decode-step --bench-warmup 1 \
    --bench-iters 10 --bench-samples 1
```

Baseline result:

```text
decode_step_ms p50 = 93.185
decode_step_tps_p50 = 10.731
sliding fused layer kernel avg = 1.981 ms
global fused layer kernel avg = 1.409 ms
```

Post-fix benchmark command:

```bash
./build/gemma4_prompt --checkpoint models/gemma-4-12B-it/model.safetensors \
  --tokenizer models/gemma-4-12B-it/tokenizer.json --prompt Hello \
  --max-new 4 --benchmark-mode decode-step --bench-warmup 3 \
  --bench-iters 5 --bench-samples 3
```

Post-fix full decode-step results:

```text
run 1:
  decode_step_ms n=3 mean_ms=37.368 p50_ms=37.125 min_ms=36.825 max_ms=38.155
  decode_step_tps_p50=26.936

run 2:
  decode_step_ms n=3 mean_ms=37.345 p50_ms=37.095 min_ms=36.830 max_ms=38.109
  decode_step_tps_p50=26.958
```

Post-fix Nsight command:

```bash
nsys profile --force-overwrite=true --trace=cuda,nvtx,osrt --sample=none \
  --stats=true -o build/nsys/decode_step_o_hidden_sharded \
  ./build/gemma4_prompt --checkpoint models/gemma-4-12B-it/model.safetensors \
    --tokenizer models/gemma-4-12B-it/tokenizer.json --prompt Hello \
    --max-new 4 --benchmark-mode decode-step --bench-warmup 1 \
    --bench-iters 10 --bench-samples 1
```

Post-fix Nsight result:

```text
decode_step_ms p50 = 36.855
decode_step_tps_p50 = 27.133
sliding fused layer kernel avg = 0.698 ms
global fused layer kernel avg = 0.751 ms
```

Conclusion:

- The regression was the head-owned O projection schedule, especially on the
  sliding layers.
- Restoring hidden-column CTA ownership recovers full-path decode from about
  `10.7 tok/s` to about `27 tok/s` on this benchmark.
- This is not the final attention+O fusion design, because it deliberately keeps
  the small BF16 `attention_out` handoff to preserve O-projection concurrency.

## 2026-07-01 - Decode token-level cooperative layer loop

Question: can the existing batch-1 decode path move the 48-layer host launch
loop onto the GPU while preserving the current per-layer math and final sampling
tail?

Change:

- Extracted the cooperative decode layer body into a device helper that receives
  the caller's `cg::grid_group`.
- Kept the old per-layer decode kernel as a wrapper for focused tests.
- Added one token-level cooperative kernel that builds per-layer args on device
  and loops over all 48 transformer layers.
- Kept final RMSNorm, sampling, and tied embedding gather outside this first
  token-layer rung.

Validation:

```bash
git diff --check
make test-decode-megakernel
make test-flash-attention-cpp
make prompt
```

All passed.

Environment:

```text
GPU: NVIDIA RTX A6000
Driver: 580.159.03
Persistence mode: disabled
Clock policy: not locked
SM clock during pre-run snapshot: 1800 MHz
Memory clock during pre-run snapshot: 8001 MHz
Power limit: 300 W
Nsight Systems: 2025.3.2.474-253236389321v0
```

Benchmark command:

```bash
./build/gemma4_prompt --checkpoint models/gemma-4-12B-it/model.safetensors \
  --tokenizer models/gemma-4-12B-it/tokenizer.json --prompt Hello \
  --max-new 4 --benchmark-mode decode-step --bench-warmup 3 \
  --bench-iters 5 --bench-samples 3
```

Benchmark contract:

- Host-visible decode-step latency after one prompt prefill.
- Batch 1, no CUDA graphs, launch overhead included.
- Warm process state after model load and benchmark warmups.
- Cache state uncontrolled/warm from repeated decode-step execution.
- Clocks were not locked.

Result:

```text
decode_step_ms n=3 mean_ms=38.302 p50_ms=37.906 p95_ms=39.214 p99_ms=39.331 min_ms=37.640 max_ms=39.360
decode_step_tps_p50=26.381
```

Nsight command:

```bash
nsys profile --force-overwrite=true --trace=cuda,nvtx,osrt --sample=none \
  --stats=true -o build/nsys/decode_token_kernel \
  ./build/gemma4_prompt --checkpoint models/gemma-4-12B-it/model.safetensors \
    --tokenizer models/gemma-4-12B-it/tokenizer.json --prompt Hello \
    --max-new 4 --benchmark-mode decode-step --bench-warmup 1 \
    --bench-iters 10 --bench-samples 1
```

Nsight result:

```text
decode_step_ms p50 = 37.612
decode_step_tps_p50 = 26.587
decode_megakernel_token_kernel instances = 11
```

The 11 token-kernel instances match 1 warmup decode step plus 10 measured decode
steps. The generated trace string table contains
`decode_megakernel_token_kernel` and no
`decode_megakernel_fused_layer_kernel` symbol.

Conclusion:

- The existing decode path now launches one cooperative token-layer kernel for
  the 48 transformer layers of each decoded token.
- Final RMSNorm/sampling remains a separate tail, as intended for this rung.

## 2026-07-01 - Decode token-kernel slowdown audit and rollback

Question: why did the one-kernel 48-layer decode loop get slower, and should it
remain the default path?

Commands:

```bash
./build/gemma4_prompt --checkpoint models/gemma-4-12B-it/model.safetensors \
  --tokenizer models/gemma-4-12B-it/tokenizer.json --prompt Hello \
  --max-new 4 --benchmark-mode decode-step --bench-warmup 3 \
  --bench-iters 5 --bench-samples 3

ncu --target-processes all --kernel-name-base demangled \
  --kernel-name 'regex:decode_megakernel_token_kernel' --launch-count 1 \
  --metrics gpu__time_duration.sum,launch__grid_size,launch__block_size,launch__registers_per_thread,launch__shared_mem_per_block_static,launch__shared_mem_per_block_dynamic,launch__waves_per_multiprocessor,smsp__warps_active.avg.pct_of_peak_sustained_active,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed \
  --page raw ./build/gemma4_prompt \
  --checkpoint models/gemma-4-12B-it/model.safetensors \
  --tokenizer models/gemma-4-12B-it/tokenizer.json --prompt Hello \
  --max-new 4 --benchmark-mode decode-step --bench-warmup 0 \
  --bench-iters 1 --bench-samples 1

make test-decode-megakernel test-flash-attention-cpp
make prompt
```

Contract:

- Hardware: NVIDIA RTX A6000, driver `580.159.03`.
- Host-visible decode-step latency after one prompt prefill.
- Batch 1, no CUDA graphs, launch overhead included.
- Warm process state after model load and benchmark warmups.
- Cache state uncontrolled/warm from repeated decode-step execution.
- Clocks were not locked; persistence mode disabled.

Evidence:

```text
token-kernel path, fresh short benchmark:
  decode_step_ms p50 = 37.934

token-kernel Nsight Compute launch sample:
  decode_megakernel_token_kernel time = 35.21 ms
  grid = 84 CTAs x 512 threads
  registers_per_thread = 128
  waves_per_sm = 1
  DRAM throughput = 89.93%

existing nsys decode_token_kernel trace:
  decode_megakernel_token_kernel avg = 34.739 ms

existing nsys hidden-sharded per-layer trace:
  sliding fused layer avg = 0.698 ms
  global fused layer avg = 0.751 ms
  40 sliding + 8 global layers ~= 33.93 ms

earlier launch-gap audit:
  total GPU idle launch gaps ~= 0.15 ms/token

restored per-layer path, fresh short benchmark:
  decode_step_ms p50 = 37.112
```

Conclusion:

- The one-kernel path did not lose occupancy; it still launched one CTA per SM.
- The expected launch-gap upside was only about `0.15 ms/token`, because the
  per-layer path already had tiny GPU idle gaps.
- The token kernel made the 48-layer GPU work about `0.8 ms/token` slower. The
  main structural cost is the required inter-layer cooperative `grid.sync()`
  after the FFN tail so the next layer can safely read the ping-ponged hidden
  row. Kernel boundaries gave that ordering in the per-layer path without an
  explicit in-kernel grid barrier.
- Removed the token-level cooperative loop from the default path and restored
  the per-layer host loop over the existing fused layer kernel.

## 2026-07-01 - Decode Attention Readiness Handoff Experiment

Question: can the fused decode layer replace only the Q/K/V-prep to attention
`grid.sync()` with per-KV-head readiness tags?

Implementation:

- Added an opt-in `GEMMA4_DECODE_ATTENTION_READY_HANDOFF=1` path.
- Default builds keep the original full-grid sync before attention.
- Experimental builds use one `uint32_t` readiness tag per model layer and
  possible sliding KV head in decode scratch.
- Producer CTAs finish one KV-head prep, run a block-local sync, fence, then
  publish the tag with release ordering.
- Attention CTAs wait on their KV-head tag with acquire polling and
  `__nanosleep(64)`.
- The projection-to-prep `grid.sync()` and all inter-layer ordering are
  unchanged.

Commands:

```bash
make test-decode-megakernel test-flash-attention-cpp

make BUILD_DIR=build_ready \
  NVCCFLAGS='-std=c++17 -O3 -arch=sm_86 -DGEMMA4_DECODE_ATTENTION_READY_HANDOFF=1' \
  test-decode-megakernel

make BUILD_DIR=build_ready \
  NVCCFLAGS='-std=c++17 -O3 -arch=sm_86 -DGEMMA4_DECODE_ATTENTION_READY_HANDOFF=1' \
  test-flash-attention-cpp

./build/gemma4_prompt --checkpoint models/gemma-4-12B-it/model.safetensors \
  --tokenizer models/gemma-4-12B-it/tokenizer.json --prompt Hello \
  --max-new 4 --benchmark-mode decode-step --bench-warmup 3 \
  --bench-iters 5 --bench-samples 3

make BUILD_DIR=build_ready \
  NVCCFLAGS='-std=c++17 -O3 -arch=sm_86 -DGEMMA4_DECODE_ATTENTION_READY_HANDOFF=1' \
  prompt

./build_ready/gemma4_prompt --checkpoint models/gemma-4-12B-it/model.safetensors \
  --tokenizer models/gemma-4-12B-it/tokenizer.json --prompt Hello \
  --max-new 4 --benchmark-mode decode-step --bench-warmup 3 \
  --bench-iters 5 --bench-samples 3
```

Nsight Compute command for both default and readiness binaries:

```bash
ncu --target-processes all --kernel-name-base demangled \
  --kernel-name 'regex:decode_megakernel_fused_layer_kernel' \
  --launch-count 1 \
  --metrics gpu__time_duration.sum,launch__grid_size,launch__block_size,launch__registers_per_thread,launch__shared_mem_per_block_static,launch__shared_mem_per_block_dynamic,dram__throughput.avg.pct_of_peak_sustained_elapsed,smsp__sass_inst_executed_op_local_ld.sum,smsp__sass_inst_executed_op_local_st.sum \
  --page raw ./build_ready/gemma4_prompt \
  --checkpoint models/gemma-4-12B-it/model.safetensors \
  --tokenizer models/gemma-4-12B-it/tokenizer.json --prompt Hello \
  --max-new 4 --benchmark-mode decode-step --bench-warmup 0 \
  --bench-iters 1 --bench-samples 1
```

Contract:

- Hardware: NVIDIA RTX A6000, driver `580.159.03`.
- CUDA/NVCC: CUDA compilation tools `13.0`, `V13.0.88`.
- Batch 1, prompt `Hello`, decode-step benchmark after one prompt prefill.
- Host-visible decode-step latency includes launch overhead.
- Warmup/repeats: `3` warmup decode steps, `5` timed steps per sample,
  `3` samples.
- Cache state: uncontrolled/warm from repeated decode-step execution.
- Clocks were not locked; persistence mode disabled.
- Correctness tolerance: existing `test_decode_megakernel` and
  `test_flash_attention_cpp` tolerances.

Results:

```text
Correctness:
  default test-decode-megakernel: pass
  default test-flash-attention-cpp: pass
  readiness test-decode-megakernel: pass
  readiness test-flash-attention-cpp: pass

Host-visible decode-step:
  default p50 = 37.077 ms/token
  readiness p50 = 37.132 ms/token
  delta = +0.055 ms/token, readiness slower

Nsight Compute first sliding fused layer:
  default time = 715.52 us
  readiness time = 712.19 us
  default registers/thread = 90
  readiness registers/thread = 90
  default DRAM throughput = 89.54% of peak sustained elapsed
  readiness DRAM throughput = 89.96% of peak sustained elapsed
  default local load/store inst = 62209 / 63552
  readiness local load/store inst = 62209 / 63552
```

Conclusion:

- The experiment did not meet the keep bar of `>= 0.5 ms/token` decode-step
  p50 improvement.
- The per-layer `ncu` sample showed only a tiny first-layer improvement, not a
  meaningful enough sync-wait removal to overcome the host-visible result.
- Keep the path only as macro-gated experimental code for future A/B work; do
  not enable it in the default production build.

## 2026-07-01 - Sliding decode K/V cp.async double-buffer ablation

Implemented an opt-in decode-attention ablation guarded by
`GEMMA4_EXPERIMENT_DECODE_KV_CP_ASYNC_PIPELINE`. The default production build
keeps direct K/V `loadg` loads. With the macro enabled, sliding decode
attention stages one token's K and V vectors into a two-stage shared-memory
buffer with `cp.async`, prefetching token `t + 1` while token `t` is consumed.
Global decode attention remains on the direct path.

Commands:

```bash
make test-flash-attention-cpp test-decode-megakernel

make BUILD_DIR=build_cp_async \
  NVCCFLAGS='-std=c++17 -O3 -arch=sm_86 -DGEMMA4_EXPERIMENT_DECODE_KV_CP_ASYNC_PIPELINE=1' \
  test-flash-attention-cpp test-decode-megakernel

make flash-attn-bench

make BUILD_DIR=build_cp_async \
  NVCCFLAGS='-std=c++17 -O3 -arch=sm_86 -DGEMMA4_EXPERIMENT_DECODE_KV_CP_ASYNC_PIPELINE=1' \
  flash-attn-bench

./build/benches/gemma4_flash_attention_bench 1024 100 20 7 1 0 warm 64
./build_cp_async/benches/gemma4_flash_attention_bench 1024 100 20 7 1 0 warm 64
./build/benches/gemma4_flash_attention_bench 1024 100 20 7 1 0 cold 64
./build_cp_async/benches/gemma4_flash_attention_bench 1024 100 20 7 1 0 cold 64
```

Contract:

- Hardware: NVIDIA RTX A6000, driver `580.159.03`.
- CUDA/NVCC: CUDA compilation tools `13.0`, `V13.0.88`.
- Shape: batch 1, sliding decode attention, sequence/window `1024`, split
  size `20`, `52` overprovisioned split CTAs per query-head row.
- Timing: CUDA events on the benchmark stream. Launch overhead included.
- Warm cache: no L2 flush inside timing loop.
- Cold cache: 64 MiB L2 flush before each timed iteration.
- Clocks were not locked; persistence mode disabled.

Results for the `decode_paged_attention` row:

```text
Default direct, warm:
  median = 0.048026 ms
  min    = 0.046326 ms
  max    = 0.048108 ms

cp.async pipeline, warm:
  median = 0.042250 ms
  min    = 0.041236 ms
  max    = 0.043028 ms

Default direct, cold:
  median = 0.048392 ms
  min    = 0.048311 ms
  max    = 0.048477 ms

cp.async pipeline, cold:
  median = 0.042372 ms
  min    = 0.042327 ms
  max    = 0.042414 ms
```

Generated-code checks:

```text
cuobjdump resource usage, sliding decode split kernel:
  direct:   REG 66, SHARED 96,   STACK 0, LOCAL 0
  cp.async: REG 72, SHARED 2144, STACK 0, LOCAL 0

ptxas -v, cp.async sliding decode split kernel:
  0 bytes stack frame
  0 bytes spill stores
  0 bytes spill loads
  Used 72 registers

SASS:
  LDGSTS.E.BYPASS.LTC128B.128 emitted in the macro build
```

Conclusion:

- The two-buffer `cp.async` ablation is faster in the attention-only decode
  benchmark by about `0.0058-0.0060 ms` per decode-attention call.
- Keep the implementation as experimental macro-gated code for A/B work.
- Do not enable it by default yet; the production direct-load path remains the
  default, and this run did not measure full decode-step impact.

## 2026-07-01 - Decode partial-O accumulation ablation

Tried a default-off `GEMMA4_EXPERIMENT_PARTIAL_O_ACCUMULATION` path that kept
attention materialization but replaced the hidden-column O projector with FP32
per-hidden-column atomic partial sums over `(q_head, hidden tile)` work. The
ablation reused post-attention tail scratch and then finalized the same
post-attention RMSNorm/residual outputs.

Commands:

```bash
make test-flash-attention-cpp test-decode-megakernel
make BUILD_DIR=build_partial_o CPPFLAGS='-Isrc -DGEMMA4_EXPERIMENT_PARTIAL_O_ACCUMULATION=1' \
  test-flash-attention-cpp test-decode-megakernel
make prompt
make BUILD_DIR=build_partial_o CPPFLAGS='-Isrc -DGEMMA4_EXPERIMENT_PARTIAL_O_ACCUMULATION=1' prompt

./build/gemma4_prompt --checkpoint models/gemma-4-12B-it/model.safetensors \
  --tokenizer models/gemma-4-12B-it/tokenizer.json --prompt Hello \
  --max-new 4 --benchmark-mode decode-step --bench-warmup 3 \
  --bench-iters 5 --bench-samples 3

./build_partial_o/gemma4_prompt --checkpoint models/gemma-4-12B-it/model.safetensors \
  --tokenizer models/gemma-4-12B-it/tokenizer.json --prompt Hello \
  --max-new 4 --benchmark-mode decode-step --bench-warmup 3 \
  --bench-iters 5 --bench-samples 3
```

Contract:

- Hardware: NVIDIA RTX A6000, driver `580.159.03`.
- CUDA/NVCC: CUDA compilation tools `13.0`, `V13.0.88`.
- Prompt: `Hello`, prompt length `1`, decode split `20`.
- Timing: host-visible decode-step benchmark, warmup `3`, iters `5`,
  samples `3`.
- Clocks were not locked.

Results:

```text
Baseline:
  decode_step_ms mean = 37.345
  decode_step_ms p50  = 37.115
  decode_step_ms min  = 36.772
  decode_step_ms max  = 38.147

Partial-O experiment:
  decode_step_ms mean = 42.231
  decode_step_ms p50  = 42.023
  decode_step_ms min  = 41.703
  decode_step_ms max  = 42.968
```

Narrow `ncu` check on the first
`decode_megakernel_fused_layer_kernel<Gemma4AttentionTraits<0>>` launch:

```text
Baseline:
  kernel time         = 0.712480 ms
  registers/thread    = 90
  shared/block static = 11024 B
  shared/block total  = 12048 B
  local ld/st bytes   = 8.481 MB / 8.653 MB
  global atom bytes   = 0.0156 MB avg
  DRAM bytes          = 38.916 MB avg

Partial-O experiment:
  kernel time         = 0.870080 ms
  registers/thread    = 96
  shared/block static = 11024 B
  shared/block total  = 12048 B
  local ld/st bytes   = 8.481 MB / 8.653 MB
  global atom bytes   = 0.0391 MB avg
  DRAM bytes          = 38.916 MB avg
```

Conclusion:

- The ablation regressed full decode-step p50 by `4.908 ms/token`.
- Kernel-level profiling also showed higher register pressure and more atomic
  traffic.
- The experimental code was removed completely. Keep the current
  hidden-column-owned O projection path.

## 2026-07-01 - Promote sliding decode K/V cp.async pipeline

Promoted the positive sliding decode K/V `cp.async` result into the default
decode attention path. The experiment macro was removed from CUDA source, and
the sliding head_dim=256 path now always uses the two-buffer K/V shared-memory
pipeline. Global head_dim=512 decode remains on direct loads because it was not
part of the measured win.

Commands:

```bash
make test-flash-attention-cpp test-decode-megakernel
make flash-attn-bench
./build/benches/gemma4_flash_attention_bench 1024 100 20 7 1 0 warm 64
./build/benches/gemma4_flash_attention_bench 1024 100 20 7 1 0 cold 64
make prompt
./build/gemma4_prompt --checkpoint models/gemma-4-12B-it/model.safetensors \
  --tokenizer models/gemma-4-12B-it/tokenizer.json --prompt Hello \
  --max-new 4 --benchmark-mode decode-step --bench-warmup 3 \
  --bench-iters 5 --bench-samples 3
```

Results:

```text
Default sliding decode_paged_attention after promotion:
  warm median = 0.042476 ms
  warm min    = 0.041196 ms
  warm max    = 0.043254 ms
  cold median = 0.042260 ms
  cold min    = 0.042146 ms
  cold max    = 0.042330 ms

Short full decode-step sanity:
  decode_step_ms mean = 37.228
  decode_step_ms p50  = 36.993
  decode_step_ms min  = 36.719
  decode_step_ms max  = 37.971
```

Conclusion:

- The promoted default stays in the faster cp.async attention-only band from
  the earlier ablation.
- The same short full decode-step command improved slightly versus the earlier
  direct-load baseline in this chat (`37.115 ms/token` p50 to `36.993`).
- The readiness handoff experiment stayed macro-gated because it was slower in
  full decode-step timing. The partial-O experiment stayed removed.
