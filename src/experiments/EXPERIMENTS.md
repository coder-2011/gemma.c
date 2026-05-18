##  Experiments Log

AI-updated and directed log of the experiments I ran throughout this project to optimize the kernels. 
Expect this to be very messy and pretty much useless for most people to look at.  it is meant to be a place for me and my agents to fuck around

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

## 2026-05-18 - Matmul prefill wrapper benchmark coverage

Runtime file: `src/gemma4_matmul_kernels.cu`

Benchmark file: `src/experiments/gemma4_decode_bench.cu`

Change:

- Validated the existing cuBLAS BF16 prefill wrappers for the same fixed dense projection shapes that already have decode entry points:
  - `ffn_gate_up`: `K=5376, N=43008`
  - `ffn_down`: `K=21504, N=5376`
  - `sliding_qkv`: `K=5376, N=16384`
  - `sliding_o`: `K=8192, N=5376`
  - `global_q`: `K=5376, N=16384`
  - `global_k`: `K=5376, N=2048`
  - `global_o`: `K=16384, N=5376`
  - `final_logits`: `K=5376, N=262144`
- All wrappers share one helper that computes row-major `Y[M, N] = X[M, K] * W[K, N]` through cuBLAS as column-major `Y^T[N, M] = W^T[N, K] * X^T[K, M]`.
- Extended `gemma4_decode_bench` so each op also calls its prefill wrapper with `M=1`. This keeps the benchmark decode-focused while still validating that the prefill wrapper table links and produces the same BF16-scale output as the existing cuBLAS references.

Build:

```bash
make decode-bench cuda-kernels
```

Smoke validation:

```bash
GEMMA4_DECODE_BENCH_SEED=0x1234 ./build/experiments/gemma4_decode_bench all 1 0 1
```

Result:

- Completed all eight projection shapes.
- Every `cublas_bf16_prefill_m1_*_diff` line matched the same BF16-scale behavior as the existing GEMV/GEMM references.
- This was a one-iteration smoke pass, not a tuning run. Do not use the timings for performance conclusions.

CUDA guide note:

- `$cuda-programming-guide` was queried for the relevant CUDA-X/cuBLAS and matrix-layout context. The guide recommends using CUDA-X libraries such as cuBLAS for Tensor Core acceleration on supported hardware, and its memory-layout examples define column-major leading dimension as the number of rows.
