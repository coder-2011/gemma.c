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
