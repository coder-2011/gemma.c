##  Experiments Log

AI-updated and directed log of the experiments I ran throughout this project to optimize the kernels. 
Expect this to be very messy and pretty much useless for most people to look at.  it is meant to be a place for me and my agents to fuck around

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
- This pass keeps normal cached loads for the reused input vector and marks the weight stream with the existing `Packed128` + `load128cs` helper.

Implementation:

- Changed the fixed-four decode dot helper from half2-style 32-bit loads to `Packed128<__nv_bfloat16>` loads.
- Each load now covers 8 BF16 values per thread.
- Weight row loads use `load128cs`, which SASS emits as `LDG.E.EF.128` on this build.
- Input-vector loads use `load128`, which SASS emits as `LDG.E.128.CONSTANT` on this build.
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
- The useful retained cache hint is streaming global load for weights through `load128cs`, because weights are streamed through the decode GEMV.
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
- Added an eight-BF16 output store helper so an eight-column block writes its result with one 128-bit store.
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

- Baseline: current `load128cs` streaming weight loads.
- `prefetch_next_iter`: inline `prefetch.global.L2` for the next K-stride `x` pack and all eight next K-stride weight packs.
- `prefetch_next_col`: inline `prefetch.global.L2` for the next output column's weight pack before loading/computing the current column.
- `normal_weight_load`: replace streaming `load128cs` weight loads with normal `load128` weight loads.

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
- Do not switch the weight loads from streaming `load128cs` to normal `load128`.
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

- Added inlined decode load helpers:
  - `gemma4_load_activation_pack(...)` uses `load128(...)`.
  - `gemma4_load_streaming_weight_pack<K>(...)` uses `load128cs(...)`.
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
- This reinforces the current decision: use `load128` for reused activation packs, `load128cs` for streaming weight packs, and avoid shared-memory async copy until the GEMV mapping has real staged-data reuse.

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
- Keep cp.async as a compile-time experiment path only. The runtime default should remain the direct `load128`/`load128cs` path.

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

## 2026-05-19 - Decode GEMV cp.async debug with load128/load128cs staging

Runtime file tested: `src/gemma4_matmul_kernels.cu`

Question:

- The cp.async double buffer should plausibly be faster, so isolate whether the slowdown comes from shared-memory staging itself or from cp.async commit/wait mechanics.

Implementation:

- Kept the same two-stage shared-memory layout used by the cp.async path:
  `Gemma4Bf16Pack weight_stages[2][Threads]`.
- Added two compile-time sibling variants:
  - `GEMMA4_DECODE_SHARED_STAGE_LOAD128`
  - `GEMMA4_DECODE_SHARED_STAGE_LOAD128CS`
- Both variants use the same shared-stage consume path as cp.async, but stage data with normal global loads instead of `__pipeline_memcpy_async`.
- The direct default path remains unchanged.

Commands:

```bash
make -B decode-bench NVCCFLAGS="-std=c++17 -O3 -arch=sm_86"
GEMMA4_DECODE_BENCH_SEED=0x1234 ./build/experiments/gemma4_decode_bench all 50 10 3

make -B decode-bench NVCCFLAGS="-std=c++17 -O3 -arch=sm_86 -DGEMMA4_DECODE_CP_ASYNC_DOUBLE_BUFFER"
GEMMA4_DECODE_BENCH_SEED=0x1234 ./build/experiments/gemma4_decode_bench all 50 10 3

make -B decode-bench NVCCFLAGS="-std=c++17 -O3 -arch=sm_86 -DGEMMA4_DECODE_SHARED_STAGE_LOAD128"
GEMMA4_DECODE_BENCH_SEED=0x1234 ./build/experiments/gemma4_decode_bench all 50 10 3

make -B decode-bench NVCCFLAGS="-std=c++17 -O3 -arch=sm_86 -DGEMMA4_DECODE_SHARED_STAGE_LOAD128CS"
GEMMA4_DECODE_BENCH_SEED=0x1234 ./build/experiments/gemma4_decode_bench all 50 10 3
```

Timing, seeded `all 50 10 3`:

| Op | Direct | cp.async | Shared `load128` | Shared `load128cs` |
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
| Shared `load128` | 87.105 |
| Shared `load128cs` | 86.983 |

PTX/resource check:

| Variant | Registers | Shared memory | Spills | PTX evidence |
| --- | --- | --- | --- | --- |
| Direct | `63`, `72` | `512B`, `1024B` | `0/0` | `ld.global.cs=56`, `ld.global.nc=7` |
| cp.async | `54`, `55` | `16896B`, `33792B` | `0/0` | `cp.async.cg.shared.global=56`, `commit_group=56`, `wait_group=56` |
| Shared `load128` | `59`, `60` | `16896B`, `33792B` | `0/0` | `ld.global.nc=63`, no cp.async |
| Shared `load128cs` | `64` | `16896B`, `33792B` | `0/0` | `ld.global.cs=56`, `ld.global.nc=7`, no cp.async |

Interpretation:

- Shared-memory staging itself is not the issue. Shared `load128cs` is essentially tied with direct.
- The weight load policy matters: shared `load128cs` beats shared `load128`, matching the direct path's streaming-weight policy.
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
| Same plus streaming stores | 0.089786 | 0.089957 | -0.48% |
| Final simplified kept code | 0.089864 | 0.090050 | -0.39% |

The streaming-store candidate moved the aggregate by only about `0.023%` versus the
previous candidate, below the requested `0.05%` stopping threshold, so tuning stopped
there. The final kept code uses the meaningful settings only:

- `__launch_bounds__(Threads, 2)` for the hidden prefill kernel
- cached `load128g` input loads for `inp1` and `inp2`
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
