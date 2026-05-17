# Project Instructions

## Architecture Reference

`gemma4_architecture.md` is the architecture document for this project. Refer to it frequently when making design, implementation, optimization, or correctness decisions.

## Goals

- Target Gemma 4 dense inference, especially the 31B model size.
- Optimize for RTX A6000-class GPUs.
- Use CUDA 12.x, tracking the latest available CUDA 12 release.
- Keep the implementation close to the metal, starting from raw math kernels.
- Prioritize correctness first, then profiling, then fusion.
- Build toward a single highly specialized inference pipeline instead of a general-purpose framework.

## Development Plan

1. Write the required CUDA kernels one by one as standalone, testable baseline kernels.
2. Validate each kernel numerically against a known-good PyTorch or CPU reference before depending on it in the full path.
3. Build the correct unfused inference path from those baseline kernels plus cuBLAS/cuBLASLt GEMMs.
4. Add a launcher/orchestration layer for prefill and decode.
5. Benchmark the unfused path before doing fusion work.
6. Fuse only after the unfused baseline is correct and measured.
7. Benchmark continuously after each kernel change, fusion, or launcher change.
8. Iterate toward a specialized fused steady-state decode path and, eventually, a mega-kernel design.

## Benchmarking

This project will be profiling-heavy. Kernel changes should be measured directly, and inference-path changes should be compared against established runtimes where possible.

Nsight Compute (`ncu`) is the main profiling tool for this project. It should be used constantly for per-kernel analysis, including memory throughput, compute utilization, warp stalls, and occupancy.

Nsight Systems (`nsys`) is used for system-level profiling. It is useful for timeline analysis, kernel launch overhead, CPU/GPU overlap, and idle gaps. The project's `benchmark_on_modal.py` script runs `nsys profile` explicitly.

Primary benchmark targets:

- TensorRT-LLM
- vLLM

Additional comparison targets:

- SGLang
- llama.cpp
- PyTorch

Google Benchmark will be used for focused C++ and CUDA microbenchmarks. PyTorch can be used for basic correctness checks, quick baseline measurements, and reference experiments.

## Model Sizes

Approximate model memory footprints:

| Model | BF16 (16-bit) | SFP8 (8-bit) | Q4_0 (4-bit) |
| --- | ---: | ---: | ---: |
| Gemma 4 E2B | 9.6 GB | 4.6 GB | 3.2 GB |
| Gemma 4 E4B | 15 GB | 7.5 GB | 5 GB |
| Gemma 4 31B | 58.3 GB | 30.4 GB | 17.4 GB |
| Gemma 4 26B A4B | 48 GB | 25 GB | 15.6 GB |

## Tooling

- CUDA 12.x, tracking the latest CUDA 12 release.
- Core CUDA libraries and tools, including cuBLAS, cuDNN, CUDA Runtime, NVCC, Nsight Systems, and Nsight Compute.
- Python 3.11 for scripts and reference tooling.
- `uv` for Python environment management.
- Latest stable PyTorch installed through `uv`.

## Source Layout

- Keep model metadata, config helpers, checkpoint loading, CPU-side orchestration, and CUDA sources under `src/`.
- Use `.cu` files for GPU inference and kernel-launch code; do not put real CUDA inference work in plain `.c` files.
- Prefer stable filenames such as `gemma4.c`, `gemma4_infer.cu`, and `gemma4_kernels.cu`; specialize behavior through config/constants rather than checkpoint-specific filenames like `gemma-4-31B.cu`.

## Kernel Inventory

The first implementation target is the Gemma 4 31B dense text inference path. Vision kernels can come later; do not block the first correct unfused text path on multimodal support.

Write baseline kernels in this order, keeping each kernel separately testable before wiring it into the full launcher:

1. Token embedding gather
   - Token IDs to hidden rows.
   - Shape: token id -> `[5376]`.
   - Used for prompt/prefill input construction.

2. RMSNorm width 5376
   - Learned-weight RMSNorm for language hidden states.
   - Used by `input_layernorm`, `post_attention_layernorm`, `pre_feedforward_layernorm`, `post_feedforward_layernorm`, and final language-model norm.
   - Baseline first, then fused variants with residual add.

3. Residual add width 5376
   - Baseline standalone residual add.
   - Later fuse with the following RMSNorm wherever possible.

4. Residual add plus RMSNorm width 5376
   - Memory-bound fused kernel.
   - Needed after attention projection and after FFN down projection.
   - Account for checkpoint `layer_scalar` once its exact semantics are verified.

5. Q/K RMSNorm
   - Per-head learned-weight RMSNorm after Q and K projections.
   - Sliding layers: head dim `256`.
   - Global layers: head dim `512`.
   - Weights are `[256]` or `[512]`, shared across heads.

6. V RMSNorm without learned weight
   - Scale-free RMSNorm for values.
   - Sliding layers normalize projected V.
   - Global layers derive V from the K source because global attention has no V projection.

7. RoPE and p-RoPE apply
   - Sliding layers: head dim `256`, rotary dims `256`, theta `10000`.
   - Global layers: head dim `512`, rotary dims `128`, theta `1000000`.
   - Baseline standalone first; later fuse with Q/K norm and KV-cache write.

8. KV-cache write/update
   - Sliding layers: K/V width `4096`, window `1024`.
   - Global layers: K/V width `2048`, full context up to `256000`.
   - Support prefill bulk writes, decode appends, and sliding-window wraparound.

9. Sliding-window attention
   - FlashAttention-style fused attention for local layers.
   - Head dim `256`.
   - GQA ratio: `32` Q heads / `16` KV heads = `2` Q heads per KV head.
   - Support prefill causal local attention and single-token decode attention.

10. Global attention
   - FlashAttention-style fused attention for full layers.
   - Head dim `512`.
   - GQA ratio: `32` Q heads / `4` KV heads = `8` Q heads per KV head.
   - Support prefill full causal attention and single-token decode over the full KV cache.

11. Attention output pack
   - Pack attention output for the output projection input layout.
   - Sliding layers produce `[M, 8192]`.
   - Global layers produce `[M, 16384]`.
   - Prefer having attention kernels write directly in projection-ready layout once the baseline is correct.

12. GeGLU tanh activation
   - Apply `gate * GELU_tanh(up)`.
   - Width `21504`.
   - Baseline standalone first; later fuse into FFN output handling.

13. Final logit softcap
   - Apply `tanh(logits / 30.0) * 30.0`.
   - Width `262144`.
   - Later fuse with sampling when useful.

14. Sampling
   - Start with greedy argmax over vocab `262144`.
   - Add temperature, top-k, top-p, and random sampling after the deterministic path works.

Dense GEMMs are required operations, but the baseline path should use cuBLAS/cuBLASLt before writing custom matmul kernels:

- Sliding Q projection: `[M, 5376] x [5376, 8192]`
- Sliding K/V projection: `[M, 5376] x [5376, 4096]`
- Sliding O projection: `[M, 8192] x [8192, 5376]`
- Global Q projection: `[M, 5376] x [5376, 16384]`
- Global K projection: `[M, 5376] x [5376, 2048]`
- Global O projection: `[M, 16384] x [16384, 5376]`
- FFN gate/up projections: `[M, 5376] x [5376, 21504]`, or packed `[M, 5376] x [5376, 43008]`
- FFN down projection: `[M, 21504] x [21504, 5376]`
- Final vocab projection: `[M, 5376] x [5376, 262144]`

Only replace cuBLAS/cuBLASLt GEMMs with custom kernels after correctness exists and profiling shows a reason. The first custom GEMM candidates are the dominant FFN shapes:

- `[M, 5376] x [5376, 21504]`
- `[M, 21504] x [21504, 5376]`
- packed gate/up: `[M, 5376] x [5376, 43008]`

## Layer Pipeline Optimization

Use this rough compute/memory classification when deciding what to fuse and what to leave to library GEMMs:

```text
input
  -> residual_add + layernorm      [memory-bound, fuse candidate]
  -> QKV matmul                    [compute-bound, leave to cuBLAS]
  -> attention (QK^T, softmax, AV) [fuse candidate, this is FlashAttention]
  -> projection matmul             [compute-bound, cuBLAS]
  -> residual_add + layernorm      [memory-bound, fuse candidate]
  -> FFN matmul                    [compute-bound, cuBLAS]
  -> GELU                          [memory-bound, fuse into FFN output]
  -> FFN matmul                    [compute-bound, cuBLAS]
  -> residual_add                  [fuse with next layernorm]
```

## Scope

This repository is intentionally narrow. It is not trying to become a broad model-serving framework. The intended path is to specialize hard for one model family, one class of GPU, and one inference workload.

## Status

Early scaffolding. The first milestone is a correct unfused Gemma inference implementation.
