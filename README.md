# gemma.c

`gemma.c` is an experimental inference runtime for Gemma dense models, with an initial focus on running the Gemma 4 31B dense model efficiently on NVIDIA RTX A6000 GPUs.

The long-term goal is a highly optimized mega-kernel inference path: start from straightforward, correct kernels, build a fast unfused implementation, then progressively fuse the hot path into a minimal set of specialized kernels.

## Goals

- Target Gemma 4 dense inference, especially the 31B model size.
- Optimize for RTX A6000-class GPUs.
- Use CUDA 12.x, tracking the latest available CUDA 12 release.
- Keep the implementation close to the metal, starting from raw math kernels.
- Prioritize correctness first, then profiling, then fusion.
- Build toward a single highly specialized inference pipeline instead of a general-purpose framework.

## Development Plan

1. Implement a simple unfused inference path.
2. Add baseline CUDA kernels for the core model operations.
3. Validate numerics against a known-good reference.
4. Profile the full decode path on A6000 hardware.
5. Optimize individual kernels.
6. Fuse the highest-impact operations.
7. Iterate toward a mega-kernel design for the steady-state inference loop.

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

## Scope

This repository is intentionally narrow. It is not trying to become a broad model-serving framework. The intended path is to specialize hard for one model family, one class of GPU, and one inference workload.

## Status

Early scaffolding. The first milestone is a correct unfused Gemma inference implementation.
