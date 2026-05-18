# TODO

## High-Priority Kernel Work

- Write a hyperoptimized CUDA kernel path for the Gemma 4 31B FFN matmuls, the dominant total-FLOP workload:
  - Gate/up projection: `[M, 5376] x [5376, 21504]`
  - Down projection: `[M, 21504] x [21504, 5376]`
  - Consider a packed gate/up variant: `[M, 5376] x [5376, 43008]`
  - Optimize first for correctness, then profile on RTX A6000-class hardware with `ncu`.
- Fuse token embedding gather with the first matrix multiplication in the language path once the unfused baseline is correct.
  - Avoid materializing the initial `[M, 5376]` hidden buffer when the first projection can consume embedding rows directly.
  - Benchmark against the standalone embedding gather plus cuBLAS/cuBLASLt baseline before keeping the fusion.
