# TODO

## High-Priority Kernel Work

- Write a hyperoptimized CUDA kernel path for the Gemma 4 31B FFN matmuls, the dominant total-FLOP workload:
  - Gate/up projection: `[M, 5376] x [5376, 21504]`
  - Down projection: `[M, 21504] x [21504, 5376]`
  - Consider a packed gate/up variant: `[M, 5376] x [5376, 43008]`
  - Optimize first for correctness, then profile on RTX A6000-class hardware with `ncu`.
