# TODO

*Agent maintained, quite messy. They never listen to me when I tell them to keep it clean...*

## High-Priority Kernel Work

- Write a hyperoptimized CUDA kernel path for the Gemma 4 31B FFN matmuls, the dominant total-FLOP workload:
  - Gate/up projection: `[M, 5376] x [5376, 21504]`
  - Down projection: `[M, 21504] x [21504, 5376]`
  - Consider a packed gate/up variant: `[M, 5376] x [5376, 43008]`
  - Optimize first for correctness, then profile on RTX A6000-class hardware with `ncu`.
- Fuse token embedding gather with the first matrix multiplication in the language path once the unfused baseline is correct.
  - Avoid materializing the initial `[M, 5376]` hidden buffer when the first projection can consume embedding rows directly.
  - Benchmark against the standalone embedding gather plus cuBLAS/cuBLASLt baseline before keeping the fusion.
- Split the RMSNorm kernels into separate decode and prefill paths.
  - Decode path should optimize for one row/token and low launch/device overhead.
  - Prefill path should optimize for many rows and sustained memory bandwidth.
  - Keep the existing shared implementation as the correctness baseline until both specialized paths pass tests and benchmarks.

## Decode GEMV Priority Order

## Decode GEMV Kernel-Shape Experiments

- Benchmark warp-tiled GEMV variants against the current CTA-cooperative baseline.
  - Current baseline: one CTA computes four output columns, with all CTA threads reducing the same four dots.
  - Variant A: one warp owns one output column.
  - Variant B: one warp owns two output columns.
  - Variant C: two warps own one output column for large-`K` shapes.
  - Keep `Packed128` loads and streaming weight-load hints in each variant.
  - Compare first on `global_k`, `sliding_o`, `sliding_qkv`, and `global_q`, where reduction overhead may matter most.
  - Preserve the existing CTA-reduction kernel as the baseline and only replace it after correctness and CUDA-event timings clearly improve.

1. FFN gate+up GEMV
   - Shape: `x[5376] -> gate_up[43008]`
   - Estimated weight traffic: `~27.7 GB/token` across 60 layers.
   - Baseline custom decode GEMV exists.
   - Biggest fixed-shape decode target.

2. FFN down GEMV
   - Shape: `ffn_hidden[21504] -> hidden[5376]`
   - Estimated weight traffic: `~13.9 GB/token`.
   - Baseline custom decode GEMV exists; needs tuning.
   - Same every layer, huge, clean.

3. Sliding QKV packed GEMV
   - Shape: `x[5376] -> qkv[16384]`
   - Estimated weight traffic: `~8.8 GB/token` across 50 sliding layers.
   - Do not optimize sliding Q/K/V separately if we can avoid it.
   - Pack as `Q 8192 + K 4096 + V 4096 = 16384`.
   - Baseline custom decode GEMV exists; needs tuning.

4. Sliding O GEMV
   - Shape: `attn_out[8192] -> hidden[5376]`
   - Estimated weight traffic: `~4.4 GB/token`.
   - Repeated across 50 sliding layers.
   - Baseline custom decode GEMV exists; needs tuning.

5. Final vocab/logits GEMV
   - Shape: `hidden[5376] -> logits[262144]`
   - Estimated weight traffic: `~2.8 GB/token`.
   - Huge single kernel.
   - High value if fused with softcap/top-k/argmax, but only runs once per token.
   - Baseline custom decode GEMV exists; needs fusion with softcap/sampling.

6. Global Q/O GEMVs
   - Global Q shape: `x[5376] -> q[16384]`
   - Global O shape: `attn_out[16384] -> hidden[5376]`
   - Estimated weight traffic: `~1.76 GB/token` each.
   - Baseline custom decode GEMVs exist for global Q, K, and O; Q/O need tuning.
