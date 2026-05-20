# TODO

*Agent maintained, quite messy. They never listen to me when I tell them to keep it clean...*

## High-Priority Kernel Work

- Fix the current RMSNorm build break before building new dependent kernels.
  - `src/gemma4_rmsnorm.cu` still references fallback kernels removed during cleanup.
  - `make test-rmsnorm` should compile and pass again before RMSNorm-dependent work proceeds.
- Fuse token embedding gather with the first matrix multiplication in the language path once the unfused baseline is correct.
  - Avoid materializing the initial `[M, 5376]` hidden buffer when the first projection can consume embedding rows directly.
  - Benchmark against the standalone embedding gather plus cuBLAS/cuBLASLt baseline before keeping the fusion.

## Remaining Unfused Inference Buildout

Build the remaining model-path pieces in this rough order. Keep each kernel
standalone, numerically tested, benchmarked, and logged in
`src/experiments/EXPERIMENTS.md` before wiring it into the full path.

1. Q/K per-head RMSNorm
   - Sliding layers: learned-weight RMSNorm over head dim `256`.
   - Global layers: learned-weight RMSNorm over head dim `512`.
   - Weights are shared across heads.

2. KV-cache write/update
   - Sliding cache: K/V width `4096`, window `1024`, wraparound.
   - Global cache: K/V width `2048`, full context up to `256000`.
   - Support prefill bulk writes, decode appends, and sliding-window wraparound.

3. Sliding-window attention
   - FlashAttention-style local causal attention.
   - Head dim `256`.
   - GQA ratio: `32` Q heads / `16` KV heads = `2` Q heads per KV head.
   - Support prefill local causal attention and single-token decode.

4. Global attention
   - FlashAttention-style full causal attention.
   - Head dim `512`.
   - GQA ratio: `32` Q heads / `4` KV heads = `8` Q heads per KV head.
   - Support prefill full causal attention and decode over the full KV cache.

5. Attention output packing
   - Sliding layers produce projection input width `8192`.
   - Global layers produce projection input width `16384`.
   - Prefer having attention kernels write directly in projection-ready layout once correct.

6. GeGLU tanh activation
   - Apply `gate * GELU_tanh(up)`.
   - Width `21504`.
   - Baseline standalone first; later fuse into FFN output handling.

7. Final logit softcap
   - Apply `tanh(logits / 30.0) * 30.0`.
   - Width `262144`.
   - Later fuse with sampling when useful.

8. Sampling
   - Start with greedy argmax over vocab `262144`.
   - Add temperature, top-k, top-p, and random sampling after the deterministic path works.

9. Full prefill/decode orchestration
   - Wire the correct unfused layer pipeline using baseline kernels plus cuBLAS/cuBLASLt GEMMs.
   - Validate end-to-end against a known-good reference before fusion work.
   - Benchmark against TensorRT-LLM and vLLM once the path is correct.

## Decode GEMV Tuning

- Benchmark warp-tiled GEMV variants against the current CTA-cooperative baseline.
  - Current baseline: one CTA computes eight output columns, with all CTA threads reducing the same eight dots.
  - Variant A: one warp owns one output column.
  - Variant B: one warp owns two output columns.
  - Variant C: two warps own one output column for large-`K` shapes.
  - Keep `Packed128` loads and streaming weight-load hints in each variant.
  - Compare first on `global_k`, `sliding_o`, `sliding_qkv`, and `global_q`, where reduction overhead may matter most.
  - Preserve the existing CTA-reduction kernel as the baseline and only replace it after correctness and CUDA-event timings clearly improve.

1. FFN gate+up GEMV
   - Shape: `x[5376] -> gate_up[43008]`
   - Estimated weight traffic: `~27.7 GB/token` across 60 layers.
   - Biggest fixed-shape decode tuning target.

2. FFN down GEMV
   - Shape: `ffn_hidden[21504] -> hidden[5376]`
   - Estimated weight traffic: `~13.9 GB/token`.
   - Same every layer, huge, clean.

3. Sliding QKV packed GEMV
   - Shape: `x[5376] -> qkv[16384]`
   - Estimated weight traffic: `~8.8 GB/token` across 50 sliding layers.
   - Do not optimize sliding Q/K/V separately if we can avoid it.
   - Pack as `Q 8192 + K 4096 + V 4096 = 16384`.

4. Sliding O GEMV
   - Shape: `attn_out[8192] -> hidden[5376]`
   - Estimated weight traffic: `~4.4 GB/token`.
   - Repeated across 50 sliding layers.

5. Final vocab/logits GEMV
   - Shape: `hidden[5376] -> logits[262144]`
   - Estimated weight traffic: `~2.8 GB/token`.
   - Huge single kernel.
   - High value if fused with softcap/top-k/argmax, but only runs once per token.

6. Global Q/O GEMVs
   - Global Q shape: `x[5376] -> q[16384]`
   - Global O shape: `attn_out[16384] -> hidden[5376]`
   - Estimated weight traffic: `~1.76 GB/token` each.
