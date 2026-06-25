# TODO

*Agent maintained, quite messy. They never listen to me when I tell them to keep it clean...*

## High-Priority Kernel Work

- Benchmark the RMSNorm kernels again after the current cleanup/recovery work.
  - Include hidden width `3840` plus Q/K widths `256` and `512`.
  - Confirm the hidden-width-only fused residual+RMSNorm path is skipped for non-hidden widths.
  - Log commands, GPU/clocks, warmup/iteration counts, cache policy, and results in `src/experiments/EXPERIMENTS.md`.
- Refactor megakernel timing instrumentation.
  - Remove standalone `__global__` timing wrappers around device functions once the megakernel path owns measurement.
  - Add flag-gated timing spots in the megakernel hot path so timings cover real in-path work without wrapper launch overhead.
- Remove the redundant C ABI wrappers across the CUDA path.
  - Most call sites are C++ now, so keep `extern "C"` only where a real Python/ctypes or shared-library boundary still needs it.
  - Prefer direct C++ declarations for internal kernel launchers and helper APIs.
- Fuse token embedding gather with the first matrix multiplication in the language path once the unfused baseline is correct.
  - Avoid materializing the initial `[M, 3840]` hidden buffer when the first projection can consume embedding rows directly.
  - Benchmark against the standalone embedding gather plus cuBLAS/cuBLASLt baseline before keeping the fusion.
- Write a custom projection GEMM/GEMV path that emits paged K/V directly.
  - Target decode first: `x @ Wk` / `x @ Wv` should produce final cache values without a contiguous K/V scratch buffer.
  - The epilogue must preserve Gemma semantics: K gets per-head RMSNorm plus RoPE, V gets scale-free RMSNorm, then both scatter into Layout-A paged cache.
  - Benchmark against the lazier baseline first: cuBLAS/cuBLASLt projection into contiguous raw K/V plus fused norm/RoPE/V-norm paged-cache write.
- Add support for different quantization types.
  - Keep weight format selection explicit so kernels do not assume BF16 weights forever.
- Later: investigate persistent producer/consumer orchestration to minimize KV-cache and prepared-Q HBM traffic.
  - Treat the unavoidable paged K/V cache write as the baseline, then look for ways to consume newly written K/V from L2 before it ages out.
  - Prototype a device-side task queue where projection/prep tasks publish ready KV groups and flash-split tasks consume them immediately.
  - Track tiny ready flags, task descriptors, and prepared-Q handoff buffers separately from large streaming weight/KV traffic so they can be kept L2-hot.
  - Explore newest-split-first flash scheduling, L2 persisting windows for prepared Q/partials, and producer-consumer ordering that avoids rereading fresh K/V from HBM when possible.
  - Keep this as a research path after the simple fused projection-to-prep path is correct and benchmarked.
- Later: investigate a persistent projection/FFN worker before trying to fuse FlashAttention.
  - Use one fixed weight layout and let the persistent task scheduler choose prefill-sized GEMM tiles or decode-sized GEMV tiles per work item.
  - Keep prefill and decode projection tasks adjacent by layer/projection where possible so recently streamed weight tiles can get partial L2 reuse.
  - Start with weight-stream-heavy tasks such as FFN gate/up, FFN down, QKV/Q/K projection, O projection, and final LM-head projection; FlashAttention mostly streams KV cache, not layer weights.
- Idea: try cooperative megakernels as the next step.
  - Start with a small cooperative launch prototype before folding more of the decode path into it.
  - Prefer atomics for synchronization instead of `cg::` primitives.

## Remaining Unfused Inference Buildout

Build the remaining model-path pieces in this rough order. Keep each kernel
standalone, numerically tested, benchmarked, and logged in
`src/experiments/EXPERIMENTS.md` before wiring it into the full path.

1. Q/K per-head RMSNorm
   - Sliding layers: learned-weight RMSNorm over head dim `256`.
   - Global layers: learned-weight RMSNorm over head dim `512`.
   - Weights are shared across heads.

2. KV-cache write/update
   - Sliding cache: K/V width `2048`, window `1024`, wraparound.
   - Global cache: K/V width `512`, full context up to `256000`.
   - Support prefill bulk writes, decode appends, and sliding-window wraparound.

3. Sliding-window attention
   - FlashAttention-style local causal attention.
   - Head dim `256`.
   - GQA ratio: `16` Q heads / `8` KV heads = `2` Q heads per KV head.
   - Support prefill local causal attention and single-token decode.

4. Global attention
   - FlashAttention-style full causal attention.
   - Head dim `512`.
   - GQA ratio: `16` Q heads / `1` KV head = `16` Q heads per KV head.
   - Support prefill full causal attention and decode over the full KV cache.

5. Attention output packing
   - Sliding layers produce projection input width `4096`.
   - Global layers produce projection input width `8192`.
   - Prefer having attention kernels write directly in projection-ready layout once correct.

6. GeGLU tanh activation
   - Apply `GELU_tanh(gate) * up`.
   - Width `15360`.
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
   - Shape: `x[3840] -> gate_up[30720]`
   - Estimated weight traffic: `~11.3 GB/token` across 48 layers.
   - Biggest fixed-shape decode tuning target.

2. FFN down GEMV
   - Shape: `ffn_hidden[15360] -> hidden[3840]`
   - Estimated weight traffic: `~5.7 GB/token`.
   - Same every layer, huge, clean.

3. Sliding QKV packed GEMV
   - Shape: `x[3840] -> qkv[8192]`
   - Estimated weight traffic: `~2.5 GB/token` across 40 sliding layers.
   - Do not optimize sliding Q/K/V separately if we can avoid it.
   - Pack as `Q 4096 + K 2048 + V 2048 = 8192`.

4. Sliding O GEMV
   - Shape: `attn_out[4096] -> hidden[3840]`
   - Estimated weight traffic: `~1.3 GB/token`.
   - Repeated across 40 sliding layers.

5. Final vocab/logits GEMV
   - Shape: `hidden[3840] -> logits[262144]`
   - Estimated weight traffic: `~2.0 GB/token`.
   - Huge single kernel.
   - High value if fused with softcap/top-k/argmax, but only runs once per token.

6. Global Q/O GEMVs
   - Global Q shape: `x[3840] -> q[8192]`
   - Global O shape: `attn_out[8192] -> hidden[3840]`
   - Estimated weight traffic: `~0.50 GB/token` each.
