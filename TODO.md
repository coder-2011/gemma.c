# TODO

*Agent maintained, quite messy. They never listen to me when I tell them to keep it clean...*

## Whole-Kernel Tuning Queue

Tune every project-owned runtime kernel until further sweeps produce only small,
non-reproducible changes. Treat this as a standing optimization ledger: each
entry needs correctness coverage, warm/cold benchmark data where relevant,
resource/SASS inspection for close calls, and an experiment-log entry before
the tuned result is accepted.

Tuning contract:

- Locked by Gemma 4 architecture: hidden size `5376`, intermediate size `21504`,
  vocab `262144`, layer count/pattern, Q/KV head counts, head dims, GQA ratios,
  sliding window `1024`, global full-context semantics, RoPE/p-RoPE dimensions
  and theta values, RMS eps/model math, GeGLU tanh semantics, global `K=V`, and
  final softcap semantics.
- Free tuning knobs: launch shapes, rows/heads/columns per block, split sizes,
  graph-compatible overprovisioning, cache page size/layout, vector widths,
  load/store cache policy, shared-memory staging, `cp.async`, CUB vs custom
  reductions, swizzles, prefetch/register staging, `__launch_bounds__`, CUTLASS
  tile configs, compile-time specialization, and benchmark harness shape ranges.
- Stop when the best candidate is correct and two follow-up sweeps fail to show
  a reproducible median improvement above the benchmark's declared minimum
  effect size, or when the improvement is smaller than the noise floor after
  process-level reruns.
- Dynamic-shape kernels must be tuned across representative ranges, not one
  lucky shape. Record the accepted range and any shape-specific dispatch rules.

Runtime kernel tuning todos:

- [x] Decision: tune `embedding_gather_kernel` in `src/gemma4_embedding_gather.cu`.
  Sweep token counts, load policy, threads per row, multi-row CTAs, and whether
  the gather should be fused into first projection paths.
  - Accepted `GEMMA4_EMBEDDING_GATHER_THREADS=128` on 2026-06-18. The main
    4096/8192-token bandwidth win over 32 threads was small, around `0.3-0.6%`,
    but 128 improved cold small-token medians without hurting large-token
    throughput. Stop here until gather is fused with first projection.
- [x] Decision: tune `gemma4_rmsnorm_bf16_decode_kernel` in `src/gemma4_rmsnorm.cu`.
  Sweep decode thread count, input staging, weight load policy, launch bounds,
  and graph-replay timing for hidden width `5376`.
  - Accepted a 512-thread decode launch on 2026-06-18. Raw
    single-row timings were noisy, but 512 was the best low-resource point in
    the focused sweep and graph timings were tied within about `0.1us`.
- [x] Decision: tune `gemma4_residual_add_rmsnorm_bf16_decode_kernel` in
  `src/gemma4_rmsnorm.cu`. Sweep decode threads, residual load/store policy,
  register lifetime, and whether the normed store should use write-back or
  streaming policy.
  - Accepted a 704-thread fused decode launch on 2026-06-18. Counts
    `704` and `768` were effectively tied under graph replay; keep 704 as the
    lower-thread default. Added a static guard that this kernel needs at least
    one thread per hidden pack.
- [x] Decision: tune `gemma4_residual_add_rmsnorm_bf16_hidden_prefill_kernel` in
  `src/gemma4_rmsnorm.cu`. Sweep rows `1..prefill max`, min blocks per SM,
  register pressure, and whether one-pack-per-thread remains best.
  - Accepted no change to the min-blocks-per-SM value `2` on
    2026-06-18. Min-blocks `1` and `2` were identical through 1024 rows, and
    one-pack-per-thread remains the only sensible ownership model for width
    `5376`.
- [x] Decision: tune `gemma4_rmsnorm_bf16_shared_kernel` in
  `src/gemma4_rmsnorm.cu`. Sweep widths `256`, `512`, `5376`, rows, rows per
  block, shared weight staging, async input staging, and direct vs staged input.
  - Accepted 3 rows per block on 2026-06-18 for learned
    hidden RMSNorm. This improves hidden width `5376` prefill timing while
    staying under the default dynamic shared-memory limit.
- [x] Decision: tune `gemma4_rmsnorm_bf16_direct_weight_kernel` in
  `src/gemma4_rmsnorm.cu`. Sweep the same width/row range as the shared-weight
  path and keep it only where direct weight loads beat staged weights.
  - Accepted width-specific direct-weight row packing on 2026-06-18:
    8 rows per block for width `256` and
    2 rows per block for width `512` and small-width
    fallback.
- [x] Decision: tune `gemma4_rmsnorm_scale_free_bf16_shared_kernel` in
  `src/gemma4_rmsnorm.cu`. Focus on V-norm widths `256` and `512`, rows,
  staging, and store policy.
  - Accepted the same width-specific row packing as the learned direct path on
    2026-06-18. Scale-free width `512` prefers the two-row global packing;
    width `256` benefits from the denser sliding packing.
- [x] Decision: tune `gemma4_residual_add_bf16_kernel` in
  `src/gemma4_rmsnorm.cu`. Sweep element counts, block size, load/store policy,
  and fusion opportunities with following RMSNorm.
  - Accepted no change to the 256-thread residual-add launch on 2026-06-18.
    Block sizes `128`, `256`, and `512` were within noise at hidden width
    `5376`; `1024` regressed at the largest measured row count.
- [x] Decision: tune `gemma4_rope_bf16_kernel` in `src/gemma4_rope.cu`.
  Sweep grid ordering, table layout/stride, packed load/store policy, sliding
  `D=256` and global p-RoPE `D=512/R=128`.
  - Accepted no default change on 2026-06-18: keep one warp per `(row, head)`,
    row-major grid ordering, and cache-streaming Q/K loads. Added
    `GEMMA4_ROPE_THREADS` as a guarded sweep knob.
- [x] Decision: tune `gemma4_rope_forward_bf16_kernel` in `src/gemma4_rope.cu`.
  Sweep the Python-facing layout separately from physical-layout RoPE and keep
  or delete if fused prep paths cover every real use.
  - Accepted the same launch defaults as physical-layout RoPE on 2026-06-18.
    The benchmark now reports physical and forward layout timings separately.
- [x] Decision: tune `kv_cache_write_kernel` in `src/gemma4_kv_cache.cu`.
  Treat as scalar fallback: benchmark odd/non-vector layouts and either improve
  it or make the fallback contract explicit.
  - Accepted a 128-thread scalar KV write launch on 2026-06-18 using write-only
    `head_dim=260` scalar-fallback benchmarks. Added scalar write correctness
    coverage with `head_dim=10`.
- [x] Decision: tune `kv_cache_write_vec_kernel` in `src/gemma4_kv_cache.cu`.
  Sweep page sizes, token counts, head dims `256/512`, cache store policy, and
  whether metadata loads should be page-span aware.
  - Accepted no default change to the 128-thread vector KV write launch on
    2026-06-18. It is the best all-around decode-write choice for sliding
    width `256` and global width `512`; `32/64` hurt decode writes and `256`
    regressed.
- [x] Decision: tune generic `paged_decode_split_kernel` in
  `src/gemma4_kv_cache.cu`. It is fallback/global-reference code today; tune
  across sliding/global configs or delete after specialized paths fully replace
  it.
  - Accepted no code change on 2026-06-18. Runtime `split_size=16` is best for
    sliding at effective key count `1024`; global preferred `16` at seq `1024`
    and `32` at seq `4096`.
- [x] Decision: tune generic `paged_decode_reduce_kernel` in
  `src/gemma4_kv_cache.cu`. Sweep split counts, vectorized partial reads, and
  whether it remains needed after specialized decode reducers mature.
  - Accepted the same runtime split-size guidance as the split kernel on
    2026-06-18. No partial-layout code change was made because specialized
    FlashAttention decode is the production path.
- [x] Decision: tune projection decode GEMV `gemma4_decode_gemv_cols_kernel`
  for `GEMMA4_PROJECTION_FFN_GATE_UP`. Shape `5376 -> 43008`; sweep threads,
  cols per block, swizzle, load policy, buffering, and warp-owned variants.
  - Accepted `ColsPerBlock=4`, `Threads=512`, streaming weight loads, one
    register-buffer stage, identity swizzle on 2026-06-18.
- [x] Decision: tune projection decode GEMV `gemma4_decode_gemv_cols_kernel`
  for `GEMMA4_PROJECTION_FFN_DOWN`. Shape `21504 -> 5376`; sweep separately
  from gate/up because reduction length and output count differ.
  - Accepted `ColsPerBlock=4`, `Threads=1024`, streaming weight loads, one
    register-buffer stage, identity swizzle on 2026-06-18.
- [x] Decision: tune projection decode GEMV `gemma4_decode_gemv_cols_kernel`
  for `GEMMA4_PROJECTION_SLIDING_QKV`. Shape `5376 -> 16384`; include packed
  Q/K/V epilogue possibilities.
  - Accepted `ColsPerBlock=4`, `Threads=512`, streaming weight loads, one
    register-buffer stage, identity swizzle on 2026-06-18.
- [x] Decision: tune projection decode GEMV `gemma4_decode_gemv_cols_kernel`
  for `GEMMA4_PROJECTION_SLIDING_O`. Shape `8192 -> 5376`; compare CTA-wide,
  warp-owned, and two-warp-per-column variants.
  - Accepted `ColsPerBlock=4`, `Threads=512`, streaming weight loads, one
    register-buffer stage, identity swizzle on 2026-06-18.
- [x] Decision: tune projection decode GEMV `gemma4_decode_gemv_cols_kernel`
  for `GEMMA4_PROJECTION_GLOBAL_Q`. Shape `5376 -> 16384`; verify it does not
  accidentally inherit sliding-QKV-specific choices.
  - Accepted `ColsPerBlock=4`, `Threads=512`, streaming weight loads, one
    register-buffer stage, identity swizzle on 2026-06-18.
- [x] Decision: tune projection decode GEMV `gemma4_decode_gemv_cols_kernel`
  for `GEMMA4_PROJECTION_GLOBAL_K`. Shape `5376 -> 2048`; prioritize launch
  overhead/reduction efficiency because this is a smaller output.
  - Accepted `ColsPerBlock=4`, `Threads=512`, streaming weight loads, one
    register-buffer stage, identity swizzle on 2026-06-18.
- [x] Decision: tune projection decode GEMV `gemma4_decode_gemv_cols_kernel`
  for `GEMMA4_PROJECTION_GLOBAL_O`. Shape `16384 -> 5376`; sweep long-K
  reductions independently.
  - Accepted `ColsPerBlock=4`, `Threads=512`, streaming weight loads, one
    register-buffer stage, identity swizzle on 2026-06-18. A `768`-thread
    confirmation was tied within noise, so the existing thread count stayed.
- [x] Decision: tune projection decode GEMV `gemma4_decode_gemv_cols_kernel`
  for `GEMMA4_PROJECTION_FINAL_LOGITS`. Shape `5376 -> 262144`; benchmark with
  softcap/argmax fusion candidates once the standalone path is stable.
  - Accepted `ColsPerBlock=4`, `Threads=1024`, streaming weight loads, one
    register-buffer stage, identity swizzle on 2026-06-18. Swizzle16 stayed a
    sub-0.1% microbenchmark-only edge, so production keeps identity traversal.
- [x] Decision: tune `gemma4_ffn_decode_accumulate_bf16_kernel` in
  `src/gemma4_ffn_decode.cu`. Sweep act tile, intermediate tile, accumulation
  block count, swizzle, weight load policy, down preload, and atomic pressure.
  - Accepted no default change on 2026-06-18. Keep `INTERMEDIATE_TILE=2`,
    `ACT_TILE=2`, `ACCUM_BLOCKS=10080`, hidden-pack swizzle on, streaming
    weight loads, and no down preload. Near-miss accum-block counts tied or
    regressed under longer confirmation.
- [x] Decision: tune `gemma4_ffn_decode_accumulate_partials_bf16_kernel` in
  `src/gemma4_ffn_decode.cu`. Sweep partial groups and compare against the
  atomic accumulator policy across warm/cold cache.
  - Accepted no production use on 2026-06-18. Partial-groups mode was much
    slower than atomic accumulation for all tested group counts because scratch
    clear/finalize traffic grew sharply.
- [x] Decision: tune `gemma4_ffn_decode_finalize_bf16_kernel` in
  `src/gemma4_ffn_decode.cu`. Sweep RMS reduction shape, accumulator layout,
  residual load policy, and final store policy.
  - Accepted no default change on 2026-06-18. Added guarded final load/store
    policy knobs, then kept `ldg` final loads, plain residual store, and
    write-back normed store after confirmation. Accumulate/finalize resources
    stayed at zero spills.
- [x] Decision: tune `swizzle_hidden_packs_kernel` in
  `src/gemma4_ffn_decode.cu` as an offline transform. Optimize only enough that
  model load/prep time is acceptable; do not trade decode speed for prep speed.
  - Accepted 96 swizzle threads and 7 blocks per row on 2026-06-18. The final
    down-weight swizzle measured `0.678330 ms` best at about `681.7 GB/s`, with zero
    spills. Further geometry/cache-policy sweeps were below the prep benchmark
    noise floor.
- [x] Decision: tune `swizzle_gate_up_interleaved_kernel` in
  `src/gemma4_ffn_decode.cu` as an offline transform. Validate layout choices
  against decode FFN performance before changing it.
  - Accepted the same swizzle launch defaults as hidden-pack swizzle on
    2026-06-18. The prepared layout did not change, only the offline launch
    shape; gate/up swizzle measured `1.352854 ms` best at about `683.6 GB/s`,
    and decode-path timing remains governed by the already-tuned FFN kernels.
- [x] Decision: tune `gemma4_ffn_prefill_geglu_bf16_kernel` in
  `src/gemma4_ffn_decode.cu`. Sweep vectorization, block size, packed BF16
  loads/stores, and fusion with prefill GEMM epilogues.
  - Accepted 256 GeGLU threads and 2 elements per thread on 2026-06-18. Final
    row-range timing improved the large prefill end to `0.190798 ms` at rows
    `1024` and avoided broad regressions across rows `1..1024`; the kernel
    uses 15 registers and zero spills.
- [x] Decision: tune FFN prefill CUTLASS gate/up and down GEMM configs in
  `src/gemma4_ffn_decode.cu`. These are library kernels, but the tile choices
  are ours; sweep rows across the expected prefill range.
  - Accepted no production tile change on 2026-06-18. Keep gate/up
    `128x128x64` threadblock, `64x64` warp, `3` stages; keep down small-row
    `64x128x64`, `32x64`, `3` stages through `128` rows; keep down large-row
    `128x128x64`, `64x64`, `3` stages. Added guarded tile knobs and a sampled
    correctness benchmark over rows `1..1024`. Rejected the opt-in small gate/up
    dispatch because its `<=64` row win regressed large rows, and rejected
    `N=256`, `K=128`, and `4`-stage variants due correctness or build failures.
- [x] Decision: tune `gemma4_flash_fwd_bf16_kernel` sliding instantiations in
  `src/gemma4_flash_attention.cu`. Prioritize no-LSE inference, but benchmark
  LSE too; sweep block M/N, warps, stages, mask specialization, and spills.
  - 2026-06-18 first pass: added compile-time tile knobs plus prepared-QKV FA
    timing to the C++ benchmark. Default `64x64x4` remains the large-seq winner
    after pruning `64x32x4`, `32x64x4`, `32x32x4`, `64x64x2`, `64x64x8`,
    `128x32x4`, and `64x48x4`. The follow-up below closes the then-remaining
    small-seq dispatch, cold-cache, and resource-check items.
  - Accepted `64x64x4` plus a local mask-specialized steady loop on
    2026-06-18. The middle K/V blocks now run unmasked; only the causal edge
    and final left-window boundary keep mask checks. Rejected small-seq
    `64x32x4` dispatch after confirmation failed to reproduce a reliable win.
    Final prepared no-LSE timing improved from `0.606250 ms` to `0.585129 ms`
    at seq `2048` and from `1.469820 ms` to `1.406290 ms` at seq `4096`.
    Sliding FA stays zero-spill; `ncu` is not installed on this machine, so the
    resource pass used ptxas and CUDA function attributes.
- [x] Decision: tune `gemma4_flash_fwd_bf16_kernel` global instantiations in
  `src/gemma4_flash_attention.cu`. Head dim `512` spills today; compare against
  upstream FA2-style traits and tune resources before changing semantics.
  - Accepted a two-tile global dispatch on 2026-06-18: keep `32x32x2` for
    seq `<256`, and use `32x16x2` for seq `>=256`. The large tile reduced the
    no-LSE global FA stack/spill footprint from `704 B`, `1432/1600 B` to
    `336 B`, `1056/1200 B`, and improved warm prepared global FA medians by
    `1.3-1.47x` across seq `256..2048`. Rejected `16x32x2`, `16x16x2`,
    `32x32x4`, `48x16x2`, `64x16x2`, and `32x16x4`; the latter three were
    either much slower or numerically invalid against the PyTorch reference.
- [x] Decision: tune `gemma4_qkv_norm_rope_kernel` sliding prefill
  instantiation in `src/gemma4_flash_attention.cu`. Sweep heads per block,
  lane mapping, table loads, V norm, and prepared tensor stores.
  - Accepted a two-shape prefill launch dispatch on 2026-06-18: `8` heads per
    block below seq `512`, and `32` heads per block from seq `512` upward.
    The retained large path improved warm prep-only medians from `0.107574 ms`
    to `0.103582 ms` at seq `1024`, and from `0.422517 ms` to `0.403156 ms`
    at seq `4096`. Cold confirmation stayed close: `0.105822 ms` at seq
    `1024` and `0.405318 ms` at seq `4096`. Both retained instantiations are
    zero-spill in ptxas; the large path uses `49` registers.
- [x] Decision: tune `gemma4_qkv_norm_rope_kernel` global prefill
  instantiation in `src/gemma4_flash_attention.cu`. Include p-RoPE `128`
  rotary dims, global `K=V`, and `D=512` register pressure.
  - Swept `4`, `8`, `16`, and `32` heads per block on 2026-06-18 with direct
    prepared-tensor correctness. Kept the default `8` heads per block: `w32`
    won some mid-context warm runs, `w16` won one long run, and the variance
    was too high for a reliable global dispatch. Final retained warm prep-only
    medians were `0.150512 ms` at seq `1024` and `0.566512 ms` at seq `4096`;
    ptxas reports `40` registers and zero spills.
- [x] Decision: tune `gemma4_sliding_decode_q_paged_kv_norm_rope_kernel` in
  `src/gemma4_flash_attention.cu`. Sweep batch shapes, metadata loads, cache
  write policy, and whether Q prep plus cache write should fuse with projection.
  - Accepted `2` heads per block on 2026-06-18. Broad warm sweeps over
    `1/2/4/8/16/32` heads per block and follow-up batches `1/2/4/8/16`
    kept `h2` as the best all-range choice; the only near-miss was `h1` at
    batch `16`, below the declared effect size and not worth a branch. Final
    warm medians span `0.002337..0.002938 ms` for batches `1..16`, and cold
    representatives were `0.005574 ms` at batch `1` and `0.006918 ms` at
    batch `16`. ptxas reports `49` registers and zero spills.
- [x] Decision: tune `gemma4_global_decode_q_paged_kv_norm_rope_kernel` in
  `src/gemma4_flash_attention.cu`. Sweep the same decode-prep knobs under
  global `D=512`, `K=V`, and p-RoPE.
  - Accepted a batch-size dispatch on 2026-06-18: `2` heads per block for
    batch `<8`, and `1` head per block for batch `>=8`. Focused warm
    confirmation kept `h2` through batch `4` and `h1` at batches `8/16`.
    Final warm medians span `0.002585..0.003429 ms`; cold representatives
    were `0.006733 ms` at batch `4`, `0.007155 ms` at batch `8`, and
    `0.007526 ms` at batch `16`. ptxas reports zero spills; the `h2` path
    uses `57` registers and the `h1` large-batch path uses `66`.
- [x] Decision: tune specialized `decode_paged_grouped_split_kernel` sliding
  instantiation in `src/gemma4_flash_attention.cu`.
  - Accepted split-size guidance/default `16` for steady-state sliding decode;
    short pre-window contexts can prefer `32`, but steady-state window timing is
    the target. `page_size=64` remains the default because page-size wins were
    shape/noise dependent. Final C++ steady-state `seq=4096` medians improved
    over split `64`: warm `0.063820 -> 0.057857 ms`, cold
    `0.070946 -> 0.064954 ms`.
- [x] Decision: tune specialized `decode_paged_reduce_kernel` sliding
  instantiation in `src/gemma4_flash_attention.cu`.
  - Kept the graph-compatible partial layout and head-dim-wide reduce CTA.
    ptxas reports zero spills: split `36` registers/thread, reduce `27`
    registers/thread. No separate reduce-thread or layout variant earned its
    complexity in the split-size/page-size sweeps.
- [x] Decision: tune specialized `decode_paged_grouped_split_kernel` global
  instantiation in `src/gemma4_flash_attention.cu`.
  - Added runtime dispatch between global query-head grouping `2` for short
    spans and `4` once `split_size * num_splits >= 4096`. This preserves Gemma's
    architectural GQA ratio `8` while tuning CTA ownership. Final medians:
    global `seq=1024`, split `16` warm `0.100891 ms`; global `seq=4096`, split
    `32` warm/cold `0.329709/0.323679 ms`; global `seq=8192`, split `32`
    warm/cold `0.619349/0.599520 ms`.
- [x] Decision: tune specialized `decode_paged_reduce_kernel` global
  instantiation in `src/gemma4_flash_attention.cu`.
  - Kept the shared partial layout across short/long global split kernels.
    ptxas reports zero spills for both global grouping variants: split `40`
    registers/thread, reduce `30` registers/thread. Long global decode should
    use split `32`; split `64` was slower at `seq=4096`
    (`0.352979 ms` warm versus `0.329709 ms`).

Experimental and benchmark-owned kernels:

- [x] Decision: audit helper kernels in `src/benches/*` and `tests/*`.
  - Audited on 2026-06-18. Fill/conversion kernels such as
    `gemma4_fill_random_bf16_kernel`, the local `fill_random_bf16_kernel`
    variants, `gemma4_rmsnorm_fill_random_bf16_kernel`,
    `gemma4_rmsnorm_fill_constant_bf16_kernel`,
    `fill_unit_rope_table_kernel`, `float_to_bf16_kernel`,
    `fill_float_kernel`, and the experiment-local `fill_bf16_kernel`
    variants are input setup only and are kept outside timed regions.
  - L2/cache controls `l2_flush_kernel` and `flush_cache_kernel` are benchmark
    policy mechanisms, not model kernels. Most harnesses record the CUDA start
    event after the flush; `gemma4_ffn_decode_load_bench.cu` intentionally times
    cold flush-plus-work and also reports matched flush-subtracted metrics.
  - Check/diff kernels such as `check_geglu_kernel`,
    `check_gate_up_swizzle_kernel`, `check_down_swizzle_kernel`,
    `check_gate_up_gemm_kernel`, `check_down_gemm_kernel`,
    `gemma4_experiment_max_abs_diff_kernel`, and
    `warp_reduce_sum_real_kernel` are correctness/test helpers. Do not tune
    them for model speed unless a future benchmark places them inside a claimed
    timed region.
- [x] Decision: exclude imported third-party experimental kernels from the
  project-owned tuning queue until promoted into Gemma code.
  - `experiments/marlin/marlin/marlin_cuda_kernel.cu::Marlin`, upstream
    FlashAttention template kernels, and vendored CUTLASS examples are not
    maintained as Gemma 4 runtime kernels in this repo. If any are promoted
    into the main inference path, add a new production tuning TODO with Gemma
    shape ranges and correctness gates.
- [x] Decision: delete retired `gemma4_experiment_hgemm_tn_cute_kernel` in
  `src/benches/16384_512_4096.cu`.
  - Retired on 2026-06-18. The standalone experiment already showed the custom
    CUTE kernel losing to cuBLAS on its isolated shape (`0.756506 ms` custom
    versus `0.690792 ms` cuBLAS, exact deterministic-output match), and the
    later matmul simplification entry explicitly kept it as experiment history
    instead of runtime code. The file was not wired into `Makefile` or any test
    target, so deleting the orphan source is the correct tune/delete decision.
    `gemma4_experiment_fill_half_kernel` and
    `gemma4_experiment_max_abs_diff_kernel` were setup/check helpers covered by
    the helper audit above.
- [x] Decision: retire the Tuna prefill experiment kernels:
  `gemma4_tuna_prefill_bf16_kernel` and
  `gemma4_tuna_prefill_bf16_smem_kernel` in
  `experiments/tuna/gemma4_prefill_bench.cu`.
  - Retired on 2026-06-18 as a production prefill candidate. The cuBLASLt
    all-shape sweep across Gemma projections and `M=16,64,256` routed every
    case to cuBLASLt; Tuna custom-only weighted speed was `0.2476x` of
    cuBLASLt. The kernels remain useful only as negative-result experiment
    scaffolding unless a materially different pipelined Tuna BF16 design is
    written. Generic imported Tuna kernels `Tuna` and `dequantize_4bit` remain
    excluded unless promoted into Gemma code.
- [x] Decision: retire the SGEMM FP32 experiment kernels:
  `kernel_basic`, `sgemm_128x128x8`, `sgemm_texld_128x128x8`, and
  `sgemm_128x256x8` under `experiments/sgemm.cu/src/kernels/`, plus the
  Gemma FP32 harness path in `experiments/sgemm.cu/gemma4_prefill_bench.cu`.
  - Retired on 2026-06-18 for inference tuning. The FP32 harness was useful as
    a shape baseline and showed narrow `global_k` wins over `cublasSgemm`, but
    it does not match Gemma 4's BF16 inference datatype. The follow-up BF16 and
    cuBLASLt sweeps supersede this path; keep it only as reference/experiment
    code, not as a production kernel family to tune.
- [x] Decision: retire the SGEMM BF16 experiment kernels in
  `experiments/sgemm.cu/gemma4_bf16_prefill_bench.cu`:
  `gemma4_sgemm_bf16_kernel`, `gemma4_sgemm_bf16_smem_kernel`,
  `gemma4_sgemm_bf16_smem_a_kernel`,
  `gemma4_sgemm_bf16_wide_warp_kernel`,
  `gemma4_sgemm_bf16_splitk_kernel`, and
  `gemma4_sgemm_bf16_splitk_reduce_kernel`.
  - Retired on 2026-06-18 as production prefill candidates. The basic BF16
    all-shape cuBLASLt sweep routed every Gemma shape to cuBLASLt and measured
    custom-only at `0.2475x` of cuBLASLt. The later `ffn_down`-focused CUTLASS,
    Stream-K, graph-timed, and multi-heuristic cuBLASLt sweeps found near-parity
    local cases but no `1.05x` dynamic-M routing win across `M=8..1024`; the
    present h32 artifacts give custom-only `0.9773x` versus cuBLASLt. Keep the
    harness only for future substantially different custom-GEMM designs.

## Future Kernel Work Beyond This Tuning Pass

- RMSNorm cleanup/build-break note resolved on 2026-06-18.
  - `make test-rmsnorm NVCC=/usr/local/cuda/bin/nvcc` compiles and passes.
  - RMSNorm tuning results above cover hidden width `5376` and Q/K/V widths
    `256` and `512`, including the hidden-only fused residual+RMSNorm guard.
- Fuse token embedding gather with the first matrix multiplication in the language path once the unfused baseline is correct.
  - Avoid materializing the initial `[M, 5376]` hidden buffer when the first projection can consume embedding rows directly.
  - Benchmark against the standalone embedding gather plus cuBLAS/cuBLASLt baseline before keeping the fusion.
- Write a custom projection GEMM/GEMV path that emits paged K/V directly.
  - Target decode first: `x @ Wk` / `x @ Wv` should produce final cache values without a contiguous K/V scratch buffer.
  - The epilogue must preserve Gemma semantics: K gets per-head RMSNorm plus RoPE, V gets scale-free RMSNorm, then both scatter into Layout-A paged cache.
  - Benchmark against the lazier baseline first: cuBLAS/cuBLASLt projection into contiguous raw K/V plus fused norm/RoPE/V-norm paged-cache write.

## Future Unfused Inference Buildout Notes

These are future model-path buildout notes, not remaining decisions for the current
whole-codebase kernel-tuning pass. Any newly added kernel should get its own
tuning TODO, numerical test, benchmark under `src/benches/`, and dated entry in
`src/experiments/EXPERIMENTS.md` before it is promoted.

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
   - Later: add chunked/paged prefill for very long prompts, prefix-cache reuse,
     and serving paths where contiguous prompt K/V is too limiting.

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

## Resolved Decode GEMV Tuning Notes

- The shape-specific `gemma4_decode_gemv_cols_kernel` decisions above supersede
  the earlier open warp-tiled variant note.
- Earlier experiments tested narrow warp-owned-column and column-grouping
  variants; they did not improve the production route enough to keep.
- The final June 18 all-shape sweep accepted `ColsPerBlock=4`, shape-specific
  thread counts, streaming weight loads, one register-buffer stage, and identity
  traversal for the retained production decode GEMV paths.
- Future GEMV work should be a new kernel/fusion design, such as producer or
  consumer fusion around paged K/V or logits sampling, not a remaining tuning
  obligation for the current `gemma4_decode_gemv_cols_kernel`.
