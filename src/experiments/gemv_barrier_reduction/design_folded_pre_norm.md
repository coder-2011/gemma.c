# Design: fold pre-FFN RMSNorm into the gate/up GEMV (scale-at-the-end)

Flag: `GEMMA4_FFN_FOLDED_PRE_NORM` (default 0 = baseline unchanged).
Prototype scope: default FFN path only (`GEMMA4_FFN_WARP_TILED=0`); `#error` if
both flags are 1.

## Identity

For pre-FFN norm `x = s2 * (gamma_pre ⊙ y)` with
`s2 = rsqrtf(sum(y^2)/HIDDEN + eps)` (gamma applied plain, no 1+w — this repo
pre-bakes the offset), every gate/up dot satisfies:

```
dot(w, x) = s2 * dot(w, gamma_pre ⊙ y)
```

so the FFN can consume the RAW post-attention residual row `y`, multiply each
x-element by gamma_pre inside the dot loop (fp32), and apply the scalar `s2`
once after the block reduction, BEFORE gelu:

```
gate = s2 * gate_raw;  up = s2 * up_raw;  act = gelu_tanh(gate) * up
```

This deletes the CTA0-only pre-FFN norm phase
(`gemma4_flash_attention_decode.cu:1059-1065`) and the grid.sync at `:1067`.
The FFN's inputs (`y` row and `s2`) are already ordered by the existing
grid.sync at `:128` inside `project_attention_out_post_attention`.

## Where s2 comes from

CTA0 already walks the full row when it writes
`ffn_residual = attention_x + normed_out` (`:116-124`). Under the flag it
additionally accumulates `sum_sq` of the bf16-ROUNDED packs it writes (use the
rounded values so s2 is bit-identical to what the deleted norm would compute),
block-reduces (same pattern as `reduce_ffn_hidden_pack_sum`), and thread 0
stores `s2 = rsqrtf(sum_sq / GEMMA4_HIDDEN_SIZE + GEMMA4_RMS_NORM_EPS)` to a
scale slot in global scratch. All of this happens before the sync at `:128`.

Scale slot: the tail of `partial_acc` reserved in `gemma4_prompt.cu:244-249`
(`GEMMA4_HIDDEN_SIZE + 1` floats after
`GEMMA4_NUM_QUERY_HEADS * max_alloc_splits * GEMMA4_GLOBAL_HEAD_DIM`); use the
final float (`tail[GEMMA4_HIDDEN_SIZE]`). The kernel cannot derive
`max_alloc_splits` from its own args (they carry active splits), so the host
computes the pointer: in `gemma4_decode_megakernel.cu`, before splits are
clamped, `max_alloc_splits = max(args.sliding_splits, args.global_splits)`;
new layer-args field `float *pre_ffn_scale` =
`args.partial_acc + NUM_QUERY_HEADS * max_alloc_splits * GLOBAL_HEAD_DIM
+ GEMMA4_HIDDEN_SIZE`.

## Changes by file

1. `src/gemma4_megakernel.cuh`: add `float *pre_ffn_scale = nullptr;` to
   `Gemma4DecodeMegakernelLayerArgs`.
2. `src/gemma4_decode_megakernel.cu` (`decode_layer_args`): set it per the
   formula above (use the caller-capacity splits from the pre-clamp args).
3. `src/gemma4_matmul_kernels.cu`: new
   `extern "C" __device__ void gemma4_ffn_gate_up_tile_folded_norm_bf16_device(
   x_raw, gamma, w_interleaved_row_major, col0, thread_idx, warp_sums, gate,
   up)` - clone of `gemma4_ffn_gate_up_tile_bf16_device` (`:152-209`) where
   each x pack is converted to fp32, multiplied elementwise by the gamma pack
   (natural, unswizzled index like x), and that product feeds the same
   per-column FMA loop. gamma pack load uses the same `x_col` natural offset.
   Do NOT modify the existing tile function.
4. `src/gemma4_ffn.cu` + `src/gemma4_ffn.cuh`: extend
   `gemma4_ffn_decode_fused_bf16_device` signature (unconditionally) with
   `const __nv_bfloat16 *pre_norm_weight, const float *pre_ffn_scale`. Under
   the flag, `accumulate_decode_intermediate_tile` calls the folded-norm tile
   helper with x = raw row and gamma = pre_norm_weight, and thread 0 computes
   `const float s2 = __ldg(pre_ffn_scale);
   activation[t] = gelu_tanh(s2 * gate[t]) * (s2 * up[t]);`
   Flag off: existing behavior, new params unused.
5. `src/gemma4_flash_attention_decode.cu`:
   - `project_attention_out_post_attention` CTA0 block: under flag, accumulate
     rounded-pack sum_sq during the `:116-124` write loop, block-reduce, store
     s2 to `args.pre_ffn_scale` (still before the `:128` sync).
   - `decode_megakernel_fused_layer_kernel` (`:1059-1072`): under flag, skip
     the CTA0 pre-FFN norm and the grid.sync at `:1067`; pass
     `x = args.ffn_residual` (raw), plus
     `args.attention_pre_ffn_norm_weight` and `args.pre_ffn_scale` to the FFN.
     Flag off: pass the two new params but keep old behavior
     (`x = args.ffn_x`, params ignored).

## Numerics

s2 matches baseline bit-for-bit (same rounded inputs, same reduction shape as
the deleted norm is acceptable if the block-reduce order differs slightly).
The dot differs from baseline by one rounding: baseline rounds
`y*s2*gamma` to bf16 before the dot; folded keeps `y*gamma` in fp32 and scales
after - slightly MORE precise. Greedy tokens must still match A/B.

## Acceptance

- `make prompt BUILD_DIR=build_sm86` (flag off) binary is behavior-identical.
- Flag-on build compiles with no spills for the fused kernel on sm_86.
- Token check A/B identical; decode-step p50 at seq 121/1001/4001 on the 3090
  reported vs baseline.
