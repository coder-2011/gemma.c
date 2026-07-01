# Design: fold the attention input RMSNorm across the layer boundary

Flag: `GEMMA4_FOLDED_INPUT_NORM` (default 0). Independent of the other flags
but benchmarked stacked with `GEMMA4_FFN_FOLDED_PRE_NORM=1`.

## Why no scale is needed

The input-normed row feeds only the QKV projection. Every consumer of those
projections is scale-invariant: Q and K pass through learned per-head RMSNorms
(RMSNorm(s*v) = RMSNorm(v)), sliding V passes through the scale-free V RMSNorm,
and global V is derived from the projected K source which is also V-normed.
Therefore `W_qkv @ (s * (gamma_in ⊙ x))` and `W_qkv @ (gamma_in ⊙ x)` produce
identical post-norm Q/K/V up to bf16 rounding. Only `gamma_in ⊙ x` must be
staged; the rsqrt scale is never computed or applied.

## Mechanism

Layer i's FFN tail (CTA0 block in `gemma4_ffn_decode_fused_bf16_device`,
src/gemma4_ffn.cu, where `scaled_residual_out_pack` is built and stored)
additionally computes `gamma_next ⊙ scaled_residual_out` per pack via
`gemma4_bf16_pack_apply_scale_weight(scaled_residual_out_pack, gamma_pack,
1.0f)` and stores it to a staging row. Layer i+1 then skips its CTA0 input-norm
phase AND the grid.sync that publishes it
(src/gemma4_flash_attention_decode.cu:492-501 in
`phase_decode_rmsnorm_project_bf16`), reading the staged row instead of
`normed_out`. Cross-kernel launch ordering on the same stream orders the
staging write before the next layer's reads; no new synchronization.

Staging buffer: `args.normed` (the per-step normed scratch row,
`layer_args.normed_out`). Timeline safety: layer i+1's QKV phase reads it
first thing; the only later writer of that buffer within layer i+1 is the
post-attention norm (decode.cu:108), which runs after every CTA has finished
its QKV-phase read (each CTA loads its pack at decode.cu:505-509 before the
tile loop). Note if `GEMMA4_FOLDED_POST_NORM` (separate design) is also on,
that writer disappears entirely.

## Plumbing

1. `Gemma4DecodeMegakernelLayerArgs` (src/gemma4_megakernel.cuh): add
   `bool attention_input_staged = false;` (kernel skips the input-norm phase
   and its grid.sync when true) and pass-through fields for the FFN tail:
   `const __nv_bfloat16 *ffn_next_input_norm_weight = nullptr;`
   `__nv_bfloat16 *ffn_staged_next_input = nullptr;`
2. `decode_layer_args` (src/gemma4_decode_megakernel.cu): under the flag set
   `attention_input_staged = (layer > 0)`,
   `ffn_next_input_norm_weight = layer + 1 < GEMMA4_NUM_LAYERS ?
   args.weights->layers[layer + 1].input_norm_weight : nullptr`,
   `ffn_staged_next_input = args.normed`. Flag off: leave defaults.
3. `gemma4_ffn_decode_fused_bf16_device` (src/gemma4_ffn.cu + .cuh): two new
   trailing params (`next_input_norm_weight`, `staged_next_input`,
   unconditional signature like the previous fold); in the CTA0 tail, when both
   non-null, compute and store the staged pack alongside the existing
   `residual_out` store. Caller (decode.cu FFN handoff, both fold-on and
   fold-off branches) passes `args.ffn_next_input_norm_weight` /
   `args.ffn_staged_next_input`.
4. `phase_decode_rmsnorm_project_bf16` (decode.cu:482+): under the flag, when
   `args.attention_input_staged`, skip the CTA0 norm and grid.sync and load
   `normed_pack` from `args.normed_out` as today (the buffer now holds the
   staged row). When false (layer 0), unchanged path.

## Numerics

Baseline rounds `x*scale*gamma` to bf16; folded rounds `residual*gamma` (and
the QK/V norms erase the missing scale). One different rounding step; greedy
tokens must still match A/B (verify).

## Expected gain

Removes one CTA0 full-row phase (~3 us) plus one grid.sync (~1.5 us) for 47 of
48 layers: ~200 us/step, ~0.5% on the 3090.
