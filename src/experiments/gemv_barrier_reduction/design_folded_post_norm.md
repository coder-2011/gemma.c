# Design: distribute the post-attention norm + residual across all CTAs

Flag: `GEMMA4_FOLDED_POST_NORM` (default 0). Requires
`GEMMA4_FFN_FOLDED_PRE_NORM=1` (#error otherwise): it replaces that fold's
CTA0 staging block with a fully distributed epilogue.

## What it replaces

In `project_attention_out_post_attention`
(src/gemma4_flash_attention_decode.cu:36-154), between the grid.sync that
publishes the O-projection row `r` (`:103`) and the trailing grid.sync
(`:153`), CTA0 alone currently: (1) RMSNorms `r` into `normed_out`
(`s1 = rsqrtf(mean(r^2)+eps)`, gamma_post applied plain), (2) writes
`ffn_residual = attention_x + normed`, (3) [pre-FFN fold] accumulates
`sum(y^2)`, stages `gamma_pre ⊙ y` into `ffn_x`, stores s2. That is two
serial full-row passes (~5-6 us) on one SM while 81 idle.

## Distributed version

Deterministic per-CTA partial-sum slots live in the reserved tail of
`partial_acc` (see `gemma4_prompt.cu:244-249`: `GEMMA4_HIDDEN_SIZE + 1` floats
after the split accumulators; `pre_ffn_scale` currently uses the final float).
Define the tail layout with named constexpr offsets (a small struct of
pointers derived from the existing `pre_ffn_scale` base minus
`GEMMA4_HIDDEN_SIZE`, or plumb a second pointer `float *norm_partials` the
same way `pre_ffn_scale` is plumbed in src/gemma4_decode_megakernel.cu):
- `r2[cta]` for cta in [0, gridDim.x): partial sum of r^2 - slots [0, 512)
- `y2[cta]`: partial sum of y^2 - slots [512, 1024)
Grid size is at most 2 CTAs/SM x SM count (<= 376 today); static_assert-style
guard that 1024 <= GEMMA4_HIDDEN_SIZE.

Producer side (before `:103`): in the O-projection col_block loop, thread 0
already holds the 8 reduced sums for its pack (`:90-98`); accumulate
`sum += sums[col]^2` OF THE bf16-ROUNDED values (square `__bfloat162float(o_pack[col])`,
matching what the deleted RMSNorm would read) into a thread-0 register across
the CTA's col_blocks; after the loop thread 0 writes `r2[blockIdx.x]` (every
CTA writes its slot, even if it owned zero col_blocks -> write 0.0f).

Epilogue (between `:103` and `:153`), replacing the whole CTA0 block: every
thread grid-strides over hidden packs
(`for (pack = blockIdx.x*blockDim.x+threadIdx.x; pack < kFfnHiddenPacks;
pack += gridDim.x*blockDim.x)`):
1. Once per CTA before the loop: `s1 = rsqrtf(sum(r2[0..gridDim.x))/HIDDEN + eps)`
   summed in fixed slot order (deterministic); each thread can compute it
   (gridDim.x reads from L2) or compute once per CTA into __shared__.
2. Per pack: load `r` (residual_out), `ax` (attention_x), `gamma_post`,
   `gamma_pre`; `normed = gemma4_bf16_pack_apply_scale_weight(r_pack,
   gamma_post_pack, s1)`; `y = gemma4_bf16_pack_add(ax_pack, normed)`; store
   `y` to `ffn_residual`; store
   `gemma4_bf16_pack_apply_scale_weight(y, gamma_pre_pack, 1.0f)` to `ffn_x`;
   accumulate `y^2` of the rounded y pack into a per-thread float.
3. Block-reduce the per-thread y^2 partials (existing warp_sums pattern,
   shared array) and thread 0 writes `y2[blockIdx.x]` (all CTAs write, 0.0f if
   no packs).
Do NOT write `normed_out` at all in this path.

Consumer side (FFN): `s2` can no longer be a single prestored float. In
`gemma4_ffn_decode_fused_bf16_device` under this flag, compute
`s2 = rsqrtf(sum(y2[0..gridDim.x))/HIDDEN + eps)` ONCE at function entry
(after its internal accum-zero grid.sync, which orders the y2 slot writes)
into a register/shared value, replacing the per-tile `__ldg(pre_ffn_scale)`.
Plumb the y2 slot base pointer + grid size the same way pre_ffn_scale is
plumbed today (extend or replace that parameter).

## Sync accounting

No sync count change (103 and 153 remain), but the two serial CTA0 row passes
become one parallel pass (~5 us -> ~0.2 us per layer). `pre_ffn_scale`'s
single-float store/load disappears in favor of slot sums.

## Numerics

s1/s2 reduction order changes (per-CTA slots, fixed-order sum) - reassociation
only, deterministic across runs. Greedy tokens must match A/B; if they differ,
investigate before benchmarking further.

## Expected gain

~5 us x 48 layers ~= 240 us/step ~= 0.6% on the 3090, stacking with the input
and pre-FFN folds.
