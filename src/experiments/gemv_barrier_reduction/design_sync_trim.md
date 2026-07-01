# Design: drop the redundant post-attention trailing grid.sync

Flag: `GEMMA4_TRIM_REDUNDANT_SYNC` (default 0). Requires
`GEMMA4_FFN_FOLDED_PRE_NORM=1`.

## Observation

Per layer, `project_attention_out_post_attention` ends with a trailing
grid.sync (src/gemma4_flash_attention_decode.cu:153) that publishes the
epilogue's `ffn_residual` / `ffn_x` / scale writes. Immediately afterwards the
kernel calls `gemma4_ffn_decode_fused_bf16_device`, whose FIRST grid-wide
action is zeroing `scratch->accum` followed by its own grid.sync
(src/gemma4_ffn.cu, top of the fused function). The FFN's internal sync
already orders BOTH the accum zeroing AND every pre-sync global write from the
epilogue: two device-wide barriers back to back with only an elementwise
zeroing loop between them.

The zeroing loop touches only `scratch->accum` (disjoint from the epilogue's
outputs), so removing the trailing sync at `:153` is safe: all reads of
`ffn_residual`/`ffn_x`/scale slots happen after the FFN's internal grid.sync.

Audit before relying on this: confirm nothing between `:153` and the FFN call
in `decode_megakernel_fused_layer_kernel` reads the epilogue outputs (under
the fold flags the CTA0 pre-FFN block is already compiled out), and confirm
the non-fold (#else) FFN handoff branch is NOT affected (flag requires the
fold, so only the fold branch changes).

## Change

Under the flag, make the trailing `grid.sync()` at `:153` conditional
(compile it out). One-line-plus-comment change; the comment must state which
downstream sync assumes the ordering role.

## Expected gain

~1.5 us x 48 layers ~= 70 us/step ~= 0.2% on the 3090. Tokens must be
identical A/B.
