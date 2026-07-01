# GEMV barrier reduction (decode FFN warp-per-column)

Branch: `exp/gemv-barrier-reduction`

## Problem

NCU PC sampling (2026-07-01 profile, see EXPERIMENTS.md) shows the fused decode
layer kernels spend 32-35% of stall cycles at CTA barriers on both the RTX 3090
and the RTX PRO 6000. The hot barriers are the block-wide reductions that close
each FFN gate/up tile (`gemma4_matmul_kernels.cu` reduce + `gemma4_ffn.cu`
activation broadcast). The baseline decode FFN processes 2 intermediate columns
per tile with two `__syncthreads()` per tile, i.e. one barrier per ~23 KB of
streamed weights, and the down-projection stream cannot start until the gate/up
reduction chain finishes, so the memory pipe drains at every tile boundary.

## Approach

`GEMMA4_FFN_WARP_TILED=1` switches `gemma4_ffn_decode_fused_bf16_device` to:

- one interleaved gate/up column pair per warp
  (`gemma4_ffn_gate_up_warp_col_bf16_device`), reduced with warp shuffles only,
  no shared memory and no block barrier in the reduction;
- a super tile of `blockDim/32` columns (16 in the fused megakernel, using the
  previously idle 16th warp) with a single `__syncthreads()` per super tile;
- double-buffered `s_act` staging so the write-after-read hazard between
  consecutive super tiles needs no second barrier.

Barrier count per layer FFN drops from 2 per 2 columns to 1 per 16 columns
(16x). Weight-load coalescing is preserved: the gate/up swizzle permutes packs
only inside 8-pack (128 B) blocks, so lane-consecutive packs still cover whole
128 B lines. Reduction reassociation changes numerics slightly (bf16 inputs,
fp32 accumulation) versus the two-level block tree.

## Measurement protocol

Same as the 2026-07-01 baseline profile: decode-step benchmark p50 over 30
steps at prompt lengths 121/1001/4001(/16001 Blackwell only), plus NCU
sliding-layer capture for DRAM%, stall mix.

Results: see `results.md` in this folder and the dated EXPERIMENTS.md entry.
