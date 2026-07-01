# Results: decode FFN warp-per-column (GEMMA4_FFN_WARP_TILED)

Date: 2026-07-01. Builds and protocol per README.md; same machine state as the
2026-07-01 baseline profile (GPU0 shared with another job; clean-window reps
reported for Blackwell).

## decode_step_ms p50 (A = baseline, B = warp-tiled v2)

| GPU / seq | A | B | delta |
| --- | ---: | ---: | ---: |
| 3090 / 121 | 33.460 | 33.436 | -0.1% |
| 3090 / 1001 | 37.446 | 37.414 | -0.1% |
| 3090 / 4001 | 40.261 | 40.232 | -0.1% |
| RTX PRO 6000 / 4001 (clean rep) | 19.574 | 20.023 | +2.3% |

v1 (runtime-bound down loop, not unrollable) was +2.8-3.1% on Blackwell across
seq 121-16001; the compile-time 16-row unroll (v2) recovered about half a
percent but the variant stays a small regression on sm_120 and is noise-level
on sm_86.

## NCU, sliding fused layer

| metric | 3090 base | 3090 wt-v1 | BW base | BW wt-v2 |
| --- | ---: | ---: | ---: | ---: |
| duration us | 684.1 | 685.1 | 366.3 | 374.7 |
| DRAM % peak | 77.7 | 77.8 | 75.3 | 75.3 |
| warp cycles/issued inst | 21.8 | 36.6 | 51.2 | 63.0 |
| barrier stall share | 31.7% | 18.8% | 34.9% | ~20% |
| long_scoreboard share | 35.3% | 61.1% | 53.2% | 58%+ |

## Conclusion

The barrier elimination worked mechanically (barrier stall share roughly
halved; 1 block barrier per 16 intermediate columns instead of 2 per 2
columns) but wall time did not improve: DRAM% and duration are unchanged on
the 3090 and slightly worse on Blackwell. The stall mass moved from `barrier`
to `long_scoreboard` on the same weight-load instructions. Interpretation: at
33% occupancy (16 warps/SM, register-limited to 1 CTA/SM) the barrier waits
were already fully overlapped with the DRAM weight stream; the binding
constraint is memory latency hiding, not synchronization. On Blackwell the
16-independent-row streaming pattern is marginally less efficient per byte
than the baseline block-wide single-row sweep.

Follow-ups this motivates, in order:

1. Occupancy: cut fused-layer registers to <= 64/thread to fit 2 CTAs/SM
   (32 warps), which attacks the long_scoreboard wall directly.
2. Bytes: quantized FFN weights; the BF16 stream is 72-78% of DRAM peak and is
   the floor itself.
3. Keep `GEMMA4_FFN_WARP_TILED` default 0. The flag and helper stay on this
   branch as documentation of the negative result.

## Addendum: occupancy and split-size levers (3090 only, same date)

2-CTA/SM variant (`GEMMA4_DECODE_MEGA_MIN_BLOCKS=2` + global `-maxrregcount=64`,
build_sm86_occ2):

| seq | baseline p50 | occ2 p50 |
| ---: | ---: | ---: |
| 121 | 33.461 | 35.212 (+5.2%) |
| 1001 | 37.455 | 38.072 (+1.6%) |
| 4001 | 40.273 | cuda OOM |

NCU sliding layer at seq 1001: grid 164 (2 CTAs/SM), achieved occupancy 66.5%
(2x), DRAM busy 80.1% (up from 77.7) - but duration 708 us vs 684 us. ptxas
spills 21,022 local-memory requests per launch (100% overhead): the extra DRAM
busy is spill traffic, not useful weight bytes. The spill backing store also
inflates context local memory enough to OOM the 24 GB card at 4k prompts.
Conclusion: the register cap costs more than the doubled warp count buys; a
spill-free <=64-reg version needs a redesign of the per-phase accumulator
tiling, not a compiler cap.

decode_split_size sweep at seq 4001 (CLI only): 20 -> 40.264 ms,
32 -> 40.441, 64 -> 40.127, 128 -> 43.114. Flat within 0.4% up to 64; keep 20.

## Overall status on sm_86

Three levers measured, none clears the keep bar: barriers (overlapped anyway),
occupancy via reg cap (spills), split size (flat). The fused layers run at
77.7%/74.7% of DRAM peak vs 97.4% for the phase-free final-logits GEMV on the
same GPU; the remaining gap lives in the fused phase structure (grid syncs and
phase dependency drains), not in the GEMV inner loops.

## Folded pre-FFN norm (GEMMA4_FFN_FOLDED_PRE_NORM), 3090

Design: design_folded_pre_norm.md. v1 multiplied gamma inside the gate/up dot
loop: +1.1-1.3% (deeper per-pack dependency chain in the hot loop). v2 stages
gamma*y in CTA0's existing residual pass and applies only the scalar s2 after
the block reduction, keeping the gate/up stream byte-identical to baseline;
this deletes the separate CTA0 pre-FFN norm phase and one grid.sync per layer.

| seq | baseline p50 | fold v2 p50 | delta |
| ---: | ---: | ---: | ---: |
| 121 | 33.481 | 33.382 | -0.30% |
| 1001 | 37.468 | 37.425 | -0.11% |
| 4001 | 40.289 | 40.189 | -0.25% |

Tokens identical A/B (24-step greedy). min agrees with p50 at all lengths.

Unit economics now measured: one removed CTA0 row phase + one grid.sync per
layer nets ~2 us/layer (~0.25%/step). Extrapolation: distributing the
post-attention norm + residual (syncs at decode.cu:103/128, needs the
3-partial-sum quadratic trick for s2) and the input norm (decode.cu:462-467)
would add roughly 2-4x this, i.e. the full norm/sync elimination program is
worth ~0.5-1% on sm_86, not the multi-percent hoped for. The 20-point DRAM%
gap vs the phase-free logits GEMV is mostly NOT serial-phase overhead; the
untested remainder is chunk-level cross-phase pipelining (consumers starting
on partial rows).
