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
