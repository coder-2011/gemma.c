# TODO

*Agent maintained, quite messy. They never listen to me when I tell them to keep it clean...*

- Create more types of graphs, and gather more data
- use ncu and nsys profiler to gather more information about our speed data
- Add support for batched decode
- Add API access

## Tune every mechanism

- [x] Establish repeatable main decode benchmark baseline.
- [ ] Tune decode orchestration, launch grouping, scratch layout, and graphability.
- [ ] Tune RMSNorm and fused residual + RMSNorm parameters.
- [ ] Tune Q/K/V norm paths and cache hints.
- [ ] Tune RoPE and p-RoPE table access, packing, and launch shape.
- [ ] Tune KV-cache writes, page-table lookup, cache layout, and sliding wraparound.
- [ ] Tune sliding-window attention split size, tile shape, swizzle, cache hints, and buffering.
- [ ] Tune global attention split size, tile shape, swizzle, cache hints, and buffering.
- [ ] Tune FFN decode GEMV, gate/up packing, swizzle params, block sizes, cache hints, and buffering.
- [ ] Tune final RMSNorm, logit softcap, sampling, and embedding gather.
- [ ] Fix benchmarks when timing, warmup, correctness, or reporting issues block tuning.
- [ ] Record before/after decode speedup and experiment notes.
