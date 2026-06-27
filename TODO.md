# TODO

*Agent maintained, quite messy. They never listen to me when I tell them to keep it clean...*

- Rewrite the main megakernel path fully, as it is quite broken right now. Just scrap it, and rewrite. DONE
- Confirm and audit all indiviual path benchmarks, and clean them up. I bet a sophisticated agent loop can do this in one shot, but I am not entirely sure. It might take some more manual work. DONE
- Get rid of the random slop that is in this repo, like the SASS files, and make sure all important things to gitignore and ignored. PARTIALLY DONE.
- We need to run a /goal to tune basic hyperparams for quasi-kernels, I think we can squeeze another 10-20% more efficency just off that alone. DONE
- Normal optimization of quasi-kernels
- It is important to look at the whole path from a more holistic perspective, but still very thoroughly.
- Move all graphs and images to public, and write the blog post in public/BLOG.md
- Generate more graphs off more tests!

## Tune every mechanism

- [x] Establish repeatable main decode benchmark baseline.
- [x] Tune decode orchestration, launch grouping, scratch layout, and graphability.
- [x] Tune RMSNorm and fused residual + RMSNorm parameters.
- [x] Tune Q/K/V norm paths and cache hints.
- [x] Tune RoPE and p-RoPE table access, packing, and launch shape.
- [x] Tune KV-cache writes, page-table lookup, cache layout, and sliding wraparound.
- [x] Tune sliding-window attention split size, tile shape, swizzle, cache hints, and buffering.
- [x] Tune global attention split size, tile shape, swizzle, cache hints, and buffering.
- [x] Tune FFN decode GEMV, gate/up packing, swizzle params, block sizes, cache hints, and buffering.
- [x] Tune final RMSNorm, logit softcap, sampling, and embedding gather.
- [x] Fix benchmarks when timing, warmup, correctness, or reporting issues block tuning.
- [x] Record before/after decode speedup and experiment notes.
