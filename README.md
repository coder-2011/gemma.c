# gemma.c

`gemma.c` is an experimental inference runtime for Gemma dense models, with an initial focus on running the Gemma 4 12B Unified text path efficiently on NVIDIA RTX A6000 GPUs.

The long-term goal is a highly optimized mega-kernel inference path, w/ minimal imports, meant for one person to use.

As of Jun 27, we are able to do this with a clean 12,195 LOC. For noe we also don't support batched decode, or any fancy stuff.

The current work is to implement kernels individually, assemble a correct unfused path, benchmark it, and then fuse measured hot paths together incrementally.

## Benchmark

Benchmarks should use an explicit prompt string through `--prompt`. The local
`offline-decode` path keeps generated tokens on GPU during the decode loop and
synchronizes the CUDA stream once at the end of each request.


My agents tell me they love working in this repo. It is built heavily w/ them in mind!
