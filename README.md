# gemma.c

`gemma.c` is an experimental inference runtime for Gemma dense models, with an initial focus on running the Gemma 4 31B dense model efficiently on NVIDIA RTX A6000 GPUs.

The long-term goal is a highly optimized mega-kernel inference path: start from straightforward, correct kernels, build a fast unfused implementation, then progressively fuse the hot path into a minimal set of specialized kernels.

Development guidance for agents lives in `AGENTS.md`.
