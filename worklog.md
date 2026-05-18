# Worklog

This file is a work log of how I went about creating the megakernel
Note that I have two work logs. experiments/EXPERMIENTS.md is fully agent maintained, annoying to read, and verbose.
This one is human maintained, and lightly AI formatted, as well as more relevant to read.

- The general approach for my work through the mega kernel is to start by creating an unfused implementation every kernel, but writing ways that is easy to maintain and fuse later. This is largely to get a feel for what the model is like, because Gemma 4 is a unique, feature-dense architecture and A6000 is a GPU I haven't created many kernels for before.


- A design decision I'm taking is writing custom mat model kernels and trying to rely as little on CU DNN as possible such that I can squeeze as much performance as possible. This is really meant to be as optimal and easy to use as possible, and that requires me to work my way through the stack.

- As of now I'm working on custom matmul and GEMV kernels for the fixed projection shapes in the Gemma 4 31B dense path. The current decode GEMV benchmark has been moved to BF16 inputs/outputs with randomized values, so the smoke tests are closer to the actual model path than the old FP16 deterministic pattern. In the latest BF16 smoke run, the custom M=1 kernels are much faster than BF16 cuBLAS for the small/medium decode projections where cuBLAS has heavy overhead or poor layout behavior: roughly 5.7x for packed FFN gate+up, 11.7x for sliding QKV, 19.7x for sliding O, 13.1x for global Q, and 56.9x for global K against the BF16 M=1 GEMM baseline. The large memory-streaming shapes are basically roofline limited instead of algorithmically interesting: FFN down and global O are around parity to ~1.07x, and final logits is around 1.002x. Earlier cuDNN 1x1-conv testing on packed gate+up showed about a 4.9x custom win, but cuDNN is no longer the main comparison for these decode GEMVs; BF16 cuBLAS is the cleaner baseline.
