# Worklog

This file is a work log of how I went about creating the megakernel
Note that I have two work logs. experiments/EXPERMIENTS.md is fully agent maintained, annoying to read, and verbose.
This one is human maintained, and lightly AI formatted, as well as more relevant to read.

- The general approach for my work through the mega kernel is to start by creating an unfused implementation every kernel, but writing ways that is easy to maintain and fuse later. This is largely to get a feel for what the model is like, because Gemma 4 is a unique, feature-dense architecture and A6000 is a GPU I haven't created many kernels for before.


- A design decision I'm taking is writing custom mat model kernels and trying to rely as little on CU DNN as possible such that I can squeeze as much performance as possible. This is really meant to be as optimal and easy to use as possible, and that requires me to work my way through the stack.

- As of now I'm working on custom matmul and GEMV kernels for the fixed projection shapes in the Gemma 4 31B dense path. 

- Matmul kernels have been finished, code quality has to be improved, and I think there's a couple of issues with cache hints. Unfortunately, I'm not running on bare metal GPUs, so I cannot use _nsys_, but hopefully if I get my hands on some baremetal A6000s, I can run nsys and check for issues. All the bells and whistles are implemented.

- For the RMS-norm, I pretty much ported llm.c's RMSNorm, and worked off that. it was really cleanly implemented so there is no need to rewrite it. Ofc, adjustments have to be made since we use RMS instead of LayerNorm.

- Back to matmul, playing around w/ warptiling to torture the chip into giving me all its FLOPs. Benchmarks had to be fixed a bit too, bc it didnt fully account for overhead from cuDNN. I used cuDNN backend API anyways to minimize ovehead. Tried double buffering, but this is GEMV decode: it mostly streams weights once, so mem constraints will not let us speed up much more than this.

- Buffering and fancy memory tricks have mostly not helped the M=1 decode GEMV path. The direct `load128cs` path streams `VRAM -> registers -> FMA`; `cp.async` double buffering changes that to `VRAM -> SMEM -> registers -> FMA`, adding shared-memory traffic plus commit/wait overhead without reducing bytes or creating reuse. If this is revisited, use a different tiling strategy where staged data is reused; don't just pipeline one-use 16-byte packs.

-cleaned up everything for readability

- exapted a boilerplate RoPE kernel. Considering how I an optimize it.

- Benchmarked RoPE vs cuDNN pointwise decomposition. Custom kernel wins by a lot, roughly 2.4-3x at seq 4096 even after graph capture. Not worth using cuDNN here; next move is fuse Q/K norm + RoPE + KV write.

- RMSNorm started as a pretty direct llm.c-style baseline, then got split into the normal learned-weight path and the scale-free V path. Rebenchmarked on real shapes with cuDNN setup/plan/build overhead factored out, using CUDA graph replay columns only. The useful result is: custom scale-free V RMSNorm is worth keeping. Width 256 is about 1.05-1.51x faster than cuDNN one-scale, and width 512 is about 1.15-1.65x faster. Hidden width 5376 is different: custom wins decode row=1, but cuDNN beats us for prefill rows 4-1024. I think that is because our prefill hidden RMSNorm is still a simple one-warp-per-row baseline that loops over 672 bf16x8 packs, stages input in smem, and reloads/caches gamma per tiny row group. cuDNN is just a better batched wide-row norm implementation there. So: keep custom V RMSNorm, keep decode hidden RMSNorm, don't pretend the hidden prefill RMSNorm baseline is better than cuDNN until we fuse it or write a more serious multi-warp row kernel.
