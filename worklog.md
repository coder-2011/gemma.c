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

- Rebenchmarked RMSNorm on real shapes, making sure cuDNN setup/plan/build overhead is not in the measured numbers. The comparison uses CUDA graph replay columns only. For the actual new target, scale-free V RMSNorm, custom is better: width 256 is about 1.05-1.51x faster than cuDNN one-scale, and width 512 is about 1.15-1.65x faster. So yes, keep the custom V RMSNorm path. For hidden width 5376, custom only wins decode row=1; cuDNN is better for prefill rows 4-1024. Fused residual+RMSNorm mostly wins too, but there are a couple of row-counts where split CUDA is basically tied/slightly better, so don't overclaim that one.
