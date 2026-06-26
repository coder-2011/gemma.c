# Worklog

**This file is a work log of how I went about creating the megakernel
Note that I have two work logs. experiments/EXPERMIENTS.md is fully agent maintained, annoying to read, and verbose.
This one is human maintained, and lightly AI formatted, as well as more relevant to read.**

- The general approach for my work through the mega kernel is to start by creating an unfused implementation every kernel, but writing ways that is easy to maintain and fuse later. This is largely to get a feel for what the model is like, because Gemma 4 is a unique, feature-dense architecture and A6000 is a GPU I haven't created many kernels for before.


- A design decision I'm taking is writing custom mat model kernels and trying to rely as little on CU DNN as possible such that I can squeeze as much performance as possible. This is really meant to be as optimal and easy to use as possible, and that requires me to work my way through the stack.

- As of now I'm working on custom matmul and GEMV kernels for the fixed projection shapes in the Gemma 4 31B dense path. 

- Matmul kernels have been finished, code quality has to be improved, and I think there's a couple of issues with cache hints. Unfortunately, I'm not running on bare metal GPUs, so I cannot use _nsys_, but hopefully if I get my hands on some baremetal A6000s, I can run nsys and check for issues. All the bells and whistles are implemented.

- For the RMS-norm, I pretty much ported llm.c's RMSNorm, and worked off that. it was really cleanly implemented so there is no need to rewrite it. Ofc, adjustments have to be made since we use RMS instead of LayerNorm.

- Back to matmul, playing around w/ warptiling to torture the chip into giving me all its FLOPs. Benchmarks had to be fixed a bit too, bc it didnt fully account for overhead from cuDNN. I used cuDNN backend API anyways to minimize ovehead. Tried double buffering, but this is GEMV decode: it mostly streams weights once, so mem constraints will not let us speed up much more than this.

- Buffering and fancy memory tricks have mostly not helped the M=1 decode GEMV path. Direct 128-bit pack loads stream `VRAM -> registers -> FMA`; `cp.async` double buffering changes that to `VRAM -> SMEM -> registers -> FMA`, adding shared-memory traffic plus commit/wait overhead without reducing bytes or creating reuse. If this is revisited, use a different tiling strategy where staged data is reused; don't just pipeline one-use 16-byte packs.

-cleaned up everything for readability

- exapted a boilerplate RoPE kernel. Considering how I an optimize it.

- Benchmarked RoPE vs cuDNN pointwise decomposition. Custom kernel wins by a lot, roughly 2.4-3x at seq 4096 even after graph capture. Not worth using cuDNN here; next move is fuse Q/K norm + RoPE + KV write.

- RMSNorm started as a direct llm.c-style baseline, then got split into learned-weight and scale-free V paths. Rebenchmarked real shapes w/ cuDNN setup overhead factored out. Custom V RMSNorm is worth keeping: width 256 is ~1.05-1.51x faster, width 512 is ~1.15-1.65x faster. Hidden width 5376 only wins decode row=1; cuDNN beats prefill because our hidden prefill path is still a simple one-warp-per-row baseline. Fix that later by fusion or a real multi-warp row kernel.

- Fixed the hidden RMSNorm prefill weakness by making fused residual add + RMSNorm use one block per row, then keeping the residual pack in registers instead of shared memory. This finally beats cuDNN split on real hidden shapes, about 1.16-1.99x for prefill rows 4-1024, so fused RMSNorm is now worth keeping for both decode and prefill.

- Plucked some low hanging fruit: SMEM buffering the data arriving from DRAM, because we load _x_ twice–once for sum of squares, and once for multiplying by _γ_. After a bit of tuning, aggregate best time across rows 4-1024 moved from 0.090220ms to 0.089864ms, about 0.39% faster; final best graph replay for rows 4/16/64/256/1024 was 0.002104/0.002157/0.002839/0.017169/0.065835ms.

- Cleaned up the RMSNorm file after that tuning pass. Removed dead scale-free fused residual paths and most of the tiny RMSNorm-local wrapper helpers, then made the remaining block reduction helper take explicit thread/lane/warp state plus shared scratch storage. Tests still pass, ptxas reports no stack/spills, and the focused hidden fused smoke benchmark stayed in the same range: best graph replay for rows 4/16/64/256/1024 was 0.002104/0.002157/0.002839/0.017169/0.065835ms.

- Removed the next batch of dead RMSNorm code: fallback warp kernels, generic fused residual+RMSNorm for non-hidden widths, the `cudaFuncSetAttribute` fallback routing, and the null-weight shortcut in learned RMSNorm. Fused residual+RMSNorm is now hidden-width-only, which is the actual residual stream case. The file is down to 6 kernels, tests pass, ptxas still reports no spills, and the focused hidden fused best graph replay for rows 4/16/64/256/1024 was 0.002080/0.002133/0.002841/0.017191/0.065856ms.

- I have discovered that the line is much finer between slop and useful software w/ CUDA. You have to really tighten the agency you give agents compared to programming at a higher level. Kinda wasted time due to this. 

-Micro-optimizations on RoPE: Ofc there is not much point to it, bc RoPE is so extremely cheap anyways, but it was fun, so I did it. Added packed 128-bit Q/K loads/stores, compact sin/cos tables, cache hints, one-warp tiling, and verified FP32 FMA consistency.

-Flash attn has been implemented. Matches Flash Attention pretty much perfectly. There's a couple of optimizations I can pull from FA3 and FA4 that are not architecture specific. Mainly:
- the software emulation of the exponential function
- conditional online softmax rescaling.

-Fused FFN decode using the custom matmul gemv kernels. implementation was clean, outperforming cuDNN by ~15 percent (factoring out overhead). Agents make alot of silly mistakes still.  I plan on experimenting with different agent harnesses for optimizing kernels. [Kernel Design Agents]([url](https://github.com/mit-han-lab/kernel-design-agents)) seems promising.

-GEMM+epilogue is pretty much what I am doing. the tri dao paper reperameterized it w/ a neat mathematical framework accounting for a whole transformer layer. It is a really impressive feat of engineering to fuse the entire transformer block without touching HBM. my ffn-decode kernel is there rn, but I have to fuse it w/ flash attn.

-  I feel like I'm about 60% there, but I've felt this way so many • in so many projects, and I'm always like 20% there (fingers crossed) this time is different.
