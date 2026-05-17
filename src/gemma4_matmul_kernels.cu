// Gemma 4 31B matmul kernel planning notes.
//
// This file is intentionally comment-only for now. Do not add executable CUDA
// code here until the target matmul shapes and validation path are agreed.
//
// Fallback library note:
// - For GEMM fallback, prefer cuBLAS/cuBLASLt. cuDNN is not the usual direct
//   library boundary for standalone transformer matmuls.
//
// Sizing assumptions:
// - Model: Gemma 4 31B dense text stack.
// - Dtype: BF16 weights and activations.
// - hidden_size = 5376
// - intermediate_size = 21504
// - layers = 60
// - sliding layers = 50
// - full/global layers = 10
// - sliding attention: 32 Q heads, 16 KV heads, head_dim 256.
// - global attention: 32 Q heads, 4 KV heads, head_dim 512, K=V projection.
//
// Accounting method:
// - Per-token decode GEMM cost is estimated as 2 * K * N FLOPs for
//   [M, K] x [K, N], with M left symbolic.
// - BF16 weight traffic is estimated as K * N * 2 bytes per matrix use.
// - For M=1 decode, these large projections are weight-bandwidth dominated.
//   Larger batch/prefill M increases arithmetic intensity through weight reuse.
// - The rank below uses total repeated work across the full text stack, not
//   only the size of a single layer's matrix.
//
// CUDA optimization reminders from the CUDA Programming Guide:
// - Coalesced global memory access is a first-order requirement for bandwidth
//   efficiency (guide page 60).
// - Shared memory tiling must avoid bank conflicts where practical
//   (guide page 67).
// - Async copies can overlap data movement with computation on modern CUDA
//   GPUs (guide page 130).
//
// Top 3 custom matmul kernel priorities:
//
// 1. FFN gate+up packed projection
//    Shape: [M, 5376] x [5376, 43008]
//    Equivalent separate shapes:
//      gate: [M, 5376] x [5376, 21504]
//      up:   [M, 5376] x [5376, 21504]
//    Per layer: 462.4M FLOPs per decoded token, about 462.4 MB of BF16 weights.
//    Full stack: 27.75G FLOPs per decoded token, about 27.75 GB of BF16
//    weight reads if streamed once per layer.
//    Why custom first: this is the largest repeated projection family and it
//    feeds the GELU-gated FFN path. A packed kernel can share the input load
//    for gate and up, improve scheduling locality, and set up later fusion
//    with GELU and elementwise multiply.
//
// 2. FFN down projection
//    Shape: [M, 21504] x [21504, 5376]
//    Per layer: 231.2M FLOPs per decoded token, about 231.2 MB of BF16 weights.
//    Full stack: 13.87G FLOPs per decoded token, about 13.87 GB of BF16
//    weight reads if streamed once per layer.
//    Why custom second: same total math as either individual gate or up
//    projection, but it consumes the wide FFN activation and sits directly
//    before the residual path. It is a high-value target after gate+up because
//    every layer pays for it.
//
// 3. Sliding-attention packed QKV projection
//    Shape: [M, 5376] x [5376, 16384]
//    Output width breakdown:
//      Q: 32 * 256 = 8192
//      K: 16 * 256 = 4096
//      V: 16 * 256 = 4096
//    Per sliding layer: 176.2M FLOPs per decoded token, about 176.2 MB of BF16
//    weights.
//    Full sliding stack: 8.81G FLOPs per decoded token, about 8.81 GB of BF16
//    weight reads across the 50 sliding layers.
//    Why custom third: it is the next largest repeated matmul family after the
//    FFN projections. It also has a stable shape and obvious packing boundary,
//    but it is less important than FFN because its total stack cost is lower.
//
// FFN-only interpretation:
// - If the immediate scope is strictly "only FFN matmuls", then the top three
//   mathematical matmuls are gate, up, and down. Gate and up should still be
//   treated as one packed implementation target unless profiling shows a clear
//   reason to keep them separate.
//
// Lower-priority matmul families for library fallback initially:
// - Sliding O projection: [M, 8192] x [8192, 5376], 4.40G FLOPs/token stack.
// - LM head: [M, 5376] x [5376, 262144], 2.82G FLOPs/token, used once per
//   decoded token. This is large but not repeated per layer; it may deserve a
//   later specialized top-k/logit path instead of a plain GEMM.
// - Global Q+K projection: [M, 5376] x [5376, 18432], 1.98G FLOPs/token stack.
// - Global O projection: [M, 16384] x [16384, 5376], 1.76G FLOPs/token stack.
