# Gemma 4 RoPE Baseline

This note tracks the first CUDA RoPE implementation for Gemma 4 attention
prep. It is based on the provided Triton kernel shape, but keeps the runtime
entry point small and Gemma-specific.

## Tensor Layout

The CUDA kernel expects Q and K in the same physical layout that the Triton
code creates after `transpose(1, 2).contiguous()`:

```text
q: [batch, seq, q_heads, head_dim]
k: [batch, seq, kv_heads, head_dim]
```

Each CUDA row is one `[batch, seq]` token position. The row stride is measured
in BF16 elements:

```text
q_row_stride = q_heads * head_dim
k_row_stride = kv_heads * head_dim
```

Cosine and sine tables are precomputed FP32 arrays. They may be shared across
the batch:

```text
cos/sin: [1, seq, rotary_dim / 2]
```

or batch-specific:

```text
cos/sin: [batch, seq, rotary_dim / 2]
```

## Rotation Math

The implementation uses the split-half form from the Triton kernel:

```text
x = [x1, x2]
y1 = x1 * cos - x2 * sin
y2 = x2 * cos + x1 * sin
```

For sliding layers:

```text
head_dim = 256
rotary_dim = 256
theta = 10000
```

For global layers:

```text
head_dim = 512
rotary_dim = 128
theta = 1000000
```

Global layers leave dimensions `[128, 512)` unchanged. This is the p-RoPE /
NoPE split described in `gemma4_architecture.md`.

## CUDA Mapping

The baseline CUDA mapping is one block per `(row, head)` pair:

```text
grid.x = batch * seq
grid.y = max(q_heads, kv_heads)
```

Within each block, threads cover `rotary_dim / 2` element pairs. If the head
index exists for Q, the block rotates that Q head. If the same head index also
exists for K, it rotates the K head too.

This is intentionally a correctness-first baseline. Later attention-prep work
can fuse Q/K RMSNorm, RoPE, and KV-cache write so Q/K are read and written once.
