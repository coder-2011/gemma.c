# Gemma 4 Architecture: Technical Reference

> **Current as of:** June 20, 2026
> **Primary implementation target for this repository:** dense text inference on RTX A6000-class Ampere GPUs
> **Current practical target models:** Gemma 4 12B Unified for BF16 bring-up, Gemma 4 31B Dense for quantized single-GPU work, and Gemma 4 26B-A4B for later MoE support.
> **Out of scope for this document:** E2B and E4B implementation details. They are part of the official Gemma 4 family, but this project is not targeting E-series execution.

---

## Table of Contents

1. [Source Status](#1-source-status)
2. [Family Overview](#2-family-overview)
3. [Repo-Relevant Model Set](#3-repo-relevant-model-set)
4. [Shared Text Architecture](#4-shared-text-architecture)
5. [Hybrid Attention](#5-hybrid-attention)
6. [RoPE and p-RoPE](#6-rope-and-p-rope)
7. [Gemma 4 12B Unified](#7-gemma-4-12b-unified)
8. [Gemma 4 31B Dense](#8-gemma-4-31b-dense)
9. [Gemma 4 26B-A4B MoE](#9-gemma-4-26b-a4b-moe)
10. [Unified Multimodal Path in 12B](#10-unified-multimodal-path-in-12b)
11. [Standard Multimodal Path in 31B and 26B](#11-standard-multimodal-path-in-31b-and-26b)
12. [Tokenizer and Chat Format](#12-tokenizer-and-chat-format)
13. [Hyperparameter Tables](#13-hyperparameter-tables)
14. [Parameter Accounting](#14-parameter-accounting)
15. [KV Cache and Memory Planning](#15-kv-cache-and-memory-planning)
16. [A6000 Implications](#16-a6000-implications)
17. [Benchmark Snapshot](#17-benchmark-snapshot)
18. [Implementation Notes for this Repo](#18-implementation-notes-for-this-repo)
19. [Sources](#19-sources)

---

## 1. Source Status

Gemma 4 changed after this document was first written. The original version predated the June 2026 release of **Gemma 4 12B Unified**, so any architecture notes that say "four models" or omit 12B are stale.

This update is based on:

- Official Google Gemma 4 model overview and model card.
- Official Google 12B launch post and 12B developer guide.
- Hugging Face model repositories and downloaded metadata under `/tmp/gemma4_hf_metadata`.
- Hugging Face Transformers `gemma4` and `gemma4_unified` configuration/modeling code.
- Exa search results used to cross-check the official pages and locate the 12B-specific docs.

The downloaded Hugging Face metadata includes:

```text
/tmp/gemma4_hf_metadata/gemma-4-12B/config.json
/tmp/gemma4_hf_metadata/gemma-4-12B/README.md
/tmp/gemma4_hf_metadata/gemma-4-12B/processor_config.json
/tmp/gemma4_hf_metadata/gemma-4-12B/generation_config.json
/tmp/gemma4_hf_metadata/gemma-4-12B/tokenizer_config.json
/tmp/gemma4_hf_metadata/gemma-4-31B-it/config.json
/tmp/gemma4_hf_metadata/gemma-4-26B-A4B/config.json
/tmp/gemma4_hf_metadata/modeling_gemma4.py
/tmp/gemma4_hf_metadata/modeling_gemma4_unified.py
```

The model weight file was intentionally not downloaded. The 12B tree contains a roughly 23.9 GB `model.safetensors`; it is not needed for architecture metadata.

---

## 2. Family Overview

Gemma 4 is the fourth generation of Google's open-weight Gemma family. The official current lineup includes five parameter sizes, but this repo is focused on the three non-E-series models:

| Model | Official architecture bucket | Parameters | Context | Modalities |
|---|---:|---:|---:|---|
| Gemma 4 12B Unified | Dense unified, encoder-free multimodal | 11.95B | 256K | Text, Image, Audio |
| Gemma 4 26B-A4B | Mixture-of-Experts | 25.2B total, 3.8B active | 256K | Text, Image |
| Gemma 4 31B Dense | Dense | 30.7B text scale | 256K | Text, Image |

The official family also includes E2B and E4B edge models. They are intentionally out of scope here.

Naming:

- **12B Unified** means the model removes the separate multimodal encoders and projects image/audio inputs directly into the language-model embedding space.
- **31B Dense** means every token runs the full dense transformer stack.
- **26B-A4B** means about 26B total parameters are loaded, but only about 4B are active for each token because MoE routing selects a sparse subset.

---

## 3. Repo-Relevant Model Set

### Recommended Bring-Up Order

1. **Gemma 4 12B Unified, text path only**
   - Same dense text recipe as 31B at smaller dimensions.
   - BF16 weights fit on one RTX A6000 with meaningful KV room.
   - Best target for proving the unfused baseline and decode orchestration.

2. **Gemma 4 31B Dense, quantized weights**
   - The original project target.
   - BF16 weights do not fit on one 48 GB A6000.
   - Q4/weight-only quantization is needed for single-GPU 31B.

3. **Gemma 4 26B-A4B, after dense path is stable**
   - Attractive for speed because only 3.8B parameters are active per token.
   - Requires MoE routing, expert layout, and sparse expert execution.
   - Not a small delta from the dense path.

### Why 12B Matters for A6000

Official memory guidance lists Gemma 4 12B at about **26.7 GB BF16**, **13.4 GB SFP8**, and **6.7 GB Q4_0** before context/KV cache. The RTX A6000 has 48 GB, so 12B BF16 is realistic. By contrast, 31B BF16 is listed at about **69.9 GB** with loading overhead, so it cannot fit on a single A6000 in full precision.

12B is not "the same model but fewer layers." It preserves the same important text architecture ideas:

- local/global hybrid attention,
- wider global attention heads,
- global K=V,
- p-RoPE on global layers,
- GeGLU MLPs,
- RMSNorm,
- logit soft-capping,
- tied embeddings,
- 256K context.

But its dimensions are different enough that this repository's hardcoded 31B constants must become model-config driven before 12B and 31B can share code cleanly.

---

## 4. Shared Text Architecture

The repo-relevant Gemma 4 text models are causal decoder-only transformers.

Shared text features:

| Feature | Value / behavior |
|---|---|
| Transformer type | Decoder-only, autoregressive |
| Attention pattern | Interleaved sliding-window and full/global attention |
| Final layer | Always full/global attention |
| Sliding RoPE | Standard RoPE, `theta=10000` |
| Global RoPE | Proportional RoPE, `theta=1000000`, `partial_rotary_factor=0.25` |
| Sliding head dim | 256 |
| Global head dim | 512 |
| Attention bias | false |
| Attention dropout | 0.0 |
| Q norm | Learned RMSNorm per head |
| K norm | Learned RMSNorm per head |
| V norm | RMSNorm without learned scale |
| MLP | GeGLU with `gelu_pytorch_tanh` |
| Decoder norms | input, post-attention, pre-FFN, post-FFN RMSNorm |
| Final norm | RMSNorm |
| Logit softcap | `tanh(logits / 30.0) * 30.0` |
| Weight tying | Input embedding and LM head tied |
| Vocabulary | 262,144 tokens |
| Default dtype | BF16 |
| RMSNorm epsilon | `1e-6` |
| Initializer range | `0.02` |

The text decoder uses four RMSNorms per block:

```text
x
  -> input_layernorm
  -> attention
  -> post_attention_layernorm
  -> residual add
  -> pre_feedforward_layernorm
  -> GeGLU MLP
  -> post_feedforward_layernorm
  -> residual add
```

Gemma 4's residual/norm order is important for kernel planning. The memory-bound operations are the residual adds, RMSNorms, Q/K/V normalizations, logit softcap, and sampling. GEMMs should stay in cuBLAS/cuBLASLt until correctness and profiling justify replacing them.

---

## 5. Hybrid Attention

Gemma 4 does not use one attention shape everywhere. It alternates local/sliding layers and full/global layers.

### Layer Pattern

For the repo-relevant models:

| Model | Layers | Sliding layers | Global layers | Pattern |
|---|---:|---:|---:|---|
| 12B Unified | 48 | 40 | 8 | five sliding, one global |
| 31B Dense | 60 | 50 | 10 | five sliding, one global |
| 26B-A4B | 30 | 25 | 5 | five sliding, one global |

The layer list repeats:

```text
sliding, sliding, sliding, sliding, sliding, full
```

and ends on a full/global layer.

### Sliding Attention

Sliding attention is local causal attention.

| Property | 12B | 31B | 26B-A4B |
|---|---:|---:|---:|
| Window | 1024 | 1024 | 1024 |
| Head dim | 256 | 256 | 256 |
| Q heads | 16 | 32 | 16 |
| KV heads | 8 | 16 | 8 |
| Q width | 4096 | 8192 | 4096 |
| K width | 2048 | 4096 | 2048 |
| V width | 2048 | 4096 | 2048 |
| GQA ratio | 2 Q heads / KV head | 2 Q heads / KV head | 2 Q heads / KV head |

Sliding attention only needs the most recent 1024 tokens in the decode KV cache for each sliding layer. For long-context memory estimates, do not multiply sliding layers by full context length unless modeling a framework that stores unnecessary history.

### Full / Global Attention

Global attention is full causal attention over the sequence.

| Property | 12B | 31B | 26B-A4B |
|---|---:|---:|---:|
| Window | full context | full context | full context |
| Head dim | 512 | 512 | 512 |
| Q heads | 16 | 32 | 16 |
| Global KV heads | 1 | 4 | 2 |
| Q width | 8192 | 16384 | 8192 |
| K width | 512 | 2048 | 1024 |
| V projection | none; V derived from K | none; V derived from K | none; V derived from K |
| GQA ratio | 16 Q heads / KV head | 8 Q heads / KV head | 8 Q heads / KV head |

The Hugging Face implementation sets `use_alternative_attention = attention_k_eq_v and not is_sliding`. In other words, K=V projection sharing applies to global layers in these configs, not to sliding layers.

### K=V Behavior

In global layers:

```text
k_raw = x @ W_k
v_raw = k_raw
k = RMSNorm(k_raw, learned_scale)
v = RMSNorm(v_raw, no_learned_scale)
```

This saves one global V projection matrix and reduces global KV width. It is especially important for the 12B model, where each global layer has only one KV head.

---

## 6. RoPE and p-RoPE

Gemma 4 uses separate RoPE settings for sliding and global layers.

| Layer type | `rope_theta` | `rope_type` | Rotated fraction | Rotated dims |
|---|---:|---|---:|---:|
| Sliding | 10,000 | default | 100% | 256 of 256 |
| Global | 1,000,000 | proportional | 25% | 128 of 512 |

The global attention head is 512 dimensions, but only the first 25% is position-rotated. The remaining 384 dimensions are effectively non-rotary semantic channels. This is the p-RoPE tradeoff: keep enough position signal for long-range ordering while preserving a large position-independent subspace.

Implementation implications:

- RoPE code must accept `rotary_dim` that differs from `head_dim`.
- Global layers use `rotary_dim = 128`, not 512.
- Sliding layers use `rotary_dim = 256`.
- The RoPE cache must be keyed by layer type because theta and rotary dimension differ.
- Q and K receive RoPE after Q/K RMSNorm.
- V does not receive RoPE.

For this repo's CUDA kernels, treat rotary application as a separate baseline kernel first, then fuse with Q/K RMSNorm and KV write only after numerical checks pass.

---

## 7. Gemma 4 12B Unified

Gemma 4 12B Unified is the most important new target for this repository. It is dense, fits on an A6000 in BF16, and shares the server-class text design of 31B without requiring quantization for initial bring-up.

### Core Text Config

Source: `google/gemma-4-12B/config.json`.

| Parameter | Value |
|---|---:|
| `architectures` | `Gemma4UnifiedForConditionalGeneration` |
| `model_type` | `gemma4_unified` |
| `text_config.model_type` | `gemma4_unified_text` |
| `dtype` | `bfloat16` |
| `hidden_size` | 3840 |
| `num_hidden_layers` | 48 |
| Sliding layers | 40 |
| Global layers | 8 |
| `num_attention_heads` | 16 |
| `num_key_value_heads` | 8 |
| `num_global_key_value_heads` | 1 |
| `head_dim` | 256 |
| `global_head_dim` | 512 |
| `intermediate_size` | 15360 |
| `sliding_window` | 1024 |
| `max_position_embeddings` | 262144 |
| `vocab_size` | 262144 |
| `attention_k_eq_v` | true |
| `hidden_size_per_layer_input` | 0 |
| `num_kv_shared_layers` | 0 |
| `use_double_wide_mlp` | false |
| `final_logit_softcapping` | 30.0 |
| `tie_word_embeddings` | true |

### Projection Shapes

For text-only inference:

| Operation | Shape |
|---|---|
| Sliding Q | `[M, 3840] x [3840, 4096]` |
| Sliding K | `[M, 3840] x [3840, 2048]` |
| Sliding V | `[M, 3840] x [3840, 2048]` |
| Sliding O | `[M, 4096] x [4096, 3840]` |
| Global Q | `[M, 3840] x [3840, 8192]` |
| Global K | `[M, 3840] x [3840, 512]` |
| Global V | no matrix; derived from K |
| Global O | `[M, 8192] x [8192, 3840]` |
| FFN gate | `[M, 3840] x [3840, 15360]` |
| FFN up | `[M, 3840] x [3840, 15360]` |
| FFN down | `[M, 15360] x [15360, 3840]` |
| LM head | `[M, 3840] x [3840, 262144]` |

### Per-Block Parameter Counts

Approximate text-only counts:

| Component | Sliding block | Global block | Notes |
|---|---:|---:|---|
| Q projection | 15.7M | 31.5M | global head dim is 512 |
| K projection | 7.9M | 2.0M | one global KV head |
| V projection | 7.9M | 0 | global K=V |
| O projection | 15.7M | 31.5M | output width follows Q width |
| Attention total | 47.2M | 64.9M | excludes norm weights |
| GeGLU FFN | 176.9M | 176.9M | `3 * 3840 * 15360` |
| Block total | 224.1M | 241.8M | approximate |

Text parameter estimate:

```text
Sliding blocks: 40 * 224.1M =  8.97B
Global blocks:   8 * 241.8M =  1.93B
Token embedding: 262144 * 3840 = 1.01B
Text total:                    = 11.91B
Official listed scale:          11.95B
```

The small difference is consistent with normalization weights and the lightweight unified multimodal components.

### Why 12B Is Not Just "31B Smaller"

12B keeps the same server-class layer pattern, but changes the compression point:

- It has half as many Q heads as 31B.
- It has only one global KV head.
- Its global K/V cache grows at 16 KB per token across all global layers in BF16.
- Its hidden size is 3840, so all dense GEMMs are meaningfully smaller.
- It removes the separate vision/audio towers entirely.

For this repo, that means 12B can be used to develop the same baseline kernels at smaller dimensions before scaling to 31B.

---

## 8. Gemma 4 31B Dense

Gemma 4 31B Dense remains the original long-term target for a specialized inference engine. It is the largest dense open Gemma 4 model and has the highest benchmark scores in the family, but it does not fit on a single A6000 in BF16.

### Core Text Config

Source: `google/gemma-4-31B-it/config.json`.

| Parameter | Value |
|---|---:|
| `architectures` | `Gemma4ForConditionalGeneration` |
| `model_type` | `gemma4` |
| `text_config.model_type` | `gemma4_text` |
| `dtype` | `bfloat16` |
| `hidden_size` | 5376 |
| `num_hidden_layers` | 60 |
| Sliding layers | 50 |
| Global layers | 10 |
| `num_attention_heads` | 32 |
| `num_key_value_heads` | 16 |
| `num_global_key_value_heads` | 4 |
| `head_dim` | 256 |
| `global_head_dim` | 512 |
| `intermediate_size` | 21504 |
| `sliding_window` | 1024 |
| `max_position_embeddings` | 262144 |
| `vocab_size` | 262144 |
| `attention_k_eq_v` | true |
| `hidden_size_per_layer_input` | 0 |
| `num_kv_shared_layers` | 0 |
| `use_double_wide_mlp` | false |
| `final_logit_softcapping` | 30.0 |
| `tie_word_embeddings` | true |

### Projection Shapes

| Operation | Shape |
|---|---|
| Sliding Q | `[M, 5376] x [5376, 8192]` |
| Sliding K | `[M, 5376] x [5376, 4096]` |
| Sliding V | `[M, 5376] x [5376, 4096]` |
| Sliding O | `[M, 8192] x [8192, 5376]` |
| Global Q | `[M, 5376] x [5376, 16384]` |
| Global K | `[M, 5376] x [5376, 2048]` |
| Global V | no matrix; derived from K |
| Global O | `[M, 16384] x [16384, 5376]` |
| FFN gate | `[M, 5376] x [5376, 21504]` |
| FFN up | `[M, 5376] x [5376, 21504]` |
| FFN down | `[M, 21504] x [21504, 5376]` |
| LM head | `[M, 5376] x [5376, 262144]` |

### Per-Block Parameter Counts

| Component | Sliding block | Global block | Notes |
|---|---:|---:|---|
| Q projection | 44.0M | 88.1M | global head dim is 512 |
| K projection | 22.0M | 11.0M | four global KV heads |
| V projection | 22.0M | 0 | global K=V |
| O projection | 44.0M | 88.1M | output width follows Q width |
| Attention total | 132.1M | 187.2M | excludes norm weights |
| GeGLU FFN | 346.8M | 346.8M | `3 * 5376 * 21504` |
| Block total | 478.9M | 534.0M | approximate |

Text parameter estimate:

```text
Sliding blocks: 50 * 478.9M = 23.95B
Global blocks:  10 * 534.0M =  5.34B
Token embedding: 262144 * 5376 = 1.41B
Text total:                    = 30.70B
```

The model also has a standard Gemma 4 vision tower with about 550M parameters, but the current implementation plan should not block the text path on multimodal support.

---

## 9. Gemma 4 26B-A4B MoE

Gemma 4 26B-A4B is not a dense model. It has about 25.2B total parameters but activates about 3.8B per token. It is attractive for throughput, but it requires a different FFN execution path.

### Core Text Config

Source: `google/gemma-4-26B-A4B/config.json`.

| Parameter | Value |
|---|---:|
| `architectures` | `Gemma4ForConditionalGeneration` |
| `model_type` | `gemma4` |
| `text_config.model_type` | `gemma4_text` |
| `dtype` | `bfloat16` |
| `hidden_size` | 2816 |
| `num_hidden_layers` | 30 |
| Sliding layers | 25 |
| Global layers | 5 |
| `num_attention_heads` | 16 |
| `num_key_value_heads` | 8 |
| `num_global_key_value_heads` | 2 |
| `head_dim` | 256 |
| `global_head_dim` | 512 |
| Dense/shared `intermediate_size` | 2112 |
| Expert `expert_intermediate_size` | 704 |
| `num_experts` | 128 |
| `top_k_experts` | 8 |
| `sliding_window` | 1024 |
| `max_position_embeddings` | 262144 |
| `vocab_size` | 262144 |
| `attention_k_eq_v` | true |
| `enable_moe_block` | true |
| `final_logit_softcapping` | 30.0 |
| `tie_word_embeddings` | true |

### Projection Shapes

| Operation | Shape |
|---|---|
| Sliding Q | `[M, 2816] x [2816, 4096]` |
| Sliding K | `[M, 2816] x [2816, 2048]` |
| Sliding V | `[M, 2816] x [2816, 2048]` |
| Sliding O | `[M, 4096] x [4096, 2816]` |
| Global Q | `[M, 2816] x [2816, 8192]` |
| Global K | `[M, 2816] x [2816, 1024]` |
| Global V | no matrix; derived from K |
| Global O | `[M, 8192] x [8192, 2816]` |
| Dense/shared FFN gate/up/down | `2816 <-> 2112` GeGLU |
| Expert FFN gate/up/down | `2816 <-> 704` GeGLU per expert |
| Router | `[M, 2816] x [2816, 128]` |

### MoE Structure

The model card describes 128 total experts, 8 active experts per token, and one shared path. In config terms:

- `num_experts = 128`
- `top_k_experts = 8`
- `expert_intermediate_size = 704`
- `intermediate_size = 2112`

The practical interpretation for implementation:

```text
dense/shared = GeGLU(x; hidden=2112)
router_logits = x @ W_router
selected = top_k(router_logits, 8)
expert_sum = weighted_sum(GeGLU_expert_i(x; hidden=704) for i in selected)
ffn_out = combine(dense/shared, expert_sum)
```

This is not a simple replacement of the dense FFN. The shared dense path remains always active, while the sparse experts add conditional capacity.

### Per-Block Parameter Counts

| Component | Sliding block | Global block | Notes |
|---|---:|---:|---|
| Attention total | 34.6M | 49.0M | global has wider Q/O, smaller K |
| Dense/shared GeGLU | 17.8M | 17.8M | `3 * 2816 * 2112` |
| All experts | 761.3M | 761.3M | `128 * 3 * 2816 * 704` |
| Active experts/token | 47.6M | 47.6M | `8 * 3 * 2816 * 704` |
| Router | 0.36M | 0.36M | `2816 * 128` |
| Capacity/block | 814.1M | 828.5M | loaded parameters |
| Active/block | 100.4M | 114.8M | per-token compute scale |

Parameter estimate:

```text
Capacity:
  Sliding blocks: 25 * 814.1M = 20.35B
  Global blocks:   5 * 828.5M =  4.14B
  Token embedding:              0.74B
  Total capacity:              25.23B

Active per token:
  Sliding blocks: 25 * 100.4M = 2.51B
  Global blocks:   5 * 114.8M = 0.57B
  Token embedding:             0.74B
  Active scale:                3.82B
```

Implementation consequence: the model's memory footprint behaves like a 25B model, but the FFN compute per token behaves closer to a 4B active model plus routing overhead.

---

## 10. Unified Multimodal Path in 12B

12B Unified is architecturally different from 31B and 26B in the multimodal path. It has no separate vision tower and no separate audio tower.

The Transformers implementation describes the unified model as:

```text
Vision: raw patches -> LayerNorm -> Dense -> LayerNorm
        -> factorized XY positional embedding -> LayerNorm
        -> RMSNorm -> Linear into text hidden space

Audio: raw waveform frames -> RMSNorm -> Linear into text hidden space
```

### 12B Vision Embedder

Source: `gemma4_unified` config and modeling code.

| Field | Value |
|---|---:|
| `vision_config.model_type` | `gemma4_unified_vision` |
| `patch_size` | 16 |
| `pooling_kernel_size` | 3 |
| `model_patch_size` | 48 |
| Raw merged patch width | `48 * 48 * 3 = 6912` |
| `mm_embed_dim` | 3840 |
| `mm_posemb_size` | 1120 |
| `num_soft_tokens` | 280 |
| `output_proj_dims` | 3840 |

The patch embedding core is:

```text
LayerNorm(6912)
Linear(6912 -> 3840)
LayerNorm(3840)
add factorized positional embedding with shape [1120, 2, 3840]
LayerNorm(3840)
RMSNorm(3840, no learned scale in multimodal embedder)
Linear(3840 -> text_hidden_size)
```

The main dense patch projection alone is about **26.5M** parameters. The factorized XY positional table is about **8.6M** parameters. This matches the official/developer-guide point that the heavy 27-layer vision encoder is replaced by a lightweight projection-style embedder.

The processor config sets:

| Processor field | Value |
|---|---:|
| `image_seq_length` | 280 |
| `image_processor.max_soft_tokens` | 280 |
| `video_processor.max_soft_tokens` | 70 |
| `video_processor.num_frames` | 32 |
| `do_normalize` for images | false |
| `do_rescale` for images | true |
| `rescale_factor` | `1 / 255` |

### 12B Audio Projection

Source: `processor_config.json` and `audio_config`.

| Field | Value |
|---|---:|
| `audio_config.model_type` | `gemma4_unified_audio` |
| `audio_embed_dim` | 640 |
| `audio_samples_per_token` | 640 |
| `audio_ms_per_token` | 40 |
| `audio_seq_length` | 750 |
| Sampling rate | 16000 Hz |
| Max audio represented by config | 30 seconds |
| Projection | RMSNorm + Linear into text hidden space |

At 16 kHz, 640 samples equals 40 ms. With 750 audio tokens, the processor budget is 30 seconds.

### Why Unified Matters

The unified path is not just a deployment trick:

- Multimodal inputs enter the same decoder-only transformer directly.
- There is no separate vision attention stack to run before the LLM.
- There is no separate Conformer-style audio encoder to run before the LLM.
- Fine-tuning can update the multimodal projection and text backbone in one pass.
- Text-only inference can ignore these modules entirely.

For this repository, the immediate recommendation is still to bring up **text-only 12B** first. The unified multimodal path is small enough to add later, but it is not needed to validate the CUDA text pipeline.

---

## 11. Standard Multimodal Path in 31B and 26B

31B and 26B-A4B use the standard Gemma 4 multimodal path rather than the 12B unified path.

### Vision Tower

Source: `vision_config` in 31B and 26B configs.

| Field | 31B / 26B-A4B |
|---|---:|
| `vision_config.model_type` | `gemma4_vision` |
| Vision hidden size | 1152 |
| Vision layers | 27 |
| Vision attention heads | 16 |
| Vision KV heads | 16 |
| Vision head dim | 72 |
| Vision intermediate size | 4304 |
| Patch size | 16 |
| Pooling kernel | 3 |
| Default output length | 280 |
| Position embedding size | 10240 |
| RoPE theta | 100 |
| Standardize | true |

The vision tower uses bidirectional attention over image tokens and 2D RoPE. Text tokens remain causal. This is useful for multimodal inference, but this project should not block the first correct text inference path on it.

### No Audio Tower

The 31B and 26B-A4B configs have `audio_config = null`. They support text and image, not audio.

---

## 12. Tokenizer and Chat Format

### Vocabulary and Token IDs

The shared vocabulary size is 262,144.

| Token / field | ID |
|---|---:|
| `<pad>` / `pad_token_id` | 0 |
| `<eos>` / base `eos_token_id` | 1 |
| `<bos>` / `bos_token_id` | 2 |
| `boi_token_id` | 255999 |
| `boa_token_id` | 256000 |
| `image_token_id` | 258880 |
| `audio_token_id` | 258881 |
| `eoi_token_id` | 258882 |
| `eoa_token_id` | 258883 |
| `video_token_id` | 258884 |

Instruction-tuned configs may use multiple EOS IDs in generation config. For example, the downloaded `31B-it` generation config lists `[1, 106, 50]`. The base 12B generation config uses EOS `1` and suppresses `258883` and `258882`.

### Generation Defaults

Official generation config:

| Field | Value |
|---|---:|
| `do_sample` | true |
| `temperature` | 1.0 |
| `top_k` | 64 |
| `top_p` | 0.95 |
| `bos_token_id` | 2 |
| `pad_token_id` | 0 |

### Chat and Thinking

Gemma 4 instruction-tuned models support standard chat roles, including a native `system` role. Official docs describe a configurable thinking mode controlled through the chat template and thinking tokens. Many libraries hide the raw template details behind `processor.apply_chat_template`.

For this repo's low-level inference path, tokenization/chat templating should remain outside the CUDA core. The CUDA path should accept token IDs and return logits/sampled token IDs.

---

## 13. Hyperparameter Tables

### Text Backbone

| Parameter | 12B Unified | 31B Dense | 26B-A4B |
|---|---:|---:|---:|
| Official listed scale | 11.95B | 30.7B | 25.2B total / 3.8B active |
| `hidden_size` | 3840 | 5376 | 2816 |
| `num_hidden_layers` | 48 | 60 | 30 |
| Sliding layers | 40 | 50 | 25 |
| Global layers | 8 | 10 | 5 |
| `num_attention_heads` | 16 | 32 | 16 |
| Sliding KV heads | 8 | 16 | 8 |
| Global KV heads | 1 | 4 | 2 |
| Sliding head dim | 256 | 256 | 256 |
| Global head dim | 512 | 512 | 512 |
| Sliding Q width | 4096 | 8192 | 4096 |
| Sliding KV width | 2048 | 4096 | 2048 |
| Global Q width | 8192 | 16384 | 8192 |
| Global KV width | 512 | 2048 | 1024 |
| FFN intermediate | 15360 | 21504 | 2112 shared, 704 expert |
| MoE experts | none | none | 128 |
| Active experts/token | none | none | 8 |
| Sliding window | 1024 | 1024 | 1024 |
| Max context | 262144 | 262144 | 262144 |
| Vocabulary | 262144 | 262144 | 262144 |
| K=V global attention | yes | yes | yes |
| PLE | no | no | no |
| KV-shared layers | 0 | 0 | 0 |
| Default dtype | BF16 | BF16 | BF16 |

### RoPE

| Parameter | Sliding layers | Global layers |
|---|---:|---:|
| `rope_theta` | 10000 | 1000000 |
| `rope_type` | default | proportional |
| `partial_rotary_factor` | not set / full | 0.25 |
| Head dim | 256 | 512 |
| Rotated dims | 256 | 128 |

### Multimodal Support

| Parameter | 12B Unified | 31B Dense | 26B-A4B |
|---|---|---|---|
| Text | yes | yes | yes |
| Image | encoder-free embedder | 27-layer vision tower | 27-layer vision tower |
| Audio | encoder-free waveform projection | no | no |
| Video | frames through image path | frames through image path | frames through image path |
| Vision hidden / embed dim | 3840 embedder | 1152 tower | 1152 tower |
| Vision output tokens | 280 image, 70/video frame budget | 280 | 280 |
| Audio budget | 750 tokens / 30 sec | none | none |

---

## 14. Parameter Accounting

These are approximate text-path counts for implementation planning. They intentionally focus on the transformer/LM path, not every tokenizer or processor artifact.

### 12B Unified

```text
Sliding attention per layer: 47.2M
Global attention per layer:  64.9M
Dense FFN per layer:        176.9M

Sliding blocks: 40 * 224.1M =  8.97B
Global blocks:   8 * 241.8M =  1.93B
Token embedding:              1.01B
Text total:                  11.91B
Official scale:              11.95B
```

### 31B Dense

```text
Sliding attention per layer: 132.1M
Global attention per layer:  187.2M
Dense FFN per layer:         346.8M

Sliding blocks: 50 * 478.9M = 23.95B
Global blocks:  10 * 534.0M =  5.34B
Token embedding:              1.41B
Text total:                  30.70B
```

### 26B-A4B

```text
Sliding attention per layer:       34.6M
Global attention per layer:        49.0M
Shared dense FFN per layer:        17.8M
All experts per layer:            761.3M
Active experts per token/layer:    47.6M
Router per layer:                   0.36M

Capacity:
  Sliding blocks: 25 * 814.1M = 20.35B
  Global blocks:   5 * 828.5M =  4.14B
  Token embedding:              0.74B
  Total capacity:              25.23B

Active:
  Sliding blocks: 25 * 100.4M =  2.51B
  Global blocks:   5 * 114.8M =  0.57B
  Token embedding:              0.74B
  Active scale:                 3.82B
```

---

## 15. KV Cache and Memory Planning

### Correct KV Cache Formula

For Gemma 4's hybrid attention, an optimized decode cache should treat sliding and global layers differently:

```text
sliding_kv_bytes =
  sliding_layers * 2 * sliding_kv_heads * head_dim * dtype_bytes * min(sequence_length, sliding_window)

global_kv_bytes =
  global_layers * 2 * global_kv_heads * global_head_dim * dtype_bytes * sequence_length

total_kv_bytes = sliding_kv_bytes + global_kv_bytes
```

Do not use `num_layers * sequence_length` for every layer. Sliding layers only need the local window for steady-state decode. This distinction is central to why Gemma 4 can target 256K context.

### BF16 KV Cache, Batch 1

Assumes:

- BF16 K/V cache, 2 bytes per scalar.
- Sliding cache capped at 1024 tokens.
- Global cache stores full context.
- No allocator fragmentation or paging metadata included.

| Context | 12B Unified | 31B Dense | 26B-A4B |
|---:|---:|---:|---:|
| 4K | 0.38 GiB | 1.09 GiB | 0.27 GiB |
| 32K | 0.81 GiB | 3.28 GiB | 0.82 GiB |
| 128K | 2.31 GiB | 10.78 GiB | 2.70 GiB |
| 256K | 4.31 GiB | 20.78 GiB | 5.20 GiB |

The old all-layers-full-context estimate is too pessimistic for a specialized implementation. It may still describe a naive framework cache, but it should not guide this repo's memory targets.

### Weight Memory

Official Google memory table includes about 20% loading overhead and excludes KV cache / context memory.

| Precision | 12B Unified | 31B Dense | 26B-A4B |
|---|---:|---:|---:|
| BF16 / 16-bit | 26.7 GB | 69.9 GB | 57.7 GB |
| SFP8 / 8-bit | 13.4 GB | 34.9 GB | 28.8 GB |
| Q4_0 / 4-bit | 6.7 GB | 17.5 GB | 14.4 GB |

Raw text parameter memory without loader overhead is lower:

| Model | Text scale | Raw BF16 text weights | Raw 8-bit text weights | Raw 4-bit text weights |
|---|---:|---:|---:|---:|
| 12B | 11.95B | ~23.9 GB | ~12.0 GB | ~6.0 GB |
| 31B | 30.7B | ~61.4 GB | ~30.7 GB | ~15.4 GB |
| 26B-A4B | 25.2B loaded | ~50.4 GB | ~25.2 GB | ~12.6 GB |

Use official load-memory numbers for practical VRAM planning. Use raw numbers for kernel bandwidth math.

---

## 16. A6000 Implications

The target GPU constraint matters:

- RTX A6000 has 48 GB VRAM.
- It is Ampere (`sm_86`).
- It supports BF16/FP16/INT8/INT4 tensor core paths.
- It does not have native FP8 tensor-core support.

### Single-GPU Fit

Approximate fit on one 48 GB A6000:

| Model / precision | Weights with overhead | 256K BF16 KV | Total before workspace | Practical? |
|---|---:|---:|---:|---|
| 12B BF16 | 26.7 GB | 4.31 GiB | ~31-32 GB | yes |
| 12B 8-bit | 13.4 GB | 4.31 GiB | ~18 GB | yes |
| 12B Q4 | 6.7 GB | 4.31 GiB | ~12 GB | yes |
| 31B BF16 | 69.9 GB | 20.78 GiB | >90 GB | no |
| 31B 8-bit | 34.9 GB | 20.78 GiB | ~57 GB | no at 256K; maybe short context |
| 31B Q4 | 17.5 GB | 20.78 GiB | ~39-40 GB | tight but plausible with careful runtime |
| 26B BF16 | 57.7 GB | 5.20 GiB | >62 GB | no |
| 26B 8-bit | 28.8 GB | 5.20 GiB | ~34 GB | yes, but MoE path required |
| 26B Q4 | 14.4 GB | 5.20 GiB | ~20 GB | yes, but MoE path required |

### Recommended Project Path

For this repository:

1. Use **12B BF16 text-only** to validate the full unfused inference pipeline on A6000.
2. Generalize constants so 12B and 31B dimensions are config-selected, not hardcoded.
3. Add weight-only quantization for 31B.
4. Treat KV cache quantization as a separate experiment after BF16 KV is correct.
5. Defer 26B-A4B until dense attention/FFN orchestration is stable.
6. Do not build an FP8-first path for A6000; use INT8/INT4 or BF16 where Ampere is strong.

---

## 17. Benchmark Snapshot

Official 12B model card benchmark table, instruction-tuned variants:

| Benchmark | 31B | 26B-A4B | 12B Unified | Gemma 3 27B no-think |
|---|---:|---:|---:|---:|
| MMLU Pro | 85.2% | 82.6% | 77.2% | 67.6% |
| AIME 2026 no tools | 89.2% | 88.3% | 77.5% | 20.8% |
| LiveCodeBench v6 | 80.0% | 77.1% | 72.0% | 29.1% |
| Codeforces ELO | 2150 | 1718 | 1659 | 110 |
| GPQA Diamond | 84.3% | 82.3% | 78.8% | 42.4% |
| Tau2 average over 3 | 76.9% | 68.2% | 69.0% | 16.2% |
| BigBench Extra Hard | 74.4% | 64.8% | 53.0% | 19.3% |
| MMMLU | 88.4% | 86.3% | 83.4% | 70.7% |
| MMMU Pro | 76.9% | 73.8% | 69.1% | 49.7% |
| OmniDocBench 1.5, lower better | 0.131 | 0.149 | 0.164 | 0.365 |
| MATH-Vision | 85.6% | 82.4% | 79.7% | 46.0% |
| MedXPertQA MM | 61.3% | 58.1% | 48.7% | - |
| CoVoST | - | - | 38.5* | - |
| FLEURS, lower better | - | - | 0.069* | - |
| MRCR v2 8-needle 128K | 66.4% | 44.1% | 43.4% | 13.5% |

`*` The 12B audio numbers exclude Chinese, per the model card note.

The benchmark takeaway for this repo is simple:

- 31B is strongest overall but needs quantization on A6000.
- 26B-A4B is very strong for active compute, but MoE complicates implementation.
- 12B is the best full-precision local development target and is much stronger than its size suggests.

---

## 18. Implementation Notes for this Repo

### Constants That Must Become Configurable

The current code was written around 31B constants such as hidden size 5376 and attention widths 8192/16384. To support 12B cleanly, these must become model-selected:

| Concept | 12B | 31B |
|---|---:|---:|
| Hidden width | 3840 | 5376 |
| FFN width | 15360 | 21504 |
| Q heads | 16 | 32 |
| Sliding KV heads | 8 | 16 |
| Global KV heads | 1 | 4 |
| Layers | 48 | 60 |
| Sliding/global counts | 40/8 | 50/10 |
| LM head input width | 3840 | 5376 |

The simplest good design is not a general framework. Use a small set of compile-time or launch-time model configs for the repo-relevant models.

### Kernel Inventory Update

For dense text, the baseline kernel order remains:

1. Token embedding gather.
2. RMSNorm.
3. Residual add.
4. Residual add + RMSNorm.
5. Q/K RMSNorm.
6. V RMSNorm without learned weight.
7. RoPE and p-RoPE.
8. KV-cache write/update with sliding/global policies.
9. Sliding-window attention.
10. Global attention.
11. Attention output pack.
12. GeGLU tanh activation.
13. Final logit softcap.
14. Sampling.

But each kernel should be parameterized over:

- hidden width,
- number of heads,
- number of KV heads,
- head dimension,
- global head dimension,
- rotary dimension,
- layer type,
- sliding window.

Do not introduce MoE kernels until the dense path works for 12B and 31B.

### GEMM Policy

Keep dense GEMMs in cuBLAS/cuBLASLt for the baseline:

| Model | Dominant dense FFN GEMMs |
|---|---|
| 12B | `[M, 3840] x [3840, 15360]`, `[M, 15360] x [15360, 3840]` |
| 31B | `[M, 5376] x [5376, 21504]`, `[M, 21504] x [21504, 5376]` |

Only replace cuBLAS/cuBLASLt after:

- numerical correctness exists,
- the unfused path is benchmarked,
- Nsight Compute shows a specific bottleneck,
- a custom kernel has a plausible roofline advantage.

### KV Cache Policy

Use two cache policies:

```text
sliding layer:
  cache length = sliding_window
  update by ring-buffer / paged-window logic

global layer:
  cache length = full sequence
  append through full context
```

For 12B:

```text
sliding cache width per layer = 2 * 8 * 256 = 4096 BF16 scalars/token
global cache width per layer  = 2 * 1 * 512 = 1024 BF16 scalars/token
```

For 31B:

```text
sliding cache width per layer = 2 * 16 * 256 = 8192 BF16 scalars/token
global cache width per layer  = 2 * 4 * 512 = 4096 BF16 scalars/token
```

### 12B as the A6000 Baseline

The most practical near-term objective is:

```text
Gemma 4 12B text-only BF16
  -> correct unfused pipeline
  -> benchmark prefill/decode
  -> add model-config selection
  -> port same kernels to 31B Q4/INT8 weight path
```

This route keeps the implementation close to the original 31B thesis while avoiding the dead end of trying to fit 31B BF16 onto 48 GB.

---

## 19. Sources

Primary:

1. Google AI for Developers, Gemma 4 model overview: https://ai.google.dev/gemma/docs/core
2. Google AI for Developers, Gemma 4 model card: https://ai.google.dev/gemma/docs/core/model_card_4
3. Google blog, "Introducing Gemma 4 12B: a unified, encoder-free multimodal model": https://blog.google/innovation-and-ai/technology/developers-tools/introducing-gemma-4-12b/
4. Google Developers Blog, "Gemma 4 12B: The Developer Guide": https://developers.googleblog.com/gemma-4-12b-the-developer-guide/
5. Hugging Face, `google/gemma-4-12B`: https://huggingface.co/google/gemma-4-12B
6. Hugging Face, `google/gemma-4-31B-it`: https://huggingface.co/google/gemma-4-31B-it
7. Hugging Face, `google/gemma-4-26B-A4B`: https://huggingface.co/google/gemma-4-26B-A4B
8. Hugging Face Transformers docs, Gemma4 Unified: https://huggingface.co/docs/transformers/model_doc/gemma4_unified
9. Hugging Face Transformers docs, Gemma4: https://huggingface.co/docs/transformers/model_doc/gemma4

Downloaded metadata and implementation references:

1. `/tmp/gemma4_hf_metadata/gemma-4-12B/config.json`
2. `/tmp/gemma4_hf_metadata/gemma-4-12B/README.md`
3. `/tmp/gemma4_hf_metadata/gemma-4-12B/processor_config.json`
4. `/tmp/gemma4_hf_metadata/gemma-4-31B-it/config.json`
5. `/tmp/gemma4_hf_metadata/gemma-4-26B-A4B/config.json`
6. `/tmp/gemma4_hf_metadata/modeling_gemma4.py`
7. `/tmp/gemma4_hf_metadata/modeling_gemma4_unified.py`

Secondary explanatory sources found via Exa:

1. Maarten Grootendorst, "A Visual Guide to Gemma 4 12B": https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-gemma-4-12b
2. Hugging Face blog, "Welcome Gemma 4": https://huggingface.co/blog/gemma4

Use official configs over prose if the two disagree. Use the Transformers implementation to resolve execution semantics such as global K=V, KV sharing, p-RoPE, and the 12B unified embedder path.
