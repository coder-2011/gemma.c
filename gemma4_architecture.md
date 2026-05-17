# Gemma 4 Architecture: Complete Technical Reference

> **Release:** April 2, 2026 — Google DeepMind  
> **License:** Apache 2.0  
> **Status:** No formal research paper published as of May 2026. Primary sources: official model card, HuggingFace model pages, HuggingFace transformers documentation, and community reverse-engineering from config.json files.

---

## Table of Contents

1. [Family Overview](#1-family-overview)
2. [Shared Architectural DNA](#2-shared-architectural-dna)
3. [Attention Mechanism — Dual-Config Hybrid](#3-attention-mechanism--dual-config-hybrid)
4. [Positional Encoding — Proportional RoPE (p-RoPE)](#4-positional-encoding--proportional-rope-p-rope)
5. [Per-Model Architecture Deep Dives](#5-per-model-architecture-deep-dives)
   - 5.1 [Gemma 4 31B — Dense Server](#51-gemma-4-31b--dense-server)
   - 5.2 [Gemma 4 26B-A4B — MoE Server](#52-gemma-4-26b-a4b--moe-server)
   - 5.3 [Gemma 4 E4B — Edge Dense](#53-gemma-4-e4b--edge-dense)
   - 5.4 [Gemma 4 E2B — On-Device Any-to-Any](#54-gemma-4-e2b--on-device-any-to-any)
6. [Per-Layer Embeddings (PLE)](#6-per-layer-embeddings-ple)
7. [Vision Encoder](#7-vision-encoder)
8. [Audio Encoder (E-Series Only)](#8-audio-encoder-e-series-only)
9. [MoE Architecture Deep Dive](#9-moe-architecture-deep-dive)
10. [Tokenizer](#10-tokenizer)
11. [Full Hyperparameter Tables](#11-full-hyperparameter-tables)
12. [Parameter Accounting](#12-parameter-accounting)
13. [Memory & Hardware Requirements](#13-memory--hardware-requirements)
14. [Benchmark Results](#14-benchmark-results)
15. [What Changed from Gemma 3](#15-what-changed-from-gemma-3)
16. [Training Details](#16-training-details)
17. [Inference & Deployment](#17-inference--deployment)

---

## 1. Family Overview

Gemma 4 is the fourth generation of Google DeepMind's open-weight model family, built directly from the same research foundations as the proprietary Gemini 3 models. It is the first Gemma generation where **architecture — not just scale or training data — is the primary axis of differentiation** across variants.

### Model Variants

| Model | Architecture | Total Params | Active Params | Context | Modalities |
|---|---|---|---|---|---|
| **Gemma 4 E2B** | Dense + PLE | ~5.1B | ~2.3B effective | 128K | Text, Image, Video, Audio |
| **Gemma 4 E4B** | Dense + PLE | ~8B | ~4.5B effective | 128K | Text, Image, Video, Audio |
| **Gemma 4 26B-A4B** | MoE | ~26B | ~3.8B active | 256K | Text, Image, Video |
| **Gemma 4 31B** | Dense | ~31B | ~31B | 256K | Text, Image, Video |

**Naming conventions:**
- `E` = "Effective" parameters — edge models where PLE tables account for a large portion of total params but only `~2–4.5B` parameters are actively computed per forward pass
- `A` = "Active" parameters — for the MoE, only `~3.8B` of `26B` total parameters fire per token
- All four models are **multimodal from the ground up** — not a bolted-on vision tower, but trained jointly from scratch
- All shipped as both **pre-trained (PT)** and **instruction-tuned (IT)** variants with thinking mode

---

## 2. Shared Architectural DNA

Every Gemma 4 model is a **decoder-only transformer** with the following universally shared properties:

### Core Architecture
- **Type:** Causal decoder-only transformer (autoregressive)
- **Normalization:** RMSNorm at pre-attention (`input_layernorm`), post-attention (`post_attention_layernorm`), pre-FFN (`pre_feedforward_layernorm`), post-FFN (`post_feedforward_layernorm`) — 4 norms per block
- **Activation:** `GELU (tanh approximation)` — `gelu_pytorch_tanh` — used inside GeGLU gated units
- **FFN type:** GeGLU (Gated Linear Units with GELU): `FFN(x) = (xW_gate ⊙ GELU(xW_up)) · W_down`
- **Attention type:** Grouped Query Attention (GQA) on all variants, with differing group ratios per model
- **QK Normalization:** RMSNorm applied to query and key projections before attention computation — stabilizes training at scale
- **V Normalization:** RMSNorm applied to values (WITHOUT a learned scale parameter) — new in Gemma 4, absent in Gemma 3
- **Logit soft-capping:** `tanh(x / 30.0) * 30.0` applied to final output logits — bounds to `[-30, 30]`
- **Weight tying:** Input embedding and output lm_head weights are tied (`tie_word_embeddings=True`)
- **Attention bias:** None (`attention_bias=False`)
- **Attention dropout:** 0.0
- **K=V sharing in global layers:** Full attention layers share K and V projections — V is derived from K before normalization, eliminating the V projection matrix entirely
- **Final layer always global:** The last decoder layer is always a full (global) attention layer, regardless of the interleaving ratio — guarantees full-context awareness in final representation
- **Vocabulary size:** 262,144 tokens (up from 262,208 in Gemma 3 — minor boundary rounding)
- **Attention — K equals V (`attention_k_eq_v`):** `true` — in global layers, K and V start from the same projection, then diverge through separate normalization paths
- **`initializer_range`:** 0.02
- **`rms_norm_eps`:** 1e-6

### Hybrid Attention Pattern

All Gemma 4 models alternate between two structurally distinct layer types:

```
Sliding Layer (local): head_dim=256, more KV heads, RoPE theta=10K, full rotation, window=512 or 1024
Full Layer (global):   head_dim=512, fewer KV heads, p-RoPE theta=1M, 25% partial rotation, no window
```

The pattern ratio is **5:1** (five sliding, one full) for 26B and 31B, and **4:1** (four sliding, one full) for E2B/E4B.

---

## 3. Attention Mechanism — Dual-Config Hybrid

This is the most architecturally significant innovation in Gemma 4. Gemma 3 used identical attention geometry across all layers; Gemma 4 makes the geometry fundamentally different depending on layer type.

### Sliding Window (Local) Attention Layers

- **Window size:** 512 tokens (E2B, E4B) or 1024 tokens (26B, 31B)
- **`head_dim`:** 256
- **KV heads:** Model-dependent (see hyperparameter table below)
- **RoPE:** Standard, `theta=10,000`, all head dimensions rotated (100%)
- **Masking:** Causal mask + window mask — token can only attend to the last `window` tokens
- **Purpose:** Efficient local pattern recognition, syntax, short-range dependencies

### Full / Global Attention Layers

- **Window size:** None — full causal attention over entire sequence
- **`global_head_dim`:** 512 (double the sliding head_dim)
- **KV heads:** 4× fewer than sliding layers (see table)
- **RoPE:** Proportional RoPE (p-RoPE), `theta=1,000,000`, only 25% of dimensions rotated
- **K=V weight sharing:** V projection matrix is eliminated; K is cloned as V before normalization
- **`k_norm`:** RMSNorm with learned scale (applied to keys)
- **`v_norm`:** RMSNorm without learned scale (applied to values derived from K)
- **Purpose:** Long-range semantic attention, global context, cross-sentence reasoning

### Why Two Head Dims?

The wider `head_dim=512` in global layers compensates for the aggressive KV compression (fewer KV heads). Each global KV head needs higher capacity to represent the full sequence. The narrower `head_dim=256` in sliding layers is fine because each head only covers a 512–1024 token window.

### Information Propagation

Sliding window layers cannot directly attend to tokens outside their window. However, information propagates through the residual stream — tokens from 5,000 positions ago influence today's window because their hidden states were modified by all prior layers. The periodic global layers (every 5th or 6th) break through this indirection by attending directly to the full sequence.

---

## 4. Positional Encoding — Proportional RoPE (p-RoPE)

### Background

Standard RoPE rotates all head dimensions. In long contexts, the lowest-frequency (slowest-rotating) dimensions eventually complete full rotations, destroying semantic signal. Gemma 3 addressed this with linear frequency scaling (8×) on global layers, but this is suboptimal.

### The p-RoPE Solution

Based on the paper *"Round and Round We Go! What makes Rotary Positional Encodings useful?"* (Barbero et al., Oxford & Google DeepMind, 2024, arXiv:2410.06205).

The key insight: **not all RoPE dimensions are equal**:
- High-frequency dimensions → **positional heads** (robust position tracking)
- Low-frequency dimensions → **semantic heads** (content meaning, position-invariant at long context)

**p-RoPE** sets the `(1 - p)` lowest-frequency dimensions to zero, making them position-independent (NoPE). With `p=0.25`, only 25% of dimensions carry positional information; 75% become pure semantic channels that never degrade regardless of context length.

### Gemma 4's Implementation

| Layer Type | theta | Partial Factor | Dims Rotated | Behavior |
|---|---|---|---|---|
| Sliding (local) | 10,000 | 1.0 (100%) | All head dims | Standard RoPE, fine-grained local positioning |
| Full (global) | 1,000,000 | 0.25 (25%) | Top 25% of dims | p-RoPE, robust long-range semantic attention |

For the 31B with `global_head_dim=512`:
- 128 dimensions are rotated (positional)
- 384 dimensions are position-free (semantic NoPE)

This is why Gemma 4 can handle 256K context reliably while Gemma 3 struggled beyond 128K.

### Validation (from p-RoPE paper, lower perplexity = better)

| Encoding | WikiText | Properties |
|---|---|---|
| NoPE (no rotation) | 4.8594 | Semantic only, no position |
| Standard RoPE (theta=10K) | 4.4627 | Baseline |
| RoPE (theta=500K) | 4.4485 | High theta |
| **0.25-RoPE (p=0.25)** | **4.4592** | **Gemma 4's setting** |
| 0.75-RoPE full model | 4.4414 | Best overall |

Gemma 4 uses `0.25-RoPE` on global layers only, giving the best of both: standard RoPE for local layers, p-RoPE for global layers.

---

## 5. Per-Model Architecture Deep Dives

### 5.1 Gemma 4 31B — Dense Server

The flagship model. No sparsity tricks — every layer computes fresh K, V, and FFN.

**Core text config:**

| Parameter | Value |
|---|---|
| `model_type` | `gemma4` |
| `hidden_size` | 5,376 |
| `num_hidden_layers` | 60 |
| `layer_pattern` | 50 sliding + 10 full (5:1 ratio) |
| `num_attention_heads` (Q) | 32 |
| `num_key_value_heads` (sliding) | 16 |
| `num_key_value_heads` (global) | 4 |
| `head_dim` (sliding) | 256 |
| `global_head_dim` (full) | 512 |
| `intermediate_size` (FFN hidden) | 21,504 |
| `FFN type` | GeGLU (dense) |
| `sliding_window` | 1,024 tokens |
| `context_window` | 256,000 tokens |
| `vocab_size` | 262,144 |
| `attention_k_eq_v` | true (global layers) |
| `final_logit_softcapping` | 30.0 |
| `rms_norm_eps` | 1e-6 |
| `hidden_activation` | gelu_pytorch_tanh |
| `attention_bias` | false |
| `attention_dropout` | 0.0 |
| `dtype` | bfloat16 |

**Per-block parameter counts:**

| Component | Sliding Block | Full Block | Notes |
|---|---|---|---|
| Q proj | 44.0M | 88.1M | `5376 × 32 × head_dim` |
| K proj | 22.0M | 11.0M | `5376 × kv_heads × head_dim` |
| V proj | 22.0M | **0** | Eliminated in full layers (K=V) |
| O proj | 44.0M | 88.1M | `q_heads × head_dim × 5376` |
| **Attention total** | **132.1M** | **187.2M** | |
| GeGLU FFN | 346.8M | 346.8M | `3 × 5376 × 21504` |
| **Block total** | **478.9M** | **534.0M** | |

**Total:** `50 × 478.9M + 10 × 534.0M + 1.4B embed ≈ 30.7B`

**Vision encoder:** ViT, 27 transformer layers, hidden dim 1152, 550M parameters (shared with 26B)

---

### 5.2 Gemma 4 26B-A4B — MoE Server

The efficiency powerhouse. Achieves 98% of the 31B's Arena score with only ~3.8B active parameters per token — a 7.5× compute reduction.

**Core text config:**

| Parameter | Value |
|---|---|
| `model_type` | `gemma4` |
| `hidden_size` | 2,816 |
| `num_hidden_layers` | 30 |
| `layer_pattern` | 25 sliding + 5 full (5:1 ratio) |
| `num_attention_heads` (Q) | 16 |
| `num_key_value_heads` (sliding) | 8 |
| `num_key_value_heads` (global) | 2 |
| `head_dim` (sliding) | 256 |
| `global_head_dim` (full) | 512 |
| `intermediate_size` (dense FFN hidden) | 2,112 |
| `FFN type` | MoE (128 experts, top-8) + parallel dense GeGLU |
| `num_experts` | 128 |
| `num_experts_per_tok` | 8 |
| `expert_hidden_size` | 704 |
| `sliding_window` | 1,024 tokens |
| `context_window` | 256,000 tokens |
| `vocab_size` | 262,144 |
| `enable_moe_block` | true |
| `attention_k_eq_v` | true (global layers) |
| `final_logit_softcapping` | 30.0 |
| `dtype` | bfloat16 |

**The unusual MoE design — parallel, not replacement:**

Most MoE models (DeepSeek, Mixtral, Qwen) **replace** the dense FFN with a MoE layer. Gemma 4's 26B instead runs them **in parallel and sums their outputs**:

```
FFN_output = (dense_GeGLU(x) + MoE_output(x)) * (1 / sqrt(2))
```

Where:
- `dense_GeGLU`: always-on, hidden=2,112, provides stable baseline capacity
- `MoE_output`: 128 experts, each with hidden=704, top-8 routing per token
- The `1/sqrt(2)` scaling prevents magnitude explosion from summation

**Expert routing:**
- Router: learned linear projection `nn.Linear(2816, 128)` + softmax
- For each token, 128 scores computed → top-8 selected by score
- Expert outputs weighted by their softmax scores and summed
- Routing is independent per token, per layer, per forward pass

**Per-block parameter counts:**

| Component | Sliding Block | Full Block | Notes |
|---|---|---|---|
| Q proj | 11.5M | 23.1M | `2816 × 16 × head_dim` |
| K proj | 5.8M | 2.9M | `2816 × kv_heads × head_dim` |
| V proj | 5.8M | **0** | Eliminated in full layers |
| O proj | 11.5M | 23.1M | |
| **Attention total** | **34.6M** | **49.0M** | |
| Dense GeGLU | 17.8M | 17.8M | `3 × 2816 × 2112` |
| MoE experts (all 128) | 761.3M | 761.3M | `128 × 3 × 2816 × 704` |
| MoE active (top-8) | 47.6M | 47.6M | `8 × 5.9M per expert` |
| Router | 0.4M | 0.4M | `2816 × 128` |
| **FFN total capacity** | **779.5M** | **779.5M** | |
| **FFN active/token** | **65.8M** | **65.8M** | dense + top-8 + router |
| **Block total capacity** | **814.1M** | **828.5M** | |
| **Block active/token** | **100.4M** | **114.8M** | |

**Total capacity:** `25 × 814M + 5 × 829M + 0.7B embed ≈ 25.2B`  
**Active per token:** `25 × 100M + 5 × 115M + 0.7B embed ≈ 3.8B`

> Note: All 26B parameters must be loaded into VRAM even though only ~3.8B are active per token — expert weights are all needed for routing decisions.

---

### 5.3 Gemma 4 E4B — Edge Dense

The mid-size on-device model. ~8B total parameters, ~4.5B effective. Shares E-series architecture with PLE and KV cache sharing.

**Key specs:**

| Parameter | Value |
|---|---|
| Total parameters | ~8B |
| Effective parameters | ~4.5B |
| Context window | 128,000 tokens |
| Sliding window | 512 tokens |
| Layer pattern | 4:1 (sliding:full) |
| Modalities | Text, Image, Video, Audio |
| Vision encoder | ViT, 16 layers, d=768, 150M params |
| Audio encoder | Conformer (USM-style) |
| PLE | Yes — per-layer embedding system |
| KV cache sharing | Yes |

> Note: Full config.json was not publicly released at time of writing. Detailed hyperparameters confirmed to be similar to E2B but scaled up.

---

### 5.4 Gemma 4 E2B — On-Device Any-to-Any

The most architecturally novel variant. The largest architectural divergence from the server models.

**Core text config:**

| Parameter | Value |
|---|---|
| `model_type` | `gemma4` |
| `hidden_size` | 1,536 |
| `num_hidden_layers` | 35 |
| `layer_pattern` | 28 sliding + 7 full (4:1 ratio) |
| `num_attention_heads` (Q) | 8 |
| `num_key_value_heads` (sliding) | 1 (extreme MQA) |
| `num_key_value_heads` (global) | 1 |
| `head_dim` (sliding) | 256 |
| `global_head_dim` (full) | 512 |
| `intermediate_size` (standard layers) | 6,144 |
| `intermediate_size` (KV-shared layers, 2× wide) | 12,288 |
| `sliding_window` | 512 tokens |
| `context_window` | 128,000 tokens |
| `vocab_size` | 262,144 |
| `num_kv_shared_layers` | 20 (of 35 total) |
| `hidden_size_per_layer_input` (PLE dim) | 256 |
| `vocab_size_per_layer_input` (PLE vocab) | 262,144 |
| `attention_k_eq_v` | false (E2B does not use K=V sharing) |
| `final_logit_softcapping` | 30.0 |
| `dtype` | bfloat16 |

**KV Cache Sharing:**

20 of 35 layers reuse KV caches computed by earlier layers of the same type. This dramatically reduces the KV cache memory footprint. Layers with shared KV compensate by using a **2× wider MLP** (`hidden=12,288` instead of `6,144`), maintaining representation capacity.

**Per-block parameter counts (E2B):**

| Component | Sliding (standard) | Full (standard) | Notes |
|---|---|---|---|
| Q proj | 3.1M | 6.3M | `1536 × 8 × head_dim` |
| K proj | 0.4M | 0.8M | `1536 × 1 × head_dim` |
| V proj | 0.4M | 0.8M | E2B does not use K=V sharing |
| O proj | 3.1M | 6.3M | |
| **Attention total** | **7.1M** | **14.2M** | |
| GeGLU FFN (standard) | 28.3M | 28.3M | `3 × 1536 × 6144` |
| GeGLU FFN (KV-shared, 2× wide) | 56.6M | 56.6M | `3 × 1536 × 12288` |

**Total breakdown:**
- 15 standard blocks + 20 double-wide (KV-shared) blocks
- + 0.4B computed embedding params
- + ~2.3B per-layer embedding table (PLE)
- **≈ 5.1B total, ~2.3B effective**

---

## 6. Per-Layer Embeddings (PLE)

PLE is the key innovation enabling E-series models to run efficiently on-device. It gives each decoder layer a **unique token-dependent signal** rather than sharing one embedding across all layers.

### Motivation

In standard transformers, all layers process the same input embedding. But different layers specialize differently (early = syntax, late = semantics). PLE lets each layer "re-read" the token with a layer-specific lens, enabling shallower models to encode information that would otherwise require more depth.

### Dual-Signal Architecture

PLE combines two complementary signals per layer:

**1. Token-Identity Component**
- A single `nn.Embedding` of shape `(vocab_size, num_layers × hidden_size_per_layer_input)`
- For E2B: shape = `(262144, 35 × 256)` = `(262144, 8960)` — this alone is ~2.3B parameters
- Lookup is direct (no context from surrounding tokens)
- Scaled by `√hidden_size_per_layer_input = √256 = 16`

**2. Context-Aware Component**
- A learned `nn.Linear(hidden_size, num_layers × hidden_size_per_layer_input)` 
- Projects the main `inputs_embeds` (which include vision/audio soft tokens for multimodal inputs) into per-layer space
- Scaled by `1/√hidden_size`, then RMSNorm applied
- This component *sees surrounding context* through the embedding

**Combination:**
```python
per_layer_input = (token_identity + context_aware) * (1 / sqrt(2))
# Reshaped to: (batch, seq_len, num_layers, hidden_size_per_layer_input)
```

**Injection into each layer:**

Each decoder layer receives a `256`-dim slice for its specific layer index. This is injected as a **gated third residual block** after attention and FFN:

```
hidden = hidden + attention_output
hidden = hidden + ffn_output
hidden = hidden + gate(per_layer_input[layer_idx])  # PLE residual
```

### Parameter Accounting

For E2B:
- Per-layer embedding table: `262144 × 35 × 256 = ~2.34B parameters`
- Context projection: `1536 × 35 × 256 = ~13.7M parameters`
- **Total PLE: ~2.35B parameters** (out of ~5.1B total, but not "computed" per token)

This is why "effective parameters" is ~2.3B — the PLE table is looked up (not computed through matrix multiplications) and does not contribute to FLOPs per token in the same way as attention and FFN.

---

## 7. Vision Encoder

**All four Gemma 4 models are natively multimodal** — the vision encoder is not a plug-in but was trained jointly with the language model.

### Encoder Sizes

| Model | Vision Encoder | Layers | Hidden Dim | Params |
|---|---|---|---|---|
| 31B | ViT | 27 | 1,152 | ~550M |
| 26B-A4B | ViT | 27 | 1,152 | ~550M |
| E4B | ViT | 16 | 768 | ~150M |
| E2B | ViT | 16 | 768 | ~150M |

> Note: Gemma 3 used SigLIP as the vision encoder. Gemma 4 replaces it with a ViT featuring 2D RoPE.

### Key Innovations vs Gemma 3

**2D RoPE for spatial awareness:**
- Independently rotates half the attention head dimensions for the x-axis and the other half for the y-axis
- Enables the model to understand spatial relationships: "above," "below," "left of," "right of"
- Position table stores up to 10,240 positions per axis (handles very large images)
- Each position is a learned vector of the same dimensions as the patch embedding

**Variable aspect ratio support:**
- Unlike standard ViT (which resizes to a fixed square), Gemma 4 preserves the original aspect ratio
- Images are resized so 16×16 pixel patches tile cleanly; minimal padding added where needed
- Supports configurable token budgets: **70, 140, 280, 560, 1,120 tokens per image**
- Default: **280 tokens** (vs 256 in Gemma 3)

**Pooling change:**
- Gemma 3: 4×4 spatial pooling after ViT
- Gemma 4: **3×3 spatial pooling** — finer spatial granularity, slightly more tokens (280 vs 256)

**Bidirectional attention in vision encoder:**
- Vision tokens attend to each other bidirectionally (`use_bidirectional_attention="vision"`)
- Text tokens still use standard causal attention
- This is set in the model config and handled transparently

### Token Budget System

Users can control image resolution vs. token cost:

| Budget | Tokens | Resolution tradeoff |
|---|---|---|
| 70 tokens | Very low | ~7×10 patches — icon-level resolution |
| 140 tokens | Low | ~10×14 patches |
| **280 tokens** | **Default** | ~14×20 patches |
| 560 tokens | High | ~20×28 patches — detailed images |
| 1,120 tokens | Max | ~28×40 patches — very high detail |

### Special Vision Tokens

```
boi_token_id: 255999   # begin of image
eoi_token_id: 258882   # end of image
image_token_id: 258880 # individual image soft token
```

---

## 8. Audio Encoder (E-Series Only)

E2B and E4B include a native audio encoder, making them **true any-to-any models** (text + image + video + audio in, text out).

### Architecture: USM-style Conformer

| Component | Spec |
|---|---|
| Encoder type | Conformer (USM — Universal Speech Model architecture) |
| Number of layers | 12 |
| Attention type | Local (chunked) attention — causal |
| Convolution | Causal Conv1d + depthwise |
| Subsampling | 2-stage Conv2d, 4× temporal reduction |
| Output | Projected to text LLM hidden space |
| Max audio input | Up to 30 seconds |
| Primary tasks | Speech recognition (ASR), speech translation |

### Special Audio Tokens

```
boa_token_id: 256000   # begin of audio
eoa_token_id: 258883   # end of audio
audio_token_id: 258881 # individual audio soft token
```

### Benchmark Performance (from model card)

| Task | E2B | E4B |
|---|---|---|
| CoVoST-2 (translation) | 33.47 BLEU | 35.54 BLEU |
| FLEURS (ASR, WER) | 0.09 | 0.08 |

---

## 9. MoE Architecture Deep Dive

### Why 128 Experts?

Most prominent MoE models use far fewer experts:

| Model | Total Experts | Active per Token | Sparsity |
|---|---|---|---|
| Mixtral 8×7B | 8 | 2 | 25% |
| DeepSeek-V3 | 256 | 8 | 3.1% |
| **Gemma 4 26B-A4B** | **128** | **8** | **6.25%** |
| Qwen MoE | 64 | 8 | 12.5% |

128 experts with top-8 routing = extreme sparsity (6.25% activation). This level of sparsity is normally **fragile** — routing failures or expert collapse are serious risks. Gemma 4 mitigates this with the **always-on dense FFN running in parallel**, which acts as a structural safety net.

### The Parallel Dense + MoE Design

```
# Per-token, per-MoE-layer forward pass:
dense_output  = GeGLU_dense(x)          # hidden=2,112, always active
expert_scores = softmax(x @ router.T)    # shape: (128,)
top8_indices  = argsort(expert_scores)[-8:]
moe_output    = sum(expert_scores[i] * expert_i(x) for i in top8_indices)
layer_output  = (dense_output + moe_output) / sqrt(2)
```

**Why this matters:**
- Dense FFN hidden (2,112) is **3× larger than each expert** (704)
- Even if routing fails for a token (all experts score poorly), the dense FFN provides a reasonable representation
- The dense path also helps gradient flow through experts during training, stabilizing the 128-expert routing

### Expert Size Rationale

Each expert is intentionally small (`hidden=704`). This forces specialization:
- With `hidden=2816` main stream and `hidden=704` per expert, each expert handles a narrow niche
- 128 narrow specialists > 8 generalist experts for capturing diverse knowledge
- The dense FFN handles the common/universal transformations

### Load Balancing

Google has not published details on the auxiliary loss used for expert load balancing. Common practice (used in DeepSeek, Mixtral, etc.) is an auxiliary load-balancing loss during training that penalizes routing distributions that collapse to a few popular experts.

---

## 10. Tokenizer

### Specifications

| Property | Value |
|---|---|
| Type | SentencePiece (BPE with byte fallback) |
| Vocabulary size | 262,144 |
| Prefix space | None |
| Normalizer | Replaces spaces with `▁` |
| `bos_token_id` | 2 |
| `eos_token_id` | 1 |
| `pad_token_id` | 0 |

### Special Token IDs

| Token | ID | Purpose |
|---|---|---|
| `<pad>` | 0 | Padding |
| `<eos>` | 1 | End of sequence |
| `<bos>` | 2 | Beginning of sequence |
| `<boi>` | 255,999 | Begin of image |
| `<boa>` | 256,000 | Begin of audio |
| `<image>` | 258,880 | Image soft token placeholder |
| `<audio>` | 258,881 | Audio soft token placeholder |
| `<eoi>` | 258,882 | End of image |
| `<eoa>` | 258,883 | End of audio |

### Thinking Mode Tokens

Gemma 4 IT models include a native reasoning/thinking mode:

```
<|think|>   — triggers internal chain-of-thought reasoning
```

When `<|think|>` appears at the start of the system prompt, the model generates internal reasoning before the final answer. This is toggleable — removing the token disables thinking mode.

### Chat Template

Gemma 4 introduces native support for the **system role** (Gemma 3 did not support this). Standard roles:

```
system, user, assistant
```

The `apply_chat_template` processor handles all formatting.

---

## 11. Full Hyperparameter Tables

### Architecture Parameters by Model

| Parameter | 31B | 26B-A4B | E2B | Gemma 3 27B (ref) |
|---|---|---|---|---|
| **Total params** | ~31B | ~26B | ~5.1B | 27B |
| **Active params** | 31B | ~3.8B | ~2.3B eff | 27B |
| **hidden_size** | 5,376 | 2,816 | 1,536 | 5,376 |
| **num_hidden_layers** | 60 | 30 | 35 | 62 |
| **layer_pattern** | 5:1 | 5:1 | 4:1 | 5:1 |
| **Sliding layers** | 50 | 25 | 28 | 52 |
| **Full layers** | 10 | 5 | 7 | 10 |
| **num_attention_heads (Q)** | 32 | 16 | 8 | 32 |
| **num_key_value_heads (sliding)** | 16 | 8 | 1 | 16 |
| **num_key_value_heads (full/global)** | 4 | 2 | 1 | 16 |
| **head_dim (sliding)** | 256 | 256 | 256 | 128 |
| **global_head_dim (full)** | 512 | 512 | 512 | 128 |
| **intermediate_size** | 21,504 | 2,112 (dense) | 6,144 / 12,288 | 21,504 |
| **FFN type** | GeGLU | MoE+GeGLU | GeGLU | GeGLU |
| **num_experts** | — | 128 | — | — |
| **num_experts_per_tok** | — | 8 | — | — |
| **expert_hidden_size** | — | 704 | — | — |
| **sliding_window** | 1,024 | 1,024 | 512 | 1,024 |
| **context_window** | 256,000 | 256,000 | 128,000 | 128,000 |
| **vocab_size** | 262,144 | 262,144 | 262,144 | 262,208 |
| **RoPE theta (sliding)** | 10,000 | 10,000 | 10,000 | 10,000 |
| **RoPE theta (full)** | 1,000,000 | 1,000,000 | 1,000,000 | 1,000,000 |
| **partial_rotary_factor (full)** | 0.25 | 0.25 | 0.25 | linear 8× |
| **QK Norm** | RMSNorm | RMSNorm | RMSNorm | RMSNorm |
| **V Norm** | RMSNorm (no scale) | RMSNorm (no scale) | RMSNorm (no scale) | none |
| **K=V sharing (full layers)** | yes | yes | no | no |
| **KV shared layers** | — | — | 20 of 35 | — |
| **PLE dim** | — | — | 256 | — |
| **PLE vocab** | — | — | 262,144 | — |
| **Logit soft-cap** | 30.0 | 30.0 | 30.0 | none |
| **Vision encoder** | ViT 27L d=1152 | ViT 27L d=1152 | ViT 16L d=768 | SigLIP 27L d=1152 |
| **Vision encoder params** | ~550M | ~550M | ~150M | ~400M |
| **Audio encoder** | — | — | Conformer 12L | — |
| **Tie word embeddings** | yes | yes | yes | yes |
| **attention_bias** | false | false | false | false |
| **rms_norm_eps** | 1e-6 | 1e-6 | 1e-6 | 1e-6 |
| **initializer_range** | 0.02 | 0.02 | 0.02 | 0.02 |
| **dtype** | bfloat16 | bfloat16 | bfloat16 | bfloat16 |

---

## 12. Parameter Accounting

### Gemma 4 31B (~30.7B actual)

```
Sliding blocks (50):   50 × 478.9M  = 23.95B
Full blocks (10):      10 × 534.0M  =  5.34B
Embedding table:                     ~1.40B
                               Total = ~30.69B
```

### Gemma 4 26B-A4B (~25.2B total, ~3.8B active)

```
Sliding blocks (25):   25 × 814.1M  = 20.35B (capacity)
Full blocks (5):        5 × 828.5M  =  4.14B (capacity)
Embedding table:                     ~0.74B
                       Total capacity = ~25.23B

Active per token:
Sliding (25):          25 × 100.4M  =  2.51B
Full (5):               5 × 114.8M  =  0.57B
Embedding:                           ~0.74B
                       Total active  =  ~3.82B
```

### Gemma 4 E2B (~5.1B total, ~2.3B effective)

```
Standard blocks (15):  ~35.4M × 15  =  0.53B
Double-wide blocks (20): ~63.7M × 20 =  1.27B
Computed embedding:                  ~0.40B
Per-layer embed table (PLE):         ~2.35B  ← lookup only, not computed
Context projection (PLE):            ~0.014B
Audio encoder:                       ~0.10B
Vision encoder (ViT 150M):           ~0.15B
                       Total         ~4.81B
                       Effective     ~2.3B (excl. PLE lookup table)
```

---

## 13. Memory & Hardware Requirements

### Weight Memory by Precision

| Precision | 31B | 26B-A4B | E4B | E2B |
|---|---|---|---|---|
| BF16 / FP16 | ~62 GB | ~52 GB | ~16 GB | ~10.2 GB |
| INT8 | ~31 GB | ~26 GB | ~8 GB | ~5.1 GB |
| INT4 | ~15.5 GB | ~13 GB | ~4 GB | ~2.6 GB |

> Note: 26B-A4B requires ALL 26B parameters loaded for routing even though only 3.8B are active per token.

### KV Cache Size (FP16, batch=1)

KV formula: `2 × kv_heads × head_dim × 2 bytes × num_layers × seq_len`

For 31B (50 sliding: 16 heads × 256 dim; 10 full: 4 heads × 512 dim):

| Context | 31B | 26B-A4B | E2B | Gemma 3 27B |
|---|---|---|---|---|
| 4K | 3.4 GB | 0.9 GB | 0.07 GB | 1.9 GB |
| 32K | 27.5 GB | 6.9 GB | 0.6 GB | 15.5 GB |
| 128K | 110 GB | 27.5 GB | 2.3 GB | 62 GB |
| 256K | 220 GB | 55 GB | — | — |

The E2B's KV sharing (20 of 35 layers reuse caches) dramatically reduces its footprint — full 128K context fits in ~2.3 GB KV cache.

### Total VRAM Requirements & GPU Recommendations (batch=1)

| Scenario | 31B | 26B-A4B | E2B |
|---|---|---|---|
| FP16, 4K ctx | 65.4 GB — 1× H100 80GB | 52.9 GB — 1× H100 80GB | 10.3 GB — RTX 4070 12GB |
| INT4, 4K ctx | 18.9 GB — RTX 4090 24GB | 13.9 GB — RTX 4070 Ti 16GB | 2.7 GB — Any GPU |
| INT4, 128K ctx | ~125.5 GB — 2× H100 | 40.5 GB — A6000 48GB | 4.9 GB — RTX 4060 8GB |

The 26B-A4B MoE's KV cache is 4× smaller than the 31B at equivalent context, making it practical at long context on a single GPU that the 31B cannot fit.

---

## 14. Benchmark Results

All results are for instruction-tuned variants. Sources: Google DeepMind model card (April 2026), LMArena public leaderboard.

### Core Reasoning & Knowledge

| Benchmark | 31B | 26B-A4B | E4B | E2B |
|---|---|---|---|---|
| **LMArena Score** (text est.) | 1,452 | 1,441 | — | — |
| **MMLU Pro** | 85.2% | 82.6% | 69.4% | 60.0% |
| **GPQA Diamond** (graduate science) | 84.3% | 82.3% | 58.6% | 43.4% |
| **AIME 2026** (math, no tools) | 89.2% | 88.3% | 42.5% | 37.5% |

### Coding

| Benchmark | 31B | 26B-A4B | E4B | E2B |
|---|---|---|---|---|
| **LiveCodeBench v6** | 80.0% | 77.1% | 52.0% | 44.0% |
| **Codeforces ELO** | 2,150 | 1,718 | 940 | 633 |

### Multimodal

| Benchmark | 31B | 26B-A4B | E4B | E2B |
|---|---|---|---|---|
| **MMMU Pro** (vision reasoning) | 76.9% | 73.8% | 52.6% | 44.2% |
| **MATH-Vision** | — | — | — | — |

### Long Context

| Benchmark | 31B | 26B-A4B | E4B | E2B |
|---|---|---|---|---|
| **MRCR v2 8-needle 128K** | 66.4% | 44.1% | 25.4% | 19.1% |

### Agentic / Tool Use

| Benchmark | 31B | 26B-A4B | Gemma 3 27B |
|---|---|---|---|
| **τ2-bench Retail** (agentic tool use) | 86.4% | — | 6.6% |

### Audio (E-Series)

| Benchmark | E4B | E2B |
|---|---|---|
| **CoVoST-2** (speech translation BLEU) | 35.54 | 33.47 |
| **FLEURS** (ASR word error rate) | 0.08 | 0.09 |

### Comparison vs Previous Generation

| Benchmark | Gemma 4 31B | Gemma 3 27B | Delta |
|---|---|---|---|
| LMArena Text | 1,452 | 1,365 | +87 (+6.4%) |
| AIME 2026 | 89.2% | 20.8% | +68.4pp |
| LiveCodeBench v6 | 80.0% | 29.1% | +50.9pp |
| GPQA Diamond | 84.3% | 42.4% | +41.9pp |
| τ2-bench Retail | 86.4% | 6.6% | +79.8pp |

---

## 15. What Changed from Gemma 3

### Summary Table

| Feature | Gemma 3 27B | Gemma 4 31B | Delta |
|---|---|---|---|
| **Attention geometry** | Uniform across all layers | Per-type (sliding: head_dim=256, full: head_dim=512) | Major change |
| **KV heads (full layers)** | 16 | 4 | 4× reduction |
| **K=V weight sharing** | No | Yes (full layers) | New |
| **V normalization** | None | RMSNorm (no scale) | New |
| **RoPE (full layers)** | theta=1M, linear 8× scaling | theta=1M, p-RoPE partial=0.25 | Changed strategy |
| **Rotated dims (full)** | 128 of 128 (100%) | 128 of 512 (25%) | 3× more semantic dims |
| **Semantic capacity** | All dims position-coupled | 384 dims pure semantic | Major long-context improvement |
| **Context window** | 128K | 256K | 2× |
| **Final layer** | Local in some models (e.g. 4B) | Always global | Enforced |
| **Logit soft-capping** | None | tanh(x/30)×30 | New |
| **Vision encoder** | SigLIP 27L, d=1152 | ViT 27L, d=1152, 2D RoPE | Upgraded |
| **Vision tokens** | 256 (4×4 pooling) | 280 (3×3 pooling) | More detail |
| **Per-Layer Embeddings** | None | Yes (E-series only) | New |
| **Audio** | None | Yes (E-series only) | New |
| **MoE variant** | None | 26B-A4B (128 experts) | New |
| **License** | Gemma Terms of Use | Apache 2.0 | More permissive |

---

## 16. Training Details

Google has not released a full technical paper. The following is known from the model card and blog posts.

### Training Data
- **Scale:** Large-scale diverse multilingual corpus
- **Cutoff:** January 2025
- **Modalities in training:** Web documents, code, images, audio
- **Languages:** 140+ languages — balanced representation across European, Asian, African, and Indic language families
- **Data filtering:** Automated filtering for personal information, quality filtering, safety filtering

### Research Lineage
- Gemma 4 is built from **Gemini 3 research** — same foundational architecture and training innovations as Google's proprietary Gemini 3 models, packaged for open release
- This is meaningfully different from prior Gemma generations, which were based on earlier Gemini research

### Post-Training
- **Instruction tuning:** Supervised fine-tuning on instruction-following data covering all supported modalities
- **Thinking mode:** Additional training to enable configurable step-by-step reasoning via `<|think|>` token
- **Function calling:** Structured tool-use training with JSON schema support
- **Language coverage in IT:** Top 40 languages with human-verified evaluations; remaining 100+ via transfer

### Safety
- Same rigorous safety evaluations as Gemini proprietary models
- Partnership with Google DeepMind internal safety and responsible AI teams
- Range of automated + human evaluations
- Sensitive data filtering applied during pre-training

---

## 17. Inference & Deployment

### Framework Support

| Framework | Status |
|---|---|
| HuggingFace Transformers | ✅ Official (`Gemma4ForConditionalGeneration`) |
| vLLM | ✅ Supported (uses TRITON_ATTN backend due to heterogeneous head dims) |
| llama.cpp (GGUF) | ✅ Available — tokenizer fix required post-launch |
| Ollama | ✅ Available |
| MLX (Apple Silicon) | ✅ Available — ~40% less memory than Ollama |
| Google AI Studio | ✅ Hosted |
| Vertex AI | ✅ Hosted |
| Kaggle | ✅ Available |
| LiteRT-LM (mobile) | ✅ E2B/E4B on Android/iOS (INT4) |

### vLLM Specific Notes

```
INFO: Gemma4 model has heterogeneous head dimensions (head_dim=256, global_head_dim=512).
Forcing TRITON_ATTN backend to prevent mixed-backend numerical divergence.
```

vLLM detects the dual head_dim configuration and routes to the Triton attention backend automatically to avoid precision mismatches between the CUDA flash attention implementation (optimized for fixed head_dim) and the heterogeneous Gemma 4 layout.

### Recommended Sampling Configuration

From the official model card:

```python
generation_config = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 64,
}
```

### Thinking Mode Usage

```python
# Enable thinking mode
system_prompt = "<|think|>You are a helpful assistant."

# Disable thinking mode  
system_prompt = "You are a helpful assistant."
```

### Quantization Notes

- **INT4 recommended** for local deployment of 26B-A4B and 31B
- **INT4 + GGUF** community builds available; use tokenizer from latest main branch (bug fix merged post-launch)
- **Unsloth MLX** builds use ~40% less memory than Ollama at ~15-20% throughput cost for E-series

### Hardware Recommendations Summary

| Model | Minimum GPU | Recommended | Production |
|---|---|---|---|
| E2B (INT4) | Any GPU with 3GB VRAM | RTX 3060 8GB | Raspberry Pi 5 viable |
| E4B (INT4) | RTX 3060 8GB | RTX 3080 10GB | M-series Mac |
| 26B-A4B (INT4) | RTX 4080 16GB | RTX 4090 24GB | A6000 48GB for long ctx |
| 31B (INT4) | RTX 4090 24GB | A6000 48GB | H100 80GB for FP16 |

---

## Sources

1. **Google DeepMind Gemma 4 model card** — https://ai.google.dev/gemma/docs/core/model_card_4
2. **HuggingFace Gemma 4 blog** — https://huggingface.co/blog/gemma4
3. **HuggingFace transformers Gemma4 docs** — https://huggingface.co/docs/transformers/model_doc/gemma4
4. **google/gemma-4-31B HuggingFace page** — https://huggingface.co/google/gemma-4-31B
5. **g4.si5.pl — Community architecture reference** (most detailed parameter breakdown)
6. **Maarten Grootendorst — Visual Guide to Gemma 4** — https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-gemma-4
7. **WaveSpeed — Architecture breakdown** — https://wavespeed.ai/blog/posts/what-is-google-gemma-4/
8. **Google DeepMind release blog** — https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/
9. **vLLM GitHub Issue #39133** — config.json dump confirming `num_hidden_layers=60`, `num_attention_heads=32`, `num_key_value_heads=16`, `head_dim=256`, `global_head_dim=512`, `sliding_window=1024`, 50 sliding + 10 full layers
10. **Barbero et al. (2024)** — "Round and Round We Go! What makes Rotary Positional Encodings useful?" — arXiv:2410.06205

---

*Document compiled May 2026. No formal Google DeepMind technical paper has been published for Gemma 4 as of this date. All hyperparameter values are sourced from config.json files, official model cards, and HuggingFace transformers implementation.*
