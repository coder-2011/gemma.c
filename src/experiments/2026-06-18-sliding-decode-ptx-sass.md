# 2026-06-18 - Sliding Paged Decode PTX/SASS Annotation

Note: this is a historical compiler dump from before the decode `cp.async`
ablation was removed from active code. The current production sliding decode
path is the direct-load kernel; prefill FlashAttention still uses its separate
CUTE `cp.async` path.

This note is an annotated compiler dump for the sliding paged decode path in
`src/gemma4_flash_attention.cu`. It focuses on the production direct-load decode
kernel, the `cp.async.cg` ablation, the split-reduction kernel, and the decode
prep-cache kernel that feeds them.

The code shape being inspected is:

```text
gemma4_sliding_decode_q_paged_kv_norm_rope_kernel
  raw Q/K/V + page table -> q_prepared + paged cache K/V

sliding_decode_paged_grouped_split_kernel<false>
  q_prepared + paged K/V -> partial_m, partial_l, partial_acc

sliding_decode_paged_grouped_split_kernel<true>
  same math, but K/V are staged through shared memory with cp.async.cg

sliding_decode_paged_reduce_kernel
  partial_m, partial_l, partial_acc -> final BF16 attention output
```

## Reproduction

Generated from this worktree:

```bash
git rev-parse --short HEAD
# 3c80283

/usr/local/cuda/bin/nvcc --version
# Cuda compilation tools, release 13.0, V13.0.48

OUT=/home/ubuntu/gemma.c/build/analysis/sliding_decode_paged_20260618
mkdir -p "$OUT"

/usr/local/cuda/bin/nvcc \
  -std=c++17 -O3 -arch=sm_86 \
  --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math \
  -D_GLIBCXX_USE_CXX11_ABI=1 \
  -Isrc -Iexperiments/flash-attention/csrc/cutlass/include \
  -lineinfo --source-in-ptx \
  -ptx src/gemma4_flash_attention.cu \
  -o "$OUT/gemma4_flash_attention_sm86.ptx"

/usr/local/cuda/bin/nvcc \
  -std=c++17 -O3 -arch=sm_86 \
  --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math \
  -D_GLIBCXX_USE_CXX11_ABI=1 \
  -Isrc -Iexperiments/flash-attention/csrc/cutlass/include \
  -lineinfo --source-in-ptx -Xptxas=-v \
  -cubin src/gemma4_flash_attention.cu \
  -o "$OUT/gemma4_flash_attention_sm86.cubin" \
  2> "$OUT/ptxas.log"

/usr/local/cuda/bin/nvdisasm \
  --print-line-info --separate-functions --print-code \
  "$OUT/gemma4_flash_attention_sm86.cubin" \
  > "$OUT/gemma4_flash_attention_sm86.sass"

/usr/local/cuda/bin/cuobjdump \
  --dump-resource-usage "$OUT/gemma4_flash_attention_sm86.cubin" \
  > "$OUT/resource_usage.txt"
```

Sidecar dumps:

```text
build/analysis/sliding_decode_paged_20260618/gemma4_flash_attention_sm86.ptx
build/analysis/sliding_decode_paged_20260618/gemma4_flash_attention_sm86.sass
build/analysis/sliding_decode_paged_20260618/gemma4_flash_attention_sm86.cubin
build/analysis/sliding_decode_paged_20260618/ptxas.log
build/analysis/sliding_decode_paged_20260618/resource_usage.txt

build/analysis/sliding_decode_paged_20260618/sliding_decode_split_direct.{ptx,sass}
build/analysis/sliding_decode_paged_20260618/sliding_decode_split_cp_async.{ptx,sass}
build/analysis/sliding_decode_paged_20260618/sliding_decode_reduce.{ptx,sass}
build/analysis/sliding_decode_paged_20260618/sliding_decode_prep.{ptx,sass}
```

CUDA guide grounding used for interpretation:

- Page 67: global memory performance depends heavily on properly coalesced access.
- Pages 130 and 132: async copy is intended to decouple memory-transfer initiation
  from completion so useful work can overlap with the transfer.
- Page 244: cooperative async movement copies global memory into shared memory.
- Page 28: registers, shared memory, and L1 are SM resources, so extra shared memory
  and barriers are not free.
- Page 56: constant memory is useful for small read-only data, but the code here
  mostly uses `__ldg`/read-only global loads rather than actual `__constant__` symbols.

## Kernel Symbol Map

| Logical kernel | Mangled symbol suffix | Meaning |
| --- | --- | --- |
| Direct split decode | `sliding_decode_paged_grouped_split_kernelILb0E...` | `UseCpAsync=false`, direct `__ldg` K/V cache reads. |
| cp.async split decode | `sliding_decode_paged_grouped_split_kernelILb1E...` | `UseCpAsync=true`, global-to-shared K/V staging. |
| Split reduction | `sliding_decode_paged_reduce_kernel...` | Merges split partials into final BF16 output. |
| Decode prep-cache | `gemma4_sliding_decode_q_paged_kv_norm_rope_kernel...` | Normalizes Q/K/V, applies RoPE, writes q-prepared and cache. |

## Resource Summary

From `ptxas.log` and `resource_usage.txt`:

| Kernel | Registers/thread | Static shared | Stack/spills | Notes |
| --- | ---: | ---: | ---: | --- |
| Direct split decode | 32 | 48 B | 0 | Only CUB reduction storage and `s_score`; no local spills. |
| cp.async split decode | 38 | 1072 B | 0 | Adds 1024 B K/V shared staging plus the same reduction storage. |
| Split reduction | 26 | 64 B | 0 | Light register footprint, one CUB block-reduce storage. |
| Decode prep-cache | 40 | 0 B | 0 | Warp-level RMS reductions, many scalar BF16 loads/stores. |

Important immediate read: the cp.async path costs +6 registers/thread and +1024 B
shared memory per CTA. It still waits immediately after issuing the copies, so the
extra resource use has to buy a lot to beat the direct path. The benchmark says it
does not.

## Direct Split Decode: Hot Loop Anatomy

Source hot loop:

```cpp
for (int32_t pos = split_begin; pos < split_end; ++pos) {
  const int32_t page_slot = gemma4_kv_cache_page_slot(config, pos);
  const int32_t physical_page =
      __ldg(page_table + batch * config.max_pages_per_seq + page_slot);
  if (physical_page < 0 || physical_page >= config.num_pages) continue;
  const int32_t page_offset = gemma4_kv_cache_page_offset(config, pos);
  const int64_t kv_base = gemma4_kv_cache_offset(config, layer, physical_page, page_offset, kv_head, 0);

  k_value = __bfloat162float(loadg(cache_k + kv_base + dim));
  v_value = __bfloat162float(loadg(cache_v + kv_base + dim));
  ...
}
```

### Page table load and page offset

PTX:

```ptx
// physical_page = __ldg(page_table + batch * max_pages_per_seq + page_slot)
cvt.s64.s32     %rd52, %r50;
add.s64         %rd53, %rd52, %rd4;
shl.b64         %rd54, %rd53, 2;
add.s64         %rd51, %rd16, %rd54;
ld.global.nc.s32 %r48, [%rd51];

// if invalid, continue
setp.lt.s32     %p6, %r48, 0;
setp.ge.s32     %p7, %r48, %r100;
or.pred         %p8, %p6, %p7;
@%p8 bra        $L__BB8_19;

// page_offset = position % config.page_size
rem.s32         %r67, %r101, %r18;
```

SASS:

```sass
/*0570*/ IABS R22, c[0x0][0x1a8] ;        // page_size absolute value
/*05b0*/ MUFU.RCP R11, R11 ;               // reciprocal sequence for division/mod
/*05e0*/ F2I.FTZ.U32.TRUNC.NTZ R9, R8 ;
/*0630*/ IMAD.HI.U32 R9, R9, R23, R8 ;
...
/*0860*/ SHF.R.S32.HI R8, RZ, 0x1f, R9 ;
/*0890*/ LEA R10, P0, R9, c[0x0][0x190], 0x2 ;
/*08b0*/ LDG.E.CONSTANT R10, [R10.64] ;    // page_table load from __ldg
```

Annotation:

- The page table access is a read-only global load in PTX: `ld.global.nc.s32`.
- `nvdisasm` prints the SASS as `LDG.E.CONSTANT`. This is the SASS read-only/global
  load form emitted for the `__ldg` path in this compile, not proof that the source
  pointer lives in `__constant__` memory.
- The hot loop pays a lot of integer instructions for `page_slot` and `page_offset`.
  The `rem.s32` in PTX lowers to reciprocal/multiply/fixup SASS because `page_size`
  is a runtime config field.
- With `page_size=64` and `split_size=64`, this work is mostly avoidable. A
  page-chunked loop should load the page table once per page and replace `% page_size`
  with a linear offset increment inside the page.

### Direct K/V cache loads

PTX:

```ptx
// k_value = __bfloat162float(loadg(cache_k + kv_base + dim))
add.s64          %rd66, %rd65, %rd3;
shl.b64          %rd67, %rd66, 1;
add.s64          %rd55, %rd14, %rd67;
ld.global.nc.b16 %rs5, [%rd55];
{ mov.b32 %f39, {0,%rs5};}

// v_value = __bfloat162float(loadg(cache_v + kv_base + dim))
add.s64          %rd56, %rd15, %rd67;
ld.global.nc.b16 %rs7, [%rd56];
{ mov.b32 %f40, {0,%rs7};}
```

Representative SASS shape:

```sass
/*0a90*/ IMAD R11, R8, c[0x0][0x1b4], RZ ;
/*0aa0*/ IMAD.WIDE.U32 R8, R10, c[0x0][0x1b4], R16 ;
/*0ab0*/ IMAD R25, R10, UR5, R11 ;
...
/*0ad0*/ LDG.E.U16.CONSTANT R10, [R10.64] ;    // K or V BF16 load
/*0b00*/ LDG.E.U16.CONSTANT R25, [R8.64] ;     // matching K/V load
/*0b20*/ PRMT R29, RZ, 0x5410, R10 ;           // BF16 -> FP32 lane packing
```

Annotation:

- The useful payload per thread per token is tiny: one BF16 K and one BF16 V.
- Coalescing is good: 256 threads cover a contiguous 256-element BF16 vector for a
  single KV head and position.
- The direct path has no shared-memory roundtrip. It loads the BF16 and immediately
  promotes it to FP32 in registers.
- Because each CTA handles two Q heads for one KV head, the same K/V load feeds two
  scores and two online softmax accumulators. That GQA grouping is the strongest
  cache/data-reuse feature in this kernel.

## cp.async Split Decode: What Actually Gets Emitted

Source:

```cpp
if (threadIdx.x < GEMMA4_SLIDING_HEAD_DIM / kBf16Packed128Elements) {
  const int32_t vec_dim = threadIdx.x * kBf16Packed128Elements;
  sliding_decode_cp_async_16(cp_async_storage.s_k + vec_dim, cache_k + kv_base + vec_dim);
  sliding_decode_cp_async_16(cp_async_storage.s_v + vec_dim, cache_v + kv_base + vec_dim);
}
asm volatile("cp.async.commit_group;\n" ::);
asm volatile("cp.async.wait_group 0;\n" ::);
__syncthreads();
k_value = __bfloat162float(cp_async_storage.s_k[dim]);
v_value = __bfloat162float(cp_async_storage.s_v[dim]);
```

PTX:

```ptx
// shared declarations
.shared .align 16 .b8 cp_async_storage[1024];
.shared .align 4  .b8 reduce_storage[44];
.shared .align 4  .f32 s_score;

// K and V copies, 16B each from the first 32 threads
cp.async.cg.shared.global [%r11], [%rd55], 16;
cp.async.cg.shared.global [%r12], [%rd56], 16;

cp.async.commit_group;
cp.async.wait_group 0;
bar.sync 0;

ld.shared.u16 %rs5, [%r13];
ld.shared.u16 %rs6, [%r13+512];
```

SASS:

```sass
/*0b30*/ @!PT LDS RZ, [RZ] ;                  // scheduling / dependency padding
/*0b40*/ @!PT LDS RZ, [RZ] ;
/*0b50*/ @!PT LDS RZ, [RZ] ;
/*0b60*/ LDGSTS.E.BYPASS.128 [R6], [R10.64] ; // cp.async K: global -> shared
/*0b70*/ LDGSTS.E.BYPASS.128 [R6+0x200], [R8.64] ; // cp.async V
/*0b90*/ LDGDEPBAR ;
/*0bc0*/ DEPBAR.LE SB0, 0x0 ;
/*0bd0*/ BAR.SYNC.DEFER_BLOCKING 0x0 ;
/*0be0*/ LDS.U16 R8, [R7] ;                   // read K back out of shared
/*0bf0*/ LDS.U16 R31, [R7+0x200] ;            // read V back out of shared
```

Annotation:

- The PTX is exactly the intended cache operator: `cp.async.cg.shared.global`.
- On SM86 SASS this lowers to `LDGSTS.E.BYPASS.128`, which is the global-to-shared
  movement instruction family.
- The code immediately emits wait/dependency/barrier machinery before consuming shared:
  `LDGDEPBAR`, `DEPBAR.LE`, and `BAR.SYNC`.
- This means the async copy is not hiding latency behind independent math. It is
  acting mostly like a more complicated direct load, plus a shared write, plus a
  shared read, plus barriers.
- This explains the measured result: direct decode was faster than `cp.async.cg`.
  The guide says async copies are useful when transfer initiation is decoupled from
  completion and overlapped with useful work; this path waits immediately.

## Block Reductions Inside Split Decode

For each token position, the split kernel computes two dot products:

```cpp
float score0 = DecodeBlockReduce(reduce_storage).Sum(q0 * k_value);
...
float score1 = DecodeBlockReduce(reduce_storage).Sum(q1 * k_value);
```

PTX shape:

```ptx
mul.ftz.f32 %f43, %f31, %f39;  // q0 * k

shfl.sync.down.b32 ...;        // warp reduce step 1
shfl.sync.down.b32 ...;        // step 2
shfl.sync.down.b32 ...;        // step 4
shfl.sync.down.b32 ...;        // step 8
shfl.sync.down.b32 ...;        // step 16

st.shared.f32 [reduce_storage + warp_slot], %f121;
bar.sync 0;
ld.shared.f32 ...;             // warp-aggregate fold in warp 0

st.shared.f32 [s_score], %f120;
bar.sync 0;
ld.shared.f32 %f14, [s_score];
bar.sync 0;
```

SASS shape:

```sass
/*0c10*/ FMUL.FTZ R9, R29, R8.reuse ;
/*0c30*/ SHFL.DOWN P0, R10, R9, 0x1, 0x1f ;
/*0c40*/ @P0 FADD R10, R9, R10 ;
/*0c50*/ SHFL.DOWN P0, R11, R10, 0x2, 0x1f ;
/*0c60*/ @P0 FADD R11, R10, R11 ;
/*0c70*/ SHFL.DOWN P0, R24, R11, 0x4, 0x1f ;
/*0c80*/ @P0 FADD R24, R11, R24 ;
/*0c90*/ SHFL.DOWN P0, R25, R24, 0x8, 0x1f ;
/*0ca0*/ @P0 FADD R25, R24, R25 ;
/*0cc0*/ SHFL.DOWN P2, R33, R25, 0x10, 0x1f ;
/*0cd0*/ @P2 FADD R33, R25, R33 ;
/*0ce0*/ @!P1 STS [R28.X4+0x408], R33 ;
/*0cf0*/ BAR.SYNC.DEFER_BLOCKING 0x0 ;
```

Annotation:

- CUB lowers to warp shuffle reductions first, then shared memory for inter-warp
  aggregation.
- There are two full block reductions per token position. That means ten shuffle
  stages per token for the two Q heads, plus CUB shared traffic and barriers.
- The split kernel uses one `s_score` scalar broadcast per score. That introduces
  additional `STS`/`LDS` and barriers.
- A hand-specialized two-score reduction could probably share more structure between
  `score0` and `score1`, but the current CUB lowering is clean and spill-free.

## Online Softmax Update

Source:

```cpp
const float new_m = fmaxf(m, score);
const float old_scale = __expf(m - new_m);
const float new_scale = __expf(score - new_m);
acc = acc * old_scale + v_value * new_scale;
l = l * old_scale + new_scale;
m = new_m;
```

PTX:

```ptx
max.ftz.f32        %f21, %f124, %f99;
sub.ftz.f32        %f107, %f124, %f21;
mul.ftz.f32        %f108, %f107, 0f3FB8AA3B; // log2(e)
ex2.approx.ftz.f32 %f109, %f108;
sub.ftz.f32        %f110, %f99, %f21;
mul.ftz.f32        %f111, %f110, 0f3FB8AA3B;
ex2.approx.ftz.f32 %f112, %f111;
mul.ftz.f32        %f113, %f40, %f112;
fma.rn.ftz.f32     %f126, %f126, %f109, %f113;
fma.rn.ftz.f32     %f123, %f123, %f109, %f112;
```

SASS:

```sass
/*0fa0*/ FMNMX.FTZ R22, R15, R10, !PT ;
/*0fb0*/ FADD.FTZ R10, R10, -R22.reuse ;
/*0fc0*/ FADD.FTZ R9, R15, -R22 ;
/*0fd0*/ FMUL.FTZ R23, R10, 1.4426950216293334961 ;
/*0fe0*/ FMUL.FTZ R10, R9, 1.4426950216293334961 ;
/*1000*/ MUFU.EX2 R13, R10 ;
/*1020*/ MUFU.EX2 R23, R23 ;
/*1040*/ FFMA.FTZ R16, R16, R13, R23 ;
/*1050*/ FFMA.FTZ R14, R14, R13, R15 ;
```

Annotation:

- `__expf` lowers to base-2 exponentials using the `log2(e)` multiplier and
  `MUFU.EX2`.
- The update stays in registers until final partial stores.
- This is the right shape for CUDA cores. There is no tensor-core instruction here,
  which is what we want for `q_len=1`.

## Split Partial Stores

PTX:

```ptx
// thread 0 writes scalar split state for both query heads
st.global.f32 [%rd70], %f125;  // partial_m[partial0]
st.global.f32 [%rd72], %f122;  // partial_l[partial0]
st.global.f32 [%rd74], %f124;  // partial_m[partial1]
st.global.f32 [%rd75], %f123;  // partial_l[partial1]

// all 256 threads write one accumulator element per q-head
st.global.f32 [%rd80], %f127;  // partial_acc[partial0, dim]
st.global.f32 [%rd84], %f126;  // partial_acc[partial1, dim]
```

SASS:

```sass
/*1270*/ STG.E [R8.64], R13 ;
/*1280*/ STG.E [R10.64], R17 ;
/*1290*/ STG.E [R18.64], R15 ;
/*12a0*/ STG.E [R4.64], R16 ;
/*12c0*/ STG.E [R2.64], R12 ;
/*12d0*/ STG.E [R6.64], R14 ;
```

Annotation:

- The scalar `m/l` stores are small. The large traffic is `partial_acc`.
- For `batch=1`, `q_heads=32`, `num_splits=16`, `head_dim=256`, `partial_acc`
  writes `1 * 32 * 16 * 256 * 4 = 524288` bytes, then the reduce kernel reads it.
- This is the biggest structural memory roundtrip left in the split design.

## Split Reduction Kernel

The reduce kernel has three phases:

1. Read `partial_m` and reduce max.
2. Read `partial_l` and `partial_m`, reduce denominator.
3. Read `partial_acc`, `partial_m`, and `partial_l`, write BF16 output.

PTX phase 1 excerpt:

```ptx
ld.global.nc.f32 %f44, [%rd26];  // partial_m[row, split]
max.ftz.f32      %f134, %f134, %f44;

shfl.sync.down.b32 ...;
max.ftz.f32 ...;
st.shared.f32 ...;
bar.sync 0;
ld.shared.f32 ...;
```

SASS phase 1 excerpt:

```sass
/*00f0*/ IMAD.WIDE R4, R4, R5, c[0x0][0x168] ;
/*0100*/ LDG.E.CONSTANT R4, [R4.64] ;
/*0130*/ FMNMX.FTZ R9, R4, R9, !PT ;
...
/*0160*/ SHFL.DOWN PT, R0, R9, 0x1, 0x1f ;
/*01e0*/ @!P1 FMNMX.FTZ R4, R9, R0, !PT ;
...
/*0310*/ BAR.SYNC.DEFER_BLOCKING 0x0 ;
/*0330*/ LDS R8, [0xc] ;
/*0340*/ LDS.128 R4, [0x10] ;
/*0350*/ LDS.64 R10, [0x20] ;
```

SASS denominator phase:

```sass
/*04c0*/ LDG.E.CONSTANT R11, [R4.64] ;       // partial_l
/*0550*/ LDG.E.CONSTANT R4, [R4.64] ;        // partial_m
/*0570*/ FMUL.FTZ R9, R9, 1.4426950216293334961 ;
/*0580*/ MUFU.EX2 R9, R9 ;
/*0590*/ FFMA.FTZ R6, R11, R9, R6 ;
...
/*06b0*/ BAR.SYNC.DEFER_BLOCKING 0x0 ;
/*06d0*/ LDS R0, [0xc] ;
/*06e0*/ LDS.128 R4, [0x10] ;
/*06f0*/ LDS.64 R10, [0x20] ;
```

Annotation:

- The reduce kernel is light on registers and has no spills, but it exists because
  the split kernel materializes `partial_acc` globally.
- The `partial_m/l` traffic is small and cache-friendly. The `partial_acc` traffic
  is the meaningful cost.
- For small sliding windows, a no-split path can avoid this entire kernel and the
  global partial roundtrip.

## Decode Prep-Cache Kernel

Source role:

```text
raw Q/K/V -> RMSNorm -> RoPE for Q/K -> q_prepared + cache_k/cache_v stores
```

### Token position and raw Q loads

PTX:

```ptx
// position = __ldg(token_position + batch)
ld.global.nc.s32 %r18, [%rd20];

// lane owns 8 values of a 256-wide head
ld.global.nc.u16 %rs1, [%rd36];
ld.global.nc.u16 %rs2, [%rd36+64];
ld.global.nc.u16 %rs3, [%rd36+128];
ld.global.nc.u16 %rs4, [%rd36+192];
ld.global.nc.u16 %rs5, [%rd36+256];
ld.global.nc.u16 %rs6, [%rd36+320];
ld.global.nc.u16 %rs7, [%rd36+384];
ld.global.nc.u16 %rs8, [%rd36+448];

fma.rn.ftz.f32 %f34, %f1, %f1, 0f00000000;
...
shfl.sync.bfly.b32 ...;
rsqrt.approx.ftz.f32 %f55, %f54;
```

SASS:

```sass
/*0190*/ LDG.E.U16.CONSTANT R0, [R4.64] ;
/*01a0*/ LDG.E.U16.CONSTANT R35, [R4.64+0x40] ;
/*01b0*/ LDG.E.U16.CONSTANT R11, [R4.64+0x80] ;
/*01c0*/ LDG.E.U16.CONSTANT R32, [R4.64+0xc0] ;
/*01d0*/ LDG.E.U16.CONSTANT R33, [R4.64+0x100] ;
/*01e0*/ LDG.E.U16.CONSTANT R17, [R4.64+0x140] ;
/*01f0*/ LDG.E.U16.CONSTANT R34, [R4.64+0x180] ;
/*0200*/ LDG.E.U16.CONSTANT R36, [R4.64+0x1c0] ;
/*02e0*/ FFMA.FTZ R8, R0, R0, RZ ;
...
/*03c0*/ SHFL.BFLY PT, R5, R8, 0x10, 0x1f ;
...
/*0520*/ MUFU.RSQ R31, R31 ;
```

Annotation:

- The lane-strided 8-load pattern is visible directly in PTX and SASS.
- Warp-level RMS reduction lowers to `SHFL.BFLY` plus `FADD`.
- `rsqrtf` lowers to `MUFU.RSQ`.
- There is no shared memory in this kernel. That is good for occupancy and avoids
  barriers.

### Norm weights, RoPE tables, packing, and stores

SASS excerpt:

```sass
/*04a0*/ LDG.E.U16.CONSTANT R23, [R12.64] ;      // norm weight
/*0510*/ LDG.E.CONSTANT R16, [R14.64] ;          // cos/sin float load
/*0530*/ LDG.E.CONSTANT R9, [R14.64+0x100] ;
/*0540*/ LDG.E.CONSTANT R10, [R14.64+0x180] ;
/*0550*/ LDG.E.CONSTANT R8, [R14.64+0x80] ;
...
/*07a0*/ FMUL.FTZ R5, R14.reuse, R9 ;
/*07b0*/ FFMA.FTZ R13, R14, R11, R13 ;
/*07d0*/ FFMA.FTZ R33, -R33, R17, R4 ;
...
/*0850*/ F2FP.BF16.PACK_AB R0, R0, R33 ;
/*0860*/ F2FP.BF16.PACK_AB R4, R35, R4 ;
/*0870*/ F2FP.BF16.PACK_AB R13, R13, R34 ;
/*0880*/ F2FP.BF16.PACK_AB R14, R31, R14 ;
...
/*08d0*/ STG.E.U16 [R22.64], R0 ;
/*08e0*/ STG.E.U16 [R22.64+0x100], R5 ;
/*08f0*/ STG.E.U16 [R22.64+0x40], R4 ;
/*0900*/ STG.E.U16 [R22.64+0x140], R21 ;
```

Annotation:

- Norm weights and RoPE cos/sin are read-only global loads. They are small and reused
  across heads, so they are good candidates for cache-policy experiments.
- Constant memory is not automatically a win because lanes access different dimensions;
  the guide's constant-memory advice is best for small read-only data, especially when
  access is broadcast-like. Here, read-only global cache or shared row staging may be
  better than `__constant__`.
- The compiler uses `F2FP.BF16.PACK_AB` before stores, so two FP32 values are packed
  into BF16 lanes. The following `STG.E.U16` stores write the prepared/cache vectors.

## Instruction-Level Takeaways

| Finding | Evidence in dump | Optimization consequence |
| --- | --- | --- |
| Direct K/V path is clean and coalesced. | PTX `ld.global.nc.b16`; SASS `LDG.E.U16.CONSTANT` plus no shared staging. | Keep direct path as production default. Benchmark `ld.global.cg` only as a narrow cache-policy variant. |
| cp.async path is real `.cg`, but not overlapped. | PTX `cp.async.cg.shared.global`; SASS `LDGSTS.E.BYPASS.128`, then `LDGDEPBAR`, `DEPBAR.LE`, `BAR.SYNC`, then `LDS`. | Keep as ablation. It needs double-buffering or more reuse to matter. |
| Runtime page modulo is expensive. | PTX `rem.s32`; SASS reciprocal/fixup sequence with `MUFU.RCP`, `F2I`, `IMAD.HI`, predicates. | Page-chunked loop should remove repeated modulo and repeated page-table lookup. |
| CUB reductions are spill-free but barrier-heavy. | Repeated `SHFL.DOWN`, `STS`, `BAR.SYNC`, `LDS`. | A custom two-score reduction may reduce synchronization, but this is lower priority than page-chunking and partial-acc roundtrip. |
| Split reduction is mostly a global-memory artifact. | Separate kernel reads `partial_m/l/acc` and writes final BF16 output. | Add no-split or fused reduce path for common sliding-window decode. |
| Decode prep has no shared memory and no spills. | Resource usage: 40 regs, 0 shared, 0 spills. | Good baseline. Cache experiments should target RoPE/norm read reuse, not add broad shared staging blindly. |

## What I Would Change Next

1. Rewrite the split loop from `for pos` to `for page span`.
   - Load `physical_page` once for a page.
   - Compute `page_offset` once at span start.
   - Increment `kv_base` linearly for positions inside the page.
   - This attacks both metadata traffic and the modulo/division SASS.

2. Keep direct K/V loads as the production path.
   - The dump explains why the cp.async ablation lost: the copy is immediately waited on.
   - Only revisit cp.async with a real two-buffer pipeline.

3. Specialize `num_splits == 1` or `key_count <= 1024`.
   - Avoid writing and rereading `partial_acc` when one CTA can own the whole sliding
     window acceptably.

4. Try a small decode-prep cache experiment for RoPE rows.
   - The same cos/sin position row is reused across heads.
   - Shared row staging might help, but only benchmarked evidence should promote it.

5. Consider a hand-rolled split dot reducer only after the above.
   - CUB is not spilling, which is good.
   - The barrier/shared pattern is visible, but not obviously the first bottleneck.
