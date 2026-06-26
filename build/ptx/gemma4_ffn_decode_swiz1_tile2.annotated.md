# Annotated PTX Walkthrough: `gemma4_ffn_decode_fused_bf16_kernel`

Artifact being annotated:

- PTX: `build/ptx/gemma4_ffn_decode_swiz1_tile2.ptx`
- SASS: `build/ptx/gemma4_ffn_decode_swiz1_tile2.sass`
- Compile mode: `GEMMA4_FFN_DECODE_SWIZZLE_X=1`, `GEMMA4_FFN_DECODE_ACT_TILE=2`, `sm_86`

This is the current default FFN decode kernel. It is a single-token fused FFN path:

1. Load hidden vector `x[5376]` into shared memory as 672 packed 128-bit BF16 chunks.
2. For each CTA-owned intermediate-column tile, compute two gate/up columns at a time.
3. Reduce gate/up dot products across the CTA.
4. Compute `gate * GELU_tanh(up)` for each of the two columns.
5. Stream two corresponding down-projection rows and accumulate eight hidden outputs per active thread.
6. Use the global scratch buffer to serialize/accumulate all intermediate CTAs.
7. In the last CTA, add residual, store residual output, compute RMS sum, compute scale, and store normalized output.

The CUDA guide grounding that matters:

- Registers are per-thread storage managed by the compiler; register pressure affects how many blocks/warps can reside on an SM. CUDA Programming Guide pages 28 and 55.
- Shared memory is block-scoped, allocated per thread block, and can have bank-conflict sensitivity depending on access pattern. CUDA Programming Guide pages 67-69.
- Global-to-shared async/TMA style copies can reduce register usage by avoiding explicit register staging, but this kernel currently uses normal vector global loads followed by shared stores. CUDA Programming Guide page 302.

## 1. Header, Target, Entry, Register Declarations

PTX lines 9-36:

```ptx
.version 8.8
.target sm_86
.address_size 64

.entry ...gemma4_ffn_decode_fused_bf16_kernel(...)
.maxntid 1024, 1, 1
.minnctapersm 1
{
  .reg .pred %p<96>;
  .reg .b16  %rs<97>;
  .reg .f32  %f<539>;
  .reg .b32  %r<370>;
  .reg .b64  %rd<67>;
```

This is PTX, not final machine code. The `.reg` numbers are **virtual PTX registers**, not the final ptxas allocation. The final ptxas result for this kernel is:

```text
Used 50 registers, 11020 bytes smem, no spills
```

So `%f<539>` does not mean 539 physical FP32 registers per thread. It means NVVM named 539 virtual float temporaries in PTX. ptxas later performs liveness analysis, scheduling, coalescing, and physical register allocation. The SASS confirms the final physical peak is around `R47`, matching the ptxas `50` register report after accounting for special/internal allocation.

The launch bounds:

```ptx
.maxntid 1024, 1, 1
.minnctapersm 1
```

come from:

```cpp
__global__ __launch_bounds__(kFfnThreads, 1)
```

That tells ptxas the kernel is launched with 1024 threads per block and at least one resident block per SM is the target. Since 1024 threads is already the architectural max per CTA on this target, this kernel is occupancy-limited by block size first. Register count still matters because too many registers could make the kernel fail launch or reduce schedulability on different shapes/targets, but this kernel is already `1 CTA/SM` by launch shape.

## 2. Shared Memory Declarations

PTX lines 38-44:

```ptx
.shared .align 16 .b8 s_x[10752];
.shared .align 4  .b8 s_reduce_warp_sums[256];
.shared .align 4  .b8 s_act[8];
.shared .align 4  .f32 s_scale;
```

Mapping back to CUDA:

```cpp
__shared__ FfnBf16Pack s_x[kHiddenPacks];
__shared__ float s_reduce_warp_sums[2][kFfnWarps];
__shared__ float s_act[kActTile];
__shared__ float s_scale;
```

Sizes:

- `s_x[10752]`: 672 chunks * 16 bytes = 10,752 bytes. This is the hidden vector `[5376]` in BF16, staged once per CTA.
- `s_reduce_warp_sums[256]`: 2 arrays * 32 warps * 4 bytes = 256 bytes. This is the gate/up pair-reduction scratch.
- `s_act[8]`: tile 2 * 4 bytes = 8 bytes.
- `s_scale`: one float = 4 bytes.

The total reported shared memory is 11,020 bytes. That is the above data plus compiler/runtime alignment details.

## 3. Parameter Loading

PTX lines 46-48 load only the parameters immediately needed at the top:

```ptx
ld.param.u64 %rd10, [param_2];  // x
ld.param.u64 %rd13, [param_5];  // w_gate_up_col_major
ld.param.u64 %rd14, [param_6];  // w_down_row_major
mov.u32      %r1, %tid.x;
setp.gt.s32 %p1, %r1, 671;
@%p1 bra    $L__BB0_7;
```

Parameter mapping from the CUDA signature:

```text
param_0 = residual_out
param_1 = normed_out
param_2 = x
param_3 = residual
param_4 = rms_weight
param_5 = w_gate_up_col_major
param_6 = w_down_row_major
param_7 = scratch
param_8 = eps
```

`%r1 = threadIdx.x`.

`setp.gt.s32 %p1, %r1, 671` computes:

```cpp
owns_hidden_pack = threadIdx.x < kHiddenPacks;  // kHiddenPacks = 672
```

Threads `0..671` own one hidden pack and participate in data-owning work. Threads `672..1023` still participate in barriers/reductions but do not own hidden output packs. That is why `%p1` is used repeatedly to skip memory work while still letting all 1024 threads reach `bar.sync`.

## 4. Shared `x` Preload With Chunk Swizzle

This corresponds to:

```cpp
for (int pack = threadIdx.x; pack < kHiddenPacks; pack += kFfnThreads) {
  s_x[shared_x_chunk_index(pack)] =
      Bf16Packed128{*reinterpret_cast<const int4 *>(x + pack * kBf16Packed128Elements)};
}
__syncthreads();
```

Because `kHiddenPacks = 672` and `kFfnThreads = 1024`, only threads `0..671` execute one useful iteration. The PTX still has generalized loop code because the compiler lowers the source loop structurally.

The core swizzle math appears at lines 70-79:

```ptx
shr.u32   %r37, %r363, 3;      // row-ish index = chunk >> 3
and.b32   %r38, %r37, 7;       // row mod 8
xor.b32   %r39, %r38, %r363;   // chunk ^ (row mod 8)
shl.b32   %r40, %r39, 4;       // byte offset = swizzled_chunk * 16
mov.u32   %r41, s_x;
add.s32   %r42, %r41, %r40;
ld.global.nc.v4.s32 {%r33,%r34,%r35,%r36}, [%rd16];
st.shared.v4.u32 [%r42], {%r33, %r34, %r35, %r36};
```

This is exactly:

```cpp
u = chunk;
row = (u >> 3) & 7;
swizzled_chunk = (u & ~7) | ((u & 7) ^ row);
byte_offset = swizzled_chunk * 16;
```

Why `xor.b32 %r39, %r38, %r363` works despite not explicitly masking `col`:

`%r38` only has bits `0..2` populated. XORing it into the whole chunk only flips those low three bits. Higher bits are unchanged. That is equivalent to:

```cpp
(u & ~7) | ((u & 7) ^ row)
```

The global load:

```ptx
ld.global.nc.v4.s32
```

is a 128-bit global load. The `.nc` comes from the project load helper using the read-only/non-coherent path for `x`. The shared store:

```ptx
st.shared.v4.u32
```

writes the 16-byte pack into shared memory.

The bulk block at `$L__BB0_6` is the same preload shape, but ptxas has unrolled it four chunks at a time:

```ptx
ld.global.nc.v4.s32 ... ; st.shared.v4.u32 ...
ld.global.nc.v4.s32 ... ; st.shared.v4.u32 ...
ld.global.nc.v4.s32 ... ; st.shared.v4.u32 ...
ld.global.nc.v4.s32 ... ; st.shared.v4.u32 ...
```

This is the region that changed register pressure. With swizzle off, ptxas recognizes a simple linear shared-memory destination stream and schedules a much larger preload cluster. With swizzle on, the destination address depends on XOR chunk mapping, so ptxas keeps fewer independent payloads and addresses live at once. The result is lower physical register allocation.

At line 157:

```ptx
bar.sync 0;
```

This is the `__syncthreads()` after loading `s_x`. It guarantees all threads see the shared `x` data before any thread enters the gate/up dot loop.

## 5. Outer Activation-Tile Loop Setup

PTX lines 156-191:

```ptx
bar.sync 0;
shl.b32     %r87, %r1, 3;
cvt.s64.s32 %rd1, %r87;
mov.u32     %r368, 0;
mov.f32     %f120, 0;
mov.u32     %r88, %ctaid.x;
shl.b32     %r89, %r88, 8;
mov.f32     %f520..%f527, %f120;
```

Meaning:

- `%rd1 = threadIdx.x * 8`, the hidden element offset for this thread's pack.
- `%r368 = local_col`, initialized to 0.
- `%r88 = blockIdx.x`.
- `%r89 = blockIdx.x * 256`, the start intermediate column for this CTA.
- `%f520..%f527` are the eight FP32 output accumulators for this thread's hidden pack.

This is the CUDA state:

```cpp
float partial[8] = {};
intermediate_begin = blockIdx.x * 256;
hidden_col = threadIdx.x * 8;

for (local_col = 0; local_col < 256; local_col += 2) {
  float gate[2] = {};
  float up[2] = {};
  ...
}
```

The accumulator ordering looks odd in PTX (`%f520..%f527`) because the compiler chooses its own register assignment. Semantically these are `partial[0..7]`.

## 6. Gate/Up Dot Loop For Two Columns

The dot-loop hot block starts at `$L__BB0_10`, around PTX lines 193-540.

It begins by computing the current intermediate columns:

```ptx
add.s32      %r90, %r368, %r89;     // intermediate_col0 = local_col + blockIdx.x*256
add.s32      %r91, %r90, 21504;     // up_col0 = gate_col0 + intermediate_size
mul.wide.s32 %rd3, %r91, 5376;      // up column 0 base offset
add.s32      %r92, %r90, 1;
mul.wide.s32 %rd4, %r92, 5376;      // gate column 1 base offset
add.s32      %r93, %r90, 21505;
mul.wide.s32 %rd5, %r93, 5376;      // up column 1 base offset
```

The current gate column 0 base is recomputed inside the loop:

```ptx
add.s32      %r359, %r368, %r89;
mul.wide.s32 %rd66, %r359, 5376;
```

That is:

```cpp
gate_col0 = w_gate_up_col_major + intermediate_col0 * 5376;
up_col0   = w_gate_up_col_major + (21504 + intermediate_col0) * 5376;
gate_col1 = w_gate_up_col_major + (intermediate_col0 + 1) * 5376;
up_col1   = w_gate_up_col_major + (21504 + intermediate_col0 + 1) * 5376;
```

The shared activation load repeats the same chunk swizzle:

```ptx
shr.u32          %r110, %r369, 3;
and.b32          %r111, %r110, 7;
xor.b32          %r112, %r111, %r369;
shl.b32          %r113, %r112, 4;
add.s32          %r115, s_x, %r113;
ld.shared.v4.u32 {%r116,%r117,%r118,%r119}, [%r115];
```

Then the PTX splits each 32-bit word into two BF16 halves:

```ptx
mov.b32 {%rs49, %rs50}, %r116;
mov.b32 {%rs53, %rs54}, %r117;
mov.b32 {%rs57, %rs58}, %r118;
mov.b32 {%rs61, %rs62}, %r119;
```

`%rs*` are 16-bit virtual registers. A 128-bit `Bf16Packed128` holds 8 BF16 values. Each `.b32` register contains two BF16 lanes, so four `.b32` registers become eight `.b16` registers.

The BF16-to-FP32 conversion is emitted as inline asm:

```ptx
{ mov.b32 %f129, {0,%rs49};}
```

This constructs a FP32 bit pattern whose high 16 bits are the BF16 value and whose low 16 bits are zero. That is the normal cheap BF16-to-FP32 widening trick.

The weight loads are streaming 128-bit loads:

```ptx
ld.global.cs.v4.s32 {%r94,%r95,%r96,%r97}, [%rd26];   // gate col 0 pack
ld.global.cs.v4.s32 {%r98,%r99,%r100,%r101}, [%rd27]; // up col 0 pack
...
ld.global.cs.v4.s32 {%r102,%r103,%r104,%r105}, [%rd28]; // gate col 1 pack
ld.global.cs.v4.s32 {%r106,%r107,%r108,%r109}, [%rd29]; // up col 1 pack
```

`.cs` is the cache streaming hint from the old streaming pack-load path. That matches the access pattern: each weight vector is streamed once for this token/tile and not reused heavily by this CTA.

The dot product is a chain of `fma.rn.f32`:

```ptx
fma.rn.f32 %f193, %f129, %f131, %f485;
fma.rn.f32 %f194, %f130, %f132, %f193;
...
fma.rn.f32 %f485, %f142, %f144, %f199;
```

This is:

```cpp
gate[0] += x0*w0 + x1*w1 + ... + x7*w7;
```

and similarly for:

```cpp
up[0], gate[1], up[1]
```

The four accumulator registers in this block are:

```text
%f485 = gate[0]
%f487 = up[0]
%f484 = gate[1]
%f486 = up[1]
```

At the end of the loop:

```ptx
add.s32      %r25, %r369, 1024;
setp.lt.s32  %p7, %r369, -352;
mov.u32      %r369, %r25;
@%p7 bra     $L__BB0_10;
```

The source loop is:

```cpp
for (int pack = threadIdx.x; pack < 672; pack += 1024)
```

For this exact shape, active threads only have one pack. The PTX condition looks strange because the compiler transformed the loop variable into a signed range form. Functionally, it is still "advance by 1024 packs; continue if still under kHiddenPacks".

## 7. Block Reduction For Gate/Up Column 0

After each thread has its partial dot sums, the kernel reduces across the CTA.

PTX lines 542-603 are the first-stage warp reduction and warp-sum store:

```ptx
and.b32 %r125, %r1, 31;       // lane = threadIdx.x & 31
...
shfl.sync.bfly.b32 ... 16
shfl.sync.bfly.b32 ... 8
shfl.sync.bfly.b32 ... 4
shfl.sync.bfly.b32 ... 2
shfl.sync.bfly.b32 ... 1
...
st.shared.f32 [%r156],     %f21;
st.shared.f32 [%r156+128], %f22;
bar.sync 0;
```

This is `block_reduce_pair(gate[0], up[0], ...)`.

The repeated `shfl.sync.bfly.b32` instructions are the warp-level butterfly sum. Each lane exchanges partial sums with lanes at offsets 16, 8, 4, 2, 1. After this, lane 0 has the sum for its warp.

Only lane 0 writes the warp's result into shared memory:

```cpp
if (lane == 0) {
  a_warp_sums[warp] = a;
  b_warp_sums[warp] = b;
}
```

The PTX uses the two halves of `s_reduce_warp_sums`:

- offset `+0` for gate warp sums
- offset `+128` for up warp sums

because `32 warps * 4 bytes = 128`.

Then lines 603-676 load those 32 warp sums back into warp 0 and reduce them:

```ptx
bar.sync 0;
ld.shared.f32 %f490, [...]
ld.shared.f32 %f491, [...+128]
...
shfl.sync.bfly.b32 ...
```

Only warp 0 does this second reduction. At the end, thread 0 owns the full-CTA gate/up totals.

## 8. GELU Tanh Approximation And `s_act[0]`

PTX lines 676-716 compute activation 0:

```ptx
mul.f32      %f259, %f491, %f491;
mul.f32      %f260, %f491, 0f3D372713;
fma.rn.f32   %f261, %f260, %f259, %f491;
mul.f32      %f262, %f261, 0f3F4C422A;
...
ex2.approx.ftz.f32 %f265, %f264;
rcp.approx.ftz.f32 %f268, %f266;
...
fma.rn.f32   ...
...
mul.f32      %f287, %f286, %f353;
mul.f32      %f288, %f490, %f287;
st.shared.f32 [s_act], %f288;
```

This is:

```cpp
s_act[0] = gate[0] * gelu_tanh(up[0]);
```

Important mapping:

- `%f490` is gate total.
- `%f491` is up total.
- `%f491 * %f491`, `0.044715`, and `sqrt(2/pi)` implement tanh-GELU's inner expression.
- `ex2.approx` and `rcp.approx` are how the compiler lowers `tanhf` approximately.
- The polynomial-looking sequence is part of the libdevice/compiler tanh approximation.

Only thread 0 stores `s_act[0]`. The following `bar.sync` ensures all threads can read it before down accumulation.

## 9. Block Reduction And Activation For Column 1

PTX lines 722-895 repeat the same pattern for `gate[1]` and `up[1]`:

1. Warp reduce.
2. Write warp sums.
3. Block reduce in warp 0.
4. Compute `gate[1] * gelu_tanh(up[1])`.
5. Store to `s_act + 4`.

The store:

```ptx
st.shared.f32 [s_act+4], %f356;
```

is `s_act[1]`. The `+4` is one FP32 element.

This two-step reduction explains why gate and up need two independent warp-sum arrays. Gate and up are reduced together in `block_reduce_pair`, so they cannot share the same scratch at the same time. The RMS sum later can reuse one of those arrays because the lifetimes do not overlap.

## 10. Down Projection Accumulation

PTX lines 895-1006:

```ptx
bar.sync 0;
@%p1 bra $L__BB0_33;

ld.global.cs.v4.s32 {%r269,%r270,%r271,%r272}, [%rd39];
ld.shared.f32 %f373, [s_act];
...
fma.rn.f32 %f374, %f373, %f357, %f523;
...
ld.global.cs.v4.s32 {%r273,%r274,%r275,%r276}, [%rd40];
ld.shared.f32 %f382, [s_act+4];
...
fma.rn.f32 %f523, %f382, %f365, %f374;
...
bar.sync 0;
add.s32 %r368, %r368, 2;
setp.lt.u32 %p63, %r368, 256;
@%p63 bra $L__BB0_8;
```

This is:

```cpp
if (owns_hidden_pack) {
  down_pack0 =
      Bf16Packed128{*reinterpret_cast<const int4 *>(
          w_down_row_major + intermediate_col0 * 5376 + hidden_col)};
  partial += s_act[0] * down_pack0;

  down_pack1 =
      Bf16Packed128{*reinterpret_cast<const int4 *>(
          w_down_row_major + (intermediate_col0 + 1) * 5376 + hidden_col)};
  partial += s_act[1] * down_pack1;
}
__syncthreads();
```

The eight output accumulators are still `%f520..%f527`. Each down row contributes one BF16 weight pack of eight values, widened to FP32, then multiplied by scalar `s_act[t]`, then accumulated.

The loop increment:

```ptx
add.s32 %r368, %r368, 2;
setp.lt.u32 %p63, %r368, 256;
@%p63 bra $L__BB0_8;
```

is the activation tile loop:

```cpp
for (local_col = 0; local_col < 256; local_col += 2)
```

Each CTA handles 256 intermediate columns. There are `21504 / 256 = 84` CTAs.

## 11. Cross-CTA Scratch Turn Lock

After a CTA has accumulated its 256-column contribution, CTAs serialize through `scratch->lock`.

PTX lines 1011-1035:

```ptx
ld.param.u64 %rd60, [param_7];  // scratch
add.s64      %rd6, %rd60, 21504; // scratch->lock

$L__BB0_35:
ld.global.acquire.gpu.b32 %r281, [%rd6];
setp.ne.s32 %p65, %r281, %r88;
@%p65 bra $L__BB0_35;

bar.sync 0;

$L__BB0_37:
ld.global.acquire.gpu.b32 %r282, [%rd6];
setp.ne.s32 %p66, %r282, %r88;
@%p66 bra $L__BB0_37;

bar.sync 0;
```

`%r88` is `blockIdx.x`. The lock value must equal this CTA's index before the CTA can read or write `scratch->accum`.

There are two acquire phases because the source does:

```cpp
if (threadIdx.x == 0) {
  spin with acquire load
}
__syncthreads();

all threads perform acquire load
__syncthreads();
```

The first phase avoids all 1024 threads pounding the lock while waiting. The second phase gives every reader thread acquire semantics before it reads `scratch->accum`.

## 12. Scratch Accumulate Or Store

PTX lines 1038-1069 handle non-first and non-last CTAs:

```ptx
setp.eq.s32 %p68, %r88, 0;
...
@%p68 bra $L__BB0_41;

ld.global.v4.f32 {%f383,%f384,%f385,%f386}, [%rd7];
ld.global.v4.f32 {%f388,%f389,%f390,%f391}, [%rd7+16];
add.f32 ...

$L__BB0_41:
setp.eq.s32 %p69, %r88, 83;
@%p69 bra $L__BB0_46;

st.global.v4.u32 [%rd7],    {...};
st.global.v4.u32 [%rd7+16], {...};
```

`blockIdx.x == 0` skips the read because there is no previous partial sum.

`blockIdx.x == 83` skips the write because the last CTA will finish the residual/RMSNorm/output path directly.

All middle CTAs do:

```cpp
partial += scratch->accum[hidden_pack];
scratch->accum[hidden_pack] = partial;
```

The scratch buffer is FP32, so these are `ld.global.v4.f32` and stores of raw `.u32` bit patterns.

## 13. Last CTA: Residual Add, Sum Sq, Residual Output Store

PTX lines 1076-1158:

```ptx
ld.param.u64 %rd64, [param_0]; // residual_out
ld.param.u64 %rd63, [param_3]; // residual
ld.global.nc.v4.s32 {%r295,%r296,%r297,%r298}, [%rd52];
...
add.f32 %f523, residual0, partial0;
add.f32 %f522, residual1, partial1;
...
fma.rn.f32 %f417, %f523, %f523, 0;
fma.rn.f32 %f418, %f522, %f522, %f417;
...
cvt.rn.bf16x2.f32 ...
st.global.v4.u32 [%rd55], {%r299, %r300, %r301, %r302};
```

This is:

```cpp
residual_pack = Bf16Packed128{*reinterpret_cast<const int4 *>(residual + hidden_col)};
values[i] = partial[i] + residual[i];
sum_sq += values[i] * values[i];
store residual_out as BF16;
```

The values stay in `%f520..%f527` after residual add. That is important: the same FP32 values are reused later for RMSNorm output, so the kernel does not reload `residual_out`.

The BF16 stores use:

```ptx
cvt.rn.bf16x2.f32
```

packing two FP32 values into one BF16x2 32-bit register. Four such registers make the 128-bit store.

## 14. RMS Sum Reduction

PTX lines 1160-1240 reduce `sum_sq`:

```ptx
shfl.sync.bfly.b32 ... 16
shfl.sync.bfly.b32 ... 8
shfl.sync.bfly.b32 ... 4
shfl.sync.bfly.b32 ... 2
shfl.sync.bfly.b32 ... 1
...
st.shared.f32 [%r324], %f107;
bar.sync 0;
ld.shared.f32 %f538, [%r328];
...
shfl.sync.bfly.b32 ... 16, 8, 4, 2, 1
```

This is `block_reduce_sum(sum_sq, s_reduce_warp_sums[0])`.

The reduction pattern is the same as gate/up, but only one scalar is reduced. This is where the earlier scratch cleanup matters: RMS sum reuses `s_reduce_warp_sums[0]`, because gate/up reduction is complete and `s_reduce_warp_sums[0]` is dead.

## 15. RMS Scale

PTX lines 1242-1250:

```ptx
ld.param.f32      %f471, [param_8]; // eps
div.rn.f32        %f443, %f538, 0f45A80000;
add.f32           %f444, %f443, %f471;
rsqrt.approx.f32  %f445, %f444;
st.shared.f32     [s_scale], %f445;
```

`0f45A80000` is `5376.0f`.

So this is:

```cpp
s_scale = rsqrtf(total / 5376.0f + eps);
```

Only thread 0 writes `s_scale`; the following barrier makes it visible to all active hidden-pack threads.

## 16. RMSNorm Output Store

PTX lines 1252-1334:

```ptx
ld.param.u64 %rd62, [param_1]; // normed_out
ld.param.u64 %rd61, [param_4]; // rms_weight
ld.global.nc.v4.s32 {%r347,%r348,%r349,%r350}, [%rd56];
ld.shared.f32 %f462, [s_scale];
mul.f32 %f463, %f462, %f523;
mul.f32 %f448, gamma0, %f463;
...
cvt.rn.bf16x2.f32 ...
st.global.v4.u32 [%rd59], {%r351,%r352,%r353,%r354};
```

This is:

```cpp
normed[i] = values[i] * s_scale * gamma[i];
store normed_out as BF16;
```

`gamma` comes from `rms_weight`. It is loaded as BF16, widened to FP32 with the same `{0,%rs}` trick, multiplied into the normalized values, then packed back to BF16x2.

## 17. Non-Last CTA Release Path

PTX lines 1337-1348:

```ptx
$L__BB0_44:
setp.ne.s32 %p94, %r1, 0;
bar.sync 0;
membar.gl;
bar.sync 0;
@%p94 bra $L__BB0_58;

mov.u32 %r294, 1;
red.relaxed.gpu.global.add.s32 [%rd6], %r294;
```

This is `release_reduce_turn`:

```cpp
__syncthreads();
__threadfence();
__syncthreads();
if (threadIdx.x == 0) {
  red.relaxed.gpu.global.add.s32(lock, 1);
}
```

`membar.gl` is the PTX lowering of `__threadfence()`. It is important because all writer threads, not only thread 0, may have written `scratch->accum`. The lock increment must not become visible before those global writes are ordered.

The second `bar.sync` ensures thread 0 does not signal before all threads have executed the fence.

The atomic-like `red.relaxed.gpu.global.add.s32` increments the lock to release the next CTA.

## 18. The Main Performance Read

The hottest region is not the final RMSNorm. It is the repeated gate/up dot and down accumulation:

- Gate/up dot: lines roughly 193-540.
- Reductions and activation: lines roughly 542-895.
- Down accumulation: lines roughly 895-1006.

The gate/up dot does scalar FP32 FMA chains after widening BF16 values. No tensor cores are used. This is expected for the current one-token decode shape and this hand-written scalar GEMV decomposition.

The shared `x` swizzle does not obviously improve memory throughput in this kernel. The benchmark says the throughput effect is small/noisy. Its measurable compile-side effect is lower physical register allocation because it changes ptxas scheduling in the preload.

## 19. What The PTX Suggests We Could Improve

1. **Async copy would target the `s_x` preload.**

   The PTX currently stages global `x` through registers:

   ```ptx
   ld.global.nc.v4.s32 {%r33,%r34,%r35,%r36}, [%rd16];
   st.shared.v4.u32 [%r42], {%r33,%r34,%r35,%r36};
   ```

   A true global-to-shared async path could reduce this explicit payload-register staging. The CUDA guide calls this out for async copy: copying directly to shared can reduce register usage. However, `s_x` is only 10.5 KB and loaded once per CTA, while the weights dominate traffic. This is probably a cleanup/latency-overlap experiment, not the main breakthrough.

2. **The real cost is scalar GEMV work, not RMSNorm.**

   The PTX shows long scalar BF16 widen + FP32 FMA chains for gate/up and down. That is the core bottleneck. Larger activation tiles help a bit until they spill; tile 8 spills and regresses.

3. **The tanh GELU sequence is expensive but only runs once per intermediate column.**

   The `ex2.approx`, `rcp.approx`, and polynomial sequence happen for each activation scalar, not for each hidden element. It is visible, but the dominant repeated work remains the vector dot/down accumulation.

4. **The global scratch handoff is correct but serializing.**

   The lock path enforces CTA order. That is necessary for the current single-kernel decomposition, but it means the final accumulation across intermediate tiles is serialized through `scratch->accum`.

5. **The unsigned swizzle-index cleanup was justified.**

   Before the cleanup, PTX emitted signed correction for `chunk / 8` even though `chunk` is nonnegative. Now the swizzle is just shift/mask/xor/shift, matching the intended chunk-level mapping.

## 20. Compact Label Map

```text
entry prologue        lines 18-64
$L__BB0_3             small/tail shared x preload
$L__BB0_6             unrolled shared x preload
$L__BB0_7             post-x-load barrier and outer-loop initialization
$L__BB0_8             activation tile loop top
$L__BB0_10            gate/up dot loop body
$L__BB0_11..19        reduce gate/up for activation 0
$L__BB0_19            GELU and s_act[0] store
$L__BB0_21..29        reduce gate/up for activation 1
$L__BB0_29            GELU and s_act[1] store
$L__BB0_31            down projection accumulation
$L__BB0_33            activation tile loop increment
$L__BB0_35..37        wait for scratch lock turn
$L__BB0_41            scratch read skipped for first CTA
$L__BB0_43/$L__BB0_44 non-last CTA store/release path
$L__BB0_46            last CTA residual/RMS path
$L__BB0_48..54        RMS sum reduction and scale
$L__BB0_56            normed output store
$L__BB0_58            return
```

## 21. Bottom Line

The PTX is doing what the CUDA source says:

- vectorized 128-bit BF16 movement,
- scalar FP32 dot products,
- warp-shuffle reductions,
- shared-memory publication of two activation scalars,
- scalar down accumulation into eight FP32 registers,
- global scratch serialization,
- final residual add, RMS sum, RMS scale, and BF16 output pack.

The surprising ptxas register drop from swizzle-on is **not** because swizzle is magically cheaper. It is because the non-linear shared destination address changes ptxas' preload scheduling and lowers the maximum number of simultaneously live physical registers.

Questionable/worth revisiting:

- The global scratch lock is correct but structurally serial.
- Async copy for `s_x` may reduce register staging, but the x preload is not the dominant traffic.
- Tensor-core use would require a different decomposition; this PTX is scalar FMA GEMV code.
- The current tile-2 default is conservative and measured; tile-4 is close but not better in the longer run, and tile-8 spills.
