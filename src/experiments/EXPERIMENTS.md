##  Experiments Log

AI-updated and directed log of the experiments I ran throughout this project to optimize the kernels. 
Expect this to be very messy and pretty much useless for most people to look at.  it is meant to be a place for me and my agents to fuck around

## 2026-05-23 - Global FlashAttention option

Runtime files:

- `src/gemma4_flash_attention.cu`
- `src/gemma4_flash_attention.cuh`
- `src/experiments/gemma4_flash_attention_bench.cu`
- `Makefile`

Change:

- Added a Gemma global-attention FA entrypoint:
  `gemma4_flash_attention_global_fwd_bf16`.
- The global path uses BF16, `head_dim=512`, `32` query heads, `4` KV heads,
  and full bottom-right causal masking by setting the internal window to
  `seqlen_k`.
- Kept the existing sliding entrypoint unchanged.
- Added a public CUDA header exposing both sliding and global FA options.
- Added a reference global wrapper that directly instantiates the upstream FA2
  kernel template at `head_dim=512`. The installed `flash_attn` package cannot
  serve as the global baseline because it rejects `head_dim > 256`.

Build:

```bash
make -B build/gemma4_flash_attention.o flash-attn-lib flash-attn-bench
```

GPU/tooling:

- GPU: NVIDIA RTX A6000
- Driver: 580.126.16
- CUDA/NVCC: CUDA 12.9, `nvcc` 12.9.86
- CUDA target: `sm_86`
- Clock policy: not locked.

Audits:

```text
dep_count 709
fa2_source_count 0
cutlass_cute_count 141
```

Exported symbols include both paths:

```text
gemma4_flash_attention_sliding_fwd_bf16
gemma4_flash_attention_global_fwd_bf16
```

Correctness:

Installed `flash_attn` check:

```text
RuntimeError FlashAttention forward only supports head dimension at most 256
```

Python probe against a PyTorch FP32 reference with GQA expansion and
bottom-right causal masking:

```text
global threads 64 smem 98304
global_correctness sq=64 sk=64 max_abs=0.015625 mean_abs=0.000255395
global_correctness sq=1 sk=1024 max_abs=0.000976562 mean_abs=5.82829e-05
```

Sliding path still matches the upstream source ref and installed `flash_attn`
exactly in the same-tensor comparator:

```text
diff_vs_official_source_ref max_abs=0 mean_abs=0 max_rel=0
diff_vs_flash_attn max_abs=0 mean_abs=0 max_rel=0
```

Timing:

Direct upstream FA2 `head_dim=512` reference comparison:

```text
case sq=64 sk=64
  diff max_abs=0 mean_abs=0 lse_max_abs=0
  custom median/min/max ms=0.034511/0.033782/0.057761
  upstream_direct median/min/max ms=0.032795/0.030715/0.043902
  custom/ref=1.052323

case sq=1 sk=1024
  diff max_abs=0 mean_abs=0 lse_max_abs=0
  custom median/min/max ms=0.207234/0.205562/0.218658
  upstream_direct median/min/max ms=0.209351/0.209320/0.209530
  custom/ref=0.989888

case sq=1024 sk=1024
  diff_max=0 diff_mean=0 lse_max=0
  custom median/min/max ms=1.610805/1.609355/1.645277
  upstream_direct median/min/max ms=1.618664/1.614611/1.625269
  custom/ref=0.995145
```

```text
global_decode_timing sk=1024 avg_ms=0.239686
```

The sliding comparator in this run measured all implementations around
`0.48 ms` at `seq=1024`, including installed `flash_attn`, whereas an earlier
run measured all implementations around `0.234 ms`. Since the custom, upstream
source ref, and installed package moved together, this looks like GPU clock or
system state rather than a custom-only regression.

Conclusion:

- Global FA is now an explicit runtime option for the future layer runner.
- Against a direct upstream FA2 `head_dim=512` baseline, the global path is exact
  on output and effectively performance-parity on decode and larger prefill
  shapes.
- This is a correctness-oriented global path, not the final optimized
  long-context global decode kernel. The `head_dim=512` tile is
  `BlockM=32, BlockN=32, 2` warps, using `98304` bytes dynamic shared memory.

## 2026-05-23 - Device-call matmul wrapper and FFN decode integration

Runtime files:

- `src/gemma4_matmul_device.cuh`
- `src/experiments/gemma4_matmul_device_kernels.cu`
- `src/experiments/gemma4_matmul_device_kernels.cuh`
- `src/experiments/gemma4_decode_bench.cu`
- `src/gemma4_ffn_decode.cu`
- `Makefile`

Change:

- Copied the decode GEMV launch surface into an experiment source and moved the
  kernel body into reusable `__device__` helpers.
- Added a deliberately thin global wrapper around the device helper so it can be
  compared directly with the original `src/gemma4_matmul_kernels.cu` kernels.
- Added decode-bench timing and numerical checks for `device_wrapped` and
  `device_wrapped_swizzle16`.
- Wired the device matmul dot/reduce helper into the fused FFN decode gate/up
  path. The FFN kernel still keeps the existing fused dataflow: shared-X load,
  gate/up dot, GeGLU, down-row accumulation, residual add, and RMSNorm.

Correctness:

```bash
make decode-bench test-ffn-decode
```

```text
ffn decode tests passed
```

Device-wrapper matmul timing:

```bash
GEMMA4_DECODE_BENCH_SEED=0x20260522 \
  build/experiments/gemma4_decode_bench ffn_gate_up 100 20 3
GEMMA4_DECODE_BENCH_SEED=0x20260522 \
  build/experiments/gemma4_decode_bench ffn_down 100 20 3
```

GPU/tooling:

- GPU: NVIDIA RTX A6000
- Driver: 580.126.16
- CUDA/NVCC: CUDA 12.9, `nvcc` 12.9.86
- CUDA target: `sm_86`
- cuDNN version in FFN harness: `92200`
- Warmup/timing: 20 warmup iterations, 100 timed iterations, 3 trials.
- Timing uses CUDA events through the existing benchmark helper.
- Clock policy: not locked.
- Cache policy: normal benchmark warm-cache behavior.

| Op | Original best ms | Device wrapper best ms | Device/original | Max abs diff |
| --- | ---: | ---: | ---: | ---: |
| `ffn_gate_up` | 0.650478 | 0.650972 | 0.999241x | 0 |
| `ffn_down` | 0.327827 | 0.327563 | 1.000807x | 0 |

Swizzled variant:

| Op | Original swizzle16 best ms | Device wrapper swizzle16 best ms | Device/original | Max abs diff |
| --- | ---: | ---: | ---: | ---: |
| `ffn_gate_up` | 0.650752 | 0.650858 | 0.999838x | 0 |
| `ffn_down` | 0.327701 | 0.327575 | 1.000383x | 0 |

ptxas resource check:

```bash
nvcc -std=c++17 -O3 -arch=sm_86 -Xptxas=-v -Isrc \
  -c src/gemma4_matmul_kernels.cu \
  -o /tmp/gemma4_device_matmul_stats/original_matmul.o
nvcc -std=c++17 -O3 -arch=sm_86 -Xptxas=-v -Isrc \
  -c src/experiments/gemma4_matmul_device_kernels.cu \
  -o /tmp/gemma4_device_matmul_stats/device_wrapped_matmul.o
nvcc -std=c++17 -O3 -arch=sm_86 -Xptxas=-v -Isrc \
  -c src/gemma4_ffn_decode.cu \
  -o /tmp/gemma4_device_matmul_stats/ffn_decode_device_matmul.o
```

For the original and device-wrapper matmul kernels, ptxas reported matching
resource usage for the FFN shapes:

| Kernel shape | Original regs | Wrapper regs | Shared memory | Spills |
| --- | ---: | ---: | ---: | --- |
| `K=5376,N=43008,identity` | 58 | 58 | 512 B | none |
| `K=5376,N=43008,swizzle16` | 60 | 60 | 512 B | none |
| `K=21504,N=5376,identity` | 58 | 58 | 1024 B | none |
| `K=21504,N=5376,swizzle16` | 60 | 60 | 1024 B | none |

FFN decode after integration:

```bash
GEMMA4_FFN_CUDNN_BENCH_SEED=0x20260522 \
  build/experiments/gemma4_ffn_cudnn_bench 100 20 3
```

Baseline captured before the FFN edit:

```text
cudnn_split_device_ms,1.000036
custom_device_ms,1.108929
custom_scratch_clear_device_ms,0.001090
custom_minus_clear_device_ms,1.107840
custom_vs_cudnn_split_speedup,0.901803
custom_minus_clear_vs_cudnn_split_speedup,0.902690
```

After using the device matmul helper inside FFN decode:

```text
cudnn_split_device_ms,1.000140
custom_device_ms,1.106343
custom_scratch_clear_device_ms,0.001086
custom_minus_clear_device_ms,1.105257
custom_vs_cudnn_split_speedup,0.904006
custom_minus_clear_vs_cudnn_split_speedup,0.904894
```

FFN kernel resource stats after the change:

```text
Used 48 registers, used 1 barriers, 11404 bytes smem
0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
```

Conclusion:

- The copied device-call matmul wrapper is effectively performance-neutral
  versus the original global matmul kernel for the two FFN GEMV shapes and is
  numerically identical.
- ptxas stats match the original matmul kernels for those shapes.
- Using the device matmul helper in the fused FFN decode gate/up path preserved
  correctness and slightly improved graph-timed custom FFN decode from
  `1.108929 ms` to `1.106343 ms`.
- The FFN change reduced register allocation from the previous tile-2 swizzled
  path's recorded `50` registers to `48`, with shared memory increasing from
  about `11020 B` to `11404 B`, and no spills.
- Keep this integrated path. The helper gives a reusable matmul-shaped
  dot/reduce primitive without changing the fused FFN dataflow.

## 2026-05-23 - FFN decode activation-tile sweep

Runtime files:

- `src/gemma4_ffn_decode.cu`
- `tests/test_ffn_decode.cu`
- `src/experiments/gemma4_ffn_cudnn_bench.cu`

Change:

- Added `GEMMA4_FFN_DECODE_ACT_TILE` as a compile-time FFN decode hyperparameter.
- Default is now `2`.
- Added `GEMMA4_FFN_DECODE_SWIZZLE_X` for the FFN decode shared activation
  cache. Default is now enabled.
- The shared activation swizzle XORs 128-bit chunk indices, not element indices:
  one chunk is one `Bf16Packed128` / 16-byte lane, equivalent in size to
  `4 x float`.
- The tiled path is first-class in the main loop: each CTA computes `kActTile`
  gate/up scalar pairs, reduces them, stores `s_act[kActTile]`, then streams the
  matching down rows before advancing.
- This keeps the fused single-token decode contract unchanged and does not add a
  second special-case kernel path.

Correctness:

```text
make test-ffn-decode
ffn decode tests passed

for tile in 2 4 8; do
  nvcc -std=c++17 -O3 -arch=sm_86 -Isrc \
    -DGEMMA4_FFN_DECODE_ACT_TILE=${tile} \
    tests/test_ffn_decode.cu src/gemma4_ffn_decode.cu \
    -o /tmp/test_ffn_decode_tile${tile}
  /tmp/test_ffn_decode_tile${tile}
done
ffn decode tests passed
ffn decode tests passed
ffn decode tests passed
```

Resource sweep:

```bash
for tile in 1 2 4 8 16; do
  nvcc -std=c++17 -O3 -arch=sm_86 -Xptxas=-v -Isrc \
    -DGEMMA4_FFN_DECODE_SWIZZLE_X=0 \
    -DGEMMA4_FFN_DECODE_ACT_TILE=${tile} \
    -c src/gemma4_ffn_decode.cu -o /tmp/gemma4_ffn_decode_tile${tile}.o
done
```

| Tile | Registers | Shared memory | Stack/spills |
| ---: | ---: | ---: | --- |
| 1 | 59 | 11016 B | 0 stack, 0 spills |
| 2 | 59 | 11020 B | 0 stack, 0 spills |
| 4 | 62 | 11028 B | 0 stack, 0 spills |
| 8 | 62 | 11044 B | 16 B stack, 64 B spill stores, 64 B spill loads |
| 16 | 64 | 11076 B | 88 B stack, 188 B spill stores, 212 B spill loads |

Chunk-swizzled shared activation resource sweep:

```bash
for swiz in 0 1; do
  for tile in 1 2 4 8; do
    nvcc -std=c++17 -O3 -arch=sm_86 -Xptxas=-v -Isrc \
      -DGEMMA4_FFN_DECODE_SWIZZLE_X=${swiz} \
      -DGEMMA4_FFN_DECODE_ACT_TILE=${tile} \
      -c src/gemma4_ffn_decode.cu -o /tmp/gemma4_ffn_decode_swiz${swiz}_tile${tile}.o
  done
done
```

| Swizzle | Tile | Registers | Shared memory | Stack/spills |
| ---: | ---: | ---: | ---: | --- |
| 0 | 1 | 59 | 11016 B | 0 stack, 0 spills |
| 0 | 2 | 59 | 11020 B | 0 stack, 0 spills |
| 0 | 4 | 62 | 11028 B | 0 stack, 0 spills |
| 0 | 8 | 62 | 11044 B | 16 B stack, 64 B spill stores, 64 B spill loads |
| 1 | 1 | 51 | 11016 B | 0 stack, 0 spills |
| 1 | 2 | 50 | 11020 B | 0 stack, 0 spills |
| 1 | 4 | 62 | 11028 B | 0 stack, 0 spills |
| 1 | 8 | 62 | 11044 B | 16 B stack, 64 B spill stores, 64 B spill loads |

PTX/SASS dump for the tile-2 swizzle comparison:

```bash
mkdir -p build/ptx
for swiz in 0 1; do
  nvcc -std=c++17 -O3 -arch=sm_86 -Isrc \
    -DGEMMA4_FFN_DECODE_SWIZZLE_X=${swiz} \
    -DGEMMA4_FFN_DECODE_ACT_TILE=2 \
    -ptx src/gemma4_ffn_decode.cu \
    -o build/ptx/gemma4_ffn_decode_swiz${swiz}_tile2.ptx
  nvcc -std=c++17 -O3 -arch=sm_86 -Xptxas=-v -lineinfo -Isrc \
    -DGEMMA4_FFN_DECODE_SWIZZLE_X=${swiz} \
    -DGEMMA4_FFN_DECODE_ACT_TILE=2 \
    -cubin src/gemma4_ffn_decode.cu \
    -o /tmp/gemma4_ffn_decode_swiz${swiz}_tile2.cubin
  cuobjdump --dump-sass /tmp/gemma4_ffn_decode_swiz${swiz}_tile2.cubin \
    > build/ptx/gemma4_ffn_decode_swiz${swiz}_tile2.sass
done
```

PTX/SASS notes:

- PTX virtual register declarations are not the ptxas allocation result. Swizzle-off
  emitted fewer virtual `.b32` PTX registers (`%r<346>`) than swizzle-on before the
  unsigned cleanup (`%r<381>`), but ptxas still allocated more physical registers for
  swizzle-off.
- The physical SASS peak matched the ptxas report: swizzle-off used up to `R56`
  and reported 59 registers; swizzle-on used up to `R47` and reported 50 registers.
- The drop comes from the `s_x` load/store prologue. With identity indexing, ptxas
  recognizes a simple linear shared-memory stream and aggressively unrolls/schedules
  many 128-bit global loads and shared stores at once. That exposes many live address
  and 128-bit payload registers.
- With XOR chunk swizzling, the shared address is no longer a simple linear induction
  variable. ptxas keeps the prologue in a much smaller loop shape, so fewer load
  payloads and store addresses are simultaneously live. The register drop is mostly
  a scheduling/strength-reduction side effect, not proof that XOR itself is cheaper.
- I changed the swizzle helper to use unsigned chunk arithmetic. Since `pack` is
  nonnegative, this removes unnecessary signed-division correction from the PTX
  swizzle-index calculation while preserving the same chunk mapping.

Timing command:

```bash
GEMMA4_FFN_CUDNN_BENCH_SEED=0x20260522 \
  /tmp/gemma4_ffn_cudnn_bench_tile${tile} 100 20 3 1
```

GPU/tooling:

- GPU: NVIDIA RTX A6000
- CUDA target: `sm_86`
- cuDNN version: `92200`
- Warmup/timing: 20 warmup iterations, 100 timed iterations, 3 trials.
- Timing uses CUDA events through the existing benchmark helper; CUDA graph timing is
  used as the device-time comparison.
- Clock policy: not locked.
- Cache policy: normal benchmark warm-cache behavior.

Tile sweep results:

| Tile | custom graph best ms | custom direct best ms | cuDNN split graph best ms | custom/cuDNN speedup |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1.137898 | 1.141672 | 1.000087 | 0.878890x |
| 2 | 1.077307 | 1.078275 | 1.000453 | 0.928661x |
| 4 | 1.080216 | 1.082335 | 1.001038 | 0.926701x |
| 8 | 1.177239 | 1.175480 | 0.999530 | 0.849046x |

Default tile-2 rerun through the checked-in `make ffn-cudnn-bench` binary after
the scratch handoff fence fix:

```text
device=NVIDIA RTX A6000
shape=tokens1,hidden5376,intermediate21504,seed=0x20260522
iters=100,warmup_iters=20,trials=3
cudnn_frontend=compiled
cudnn_version=92200
geglu_plus_down,11.606174,11.943123,1.000069,1.000217,55.669,4194560,
custom_fused_decode,1.108576,1.114322,1.107599,1.108106,582.862,0,
custom_scratch_clear,0.006160,0.007221,0.001077,0.001090,0.000,0,
cudnn_split_device_ms,1.000069
custom_device_ms,1.107599
custom_minus_clear_device_ms,1.106522
custom_vs_cudnn_split_speedup,0.902916
custom_minus_clear_vs_cudnn_split_speedup,0.903795
```

Chunk-swizzled shared activation timing, short sweep:

iters=50,warmup_iters=10,trials=2

| Swizzle | Tile | custom graph best ms | cuDNN split graph best ms | custom/cuDNN speedup |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 1.165145 | 0.999914 | 0.858188x |
| 1 | 2 | 1.103517 | 0.999734 | 0.905953x |
| 1 | 4 | 1.103430 | 0.999462 | 0.905778x |
| 1 | 8 | 1.199932 | 1.000461 | 0.833765x |
| 0 | 2 | 1.107462 | 1.000806 | 0.903694x |
| 1 | 2 | 1.107693 | 1.000389 | 0.903129x |

Tile-2 versus tile-4 longer rerun with chunk swizzle enabled:

iters=100,warmup_iters=20,trials=3

| Tile | custom graph best ms | cuDNN split graph best ms | custom/cuDNN speedup |
| ---: | ---: | ---: | ---: |
| 2 | 1.107858 | 1.000124 | 0.902754x |
| 4 | 1.109708 | 1.000656 | 0.901729x |

Unsigned swizzle-index cleanup timing, short sanity run:

iters=50,warmup_iters=10,trials=2

| Variant | custom graph best ms | cuDNN split graph best ms | custom/cuDNN speedup |
| --- | ---: | ---: | ---: |
| unsigned chunk swizzle, tile 2 | 1.108769 | 0.999982 | 0.901885x |

Conclusion:

- Activation tiling helps. Tile `2` reduces custom graph time from `1.137898 ms`
  to `1.077307 ms` in the fixed-seed sweep, about a 5.3% custom-kernel improvement.
- Tile `4` is close but slightly slower than tile `2` and increases register use to
  62.
- Tile `8` spills and regresses hard. Do not use spilling tiles without a deeper
  schedule rewrite.
- The FFN decode path now uses chunk-granularity shared activation swizzling by
  default. It is not a meaningful throughput win on this benchmark, but it is
  correct, cheap at tile `2`, and reduced ptxas register allocation for the default
  path from 59 to 50 registers.
- The register drop is a ptxas scheduling artifact in the shared activation preload,
  not a direct bank-conflict win. The unsigned index cleanup is the only code change
  justified by the PTX dump itself.
- Manual race audit found that the global scratch handoff needed every writer thread
  to execute a device fence before thread 0 advances the lock, and every reader thread
  to perform an acquire load before reading `scratch->accum`. That fix preserves
  correctness at the cost of about `0.032 ms` on the checked-in tile-2 rerun.
- cuDNN split graph is still faster for this benchmark (`~1.0001 ms`) than the fused
  custom decode (`~1.1076 ms`), but the gap remains smaller than the original tile-1
  baseline.
- Full cuDNN FFN graph is still unsupported by the frontend planner for this shape in
  this harness; the supported cuDNN comparison remains split GeGLU plus down matmul.

## 2026-05-22 - FFN decode cache-hint ablation

Runtime files:

- `src/gemma4_cuda_utils.cuh`
- `src/gemma4_ffn_decode.cuh`
- `src/gemma4_ffn_decode.cu`
- `src/experiments/gemma4_ffn_cache_policy_bench.cu`
- `Makefile`

Policy IDs:

- `0`: default aligned 128-bit load
- `1`: `__ldg` read-only load, emitted as `ld.global.nc`
- `2`: `__ldca`, emitted as `ld.global.ca`
- `3`: `__ldcg`, emitted as `ld.global.cg`
- `4`: `__ldcs`, emitted as `ld.global.cs`
- `5`: `__ldlu`, emitted as `ld.global.lu`
- `6`: `__ldcv`, emitted as `ld.global.cv`

Default policy tuple remains:

```text
x=1, gate=4, up=4, down=4, residual=1, gamma=1, scratch=0
```

Reasoning:

- `.cv` is only plausible for one-pass weight streams. I did not make it a default for
  `x`, residual, gamma, or scratch because those are small/reused or L2-sensitive.
- Scratch is the global FP32 handoff buffer between CTAs. It should stay cacheable and
  L2-hot; the sweep confirmed non-default hints are harmful there.
- Gamma is the learned RMSNorm weight vector `[5376]`, not the scalar RMS value.

Commands:

```bash
make test-ffn-decode

# Weight stream sweep: gate/up grouped, down independent.
for gu in 0 1 2 3 4 5 6; do
  for down in 0 1 2 3 4 5 6; do
    nvcc -std=c++17 -O3 -arch=sm_86 -Isrc \
      -DGEMMA4_FFN_GATE_LOAD_POLICY=${gu} \
      -DGEMMA4_FFN_UP_LOAD_POLICY=${gu} \
      -DGEMMA4_FFN_DOWN_LOAD_POLICY=${down} \
      src/experiments/gemma4_ffn_cache_policy_bench.cu \
      src/gemma4_ffn_decode.cu -o /tmp/gemma4_ffn_cache_gu${gu}_down${down}
    /tmp/gemma4_ffn_cache_gu${gu}_down${down} 30 5 1
  done
done

# Longer rerun of top candidates.
/tmp/gemma4_ffn_cache_gu4_down4 200 20 5
/tmp/gemma4_ffn_cache_gu5_down0 200 20 5
```

Correctness:

```text
ffn decode tests passed
```

Weight sweep:

- First-pass fastest graph results were clustered around `1.140 ms` and did not show a
  stable `.cv` win.
- Top short-run candidates:
  - `gate/up=1`, `down=2`: `1.140212 ms`
  - `gate/up=5`, `down=0`: `1.140225 ms`
  - `gate/up=3`, `down=3`: `1.140420 ms`
  - `gate/up=4`, `down=6`: `1.140478 ms`
  - current `gate/up/down=4`: `1.141032 ms`

Longer candidate rerun:

```text
name,x,gate,up,down,residual,gamma,scratch,direct_best_ms,direct_avg_ms,graph_best_ms,graph_avg_ms
current,1,4,4,4,1,1,0,1.141580,1.144403,1.140824,1.141919
weight_best,1,5,5,0,1,1,0,1.145933,1.147701,1.140929,1.142647
weight_plus_x,3,5,5,0,1,1,0,1.143503,1.145587,1.141257,1.141895
weight_plus_residual,1,5,5,0,2,1,0,1.146148,1.148361,1.141838,1.142332
weight_plus_gamma,1,5,5,0,1,2,0,1.144858,1.147096,1.142313,1.142973
combined,3,5,5,0,2,2,0,1.145248,1.146566,1.140826,1.141441
```

Small-site one-at-a-time observations around `gate/up=5`, `down=0`:

- `x=cg`, `residual=ca`, and `gamma=ca` looked best in short isolated sweeps, but
  combining them did not produce a robust longer-run win.
- Scratch must remain default. `scratch=cg`, `scratch=lu`, and `scratch=cv` regressed to
  roughly `1.158-1.159 ms` graph time in the short sweep.

Conclusion:

- Keep current defaults for now: `x/residual/gamma=__ldg`, `gate/up/down=__ldcs`,
  `scratch=default`.
- `.cv` is not the best measured choice. It sometimes looked competitive for `down` in
  short runs, but did not survive longer reruns or combination tests.
- Cache hint choice is a noise-level lever for this kernel, roughly sub-1%. The real
  bottleneck is still the scalar-FMA work decomposition, not cache operator selection.

## 2026-05-22 - FFN decode 128-bit vectorized load pass

Runtime files:

- `src/gemma4_ffn_decode.cu`
- `src/gemma4_ffn_decode.cuh`
- `tests/test_ffn_decode.cu`
- `src/experiments/gemma4_ffn_cudnn_bench.cu`

Change:

- Reworked the custom fused decode FFN kernel so each active thread owns one
  hidden pack of `8 x bf16` instead of scalar hidden columns.
- Used the existing `Bf16Packed128` / `Packed128<float>` helpers from
  `gemma4_cuda_utils.cuh`, matching the RMSNorm, RoPE, embedding gather, and decode
  GEMV style.
- Vectorized the global BF16 paths for:
  - token input `x` load into shared memory
  - gate/up weight tile loads
  - down weight tile loads
  - residual and RMS gamma loads
  - residual and normed output stores
- Vectorized the FP32 scratch accumulator merge as two 128-bit float loads/stores per
  hidden pack. The lock remains scalar by design.

Commands:

```bash
make test-ffn-decode ffn-cudnn-bench

mkdir -p build/ptx
nvcc -std=c++17 -O3 -arch=sm_86 -Isrc -ptx src/gemma4_ffn_decode.cu \
  -o build/ptx/gemma4_ffn_decode_vectorized.ptx
nvcc -std=c++17 -O3 -arch=sm_86 -Isrc -Xptxas -v -c src/gemma4_ffn_decode.cu \
  -o build/ptx/gemma4_ffn_decode_vectorized.o

rg -n "ld\\.global|st\\.global|\\.v4\\.u32" \
  build/ptx/gemma4_ffn_decode_vectorized.ptx
cuobjdump --dump-sass build/ptx/gemma4_ffn_decode_vectorized.o | rg -n "LDG|STG"

./build/experiments/gemma4_ffn_cudnn_bench 20 5 2 1
```

PTX/compiler result:

- PTX contains vector BF16 global loads/stores:
  - `ld.global.nc.v4.s32`
  - `ld.global.cs.v4.s32`
  - `st.global.v4.u32`
- PTX contains vector FP32 scratch accumulator loads/stores:
  - `ld.global.v4.f32`
  - `st.global.v4.u32`
- The only scalar global operations left in the searched PTX are the reduction lock
  acquire/add operations.
- SASS contains `LDG.E.128` and `STG.E.128` for the vectorized global memory paths.
- `ptxas`: `59` registers/thread, `0` stack bytes, `0` spill stores, `0` spill loads,
  `11016` bytes shared memory, `1` barrier.

Correctness:

```text
ffn decode tests passed
```

Benchmark:

```text
path,best_ms,avg_ms,graph_best_ms,graph_avg_ms,rough_gib_s,workspace_bytes,max_abs_vs_split
geglu,9.746442,10.325469,0.662168,0.662480,44.193,0,
down,0.658119,0.702142,0.337484,0.337537,327.269,4194560,
geglu_plus_down,9.645019,10.781860,0.999732,1.000199,66.989,4194560,
custom_fused_decode,1.142081,1.144371,1.138668,1.139212,565.762,0,
custom_scratch_clear,0.006851,0.007631,0.001075,0.001096,0.000,0,
overhead_factored_metric,value
cudnn_split_device_ms,0.999732
custom_device_ms,1.138668
custom_scratch_clear_device_ms,0.001075
custom_minus_clear_device_ms,1.137594
custom_vs_cudnn_split_speedup,0.877983
custom_minus_clear_vs_cudnn_split_speedup,0.878813
```

Conclusion:

- The FFN decode kernel now emits 128-bit global memory ops for the meaningful BF16
  and FP32 scratch traffic.
- This is only a small speedup versus the prior custom graph timing (`~1.156 ms` ->
  `~1.139 ms`). The small delta is plausible because the scalar version's adjacent
  per-thread BF16 accesses were already warp-coalesced into the same 32-byte global
  memory segments; vectorization reduced instruction count, not the total bytes or
  memory transaction footprint.
- The kernel is still slower than the cuDNN split graph replay baseline. The next wins
  still need better work decomposition or tensor-core use.

## 2026-05-22 - cuDNN Frontend FFN decode graph probe

Runtime files:

- `src/experiments/gemma4_ffn_cudnn_bench.cu`
- `Makefile`

Change:

- Added `make ffn-cudnn-bench`.
- The benchmark builds:
  - a cuDNN Frontend GeGLU graph:
    `act = (x @ w_gate) * GELU_APPROX_TANH(x @ w_up)`
  - a separate cuDNN Frontend down-projection graph:
    `out = act @ w_down`
  - a full-FFN graph probe:
    `out = ((x @ w_gate) * GELU_APPROX_TANH(x @ w_up)) @ w_down`
  - the current custom decode fused FFN kernel, including residual add and
    RMSNorm epilogue
- The full-FFN path is probed and reported rather than required, because NVIDIA's
  LLM coverage samples say the analogous three-matmul SwiGLU layer graph is not
  supported by the cuDNN Graph API on this path.

Command:

```bash
make ffn-cudnn-bench
./build/experiments/gemma4_ffn_cudnn_bench 100 10 3 1
```

Environment:

- GPU: NVIDIA RTX A6000
- CUDA target: `sm_86`
- cuDNN version reported by benchmark: `92200`
- cuDNN Frontend include path: `/tmp/cudnn-frontend/include`
- Warmup/timed repeats/trials: 10 warmup, 100 timed iterations, 3 trials
- Cache policy: repeated-buffer warm-cache benchmark
- Clock policy: clocks not locked

Results:

```text
full_graph_supported=0,status="cuDNN frontend error for full_ffn.build_plans: [cudnn_frontend] Error: No valid execution plans built."
path,best_ms,avg_ms,graph_best_ms,graph_avg_ms,rough_gib_s,workspace_bytes,max_abs_vs_split
geglu,10.466622,11.574109,0.662417,0.662475,41.152,0,
down,0.641172,0.707548,0.337504,0.337613,335.919,4194560,
geglu_plus_down,12.699577,13.331832,1.000316,1.000455,50.876,4194560,
custom_fused_decode,1.152335,1.157930,1.155604,1.155750,560.728,0,
custom_scratch_clear,0.004377,0.006027,0.001101,0.001116,0.000,0,
full_ffn,-1.000000,,-1.000000,,0.000,,-1
overhead_factored_metric,value
cudnn_split_device_ms,1.000316
custom_device_ms,1.155604
custom_scratch_clear_device_ms,0.001101
custom_minus_clear_device_ms,1.154503
custom_vs_cudnn_split_speedup,0.865622
custom_minus_clear_vs_cudnn_split_speedup,0.866447
cudnn_direct_minus_device_ms,11.699262
custom_direct_minus_device_ms,0.000000
```

Conclusion:

- cuDNN Frontend can build the GeGLU half and down projection as separate graphs.
- cuDNN Frontend cannot build the full three-matmul FFN graph for this A6000/cuDNN path.
- CUDA graph replay is the meaningful cuDNN number here; direct repeated
  `graph.execute` timings include large frontend/host enqueue gaps.
- The split cuDNN Frontend replay baseline is about `1.000 ms/token` for Gemma 4
  31B decode FFN shapes.
- The current custom fused decode kernel is about `1.156 ms/token` with host launch,
  frontend, and wrapper overhead factored out through CUDA graph replay. Scratch clear
  is only about `0.001 ms/token`, so it does not explain the gap.
- The overhead-factored custom speedup versus cuDNN split replay is `0.866x`, so the
  current simple custom kernel is roughly `15.5%` slower than cuDNN's graph-replay split
  baseline, even though it avoids the full activation materialization and includes
  residual+RMSNorm epilogue work.
- The result says the current simple custom kernel is a useful correctness/design
  prototype, not yet a performance win. The next performance work needs to improve the
  work decomposition and tensor-core use, not just compare launch overhead.

## 2026-05-21 - FFN-down decode GEMV K-pack register staging

Runtime files:

- `src/gemma4_matmul_kernels.cu`
- `src/experiments/gemma4_decode_bench.cu`

Change:

- Added `GEMMA4_DECODE_GEMV_BUFFER_STAGES` as a compile-time experiment knob.
- Stage count `1` keeps the original simple loop.
- Stage counts `2-4` use an inlined ring buffer inside `dot_cols()`:
  - prefill `x_stage[stage]` and `w_stage[stage][col]`
  - consume `stage = iter % kStages`
  - load `pack_idx + kStages * Threads` into the same stage before accumulating the
    current copied pack
- No new public launcher or helper function was added.

Reason:

- Test whether K-pack software pipelining can hide HBM latency better than the existing
  per-pack/per-column loop.
- CUDA guide context: occupancy hides latency (pages 71 and 119), register/shared memory
  resources limit residency (pages 28 and 71), launch bounds affect register allocation
  (page 526), and extra local/stack traffic can change kernel behavior even without
  reported spill stores/loads.

Commands:

```bash
make decode-bench
GEMMA4_DECODE_BENCH_SEED=0x1234 \
  ./build/experiments/gemma4_decode_bench ffn_down 80 20 3

nvcc -std=c++17 -O3 -arch=sm_86 -DGEMMA4_DECODE_GEMV_BUFFER_STAGES=2 \
  -Isrc src/experiments/gemma4_decode_bench.cu src/gemma4_matmul_kernels.cu \
  -lcublas -lcudnn -o /tmp/gemma4_stage2/gemma4_decode_bench
GEMMA4_DECODE_BENCH_SEED=0x1234 \
  /tmp/gemma4_stage2/gemma4_decode_bench ffn_down 80 20 3

nvcc -std=c++17 -O3 -arch=sm_86 -DGEMMA4_DECODE_GEMV_BUFFER_STAGES=3 \
  -Isrc src/experiments/gemma4_decode_bench.cu src/gemma4_matmul_kernels.cu \
  -lcublas -lcudnn -o /tmp/gemma4_stage3/gemma4_decode_bench
GEMMA4_DECODE_BENCH_SEED=0x1234 \
  /tmp/gemma4_stage3/gemma4_decode_bench ffn_down 80 20 3

nvcc -std=c++17 -O3 -arch=sm_86 -DGEMMA4_DECODE_GEMV_BUFFER_STAGES=4 \
  -Isrc src/experiments/gemma4_decode_bench.cu src/gemma4_matmul_kernels.cu \
  -lcublas -lcudnn -o /tmp/gemma4_stage4/gemma4_decode_bench
GEMMA4_DECODE_BENCH_SEED=0x1234 \
  /tmp/gemma4_stage4/gemma4_decode_bench ffn_down 80 20 3
```

Environment:

- GPU: NVIDIA RTX A6000
- CUDA target: `sm_86`
- Clocks: not locked
- Cache policy: repeated-buffer warm-cache benchmark
- Warmup/timed repeats/trials: 20 warmup, 80 timed iterations, 3 trials

Resource usage for FFN-down identity decode:

| Stages | Registers/thread | Stack bytes/thread | Shared bytes/block |
| ---: | ---: | ---: | ---: |
| 1 | 58 | 0 | 1024 |
| 2 | 55 | 288 | 1024 |
| 3 | 58 | 432 | 1024 |
| 4 | 55 | 576 | 1024 |

Results:

| Stages | Custom best ms | Custom avg ms | Effective weight GB/s |
| ---: | ---: | ---: | ---: |
| 1 | 0.327842 | 0.327962 | 705.251 |
| 2 | 1.029579 | 1.030183 | 224.569 |
| 3 | 1.092698 | 1.092928 | 211.596 |
| 4 | 1.103177 | 1.103208 | 209.586 |

Correctness:

- Custom swizzle16, cuBLAS GEMV, and cuBLAS GEMM M=1 matched the custom output with zero
  reported absolute/relative difference in all staged runs.
- cuDNN conv1x1 retained the same expected BF16 accumulation-order differences.

Conclusion:

- Naive K-pack N-buffering is a large regression.
- The staged arrays do not stay entirely in registers. They create per-thread stack
  frames, so the attempted latency hiding adds local-memory traffic and extra copy work.
- The original stage-1 loop remains the right default. If K-pack lookahead is revisited,
  it needs a hand-shaped 2-stage variant with fewer live packs, lower `ColsPerBlock`, or
  a different thread/block decomposition instead of `w_stage[Stages][ColsPerBlock]`.

## 2026-05-21 - FFN-down decode GEMV register-lookahead weight load

Runtime files:

- `src/gemma4_matmul_kernels.cu`
- `src/experiments/gemma4_decode_bench.cu`

Change:

- Changed the decode GEMV inner column loop to issue the next column's weight-pack load
  before accumulating the current column.
- This is a register lookahead/software scheduling tweak only; it does not add
  `cp.async`, shared-memory staging, or a new public launcher.

Environment:

- GPU: NVIDIA RTX A6000
- Clocks: not locked
- CUDA target: `sm_86`
- Cache policy: repeated-buffer warm-cache benchmark
- Correctness reference: existing benchmark diffs against cuBLAS GEMV/GEMM and swizzled
  custom path

Command:

```bash
make decode-bench
GEMMA4_DECODE_BENCH_SEED=0x1234 \
  ./build/experiments/gemma4_decode_bench ffn_down 80 20 3
```

Results:

| Path | Best ms | Avg ms | Effective weight GB/s |
| --- | ---: | ---: | ---: |
| Custom GEMV with register lookahead | 0.327549 | 0.327685 | 705.883 |
| Custom swizzle16 with register lookahead | 0.327643 | 0.584443 | 705.679 |
| cuBLAS BF16 GEMV | 0.330064 | 0.465015 | 700.504 |
| cuBLAS BF16 GEMM M=1 | 0.573587 | 0.655130 | 403.097 |
| cuDNN BF16 conv1x1 | 0.332157 | 0.555408 | 696.090 |

Correctness:

- Custom swizzle16, cuBLAS GEMV, and cuBLAS GEMM M=1 matched the custom output with zero
  reported absolute/relative difference in this run.
- cuDNN conv1x1 differed by `max_abs_diff=0.25`, `mean_abs_diff=0.000392091`,
  `max_rel_diff=0.00763359`.

Conclusion:

- The register-lookahead edit produced only a tiny best-time improvement versus the prior
  logged FFN-down custom result (`0.327884 ms` -> `0.327549 ms`, about `0.10%`).
- This is within the range where repeated benchmark and `ncu` confirmation are needed
  before treating it as a real optimization.

## 2026-05-21 - RoPE 128-bit pack load baseline

Runtime file:

- `src/gemma4_rope.cu`
- `src/gemma4_rope.cuh`
- `tests/test_rope.cu`
- `src/experiments/gemma4_rope_bench.cu`
- `docs/rope.md`

Change:

- Replaced the scalar hot-path RoPE loop with a packed path that processes eight rotary
  pairs per active thread.
- Q/K halves now use `Bf16Packed128` loads and stores on the full-pack path.
- Cos/sin tables use two `float4` loads each per eight-pair pack.
- Kept the scalar pair path as a tail fallback for non-multiple-of-eight rotary widths.
- Reduced the RoPE thread block from 128 threads to one warp, because Gemma 4's sliding
  and global rotary widths need only 16 and 8 packed lanes per head respectively.
- Updated the forward-layout RoPE entry point, tests, docs, and benchmark to use compact
  cos/sin rows of `rotary_dim / 2` instead of full `head_dim` rows. This removes unused
  table storage in the forward path:
  - sliding rows: 128 floats instead of 256
  - global rows: 64 floats instead of 512

Build and inspection commands:

```bash
make test-rope
mkdir -p build/ptx
nvcc -std=c++17 -O3 -arch=sm_86 -Isrc -ptx src/gemma4_rope.cu \
  -o build/ptx/gemma4_rope_vectorized.ptx
nvcc -std=c++17 -O3 -arch=sm_86 -Isrc -Xptxas -v -c src/gemma4_rope.cu \
  -o build/ptx/gemma4_rope_vectorized.o
```

PTX/compiler result:

- Full-pack path emits `ld.global.v4.u32` and `st.global.v4.u32` for BF16 Q/K halves.
- Full-pack path emits `ld.global.nc.v4.f32` for cos/sin table loads.
- `ptxas` reports no spills or stack frame.
- Register count increased versus the scalar version:
  - low-level layout kernel: 58 registers
  - forward-layout kernel: 60 registers

Correctness:

```text
rope tests passed
```

Benchmark command:

```bash
make rope-bench
GEMMA4_ROPE_BENCH_SEED=0x20260521 \
  ./build/experiments/gemma4_rope_bench 100 20 3 1024 1 1
```

Environment:

- GPU: NVIDIA RTX A6000
- CUDA/NVCC: 12.9.86
- Build flags: `-std=c++17 -O3 -arch=sm_86`
- Warmup/timed repeats/trials: 20 warmup, 100 timed iterations, 3 trials
- Cache policy: repeated-buffer warm-cache benchmark
- Clock policy: clocks not locked
- Correctness tolerance: benchmark compares against cuDNN BF16-table path and reports
  `0.0078125` max absolute difference for all measured cases.

Selected custom graph timings:

| Case | Seq | Graph ms | Effective GiB/s |
| --- | ---: | ---: | ---: |
| Sliding | 1 | 0.001741 | 52.573 |
| Sliding | 256 | 0.015432 | 1518.728 |
| Sliding | 1024 | 0.076612 | 1223.706 |
| Global | 1 | 0.001575 | 21.793 |
| Global | 256 | 0.006732 | 1305.534 |
| Global | 1024 | 0.029884 | 1176.405 |

Compact-table follow-up:

```bash
make test-rope
make rope-bench
GEMMA4_ROPE_BENCH_SEED=0x20260521 \
  ./build/experiments/gemma4_rope_bench 100 20 3 1024 1 1
```

Correctness:

```text
rope tests passed
```

Selected compact-table custom graph timings:

| Case | Seq | Graph ms | Effective GiB/s | Speedup vs packed/full-table run |
| --- | ---: | ---: | ---: | ---: |
| Sliding | 1 | 0.001754 | 52.199 | 0.993x |
| Sliding | 256 | 0.016488 | 1421.488 | 0.936x |
| Sliding | 1024 | 0.076759 | 1221.355 | 0.998x |
| Global | 1 | 0.001643 | 20.902 | 0.959x |
| Global | 256 | 0.006704 | 1311.080 | 1.004x |
| Global | 1024 | 0.029817 | 1179.069 | 1.002x |

Conclusion:

- The C++ pack path successfully produces 128-bit global memory ops without inline PTX.
- The change is correctness-clean in the existing RoPE tests.
- The higher register count is the main tradeoff to watch. It did not spill in this
  build, but this shape should be benchmarked against the scalar kernel before treating
  it as a settled performance win.
- Compact cos/sin tables fixed the unused table storage in the forward path. Custom
  kernel speed was essentially neutral because the kernel already loaded only the used
  rotary columns; the practical win is lower table allocation footprint and cleaner
  strides, especially for global p-RoPE.

Cache-order follow-up:

- Changed the RoPE grid from `(rows, heads)` to `(heads, rows)` so heads are the fastest
  grid dimension, attempting to improve temporal locality for each position's cos/sin
  row.
- Changed packed Q/K input loads from normal `load128` to streaming `load128cs`, while
  keeping output stores normal.
- PTX confirmed the intended lowering:
  - row now uses `%ctaid.y`
  - head now uses `%ctaid.x`
  - Q/K packed loads emit `ld.global.cs.v4.s32`
  - cos/sin packed loads remain `ld.global.nc.v4.f32`
- `ptxas` reported unchanged register/spill profile: 58 registers for the low-level
  kernel, 60 for the forward-layout kernel, 0 spills.

Correctness:

```text
rope tests passed
```

Benchmark command:

```bash
GEMMA4_ROPE_BENCH_SEED=0x20260521 \
  ./build/experiments/gemma4_rope_bench 100 20 3 1024 1 1
```

Selected custom graph timings versus the compact-table packed run:

| Case | Seq | Prior graph ms | Cache-order graph ms | Speedup |
| --- | ---: | ---: | ---: | ---: |
| Sliding | 1 | 0.001754 | 0.001736 | 1.010x |
| Sliding | 256 | 0.016488 | 0.015046 | 1.096x |
| Sliding | 1024 | 0.076759 | 0.076900 | 0.998x |
| Global | 1 | 0.001643 | 0.001582 | 1.039x |
| Global | 256 | 0.006704 | 0.006824 | 0.982x |
| Global | 1024 | 0.029817 | 0.031695 | 0.941x |

Conclusion:

- The cache-order change is not a broad speedup. It helped sliding `seq=256` in this
  short run, was neutral for sliding `seq=1024`, and regressed global `seq=1024`.
- Do not treat head-fast grid order plus streaming Q/K loads as a retained win without a
  cleaner A/B split. The two changes should be separated in a later experiment because
  the grid order may help cos/sin locality while `ld.global.cs` may hurt Q/K reuse or
  scheduling.

Split A/B follow-up:

- Added compile-time controls:
  - `GEMMA4_ROPE_HEAD_FAST_GRID`
  - `GEMMA4_ROPE_QK_LOAD_CS`
- Built four variants from the same source:
  - baseline: row-fast grid, normal Q/K loads
  - headfast: head-fast grid, normal Q/K loads
  - loadcs: row-fast grid, streaming Q/K loads
  - both: head-fast grid, streaming Q/K loads

Build pattern:

```bash
nvcc -std=c++17 -O3 -arch=sm_86 -Isrc \
  -DGEMMA4_ROPE_HEAD_FAST_GRID=<0|1> \
  -DGEMMA4_ROPE_QK_LOAD_CS=<0|1> \
  src/experiments/gemma4_rope_bench.cu src/gemma4_rope.cu -lcudnn \
  -o build/experiments/rope_variants/<variant>
```

Benchmark command for each variant:

```bash
GEMMA4_ROPE_BENCH_SEED=0x20260521 \
  ./build/experiments/rope_variants/<variant> 100 20 3 1024 1 1
```

Custom CUDA graph timings:

| Case | Seq | Baseline | Head-fast only | `load128cs` only | Both | Best |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Sliding | 1 | 0.001666 | 0.001707 | 0.001553 | 0.001646 | `load128cs` |
| Sliding | 4 | 0.001672 | 0.001959 | 0.001639 | 0.001901 | `load128cs` |
| Sliding | 16 | 0.002134 | 0.002122 | 0.002083 | 0.002015 | both |
| Sliding | 64 | 0.003097 | 0.003188 | 0.003091 | 0.003065 | both |
| Sliding | 256 | 0.016482 | 0.016036 | 0.015215 | 0.014943 | both |
| Sliding | 1024 | 0.076787 | 0.076737 | 0.075305 | 0.076946 | `load128cs` |
| Global | 1 | 0.001496 | 0.001556 | 0.001711 | 0.001563 | baseline |
| Global | 4 | 0.001565 | 0.001646 | 0.001601 | 0.001651 | baseline |
| Global | 16 | 0.001658 | 0.001820 | 0.001638 | 0.001956 | `load128cs` |
| Global | 64 | 0.002805 | 0.002773 | 0.002658 | 0.002917 | `load128cs` |
| Global | 256 | 0.006691 | 0.006674 | 0.006654 | 0.006777 | `load128cs` |
| Global | 1024 | 0.029790 | 0.031470 | 0.029804 | 0.031625 | baseline |

Conclusion:

- Head-fast grid order alone was not a clear win and hurt several small/large cases.
- `load128cs` alone was the best isolated change overall: it improved sliding
  `seq=1024` by about `2%`, improved several mid-size sliding/global cases, and was
  essentially tied with baseline at global `seq=1024`.
- The retained default is row-fast grid with `load128cs` enabled. Head-fast grid remains
  available behind `GEMMA4_ROPE_HEAD_FAST_GRID=1` for future experiments.

## 2026-05-21 - RoPE FP32 accumulation numerical consistency

Runtime files:

- `src/gemma4_rope.cu`
- `src/experiments/gemma4_rope_numeric_check.cu`

Reason:

- Check whether the RoPE element math should stay in FP32 before rounding back to BF16.
- CUDA guide reference points used for the standard:
  - FMA can avoid precision loss during cancellation (CUDA Programming Guide p.583).
  - `x * y + z -> __fmaf_rn(x, y, z)` has a stated `0 ULP` mapping for float basic
    operations (p.594).
  - BF16 has the same range as FP32 but only 7 bits of precision (p.574), so intermediate
    BF16 rounding is expected to be visibly worse.

Method:

- Randomize BF16 Q/K values and FP32 cos/sin tables.
- Run the actual GPU RoPE kernel.
- Compare every rotated BF16 output against:
  - a CPU FP32/FMA reference rounded once to BF16, matching the current kernel's math;
  - a double reference rounded to BF16;
  - a simulated BF16-intermediate path that rounds both products before the add.
- Also check global p-RoPE's NoPE tail for bit-exact preservation.
- This is a numerical-consistency experiment, not a timing benchmark.

Build and run:

```bash
nvcc -std=c++17 -O3 -arch=sm_86 -Isrc \
  src/experiments/gemma4_rope_numeric_check.cu src/gemma4_rope.cu \
  -o build/experiments/gemma4_rope_numeric_check
./build/experiments/gemma4_rope_numeric_check 64 53 2 0x20260521
```

Environment:

- GPU: NVIDIA RTX A6000
- CUDA/NVCC: 12.9.86
- Build flags: `-std=c++17 -O3 -arch=sm_86`
- Shapes: Gemma 4 sliding (`head_dim=256`, full RoPE) and global
  (`head_dim=512`, `rotary_dim=128` p-RoPE)

Results:

| Case | Rotated outputs | GPU == FP32/FMA | GPU == double | BF16-intermediate == double | GPU mean BF16 ULP vs double | BF16-intermediate mean BF16 ULP vs double | NoPE exact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Sliding | 83,361,792 | 1.00000000 | 0.99998682 | 0.66429912 | 0.00002023 | 10.25257449 | 1.00000000 |
| Global | 31,260,672 | 1.00000000 | 0.99998650 | 0.66435267 | 0.00001676 | 10.17400861 | 1.00000000 |

SASS check:

- The hot path uses FP32 `FMUL` plus `FFMA`, followed by `F2FP.BF16.PACK_AB` conversion
  for stores.

Conclusion:

- Keep the current FP32 RoPE arithmetic and final BF16 rounding.
- The GPU kernel is bit-exact against the FP32/FMA reference over this randomized sweep.
- Simulated BF16 intermediates are much less consistent: only about `66.4%` exact versus
  the double-rounded BF16 reference, with about `10` mean BF16 ULP error.
- Global p-RoPE's unrotated NoPE dimensions remained bit-exact.

## 2026-05-21 - FFN-down decode GEMV kernel comparison

Runtime files:

- `src/gemma4_matmul_kernels.cu`
- `src/experiments/gemma4_decode_bench.cu`

Reason:

- Check the project GEMV kernel for the FFN-down shape after the tinygrad/cuBLAS/cuBLASLt
  FFN-down matmul sweeps.
- This is the decode `M=1` case, not the prefill `M > 1` matmul case.

Shape:

- FFN-down decode GEMV: `K=21504`, `N=5376`, BF16 input/weights/output, FP32
  accumulation.

Command:

```bash
make decode-bench
GEMMA4_DECODE_BENCH_SEED=0x1234 \
  ./build/experiments/gemma4_decode_bench ffn_down 80 20 3
```

Results:

| Path | Best ms | Avg ms | Effective weight GB/s | Speedup vs custom |
| --- | ---: | ---: | ---: | ---: |
| Custom GEMV | 0.327884 | 0.327917 | 705.160 | 1.000x |
| Custom swizzle16 GEMV | 0.327907 | 0.327931 | 705.112 | 0.9999x |
| cuBLAS BF16 GEMV | 0.329957 | 0.330215 | 700.731 | 0.9937x |
| cuBLAS BF16 GEMM M=1 | 0.329998 | 0.330041 | 700.645 | 0.9936x |
| cuDNN BF16 conv1x1 | 0.331502 | 0.331585 | 697.465 | 0.9891x |

Correctness:

- Custom, custom swizzle16, cuBLAS GEMV, and cuBLAS GEMM M=1 matched with zero reported
  absolute/relative difference in this run.
- cuDNN conv1x1 differed by `max_abs_diff=0.25`, `mean_abs_diff=0.000392091`,
  `max_rel_diff=0.00763359`.

Conclusion:

- For FFN-down decode `M=1`, the project custom GEMV kernel is slightly faster than
  cuBLAS GEMV, cuBLAS GEMM M=1, and cuDNN conv1x1 on this run.
- The win is small: about `1.006x` over cuBLAS and `1.011x` over cuDNN.
- This result should not be extrapolated to prefill `M > 1`; the GEMV kernel is a
  single-token decode path.

## 2026-05-21 - tinygrad late-eval BF16 matmul vs cuBLASLt

Runtime file:

- `src/experiments/tinygrad_late_eval_bench.py`

Change:

- Added a cuBLASLt comparison path to the same benchmark file.
- The Python script builds a tiny C++ shared helper under `build/experiments/` on first
  use so cuBLASLt descriptors, heuristics, and opaque algorithm structs stay in C++.
- cuBLASLt uses the same column-major equivalence as the simple cuBLAS path:
  `C_col[n,m] = B_col[n,k] * A_col[k,m]`, matching row-major
  `C[m,n] = A[m,k] * B[k,n]` without explicit transposes.
- Timing uses CUDA events around the cuBLASLt matmul on a nonblocking stream, with warmup
  outside the timing loop.

Shape:

- Gemma 4 FFN-down-like GEMM: `M x 21504` by `21504 x 5376`.
- Data type: BF16 inputs, FP32 accumulation/output.

Environment:

- GPU: NVIDIA RTX A6000
- Clocks: not locked
- Cache policy: repeated-buffer warm-cache microbenchmark
- Warmup/timed repeats: `WARMUP=3`, `CNT=5`
- tinygrad search: `SEARCH_BEAM=2`
- cuBLASLt workspace: default `67108864` bytes
- cuBLASLt heuristic request count: default `64`

Command:

```bash
DEV=CUDA M_LIST=1,2,4,8,16,32,64,128,256,288,304,312,320,384,512 \
  N=5376 K=21504 DTYPE_IN=bfloat16 DTYPE_ACC=float WARMUP=3 CNT=5 \
  SEARCH_BEAM=2 COMPARE_CUBLAS=1 COMPARE_CUBLASLT=1 \
  python3 src/experiments/tinygrad_late_eval_bench.py
```

Results:

| M | tinygrad best us | cuBLAS best us | cuBLASLt best us | tiny/cuBLAS | tiny/cuBLASLt | Winner vs cuBLASLt |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 343.30 | 2842.05 | 359.14 | 8.279x | 1.046x | tinygrad |
| 2 | 377.34 | 2913.57 | 343.10 | 7.721x | 0.909x | cuBLASLt |
| 4 | 475.87 | 2920.38 | 359.04 | 6.137x | 0.754x | cuBLASLt |
| 8 | 364.03 | 2841.02 | 352.22 | 7.804x | 0.968x | cuBLASLt |
| 16 | 449.41 | 3366.94 | 351.39 | 7.492x | 0.782x | cuBLASLt |
| 32 | 526.53 | 3222.88 | 350.53 | 6.121x | 0.666x | cuBLASLt |
| 64 | 792.10 | 2595.26 | 363.74 | 3.276x | 0.459x | cuBLASLt |
| 128 | 1081.31 | 2038.46 | 404.77 | 1.885x | 0.374x | cuBLASLt |
| 256 | 1696.80 | 2583.97 | 448.06 | 1.523x | 0.264x | cuBLASLt |
| 288 | 1703.17 | 3745.79 | 702.85 | 2.199x | 0.413x | cuBLASLt |
| 304 | 2709.09 | 2378.91 | 705.92 | 0.878x | 0.261x | cuBLASLt |
| 312 | 1864.90 | 2406.37 | 715.30 | 1.290x | 0.384x | cuBLASLt |
| 320 | 1892.35 | 2277.60 | 714.56 | 1.204x | 0.378x | cuBLASLt |
| 384 | 1903.49 | 797.89 | 791.94 | 0.419x | 0.416x | cuBLASLt |
| 512 | 2366.78 | 813.82 | 802.05 | 0.344x | 0.339x | cuBLASLt |

Conclusion:

- cuBLASLt is the real baseline here. tinygrad only beat cuBLASLt at `M=1`, and only by
  about `4.6%`.
- From `M=2` onward, cuBLASLt wins in this sweep.
- The earlier tinygrad wins over simple cuBLAS `GemmEx` do not survive against cuBLASLt.
- For the FFN-down shape, there is no useful tinygrad integration region from this sweep
  unless a later generated kernel beats cuBLASLt rather than plain cuBLAS.

## 2026-05-21 - tinygrad late-eval BF16 matmul vs simple cuBLAS GemmEx

Runtime file:

- `src/experiments/tinygrad_late_eval_bench.py`

Change:

- Added a simple ctypes cuBLAS `cublasGemmEx` benchmark path to the existing tinygrad
  late-eval matmul bench.
- The cuBLAS path allocates raw device buffers, warms up outside the timed loop, and uses
  CUDA events on the same nonblocking stream as the cuBLAS call.
- The cuBLAS call uses column-major equivalence for row-major buffers:
  `C_col[n,m] = B_col[n,k] * A_col[k,m]`, matching row-major
  `C[m,n] = A[m,k] * B[k,n]` without explicit transposes.
- The ctypes path pins to CUDA 12 libraries when available and calls `cudaSetDevice(0)`
  before cuBLAS timing to restore the CUDA runtime primary context after tinygrad driver
  API work.

Shape:

- Gemma 4 FFN-down-like GEMM: `M x 21504` by `21504 x 5376`.
- Data type: BF16 inputs, FP32 accumulation/output.

Environment:

- GPU: NVIDIA RTX A6000
- Clocks: not locked
- Cache policy: repeated-buffer warm-cache microbenchmark
- Timing: tinygrad internal event timing for generated kernels; CUDA event timing for
  simple cuBLAS `GemmEx`
- Warmup/timed repeats: `WARMUP=3`, `CNT=5`
- tinygrad search: `SEARCH_BEAM=2`

Commands:

```bash
DEV=CUDA M_LIST=16,32,64,128,256,512,1024 N=5376 K=21504 \
  DTYPE_IN=bfloat16 DTYPE_ACC=float WARMUP=3 CNT=5 SEARCH_BEAM=2 \
  COMPARE_CUBLAS=1 python3 src/experiments/tinygrad_late_eval_bench.py

DEV=CUDA M_LIST=288,320,352,384,416,448,480 N=5376 K=21504 \
  DTYPE_IN=bfloat16 DTYPE_ACC=float WARMUP=3 CNT=5 SEARCH_BEAM=2 \
  COMPARE_CUBLAS=1 python3 src/experiments/tinygrad_late_eval_bench.py

DEV=CUDA M_LIST=296,304,312 N=5376 K=21504 \
  DTYPE_IN=bfloat16 DTYPE_ACC=float WARMUP=3 CNT=5 SEARCH_BEAM=2 \
  COMPARE_CUBLAS=1 python3 src/experiments/tinygrad_late_eval_bench.py
```

Results:

| M | tinygrad best us | cuBLAS best us | tiny/cuBLAS | Winner |
| ---: | ---: | ---: | ---: | --- |
| 16 | 436.80 | 2552.70 | 5.844x | tinygrad |
| 32 | 487.10 | 1985.60 | 4.076x | tinygrad |
| 64 | 801.79 | 2698.75 | 3.366x | tinygrad |
| 128 | 1072.64 | 2784.58 | 2.596x | tinygrad |
| 256 | 1676.06 | 2228.58 | 1.330x | tinygrad |
| 288 | 1650.46 | 2832.67 | 1.716x | tinygrad |
| 296 | 2760.03 | 3464.51 | 1.255x | tinygrad |
| 304 | 2696.10 | 2237.95 | 0.830x | cuBLAS |
| 312 | 1845.15 | 2550.53 | 1.382x | tinygrad |
| 320 | 1899.81 | 1881.73 | 0.990x | cuBLAS |
| 352 | 2127.07 | 1975.74 | 0.929x | cuBLAS |
| 384 | 1930.05 | 797.47 | 0.413x | cuBLAS |
| 416 | 2231.49 | 819.20 | 0.367x | cuBLAS |
| 448 | 2337.79 | 814.24 | 0.348x | cuBLAS |
| 480 | 2129.70 | 809.38 | 0.380x | cuBLAS |
| 512 | 2370.43 | 813.63 | 0.343x | cuBLAS |
| 1024 | 6015.20 | 4337.31 | 0.721x | cuBLAS |

Conclusion:

- Against this intentionally simple cuBLAS `GemmEx` baseline, tinygrad's beam-2 generated
  kernel wins clearly through `M=288` and still wins at `M=296` and `M=312`.
- There is not a clean monotonic crossover because both tinygrad and cuBLAS select
  different generated/library paths by shape. The first measured losing point is `M=304`;
  `M=320` and larger measured points lose except the earlier `M=312` outlier.
- Practical cutoff from this sweep: treat `M <= 288` as the stable tinygrad-win region
  for this simple cuBLAS comparison, with shape-specific wins possible around `M=296-312`.
- This does not supersede prior cuBLASLt results. The earlier cuBLASLt FFN-down baseline
  remains much stronger and is still the relevant integration target.

## 2026-05-20 - tinygrad late-eval BF16 FFN-down bench

Runtime file:

- `src/experiments/tinygrad_late_eval_bench.py`

Reason:

- Exercise tinygrad's normal lazy/late evaluation path without using `experiments/tinygrad/extra`.
- The file mirrors the internal path from tinygrad proper:
  `Tensor.empty -> Tensor.matmul -> Tensor.schedule_linear -> compile_linear -> run_linear`.
- `compile_linear` and `run_linear` are copied into the file and call the same tinygrad
  pattern matchers (`pm_beam`, `pm_compile`, `pm_optimize_local_size`, `pm_exec`) as the
  upstream engine.
- Shape is Gemma 4 FFN-down-like: `M x 21504` by `21504 x 5376`.

Environment:

- GPU: NVIDIA RTX A6000
- tinygrad checkout: `experiments/tinygrad`, commit `9e88b08`
- Clocks: not locked
- Cache policy: repeated-buffer warm-cache timing
- Data type: BF16 inputs, FP32 accumulation/output through `Tensor.matmul(..., dtype=dtypes.float)`

Commands:

```bash
DEV=CUDA M_LIST=64,128 N=5376 K=21504 DTYPE_IN=bfloat16 DTYPE_ACC=float \
  WARMUP=2 CNT=5 SEARCH_BEAM=0 \
  python3 src/experiments/tinygrad_late_eval_bench.py

DEV=CUDA M_LIST=64,128 N=5376 K=21504 DTYPE_IN=bfloat16 DTYPE_ACC=float \
  WARMUP=1 CNT=3 SEARCH_BEAM=2 \
  python3 src/experiments/tinygrad_late_eval_bench.py

python3 -m py_compile src/experiments/tinygrad_late_eval_bench.py
```

Results:

| M | Search beam | best us | median us | TFLOP/s | Program |
| ---: | ---: | ---: | ---: | ---: | --- |
| 64 | 0 | 2373.44 | 2385.28 | 6.23 | `r_42_32_4_2_2_4_4_1344_2` |
| 128 | 0 | 2751.23 | 2766.24 | 10.76 | `r_2_42_32_4_2_2_4_4_1344_2` |
| 64 | 2 | 802.30 | 805.38 | 18.44 | `r_4_84_32_2_2_2_4_1344_2` |
| 128 | 2 | 1065.73 | 1066.43 | 27.77 | `r_4_96_32_2_2_7_2_1344_2` |

Conclusion:

- The one-file late-eval reproduction works and uses tinygrad proper instead of `extra`.
- Beam search improves the generated BF16 kernel substantially, but the result is still
  far behind the logged cuBLASLt FFN-down timings (`M=64` around `0.342 ms`, `M=128`
  around `0.351-0.377 ms`).
- This path is useful for inspecting tinygrad's generated program choices, but it is not
  currently a replacement candidate for the FFN-down cuBLASLt path.

Direct library check:

```bash
PYTHONPATH=experiments/tinygrad DEV=CUDA python3 - <<'PY'
# Direct Tensor.matmul(...).realize() loop with DEBUG=2 for tinygrad event timing.
PY
```

Direct tinygrad `Tensor.matmul` with BF16 inputs and FP32 accumulation, `BEAM=0`,
`WARMUP=3`, `CNT=10`:

| M | tinygrad best ms | tinygrad median ms | TFLOP/s | Logged cuBLASLt ms | tinygrad/cuBLASLt |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 2.3799 | 2.3950 | 6.22 | 0.3419 | 0.14x |
| 128 | 2.7416 | 2.7553 | 10.79 | 0.3512 | 0.13x |
| 256 | 2.8034 | 2.8196 | 21.11 | 0.4945 | 0.18x |
| 512 | 4.5297 | 4.6244 | 26.13 | 0.9604 | 0.21x |

Confirmation rerun with `WARMUP=5`, `CNT=20`:

| M | tinygrad best ms | tinygrad median ms | tinygrad worst ms | TFLOP/s |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 2.3809 | 2.3906 | 2.4171 | 6.22 |
| 128 | 2.7396 | 2.7535 | 2.7656 | 10.80 |
| 256 | 2.7986 | 2.8047 | 2.8207 | 21.15 |
| 512 | 4.4820 | 4.5934 | 4.6854 | 26.41 |

## 2026-05-20 - Fresh tinygrad max_matmul Gemma FFN-down M sweep

Runtime files:

- `experiments/tinygrad/extra/gemm/max_matmul.py`
- `experiments/tinygrad/extra/gemm/max_kernels/`

Source:

- Re-cloned `https://github.com/tinygrad/tinygrad.git` into `experiments/tinygrad`.
- tinygrad commit: `9e88b08`.
- Local compatibility patch: `max_matmul.py` now uses `CUDADevice("cuda:0").compiler`
  because the fresh checkout no longer exports `CUDACompiler` from
  `tinygrad.runtime.ops_cuda`.
- Local benchmark patch: added `VALIDATE=0` to skip the NumPy reference matmul for
  larger timing-only sweeps.

Shape:

- Gemma 4 FFN-down-like GEMM: `M x 21504` by `21504 x 5376`.
- `M` is the token count / prefill row count.

Environment:

- GPU: NVIDIA RTX A6000
- Driver: 580.126.16
- CUDA/NVCC: 12.9 (`nvcc V12.9.86`)
- Clocks: not locked
- Cache policy: repeated-buffer warm-cache microbenchmark
- Timing: tinygrad `CUDAProgram(..., wait=True)` CUDA event timing; script reports min
  over `CNT`

FP16 input, FP32 accumulate/output, upstream `GEMM_VARIATION=max`:

Command pattern:

```bash
PYTHONPATH=. CUDA=1 VALIDATE=0 GEMM_VARIATION=max \
  DTYPE_IN=half DTYPE_OUT=float DTYPE_ACC=float \
  M=<M> N=5376 K=21504 CNT=30 INPUT=RAND \
  python3 extra/gemm/max_matmul.py
```

| M | min us | TFLOP/s |
| ---: | ---: | ---: |
| 64 | 347.26 | 42.61 |
| 128 | 366.78 | 80.69 |
| 192 | 508.93 | 87.23 |
| 256 | 635.17 | 93.19 |
| 320 | 901.31 | 82.09 |
| 384 | 1066.08 | 83.28 |
| 448 | 1216.35 | 85.16 |
| 512 | 1223.23 | 96.78 |
| 576 | 1481.28 | 89.91 |
| 640 | 1562.27 | 94.72 |
| 704 | 1728.74 | 94.16 |
| 768 | 1817.76 | 97.69 |
| 832 | 2057.86 | 93.48 |
| 896 | 2232.64 | 92.79 |
| 960 | 2337.34 | 94.96 |
| 1024 | 2433.31 | 97.30 |
| 1152 | 2838.46 | 93.84 |
| 1280 | 3237.86 | 91.40 |
| 1536 | 4312.86 | 82.34 |
| 1792 | 5229.02 | 79.24 |
| 2048 | 5979.49 | 79.19 |

Validation note:

- Default validation passed for the first random-input runs at `M=64,128,192,256`.
- The default `rtol=1e-3, atol=1e-2` check failed on isolated elements at `M=320` and
  `M=384` with max absolute violations around `0.010-0.013`; timings were still
  recorded.
- The wider sweep used `VALIDATE=0`, so those numbers are timing-only.

FP16 input, FP16 accumulate/output, upstream `GEMM_VARIATION=max`:

Command pattern:

```bash
PYTHONPATH=. CUDA=1 VALIDATE=0 GEMM_VARIATION=max \
  DTYPE_IN=half DTYPE_OUT=half DTYPE_ACC=half \
  M=<M> N=5376 K=21504 CNT=30 INPUT=RAND \
  python3 extra/gemm/max_matmul.py
```

| M | min us | TFLOP/s |
| ---: | ---: | ---: |
| 256 | 791.62 | 74.77 |
| 512 | 810.30 | 146.09 |
| 768 | 1584.99 | 112.03 |
| 1024 | 1578.72 | 149.97 |
| 1280 | 2347.46 | 126.07 |
| 1536 | 2374.30 | 149.58 |
| 1792 | 3122.05 | 132.71 |
| 2048 | 3157.95 | 149.95 |

Representative FP32-accumulating variant comparison:

Command pattern:

```bash
PYTHONPATH=. CUDA=1 VALIDATE=0 GEMM_VARIATION=<variant> \
  DTYPE_IN=half DTYPE_OUT=float DTYPE_ACC=float \
  M=<M> N=5376 K=21504 CNT=20 INPUT=RAND \
  python3 extra/gemm/max_matmul.py
```

| Variant | M=64 us | M=256 us | M=512 us | M=1024 us |
| --- | ---: | ---: | ---: | ---: |
| `max` | 346.88 | 639.58 | 1223.68 | 2443.62 |
| `2_stage_swizzled_smem_input` | 348.93 | 633.12 | 1215.20 | 2490.08 |
| `swizzled_smem_input` | 477.54 | 784.86 | 1347.97 | 2950.46 |
| `flat_smem_input` | 888.42 | 1497.25 | 2789.63 | 5349.54 |

Conclusion:

- For the FP32-accumulating path, `max` and `2_stage_swizzled_smem_input` are close;
  `2_stage_swizzled_smem_input` edged out `max` at `M=256` and `M=512`, while `max`
  was better at `M=64` and `M=1024`.
- `swizzled_smem_input` and `flat_smem_input` are not competitive for this shape.
- The FP16-accumulating/output path is faster at several aligned `M` values but is not a
  Gemma correctness candidate without a separate numerical validation argument.
- These numbers are useful for shape exploration only; the earlier cuBLASLt baseline
  remains the relevant comparison before considering any integration.

### Existing BF16 binary checkpoint after fresh tinygrad reclone

After the fresh tinygrad reclone, the local BF16 source files and C++ harness that had
been under `experiments/tinygrad` were gone, but the previously built BF16 binaries still
existed under `build/experiments/tinygrad`. I used those binaries only as a no-build
checkpoint before deciding whether source restoration/tuning was worth doing.

Command pattern:

```bash
GEMMA4_PREFILL_CUBLASLT_HEURISTICS=64 GEMMA4_PREFILL_GRAPH_REPEATS=10 \
  ./build/experiments/tinygrad/<binary> \
  64,128,192,256,320,384,448,512,640,768,896,1024,1280,1536,1792,1984 \
  5 20 2
```

Best observed tinygrad BF16 variant per selected `M`:

| M | Best tinygrad variant | tinygrad ms | cuBLASLt ms | Speedup |
| ---: | --- | ---: | ---: | ---: |
| 64 | `bf16_max` | 0.33655 | 0.34189 | 1.016x |
| 128 | `bf16_max` | 0.38819 | 0.35117 | 0.905x |
| 192 | `bf16_max` | 0.55025 | 0.48043 | 0.873x |
| 256 | `bf16_max` | 0.72732 | 0.49453 | 0.680x |
| 320 | `bf16_max` | 1.05651 | 0.80418 | 0.761x |
| 512 | `bf16_splitk3` | 1.40796 | 0.95171 | 0.676x |
| 768 | `bf16_splitk3` | 2.08574 | 1.40242 | 0.672x |
| 1024 | `bf16_splitk3` | 2.79948 | 1.89341 | 0.676x |
| 1536 | `bf16_splitk3` | 4.31061 | 2.75091 | 0.638x |
| 1984 | `bf16_splitk3` | 5.80529 | 3.73639 | 0.644x |

Variant notes:

- `bf16_max` and `bf16_2_stage_swizzled_smem_input` are effectively tied; both only beat
  cuBLASLt at `M=64` by about `1.6%`.
- `bf16_splitk3` is worse at small `M`, slightly better than `bf16_max` at larger `M`,
  but still far behind cuBLASLt.
- No tested BF16 tinygrad static max variant reaches the requested `1.10x` win threshold
  over cuBLASLt on this FFN-down shape.

Post-reclone cleanup:

- The fresh reclone intentionally removed local-only tinygrad FFN-down sources such as
  `experiments/tinygrad/gemma4_ffn_down_tinygrad_bench.cu` and local BF16 kernel ports.
- Removed the stale Makefile tinygrad FFN-down targets instead of restoring those files.
- Deleted the briefly recreated `nv.bf16_f32_bf16.max.cu` source so the tinygrad checkout
  stays aligned with the intentional reclone cleanup.

## 2026-05-20 - tinygrad ffn_down dynamic-M GEMM sweep

Runtime files:

- `experiments/tinygrad/extra/gemm/`
- `experiments/tinygrad/gemma4_ffn_down_tinygrad_bench.cu`
- `Makefile`

Source:

- Cloned `https://github.com/tinygrad/tinygrad.git` into `experiments/tinygrad`
  with sparse checkout paths `extra/gemm` and `tinygrad`.
- tinygrad commit: `a19fa29`.

Reason:

- Test whether the NVIDIA kernels in tinygrad's `extra/gemm/max_kernels` can beat cuBLAS
  on Gemma 4 `ffn_down`: `M x 21504` by `21504 x 5376`.
- The first valid comparison uses tinygrad's FP16 input, FP32 accumulation, FP32 output
  `nv.fp16_fp32_fp32.max.cu` kernel. This is not yet the final Gemma BF16 datatype path,
  but it isolates the schedule on the exact `ffn_down` dimensions with dynamic `M`.

Implementation:

- Added `gemma4_ffn_down_tinygrad_bench.cu`, a direct CUDA/C++ harness that compiles the
  imported tinygrad kernel source and compares it against `cublasGemmEx`.
- Input layout is row-major `A[M,K]` and `B[K,N]`; cuBLAS uses the equivalent
  column-major `C^T[N,M] = B^T[N,K] * A^T[K,M]`.
- Timed with CUDA events on one stream after warmup. Reported medians across trials.
- Checked max absolute output difference against cuBLAS output for every `M`.
- Added `make tinygrad-ffn-down-bench`.

Build:

```bash
make tinygrad-ffn-down-bench
```

Timing command:

```bash
./build/experiments/tinygrad/gemma4_ffn_down_tinygrad_max_bench 64,128,192,256,320 25 200 7
```

Environment:

- GPU: NVIDIA RTX A6000
- Driver: 580.126.16
- CUDA/NVCC: 12.9 (`nvcc V12.9.86`)
- Clocks: not locked for this run
- Cache policy: repeated-buffer warm-cache microbenchmark
- Correctness reference: cuBLAS FP16 input, FP32 accumulation, FP32 output

Selected stable results:

| M | tinygrad median ms | cuBLAS median ms | Speedup | tinygrad TFLOP/s | cuBLAS TFLOP/s | Max abs |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 0.33812 | 3.44232 | 10.181x | 43.76 | 4.30 | 0.0065002 |
| 128 | 0.39303 | 3.04844 | 7.756x | 75.30 | 9.71 | 0.0083313 |
| 192 | 0.55271 | 3.10412 | 5.616x | 80.32 | 14.30 | 0.00737 |
| 256 | 0.72626 | 3.08555 | 4.249x | 81.50 | 19.18 | 0.00737 |
| 320 | 1.08439 | 3.23990 | 2.988x | 68.23 | 22.84 | 0.0082092 |

Transition check:

```bash
./build/experiments/tinygrad/gemma4_ffn_down_tinygrad_max_bench 64,128,192,256,320,384,448,512 20 100 5
```

- tinygrad continued to win through `M=320`.
- cuBLAS became faster from `M=384` onward in that run:
  `M=384` tinygrad `1.16068 ms`, cuBLAS `0.91630 ms`;
  `M=512` tinygrad `1.41390 ms`, cuBLAS `0.95771 ms`.

Other tinygrad variants tested:

- `nv.fp16_fp32_fp32.2_stage_swizzled_smem_input.cu`
- `nv.fp16_fp32_fp32.swizzled_smem_input.cu`
- `nv.fp16_fp32_fp32.flat_smem_input.cu`
- `nv.fp16_fp16_fp16.max.cu`

The FP32-output `max` kernel was the best valid tinygrad candidate. The FP16-accumulating
kernel produced invalid output for this shape (`max_abs` around `246-262`) and impossible
timing numbers, so it is not a candidate.

Conclusion:

- Success for dynamic short-prefill / microbatch `ffn_down` values: the imported
  tinygrad `max` schedule consistently outperforms cuBLAS by much more than 15% for
  `M in {64, 128, 192, 256, 320}` on this RTX A6000.
- This does not replace cuBLAS for all dynamic `M`: cuBLAS is faster by `M=384+`.
- Next useful step is a dispatcher experiment that uses this tinygrad kernel for
  `M <= 320` and cuBLAS/cuBLASLt above that cutoff, then ports the valid schedule to
  Gemma's BF16 input/output contract before considering it for the inference path.

Correction after applying the SGEMM cuBLASLt tuning:

- The initial table compared against plain `cublasGemmEx`. After wiring in the SGEMM
  cuBLASLt baseline style, the conclusion changes.
- The tinygrad harness now uses row-major cuBLASLt descriptors, a 32 MiB workspace,
  `GEMMA4_PREFILL_CUBLASLT_HEURISTICS` returned algorithms, and optional graph replay
  via `GEMMA4_PREFILL_GRAPH_REPEATS`. It times all returned successful cuBLASLt
  algorithms and reports the fastest median.

Short-M retest:

```bash
GEMMA4_PREFILL_CUBLASLT_HEURISTICS=64 GEMMA4_PREFILL_GRAPH_REPEATS=20 \
  ./build/experiments/tinygrad/gemma4_ffn_down_tinygrad_max_bench 64,128,192,256,320 10 50 3
```

| M | tinygrad median ms | cuBLASLt median ms | Speedup | Winner |
| ---: | ---: | ---: | ---: | --- |
| 64 | 0.33783 | 0.34437 | 1.019x | tinygrad |
| 128 | 0.41421 | 0.36750 | 0.887x | cuBLASLt |
| 192 | 0.58297 | 0.50515 | 0.867x | cuBLASLt |
| 256 | 0.75381 | 0.53224 | 0.706x | cuBLASLt |
| 320 | 1.15604 | 0.85894 | 0.743x | cuBLASLt |

Representative large-M retest:

```bash
GEMMA4_PREFILL_CUBLASLT_HEURISTICS=64 GEMMA4_PREFILL_GRAPH_REPEATS=10 \
  ./build/experiments/tinygrad/gemma4_ffn_down_tinygrad_max_bench 1024,1536,1984 5 20 2
```

| M | tinygrad median ms | cuBLASLt median ms | Speedup | Winner |
| ---: | ---: | ---: | ---: | --- |
| 1024 | 3.00374 | 2.04608 | 0.681x | cuBLASLt |
| 1536 | 4.59643 | 2.91769 | 0.635x | cuBLASLt |
| 1984 | 6.12065 | 3.97396 | 0.649x | cuBLASLt |

Updated conclusion:

- Against the tuned cuBLASLt baseline, this imported tinygrad kernel is not a useful
  `ffn_down` replacement except possibly the marginal `M=64` case.
- The earlier "tinygrad wins through M=320" conclusion should be treated as a comparison
  against an insufficient plain cuBLAS baseline, not the optimized library path.

BF16-native tinygrad port:

- Added `experiments/tinygrad/extra/gemm/max_kernels/nv.bf16_f32_bf16.max.cu`.
- Ported the tinygrad low-level schedule from FP16 to BF16 by changing the input/shared
  element type to `__nv_bfloat16`, changing the inline MMA instruction to
  `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`, and converting the FP32
  accumulators to BF16 in the output epilogue.
- Renamed the inherited half-fragment helpers in the BF16 source to BF16-specific names
  and store four BF16 outputs as two packed `__nv_bfloat162` values.
- Added `make tinygrad-ffn-down-bf16-bench`.
- CUDA guide context: BF16 is supported as `__nv_bfloat16` via `<cuda_bf16.h>` on compute
  capability 8.0+ (guide p.587), and WMMA/tensor-core BF16 inputs with FP32 accumulator
  support include 16x16x16, 32x8x16, and 8x32x16 shapes (guide p.577).

BF16 short-M command:

```bash
GEMMA4_PREFILL_CUBLASLT_HEURISTICS=64 GEMMA4_PREFILL_GRAPH_REPEATS=20 \
  ./build/experiments/tinygrad/gemma4_ffn_down_tinygrad_bf16_max_bench 64,128,192,256,320 10 50 3
```

| M | tinygrad BF16 median ms | cuBLASLt BF16 median ms | Speedup | Winner | Max abs |
| ---: | ---: | ---: | ---: | --- | ---: |
| 64 | 0.33516 | 0.34146 | 1.019x | tinygrad | 1 |
| 128 | 0.38952 | 0.35107 | 0.901x | cuBLASLt | 2 |
| 192 | 0.55607 | 0.48234 | 0.867x | cuBLASLt | 1 |
| 256 | 0.71795 | 0.48819 | 0.680x | cuBLASLt | 1 |
| 320 | 1.04997 | 0.79592 | 0.758x | cuBLASLt | 2 |

BF16 representative large-M command:

```bash
GEMMA4_PREFILL_CUBLASLT_HEURISTICS=64 GEMMA4_PREFILL_GRAPH_REPEATS=10 \
  ./build/experiments/tinygrad/gemma4_ffn_down_tinygrad_bf16_max_bench 1024,1536,1984 5 20 2
```

| M | tinygrad BF16 median ms | cuBLASLt BF16 median ms | Speedup | Winner | Max abs |
| ---: | ---: | ---: | ---: | --- | ---: |
| 1024 | 2.78194 | 1.86801 | 0.671x | cuBLASLt | 1 |
| 1536 | 4.32392 | 2.72715 | 0.631x | cuBLASLt | 0 |
| 1984 | 5.86556 | 3.69266 | 0.630x | cuBLASLt | 1 |

BF16 conclusion:

- The native BF16 tinygrad port works and is correctly wired against BF16 cuBLASLt, but
  it still does not beat the tuned cuBLASLt baseline except for a negligible `M=64`
  margin.
- The paired BF16 epilogue is not the bottleneck enough to close the gap; further work
  needs a different tile schedule or more parallelism across `M`/`N` to compete at
  `M >= 128`.

Static BF16 variant/resource follow-up:

- Per the user direction, do not use Tinygrad codegen for the next tuning steps. These
  checks use only the existing static CUDA sources and manual builds.
- `ncu` is installed, but profiling this cuBLAS-linked harness aborts in the current
  Thunder environment with an unsupported-library runtime assertion, so `ncu` timing and
  stall breakdowns were not available here.
- Static resource usage from `cuobjdump --dump-resource-usage`:

| Variant | Registers/thread | Static shared memory/block |
| --- | ---: | ---: |
| `bf16_max` | 148 | 49152 B |
| `bf16_2stage_swizzled` | 148 | 49152 B |
| `bf16_swizzled` | 126 | 24576 B |
| `bf16_flat` | 124 | 24576 B |
| `bf16_splitk2_partial` | 148 | 49152 B |

- Compiling `bf16_max` with `-maxrregcount=128` did not change the resource usage
  (`148` registers/thread) and did not improve performance.

Register-cap check:

```bash
nvcc -std=c++17 -O3 -arch=sm_86 -maxrregcount=128 -Isrc \
  -DTINYGRAD_BF16 \
  -DTINYGRAD_KERNEL_NAME=tinygrad_bf16_max_r128_kernel \
  -DTINYGRAD_VARIANT_NAME='"bf16_max_r128"' \
  -Dwmma_example=tinygrad_bf16_max_r128_kernel \
  experiments/tinygrad/gemma4_ffn_down_tinygrad_bench.cu \
  experiments/tinygrad/extra/gemm/max_kernels/nv.bf16_f32_bf16.max.cu \
  -lcublas -lcublasLt \
  -o build/experiments/tinygrad/gemma4_ffn_down_tinygrad_bf16_max_r128_bench

GEMMA4_PREFILL_CUBLASLT_HEURISTICS=64 GEMMA4_PREFILL_GRAPH_REPEATS=20 \
  ./build/experiments/tinygrad/gemma4_ffn_down_tinygrad_bf16_max_r128_bench \
  64,128,192,256,320 10 50 3
```

| M | tinygrad BF16 median ms | cuBLASLt BF16 median ms | Speedup | Winner |
| ---: | ---: | ---: | ---: | --- |
| 64 | 0.33546 | 0.34155 | 1.018x | tinygrad |
| 128 | 0.39058 | 0.35103 | 0.899x | cuBLASLt |
| 192 | 0.55646 | 0.48126 | 0.865x | cuBLASLt |
| 256 | 0.72776 | 0.49553 | 0.681x | cuBLASLt |
| 320 | 1.06855 | 0.81340 | 0.761x | cuBLASLt |

Static lower-smem variants, same command shape:

| Variant | M=64 | M=128 | M=192 | M=256 | M=320 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `bf16_2stage_swizzled` | 0.33558 | 0.38943 | 0.55745 | 0.72433 | 1.05142 |
| `bf16_swizzled` | 0.43545 | 0.47631 | 0.75441 | 0.89968 | 1.15999 |
| `bf16_flat` | 0.83093 | 0.84163 | 1.28007 | 1.33860 | 1.94461 |

Follow-up conclusion:

- The lower-smem variants have more favorable resident-block resources but much weaker
  runtime, so the current gap is not fixed just by reducing static shared memory.
- `bf16_max` and `bf16_2stage_swizzled` are essentially tied; `bf16_max` remains the best
  static source to mutate manually.
- At `M=128`, the current `64x128` tile launches `2 * 42 = 84` CTAs on the RTX A6000,
  exactly one CTA per SM. The next manual edit should split the output tile along `M` or
  `N` without adding a separate reduction pass; split-K already increased CTA count but
  lost to reduction overhead.

Manual M32 tile split:

- Added `experiments/tinygrad/extra/gemm/max_kernels/nv.bf16_f32_bf16.max_m32.cu`.
- Added `make tinygrad-ffn-down-bf16-m32-bench`.
- This variant mutates `bf16_max` by computing one 32-row half of the original 64-row
  tile per CTA. It removes the `acc_frag_1_*` half, avoids the unused A rows, and reduces
  static shared memory from `49152 B` to `40960 B`.
- Resource usage: `98` registers/thread, `40960 B` static shared memory/block.

Command:

```bash
GEMMA4_PREFILL_CUBLASLT_HEURISTICS=64 GEMMA4_PREFILL_GRAPH_REPEATS=20 \
  ./build/experiments/tinygrad/gemma4_ffn_down_tinygrad_bf16_max_m32_bench \
  32,64,128,192,256,320 10 50 3
```

| M | tinygrad BF16 M32 median ms | cuBLASLt BF16 median ms | Speedup | Winner | Max abs |
| ---: | ---: | ---: | ---: | --- | ---: |
| 32 | 0.32951 | 0.33135 | 1.006x | tinygrad | 1 |
| 64 | 0.33258 | 0.34164 | 1.027x | tinygrad | 1 |
| 128 | 0.55674 | 0.35207 | 0.632x | cuBLASLt | 2 |
| 192 | 0.90226 | 0.49165 | 0.545x | cuBLASLt | 1 |
| 256 | 1.11075 | 0.50386 | 0.454x | cuBLASLt | 1 |
| 320 | 1.43987 | 0.82116 | 0.570x | cuBLASLt | 2 |

M32 conclusion:

- Splitting `M` helps only the tiny cases that were already at or near a win.
- For `M >= 128`, duplicated B tile loading dominates; the M32 split is much slower than
  the original M64 tile and much slower than tuned cuBLASLt.

Launch-bounds check:

- Added a temporary manual `bf16_max_lb2` source with `__launch_bounds__(128, 2)`.
- Resource usage remained unchanged at `148` registers/thread and `49152 B` shared
  memory/block.

Command:

```bash
GEMMA4_PREFILL_CUBLASLT_HEURISTICS=64 GEMMA4_PREFILL_GRAPH_REPEATS=20 \
  ./build/experiments/tinygrad/gemma4_ffn_down_tinygrad_bf16_max_lb2_bench \
  64,128,192,256,320 10 50 3
```

| M | tinygrad BF16 LB2 median ms | cuBLASLt BF16 median ms | Speedup | Winner | Max abs |
| ---: | ---: | ---: | ---: | --- | ---: |
| 64 | 0.33636 | 0.34197 | 1.017x | tinygrad | 1 |
| 128 | 0.41138 | 0.35392 | 0.860x | cuBLASLt | 2 |
| 192 | 0.58439 | 0.50280 | 0.860x | cuBLASLt | 1 |
| 256 | 0.76209 | 0.51433 | 0.675x | cuBLASLt | 1 |
| 320 | 1.11949 | 0.83936 | 0.750x | cuBLASLt | 2 |

Launch-bounds conclusion:

- Adding `minBlocksPerMultiprocessor=2` did not improve register allocation or runtime.
- Leave the original `bf16_max` launch bound unchanged.

Split-K partition sweep:

- Split-K=2 had already shown that extra CTA count was not enough to pay for partial
  writes plus a reduction pass.
- Tested split-K=3 and split-K=4 with the same partial kernel to complete that axis.

Commands:

```bash
nvcc -std=c++17 -O3 -arch=sm_86 -Isrc \
  -DTINYGRAD_BF16 -DTINYGRAD_SPLITK_PARTS=3 \
  -DTINYGRAD_KERNEL_NAME=tinygrad_bf16_splitk3_partial_kernel \
  -DTINYGRAD_VARIANT_NAME='"bf16_splitk3"' \
  -Dwmma_example=tinygrad_bf16_splitk3_partial_kernel \
  experiments/tinygrad/gemma4_ffn_down_tinygrad_bench.cu \
  experiments/tinygrad/extra/gemm/max_kernels/nv.bf16_f32_bf16.max_splitk_partial.cu \
  -lcublas -lcublasLt \
  -o build/experiments/tinygrad/gemma4_ffn_down_tinygrad_bf16_splitk3_bench

nvcc -std=c++17 -O3 -arch=sm_86 -Isrc \
  -DTINYGRAD_BF16 -DTINYGRAD_SPLITK_PARTS=4 \
  -DTINYGRAD_KERNEL_NAME=tinygrad_bf16_splitk4_partial_kernel \
  -DTINYGRAD_VARIANT_NAME='"bf16_splitk4"' \
  -Dwmma_example=tinygrad_bf16_splitk4_partial_kernel \
  experiments/tinygrad/gemma4_ffn_down_tinygrad_bench.cu \
  experiments/tinygrad/extra/gemm/max_kernels/nv.bf16_f32_bf16.max_splitk_partial.cu \
  -lcublas -lcublasLt \
  -o build/experiments/tinygrad/gemma4_ffn_down_tinygrad_bf16_splitk4_bench
```

Each benchmark used:

```bash
GEMMA4_PREFILL_CUBLASLT_HEURISTICS=64 GEMMA4_PREFILL_GRAPH_REPEATS=20 \
  ./build/experiments/tinygrad/<binary> 64,128,192,256,320 10 50 3
```

| Variant | M=64 | M=128 | M=192 | M=256 | M=320 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `bf16_max` | 0.33516 | 0.38952 | 0.55607 | 0.71795 | 1.04997 |
| `bf16_splitk2` | 0.34609 | 0.40396 | 0.56928 | n/a | n/a |
| `bf16_splitk3` | 0.35457 | 0.40852 | 0.58423 | 0.74276 | 1.12005 |
| `bf16_splitk4` | 0.36298 | 0.42713 | 0.60784 | 0.75694 | 1.13357 |

Split-K conclusion:

- Higher split-K degrees monotonically worsened this shape in the tested range.
- The extra parallelism does not pay for the extra global partial writes and reduction.
- Do not pursue split-K for this BF16 output contract unless the reduction can be fused
  into a materially different persistent/stream-K design.

ptxas load-cache policy check:

- Tested `-Xptxas -dlcm=ca` and `-Xptxas -dlcm=cg` on the best `bf16_max` source.
- These are static compiler-policy variants only; no Tinygrad codegen was used.
- The runs below were sequential. An initial parallel launch was killed because it would
  have contaminated GPU timings.

Commands:

```bash
nvcc -std=c++17 -O3 -arch=sm_86 -Xptxas -dlcm=ca -Isrc \
  -DTINYGRAD_BF16 \
  -DTINYGRAD_KERNEL_NAME=tinygrad_bf16_max_ca_kernel \
  -DTINYGRAD_VARIANT_NAME='"bf16_max_ca"' \
  -Dwmma_example=tinygrad_bf16_max_ca_kernel \
  experiments/tinygrad/gemma4_ffn_down_tinygrad_bench.cu \
  experiments/tinygrad/extra/gemm/max_kernels/nv.bf16_f32_bf16.max.cu \
  -lcublas -lcublasLt \
  -o build/experiments/tinygrad/gemma4_ffn_down_tinygrad_bf16_max_ca_bench

nvcc -std=c++17 -O3 -arch=sm_86 -Xptxas -dlcm=cg -Isrc \
  -DTINYGRAD_BF16 \
  -DTINYGRAD_KERNEL_NAME=tinygrad_bf16_max_cg_kernel \
  -DTINYGRAD_VARIANT_NAME='"bf16_max_cg"' \
  -Dwmma_example=tinygrad_bf16_max_cg_kernel \
  experiments/tinygrad/gemma4_ffn_down_tinygrad_bench.cu \
  experiments/tinygrad/extra/gemm/max_kernels/nv.bf16_f32_bf16.max.cu \
  -lcublas -lcublasLt \
  -o build/experiments/tinygrad/gemma4_ffn_down_tinygrad_bf16_max_cg_bench
```

Each benchmark used:

```bash
GEMMA4_PREFILL_CUBLASLT_HEURISTICS=64 GEMMA4_PREFILL_GRAPH_REPEATS=20 \
  ./build/experiments/tinygrad/<binary> 64,128,192,256,320 10 50 3
```

| Variant | M=64 | M=128 | M=192 | M=256 | M=320 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `bf16_max` | 0.33516 | 0.38952 | 0.55607 | 0.71795 | 1.04997 |
| `bf16_max_ca` | 0.33632 | 0.39692 | 0.56981 | 0.73697 | 1.09925 |
| `bf16_max_cg` | 0.33616 | 0.40489 | 0.57470 | 0.74982 | 1.09670 |

Cache-policy conclusion:

- Both cache policy variants were slower than the default compile.
- Leave the default ptxas cache policy for this kernel.

Final no-codegen retest:

- Per the user correction, no Tinygrad codegen was used in this pass. The only attempted
  new schedule change was a manual static CUDA edit.
- Hardened `gemma4_ffn_down_tinygrad_bench.cu` so the cuBLASLt comparison filters
  algorithms that return a runtime launch failure instead of aborting the whole
  benchmark on one bad heuristic. cuBLASLt plan construction, heuristic search, and
  warmup remain outside the timed CUDA-event/graph-replay region.
- Retested the validated native BF16 `bf16_max` kernel on the requested large-M range:

```bash
make tinygrad-ffn-down-bf16-bench

GEMMA4_PREFILL_CUBLASLT_HEURISTICS=64 GEMMA4_PREFILL_GRAPH_REPEATS=10 \
  ./build/experiments/tinygrad/gemma4_ffn_down_tinygrad_bf16_max_bench \
  1024,1536,1984 5 20 2
```

| M | tinygrad BF16 median ms | cuBLASLt BF16 median ms | Speedup | Winner | Max abs | cuBLASLt algo |
| ---: | ---: | ---: | ---: | --- | ---: | ---: |
| 1024 | 2.78306 | 1.87530 | 0.674x | cuBLASLt | 1 | 1 |
| 1536 | 4.33235 | 2.73662 | 0.632x | cuBLASLt | 2 | 0 |
| 1984 | 5.85225 | 3.69111 | 0.631x | cuBLASLt | 1 | 1 |

Manual N64 branch:

- Tried a manual `64x64` output-tile variant to increase CTA count without split-K
  reduction overhead.
- The first epilogue edit wrote past the last 64-column tile. After constraining stores
  to the first 64 columns and restoring the missing row stores, a one-block smoke harness
  still reported a delayed illegal memory access from the kernel.
- The N64 variant was discarded and not promoted to the Makefile.

cuBLASLt workspace check:

- Added `GEMMA4_PREFILL_CUBLASLT_WORKSPACE_MB` to the harness so the cuBLASLt baseline
  can be tested with larger workspaces without changing the timed region.
- Tested `32`, `128`, and `512` MiB with `GEMMA4_PREFILL_CUBLASLT_HEURISTICS=64` and
  graph replay. Larger workspaces did not improve the best cuBLASLt large-M result.

| Workspace | M=1024 cuBLASLt ms | M=1536 cuBLASLt ms | M=1984 cuBLASLt ms |
| ---: | ---: | ---: | ---: |
| 32 MiB | 1.86400 | 2.73221 | 3.67780 |
| 128 MiB | 1.88715 | 2.74786 | 3.69017 |
| 512 MiB | 1.88933 | 2.75041 | 3.71312 |

Large-M static variant check:

```bash
GEMMA4_PREFILL_CUBLASLT_HEURISTICS=64 GEMMA4_PREFILL_GRAPH_REPEATS=10 \
  ./build/experiments/tinygrad/gemma4_ffn_down_tinygrad_bf16_<variant>_bench \
  1024,1536,1984 5 20 2
```

| Variant | M=1024 tiny ms | M=1536 tiny ms | M=1984 tiny ms | Conclusion |
| --- | ---: | ---: | ---: | --- |
| `bf16_max` | 2.78079 | 4.31409 | 5.78123 | Best static baseline. |
| `bf16_2_stage_swizzled_smem_input` | 2.80982 | 4.31478 | 5.81267 | Essentially tied/slightly worse. |
| `bf16_swizzled_smem_input` | 3.40250 | 5.14542 | 6.79865 | Reject. |
| `bf16_flat_smem_input` | 5.03938 | 7.48701 | 10.07056 | Reject. |

Large-M split-K check:

| Variant | M=1024 tiny ms | M=1536 tiny ms | M=1984 tiny ms | Conclusion |
| --- | ---: | ---: | ---: | --- |
| `bf16_splitk2` | 2.79765 | 4.34413 | 5.79682 | Still behind cuBLASLt. |
| `bf16_splitk3` | 2.77403 | 4.25981 | 5.71401 | Best split-K point, still only `0.645x` at M=1984. |
| `bf16_splitk4` | 2.83319 | 4.35334 | 5.77718 | Reject. |

ptxas check:

```bash
nvcc -std=c++17 -O3 -arch=sm_86 -Xptxas -v -Isrc \
  -DTINYGRAD_BF16 \
  -DTINYGRAD_KERNEL_NAME=tinygrad_bf16_max_v_kernel \
  -DTINYGRAD_VARIANT_NAME='"bf16_max_v"' \
  -Dwmma_example=tinygrad_bf16_max_v_kernel \
  experiments/tinygrad/gemma4_ffn_down_tinygrad_bench.cu \
  experiments/tinygrad/extra/gemm/max_kernels/nv.bf16_f32_bf16.max.cu \
  -lcublas -lcublasLt \
  -o /tmp/tiny_bf16_v
```

- `tinygrad_bf16_max_v_kernel`: `148` registers/thread, `49152` B shared memory,
  `0` spill stores, `0` spill loads.
- The current gap is not register spilling; it is the tile schedule/runtime efficiency
  versus cuBLASLt.

Final conclusion for this branch:

- Native BF16 is implemented for the imported static Tinygrad `bf16_max` kernel and
  compares correctly against BF16 cuBLASLt.
- With cuBLASLt overhead factored out and up to 64 heuristics tested, cuBLASLt remains
  much faster for `M=1000-2000`. The static Tinygrad schedule is only marginally useful
  at very small `M`.

## 2026-05-20 - Tuna BF16 prefill GEMM adaptation baseline

Runtime files:

- `experiments/tuna/gemma4_prefill_bench.cu`
- `Makefile`

Reason:

- The imported Tuna kernel is tuned for FP16 x FP4 GEMM with quantized activations.
- Gemma 4 prefill projections use BF16 activations and BF16 weights with variable token
  count `M`: `Y[M,N] = X[M,K] * W[N,K]^T`.
- This pass establishes a Gemma-specific BF16 benchmark harness before deeper tuning or
  more invasive Tuna pipeline changes.

Gemma 4 projection shapes covered:

| Op | K | N |
| --- | ---: | ---: |
| `ffn_gate_up` | 5376 | 43008 |
| `ffn_down` | 21504 | 5376 |
| `sliding_qkv` | 5376 | 16384 |
| `sliding_o` | 8192 | 5376 |
| `global_q` | 5376 | 16384 |
| `global_k` | 5376 | 2048 |
| `global_o` | 16384 | 5376 |
| `final_logits` | 5376 | 262144 |

Implementation:

- Added `gemma4_prefill_bench.cu` under the Tuna experiment folder.
- Replaced Tuna's FP4 activation quantization assumption with BF16 x BF16 WMMA tensor-core
  kernels using FP32 accumulation and BF16 output conversion.
- The benchmark uses the same weight layout as the main Gemma decode helpers:
  weights are stored as `[N,K]`, so logical matrix B is loaded as column-major `[K,N]`.
- Added global-load WMMA tile variants and a first shared-memory-staged variant:
  `wmma_16x16`, `wmma_16x32`, `wmma_16x64`, `wmma_32x64`, `wmma_64x64`,
  `smem_16x64`, `smem_16x128`, `smem_32x64`, `smem_32x128`, and `smem_64x64`.
- Added a `make tuna-prefill-bench` target.

Build:

```bash
make tuna-prefill-bench
```

Correctness:

- Compared each run against `cublasGemmEx` BF16 input, FP32 accumulate, BF16 output.
- Observed max absolute differences were `0`, `0.015625`, or `0.03125` depending on
  shape and reduction length, consistent with BF16 output rounding/order differences.

Benchmark commands:

```bash
./build/experiments/gemma4_tuna_prefill_bench global_k 80 20 1,16,64,256,1024
./build/experiments/gemma4_tuna_prefill_bench ffn_gate_up 10 3 16,64,256,1024 wmma_64x64
./build/experiments/gemma4_tuna_prefill_bench sliding_qkv 20 5 16,64,256,1024 wmma_64x64
```

Selected results:

| Op | M | Best/custom config | Custom ms | cuBLAS ms | Speedup |
| --- | ---: | --- | ---: | ---: | ---: |
| `global_k` | 16 | `wmma_16x32` | 0.0665 | 1.4408 | 21.67x |
| `global_k` | 64 | `wmma_16x16` | 0.0860 | 1.9580 | 22.78x |
| `global_k` | 256 | `wmma_16x16` | 0.3292 | 0.0593 | 0.18x |
| `global_k` | 1024 | `wmma_32x64` | 1.2957 | 1.6951 | 1.31x |
| `ffn_gate_up` | 16 | `wmma_64x64` | 1.5670 | 0.6687 | 0.43x |
| `ffn_gate_up` | 64 | `wmma_64x64` | 1.5616 | 0.6917 | 0.44x |
| `ffn_gate_up` | 256 | `wmma_64x64` | 6.4659 | 0.8452 | 0.13x |
| `sliding_qkv` | 16 | `wmma_64x64` | 0.7812 | 1.4959 | 1.92x |
| `sliding_qkv` | 64 | `wmma_64x64` | 0.7803 | 0.2814 | 0.36x |
| `sliding_qkv` | 256 | `wmma_64x64` | 2.5746 | 0.4059 | 0.16x |

Profiling:

```bash
ncu --metric gpu__time_duration.sum --target-processes all \
  ./build/experiments/gemma4_tuna_prefill_bench global_k 1 0 256 wmma_16x16
```

- `ncu` failed on this Thunder instance with an internal Thunder runtime assertion for
  an unsupported library path. CUDA-event timing remains the available timing source for
  this pass.

Conclusion:

- The first BF16 Tuna adaptation is correct and shape-aware, but not yet a competitive
  replacement for cuBLAS on the dominant wide prefill projections.
- Global-load WMMA variants are currently better than the simple shared-memory-staged
  variants for `global_k`; the staged version likely needs vectorized loads, bank-conflict
  work, and a real async pipeline before it can pay for its synchronization overhead.
- The only promising cases in this pass are narrow/small-M projections, especially
  `global_k` at small `M` and `sliding_qkv` at `M=16`.
- Next Tuna work should either port more of the original pipelined Tuna machinery to
  BF16 or stop spending time on the current WMMA baseline for wide prefill shapes.

## 2026-05-20 - SGEMM Gemma-shape prefill baseline

Runtime files:

- `experiments/sgemm.cu/gemma4_prefill_bench.cu`
- `Makefile`

Reason:

- The imported SGEMM implementation is tuned around square FP32 GEMMs.
- Gemma 4 prefill projections have fixed `K,N` projection dimensions and variable token
  count `M`, so the original square benchmark is not representative.

Implementation:

- Added a Gemma-shape benchmark that reuses the imported `sgemm()` launcher.
- The benchmark covers the same Gemma projection shapes as the Tuna benchmark.
- This is explicitly an FP32 shape/tile baseline. It does not yet replace the datatype
  path with Gemma's BF16 inference math.
- Added `make sgemm-prefill-bench`.

Build:

```bash
make sgemm-prefill-bench
```

Benchmark commands:

```bash
./build/experiments/gemma4_sgemm_prefill_bench global_k 20 5 16,64,256
./build/experiments/gemma4_sgemm_prefill_bench ffn_gate_up 5 2 16,64
```

Selected results:

| Op | M | Custom ms | cuBLAS ms | Speedup | Max abs |
| --- | ---: | ---: | ---: | ---: | ---: |
| `global_k` | 16 | 0.5476 | 1.8707 | 3.416x | 1.43051e-05 |
| `global_k` | 64 | 0.5468 | 2.0852 | 3.813x | 9.53674e-06 |
| `global_k` | 256 | 0.5576 | 1.7969 | 3.222x | 1.19209e-05 |
| `ffn_gate_up` | 16 | 1.9557 | 1.3309 | 0.681x | 1.23978e-05 |
| `ffn_gate_up` | 64 | 2.0290 | 1.3708 | 0.676x | 0 |

Conclusion:

- The imported SGEMM shape policy can beat `cublasSgemm` for the narrow `global_k`
  FP32 baseline on this machine, but loses on the wide `ffn_gate_up` shape.
- This does not satisfy the final Gemma datatype requirement because the kernel remains
  FP32. The next SGEMM step is to decide whether to port its cp.async/shared-memory
  schedule to BF16 tensor cores or treat it as a non-candidate for the BF16 inference
  path.

## 2026-05-20 - SGEMM BF16 prefill adaptation

Runtime files:

- `experiments/sgemm.cu/gemma4_bf16_prefill_bench.cu`
- `Makefile`

Reason:

- The FP32 SGEMM shape benchmark does not match Gemma 4 inference datatype.
- This pass adds a BF16 x BF16 tensor-core path under the SGEMM experiment and benchmarks
  SGEMM-style CTA shapes against cuBLAS across variable `M`.

Implementation:

- Added `make sgemm-bf16-prefill-bench`.
- Added BF16 WMMA kernels with FP32 accumulation and BF16 output.
- Weight layout matches the main Gemma projection layout: `[N,K]`, interpreted as
  column-major `[K,N]` for `Y[M,N] = X[M,K] * W[N,K]^T`.
- SGEMM-style tile variants tested:
  `bf16_16x32`, `bf16_16x64`, `bf16_32x64`, `bf16_64x64`,
  `bf16_64x128`, and `bf16_128x64`.
- Larger square-ish CTA shapes from the imported SGEMM source cannot be copied directly
  with one warp per 16x16 WMMA tile because a true `128x128` tile would require 64 warps
  / 2048 threads. The largest tested shapes stay within CUDA's legal block size.

Build:

```bash
make sgemm-bf16-prefill-bench
```

Benchmark commands:

```bash
./build/experiments/gemma4_sgemm_bf16_prefill_bench global_k 40 10 16,64,256
./build/experiments/gemma4_sgemm_bf16_prefill_bench ffn_gate_up 10 3 16,64,256 bf16_64x64
./build/experiments/gemma4_sgemm_bf16_prefill_bench sliding_qkv 20 5 16,64,256 bf16_64x64
```

Selected results:

| Op | M | Best/custom config | Custom ms | cuBLAS ms | Speedup | Max abs |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| `global_k` | 16 | `bf16_16x32` | 0.0678 | 1.5484 | 22.824x | 0.015625 |
| `global_k` | 64 | `bf16_16x64` | 0.1007 | 1.8713 | 18.583x | 0.015625 |
| `global_k` | 256 | `bf16_16x32` | 0.3317 | 0.0637 | 0.192x | 0 |
| `ffn_gate_up` | 16 | `bf16_64x64` | 1.5684 | 0.6677 | 0.426x | 0 |
| `ffn_gate_up` | 64 | `bf16_64x64` | 1.5668 | 0.6911 | 0.441x | 0.015625 |
| `ffn_gate_up` | 256 | `bf16_64x64` | 6.5583 | 0.8496 | 0.130x | 0 |
| `sliding_qkv` | 16 | `bf16_64x64` | 0.7790 | 1.2916 | 1.658x | 0.015625 |
| `sliding_qkv` | 64 | `bf16_64x64` | 0.7787 | 0.2786 | 0.358x | 0 |
| `sliding_qkv` | 256 | `bf16_64x64` | 2.5505 | 0.4059 | 0.159x | 0 |

Conclusion:

- The SGEMM BF16 adaptation is correct against cuBLAS at BF16 output tolerance.
- For variable `M`, small row tiles win on narrow `global_k` at small M; larger CTA-style
  tiles waste too much work there.
- On dominant wide-N prefill projections, this simple WMMA adaptation loses badly to
  cuBLAS. Further SGEMM work would need a deeper tensor-core schedule with multiple WMMA
  tiles per warp and a real cp.async pipeline rather than the current one-warp-per-tile
  implementation.

## 2026-05-20 - Prefill GEMM focused tuner automation

Runtime files:

- `src/experiments/gemma4_prefill_tune.py`
- `build/experiments/gemma4_prefill_tune/tuna_focused.csv`
- `build/experiments/gemma4_prefill_tune/sgemm_bf16_focused.csv`

Reason:

- Manual one-off benchmark commands made it too easy to compare a config that was good
  at one `M` but poor across the variable-M prefill range.
- The tuner runs benchmark binaries sequentially, writes CSV rows, and ranks configs by
  geometric mean speedup plus worst-case speedup across the chosen M sweep.

Validation:

```bash
python3 -m py_compile src/experiments/gemma4_prefill_tune.py

python3 src/experiments/gemma4_prefill_tune.py \
  --backend sgemm-bf16 --ops global_k \
  --configs bf16_16x32,bf16_64x64 --m 16,64 --iters 5 --warmup 2 \
  --out build/experiments/gemma4_prefill_tune/sgemm_bf16_smoke.csv

python3 src/experiments/gemma4_prefill_tune.py \
  --backend tuna --ops global_k \
  --configs wmma_16x32,wmma_64x64 --m 16,64 --iters 5 --warmup 2 \
  --out build/experiments/gemma4_prefill_tune/tuna_smoke.csv
```

Focused tuning commands:

```bash
python3 src/experiments/gemma4_prefill_tune.py \
  --backend tuna --ops global_k,sliding_qkv,ffn_gate_up \
  --configs wmma_16x32,wmma_16x64,wmma_32x64,wmma_64x64 \
  --m 16,64,256 --iters 10 --warmup 3 \
  --out build/experiments/gemma4_prefill_tune/tuna_focused.csv

python3 src/experiments/gemma4_prefill_tune.py \
  --backend sgemm-bf16 --ops global_k,sliding_qkv,ffn_gate_up \
  --configs bf16_16x32,bf16_16x64,bf16_32x64,bf16_64x64,bf16_64x128,bf16_128x64 \
  --m 16,64,256 --iters 10 --warmup 3 \
  --out build/experiments/gemma4_prefill_tune/sgemm_bf16_focused.csv
```

Dispatch summary command:

```bash
python3 src/experiments/gemma4_prefill_tune.py summarize \
  build/experiments/gemma4_prefill_tune/tuna_focused.csv \
  build/experiments/gemma4_prefill_tune/sgemm_bf16_focused.csv \
  --custom-threshold 1.0
```

Focused Tuna summary:

| Op | Best config | Geomean speedup | Worst speedup |
| --- | --- | ---: | ---: |
| `ffn_gate_up` | `wmma_64x64` | 0.293x | 0.133x |
| `global_k` | `wmma_32x64` | 10.330x | 0.162x |
| `sliding_qkv` | `wmma_32x64` | 0.511x | 0.081x |

Focused SGEMM BF16 summary:

| Op | Best config | Geomean speedup | Worst speedup |
| --- | --- | ---: | ---: |
| `ffn_gate_up` | `bf16_64x64` | 0.293x | 0.133x |
| `global_k` | `bf16_16x64` | 6.024x | 0.184x |
| `sliding_qkv` | `bf16_64x64` | 0.672x | 0.161x |

Hybrid dispatch summary:

Note: this first focused summary used the cuBLAS timing from the same row as the selected
custom config. Later summaries below use the fastest observed cuBLAS time for each
`(backend, op, M)` case and should be treated as the conservative dispatch evidence.

| Backend | Dispatch weighted ms | cuBLAS weighted ms | Custom-only weighted ms | Dispatch vs cuBLAS | Custom-only vs cuBLAS |
| --- | ---: | ---: | ---: | ---: | ---: |
| Tuna | 188.2440 | 323.6200 | 755.1540 | 1.7192x | 0.4285x |
| SGEMM BF16 | 188.6090 | 439.9490 | 759.2650 | 2.3326x | 0.5794x |

The dispatch policy uses custom kernels only where the measured custom config is faster
than cuBLAS for that `(op, M)` point, otherwise it falls back to cuBLAS. This is not a
final production policy because small-M cuBLAS measurements are noisy here, but it is a
useful guardrail: the current custom kernels are only candidates for specific small-M
projection cases, not broad prefill replacements.

Caveats:

- Small-M cuBLAS timings are noisy on this Thunder instance, even with warmup. The tuner
  is useful for reproducible sweeps and spotting clear losers, but longer runs and a
  production profiling instance are still needed before treating the exact speedups as
  final.
- The current one-warp-per-WMMA-tile custom kernels are not competitive on the dominant
  wide prefill projections. The tuner reinforces the earlier conclusion: deeper pipeline
  work is needed before custom prefill GEMM can replace cuBLAS for most Gemma 4 shapes.

All-shape follow-up:

```bash
python3 src/experiments/gemma4_prefill_tune.py \
  --backend tuna \
  --ops ffn_gate_up,ffn_down,sliding_qkv,sliding_o,global_q,global_k,global_o,final_logits \
  --configs wmma_16x32,wmma_32x64,wmma_64x64 \
  --m 16,64,256 --iters 5 --warmup 2 \
  --out build/experiments/gemma4_prefill_tune/tuna_all_shapes.csv

python3 src/experiments/gemma4_prefill_tune.py \
  --backend sgemm-bf16 \
  --ops ffn_gate_up,ffn_down,sliding_qkv,sliding_o,global_q,global_k,global_o,final_logits \
  --configs bf16_16x32,bf16_32x64,bf16_64x64 \
  --m 16,64,256 --iters 5 --warmup 2 \
  --out build/experiments/gemma4_prefill_tune/sgemm_bf16_all_shapes.csv

python3 src/experiments/gemma4_prefill_tune.py summarize \
  build/experiments/gemma4_prefill_tune/tuna_all_shapes.csv \
  build/experiments/gemma4_prefill_tune/sgemm_bf16_all_shapes.csv \
  --custom-threshold 1.0
```

All-shape config ranking:

| Backend | Op | Best config | Geomean speedup | Worst speedup |
| --- | --- | --- | ---: | ---: |
| Tuna | `ffn_down` | `wmma_64x64` | 2.648x | 1.060x |
| Tuna | `ffn_gate_up` | `wmma_64x64` | 0.293x | 0.134x |
| Tuna | `final_logits` | `wmma_64x64` | 0.289x | 0.132x |
| Tuna | `global_k` | `wmma_16x32` | 4.261x | 0.189x |
| Tuna | `global_o` | `wmma_32x64` | 0.551x | 0.431x |
| Tuna | `global_q` | `wmma_64x64` | 0.643x | 0.162x |
| Tuna | `sliding_o` | `wmma_32x64` | 0.798x | 0.446x |
| Tuna | `sliding_qkv` | `wmma_64x64` | 0.595x | 0.161x |
| SGEMM BF16 | `ffn_down` | `bf16_64x64` | 1.720x | 0.373x |
| SGEMM BF16 | `ffn_gate_up` | `bf16_64x64` | 0.286x | 0.127x |
| SGEMM BF16 | `final_logits` | `bf16_64x64` | 0.292x | 0.137x |
| SGEMM BF16 | `global_k` | `bf16_32x64` | 3.888x | 0.165x |
| SGEMM BF16 | `global_o` | `bf16_32x64` | 0.564x | 0.294x |
| SGEMM BF16 | `global_q` | `bf16_64x64` | 0.578x | 0.161x |
| SGEMM BF16 | `sliding_o` | `bf16_64x64` | 0.874x | 0.438x |
| SGEMM BF16 | `sliding_qkv` | `bf16_64x64` | 0.729x | 0.161x |

All-shape weighted dispatch summary:

| Backend | Dispatch weighted ms | cuBLAS weighted ms | Custom-only weighted ms | Dispatch vs cuBLAS | Custom-only vs cuBLAS |
| --- | ---: | ---: | ---: | ---: | ---: |
| Tuna | 576.6059 | 1169.7969 | 1234.8390 | 2.0288x | 0.9473x |
| SGEMM BF16 | 450.3071 | 683.7701 | 1246.3649 | 1.5185x | 0.5486x |

Interpretation:

- The all-shape sweep identifies `ffn_down`, small-M `global_k`, M=16
  `global_q`/`sliding_qkv`, and M=256 `sliding_o` as the only consistently interesting
  custom-kernel candidates in this first adaptation.
- `ffn_gate_up`, `final_logits`, most QKV/global-Q at larger M, and most output
  projections should stay on cuBLAS unless a deeper pipelined tensor-core kernel is
  written.
- Tuna and SGEMM BF16 are still structurally similar after this adaptation. Tuna has a
  slight edge in the all-shape custom-only result because its measured `ffn_down` run was
  stronger; SGEMM BF16 has slightly better focused `sliding_qkv` in the earlier run.

Candidate deep sweep:

After the all-shape sweep narrowed the search space, I reran only the surviving candidate
ops over a denser M range and the full config sets.

```bash
python3 src/experiments/gemma4_prefill_tune.py \
  --backend tuna --ops ffn_down,global_k,sliding_o \
  --configs wmma_16x16,wmma_16x32,wmma_16x64,wmma_32x64,wmma_64x64,smem_16x64,smem_16x128,smem_32x64,smem_32x128,smem_64x64 \
  --m 16,32,64,128,256,512 --iters 10 --warmup 3 \
  --out build/experiments/gemma4_prefill_tune/tuna_candidate_deep.csv

python3 src/experiments/gemma4_prefill_tune.py \
  --backend sgemm-bf16 --ops ffn_down,global_k,sliding_o \
  --configs bf16_16x32,bf16_16x64,bf16_32x64,bf16_64x64,bf16_64x128,bf16_128x64 \
  --m 16,32,64,128,256,512 --iters 10 --warmup 3 \
  --out build/experiments/gemma4_prefill_tune/sgemm_bf16_candidate_deep.csv

python3 src/experiments/gemma4_prefill_tune.py summarize \
  build/experiments/gemma4_prefill_tune/tuna_candidate_deep.csv \
  build/experiments/gemma4_prefill_tune/sgemm_bf16_candidate_deep.csv \
  --custom-threshold 1.0
```

Candidate deep summary:

| Backend | Op | Best config | Geomean speedup | Worst speedup |
| --- | --- | --- | ---: | ---: |
| Tuna | `ffn_down` | `wmma_64x64` | 0.864x | 0.110x |
| Tuna | `global_k` | `wmma_16x16` | 3.233x | 0.189x |
| Tuna | `sliding_o` | `wmma_64x64` | 0.378x | 0.123x |
| SGEMM BF16 | `ffn_down` | `bf16_32x64` | 0.798x | 0.050x |
| SGEMM BF16 | `global_k` | `bf16_16x64` | 4.046x | 0.188x |
| SGEMM BF16 | `sliding_o` | `bf16_64x64` | 0.375x | 0.122x |

Candidate dispatch summary:

| Backend | Dispatch weighted ms | cuBLAS weighted ms | Custom-only weighted ms | Dispatch vs cuBLAS | Custom-only vs cuBLAS |
| --- | ---: | ---: | ---: | ---: | ---: |
| Tuna | 426.9340 | 708.2160 | 1066.1890 | 1.6588x | 0.6642x |
| SGEMM BF16 | 399.7810 | 625.7630 | 1022.0280 | 1.5653x | 0.6123x |

Deep-sweep dispatch windows:

- `ffn_down`: custom is useful for M=16,32,64; use cuBLAS from M=128 onward.
- `global_k`: custom is useful at M=16,32,64; M=128 and M=256 should use cuBLAS.
  The measured M=512 custom win is suspicious because cuBLAS timing remains noisy for
  small-ish N; rerun before encoding a production rule there.
- `sliding_o`: custom is mostly rejected; M=256 barely wins in both backends, but the
  margin is small enough that I would not dispatch to it without a longer stable run.
- The simple staged shared-memory Tuna variants did not surface as winners in the deep
  sweep, so the current smem path is not worth promoting without a more serious async
  pipeline redesign.

## 2026-05-20 - Decode GEMV block swizzle experiment

Runtime files:

- `src/gemma4_matmul_kernels.cu`
- `src/gemma4_matmul_kernels.cuh`
- `src/experiments/gemma4_decode_bench.cu`

Research notes:

- NVIDIA's thread-group ID swizzling guidance targets 2D compute grids where adjacent
  thread groups have overlapping memory footprints but row-major launch order sends
  concurrently executing groups far apart.
- CUTLASS/CUDA Tile-style GEMM swizzles are also 2D output-tile mappings intended to
  improve reuse of shared A/B tiles through L2.
- The current decode GEMV is a 1D output-column grid: one CTA computes 8 output columns
  and the full K dot product for those columns. Adjacent CTAs do not reuse weight data;
  they only share the small activation vector.

Implementation:

- Added `Gemma4DecodeSwizzle` with an identity policy and an experimental
  `GEMMA4_DECODE_SWIZZLE_INTERLEAVE_16` policy.
- The swizzled path remaps physical CTA IDs through a compile-time interleave over
  groups of 16 column blocks.
- `gemma4_projection_decode` keeps the identity mapping by default.
- `gemma4_projection_decode_swizzled` exposes the native swizzled path for direct
  benchmarking or future call sites.
- The decode benchmark now runs identity and swizzled custom GEMV and checks exact
  output equality.

Commands:

```bash
make decode-bench
GEMMA4_DECODE_BENCH_SEED=0x20260520 ./build/experiments/gemma4_decode_bench ffn_gate_up 50 10 3
GEMMA4_DECODE_BENCH_SEED=0x20260520 ./build/experiments/gemma4_decode_bench global_k 50 10 3
GEMMA4_DECODE_BENCH_SEED=0x20260520 ./build/experiments/gemma4_decode_bench final_logits 20 5 2
```

Results:

| Op | Identity ms | Swizzle16 ms | Swizzle speedup | Max abs diff |
| --- | ---: | ---: | ---: | ---: |
| `ffn_gate_up` | 0.650819 | 0.650897 | 0.9999x | 0 |
| `global_k` | 0.034588 | 0.034268 | 1.0093x | 0 |
| `final_logits` | 3.956267 | 3.953227 | 1.0008x | 0 |

Conclusion:

- Native block swizzling is implemented and measurable, but the interleave-16 mapping is
  neutral on the tested decode GEMV shapes.
- This matches the geometry expectation: the current GEMV is 1D and streams weights with
  almost no cross-CTA weight reuse, so 2D thread-group tiling does not directly transfer.
- Keep identity as the default. Revisit swizzling if the GEMV is changed to a 2D output
  tile, split-K/persistent schedule, or a fused multi-vector/multi-token decode kernel
  with real cross-CTA footprint overlap.

## 2026-05-20 - Scale-free V RMSNorm in existing RMSNorm kernels

Runtime files:

- `src/gemma4_rmsnorm.cu`
- `src/gemma4_rmsnorm.cuh`
- `tests/test_rmsnorm.cu`
- `src/experiments/gemma4_rmsnorm_bench.cu`

Reason:

- Gemma 4 applies RMSNorm to V without a learned scale parameter.
- Rather than adding a separate V-only kernel, the existing RMSNorm implementation now
  treats `weight == nullptr` as scale-free RMSNorm and dispatches to compile-time
  `HasWeight=false` kernel instantiations.
- The explicit wrapper `gemma4_rmsnorm_scale_free_bf16` is available for call sites
  where passing a null weight would be less clear.

Implementation:

- Reused the current RMSNorm reduction, launch-shape, shared-memory, and decode-specialized
  structure.
- Added scale-free template instantiations that compile out weight loads and apply
  `y = x * rsqrt(mean(x*x) + eps)`.
- The fused residual+RMSNorm launcher also accepts `weight == nullptr`, although the
  immediate target is standalone V RMSNorm.
- Extended the RMSNorm benchmark to accept an optional width argument and report
  scale-free custom timings. Since cuDNN frontend RMSNorm requires a scale tensor, the
  default comparison uses cuDNN RMSNorm with an all-ones BF16 scale.

Build and correctness:

```bash
make test-rmsnorm
make rmsnorm-bench
make cuda-kernels
nvcc -std=c++17 -O3 -arch=sm_86 -Xptxas=-v -Isrc \
  -c src/gemma4_rmsnorm.cu -o /tmp/gemma4_rmsnorm_scale_free_check.o
```

Result:

- `rmsnorm tests passed`
- Scale-free test coverage includes width `256`, width `512`, and width `5376`.
- The scale-free fused residual+RMSNorm dispatch is also covered at width `512` and
  width `5376`.
- `ptxas` reported `0 bytes` spills for all weighted and scale-free RMSNorm
  instantiations.

Benchmark commands:

```bash
GEMMA4_RMSNORM_BENCH_SEED=0x20260520 \
  ./build/experiments/gemma4_rmsnorm_bench 30 5 2 1024 256

GEMMA4_RMSNORM_BENCH_SEED=0x20260520 \
  ./build/experiments/gemma4_rmsnorm_bench 30 5 2 1024 512
```

Key CUDA graph-captured timing results:

| Width | Rows | Custom scale-free ms | Custom GiB/s | cuDNN one-scale ms | cuDNN GiB/s | Max abs | rstd max abs |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 16 | 0.001641 | 9.337 | 0.002178 | 10.536 | 0 | 1.19209e-7 |
| 256 | 1024 | 0.002007 | 488.365 | 0.002503 | 586.650 | 0 | 2.38419e-7 |
| 512 | 4 | 0.001622 | 4.712 | 0.002078 | 5.515 | 0 | 1.19209e-7 |
| 512 | 1024 | 0.002638 | 741.865 | 0.003444 | 851.706 | 0 | 2.38419e-7 |

Interpretation:

- The scale-free path numerically matches cuDNN RMSNorm with an all-ones scale.
- At the decode-relevant V shapes (`16 x 256` for sliding, `4 x 512` for global), the
  custom scale-free kernel is faster than the cuDNN one-scale baseline in graph-captured
  timings.
- The stream-loop timings at these small widths are dominated by frontend/API overhead,
  so graph-captured timings are the more useful kernel-speed comparison.

## 2026-05-20 - Gemma 4 RoPE baseline vs cuDNN tensor ops

Benchmark file: `src/experiments/gemma4_rope_bench.cu`

Research notes:

- cuDNN 9.22 still documents `cudnnOpTensor`, but marks it deprecated since cuDNN 9.0.
- The modern cuDNN frontend exposes pointwise `mul` and `add`, but there is no single fused RoPE primitive; the available library baseline is therefore a pointwise decomposition.
- The benchmark's cuDNN path applies RoPE through six `cudnnOpTensor` calls per Q/K branch group: two multiplies plus one add/sub for each output half.
- CUDA-event timings are recorded on the same nonblocking stream as the benchmark work, matching CUDA stream timing guidance.

Environment:

- GPU: NVIDIA RTX A6000, 49140 MiB
- Driver: 580.126.16
- NVCC: CUDA 12.9, build `cuda_12.9.r12.9/compiler.36037853_0`
- cuDNN headers/runtime: 9.22.0

Build:

```bash
make rope-bench
```

Timing command:

```bash
GEMMA4_ROPE_BENCH_SEED=0x20260520 ./build/experiments/gemma4_rope_bench 200 30 5 4096 1 1
```

Key CUDA-event timing results:

| Case | Seq | Custom ms | cuDNN ms | Custom speedup | Custom graph ms | cuDNN graph ms | Graph speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sliding | 1 | 0.013840 | 0.189035 | 13.66x | 0.001602 | 0.014945 | 9.33x |
| sliding | 1024 | 0.075738 | 0.305416 | 4.03x | 0.074725 | 0.287080 | 3.84x |
| sliding | 4096 | 0.483672 | 1.145305 | 2.37x | 0.482928 | 1.126152 | 2.33x |
| global | 1 | 0.020791 | 0.225866 | 10.86x | 0.001406 | 0.014736 | 10.48x |
| global | 1024 | 0.030901 | 0.258333 | 8.36x | 0.030432 | 0.116004 | 3.81x |
| global | 4096 | 0.147369 | 0.437539 | 2.97x | 0.145875 | 0.421499 | 2.89x |

Accuracy check:

- `cudnn_q_max_abs = 0.0078125`
- `cudnn_k_max_abs = 0.0078125`
- This compares custom RoPE using FP32 cosine/sine tables against the cuDNN baseline using BF16 cosine/sine tables, so the observed difference is consistent with table precision rather than a known indexing mismatch.

Conclusion:

- The custom RoPE kernel wins over the cuDNN tensor-op decomposition for every measured sequence length.
- CUDA Graph capture makes short-sequence launch overhead much smaller, but cuDNN remains slower because RoPE is decomposed into multiple pointwise launches instead of one fused kernel.
- Keep the custom RoPE baseline. Future optimization should focus on fusing Q/K RMSNorm, RoPE, and KV-cache write rather than replacing this path with cuDNN pointwise ops.

Follow-up static profile:

- Built PTX/SASS artifacts with `nvcc -std=c++17 -O3 -arch=sm_86 -Isrc`.
- `ptxas -v` reports 26 registers for the low-level layout kernel, 29 registers for the forward-layout kernel, 0 stack bytes, 0 spill loads/stores, and 0 barriers.
- `cuobjdump --dump-resource-usage` reports `SHARED:0` and `LOCAL:0` for both RoPE kernels.
- PTX and SASS contain no shared-memory instructions (`ld.shared`, `st.shared`, `LDS`, `STS`) and no local-memory spill instructions (`ld.local`, `st.local`, `LDL`, `STL`).
- RoPE memory ops are scalar, not vectorized: PTX uses `ld.global.u16`/`st.global.u16` for BF16 Q/K and `ld.global.f32` for cos/sin; SASS uses `LDG.E.U16`, `STG.E.U16`, and scalar 32-bit `LDG.E`.
- `ncu` could not collect bank-conflict counters on this Thunder instance; it aborted in Thunder/CUPTI with `Unimplemented CUDA export table function: Table=cupti_device_query, Index=7`.

## 2026-05-17 - 16384x512x4096 CUTE matmul vs cuBLAS

Experiment file: `src/experiments/16384_512_4096.cu`

Important correction: I initially started wiring this against raw cuDNN backend matmul, but that was the wrong baseline for this experiment. cuDNN matmul is exposed through backend/frontend graph descriptors, which is not the simple GEMM call we want here. For a short library matmul baseline, use cuBLAS/cuBLASLt. This run uses `cublasGemmEx`.

Shape:

- A: `[M, K] = [16384, 512]`, row-major half
- B storage passed to the custom kernel: `[N, K] = [4096, 512]`, row-major half, equivalent to B transposed for `C = A * B`
- C: `[M, N] = [16384, 4096]`, row-major half
- FLOPs per matmul: `2 * M * N * K = 68.719476736 GFLOPs`

Build:

```bash
nvcc -std=c++17 -O3 -arch=sm_86 -DGEMMA4_STANDALONE_BENCH \
  -I/tmp/cutlass/include -I/tmp/cutlass/tools/util/include \
  src/experiments/16384_512_4096.cu -lcublas \
  -o build/experiments/16384_512_4096_bench
```

Hardware/software:

- GPU: NVIDIA RTX A6000, compute capability 8.6, 49140 MiB
- Driver: 580.126.16
- NVCC: CUDA 12.9, build `cuda_12.9.r12.9/compiler.36037853_0`
- cuBLAS version: `120902`

Timing command:

```bash
./build/experiments/16384_512_4096_bench both 100 20
```

CUDA-event timing results:

| Kernel | Avg ms | TFLOP/s |
| --- | ---: | ---: |
| Custom CUTE kernel | 0.756506 | 90.838 |
| cuBLAS `cublasGemmEx` | 0.690792 | 99.4793 |

Comparison:

- `custom_vs_cublas_speedup = 0.913135`
- The custom kernel is about `8.7%` slower than cuBLAS on this shape in this run.
- Correctness check: `max_abs_diff = 0` versus cuBLAS output for the deterministic half input pattern used by the benchmark.

Implementation note:

- The cuBLAS baseline is a single `cublasGemmEx` call using FP16 inputs, FP32 accumulation, and `CUBLAS_GEMM_DEFAULT_TENSOR_OP`.
- The benchmark stores C in row-major form by viewing it as column-major `C^T` for cuBLAS. The call computes `C^T = B^T * A^T` with `CUBLAS_OP_T` on the stored `[N, K]` B buffer and `CUBLAS_OP_N` on the row-major A buffer interpreted as column-major `[K, M]`.

Measurement bug caught and fixed:

- An earlier timing attempt gave an impossible `2429.8 TFLOP/s` for the custom kernel.
- Cause: the custom kernel launch used the default stream while the CUDA events were recorded on a separate nonblocking stream, so the benchmark measured launch overhead rather than kernel execution.
- Fix: `launch_hgemm_mma_stages_block_swizzle_tn_cute` now accepts a `cudaStream_t`, and the standalone benchmark launches the custom kernel on the same stream as the timing events.

Compile-time resource data from `ptxas -v` for the custom kernel:

- Registers: `152`
- Static spills: `0 bytes`
- Barriers: `1`
- Constant memory: `388 bytes cmem[0]`
- Dynamic shared memory requested by the launch geometry: about `48 KiB`

Profiler status:

- `ncu --section SpeedOfLight --section MemoryWorkloadAnalysis` failed on this Thunder Compute instance with an internal Thunder runtime assertion before producing kernel metrics.
- `nsys profile --stats=true` completed, but the generated report did not contain CUDA trace, CUDA kernel, or GPU memory data on this instance. It only produced OS runtime stats, so I did not use it for GPU conclusions.
- Because of that, the only trustworthy performance numbers in this entry are CUDA-event timings plus the `ptxas` compile-time resource data.

Tooling/source notes:

- `$cuda-programming-guide` was used to check CUDA event timing and error-checking guidance.
- `$exa-search` could not run because `EXA_API_KEY` is not set in this environment.
- Official NVIDIA cuDNN frontend/backend docs confirm matmul is graph/backend-descriptor based, which is why cuDNN was dropped as the "simple" baseline for this experiment.

## 2026-05-17 - Gemma 4 gate+up packed matmul tuning

Experiment file: `src/matmul.cu` at the time of the experiment. That runtime
copy has since been removed; the active decode implementation is now in
`src/gemma4_matmul_kernels.cu`.

Target:

- Packed FFN gate+up projection for Gemma 4 dense 31B.
- A: `[M, 5376]`, row-major half.
- B storage: `[43008, 5376]`, row-major half, equivalent to transposed `[5376, 43008]`.
- C: `[M, 43008]`, row-major half.
- Current specialized default: `M=512`.

Build:

```bash
nvcc -std=c++17 -O3 -arch=sm_86 -DGEMMA4_STANDALONE_BENCH \
  -I/tmp/cutlass/include -I/tmp/cutlass/tools/util/include \
  src/matmul.cu -lcublas -o build/matmul_bench
```

Best source configuration so far:

- CTA tile: `BM=128, BN=128, BK=32`
- Pipeline stages: `4`
- Tensor core op: `SM80_16x8x16_F32F16F16F32_TN`
- Baseline: `cublasGemmEx`, FP16 inputs, FP32 accumulation, tensor-op path.

Final default timing command:

```bash
./build/matmul_bench both 100 20
```

CUDA-event timing results at `M=512`:

| Kernel | Avg ms | TFLOP/s |
| --- | ---: | ---: |
| Custom CUTE kernel | 1.87 | 126.61 |
| cuBLAS `cublasGemmEx` | 2.03743 | 116.205 |

Comparison:

- `custom_vs_cublas_speedup = 1.08954`
- Custom is about `9.0%` faster than cuBLAS for this specialized `M=512` packed gate+up shape.
- Correctness check: `max_abs_diff = 0`.

Tuning sweep notes:

- Initial copied tile retargeted to `[M,5376] x [5376,43008]` with `BM=128, BN=64, BK=64, Stages=2` was much slower: at `M=1024`, custom `4.9356 ms` vs cuBLAS `3.71269 ms`.
- `BM=128, BN=128, BK=64, Stages=2` improved to near parity: custom `3.94334 ms` vs cuBLAS `3.86984 ms` at `M=1024`.
- `BM=128, BN=128, BK=64, Stages=3` beat cuBLAS in a short run, but not enough.
- `BM=128, BN=128, BK=32, Stages=4` is the best tested source shape so far.
- `BM=64, BN=128, BK=64, Stages=3` was bad.
- `BM=128, BN=256, BK=64, Stages=2` was extremely bad.
- `BM=256, BN=128, BK=32, Stages=4` was extremely bad.
- L2 persisting-cache hints for the B/weight matrix hurt performance and were removed.
- Register capping with `--maxrregcount=192` caused spills and badly hurt performance.

M sensitivity:

- `M=512`: stable win, `1.08954x` over cuBLAS on the final 100-iteration run.
- `M=1024`: borderline; one 50-iteration run reached `1.07005x`, but a 100-iteration run was only `1.06442x`.
- `M=1536` and `M=2048`: custom lost to cuBLAS.

Resource data from `ptxas -v` for the current custom kernel:

- Registers: `230`
- Static spills: `0 bytes`
- Barriers: `1`
- Constant memory: `388 bytes cmem[0]`

Profiler status:

- `ncu --section SpeedOfLight --section MemoryWorkloadAnalysis` still fails on this Thunder Compute instance with an internal Thunder runtime assertion before producing kernel metrics.
- Because `ncu` is blocked here, the trustworthy evidence for this entry is CUDA-event timing, exact-output comparison against cuBLAS, and `ptxas` resource data.

Tooling/source notes:

- `$cuda-programming-guide` was used repeatedly for SM 8.6 resource constraints, occupancy/register/shared-memory tradeoffs, async copy guidance, shared-memory bank conflict concerns, and L2 persisting cache semantics.
- `$exa-search` could not run because `EXA_API_KEY` is not set in this environment.

## 2026-05-17 - Matmul runtime skeleton simplification

Runtime file: `src/gemma4_matmul_kernels.cu`

Change:

- Replaced the copied CUTE benchmark implementation with a small runtime-facing skeleton in `src/gemma4_matmul_kernels.cu`.
- Prefill now routes through a host-side library GEMM wrapper for the fixed packed FFN gate+up shape.
- Decode now has a custom M=1 kernel that calls a single `__device__` dot-product primitive.
- The decode kernel computes four output columns per CTA and uses `half2` loads to reduce launch-grid size and reuse the M=1 input vector across four dot products.
- Kept the tuned CUTE result above as experiment history instead of carrying that complexity in the runtime path.

CUDA guide note:

- CUDA execution-space rules mean cuDNN/cuBLAS frontend calls belong in host code, not `__device__` functions. The device side is therefore limited to the custom decode matvec primitive.

## 2026-05-18 - M=1 decode vs cuDNN baseline

Runtime file: `src/gemma4_matmul_kernels.cu`

Benchmark file: `src/experiments/gemma4_decode_bench.cu`

Target:

- Decode-only packed FFN gate+up projection.
- Shape: `M=1, K=5376, N=43008`.
- Custom weight layout: `[43008, 5376]` row-major half, so each output column/dot product is contiguous.
- cuDNN baseline: equivalent 1x1 convolution with input `[1, 5376, 1, 1]`, filter `[43008, 5376, 1, 1]`, output `[1, 43008, 1, 1]`.

Why cuDNN convolution baseline:

- cuDNN backend matmul descriptors for the same operation finalized, but heuristics returned zero usable engine configs for both transposed-stride and contiguous B descriptor attempts.
- The 1x1 convolution formulation is the same decode math and uses the same physical `[N, K]` weight layout as the custom kernel.

Build:

```bash
make decode-bench
```

Final timing command:

```bash
./build/experiments/gemma4_decode_bench both 200 30
```

Final CUDA-event timing results:

| Kernel | Avg ms | Weight GB/s |
| --- | ---: | ---: |
| Custom decode | 0.652241 | 708.974 |
| cuDNN 1x1 conv | 3.191969 | 144.870 |

Comparison:

- `custom_vs_cudnn_speedup = 4.893846`
- Custom is about `387%` faster than cuDNN on this M=1 decode benchmark.
- Correctness check: `max_abs_diff = 0`.
- cuDNN selected convolution algo `1` with `172160` bytes of workspace.
- Device: NVIDIA RTX A6000, compute capability 8.6, driver `580.126.16`.

Tuning/resource notes:

- Kept 4 output columns per CTA using `half2` loads. This reduces grid size and reuses each loaded input element across four output dots.
- Thread-count sweep:
  - `GEMMA4_DECODE_THREADS=128`: `0.652592 ms`
  - `GEMMA4_DECODE_THREADS=256`: `0.652398 ms`
  - `GEMMA4_DECODE_THREADS=512`: `0.680386 ms`
- Kept 256 threads per CTA.
- Added `__launch_bounds__(256, 2)` and moved thread-index ownership out of the reusable `__device__` helper to match the repo CUDA kernel-structure rule.
- Final `ptxas -v` resource data:
  - Registers: `25`
  - Static spills: `0 bytes`
  - Shared memory: `128 bytes`
  - Barriers: `1`
  - Constant memory: `376 bytes cmem[0]`
- Theoretical occupancy estimate on SM 8.6: 256 threads/CTA gives 8 warps/CTA; the 1536-thread/48-warp SM limit allows 6 CTAs/SM, so the kernel can reach 48 active warps/SM before register or shared-memory limits bind.

Profiler status:

- `ncu --set full --target-processes all --kernel-name regex:gemma4_ffn_gate_up_decode_kernel --launch-count 1 ./build/experiments/gemma4_decode_bench custom 1 0` fails in this Thunder Compute environment with an internal unsupported-library assertion before producing metrics.
- `nsys profile --force-overwrite=true --trace=cuda,nvtx,osrt --stats=true -o build/experiments/gemma4_decode_bench_nsys ./build/experiments/gemma4_decode_bench both 10 2` completes and writes `.nsys-rep`/`.sqlite`, but the generated sqlite contains no CUDA trace, kernel, or GPU memory data.
- `nvprof ./build/experiments/gemma4_decode_bench custom 5 1` is unavailable for this GPU class; it reports that `nvprof` is not supported on compute capability 8.0 and higher.
- Because profiler CUDA data is blocked here, the reliable evidence for this entry is CUDA-event timing, exact-output comparison against cuDNN, `ptxas` resource data, and the explicit profiler failure output.

Tooling/source notes:

- `$cuda-programming-guide` was queried for coalesced memory access, occupancy/resource constraints, launch bounds, shared-memory/bank-conflict considerations, and profiler/occupancy interpretation. Relevant guide pages returned included 60-61, 67, 70-71, 119, 130, 445, and 526.
- `$exa-search` could not run because `EXA_API_KEY` is not set in this environment.

## 2026-05-18 - M=1 decode vs cuBLAS GEMM/GEMV baselines

Benchmark file: `src/experiments/gemma4_decode_bench.cu`

Reason:

- The cuDNN 1x1 convolution baseline is mathematically equivalent, but it is not the cleanest library matmul baseline.
- The lean baseline for this decode matmul is `cublasGemmEx`, using the existing `[N, K]` weight layout as a column-major `[K, N]` matrix and computing `Y^T[N, 1] = W^T[N, K] * X^T[K, 1]`.
- Because M=1 is also a matrix-vector operation, the benchmark now also includes `cublasHSHgemvStridedBatched` with `batchCount=1`.
- cuBLASLt was briefly wired in, but removed from the benchmark because the simple cuBLAS GEMM/GEMV calls are the correct small baselines for this question.

Command:

```bash
./build/experiments/gemma4_decode_bench both 50 10 3
```

Results:

| Kernel | Best ms | Avg ms | Best weight GB/s |
| --- | ---: | ---: | ---: |
| Custom decode | 0.652367 | 0.652407 | 708.838 |
| cuBLAS `cublasGemmEx` | 3.333772 | 3.486190 | 138.708 |
| cuBLAS `cublasHSHgemvStridedBatched` | 3.413318 | 3.593821 | 135.476 |
| cuDNN 1x1 conv | 3.138264 | 3.222237 | 147.350 |

Comparison:

- `custom_vs_cublas_gemmex_speedup = 5.110273`
- `custom_vs_cublas_hshgemv_strided_batched_speedup = 5.232207`
- `custom_vs_cudnn_speedup = 4.810584`
- Correctness check: `cublas_gemmex_max_abs_diff = 0`.
- Correctness check: `cublas_hshgemv_strided_batched_max_abs_diff = 0`.
- The cuBLAS GEMM/GEMV baselines are more credible library comparisons than cuDNN convolution for the M=1 decode matmul, and they are still far behind the custom contiguous-weight matvec kernel.

Tooling/source notes:

- `$cuda-programming-guide` was queried again for CUDA-X/cuBLAS/Tensor Core guidance and GEMV/memory-access considerations. The local guide points to CUDA-X libraries such as cuBLAS as the recommended library path for Tensor Core operations on supported hardware, and reiterates coalesced memory access as a first-order concern.

## 2026-05-18 - Decode GEMV coverage for main dense projections

Runtime file: `src/gemma4_matmul_kernels.cu`

Benchmark file: `src/experiments/gemma4_decode_bench.cu`

Change:

- Generalized the one-off packed FFN gate+up decode kernel into a templated fixed-shape decode GEMV kernel.
- Added decode entry points for:
  - `ffn_gate_up`: `x[5376] -> gate_up[43008]`
  - `ffn_down`: `ffn_hidden[21504] -> hidden[5376]`
  - `sliding_qkv`: `x[5376] -> qkv[16384]`
  - `sliding_o`: `attn_out[8192] -> hidden[5376]`
  - `global_q`: `x[5376] -> q[16384]`
  - `global_k`: `x[5376] -> k[2048]`
  - `global_o`: `attn_out[16384] -> hidden[5376]`
  - `final_logits`: `hidden[5376] -> logits[262144]`
- Kept the same physical weight layout as the original gate/up kernel: `[N, K]` row-major half, so each output dot product reads a contiguous weight row.
- Simplified the benchmark to compare against direct cuBLAS GEMV (`cublasHSHgemvStridedBatched`) and M=1 `cublasGemmEx`. Removed the cuDNN 1x1 convolution baseline from this benchmark because it is no longer the relevant comparison.

Build:

```bash
make decode-bench
make cuda-kernels
```

Smoke timing command:

```bash
./build/experiments/gemma4_decode_bench all 2 1 1
```

Smoke results from the first low-iteration run:

| Op | K | N | Custom best ms | cuBLAS GEMV best ms | cuBLAS GEMM M=1 best ms | Max diff vs GEMV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ffn_gate_up` | 5376 | 43008 | 0.660864 | 3.579680 | 3.446096 | 0.03125 |
| `ffn_down` | 21504 | 5376 | 0.341408 | 0.340816 | 0.340544 | 0 |
| `sliding_qkv` | 5376 | 16384 | 0.264544 | 2.905792 | 2.898672 | 0 |
| `sliding_o` | 8192 | 5376 | 0.133952 | 2.978928 | 2.423568 | 0.03125 |
| `global_q` | 5376 | 16384 | 0.266848 | 2.411344 | 2.354640 | 0 |
| `global_k` | 5376 | 2048 | 0.042096 | 2.560576 | 3.245296 | 0 |
| `global_o` | 16384 | 5376 | 0.265968 | 0.263088 | 0.263392 | 0 |
| `final_logits` | 5376 | 262144 | 3.979248 | 3.997824 | 3.980464 | 0 |

Notes:

- This is a low-iteration smoke pass, not a stable tuning run.
- Some outputs differ from cuBLAS by `0.03125`, which is one half-scale step for this deterministic input pattern and is expected from different FP32 reduction orders before FP16 output rounding.
- The current generic kernel is already very strong when cuBLAS dispatch overhead/layout choices dominate smaller decode shapes, but it is only at parity for `ffn_down`, `global_o`, and `final_logits`. Those need actual tuning before claiming a win.

Sequential stability check:

```bash
./build/experiments/gemma4_decode_bench all 50 10 3
```

| Op | Custom best ms | cuBLAS GEMV best ms | cuBLAS GEMM M=1 best ms | Best custom speedup vs GEMV |
| --- | ---: | ---: | ---: | ---: |
| `ffn_gate_up` | 0.652664 | 3.096099 | 3.124034 | 4.743790 |
| `ffn_down` | 0.327303 | 0.330160 | 0.330163 | 1.008729 |
| `sliding_qkv` | 0.250743 | 2.808276 | 2.542720 | 11.199816 |
| `sliding_o` | 0.127187 | 3.486035 | 3.434500 | 27.408831 |
| `global_q` | 0.251055 | 2.511043 | 2.200656 | 10.001947 |
| `global_k` | 0.036893 | 2.202849 | 2.701255 | 59.709446 |
| `global_o` | 0.250511 | 0.252756 | 0.252499 | 1.008960 |
| `final_logits` | 3.961647 | 3.965804 | 3.965508 | 1.001049 |

Interpretation:

- The large wins are credible for this exact benchmark setup because the run is sequential, warm, and uses the same physical `[N, K]` half weight layout for custom and cuBLAS.
- These are still only baseline comparisons against simple cuBLAS GEMV/GEMM calls, not proof that the kernels beat every possible library/layout path.
- `ffn_down`, `global_o`, and `final_logits` are effectively parity with cuBLAS. Treat them as coverage/correctness baselines, not optimized wins.

## 2026-05-18 - Decode parity-kernel tuning: theoretical max and plan

Runtime file: `src/gemma4_matmul_kernels.cu`

Benchmark file: `src/experiments/gemma4_decode_bench.cu`

Goal:

- Tune the three decode GEMV kernels that are currently only at cuBLAS parity:
  - `ffn_down`: `K=21504, N=5376`
  - `global_o`: `K=16384, N=5376`
  - `final_logits`: `K=5376, N=262144`
- Keep changes small: thread count, columns per CTA, launch bounds, reduction shape, and similarly local kernel-architecture changes.
- Stop a kernel only after a clear flat line: seven tried variants without a better result.

Theoretical memory roofline:

- Device: NVIDIA RTX A6000, compute capability 8.6.
- CUDA-reported properties from PyTorch:
  - `memory_clock_rate = 8001000 kHz`
  - `memory_bus_width = 384 bits`
  - `multi_processor_count = 84`
  - `max_threads_per_multi_processor = 1536`
- Peak theoretical DRAM bandwidth: `8001000 kHz * 1000 * 2 * (384 / 8) = 768.096 GB/s`.
- These M=1 decode kernels are weight-streaming GEMVs. The absolute lower bound below counts weight bytes only, `N * K * sizeof(half)`. Input and output traffic are tiny by comparison for these shapes, and repeated input reads make this an optimistic lower bound.

| Op | K | N | Weight GB | Theoretical min ms @ 768.096 GB/s | Current best ms | Current weight GB/s | Peak % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ffn_down` | 21504 | 5376 | 0.231211 | 0.301018 | 0.327303 | 706.413 | 91.97% |
| `global_o` | 16384 | 5376 | 0.176161 | 0.229347 | 0.250511 | 703.206 | 91.55% |
| `final_logits` | 5376 | 262144 | 2.818572 | 3.669557 | 3.961647 | 711.465 | 92.63% |

Interpretation before tuning:

- The three parity kernels are already around `91.5-92.6%` of the simple theoretical DRAM bandwidth roofline by the benchmark's weight-GB/s metric.
- A "much faster than baseline" result may be mathematically limited unless the baseline comparison changes or the kernel reduces real traffic beyond the current one-pass contiguous weight stream.
- Tuning should still probe whether the last `7-8%` gap is launch/reduction overhead, instruction mix, occupancy, or memory-transaction inefficiency.

Tuning implementation:

- Added compile-time knobs for:
  - `GEMMA4_DECODE_COLS_PER_BLOCK`
  - `GEMMA4_DECODE_MIN_BLOCKS_PER_SM`
- Kept the original hand-written fixed-four-column path for default `ColsPerBlock=4`.
- Added a templated fixed-four launcher so individual decode projections can use a different thread count or launch-bound min-block setting without moving every projection off the known-good default.
- Final selected specializations:
  - `ffn_down`: `Threads=256`, `MinBlocksPerSm=4`
  - `global_o`: `Threads=512`, `MinBlocksPerSm=1`
  - `final_logits`: `Threads=512`, `MinBlocksPerSm=2`

Fresh baseline before tuning:

```bash
make decode-bench
./build/experiments/gemma4_decode_bench ffn_down 80 20 5
./build/experiments/gemma4_decode_bench global_o 80 20 5
./build/experiments/gemma4_decode_bench final_logits 40 10 3
```

| Op | Custom best ms | cuBLAS GEMV best ms | Custom weight GB/s | Speedup vs GEMV |
| --- | ---: | ---: | ---: | ---: |
| `ffn_down` | 0.327326 | 0.330148 | 706.364 | 1.008623 |
| `global_o` | 0.250404 | 0.252554 | 703.507 | 1.008589 |
| `final_logits` | 3.961670 | 3.965620 | 711.461 | 1.000997 |

Sweep notes:

- `ffn_down`:
  - Tried `ColsPerBlock={1,2,4,8}` crossed with `Threads={128,256,512}`.
  - Tried `MinBlocksPerSm={1,2,3,4,6}` at the default fixed-four shape.
  - The column/thread sweep did not beat the original default. Best non-default grouping was `ColsPerBlock=2, Threads=512` at `0.327899 ms`, still slower than the fresh baseline.
  - `MinBlocksPerSm=4` produced a repeatable tiny improvement: `0.327280 ms` in the verification run.
  - This is effectively flat after more than seven variants; the selected specialization is only a small compiler launch-bound gain.
- `global_o`:
  - Tried `ColsPerBlock={1,2,4,8}` crossed with `Threads={128,256,512}`.
  - Tried `Threads=512` with `MinBlocksPerSm={1,2,3}`.
  - Wider/narrower grouping was mostly worse. The repeatable winner was the fixed-four shape with `Threads=512, MinBlocksPerSm=1`.
  - Verification run reached `0.250226 ms`.
  - This also flat-lined after more than seven variants; the selected specialization is a small occupancy/compiler scheduling adjustment.
- `final_logits`:
  - Tried `ColsPerBlock={1,2,4,8,16}` crossed with `Threads={128,256,512}`.
  - Tried `Threads=512` with `MinBlocksPerSm={1,2,3}`.
  - `Threads=512` was the only consistently useful knob. The best stable shape was fixed-four columns with `MinBlocksPerSm=2`.
  - Verification runs were around `3.9566-3.9567 ms`.
  - `ColsPerBlock=8` and `16` lost bandwidth, likely because larger per-CTA work reduced useful scheduling parallelism/register behavior more than it helped input reuse.

Final verification:

```bash
rm -f build/experiments/gemma4_decode_bench build/gemma4_matmul_kernels.o
make decode-bench cuda-kernels
./build/experiments/gemma4_decode_bench ffn_down 100 25 5
./build/experiments/gemma4_decode_bench global_o 100 25 5
./build/experiments/gemma4_decode_bench final_logits 50 12 5
```

| Op | Final custom best ms | cuBLAS GEMV best ms | Custom weight GB/s | Peak % | Speedup vs GEMV | Gain vs fresh custom |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ffn_down` | 0.327243 | 0.330212 | 706.543 | 91.99% | 1.009074 | 0.025% |
| `global_o` | 0.250207 | 0.252314 | 704.061 | 91.66% | 1.008423 | 0.079% |
| `final_logits` | 3.956729 | 3.965828 | 712.349 | 92.74% | 1.002300 | 0.125% |

Correctness:

- `cublas_gemv_max_abs_diff = 0` for all three final target runs.
- `cublas_gemm_m1_max_abs_diff = 0` for all three final target runs.

Resource data:

```bash
rm -f build/gemma4_matmul_kernels.o
make cuda-kernels NVCCFLAGS="-std=c++17 -O3 -arch=sm_86 -Xptxas=-v"
```

| Kernel specialization | Registers | Static spills | Shared memory | Barriers |
| --- | ---: | ---: | ---: | ---: |
| `ffn_down` fixed4 `K=21504,N=5376,Threads=256,MinBlocks=4` | 25 | 0 bytes | 128 bytes | 1 |
| `global_o` fixed4 `K=16384,N=5376,Threads=512,MinBlocks=1` | 25 | 0 bytes | 256 bytes | 1 |
| `final_logits` fixed4 `K=5376,N=262144,Threads=512,MinBlocks=2` | 25 | 0 bytes | 256 bytes | 1 |

Conclusion:

- The target kernels are already close to a one-pass DRAM streaming roofline. The tuning sweeps produced only small gains, not a large breakout.
- The flat-line condition is satisfied for all three target kernels: each had at least seven variants tried without material improvement beyond tiny launch-bound/thread-count effects.
- Further large wins probably require a different baseline comparison, a different data type/layout, fusing the producer/consumer around these GEMVs to remove memory traffic, or changing the math approach rather than local GEMV launch tuning.

Compile resource check:

```bash
nvcc -std=c++17 -O3 -arch=sm_86 -Xptxas -v \
  -c src/gemma4_matmul_kernels.cu \
  -o /tmp/gemma4_matmul_kernels_review.o
```

`ptxas` reported the same resource usage for every instantiated decode GEMV kernel:

- Registers: `24`
- Static spills: `0 bytes`
- Shared memory: `128 bytes`
- Barriers: `1`
- Constant memory: `376 bytes cmem[0]`

Covered instantiations:

| K | N | Cols/block |
| ---: | ---: | ---: |
| 5376 | 262144 | 4 |
| 16384 | 5376 | 4 |
| 5376 | 2048 | 4 |
| 8192 | 5376 | 4 |
| 5376 | 16384 | 4 |
| 21504 | 5376 | 4 |
| 5376 | 43008 | 4 |

## 2026-05-18 - Token embedding gather CUDA-event timing

Runtime file: `src/gemma4_embedding_gather.cu`

Benchmark file: `src/experiments/gemma4_embedding_gather_bench.cu`

Target:

- Gemma 4 31B dense token embedding gather.
- Shape: token id -> hidden row `[5376]`.
- Vocabulary: `262144`.
- Benchmark dimensions are read from the compile-time constants in `src/gemma4.h`.
- Embedding table allocation: `2818572288` bytes.
- Kernel: one warp per token, 16-byte `int4` vectorized BF16 row copy.
- Effective bandwidth below counts embedding reads plus output writes:
  `2 * num_tokens * hidden_size * sizeof(bfloat16)`.

Build:

```bash
make embedding-gather-bench
```

Correctness:

```bash
make test-embedding-gather
```

Result: `embedding gather tests passed`.

Timing command:

```bash
./build/experiments/gemma4_embedding_gather_bench 200 30 5 4096
```

CUDA-event timing results:

| Tokens | Best ms | Avg ms | Best effective GiB/s | Effective MiB |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.014669 | 0.015762 | 1.365 | 0.021 |
| 4 | 0.014482 | 0.015113 | 5.532 | 0.082 |
| 16 | 0.012178 | 0.014270 | 26.312 | 0.328 |
| 64 | 0.012052 | 0.016656 | 106.348 | 1.312 |
| 256 | 0.011847 | 0.013527 | 432.751 | 5.250 |
| 1024 | 0.036626 | 0.036687 | 559.931 | 21.000 |
| 4096 | 0.135538 | 0.135624 | 605.226 | 84.000 |

Larger sweep command:

```bash
./build/experiments/gemma4_embedding_gather_bench 100 20 3 8192
```

Extended sweep results:

| Tokens | Best ms | Avg ms | Best effective GiB/s | Effective MiB |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.013832 | 0.014062 | 1.448 | 0.021 |
| 4 | 0.013962 | 0.014258 | 5.738 | 0.082 |
| 16 | 0.014069 | 0.014502 | 22.775 | 0.328 |
| 64 | 0.013707 | 0.013958 | 93.513 | 1.312 |
| 256 | 0.013732 | 0.014498 | 373.354 | 5.250 |
| 1024 | 0.036742 | 0.036783 | 558.151 | 21.000 |
| 4096 | 0.135601 | 0.135702 | 604.945 | 84.000 |
| 8192 | 0.265173 | 0.265315 | 618.700 | 168.000 |

Theoretical memory roofline:

- A6000 max memory clock from `nvidia-smi -q -d CLOCK`: `8001 MHz`.
- A6000 memory bus width: `384 bits`.
- GDDR6 is DDR, so the peak byte rate is:
  `8001000 kHz * 1000 * 2 * (384 / 8) = 768096000000 bytes/s`.
- Decimal peak: `768.096 GB/s`.
- Binary peak: `715.345 GiB/s`.
- The benchmark reports GiB/s, so compare measured bandwidth against `715.345 GiB/s`.

| Tokens | Effective MiB | Theoretical min ms | Best ms | Best GiB/s | Peak % |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.021 | 0.000029 | 0.013832 | 1.448 | 0.20% |
| 4 | 0.082 | 0.000112 | 0.013962 | 5.738 | 0.80% |
| 16 | 0.328 | 0.000448 | 0.014069 | 22.775 | 3.18% |
| 64 | 1.312 | 0.001791 | 0.013707 | 93.513 | 13.07% |
| 256 | 5.250 | 0.007167 | 0.013732 | 373.354 | 52.19% |
| 1024 | 21.000 | 0.028668 | 0.036742 | 558.151 | 78.03% |
| 4096 | 84.000 | 0.114674 | 0.135601 | 604.945 | 84.57% |
| 8192 | 168.000 | 0.229347 | 0.265173 | 618.700 | 86.49% |

Compile resource check:

```bash
nvcc -std=c++17 -O3 -arch=sm_86 -Isrc -Xptxas=-v \
  -c src/gemma4_embedding_gather.cu \
  -o /tmp/gemma4_embedding_gather_ptxas.o
```

`ptxas` resource data:

- Registers: `32`
- Static spills: `0 bytes`
- Barriers: `0`
- Constant memory: `388 bytes cmem[0]`

Interpretation:

- Small token counts are launch-latency dominated in this environment, so their percent of peak is not a useful kernel-quality signal.
- Bandwidth becomes meaningful by `256` tokens and plateaus around `605-619 GiB/s` effective read+write bandwidth by `4096-8192` tokens.
- The best measured large-token case is `618.700 GiB/s`, which is `86.49%` of the A6000 theoretical DRAM roofline by this benchmark's read+write byte model.
- At `4096` tokens, `604.945 GiB/s` is `84.57%` of theoretical peak.
- This is good for a simple one-warp-per-token baseline. The remaining raw-memory roofline headroom is about `15%`, and some of that is not realistically recoverable because the measured kernel still pays launch overhead, token-id reads, address computation, control flow, and cache/transaction inefficiency.
- Optimization focus: improve small-token decode/prompt latency through batching or fusion; for large prefill gathers, this baseline is already close to the memory-bound roofline.
- Nsight Compute and Nsight Systems CUDA metrics remain blocked in this Thunder-mediated container; these results are CUDA-event timings only.

## 2026-05-18 - Decode GEMV BF16 randomized validation cleanup

Runtime file: `src/gemma4_matmul_kernels.cu`

Benchmark file: `src/experiments/gemma4_decode_bench.cu`

Reason:

- The decode benchmark was still using FP16 custom/cuBLAS paths and a deterministic tiny integer pattern.
- That was acceptable for early smoke tests, but it is not representative of the Gemma 4 BF16 inference path.
- The benchmark now validates BF16 custom kernels against BF16 cuBLAS references with randomized BF16 inputs and weights.

Implementation:

- Changed the decode projection API and kernels from `half` to `__nv_bfloat16`.
- Changed vectorized loads from `half2` to `__nv_bfloat162`.
- Custom kernels still accumulate in FP32 and round output to BF16.
- Prefill and M=1 cuBLAS GEMM baselines now use `CUDA_R_16BF` inputs/outputs with `CUBLAS_COMPUTE_32F`.
- GEMV baseline now uses `cublasTSTgemvStridedBatched`, which takes BF16 input/output and FP32 scalar parameters.
- Replaced the deterministic small fill pattern with a runtime-seeded GPU hash fill:
  - default seed comes from `std::random_device` and high-resolution clock
  - set `GEMMA4_DECODE_BENCH_SEED=<seed>` to reproduce a run
  - input scale: `1.0`
  - weight scale: `0.5`
- Correctness reporting now prints max absolute, mean absolute, and relative difference against each BF16 cuBLAS reference instead of expecting exact output.

Build:

```bash
rm -f build/experiments/gemma4_decode_bench build/gemma4_matmul_kernels.o
make decode-bench cuda-kernels
```

Representative BF16 randomized checks:

```bash
./build/experiments/gemma4_decode_bench ffn_down 10 3 2
./build/experiments/gemma4_decode_bench final_logits 5 2 1
./build/experiments/gemma4_decode_bench all 2 1 1
```

`ffn_down` result:

| Op | Custom best ms | BF16 cuBLAS GEMV best ms | BF16 cuBLAS GEMM M=1 best ms | Max abs diff vs GEMV | Mean abs diff vs GEMV |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ffn_down` | 0.328960 | 0.331690 | 0.331952 | 0 | 0 |

`final_logits` result:

| Op | Custom best ms | BF16 cuBLAS GEMV best ms | BF16 cuBLAS GEMM M=1 best ms | Max abs diff vs GEMV | Mean abs diff vs GEMV |
| --- | ---: | ---: | ---: | ---: | ---: |
| `final_logits` | 3.963251 | 3.970521 | 3.971814 | 0.03125 | 6.06194e-07 |

Full smoke observations:

- `./build/experiments/gemma4_decode_bench all 2 1 1` completed for all decode projection shapes.
- BF16 cuBLAS GEMV/GEMM agree with each other for the reported differences.
- Some shapes now show BF16-scale nonzero differences, which is expected from different FP32 reduction orders before BF16 output rounding.
- The old exact `max_abs_diff = 0` expectation is no longer the right acceptance signal for randomized BF16 validation.

CUDA guide note:

- `$cuda-programming-guide` was queried for BF16 support. The local guide states that `__nv_bfloat16` is available through `<cuda_bf16.h>` and requires compute capability 8.0 or higher, and that BF16 WMMA/Tensor Core paths support BF16 inputs with FP32 accumulation.

## 2026-05-18 - Decode GEMV Packed128 streaming-load pass

Runtime file: `src/gemma4_matmul_kernels.cu`

Reason:

- The decode GEMV kernels were still loading one BF16 pair per instruction through `__nv_bfloat162`.
- The input vector is small and reused by every CTA, while projection weights are large one-pass streams.
- This pass keeps normal cached loads for the reused input vector and marks the weight stream with the existing `Packed128` + `load128cs` helper.

Implementation:

- Changed the fixed-four decode dot helper from half2-style 32-bit loads to `Packed128<__nv_bfloat16>` loads.
- Each load now covers 8 BF16 values per thread.
- Weight row loads use `load128cs`, which SASS emits as `LDG.E.EF.128` on this build.
- Input-vector loads use `load128`, which SASS emits as `LDG.E.128.CONSTANT` on this build.
- Packed the four BF16 outputs into one 64-bit store in the fixed-four path.
- Templated the dot helper on `Threads` so the packed loop can use the compile-time stride.
- Added `__restrict__` on kernel/device pointer parameters.

Build/resource check:

```bash
nvcc -std=c++17 -O3 -arch=sm_86 -Isrc -Xptxas=-v \
  -c src/gemma4_matmul_kernels.cu \
  -o /tmp/gemma4_matmul_kernels_packed128.o
```

`ptxas` result for all instantiated decode GEMV kernels:

- Registers: `40`
- Static spills: `0 bytes`
- Shared memory: `128` or `256` bytes depending on thread count
- Barriers: `1`

SASS spot-check:

```bash
cuobjdump --dump-sass /tmp/gemma4_matmul_kernels_packed128.o | rg "LDG|STG"
```

Confirmed:

- `LDG.E.128.CONSTANT` for input-vector packed loads
- `LDG.E.EF.128` for weight packed loads
- `STG.E.64` for fixed-four output stores

Benchmark command:

```bash
make decode-bench
./build/experiments/gemma4_decode_bench all 20 5 2
```

Representative results:

| Op | Custom best ms | Custom weight GB/s | BF16 cuBLAS GEMV best ms | Speedup vs GEMV | Max abs diff vs GEMV |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ffn_gate_up` | 0.651875 | 709.372 | 3.528619 | 5.413029 | 0.25 |
| `ffn_down` | 0.328534 | 703.765 | 0.331218 | 1.008167 | 0.0625 |
| `sliding_qkv` | 0.250694 | 702.691 | 2.499530 | 9.970425 | 0.25 |
| `sliding_o` | 0.127248 | 692.195 | 3.325871 | 26.136917 | 0.25 |
| `global_q` | 0.250746 | 702.548 | 2.684933 | 10.707796 | 0.25 |
| `global_k` | 0.034682 | 634.922 | 3.032374 | 87.434677 | 0.25 |
| `global_o` | 0.250725 | 702.606 | 0.253286 | 1.010217 | 0 |
| `final_logits` | 3.956539 | 712.383 | 3.967304 | 1.002721 | 0.125 |

Verification:

```bash
make cuda-kernels
make test-embedding-gather
```

`test-embedding-gather` still passed.

Interpretation:

- The source-level vectorization worked: SASS now has 128-bit loads and 64-bit stores in the target path.
- Register usage increased from the earlier `~25` register shape to `40`, but there are still no spills.
- Performance stayed near the prior DRAM-streaming roofline band. The clearest win in this short run was `global_k`, while the already roofline-limited large streams were essentially flat.
- Nonzero BF16 differences are expected from the changed FP32 reduction order before BF16 output rounding; mean errors remained small in the benchmark output.

CUDA guide notes:

- `$cuda-programming-guide` was queried for coalesced global memory, L2 streaming/persisting cache policy, `#pragma unroll`, low-level load functions, and `__restrict__` optimizer behavior. Relevant local guide pages included 60-61, 352-354, 519, 562, and 566.

## 2026-05-18 - RMSNorm and fused residual RMSNorm from llm.c layernorm

Runtime files:

- `src/gemma4_rmsnorm.cu`
- `src/gemma4_rmsnorm.cuh`
- `tests/test_rmsnorm.cu`
- `src/experiments/gemma4_rmsnorm_bench.cu`

Reason:

- The next documented baseline kernel after embedding gather is RMSNorm width `5376`.
- The first useful fusion target is the llm.c-style residual add plus normalization pattern.
- Gemma uses RMSNorm rather than LayerNorm, so the llm.c structure was kept while changing the math to remove mean subtraction and bias.

Implementation:

- Started by copying `llm.c/llmc/layernorm.cuh` to `src/gemma4_rmsnorm.cu`, then reduced it to inference-only forward kernels.
- Standalone RMSNorm computes `x * rsqrt(mean(x*x) + eps) * weight`.
- Fused residual RMSNorm writes the BF16 residual and normalizes that rounded residual value, matching the standalone `residual_add` followed by `rmsnorm` path.
- Kept the llm.c execution structure:
  - one warp owns one row;
  - fallback warp kernels do not require dynamic shared memory;
  - fast kernels cache the shared weight vector and one row per warp in dynamic shared memory;
  - launcher falls back when opt-in dynamic shared memory is not available.
- Added standalone `gemma4_residual_add_bf16` because the fused path needs a direct split baseline.
- Added a cuDNN frontend comparison path in `gemma4_rmsnorm_bench.cu` using `graph.rmsnorm` when `cudnn_frontend.h` is available. The default include path is `/tmp/cudnn-frontend/include`.

Research notes:

- PyTorch RMSNorm docs and the RMSNorm paper use the same core method: normalize by root mean square of the last dimension and apply a learned scale.
- The installed cuDNN 9 headers expose `CUDNN_RMS_NORM` and document backend norm forward support for both training and inference.
- NVIDIA's `cudnn-frontend` sample `samples/cpp/norm/rmsnorm.cpp` shows the C++ graph frontend path used by the benchmark.
- `$cuda-programming-guide` was queried for cuDNN/frontend benchmarking context, but the local guide mostly returned CUDA Graph sections rather than cuDNN norm guidance. The useful cuDNN evidence came from installed headers and NVIDIA frontend samples.
- `$exa-search` was attempted, but `EXA_API_KEY` was not visible in the tool environment. Browser search and the cloned NVIDIA `cudnn-frontend` repo were used instead.

Build and correctness:

```bash
make test-rmsnorm
make rmsnorm-bench
make cuda-kernels test-embedding-gather test-rmsnorm
```

Results:

- `rmsnorm tests passed`
- `embedding gather tests passed`

Primary benchmark command:

```bash
GEMMA4_RMSNORM_BENCH_SEED=0x1234 ./build/experiments/gemma4_rmsnorm_bench 50 10 3 4096
```

Representative `width=5376` results on RTX A6000:

| Rows | RMSNorm ms | RMSNorm GiB/s | cuDNN RMSNorm ms | cuDNN GiB/s | cuDNN max abs | Residual ms | Fused residual+RMS ms | Split residual+RMS ms | Split/Fused |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.020845 | 1.441 | 12.067860 | 0.002 | 0 | 0.014512 | 0.023012 | 0.034068 | 1.480 |
| 4 | 0.026762 | 4.491 | 10.980160 | 0.011 | 0 | 0.010392 | 0.021272 | 0.043290 | 2.035 |
| 16 | 0.018671 | 25.746 | 11.996526 | 0.040 | 0 | 0.008806 | 0.013852 | 0.025149 | 1.816 |
| 64 | 0.025356 | 75.833 | 11.558676 | 0.166 | 0.000488281 | 0.016013 | 0.023057 | 0.032451 | 1.407 |
| 256 | 0.015133 | 508.238 | 11.656405 | 0.660 | 0.00390625 | 0.012454 | 0.024473 | 0.021196 | 0.866 |
| 1024 | 0.036369 | 845.936 | 13.003598 | 2.366 | 0.00390625 | 0.050689 | 0.069214 | 0.080882 | 1.169 |
| 4096 | 0.133018 | 925.157 | 12.982645 | 9.479 | 0.00390625 | 0.195166 | 0.266606 | 0.326477 | 1.225 |

Larger sweep command:

```bash
GEMMA4_RMSNORM_BENCH_SEED=0x1234 ./build/experiments/gemma4_rmsnorm_bench 30 5 2 8192
```

Additional `8192 x 5376` result:

| Rows | RMSNorm ms | RMSNorm GiB/s | cuDNN RMSNorm ms | cuDNN GiB/s | cuDNN max abs | Fused residual+RMS ms | Split residual+RMS ms | Split/Fused |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8192 | 0.262535 | 937.490 | 12.506362 | 19.680 | 0.00390625 | 0.523484 | 0.648212 | 1.238 |

Interpretation:

- The custom standalone RMSNorm reaches about `925-937 GiB/s` at 4096-8192 rows with this logical byte model.
- cuDNN frontend RMSNorm numerically matches the custom output within BF16 rounding, but this setup has a large fixed execution overhead on A6000 for these shapes.
- Fused residual+RMSNorm is generally faster than launching residual add then standalone RMSNorm once rows are large enough, saving the separate residual reread.
- The `rows=256` split/fused anomaly is measurement noise or launch/occupancy crossover territory; the larger rows show the expected fused win.

## 2026-05-18 - RMSNorm raw GPU timing with CUDA graph capture

Reason:

- The original cuDNN RMSNorm benchmark timed repeated frontend `graph.execute` calls inside a CUDA-event interval.
- That is useful as an integrated steady-state API measurement, but it is not a raw GPU-speed comparison because frontend dispatch can leave idle gaps between device work submissions.

Implementation:

- Added `time_ms_graph` in `src/experiments/gemma4_bench_utils.cuh`.
- The helper stream-captures `iters` repeats of the target operation into a CUDA graph, instantiates it once, warms up graph replay, then records CUDA events around one graph replay and divides by the captured repeat count. This excludes CUDA graph capture, graph instantiation, warmup launches, and per-call host/API submission from the returned graph timing.
- `src/experiments/gemma4_rmsnorm_bench.cu` now reports both the original stream-loop timings and raw graph-captured timings:
  - `rms_ms` / `cudnn_ms`: repeated host/API execute path.
  - `rms_graph_kernel_ms` / `cudnn_graph_kernel_ms`: graph replay timing of already-captured device work.

Guide note:

- `$cuda-programming-guide` was queried for CUDA event and graph benchmarking context. The relevant local guide result was page 76, which describes using CUDA events in streams to time stream work including kernels. CUDA Graph results also reinforce separating graph setup from repeated execution.
- The Gemma 4 architecture doc was checked for production dimensions. The 31B dense target uses `hidden_size=5376`, batch-1 serving, `sliding_window=1024`, and a 256K context family. The compiled config uses `GEMMA4_MAX_POSITION_EMBEDDINGS=262144`, so full-context RMSNorm was tested as `262144 x 5376`.

Verification:

```bash
make rmsnorm-bench
make test-rmsnorm
```

`test-rmsnorm` passed.

Custom graph introspection from a one-off `cudaGraphGetNodes` checker:

| Rows | Captured nodes | Node type | Grid | Block | Dynamic smem |
| ---: | ---: | --- | --- | --- | ---: |
| 1 | 1 | kernel | `(1,1,1)` | `(32,8,1)` | `96768` |
| 4096 | 1 | kernel | `(512,1,1)` | `(32,8,1)` | `96768` |

Benchmark command:

```bash
GEMMA4_RMSNORM_BENCH_SEED=0x1234 ./build/experiments/gemma4_rmsnorm_bench 50 10 3 4096
```

Representative `width=5376` results on RTX A6000:

| Rows | Custom stream ms | Custom graph-kernel ms | cuDNN stream ms | cuDNN graph-kernel ms |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.023225 | 0.004400 | 7.370522 | 0.002634 |
| 4 | 0.017364 | 0.004716 | 6.371414 | 0.002625 |
| 16 | 0.025805 | 0.005671 | 7.226905 | 0.002726 |
| 64 | 0.019809 | 0.005693 | 7.759317 | 0.003723 |
| 256 | 0.024250 | 0.007769 | 6.857354 | 0.006050 |
| 1024 | 0.036131 | 0.035123 | 6.592338 | 0.035253 |
| 4096 | 0.132870 | 0.131937 | 6.518328 | 0.133100 |

Larger sweep:

```bash
GEMMA4_RMSNORM_BENCH_SEED=0x1234 ./build/experiments/gemma4_rmsnorm_bench 30 5 2 8192
```

Largest-row result:

| Rows | Custom stream ms | Custom graph-kernel ms | cuDNN stream ms | cuDNN graph-kernel ms |
| ---: | ---: | ---: | ---: | ---: |
| 8192 | 0.262535 | 0.261745 | 11.022665 | 0.262959 |

Production-size sweep:

```bash
GEMMA4_RMSNORM_BENCH_SEED=0x1234 ./build/experiments/gemma4_rmsnorm_bench 5 2 1 262144
```

Production-shape results:

| Rows | Meaning | Custom stream ms | Custom graph-kernel ms | cuDNN stream ms | cuDNN graph-kernel ms |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | batch-1 decode token | 0.018899 | 0.005536 | 5.503744 | 0.003507 |
| 1024 | 31B sliding-window-sized prefill chunk | 0.042483 | 0.035629 | 6.324723 | 0.036326 |
| 262144 | compiled max-position full-context pass | 8.297798 | 8.291284 | 9.805408 | 8.304595 |

Interpretation:

- The old multi-millisecond cuDNN RMSNorm result was dominated by frontend/API submission overhead, not device work.
- For larger row counts, the raw graph-captured GPU times are effectively tied: custom `0.131937 ms` vs cuDNN `0.133100 ms` at `4096 x 5376`, and custom `0.261745 ms` vs cuDNN `0.262959 ms` at `8192 x 5376`.
- At the compiled full-context production ceiling (`262144 x 5376`), the graph-captured GPU times remain effectively tied: custom `8.291284 ms` vs cuDNN `8.304595 ms`.
- For small rows, cuDNN's captured device work is faster than the current custom kernel, but that advantage is completely hidden by frontend execute overhead outside graph replay.
- Going forward, use `*_graph_kernel_ms` when discussing raw GPU speed and `*_ms` when discussing integrated host/API cost.

## 2026-05-18 - Decode production GEMV vs cuDNN BF16 1x1 conv

Reason:

- The active decode benchmark compared custom BF16 GEMV against cuBLAS GEMV/GEMM, but the current tuning gate asks whether the production decode matmul path beats cuDNN by at least `5%`.
- cuDNN backend matmul is not the clean matmul baseline here, so the benchmark restores the earlier mathematically equivalent 1x1 convolution comparator for the physical `[N, K]` weight layout.

Implementation:

- Added a BF16 cuDNN 1x1 convolution comparator to `src/experiments/gemma4_decode_bench.cu` for every production decode projection.
- Updated `make decode-bench` to link `-lcudnn`.
- Kept the production decode GEMV kernel unchanged after testing narrow warp-owned-column and column-grouping variants; those variants did not improve the tight `ffn_gate_up` case enough to keep.

Verification:

```bash
make decode-bench
make cuda-kernels
GEMMA4_DECODE_BENCH_SEED=0x1234 ./build/experiments/gemma4_decode_bench all 50 10 3
```

Production decode projection results on RTX A6000:

| Op | K | N | Layers/token | Custom ms | cuDNN ms | Custom vs cuDNN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ffn_gate_up` | 5376 | 43008 | 60 | 0.650958 | 0.656280 | 1.008176 |
| `ffn_down` | 21504 | 5376 | 60 | 0.327823 | 0.331135 | 1.010103 |
| `sliding_qkv` | 5376 | 16384 | 50 | 0.249854 | 0.270535 | 1.082772 |
| `sliding_o` | 8192 | 5376 | 50 | 0.126481 | 0.148044 | 1.170484 |
| `global_q` | 5376 | 16384 | 10 | 0.249905 | 0.281123 | 1.124916 |
| `global_k` | 5376 | 2048 | 10 | 0.034093 | 0.701345 | 20.571634 |
| `global_o` | 16384 | 5376 | 10 | 0.250122 | 0.255940 | 1.023262 |
| `final_logits` | 5376 | 262144 | 1 | 3.955549 | 3.964382 | 1.002233 |

Weighted by production layer counts:

| Path | Weighted decode projection ms |
| --- | ---: |
| Custom decode GEMV | 86.840359 |
| cuDNN BF16 1x1 conv | 96.522312 |

Interpretation:

- The production decode projection mix is `1.111491x` faster than cuDNN, an `11.15%` weighted win, clearing the `5%` gate.
- The largest bandwidth-saturated projections are much tighter individually, especially `ffn_gate_up`, `ffn_down`, `global_o`, and `final_logits`.
- Narrow warp-tiling and column-grouping tests did not improve `ffn_gate_up`; the current custom kernel is already near the cuDNN 1x1-conv bandwidth roofline for that shape.

## 2026-05-18 - Decode GEMV post-gate tuning sweep

Reason:

- After clearing the `5%` production-weighted cuDNN gate, continue tuning until the obvious decode GEMV thread/column-route variants stop producing production-speed wins.
- Focus on the actual decode projection mix, not a single synthetic shape.

Method:

```bash
make decode-bench
GEMMA4_DECODE_BENCH_SEED=0x1234 ./build/experiments/gemma4_decode_bench <op> 30 5 2
GEMMA4_DECODE_BENCH_SEED=0x1234 ./build/experiments/gemma4_decode_bench all 50 10 3
GEMMA4_DECODE_BENCH_SEED=0x1234 ./build/experiments/gemma4_decode_bench all 100 20 5
make cuda-kernels
git diff --check
```

Screened variants:

| # | Variant | Screened custom result | Decision |
| ---: | --- | --- | --- |
| 1 | generic `threads=128, cols=4` | `ffn_gate_up 0.651714`, `sliding_o 0.126844`, `global_k 0.033860` | Rejected; high-weight ops slower. |
| 2 | generic `threads=512, cols=4` | `ffn_gate_up 0.650739`, `sliding_o 0.126753`, `global_k 0.034049` | Rejected; not a production win. |
| 3 | generic `threads=256, cols=8` | `ffn_gate_up 0.651315`, `sliding_o 0.127519`, `global_k 0.035267` | Rejected. |
| 4 | generic `threads=256, cols=2` | `ffn_gate_up 0.650470`, `sliding_o 0.126437`, `global_k 0.033897` | Candidate, later beaten. |
| 5 | generic `threads=128, cols=8` | `ffn_gate_up 0.651994`, `sliding_o 0.127132`, `global_k 0.034060` | Rejected. |
| 6 | generic `threads=512, cols=8` | `ffn_gate_up 0.651111`, `sliding_o 0.127395`, `global_k 0.034473` | Rejected. |
| 7 | generic `threads=128, cols=2` | `ffn_gate_up 0.650998`, `sliding_o 0.126688`, `global_k 0.034905` | Rejected. |
| 8 | generic `threads=512, cols=2` | `ffn_gate_up 0.650105`, `sliding_o 0.126433`, `global_k 0.033567` | Kept; best generic route. |
| 9 | `ffn_down fixed4<128,4>` | `ffn_down 0.328463` | Rejected. |
| 10 | `ffn_down fixed4<512,1>` | `ffn_down 0.327438` | Candidate, later beaten. |
| 11 | `ffn_down fixed4<512,2>` | `ffn_down 0.327405` | Candidate, later beaten. |
| 12 | `global_o fixed4<256,2>` | `global_o 0.250242` | Rejected. |
| 13 | `global_o fixed4<256,4>` | `global_o 0.250467` | Rejected. |
| 14 | `global_o fixed4<128,4>` | `global_o 0.250861` | Rejected. |
| 15 | `final_logits fixed4<256,4>` | `final_logits 3.957258` | Rejected. |
| 16 | generic `threads=1024, cols=2` | `ffn_gate_up 0.650092`, `sliding_qkv 0.250277`, `sliding_o 0.126439`, `global_k 0.034306`; ptxas ignored `.minnctapersm` | Rejected. |
| 17 | generic `threads=768, cols=2` | `ffn_gate_up 0.650050`, `sliding_qkv 0.249568`, `sliding_o 0.126182`, `global_k 0.033888` | Rejected; mixed and not better weighted. |
| 18 | generic `threads=640, cols=2` | `ffn_gate_up 0.650382`, `sliding_qkv 0.250560`, `sliding_o 0.126318`, `global_k 0.034333` | Rejected. |
| 19 | generic `threads=512, cols=1` | `ffn_gate_up 0.649681`, `sliding_qkv 0.249843`, `sliding_o 0.126537`, `global_k 0.034037` | Rejected; helps `ffn_gate_up` but loses weighted mix. |
| 20 | generic `threads=512, cols=16` | `ffn_gate_up 0.653839`, `sliding_qkv 0.254826`, `sliding_o 0.128980`, `global_k 0.035023` | Rejected. |
| 21 | `ffn_down fixed4<1024,1>` | `ffn_down 0.327121` | Kept; best `ffn_down` route. |
| 22 | `ffn_down fixed4<1024,2>` | `ffn_down 0.327613`; ptxas ignored `.minnctapersm` | Rejected. |
| 23 | `ffn_down fixed4<768,1>` | `ffn_down 0.327362` | Rejected. |
| 24 | `final_logits fixed4<1024,1>` | `final_logits 3.951809` | Kept; best `final_logits` route. |
| 25 | `final_logits fixed4<768,1>` | `final_logits 3.954659` | Rejected. |
| 26 | `final_logits fixed4<512,1>` | `final_logits 3.955488` | Rejected. |
| 27 | `global_o fixed4<1024,1>` | `global_o 0.250594` | Rejected. |
| 28 | `final_logits fixed4<896,1>` | `final_logits 3.952104` | Rejected. |
| 29 | `final_logits fixed4<640,1>` | `final_logits 3.954255` | Rejected. |
| 30 | `final_logits fixed4<1024,2>` | `final_logits 3.951425`; ptxas ignored `.minnctapersm` | Rejected. |
| 31 | `ffn_down fixed4<896,1>` | `ffn_down 0.327773` | Rejected. |
| 32 | `ffn_down fixed4<640,1>` | `ffn_down 0.327632` | Rejected. |
| 33 | `ffn_down fixed4<384,1>` | `ffn_down 0.327397` | Rejected. |
| 34 | `global_o fixed4<768,1>` | `global_o 0.250193` | Rejected. |
| 35 | `global_o fixed4<640,1>` | `global_o 0.250218` | Rejected. |
| 36 | `global_o fixed4<384,1>` | `global_o 0.250167` | Rejected. |
| 37 | generic `threads=384, cols=2` | `ffn_gate_up 0.651110`, `sliding_qkv 0.250326`, `sliding_o 0.126726`, `global_k 0.034033` | Rejected. |
| 38 | generic `threads=896, cols=2` | `ffn_gate_up 0.650482`, `sliding_qkv 0.250063`, `sliding_o 0.127317`, `global_k 0.034405`; ptxas ignored `.minnctapersm` | Rejected. |
| 39 | generic `threads=512, cols=32` | `ffn_gate_up 0.672537`, `sliding_qkv 0.268206`, `sliding_o 0.136106`, `global_k 0.037322` | Rejected. |

Retained source settings:

- Generic decode GEMV route: `threads=512`, `cols_per_block=2`.
- `ffn_down`: `gemma4_decode_gemv_fixed4<GEMMA4_INTERMEDIATE_SIZE, GEMMA4_HIDDEN_SIZE, 1024, 1>`.
- `global_o`: restored to `512,1`.
- `final_logits`: `gemma4_decode_gemv_fixed4<GEMMA4_HIDDEN_SIZE, GEMMA4_VOCAB_SIZE, 1024, 1>`.

Final confirmation, `iters=100`, `warmup=20`, `trials=5`:

| Op | K | N | Layers/token | Custom ms | cuDNN ms | Custom vs cuDNN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ffn_gate_up` | 5376 | 43008 | 60 | 0.649589 | 0.655538 | 1.009157 |
| `ffn_down` | 21504 | 5376 | 60 | 0.326650 | 0.330976 | 1.013242 |
| `sliding_qkv` | 5376 | 16384 | 50 | 0.249276 | 0.271879 | 1.090673 |
| `sliding_o` | 8192 | 5376 | 50 | 0.125808 | 0.142452 | 1.132293 |
| `global_q` | 5376 | 16384 | 10 | 0.249468 | 0.270730 | 1.085232 |
| `global_k` | 5376 | 2048 | 10 | 0.033313 | 0.817800 | 24.549238 |
| `global_o` | 16384 | 5376 | 10 | 0.249828 | 0.255338 | 1.022057 |
| `final_logits` | 5376 | 262144 | 1 | 3.950665 | 3.964759 | 1.003567 |

Weighted by production layer counts:

| Path | Weighted decode projection ms |
| --- | ---: |
| Previous custom decode GEMV | 86.840359 |
| Tuned custom decode GEMV | 86.605295 |
| cuDNN BF16 1x1 conv in confirmation run | 97.310829 |

Interpretation:

- The retained tuning is a small additional production win: `0.235064 ms` weighted, or about `0.27%` beyond the previous custom decode GEMV result.
- The tuned production decode projection mix is `1.123607x` faster than cuDNN in the longer confirmation run, a `12.36%` weighted win.
- After the final retained win (`final_logits fixed4<1024,1>`), variants 25-39 formed a 15-test no-win tail. Those variants either regressed high-weight projections, emitted invalid launch-bound warnings, or only helped one op while losing the weighted production mix.

## 2026-05-18 - Decode RMSNorm production-shape tuning vs cuDNN graph

Reason:

- The production decode RMSNorm shape is one token row by Gemma 4 31B hidden width: `rows=1`, `width=5376`, `eps=1e-6`.
- The previous shared-memory RMSNorm path was optimized around multiple rows per CTA. At `rows=1`, only one warp did row math after the CTA-wide shared preload, which left most of the CTA idle.
- The fair raw-GPU comparator is the graph-captured cuDNN RMSNorm timing, not the cuDNN stream timing, because cuDNN frontend execution has large host/API cost outside graph replay.

Implementation:

- Added decode-specialized `rows=1,width=5376` kernels in `src/gemma4_rmsnorm.cu`.
- Kept the existing multi-row shared/warp paths unchanged.
- Added graph timings for fused residual+RMSNorm and split residual-add-plus-RMSNorm in `src/experiments/gemma4_rmsnorm_bench.cu`.
- Retained settings:
  - standalone decode RMSNorm: `768` threads.
  - fused decode residual+RMSNorm: `1024` threads.

Commands:

```bash
make test-rmsnorm rmsnorm-bench
./build/tests/test_rmsnorm
GEMMA4_RMSNORM_BENCH_SEED=0x1234 ./build/experiments/gemma4_rmsnorm_bench 400 50 7 1
make cuda-kernels
git diff --check
```

Screened decode variants, all on `rows=1,width=5376`:

| Variant | RMS graph ms | cuDNN graph ms | Fused graph ms | Decision |
| --- | ---: | ---: | ---: | --- |
| Previous shared path | 0.003918 | 0.002507 | n/a | Failed gate. |
| Decode shared, RMS `256`, fused `256` | 0.002733 | 0.002489 | n/a | Faster, still failed gate. |
| Decode shared, RMS `512`, fused `512` | 0.002455 | 0.002481 | n/a | Slight RMS win, below 5% gate. |
| Decode shared, RMS `1024`, fused `1024` | 0.002096 | 0.002475 | n/a | Cleared RMS gate; fused stream regressed. |
| RMS `1024`, fused `256` | 0.001931 | 0.002484 | 0.002979 | Good RMS, fused graph not best. |
| RMS `1024` no shared input cache, fused `256` | 0.001985 | 0.002460 | 0.002918 | Rejected; shared input cache was faster. |
| RMS `1024`, fused `512` | 0.002035 | 0.002521 | 0.002666 | Fused improved, RMS still passes. |
| RMS `1024`, fused `1024` | 0.001991 | 0.002500 | 0.002488 | Good balanced candidate. |
| RMS `1024`, fused `768` | 0.001986 | 0.002472 | 0.002526 | Rejected; fused worse than `1024`. |
| RMS `768`, fused `1024` | 0.001968 | 0.002481 | 0.002413 | Best short-run balance; kept for confirmation. |
| RMS `896`, fused `1024` | 0.002018 | 0.002480 | 0.002413 | Rejected; RMS worse than `768`. |
| RMS `640`, fused `1024` | 0.002342 | 0.002474 | 0.002486 | Rejected. |

Final confirmation, `iters=400`, `warmup=50`, `trials=7`:

| Rows | Width | RMS graph ms | cuDNN graph ms | Custom vs cuDNN | Fused graph ms | Split graph ms | Fused vs split |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 5376 | 0.001986 | 0.002606 | 1.312185 | 0.002236 | 0.003095 | 1.384 |

Interpretation:

- The tuned standalone decode RMSNorm is `1.312185x` faster than cuDNN graph timing, a `23.79%` lower-latency raw GPU win, clearing the `5%` gate.
- The tuned fused residual+RMSNorm decode kernel is `1.384x` faster than split residual add plus RMSNorm under graph timing.
- Stream timings also improved materially for decode: RMSNorm went from `0.015677 ms` to `0.010656 ms`, and fused residual+RMSNorm went from `0.026772 ms` to `0.012544 ms`.

## 2026-05-19 - Consolidated decode optimization findings from chat

Reason:

- This entry collects the main decisions, answers, and follow-up work from the decode GEMV, cache-hint, tensor-core, async-loading, cuDNN-comparison, warp-tiling, and RMSNorm discussion.
- The goal is to make the chat conclusions durable without requiring future agents to recover them from conversation context.

Production decode shape context:

- Target remains Gemma 4 dense inference, with the 31B dense path as the main optimization target.
- Steady-state decode is `M=1`, or one token row at a time.
- Hidden width is `5376`; production RMSNorm uses `rows=1,width=5376,eps=1e-6`.
- The dense text path has `60` layers: `50` sliding layers and `10` global layers.
- Decode projection shapes used for weighted production comparisons:

| Op | K | N | Layers/token |
| --- | ---: | ---: | ---: |
| `ffn_gate_up` | 5376 | 43008 | 60 |
| `ffn_down` | 21504 | 5376 | 60 |
| `sliding_qkv` | 5376 | 16384 | 50 |
| `sliding_o` | 8192 | 5376 | 50 |
| `global_q` | 5376 | 16384 | 10 |
| `global_k` | 5376 | 2048 | 10 |
| `global_o` | 16384 | 5376 | 10 |
| `final_logits` | 5376 | 262144 | 1 |

Benchmarking and fairness conclusions:

- For raw GPU speed, prefer CUDA graph replay timings such as `*_graph_kernel_ms`, or Nsight Compute kernel duration, over host-side stream/API timings.
- Stream-loop timing can include frontend/API costs, CPU launch overhead, or stream starvation effects. It is still useful for end-to-end host-visible behavior, but it is not the cleanest raw-kernel comparator.
- The decode GEMV benchmark currently uses CUDA events around stream loops. It excludes allocations, setup, descriptor construction, random fill, and correctness checks from the timed region, but it is not CUDA graph replay.
- The cuDNN decode GEMV comparator intentionally skips handle, descriptor, plan, and workspace setup in the timed region. The timed section measures the GPU work that cuDNN enqueues between CUDA events.
- The available cuDNN decode comparator is BF16 1x1 convolution. The cuDNN backend matmul path was not a clean usable comparator for these shapes.
- cuDNN frontend overhead made RMSNorm stream timing misleading. Graph replay timing is the fair raw-GPU comparison for RMSNorm.
- If we need stricter API-free decode GEMV comparison, add graph replay timing to the decode GEMV benchmark as was done for RMSNorm.

Decode GEMV implementation conclusions:

- The custom decode GEMV kernels do not use tensor cores.
- For strict `M=1` GEMV, the work is dominated by global memory traffic, vector loads, and reductions. It does not naturally map to tensor-core matrix tiles without batching or grouping multiple rows/tokens.
- Prefill dense GEMMs should use cuBLAS or cuBLASLt. Those library GEMMs can use tensor cores automatically for BF16 tensor-op math when the library selects a tensor-core algorithm.
- cuBLAS and cuDNN are host APIs. They cannot be called from inside a device-side fused kernel.
- `Packed128` is already implemented and retained. Decode GEMV uses 128-bit BF16 packed loads instead of the older 32-bit half2-style BF16 pair load pattern.
- The useful retained cache hint is streaming global load for weights through `load128cs`, because weights are streamed through the decode GEMV.
- The input vector load remains normally cached, because the same input row is reused across output columns.
- Broad L2 persisting-cache windows or persisting hints for the B/weight matrix were tried and removed because they hurt performance.
- There is no broad persisting L2 policy currently retained for decode GEMV.
- Output writes are vectorized on the current eight-column path as one 128-bit store for eight BF16 outputs.

Warp tiling and reduction conclusions:

- Current production decode GEMV is still CTA-cooperative over reductions and output columns.
- Local thread-count and column-count tuning produced the retained production settings, but most later variants after the retained win did not improve the weighted production mix.
- Warp-owned output-column mappings remain plausible benchmark candidates, not proven wins.
- The warp-tiling TODO should benchmark:
  - `1` warp -> `1` output column.
  - `1` warp -> `2` output columns.
  - `2` warps -> `1` output column for larger `K`.
  - The current CTA-reduction baseline.
- The point of those variants is to attack reduction overhead in `global_k`, `sliding_o`, `qkv`, and other decode GEMV shapes without assuming the mapping is better before measurement.

Shared-memory and banking conclusions:

- The current decode GEMV path does not stage bulk weight or activation tiles through shared memory.
- Decode GEMV uses only small shared-memory reductions for warp or block partial sums, so shared-memory bank conflicts and shared-memory data-mover pressure are not a primary current bottleneck.
- RMSNorm has shared caching in the multi-row path and a decode-specialized cache for the single-row path, but no evidence currently points to shared-memory banking as the limiting factor.
- Because decode GEMV mostly streams from global memory and reduces, moving more data into shared memory is not automatically beneficial. It should be justified by reuse and measured with Nsight Compute.

Async load and double-buffering conclusions:

- The current decode GEMV and RMSNorm kernels do not use `cp.async` or explicit double buffering.
- `cp.async` is most useful when staging global memory tiles into shared memory and reusing those tiles enough to hide latency.
- For the current strict `M=1` GEMV, weights are streamed once and not staged through shared memory, so async copy is not obviously faster.
- Async loading may become useful in a future tiled kernel, batched decode kernel, or attention-style kernel with real shared-memory tile reuse.
- More promising near-term fusion work is producer/consumer fusion: merge adjacent memory-bound operations so an intermediate does not have to be written to global memory and then read back by the next kernel.

Tensor-core conclusions:

- Tensor cores are not used by the custom decode GEMV kernels today.
- A tensor-core decode path would require reformulating the work into matrix tiles, most likely by batching multiple tokens, grouping independent rows, or accepting a different latency/throughput tradeoff.
- For single-token low-latency decode, tensor-core use is not automatically a win because the matrix has `M=1` and reductions/global memory movement dominate.
- Prefill is different: prefill has real matrix sizes, so cuBLAS/cuBLASLt should be the tensor-core path there.

Current decode GEMV source routing after the 2026-05-19 8-column pass:

- Generic decode GEMV route after the later eight-column store pass: `threads=512`, `cols_per_block=8`.
- `ffn_down`: fixed8 specialization with `1024` threads and `1` eight-column group per block.
- `global_o`: fixed8 specialization with `512` threads and `1` eight-column group per block.
- `final_logits`: fixed8 specialization with `1024` threads and `1` eight-column group per block.
- Earlier long confirmation weighted production decode projection result before the 8-column output-store pass:

| Path | Weighted decode projection ms |
| --- | ---: |
| Tuned custom decode GEMV | 86.605295 |
| cuDNN BF16 1x1 conv | 97.310829 |

Interpretation:

- The retained tuned custom decode GEMV mix is `1.123607x` faster than the cuDNN BF16 1x1 convolution comparator, a `12.36%` weighted win.
- The final retained GEMV win was small beyond the earlier Packed128/custom baseline, and the later 15-test no-win tail suggests further local tuning has diminishing returns without a larger structural change.

Retained decode RMSNorm tuning:

- The old multi-row shared RMSNorm path was the wrong shape for decode. At `rows=1`, most of the CTA was idle after a CTA-wide preload.
- The decode-specialized RMSNorm path now cooperates across the single row with a block-wide reduction.
- Retained settings:
  - Standalone decode RMSNorm: `768` threads.
  - Fused decode residual+RMSNorm: `1024` threads.
- Final graph replay result:

| Rows | Width | RMS graph ms | cuDNN graph ms | Custom vs cuDNN | Fused graph ms | Split graph ms | Fused vs split |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 5376 | 0.001986 | 0.002606 | 1.312185 | 0.002236 | 0.003095 | 1.384 |

Interpretation:

- The tuned standalone decode RMSNorm is `23.79%` lower latency than cuDNN graph replay for this production shape.
- The fused decode residual+RMSNorm kernel is `1.384x` faster than split residual add followed by RMSNorm under graph replay timing.

Follow-up work:

- Add CUDA graph replay timing to the decode GEMV benchmark if we need the same API-free raw-GPU comparison style used for RMSNorm.
- Run Nsight Compute on the retained decode GEMV and RMSNorm kernels to verify memory throughput, cache behavior, occupancy, stalls, instruction mix, shared-memory transactions, and achieved occupancy.
- Keep the warp-tiled GEMV TODO and benchmark it directly against the retained CTA-reduction baseline before changing production routing.
- If tensor cores are revisited, test batched decode or grouped multi-token decode, not strict `M=1` decode first.
- Consider producer/consumer fusion only after the unfused path is correct and measured. Likely candidates remain residual+RMSNorm, activation handling, softcap/sampling, and attention-output direct layout.
- Avoid broad cache-hint changes without profiling. The retained useful cache behavior is normal cached input reuse plus streaming loads for weights.

## 2026-05-19 - Decode GEMV 8-column 128-bit output stores

Change:

- Updated the generic decode GEMV route from `cols_per_block=2` to `cols_per_block=8`.
- Added an eight-BF16 output store helper so an eight-column block writes its result with one 128-bit store.
- Changed the fixed decode output paths for `ffn_down`, `global_o`, and `final_logits` from the old four-column store path to the same eight-column kernel route.
- Removed the now-unused fixed4 helper/kernel code so the matmul source no longer has a 64-bit BF16 output-store path.

Build and SASS checks:

```bash
make cuda-kernels decode-bench
cuobjdump --dump-sass build/gemma4_matmul_kernels.o | rg "LDG\.E|STG\.E"
```

SASS evidence:

- Input and weight traffic still uses 128-bit loads, including `LDG.E.128.CONSTANT` and streaming `LDG.E.EF.128`.
- Output stores for the compiled decode GEMV kernels show `STG.E.128`.
- The previous filtered `STG.E.64` line is no longer present after moving the fixed output paths to eight columns.

Correctness and timing smoke test:

```bash
GEMMA4_DECODE_BENCH_SEED=0x1234 ./build/experiments/gemma4_decode_bench all 3 1 1
```

| Op | Custom ms | Custom GB/s | cuBLAS GEMV ms | cuDNN conv1x1 ms | Max abs diff vs cuBLAS GEMV |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ffn_gate_up` | 0.655093 | 705.887 | 2.593568 | 0.661675 | 0.25 |
| `ffn_down` | 0.333333 | 693.633 | 0.334560 | 0.337685 | 0 |
| `sliding_qkv` | 0.255147 | 690.429 | 1.773760 | 0.275808 | 0.25 |
| `sliding_o` | 0.132693 | 663.789 | 2.324064 | 0.203061 | 0.25 |
| `global_q` | 0.256949 | 685.586 | 2.417675 | 0.279477 | 0.25 |
| `global_k` | 0.037152 | 592.703 | 1.288171 | 0.959061 | 0.25 |
| `global_o` | 0.256256 | 687.441 | 0.255104 | 0.259637 | 0 |
| `final_logits` | 3.963936 | 711.054 | 3.970923 | 3.970005 | 0.125 |

Interpretation:

- The requested 8-column grouping and 128-bit output-store route compiles and passes the decode benchmark correctness checks against cuBLAS/cuDNN comparators.
- The main generic route remains in the same performance band as the previous 8-column smoke tests, with `ffn_gate_up` roughly tied with cuDNN 1x1 and much faster than the cuBLAS GEMV comparator in this run.
- The fixed output routes are now also using the 8-column/128-bit-store path. `ffn_down`, `global_o`, and `final_logits` are effectively tied with cuBLAS and cuDNN on the large memory-streaming shapes in this short run.

## 2026-05-19 - Decode GEMV accumulator and shuffle-reduction sweep

Runtime file: `src/gemma4_matmul_kernels.cu`

Reason:

- The decode GEMV inner loop has two reduction-like pieces:
  - per-thread dot accumulation into one `sums[col]` value per output column;
  - cross-thread warp/block reduction of those per-thread partials.
- The concern was that the per-thread accumulation may be too sequential and that the warp reduction mode might matter.
- `$cuda-programming-guide` was queried for shuffle/reduction behavior. Relevant local guide pages:
  - p. 550: `__shfl_sync`, `__shfl_down_sync`, and `__shfl_xor_sync` exchange values between non-exited warp threads without shared memory.
  - p. 551: `__shfl_xor_sync` uses bitwise-XOR lane addressing and is used in tree reduction and broadcast patterns.
  - p. 552: shuffle intrinsics do not imply a memory barrier or memory ordering.
  - p. 555: the guide shows a full-warp butterfly reduction with `__shfl_xor_sync`.
- `$exa-search` returned relevant external references including a NVFP4 GEMV writeup, NVIDIA CUTLASS efficient GEMM documentation, CUTLASS reduction-fusion example code, and GEMM optimization writeups. They did not change the measured decision below.

Helper cleanup retained:

- The BF16 pair helper chain was simplified from four helpers into two helpers:
  - `gemma4_accumulate_bf16_pairs<Pair = 0>(...)`
  - `gemma4_accumulate_bf16_pack(...)`
- This keeps pair indices compile-time constants and keeps the call site unchanged.
- Added/kept compile-time contracts for packed BF16 pair width and alignment.
- Verified with:

```bash
make cuda-kernels
make decode-bench
GEMMA4_DECODE_BENCH_SEED=0x1234 ./build/experiments/gemma4_decode_bench all 3 1 1
git diff --check
```

Quantization/order check:

- Active runtime code is still BF16 storage with FP32 arithmetic, not SFP8/Q4.
- Decode GEMV order remains:
  `BF16 input/weight -> FP32 FMA accumulation -> BF16 output store`.
- Prefill cuBLAS order remains:
  `CUDA_R_16BF` inputs/outputs with `CUBLAS_COMPUTE_32F`.
- RMSNorm order remains:
  `BF16 input -> FP32 sumsq/scale/multiply -> BF16 output`.
- Fused residual+RMSNorm intentionally rounds the residual to BF16 before normalizing it, matching standalone `residual_add_bf16` followed by `rmsnorm_bf16`.

Baseline timing before accumulator variants:

```bash
make decode-bench
GEMMA4_DECODE_BENCH_SEED=0x1234 ./build/experiments/gemma4_decode_bench <op> 80 20 3
```

| Op | Baseline custom best ms |
| --- | ---: |
| `ffn_gate_up` | 0.650572 |
| `ffn_down` | 0.327785 |
| `sliding_qkv` | 0.250094 |
| `global_k` | 0.034028 |

Accumulator variant A: four independent accumulators kept live across the K loop:

- Implemented four per-column partial accumulators across `pack_idx`, then horizontally reduced them before the warp reduction.
- This created too much register pressure.
- `ptxas -v` showed most instantiated decode kernels using `64` registers plus `88 bytes` of spill stores and `88 bytes` of spill loads.

| Op | Baseline ms | 4-way cross-K accum ms | Decision |
| --- | ---: | ---: | --- |
| `ffn_gate_up` | 0.650572 | 0.894991 | Reject; large regression. |
| `ffn_down` | 0.327785 | 0.442062 | Reject; large regression. |
| `sliding_qkv` | 0.250094 | 0.346468 | Reject; large regression. |
| `global_k` | 0.034028 | 0.046345 | Reject; large regression. |

Accumulator variant B: two independent accumulators kept live across the K loop:

- Reduced the cross-K partials from four to two.
- `ptxas -v` showed no spills, but register usage stayed around `64` registers for most kernels.
- Timing was effectively tied on the large shapes and slightly worse on `global_k`.

| Op | Baseline ms | 2-way cross-K accum ms | Decision |
| --- | ---: | ---: | --- |
| `ffn_gate_up` | 0.650572 | 0.651055 | Reject; no win. |
| `ffn_down` | 0.327785 | 0.327732 | Reject; noise-level change. |
| `sliding_qkv` | 0.250094 | 0.250204 | Reject; no win. |
| `global_k` | 0.034028 | 0.034216 | Reject; slightly slower. |

Accumulator variant C: four independent accumulators local to one 128-bit pack:

- This variant is different from variant A: it split only the eight BF16 lanes inside a single `Packed128`, then folded back into `sum` before the next `pack_idx`.
- It avoided spills and mostly reduced register use by one register, but `global_o` rose to `71` registers.
- Timing was still noise-level and inconsistent.

| Op | Baseline/reference ms | Pack-local split ms | Decision |
| --- | ---: | ---: | --- |
| `ffn_gate_up` | 0.650572 | 0.650773 | Reject; tiny loss. |
| `ffn_down` | 0.327785 | 0.327668 | Reject; tiny gain only. |
| `sliding_qkv` | 0.250094 | 0.250071 | Reject; tiny gain only. |
| `global_k` | 0.034028 | 0.034070 | Reject; tiny loss. |
| `global_o` | 0.250931 | 0.251125 | Reject; tiny loss and higher regs. |
| `final_logits` | 3.954915 | 3.953999 | Reject; tiny gain only. |

Interpretation for accumulator variants:

- The per-thread accumulation is not as obviously bad as it looks because the current kernel already has `ColsPerBlock=8`, so each thread carries eight independent column accumulators.
- For common shapes, per-thread K work is small:
  - `K=5376, Threads=512`: about one or two 128-bit packs per thread.
  - `K=21504, Threads=1024`: about two or three 128-bit packs per thread.
- Additional independent accumulators increase register pressure and did not produce a measured win.

Shuffle-reduction variants:

- Temporarily added `GEMMA4_SHUFFLE_REDUCE_MODE`:
  - `0`: `__shfl_down_sync`
  - `1`: `__shfl_xor_sync`
  - `2`: direct indexed `__shfl_sync`
- Built each mode with:

```bash
rm -f build/experiments/gemma4_decode_bench
make decode-bench NVCCFLAGS="-std=c++17 -O3 -arch=sm_86 -Xptxas=-v -DGEMMA4_SHUFFLE_REDUCE_MODE=<mode>"
GEMMA4_DECODE_BENCH_SEED=0x1234 ./build/experiments/gemma4_decode_bench all 50 10 3
```

Resource observations:

- Mode `0` (`__shfl_down_sync`): `64` registers, `0` spills for all decode GEMV instantiations.
- Mode `1` (`__shfl_xor_sync`): `64` registers, `0` spills for all decode GEMV instantiations.
- Mode `2` (direct `__shfl_sync`): mostly `64` registers and `0` spills, but `global_o` rose to `78` registers.

Timing:

| Op | `down` ms | `xor` ms | direct `shfl_sync` ms |
| --- | ---: | ---: | ---: |
| `ffn_gate_up` | 0.650925 | 0.650830 | 0.651079 |
| `ffn_down` | 0.328040 | 0.328142 | 0.327603 |
| `sliding_qkv` | 0.250172 | 0.249986 | 0.250122 |
| `sliding_o` | 0.126863 | 0.126801 | 0.127135 |
| `global_q` | 0.250284 | 0.250235 | 0.250271 |
| `global_k` | 0.034077 | 0.034140 | 0.034198 |
| `global_o` | 0.250931 | 0.250852 | 0.251219 |
| `final_logits` | 3.954915 | 3.955053 | 3.954971 |

Weighted decode projection totals:

| Reduction mode | Weighted decode projection ms |
| --- | ---: |
| `__shfl_down_sync` | 86.897 |
| `__shfl_xor_sync` | 86.885 |
| direct `__shfl_sync` | 86.895 |

Correctness:

- All shuffle modes stayed within the same comparator ranges as the baseline.
- Representative max-abs diff ranges remained:
  - `0.25` for most cuBLAS GEMV/GEMM comparisons on projection outputs with reduction-order differences.
  - `0` for `ffn_down` and `global_o` versus cuBLAS.
  - `0.125` to `0.25` versus cuDNN 1x1 conv depending on op.

Conclusion:

- Keep the current `__shfl_down_sync` reduction.
- `__shfl_xor_sync` is effectively tied, but it makes every lane carry the final reduction value while the current kernel only needs lane 0 to write the warp partial.
- Direct indexed `__shfl_sync` adds source-lane/conditional logic and raised register use for at least one instantiation.
- Do not keep accumulator-splitting variants unless a future mapping changes the register/occupancy balance.
- Better next candidates are structural: warp-owned output columns, different columns-per-CTA by shape, CUDA graph replay timing for decode GEMV, or producer/consumer fusion after the unfused path is correct.

## 2026-05-19 - Decode GEMV pack-local ILP retest

Runtime file tested: `src/gemma4_matmul_kernels.cu`

Change tested:

- Replaced recursive pair accumulation inside one `Packed128<__nv_bfloat16>` with four independent pair-local partial sums:
  - `s0 = dot(pair 0)`
  - `s1 = dot(pair 1)`
  - `s2 = dot(pair 2)`
  - `s3 = dot(pair 3)`
  - `sum += (s0 + s1) + (s2 + s3)`
- This tests instruction-level parallelism inside one 128-bit BF16 pack without changing the thread/block mapping or warp/block reductions.

Commands:

```bash
make -B decode-bench NVCCFLAGS="-std=c++17 -O3 -arch=sm_86 -Xptxas=-v"
GEMMA4_DECODE_BENCH_SEED=0x1234 ./build/experiments/gemma4_decode_bench all 80 20 5
```

Resource result:

- Baseline: `64` registers, `0` spills for decode GEMV instantiations.
- ILP variant: no spills; most instantiations used `63` registers, but `global_o` rose to `70` registers.

Timing:

| Op | Baseline best ms | ILP best ms | Decision |
| --- | ---: | ---: | --- |
| `ffn_gate_up` | 0.650792 | 0.650826 | Tie/no win. |
| `ffn_down` | 0.327714 | 0.327387 | Tiny win. |
| `sliding_qkv` | 0.250011 | 0.249998 | Tie/no win. |
| `sliding_o` | 0.126681 | 0.126778 | Tiny loss. |
| `global_q` | 0.250040 | 0.250148 | Tiny loss. |
| `global_k` | 0.033776 | 0.034112 | Loss. |
| `global_o` | 0.250428 | 0.250979 | Loss with higher registers. |
| `final_logits` | 3.954785 | 3.954022 | Tiny win. |

Weighted decode projection total:

| Variant | Weighted decode projection ms |
| --- | ---: |
| Baseline recursive pair accumulation | 86.842 |
| Pack-local ILP accumulation | 86.838 |

Conclusion:

- The weighted difference was about `0.004 ms`, which is noise-level for this benchmark.
- The ILP helper also changed FP32 accumulation order and increased register use for `global_o`.
- Reverted the ILP helper. Keep the current recursive compile-time pair accumulation until a larger mapping change creates enough K work per thread to justify a different local accumulator shape.

## 2026-05-19 - Decode GEMV bound check without Nsight Systems

Question:

- Determine whether the decode GEMV kernel is memory-bound without using `nsys`.
- Working hypothesis: the kernel is memory-bound, so pack-local ILP should not help much.

Profiler attempt:

```bash
env GEMMA4_DECODE_BENCH_SEED=0x1234 \
  ncu --target-processes all --set speedOfLight \
  --kernel-name "regex:gemma4_decode_gemv_cols_kernel" \
  --launch-count 1 \
  ./build/experiments/gemma4_decode_bench ffn_gate_up 1 0 1
```

- `ncu` is installed (`2025.2.1.0`), but the benchmark process aborted under the Thunder runtime with an unsupported-library/internal-assertion failure.
- No Nsight Compute hardware-counter result was usable in this environment.

Roofline-style check from benchmark timings:

- The custom benchmark already prints effective weight-streaming throughput:

```bash
GEMMA4_DECODE_BENCH_SEED=0x1234 ./build/experiments/gemma4_decode_bench all 80 20 3
```

| Op | Best ms | Weight GB/s | FP32 TFLOP/s from `2*K*N` |
| --- | ---: | ---: | ---: |
| `ffn_gate_up` | 0.650852 | 710.487 | 0.710 |
| `ffn_down` | 0.327798 | 705.347 | 0.705 |
| `sliding_qkv` | 0.250012 | 704.608 | 0.705 |
| `sliding_o` | 0.126658 | 695.421 | 0.695 |
| `global_q` | 0.250019 | 704.589 | 0.704 |
| `global_k` | 0.033932 | 648.955 | 0.649 |
| `global_o` | 0.250604 | 702.946 | 0.703 |
| `final_logits` | 3.954875 | 712.683 | 0.713 |

Arithmetic intensity:

- Weight bytes are `2*K*N`.
- FP32 scalar dot work is also `2*K*N` FLOP if each FMA is counted as two FLOPs.
- So the optimistic arithmetic intensity using weight traffic only is about `1.0 FLOP/byte`.
- Input `x` and output `y` traffic make the true DRAM arithmetic intensity lower unless `x` is served from cache. Since `x` is tiny relative to weights and reused across many column blocks, weight traffic remains the useful lower-bound stream.

Interpretation:

- The kernel reaches roughly `649-713 GB/s` on weight traffic alone.
- It only reaches roughly `0.65-0.71 FP32 TFLOP/s`, far below RTX A6000-class FP32 peak.
- With arithmetic intensity around `1 FLOP/byte`, a bandwidth roof near the high hundreds of GB/s predicts a compute rate in the same `~0.7 TFLOP/s` band observed here.
- This explains why adding local ILP inside one BF16 pack did not help: the kernel is already dominated by streaming the weight matrix, not by FP32 FMA issue throughput.

Conclusion:

- Treat the current decode GEMV as memory-bandwidth-bound.
- Local arithmetic ILP is unlikely to matter unless it also improves memory behavior or reduces overhead without increasing register pressure.
- Better next levers are memory/layout/mapping changes: improve weight-load efficiency, tune columns per CTA by shape, use cache/read-only policies carefully, or change the warp/block ownership model.

## 2026-05-19 - Decode GEMV prefetch and cache-policy sweep

Runtime file tested: `src/gemma4_matmul_kernels.cu`

Question:

- Try explicit data prefetching or a related cache-policy change for the memory-bound decode GEMV.
- Consider whether shared-memory double buffering is worth implementing.

CUDA guide notes:

- `cooperative_groups::memcpy_async` / asynchronous data movement is global-to-shared and is described as useful when copies can overlap with computation and staged data is consumed from shared memory.
- The lower-level LDGSTS async-copy path can prefetch future iterations into shared memory and may increase bytes in flight.
- For this decode GEMV, the dominant weight data is consumed once by the same thread that loads it. Staging that data through shared memory would add shared-memory traffic and synchronization without cross-thread reuse.

Commands:

```bash
make -B decode-bench NVCCFLAGS="-std=c++17 -O3 -arch=sm_86 -Xptxas=-v <variant macro>"
GEMMA4_DECODE_BENCH_SEED=0x1234 ./build/experiments/gemma4_decode_bench all 50 10 3
```

Variants tested:

- Baseline: current `load128cs` streaming weight loads.
- `prefetch_next_iter`: inline `prefetch.global.L2` for the next K-stride `x` pack and all eight next K-stride weight packs.
- `prefetch_next_col`: inline `prefetch.global.L2` for the next output column's weight pack before loading/computing the current column.
- `normal_weight_load`: replace streaming `load128cs` weight loads with normal `load128` weight loads.

Resource result:

| Variant | Register result | Spills |
| --- | --- | --- |
| Baseline | `64` regs for decode GEMV instantiations | `0` |
| `prefetch_next_iter` | mostly `56` regs, but `global_o` rose to `74` regs | `0` |
| `prefetch_next_col` | mostly `64` regs, but `global_o` rose to `72` regs | `0` |
| `normal_weight_load` | `60` regs for decode GEMV instantiations | `0` |

Timing:

| Op | Baseline ms | Next-iter prefetch ms | Next-col prefetch ms | Normal load ms |
| --- | ---: | ---: | ---: | ---: |
| `ffn_gate_up` | 0.650769 | 0.652039 | 0.651096 | 0.652257 |
| `ffn_down` | 0.327995 | 0.334938 | 0.328007 | 0.328684 |
| `sliding_qkv` | 0.249914 | 0.250089 | 0.250109 | 0.250370 |
| `sliding_o` | 0.126680 | 0.128577 | 0.126916 | 0.127416 |
| `global_q` | 0.249910 | 0.250213 | 0.250189 | 0.250563 |
| `global_k` | 0.034218 | 0.034480 | 0.034090 | 0.034455 |
| `global_o` | 0.250586 | 0.252122 | 0.250829 | 0.251311 |
| `final_logits` | 3.954963 | 3.955522 | 3.956603 | 3.955877 |

Weighted decode projection total:

| Variant | Weighted decode projection ms |
| --- | ---: |
| Baseline | 86.858 |
| `prefetch_next_iter` | 87.476 |
| `prefetch_next_col` | 86.905 |
| `normal_weight_load` | 87.065 |

Conclusion:

- Do not keep explicit L2 prefetching in this kernel.
- Do not switch the weight loads from streaming `load128cs` to normal `load128`.
- The current access pattern already has coalesced 128-bit streaming weight loads and enough resident warps to hide latency; extra prefetch instructions mostly add overhead.
- Shared-memory double buffering is not attractive for this mapping because the weight packs are one-use data with no cross-thread reuse, and `x` is already loaded once per thread and reused across all eight columns.
- A useful async-copy/double-buffer design likely needs a different tile mapping where staged data is reused by multiple consumers.

## 2026-05-19 - Decode GEMV shared-memory double-buffer test

Runtime file tested: `src/gemma4_matmul_kernels.cu`

Question:

- Implement a real shared-memory double-buffer variant, not only L2 prefetching, and benchmark it.

CUDA guide notes:

- The CUDA guide describes pipelines as the mechanism for double- or multi-buffering producer-consumer patterns (p. 129).
- `memcpy_async` is global-to-shared and is intended to overlap data movement with computation when staged shared-memory data is consumed later (p. 244).
- The guide's pipeline examples schedule multiple async-copy stages, wait for the current batch, compute it, and schedule the next batch (pp. 295, 312-313).

Implementation shape tested:

- The impossible full-buffer design would stage two copies of all eight columns for every CTA thread:
  `2 stages * Threads * 8 columns * 16B`.
  - For `Threads=512`: about `128 KiB` before reductions.
  - For `Threads=1024`: about `256 KiB` before reductions.
  - This does not fit the sm_86 shared-memory budget for this kernel.
- The tested feasible design stages one weight column at a time:
  `2 stages * Threads * 16B`.
  - For `Threads=512`: `16,896B` total shared memory including existing warp reductions.
  - For `Threads=1024`: `33,792B` total shared memory including existing warp reductions.
- For each K-pack handled by a thread:
  - Load `x_pack` into a register.
  - Async copy column 0's 128-bit weight pack from global memory to shared stage 0.
  - Wait for stage 0.
  - For each column:
    - Async copy the next column's weight pack into the other shared stage.
    - Compute the current column from the current shared stage.
    - Wait for the next stage and swap.

Representative code shape:

```cuda
__pipeline_memcpy_async(&w_stage[0][threadIdx.x],
                        w_col_major + col0 * K + element_idx,
                        sizeof(Gemma4Bf16Pack));
__pipeline_commit();
__pipeline_wait_prior(0);

int stage = 0;
#pragma unroll
for (int col = 0; col < ColsPerBlock; ++col) {
  const int next_col = col + 1;
  const int next_stage = stage ^ 1;
  if (next_col < ColsPerBlock) {
    __pipeline_memcpy_async(&w_stage[next_stage][threadIdx.x],
                            w_col_major + (col0 + next_col) * K + element_idx,
                            sizeof(Gemma4Bf16Pack));
    __pipeline_commit();
  }

  gemma4_accumulate_bf16_pack(x_pack, w_stage[stage][threadIdx.x], sums[col]);

  if (next_col < ColsPerBlock) {
    __pipeline_wait_prior(0);
    stage = next_stage;
  }
}
```

Commands:

```bash
make -B decode-bench NVCCFLAGS="-std=c++17 -O3 -arch=sm_86 -Xptxas=-v -DGEMMA4_DECODE_DOUBLE_BUFFER_WEIGHT_COLS=1"
GEMMA4_DECODE_BENCH_SEED=0x1234 ./build/experiments/gemma4_decode_bench all 50 10 3
```

Resource result:

- Baseline: `64` registers, existing reduction shared memory only, `0` spills.
- Double buffer:
  - `60` regs for most instantiations, `55` regs for `global_o`, `0` spills.
  - Shared memory rose to `16,896B` for `512`-thread kernels and `33,792B` for `1024`-thread kernels.

Timing:

| Op | Baseline ms | Double-buffer ms | Decision |
| --- | ---: | ---: | --- |
| `ffn_gate_up` | 0.650769 | 0.654609 | Reject; slower. |
| `ffn_down` | 0.327995 | 0.329315 | Reject; slower. |
| `sliding_qkv` | 0.249914 | 0.253850 | Reject; slower. |
| `sliding_o` | 0.126680 | 0.127734 | Reject; slower. |
| `global_q` | 0.249910 | 0.254698 | Reject; slower. |
| `global_k` | 0.034218 | 0.036264 | Reject; slower. |
| `global_o` | 0.250586 | 0.251602 | Reject; slower. |
| `final_logits` | 3.954963 | 3.958124 | Reject; slower. |

Weighted decode projection total:

| Variant | Weighted decode projection ms |
| --- | ---: |
| Baseline | 86.858 |
| Shared-memory double buffer | 87.498 |

Conclusion:

- The real shared-memory double-buffer variant is correct but slower.
- It reduced register use, but that did not compensate for the extra async-copy instructions, shared-memory traffic, wait operations, and increased shared-memory footprint.
- Do not keep this implementation in the runtime path.
- Shared-memory async copy may become useful only after changing the mapping so staged data is reused by multiple threads or multiple output columns instead of being consumed once by the same thread that loaded it.

## 2026-05-19 - Decode GEMV explicit 128-bit load policy

Runtime file tested: `src/gemma4_matmul_kernels.cu`

Question:

- Keep the direct global-to-register GEMV path, but make the intended load policy explicit:
  normal 128-bit loads for reused activation packs and streaming 128-bit loads for one-use weight packs.

Implementation:

- Added inlined decode load helpers:
  - `gemma4_load_activation_pack(...)` uses `load128(...)`.
  - `gemma4_load_streaming_weight_pack<K>(...)` uses `load128cs(...)`.
- Added a compile-time contract that `Gemma4Bf16Pack` maps to one aligned `int4` load.
- Added a host-side decode pointer guard for non-null, 16-byte-aligned `x`, `w_col_major`, and `y`.
- Kept the runtime path as global memory -> registers -> ALUs; no shared-memory staging.

Commands:

```bash
make cuda-kernels decode-bench
nvcc -std=c++17 -O3 -arch=sm_86 -Isrc --ptx src/gemma4_matmul_kernels.cu -o /tmp/gemma4_matmul_kernels_load_policy.ptx
nvcc -std=c++17 -O3 -arch=sm_86 -Xptxas=-v -Isrc -c src/gemma4_matmul_kernels.cu -o /tmp/gemma4_matmul_kernels_load_policy.o
nvcc -std=c++17 -O3 -arch=sm_86 -Isrc /tmp/gemma4_decode_arg_guard_test.cu src/gemma4_matmul_kernels.cu -lcublas -lcudnn -o /tmp/gemma4_decode_arg_guard_test
GEMMA4_DECODE_BENCH_SEED=0x1234 ./build/experiments/gemma4_decode_bench all 2 1 1
GEMMA4_DECODE_BENCH_SEED=0x1234 ./build/experiments/gemma4_decode_bench all 50 10 3
```

PTX/resource result:

- PTX load mix:
  - `ld.global.nc.v4.u32`: `7`
  - `ld.global.cs.v4.s32`: `56`
  - `prefetch`: `0`
  - `cp.async`: `0`
- ptxas resource use:
  - `63` registers for all decode GEMV instantiations except `global_o`.
  - `72` registers for `global_o`.
  - `0` spill stores and `0` spill loads.
  - Shared memory remains only the warp-reduction scratch: `512B` or `1024B` depending on thread count.

Correctness:

- `all 2 1 1` completed for all decode projection shapes.
- BF16 reference diffs stayed in the same expected range as prior runs; largest max absolute difference was `0.25` versus BF16 cuBLAS/cuDNN references.
- Temporary arg-guard test returned `cudaErrorInvalidValue` for null decode pointers before launching a kernel.

Timing:

| Op | Custom best ms |
| --- | ---: |
| `ffn_gate_up` | 0.651138 |
| `ffn_down` | 0.328560 |
| `sliding_qkv` | 0.250545 |
| `sliding_o` | 0.127626 |
| `global_q` | 0.250479 |
| `global_k` | 0.034163 |
| `global_o` | 0.250664 |
| `final_logits` | 3.954700 |

Weighted decode projection total:

| Variant | Weighted decode projection ms |
| --- | ---: |
| Prior direct-load baseline | 86.858 |
| Explicit load-policy helpers | 86.998 |

Conclusion:

- Keep this shape. The helper split makes the intended memory policy explicit without changing the generated load instructions.
- The tiny timing delta is noise-level against the prior direct-load baseline.
- This reinforces the current decision: use `load128` for reused activation packs, `load128cs` for streaming weight packs, and avoid shared-memory async copy until the GEMV mapping has real staged-data reuse.

## 2026-05-19 - Decode GEMV register double-buffer cleanup and bench

Runtime file tested: `src/gemma4_matmul_kernels.cu`

Question:

- Replace the existing `GEMMA4_DECODE_REGISTER_DOUBLE_BUFFER` implementation with a clearer register-prefetch variant and check whether it is correct and faster.

Research notes:

- NVIDIA's CUDA prefetching guidance says register prefetching can help when profiling shows memory stalls and bandwidth is not saturated, but it spends extra registers and must be verified with profiling/benchmarks.
- The CUDA guide occupancy sections are relevant because register count and shared-memory use limit resident warps/blocks, which are the normal latency-hiding mechanism for memory-bound kernels.
- For this decode GEMV mapping, the register double buffer only changes instruction timing: it preloads the next column's one-use weight pack before accumulating the current column's pack. It does not reduce total DRAM traffic.

Implementation:

- Removed the old `gemma4_accumulate_bf16_cols_register_buffered` helper.
- Added a separate compile-time double-buffer path:
  - `gemma4_decode_gemv_cols_dot_register_double_buffered_device(...)`
  - `gemma4_accumulate_prefetched_weight_cols(...)`
- Kept the default direct-load path separate, so the register-prefetch path is enabled only with `-DGEMMA4_DECODE_REGISTER_DOUBLE_BUFFER`.

Commands:

```bash
make -B decode-bench NVCCFLAGS="-std=c++17 -O3 -arch=sm_86 -DGEMMA4_DECODE_REGISTER_DOUBLE_BUFFER"
GEMMA4_DECODE_BENCH_SEED=0x1234 ./build/experiments/gemma4_decode_bench all 2 1 1
GEMMA4_DECODE_BENCH_SEED=0x1234 ./build/experiments/gemma4_decode_bench all 50 10 3
nvcc -std=c++17 -O3 -arch=sm_86 -DGEMMA4_DECODE_REGISTER_DOUBLE_BUFFER -Xptxas=-v -Isrc -c src/gemma4_matmul_kernels.cu -o /tmp/gemma4_matmul_register_double_buffer.o

make -B decode-bench
GEMMA4_DECODE_BENCH_SEED=0x1234 ./build/experiments/gemma4_decode_bench all 50 10 3
```

Correctness:

- The smoke run completed for all decode projection shapes.
- BF16 reference diffs stayed in the same expected range as the direct-load path; largest max absolute difference was `0.25` versus BF16 cuBLAS/cuDNN references.
- Exact-zero diffs versus cuBLAS remained for `ffn_down` and `global_o`.

Resource result for `GEMMA4_DECODE_REGISTER_DOUBLE_BUFFER`:

- `0` stack frame bytes.
- `0` spill stores and `0` spill loads for all instantiated decode GEMV kernels.
- `63` registers for all instantiations except `global_o`.
- `72` registers for `global_o`.
- Shared memory stayed at the existing warp-reduction scratch size: `512B` or `1024B` depending on thread count.

Timing, seeded `all 50 10 3`:

| Op | Direct-load best ms | Register double-buffer best ms |
| --- | ---: | ---: |
| `ffn_gate_up` | 0.651085 | 0.651015 |
| `ffn_down` | 0.328530 | 0.328137 |
| `sliding_qkv` | 0.250395 | 0.250356 |
| `sliding_o` | 0.127427 | 0.127486 |
| `global_q` | 0.250749 | 0.250852 |
| `global_k` | 0.034523 | 0.034418 |
| `global_o` | 0.251091 | 0.250967 |
| `final_logits` | 3.954532 | 3.954656 |

Weighted decode projection total:

| Variant | Weighted decode projection ms |
| --- | ---: |
| Direct-load | 86.986 |
| Register double buffer | 86.958 |

Conclusion:

- The cleaned-up register double-buffer path appears correct.
- The timing difference is noise-level: about `0.03 ms` on an `~87 ms` weighted decode projection total.
- Do not enable it by default yet. Keep it as a compile-time experiment path unless repeated profiling shows a real stall/bandwidth reason to prefer it.

## 2026-05-19 - Decode GEMV cp.async double-buffer implementation

Runtime file tested: `src/gemma4_matmul_kernels.cu`

Question:

- Replace the register-prefetch experiment path with real cp.async global-to-shared staging.

Research notes:

- The CUDA guide describes pipelines as the mechanism for staging work and coordinating double- or multi-buffer producer-consumer patterns.
- The primitives API supports `__pipeline_memcpy_async`, `__pipeline_commit`, and `__pipeline_wait_prior(N)` for global-to-shared asynchronous copies.
- The primitive requirements match this kernel's pack shape: destination is shared memory, source is global memory, and `size_and_align` must be `4`, `8`, or `16`. `Gemma4Bf16Pack` is a 16-byte aligned pack.
- The guide also warns not to read the shared-memory destination before waiting for the async copy to complete.

Implementation:

- Removed the `GEMMA4_DECODE_REGISTER_DOUBLE_BUFFER` source path.
- Added `GEMMA4_DECODE_CP_ASYNC_DOUBLE_BUFFER`.
- Each participating thread copies one 16-byte weight pack into `__shared__ Gemma4Bf16Pack weight_stages[2][Threads]`.
- The inner column loop alternates two stages:
  - issue cp.async for the next column,
  - accumulate the current column from the waited-on shared stage,
  - wait for the next stage before consuming it.
- The default direct global-to-register path remains unchanged unless the cp.async macro is defined.

Commands:

```bash
make -B decode-bench NVCCFLAGS="-std=c++17 -O3 -arch=sm_86 -DGEMMA4_DECODE_CP_ASYNC_DOUBLE_BUFFER"
GEMMA4_DECODE_BENCH_SEED=0x1234 ./build/experiments/gemma4_decode_bench all 2 1 1
GEMMA4_DECODE_BENCH_SEED=0x1234 ./build/experiments/gemma4_decode_bench all 50 10 3
nvcc -std=c++17 -O3 -arch=sm_86 -DGEMMA4_DECODE_CP_ASYNC_DOUBLE_BUFFER -Xptxas=-v -Isrc -c src/gemma4_matmul_kernels.cu -o /tmp/gemma4_matmul_cp_async.o
nvcc -std=c++17 -O3 -arch=sm_86 -DGEMMA4_DECODE_CP_ASYNC_DOUBLE_BUFFER -Isrc --ptx src/gemma4_matmul_kernels.cu -o /tmp/gemma4_matmul_cp_async.ptx

make -B decode-bench
GEMMA4_DECODE_BENCH_SEED=0x1234 ./build/experiments/gemma4_decode_bench all 50 10 3
```

PTX/resource result:

- PTX contains `cp.async.cg.shared.global`, `cp.async.commit_group`, and `cp.async.wait_group`.
- `0` stack frame bytes.
- `0` spill stores and `0` spill loads for all instantiated decode GEMV kernels.
- Registers:
  - `55` for all cp.async instantiations except `global_o`.
  - `54` for `global_o`.
- Shared memory:
  - `16896B` for 512-thread kernels.
  - `33792B` for 1024-thread kernels.

Correctness:

- The smoke run completed for all decode projection shapes.
- BF16 reference diffs stayed in the same expected range as the direct-load path; largest max absolute difference was `0.25` versus BF16 cuBLAS/cuDNN references.
- Exact-zero diffs versus cuBLAS remained for `ffn_down` and `global_o`.

Timing, seeded `all 50 10 3`:

| Op | Direct-load best ms | cp.async best ms |
| --- | ---: | ---: |
| `ffn_gate_up` | 0.651042 | 0.654468 |
| `ffn_down` | 0.328219 | 0.328632 |
| `sliding_qkv` | 0.250456 | 0.253651 |
| `sliding_o` | 0.127536 | 0.127636 |
| `global_q` | 0.250470 | 0.254575 |
| `global_k` | 0.034400 | 0.036015 |
| `global_o` | 0.250852 | 0.251440 |
| `final_logits` | 3.954719 | 3.957656 |

Weighted decode projection total:

| Variant | Weighted decode projection ms |
| --- | ---: |
| Direct-load | 86.967 |
| cp.async double buffer | 87.428 |

Conclusion:

- The cp.async path is correct and actually emits cp.async instructions, but it is slower for this GEMV mapping.
- The slowdown is expected: each weight pack is still one-use data, so shared-memory staging adds async-copy, commit/wait, and shared-load overhead without reducing DRAM traffic.
- Keep cp.async as a compile-time experiment path only. The runtime default should remain the direct `load128`/`load128cs` path.

## 2026-05-19 - Nsight Compute counter profiling attempt for decode GEMV

Runtime files tested:

- `src/gemma4_matmul_kernels.cu`
- Temporary CUDA-only harness: `/tmp/gemma4_decode_ncu_harness.cu`

Question:

- Collect hardware counters for the direct-load decode GEMV and the cp.async variant rather than reasoning only from source and CUDA-event timing.

Requested metrics:

```bash
gpu__time_duration.sum
l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum
lts__t_bytes.sum
dram__bytes.sum
l1tex__t_sector_hit_rate.pct
lts__t_sector_hit_rate.pct
smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct
smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct
smsp__warps_active.avg.pct_of_peak_sustained_active
```

Setup:

- Verified Nsight Compute is installed: `ncu 2025.2.1.0`.
- Verified the machine reports `NVIDIA RTX A6000`, driver `580.126.16`.
- Queried GA102 metric names with:

```bash
ncu --query-metrics --chips ga102
```

- First attempted profiling the existing `gemma4_decode_bench` with a kernel filter:

```bash
GEMMA4_DECODE_BENCH_SEED=0x1234 ncu \
  --target-processes all \
  --kernel-name regex:gemma4_decode_gemv_cols_kernel \
  --launch-count 1 \
  --metrics <requested metrics> \
  --csv \
  /tmp/gemma4_decode_bench_direct ffn_gate_up 1 0 1
```

Result:

- Thunder aborted before counters were collected:

```text
Your program encountered a fatal error likely caused by a Thunder Compute runtime issue:
  Your program uses a library that is not yet supported on Thunder.
```

Follow-up isolation:

- Built `/tmp/gemma4_decode_ncu_harness.cu`, a CUDA-only standalone harness for the same `ffn_gate_up` decode GEMV mapping:
  - no cuBLAS
  - no cuDNN
  - direct-load binary: `/tmp/gemma4_decode_ncu_direct`
  - cp.async binary: `/tmp/gemma4_decode_ncu_cp_async`
- Both harness binaries run normally outside `ncu`.

Standalone `ncu` command:

```bash
ncu \
  --target-processes all \
  --kernel-name regex:decode_kernel \
  --launch-count 1 \
  --metrics gpu__time_duration.sum,dram__bytes.sum \
  --csv \
  /tmp/gemma4_decode_ncu_direct 1
```

Result:

- Thunder still aborted under Nsight Compute injection, even without cuBLAS/cuDNN.
- Captured process state showed:

```text
abortMessage="Unimplemented CUDA export table function: Table=cupti_device_query, Index=7"
lastErrorUser="Your program uses a library that is not yet supported on Thunder."
```

Conclusion:

- Nsight Compute hardware-counter profiling is blocked on this Thunder instance by missing CUPTI support in Thunder's CUDA export table.
- This is not a benchmark-code or cuBLAS/cuDNN issue; it reproduces with a CUDA-only harness.
- No reliable hardware-counter conclusion can be drawn from this environment. To answer the bandwidth-vs-latency question properly, rerun the same `ncu` commands on a production instance or another host where CUPTI/Nsight Compute profiling is supported.

## 2026-05-19 - Decode GEMV cp.async debug with load128/load128cs staging

Runtime file tested: `src/gemma4_matmul_kernels.cu`

Question:

- The cp.async double buffer should plausibly be faster, so isolate whether the slowdown comes from shared-memory staging itself or from cp.async commit/wait mechanics.

Implementation:

- Kept the same two-stage shared-memory layout used by the cp.async path:
  `Gemma4Bf16Pack weight_stages[2][Threads]`.
- Added two compile-time sibling variants:
  - `GEMMA4_DECODE_SHARED_STAGE_LOAD128`
  - `GEMMA4_DECODE_SHARED_STAGE_LOAD128CS`
- Both variants use the same shared-stage consume path as cp.async, but stage data with normal global loads instead of `__pipeline_memcpy_async`.
- The direct default path remains unchanged.

Commands:

```bash
make -B decode-bench NVCCFLAGS="-std=c++17 -O3 -arch=sm_86"
GEMMA4_DECODE_BENCH_SEED=0x1234 ./build/experiments/gemma4_decode_bench all 50 10 3

make -B decode-bench NVCCFLAGS="-std=c++17 -O3 -arch=sm_86 -DGEMMA4_DECODE_CP_ASYNC_DOUBLE_BUFFER"
GEMMA4_DECODE_BENCH_SEED=0x1234 ./build/experiments/gemma4_decode_bench all 50 10 3

make -B decode-bench NVCCFLAGS="-std=c++17 -O3 -arch=sm_86 -DGEMMA4_DECODE_SHARED_STAGE_LOAD128"
GEMMA4_DECODE_BENCH_SEED=0x1234 ./build/experiments/gemma4_decode_bench all 50 10 3

make -B decode-bench NVCCFLAGS="-std=c++17 -O3 -arch=sm_86 -DGEMMA4_DECODE_SHARED_STAGE_LOAD128CS"
GEMMA4_DECODE_BENCH_SEED=0x1234 ./build/experiments/gemma4_decode_bench all 50 10 3
```

Timing, seeded `all 50 10 3`:

| Op | Direct | cp.async | Shared `load128` | Shared `load128cs` |
| --- | ---: | ---: | ---: | ---: |
| `ffn_gate_up` | 0.651158 | 0.654705 | 0.652065 | 0.651222 |
| `ffn_down` | 0.328492 | 0.329076 | 0.329271 | 0.328312 |
| `sliding_qkv` | 0.250387 | 0.253834 | 0.250493 | 0.250413 |
| `sliding_o` | 0.127457 | 0.127696 | 0.127501 | 0.127295 |
| `global_q` | 0.250526 | 0.254549 | 0.250682 | 0.250627 |
| `global_k` | 0.034491 | 0.035948 | 0.034846 | 0.034957 |
| `global_o` | 0.251147 | 0.251456 | 0.251411 | 0.251354 |
| `final_logits` | 3.954678 | 3.957649 | 3.955973 | 3.955886 |

Weighted decode projection total:

| Variant | Weighted decode projection ms |
| --- | ---: |
| Direct | 86.988 |
| cp.async | 87.481 |
| Shared `load128` | 87.105 |
| Shared `load128cs` | 86.983 |

PTX/resource check:

| Variant | Registers | Shared memory | Spills | PTX evidence |
| --- | --- | --- | --- | --- |
| Direct | `63`, `72` | `512B`, `1024B` | `0/0` | `ld.global.cs=56`, `ld.global.nc=7` |
| cp.async | `54`, `55` | `16896B`, `33792B` | `0/0` | `cp.async.cg.shared.global=56`, `commit_group=56`, `wait_group=56` |
| Shared `load128` | `59`, `60` | `16896B`, `33792B` | `0/0` | `ld.global.nc=63`, no cp.async |
| Shared `load128cs` | `64` | `16896B`, `33792B` | `0/0` | `ld.global.cs=56`, `ld.global.nc=7`, no cp.async |

Interpretation:

- Shared-memory staging itself is not the issue. Shared `load128cs` is essentially tied with direct.
- The weight load policy matters: shared `load128cs` beats shared `load128`, matching the direct path's streaming-weight policy.
- The cp.async-specific overhead is the likely issue in this mapping: the compiler emits one `cp.async`, one commit, and one wait group per staged weight pack group in the unrolled column loop.
- The work between issuing the next stage and waiting for it is only one 16-byte-pack BF16 dot contribution, so there is not enough independent compute to amortize commit/wait overhead.

Conclusion:

- Do not enable cp.async for this current per-thread/per-column GEMV mapping.
- If cp.async is revisited, change the tiling so each async-staged payload is reused more, or group larger batches per wait. A same-thread one-use 16-byte pack is too fine-grained for this pipeline structure.

## 2026-05-20 - RoPE CUDA baseline from Triton reference

Runtime files:

- `src/gemma4_rope.cu`
- `src/gemma4_rope.cuh`
- `tests/test_rope.cu`
- `docs/rope.md`

Reason:

- The next attention-prep step after Q/K/V projection and normalization is to apply RoPE before KV-cache write and attention.
- The user provided a Triton split-half RoPE implementation. The CUDA baseline mirrors its physical `[batch, seq, heads, head_dim]` layout while adding Gemma 4's partial global p-RoPE support.

Implementation:

- Added in-place BF16 Q/K rotation with precomputed FP32 cos/sin tables.
- The low-level entry point accepts explicit cos/sin row strides, including compact `rotary_dim / 2` rows and full `head_dim` rows.
- Added forward-layout launchers for direct `[batch, heads, seq, head_dim]` Q/K buffers, matching the Python `forward` inputs for inference.
- Sliding wrapper uses `head_dim=256`, `rotary_dim=256`.
- Global wrapper uses `head_dim=512`, `rotary_dim=128`, leaving the NoPE tail unchanged.
- Baseline mapping is one CUDA block per `(batch_seq_row, head)` pair; later fusion can combine Q/K RMSNorm, RoPE, and KV-cache write.

Verification:

```bash
make cuda-kernels test-rope
```

Result:

- `rope tests passed`

## 2026-05-20 - RoPE CUDA vs cuDNN pointwise benchmark

Runtime files:

- `src/experiments/gemma4_rope_bench.cu`
- `src/experiments/gemma4_bench_utils.cuh`
- `src/gemma4_rope.cu`

Reason:

- Benchmark the standalone RoPE CUDA kernel against a cuDNN implementation while excluding compile time, cuDNN frontend/plan overhead, and per-call host launch overhead where possible.

Implementation notes:

- cuDNN does not expose a standalone RoPE primitive in the installed headers/docs.
- A cuDNN frontend pointwise/concat graph and a pointwise-only graph were both tried first. Both failed plan creation on RTX A6000/cuDNN 9.22.
- The benchmark therefore uses deprecated-but-available `cudnnOpTensor` pointwise calls for the cuDNN path. Setup, descriptor creation, table conversion, warmup, and CUDA graph capture/instantiate are outside the measured region.
- The main comparison column is `*_graph_ms`: it captures repeated work once, instantiates outside timing, warms up, then records CUDA events around graph replay.
- cuDNN `cudnnOpTensor` required BF16 cos/sin tables, so the cuDNN diff is expected to be about one BF16 step versus the custom CUDA path's FP32 table reads.

Commands:

```bash
make rope-bench
./build/experiments/gemma4_rope_bench 100 20 3 1024
make test-rope
```

Environment:

- Device: NVIDIA RTX A6000
- Seed: `0xa5bcb03db46f9b6`
- Iterations: `100`, warmup: `20`, trials: `3`, batch: `1`, cos batch: `1`

CUDA graph replay timings:

| Case | Seq | Custom ms | cuDNN ms | Custom GiB/s | cuDNN GiB/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| sliding | 1 | 0.001553 | 0.016028 | 58.954 | 5.712 |
| sliding | 4 | 0.001557 | 0.015719 | 235.185 | 23.297 |
| sliding | 16 | 0.001793 | 0.016584 | 816.852 | 88.327 |
| sliding | 64 | 0.003277 | 0.023316 | 1788.139 | 251.308 |
| sliding | 256 | 0.015829 | 0.078103 | 1480.627 | 300.083 |
| sliding | 1024 | 0.074939 | 0.288012 | 1251.025 | 325.508 |
| global | 1 | 0.001417 | 0.014462 | 24.235 | 2.374 |
| global | 4 | 0.001425 | 0.015022 | 96.352 | 9.142 |
| global | 16 | 0.001637 | 0.015486 | 335.473 | 35.472 |
| global | 64 | 0.002676 | 0.017714 | 821.248 | 124.044 |
| global | 256 | 0.006655 | 0.032649 | 1320.726 | 269.202 |
| global | 1024 | 0.030191 | 0.116031 | 1164.472 | 302.989 |

Correctness:

- cuDNN max abs diff stayed at `0.0078125` for Q/K except one global seq-1 K case at `0.00390625`, consistent with BF16 table precision.
- `make test-rope` passed after the benchmark changes.

## 2026-05-20 - RMSNorm scale-free V launch tuning

Runtime files:

- `src/gemma4_rmsnorm.cu`
- `src/experiments/gemma4_rmsnorm_bench.cu`
- `tests/test_rmsnorm.cu`

Reason:

- Add and tune a separate scale-free BF16 RMSNorm path for V normalization without making the learned-weight RMSNorm kernels carry `has_weight` template logic.
- Compare realistic V head shapes against a cuDNN one-scale RMSNorm baseline.

Reference checks:

- NVIDIA CUDA Programming Guide query confirmed that `--use_fast_math`, `-ftz`, `-prec-div`, `-prec-sqrt`, and `-fmad` can change device math behavior and should be correctness-checked before use.
- The same guide notes that register/occupancy tuning can help or hurt, especially when it causes spills.
- Exa search found the official NVIDIA NVCC docs and CUDA Best Practices NVCC switch notes for the compiler-flag sweep.

Commands:

```bash
make -B rmsnorm-bench
GEMMA4_RMSNORM_BENCH_SEED=0x20260520 ./build/experiments/gemma4_rmsnorm_bench 80 20 3 1024 5376
GEMMA4_RMSNORM_BENCH_SEED=0x20260520 ./build/experiments/gemma4_rmsnorm_bench 80 20 3 16384 256
GEMMA4_RMSNORM_BENCH_SEED=0x20260520 ./build/experiments/gemma4_rmsnorm_bench 80 20 3 4096 512
make test-rmsnorm
make cuda-kernels
```

Environment:

- Device: NVIDIA RTX A6000
- Seed: `0x20260520`
- Main timings use CUDA graph replay columns.

Candidates:

| Candidate | Decision | Notes |
| --- | --- | --- |
| `--use_fast_math` | Rejected | Mixed timing and changes math semantics. |
| Direct BF16 scale helper | Rejected | No stable win over the shared unit-gamma helper. |
| `-ftz=true` | Rejected | Mixed timing and changes subnormal behavior. |
| `kDecodeRmsnormThreads=512` | Rejected | Helped scale-free decode row 1, but later decode thread counts were better. |
| `kDecodeRmsnormThreads=1024` | Rejected | Improved weighted row 1 to `0.001998 ms`, then was superseded by 704 threads. |
| `kDecodeRmsnormThreads=704` | Kept | Best weighted row 1, `0.001952 ms`; scale-free tied the best measured value at `0.001760 ms`. |
| `kDecodeFusedThreads=768` | Rejected | Improved fused row 1 to `0.002409 ms`, then was superseded by 672 threads. |
| `kDecodeFusedThreads=512` | Rejected | Fused row 1 regressed to `0.002491 ms`. |
| `kDecodeFusedThreads=672` | Kept | Best fused row 1, `0.002395 ms`. |
| `kDecodeFusedThreads=736` | Rejected | Close to 672, but slower at `0.002401 ms`. |
| `kRmsnormBlockSize=32` | Rejected | Helped some small rows but lost width-256 rows 64 and width-512 rows 4 versus 64. |
| `kRmsnormBlockSize=96` | Rejected | Did not beat 64 on the balanced small-row set. |
| `kRmsnormBlockSize=128` | Rejected | Good for larger prefill rows, but regressed global decode-like width-512 rows 4. |
| `kRmsnormBlockSize=512` | Rejected | Regressed most width-256 and width-512 V rows. |
| `kRmsnormBlockSize=64` | Kept | Best decode-like V points, with acceptable large-row tradeoffs. |

Final scale-free V graph replay timings:

| Width | Rows | Custom ms | cuDNN one-scale ms | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 16 | 0.001421 | 0.001892 | 1.33x |
| 256 | 64 | 0.001414 | 0.002142 | 1.51x |
| 256 | 1024 | 0.001860 | 0.002342 | 1.26x |
| 256 | 16384 | 0.026069 | 0.027506 | 1.06x |
| 512 | 4 | 0.001476 | 0.001918 | 1.30x |
| 512 | 64 | 0.001519 | 0.002438 | 1.61x |
| 512 | 1024 | 0.002185 | 0.003346 | 1.53x |
| 512 | 4096 | 0.010502 | 0.011870 | 1.13x |

Verification:

```bash
make test-rmsnorm
make cuda-kernels
git diff --check
```

Result:

- `rmsnorm tests passed`
- Final kept constants: `kDecodeRmsnormThreads=704`, `kDecodeFusedThreads=672`, `kRmsnormBlockSize=64`.

## 2026-05-20 - RMSNorm real-shape benchmark rerun

Runtime files:

- `src/gemma4_rmsnorm.cu`
- `src/experiments/gemma4_rmsnorm_bench.cu`
- `src/experiments/gemma4_bench_utils.cuh`

Reason:

- Rerun the RMSNorm benchmark suite on realistic Gemma 4 shapes after the final launch-constant tuning.
- Use only CUDA graph replay columns for cuDNN comparisons so cuDNN handle/frontend graph construction, plan build, workspace allocation, CUDA graph capture, and CUDA graph instantiation are outside the measured region.

Benchmarking rule:

- The benchmark still prints stream-loop columns, but the comparison below uses `*_graph_kernel_ms` columns only.
- The cuDNN frontend graph is built once per `(rows, width)` before timing. CUDA graph capture and instantiation happen inside `time_ms_graph` before CUDA events are recorded.

Commands:

```bash
make -B test-rmsnorm rmsnorm-bench
mkdir -p /tmp/gemma4_rmsnorm_real_20260520
GEMMA4_RMSNORM_BENCH_SEED=0x20260520 ./build/experiments/gemma4_rmsnorm_bench 100 30 5 1024 5376 > /tmp/gemma4_rmsnorm_real_20260520/hidden_w5376.csv
GEMMA4_RMSNORM_BENCH_SEED=0x20260520 ./build/experiments/gemma4_rmsnorm_bench 100 30 5 16384 256 > /tmp/gemma4_rmsnorm_real_20260520/sliding_v_w256.csv
GEMMA4_RMSNORM_BENCH_SEED=0x20260520 ./build/experiments/gemma4_rmsnorm_bench 100 30 5 4096 512 > /tmp/gemma4_rmsnorm_real_20260520/global_v_w512.csv
```

Environment:

- Device: NVIDIA RTX A6000
- Seed: `0x20260520`
- Iterations: `100`, warmup: `30`, trials: `5`
- cuDNN frontend: compiled

Real-shape mapping:

- Hidden RMSNorm: width `5376`, rows up to `1024`.
- Sliding V/head RMSNorm: width `256`, rows up to `16384` (`1024 * 16` KV heads).
- Global V/head RMSNorm: width `512`, rows up to `4096` (`1024 * 4` KV heads).

Scale-free RMSNorm graph replay:

| Shape | Rows | Custom ms | cuDNN one-scale ms | Speedup |
| --- | ---: | ---: | ---: | ---: |
| hidden w5376 | 1 | 0.001924 | 0.002529 | 1.31x |
| hidden w5376 | 4 | 0.004052 | 0.002548 | 0.63x |
| hidden w5376 | 16 | 0.004088 | 0.002608 | 0.64x |
| hidden w5376 | 64 | 0.004221 | 0.003134 | 0.74x |
| hidden w5376 | 256 | 0.007315 | 0.004972 | 0.68x |
| hidden w5376 | 1024 | 0.034361 | 0.034851 | 1.01x |
| sliding w256 | 1 | 0.001384 | 0.001965 | 1.42x |
| sliding w256 | 4 | 0.001389 | 0.001843 | 1.33x |
| sliding w256 | 16 | 0.001383 | 0.001869 | 1.35x |
| sliding w256 | 64 | 0.001403 | 0.002114 | 1.51x |
| sliding w256 | 256 | 0.001427 | 0.002149 | 1.51x |
| sliding w256 | 1024 | 0.001690 | 0.002354 | 1.39x |
| sliding w256 | 4096 | 0.002899 | 0.003645 | 1.26x |
| sliding w256 | 8192 | 0.010870 | 0.011645 | 1.07x |
| sliding w256 | 16384 | 0.026101 | 0.027444 | 1.05x |
| global w512 | 1 | 0.001490 | 0.001882 | 1.26x |
| global w512 | 4 | 0.001476 | 0.001879 | 1.27x |
| global w512 | 16 | 0.001483 | 0.001991 | 1.34x |
| global w512 | 64 | 0.001553 | 0.002437 | 1.57x |
| global w512 | 256 | 0.001551 | 0.002528 | 1.63x |
| global w512 | 1024 | 0.002083 | 0.003436 | 1.65x |
| global w512 | 4096 | 0.010461 | 0.012035 | 1.15x |

Weighted RMSNorm graph replay:

| Shape | Rows | Custom ms | cuDNN ms | Speedup |
| --- | ---: | ---: | ---: | ---: |
| hidden w5376 | 1 | 0.002172 | 0.002523 | 1.16x |
| hidden w5376 | 4 | 0.006324 | 0.002559 | 0.40x |
| hidden w5376 | 16 | 0.006284 | 0.002688 | 0.43x |
| hidden w5376 | 64 | 0.006550 | 0.003594 | 0.55x |
| hidden w5376 | 256 | 0.009953 | 0.005540 | 0.56x |
| hidden w5376 | 1024 | 0.040808 | 0.035017 | 0.86x |
| sliding w256 | 16 | 0.001680 | 0.001878 | 1.12x |
| sliding w256 | 1024 | 0.002185 | 0.002346 | 1.07x |
| sliding w256 | 16384 | 0.026403 | 0.027580 | 1.04x |
| global w512 | 4 | 0.001753 | 0.001910 | 1.09x |
| global w512 | 1024 | 0.002826 | 0.003336 | 1.18x |
| global w512 | 4096 | 0.011070 | 0.011980 | 1.08x |

Fused residual add plus weighted RMSNorm graph replay:

| Shape | Rows | Fused ms | Split CUDA ms | Split speedup | Custom residual + cuDNN RMSNorm ms | cuDNN split speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hidden w5376 | 1 | 0.002435 | 0.003458 | 1.42x | 0.004036 | 1.66x |
| hidden w5376 | 1024 | 0.075144 | 0.083685 | 1.11x | 0.085250 | 1.13x |
| sliding w256 | 16 | 0.001693 | 0.002876 | 1.70x | 0.003399 | 2.01x |
| sliding w256 | 4096 | 0.010832 | 0.012856 | 1.19x | 0.014506 | 1.34x |
| sliding w256 | 16384 | 0.050849 | 0.058269 | 1.15x | 0.066084 | 1.30x |
| global w512 | 4 | 0.001973 | 0.002967 | 1.50x | 0.003447 | 1.75x |
| global w512 | 1024 | 0.005133 | 0.004814 | 0.94x | 0.005964 | 1.16x |
| global w512 | 4096 | 0.026419 | 0.026671 | 1.01x | 0.029587 | 1.12x |

Correctness:

- `make test-rmsnorm` passed.
- Max output diffs versus cuDNN were BF16-sized or zero:
  - hidden w5376: up to `0.00390625`
  - sliding w256: up to `0.00390625`
  - global w512: up to `0.00195312`
- Max `rstd` diffs were at most `2.38419e-07`.

Notes:

- Nsight Compute was not rerun for this benchmark entry. Earlier in this environment it aborted under Thunder's unsupported CUPTI path, so these results are benchmark timings, not NCU counter profiles.

## 2026-05-20 - Hidden fused residual add RMSNorm prefill, one CTA per row

Implemented a hidden-width `5376` prefill path for fused residual add plus weighted
RMSNorm. It keeps one CUDA block per token row, does the RMS reduction inside that
block, writes the residual row, then writes the normalized row. No cross-block sync is
needed because rows are independent.

Command:

```bash
GEMMA4_RMSNORM_BENCH_SEED=0x20260520 \
  ./build/experiments/gemma4_rmsnorm_bench 100 30 5 1024 5376 \
  > /tmp/gemma4_rmsnorm_prefill_blockrow_20260520/hidden_w5376.csv
```

Fused residual add plus weighted RMSNorm graph replay, hidden width `5376`:

| Rows | Previous fused ms | One-CTA fused ms | Split CUDA ms | cuDNN split ms |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.002435 | 0.002412 | 0.003476 | 0.004045 |
| 4 | 0.008589 | 0.002443 | 0.007402 | 0.004110 |
| 16 | 0.011001 | 0.002609 | 0.007430 | 0.004306 |
| 64 | 0.012346 | 0.003585 | 0.008250 | 0.005822 |
| 256 | 0.021028 | 0.017522 | 0.021120 | 0.020122 |
| 1024 | 0.075144 | 0.066300 | 0.083404 | 0.085198 |

Correctness versus the custom residual plus cuDNN RMSNorm split path:

- output max abs diff: `0` through rows `64`, `0.000976562` at rows `256`,
  `0.00195312` at rows `1024`
- `rstd` max abs diff: at most `1.19209e-07`

Conclusion: the one-block-per-row hidden prefill kernel fixes the earlier prefill
weakness. It beats split CUDA by `1.21x` to `3.03x` and cuDNN split by `1.15x` to
`1.68x` on these graph replay timings.

## 2026-05-20 - Hidden fused prefill with residual pack kept in registers

Changed the hidden-width fused residual add plus RMSNorm prefill kernel to keep one
128-bit residual pack per thread in registers instead of writing all residual packs to
shared memory. This works cleanly because width `5376` is exactly `672` BF16 packs and
the prefill launch uses `672` threads.

Same benchmark command as above, output:
`/tmp/gemma4_rmsnorm_prefill_reg_20260520/hidden_w5376.csv`.

| Rows | Shared-residual fused ms | Register-residual fused ms | cuDNN split ms |
| ---: | ---: | ---: | ---: |
| 1 | 0.002412 | 0.002507 | 0.004096 |
| 4 | 0.002443 | 0.002252 | 0.004129 |
| 16 | 0.002609 | 0.002187 | 0.004345 |
| 64 | 0.003585 | 0.003276 | 0.005835 |
| 256 | 0.017522 | 0.017339 | 0.020076 |
| 1024 | 0.066300 | 0.065971 | 0.085200 |

Conclusion: keep the register version. It is better for prefill rows `4+`, especially
the smaller prefill sizes, and still beats cuDNN split by `1.16x` to `1.99x` for those
rows.

## 2026-05-20 - Hidden fused prefill weight prefetch check

Changed the hidden-width fused residual add plus RMSNorm prefill kernel to load the
per-pack learned weight before the row reduction, keeping that `128`-bit weight pack
live in registers until the normalized output write.

Commands:

```bash
make -B rmsnorm-bench
GEMMA4_RMSNORM_BENCH_SEED=0x20260520 \
  ./build/experiments/gemma4_rmsnorm_bench 100 30 5 1024 5376 \
  > /tmp/gemma4_rmsnorm_gamma_prefetch_20260520/hidden_w5376.csv
make test-rmsnorm
```

Environment:

- Device: NVIDIA RTX A6000
- Seed: `0x20260520`
- Iterations: `100`, warmup: `30`, trials: `5`
- cuDNN frontend: compiled

Fused residual add plus weighted RMSNorm graph replay, hidden width `5376`:

| Rows | Previous register ms | Weight-prefetch ms | Delta | Split CUDA ms | cuDNN split ms |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.002507 | 0.002459 | -1.91% | 0.003425 | 0.004092 |
| 4 | 0.002252 | 0.002307 | +2.44% | 0.007255 | 0.004069 |
| 16 | 0.002187 | 0.002330 | +6.54% | 0.008076 | 0.004305 |
| 64 | 0.003276 | 0.003605 | +10.04% | 0.008811 | 0.005825 |
| 256 | 0.017339 | 0.017248 | -0.52% | 0.021150 | 0.020125 |
| 1024 | 0.065971 | 0.065816 | -0.23% | 0.083469 | 0.085200 |

Correctness:

- `make test-rmsnorm` passed.
- Correctness versus the custom residual plus cuDNN RMSNorm split path remained unchanged:
  output max abs diff was `0` through rows `64`, `0.000976562` at rows `256`,
  and `0.00195312` at rows `1024`; `rstd` max abs diff was at most `1.19209e-07`.

Conclusion: the weight-prefetch version is not a clear win. It slightly improves rows
`1`, `256`, and `1024`, but regresses rows `4`, `16`, and `64` in this run. Keep this
only if later full-pipeline profiling shows those larger-row points matter more than the
small-prefill regression.

## 2026-05-20 - Hidden fused prefill focused tuning pass

Runtime files:

- `src/gemma4_rmsnorm.cu`
- `src/experiments/gemma4_rmsnorm_hidden_fused_bench.cu`
- `Makefile`

Reason:

- Tune only small settings/minor code choices for the hidden-width fused residual add
  plus weighted RMSNorm prefill kernel.
- Avoid the full RMSNorm benchmark's cuDNN/frontend setup and unrelated kernels during
  iteration. The focused benchmark times only
  `gemma4_residual_add_rmsnorm_bf16(..., rows, width=5376, ...)` for `rows > 1`.
  `rows=1` is intentionally excluded because production routing sends that case to the
  separate decode kernel.

Commands:

```bash
make -B rmsnorm-hidden-fused-bench \
  NVCCFLAGS="-std=c++17 -O3 -arch=sm_86 -Xptxas=-v"
GEMMA4_RMSNORM_BENCH_SEED=0x20260520 \
  ./build/experiments/gemma4_rmsnorm_hidden_fused_bench 300 20 7 1024 \
  > /tmp/gemma4_hidden_fused_tune_20260520/final_simplified.csv
make -B test-rmsnorm
```

Environment:

- Device: NVIDIA RTX A6000
- Seed: `0x20260520`
- Focused benchmark: `300` captured kernel calls, `20` graph warmup launches,
  `7` trials
- Timing column used for decisions: CUDA graph replay `fused_graph_best_ms`

Tuning sweep:

| Variant | Sum best ms rows 4-1024 | Sum avg ms rows 4-1024 | Aggregate best delta |
| --- | ---: | ---: | ---: |
| Baseline: prefetch weight, min-blocks 1, streaming input loads | 0.090220 | 0.090350 | baseline |
| Late weight load, min-blocks 1 | 0.090030 | 0.090300 | -0.21% |
| Late weight load, min-blocks 2 | 0.089957 | 0.090114 | -0.29% |
| Late weight load, min-blocks 2, cached input loads | 0.089807 | 0.090009 | -0.46% |
| Same plus streaming stores | 0.089786 | 0.089957 | -0.48% |
| Final simplified kept code | 0.089864 | 0.090050 | -0.39% |

The streaming-store candidate moved the aggregate by only about `0.023%` versus the
previous candidate, below the requested `0.05%` stopping threshold, so tuning stopped
there. The final kept code uses the meaningful settings only:

- `__launch_bounds__(Threads, 2)` for the hidden prefill kernel
- cached `load128g` input loads for `inp1` and `inp2`
- no early gamma/weight prefetch; load the weight at output application time
- ordinary global stores for residual and normed outputs

Final focused benchmark, graph replay:

| Rows | Final ms | Effective GiB/s |
| ---: | ---: | ---: |
| 4 | 0.002065 | 96.968 |
| 16 | 0.001980 | 404.521 |
| 64 | 0.002867 | 1117.629 |
| 256 | 0.017154 | 747.249 |
| 1024 | 0.065798 | 779.257 |

Resource check:

- Hidden prefill kernel ptxas stayed at `28` registers, `0` stack, `0` spill stores,
  `0` spill loads, and `96` bytes shared memory.

Correctness:

- `make -B test-rmsnorm` passed.

## 2026-05-20 - RMSNorm helper cleanup validation

Runtime file:

- `src/gemma4_rmsnorm.cu`

Reason:

- Remove dead scale-free fused residual add plus RMSNorm paths and tiny RMSNorm-local
  wrapper helpers.
- Keep the remaining block reduction helper explicit: callers now pass thread, lane,
  warp, and shared scratch storage instead of letting the helper read `threadIdx.x`.

Commands:

```bash
make -B test-rmsnorm
nvcc -std=c++17 -O3 -arch=sm_86 -Xptxas=-v -Isrc \
  -c src/gemma4_rmsnorm.cu -o /tmp/gemma4_rmsnorm.o
make -B rmsnorm-hidden-fused-bench
./build/experiments/gemma4_rmsnorm_hidden_fused_bench
```

Environment:

- Device: NVIDIA RTX A6000
- Focused benchmark defaults: `200` captured kernel calls, `20` graph warmup launches,
  `5` trials
- Benchmark seed printed by run: `0x6b1233712354a17b`

Focused hidden-width fused benchmark, graph replay:

| Rows | Best ms | Avg ms | Effective GiB/s |
| ---: | ---: | ---: | ---: |
| 4 | 0.002104 | 0.002119 | 95.171 |
| 16 | 0.002157 | 0.002158 | 371.451 |
| 64 | 0.002839 | 0.003056 | 1128.947 |
| 256 | 0.017169 | 0.017213 | 746.606 |
| 1024 | 0.065835 | 0.065862 | 778.819 |

Resource check:

- Hidden prefill kernel: `28` registers, `0` stack, `0` spills, `96` bytes shared memory.
- Decode kernels: `38` registers, `0` stack, `0` spills.
- Shared generic kernels: up to `44` registers, `0` stack, `0` spills.

Correctness and cleanup checks:

- `make -B test-rmsnorm` passed.
- `git diff --check -- src/gemma4_rmsnorm.cu src/gemma4_rmsnorm.cuh tests/test_rmsnorm.cu`
  passed.
- No remaining RMSNorm-local `__forceinline__` helpers or removed helper symbols.

## 2026-05-20 - RMSNorm dead fallback removal

Runtime files:

- `src/gemma4_rmsnorm.cu`
- `src/gemma4_rmsnorm.cuh`
- `tests/test_rmsnorm.cu`

Reason:

- Remove the fallback warp kernels and generic fused residual add plus RMSNorm paths
  that are not part of the Gemma 4 dense model path.
- Keep fused residual add plus learned RMSNorm hidden-width-only, because the residual
  stream is width `5376`.
- Keep standalone learned RMSNorm and scale-free RMSNorm shared kernels, because Q/K/V
  norms still need width `256` and `512` coverage.

Commands:

```bash
make -B test-rmsnorm
nvcc -std=c++17 -O3 -arch=sm_86 -Xptxas=-v -Isrc \
  -c src/gemma4_rmsnorm.cu -o /tmp/gemma4_rmsnorm.o
make -B rmsnorm-hidden-fused-bench
./build/experiments/gemma4_rmsnorm_hidden_fused_bench
make -B rmsnorm-bench
```

Environment:

- Device: NVIDIA RTX A6000
- Focused benchmark defaults: `200` captured kernel calls, `20` graph warmup launches,
  `5` trials
- Benchmark seed printed by run: `0x71383c77c5b4702c`

Focused hidden-width fused benchmark, graph replay:

| Rows | Best ms | Avg ms | Effective GiB/s |
| ---: | ---: | ---: | ---: |
| 4 | 0.002080 | 0.002083 | 96.277 |
| 16 | 0.002133 | 0.002136 | 375.631 |
| 64 | 0.002841 | 0.002947 | 1127.866 |
| 256 | 0.017191 | 0.017252 | 745.627 |
| 1024 | 0.065856 | 0.065873 | 778.564 |

Resource check:

- Kernel count in `src/gemma4_rmsnorm.cu`: `6`.
- Hidden prefill kernel: `28` registers, `0` stack, `0` spills, `96` bytes shared memory.
- Decode kernels: `38` registers, `0` stack, `0` spills.
- Remaining shared kernels: up to `44` registers, `0` stack, `0` spills.

Correctness and build checks:

- `make -B test-rmsnorm` passed.
- `make -B rmsnorm-hidden-fused-bench` passed.
- `make -B rmsnorm-bench` passed.
- `git diff --check` passed before this note was added.

## 2026-05-20 - Prefill GEMM stability rerun and dispatch manifests

Runtime files:

- `src/experiments/gemma4_prefill_tune.py`
- `build/experiments/gemma4_prefill_tune/candidate_deep_dispatch.csv`
- `build/experiments/gemma4_prefill_tune/all_shapes_dispatch.csv`
- `build/experiments/gemma4_prefill_tune/tuna_stability_globalk_slidingo.csv`
- `build/experiments/gemma4_prefill_tune/sgemm_bf16_stability_globalk_slidingo.csv`
- `build/experiments/gemma4_prefill_tune/stability_globalk_slidingo_dispatch.csv`
- `build/experiments/gemma4_prefill_tune/stability_globalk_slidingo_dispatch_threshold_1p25.csv`
- `build/experiments/gemma4_prefill_tune/tuna_cublas_algo_probe_global_k_m512.txt`
- `build/experiments/gemma4_prefill_tune/tuna_cublas_algo_probe_sliding_o_m256.txt`

Reason:

- The previous deep sweep showed suspicious or marginal custom wins for `global_k` at
  `M=512` and `sliding_o` at `M=256`.
- The tuner summary was useful interactively, but did not write the chosen
  custom-vs-cuBLAS route as a machine-readable artifact.

Implementation:

- Added `--dispatch-out` to the tuner `summarize` mode. It writes one CSV row per
  `(backend, op, M)` with route, config, chosen time, cuBLAS time, custom time, speedup,
  and layer-count weight.
- Fixed the combined summary printout to rank and display best configs per backend and
  op, instead of printing backend-ambiguous op summaries.
- Added `GEMMA4_PREFILL_CUBLAS_ALGO` support to both BF16 prefill benchmark binaries.
  Accepted values are `default`, `default_tensor`, and `algo0` through `algo15`, mapping
  to the corresponding `cublasGemmEx` algorithms. The default remains
  `CUBLAS_GEMM_DEFAULT_TENSOR_OP`.

Validation:

```bash
python3 -m py_compile src/experiments/gemma4_prefill_tune.py
rm -rf src/experiments/__pycache__

python3 src/experiments/gemma4_prefill_tune.py summarize \
  build/experiments/gemma4_prefill_tune/tuna_candidate_deep.csv \
  build/experiments/gemma4_prefill_tune/sgemm_bf16_candidate_deep.csv \
  --custom-threshold 1.0 \
  --dispatch-out build/experiments/gemma4_prefill_tune/candidate_deep_dispatch.csv

python3 src/experiments/gemma4_prefill_tune.py summarize \
  build/experiments/gemma4_prefill_tune/tuna_all_shapes.csv \
  build/experiments/gemma4_prefill_tune/sgemm_bf16_all_shapes.csv \
  --custom-threshold 1.0 \
  --dispatch-out build/experiments/gemma4_prefill_tune/all_shapes_dispatch.csv

make tuna-prefill-bench sgemm-bf16-prefill-bench
GEMMA4_PREFILL_CUBLAS_ALGO=algo7 \
  ./build/experiments/gemma4_sgemm_bf16_prefill_bench global_k 5 2 512 bf16_16x64
```

Focused stability commands:

```bash
python3 src/experiments/gemma4_prefill_tune.py \
  --backend tuna --ops global_k,sliding_o \
  --configs wmma_16x16,wmma_16x32,wmma_16x64,wmma_32x64,wmma_64x64 \
  --m 256,512 --iters 100 --warmup 25 \
  --out build/experiments/gemma4_prefill_tune/tuna_stability_globalk_slidingo.csv

python3 src/experiments/gemma4_prefill_tune.py \
  --backend sgemm-bf16 --ops global_k,sliding_o \
  --configs bf16_16x32,bf16_16x64,bf16_32x64,bf16_64x64,bf16_64x128,bf16_128x64 \
  --m 256,512 --iters 100 --warmup 25 \
  --out build/experiments/gemma4_prefill_tune/sgemm_bf16_stability_globalk_slidingo.csv

python3 src/experiments/gemma4_prefill_tune.py summarize \
  build/experiments/gemma4_prefill_tune/tuna_stability_globalk_slidingo.csv \
  build/experiments/gemma4_prefill_tune/sgemm_bf16_stability_globalk_slidingo.csv \
  --custom-threshold 1.25 \
  --dispatch-out build/experiments/gemma4_prefill_tune/stability_globalk_slidingo_dispatch_threshold_1p25.csv
```

Focused stability results:

| Backend | Op | Best config | Geomean speedup | Worst speedup |
| --- | --- | --- | ---: | ---: |
| Tuna | `global_k` | `wmma_16x16` | 0.608x | 0.182x |
| Tuna | `sliding_o` | `wmma_64x64` | 0.350x | 0.120x |
| SGEMM BF16 | `global_k` | `bf16_16x64` | 0.629x | 0.176x |
| SGEMM BF16 | `sliding_o` | `bf16_64x128` | 0.382x | 0.125x |

Conservative dispatch with threshold `1.25`:

| Backend | Dispatch weighted ms | cuBLAS weighted ms | Custom-only weighted ms | Dispatch vs cuBLAS | Custom-only vs cuBLAS |
| --- | ---: | ---: | ---: | ---: | ---: |
| Tuna | 85.9960 | 92.8600 | 198.7840 | 1.0798x | 0.4671x |
| SGEMM BF16 | 92.3180 | 100.4880 | 190.1170 | 1.0885x | 0.5286x |

Direct repeat checks:

```bash
./build/experiments/gemma4_tuna_prefill_bench global_k 200 50 512 wmma_16x64
./build/experiments/gemma4_tuna_prefill_bench global_k 200 50 512 wmma_16x64
./build/experiments/gemma4_tuna_prefill_bench global_k 200 50 256 wmma_16x64
./build/experiments/gemma4_tuna_prefill_bench sliding_o 200 50 256 wmma_64x64
./build/experiments/gemma4_tuna_prefill_bench sliding_o 200 50 512 wmma_64x64
```

Observed direct checks:

| Op | M | Config | Custom ms | cuBLAS ms | Speedup |
| --- | ---: | --- | ---: | ---: | ---: |
| `global_k` | 512 | `wmma_16x64` | 0.6590 | 3.1496 | 4.779x |
| `global_k` | 512 | `wmma_16x64` | 0.6846 | 1.7154 | 2.506x |
| `global_k` | 256 | `wmma_16x64` | 0.3422 | 0.0611 | 0.179x |
| `sliding_o` | 256 | `wmma_64x64` | 1.2435 | 1.8639 | 1.499x |
| `sliding_o` | 512 | `wmma_64x64` | 2.5940 | 0.3484 | 0.134x |

Interpretation:

- The direct repeats confirmed that cuBLAS has shape-specific timing cliffs in this
  harness: `global_k M=256` and `sliding_o M=512` are fast, while adjacent larger or
  smaller shapes can be much slower.
- These results justify keeping the custom kernels as measured candidates for narrow
  small-M or cuBLAS-cliff cases, but not as broad prefill GEMM replacements.
- `sliding_o M=256` is too marginal under the conservative `1.25` threshold and should
  stay on cuBLAS unless repeated on a production profiling setup or against a tuned
  cuBLASLt baseline.
- `global_k M=512` remains a candidate against this `cublasGemmEx` baseline, but it is
  suspicious enough that it should not be hard-coded without checking cuBLASLt or
  explicit cuBLAS algorithm selection.

Explicit cuBLAS algorithm probe:

```bash
for algo in default_tensor algo0 algo1 algo2 algo3 algo4 algo5 algo6 algo7 \
    algo8 algo9 algo10 algo11 algo12 algo13 algo14 algo15; do
  GEMMA4_PREFILL_CUBLAS_ALGO=$algo \
    ./build/experiments/gemma4_tuna_prefill_bench global_k 50 15 512 wmma_16x64 |
    awk -v algo=$algo '/global_k/{print algo ",global_k," $0} /wmma_16x64/{print algo ",global_k," $0}'
done > build/experiments/gemma4_prefill_tune/tuna_cublas_algo_probe_global_k_m512.txt

for algo in default_tensor algo0 algo1 algo2 algo3 algo4 algo5 algo6 algo7 \
    algo8 algo9 algo10 algo11 algo12 algo13 algo14 algo15; do
  GEMMA4_PREFILL_CUBLAS_ALGO=$algo \
    ./build/experiments/gemma4_tuna_prefill_bench sliding_o 50 15 256 wmma_64x64 |
    awk -v algo=$algo '/sliding_o/{print algo ",sliding_o," $0} /wmma_64x64/{print algo ",sliding_o," $0}'
done > build/experiments/gemma4_prefill_tune/tuna_cublas_algo_probe_sliding_o_m256.txt
```

Best observed explicit-algorithm checks:

| Op | M | Best cuBLAS algorithm | cuBLAS ms | Custom config | Custom ms | Speedup |
| --- | ---: | --- | ---: | --- | ---: | ---: |
| `global_k` | 512 | `algo7` | 1.3571 | `wmma_16x64` | 0.6461 | 2.100x |
| `sliding_o` | 256 | `algo14` | 1.5024 | `wmma_64x64` | 1.2408 | 1.211x |

Conclusion after the algorithm probe:

- The explicit `cublasGemmEx` tensor-op algorithms did not remove the timing cliff for
  these two cases.
- `global_k M=512` is still a plausible custom dispatch candidate against cuBLAS GEMMEx.
- `sliding_o M=256` is still a weak candidate because the best explicit cuBLAS algorithm
  narrows the margin to roughly `1.21x`; keep the conservative `1.25x` route on cuBLAS
  until cuBLASLt is tested.

## 2026-05-20 - Prefill GEMM cuBLASLt baseline

Runtime files:

- `experiments/tuna/gemma4_prefill_bench.cu`
- `experiments/sgemm.cu/gemma4_bf16_prefill_bench.cu`
- `src/experiments/gemma4_prefill_tune.py`
- `build/experiments/gemma4_prefill_tune/tuna_candidate_cublaslt.csv`
- `build/experiments/gemma4_prefill_tune/sgemm_bf16_candidate_cublaslt.csv`
- `build/experiments/gemma4_prefill_tune/candidate_cublaslt_dispatch.csv`
- `build/experiments/gemma4_prefill_tune/tuna_all_shapes_cublaslt.csv`
- `build/experiments/gemma4_prefill_tune/sgemm_bf16_all_shapes_cublaslt.csv`
- `build/experiments/gemma4_prefill_tune/all_shapes_cublaslt_dispatch.csv`

Reason:

- Earlier custom wins were measured against `cublasGemmEx` with
  `CUBLAS_GEMM_DEFAULT_TENSOR_OP`.
- Direct explicit-algorithm probing showed that GEMMEx has timing cliffs on the remaining
  candidate shapes, so cuBLASLt needed to be checked before any production dispatch rule.

Implementation:

- Added `GEMMA4_PREFILL_CUBLAS_BACKEND=gemmex|lt` to both BF16 prefill benchmark
  binaries. The default remains `gemmex`.
- The `lt` path uses row-major cuBLASLt descriptors for the actual Gemma layout:
  `Y[M,N] = X[M,K] * W[N,K]^T`, with BF16 inputs, FP32 compute, BF16 output, and a
  `32 MiB` workspace preference.
- cuBLASLt descriptor creation and heuristic selection happen outside the timed loop.
  Timed work is only warmup plus repeated `cublasLtMatmul` calls on the same benchmark
  path.
- The tuner now accepts `--cublas-backend gemmex|lt` and records `cublas_backend` and
  `cublas_algo` columns in new CSVs and dispatch manifests.

Validation:

```bash
make tuna-prefill-bench sgemm-bf16-prefill-bench

GEMMA4_PREFILL_CUBLAS_BACKEND=lt \
  ./build/experiments/gemma4_tuna_prefill_bench global_k 5 2 16 wmma_16x32

GEMMA4_PREFILL_CUBLAS_BACKEND=lt \
  ./build/experiments/gemma4_sgemm_bf16_prefill_bench global_k 5 2 16 bf16_16x32

python3 -m py_compile src/experiments/gemma4_prefill_tune.py
rm -rf src/experiments/__pycache__
```

Smoke results:

| Backend | Op | M | Library | Library ms | Custom config | Custom ms | Max abs |
| --- | --- | ---: | --- | ---: | --- | ---: | ---: |
| Tuna | `global_k` | 16 | cuBLASLt | 0.0458 | `wmma_16x32` | 0.0717 | 0.015625 |
| SGEMM BF16 | `global_k` | 16 | cuBLASLt | 0.0455 | `bf16_16x32` | 0.0717 | 0.015625 |

Candidate cuBLASLt sweeps:

```bash
python3 src/experiments/gemma4_prefill_tune.py \
  --backend tuna --ops ffn_down,global_k,sliding_o \
  --configs wmma_16x16,wmma_16x32,wmma_16x64,wmma_32x64,wmma_64x64,smem_16x64,smem_16x128,smem_32x64,smem_32x128,smem_64x64 \
  --m 16,32,64,128,256,512 --iters 20 --warmup 5 \
  --cublas-backend lt \
  --out build/experiments/gemma4_prefill_tune/tuna_candidate_cublaslt.csv

python3 src/experiments/gemma4_prefill_tune.py \
  --backend sgemm-bf16 --ops ffn_down,global_k,sliding_o \
  --configs bf16_16x32,bf16_16x64,bf16_32x64,bf16_64x64,bf16_64x128,bf16_128x64 \
  --m 16,32,64,128,256,512 --iters 20 --warmup 5 \
  --cublas-backend lt \
  --out build/experiments/gemma4_prefill_tune/sgemm_bf16_candidate_cublaslt.csv
```

Candidate cuBLASLt summary:

| Backend | Op | Best config | Geomean speedup vs cuBLASLt | Worst speedup |
| --- | --- | --- | ---: | ---: |
| Tuna | `ffn_down` | `wmma_64x64` | 0.249x | 0.110x |
| Tuna | `global_k` | `wmma_16x16` | 0.348x | 0.188x |
| Tuna | `sliding_o` | `wmma_64x64` | 0.276x | 0.122x |
| SGEMM BF16 | `ffn_down` | `bf16_64x64` | 0.254x | 0.114x |
| SGEMM BF16 | `global_k` | `bf16_16x64` | 0.329x | 0.181x |
| SGEMM BF16 | `sliding_o` | `bf16_64x64` | 0.276x | 0.120x |

All-shape cuBLASLt sweeps:

```bash
python3 src/experiments/gemma4_prefill_tune.py \
  --backend tuna \
  --ops ffn_gate_up,ffn_down,sliding_qkv,sliding_o,global_q,global_k,global_o,final_logits \
  --configs wmma_16x32,wmma_32x64,wmma_64x64 \
  --m 16,64,256 --iters 10 --warmup 3 --cublas-backend lt \
  --out build/experiments/gemma4_prefill_tune/tuna_all_shapes_cublaslt.csv

python3 src/experiments/gemma4_prefill_tune.py \
  --backend sgemm-bf16 \
  --ops ffn_gate_up,ffn_down,sliding_qkv,sliding_o,global_q,global_k,global_o,final_logits \
  --configs bf16_16x32,bf16_32x64,bf16_64x64 \
  --m 16,64,256 --iters 10 --warmup 3 --cublas-backend lt \
  --out build/experiments/gemma4_prefill_tune/sgemm_bf16_all_shapes_cublaslt.csv

python3 src/experiments/gemma4_prefill_tune.py summarize \
  build/experiments/gemma4_prefill_tune/tuna_all_shapes_cublaslt.csv \
  build/experiments/gemma4_prefill_tune/sgemm_bf16_all_shapes_cublaslt.csv \
  --custom-threshold 1.0 \
  --dispatch-out build/experiments/gemma4_prefill_tune/all_shapes_cublaslt_dispatch.csv
```

All-shape cuBLASLt dispatch:

| Backend | Dispatch weighted ms | cuBLASLt weighted ms | Custom-only weighted ms | Dispatch vs cuBLASLt | Custom-only vs cuBLASLt |
| --- | ---: | ---: | ---: | ---: | ---: |
| Tuna | 306.8089 | 306.8089 | 1238.9731 | 1.0000x | 0.2476x |
| SGEMM BF16 | 305.9398 | 305.9398 | 1236.2197 | 1.0000x | 0.2475x |

Conclusion:

- cuBLASLt removes the GEMMEx timing cliffs that made the simple custom WMMA kernels look
  useful for small-M `global_k`, `ffn_down`, and `sliding_o`.
- After checking all Gemma 4 projection shapes over `M=16,64,256`, every dispatch route
  goes to cuBLASLt. The custom-only path is roughly `4x` slower in the weighted all-shape
  sweep.
- Do not promote the current Tuna or SGEMM BF16 adaptations into the inference path for
  prefill GEMM. The useful artifact is the shape-aware benchmark/tuner plus the negative
  result: cuBLASLt is the baseline to use for the unfused prefill path.
- Further custom prefill GEMM work should only resume with a substantially different
  design, such as a real CUTLASS/Tuna-style pipelined tensor-core kernel with multiple
  MMA tiles per warp and async staging. The current one-warp-per-WMMA-tile variants have
  no remaining low-hanging parameter-tuning path against cuBLASLt.

## 2026-05-20 - SGEMM BF16 ffn_down production-shape tuning

Goal: tune only the SGEMM BF16 prefill GEMM path as hard as practical for one Gemma 4
production shape while keeping it valid across a token-count sweep. The chosen shape was
`ffn_down`, because Gemma 4 31B has 60 dense FFN down projections with
`intermediate_size=21504` and `hidden_size=5376`:

```text
Y[M,5376] = X[M,21504] * W[5376,21504]^T
```

Architecture basis:

- `gemma4_architecture.md`: Gemma 4 31B dense text config has `hidden_size=5376`,
  `intermediate_size=21504`, and `num_hidden_layers=60`.
- `src/gemma4.h`: `GEMMA4_INTERMEDIATE_SIZE=21504`,
  `GEMMA4_HIDDEN_SIZE=5376`, and `GEMMA4_PACKED_FFN_SIZE=43008`.

Files changed:

- `experiments/sgemm.cu/gemma4_bf16_prefill_bench.cu`
  - Added a broader SGEMM BF16 config space:
    `bf16_16x16`, `bf16_16x32`, `bf16_16x64`, `bf16_16x128`,
    `bf16_32x32`, `bf16_32x64`, `bf16_32x128`, `bf16_64x64`,
    `bf16_64x128`, `bf16_128x64`.
  - Added staged shared-memory variants: `bf16_smem64_32x64`,
    `bf16_smem64_64x64`.
  - Added wide-warp variants that reuse one A fragment across multiple N tiles:
    `bf16_warp_16x32`, `bf16_warp_16x64`, `bf16_warp_16x128`.
  - Added split-K variants for `ffn_down`: `bf16_splitk4_16x32`,
    `bf16_splitk4_32x64`.
- `src/experiments/gemma4_prefill_tune.py`
  - Added the new SGEMM BF16 configs to the tuner.
  - Fixed duplicated CSV field names for `cublas_backend` and `cublas_algo`.

CUDA guide check:

- Queried the local CUDA Programming Guide KB for WMMA BF16, shared-memory tiling,
  occupancy, and launch bounds. Relevant guide areas were kernel occupancy, shared-memory
  access patterns, and launch bounds. This matched the tuning directions tried here:
  tile geometry, shared-memory staging, register/occupancy pressure, and launch-bound
  constraints.

Commands:

```bash
make sgemm-bf16-prefill-bench

GEMMA4_PREFILL_CUBLAS_BACKEND=lt \
  ./build/experiments/gemma4_sgemm_bf16_prefill_bench \
  ffn_down 10 3 16,64 bf16_smem64_64x64

GEMMA4_PREFILL_CUBLAS_BACKEND=lt \
  ./build/experiments/gemma4_sgemm_bf16_prefill_bench \
  ffn_down 10 3 16,64 bf16_warp_16x64

GEMMA4_PREFILL_CUBLAS_BACKEND=lt \
  ./build/experiments/gemma4_sgemm_bf16_prefill_bench \
  ffn_down 10 3 16,64 bf16_splitk4_32x64

python3 src/experiments/gemma4_prefill_tune.py \
  --backend sgemm-bf16 --ops ffn_down \
  --configs bf16_16x32,bf16_16x64,bf16_32x64,bf16_64x64,bf16_128x64,bf16_smem64_32x64,bf16_smem64_64x64,bf16_warp_16x64,bf16_splitk4_32x64 \
  --m 8,16,32,64,128,256 --iters 10 --warmup 3 \
  --cublas-backend lt \
  --out build/experiments/gemma4_prefill_tune/sgemm_bf16_ffn_down_selected_arch.csv

python3 src/experiments/gemma4_prefill_tune.py \
  --backend sgemm-bf16 --ops ffn_down --configs bf16_32x64 \
  --m 8,16,32,64,128,256 --iters 100 --warmup 25 \
  --cublas-backend lt \
  --out build/experiments/gemma4_prefill_tune/sgemm_bf16_ffn_down_best_stability.csv

python3 src/experiments/gemma4_prefill_tune.py \
  --backend sgemm-bf16 --ops ffn_down \
  --configs bf16_16x16,bf16_16x32,bf16_16x64,bf16_16x128,bf16_32x32,bf16_32x64,bf16_32x128,bf16_64x64,bf16_64x128,bf16_128x64,bf16_smem64_32x64,bf16_smem64_64x64,bf16_warp_16x32,bf16_warp_16x64,bf16_warp_16x128,bf16_splitk4_16x32,bf16_splitk4_32x64 \
  --m 8,16,32,64,128,256 --iters 10 --warmup 3 \
  --cublas-backend lt \
  --out build/experiments/gemma4_prefill_tune/sgemm_bf16_ffn_down_extreme.csv

python3 src/experiments/gemma4_prefill_tune.py summarize \
  build/experiments/gemma4_prefill_tune/sgemm_bf16_ffn_down_extreme.csv \
  --custom-threshold 1.0 \
  --dispatch-out build/experiments/gemma4_prefill_tune/sgemm_bf16_ffn_down_extreme_dispatch.csv
```

Tried and rejected:

- Shared-memory K-stage 64 (`bf16_smem64_*`): correct, but slower than direct WMMA
  loads and slower than cuBLASLt.
- Wide-warp N reuse (`bf16_warp_*`): correct, but the extra long-lived accumulator
  fragments reduced useful parallelism and did not beat the simpler tilings.
- Split-K=4 (`bf16_splitk4_*`): correct, but the partial write plus reduction overhead
  dominated for this shape.
- `__launch_bounds__(..., 2)` on the plain WMMA kernels: produced ptxas warnings for
  1024-thread block variants and did not improve the sweep; reverted to
  `__launch_bounds__(..., 1)`.
- Nsight Compute command:

```bash
ncu --metric gpu__time_duration.sum --target-processes all \
  ./build/experiments/gemma4_sgemm_bf16_prefill_bench ffn_down 1 0 32 bf16_32x64
```

This failed in the Thunder runtime with an unsupported-library assertion, so the recorded
numbers below use CUDA event timing after warmup.

Selected-architecture sweep, `M=8,16,32,64,128,256`, `iters=10`, `warmup=3`:

| M | Best custom config | Custom ms | cuBLASLt ms | Speedup vs cuBLASLt | Max abs |
| ---: | --- | ---: | ---: | ---: | ---: |
| 8 | `bf16_16x64` | 0.3856 | 0.3388 | 0.879x | 0.03125 |
| 16 | `bf16_16x32` | 0.3868 | 0.3420 | 0.884x | 0.03125 |
| 32 | `bf16_16x64` | 0.4135 | 0.3421 | 0.827x | 0.03125 |
| 64 | `bf16_16x64` | 0.7776 | 0.3549 | 0.456x | 0.03125 |
| 128 | `bf16_128x64` | 1.5976 | 0.3768 | 0.236x | 0.0625 |
| 256 | `bf16_128x64` | 3.1981 | 0.4160 | 0.130x | 0.03125 |

Full extreme sweep, `M=8,16,32,64,128,256`, `iters=10`, `warmup=3`:

| M | Best custom config | Custom ms | cuBLASLt ms | Speedup vs cuBLASLt | Custom TFLOP/s | Max abs |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 8 | `bf16_16x32` | 0.3852 | 0.3383 | 0.878x | 4.80 | 0.03125 |
| 16 | `bf16_16x32` | 0.3853 | 0.3407 | 0.884x | 9.60 | 0.03125 |
| 32 | `bf16_32x32` | 0.4119 | 0.3407 | 0.827x | 17.96 | 0.03125 |
| 64 | `bf16_16x64` | 0.7778 | 0.3542 | 0.455x | 19.03 | 0.03125 |
| 128 | `bf16_128x64` | 1.4910 | 0.3753 | 0.252x | 19.85 | 0.0625 |
| 256 | `bf16_128x64` | 2.9172 | 0.4127 | 0.141x | 20.29 | 0.03125 |

Best full-sweep config by geomean:

| Config | Geomean speedup vs cuBLASLt | Worst speedup |
| --- | ---: | ---: |
| `bf16_32x64` | 0.344x | 0.070x |
| `bf16_32x32` | 0.326x | 0.046x |
| `bf16_64x64` | 0.318x | 0.126x |
| `bf16_32x128` | 0.308x | 0.104x |
| `bf16_16x128` | 0.296x | 0.035x |
| `bf16_16x64` | 0.286x | 0.029x |
| `bf16_splitk4_32x64` | 0.255x | 0.069x |
| `bf16_128x64` | 0.208x | 0.141x |

Stability pass for `bf16_32x64`, `M=8,16,32,64,128,256`, `iters=100`,
`warmup=25`:

| M | Custom ms | cuBLASLt ms | Speedup vs cuBLASLt | Custom TFLOP/s | Max abs |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 0.4123 | 0.3376 | 0.819x | 4.49 | 0.03125 |
| 16 | 0.4087 | 0.3400 | 0.832x | 9.05 | 0.03125 |
| 32 | 0.4028 | 0.3398 | 0.844x | 18.37 | 0.03125 |
| 64 | 1.3972 | 0.3536 | 0.253x | 10.59 | 0.03125 |
| 128 | 2.1470 | 0.3769 | 0.176x | 13.78 | 0.0625 |
| 256 | 5.7529 | 0.4681 | 0.081x | 10.29 | 0.03125 |

Dispatch result from the full extreme sweep:

- All `M` values route to cuBLASLt when the custom threshold is `1.0`.
- Weighted dispatch ms: `129.7140`.
- Weighted cuBLASLt ms: `129.7140`.
- Weighted custom-only ms: `382.1040`.
- Custom-only vs cuBLASLt: `0.3395x`.

Conclusion:

- The SGEMM BF16 kernels were tuned heavily for the production `ffn_down` shape and
  validated over a variety of token counts (`M=8,16,32,64,128,256`).
- Correctness against cuBLASLt stayed within BF16-level tolerances for the tested random
  inputs (`max_abs` generally `0.03125`, worst observed `0.0625`).
- The best custom kernels are only close for `M<=32`, where they still lose by roughly
  12-17%. For larger M, cuBLASLt is much faster.
- No variant in this one-warp-per-WMMA/direct-or-lightly-staged kernel family should be
  promoted into the production prefill path. Keep cuBLASLt as the production baseline
  for this shape. Further custom GEMM work should start from a substantially different
  architecture, such as a CUTLASS-style pipelined CTA with deeper K staging and a better
  accumulator/store strategy.

## 2026-05-20 - SGEMM BF16 dynamic ffn_down CUTLASS route

Follow-up goal: make the custom SGEMM BF16 path beat cuBLAS for the Gemma 4 `ffn_down`
shape with dynamic `M`.

Shape:

```text
Y[M,5376] = X[M,21504] * W[5376,21504]^T
```

Changes:

- Added CUTLASS-backed BF16 SGEMM configs to
  `experiments/sgemm.cu/gemma4_bf16_prefill_bench.cu`.
- Added `bf16_auto_ffn_down`, a dynamic route for this shape:
  - `M <= 128`: CUTLASS `64x128x64`, warp `32x64x64`, 3 stages.
  - `M > 128`: CUTLASS `128x128x64`, warp `64x64x64`, 3 stages.
- Kept the previous hand-WMMA variants in the benchmark for comparison.
- Added correctness fail-fast at `max_abs > 0.125` so invalid fast configs cannot be
  accidentally reported as wins.
- Added `CUTLASS_INCLUDE ?= /tmp/cutlass/include` to the benchmark build rule.

Important rejected configs:

- `bf16_smemA*` A-only staging variants: correct, but slower than direct WMMA and
  slower than CUTLASS.
- Very wide direct WMMA configs such as `bf16_16x256`, `bf16_32x256`, and
  `bf16_256x32`: correct, but slower.
- CUTLASS `256x*` and `128x128x64_s4`: produced invalid results (`max_abs` around
  `7`), so they were removed and the benchmark now fails correctness instead of timing
  such configs.

Commands:

```bash
python3 src/experiments/gemma4_prefill_tune.py \
  --backend sgemm-bf16 --ops ffn_down \
  --configs bf16_auto_ffn_down \
  --m 8,16,32,64,96,128,256 --iters 100 --warmup 20 \
  --out build/experiments/gemma4_prefill_tune/sgemm_bf16_ffn_down_auto_gemmex_win.csv

python3 src/experiments/gemma4_prefill_tune.py \
  --backend sgemm-bf16 --ops ffn_down \
  --configs bf16_auto_ffn_down,bf16_cutlass_64x128x64,bf16_cutlass_128x128x64,bf16_cutlass_64x64 \
  --m 8,16,32,64,96,128,256 --iters 50 --warmup 10 \
  --cublas-backend lt \
  --out build/experiments/gemma4_prefill_tune/sgemm_bf16_ffn_down_cutlass_cublaslt.csv

python3 src/experiments/gemma4_prefill_tune.py summarize \
  build/experiments/gemma4_prefill_tune/sgemm_bf16_ffn_down_auto_gemmex_win.csv \
  build/experiments/gemma4_prefill_tune/sgemm_bf16_ffn_down_cutlass_cublaslt.csv \
  --custom-threshold 1.05 \
  --dispatch-out build/experiments/gemma4_prefill_tune/sgemm_bf16_ffn_down_goal_dispatch.csv
```

Dynamic route versus default cuBLAS `cublasGemmEx`:

| M | cuBLAS ms | Custom ms | Speedup |
| ---: | ---: | ---: | ---: |
| 8 | 3.6761 | 0.3504 | 10.491x |
| 16 | 4.0123 | 0.3506 | 11.444x |
| 32 | 2.8335 | 0.3527 | 8.034x |
| 64 | 3.2531 | 0.3541 | 9.187x |
| 96 | 2.9703 | 0.3570 | 8.320x |
| 128 | 3.8941 | 0.3634 | 10.716x |
| 256 | 3.8283 | 0.4881 | 7.843x |

Summary: `bf16_auto_ffn_down` beats default cuBLAS `cublasGemmEx` by more than 5% for
every tested dynamic M value. Geomean speedup is `9.339x`; worst speedup is `7.843x`.

Harder cuBLASLt comparison:

| M | cuBLASLt ms | Best custom ms | Best config | Speedup |
| ---: | ---: | ---: | --- | ---: |
| 8 | 0.3374 | 0.3501 | `bf16_auto_ffn_down` | 0.964x |
| 16 | 0.3398 | 0.3511 | `bf16_auto_ffn_down` | 0.968x |
| 32 | 0.3394 | 0.3527 | `bf16_cutlass_64x128x64` | 0.962x |
| 64 | 0.3530 | 0.3547 | `bf16_cutlass_64x128x64` | 0.995x |
| 96 | 0.3718 | 0.3564 | `bf16_cutlass_64x128x64` | 1.043x |
| 128 | 0.3779 | 0.3605 | `bf16_cutlass_64x128x64` | 1.048x |
| 256 | 0.4166 | 0.4264 | `bf16_auto_ffn_down` | 0.977x |

Summary: the new CUTLASS route nearly matches cuBLASLt overall (`0.993x` geomean), and
beats cuBLASLt for `M=96` and `M=128`, but it does not meet the 5% win threshold against
cuBLASLt. The achieved 5%+ result is specifically against default cuBLAS `cublasGemmEx`.

### Correction: cuBLAS graph timing with enqueue overhead factored out

The default `cublasGemmEx` result above was not a valid production comparison: the timing
loop could include host enqueue gaps between repeated cuBLAS calls. The benchmark now
supports `GEMMA4_PREFILL_GRAPH_REPEATS`; when set above `1`, cuBLAS/cuBLASLt is captured
on a nonblocking CUDA stream with repeated GEMMs inside one CUDA graph. The reported
cuBLAS time is divided by the number of captured GEMMs, so one-time setup and per-call
host enqueue overhead are factored out of the per-GEMM result.

Code changes:

- `time_cuda_graph_stream` captures repeated cuBLAS calls into one CUDA graph on an
  explicit nonblocking stream.
- cuBLASLt timing now passes the same stream into `cublasLtMatmul`.
- CUTLASS/custom timing remains the direct CUDA-event path, which is conservative for the
  custom side in this comparison.

Commands:

```bash
GEMMA4_PREFILL_GRAPH_REPEATS=16 \
  python3 src/experiments/gemma4_prefill_tune.py \
  --backend sgemm-bf16 --ops ffn_down \
  --configs bf16_auto_ffn_down,bf16_cutlass_64x128x64,bf16_cutlass_128x128x64,bf16_cutlass_64x64 \
  --m 8,16,32,64,96,128,256 --iters 50 --warmup 10 \
  --out build/experiments/gemma4_prefill_tune/sgemm_bf16_ffn_down_graph_gemmex.csv

GEMMA4_PREFILL_GRAPH_REPEATS=16 \
  python3 src/experiments/gemma4_prefill_tune.py \
  --backend sgemm-bf16 --ops ffn_down \
  --configs bf16_auto_ffn_down,bf16_cutlass_64x128x64,bf16_cutlass_128x128x64,bf16_cutlass_64x64 \
  --m 8,16,32,64,96,128,256 --iters 50 --warmup 10 \
  --cublas-backend lt \
  --out build/experiments/gemma4_prefill_tune/sgemm_bf16_ffn_down_graph_cublaslt.csv

python3 src/experiments/gemma4_prefill_tune.py summarize \
  build/experiments/gemma4_prefill_tune/sgemm_bf16_ffn_down_graph_gemmex.csv \
  build/experiments/gemma4_prefill_tune/sgemm_bf16_ffn_down_graph_cublaslt.csv \
  --custom-threshold 1.05 \
  --dispatch-out build/experiments/gemma4_prefill_tune/sgemm_bf16_ffn_down_graph_dispatch.csv
```

Graph-timed default cuBLAS `cublasGemmEx` comparison:

| M | Best config | cuBLAS ms | Custom ms | Speedup |
| ---: | --- | ---: | ---: | ---: |
| 8 | `bf16_auto_ffn_down` | 0.3349 | 0.3489 | 0.960x |
| 16 | `bf16_cutlass_64x128x64` | 0.3378 | 0.3489 | 0.968x |
| 32 | `bf16_cutlass_64x128x64` | 0.3383 | 0.3494 | 0.968x |
| 64 | `bf16_cutlass_64x128x64` | 0.3516 | 0.3533 | 0.995x |
| 96 | `bf16_cutlass_64x128x64` | 0.3716 | 0.3565 | 1.042x |
| 128 | `bf16_cutlass_64x128x64` | 0.3777 | 0.3596 | 1.050x |
| 256 | `bf16_cutlass_128x128x64` | 0.4899 | 0.5198 | 0.942x |

Summary: after factoring out cuBLAS host/setup overhead, the custom route no longer
achieves a 5% dynamic-M win. It only reaches the threshold at `M=128`; geomean speedup
versus graph-timed default cuBLAS is `0.974x`, and worst speedup is `0.873x`.

Graph-timed cuBLASLt comparison:

- Best config: `bf16_auto_ffn_down`.
- Geomean speedup: `0.988x`.
- Worst speedup: `0.946x`.
- No `M` routes to custom at a `1.05x` threshold.

Corrected conclusion: the previous default-cuBLAS win was an overhead artifact. With
cuBLAS overhead factored out, this SGEMM BF16 path still does not beat cuBLAS/cuBLASLt by
5% across dynamic M.

## 2026-05-20 - cuBLASLt multi-heuristic and Stream-K ffn_down check

Follow-up reason:

- The earlier cuBLASLt baseline used only the first heuristic returned by
  `cublasLtMatmulAlgoGetHeuristic`.
- For a fair "fastest CUDA library path" comparison, the benchmark now asks cuBLASLt for
  multiple heuristics, graph-times each returned successful algorithm, and reports the
  fastest measured candidate.
- CUTLASS Stream-K was added as a custom architecture candidate for small-`M`, large
  `K,N` GEMMs where ordinary CTA scheduling can underfeed SMs.

Code changes:

- Added `GEMMA4_PREFILL_CUBLASLT_HEURISTICS`, defaulting to `1`, capped at `64`.
- `CublasLtPrefillPlan` now stores all successful heuristic candidates returned by
  cuBLASLt.
- cuBLASLt timing loops over those candidates, using the existing graph-timed stream path
  when `GEMMA4_PREFILL_GRAPH_REPEATS > 1`, and reports the fastest candidate.
- Added Python sweep option `--cublaslt-heuristics`; CSV labels use `h<N>` for cuBLASLt
  heuristic-count baselines.
- Added Stream-K configs:
  `bf16_streamk_64x64x64`, `bf16_streamk_64x128x64`, and
  `bf16_streamk_128x128x64`.

Commands:

```bash
GEMMA4_PREFILL_GRAPH_REPEATS=16 \
  python3 src/experiments/gemma4_prefill_tune.py \
  --backend sgemm-bf16 --ops ffn_down \
  --configs bf16_auto_ffn_down,bf16_cutlass_64x128x64,bf16_cutlass_128x128x64,bf16_cutlass_64x64 \
  --m 8,16,32,64,96,128,256 --iters 30 --warmup 8 \
  --cublas-backend lt --cublaslt-heuristics 32 \
  --out build/experiments/gemma4_prefill_tune/sgemm_bf16_ffn_down_graph_cublaslt_h32.csv

GEMMA4_PREFILL_GRAPH_REPEATS=16 \
  python3 src/experiments/gemma4_prefill_tune.py \
  --backend sgemm-bf16 --ops ffn_down \
  --configs bf16_auto_ffn_down,bf16_cutlass_64x128x64,bf16_streamk_64x64x64,bf16_streamk_64x128x64,bf16_streamk_128x128x64 \
  --m 8,16,32,64,96,128,256 --iters 30 --warmup 8 \
  --cublas-backend lt --cublaslt-heuristics 32 \
  --out build/experiments/gemma4_prefill_tune/sgemm_bf16_ffn_down_streamk_cublaslt_h32.csv

GEMMA4_PREFILL_GRAPH_REPEATS=16 \
  python3 src/experiments/gemma4_prefill_tune.py \
  --backend sgemm-bf16 --ops ffn_down \
  --configs bf16_auto_ffn_down,bf16_streamk_64x64x64,bf16_streamk_64x128x64,bf16_streamk_128x128x64 \
  --m 512,1024 --iters 20 --warmup 5 \
  --cublas-backend lt --cublaslt-heuristics 32 \
  --out build/experiments/gemma4_prefill_tune/sgemm_bf16_ffn_down_streamk_large_m_cublaslt_h32.csv
```

Best custom versus graph-timed cuBLASLt h32 over `M=8..1024`:

| M | Best custom config | cuBLASLt h32 ms | Custom ms | Speedup |
| ---: | --- | ---: | ---: | ---: |
| 8 | `bf16_streamk_64x64x64` | 0.3299 | 0.3347 | 0.986x |
| 16 | `bf16_streamk_64x64x64` | 0.3304 | 0.3368 | 0.981x |
| 32 | `bf16_streamk_64x64x64` | 0.3327 | 0.3363 | 0.989x |
| 64 | `bf16_streamk_64x64x64` | 0.3415 | 0.3405 | 1.003x |
| 96 | `bf16_cutlass_64x128x64` | 0.3737 | 0.3598 | 1.039x |
| 128 | `bf16_streamk_128x128x64` | 0.3789 | 0.3650 | 1.038x |
| 256 | `bf16_auto_ffn_down` | 0.4982 | 0.5288 | 0.942x |
| 512 | `bf16_streamk_128x128x64` | 0.9061 | 0.9421 | 0.962x |
| 1024 | `bf16_streamk_128x128x64` | 1.8709 | 1.9429 | 0.963x |

Summary:

- Multi-heuristic cuBLASLt h32 is a stronger baseline than the previous single-heuristic
  cuBLASLt run. Without Stream-K, the custom geomean over `M=8..256` drops to `0.968x`.
- Stream-K helps the best custom choice for small `M`, especially `M=8..64`, but still
  does not reach the 1.05x custom-routing threshold.
- Over `M=8..1024`, no tested `M` routes to custom at a `1.05x` threshold.
- Weighted custom-only speedup over this `M` sweep is `0.977x` versus graph-timed
  cuBLASLt h32.

Conclusion: after comparing against a stronger, overhead-factored cuBLASLt baseline,
the current CUTLASS and Stream-K custom routes do not outperform cuBLASLt for Gemma 4
`ffn_down`. Further work needs a materially different custom architecture or a narrower
deployment target than "15-20% average over a large M range."

## 2026-05-20 - Custom CUDA graph timing and larger-M tile sweep

Follow-up reason:

- The previous correction graph-timed cuBLAS/cuBLASLt but left custom kernels in the
  direct launch loop. CUDA events can include GPU idle gaps between host launches when
  kernels are short.
- This pass adds stream-aware CUTLASS launches and graph-times custom CUTLASS/Stream-K
  routes too, so both sides can be compared with host-launch gaps factored out.
- It also checks whether the large-M losses were caused by `bf16_auto_ffn_down` choosing
  the wrong existing CUTLASS tile.

Code changes:

- CUTLASS and Stream-K launchers now accept an explicit `cudaStream_t`.
- `launch_bf16` forwards the stream to CUTLASS-backed configs.
- When `GEMMA4_PREFILL_GRAPH_REPEATS > 1`, custom CUTLASS-backed configs are captured
  and timed on a nonblocking CUDA stream, just like the cuBLASLt candidates.

Commands:

```bash
GEMMA4_PREFILL_GRAPH_REPEATS=16 \
  python3 src/experiments/gemma4_prefill_tune.py \
  --backend sgemm-bf16 --ops ffn_down \
  --configs bf16_auto_ffn_down,bf16_cutlass_64x128x64,bf16_streamk_64x64x64,bf16_streamk_64x128x64,bf16_streamk_128x128x64 \
  --m 8,16,32,64,96,128,256 --iters 30 --warmup 8 \
  --cublas-backend lt --cublaslt-heuristics 32 \
  --out build/experiments/gemma4_prefill_tune/sgemm_bf16_ffn_down_both_graph_streamk_cublaslt_h32.csv

GEMMA4_PREFILL_GRAPH_REPEATS=16 \
  python3 src/experiments/gemma4_prefill_tune.py \
  --backend sgemm-bf16 --ops ffn_down \
  --configs bf16_auto_ffn_down,bf16_cutlass_64x64,bf16_cutlass_64x128,bf16_cutlass_64x128x64,bf16_cutlass_128x64,bf16_cutlass_128x64x64,bf16_cutlass_128x128,bf16_cutlass_128x128x64,bf16_cutlass_128x256,bf16_streamk_64x64x64,bf16_streamk_64x128x64,bf16_streamk_128x128x64 \
  --m 256,512 --iters 20 --warmup 5 \
  --cublas-backend lt --cublaslt-heuristics 32 \
  --out build/experiments/gemma4_prefill_tune/sgemm_bf16_ffn_down_large_m_tiles_both_graph_cublaslt_h32.csv
```

Both-sides graph timing over `M=8..256`:

| M | Best custom config | cuBLASLt h32 ms | Custom ms | Speedup |
| ---: | --- | ---: | ---: | ---: |
| 8 | `bf16_streamk_64x64x64` | 0.3299 | 0.3330 | 0.991x |
| 16 | `bf16_streamk_64x64x64` | 0.3305 | 0.3337 | 0.990x |
| 32 | `bf16_streamk_64x64x64` | 0.3328 | 0.3355 | 0.992x |
| 64 | `bf16_streamk_64x64x64` | 0.3423 | 0.3389 | 1.010x |
| 96 | `bf16_cutlass_64x128x64` | 0.3747 | 0.3620 | 1.035x |
| 128 | `bf16_streamk_128x128x64` | 0.3806 | 0.3758 | 1.013x |
| 256 | `bf16_auto_ffn_down` | 0.5087 | 0.5470 | 0.930x |

Summary for `M=8..256`: best custom geomean is `0.962x`, worst is `0.930x`, and no
case reaches the `1.05x` custom routing threshold. Graph-timing the custom side does not
reveal a hidden win.

Broader existing-tile check at larger `M`:

| M | Best custom config | cuBLASLt h32 ms | Custom ms | Speedup |
| ---: | --- | ---: | ---: | ---: |
| 256 | `bf16_cutlass_128x128` | 0.4924 | 0.5482 | 0.898x |
| 512 | `bf16_cutlass_128x256` | 0.9189 | 0.9639 | 0.953x |

Conclusion: using CUDA graphs for both custom and cuBLASLt makes the comparison cleaner
but does not change the direction. The current CUTLASS tile inventory and Stream-K route
still do not beat cuBLASLt h32 for `ffn_down`, and a broader large-M tile sweep did not
find a better existing dispatch choice.

## 2026-05-20 - Stream-K split-factor ffn_down check

Follow-up reason:

- CUTLASS example 47 exposes Stream-K's `batch_split` argument as a tile-splitting factor
  for `GemmUniversalMode::kGemm`.
- The previous Stream-K configs only measured the default split factor `1`, so this pass
  checks whether Stream-K split-K emulation helps the small/medium `M` range where custom
  was closest to cuBLASLt.

Code changes:

- `launch_cutlass_streamk_gemm` now accepts a `split_k_factor` and passes it to
  `GemmUniversal::Arguments`.
- Added split-factor configs:
  `bf16_streamk_s2_64x64x64`, `bf16_streamk_s2_64x128x64`,
  `bf16_streamk_s2_128x128x64`, `bf16_streamk_s4_64x64x64`,
  `bf16_streamk_s4_64x128x64`, and `bf16_streamk_s4_128x128x64`.

Commands:

```bash
GEMMA4_PREFILL_GRAPH_REPEATS=16 \
  python3 src/experiments/gemma4_prefill_tune.py \
  --backend sgemm-bf16 --ops ffn_down \
  --configs bf16_streamk_64x64x64,bf16_streamk_s2_64x64x64,bf16_streamk_s4_64x64x64,bf16_streamk_64x128x64,bf16_streamk_s2_64x128x64,bf16_streamk_s4_64x128x64,bf16_streamk_128x128x64,bf16_streamk_s2_128x128x64,bf16_streamk_s4_128x128x64 \
  --m 64,96,128 --iters 20 --warmup 5 \
  --cublas-backend lt --cublaslt-heuristics 32 \
  --out build/experiments/gemma4_prefill_tune/sgemm_bf16_ffn_down_streamk_split_both_graph_cublaslt_h32.csv

python3 src/experiments/gemma4_prefill_tune.py summarize \
  build/experiments/gemma4_prefill_tune/sgemm_bf16_ffn_down_both_graph_streamk_cublaslt_h32.csv \
  build/experiments/gemma4_prefill_tune/sgemm_bf16_ffn_down_streamk_split_both_graph_cublaslt_h32.csv \
  --custom-threshold 1.05 \
  --dispatch-out build/experiments/gemma4_prefill_tune/sgemm_bf16_ffn_down_both_graph_with_split_cublaslt_h32_dispatch.csv
```

Targeted split-factor sweep:

| M | Best config | cuBLASLt h32 ms | Custom ms | Speedup |
| ---: | --- | ---: | ---: | ---: |
| 64 | `bf16_streamk_64x64x64` | 0.3416 | 0.3381 | 1.010x |
| 96 | `bf16_streamk_s2_128x128x64` | 0.3721 | 0.3602 | 1.033x |
| 128 | `bf16_streamk_s2_128x128x64` | 0.3760 | 0.3611 | 1.041x |

Combined dynamic route over `M=8..256` after adding split-factor configs:

- No `M` routes to custom at the `1.05x` threshold.
- Weighted custom-only speedup is `0.9935x` versus graph-timed cuBLASLt h32.
- The best split-factor config helps `M=96` and `M=128`, but not enough for the requested
  15-20% average win, and split factors are much worse at larger `M`.

Conclusion: Stream-K split-K emulation is not the missing architecture lever for this
`ffn_down` sweep. It narrows the gap in a small band but does not outperform cuBLASLt
over a large dynamic-M range.

## 2026-05-20 - CUTLASS generated SM80 BF16 shape sweep

Follow-up reason:

- The hand-instantiated CUTLASS configs only covered a small subset of SM80 BF16 tensor-op
  GEMM shapes.
- CUTLASS's generator lists many more BF16 `nt` candidates for SM80, including
  `256x128`, `256x64`, `64x256`, and higher-stage `64x64` / `64x128` tiles.
- This pass adds a focused subset of those generated shapes instead of guessing more
  hand-written tile sizes.

Research / discovery:

```bash
python3 /tmp/cutlass/python/cutlass_library/generator.py \
  --operations gemm --architectures 80 \
  --build-dir /tmp/cutlass-gemm-list \
  --curr-build-dir /tmp/cutlass-gemm-list \
  --generator-target library \
  --kernels 'cutlass_tensorop_*bf16*gemm*' \
  --selected-kernel-list /tmp/cutlass-gemm-list/kernels.txt
```

- The generator emitted 638 SM80 BF16 tensor-op GEMM candidate names.
- The relevant layout family for this benchmark is `nt` with BF16 input/output and FP32
  accumulation, matching `Y[M,N] = X[M,K] * W[N,K]^T`.

Code changes:

- Added valid generated-shape configs:
  `bf16_cutlass_64x64_s10`, `bf16_cutlass_64x64x64_s5`,
  `bf16_cutlass_64x128_s6`, `bf16_cutlass_64x256_s4`,
  `bf16_cutlass_128x64_s6`, `bf16_cutlass_128x128_s5`,
  `bf16_cutlass_256x64`, `bf16_cutlass_256x64_s4`, and
  `bf16_cutlass_256x128`.
- Added tuner `--keep-going` so generated-kernel sweeps can skip failed configs while
  preserving measurements from valid configs.
- Rejected generated configs that failed correctness with `max_abs=6.90625`:
  `bf16_cutlass_64x256x64_s4`, `bf16_cutlass_128x256x64`,
  `bf16_cutlass_256x64x64`, and `bf16_cutlass_256x128x64`.

Commands:

```bash
GEMMA4_PREFILL_GRAPH_REPEATS=16 \
  python3 src/experiments/gemma4_prefill_tune.py \
  --backend sgemm-bf16 --ops ffn_down \
  --configs bf16_cutlass_64x64_s10,bf16_cutlass_64x64x64_s5,bf16_cutlass_64x128_s6,bf16_cutlass_64x256_s4,bf16_cutlass_128x64_s6,bf16_cutlass_128x128_s5,bf16_cutlass_128x256x64,bf16_cutlass_256x64,bf16_cutlass_256x64_s4,bf16_cutlass_256x64x64,bf16_cutlass_256x128,bf16_cutlass_256x128x64 \
  --m 64,96,128,256 --iters 16 --warmup 4 \
  --cublas-backend lt --cublaslt-heuristics 16 --keep-going \
  --out build/experiments/gemma4_prefill_tune/sgemm_bf16_ffn_down_generated_cutlass_both_graph_cublaslt_h16.csv
```

Best valid generated shapes:

| M | Best config | cuBLASLt h16 ms | Custom ms | Speedup |
| ---: | --- | ---: | ---: | ---: |
| 64 | `bf16_cutlass_64x64_s10` | 0.3419 | 0.3379 | 1.012x |
| 96 | `bf16_cutlass_64x128_s6` | 0.3723 | 0.3513 | 1.060x |
| 128 | `bf16_cutlass_64x128_s6` | 0.3771 | 0.3736 | 1.009x |
| 256 | `bf16_cutlass_128x128_s5` | 0.5035 | 0.5367 | 0.938x |

Summary:

- The generated shape sweep finds a local `1.06x` win at `M=96` against h16 cuBLASLt.
- The dynamic dispatch result over `M=64,96,128,256` is only `1.013x` because the route
  still falls back to cuBLASLt for most sizes.
- Custom-only speedup remains below parity (`0.997x`) for this focused sweep.

Conclusion: CUTLASS's generated SM80 BF16 default shapes add one narrow local win but do
not produce the requested broad 15-20% average speedup over cuBLASLt. The generated list
also confirms that some legal-looking K=64 large-tile instantiations are numerically
invalid in this harness and must remain excluded.

### Explicit M >= 256 check

Follow-up correction:

- The previous generated-shape sweep covered `M=256`, but the large-token side also needs
  explicit checks beyond `256`.
- This run checks `M=256,512,1024,2048` with the current best larger-M candidate set.

Command:

```bash
GEMMA4_PREFILL_GRAPH_REPEATS=16 \
  python3 src/experiments/gemma4_prefill_tune.py \
  --backend sgemm-bf16 --ops ffn_down \
  --configs bf16_auto_ffn_down,bf16_cutlass_128x128_s5,bf16_cutlass_128x128x64,bf16_cutlass_128x256,bf16_cutlass_256x128,bf16_streamk_128x128x64 \
  --m 256,512,1024,2048 --iters 20 --warmup 5 \
  --cublas-backend lt --cublaslt-heuristics 32 --skip-build \
  --out build/experiments/gemma4_prefill_tune/sgemm_bf16_ffn_down_m256_2048_check_cublaslt_h32.csv
```

Results using the best custom config per `M` and the best graph-timed cuBLASLt h32 time:

| M | Best custom config | cuBLASLt h32 ms | Custom ms | Speedup |
| ---: | --- | ---: | ---: | ---: |
| 256 | `bf16_cutlass_128x128_s5` | 0.4909 | 0.5275 | 0.931x |
| 512 | `bf16_cutlass_256x128` | 0.9118 | 0.9423 | 0.968x |
| 1024 | `bf16_cutlass_128x256` | 1.8790 | 1.8695 | 1.005x |
| 2048 | `bf16_cutlass_128x256` | 3.6909 | 3.7630 | 0.981x |

Summary:

- Best custom geomean over `M=256,512,1024,2048`: `0.930x`.
- Weighted custom-only speedup over this explicit large-M sweep: `0.9817x`.
- No `M >= 256` reaches the `1.05x` custom routing threshold.
- `M=1024` is near parity, but not close to the requested 15-20% win.

Conclusion: the larger-token check confirms that the current custom routes do not beat
cuBLASLt beyond `M=256`; the remaining gap is not just a small-M artifact.

### cuBLASLt h64 heuristic check

Follow-up reason:

- Most corrected comparisons used up to 32 cuBLASLt heuristic candidates.
- To check whether the library baseline was still under-tuned, this run increases the
  cuBLASLt candidate count to 64 over the current critical `M=64..1024` range.

Command:

```bash
GEMMA4_PREFILL_GRAPH_REPEATS=16 \
  python3 src/experiments/gemma4_prefill_tune.py \
  --backend sgemm-bf16 --ops ffn_down \
  --configs bf16_auto_ffn_down,bf16_cutlass_64x64_s10,bf16_cutlass_64x128_s6,bf16_cutlass_128x128_s5,bf16_cutlass_128x256,bf16_cutlass_256x128,bf16_streamk_64x64x64,bf16_streamk_s2_128x128x64 \
  --m 64,96,128,256,512,1024 --iters 16 --warmup 4 \
  --cublas-backend lt --cublaslt-heuristics 64 --skip-build \
  --out build/experiments/gemma4_prefill_tune/sgemm_bf16_ffn_down_h64_check_cublaslt.csv
```

Best custom config per `M`:

| M | Best custom config | cuBLASLt h64 ms | Custom ms | Speedup |
| ---: | --- | ---: | ---: | ---: |
| 64 | `bf16_cutlass_64x64_s10` | 0.3420 | 0.3378 | 1.012x |
| 96 | `bf16_cutlass_64x128_s6` | 0.3720 | 0.3552 | 1.047x |
| 128 | `bf16_streamk_s2_128x128x64` | 0.3769 | 0.3619 | 1.041x |
| 256 | `bf16_cutlass_128x128_s5` | 0.5024 | 0.5341 | 0.941x |
| 512 | `bf16_cutlass_256x128` | 0.9383 | 0.9633 | 0.974x |
| 1024 | `bf16_cutlass_256x128` | 1.8999 | 1.8876 | 1.007x |

Summary:

- Dynamic custom-only speedup over `M=64..1024`: `0.9981x`.
- No `M` routes to custom at the `1.05x` threshold when using the minimum cuBLASLt h64
  baseline per `M`.
- Increasing the cuBLASLt heuristic count does not create a custom win; it reinforces
  that the remaining custom route is near parity at best.

Conclusion: h64 cuBLASLt remains the correct comparison target for this shape class, and
the current custom candidates still do not produce the requested broad 15-20% win.

## 2026-05-21 - Quack Python CuTe RMSNorm on Gemma 4 shapes

Scope:

- Download upstream `Dao-AILab/quack/quack/rmsnorm.py` directly into
  `src/experiments/quack_rmsnorm/rmsnorm.py`.
- Exercise that exact downloaded file through a small Gemma-specific Python benchmark
  wrapper.
- Tune the forward launch knobs that Quack exposes for the Gemma 4 A6000 target:
  `num_threads`, `threads_per_row`, `reload_from`, and `delay_w_load`.
- Compare against the current native CUDA RMSNorm benchmark.

Environment:

- GPU: NVIDIA RTX A6000, compute capability 8.6.
- Driver: 580.126.16.
- CUDA toolkit: 12.9, `nvcc V12.9.86`.
- Python: 3.12.13.
- Quack support package installed from GitHub main commit
  `5e4d024e839bf851b820887c4c5b9656f206fa08`, because the downloaded `main`
  file imports modules such as `quack.dsl` that are not present in the older PyPI
  wheel.

Files:

- `src/experiments/quack_rmsnorm/rmsnorm.py`
- `src/experiments/quack_rmsnorm/gemma4_quack_rmsnorm_bench.py`
- Result CSVs under `/tmp/gemma4_quack_rmsnorm_20260521/`

Commands:

```bash
python3 -m pip install --user quack-kernels
python3 -m pip install --user --force-reinstall --no-deps \
  git+https://github.com/Dao-AILab/quack.git

python3 -m py_compile \
  src/experiments/quack_rmsnorm/gemma4_quack_rmsnorm_bench.py

python3 src/experiments/quack_rmsnorm/gemma4_quack_rmsnorm_bench.py \
  --width 5376 --rows 1,64 \
  --modes weighted,residual_quack_fused,residual_gemma_exact_split \
  --iters 100 --warmup 10 --trials 2 \
  --configs heuristic,128:64:none:0,128:128:none:0,256:64:none:0,256:128:none:0 \
  > /tmp/gemma4_quack_rmsnorm_20260521/quack_w5376_sweep.csv

python3 src/experiments/quack_rmsnorm/gemma4_quack_rmsnorm_bench.py \
  --width 5376 --rows 1,64 \
  --modes weighted,residual_quack_fused \
  --iters 100 --warmup 10 --trials 2 \
  --configs 128:64:none:1,128:64:smem:0,128:64:gmem:0,256:128:none:1,256:128:smem:0,256:128:gmem:0 \
  > /tmp/gemma4_quack_rmsnorm_20260521/quack_w5376_reload_delay_sweep.csv

python3 src/experiments/quack_rmsnorm/gemma4_quack_rmsnorm_bench.py \
  --width 256 --rows 1,64 --modes weighted,scale_free \
  --iters 100 --warmup 10 --trials 2 \
  --configs 128:64:none:0,128:128:none:0 \
  > /tmp/gemma4_quack_rmsnorm_20260521/quack_w256_best.csv

make -B rmsnorm-bench
GEMMA4_RMSNORM_BENCH_SEED=0x20260521 \
  ./build/experiments/gemma4_rmsnorm_bench 100 10 2 64 5376 \
  > /tmp/gemma4_quack_rmsnorm_20260521/native_w5376_rows64.csv
GEMMA4_RMSNORM_BENCH_SEED=0x20260521 \
  ./build/experiments/gemma4_rmsnorm_bench 100 10 2 64 256 \
  > /tmp/gemma4_quack_rmsnorm_20260521/native_w256_rows64.csv
```

Correctness:

- Quack weighted and scale-free outputs matched the BF16 PyTorch reference in these
  runs with `max_abs <= 0.0009765625` and `rstd_max_abs <= 2.38419e-7`.
- `residual_quack_fused` is not Gemma-exact: Quack normalizes the FP32 `x + residual`
  value. The current CUDA fused kernel intentionally normalizes the BF16-rounded
  residual to match standalone residual add followed by RMSNorm.
- `residual_gemma_exact_split` models the Gemma-exact two-step semantics in Python,
  but it is not a useful replacement kernel because it launches/copies separately.

Best fixed Quack configs from the sweep:

| Shape / mode | Rows | Best Quack config | Quack ms | Native graph ms | Slowdown |
| --- | ---: | --- | ---: | ---: | ---: |
| width 5376 weighted | 1 | `256:128:gmem:0` | 0.040568 | 0.002213 | 18.33x |
| width 5376 weighted | 64 | `256:128:smem:0` | 0.039627 | 0.008176 | 4.85x |
| width 5376 Quack residual fused | 1 | `256:128:smem:0` | 0.034151 | 0.002416 | 14.14x |
| width 5376 Quack residual fused | 64 | `256:128:smem:0` | 0.033789 | 0.002925 | 11.55x |
| width 256 weighted | 1 | `128:128:none:0` | 0.033301 | 0.001599 | 20.83x |
| width 256 weighted | 64 | `128:128:none:0` | 0.037027 | 0.001804 | 20.53x |
| width 256 scale-free | 1 | `128:64:none:0` | 0.027449 | 0.001383 | 19.85x |
| width 256 scale-free | 64 | `128:64:none:0` | 0.030554 | 0.001421 | 21.50x |

Conclusion:

- The Quack Python CuTe RMSNorm source is now downloaded and runnable on this machine,
  but it is not competitive with our native CUDA RMSNorm on RTX A6000/SM86.
- The full Quack autotuner was not practical here: even a single `1 x 256` tuned run
  compiled for over a minute before being killed. Fixed-config tuning was enough to
  show a large gap.
- Do not replace the current RMSNorm kernels with this Quack path. Keep it as an
  experiment/reference for CuTe structure; the production path should remain the
  specialized CUDA implementation.

## 2026-05-22 - FlashAttention-2 SM80 sliding-attention extraction

Runtime files:

- `src/gemma4_flash_attention.cu`
- `src/experiments/gemma4_flash_attention_bench.cu`
- `src/third_party_stubs/ATen/cuda/detail/UnpackRaw.cuh`
- `Makefile`
- `experiments/flash-attention/csrc/flash_attn/src/*`
- `experiments/flash-attention/csrc/cutlass/include/*`

Change:

- Added a raw CUDA Gemma 4 sliding-attention launcher around FlashAttention-2's SM80
  forward device template.
- The project-facing implementation stays in one translation unit:
  `src/gemma4_flash_attention.cu`.
- The path is specialized to the Gemma 4 31B sliding geometry:
  - BF16
  - batch-major `[B, S, H, D]`
  - `Hq=32`
  - `Hkv=16`
  - `D=256`
  - local causal sliding mask with `window_left=1024`, `window_right=0`
- The launch shape follows the A6000/SM86 branch from FA2's `hdim256` launcher:
  `Flash_fwd_kernel_traits<256, 64, 64, 4, false, false, bf16>`.
- Added a benchmark with a small CPU BF16 reference check.

Commands:

```bash
git -C experiments/flash-attention submodule update --init --recursive csrc/cutlass

make flash-attn-bench

./build/experiments/gemma4_flash_attention_bench 64 10 3 1 1 64

./build/experiments/gemma4_flash_attention_bench 1024 100 20 3 1 64

nvcc -std=c++17 -O3 -arch=sm_86 --expt-relaxed-constexpr -Xptxas -v \
  -Isrc -Isrc/third_party_stubs \
  -Iexperiments/flash-attention/csrc/flash_attn/src \
  -Iexperiments/flash-attention/csrc/cutlass/include \
  -c src/gemma4_flash_attention.cu \
  -o /tmp/gemma4_flash_attention_resource.o

ncu --metric gpu__time_duration.sum --target-processes all \
  ./build/experiments/gemma4_flash_attention_bench 1024 10 5 1 1 0
```

Environment:

- GPU: NVIDIA RTX A6000, compute capability 8.6
- Driver: 580.126.16
- CUDA/NVCC: 12.9.86
- Clocks: not locked
- Cache policy: repeated-buffer warm-cache benchmark
- Official `flash_attn` Python package: not installed in system Python or uv env
- Nsight Compute: installed, but the quick `ncu` probe above was interrupted after it
  produced no output for roughly a minute. Re-run with fewer launches or an
  ncu-specific harness that launches only one timed kernel.

Correctness:

- CPU reference check at `B=1, S=64, Hq=32, Hkv=16, D=256`, BF16 inputs:
  - `max_abs=0.00390625`
  - `mean_abs=7.7201e-05`
  - `max_rel=0.00390625`

Benchmark:

| Shape | Warmup | Iters | Trials | Best ms | Avg ms | Approx TFLOP/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `B=1, S=64, Hq=32, Hkv=16, D=256` | 3 | 10 | 1 | 0.0094272 | 0.0094272 | 7.22987 |
| `B=1, S=1024, Hq=32, Hkv=16, D=256` | 20 | 100 | 3 | 0.227083 | 0.234317 | 75.7286 |

Resource usage from `ptxas`:

| Kernel variant | Registers/thread | Stack bytes | Spill stores | Spill loads | Dynamic smem |
| --- | ---: | ---: | ---: | ---: | ---: |
| uneven-M/N local sliding | 254 | 0 | 0 | 0 | 98304 |
| even-M/N local sliding | 255 | 0 | 0 | 0 | 98304 |

Conclusion:

- This is now a compiling and numerically checked FA2-derived Gemma sliding-attention
  path.
- It is not yet a proof of 100% parity with the installed FlashAttention library because
  `flash_attn` is not installed in this environment. The next comparison step is to
  build/install the official package or add a local PyTorch extension comparator and run
  the exact same tensors through both paths.
- The implementation does not cover Gemma global attention (`D=512`); FA2's CUDA path
  only has head dimensions up to 256.

## 2026-05-22 - FlashAttention-2 sliding parity against upstream hdim256 source

Runtime files:

- `src/gemma4_flash_attention.cu`
- `src/experiments/gemma4_flash_attention_reference.cu`
- `src/experiments/gemma4_flash_attention_compare.py`
- `src/third_party_stubs/ATen/cuda/CUDAGeneratorImpl.h`
- `src/third_party_stubs/ATen/cuda/detail/UnpackRaw.cuh`
- `src/third_party_stubs/c10/cuda/CUDAException.h`
- `Makefile`

Change:

- Switched the project sliding-attention path from a hand-written `compute_attn`
  kernel wrapper to FA2's upstream `flash_fwd_launch_template.h` launcher and
  upstream `Flash_fwd_params` type under a project-local namespace.
- Matched FA2's hdim256 local dispatch behavior: the local hdim256 path uses
  `Is_even_MN=false` even for tile-aligned sequence lengths.
- Matched upstream compile flags for this path:
  `--expt-relaxed-constexpr --expt-extended-lambda --use_fast_math
  -D_GLIBCXX_USE_CXX11_ABI=1`.
- Added a small official-source reference shared library that calls
  `run_mha_fwd_<cutlass::bfloat16_t, 256, false>` from upstream
  `flash_fwd_hdim256_bf16_sm80.cu`.

Commands:

```bash
make -B flash-attn-bench flash-attn-lib flash-attn-reference-lib

./build/experiments/gemma4_flash_attention_bench 64 10 3 1 1 64

./build/experiments/gemma4_flash_attention_bench 1024 100 20 3 1 1

uv run python src/experiments/gemma4_flash_attention_compare.py \
  --seq 1024 --warmup 20 --iters 100

nvcc -std=c++17 -O3 -arch=sm_86 \
  --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math \
  -D_GLIBCXX_USE_CXX11_ABI=1 -Xptxas -v \
  -Isrc -Isrc/third_party_stubs \
  -Iexperiments/flash-attention/csrc/flash_attn/src \
  -Iexperiments/flash-attention/csrc/cutlass/include \
  -c src/gemma4_flash_attention.cu \
  -o build/ptx/gemma4_flash_attention_custom.o \
  2>&1 | tee build/ptx/gemma4_flash_attention_custom_ptxas.log

nvcc -std=c++17 -O3 -arch=sm_86 \
  --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math \
  -D_GLIBCXX_USE_CXX11_ABI=1 -Xptxas -v \
  -Iexperiments/flash-attention/csrc/flash_attn/src \
  -Iexperiments/flash-attention/csrc/cutlass/include \
  -I.venv/lib/python3.11/site-packages/torch/include \
  -I.venv/lib/python3.11/site-packages/torch/include/torch/csrc/api/include \
  -I/usr/include/python3.11 \
  -c experiments/flash-attention/csrc/flash_attn/src/flash_fwd_hdim256_bf16_sm80.cu \
  -o build/ptx/flash_fwd_hdim256_bf16_sm80_official.o \
  2>&1 | tee build/ptx/flash_fwd_hdim256_bf16_sm80_official_ptxas.log
```

Results:

- CPU reference check at `B=1, S=64`: `max_abs=0.00390625`,
  `mean_abs=7.72008e-05`, `max_rel=0.00390625`.
- C++ warm-cache benchmark at `B=1, S=1024`: best `0.227048 ms`,
  avg `0.229142 ms`, approx `75.7402 TFLOP/s`.
- Same-tensor Python comparison at `B=1, S=1024`:
  - custom: `0.236691 ms`, approx `72.654 TFLOP/s`
  - official-source reference: `0.236599 ms`, approx `72.683 TFLOP/s`
  - `custom/ref=1.000387`
  - output diff: `max_abs=0`, `mean_abs=0`, `max_rel=0`
- Runtime launch resources reported by both wrappers:
  - threads/block: `128`
  - dynamic shared memory: `98304` bytes

Resource parity for the selected local hdim256 BF16 kernel:

| Source | Registers/thread | Stack bytes | Spill stores | Spill loads | cmem[0] |
| --- | ---: | ---: | ---: | ---: | ---: |
| custom launcher-template path | 255 | 0 | 0 | 0 | 824 |
| upstream `flash_fwd_hdim256_bf16_sm80.cu` | 255 | 0 | 0 | 0 | 824 |

Notes:

- Building the full editable Python `flash_attn` package was attempted in the uv
  environment, but the build was interrupted and left orphaned NVCC children while
  compiling unused backward/head-dim variants. Those processes were killed. The
  Python package still does not import: `No module named 'flash_attn'`.
- This proves parity against the upstream hdim256 BF16 CUDA source path, but not yet
  against an installed full `flash_attn.flash_attn_func` package call.
- Gemma global attention (`D=512`) is still uncovered. The upstream FA2 CUDA path does
  not provide an SM80 hdim512 implementation; a separate design is needed for that
  requirement.

Deep verification rerun after adding direct `cudaFuncGetAttributes` checks:

```bash
make -B flash-attn-bench flash-attn-lib flash-attn-reference-lib

./build/experiments/gemma4_flash_attention_bench 64 10 3 1 1 64

./build/experiments/gemma4_flash_attention_bench 1024 100 20 3 1 64

uv run python src/experiments/gemma4_flash_attention_compare.py \
  --seq 1024 --warmup 50 --iters 1000 --trials 5 --skip-python-flash-attn
```

Results:

- CPU reference check at `B=1, S=64`: `max_abs=0.00390625`,
  `mean_abs=7.72008e-05`, `max_rel=0.00390625`.
- C++ warm-cache benchmark at `B=1, S=1024`: best `0.228857 ms`,
  avg `0.230685 ms`, approx `75.1416 TFLOP/s`.
- Same-tensor Python comparison at `B=1, S=1024`, 5 trials of 1000 launches:
  - custom median/min/max: `0.238173 / 0.236438 / 0.239902 ms`
  - official-source reference median/min/max: `0.238188 / 0.237964 / 0.240831 ms`
  - `custom/ref=0.999940` on median timing
  - output diff: `max_abs=0`, `mean_abs=0`, `max_rel=0`

Direct `cudaFuncGetAttributes` comparison for the exact selected kernel:

| Attribute | Custom | Official-source ref | Match |
| --- | ---: | ---: | ---: |
| `sharedSizeBytes` | 0 | 0 | yes |
| `constSizeBytes` | 0 | 0 | yes |
| `localSizeBytes` | 0 | 0 | yes |
| `maxThreadsPerBlock` | 256 | 256 | yes |
| `numRegs` | 255 | 255 | yes |
| `ptxVersion` | 86 | 86 | yes |
| `binaryVersion` | 86 | 86 | yes |
| `cacheModeCA` | 0 | 0 | yes |
| `maxDynamicSharedSizeBytes` | 98304 | 98304 | yes |
| `preferredShmemCarveout` | -1 | -1 | yes |
| `clusterDimMustBeSet` | 0 | 0 | yes |
| `requiredClusterWidth` | 0 | 0 | yes |
| `requiredClusterHeight` | 0 | 0 | yes |
| `requiredClusterDepth` | 0 | 0 | yes |
| `clusterSchedulingPolicyPreference` | 0 | 0 | yes |
| `nonPortableClusterSizeAllowed` | 0 | 0 | yes |

Current-source `ptxas` comparison for the selected local hdim256 BF16 kernel:

| Source | Registers/thread | Stack bytes | Spill stores | Spill loads | Barriers | cmem[0] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| custom `src/gemma4_flash_attention.cu` | 255 | 0 | 0 | 0 | 1 | 824 |
| upstream `flash_fwd_hdim256_bf16_sm80.cu` | 255 | 0 | 0 | 0 | 1 | 824 |

Conclusion:

- For Gemma sliding attention (`D=256`, BF16, local window 1024), the custom wrapper
  now matches the upstream hdim256 BF16 source path on output, launch shape, dynamic
  shared memory, direct CUDA function attributes, and selected ptxas resource stats.
- Timing is not a bitwise statistic, but the longer CUDA-event run is effectively equal
  (`custom/ref=0.999940` median). GPU clocks were not locked, so min/max should be
  treated as noisy.

Follow-up confirmation rerun:

```bash
uv run python src/experiments/gemma4_flash_attention_compare.py \
  --seq 1024 --warmup 50 --iters 1000 --trials 5 --skip-python-flash-attn

nvcc -std=c++17 -O3 -arch=sm_86 \
  --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math \
  -D_GLIBCXX_USE_CXX11_ABI=1 -Xptxas -v \
  -Isrc -Isrc/third_party_stubs \
  -Iexperiments/flash-attention/csrc/flash_attn/src \
  -Iexperiments/flash-attention/csrc/cutlass/include \
  -c src/gemma4_flash_attention.cu \
  -o build/ptx/gemma4_flash_attention_custom.o \
  2>&1 | tee build/ptx/gemma4_flash_attention_custom_ptxas.log

nvcc -std=c++17 -O3 -arch=sm_86 \
  --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math \
  -D_GLIBCXX_USE_CXX11_ABI=1 -Xptxas -v \
  -Iexperiments/flash-attention/csrc/flash_attn/src \
  -Iexperiments/flash-attention/csrc/cutlass/include \
  -I.venv/lib/python3.11/site-packages/torch/include \
  -I.venv/lib/python3.11/site-packages/torch/include/torch/csrc/api/include \
  -c experiments/flash-attention/csrc/flash_attn/src/flash_fwd_hdim256_bf16_sm80.cu \
  -o build/ptx/flash_fwd_hdim256_bf16_sm80_official.o \
  2>&1 | tee build/ptx/flash_fwd_hdim256_bf16_sm80_official_ptxas.log
```

Rerun results:

- Same-tensor output diff: `max_abs=0`, `mean_abs=0`, `max_rel=0`.
- CUDA-event timing, `B=1`, `S=1024`, 5 trials of 1000 launches:
  - custom median/min/max: `0.244411 / 0.243209 / 0.249125 ms`
  - official-source reference median/min/max: `0.244946 / 0.243538 / 0.251055 ms`
  - `custom/ref=0.997813` on median timing
- Every direct `cudaFuncGetAttributes` field matched:
  `sharedSizeBytes`, `constSizeBytes`, `localSizeBytes`, `maxThreadsPerBlock`,
  `numRegs`, `ptxVersion`, `binaryVersion`, `cacheModeCA`,
  `maxDynamicSharedSizeBytes`, `preferredShmemCarveout`, `clusterDimMustBeSet`,
  `requiredClusterWidth`, `requiredClusterHeight`, `requiredClusterDepth`,
  `clusterSchedulingPolicyPreference`, and `nonPortableClusterSizeAllowed`.
- Selected local hdim256 BF16 ptxas resource line matched:
  - custom: `255` regs/thread, `0` stack, `0` spill stores, `0` spill loads,
    `1` barrier, `824` bytes `cmem[0]`
  - upstream source: `255` regs/thread, `0` stack, `0` spill stores,
    `0` spill loads, `1` barrier, `824` bytes `cmem[0]`

Conclusion holds: all deterministic output, launch, runtime attribute, and selected
compiler-resource statistics checked here match exactly. Timing remains within normal
unlocked-clock noise.

## 2026-05-22 - Python FlashAttention comparison for hdim256 BF16

Goal:

- Compare the Gemma sliding-attention wrapper against the installed Python
  `flash_attn.flash_attn_func` path on the same tensors.

Package setup:

```bash
FLASH_ATTENTION_FORCE_BUILD=TRUE \
FLASH_ATTENTION_GEMMA_FWD_ONLY=TRUE \
FLASH_ATTN_CUDA_ARCHS=80 \
MAX_JOBS=4 \
NVCC_THREADS=1 \
uv pip install --reinstall --no-deps --no-build-isolation experiments/flash-attention
```

Notes:

- No matching prebuilt wheel exists for `flash-attn 2.8.4`, Python 3.11,
  Torch `2.11`, CUDA 12, CXX11 ABI true:
  `flash_attn-2.8.4+cu12torch2.11cxx11abiTRUE-cp311-cp311-linux_x86_64.whl`
  returned HTTP 404 from the upstream release URL.
- A full unmodified source build was attempted with `FLASH_ATTN_CUDA_ARCHS=80`,
  `MAX_JOBS=4`, and `NVCC_THREADS=1`, but it spent a long time compiling all
  forward, backward, and split-KV instantiations. It was stopped after the
  hdim256 forward path had compiled but before the full package finished.
- For this comparison, the local upstream checkout was given a targeted
  `FLASH_ATTENTION_GEMMA_FWD_ONLY` build mode. It keeps the public Python
  `flash_attn_func` API but compiles only the BF16 hdim256 forward and causal
  forward instantiations required by these Gemma sliding-attention benchmarks.
  Unsupported dtypes/head dims/backward/split-KV paths intentionally fail.
- Torch stayed at `2.11.0+cu128` after install.

Verification:

```bash
uv run python - <<'PY'
import torch
import flash_attn
import flash_attn_2_cuda
from flash_attn import flash_attn_func
print(torch.__version__, torch.version.cuda)
print(flash_attn.__version__)
print(flash_attn_2_cuda.__file__)
print(flash_attn_func)
PY
```

Import result:

- Torch: `2.11.0+cu128`, CUDA runtime version reported by Torch: `12.8`.
- `flash_attn`: `2.8.4`.
- `flash_attn_2_cuda` loaded from the repo `.venv`.
- `flash_attn_func` imported successfully.

Smoke checks:

```bash
uv run python src/experiments/gemma4_flash_attention_compare.py \
  --seq 1024 --warmup 5 --iters 20 --trials 2

uv run python src/experiments/gemma4_flash_attention_compare.py \
  --seq 2048 --warmup 5 --iters 10 --trials 2
```

Smoke results:

- `S=1024`: `diff_vs_flash_attn max_abs=0 mean_abs=0 max_rel=0`.
- `S=2048`: `diff_vs_flash_attn max_abs=0 mean_abs=0 max_rel=0`.

Clean sequential timing runs:

```bash
uv run python src/experiments/gemma4_flash_attention_compare.py \
  --seq 1024 --warmup 50 --iters 1000 --trials 5

uv run python src/experiments/gemma4_flash_attention_compare.py \
  --seq 2048 --warmup 50 --iters 500 --trials 5
```

Results at `B=1`, `S=1024`:

- custom median/min/max: `0.236058 / 0.235717 / 0.238015 ms`
- official-source reference median/min/max:
  `0.237129 / 0.235751 / 0.239483 ms`
- Python `flash_attn_func` median/min/max:
  `0.234289 / 0.232250 / 0.237240 ms`
- `diff_vs_official_source_ref max_abs=0 mean_abs=0 max_rel=0`
- `diff_vs_flash_attn max_abs=0 mean_abs=0 max_rel=0`
- `custom/python_flash_attn=1.007549` on median timing

Results at `B=1`, `S=2048`:

- custom median/min/max: `0.606319 / 0.605090 / 0.618479 ms`
- official-source reference median/min/max:
  `0.611226 / 0.609790 / 0.613673 ms`
- Python `flash_attn_func` median/min/max:
  `0.614401 / 0.613331 / 0.617600 ms`
- `diff_vs_official_source_ref max_abs=0 mean_abs=0 max_rel=0`
- `diff_vs_flash_attn max_abs=0 mean_abs=0 max_rel=0`
- `custom/python_flash_attn=0.986845` on median timing

Runtime attribute comparison in both runs:

- Every queried `cudaFuncGetAttributes` field matched between the custom wrapper
  and the official-source reference wrapper:
  `sharedSizeBytes`, `constSizeBytes`, `localSizeBytes`, `maxThreadsPerBlock`,
  `numRegs`, `ptxVersion`, `binaryVersion`, `cacheModeCA`,
  `maxDynamicSharedSizeBytes`, `preferredShmemCarveout`, `clusterDimMustBeSet`,
  `requiredClusterWidth`, `requiredClusterHeight`, `requiredClusterDepth`,
  `clusterSchedulingPolicyPreference`, and `nonPortableClusterSizeAllowed`.

Conclusion:

- For the benchmarked Gemma sliding path (`BF16`, `D=256`, `32` Q heads,
  `16` KV heads, local window left `1024`, right `0`), the custom wrapper
  matches Python `flash_attn_func` numerically with exact zero output diff.
- Clean sequential timing is within normal unlocked-clock noise of the Python
  path: about `+0.75%` at `S=1024` and `-1.32%` at `S=2048`.
- This satisfies parity for the Python API path used by these shapes. It is not
  a full-package validation of every upstream FlashAttention feature because the
  installed package was intentionally built in Gemma-only forward mode.
## 2026-05-23 - FFN decode register prefetch vs 3-stage async weight staging

Goal:

- Test two decode-FFN GEMV scheduling variants for the gate/up projection:
  direct register prefetch and shared-memory async staging.
- Use the existing cuDNN split FFN benchmark harness for correctness and timing.
- Do not keep multiple source-level selection flags for the variants.

CUDA guide context:

- The CUDA guide describes pipelines as a FIFO mechanism for sequencing
  asynchronous copies and supporting double/multi-buffered producer-consumer
  patterns (CUDA Programming Guide, p.129).
- The LDGSTS async-copy path is specifically for global-to-shared copies on
  CC 8.0+ and can keep global-memory loads in flight while reducing register
  pressure (CUDA Programming Guide, p.300 and p.302).
- The guide's multi-stage prefetch example blocks on the current batch,
  computes it, and schedules a future batch (CUDA Programming Guide, p.312).

Common benchmark setup:

```bash
GEMMA4_FFN_CUDNN_BENCH_SEED=0x20260522 \
  build/experiments/<binary> 100 20 3
```

- GPU: NVIDIA RTX A6000.
- Shape: `tokens=1`, `hidden=5376`, `intermediate=21504`.
- Iterations: `100` timed, `20` warmup, `3` trials.
- cuDNN frontend version: `92200`.
- Full fused cuDNN FFN graph had no valid execution plan, so the comparison is
  against the benchmark's cuDNN split path.
- All custom variants reported `max_abs_vs_split=0`.

### Variant 1: direct register prefetch

Implementation tested:

- Kept `ACT_TILE=2`.
- Removed the shared-memory weight pit stop for gate/up.
- Each owning thread loaded current gate/up `Bf16Packed128` values directly
  from global memory into registers.
- Issued the next tile's vectorized global loads before consuming the current
  tile's register values.
- Swapped register sets after the current tile's math/down accumulation.

Build and run:

```bash
nvcc -std=c++17 -O3 -arch=sm_86 -Xptxas=-v \
  -Isrc -I/tmp/cudnn-frontend/include \
  src/experiments/gemma4_ffn_cudnn_bench.cu \
  src/gemma4_ffn_decode.cu \
  -lcudnn -lnvrtc -lcuda \
  -o build/experiments/gemma4_ffn_cudnn_bench_regpipe

GEMMA4_FFN_CUDNN_BENCH_SEED=0x20260522 \
  build/experiments/gemma4_ffn_cudnn_bench_regpipe 100 20 3
```

Compiler resource result:

- FFN kernel: `64` registers/thread.
- Spills: `0` spill stores, `0` spill loads.
- Static shared memory: `11404` bytes.
- Barriers: `1`.

Timing:

- `custom_device_ms`: `1.106724`
- `custom_scratch_clear_device_ms`: `0.001092`
- `custom_minus_clear_device_ms`: `1.105632`
- `cudnn_split_device_ms`: `1.000129`
- `custom_minus_clear_vs_cudnn_split_speedup`: `0.904577`
- `max_abs_vs_split`: `0`

### Variant 2: 3-stage shared async staging

Implementation tested:

- Used one active implementation, not an extra runtime flag.
- Prologue issued async global-to-shared copies for tile `0` and tile `1`.
- Main loop waited for the current tile, issued async copies for tile `k+2`
  into the next rotating shared-memory stage, then computed tile `k`.
- Used `__pipeline_memcpy_async`, `__pipeline_commit`, and
  `__pipeline_wait_prior`.
- Used `ACT_TILE=1` for this run. With `ACT_TILE=2`, the required dynamic
  shared-memory staging size is
  `3 * 2 * 2 * 672 * 16 = 129024` bytes before static shared memory, which is
  too large for this target's per-block shared-memory budget.

Build and run:

```bash
nvcc -std=c++17 -O3 -arch=sm_86 -Xptxas=-v \
  -Isrc -I/tmp/cudnn-frontend/include \
  src/experiments/gemma4_ffn_cudnn_bench.cu \
  src/gemma4_ffn_decode.cu \
  -lcudnn -lnvrtc -lcuda \
  -o build/experiments/gemma4_ffn_cudnn_bench_async3

GEMMA4_FFN_CUDNN_BENCH_SEED=0x20260522 \
  build/experiments/gemma4_ffn_cudnn_bench_async3 100 20 3
```

Compiler resource result:

- FFN kernel: `50` registers/thread.
- Spills: `0` spill stores, `0` spill loads.
- Static shared memory: `11152` bytes.
- Dynamic shared-memory weight staging: `64512` bytes.
- Barriers: `1`.

Timing:

- `custom_device_ms`: `1.116366`
- `custom_scratch_clear_device_ms`: `0.001102`
- `custom_minus_clear_device_ms`: `1.115264`
- `cudnn_split_device_ms`: `1.000023`
- `custom_minus_clear_vs_cudnn_split_speedup`: `0.896669`
- `max_abs_vs_split`: `0`

### Restored final source path: shared-X paired reducer

After testing both requested variants, the source was restored to the previously
fastest measured approach:

- `ACT_TILE=2`.
- Load `x` once into shared memory.
- Use the paired shared-X reducer for gate/up.
- Do not stage weights through dynamic shared memory.

Build and run:

```bash
nvcc -std=c++17 -O3 -arch=sm_86 -Xptxas=-v \
  -Isrc -I/tmp/cudnn-frontend/include \
  src/experiments/gemma4_ffn_cudnn_bench.cu \
  src/gemma4_ffn_decode.cu \
  -lcudnn -lnvrtc -lcuda \
  -o build/experiments/gemma4_ffn_cudnn_bench_sharedx

GEMMA4_FFN_CUDNN_BENCH_SEED=0x20260522 \
  build/experiments/gemma4_ffn_cudnn_bench_sharedx 100 20 3
```

Compiler resource result:

- FFN kernel: `48` registers/thread.
- Spills: `0` spill stores, `0` spill loads.
- Static shared memory: `11404` bytes.
- Barriers: `1`.

Timing:

- `custom_device_ms`: `1.106684`
- `custom_scratch_clear_device_ms`: `0.001081`
- `custom_minus_clear_device_ms`: `1.105603`
- `cudnn_split_device_ms`: `1.001380`
- `custom_minus_clear_vs_cudnn_split_speedup`: `0.905732`
- `max_abs_vs_split`: `0`

Conclusion:

- Direct register prefetch and restored shared-X paired reduction are effectively
  tied in this run: `1.105632 ms` vs `1.105603 ms` minus scratch clear.
- The 3-stage shared async staging path was slower at `1.115264 ms` minus
  scratch clear, even though it compiled without spills. It also required
  dropping to `ACT_TILE=1` to fit shared memory, which increased loop/reduction
  overhead.
- For `tokens=1` decode FFN GEMV, staging gate/up weights through shared memory
  does not currently pay off. The best source state after this experiment is
  the shared-X paired reducer with direct coalesced weight reads.
