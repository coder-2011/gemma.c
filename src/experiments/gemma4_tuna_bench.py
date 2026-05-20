#!/usr/bin/env python3
"""Benchmark Tuna FP16 x FP4 GEMM on Gemma 4 production projection shapes.

This keeps Tuna as an experiment: it generates a minimal PyTorch CUDA extension under
build/tuna_gemma4/<variant> from experiments/tuna/kernel.cu, with dispatch entries only
for the requested Gemma projection shapes.
"""

import argparse
import json
import math
import os
import shutil
import statistics
import textwrap
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

REPO = Path(__file__).resolve().parents[2]
TUNA = REPO / "experiments" / "tuna"
BUILD_ROOT = REPO / "build" / "tuna_gemma4"

OPS = {
    "sliding_qkv": (16384, 5376),
    "sliding_o": (5376, 8192),
    "global_q": (16384, 5376),
    "global_k": (2048, 5376),
    "global_o": (5376, 16384),
    "ffn_gate_up": (43008, 5376),
    "ffn_down": (5376, 21504),
    "final_logits": (262144, 5376),
}

CONFIG_HEADER = """#define TRANSFORM_N true
#define TRANSFORM_T false
#define LAYOUT_C true
#define LAYOUT_F false
#define QUANTIZED true
#define NOT_QUANTIZED false

typedef bool MATRIX_QUANTIZE_STATUS;
typedef bool MATRIX_TRANSFORM;

struct KernelConfig
{
    const int tileM;
    const int tileN;
    const int tileK;
    const int patchM;
    const int patchN;
    const int k;
    const int m;
    const int splitK;
    const int warpCountM;
    const int warpCountN;
    const int mmaSizeM;
    const int mmaSizeN;
    const int mmaSizeK;
    const int warpMmaCountM;
    const int warpMmaCountN;
    const int warpMmaCountK;
    const int contiguousBytesA;
    const int contiguousBytesB;
    const int deqBlockSize;
    const int stages;
    const int absMaxPerBlock;
    const int threadsPerBlock;
    const int pipelineStrat;
    const int paddingC;

    static constexpr int codeSize = 16;
    static constexpr bool isSafe = false;
    static constexpr int alignSizeBytesA = 16;
    static constexpr int alignSizeBytesB = 16;
    static constexpr MATRIX_TRANSFORM transformA = TRANSFORM_N;
    static constexpr MATRIX_TRANSFORM transformB = TRANSFORM_T;
    static constexpr MATRIX_TRANSFORM transformC = TRANSFORM_T;
    static constexpr MATRIX_QUANTIZE_STATUS quantStatA = QUANTIZED;
    static constexpr MATRIX_QUANTIZE_STATUS quantStatB = NOT_QUANTIZED;
    static constexpr int deqBlockCount = 1;
};

"""

CONFIG_TEMPLATE = """constexpr KernelConfig {name} = {{
    /* tileM */ {tile_m},
    /* tileN */ {tile_n},
    /* tileK */ {tile_k},
    /* patchM */ {patch_m},
    /* patchN */ {patch_n},
    /* k */ {k},
    /* m */ {m},
    /* splitK */ {split_k},
    /* warpCountM */ {warp_count_m},
    /* warpCountN */ {warp_count_n},
    /* mmaSizeM */ {mma_size_m},
    /* mmaSizeN */ {mma_size_n},
    /* mmaSizeK */ {mma_size_k},
    /* warpMmaCountM */ {warp_mma_count_m},
    /* warpMmaCountN */ {warp_mma_count_n},
    /* warpMmaCountK */ {warp_mma_count_k},
    /* contiguousBytesA */ {contiguous_bytes_A},
    /* contiguousBytesB */ {contiguous_bytes_B},
    /* deqBlockSize */ {deq_block_size},
    /* stages */ {stages},
    /* absMaxPerBlock */ {abs_max_per_block},
    /* threadsPerBlock */ {tpb},
    /* pipelineStrat */ {pipeline_strat},
    /* paddingC */ {padding_C}}};

"""

KEEP_FIELDS = [
    "patch_m", "patch_n", "tile_m", "tile_n", "tile_k", "pipeline_strat",
    "stages", "warp_count_m", "warp_count_n", "split_k", "mma_size_m",
    "mma_size_n", "mma_size_k", "contiguous_bytes_B", "padding_C",
    "contiguous_bytes_A", "tpb", "warp_mma_count_m", "warp_mma_count_n",
    "warp_mma_count_k", "abs_max_per_block", "deq_block_size",
]

CODEBOOK = [
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
    0.33791524171829224, 0.4407098293345947, 0.5626170039176941,
    0.7229568362236023, 1.0,
]


def nearest_params(param_map, m, n, k):
    best = None
    best_score = float("inf")
    for key, value in param_map.items():
        km, kn, kk = [int(part) for part in key.split("_")]
        score = abs(math.log2(m / km)) + abs(math.log2(k / kk)) * 1.5
        score += abs(math.log2(max(n, 1) / max(kn, 1))) * 0.5
        if kk % 32 != 0 or km % 8 != 0:
            continue
        if score < best_score:
            best = value
            best_score = score
    if best is None:
        raise RuntimeError(f"no Tuna template config for m={m}, n={n}, k={k}")
    return {field: best[field] for field in KEEP_FIELDS}


def apply_variant(params, variant, n):
    p = dict(params)
    if variant == "nearest":
        return p
    if variant == "splitk1":
        p["split_k"] = 1
        return p
    if variant == "wide_n":
        if p["tile_n"] < 128:
            p["tile_n"] *= 2
            p["warp_mma_count_n"] = max(1, p["tile_n"] // (8 * p["warp_count_n"]))
        p["split_k"] = 1 if n >= 16 else p["split_k"]
        return p
    raise ValueError(f"unknown variant {variant}")


def make_config_name(op, n):
    return f"gemma4_{op}_{n}".replace("-", "_")


def write_extension_sources(variant, ops, ns):
    variant_dir = BUILD_ROOT / variant
    variant_dir.mkdir(parents=True, exist_ok=True)

    param_map = json.loads((TUNA / "param_map.json").read_text())
    configs = [CONFIG_HEADER]
    dispatch = []
    config_report = []

    for op in ops:
        m, k = OPS[op]
        for n in ns:
            base = nearest_params(param_map, m, n, k)
            params = apply_variant(base, variant, n)
            params["m"] = m
            params["n"] = n
            params["k"] = k
            params["deq_block_size"] = k
            params["abs_max_per_block"] = params["tile_m"]
            name = make_config_name(op, n)
            configs.append(CONFIG_TEMPLATE.format(name=name, **params))
            dispatch.append(
                f"    LAUNCH_KERNEL_IF_CONDITION({name}, {m}, {n}, {n}, {k})"
            )
            config_report.append((op, n, params))

    (variant_dir / "configs.cu").write_text("".join(configs))

    kernel = (TUNA / "kernel.cu").read_text()
    start = kernel.index("    if (false)\n    {\n    }\n")
    end = kernel.index("}\n\n__global__ void dequantize_4bit", start)
    wrapper_body = "    if (false)\n    {\n    }\n\n" + "\n".join(dispatch) + "\n"
    patched = kernel[:start] + wrapper_body + kernel[end:]
    (variant_dir / "kernel.cu").write_text(patched)
    shutil.copyfile(TUNA / "extension.cpp", variant_dir / "extension.cpp")
    return variant_dir, config_report


def build_extension(variant, ops, ns):
    variant_dir, config_report = write_extension_sources(variant, ops, ns)
    module = load(
        name=f"tuna_gemma4_{variant}",
        sources=[str(variant_dir / "extension.cpp"), str(variant_dir / "kernel.cu")],
        build_directory=str(variant_dir),
        extra_cuda_cflags=["--expt-relaxed-constexpr", "-O3", "-arch=sm_86"],
        verbose=False,
    )
    return module, config_report


def quantize_blockwise_dynamic_4bit(inp, block_size, preprocess=True):
    assert inp.numel() % 128 == 0
    assert inp.shape[-1] == block_size
    device = inp.device
    m, k = inp.shape[-2:]
    temp = inp.flatten().reshape(inp.numel() // block_size, block_size)
    absmax = torch.max(temp.abs(), dim=1)[0].float()
    code = torch.tensor(CODEBOOK, device=device, dtype=torch.float)
    normalized = torch.divide(temp, absmax.unsqueeze(-1)).flatten()
    chunk_size = 512 * 1024
    chunks = math.ceil(normalized.numel() / chunk_size)
    results = torch.empty((normalized.numel(),), dtype=torch.uint8, device=device)
    for i in range(chunks):
        begin = i * chunk_size
        end = min((i + 1) * chunk_size, normalized.numel())
        candidates = torch.sub(normalized[begin:end].unsqueeze(0), code.unsqueeze(-1)).abs()
        results[begin:end] = torch.argmin(candidates, keepdim=True, dim=0).flatten()
    if preprocess:
        results = (
            results.reshape(-1, m // 8, 8, k // 8, 4, 2)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(-1, m // 8, k // 8, 32, 2)
        )
        results = results[..., 0] * 16 + results[..., 1]
        results = results.reshape(*inp.shape[:-2], m // 8, k // 8, 32)
    return results, absmax, code


def time_cuda(fn, warmup, iters, trials):
    samples = []
    for _ in range(trials):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        stop.record()
        stop.synchronize()
        samples.append(start.elapsed_time(stop) / iters)
    return min(samples), statistics.median(samples), max(samples)


def gbps(weight_bytes, ms):
    return weight_bytes / (ms * 1.0e6)


def run_case(module, op, n, warmup, iters, trials, seed):
    m, k = OPS[op]
    torch.manual_seed(seed + m * 17 + n * 131 + k)
    x = torch.randn((n, k), dtype=torch.float16, device="cuda") * 0.1
    w = torch.randn((m, k), dtype=torch.float16, device="cuda") * 0.05
    out_tuna = torch.empty((n, m), dtype=torch.float16, device="cuda")
    out_cublas = torch.empty((n, m), dtype=torch.float16, device="cuda")
    q, absmax, code = quantize_blockwise_dynamic_4bit(w, block_size=k, preprocess=True)

    module.matmul(x, q, absmax, code, out_tuna, m, n, k)
    torch.mm(x, w.t(), out=out_cublas)
    torch.cuda.synchronize()

    diff = (out_tuna.float() - out_cublas.float()).abs()
    mean_abs = float(diff.mean().item())
    max_abs = float(diff.max().item())

    tuna = time_cuda(lambda: module.matmul(x, q, absmax, code, out_tuna, m, n, k), warmup, iters, trials)
    cublas = time_cuda(lambda: torch.mm(x, w.t(), out=out_cublas), warmup, iters, trials)

    weight_bytes_fp16 = m * k * 2
    weight_bytes_fp4 = m * k // 2 + m * 4
    return {
        "op": op,
        "m": m,
        "n": n,
        "k": k,
        "tuna_min_ms": tuna[0],
        "tuna_median_ms": tuna[1],
        "tuna_max_ms": tuna[2],
        "cublas_min_ms": cublas[0],
        "cublas_median_ms": cublas[1],
        "cublas_max_ms": cublas[2],
        "speedup_median": cublas[1] / tuna[1],
        "tuna_fp4_weight_gbps": gbps(weight_bytes_fp4, tuna[1]),
        "cublas_fp16_weight_gbps": gbps(weight_bytes_fp16, cublas[1]),
        "mean_abs_vs_fp16_weight": mean_abs,
        "max_abs_vs_fp16_weight": max_abs,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="nearest", choices=["nearest", "splitk1", "wide_n"])
    parser.add_argument("--ops", default="sliding_qkv,global_o,ffn_gate_up,ffn_down,final_logits")
    parser.add_argument("--n", default="1,16,128")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0x20260520)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    ops = [op.strip() for op in args.ops.split(",") if op.strip()]
    ns = [int(x) for x in args.n.split(",") if x]
    unknown = [op for op in ops if op not in OPS]
    if unknown:
        raise ValueError(f"unknown ops: {unknown}")

    module, config_report = build_extension(args.variant, ops, ns)
    print(f"device={torch.cuda.get_device_name()},torch={torch.__version__},cuda={torch.version.cuda}")
    print(f"variant={args.variant},warmup={args.warmup},iters={args.iters},trials={args.trials},seed={args.seed}")
    for op, n, params in config_report:
        print(
            "config="
            f"op={op},n={n},tile_m={params['tile_m']},tile_n={params['tile_n']},"
            f"tile_k={params['tile_k']},patch_m={params['patch_m']},"
            f"patch_n={params['patch_n']},split_k={params['split_k']},"
            f"stages={params['stages']},tpb={params['tpb']},"
            f"pipeline={params['pipeline_strat']}"
        )
    print("op,n,m,k,tuna_median_ms,cublas_median_ms,speedup_median,tuna_min_ms,tuna_max_ms,cublas_min_ms,cublas_max_ms,tuna_fp4_weight_gbps,cublas_fp16_weight_gbps,mean_abs_vs_fp16_weight,max_abs_vs_fp16_weight")
    for op in ops:
        for n in ns:
            result = run_case(module, op, n, args.warmup, args.iters, args.trials, args.seed)
            print(
                f"{result['op']},{result['n']},{result['m']},{result['k']},"
                f"{result['tuna_median_ms']:.6f},{result['cublas_median_ms']:.6f},"
                f"{result['speedup_median']:.6f},{result['tuna_min_ms']:.6f},"
                f"{result['tuna_max_ms']:.6f},{result['cublas_min_ms']:.6f},"
                f"{result['cublas_max_ms']:.6f},{result['tuna_fp4_weight_gbps']:.3f},"
                f"{result['cublas_fp16_weight_gbps']:.3f},"
                f"{result['mean_abs_vs_fp16_weight']:.6g},"
                f"{result['max_abs_vs_fp16_weight']:.6g}"
            )


if __name__ == "__main__":
    main()
