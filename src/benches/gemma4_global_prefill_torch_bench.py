#!/usr/bin/env python3
import argparse
import ctypes
import json
import math
import os
import statistics
import subprocess
import time

import torch
import torch.nn.functional as F


GEMMA4_NUM_QUERY_HEADS = 32
GEMMA4_GLOBAL_KV_HEADS = 4
GEMMA4_GLOBAL_HEAD_DIM = 512
GEMMA4_GLOBAL_ROTARY_DIM = 128
GEMMA4_RMS_NORM_EPS = 1e-6
GEMMA4_ROPE_THETA_GLOBAL = 1000000.0


def ptr(tensor):
    return ctypes.c_void_p(tensor.data_ptr())


def current_stream_ptr():
    return ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)


def percentile(values, pct):
    if len(values) == 1:
        return values[0]
    sorted_values = sorted(values)
    index = pct * 0.01 * (len(sorted_values) - 1)
    lo = math.floor(index)
    hi = math.ceil(index)
    frac = index - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def summarize(values):
    sorted_values = sorted(values)
    trim = math.floor(len(sorted_values) * 0.1)
    trimmed = sorted_values[trim : len(sorted_values) - trim] or sorted_values
    return {
        "median_ms": statistics.median(values),
        "mean_ms": statistics.fmean(values),
        "trimmed_mean_ms": statistics.fmean(trimmed),
        "min_ms": min(values),
        "max_ms": max(values),
        "p95_ms": percentile(values, 95.0),
        "p99_ms": percentile(values, 99.0),
        "stddev_ms": statistics.pstdev(values),
        "iqr_ms": percentile(values, 75.0) - percentile(values, 25.0),
        "samples_ms": values,
    }


def load_lib(path):
    lib = ctypes.CDLL(path)

    global_fwd = lib.gemma4_flash_attention_global_fwd_bf16
    global_fwd.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_float,
        ctypes.c_void_p,
    ]
    global_fwd.restype = ctypes.c_int

    global_fwd_norm_rope = lib.gemma4_flash_attention_global_fwd_bf16_norm_rope
    global_fwd_norm_rope.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_float,
        ctypes.c_void_p,
    ]
    global_fwd_norm_rope.restype = ctypes.c_int

    global_prepare = lib.gemma4_flash_attention_global_prepare_qkv_norm_rope_bf16
    global_prepare.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_void_p,
    ]
    global_prepare.restype = ctypes.c_int

    return global_fwd, global_fwd_norm_rope, global_prepare


def check_status(status, label):
    if status != 0:
        raise RuntimeError(f"{label} returned cudaError_t={status}")


def flush_l2(flush_buf):
    flush_buf.zero_()
    torch.cuda.synchronize()


def make_global_rope_tables(seq_len, device):
    half = GEMMA4_GLOBAL_ROTARY_DIM // 2
    pos = torch.arange(seq_len, device=device, dtype=torch.float32)[:, None]
    idx = torch.arange(half, device=device, dtype=torch.float32)[None, :]
    freq = torch.pow(GEMMA4_ROPE_THETA_GLOBAL, -2.0 * idx / GEMMA4_GLOBAL_HEAD_DIM)
    angles = pos * freq
    return torch.cos(angles).contiguous(), torch.sin(angles).contiguous()


def rms_scale(x):
    return torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + GEMMA4_RMS_NORM_EPS)


def apply_global_weighted_rope(x, weight, cos, sin):
    y = x.float() * rms_scale(x) * weight.float().view(1, 1, 1, -1)
    lo = y[..., : GEMMA4_GLOBAL_ROTARY_DIM // 2]
    hi = y[..., GEMMA4_GLOBAL_ROTARY_DIM // 2 : GEMMA4_GLOBAL_ROTARY_DIM]
    c = cos.view(1, cos.shape[0], 1, cos.shape[1])
    s = sin.view(1, sin.shape[0], 1, sin.shape[1])
    out = torch.empty_like(y)
    out[..., : GEMMA4_GLOBAL_ROTARY_DIM // 2] = lo * c - hi * s
    out[..., GEMMA4_GLOBAL_ROTARY_DIM // 2 : GEMMA4_GLOBAL_ROTARY_DIM] = lo * s + hi * c
    out[..., GEMMA4_GLOBAL_ROTARY_DIM :] = y[..., GEMMA4_GLOBAL_ROTARY_DIM :]
    return out.to(torch.bfloat16).contiguous()


def apply_global_v_norm(k):
    return (k.float() * rms_scale(k)).to(torch.bfloat16).contiguous()


def sdpa_bshd(q, k, v, out, scale):
    group = GEMMA4_NUM_QUERY_HEADS // GEMMA4_GLOBAL_KV_HEADS
    result = F.scaled_dot_product_attention(
        q.permute(0, 2, 1, 3),
        k.permute(0, 2, 1, 3).repeat_interleave(group, dim=1),
        v.permute(0, 2, 1, 3).repeat_interleave(group, dim=1),
        dropout_p=0.0,
        is_causal=True,
        scale=scale,
    )
    out.copy_(result.permute(0, 2, 1, 3).contiguous())


def make_inputs(args, device):
    dtype = torch.bfloat16
    torch.manual_seed(args.seed)
    q = (torch.randn(
        args.batch_size,
        args.seq_len,
        GEMMA4_NUM_QUERY_HEADS,
        GEMMA4_GLOBAL_HEAD_DIM,
        device=device,
        dtype=dtype,
    ) * 0.02).contiguous()
    k = (torch.randn(
        args.batch_size,
        args.seq_len,
        GEMMA4_GLOBAL_KV_HEADS,
        GEMMA4_GLOBAL_HEAD_DIM,
        device=device,
        dtype=dtype,
    ) * 0.02).contiguous()
    q_weight = (0.8 + 0.001 * torch.arange(
        GEMMA4_GLOBAL_HEAD_DIM, device=device, dtype=torch.float32
    ).remainder(29)).to(dtype).contiguous()
    k_weight = (0.9 - 0.001 * torch.arange(
        GEMMA4_GLOBAL_HEAD_DIM, device=device, dtype=torch.float32
    ).remainder(31)).to(dtype).contiguous()
    cos, sin = make_global_rope_tables(args.seq_len, device)
    return q, k, q_weight, k_weight, cos, sin


def time_paths(paths, samples, delay_s, flush_buf, cache_mode):
    timings = {name: [] for name in paths}
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    names = list(paths.keys())
    for sample in range(samples):
        order = names if sample % 2 == 0 else list(reversed(names))
        for name in order:
            if cache_mode == "cold":
                flush_l2(flush_buf)
            if delay_s > 0.0:
                time.sleep(delay_s)
            start.record()
            paths[name]()
            stop.record()
            stop.synchronize()
            timings[name].append(start.elapsed_time(stop))
    return {name: summarize(values) for name, values in timings.items()}


def nvidia_smi_snapshot():
    query = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,persistence_mode,ecc.mode.current,mig.mode.current,"
        "power.limit,clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        return subprocess.check_output(query, text=True).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def print_table(result):
    rows = [
        ("prep custom", "custom_global_prefill_norm_rope_prep"),
        ("direct custom", "direct_custom_global_prefill"),
        ("direct PyTorch", "direct_pytorch_sdpa"),
        ("fused custom", "fused_custom_norm_rope_global_prefill"),
        ("fused PyTorch", "fused_pytorch_norm_rope_sdpa"),
    ]
    print("| path | median ms | mean ms | min ms | p95 ms | max ms |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    for label, key in rows:
        if key not in result["timings"]:
            continue
        stats = result["timings"][key]
        print(
            f"| {label} | {stats['median_ms']:.6f} | {stats['mean_ms']:.6f} | "
            f"{stats['min_ms']:.6f} | {stats['p95_ms']:.6f} | {stats['max_ms']:.6f} |"
        )
    print()
    print("| comparison | PyTorch / custom median | custom result |")
    print("| --- | ---: | --- |")
    for label, key in result["speedups"].items():
        verdict = "faster" if key > 1.0 else "slower"
        print(f"| {label} | {key:.3f}x | custom {verdict} |")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lib", default="build/libgemma4_flash_attention.so")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--delay-s", type=float, default=0.1)
    parser.add_argument("--flush-mib", type=int, default=256)
    parser.add_argument("--cache", choices=("cold", "warm"), default="cold")
    parser.add_argument("--paths", choices=("all", "custom", "prep"), default="all")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not os.path.exists(args.lib):
        raise RuntimeError(f"missing custom library: {args.lib}")
    if args.samples <= 0 or args.warmup < 0:
        raise RuntimeError("--samples must be > 0 and --warmup must be >= 0")
    if args.delay_s < 0.0:
        raise RuntimeError("--delay-s must be nonnegative")
    if args.cache == "cold" and args.flush_mib <= 0:
        raise RuntimeError("--flush-mib must be > 0 for cold-cache timing")

    device = torch.device("cuda")
    global_fwd, global_fwd_norm_rope, global_prepare = load_lib(args.lib)
    scale = 1.0 / math.sqrt(GEMMA4_GLOBAL_HEAD_DIM)
    q, k, q_weight, k_weight, cos, sin = make_inputs(args, device)

    q_prepared = torch.empty_like(q)
    k_prepared = torch.empty_like(k)
    v_prepared = torch.empty_like(k)
    direct_custom_out = torch.empty_like(q)
    direct_torch_out = torch.empty_like(q)
    fused_custom_out = torch.empty_like(q)
    fused_torch_out = torch.empty_like(q)
    flush_buf = torch.empty(
        (args.flush_mib * 1024 * 1024) // 4,
        device=device,
        dtype=torch.int32,
    )

    def custom_fused():
        check_status(
            global_fwd_norm_rope(
                ptr(fused_custom_out),
                ctypes.c_void_p(0),
                ptr(q_prepared),
                ptr(k_prepared),
                ptr(v_prepared),
                ptr(q),
                ptr(k),
                ptr(q_weight),
                ptr(k_weight),
                ptr(cos),
                ptr(sin),
                args.batch_size,
                args.seq_len,
                args.seq_len,
                ctypes.c_float(scale),
                current_stream_ptr(),
            ),
            "gemma4_flash_attention_global_fwd_bf16_norm_rope",
        )

    def custom_prepare():
        check_status(
            global_prepare(
                ptr(q_prepared),
                ptr(k_prepared),
                ptr(v_prepared),
                ptr(q),
                ptr(k),
                ptr(q_weight),
                ptr(k_weight),
                ptr(cos),
                ptr(sin),
                args.batch_size,
                args.seq_len,
                current_stream_ptr(),
            ),
            "gemma4_flash_attention_global_prepare_qkv_norm_rope_bf16",
        )

    def custom_direct():
        check_status(
            global_fwd(
                ptr(direct_custom_out),
                ctypes.c_void_p(0),
                ptr(q_prepared),
                ptr(k_prepared),
                ptr(v_prepared),
                args.batch_size,
                args.seq_len,
                args.seq_len,
                ctypes.c_float(scale),
                current_stream_ptr(),
            ),
            "gemma4_flash_attention_global_fwd_bf16",
        )

    def torch_direct():
        sdpa_bshd(q_prepared, k_prepared, v_prepared, direct_torch_out, scale)

    def torch_fused():
        tq = apply_global_weighted_rope(q, q_weight, cos, sin)
        tk = apply_global_weighted_rope(k, k_weight, cos, sin)
        tv = apply_global_v_norm(k)
        sdpa_bshd(tq, tk, tv, fused_torch_out, scale)

    with torch.no_grad():
        timed_paths = {
            "direct_custom_global_prefill": custom_direct,
            "fused_custom_norm_rope_global_prefill": custom_fused,
        }
        if args.paths == "prep":
            timed_paths = {
                "custom_global_prefill_norm_rope_prep": custom_prepare,
            }
        if args.paths == "all":
            timed_paths = {
                "custom_global_prefill_norm_rope_prep": custom_prepare,
                "direct_custom_global_prefill": custom_direct,
                "direct_pytorch_sdpa": torch_direct,
                "fused_custom_norm_rope_global_prefill": custom_fused,
                "fused_pytorch_norm_rope_sdpa": torch_fused,
            }

        for _ in range(args.warmup):
            for path in timed_paths.values():
                path()
        torch.cuda.synchronize()

        correctness = {
            "fused_custom_vs_direct_custom_max_abs": None,
            "direct_custom_vs_pytorch_max_abs": None,
            "fused_custom_vs_pytorch_max_abs": None,
            "prepared_q_custom_vs_pytorch_max_abs": None,
            "prepared_k_custom_vs_pytorch_max_abs": None,
            "prepared_v_custom_vs_pytorch_max_abs": None,
        }
        if args.paths == "prep":
            custom_prepare()
            tq = apply_global_weighted_rope(q, q_weight, cos, sin)
            tk = apply_global_weighted_rope(k, k_weight, cos, sin)
            tv = apply_global_v_norm(k)
            torch.cuda.synchronize()
            correctness["prepared_q_custom_vs_pytorch_max_abs"] = float(
                (q_prepared.float() - tq.float()).abs().max().item()
            )
            correctness["prepared_k_custom_vs_pytorch_max_abs"] = float(
                (k_prepared.float() - tk.float()).abs().max().item()
            )
            correctness["prepared_v_custom_vs_pytorch_max_abs"] = float(
                (v_prepared.float() - tv.float()).abs().max().item()
            )
        else:
            custom_fused()
            custom_direct()
            torch_direct()
            torch_fused()
            torch.cuda.synchronize()
            correctness["fused_custom_vs_direct_custom_max_abs"] = float(
                (fused_custom_out.float() - direct_custom_out.float()).abs().max().item()
            )
            correctness["direct_custom_vs_pytorch_max_abs"] = float(
                (direct_custom_out.float() - direct_torch_out.float()).abs().max().item()
            )
            correctness["fused_custom_vs_pytorch_max_abs"] = float(
                (fused_custom_out.float() - fused_torch_out.float()).abs().max().item()
            )

        timings = time_paths(
            timed_paths, args.samples, args.delay_s, flush_buf, args.cache)
        if args.paths == "prep":
            checksum = float(
                q_prepared.float().sum().item()
                + k_prepared.float().sum().item()
                + v_prepared.float().sum().item()
            )
        else:
            checksum = float(
                direct_custom_out.float().sum().item()
                + direct_torch_out.float().sum().item()
                + fused_custom_out.float().sum().item()
                + fused_torch_out.float().sum().item()
            )

    result = {
        "contract": {
            "benchmark": f"global prefill BF16, {args.cache} cache custom/library path timing",
            "timing": "CUDA events on PyTorch current stream for both custom and PyTorch paths",
            "cache_mode": args.cache,
            "l2_flush_bytes": flush_buf.numel() * 4 if args.cache == "cold" else 0,
            "delay_s": args.delay_s,
            "delay_location": "before each timed path invocation",
            "samples": args.samples,
            "warmup": args.warmup,
            "launch_overhead": (
                "host enqueue overhead excluded; GPU timeline includes all "
                "kernels in the path"
            ),
            "path_order": "alternates forward/reverse each sample to reduce drift bias",
            "timed_paths": list(timings.keys()),
        },
        "env": {
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "nvidia_smi": nvidia_smi_snapshot(),
            "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "custom_lib": args.lib,
        },
        "shape": {
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "q_heads": GEMMA4_NUM_QUERY_HEADS,
            "kv_heads": GEMMA4_GLOBAL_KV_HEADS,
            "head_dim": GEMMA4_GLOBAL_HEAD_DIM,
            "rotary_dim": GEMMA4_GLOBAL_ROTARY_DIM,
            "dtype": "bf16",
        },
        "correctness": correctness,
        "timings": timings,
        "speedups": {},
        "checksum": checksum,
    }
    if "direct_pytorch_sdpa" in timings:
        result["speedups"]["direct attention"] = (
            timings["direct_pytorch_sdpa"]["median_ms"]
            / timings["direct_custom_global_prefill"]["median_ms"]
        )
    if "fused_pytorch_norm_rope_sdpa" in timings:
        result["speedups"]["fused norm_rope+attention"] = (
            timings["fused_pytorch_norm_rope_sdpa"]["median_ms"]
            / timings["fused_custom_norm_rope_global_prefill"]["median_ms"]
        )
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print()
    print_table(result)


if __name__ == "__main__":
    main()
