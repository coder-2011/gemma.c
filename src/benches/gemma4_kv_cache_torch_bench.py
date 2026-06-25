#!/usr/bin/env python3
import argparse
import json
import math
import statistics
import time

import torch
import torch.nn.functional as F


def percentile(values, pct):
    if not values:
        return 0.0
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


def flush_l2(flush_buf):
    if flush_buf is not None:
        flush_buf.add_(1)


def time_cuda(fn, warmup, iters, samples, flush_buf):
    for _ in range(warmup):
        flush_l2(flush_buf)
        fn()
    torch.cuda.synchronize()

    values = []
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    for _ in range(samples):
        if flush_buf is None:
            start.record()
            for _ in range(iters):
                fn()
            stop.record()
            stop.synchronize()
            values.append(start.elapsed_time(stop) / iters)
        else:
            total_ms = 0.0
            for _ in range(iters):
                flush_l2(flush_buf)
                start.record()
                fn()
                stop.record()
                stop.synchronize()
                total_ms += start.elapsed_time(stop)
            values.append(total_ms / iters)
    return summarize(values)


def sdpa(q, k, v, scale):
    try:
        return F.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, is_causal=False, scale=scale, enable_gqa=True
        )
    except TypeError:
        group = q.shape[1] // k.shape[1]
        return F.scaled_dot_product_attention(
            q,
            k.repeat_interleave(group, dim=1),
            v.repeat_interleave(group, dim=1),
            dropout_p=0.0,
            is_causal=False,
            scale=scale,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--q-heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=1)
    parser.add_argument("--head-dim", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--cache", choices=["warm", "cold"], default="warm")
    parser.add_argument("--flush-mib", type=int, default=64)
    args = parser.parse_args()

    torch.manual_seed(1234)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    scale = args.head_dim ** -0.5

    q = torch.randn(
        1, args.q_heads, 1, args.head_dim, device=device, dtype=dtype
    )
    k = torch.randn(
        1, args.kv_heads, args.seq_len, args.head_dim, device=device, dtype=dtype
    )
    v = torch.randn_like(k)
    new_k = torch.randn(
        1, args.kv_heads, 1, args.head_dim, device=device, dtype=dtype
    )
    new_v = torch.randn_like(new_k)
    flush_buf = None
    if args.cache == "cold":
        flush_buf = torch.empty(
            (args.flush_mib * 1024 * 1024) // 4,
            device=device,
            dtype=torch.int32,
        )
        flush_buf.zero_()

    def attention_only():
        out = sdpa(q, k, v, scale)
        return out

    def full_decode():
        k[:, :, -1:, :].copy_(new_k)
        v[:, :, -1:, :].copy_(new_v)
        return sdpa(q, k, v, scale)

    out = attention_only()
    torch.cuda.synchronize()
    checksum = float(out.float().sum().item())

    result = {
        "contract": {
            "timing": "CUDA-event PyTorch SDPA eager operation timing",
            "cache_mode": args.cache,
            "l2_flush_bytes": 0 if flush_buf is None else flush_buf.numel() * 4,
            "launch_overhead": "queued operation launches only",
            "host_wall_time": "excluded",
            "dtype": "bfloat16",
            "layout": "q=[B,Hq,1,D], k/v=[B,Hkv,S,D]",
            "warmup": args.warmup,
            "iters_per_sample": args.iters,
            "samples": args.samples,
            "min_effect_for_claim_pct": 5,
        },
        "env": {
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "shape": vars(args),
        "checksum": checksum,
        "attention_only": time_cuda(
            attention_only, args.warmup, args.iters, args.samples, flush_buf
        ),
        "full_decode_write_plus_attention": time_cuda(
            full_decode, args.warmup, args.iters, args.samples, flush_buf
        ),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
