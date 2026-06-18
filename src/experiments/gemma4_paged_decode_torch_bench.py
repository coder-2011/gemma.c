#!/usr/bin/env python3
import argparse
import ctypes
import json
import math
import os
import statistics
import time

import torch
import torch.nn.functional as F


GEMMA4_NUM_QUERY_HEADS = 32
GEMMA4_SLIDING_KV_HEADS = 16
GEMMA4_SLIDING_HEAD_DIM = 256
GEMMA4_SLIDING_WINDOW = 1024


class Gemma4KvCacheConfig(ctypes.Structure):
    _fields_ = [
        ("num_layers", ctypes.c_int32),
        ("num_pages", ctypes.c_int32),
        ("page_size", ctypes.c_int32),
        ("max_pages_per_seq", ctypes.c_int32),
        ("num_heads", ctypes.c_int32),
        ("head_dim", ctypes.c_int32),
        ("window_size", ctypes.c_int32),
    ]


def ptr(tensor):
    return ctypes.c_void_p(tensor.data_ptr())


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


def make_cuda_graph(fn, inner_iters):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(inner_iters):
            fn()
    return graph.replay


def flush_l2(flush_buf):
    if flush_buf is not None:
        flush_buf.zero_()


def time_cuda_graph(
    replay,
    warmup,
    iters,
    samples,
    cache_mode,
    flush_buf,
    ops_per_replay,
    sample_delay_s,
):
    for _ in range(warmup):
        flush_l2(flush_buf)
        replay()
    torch.cuda.synchronize()

    values = []
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    for _ in range(samples):
        time.sleep(sample_delay_s)
        if cache_mode == "cold":
            total_ms = 0.0
            for _ in range(iters):
                flush_l2(flush_buf)
                start.record()
                replay()
                stop.record()
                stop.synchronize()
                total_ms += start.elapsed_time(stop) / ops_per_replay
            values.append(total_ms / iters)
        else:
            start.record()
            replay()
            stop.record()
            stop.synchronize()
            values.append(start.elapsed_time(stop) / ops_per_replay)
    return summarize(values)


def load_lib(path):
    lib = ctypes.CDLL(path)

    kv_write = lib.gemma4_kv_cache_write_bf16
    kv_write.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        Gemma4KvCacheConfig,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    kv_write.restype = ctypes.c_int

    decode_direct = lib.gemma4_flash_attention_sliding_decode_paged_bf16
    decode_direct.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        Gemma4KvCacheConfig,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_float,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_void_p,
    ]
    decode_direct.restype = ctypes.c_int

    prefill = lib.gemma4_flash_attention_sliding_fwd_bf16
    prefill.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_float,
        ctypes.c_void_p,
    ]
    prefill.restype = ctypes.c_int
    return kv_write, decode_direct, prefill


def check_status(status, label):
    if status != 0:
        raise RuntimeError(f"{label} returned cudaError_t={status}")


def build_page_table(batch_size, seq_len, page_size):
    first_key = max(0, seq_len - GEMMA4_SLIDING_WINDOW)
    key_count = seq_len - first_key
    max_pages_per_seq = max(
        math.ceil(GEMMA4_SLIDING_WINDOW / page_size) + 1,
        math.ceil(key_count / page_size) + 1,
    )
    page_table = torch.full((batch_size, max_pages_per_seq), -1, dtype=torch.int32)
    next_page = 0
    positions = list(range(first_key, seq_len))
    for batch in range(batch_size):
        for pos in positions:
            slot = (pos // page_size) % max_pages_per_seq
            if page_table[batch, slot] < 0:
                page_table[batch, slot] = next_page
                next_page += 1
    return page_table, first_key, key_count, max_pages_per_seq, batch_size * max_pages_per_seq


def make_decode_inputs(args, device):
    dtype = torch.bfloat16
    torch.manual_seed(args.seed)
    page_table_cpu, first_key, key_count, max_pages_per_seq, num_pages = build_page_table(
        args.batch_size, args.seq_len, args.page_size
    )
    config = Gemma4KvCacheConfig(
        1,
        num_pages,
        args.page_size,
        max_pages_per_seq,
        GEMMA4_SLIDING_KV_HEADS,
        GEMMA4_SLIDING_HEAD_DIM,
        GEMMA4_SLIDING_WINDOW,
    )

    q = torch.randn(
        args.batch_size,
        GEMMA4_NUM_QUERY_HEADS,
        GEMMA4_SLIDING_HEAD_DIM,
        device=device,
        dtype=dtype,
    )
    k_window = torch.randn(
        args.batch_size,
        key_count,
        GEMMA4_SLIDING_KV_HEADS,
        GEMMA4_SLIDING_HEAD_DIM,
        device=device,
        dtype=dtype,
    )
    v_window = torch.randn_like(k_window)
    token_batch = torch.arange(args.batch_size, device=device, dtype=torch.int32).repeat_interleave(
        key_count
    )
    token_positions = torch.arange(first_key, args.seq_len, device=device, dtype=torch.int32).repeat(
        args.batch_size
    )
    k_flat = k_window.reshape(args.batch_size * key_count, GEMMA4_SLIDING_KV_HEADS, -1)
    v_flat = v_window.reshape_as(k_flat)
    cache_shape = (
        config.num_layers,
        config.num_pages,
        config.page_size,
        config.num_heads,
        config.head_dim,
    )
    return {
        "config": config,
        "first_key": first_key,
        "key_count": key_count,
        "page_table": page_table_cpu.to(device),
        "seq_lengths": torch.full((args.batch_size,), args.seq_len, device=device, dtype=torch.int32),
        "token_batch": token_batch.contiguous(),
        "token_positions": token_positions.contiguous(),
        "q": q.contiguous(),
        "k_window": k_window.contiguous(),
        "v_window": v_window.contiguous(),
        "k_flat": k_flat.contiguous(),
        "v_flat": v_flat.contiguous(),
        "cache_shape": cache_shape,
    }


def torch_decode_attention(q, k_window, v_window, out, scale):
    kv_map = torch.arange(GEMMA4_NUM_QUERY_HEADS, device=q.device) // (
        GEMMA4_NUM_QUERY_HEADS // GEMMA4_SLIDING_KV_HEADS
    )
    k_gqa = k_window[:, :, kv_map, :].float()
    v_gqa = v_window[:, :, kv_map, :].float()
    scores = (q[:, None, :, :].float() * k_gqa).sum(dim=-1) * scale
    probs = torch.softmax(scores, dim=1)
    out.copy_((probs[..., None] * v_gqa).sum(dim=1).to(torch.bfloat16))


def make_prefill_inputs(args, device):
    dtype = torch.bfloat16
    torch.manual_seed(args.seed + 1)
    q = torch.randn(
        args.batch_size,
        args.prefill_seq_len,
        GEMMA4_NUM_QUERY_HEADS,
        GEMMA4_SLIDING_HEAD_DIM,
        device=device,
        dtype=dtype,
    )
    k = torch.randn(
        args.batch_size,
        args.prefill_seq_len,
        GEMMA4_SLIDING_KV_HEADS,
        GEMMA4_SLIDING_HEAD_DIM,
        device=device,
        dtype=dtype,
    )
    v = torch.randn_like(k)
    return q.contiguous(), k.contiguous(), v.contiguous()


def torch_prefill_attention(q, k, v, out, scale):
    group = GEMMA4_NUM_QUERY_HEADS // GEMMA4_SLIDING_KV_HEADS
    q_t = q.permute(0, 2, 1, 3)
    k_t = k.permute(0, 2, 1, 3).repeat_interleave(group, dim=1)
    v_t = v.permute(0, 2, 1, 3).repeat_interleave(group, dim=1)
    result = F.scaled_dot_product_attention(
        q_t,
        k_t,
        v_t,
        dropout_p=0.0,
        is_causal=True,
        scale=scale,
    )
    out.copy_(result.permute(0, 2, 1, 3).contiguous())


def max_abs(a, b):
    return float((a.float() - b.float()).abs().max().item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lib", default="build/libgemma4_flash_attention.so")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--prefill-seq-len", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--split-size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--cache", choices=["warm", "cold"], default="warm")
    parser.add_argument("--flush-mib", type=int, default=128)
    parser.add_argument("--sample-delay-s", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    if args.sample_delay_s < 1.0:
        raise RuntimeError("--sample-delay-s must be at least 1.0")
    if args.prefill_seq_len <= 1:
        raise RuntimeError("--prefill-seq-len must be > 1 for the tensor-core section")
    if args.prefill_seq_len > GEMMA4_SLIDING_WINDOW:
        raise RuntimeError("--prefill-seq-len must be <= sliding window for this benchmark")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not os.path.exists(args.lib):
        raise RuntimeError(f"missing custom library: {args.lib}")

    device = torch.device("cuda")
    kv_write, decode_direct, prefill = load_lib(args.lib)
    stream = lambda: ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    scale = 1.0 / math.sqrt(GEMMA4_SLIDING_HEAD_DIM)

    decode = make_decode_inputs(args, device)
    config = decode["config"]
    key_count = decode["key_count"]
    num_splits = math.ceil(key_count / args.split_size)
    cache_k = torch.zeros(decode["cache_shape"], device=device, dtype=torch.bfloat16)
    cache_v = torch.zeros_like(cache_k)
    decode_out_direct = torch.empty_like(decode["q"])
    decode_out_torch = torch.empty_like(decode["q"])
    partial_m = torch.empty(
        args.batch_size * GEMMA4_NUM_QUERY_HEADS * num_splits,
        device=device,
        dtype=torch.float32,
    )
    partial_l = torch.empty_like(partial_m)
    partial_acc = torch.empty(
        args.batch_size * GEMMA4_NUM_QUERY_HEADS * num_splits * GEMMA4_SLIDING_HEAD_DIM,
        device=device,
        dtype=torch.float32,
    )

    check_status(
        kv_write(
            ptr(cache_k),
            ptr(cache_v),
            config,
            ptr(decode["page_table"]),
            ptr(decode["token_batch"]),
            ptr(decode["token_positions"]),
            args.batch_size * key_count,
            0,
            ptr(decode["k_flat"]),
            ptr(decode["v_flat"]),
            stream(),
        ),
        "gemma4_kv_cache_write_bf16",
    )

    def custom_decode_direct():
        check_status(
            decode_direct(
                ptr(decode_out_direct),
                ptr(partial_m),
                ptr(partial_l),
                ptr(partial_acc),
                ptr(decode["q"]),
                ptr(cache_k),
                ptr(cache_v),
                ptr(decode["page_table"]),
                ptr(decode["seq_lengths"]),
                config,
                0,
                args.batch_size,
                ctypes.c_float(scale),
                args.split_size,
                num_splits,
                stream(),
            ),
            "gemma4_flash_attention_sliding_decode_paged_bf16",
        )

    def torch_decode():
        torch_decode_attention(
            decode["q"],
            decode["k_window"],
            decode["v_window"],
            decode_out_torch,
            scale,
        )

    q_prefill, k_prefill, v_prefill = make_prefill_inputs(args, device)
    custom_prefill_out = torch.empty_like(q_prefill)
    torch_prefill_out = torch.empty_like(q_prefill)

    def custom_prefill():
        check_status(
            prefill(
                ptr(custom_prefill_out),
                ctypes.c_void_p(0),
                ptr(q_prefill),
                ptr(k_prefill),
                ptr(v_prefill),
                args.batch_size,
                args.prefill_seq_len,
                args.prefill_seq_len,
                GEMMA4_SLIDING_WINDOW,
                ctypes.c_float(scale),
                stream(),
            ),
            "gemma4_flash_attention_sliding_fwd_bf16",
        )

    def torch_prefill():
        torch_prefill_attention(q_prefill, k_prefill, v_prefill, torch_prefill_out, scale)

    with torch.no_grad():
        torch_decode()
        custom_decode_direct()
        torch_prefill()
        custom_prefill()
        torch.cuda.synchronize()

        correctness = {
            "decode_direct_vs_torch_max_abs": max_abs(decode_out_direct, decode_out_torch),
            "prefill_custom_vs_torch_max_abs": max_abs(custom_prefill_out, torch_prefill_out),
        }

        flush_buf = None
        if args.cache == "cold":
            flush_buf = torch.empty(
                (args.flush_mib * 1024 * 1024) // 4,
                device=device,
                dtype=torch.int32,
            )
            flush_buf.zero_()

        graph_inner_iters = 1 if args.cache == "cold" else args.iters
        graphs = {
            "decode_custom_direct": make_cuda_graph(custom_decode_direct, graph_inner_iters),
            "decode_torch_graph": make_cuda_graph(torch_decode, graph_inner_iters),
            "prefill_custom_tensor_core": make_cuda_graph(custom_prefill, graph_inner_iters),
            "prefill_torch_sdpa_graph": make_cuda_graph(torch_prefill, graph_inner_iters),
        }

        timings = {
            name: time_cuda_graph(
                replay,
                args.warmup,
                args.iters,
                args.samples,
                args.cache,
                flush_buf,
                graph_inner_iters,
                args.sample_delay_s,
            )
            for name, replay in graphs.items()
        }
        torch.cuda.synchronize()
        checksum = float(
            decode_out_direct.float().sum().item()
            + decode_out_torch.float().sum().item()
            + custom_prefill_out.float().sum().item()
            + torch_prefill_out.float().sum().item()
        )

    result = {
        "contract": {
            "timing": "CUDA events on current PyTorch CUDA stream",
            "execution": "CUDA graph replay for custom CUDA and PyTorch paths",
            "cache_mode": args.cache,
            "l2_flush_bytes": 0 if flush_buf is None else flush_buf.numel() * 4,
            "sample_delay_s": args.sample_delay_s,
            "delay_location": "host sleep before each measured sample, outside event window",
            "launch_overhead": "excluded by CUDA graph replay",
            "host_wall_time": "excluded from elapsed timings",
            "warmup": args.warmup,
            "iters_per_sample": args.iters,
            "graph_inner_iters": graph_inner_iters,
            "samples": args.samples,
            "min_effect_for_claim_pct": 5,
        },
        "env": {
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "custom_lib": args.lib,
        },
        "shape": {
            "batch_size": args.batch_size,
            "decode_q_len": 1,
            "decode_seq_len": args.seq_len,
            "decode_key_count": key_count,
            "page_size": args.page_size,
            "max_pages_per_seq": config.max_pages_per_seq,
            "num_pages": config.num_pages,
            "split_size": args.split_size,
            "num_splits": num_splits,
            "prefill_seq_len": args.prefill_seq_len,
            "q_heads": GEMMA4_NUM_QUERY_HEADS,
            "kv_heads": GEMMA4_SLIDING_KV_HEADS,
            "head_dim": GEMMA4_SLIDING_HEAD_DIM,
        },
        "correctness": correctness,
        "checksum": checksum,
        "timings": timings,
        "speedups": {
            "decode_direct_vs_torch_median": timings["decode_torch_graph"]["median_ms"]
            / timings["decode_custom_direct"]["median_ms"],
            "prefill_custom_vs_torch_median": timings["prefill_torch_sdpa_graph"]["median_ms"]
            / timings["prefill_custom_tensor_core"]["median_ms"],
        },
    }
    text = json.dumps(result, indent=2)
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
