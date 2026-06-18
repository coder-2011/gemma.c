#!/usr/bin/env python3
import argparse
import ctypes
import json
import math
import os
import statistics
import time

import torch


GEMMA4_NUM_QUERY_HEADS = 32
GEMMA4_SLIDING_KV_HEADS = 16
GEMMA4_SLIDING_HEAD_DIM = 256
GEMMA4_GLOBAL_KV_HEADS = 4
GEMMA4_GLOBAL_HEAD_DIM = 512
GEMMA4_GLOBAL_ROTARY_DIM = 128
GEMMA4_RMS_NORM_EPS = 1.0e-6
GEMMA4_ROPE_THETA_SLIDING = 10000.0
GEMMA4_ROPE_THETA_GLOBAL = 1000000.0


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
        flush_buf.zero_()


def make_enqueue_delay(device, sleep_cycles):
    if hasattr(torch.cuda, "_sleep"):
        def delay():
            torch.cuda._sleep(sleep_cycles)

        return delay, f"torch.cuda._sleep({sleep_cycles})"

    a = torch.randn((512, 512), device=device)
    b = torch.randn((512, 512), device=device)
    out = torch.empty_like(a)

    def delay():
        torch.mm(a, b, out=out)

    return delay, "fallback untimed 512x512 FP32 matmul"


def make_cuda_graph(fn, inner_iters):
    # Warm the exact static allocations before capture.
    for _ in range(3):
        fn()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(inner_iters):
            fn()
    return graph.replay


def time_cuda_graph(replay, warmup, iters, samples, flush_buf, enqueue_delay, ops_per_replay):
    for _ in range(warmup):
        flush_l2(flush_buf)
        replay()
    torch.cuda.synchronize()

    values = []
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    for _ in range(samples):
        if flush_buf is None:
            enqueue_delay()
            start.record()
            replay()
            stop.record()
            stop.synchronize()
            values.append(start.elapsed_time(stop) / ops_per_replay)
        else:
            total_ms = 0.0
            for _ in range(iters):
                flush_l2(flush_buf)
                enqueue_delay()
                start.record()
                replay()
                stop.record()
                stop.synchronize()
                total_ms += start.elapsed_time(stop) / ops_per_replay
            values.append(total_ms / iters)
    return summarize(values)


def layer_spec(layer_type):
    if layer_type == "sliding":
        return {
            "kv_heads": GEMMA4_SLIDING_KV_HEADS,
            "head_dim": GEMMA4_SLIDING_HEAD_DIM,
            "rotary_dim": GEMMA4_SLIDING_HEAD_DIM,
            "theta": GEMMA4_ROPE_THETA_SLIDING,
            "window_size": 1024,
        }
    return {
        "kv_heads": GEMMA4_GLOBAL_KV_HEADS,
        "head_dim": GEMMA4_GLOBAL_HEAD_DIM,
        "rotary_dim": GEMMA4_GLOBAL_ROTARY_DIM,
        "theta": GEMMA4_ROPE_THETA_GLOBAL,
        "window_size": 0,
    }


def load_custom_lib(path, layer_type):
    lib = ctypes.CDLL(path)
    if layer_type == "sliding":
        fn = lib.gemma4_flash_attention_sliding_decode_prepare_q_paged_kv_bf16
        fn.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            Gemma4KvCacheConfig,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
    else:
        fn = lib.gemma4_flash_attention_global_decode_prepare_q_paged_kv_bf16
        fn.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            Gemma4KvCacheConfig,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
    fn.restype = ctypes.c_int
    return lib, fn


def ptr(tensor):
    return ctypes.c_void_p(tensor.data_ptr())


def fill_rope_tables(seq_len, spec, device):
    half = spec["rotary_dim"] // 2
    row = torch.arange(seq_len, device=device, dtype=torch.float32)[:, None]
    dim = torch.arange(half, device=device, dtype=torch.float32)[None, :]
    freq = torch.pow(
        torch.tensor(spec["theta"], device=device),
        -(2.0 * dim) / float(spec["head_dim"]),
    )
    angle = row * freq
    return torch.cos(angle).contiguous(), torch.sin(angle).contiguous()


def rmsnorm(x, weight):
    y = x.float()
    inv_rms = torch.rsqrt(y.square().mean(dim=-1, keepdim=True) + GEMMA4_RMS_NORM_EPS)
    y = y * inv_rms
    if weight is not None:
        y = y * weight.float().view(1, 1, -1)
    return y


def apply_rope(x, cos, sin, token_position, spec):
    rotary_half = spec["rotary_dim"] // 2
    c = cos.index_select(0, token_position.long())[:, None, :]
    s = sin.index_select(0, token_position.long())[:, None, :]
    lo = x[..., :rotary_half]
    hi = x[..., rotary_half:spec["rotary_dim"]]
    out = x.clone()
    out[..., :rotary_half] = lo * c - hi * s
    out[..., rotary_half:spec["rotary_dim"]] = lo * s + hi * c
    return out.to(torch.bfloat16)


def torch_prep_cache_body(
    q,
    k,
    v,
    q_weight,
    k_weight,
    cos,
    sin,
    token_position,
    page_table,
    batch_index,
    page_size,
    max_pages_per_seq,
    q_prepared,
    cache_k,
    cache_v,
    spec,
    global_layer,
):
    q_out = apply_rope(rmsnorm(q, q_weight), cos, sin, token_position, spec)
    k_out = apply_rope(rmsnorm(k, k_weight), cos, sin, token_position, spec)
    v_source = k if global_layer else v
    v_out = rmsnorm(v_source, None).to(torch.bfloat16)
    q_prepared.copy_(q_out)

    logical_page = torch.div(token_position, page_size, rounding_mode="floor")
    slot = torch.remainder(logical_page, max_pages_per_seq)
    physical_page = page_table[batch_index, slot.long()].long()
    page_offset = torch.remainder(token_position, page_size).long()
    cache_k[0, physical_page, page_offset, :, :] = k_out
    cache_v[0, physical_page, page_offset, :, :] = v_out


def max_abs(a, b):
    return float((a.float() - b.float()).abs().max().item())


def make_inputs(args, device):
    dtype = torch.bfloat16
    spec = layer_spec(args.layer_type)
    torch.manual_seed(1234)
    q = torch.randn(
        args.batch_size,
        GEMMA4_NUM_QUERY_HEADS,
        spec["head_dim"],
        device=device,
        dtype=dtype,
    )
    k = torch.randn(
        args.batch_size,
        spec["kv_heads"],
        spec["head_dim"],
        device=device,
        dtype=dtype,
    )
    v = torch.randn_like(k)
    q_weight = (0.95 + 0.05 * torch.randn(spec["head_dim"], device=device)).to(dtype)
    k_weight = (0.95 + 0.05 * torch.randn(spec["head_dim"], device=device)).to(dtype)
    cos, sin = fill_rope_tables(args.seq_len, spec, device)

    pages_per_seq = max(1, (args.seq_len + args.page_size - 1) // args.page_size)
    num_pages = args.batch_size * pages_per_seq
    token_position = torch.full(
        (args.batch_size,),
        args.seq_len - 1,
        device=device,
        dtype=torch.int32,
    )
    page_table_cpu = torch.full(
        (args.batch_size, pages_per_seq),
        -1,
        dtype=torch.int32,
    )
    slot = ((args.seq_len - 1) // args.page_size) % pages_per_seq
    for batch in range(args.batch_size):
        page_table_cpu[batch, slot] = batch * pages_per_seq + slot
    page_table = page_table_cpu.to(device)
    config = Gemma4KvCacheConfig(
        1,
        num_pages,
        args.page_size,
        pages_per_seq,
        spec["kv_heads"],
        spec["head_dim"],
        spec["window_size"],
    )

    cache_shape = (
        config.num_layers,
        config.num_pages,
        config.page_size,
        config.num_heads,
        config.head_dim,
    )
    return {
        "q": q,
        "k": k,
        "v": v,
        "q_weight": q_weight,
        "k_weight": k_weight,
        "cos": cos,
        "sin": sin,
        "token_position": token_position,
        "page_table": page_table,
        "config": config,
        "cache_shape": cache_shape,
        "spec": spec,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lib", default="build/libgemma4_flash_attention.so")
    parser.add_argument("--layer-type", choices=["sliding", "global"], default="sliding")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--cache", choices=["warm", "cold"], default="cold")
    parser.add_argument("--paths", choices=["all", "custom"], default="all")
    parser.add_argument("--flush-mib", type=int, default=128)
    parser.add_argument("--sleep-cycles", type=int, default=1_000_000)
    parser.add_argument("--torch-compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not os.path.exists(args.lib):
        raise RuntimeError(f"missing custom library: {args.lib}")

    device = torch.device("cuda")
    lib, custom_fn = load_custom_lib(args.lib, args.layer_type)
    tensors = make_inputs(args, device)
    config = tensors["config"]
    spec = tensors["spec"]

    q_prepared_custom = torch.empty(
        args.batch_size,
        GEMMA4_NUM_QUERY_HEADS,
        spec["head_dim"],
        device=device,
        dtype=torch.bfloat16,
    )
    cache_k_custom = torch.zeros(tensors["cache_shape"], device=device, dtype=torch.bfloat16)
    cache_v_custom = torch.zeros_like(cache_k_custom)
    q_prepared_torch = torch.empty_like(q_prepared_custom)
    cache_k_torch = torch.zeros_like(cache_k_custom)
    cache_v_torch = torch.zeros_like(cache_v_custom)
    batch_index = torch.arange(args.batch_size, device=device)
    torch_prep_cache_impl = torch_prep_cache_body
    torch_compile_mode = "disabled"
    if args.torch_compile:
        torch_prep_cache_impl = torch.compile(
            torch_prep_cache_body,
            mode="reduce-overhead",
            fullgraph=False,
        )
        torch_compile_mode = "torch.compile(mode='reduce-overhead')"

    def custom_prep_cache():
        stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        common_args = [
            ptr(q_prepared_custom),
            ptr(cache_k_custom),
            ptr(cache_v_custom),
            config,
            ptr(tensors["page_table"]),
            ptr(tensors["token_position"]),
            args.batch_size,
            0,
            ptr(tensors["q"]),
            ptr(tensors["k"]),
        ]
        if args.layer_type == "sliding":
            common_args.append(ptr(tensors["v"]))
        common_args.extend([
            ptr(tensors["q_weight"]),
            ptr(tensors["k_weight"]),
            ptr(tensors["cos"]),
            ptr(tensors["sin"]),
            stream,
        ])
        status = custom_fn(*common_args)
        if status != 0:
            raise RuntimeError(f"custom kernel returned cudaError_t={status}")

    def torch_prep_cache():
        torch_prep_cache_impl(
            tensors["q"],
            tensors["k"],
            tensors["v"],
            tensors["q_weight"],
            tensors["k_weight"],
            tensors["cos"],
            tensors["sin"],
            tensors["token_position"],
            tensors["page_table"],
            batch_index,
            args.page_size,
            config.max_pages_per_seq,
            q_prepared_torch,
            cache_k_torch,
            cache_v_torch,
            spec,
            args.layer_type == "global",
        )

    with torch.no_grad():
        torch_prep_cache()
        custom_prep_cache()
        torch.cuda.synchronize()
        correctness = {
            "q_max_abs": max_abs(q_prepared_custom, q_prepared_torch),
            "cache_k_max_abs": max_abs(cache_k_custom, cache_k_torch),
            "cache_v_max_abs": max_abs(cache_v_custom, cache_v_torch),
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
        custom_graph = make_cuda_graph(custom_prep_cache, graph_inner_iters)
        torch_graph = None
        if args.paths == "all":
            torch_graph = make_cuda_graph(torch_prep_cache, graph_inner_iters)
        enqueue_delay, enqueue_delay_description = make_enqueue_delay(
            device, args.sleep_cycles
        )
        # Run once so the untimed delay path itself is initialized.
        enqueue_delay()
        torch.cuda.synchronize()

        custom_stats = time_cuda_graph(
            custom_graph,
            args.warmup,
            args.iters,
            args.samples,
            flush_buf,
            enqueue_delay,
            graph_inner_iters,
        )
        torch_stats = None
        if args.paths == "all":
            torch_stats = time_cuda_graph(
                torch_graph,
                args.warmup,
                args.iters,
                args.samples,
                flush_buf,
                enqueue_delay,
                graph_inner_iters,
            )
        torch.cuda.synchronize()
        checksum = float(
            q_prepared_custom.float().sum().item()
            + cache_k_custom.float().sum().item()
            + cache_v_custom.float().sum().item()
        )

    result = {
        "contract": {
            "timing": "CUDA-event timing on the current PyTorch CUDA stream",
            "execution": "CUDA graph replay; PyTorch work captured after optional torch.compile",
            "cache_mode": args.cache,
            "l2_flush_bytes": 0 if flush_buf is None else flush_buf.numel() * 4,
            "launch_overhead": (
                "excluded from per-op timing by CUDA graph replay "
                "and batched warm replay"
            ),
            "enqueue_delay": f"untimed {enqueue_delay_description} before start event",
            "host_wall_time": "excluded",
            "dtype": "bfloat16",
            "layout": (
                "raw q=[B,32,D], raw k/v=[B,KV,D], "
                "cache=[1,pages,page,KV,D]"
            ),
            "warmup": args.warmup,
            "iters_per_sample": args.iters,
            "graph_inner_iters": graph_inner_iters,
            "sleep_cycles": args.sleep_cycles,
            "samples": args.samples,
            "timed_paths": args.paths,
            "min_effect_for_claim_pct": 5,
        },
        "env": {
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "custom_lib": args.lib,
            "torch_compile": torch_compile_mode,
        },
        "shape": {
            "layer_type": args.layer_type,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "page_size": args.page_size,
            "pages_per_seq": config.max_pages_per_seq,
            "num_pages": config.num_pages,
            "q_heads": GEMMA4_NUM_QUERY_HEADS,
            "kv_heads": spec["kv_heads"],
            "head_dim": spec["head_dim"],
            "rotary_dim": spec["rotary_dim"],
            "rope_theta": spec["theta"],
        },
        "correctness": correctness,
        "checksum": checksum,
        "custom_cuda_graph_decode_norm_rope_paged_kv_write": custom_stats,
    }
    if torch_stats is not None:
        result["torch_non_eager_decode_norm_rope_paged_kv_write"] = torch_stats
        result["speedup_median"] = (
            torch_stats["median_ms"] / custom_stats["median_ms"]
        )
    text = json.dumps(result, indent=2)
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
