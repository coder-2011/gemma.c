#!/usr/bin/env python3
import argparse
import ctypes
import json
import math
import statistics
import subprocess
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
Q_HEADS = 32
KV_HEADS = 4
HEAD_DIM = 512
GQA = Q_HEADS // KV_HEADS


class KvConfig(ctypes.Structure):
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
    values = sorted(values)
    index = pct * 0.01 * (len(values) - 1)
    lo = math.floor(index)
    hi = math.ceil(index)
    frac = index - lo
    return values[lo] * (1.0 - frac) + values[hi] * frac


def summarize(values):
    values = list(values)
    values_sorted = sorted(values)
    trim = math.floor(len(values_sorted) * 0.1)
    trimmed = values_sorted[trim : len(values_sorted) - trim] or values_sorted
    return {
        "median_ms": statistics.median(values),
        "mean_ms": statistics.fmean(values),
        "trimmed_mean_ms": statistics.fmean(trimmed),
        "min_ms": min(values),
        "max_ms": max(values),
        "p95_ms": percentile(values, 95.0),
        "p99_ms": percentile(values, 99.0),
        "stddev_ms": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "iqr_ms": percentile(values, 75.0) - percentile(values, 25.0),
        "samples_ms": values,
    }


def run(cmd, cwd=ROOT):
    return subprocess.check_output(cmd, cwd=cwd, text=True, stderr=subprocess.STDOUT).strip()


def try_run(cmd, cwd=ROOT):
    try:
        return run(cmd, cwd)
    except Exception as exc:
        return f"unavailable: {exc}"


def load_decode_kernel():
    build_cmd = [
        "make",
        "-B",
        "build/libgemma4_flash_attention.so",
        "NVCC=/usr/local/cuda/bin/nvcc",
    ]
    subprocess.run(build_cmd, cwd=ROOT, check=True)
    lib = ctypes.CDLL(str(ROOT / "build/libgemma4_flash_attention.so"))
    fn = lib.gemma4_flash_attention_decode_paged_bf16
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        KvConfig,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_float,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    return fn


def make_case(args):
    device = "cuda"
    torch.manual_seed(args.seed)
    pages = math.ceil(args.seq_len / args.page_size)
    cfg = KvConfig(1, args.batch_size * pages, args.page_size, pages, KV_HEADS, HEAD_DIM, 0)

    page_table = torch.arange(
        args.batch_size * pages, device=device, dtype=torch.int32
    ).view(args.batch_size, pages)
    seq_lengths = torch.full(
        (args.batch_size,), args.seq_len, device=device, dtype=torch.int32
    )
    q = torch.randn(
        args.batch_size, Q_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16
    )
    k_dense = torch.randn(
        args.batch_size, args.seq_len, KV_HEADS, HEAD_DIM,
        device=device, dtype=torch.bfloat16
    )
    v_dense = torch.randn_like(k_dense)

    cache_shape = (1, cfg.num_pages, args.page_size, KV_HEADS, HEAD_DIM)
    cache_k = torch.empty(cache_shape, device=device, dtype=torch.bfloat16)
    cache_v = torch.empty_like(cache_k)
    positions = torch.arange(args.seq_len, device=device)
    slots = positions // args.page_size
    offsets = positions % args.page_size
    for batch in range(args.batch_size):
        pages_for_batch = page_table[batch, slots].long()
        cache_k[0, pages_for_batch, offsets] = k_dense[batch]
        cache_v[0, pages_for_batch, offsets] = v_dense[batch]

    num_splits = math.ceil(args.seq_len / args.split_size)
    partial_shape = (args.batch_size, Q_HEADS, num_splits)
    out = torch.empty(args.batch_size, Q_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16)
    ref = torch.empty_like(out)
    partial_m = torch.empty(partial_shape, device=device, dtype=torch.float32)
    partial_l = torch.empty_like(partial_m)
    partial_acc = torch.empty(
        args.batch_size, Q_HEADS, num_splits, HEAD_DIM,
        device=device, dtype=torch.float32
    )
    return {
        "cfg": cfg,
        "page_table": page_table,
        "seq_lengths": seq_lengths,
        "q": q,
        "cache_k": cache_k,
        "cache_v": cache_v,
        "out": out,
        "ref": ref,
        "partial_m": partial_m,
        "partial_l": partial_l,
        "partial_acc": partial_acc,
        "positions": positions,
        "slots": slots,
        "offsets": offsets,
        "num_splits": num_splits,
    }


def custom_decode(fn, case, args):
    status = fn(
        ptr(case["out"]),
        ptr(case["partial_m"]),
        ptr(case["partial_l"]),
        ptr(case["partial_acc"]),
        ptr(case["q"]),
        ptr(case["cache_k"]),
        ptr(case["cache_v"]),
        ptr(case["page_table"]),
        ptr(case["seq_lengths"]),
        case["cfg"],
        0,
        args.batch_size,
        ctypes.c_float(args.softmax_scale),
        args.split_size,
        case["num_splits"],
        ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),
    )
    if status:
        raise RuntimeError(f"gemma4_flash_attention_decode_paged_bf16 returned {status}")


def pytorch_decode(case, args):
    table = case["page_table"].long()
    page = table[:, case["slots"].long()]
    offset = case["offsets"].view(1, -1)
    k = case["cache_k"][0, page, offset]
    v = case["cache_v"][0, page, offset]
    groups = []
    for kv_head in range(KV_HEADS):
        q = case["q"][:, kv_head * GQA : (kv_head + 1) * GQA].float()
        kg = k[:, :, kv_head].float()
        vg = v[:, :, kv_head].float()
        scores = torch.einsum("bgd,bsd->bgs", q, kg) * args.softmax_scale
        probs = torch.softmax(scores, dim=-1)
        groups.append(torch.einsum("bgs,bsd->bgd", probs, vg))
    case["ref"].copy_(torch.cat(groups, dim=1).to(torch.bfloat16))


def make_graph(fn, inner_iters):
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


def time_replay(replay, args, flush_buf=None, ops_per_replay=1):
    for _ in range(args.warmup):
        flush_l2(flush_buf)
        replay()
    torch.cuda.synchronize()

    samples = []
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    for _ in range(args.samples):
        time.sleep(args.sample_delay_s)
        total_ms = 0.0
        for _ in range(args.iters):
            flush_l2(flush_buf)
            start.record()
            replay()
            stop.record()
            stop.synchronize()
            total_ms += start.elapsed_time(stop) / ops_per_replay
        samples.append(total_ms / args.iters)
    return summarize(samples)


def time_empty(args, flush_buf=None):
    for _ in range(args.warmup):
        flush_l2(flush_buf)
    torch.cuda.synchronize()

    samples = []
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    for _ in range(args.samples):
        time.sleep(args.sample_delay_s)
        total_ms = 0.0
        for _ in range(args.iters):
            flush_l2(flush_buf)
            start.record()
            stop.record()
            stop.synchronize()
            total_ms += start.elapsed_time(stop)
        samples.append(total_ms / args.iters)
    return summarize(samples)


def env_snapshot():
    query = (
        "name,gpu_bus_id,driver_version,persistence_mode,ecc.mode.current,"
        "mig.mode.current,power.limit,power.draw,clocks.sm,clocks.mem,"
        "temperature.gpu,pstate,utilization.gpu,memory.total"
    )
    return {
        "nvidia_smi": try_run([
            "nvidia-smi",
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
        ]),
        "nvcc": try_run(["/usr/local/cuda/bin/nvcc", "--version"]),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(),
        "git_head": try_run(["git", "rev-parse", "HEAD"]),
        "git_status_short": try_run(["git", "status", "--short"]),
    }


def add_corrected(summary, timer_overhead_ms):
    corrected = max(0.0, summary["median_ms"] - timer_overhead_ms)
    summary["timer_overhead_ms"] = timer_overhead_ms
    summary["corrected_median_ms"] = corrected


def print_table(timings, raw_speedup, corrected_speedup):
    print("\n| path | median ms | corrected median ms | IQR ms | p95 ms | raw speedup | corrected speedup |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for name, label in [
        ("pytorch_global_decode_graph", "PyTorch global decode graph"),
        ("custom_global_decode", "custom global decode"),
    ]:
        row = timings[name]
        raw = "baseline" if name.startswith("pytorch") else f"{raw_speedup:.2f}x"
        corrected = "baseline" if name.startswith("pytorch") else f"{corrected_speedup:.2f}x"
        print(
            f"| {label} | {row['median_ms']:.6f} | "
            f"{row['corrected_median_ms']:.6f} | {row['iqr_ms']:.6f} | "
            f"{row['p95_ms']:.6f} | {raw} | {corrected} |"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--split-size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--cache", choices=["cold", "warm"], default="cold")
    parser.add_argument("--flush-mib", type=int, default=128)
    parser.add_argument("--sample-delay-s", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--correctness-atol", type=float, default=0.002)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.seq_len % args.page_size != 0:
        raise RuntimeError("--seq-len must be a multiple of --page-size for this simple harness")

    args.softmax_scale = 1.0 / math.sqrt(HEAD_DIM)
    fn = load_decode_kernel()
    case = make_case(args)
    flush_buf = None
    if args.cache == "cold":
        flush_buf = torch.empty(args.flush_mib * 1024 * 1024 // 4, device="cuda", dtype=torch.int32)

    custom = lambda: custom_decode(fn, case, args)
    pytorch = lambda: pytorch_decode(case, args)
    custom()
    pytorch()
    torch.cuda.synchronize()
    max_abs = (case["out"].float() - case["ref"].float()).abs().max().item()
    print(f"custom_vs_pytorch max_abs={max_abs:.8f}")
    if max_abs > args.correctness_atol:
        raise RuntimeError(f"correctness failed: max_abs={max_abs}")

    inner_iters = 1 if args.cache == "cold" else args.iters
    custom_replay = make_graph(custom, inner_iters)
    pytorch_replay = make_graph(pytorch, inner_iters)
    overhead = time_empty(args, flush_buf)
    overhead_ms = overhead["median_ms"]
    custom_stats = time_replay(custom_replay, args, flush_buf, inner_iters)
    pytorch_stats = time_replay(pytorch_replay, args, flush_buf, inner_iters)
    add_corrected(custom_stats, overhead_ms)
    add_corrected(pytorch_stats, overhead_ms)
    raw_speedup = pytorch_stats["median_ms"] / custom_stats["median_ms"]
    corrected_speedup = pytorch_stats["corrected_median_ms"] / custom_stats["corrected_median_ms"]

    timings = {
        "timer_empty_event_pair": overhead,
        "custom_global_decode": custom_stats,
        "pytorch_global_decode_graph": pytorch_stats,
    }
    print_table(timings, raw_speedup, corrected_speedup)

    report = {
        "contract": {
            "measurement": "single-token global paged decode attention latency",
            "timing": "CUDA events on current PyTorch CUDA stream",
            "execution": "CUDA graph replay for custom CUDA and PyTorch paths",
            "cache_mode": args.cache,
            "l2_flush_bytes": args.flush_mib * 1024 * 1024 if args.cache == "cold" else 0,
            "flush_timing": "flush enqueued before start event and excluded from elapsed time",
            "sample_delay_s": args.sample_delay_s,
            "delay_location": "host sleep before each measured sample, outside event window",
            "launch_overhead": "CPU launch overhead excluded by CUDA graph replay and CUDA events",
            "timer_overhead": "empty CUDA event-pair median reported and subtracted in corrected medians",
            "warmup": args.warmup,
            "iters_per_sample": args.iters,
            "graph_inner_iters": inner_iters,
            "samples": args.samples,
            "min_effect_for_claim_pct": 5,
        },
        "shape": {
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "page_size": args.page_size,
            "split_size": args.split_size,
            "num_splits": case["num_splits"],
            "q_heads": Q_HEADS,
            "kv_heads": KV_HEADS,
            "head_dim": HEAD_DIM,
            "dtype": "bf16",
        },
        "environment": env_snapshot(),
        "correctness": {"custom_vs_pytorch_max_abs": max_abs, "atol": args.correctness_atol},
        "timings": timings,
        "speedups": {
            "custom_vs_pytorch_raw_median": raw_speedup,
            "custom_vs_pytorch_corrected_median": corrected_speedup,
        },
        "threats": [
            "clocks were recorded but not locked",
            "single process on one GPU only",
            "synthetic L2 flush is best-effort and not a serving trace",
            "PyTorch baseline is graph-captured operator code, not an end-to-end server path",
        ],
    }
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
