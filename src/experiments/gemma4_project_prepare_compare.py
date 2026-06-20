#!/usr/bin/env python3
import argparse
import ctypes
import importlib
import json
import statistics
import subprocess
from pathlib import Path

import torch

H, QH, KVH, D = 3840, 16, 8, 256
QS, KVS, QKVS = QH * D, KVH * D, (QH + 2 * KVH) * D
EPS, THETA = 1e-6, 10000.0


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


def addr(t):
    return ctypes.c_void_p(t.data_ptr())


def rms(x, w=None):
    y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + EPS)
    return (y if w is None else y * w.float()).to(torch.bfloat16)


def rope(x, pos, cos, sin):
    half = D // 2
    c, s = cos[pos].view(x.shape[0], 1, half), sin[pos].view(x.shape[0], 1, half)
    lo, hi = x[..., :half].float(), x[..., half:].float()
    return torch.cat((lo * c - hi * s, lo * s + hi * c), -1).to(torch.bfloat16)


def put_cache(cache_k, cache_v, k, v, table, pos):
    b = torch.arange(k.shape[0], device=k.device)
    page = table[b, pos // cache_k.shape[2]]
    off = pos % cache_k.shape[2]
    cache_k[0, page, off], cache_v[0, page, off] = k, v


def load_kernel():
    root = Path(__file__).resolve().parents[2]
    so = root / "build/libgemma4_flash_attention.so"
    subprocess.run(
        ["make", "-B", "build/libgemma4_flash_attention.so", "NVCC=/usr/local/cuda/bin/nvcc"],
        cwd=root,
        check=True,
    )
    fn = ctypes.CDLL(str(so)).gemma4_flash_attention_sliding_decode_project_prepare_paged_kv_bf16
    fn.restype = ctypes.c_int
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, KvConfig,
                   ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32,
                   ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                   ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    return fn


def make_case(batch, seq, page_size):
    device = "cuda"
    torch.manual_seed(0)
    pages = (seq + page_size - 1) // page_size
    cfg = KvConfig(1, batch * pages, page_size, pages, KVH, D, min(seq, 1024))
    pos = torch.full((batch,), seq - 1, device=device, dtype=torch.int32)
    table = torch.arange(batch * pages, device=device, dtype=torch.int32).view(batch, pages)
    dim = torch.arange(D // 2, device=device)
    phase = torch.arange(seq, device=device)[:, None] * (THETA ** (-(2 * dim) / D))[None, :]
    cache_shape = (1, cfg.num_pages, page_size, KVH, D)
    return cfg, pos, table, phase.cos(), phase.sin(), {
        "x": torch.randn(batch, H, device=device, dtype=torch.bfloat16),
        "w": (torch.randn(QKVS, H, device=device, dtype=torch.bfloat16) / H**0.5).contiguous(),
        "qw": torch.randn(D, device=device, dtype=torch.bfloat16) * 0.01 + 1,
        "kw": torch.randn(D, device=device, dtype=torch.bfloat16) * 0.01 + 1,
        "q": torch.empty(batch, QH, D, device=device, dtype=torch.bfloat16),
        "ck": torch.zeros(cache_shape, device=device, dtype=torch.bfloat16),
        "cv": torch.zeros(cache_shape, device=device, dtype=torch.bfloat16),
        "qr": torch.empty(batch, QH, D, device=device, dtype=torch.bfloat16),
        "rck": torch.zeros(cache_shape, device=device, dtype=torch.bfloat16),
        "rcv": torch.zeros(cache_shape, device=device, dtype=torch.bfloat16),
    }


def pytorch_case(t, pos, table, cos, sin, vllm_ops=None):
    qkv = t["x"] @ t["w"].t()
    q = qkv[:, :QS].reshape(-1, QH, D)
    k = qkv[:, QS:QS + KVS].reshape(-1, KVH, D)
    v = qkv[:, QS + KVS:].reshape(-1, KVH, D)
    norm = rms
    if vllm_ops:
        def norm(x, w=None):
            flat = x.reshape(-1, D).contiguous()
            out = torch.empty_like(flat)
            vllm_ops.rms_norm(out, flat, torch.ones(D, device=x.device, dtype=x.dtype) if w is None else w, EPS)
            return out.view_as(x)
    t["qr"].copy_(rope(norm(q, t["qw"]), pos, cos, sin))
    put_cache(t["rck"], t["rcv"], rope(norm(k, t["kw"]), pos, cos, sin), norm(v), table, pos)


def custom_case(fn, cfg, t, pos, table, cos, sin):
    err = fn(addr(t["q"]), addr(t["ck"]), addr(t["cv"]), cfg, addr(table), addr(pos),
             t["x"].shape[0], 0, addr(t["x"]), addr(t["w"]), addr(t["qw"]),
             addr(t["kw"]), addr(cos), addr(sin),
             ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
    if err:
        raise RuntimeError(f"cudaError_t={err}")


def bench(f, warmup, iters, samples):
    for _ in range(warmup):
        f()
    torch.cuda.synchronize()
    times = []
    for _ in range(samples):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record()
        for _ in range(iters):
            f()
        b.record(); b.synchronize()
        times.append(a.elapsed_time(b) / iters)
    return times


def pct(xs, q):
    xs = sorted(xs)
    i = (len(xs) - 1) * q
    lo, hi = int(i), min(int(i) + 1, len(xs) - 1)
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - i) + xs[hi] * (i - lo)


def stats(xs):
    ys = sorted(xs)
    mid = ys[1:-1] if len(ys) > 2 else ys
    return {
        "samples_ms": xs,
        "median_ms": statistics.median(xs),
        "trimmed_mean_ms": statistics.fmean(mid),
        "min_ms": min(xs),
        "max_ms": max(xs),
        "iqr_ms": pct(xs, 0.75) - pct(xs, 0.25),
        "p95_ms": pct(xs, 0.95),
        "p99_ms": pct(xs, 0.99),
        "stddev_ms": statistics.pstdev(xs) if len(xs) > 1 else 0.0,
    }


def out(cmd, cwd=None):
    try:
        return subprocess.check_output(cmd, cwd=cwd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as e:
        return f"unavailable: {e}"


def env_snapshot():
    root = Path(__file__).resolve().parents[2]
    query = (
        "name,gpu_bus_id,driver_version,persistence_mode,ecc.mode.current,"
        "mig.mode.current,power.limit,power.draw,clocks.sm,clocks.mem,"
        "temperature.gpu,utilization.gpu,memory.total"
    )
    return {
        "nvidia_smi": out(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"]),
        "nvcc": out(["/usr/local/cuda/bin/nvcc", "--version"]),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(),
        "git_head": out(["git", "rev-parse", "HEAD"], root),
        "git_status_short": out(["git", "status", "--short"], root),
        "build_command": "make -B build/libgemma4_flash_attention.so NVCC=/usr/local/cuda/bin/nvcc",
    }


def row(name, xs, base=None):
    s = stats(xs)
    msg = (
        f"{name} median_ms={s['median_ms']:.6f} "
        f"trimmed_mean_ms={s['trimmed_mean_ms']:.6f} "
        f"iqr_ms={s['iqr_ms']:.6f} samples_ms={xs}"
    )
    if base:
        msg += f" delta_vs_pytorch={(s['median_ms'] / statistics.median(base) - 1) * 100:+.2f}%"
    print(msg)
    return s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--page-size", type=int, default=64)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--samples", type=int, default=5)
    p.add_argument("--json", default=None)
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    fn = load_kernel()
    cfg, pos, table, cos, sin, t = make_case(args.batch, args.seq_len, args.page_size)
    custom = lambda: custom_case(fn, cfg, t, pos, table, cos, sin)
    pt = lambda: pytorch_case(t, pos, table, cos, sin)

    custom(); pt(); torch.cuda.synchronize()
    diff = max((t["q"].float() - t["qr"].float()).abs().max().item(),
               (t["ck"].float() - t["rck"].float()).abs().max().item(),
               (t["cv"].float() - t["rcv"].float()).abs().max().item())
    print(f"shape batch={args.batch} seq_len={args.seq_len} hidden={H} qkv={QKVS} dtype=bf16")
    print(f"custom_vs_pytorch max_abs={diff:.6f}")
    assert diff <= 0.0625

    contract = {
        "measurement": "typical latency for project->prepare stream work",
        "timing": "CUDA events on current stream; CPU enqueue/host wall time excluded",
        "cache": "warm-L2 repeated buffers; no L2 flush",
        "launch_overhead": "CPU launch overhead excluded by CUDA event timing",
        "stability_scope": "single process, single GPU",
        "minimum_effect_size_pct": 5.0,
        "warmup": args.warmup,
        "iters_per_sample": args.iters,
        "sample_count": args.samples,
        "shape": {"batch": args.batch, "seq_len": args.seq_len, "hidden": H, "qkv": QKVS, "dtype": "bf16"},
        "seed": 0,
        "correctness_max_abs": diff,
    }

    pt_ms = bench(pt, args.warmup, args.iters, args.samples)
    results = {"pytorch_project_prepare": stats(pt_ms)}
    row("pytorch_project_prepare", pt_ms)
    custom_ms = bench(custom, args.warmup, args.iters, args.samples)
    results["custom_project_prepare"] = stats(custom_ms)
    row("custom_project_prepare", custom_ms, pt_ms)

    try:
        ops = importlib.import_module("vllm._custom_ops")
        xs = bench(lambda: pytorch_case(t, pos, table, cos, sin, ops),
                   args.warmup, args.iters, args.samples)
        results["vllm_rmsnorm_project_prepare"] = stats(xs)
        row("vllm_rmsnorm_project_prepare",
            xs, pt_ms)
    except Exception as e:
        results["vllm_project_prepare"] = {"skipped": str(e)}
        print(f"vllm_project_prepare skipped: {e}")

    report = {
        "contract": contract,
        "environment": env_snapshot(),
        "results": results,
        "threats": [
            "clocks not locked",
            "single process only",
            "warm-cache only",
            "vLLM unavailable on this machine" if "vllm_project_prepare" in results else "vLLM baseline is operator-level, not server end-to-end",
        ],
    }
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
