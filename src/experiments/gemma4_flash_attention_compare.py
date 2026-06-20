#!/usr/bin/env python3
"""Same-tensor comparator for Gemma's FA2-derived sliding attention kernel."""

from __future__ import annotations

import argparse
import ctypes
import math
import statistics
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LIB = ROOT / "build" / "libgemma4_flash_attention.so"
DEFAULT_REFERENCE_LIB = ROOT / "build" / "libgemma4_flash_attention_reference.so"

Q_HEADS = 16
KV_HEADS = 8
HEAD_DIM = 256
WINDOW_LEFT = 1024
ATTR_NAMES = [
    "sharedSizeBytes",
    "constSizeBytes",
    "localSizeBytes",
    "maxThreadsPerBlock",
    "numRegs",
    "ptxVersion",
    "binaryVersion",
    "cacheModeCA",
    "maxDynamicSharedSizeBytes",
    "preferredShmemCarveout",
]


def load_custom(lib_path: Path) -> ctypes.CDLL:
    lib = ctypes.CDLL(str(lib_path))
    lib.gemma4_flash_attention_sliding_fwd_bf16.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_void_p,
    ]
    lib.gemma4_flash_attention_sliding_fwd_bf16.restype = ctypes.c_int
    lib.gemma4_flash_attention_sliding_smem_bytes.argtypes = []
    lib.gemma4_flash_attention_sliding_smem_bytes.restype = ctypes.c_size_t
    lib.gemma4_flash_attention_sliding_threads_per_block.argtypes = []
    lib.gemma4_flash_attention_sliding_threads_per_block.restype = ctypes.c_int
    lib.gemma4_flash_attention_sliding_kernel_attributes.argtypes = [
        ctypes.POINTER(ctypes.c_longlong),
        ctypes.c_int,
    ]
    lib.gemma4_flash_attention_sliding_kernel_attributes.restype = ctypes.c_int
    return lib


def load_reference(lib_path: Path) -> ctypes.CDLL:
    lib = ctypes.CDLL(str(lib_path))
    lib.gemma4_flash_attention_reference_sliding_fwd_bf16.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_void_p,
    ]
    lib.gemma4_flash_attention_reference_sliding_fwd_bf16.restype = ctypes.c_int
    lib.gemma4_flash_attention_reference_sliding_smem_bytes.argtypes = []
    lib.gemma4_flash_attention_reference_sliding_smem_bytes.restype = ctypes.c_size_t
    lib.gemma4_flash_attention_reference_sliding_threads_per_block.argtypes = []
    lib.gemma4_flash_attention_reference_sliding_threads_per_block.restype = ctypes.c_int
    lib.gemma4_flash_attention_reference_sliding_kernel_attributes.argtypes = [
        ctypes.POINTER(ctypes.c_longlong),
        ctypes.c_int,
    ]
    lib.gemma4_flash_attention_reference_sliding_kernel_attributes.restype = ctypes.c_int
    return lib


def call_custom(
    lib: ctypes.CDLL,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
) -> None:
    stream = torch.cuda.current_stream(q.device)
    status = lib.gemma4_flash_attention_sliding_fwd_bf16(
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_void_p(lse.data_ptr()),
        ctypes.c_void_p(q.data_ptr()),
        ctypes.c_void_p(k.data_ptr()),
        ctypes.c_void_p(v.data_ptr()),
        ctypes.c_int(q.shape[0]),
        ctypes.c_int(q.shape[1]),
        ctypes.c_int(k.shape[1]),
        ctypes.c_int(WINDOW_LEFT),
        ctypes.c_float(1.0 / math.sqrt(HEAD_DIM)),
        ctypes.c_void_p(stream.cuda_stream),
    )
    if status != 0:
        raise RuntimeError(f"custom CUDA launcher returned cudaError_t={status}")


def call_reference(
    lib: ctypes.CDLL,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
) -> None:
    stream = torch.cuda.current_stream(q.device)
    status = lib.gemma4_flash_attention_reference_sliding_fwd_bf16(
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_void_p(lse.data_ptr()),
        ctypes.c_void_p(q.data_ptr()),
        ctypes.c_void_p(k.data_ptr()),
        ctypes.c_void_p(v.data_ptr()),
        ctypes.c_int(q.shape[0]),
        ctypes.c_int(q.shape[1]),
        ctypes.c_int(k.shape[1]),
        ctypes.c_int(WINDOW_LEFT),
        ctypes.c_float(1.0 / math.sqrt(HEAD_DIM)),
        ctypes.c_void_p(stream.cuda_stream),
    )
    if status != 0:
        raise RuntimeError(f"reference CUDA launcher returned cudaError_t={status}")


def time_cuda(fn, warmup: int, iters: int) -> float:
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
    return start.elapsed_time(stop) / iters


def time_cuda_samples(fn, warmup: int, iters: int, trials: int) -> list[float]:
    return [time_cuda(fn, warmup, iters) for _ in range(trials)]


def timing_summary(samples: list[float]) -> tuple[float, float, float]:
    return min(samples), statistics.median(samples), max(samples)


def attr_values(lib: ctypes.CDLL, prefix: str) -> dict[str, int]:
    values = (ctypes.c_longlong * len(ATTR_NAMES))()
    fn = getattr(lib, f"{prefix}_sliding_kernel_attributes")
    status = fn(values, len(ATTR_NAMES))
    if status != 0:
        raise RuntimeError(f"{prefix}_sliding_kernel_attributes returned cudaError_t={status}")
    return {name: int(values[i]) for i, name in enumerate(ATTR_NAMES)}


def stats(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float, float]:
    diff = (a.float() - b.float()).abs()
    denom = torch.maximum(torch.maximum(a.float().abs(), b.float().abs()), torch.ones_like(diff))
    rel = diff / denom
    return diff.max().item(), diff.mean().item(), rel.max().item()


def approx_tflops(batch: int, seq_len: int, ms: float) -> float:
    keys = 0
    for row in range(seq_len):
        keys += row - max(0, row - WINDOW_LEFT) + 1
    flops = batch * Q_HEADS * keys * (4 * HEAD_DIM)
    return flops / (ms * 1.0e-3) / 1.0e12


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lib", type=Path, default=DEFAULT_LIB)
    parser.add_argument("--reference-lib", type=Path, default=DEFAULT_REFERENCE_LIB)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--custom-only", action="store_true")
    parser.add_argument("--skip-python-flash-attn", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if not args.lib.exists():
        raise RuntimeError(f"missing custom shared library: {args.lib}")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    q = torch.randn(args.batch, args.seq, Q_HEADS, HEAD_DIM, device=device, dtype=dtype)
    k = torch.randn(args.batch, args.seq, KV_HEADS, HEAD_DIM, device=device, dtype=dtype)
    v = torch.randn(args.batch, args.seq, KV_HEADS, HEAD_DIM, device=device, dtype=dtype)
    out_custom = torch.empty_like(q)
    lse_custom = torch.empty((args.batch, Q_HEADS, args.seq), device=device, dtype=torch.float32)

    lib = load_custom(args.lib)
    call_custom(lib, q, k, v, out_custom, lse_custom)
    torch.cuda.synchronize()

    custom_samples = time_cuda_samples(
        lambda: call_custom(lib, q, k, v, out_custom, lse_custom),
        args.warmup,
        args.iters,
        args.trials,
    )
    custom_min, custom_ms, custom_max = timing_summary(custom_samples)
    print(
        f"custom median_ms={custom_ms:.6f} min_ms={custom_min:.6f} max_ms={custom_max:.6f} "
        f"approx_tflops={approx_tflops(args.batch, args.seq, custom_ms):.3f} "
        f"threads={lib.gemma4_flash_attention_sliding_threads_per_block()} "
        f"smem={lib.gemma4_flash_attention_sliding_smem_bytes()}"
    )

    if args.custom_only:
        return

    compared_any_reference = False
    if args.reference_lib.exists():
        ref_lib = load_reference(args.reference_lib)
        out_reference = torch.empty_like(q)
        lse_reference = torch.empty_like(lse_custom)
        call_reference(ref_lib, q, k, v, out_reference, lse_reference)
        torch.cuda.synchronize()
        max_abs, mean_abs, max_rel = stats(out_custom, out_reference)
        print(
            "diff_vs_official_source_ref "
            f"max_abs={max_abs:.8g} mean_abs={mean_abs:.8g} max_rel={max_rel:.8g}"
        )
        custom_attrs = attr_values(lib, "gemma4_flash_attention")
        ref_attrs = attr_values(ref_lib, "gemma4_flash_attention_reference")
        for name in ATTR_NAMES:
            print(
                f"kernel_attr {name} custom={custom_attrs[name]} "
                f"official_source_ref={ref_attrs[name]} "
                f"match={int(custom_attrs[name] == ref_attrs[name])}"
            )
        ref_samples = time_cuda_samples(
            lambda: call_reference(ref_lib, q, k, v, out_reference, lse_reference),
            args.warmup,
            args.iters,
            args.trials,
        )
        ref_min, ref_ms, ref_max = timing_summary(ref_samples)
        print(
            f"official_source_ref median_ms={ref_ms:.6f} min_ms={ref_min:.6f} max_ms={ref_max:.6f} "
            f"approx_tflops={approx_tflops(args.batch, args.seq, ref_ms):.3f} "
            f"custom/ref={custom_ms / ref_ms:.6f} "
            f"threads={ref_lib.gemma4_flash_attention_reference_sliding_threads_per_block()} "
            f"smem={ref_lib.gemma4_flash_attention_reference_sliding_smem_bytes()}"
        )
        compared_any_reference = True

    if args.skip_python_flash_attn:
        return

    try:
        from flash_attn import flash_attn_func
    except Exception as exc:  # pragma: no cover - environment dependent
        if compared_any_reference:
            print(f"flash_attn_python missing: {exc}")
            return
        raise RuntimeError("flash_attn is not importable; install/build it before parity runs") from exc

    out_ref = flash_attn_func(
        q,
        k,
        v,
        dropout_p=0.0,
        softmax_scale=1.0 / math.sqrt(HEAD_DIM),
        causal=False,
        window_size=(WINDOW_LEFT, 0),
        alibi_slopes=None,
        deterministic=False,
    )
    torch.cuda.synchronize()
    max_abs, mean_abs, max_rel = stats(out_custom, out_ref)
    print(f"diff_vs_flash_attn max_abs={max_abs:.8g} mean_abs={mean_abs:.8g} max_rel={max_rel:.8g}")

    ref_samples = time_cuda_samples(
        lambda: flash_attn_func(
            q,
            k,
            v,
            dropout_p=0.0,
            softmax_scale=1.0 / math.sqrt(HEAD_DIM),
            causal=False,
            window_size=(WINDOW_LEFT, 0),
            alibi_slopes=None,
            deterministic=False,
        ),
        args.warmup,
        args.iters,
        args.trials,
    )
    ref_min, ref_ms, ref_max = timing_summary(ref_samples)
    print(
        f"flash_attn median_ms={ref_ms:.6f} min_ms={ref_min:.6f} max_ms={ref_max:.6f} "
        f"approx_tflops={approx_tflops(args.batch, args.seq, ref_ms):.3f} "
        f"custom/ref={custom_ms / ref_ms:.6f}"
    )


if __name__ == "__main__":
    main()
