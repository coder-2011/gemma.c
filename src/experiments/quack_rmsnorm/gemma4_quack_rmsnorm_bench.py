#!/usr/bin/env python3
"""Benchmark the downloaded Quack RMSNorm CuTe source on Gemma 4 shapes."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import pathlib
import sys
from dataclasses import dataclass
from typing import Callable

import torch


GEMMA4_HIDDEN_SIZE = 5376
GEMMA4_RMS_NORM_EPS = 1.0e-6


@dataclass
class TimingStats:
    best_ms: float
    avg_ms: float


@dataclass(frozen=True)
class ConfigSpec:
    label: str
    config: object | None


def load_downloaded_quack_rmsnorm():
    path = pathlib.Path(__file__).with_name("rmsnorm.py")
    spec = importlib.util.spec_from_file_location("gemma4_downloaded_quack_rmsnorm", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_csv_ints(value: str) -> list[int]:
    result = [int(part) for part in value.split(",") if part]
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return result


def parse_config_specs(module, value: str) -> list[ConfigSpec]:
    specs: list[ConfigSpec] = []
    for raw_spec in [part for part in value.split(",") if part]:
        if raw_spec == "heuristic":
            specs.append(ConfigSpec("heuristic", None))
            continue

        parts = raw_spec.split(":")
        if len(parts) != 4:
            raise argparse.ArgumentTypeError(
                "configs must be heuristic or threads:tpr:reload:delay"
            )
        num_threads = int(parts[0])
        threads_per_row = int(parts[1])
        reload_from = None if parts[2] == "none" else parts[2]
        delay_w_load = bool(int(parts[3]))
        if num_threads <= 0 or threads_per_row <= 0:
            raise argparse.ArgumentTypeError("thread counts must be positive")
        if reload_from not in {None, "smem", "gmem"}:
            raise argparse.ArgumentTypeError("reload must be none, smem, or gmem")
        config = module.RmsNormFwdConfig(
            num_threads=num_threads,
            threads_per_row=threads_per_row,
            cluster_n=1,
            reload_from=reload_from,
            delay_w_load=delay_w_load,
        )
        specs.append(ConfigSpec(raw_spec, config))
    if not specs:
        raise argparse.ArgumentTypeError("at least one config is required")
    return specs


def default_rows(max_rows: int) -> list[int]:
    rows = [row for row in [1, 4, 16, 64, 256, 1024, 4096, 8192] if row <= max_rows]
    if not rows or rows[-1] != max_rows:
        rows.append(max_rows)
    return rows


def time_ms(fn: Callable[[], None],
            warmup: int,
            iters: int,
            trials: int) -> TimingStats:
    best_ms = math.inf
    total_ms = 0.0
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

        ms = start.elapsed_time(stop) / iters
        best_ms = min(best_ms, ms)
        total_ms += ms
    return TimingStats(best_ms=best_ms, avg_ms=total_ms / trials)


def gib_per_second(bytes_touched: float, ms: float) -> float:
    gib = bytes_touched / (1024.0 * 1024.0 * 1024.0)
    return gib / (ms / 1000.0)


def logical_bytes(mode: str, rows: int, width: int) -> float:
    elems = float(rows * width)
    if mode == "weighted":
        return elems * 2.0 * 3.0 + rows * 4.0
    if mode == "scale_free":
        return elems * 2.0 * 2.0 + rows * 4.0
    if mode == "residual_quack_fused":
        return elems * 2.0 * 5.0 + rows * 4.0
    if mode == "residual_gemma_exact_split":
        return elems * 2.0 * 6.0 + rows * 4.0
    raise ValueError(mode)


def make_inputs(rows: int,
                width: int,
                seed: int,
                has_weight: bool,
                has_residual: bool):
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    x = torch.empty((rows, width), dtype=torch.bfloat16, device="cuda")
    x.uniform_(-1.0, 1.0, generator=generator)

    weight = None
    if has_weight:
        weight = torch.empty((width,), dtype=torch.bfloat16, device="cuda")
        weight.uniform_(-0.5, 0.5, generator=generator)

    residual = None
    if has_residual:
        residual = torch.empty((rows, width), dtype=torch.bfloat16, device="cuda")
        residual.uniform_(-1.0, 1.0, generator=generator)
    return x, weight, residual


def rmsnorm_reference(x: torch.Tensor,
                      weight: torch.Tensor | None,
                      eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    x_float = x.float()
    rstd = torch.rsqrt(torch.mean(x_float * x_float, dim=-1) + eps)
    out = x_float * rstd[:, None]
    if weight is not None:
        out = out * weight.float()[None, :]
    return out.to(torch.bfloat16), rstd


def quack_forward(module,
                  x: torch.Tensor,
                  weight: torch.Tensor | None,
                  out: torch.Tensor,
                  rstd: torch.Tensor,
                  residual: torch.Tensor | None,
                  residual_out: torch.Tensor | None,
                  eps: float,
                  config: object | None) -> None:
    if config is None:
        module._rmsnorm_fwd(
            x, weight, out, None, rstd, None, residual, residual_out, eps, False
        )
        return

    per_head = (weight is not None and weight.dim() == 2)
    dtype, out_dtype, weight_dtype, bias_dtype, res_dtype, res_out_dtype = [
        module.torch2cute_dtype_map[t.dtype] if t is not None else None
        for t in [x, out, weight, None, residual, residual_out]
    ]
    module._compile_rmsnorm_fwd(
        dtype,
        out_dtype,
        res_dtype,
        weight_dtype,
        bias_dtype,
        res_out_dtype,
        x.size(-1),
        True,
        False,
        False,
        per_head,
        config=config,
    )(x, weight, None, residual, out, residual_out, rstd, None, eps)


def run_case(module,
             mode: str,
             rows: int,
             width: int,
             eps: float,
             seed: int,
             warmup: int,
             iters: int,
             trials: int,
             config_spec: ConfigSpec) -> dict[str, object]:
    has_weight = mode != "scale_free"
    has_residual = mode.startswith("residual_")
    x, weight, residual = make_inputs(rows, width, seed, has_weight, has_residual)
    out = torch.empty_like(x)
    rstd = torch.empty((rows,), dtype=torch.float32, device="cuda")
    residual_out = torch.empty_like(x) if has_residual else None

    if mode == "residual_gemma_exact_split":
        residual_rounded = torch.empty_like(x)

        def fn() -> None:
            residual_rounded.copy_((x.float() + residual.float()).to(torch.bfloat16))
            quack_forward(
                module, residual_rounded, weight, out, rstd, None, None, eps,
                config_spec.config,
            )

        expected_residual = (x.float() + residual.float()).to(torch.bfloat16)
        expected_out, expected_rstd = rmsnorm_reference(expected_residual, weight, eps)
    else:

        def fn() -> None:
            quack_forward(
                module, x, weight, out, rstd, residual, residual_out, eps,
                config_spec.config,
            )

        if residual is None:
            expected_out, expected_rstd = rmsnorm_reference(x, weight, eps)
        else:
            # Quack's fused residual mode normalizes the FP32 sum, while the
            # Gemma CUDA fused kernel normalizes the BF16-rounded residual.
            residual_float = x.float() + residual.float()
            expected_rstd = torch.rsqrt(torch.mean(residual_float * residual_float, dim=-1) + eps)
            expected_out = residual_float * expected_rstd[:, None] * weight.float()[None, :]
            expected_out = expected_out.to(torch.bfloat16)

    fn()
    torch.cuda.synchronize()
    max_abs = (out.float() - expected_out.float()).abs().max().item()
    rstd_max_abs = (rstd - expected_rstd).abs().max().item()
    stats = time_ms(fn, warmup, iters, trials)
    bytes_touched = logical_bytes(mode, rows, width)
    return {
        "mode": mode,
        "rows": rows,
        "width": width,
        "config": config_spec.label,
        "best_ms": stats.best_ms,
        "avg_ms": stats.avg_ms,
        "gib_s": gib_per_second(bytes_touched, stats.best_ms),
        "max_abs": max_abs,
        "rstd_max_abs": rstd_max_abs,
    }


def parse_modes(value: str) -> list[str]:
    allowed = {
        "weighted",
        "scale_free",
        "residual_quack_fused",
        "residual_gemma_exact_split",
    }
    modes = [mode for mode in value.split(",") if mode]
    unknown = sorted(set(modes) - allowed)
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown modes: {', '.join(unknown)}")
    return modes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=GEMMA4_HIDDEN_SIZE)
    parser.add_argument("--max-rows", type=int, default=1024)
    parser.add_argument("--rows", type=parse_csv_ints)
    parser.add_argument("--modes", type=parse_modes, default=parse_modes("weighted"))
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--eps", type=float, default=GEMMA4_RMS_NORM_EPS)
    parser.add_argument("--seed", type=int, default=0x20260521)
    parser.add_argument(
        "--configs",
        default="heuristic",
        help="comma-separated heuristic or threads:tpr:reload:delay specs",
    )
    args = parser.parse_args()

    if args.width <= 0 or (args.width % 8) != 0:
        raise SystemExit("--width must be positive and divisible by 8")
    if args.max_rows <= 0 or args.iters <= 0 or args.warmup < 0 or args.trials <= 0:
        raise SystemExit("timing counts and max rows must be positive")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    module = load_downloaded_quack_rmsnorm()
    config_specs = parse_config_specs(module, args.configs)
    rows = args.rows if args.rows is not None else default_rows(args.max_rows)

    fieldnames = [
        "mode",
        "rows",
        "width",
        "config",
        "best_ms",
        "avg_ms",
        "gib_s",
        "max_abs",
        "rstd_max_abs",
    ]
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()
    for config_spec in config_specs:
        for mode in args.modes:
            for row_count in rows:
                case_seed = args.seed ^ args.width ^ (row_count * 0x9E3779B1)
                row = run_case(
                    module,
                    mode,
                    row_count,
                    args.width,
                    args.eps,
                    case_seed,
                    args.warmup,
                    args.iters,
                    args.trials,
                    config_spec,
                )
                writer.writerow(row)
                sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
