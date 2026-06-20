#!/usr/bin/env python3
"""Run Gemma 4 prefill GEMM tuning sweeps for Tuna and SGEMM BF16 benches."""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
BUILD_DIR = REPO_ROOT / "build" / "experiments"

OPS = [
    "ffn_gate_up",
    "ffn_down",
    "sliding_qkv",
    "sliding_o",
    "global_q",
    "global_k",
    "global_o",
    "final_logits",
]

OP_WEIGHTS = {
    "ffn_gate_up": 48,
    "ffn_down": 48,
    "sliding_qkv": 40,
    "sliding_o": 40,
    "global_q": 8,
    "global_k": 8,
    "global_o": 8,
    "final_logits": 1,
}

BACKENDS = {
    "tuna": {
        "make_target": "tuna-prefill-bench",
        "binary": BUILD_DIR / "gemma4_tuna_prefill_bench",
        "configs": [
            "wmma_16x16",
            "wmma_16x32",
            "wmma_16x64",
            "wmma_32x64",
            "wmma_64x64",
            "smem_16x64",
            "smem_16x128",
            "smem_32x64",
            "smem_32x128",
            "smem_64x64",
        ],
    },
    "sgemm-bf16": {
        "make_target": "sgemm-bf16-prefill-bench",
        "binary": BUILD_DIR / "gemma4_sgemm_bf16_prefill_bench",
        "configs": [
            "bf16_cutlass_64x64",
            "bf16_cutlass_64x64_s10",
            "bf16_cutlass_64x64x64_s5",
            "bf16_cutlass_64x128",
            "bf16_cutlass_64x128x64",
            "bf16_cutlass_64x128_s2",
            "bf16_cutlass_64x128_s4",
            "bf16_cutlass_64x128_s6",
            "bf16_cutlass_64x256_s4",
            "bf16_cutlass_128x64",
            "bf16_cutlass_128x64x64",
            "bf16_cutlass_128x64_s6",
            "bf16_cutlass_128x128",
            "bf16_cutlass_128x128x64",
            "bf16_cutlass_128x128_s5",
            "bf16_cutlass_128x256",
            "bf16_cutlass_256x64",
            "bf16_cutlass_256x64_s4",
            "bf16_cutlass_256x128",
            "bf16_streamk_64x64x64",
            "bf16_streamk_64x128x64",
            "bf16_streamk_128x128x64",
            "bf16_streamk_s2_64x64x64",
            "bf16_streamk_s2_64x128x64",
            "bf16_streamk_s2_128x128x64",
            "bf16_streamk_s4_64x64x64",
            "bf16_streamk_s4_64x128x64",
            "bf16_streamk_s4_128x128x64",
            "bf16_auto_ffn_down",
            "bf16_16x16",
            "bf16_16x32",
            "bf16_16x64",
            "bf16_16x128",
            "bf16_16x256",
            "bf16_32x32",
            "bf16_32x64",
            "bf16_32x128",
            "bf16_32x256",
            "bf16_64x64",
            "bf16_64x128",
            "bf16_128x64",
            "bf16_256x32",
            "bf16_smem64_32x64",
            "bf16_smem64_64x64",
            "bf16_smemA64_16x32",
            "bf16_smemA64_16x64",
            "bf16_smemA64_16x128",
            "bf16_smemA128_16x32",
            "bf16_smemA128_16x64",
            "bf16_smemA128_16x128",
            "bf16_warp_16x32",
            "bf16_warp_16x64",
            "bf16_warp_16x128",
            "bf16_splitk4_16x32",
            "bf16_splitk4_32x64",
        ],
    },
}

CASE_RE = re.compile(
    r"^(?P<op>\S+)\s+M=\s*(?P<m>\d+)\s+K=\s*(?P<k>\d+)\s+N=\s*(?P<n>\d+)"
    r"\s+cublas_ms=\s*(?P<cublas_ms>[0-9.]+).*max_abs=(?P<max_abs>[0-9.eE+-]+)"
)
CONFIG_RE = re.compile(
    r"^\s+(?P<config>\S+)\s+custom_ms=\s*(?P<custom_ms>[0-9.]+)"
    r"\s+custom_tflops=\s*(?P<custom_tflops>[0-9.]+)\s+speedup=\s*(?P<speedup>[0-9.]+)x"
)


@dataclass(frozen=True)
class Result:
    backend: str
    op: str
    config: str
    m: int
    k: int
    n: int
    cublas_ms: float
    custom_ms: float
    speedup: float
    custom_tflops: float
    max_abs: float
    cublas_backend: str = "gemmex"
    cublas_algo: str = "default_tensor"


@dataclass(frozen=True)
class DispatchChoice:
    backend: str
    cublas_backend: str
    cublas_algo: str
    op: str
    m: int
    k: int
    n: int
    config: str
    use_custom: bool
    chosen_ms: float
    cublas_ms: float
    custom_ms: float
    speedup: float
    weight: int


def parse_csv_arg(value: str) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("empty comma-separated list")
    return items


def parse_int_csv_arg(value: str) -> list[int]:
    try:
        items = [int(item) for item in parse_csv_arg(value)]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if any(item <= 0 for item in items):
        raise argparse.ArgumentTypeError("M values must be positive")
    return items


def run_command(cmd: list[str], env: dict[str, str] | None = None) -> str:
    print("+ " + " ".join(cmd), flush=True)
    command_env = os.environ.copy()
    if env:
        command_env.update(env)
    completed = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=command_env,
    )
    if completed.returncode != 0:
        print(completed.stdout, file=sys.stderr)
        raise RuntimeError(f"command failed with exit code {completed.returncode}: {cmd}")
    return completed.stdout


def parse_results(
    backend: str,
    config: str,
    output: str,
    cublas_backend: str,
    cublas_algo: str,
) -> list[Result]:
    results: list[Result] = []
    current: dict[str, str] | None = None

    for line in output.splitlines():
        case_match = CASE_RE.match(line)
        if case_match:
            current = case_match.groupdict()
            continue

        config_match = CONFIG_RE.match(line)
        if config_match and current is not None:
            config_row = config_match.groupdict()
            if config_row["config"] != config:
                continue
            results.append(
                Result(
                    backend=backend,
                    op=current["op"],
                    config=config,
                    m=int(current["m"]),
                    k=int(current["k"]),
                    n=int(current["n"]),
                    cublas_ms=float(current["cublas_ms"]),
                    custom_ms=float(config_row["custom_ms"]),
                    speedup=float(config_row["speedup"]),
                    custom_tflops=float(config_row["custom_tflops"]),
                    max_abs=float(current["max_abs"]),
                    cublas_backend=cublas_backend,
                    cublas_algo=cublas_algo,
                )
            )

    return results


def read_csv(path: Path) -> list[Result]:
    rows: list[Result] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                Result(
                    backend=row["backend"],
                    op=row["op"],
                    config=row["config"],
                    m=int(row["m"]),
                    k=int(row["k"]),
                    n=int(row["n"]),
                    cublas_ms=float(row["cublas_ms"]),
                    custom_ms=float(row["custom_ms"]),
                    speedup=float(row["speedup"]),
                    custom_tflops=float(row["custom_tflops"]),
                    max_abs=float(row["max_abs"]),
                    cublas_backend=row.get("cublas_backend", "gemmex"),
                    cublas_algo=row.get("cublas_algo", "default_tensor"),
                )
            )
    return rows


def geometric_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return math.exp(sum(math.log(max(value, 1.0e-12)) for value in values) / len(values))


def write_csv(path: Path, rows: list[Result]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "backend",
                "cublas_backend",
                "cublas_algo",
                "op",
                "config",
                "m",
                "k",
                "n",
                "cublas_ms",
                "custom_ms",
                "speedup",
                "custom_tflops",
                "max_abs",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def print_summary(rows: list[Result]) -> None:
    grouped: dict[tuple[str, str, str, str, str], list[Result]] = defaultdict(list)
    best_cublas_by_case: dict[tuple[str, str, str, str, int], float] = {}
    for row in rows:
        grouped[
            (row.backend, row.cublas_backend, row.cublas_algo, row.op, row.config)
        ].append(row)
        key = (row.backend, row.cublas_backend, row.cublas_algo, row.op, row.m)
        best_cublas_by_case[key] = min(best_cublas_by_case.get(key, row.cublas_ms),
                                       row.cublas_ms)

    best_by_op: dict[tuple[str, str, str, str], tuple[str, float, float]] = {}
    for (backend, cublas_backend, cublas_algo, op, config), config_rows in grouped.items():
        speedups = [
            best_cublas_by_case[
                (row.backend, row.cublas_backend, row.cublas_algo, row.op, row.m)
            ] / row.custom_ms
            for row in config_rows
        ]
        geomean = geometric_mean(speedups)
        worst = min(speedups)
        key = (backend, cublas_backend, cublas_algo, op)
        current = best_by_op.get(key)
        if current is None or (geomean, worst) > (current[1], current[2]):
            best_by_op[key] = (config, geomean, worst)

    print("\nSummary by op:")
    for backend, cublas_backend, cublas_algo, op in sorted(best_by_op):
        config, geomean, worst = best_by_op[
            (backend, cublas_backend, cublas_algo, op)
        ]
        print(
            f"  {backend:10s} cublas={cublas_backend:6s}/{cublas_algo:14s} "
            f"{op:14s} best={config:14s} "
            f"geomean={geomean:7.3f} worst={worst:7.3f}"
        )


def choose_dispatch(rows: list[Result], custom_threshold: float) -> list[DispatchChoice]:
    grouped: dict[tuple[str, str, str, str, int], list[Result]] = defaultdict(list)
    for row in rows:
        grouped[(row.backend, row.cublas_backend, row.cublas_algo, row.op, row.m)].append(row)

    choices: list[DispatchChoice] = []
    for (backend, cublas_backend, cublas_algo, op, m), config_rows in sorted(grouped.items()):
        best = min(config_rows, key=lambda row: row.custom_ms)
        cublas_ms = min(row.cublas_ms for row in config_rows)
        custom_speedup = cublas_ms / best.custom_ms
        use_custom = custom_speedup >= custom_threshold
        chosen_ms = best.custom_ms if use_custom else cublas_ms
        choices.append(
            DispatchChoice(
                backend=backend,
                cublas_backend=cublas_backend,
                cublas_algo=cublas_algo,
                op=op,
                m=m,
                k=best.k,
                n=best.n,
                config=best.config,
                use_custom=use_custom,
                chosen_ms=chosen_ms,
                cublas_ms=cublas_ms,
                custom_ms=best.custom_ms,
                speedup=cublas_ms / chosen_ms,
                weight=OP_WEIGHTS.get(op, 1),
            )
        )
    return choices


def write_dispatch_csv(path: Path, choices: list[DispatchChoice]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "backend",
                "cublas_backend",
                "cublas_algo",
                "op",
                "m",
                "k",
                "n",
                "route",
                "config",
                "chosen_ms",
                "cublas_ms",
                "custom_ms",
                "speedup",
                "weight",
            ],
        )
        writer.writeheader()
        for choice in choices:
            writer.writerow(
                {
                    "backend": choice.backend,
                    "cublas_backend": choice.cublas_backend,
                    "cublas_algo": choice.cublas_algo,
                    "op": choice.op,
                    "m": choice.m,
                    "k": choice.k,
                    "n": choice.n,
                    "route": "custom" if choice.use_custom else "cublas",
                    "config": choice.config,
                    "chosen_ms": choice.chosen_ms,
                    "cublas_ms": choice.cublas_ms,
                    "custom_ms": choice.custom_ms,
                    "speedup": choice.speedup,
                    "weight": choice.weight,
                }
            )


def summarize_csv(
    paths: list[Path], custom_threshold: float, dispatch_out: Path | None
) -> None:
    rows: list[Result] = []
    for path in paths:
        rows.extend(read_csv(path))

    print_summary(rows)
    choices = choose_dispatch(rows, custom_threshold)
    by_backend: dict[tuple[str, str, str], list[DispatchChoice]] = defaultdict(list)
    for choice in choices:
        by_backend[(choice.backend, choice.cublas_backend, choice.cublas_algo)].append(choice)

    print(f"\nDispatch summary, custom_threshold={custom_threshold:g}:")
    for (backend, cublas_backend, cublas_algo), backend_choices in sorted(by_backend.items()):
        weighted_chosen = 0.0
        weighted_cublas = 0.0
        weighted_custom_only = 0.0
        weighted_cases = 0
        print(f"\n{backend} (cublas={cublas_backend}/{cublas_algo}):")
        for choice in backend_choices:
            weighted_chosen += choice.chosen_ms * choice.weight
            weighted_cublas += choice.cublas_ms * choice.weight
            weighted_custom_only += choice.custom_ms * choice.weight
            weighted_cases += choice.weight
            route = "custom" if choice.use_custom else "cublas"
            print(
                f"  {choice.op:14s} M={choice.m:4d} route={route:6s} "
                f"config={choice.config:14s} speedup={choice.speedup:7.3f} "
                f"custom_ms={choice.custom_ms:8.4f} cublas_ms={choice.cublas_ms:8.4f}"
            )

        print(
            f"  weighted_ms_per_m_sweep: dispatch={weighted_chosen:.4f} "
            f"cublas={weighted_cublas:.4f} custom_only={weighted_custom_only:.4f} "
            f"dispatch_vs_cublas={weighted_cublas / weighted_chosen:.4f} "
            f"custom_only_vs_cublas={weighted_cublas / weighted_custom_only:.4f} "
            f"weight_sum={weighted_cases}"
        )

    if dispatch_out is not None:
        write_dispatch_csv(dispatch_out, choices)
        print(f"\nwrote dispatch choices to {dispatch_out}")


def add_tune_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend", choices=BACKENDS, required=True)
    parser.add_argument("--ops", type=parse_csv_arg, default="global_k,sliding_qkv,ffn_gate_up")
    parser.add_argument("--configs", type=parse_csv_arg, default=None)
    parser.add_argument("--m", type=parse_int_csv_arg, default="16,64,256,1024")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--cublas-backend", choices=["gemmex", "lt"], default="gemmex")
    parser.add_argument("--cublas-algo", default="default_tensor")
    parser.add_argument("--cublaslt-heuristics", type=int, default=1)
    parser.add_argument("--graph-repeats", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--keep-going", action="store_true")


def run_tune(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    backend_info = BACKENDS[args.backend]
    ops = args.ops if isinstance(args.ops, list) else parse_csv_arg(args.ops)
    configs = args.configs or backend_info["configs"]
    m_values = args.m if isinstance(args.m, list) else parse_int_csv_arg(args.m)

    unknown_ops = sorted(set(ops) - set(OPS))
    if unknown_ops:
        parser.error(f"unknown ops: {','.join(unknown_ops)}")
    unknown_configs = sorted(set(configs) - set(backend_info["configs"]))
    if unknown_configs:
        parser.error(f"unknown configs: {','.join(unknown_configs)}")
    if args.iters <= 0 or args.warmup < 0:
        parser.error("--iters must be positive and --warmup must be non-negative")
    if args.cublas_backend == "lt" and args.cublas_algo != "default_tensor":
        parser.error("--cublas-algo only applies to the gemmex cuBLAS backend")
    if args.cublaslt_heuristics <= 0:
        parser.error("--cublaslt-heuristics must be positive")
    if args.graph_repeats < 0:
        parser.error("--graph-repeats must be non-negative")

    if not args.skip_build:
        run_command(["make", str(backend_info["make_target"])])

    rows: list[Result] = []
    binary = str(backend_info["binary"])
    m_csv = ",".join(str(value) for value in m_values)
    benchmark_env = {
        "GEMMA4_PREFILL_CUBLAS_BACKEND": args.cublas_backend,
        "GEMMA4_PREFILL_CUBLAS_ALGO": args.cublas_algo,
        "GEMMA4_PREFILL_CUBLASLT_HEURISTICS": str(args.cublaslt_heuristics),
        "GEMMA4_PREFILL_GRAPH_REPEATS": str(args.graph_repeats),
    }
    cublas_algo_label = (
        f"h{args.cublaslt_heuristics}"
        if args.cublas_backend == "lt"
        else args.cublas_algo
    )
    for op in ops:
        for config in configs:
            try:
                output = run_command(
                    [binary, op, str(args.iters), str(args.warmup), m_csv, config],
                    env=benchmark_env,
                )
            except RuntimeError:
                if not args.keep_going:
                    raise
                print(f"skipping failed config {op}/{config}", file=sys.stderr)
                continue
            parsed = parse_results(
                args.backend, config, output, args.cublas_backend, cublas_algo_label
            )
            if len(parsed) != len(m_values):
                raise RuntimeError(
                    f"expected {len(m_values)} rows for {op}/{config}, parsed {len(parsed)}"
                )
            rows.extend(parsed)

    output_path = args.out
    if output_path is None:
        output_path = BUILD_DIR / "gemma4_prefill_tune" / f"{args.backend}.csv"
    write_csv(output_path, rows)
    print(f"\nwrote {output_path}")
    print_summary(rows)
    return 0


def main() -> int:
    argv = sys.argv[1:]
    if argv and argv[0] == "summarize":
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("command")
        parser.add_argument("csv", nargs="+", type=Path)
        parser.add_argument("--custom-threshold", type=float, default=1.0)
        parser.add_argument("--dispatch-out", type=Path, default=None)
        args = parser.parse_args(argv)
        summarize_csv(args.csv, args.custom_threshold, args.dispatch_out)
        return 0

    if argv and argv[0] == "tune":
        argv = argv[1:]
    parser = argparse.ArgumentParser(description=__doc__)
    add_tune_args(parser)
    args = parser.parse_args(argv)
    return run_tune(args, parser)


if __name__ == "__main__":
    raise SystemExit(main())
