#!/usr/bin/env python3
"""Run a real-prompt gemma.c decode benchmark."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


# Parses the benchmark contract and runner locations.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark gemma.c from explicit prompt text."
    )
    parser.add_argument(
        "--checkpoint",
        default="models/gemma-4-12B-it/model.safetensors",
    )
    parser.add_argument(
        "--tokenizer",
        default="models/gemma-4-12B-it/tokenizer.json",
    )
    parser.add_argument("--prompt-binary", default="build/gemma4_prompt")
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--baseline-output-len", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--out-dir", default="build/bench_results")
    parser.add_argument(
        "--summary",
        default="docs/benchmarks/decode_prompt_b1_no_spec_summary.json",
    )
    parser.add_argument("--chart-dir", default="docs/benchmarks")
    parser.add_argument("--skip-build", action="store_true")
    return parser.parse_args()


# Finds an executable by exact path first, then by PATH lookup.
def find_executable(path_or_name: str) -> str | None:
    path = Path(path_or_name)
    if path.exists():
        return str(path)
    return shutil.which(path_or_name)


# Returns the number of measured requests used by runners with repeated trials.
def measured_iterations(args: argparse.Namespace) -> int:
    return args.iters * args.samples


# Runs a command and stores stdout/stderr beside the summary data.
def run_logged(
    cmd: list[str],
    log_base: Path,
    env: dict[str, str] | None = None,
    timeout_s: int | None = None,
) -> dict:
    start = time.perf_counter()
    proc = subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        timeout=timeout_s,
    )
    elapsed_s = time.perf_counter() - start
    stdout_path = log_base.with_suffix(".stdout.txt")
    stderr_path = log_base.with_suffix(".stderr.txt")
    stdout_path.write_text(proc.stdout)
    stderr_path.write_text(proc.stderr)
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "elapsed_s": elapsed_s,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
    }


# Reads one compact stats line printed by gemma4_prompt benchmark modes.
def parse_local_stats(text: str, name: str) -> dict[str, float]:
    for line in text.splitlines():
        if not line.startswith(f"{name} "):
            continue
        fields: dict[str, float] = {}
        for item in line.split()[1:]:
            if "=" not in item:
                continue
            key, value = item.split("=", 1)
            fields[key] = float(value)
        return fields
    raise ValueError(f"missing {name} stats line")


# Produces a standard error row while keeping the raw log locations.
def error_result(runner: str, reason: str, details: dict | None = None) -> dict:
    result = {
        "runner": runner,
        "status": "unsupported",
        "reason": reason,
    }
    if details is not None:
        safe_details = dict(details)
        stdout = safe_details.pop("stdout", "")
        stderr = safe_details.pop("stderr", "")
        if stdout:
            safe_details["stdout_tail"] = stdout[-2000:]
        if stderr:
            safe_details["stderr_tail"] = stderr[-2000:]
        result["details"] = safe_details
    return result


# Computes decode-only latency by subtracting a 1-token generation baseline.
def decode_from_baseline(
    runner: str,
    baseline_s: float,
    run_s: float,
    args: argparse.Namespace,
) -> dict:
    decode_tokens = args.output_len - args.baseline_output_len
    decode_s = run_s - baseline_s
    if decode_tokens <= 0 or decode_s <= 0.0:
        return error_result(
            runner,
            "baseline subtraction produced a non-positive decode duration",
            {"baseline_s": baseline_s, "run_s": run_s},
        )
    return {
        "runner": runner,
        "status": "ok",
        "decode_tokens": decode_tokens,
        "decode_s": decode_s,
        "decode_ms_per_token": 1000.0 * decode_s / decode_tokens,
        "decode_tps": decode_tokens / decode_s,
        "baseline_s": baseline_s,
        "run_s": run_s,
    }


# Captures the hardware and toolchain context needed to interpret the run.
def capture_environment(out_dir: Path) -> dict:
    commands = {
        "nvidia_smi": [
            "nvidia-smi",
            "--query-gpu=name,gpu_bus_id,driver_version,persistence_mode,"
            "ecc.mode.current,mig.mode.current,power.limit,clocks.sm,clocks.mem,"
            "temperature.gpu,power.draw,utilization.gpu,memory.used,memory.total",
            "--format=csv",
        ],
        "nvcc": ["/usr/local/cuda/bin/nvcc", "--version"],
    }
    env_info: dict[str, dict] = {}
    for name, cmd in commands.items():
        executable = find_executable(cmd[0])
        if executable is None:
            env_info[name] = {"status": "missing"}
            continue
        actual_cmd = [executable] + cmd[1:]
        env_info[name] = run_logged(actual_cmd, out_dir / f"env_{name}")
    env_info["ld_preload"] = os.environ.get("LD_PRELOAD", "")
    return env_info


# Benchmarks the local gemma.c prompt runner with the requested prompt text.
def bench_local(args: argparse.Namespace, out_dir: Path) -> dict:
    if not args.skip_build:
        build = run_logged(["make", "prompt"], out_dir / "local_build")
        if build["returncode"] != 0:
            return error_result("gemma.c", "make prompt failed", build)

    binary = find_executable(args.prompt_binary)
    if binary is None:
        return error_result("gemma.c", "prompt binary not found")

    # Runs one local output length and returns mean host-visible E2E seconds.
    def run_once(output_len: int, label: str) -> tuple[float, dict]:
        cmd = [
            binary,
            "--checkpoint",
            args.checkpoint,
            "--tokenizer",
            args.tokenizer,
            "--benchmark-mode",
            "offline-decode",
            "--bench-warmup",
            str(args.warmup),
            "--bench-iters",
            str(args.iters),
            "--bench-samples",
            str(args.samples),
            "--prompt",
            args.prompt,
            "--max-new",
            str(output_len),
        ]
        run = run_logged(cmd, out_dir / f"local_{label}")
        if run["returncode"] != 0:
            raise RuntimeError(f"gemma.c {label} run failed; see {run['stderr_path']}")
        e2e_ms = parse_local_stats(run["stdout"], "e2e_ms")["mean_ms"]
        return e2e_ms / 1000.0, run

    try:
        baseline_s, baseline = run_once(args.baseline_output_len, "baseline")
        run_s, run = run_once(args.output_len, "run")
    except Exception as exc:
        return error_result("gemma.c", str(exc))

    result = decode_from_baseline("gemma.c", baseline_s, run_s, args)
    result["timing_method"] = (
        "gemma4_prompt offline-decode host latency subtraction; one final "
        "stream sync per request and no per-token D2H token copy"
    )
    result["logs"] = {
        "baseline_stdout": baseline["stdout_path"],
        "run_stdout": run["stdout_path"],
    }
    return result


# Writes bar charts from the benchmark summary.
def write_chart(
    results: list[dict],
    field: str,
    ylabel: str,
    title: str,
    path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [result["runner"] for result in results]
    values = [
        result.get(field, 0.0) if result["status"] == "ok" else 0.0
        for result in results
    ]
    max_value = max(values) if values else 1.0
    colors = ["#2563eb", "#059669", "#7c3aed"]

    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=180)
    bars = ax.bar(names, values, color=colors[: len(values)], width=0.55)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0.0, max(1.0, max_value * 1.25))
    ax.grid(axis="y", color="#d4d4d8", linewidth=0.8, alpha=0.65)
    ax.set_axisbelow(True)
    for bar, result, value in zip(bars, results, values):
        if result["status"] == "ok":
            label = f"{value:.2f}" if value < 100.0 else f"{value:.1f}"
        else:
            label = "unsupported"
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + max(1.0, max_value) * 0.03,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# Writes the compact JSON artifact and the two graphs.
def write_outputs(summary: dict, args: argparse.Namespace) -> None:
    summary_path = Path(args.summary)
    chart_dir = Path(args.chart_dir)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    chart_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    write_chart(
        summary["results"],
        "decode_tps",
        "Decode tokens / second",
        f"Decode throughput, prompt / {args.output_len} generated",
        chart_dir / "decode_tps_prompt_b1_no_spec.png",
    )
    write_chart(
        summary["results"],
        "decode_ms_per_token",
        "Milliseconds / decode token",
        f"Decode latency, prompt / {args.output_len} generated",
        chart_dir / "decode_ms_per_token_prompt_b1_no_spec.png",
    )


# Runs the local benchmark and records its explicit-prompt contract.
def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = [bench_local(args, out_dir)]

    summary = {
        "contract": {
            "prompt": args.prompt,
            "output_len": args.output_len,
            "baseline_output_len": args.baseline_output_len,
            "warmup": args.warmup,
            "measured_iterations": measured_iterations(args),
            "speculation": "not configured; no draft/speculative options passed",
            "local_timing": (
                "offline-decode host wall time with one final stream sync per "
                "request; generated tokens stay on GPU during decode"
            ),
        },
        "environment": capture_environment(out_dir),
        "results": results,
    }
    write_outputs(summary, args)
    print(json.dumps(summary["results"], indent=2))
    return 0 if any(result["status"] == "ok" for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
