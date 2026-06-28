#!/usr/bin/env python3
"""Run endpoint-free 32-token prompt / 128-token gemma.c/vLLM benchmarks."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import time


SOURCE_LINKS = {
    "vllm_latency_docs": "https://docs.vllm.ai/en/v0.23.0/cli/bench/latency/",
    "vllm_latency_source": (
        "https://github.com/vllm-project/vllm/blob/3e49479c/vllm/benchmarks/latency.py"
    ),
    "nvidia_llm_metrics": (
        "https://developer.nvidia.com/blog/llm-benchmarking-fundamental-concepts/"
    ),
}


# Parses the benchmark contract and runner locations.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark gemma.c and vLLM without serving endpoints."
    )
    parser.add_argument("--engines", default="gemma.c,vllm")
    parser.add_argument("--model", default="models/gemma-4-12B-it")
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
    parser.add_argument("--vllm-bin", default="/tmp/vllm-bench-venv/bin/vllm")
    parser.add_argument(
        "--sglang-python",
        default="/tmp/sglang-bench-venv/bin/python",
    )
    parser.add_argument("--input-len", type=int, default=32)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--baseline-output-len", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--out-dir", default="build/bench_results")
    parser.add_argument(
        "--summary",
        default="docs/benchmarks/decode_32p128g_b1_no_spec_summary.json",
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


# Benchmarks vLLM through its official in-process latency CLI, not an endpoint.
def bench_vllm(args: argparse.Namespace, out_dir: Path) -> dict:
    vllm_bin = find_executable(args.vllm_bin)
    if vllm_bin is None:
        return error_result("vLLM", "vLLM benchmark executable not found")
    # Child JIT/build helpers should resolve against the same venv as vLLM.
    env = os.environ.copy()
    env["PATH"] = str(Path(vllm_bin).parent) + os.pathsep + env.get("PATH", "")

    # Runs one vLLM output length and returns average offline latency seconds.
    def run_once(output_len: int, label: str) -> tuple[float, dict]:
        output_json = out_dir / f"vllm_{label}.json"
        cmd = [
            vllm_bin,
            "bench",
            "latency",
            "--model",
            args.model,
            "--tokenizer",
            args.model,
            "--trust-remote-code",
            "--dtype",
            args.dtype,
            "--max-model-len",
            str(args.input_len + args.output_len),
            "--input-len",
            str(args.input_len),
            "--output-len",
            str(output_len),
            "--batch-size",
            str(args.batch_size),
            "--num-iters-warmup",
            str(args.warmup),
            "--num-iters",
            str(measured_iterations(args)),
            "--disable-detokenize",
            "--no-enable-prefix-caching",
            "--output-json",
            str(output_json),
        ]
        run = run_logged(cmd, out_dir / f"vllm_{label}", env=env, timeout_s=7200)
        if run["returncode"] != 0:
            raise RuntimeError(f"vLLM {label} run failed; see {run['stderr_path']}")
        data = json.loads(output_json.read_text())
        latency_s = float(data.get("avg_latency", statistics.mean(data["latencies"])))
        return latency_s, run

    try:
        baseline_s, baseline = run_once(args.baseline_output_len, "baseline")
        run_s, run = run_once(args.output_len, "run")
    except Exception as exc:
        return error_result("vLLM", str(exc))

    result = decode_from_baseline("vLLM", baseline_s, run_s, args)
    result["timing_method"] = "vllm bench latency avg latency subtraction"
    result["logs"] = {
        "baseline_stdout": baseline["stdout_path"],
        "run_stdout": run["stdout_path"],
    }
    return result


# Benchmarks SGLang with its low-level single-batch offline latency runner.
def bench_sglang(args: argparse.Namespace, out_dir: Path) -> dict:
    sglang_python = find_executable(args.sglang_python)
    if sglang_python is None:
        return error_result("SGLang", "SGLang python executable not found")
    # FlashInfer JIT launches ninja by name, so keep the SGLang venv on PATH.
    env = os.environ.copy()
    env["PATH"] = str(Path(sglang_python).parent) + os.pathsep + env.get("PATH", "")

    result_file = out_dir / "sglang_one_batch.jsonl"
    if result_file.exists():
        result_file.unlink()

    # Batch-1 capture avoids default graph sizes that are outside this contract.
    cmd = [
        sglang_python,
        "-m",
        "sglang.bench_one_batch",
        "--model-path",
        args.model,
        "--trust-remote-code",
        "--dtype",
        args.dtype,
        "--context-length",
        str(args.input_len + args.output_len),
        "--batch-size",
        str(args.batch_size),
        "--input-len",
        str(args.input_len),
        "--output-len",
        str(args.output_len),
        "--disable-radix-cache",
        "--cuda-graph-max-bs",
        str(args.batch_size),
        "--cuda-graph-bs",
        str(args.batch_size),
        "--result-filename",
        str(result_file),
    ]
    run = run_logged(cmd, out_dir / "sglang_run", env=env, timeout_s=7200)
    if run["returncode"] != 0:
        return error_result("SGLang", "SGLang bench_one_batch failed", run)
    if not result_file.exists():
        return error_result("SGLang", "SGLang did not write a result file", run)

    lines = [line for line in result_file.read_text().splitlines() if line.strip()]
    if not lines:
        return error_result("SGLang", "SGLang wrote an empty result file", run)
    data = json.loads(lines[-1])
    latency_s = float(data["median_decode_latency"])
    decode_tps = float(data["median_decode_throughput"])
    return {
        "runner": "SGLang",
        "status": "ok",
        "decode_tokens": args.output_len - args.baseline_output_len,
        "decode_s": latency_s * (args.output_len - args.baseline_output_len),
        "decode_ms_per_token": 1000.0 * latency_s,
        "decode_tps": decode_tps,
        "timing_method": "SGLang median decode step latency",
        "prefill_s": float(data["prefill_latency"]),
        "run_s": float(data["total_latency"]),
        "logs": {"run_stdout": run["stdout_path"]},
    }


# Writes README-ready bar charts from the benchmark summary.
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


# Writes the compact JSON artifact and the two README graphs.
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
        "Decode throughput, 32 prompt tokens / 128 generated",
        chart_dir / "decode_tps_32p128g_b1_no_spec.png",
    )
    write_chart(
        summary["results"],
        "decode_ms_per_token",
        "Milliseconds / decode token",
        "Decode latency, 32 prompt tokens / 128 generated",
        chart_dir / "decode_ms_per_token_32p128g_b1_no_spec.png",
    )


# Runs the requested engines and records the shared benchmark contract.
def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    engines = [engine.strip() for engine in args.engines.split(",") if engine.strip()]
    runners = {
        "gemma.c": bench_local,
        "vllm": bench_vllm,
        "sglang": bench_sglang,
    }

    results = []
    for engine in engines:
        runner = runners.get(engine)
        if runner is None:
            results.append(error_result(engine, "unknown engine"))
            continue
        results.append(runner(args, out_dir))

    summary = {
        "contract": {
            "input_len": args.input_len,
            "output_len": args.output_len,
            "baseline_output_len": args.baseline_output_len,
            "batch_size": args.batch_size,
            "dtype": args.dtype,
            "warmup": args.warmup,
            "measured_iterations": measured_iterations(args),
            "endpoint_free": True,
            "speculation": "not configured; no draft/speculative options passed",
            "prefix_cache": "disabled where exposed",
            "local_timing": (
                "offline-decode host wall time with one final stream sync per "
                "request; generated tokens stay on GPU during decode"
            ),
            "detokenization": "disabled for vLLM; not part of local timing",
        },
        "sources": SOURCE_LINKS,
        "environment": capture_environment(out_dir),
        "results": results,
    }
    write_outputs(summary, args)
    print(json.dumps(summary["results"], indent=2))
    return 0 if any(result["status"] == "ok" for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
