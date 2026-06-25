import json
import subprocess

import torch


def _capture(command):
    """Return one command output line for best-effort benchmark metadata."""
    try:
        result = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except OSError:
        return "unavailable"
    line = result.stdout.splitlines()[0] if result.stdout.splitlines() else ""
    return line if line else "unavailable"


def benchmark_environment(name):
    """Return CUDA, device, and nvidia-smi metadata for benchmark comparisons."""
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    smi = _capture(
        [
            "nvidia-smi",
            "--query-gpu=name,gpu_bus_id,driver_version,persistence_mode,"
            "ecc.mode.current,mig.mode.current,power.limit,clocks.sm,"
            "clocks.mem,temperature.gpu,power.draw,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    return {
        "name": name,
        "cuda_device": device,
        "gpu": props.name,
        "global_mem_bytes": props.total_memory,
        "l2_cache_bytes": getattr(props, "L2_cache_size", None),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "nvidia_smi_fields": (
            "name,bus_id,driver,persistence,ecc,mig,power_limit_w,"
            "sm_clock_mhz,mem_clock_mhz,temp_c,power_draw_w,gpu_util_pct"
        ),
        "nvidia_smi": smi,
    }


def print_benchmark_environment(name):
    """Print CUDA, device, and nvidia-smi metadata for benchmark comparisons."""
    payload = benchmark_environment(name)
    print("benchmark_env " + json.dumps(payload, sort_keys=True))


def benchmark_contract(**kwargs):
    """Return the measurement contract for JSON benchmark reports."""
    payload = {
        "measurement": "kernel_only",
        "timing": "cuda_events_same_stream",
        "cache": "unspecified",
        "launch_overhead": "graph_replay_excluded_capture_cost",
        "aggregation": "median_trimmed_mean_p95_p99_stddev_iqr_raw_samples",
        "correctness": "checked_before_timing",
        "clock_policy": "unlocked_boost",
        "no_sleep_between_trials": True,
        "ncu_clock_control": "use_--clock-control_none",
    }
    payload.update(kwargs)
    return payload


def print_benchmark_contract(**kwargs):
    """Print the measurement contract beside timing rows."""
    payload = benchmark_contract(**kwargs)
    print("benchmark_contract " + json.dumps(payload, sort_keys=True))
