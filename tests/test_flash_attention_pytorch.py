#!/usr/bin/env python3
import ctypes
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.benches.gemma4_paged_decode_torch_bench import (  # noqa: E402
    check_status,
    load_lib,
    make_decode_inputs,
    max_abs,
    ptr,
    torch_decode_attention,
)

MAX_ABS_TOLERANCE = 0.015625


# Compare sliding paged decode against the PyTorch softmax reference.
def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    args = SimpleNamespace(
        batch_size=2,
        seq_len=10,
        page_size=4,
        split_size=3,
        seed=1234,
    )
    lib_path = os.environ.get(
        "GEMMA4_FLASH_ATTENTION_LIB",
        str(ROOT / "build/libgemma4_flash_attention.so"),
    )
    kv_write, decode_direct, _ = load_lib(lib_path)
    device = torch.device("cuda")
    decode = make_decode_inputs(args, device)
    config = decode["config"]
    key_count = decode["key_count"]
    # Keep this tiny fixture inside the decode launcher's split-capacity guard.
    config.window_size = key_count
    q_heads = decode["q"].shape[1]
    head_dim = decode["q"].shape[2]
    num_splits = math.ceil(key_count / args.split_size)
    scale = 1.0 / math.sqrt(head_dim)

    cache_k = torch.zeros(decode["cache_shape"], device=device, dtype=torch.bfloat16)
    cache_v = torch.zeros_like(cache_k)
    out_custom = torch.empty_like(decode["q"])
    out_torch = torch.empty_like(decode["q"])
    partial_m = torch.empty(
        args.batch_size * q_heads * num_splits,
        device=device,
        dtype=torch.float32,
    )
    partial_l = torch.empty_like(partial_m)
    partial_acc = torch.empty(
        partial_m.numel() * head_dim,
        device=device,
        dtype=torch.float32,
    )
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)

    # First write the PyTorch K/V window into the same paged cache used by CUDA.
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
            stream,
        ),
        "gemma4_kv_cache_write_bf16",
    )

    # Then compare the CUDA decode output with the same tensors evaluated by PyTorch.
    check_status(
        decode_direct(
            ptr(out_custom),
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
            stream,
        ),
        "gemma4_flash_attention_sliding_decode_paged_bf16",
    )
    torch_decode_attention(
        decode["q"],
        decode["k_window"],
        decode["v_window"],
        out_torch,
        scale,
    )
    torch.cuda.synchronize()

    diff = max_abs(out_custom, out_torch)
    if diff > MAX_ABS_TOLERANCE:
        raise RuntimeError(f"sliding decode vs PyTorch max_abs={diff}")
    print(f"flash attention PyTorch decode parity max_abs={diff:.8g}")


if __name__ == "__main__":
    main()
