#!/usr/bin/env python3
import ctypes
import math
import os
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]

GEMMA4_NUM_QUERY_HEADS = 16
GEMMA4_SLIDING_KV_HEADS = 8
GEMMA4_SLIDING_HEAD_DIM = 256
GEMMA4_SLIDING_WINDOW = 1024
MAX_ABS_TOLERANCE = 0.015625


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


def ptr(tensor):
    return ctypes.c_void_p(tensor.data_ptr())


def check_status(status, label):
    if status != 0:
        raise RuntimeError(f"{label} returned cudaError_t={status}")


def load_lib(path):
    lib = ctypes.CDLL(path)
    kv_write = lib.gemma4_kv_cache_write_bf16
    kv_write.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        Gemma4KvCacheConfig,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    kv_write.restype = ctypes.c_int

    decode_direct = lib.gemma4_flash_attention_sliding_decode_paged_bf16
    decode_direct.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        Gemma4KvCacheConfig,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_float,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_void_p,
    ]
    decode_direct.restype = ctypes.c_int
    return kv_write, decode_direct


def build_page_table(batch_size, seq_len, page_size):
    first_key = max(0, seq_len - GEMMA4_SLIDING_WINDOW)
    key_count = seq_len - first_key
    max_pages_per_seq = max(
        math.ceil(GEMMA4_SLIDING_WINDOW / page_size) + 1,
        math.ceil(key_count / page_size) + 1,
    )
    page_table = torch.full((batch_size, max_pages_per_seq), -1, dtype=torch.int32)
    next_page = 0
    for batch in range(batch_size):
        for pos in range(first_key, seq_len):
            slot = (pos // page_size) % max_pages_per_seq
            if page_table[batch, slot] < 0:
                page_table[batch, slot] = next_page
                next_page += 1
    return page_table, first_key, key_count, max_pages_per_seq, batch_size * max_pages_per_seq


def make_decode_inputs(args, device):
    torch.manual_seed(args.seed)
    page_table_cpu, first_key, key_count, max_pages_per_seq, num_pages = (
        build_page_table(args.batch_size, args.seq_len, args.page_size)
    )
    config = Gemma4KvCacheConfig(
        1,
        num_pages,
        args.page_size,
        max_pages_per_seq,
        GEMMA4_SLIDING_KV_HEADS,
        GEMMA4_SLIDING_HEAD_DIM,
        GEMMA4_SLIDING_WINDOW,
    )
    dtype = torch.bfloat16
    q = torch.randn(
        args.batch_size,
        GEMMA4_NUM_QUERY_HEADS,
        GEMMA4_SLIDING_HEAD_DIM,
        device=device,
        dtype=dtype,
    )
    k_window = torch.randn(
        args.batch_size,
        key_count,
        GEMMA4_SLIDING_KV_HEADS,
        GEMMA4_SLIDING_HEAD_DIM,
        device=device,
        dtype=dtype,
    )
    v_window = torch.randn_like(k_window)
    token_batch = torch.arange(args.batch_size, device=device, dtype=torch.int32)
    token_batch = token_batch.repeat_interleave(key_count)
    token_positions = torch.arange(first_key, args.seq_len, device=device, dtype=torch.int32)
    token_positions = token_positions.repeat(args.batch_size)
    k_flat = k_window.reshape(args.batch_size * key_count, GEMMA4_SLIDING_KV_HEADS, -1)
    v_flat = v_window.reshape_as(k_flat)
    cache_shape = (
        config.num_layers,
        config.num_pages,
        config.page_size,
        config.num_heads,
        config.head_dim,
    )
    return {
        "config": config,
        "key_count": key_count,
        "page_table": page_table_cpu.to(device),
        "seq_lengths": torch.full((args.batch_size,), args.seq_len, device=device, dtype=torch.int32),
        "token_batch": token_batch.contiguous(),
        "token_positions": token_positions.contiguous(),
        "q": q.contiguous(),
        "k_window": k_window.contiguous(),
        "v_window": v_window.contiguous(),
        "k_flat": k_flat.contiguous(),
        "v_flat": v_flat.contiguous(),
        "cache_shape": cache_shape,
    }


def torch_decode_attention(q, k_window, v_window, out, scale):
    group = GEMMA4_NUM_QUERY_HEADS // GEMMA4_SLIDING_KV_HEADS
    kv_map = torch.arange(GEMMA4_NUM_QUERY_HEADS, device=q.device) // group
    k_gqa = k_window[:, :, kv_map, :].float()
    v_gqa = v_window[:, :, kv_map, :].float()
    scores = (q[:, None, :, :].float() * k_gqa).sum(dim=-1) * scale
    probs = torch.softmax(scores, dim=1)
    out.copy_((probs[..., None] * v_gqa).sum(dim=1).to(torch.bfloat16))


def max_abs(a, b):
    return float((a.float() - b.float()).abs().max().item())


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
    kv_write, decode_direct = load_lib(lib_path)
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
