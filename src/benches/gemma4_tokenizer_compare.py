#!/usr/bin/env python3
"""Compare custom, tokenizers, and transformers tokenizer throughput."""

from __future__ import annotations

import argparse
import os
import statistics
import subprocess
import time
from pathlib import Path


GIST_URL = "https://gist.github.com/dde3a2b7e698f52f389532b4b52bc254.git"
GIST_FILE = "shakespeare.txt"


# Parses command-line options for a small tokenizer throughput comparison.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer-dir", default="models/gemma-4-12B")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--sleep-ms", type=int, default=250)
    return parser.parse_args()


# Runs one setup command outside the measured tokenizer loops.
def run_setup(command: list[str]) -> str:
    result = subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


# Clones the requested gist once and returns the local text file path.
def ensure_gist_data(repo_root: Path) -> tuple[Path, str]:
    gist_dir = repo_root / "build" / "benches" / "tokenizer_bench_gist"
    if not gist_dir.exists():
        gist_dir.parent.mkdir(parents=True, exist_ok=True)
        run_setup(["git", "clone", "--depth", "1", GIST_URL, str(gist_dir)])

    commit = run_setup(["git", "-C", str(gist_dir), "rev-parse", "HEAD"])
    data_path = gist_dir / GIST_FILE
    if not data_path.exists():
        raise FileNotFoundError(data_path)
    return data_path, commit


# Loads non-empty lines before timing so disk I/O and splitting are excluded.
def load_texts(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    texts = [line for line in text.splitlines() if line]
    if not texts:
        raise ValueError("benchmark data has no non-empty lines")
    return texts


# Repeats one full-dataset encode callable and returns the consumed token count.
def run_iters(encode_once, iters: int) -> int:
    tokens = 0
    for _ in range(iters):
        tokens += encode_once()
    return tokens


# Times one initialized tokenizer without printing inside the measured loop.
def bench_impl(name: str, encode_once, texts: list[str], args: argparse.Namespace) -> None:
    checksum = 0
    for _ in range(args.warmup):
        checksum += run_iters(encode_once, 1)

    time.sleep(args.sleep_ms / 1000.0)

    samples = []
    sample_tokens = 0
    for _ in range(args.samples):
        start = time.perf_counter()
        sample_tokens = run_iters(encode_once, args.iters)
        elapsed = time.perf_counter() - start
        checksum += sample_tokens
        samples.append(elapsed * 1000.0)

    median_ms = statistics.median(samples)
    seconds = median_ms / 1000.0
    docs = len(texts) * args.iters
    mib = sum(len(text.encode("utf-8")) for text in texts) * args.iters
    mib /= 1024.0 * 1024.0
    print(
        f"tokenizer_bench impl={name} warmup={args.warmup} "
        f"iters={args.iters} samples={args.samples} docs_per_iter={len(texts)} "
        f"median_ms={median_ms:.3f} min_ms={min(samples):.3f} "
        f"max_ms={max(samples):.3f} docs_per_s={docs / seconds:.2f} "
        f"mib_per_s={mib / seconds:.2f} "
        f"tokens_per_s={sample_tokens / seconds:.2f} "
        f"tokens_per_sample={sample_tokens} checksum={checksum}"
    )


# Runs the compiled custom-tokenizer benchmark in its own initialized process.
def run_custom(repo_root: Path, tokenizer_json: Path, data_path: Path, args) -> None:
    binary = repo_root / "build" / "benches" / "gemma4_tokenizer_bench"
    if not binary.exists():
        raise FileNotFoundError(f"{binary}; run `make tokenizer-bench` first")
    subprocess.run(
        [
            str(binary),
            str(tokenizer_json),
            str(data_path),
            str(args.warmup),
            str(args.iters),
            str(args.samples),
            str(args.sleep_ms),
        ],
        check=True,
    )


# Loads all tokenizers once, then prints comparable throughput rows.
def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    tokenizer_dir = (repo_root / args.tokenizer_dir).resolve()
    tokenizer_json = tokenizer_dir / "tokenizer.json"

    if args.data is None:
        data_path, commit = ensure_gist_data(repo_root)
    else:
        data_path = args.data.resolve()
        commit = "external"

    texts = load_texts(data_path)
    byte_count = sum(len(text.encode("utf-8")) for text in texts)
    print(
        f"tokenizer_bench_source gist_url={GIST_URL} gist_commit={commit} "
        f"data_file={data_path} docs={len(texts)} bytes={byte_count}",
        flush=True,
    )
    print(
        "tokenizer_bench_contract init_excluded=true data_load_excluded=true "
        "timing=steady_clock_cpp_or_perf_counter_python "
        "aggregation=median_min_max warmup_before_timing=true "
        "no_output_inside_timed_loop=true "
        f"tokenizers_parallelism={os.environ.get('TOKENIZERS_PARALLELISM', 'default')}",
        flush=True,
    )

    run_custom(repo_root, tokenizer_json, data_path, args)

    from tokenizers import Tokenizer
    from transformers import AutoTokenizer

    uv_tokenizer = Tokenizer.from_file(str(tokenizer_json))
    hf_tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), use_fast=True)

    bench_impl(
        "uv_tokenizers_batch",
        lambda: sum(
            len(encoding.ids)
            for encoding in uv_tokenizer.encode_batch(
                texts,
                add_special_tokens=False,
            )
        ),
        texts,
        args,
    )
    bench_impl(
        "transformers_auto_batch",
        lambda: sum(
            len(ids)
            for ids in hf_tokenizer(
                texts,
                add_special_tokens=False,
                padding=False,
                truncation=False,
            )["input_ids"]
        ),
        texts,
        args,
    )


if __name__ == "__main__":
    main()
