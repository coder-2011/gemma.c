#!/usr/bin/env python3
import argparse, re, statistics, subprocess

import torch


HIDDEN, VOCAB, EMBED_SCALE = 3840, 262144, 62.0
CUSTOM_BENCH = "build/benches/gemma4_sampling_bench"

parser = argparse.ArgumentParser()
parser.add_argument("--warmup", type=int, default=25)
parser.add_argument("--iters", type=int, default=100)
parser.add_argument("--samples", type=int, default=21)
args = parser.parse_args()

torch.manual_seed(0x5678)
lm_head = torch.empty((VOCAB, HIDDEN), device="cuda", dtype=torch.bfloat16)
lm_head.uniform_(-0.05, 0.05)
lm_head_t = lm_head.t()

torch.manual_seed(0x2468)
hidden = torch.empty((1, HIDDEN), device="cuda", dtype=torch.bfloat16)
hidden.uniform_(-0.05, 0.05)

logits = torch.empty((1, VOCAB), device="cuda", dtype=torch.bfloat16)
values = torch.empty((1,), device="cuda", dtype=torch.bfloat16)
token = torch.empty((1,), device="cuda", dtype=torch.long)
selected = torch.empty_like(hidden)
next_hidden = torch.empty_like(hidden)

for _ in range(3):
    torch.mm(hidden, lm_head_t, out=logits)
    torch.max(logits, dim=1, out=(values, token))
    torch.index_select(lm_head, 0, token, out=selected)
    torch.mul(selected, EMBED_SCALE, out=next_hidden)
torch.cuda.synchronize()

# Capture the fixed native PyTorch LM-head, argmax, and gather work.
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    torch.mm(hidden, lm_head_t, out=logits)
    torch.max(logits, dim=1, out=(values, token))
    torch.index_select(lm_head, 0, token, out=selected)
    torch.mul(selected, EMBED_SCALE, out=next_hidden)
graph.replay()
torch.cuda.synchronize()

chosen = int(token.item())
assert torch.equal(next_hidden, lm_head[chosen : chosen + 1] * EMBED_SCALE)

for _ in range(args.warmup):
    graph.replay()
torch.cuda.synchronize()

pytorch = []
start = torch.cuda.Event(enable_timing=True)
stop = torch.cuda.Event(enable_timing=True)
for _ in range(args.samples):
    start.record()
    for _ in range(args.iters):
        graph.replay()
    stop.record()
    stop.synchronize()
    pytorch.append(start.elapsed_time(stop) * 1000.0 / args.iters)
pytorch_median = statistics.median(pytorch)

output = subprocess.check_output(
    [CUSTOM_BENCH, str(args.warmup), str(args.iters), str(args.samples)],
    stderr=subprocess.STDOUT,
    text=True,
)
custom = float(re.search(
    r"variant=fused_lm_head_sample_full_vocab .*?median_us=([0-9.]+)",
    output,
).group(1))

print(f"benchmark_env gpu=\"{torch.cuda.get_device_name()}\" torch={torch.__version__}")
print("benchmark_contract timing=cuda_events_cuda_graph cache=warm_repeated_buffers")
print(f"correctness selected_row_gather=passed token={chosen}")
print(f"variant=native_pytorch_cuda_graph median_us={pytorch_median:.3f} samples_us={pytorch}")
print(f"variant=custom_cuda_bench median_us={custom:.3f}")
print(f"speedup custom_vs_pytorch={pytorch_median / custom:.3f}x")
