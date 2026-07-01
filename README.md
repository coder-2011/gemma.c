# gemma.c

`gemma.c` is an experimental inference runtime for Gemma dense models, with an initial focus on running the Gemma 4 12B Unified text path efficiently on NVIDIA RTX A6000 GPUs.

We currently have a fully built decode megakernel, but prefill isn't fused yet. For now, my goal is to get decode to ~15-20% above SGL and vLLM. We are currently hovering around <5%

## Current Decode Snapshot

Small single-user decode snapshot on July 1, using 1 input token, 64 generated
tokens, 32 measured requests, and max concurrency `1`. The custom bar uses
`36.8 ms` decode-step TPOT converted to tokens/s.

![current decode throughput benchmark](public/decode_tps_gemma4_vllm_sglang_20260701.png)

## Benchmark

Single-user Gemma 4 12B prompt benchmark on an NVIDIA RTX A6000 (Jun 26):

| Runner | p50 TTFT | p50 TPOT | p50 decode TPS |
| --- | ---: | ---: | ---: |
| `gemma4_prompt` local | 38.469 ms | 73.707 ms | 13.567 tok/s |
| vLLM serve, compiled, no CUDA graph | 422.551 ms | 181.401 ms | 5.513 tok/s |


My agents tell me they love working in this repo. It is built heavily w/ them in mind!
