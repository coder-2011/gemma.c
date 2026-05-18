# llama.cpp Reference Snapshot

Source repository: `ggml-org/llama.cpp`

Pinned commit: `dd7cad7197f991b18ded6aca46ff095972b95318`

Downloaded on: 2026-05-18

This folder is a local reference snapshot for Gemma 4 dense inference work in
`gemma.c`. It preserves upstream llama.cpp paths under this directory so files
can be compared against the original tree without path translation.

Included material:

- CUDA GEMM/GEMV and matrix-vector references: `ggml/src/ggml-cuda/mmf*`,
  `mmvf*`, `mmq*`, `mmvq*`, `mmid*`, `mma.cuh`, `vecdotq.cuh`, quantize and
  dequantize helpers, plus the main CUDA backend dispatch file.
- CPU and library fallback references: CPU `mul_mat` flow, vector dot and
  quantization traits, repacked paths, AMX/MMQ, BLAS, and llamafile SGEMM.
- Cross-backend GEMM/GEMV references: OpenCL GEMM/GEMV kernels plus compact
  SYCL, Vulkan, WebGPU, and zDNN matmul files.
- Gemma 4 references: converter, GGUF tensor constants/mapping, model graph,
  architecture/hparams/model/vocab files, chat templates, tokenizer fixtures,
  tests, and multimodal projector code.

Generated files:

- `SOURCE_PATHS.txt` lists the upstream paths copied into this snapshot.
- `MANIFEST.tsv` maps each copied path to its pinned GitHub source URL.

The binary tokenizer fixture `models/ggml-vocab-gemma-4.gguf` is included
because it is useful for validating Gemma 4 tokenization behavior.
