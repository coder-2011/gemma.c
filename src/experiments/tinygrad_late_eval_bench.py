#!/usr/bin/env python3
import ctypes
import os
import statistics
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TINYGRAD_ROOT = REPO_ROOT / "experiments" / "tinygrad"
CUDA12_LIB = Path("/usr/local/cuda-12/targets/x86_64-linux/lib")
CUDA12_INCLUDE = Path("/usr/local/cuda-12/include")
BUILD_DIR = REPO_ROOT / "build" / "experiments"
sys.path.insert(0, str(TINYGRAD_ROOT))

from tinygrad import Context, Tensor, dtypes
from tinygrad.dtype import DTYPES_DICT
from tinygrad.engine.realize import ExecContext, pm_beam, pm_compile, pm_exec, pm_optimize_local_size, pm_validate
from tinygrad.helpers import BEAM, DEBUG, GlobalCounters, VALIDATE_WITH_CPU, getenv
from tinygrad.uop.ops import Ops, UOp, graph_rewrite

CUDA_SUCCESS = 0
CUBLAS_STATUS_SUCCESS = 0
CUDA_STREAM_NON_BLOCKING = 1
CUDA_R_32F = 0
CUDA_R_16F = 2
CUDA_R_16BF = 14
CUBLAS_OP_N = 0
CUBLAS_COMPUTE_32F = 68
CUBLAS_COMPUTE_32F_FAST_16BF = 75
CUBLAS_GEMM_DEFAULT_TENSOR_OP = 99


def parse_int_list(raw: str) -> list[int]:
  values = []
  for part in raw.split(","):
    part = part.strip()
    if not part:
      continue
    if "-" in part:
      lo, hi = [int(x) for x in part.split("-", 1)]
      values.extend(range(lo, hi + 1))
    else:
      values.append(int(part))
  return values


def copied_compile_linear(linear: UOp, beam: int | None = None, validate=False) -> UOp:
  if validate:
    linear = graph_rewrite(linear, pm_validate, name="validate", walk=True)
  if (beam_val := BEAM.value if beam is None else beam) >= 1:
    linear = graph_rewrite(linear, pm_beam, ctx=beam_val, walk=True)
  linear = graph_rewrite(linear, pm_compile, name="precompile kernels", walk=True)
  return graph_rewrite(linear, pm_optimize_local_size, name="optimize local size", walk=True)


def copied_run_linear(linear: UOp, var_vals: dict[str, int] | None = None, input_uops: tuple[UOp, ...] = (),
                      update_stats=True, jit=False, wait=False):
  if not jit:
    linear = copied_compile_linear(linear, validate=VALIDATE_WITH_CPU)
  ctx = ExecContext(var_vals or {}, input_uops, update_stats, jit, wait or DEBUG >= 2)
  for call in linear.src:
    pm_exec.rewrite(call, ctx)


def parse_m_list() -> list[int]:
  raw = os.environ.get("M_LIST", os.environ.get("M", "64"))
  return [int(x) for x in raw.split(",") if x.strip()]


def program_names(linear) -> list[str]:
  names = []
  for call in linear.src:
    ast = call.src[0]
    if ast.op is Ops.PROGRAM:
      names.append(ast.arg.name)
  return names


def build_late_matmul(m: int, n: int, k: int, dtype_in, dtype_acc):
  a = Tensor.empty(m, k, dtype=dtype_in)
  b = Tensor.empty(k, n, dtype=dtype_in)
  c = a.matmul(b, dtype=dtype_acc)
  return c.schedule_linear()


def time_tinygrad_beams(m: int, n: int, k: int, dtype_in, dtype_acc, warmup: int,
                        cnt: int, beams: list[int]) -> tuple[float, float, int, str]:
  best_time = None
  best_median = 0.0
  best_beam = beams[0]
  best_programs = "none"
  for beam in beams:
    linear = build_late_matmul(m, n, k, dtype_in, dtype_acc)
    times, compiled = time_late_linear(linear, warmup, cnt, beam)
    candidate_best = min(times)
    if best_time is None or candidate_best < best_time:
      best_time = candidate_best
      best_median = statistics.median(times)
      best_beam = beam
      best_programs = ",".join(program_names(compiled)) or "none"
  return best_time or 0.0, best_median, best_beam, best_programs


def time_late_linear(linear, warmup: int, cnt: int, beam: int) -> tuple[list[float], object]:
  # This mirrors tinygrad's late path:
  # Tensor.schedule_linear builds a LINEAR UOp, compile_linear lowers it to PROGRAM UOps,
  # and run_linear(..., jit=True, wait=True) executes the already-compiled program.
  with Context(DEBUG=getenv("DEBUG", 0), BEAM=0):
    compiled = copied_compile_linear(linear, beam=beam)
    for _ in range(warmup):
      copied_run_linear(compiled, jit=True, wait=True)
    times = []
    for _ in range(cnt):
      GlobalCounters.reset()
      copied_run_linear(compiled, jit=True, wait=True)
      times.append(GlobalCounters.time_sum_s)
  return times, compiled


def dtype_cuda_info(dtype) -> tuple[int, int]:
  if dtype is dtypes.float:
    return CUDA_R_32F, 4
  if dtype is dtypes.half:
    return CUDA_R_16F, 2
  if dtype is dtypes.bfloat16:
    return CUDA_R_16BF, 2
  raise ValueError(f"unsupported cuBLAS dtype: {dtype}")


class CudaCublas:
  def __init__(self):
    cudart_path = CUDA12_LIB / "libcudart.so.12"
    cublas_path = CUDA12_LIB / "libcublas.so.12"
    self.cudart = ctypes.CDLL(str(cudart_path) if cudart_path.exists() else "libcudart.so")
    self.cublas = ctypes.CDLL(str(cublas_path) if cublas_path.exists() else "libcublas.so")
    self.stream = ctypes.c_void_p()
    self.handle = ctypes.c_void_p()
    self._configure_apis()
    self._cuda(self.cudart.cudaSetDevice(0), "cudaSetDevice")
    self._cuda(self.cudart.cudaStreamCreateWithFlags(ctypes.byref(self.stream),
                                                     CUDA_STREAM_NON_BLOCKING),
               "cudaStreamCreateWithFlags")
    self._cublas(self.cublas.cublasCreate_v2(ctypes.byref(self.handle)), "cublasCreate")
    self._cublas(self.cublas.cublasSetStream_v2(self.handle, self.stream), "cublasSetStream")

  def _configure_apis(self) -> None:
    self.cudart.cudaSetDevice.argtypes = [ctypes.c_int]
    self.cudart.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    self.cudart.cudaFree.argtypes = [ctypes.c_void_p]
    self.cudart.cudaMemset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
    self.cudart.cudaStreamCreateWithFlags.argtypes = [ctypes.POINTER(ctypes.c_void_p),
                                                      ctypes.c_uint]
    self.cudart.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
    self.cudart.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
    self.cudart.cudaEventCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    self.cudart.cudaEventRecord.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    self.cudart.cudaEventSynchronize.argtypes = [ctypes.c_void_p]
    self.cudart.cudaEventElapsedTime.argtypes = [ctypes.POINTER(ctypes.c_float),
                                                 ctypes.c_void_p, ctypes.c_void_p]
    self.cudart.cudaEventDestroy.argtypes = [ctypes.c_void_p]
    self.cublas.cublasCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    self.cublas.cublasDestroy_v2.argtypes = [ctypes.c_void_p]
    self.cublas.cublasSetStream_v2.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    self.cublas.cublasGemmEx.argtypes = [
      ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
      ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
      ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
      ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ]

  def _cuda(self, status: int, name: str) -> None:
    if status != CUDA_SUCCESS:
      raise RuntimeError(f"{name} failed with cuda status {status}")

  def _cublas(self, status: int, name: str) -> None:
    if status != CUBLAS_STATUS_SUCCESS:
      raise RuntimeError(f"{name} failed with cuBLAS status {status}")

  def malloc(self, size: int) -> ctypes.c_void_p:
    ptr = ctypes.c_void_p()
    self._cuda(self.cudart.cudaMalloc(ctypes.byref(ptr), size), "cudaMalloc")
    self._cuda(self.cudart.cudaMemset(ptr, 0x3f, size), "cudaMemset")
    return ptr

  def free(self, ptr: ctypes.c_void_p) -> None:
    if ptr:
      self._cuda(self.cudart.cudaFree(ptr), "cudaFree")

  def destroy(self) -> None:
    if self.handle:
      self._cublas(self.cublas.cublasDestroy_v2(self.handle), "cublasDestroy")
      self.handle = ctypes.c_void_p()
    if self.stream:
      self._cuda(self.cudart.cudaStreamDestroy(self.stream), "cudaStreamDestroy")
      self.stream = ctypes.c_void_p()

  def gemm_ex_status(self, m: int, n: int, k: int, a, b, c, input_type: int,
                     output_type: int, compute_type: int, algo: int) -> int:
    alpha = ctypes.c_float(1.0)
    beta = ctypes.c_float(0.0)
    # cuBLAS is column-major. This computes the byte-equivalent of row-major
    # C[m,n] = A[m,k] * B[k,n] as C_col[n,m] = B_col[n,k] * A_col[k,m].
    return self.cublas.cublasGemmEx(
      self.handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, ctypes.byref(alpha),
      b, input_type, n, a, input_type, k, ctypes.byref(beta), c, output_type, n,
      compute_type, algo,
    )

  def gemm_ex(self, m: int, n: int, k: int, a, b, c, input_type: int,
              output_type: int, compute_type: int, algo: int) -> None:
    self._cublas(self.gemm_ex_status(m, n, k, a, b, c, input_type, output_type,
                                     compute_type, algo), "cublasGemmEx")

  def time_gemm_ex(self, m: int, n: int, k: int, dtype_in, dtype_acc,
                   warmup: int, cnt: int) -> list[float]:
    results = self.time_gemm_candidates(m, n, k, dtype_in, dtype_acc, warmup, cnt, [
      ("fast16bf" if getenv("CUBLAS_FAST_16BF", 0) else "32f",
       CUBLAS_COMPUTE_32F_FAST_16BF if getenv("CUBLAS_FAST_16BF", 0) else CUBLAS_COMPUTE_32F,
       CUBLAS_GEMM_DEFAULT_TENSOR_OP),
    ])
    if not results:
      raise RuntimeError("no working cuBLAS candidate")
    return results[0]["times"]

  def time_gemm_candidates(self, m: int, n: int, k: int, dtype_in, dtype_acc,
                           warmup: int, cnt: int,
                           candidates: list[tuple[str, int, int]]) -> list[dict]:
    self._cuda(self.cudart.cudaSetDevice(0), "cudaSetDevice")
    input_type, input_bytes = dtype_cuda_info(dtype_in)
    output_type, output_bytes = dtype_cuda_info(dtype_acc)
    a = b = c = start = stop = None
    try:
      a = self.malloc(m * k * input_bytes)
      b = self.malloc(k * n * input_bytes)
      c = self.malloc(m * n * output_bytes)
      start = ctypes.c_void_p()
      stop = ctypes.c_void_p()
      self._cuda(self.cudart.cudaEventCreate(ctypes.byref(start)), "cudaEventCreate(start)")
      self._cuda(self.cudart.cudaEventCreate(ctypes.byref(stop)), "cudaEventCreate(stop)")
      results = []
      for name, compute_type, algo in candidates:
        if self.gemm_ex_status(m, n, k, a, b, c, input_type, output_type, compute_type,
                               algo) != CUBLAS_STATUS_SUCCESS:
          continue
        self._cuda(self.cudart.cudaStreamSynchronize(self.stream), "cudaStreamSynchronize")
        for _ in range(warmup):
          self.gemm_ex(m, n, k, a, b, c, input_type, output_type, compute_type, algo)
        self._cuda(self.cudart.cudaStreamSynchronize(self.stream), "cudaStreamSynchronize")
        times = []
        for _ in range(cnt):
          elapsed_ms = ctypes.c_float()
          self._cuda(self.cudart.cudaEventRecord(start, self.stream), "cudaEventRecord(start)")
          self.gemm_ex(m, n, k, a, b, c, input_type, output_type, compute_type, algo)
          self._cuda(self.cudart.cudaEventRecord(stop, self.stream), "cudaEventRecord(stop)")
          self._cuda(self.cudart.cudaEventSynchronize(stop), "cudaEventSynchronize(stop)")
          self._cuda(self.cudart.cudaEventElapsedTime(ctypes.byref(elapsed_ms), start, stop),
                     "cudaEventElapsedTime")
          times.append(elapsed_ms.value / 1e3)
        results.append({
          "name": name, "compute_type": compute_type, "algo": algo, "times": times,
          "best": min(times), "median": statistics.median(times),
        })
      return sorted(results, key=lambda x: x["best"])
    finally:
      if start:
        self._cuda(self.cudart.cudaEventDestroy(start), "cudaEventDestroy(start)")
      if stop:
        self._cuda(self.cudart.cudaEventDestroy(stop), "cudaEventDestroy(stop)")
      self.free(a)
      self.free(b)
      self.free(c)


def build_cublaslt_helper() -> Path:
  BUILD_DIR.mkdir(parents=True, exist_ok=True)
  src = BUILD_DIR / "tinygrad_late_eval_cublaslt_helper.cc"
  so = BUILD_DIR / "libtinygrad_late_eval_cublaslt_helper.so"
  code = r'''
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include <cuda_runtime_api.h>
#include <cublasLt.h>

#define CHECK_CUDA(call) do { cudaError_t st = (call); if (st != cudaSuccess) return 100000 + (int)st; } while (0)
#define CHECK_CUBLAS(call) do { cublasStatus_t st = (call); if (st != CUBLAS_STATUS_SUCCESS) return 200000 + (int)st; } while (0)

extern "C" int gemma4_bench_cublaslt_bf16_f32(int m, int n, int k, int warmup, int cnt,
                                              size_t workspace_size, int requested_algos,
                                              int tune_algos, float *best_us,
                                              float *median_us, int *best_algo_rank,
                                              int *returned_algos) {
  if (m <= 0 || n <= 0 || k <= 0 || warmup < 0 || cnt <= 0) return 1;
  CHECK_CUDA(cudaSetDevice(0));

  cudaStream_t stream = nullptr;
  CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  cublasLtHandle_t handle = nullptr;
  CHECK_CUBLAS(cublasLtCreate(&handle));

  void *a = nullptr, *b = nullptr, *c = nullptr, *workspace = nullptr;
  CHECK_CUDA(cudaMalloc(&a, (size_t)m * (size_t)k * 2));
  CHECK_CUDA(cudaMalloc(&b, (size_t)k * (size_t)n * 2));
  CHECK_CUDA(cudaMalloc(&c, (size_t)m * (size_t)n * 4));
  CHECK_CUDA(cudaMemset(a, 0x3f, (size_t)m * (size_t)k * 2));
  CHECK_CUDA(cudaMemset(b, 0x3f, (size_t)k * (size_t)n * 2));
  CHECK_CUDA(cudaMemset(c, 0, (size_t)m * (size_t)n * 4));
  if (workspace_size > 0) CHECK_CUDA(cudaMalloc(&workspace, workspace_size));

  cublasLtMatmulDesc_t op_desc = nullptr;
  cublasLtMatrixLayout_t a_desc = nullptr, b_desc = nullptr, c_desc = nullptr;
  cublasLtMatmulPreference_t pref = nullptr;

  CHECK_CUBLAS(cublasLtMatmulDescCreate(&op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  // cuBLASLt is column-major here. This computes the byte-equivalent of
  // row-major C[m,n] = A[m,k] * B[k,n] as C_col[n,m] = B_col[n,k] * A_col[k,m].
  CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&a_desc, CUDA_R_16BF, (uint64_t)n, (uint64_t)k, (int64_t)n));
  CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&b_desc, CUDA_R_16BF, (uint64_t)k, (uint64_t)m, (int64_t)k));
  CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&c_desc, CUDA_R_32F, (uint64_t)n, (uint64_t)m, (int64_t)n));
  CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&pref));
  CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(
      pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size, sizeof(workspace_size)));

  requested_algos = std::max(1, requested_algos);
  std::vector<cublasLtMatmulHeuristicResult_t> heuristics((size_t)requested_algos);
  int returned = 0;
  CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic(
      handle, op_desc, a_desc, b_desc, c_desc, c_desc, pref, requested_algos,
      heuristics.data(), &returned));
  if (returned <= 0 || heuristics[0].state != CUBLAS_STATUS_SUCCESS) return 2;
  *returned_algos = returned;

  float alpha = 1.0f, beta = 0.0f;
  cudaEvent_t start = nullptr, stop = nullptr;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));
  tune_algos = std::max(1, std::min(tune_algos, returned));
  float best_candidate = 1.0e30f;
  float best_candidate_median = 0.0f;
  int best_rank = -1;
  for (int rank = 0; rank < tune_algos; ++rank) {
    if (heuristics[rank].state != CUBLAS_STATUS_SUCCESS) continue;
    bool usable = true;
    for (int i = 0; i < warmup; ++i) {
      cublasStatus_t st = cublasLtMatmul(
          handle, op_desc, &alpha, b, a_desc, a, b_desc, &beta, c, c_desc, c, c_desc,
          &heuristics[rank].algo, workspace, workspace_size, stream);
      if (st != CUBLAS_STATUS_SUCCESS) { usable = false; break; }
    }
    if (!usable) continue;
    CHECK_CUDA(cudaStreamSynchronize(stream));

    std::vector<float> times;
    times.reserve((size_t)cnt);
    for (int i = 0; i < cnt; ++i) {
      CHECK_CUDA(cudaEventRecord(start, stream));
      cublasStatus_t st = cublasLtMatmul(
          handle, op_desc, &alpha, b, a_desc, a, b_desc, &beta, c, c_desc, c, c_desc,
          &heuristics[rank].algo, workspace, workspace_size, stream);
      if (st != CUBLAS_STATUS_SUCCESS) { usable = false; break; }
      CHECK_CUDA(cudaEventRecord(stop, stream));
      CHECK_CUDA(cudaEventSynchronize(stop));
      float ms = 0.0f;
      CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
      times.push_back(ms * 1000.0f);
    }
    if (!usable || times.empty()) continue;
    std::sort(times.begin(), times.end());
    if (times.front() < best_candidate) {
      best_candidate = times.front();
      best_candidate_median = times[times.size() / 2];
      best_rank = rank;
    }
  }
  if (best_rank < 0) return 3;
  *best_us = best_candidate;
  *median_us = best_candidate_median;
  *best_algo_rank = best_rank;

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cublasLtMatmulPreferenceDestroy(pref);
  cublasLtMatrixLayoutDestroy(c_desc);
  cublasLtMatrixLayoutDestroy(b_desc);
  cublasLtMatrixLayoutDestroy(a_desc);
  cublasLtMatmulDescDestroy(op_desc);
  cudaFree(workspace);
  cudaFree(c);
  cudaFree(b);
  cudaFree(a);
  cublasLtDestroy(handle);
  cudaStreamDestroy(stream);
  return 0;
}
'''
  if not so.exists() or not src.exists() or src.read_text() != code:
    src.write_text(code)
    include_dir = CUDA12_INCLUDE if CUDA12_INCLUDE.exists() else Path("/usr/local/cuda/include")
    lib_dir = CUDA12_LIB if CUDA12_LIB.exists() else Path("/usr/local/cuda/lib64")
    subprocess.run([
      "g++", "-O2", "-std=c++17", "-fPIC", "-shared", str(src), "-o", str(so),
      f"-I{include_dir}", f"-L{lib_dir}", "-Wl,-rpath," + str(lib_dir),
      "-lcublasLt", "-lcudart",
    ], check=True)
  return so


class CublasLtBench:
  def __init__(self):
    self.lib = ctypes.CDLL(str(build_cublaslt_helper()))
    self.lib.gemma4_bench_cublaslt_bf16_f32.argtypes = [
      ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
      ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_float),
      ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int),
      ctypes.POINTER(ctypes.c_int),
    ]

  def time_bf16_f32(self, m: int, n: int, k: int, warmup: int, cnt: int) -> tuple[float, float, int, int]:
    best_us = ctypes.c_float()
    median_us = ctypes.c_float()
    best_rank = ctypes.c_int()
    returned = ctypes.c_int()
    workspace = getenv("CUBLASLT_WORKSPACE_BYTES", 64 * 1024 * 1024)
    requested_algos = getenv("CUBLASLT_HEURISTICS", 64)
    tune_algos = getenv("CUBLASLT_TUNE_ALGOS", 8)
    status = self.lib.gemma4_bench_cublaslt_bf16_f32(
      m, n, k, warmup, cnt, workspace, requested_algos, tune_algos,
      ctypes.byref(best_us), ctypes.byref(median_us), ctypes.byref(best_rank),
      ctypes.byref(returned),
    )
    if status != 0:
      raise RuntimeError(f"cuBLASLt helper failed with status {status}")
    return best_us.value / 1e6, median_us.value / 1e6, best_rank.value, returned.value


def cublas_candidates_from_env() -> list[tuple[str, int, int]]:
  compute_modes = os.environ.get("CUBLAS_COMPUTES", "32f,fast16bf").split(",")
  algos = parse_int_list(os.environ.get("CUBLAS_ALGOS", "99,100-115"))
  candidates = []
  for mode in [x.strip().lower() for x in compute_modes if x.strip()]:
    if mode in ("32f", "default"):
      compute_type = CUBLAS_COMPUTE_32F
      mode_name = "32f"
    elif mode in ("fast16bf", "fast_bf16", "fast"):
      compute_type = CUBLAS_COMPUTE_32F_FAST_16BF
      mode_name = "fast16bf"
    else:
      raise ValueError(f"unknown CUBLAS_COMPUTES entry: {mode}")
    for algo in algos:
      candidates.append((mode_name, compute_type, algo))
  return candidates


def main() -> None:
  n = getenv("N", 5376)
  k = getenv("K", 21504)
  warmup = getenv("WARMUP", 2)
  cnt = getenv("CNT", 5)
  beam = getenv("SEARCH_BEAM", 0)
  tiny_beams = parse_int_list(os.environ.get("TINYGRAD_BEAMS", str(beam)))
  compare_cublas = getenv("COMPARE_CUBLAS", 1)
  compare_cublaslt = getenv("COMPARE_CUBLASLT", 1)
  dtype_in = DTYPES_DICT[getenv("DTYPE_IN", "bfloat16")]
  dtype_acc = DTYPES_DICT[getenv("DTYPE_ACC", "float")]
  if cnt <= 0:
    raise ValueError("CNT must be positive")

  print("tinygrad late-eval matmul bench")
  print(f"tinygrad_root={TINYGRAD_ROOT}")
  print(f"N={n} K={k} dtype_in={dtype_in} dtype_acc={dtype_acc} tiny_beams={tiny_beams}")

  cublas = CudaCublas() if compare_cublas else None
  cublaslt = CublasLtBench() if compare_cublaslt else None
  cublas_candidates = cublas_candidates_from_env()
  try:
    for m in parse_m_list():
      best, median, best_beam, names = time_tinygrad_beams(
        m, n, k, dtype_in, dtype_acc, warmup, cnt, tiny_beams)
      flops = 2 * m * n * k
      line = (
        f"M={m} tiny_best_us={best * 1e6:.2f} tiny_median_us={median * 1e6:.2f} "
        f"tiny_tflops={flops * 1e-12 / best:.2f} tiny_beam={best_beam}"
      )
      if cublas:
        cublas_results = cublas.time_gemm_candidates(
          m, n, k, dtype_in, dtype_acc, warmup, cnt, cublas_candidates)
        if not cublas_results:
          raise RuntimeError(f"no working cuBLAS candidates for M={m}")
        cublas_result = cublas_results[0]
        cublas_best = cublas_result["best"]
        cublas_median = cublas_result["median"]
        speedup = cublas_best / best
        winner = "tinygrad" if speedup > 1.0 else "cublas"
        line += (
          f" cublas_best_us={cublas_best * 1e6:.2f}"
          f" cublas_median_us={cublas_median * 1e6:.2f}"
          f" cublas_tflops={flops * 1e-12 / cublas_best:.2f}"
          f" cublas_compute={cublas_result['name']} cublas_algo={cublas_result['algo']}"
          f" tiny_vs_cublas={speedup:.3f}x winner={winner}"
        )
      if cublaslt:
        if dtype_in is not dtypes.bfloat16 or dtype_acc is not dtypes.float:
          raise ValueError("cuBLASLt helper currently supports only BF16 input and FP32 output")
        cublaslt_best, cublaslt_median, lt_rank, lt_returned = cublaslt.time_bf16_f32(
          m, n, k, warmup, cnt)
        lt_speedup = cublaslt_best / best
        lt_winner = "tinygrad" if lt_speedup > 1.0 else "cublaslt"
        line += (
          f" cublaslt_best_us={cublaslt_best * 1e6:.2f}"
          f" cublaslt_median_us={cublaslt_median * 1e6:.2f}"
          f" cublaslt_tflops={flops * 1e-12 / cublaslt_best:.2f}"
          f" cublaslt_rank={lt_rank}/{lt_returned}"
          f" tiny_vs_cublaslt={lt_speedup:.3f}x lt_winner={lt_winner}"
        )
      print(f"{line} programs={names}")
  finally:
    if cublas:
      cublas.destroy()


if __name__ == "__main__":
  main()
