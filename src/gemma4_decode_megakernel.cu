#include "gemma4_decode_megakernel.cuh"

#include "gemma4_decode_megakernel_phases.cu"

#include <cooperative_groups.h>

cudaError_t gemma4_decode_megakernel_ffn_tail_flash_attention_bf16(
    const Gemma4DecodeMegakernelFfnTailArgs &args,
    void *__restrict__ scratch,
    size_t scratch_bytes,
    cudaStream_t stream);
cudaError_t gemma4_decode_megakernel_attention_ffn_flash_attention_bf16(
    const Gemma4DecodeMegakernelFfnTailArgs &args,
    void *__restrict__ scratch,
    size_t scratch_bytes,
    cudaStream_t stream);

namespace {

namespace phase = gemma4_decode_megakernel_phases;
namespace cg = cooperative_groups;

// Runs the final decode spine schedule after the current residual row is ready.
__device__ inline void run_final_decode_spine(
    cg::grid_group grid,
    __nv_bfloat16 *__restrict__ next_hidden,
    int32_t *__restrict__ next_token,
    Gemma4SampleCandidate *__restrict__ candidates,
    __nv_bfloat16 *__restrict__ normed_hidden,
    const __nv_bfloat16 *__restrict__ state,
    const __nv_bfloat16 *__restrict__ final_norm_weight,
    const __nv_bfloat16 *__restrict__ lm_head_col_major,
    int32_t active_candidate_count) {
  phase::phase_final_rmsnorm_hidden(
      normed_hidden, state, final_norm_weight);
  grid.sync();

  const Gemma4SampleCandidate candidate =
      phase::phase_final_logits_block_candidate(
          normed_hidden, lm_head_col_major);
  if (threadIdx.x == 0) {
    candidates[blockIdx.x] = candidate;
  }
  grid.sync();

  phase::phase_reduce_candidates_and_gather(
      next_hidden, next_token, lm_head_col_major, candidates,
      active_candidate_count);
}

// Runs the resident final decode spine as a cooperative phase schedule.
__global__ __launch_bounds__(phase::kMegaThreads, 1)
void decode_megakernel_spine_kernel(
    Gemma4DecodeMegakernelSpineArgs args,
    Gemma4SampleCandidate *__restrict__ candidates,
    __nv_bfloat16 *__restrict__ normed_hidden,
    int32_t active_candidate_count) {
  cg::grid_group grid = cg::this_grid();

  run_final_decode_spine(
      grid, args.next_hidden, args.next_token, candidates, normed_hidden,
      args.state, args.final_norm_weight, args.lm_head_col_major,
      active_candidate_count);
}

// Applies the checkpoint layer scalar after the CUTLASS FFN tail writes hidden.
__global__ __launch_bounds__(phase::kMegaThreads, 1)
void scale_layer_hidden_kernel(
    __nv_bfloat16 *__restrict__ hidden,
    const __nv_bfloat16 *__restrict__ layer_scalar) {
  phase::phase_scale_layer_hidden(hidden, layer_scalar);
}

// Returns resident spine CTAs for scratch sizing, falling back to the old
// max-candidate allocation because this public size API has no error channel.
int32_t spine_candidate_count_for_scratch(void) {
  int32_t active_candidate_count = 0;
  const cudaError_t status = phase::active_candidate_count_for_kernel(
      decode_megakernel_spine_kernel, &active_candidate_count);
  if (status != cudaSuccess) {
    return phase::kMegaCandidateCount;
  }
  return active_candidate_count;
}

// Launches the final decode spine using scratch already split for the FFN tail.
cudaError_t launch_final_decode_spine(
    const Gemma4DecodeMegakernelFfnTailArgs &args,
    const phase::Gemma4DecodeFfnTailScratch &scratch_parts,
    int32_t active_candidate_count,
    cudaStream_t stream) {
  Gemma4DecodeMegakernelSpineArgs spine_args = {};
  spine_args.state = args.residual_out;
  spine_args.next_hidden = args.next_hidden;
  spine_args.next_token = args.next_token;
  spine_args.final_norm_weight = args.final_norm_weight;
  spine_args.lm_head_col_major = args.lm_head_col_major;

  Gemma4SampleCandidate *candidates_arg = scratch_parts.candidates;
  __nv_bfloat16 *normed_hidden_arg = scratch_parts.normed_hidden;
  int32_t active_candidate_count_arg = active_candidate_count;
  void *kernel_args[] = {
      &spine_args,
      &candidates_arg,
      &normed_hidden_arg,
      &active_candidate_count_arg,
  };

  cudaError_t status = cudaLaunchCooperativeKernel(
      reinterpret_cast<void *>(decode_megakernel_spine_kernel),
      active_candidate_count,
      phase::kMegaThreads,
      kernel_args,
      0,
      stream);
  if (status != cudaSuccess) {
    return status;
  }
  return cudaGetLastError();
}

}  // namespace

// Returns scratch for all per-block candidates plus the normalized hidden row.
size_t gemma4_decode_megakernel_spine_scratch_bytes(void) {
  return phase::spine_scratch_bytes(spine_candidate_count_for_scratch());
}

// Returns scratch for CUTLASS FFN decode plus the existing sampling-tail scratch.
size_t gemma4_decode_megakernel_ffn_tail_scratch_bytes(void) {
  return phase::ffn_tail_scratch_bytes(spine_candidate_count_for_scratch());
}

// Launches the cooperative final decode spine.
cudaError_t gemma4_decode_megakernel_spine_bf16(
    const Gemma4DecodeMegakernelSpineArgs &args,
    void *__restrict__ scratch,
    size_t scratch_bytes,
    cudaStream_t stream) {
  return phase::launch_spine(
      args, scratch, scratch_bytes, stream, decode_megakernel_spine_kernel);
}

// Finishes the decode FFN tail with the CUTLASS-backed standalone FFN path.
cudaError_t gemma4_decode_megakernel_finish_ffn_tail_bf16(
    const Gemma4DecodeMegakernelFfnTailArgs &args,
    void *__restrict__ scratch,
    size_t scratch_bytes,
    bool run_final_spine,
    cudaStream_t stream) {
  int32_t active_candidate_count = 0;
  if (run_final_spine) {
    cudaError_t status = phase::active_candidate_count_for_kernel(
        decode_megakernel_spine_kernel, &active_candidate_count);
    if (status != cudaSuccess) {
      return status;
    }
  }

  const size_t required_scratch =
      run_final_spine ? phase::ffn_tail_scratch_bytes(active_candidate_count)
                      : sizeof(Gemma4FfnDecodeScratch);
  if (scratch_bytes < required_scratch) {
    return cudaErrorInvalidValue;
  }

  const phase::Gemma4DecodeFfnTailScratch scratch_parts =
      phase::ffn_tail_scratch_from_buffer(scratch, active_candidate_count);
  cudaError_t status = gemma4_ffn_decode_fused_bf16(
      args.residual_out, args.normed_out, args.ffn_x, args.ffn_residual,
      args.ffn_norm_weight, args.ffn_gate_up_decode, args.ffn_down_decode,
      scratch_parts.ffn, GEMMA4_RMS_NORM_EPS, stream);
  if (status != cudaSuccess) {
    return status;
  }

  const dim3 grid_dim(1);
  const dim3 block_dim(phase::kMegaThreads);
  scale_layer_hidden_kernel<<<grid_dim, block_dim, 0, stream>>>(
      args.residual_out, args.layer_scalar);
  status = cudaGetLastError();
  if (status != cudaSuccess || !run_final_spine) {
    return status;
  }
  return launch_final_decode_spine(
      args, scratch_parts, active_candidate_count, stream);
}

// Launches optional attention preparation and the CUTLASS FFN tail without final sampling.
cudaError_t gemma4_decode_megakernel_attention_ffn_bf16(
    const Gemma4DecodeMegakernelFfnTailArgs &args,
    void *__restrict__ scratch,
    size_t scratch_bytes,
    cudaStream_t stream) {
  if ((args.flags & GEMMA4_DECODE_MEGAKERNEL_FLAG_FLASH_ATTENTION) != 0) {
    return gemma4_decode_megakernel_attention_ffn_flash_attention_bf16(
        args, scratch, scratch_bytes, stream);
  }
  return gemma4_decode_megakernel_finish_ffn_tail_bf16(
      args, scratch, scratch_bytes, false, stream);
}

// Launches the CUTLASS FFN tail followed by the final decode spine.
cudaError_t gemma4_decode_megakernel_ffn_tail_bf16(
    const Gemma4DecodeMegakernelFfnTailArgs &args,
    void *__restrict__ scratch,
    size_t scratch_bytes,
    cudaStream_t stream) {
  if ((args.flags & GEMMA4_DECODE_MEGAKERNEL_FLAG_FLASH_ATTENTION) != 0) {
    return gemma4_decode_megakernel_ffn_tail_flash_attention_bf16(
        args, scratch, scratch_bytes, stream);
  }
  return gemma4_decode_megakernel_finish_ffn_tail_bf16(
      args, scratch, scratch_bytes, true, stream);
}
