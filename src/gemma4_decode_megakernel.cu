#include "gemma4_decode_megakernel.cuh"

#include "gemma4_decode_megakernel_phases.cuh"

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

// Runs the resident FFN phase, optionally followed by the final decode spine.
template <bool RunFinalSpine>
__global__ __launch_bounds__(phase::kMegaThreads, 2)
void decode_megakernel_ffn_kernel(
    Gemma4DecodeMegakernelFfnTailArgs args,
    Gemma4FfnDecodeScratch *__restrict__ ffn_scratch,
    Gemma4SampleCandidate *__restrict__ candidates,
    __nv_bfloat16 *__restrict__ normed_hidden,
    int32_t active_candidate_count) {
  cg::grid_group grid = cg::this_grid();

  phase::phase_ffn_zero_accum(ffn_scratch);
  grid.sync();

  phase::phase_ffn_accumulate(args, ffn_scratch);
  grid.sync();

  phase::phase_ffn_finalize_rmsnorm_residual(args, ffn_scratch);
  grid.sync();

  phase::phase_scale_layer_hidden(args.residual_out, args.layer_scalar);
  if constexpr (RunFinalSpine) {
    grid.sync();
    run_final_decode_spine(
        grid, args.next_hidden, args.next_token, candidates, normed_hidden,
        args.residual_out, args.final_norm_weight, args.lm_head_col_major,
        active_candidate_count);
  } else {
    (void)candidates;
    (void)normed_hidden;
    (void)active_candidate_count;
  }
}

}  // namespace

// Returns scratch for all per-block candidates plus the normalized hidden row.
size_t gemma4_decode_megakernel_spine_scratch_bytes(void) {
  return phase::spine_scratch_bytes();
}

// Returns scratch for FFN accumulation plus the existing sampling-tail scratch.
size_t gemma4_decode_megakernel_ffn_tail_scratch_bytes(void) {
  return phase::ffn_tail_scratch_bytes();
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

// Launches the cooperative attention and FFN phase without final sampling.
cudaError_t gemma4_decode_megakernel_attention_ffn_bf16(
    const Gemma4DecodeMegakernelFfnTailArgs &args,
    void *__restrict__ scratch,
    size_t scratch_bytes,
    cudaStream_t stream) {
  if (phase::ffn_tail_uses_flash_attention(args)) {
    return gemma4_decode_megakernel_attention_ffn_flash_attention_bf16(
        args, scratch, scratch_bytes, stream);
  }
  return phase::launch_ffn_tail(
      args, scratch, scratch_bytes, stream,
      decode_megakernel_ffn_kernel<false>);
}

// Launches the cooperative FFN-tail phase followed by the final decode spine.
cudaError_t gemma4_decode_megakernel_ffn_tail_bf16(
    const Gemma4DecodeMegakernelFfnTailArgs &args,
    void *__restrict__ scratch,
    size_t scratch_bytes,
    cudaStream_t stream) {
  if (phase::ffn_tail_uses_flash_attention(args)) {
    return gemma4_decode_megakernel_ffn_tail_flash_attention_bf16(
        args, scratch, scratch_bytes, stream);
  }
  return phase::launch_ffn_tail(
      args, scratch, scratch_bytes, stream,
      decode_megakernel_ffn_kernel<true>);
}
