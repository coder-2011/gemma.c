// BSD-3-Clause FlashAttention-derived Gemma 4 attention path.
//
// This file is a Gemma-specific inline specialization of the FlashAttention-2
// SM80 forward path. It keeps only the BF16 forward kernels used by Gemma 4
// 31B text attention: local sliding layers and full-causal global layers. The
// implementation is limited to fixed-shape model paths and omits the upstream
// generic feature matrix.
//
// The cloned upstream project carries the full BSD-3-Clause license in
// experiments/flash-attention/LICENSE.

#include <cuda_bf16.h>
#include <cuda/cmath>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>

#include <cute/algorithm/tensor_reduce.hpp>
#include <cute/tensor.hpp>
#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/layout/layout.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

#include "gemma4_flash_attention.cuh"
#include "gemma4_rope.cuh"
#include "gemma4.h"

#ifndef GEMMA4_FA_USE_SHARED_ROPE_HELPER
#define GEMMA4_FA_USE_SHARED_ROPE_HELPER 1
#endif

namespace gemma4_flash_attention {

using namespace cute;

// FlashAttention code is extremely shape-specialized. These constants and
// aliases pin this file to Gemma 4's BF16 attention path on SM80+ GPUs.
constexpr int kWarpSize = 32;

// Runtime parameters copied into constant-kernel-parameter space at launch.
// The tensor layouts are batch-major and row-major inside each head:
//   Q/O: [batch, seqlen_q, q_heads, head_dim]
//   K/V: [batch, seqlen_k, kv_heads, head_dim]
struct Gemma4FlashFwdParams {
  using index_t = int64_t;

  // ponytail: fixed Gemma path; add fields only when a real kernel reads them.
  const cutlass::bfloat16_t *__restrict__ q_ptr;
  const cutlass::bfloat16_t *__restrict__ k_ptr;
  const cutlass::bfloat16_t *__restrict__ v_ptr;
  cutlass::bfloat16_t *__restrict__ o_ptr;
  float *__restrict__ softmax_lse_ptr;

  int seqlen_q;
  int seqlen_k;

  float scale_softmax;
  float scale_softmax_log2;
  int window_size_left;
};

// Forward-kernel traits derive every CUTE layout and copy atom from the fixed
// tile geometry. This is where "block size" becomes concrete shared-memory
// swizzles, vectorized global-memory loads, and tensor-core MMA layout.
template <int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_>
struct Gemma4FlashFwdKernelTraits {
  // ponytail: RTX A6000/sm86 only; restore a trait base if pre-SM80 grows back.
  using Element = cutlass::bfloat16_t;

  using MMA_Atom_Arch = MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>;

  // LDSM copies move shared-memory tiles into tensor-core register fragments.
  // The transposed copy is used for V because the second GEMM consumes P * V.
  using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, Element>;
  using SmemCopyAtomTransposed = Copy_Atom<SM75_U16x8_LDSM_T, Element>;

  static constexpr int kNWarps = kNWarps_;
  static constexpr int kNThreads = kNWarps * kWarpSize;
  static constexpr int kBlockM = kBlockM_;
  static constexpr int kBlockN = kBlockN_;
  static constexpr int kHeadDim = kHeadDim_;
  static_assert(kHeadDim % 32 == 0, "FA head dim must be a multiple of 32");

  // Global memory is loaded as 128-bit vectors. kBlockKSmem controls the
  // shared-memory row width used by the swizzled Q/K/V tiles.
  static constexpr int kBlockKSmem = kHeadDim % 64 == 0 ? 64 : 32;
  static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
  static constexpr int kGmemThreadsPerRow = kBlockKSmem / kGmemElemsPerLoad;
  static constexpr int kSwizzle = kBlockKSmem == 32 ? 2 : 3;
  static_assert(kHeadDim % kGmemElemsPerLoad == 0,
                "FA head dim must be a multiple of vector load width");
  static_assert(kNThreads % kGmemThreadsPerRow == 0, "FA thread count must divide row load layout");

  // QK stays row-split because softmax state is row-owned. P*V gets a separate
  // MMA type but keeps the same row ownership; true N-splits need P staging.
  using TiledMmaQK = TiledMMA< // ablate layout and tile size/shape
      MMA_Atom_Arch,
      Layout<Shape<Int<kNWarps>, _1, _1>>,
      Tile<Int<16 * kNWarps>, _16, _16>>;
  using TiledMmaPV = TiledMMA<
      MMA_Atom_Arch,
      Layout<Shape<Int<kNWarps>, _1, _1>>,
      Tile<Int<16 * kNWarps>, _16, _16>>;

  // Shared-memory layouts use a CUTE swizzle to make tensor-core LDSM loads
  // bank-conflict-friendly. Q, K, and V share the same logical tile shape, but
  // V also gets a transposed view for the P*V multiply.
  using SmemLayoutAtomQ = decltype(composition(Swizzle<kSwizzle, 3, 3>{}, Layout<Shape<_8, Int<kBlockKSmem>>, Stride<Int<kBlockKSmem>, _1>>{})); // 3, 3 changes, if we adjust layout
  using SmemLayoutQ = decltype(tile_to_shape(SmemLayoutAtomQ{}, Shape<Int<kBlockM>, Int<kHeadDim>>{}));

  using SmemLayoutKV = decltype(tile_to_shape(SmemLayoutAtomQ{}, Shape<Int<kBlockN>, Int<kHeadDim>>{}));
  using SmemLayoutVtransposed = decltype(composition(SmemLayoutKV{}, make_layout(Shape<Int<kHeadDim>, Int<kBlockN>>{}, GenRowMajor{})));
  using SmemLayoutVtransposedNoSwizzle = decltype(get_nonswizzle_portion(SmemLayoutVtransposed{}));

  using SmemLayoutAtomO = decltype(composition(Swizzle<kSwizzle, 3, 3>{}, Layout<Shape<Int<8>, Int<kBlockKSmem>>, Stride<Int<kBlockKSmem>, _1>>{}));
  using SmemLayoutO = decltype(tile_to_shape(SmemLayoutAtomO{}, Shape<Int<kBlockM>, Int<kHeadDim>>{}));
  using SmemCopyAtomO = Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>;

  static constexpr int kSmemQSize = size(SmemLayoutQ{}) * sizeof(Element);
  static constexpr int kSmemKVSize = size(SmemLayoutKV{}) * 2 * sizeof(Element);
  static constexpr int kSmemSize = kSmemQSize + kSmemKVSize;

  // The global-memory copy layout assigns threads to 128-bit vector chunks.
  // Q/K/V use cp.async into shared memory; O uses normal vectorized stores.
  using GmemLayoutAtom =
      Layout<Shape<Int<kNThreads / kGmemThreadsPerRow>,
                   Int<kGmemThreadsPerRow>>,
             Stride<Int<kGmemThreadsPerRow>, _1>>;
  using GmemTiledCopyQKV = decltype(
      make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, Element>{},
                      GmemLayoutAtom{},
                      Layout<Shape<_1, _8>>{}));
  using GmemTiledCopyO = decltype(
      make_tiled_copy(
          Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>{},
          GmemLayoutAtom{},
          Layout<Shape<_1, _8>>{}));
};

using Gemma4SlidingFa2KernelTraits =
    Gemma4FlashFwdKernelTraits<GEMMA4_SLIDING_HEAD_DIM, 64, 64, 4>;
using Gemma4GlobalFa2KernelTraits =
    Gemma4FlashFwdKernelTraits<GEMMA4_GLOBAL_HEAD_DIM, 32, 32, 2>;

// Return the maximum value across the 4 lanes that jointly own one score row.
__device__ __forceinline__ float gemma4_fa_quad_reduce_max(float x) {
  x = max(x, __shfl_xor_sync(uint32_t(-1), x, 2));
  x = max(x, __shfl_xor_sync(uint32_t(-1), x, 1));
  return x;
}

// Return the sum across the 4 lanes that jointly own one score row.
__device__ __forceinline__ float gemma4_fa_quad_reduce_sum(float x) {
  x += __shfl_xor_sync(uint32_t(-1), x, 2);
  x += __shfl_xor_sync(uint32_t(-1), x, 1);
  return x;
}

// Generic tiled tensor-core GEMM helper for operands staged in shared memory.
// It overlaps the next LDSM copy into registers with the current MMA step.
template <bool AInRegs = false, bool BInRegs = false, typename Tensor0,
          typename Tensor1, typename Tensor2, typename Tensor3,
          typename Tensor4, typename TiledMma, typename TiledCopyA,
          typename TiledCopyB, typename ThrCopyA, typename ThrCopyB>
__forceinline__ __device__ void gemma4_fa_gemm(
    Tensor0 &acc,
    Tensor1 &tCrA,
    Tensor2 &tCrB,
    Tensor3 const &tCsA,
    Tensor4 const &tCsB,
    TiledMma tiled_mma,
    TiledCopyA smem_tiled_copy_A,
    TiledCopyB smem_tiled_copy_B,
    ThrCopyA smem_thr_copy_A,
    ThrCopyB smem_thr_copy_B) {
  Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);
  Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
  if constexpr (!AInRegs) {
    cute::copy(smem_tiled_copy_A, tCsA(_, _, _0{}), tCrA_copy_view(_, _, _0{}));
  }
  if constexpr (!BInRegs) {
    cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{}));
  }
#pragma unroll
  for (int i = 0; i < size<2>(tCrA); ++i) {
    if (i < size<2>(tCrA) - 1) {
      if constexpr (!AInRegs) {
        cute::copy(smem_tiled_copy_A, tCsA(_, _, i + 1), tCrA_copy_view(_, _, i + 1));
      }
      if constexpr (!BInRegs) {
        cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1));
      }
    }
    cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
  }
}

// Register/shared GEMM helper: A is already in registers and only B is streamed
// from shared memory. This is used for the probability-times-value multiply.
template <typename Tensor0, typename Tensor1, typename Tensor2,
          typename Tensor3, typename TiledMma, typename TiledCopy,
          typename ThrCopy>
__forceinline__ __device__ void gemma4_fa_gemm_rs(
    Tensor0 &acc,
    Tensor1 &tCrA,
    Tensor2 &tCrB,
    Tensor3 const &tCsB,
    TiledMma tiled_mma,
    TiledCopy smem_tiled_copy_B,
    ThrCopy smem_thr_copy_B) {
  Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
  cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{}));
#pragma unroll
  for (int i = 0; i < size<2>(tCrA); ++i) {
    if (i < size<2>(tCrA) - 1) {
      cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1));
    }
    cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
  }
}

// Convert the tensor-core accumulator fragment layout into a row/column view.
// The softmax and masks need logical score coordinates, not raw MMA fragments.
template <typename Layout>
__forceinline__ __device__ auto gemma4_fa_acc_rowcol(Layout acc_layout) {
  static_assert(decltype(size<0>(acc_layout))::value == 4);
  auto l = logical_divide(acc_layout, Shape<_2>{});
  return make_layout(make_layout(get<0, 1>(l), get<1>(l)),
                     make_layout(get<0, 0>(l), get<2>(l)));
}

// Reinterpret an accumulator fragment as the A operand layout expected by the
// second MMA. The shape adjustment depends on the MMA K dimension.
template <typename MMA_traits, typename Layout>
__forceinline__ __device__ auto gemma4_fa_acc_Aregs(Layout acc_layout) {
  using X = Underscore;
  static_assert(decltype(size<0>(acc_layout))::value == 4);
  constexpr int mma_shape_K = get<2>(typename MMA_traits::Shape_MNK{});
  static_assert(mma_shape_K == 8 || mma_shape_K == 16);
  if constexpr (mma_shape_K == 8) {
    return acc_layout;
  } else {
    auto l = logical_divide(acc_layout, Shape<X, X, _2>{});
    return make_layout(make_layout(get<0>(l), get<2, 0>(l)), get<1>(l), get<2, 1>(l));
  }
}

// Convert all elements of a register fragment at once using CUTLASS numeric
// conversion. The common case here is FP32 softmax probabilities -> BF16.
template <typename ToType, typename Engine, typename Layout>
__forceinline__ __device__ auto gemma4_fa_convert_type(
    Tensor<Engine, Layout> const &tensor) {
  using FromType = typename Engine::value_type;
  constexpr int numel = decltype(size(tensor))::value;
  cutlass::NumericArrayConverter<ToType, FromType, numel> convert_op;
  auto frag = convert_op(*reinterpret_cast<const cutlass::Array<FromType, numel> *>(tensor.data()));
  return make_tensor(make_rmem_ptr<ToType>(&frag), tensor.layout());
}

// Thin wrapper around the cp.async wait instruction. N is the number of async
// groups that may remain outstanding after the wait.
template <int N>
__device__ __forceinline__ void gemma4_fa_cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

// Copy a full tile with no predicates; used when the caller has already proven
// the whole tile is in-bounds.
template <typename TiledCopy, typename Engine0, typename Layout0,
          typename Engine1, typename Layout1>
__forceinline__ __device__ void gemma4_fa_copy(
    TiledCopy tiled_copy,
    Tensor<Engine0, Layout0> const &src,
    Tensor<Engine1, Layout1> &dst) {
  cute::copy(tiled_copy, src, dst);
}

// Copy only rows/columns that are in-bounds, leaving out-of-bounds destination
// entries untouched.
template <typename TiledCopy, typename Engine0, typename Layout0,
          typename Engine1, typename Layout1, typename Engine2,
          typename Layout2>
__forceinline__ __device__ void gemma4_fa_copy_mn_guarded(
    TiledCopy tiled_copy,
    Tensor<Engine0, Layout0> const &src,
    Tensor<Engine1, Layout1> &dst,
    Tensor<Engine2, Layout2> const &identity_mn,
    const int max_mn) {
#pragma unroll
  for (int m = 0; m < size<1>(src); ++m) {
    // identity_mn carries the logical tile coordinate for this vector chunk.
    if (get<0>(identity_mn(0, m, 0)) < max_mn) {
#pragma unroll
      for (int k = 0; k < size<2>(src); ++k) {
        cute::copy(tiled_copy, src(_, m, k), dst(_, m, k));
      }
    }
  }
}

// Copy only rows/columns that are in-bounds, and zero any out-of-bounds
// destination rows so later math never consumes stale shared memory.
template <typename TiledCopy, typename Engine0, typename Layout0,
          typename Engine1, typename Layout1, typename Engine2,
          typename Layout2>
__forceinline__ __device__ void gemma4_fa_copy_mn_guarded_clear(
    TiledCopy tiled_copy,
    Tensor<Engine0, Layout0> const &src,
    Tensor<Engine1, Layout1> &dst,
    Tensor<Engine2, Layout2> const &identity_mn,
    const int max_mn) {
#pragma unroll
  for (int m = 0; m < size<1>(src); ++m) {
    // The first mode is the row/column coordinate; K is always even here.
    if (get<0>(identity_mn(0, m, 0)) < max_mn) {
#pragma unroll
      for (int k = 0; k < size<2>(src); ++k) {
        cute::copy(tiled_copy, src(_, m, k), dst(_, m, k));
      }
    } else {
      cute::clear(dst(_, m, _));
    }
  }
}

// Reduce each row-owned scalar fragment with a 4-lane maximum.
template <typename Engine0, typename Layout0, typename Engine1,
          typename Layout1>
__device__ __forceinline__ void gemma4_fa_quad_reduce_max(
    Tensor<Engine0, Layout0> &dst,
    Tensor<Engine1, Layout1> const &src) {
#pragma unroll
  for (int i = 0; i < size(dst); i++) {
    dst(i) = gemma4_fa_quad_reduce_max(src(i));
  }
}

// Reduce each row-owned scalar fragment with a 4-lane sum.
template <typename Engine0, typename Layout0, typename Engine1,
          typename Layout1>
__device__ __forceinline__ void gemma4_fa_quad_reduce_sum(
    Tensor<Engine0, Layout0> &dst,
    Tensor<Engine1, Layout1> const &src) {
#pragma unroll
  for (int i = 0; i < size(dst); i++) {
    dst(i) = gemma4_fa_quad_reduce_sum(src(i));
  }
}

// Apply exp2(score * scale - row_max * scale). Using exp2 lets the kernel keep
// the softmax scale in log2 units, matching upstream FlashAttention.
template <bool ScaleMax>
__forceinline__ __device__ float gemma4_fa_scaled_max(
    float max_value,
    float scale) {
  if (max_value == -INFINITY) return 0.0f;
  if constexpr (ScaleMax) return max_value * scale;
  return max_value * float(M_LOG2E);
}

template <bool CheckInf>
__forceinline__ __device__ float gemma4_fa_checked_max(float max_value) {
  if constexpr (CheckInf) return max_value == -INFINITY ? 0.0f : max_value;
  return max_value;
}

template <bool ScaleMax = true, typename Engine0, typename Layout0,
          typename Engine1, typename Layout1>
__forceinline__ __device__ void gemma4_fa_scale_apply_exp2(
    Tensor<Engine0, Layout0> &tensor,
    Tensor<Engine1, Layout1> const &max_values,
    const float scale) {
#pragma unroll
  for (int mi = 0; mi < size<0>(tensor); ++mi) {
    const float max_scaled =
        gemma4_fa_scaled_max<ScaleMax>(max_values(mi), scale);
#pragma unroll
    for (int ni = 0; ni < size<1>(tensor); ++ni) {
      tensor(mi, ni) = exp2f(tensor(mi, ni) * scale - max_scaled);
    }
  }
}

// Apply exp2 and accumulate the per-thread row sums in the same pass over the
// score fragment. The cross-lane sum reduction still happens once at final
// normalization, matching FA2's online-softmax state layout.
template <bool ZeroInit, bool ScaleMax = true, typename Engine0,
          typename Layout0, typename Engine1, typename Layout1,
          typename Engine2, typename Layout2>
__forceinline__ __device__ void gemma4_fa_scale_apply_exp2_sum(
    Tensor<Engine0, Layout0> &tensor,
    Tensor<Engine1, Layout1> const &max_values,
    Tensor<Engine2, Layout2> &row_sum,
    const float scale) {
#pragma unroll
  for (int mi = 0; mi < size<0>(tensor); ++mi) {
    const float max_scaled =
        gemma4_fa_scaled_max<ScaleMax>(max_values(mi), scale);
    float sum = ZeroInit ? 0.0f : row_sum(mi);
#pragma unroll
    for (int ni = 0; ni < size<1>(tensor); ++ni) {
      const float score = exp2f(tensor(mi, ni) * scale - max_scaled);
      tensor(mi, ni) = score;
      sum += score;
    }
    row_sum(mi) = sum;
  }
}

// Online softmax state for one query tile. As K/V blocks stream right-to-left,
// this tracks row maxima and row sums and rescales the accumulated output when
// a later block changes the row maximum.
template <int Rows, bool UseFusedExpSum>
struct Gemma4FlashSoftmax {
  using TensorT = decltype(make_tensor<float>(Shape<Int<Rows>>{}));
  TensorT row_max;
  TensorT row_sum;

  // Update online-softmax state for one score tile and rescale O when the row
  // maximum changes.
  template <bool IsFirst, bool CheckInf = false, typename Tensor0, typename Tensor1>
  __forceinline__ __device__ void softmax_rescale_o(
      Tensor0 &acc_s,
      Tensor1 &acc_o,
      float softmax_scale_log2) {
    Tensor scores = make_tensor(acc_s.data(), gemma4_fa_acc_rowcol(acc_s.layout()));
    if (IsFirst) {
      // First K block: initialize max/sum and leave O unscaled because it is
      // still zero.
      cute::fill(row_max, -INFINITY);
#pragma unroll
      for (int mi = 0; mi < size<0>(scores); ++mi) {
        // Per-thread row max first; the quad reduction below joins row lanes.
        row_max(mi) = cute::reduce(scores(mi, _), row_max(mi), cute::max_fn{});
      }
      gemma4_fa_quad_reduce_max(row_max, row_max);
      if constexpr (UseFusedExpSum) {
        // Sliding attention measured best when exp2 and local sum share a pass.
        gemma4_fa_scale_apply_exp2_sum</*ZeroInit=*/true>(
            scores, row_max, row_sum, softmax_scale_log2);
      } else {
        // Global attention keeps the older two-pass path to avoid extra spills.
        gemma4_fa_scale_apply_exp2(scores, row_max, softmax_scale_log2);
        cute::clear(row_sum);
#pragma unroll
        for (int mi = 0; mi < size<0>(scores); ++mi) {
          row_sum(mi) = cute::reduce(scores(mi, _), row_sum(mi), cute::plus{});
        }
      }
    } else {
      // Later K blocks: if the max increases, previous probabilities and the
      // accumulated O must be scaled down to remain in the same softmax frame.
      Tensor scores_max_prev = make_fragment_like(row_max);
      cute::copy(row_max, scores_max_prev);
#pragma unroll
      for (int mi = 0; mi < size<0>(scores); ++mi) {
        // Seed with the previous max so the result is max(old tiles, new tile).
        row_max(mi) = cute::reduce(scores(mi, _), row_max(mi), cute::max_fn{});
      }
      gemma4_fa_quad_reduce_max(row_max, row_max);
      Tensor acc_o_rowcol = make_tensor(acc_o.data(), gemma4_fa_acc_rowcol(acc_o.layout()));
#pragma unroll
      for (int mi = 0; mi < size(row_max); ++mi) {
        float scores_max_cur = gemma4_fa_checked_max<CheckInf>(row_max(mi));
        float scores_scale = exp2f((scores_max_prev(mi) - scores_max_cur) * softmax_scale_log2);
        // Bring the old denominator and O accumulator into the new max frame.
        row_sum(mi) *= scores_scale;
#pragma unroll
        for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) {
          acc_o_rowcol(mi, ni) *= scores_scale;
        }
      }
      if constexpr (UseFusedExpSum) {
        gemma4_fa_scale_apply_exp2_sum</*ZeroInit=*/false>(
            scores, row_max, row_sum, softmax_scale_log2);
      } else {
        gemma4_fa_scale_apply_exp2(scores, row_max, softmax_scale_log2);
#pragma unroll
        for (int mi = 0; mi < size<0>(scores); ++mi) {
          row_sum(mi) = cute::reduce(scores(mi, _), row_sum(mi), cute::plus{});
        }
      }
    }
  }

  template <bool IsFirst, typename Tensor0, typename Tensor1>
  __forceinline__ __device__ void softmax_rescale_visible(
      Tensor0 &acc_s,
      Tensor1 &acc_o,
      float softmax_scale_log2,
      bool score_block_fully_visible) {
    if (score_block_fully_visible) {
      this->template softmax_rescale_o<IsFirst, /*CheckInf=*/false>(
          acc_s, acc_o, softmax_scale_log2);
    } else {
      this->template softmax_rescale_o<IsFirst, /*CheckInf=*/true>(
          acc_s, acc_o, softmax_scale_log2);
    }
  }

  // Finish the softmax denominator, normalize O, and optionally return LSE.
  template <bool ReturnLse, typename Tensor0>
  __forceinline__ __device__ TensorT normalize_softmax_lse(
      Tensor0 &acc_o,
      float softmax_scale) {
    // Complete the row-sum reduction, normalize O by the softmax denominator,
    // and optionally produce the log-sum-exp side output for diagnostics/tests.
    gemma4_fa_quad_reduce_sum(row_sum, row_sum);
    TensorT lse = make_fragment_like(row_sum);
    Tensor acc_o_rowcol = make_tensor(acc_o.data(), gemma4_fa_acc_rowcol(acc_o.layout()));
#pragma unroll
    for (int mi = 0; mi < size<0>(acc_o_rowcol); ++mi) {
      float sum = row_sum(mi);
      // Empty or fully masked rows produce a harmless scale and diagnostic LSE.
      const bool invalid_sum = sum == 0.0f || sum != sum;
      float inv_sum = invalid_sum ? 1.0f : 1.0f / sum;
      if constexpr (ReturnLse) {
        lse(mi) = invalid_sum ? INFINITY
                              : row_max(mi) * softmax_scale + __logf(sum);
      }
#pragma unroll
      for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) {
        acc_o_rowcol(mi, ni) *= inv_sum;
      }
    }
    return lse;
  }
};

// Score mask for Gemma's causal attention variants. Local layers additionally
// clamp the visible keys to the sliding window.
template <bool IsLocal, typename Engine, typename Layout>
__forceinline__ __device__ void gemma4_apply_score_mask(
    Tensor<Engine, Layout> &tensor_,
    const int col_idx_offset_,
    const int row_idx_offset,
    const int warp_row_stride,
    const int max_seqlen_k,
    const int max_seqlen_q,
    const int window_size_left) {
  Tensor tensor = make_tensor(tensor_.data(), gemma4_fa_acc_rowcol(tensor_.layout()));
  const int lane_id = threadIdx.x % kWarpSize;
  // Each lane owns two adjacent logical columns in the FA2 accumulator view.
  const int col_idx_offset = col_idx_offset_ + (lane_id % 4) * 2;
#pragma unroll
  for (int mi = 0; mi < size<0, 1>(tensor); ++mi) {
    const int row_idx_base = row_idx_offset + mi * warp_row_stride;
#pragma unroll
    for (int i = 0; i < size<0, 0>(tensor); ++i) {
      const int row_idx = row_idx_base + i * 8;
      const int key_row = row_idx + max_seqlen_k - max_seqlen_q;
      const int right = std::min(max_seqlen_k, key_row + 1);
#pragma unroll
      for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
        const int col_idx_base = col_idx_offset + nj * 8;
#pragma unroll
        for (int j = 0; j < size<1, 0>(tensor); ++j) {
          const int col_idx = col_idx_base + j;
          bool masked = col_idx >= right;
          if constexpr (IsLocal) {
            const int left = std::max(0, key_row - window_size_left);
            masked = masked || col_idx < left;
          }
          if (masked) {
            tensor(make_coord(i, mi), make_coord(j, nj)) = -INFINITY;
          }
        }
      }
    }
  }
}

// Return true when every score in a K tile is causally visible to every valid Q
// row in the current query tile.
__device__ __forceinline__ bool gemma4_causal_block_fully_visible(
    int col_idx_offset,
    int block_n,
    int row_idx_offset,
    int valid_rows,
    int seqlen_delta,
    int max_seqlen_k) {
  if (valid_rows <= 0) return false;
  // The last key column in this K tile must be no later than the earliest
  // allowed causal key for the first valid query row.
  const int block_end = std::min(max_seqlen_k, col_idx_offset + block_n) - 1;
  const int earliest_right = row_idx_offset + seqlen_delta;
  return block_end <= earliest_right;
}

// Return true when every score in a local K tile is both causal and inside the
// sliding window for every valid Q row in the current query tile.
__device__ __forceinline__ bool gemma4_local_block_fully_visible(
    int col_idx_offset,
    int block_n,
    int row_idx_offset,
    int valid_rows,
    int seqlen_delta,
    int max_seqlen_k,
    int window_size_left) {
  if (!gemma4_causal_block_fully_visible(
          col_idx_offset, block_n, row_idx_offset, valid_rows, seqlen_delta,
          max_seqlen_k)) {
    return false;
  }
  // The latest valid query row has the strictest left-window boundary.
  const int latest_key_row = row_idx_offset + valid_rows - 1 + seqlen_delta;
  const int latest_left = std::max(0, latest_key_row - window_size_left);
  return col_idx_offset >= latest_left;
}

template <bool IsLocal>
__device__ __forceinline__ bool gemma4_score_block_fully_visible(
    int col_idx_offset,
    int block_n,
    int row_idx_offset,
    int valid_rows,
    int seqlen_delta,
    int max_seqlen_k,
    int window_size_left) {
  if constexpr (IsLocal) {
    return gemma4_local_block_fully_visible(
        col_idx_offset, block_n, row_idx_offset, valid_rows, seqlen_delta,
        max_seqlen_k, window_size_left);
  } else {
    return gemma4_causal_block_fully_visible(
        col_idx_offset, block_n, row_idx_offset, valid_rows, seqlen_delta,
        max_seqlen_k);
  }
}

// Build a [BlockM] tile view into the softmax log-sum-exp output for one
// (batch, query-head, row-block).
template <int BlockM>
__forceinline__ __device__ auto gemma4_get_lse_tile(
    const Gemma4FlashFwdParams &params,
    const int bidb,
    const int bidh,
    const int m_block) {
  using index_t = Gemma4FlashFwdParams::index_t;
  const index_t offset =
      (index_t(bidb) * GEMMA4_NUM_QUERY_HEADS + bidh) * params.seqlen_q +
      index_t(m_block) * BlockM;
  return make_tensor(make_gmem_ptr(params.softmax_lse_ptr + offset), Shape<Int<BlockM>>{});
}

// Compute one block of query rows for one batch and one query head. This is the
// heart of the kernel:
//   1. stage Q once;
//   2. stream K/V blocks from right to left;
//   3. compute QK^T, mask, online softmax, and P*V;
//   4. normalize and write O plus LSE.
template <typename KernelTraits, bool IsLocal, bool ReturnLse>
inline __device__ void gemma4_compute_attn_1rowblock(
    const Gemma4FlashFwdParams &params,
    const int bidb,
    const int bidh,
    const int m_block) {
  using Element = typename KernelTraits::Element;
  using index_t = Gemma4FlashFwdParams::index_t;

  extern __shared__ char smem_[];

  const int tidx = threadIdx.x;
  constexpr int kBlockM = KernelTraits::kBlockM;
  constexpr int kBlockN = KernelTraits::kBlockN;
  constexpr int kHeadDim = KernelTraits::kHeadDim;
  constexpr int kNWarps = KernelTraits::kNWarps;
  constexpr int kKVHeads = IsLocal ? GEMMA4_SLIDING_KV_HEADS : GEMMA4_GLOBAL_KV_HEADS;
  constexpr int kHeadRatio = GEMMA4_NUM_QUERY_HEADS / kKVHeads;
  constexpr int kWarpRowStride = kNWarps * 16;
  constexpr int kQRowStride = GEMMA4_NUM_QUERY_HEADS * kHeadDim;
  constexpr int kKVRowStride = kKVHeads * kHeadDim;
  static_assert(GEMMA4_NUM_QUERY_HEADS % kKVHeads == 0,
                "Gemma GQA ratio must be integral");

  // ponytail: fixed batch-major tensors; add block-info back only for ragged batches.
  const index_t batch = index_t(bidb);
  const int q_tile_start = m_block * kBlockM;
  const int q_tile_remaining = params.seqlen_q - q_tile_start;
  if (q_tile_remaining <= 0) return;
  const int valid_q_rows = std::min(kBlockM, q_tile_remaining);

  const int seqlen_delta = params.seqlen_k - params.seqlen_q;
  const int mask_row_offset =
      q_tile_start + (tidx / kWarpSize) * 16 + (tidx % kWarpSize) / 4;
  const index_t q_batch_offset = batch * index_t(params.seqlen_q) * kQRowStride;
  const index_t kv_batch_offset = batch * index_t(params.seqlen_k) * kKVRowStride;

  // Work backward over only the visible K blocks. Local layers clamp the left
  // side to the sliding window; global layers start at block zero.
  const int n_block_min =
      IsLocal ? std::max(0, (q_tile_start + seqlen_delta - params.window_size_left) / kBlockN) : 0;
  int n_block_max = cute::ceil_div(params.seqlen_k, kBlockN);
  // The right edge is causal: queries in this tile never need blocks wholly in
  // the future, even when seqlen_q != seqlen_k during decode.
  n_block_max = std::min(n_block_max, cute::ceil_div(q_tile_start + kBlockM + seqlen_delta, kBlockN));

  // If a local tile has no visible keys, write zero output and +inf LSE. This
  // keeps edge cases defined instead of relying on stale output memory.
  if (n_block_max <= n_block_min) {
    Tensor mO = make_tensor(
        make_gmem_ptr(params.o_ptr + q_batch_offset),
        make_shape(params.seqlen_q, GEMMA4_NUM_QUERY_HEADS, kHeadDim),
        make_stride(kQRowStride, kHeadDim, _1{}));
    Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_coord(m_block, 0));
    typename KernelTraits::GmemTiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
    Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
    Tensor tOrO = make_tensor<Element>(shape(tOgO));
    clear(tOrO);
    Tensor cO = make_identity_tensor(make_shape(size<0>(gO), size<1>(gO)));
    Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
    gemma4_fa_copy_mn_guarded(
        gmem_tiled_copy_O, tOrO, tOgO, tOcO, q_tile_remaining);
    if constexpr (ReturnLse) {
      Tensor gLSE = gemma4_get_lse_tile<kBlockM>(params, bidb, bidh, m_block);
#pragma unroll
      for (int m = 0; m < size<1>(tOgO); ++m) {
        const int row = get<0>(tOcO(0, m, 0));
        // Only one vector lane owns the scalar LSE for each row.
        if (row < q_tile_remaining && get<1>(tOcO(0, m, 0)) == 0) {
          gLSE(row) = INFINITY;
        }
      }
    }
    return;
  }

  // Create global-memory tensor views, then select the tile for this query
  // block and the KV head mapped by Gemma's grouped-query attention ratio.
  Tensor mQ = make_tensor(
      make_gmem_ptr(params.q_ptr + q_batch_offset),
      make_shape(params.seqlen_q, GEMMA4_NUM_QUERY_HEADS, kHeadDim),
      make_stride(kQRowStride, kHeadDim, _1{}));
  Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_coord(m_block, 0));
  const int kv_head = bidh / kHeadRatio;
  Tensor mK = make_tensor(
      make_gmem_ptr(params.k_ptr + kv_batch_offset),
      make_shape(params.seqlen_k, kKVHeads, kHeadDim),
      make_stride(kKVRowStride, kHeadDim, _1{}));
  Tensor gK = local_tile(mK(_, kv_head, _), Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_coord(_, 0));
  Tensor mV = make_tensor(
      make_gmem_ptr(params.v_ptr + kv_batch_offset),
      make_shape(params.seqlen_k, kKVHeads, kHeadDim),
      make_stride(kKVRowStride, kHeadDim, _1{}));
  Tensor gV = local_tile(mV(_, kv_head, _), Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_coord(_, 0));

  // Lay out dynamic shared memory as Q | K | V. V also receives a transposed
  // logical view over the same bytes for the P*V tensor-core operation.
  Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)), typename KernelTraits::SmemLayoutQ{});
  Tensor sK = make_tensor(sQ.data() + size(sQ), typename KernelTraits::SmemLayoutKV{});
  Tensor sV = make_tensor(sK.data() + size(sK), typename KernelTraits::SmemLayoutKV{});
  Tensor sVt = make_tensor(sV.data(), typename KernelTraits::SmemLayoutVtransposed{});
  Tensor sVtNoSwizzle = make_tensor(sV.data().get(), typename KernelTraits::SmemLayoutVtransposedNoSwizzle{});

  // Partition global/shared copies by thread. These objects encode which
  // vector lanes each thread owns for Q, K, and V movement.
  typename KernelTraits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
  auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);
  Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
  Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
  Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);
  Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
  Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);
  Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);

  // Partition tensor-core MMA fragments. acc_o stays in FP32 registers across
  // all streamed K/V blocks.
  typename KernelTraits::TiledMmaQK tiled_mma_qk;
  typename KernelTraits::TiledMmaPV tiled_mma_pv;
  auto thr_mma_qk = tiled_mma_qk.get_thread_slice(tidx);
  auto thr_mma_pv = tiled_mma_pv.get_thread_slice(tidx);
  Tensor tSrQ = thr_mma_qk.partition_fragment_A(sQ);
  Tensor tSrK = thr_mma_qk.partition_fragment_B(sK);
  Tensor tOrVt = thr_mma_pv.partition_fragment_B(sVtNoSwizzle);
  Tensor acc_o = partition_fragment_C(tiled_mma_pv, Shape<Int<kBlockM>, Int<kHeadDim>>{});

  auto smem_tiled_copy_Q = make_tiled_copy_A(typename KernelTraits::SmemCopyAtom{}, tiled_mma_qk);
  auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
  Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);

  auto smem_tiled_copy_K = make_tiled_copy_B(typename KernelTraits::SmemCopyAtom{}, tiled_mma_qk);
  auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
  Tensor tSsK = smem_thr_copy_K.partition_S(sK);

  auto smem_tiled_copy_V =
      make_tiled_copy_B(typename KernelTraits::SmemCopyAtomTransposed{}, tiled_mma_pv);
  auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
  Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);

  // Identity tensors carry logical row/column coordinates through the same
  // partitioning as data. The copy helper uses them for edge predicates.
  Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));
  Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));
  Tensor tQcQ = gmem_thr_copy_QKV.partition_S(cQ);
  Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(cKV);

  // Q is reused for every K/V block in this row tile, so it is loaded once.
  gemma4_fa_copy_mn_guarded(
      gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, q_tile_remaining);

  // Preload the rightmost K block. The loop then visits K/V blocks in
  // descending order so the causal edge is handled before the visible prefix.
  int n_block = n_block_max - 1;
  gemma4_fa_copy_mn_guarded(
      gmem_tiled_copy_QKV, tKgK(_, _, _, n_block), tKsK, tKVcKV,
      params.seqlen_k - n_block * kBlockN);
  cute::cp_async_fence();

  clear(acc_o);
  Gemma4FlashSoftmax<2 * size<1>(acc_o), IsLocal> softmax;

  // The first few blocks may touch the causal/window boundary and need masks.
  // After that, local/global fully-visible blocks can skip causal masking.
  constexpr int kMaskingSteps = cuda::ceil_div(kBlockM, kBlockN) + 1;
#pragma unroll
  for (int masking_step = 0; masking_step < kMaskingSteps; ++masking_step, --n_block) {
    Tensor acc_s = partition_fragment_C(tiled_mma_qk, Shape<Int<kBlockM>, Int<kBlockN>>{});
    clear(acc_s);
    gemma4_fa_cp_async_wait<0>();
    __syncthreads();

    // Stage V for the current block while K for this block is already in SMEM.
    // The first V tile may be partial at seqlen_k's right edge.
    if (masking_step > 0) {
      gemma4_fa_copy(gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVsV);
    } else {
      gemma4_fa_copy_mn_guarded_clear(
          gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVsV, tKVcKV,
          params.seqlen_k - n_block * kBlockN);
    }
    cute::cp_async_fence();

    gemma4_fa_gemm</*AInRegs=*/false>(
        acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma_qk, smem_tiled_copy_Q,
        smem_tiled_copy_K, smem_thr_copy_Q, smem_thr_copy_K);

    // Boundary blocks need either sliding-window or full-causal masking before
    // their scores enter the online softmax.
    const int score_col_offset = n_block * kBlockN;
    const bool score_block_fully_visible =
        gemma4_score_block_fully_visible<IsLocal>(
            score_col_offset, kBlockN, q_tile_start, valid_q_rows,
            seqlen_delta, params.seqlen_k, params.window_size_left);
    if (!score_block_fully_visible) {
      gemma4_apply_score_mask<IsLocal>(
          acc_s, score_col_offset, mask_row_offset, kWarpRowStride,
          params.seqlen_k, params.seqlen_q, params.window_size_left);
    }

    gemma4_fa_cp_async_wait<0>();
    __syncthreads();
    // K for the next iteration is prefetched while this iteration consumes V.
    if (n_block > n_block_min) {
      gemma4_fa_copy(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block - 1), tKsK);
      cute::cp_async_fence();
    }

    if (masking_step == 0) {
      softmax.template softmax_rescale_visible</*IsFirst=*/true>(
          acc_s, acc_o, params.scale_softmax_log2, score_block_fully_visible);
    } else {
      softmax.template softmax_rescale_visible</*IsFirst=*/false>(
          acc_s, acc_o, params.scale_softmax_log2, score_block_fully_visible);
    }

    // Convert probabilities to BF16 and multiply by V, accumulating O in FP32.
    Tensor rP = gemma4_fa_convert_type<Element>(acc_s);
    Tensor tOrP = make_tensor(
        rP.data(),
        gemma4_fa_acc_Aregs<typename KernelTraits::TiledMmaPV>(rP.layout()));
    gemma4_fa_gemm_rs(
        acc_o, tOrP, tOrVt, tOsVt, tiled_mma_pv, smem_tiled_copy_V,
        smem_thr_copy_V);

    if (kMaskingSteps > 1 && n_block <= n_block_min) {
      --n_block;
      break;
    }
  }

  // Remaining blocks are fully causal for global attention. Local attention
  // still checks the left side of the sliding window.
  for (; n_block >= n_block_min; --n_block) {
    Tensor acc_s = partition_fragment_C(tiled_mma_qk, Shape<Int<kBlockM>, Int<kBlockN>>{});
    clear(acc_s);
    gemma4_fa_cp_async_wait<0>();
    __syncthreads();
    gemma4_fa_copy(gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVsV);
    cute::cp_async_fence();

    gemma4_fa_gemm</*AInRegs=*/false>(
        acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma_qk, smem_tiled_copy_Q,
        smem_tiled_copy_K, smem_thr_copy_Q, smem_thr_copy_K);

    gemma4_fa_cp_async_wait<0>();
    __syncthreads();
    if (n_block > n_block_min) {
      gemma4_fa_copy(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block - 1), tKsK);
      cute::cp_async_fence();
    }

    if constexpr (IsLocal) {
      const int score_col_offset = n_block * kBlockN;
      const bool score_block_fully_visible =
          gemma4_score_block_fully_visible<IsLocal>(
              score_col_offset, kBlockN, q_tile_start, valid_q_rows,
              seqlen_delta, params.seqlen_k, params.window_size_left);
      if (!score_block_fully_visible) {
        gemma4_apply_score_mask<IsLocal>(
            acc_s, score_col_offset, mask_row_offset, kWarpRowStride,
            params.seqlen_k, params.seqlen_q, params.window_size_left);
      }
      softmax.template softmax_rescale_visible</*IsFirst=*/false>(
          acc_s, acc_o, params.scale_softmax_log2, score_block_fully_visible);
    } else {
      softmax.template softmax_rescale_o</*IsFirst=*/false, /*CheckInf=*/false>(
          acc_s, acc_o, params.scale_softmax_log2);
    }

    // The score accumulator now holds probabilities, so it can serve as the A
    // operand for the P*V tensor-core multiply.
    Tensor rP = gemma4_fa_convert_type<Element>(acc_s);
    Tensor tOrP = make_tensor(
        rP.data(),
        gemma4_fa_acc_Aregs<typename KernelTraits::TiledMmaPV>(rP.layout()));
    gemma4_fa_gemm_rs(
        acc_o, tOrP, tOrVt, tOsVt, tiled_mma_pv, smem_tiled_copy_V,
        smem_thr_copy_V);
  }

  // Finalize O = softmax(QK^T)V and store through shared memory so the MMA
  // accumulator fragment can be rearranged into coalesced global writes.
  Tensor lse = softmax.template normalize_softmax_lse<ReturnLse>(acc_o, params.scale_softmax);
  Tensor rO = gemma4_fa_convert_type<Element>(acc_o);
  Tensor sO = make_tensor(sQ.data(), typename KernelTraits::SmemLayoutO{});
  auto smem_tiled_copy_O = make_tiled_copy_C(typename KernelTraits::SmemCopyAtomO{}, tiled_mma_pv);
  auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
  Tensor taccOrO = smem_thr_copy_O.retile_S(rO);
  Tensor taccOsO = smem_thr_copy_O.partition_D(sO);
  // Shared memory is used as a small layout-conversion buffer before global
  // stores, matching the FA2 epilogue pattern.
  cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);

  Tensor mO = make_tensor(
      make_gmem_ptr(params.o_ptr + q_batch_offset),
      make_shape(params.seqlen_q, GEMMA4_NUM_QUERY_HEADS, kHeadDim),
      make_stride(kQRowStride, kHeadDim, _1{}));
  Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_coord(m_block, 0));
  if constexpr (ReturnLse) {
    Tensor gLSE = gemma4_get_lse_tile<kBlockM>(params, bidb, bidh, m_block);

    Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});
    Tensor taccOcO = thr_mma_pv.partition_C(caccO);
    Tensor taccOcO_row = logical_divide(taccOcO, Shape<_2>{})(make_coord(0, _), _, 0);
    // A single column-owner writes each row's scalar LSE.
    if (get<1>(taccOcO_row(0)) == 0) {
#pragma unroll
      for (int mi = 0; mi < size(lse); ++mi) {
        const int row = get<0>(taccOcO_row(mi));
        if (row < q_tile_remaining) {
          gLSE(row) = lse(mi);
        }
      }
    }
  }

  typename KernelTraits::GmemTiledCopyO gmem_tiled_copy_O;
  auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
  Tensor tOsO = gmem_thr_copy_O.partition_S(sO);
  Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
  __syncthreads();

  Tensor tOrO = make_tensor<Element>(shape(tOgO));
  // Bring O back out of shared memory in the same vector layout used for the
  // final coalesced global store.
  cute::copy(gmem_tiled_copy_O, tOsO, tOrO);

  Tensor cO = make_identity_tensor(make_shape(size<0>(sO), size<1>(sO)));
  Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
  gemma4_fa_copy_mn_guarded(
      gmem_tiled_copy_O, tOrO, tOgO, tOcO, q_tile_remaining);
}

// One CTA computes one query row-block for one batch and one query head.
template <typename KernelTraits, bool IsLocal, bool ReturnLse>
__global__ void gemma4_flash_fwd_bf16_kernel(
    __grid_constant__ const Gemma4FlashFwdParams params) {
  const int m_block = blockIdx.x;
  const int bidb = blockIdx.y;
  const int bidh = blockIdx.z;
  gemma4_compute_attn_1rowblock<KernelTraits, IsLocal, ReturnLse>(
      params, bidb, bidh, m_block);
}

// Fill the compact launch-parameter struct expected by the specialized kernel.
// The public C API passes only Gemma-level dimensions; strides are derived here.
Gemma4FlashFwdParams make_params(
    __nv_bfloat16 *__restrict__ d_out,
    float *__restrict__ d_softmax_lse,
    const __nv_bfloat16 *__restrict__ d_q,
    const __nv_bfloat16 *__restrict__ d_k,
    const __nv_bfloat16 *__restrict__ d_v,
    int seqlen_q,
    int seqlen_k,
    int window_left,
    float softmax_scale) {
  Gemma4FlashFwdParams params{};
  params.q_ptr = reinterpret_cast<const cutlass::bfloat16_t *>(d_q);
  params.k_ptr = reinterpret_cast<const cutlass::bfloat16_t *>(d_k);
  params.v_ptr = reinterpret_cast<const cutlass::bfloat16_t *>(d_v);
  params.o_ptr = reinterpret_cast<cutlass::bfloat16_t *>(d_out);
  params.softmax_lse_ptr = d_softmax_lse;

  params.seqlen_q = seqlen_q;
  params.seqlen_k = seqlen_k;

  params.scale_softmax = softmax_scale;
  params.scale_softmax_log2 = softmax_scale * float(M_LOG2E);
  params.window_size_left = window_left;
  return params;
}

bool gemma4_fa_valid_qkv_args(
    const void *d_out,
    const void *d_q,
    const void *d_k,
    const void *d_v,
    int batch_size,
    int seqlen_q,
    int seqlen_k) {
  return d_out != nullptr && d_q != nullptr && d_k != nullptr &&
         d_v != nullptr && batch_size > 0 && seqlen_q > 0 && seqlen_k > 0;
}

bool gemma4_fa_valid_sliding_args(
    const void *d_out,
    const void *d_q,
    const void *d_k,
    const void *d_v,
    int batch_size,
    int seqlen_q,
    int seqlen_k,
    int window_left) {
  return gemma4_fa_valid_qkv_args(
             d_out, d_q, d_k, d_v, batch_size, seqlen_q, seqlen_k) &&
         window_left >= 0;
}

bool gemma4_fa_valid_sliding_cache_config(
    const Gemma4KvCacheConfig &config,
    int32_t cache_layer) {
  return cache_layer >= 0 && cache_layer < config.num_layers &&
         config.num_layers > 0 && config.num_pages > 0 &&
         config.page_size > 0 && config.max_pages_per_seq > 0 &&
         config.num_heads == GEMMA4_SLIDING_KV_HEADS &&
         config.head_dim == GEMMA4_SLIDING_HEAD_DIM;
}

// Dynamic shared memory exceeds the default limit for these FA tiles, so the
// selected kernel must opt in before launch or attribute introspection.
template <typename KernelTraits, bool IsLocal, bool ReturnLse>
cudaError_t set_kernel_smem() {
  static bool initialized = false;
  if (initialized) return cudaSuccess;
  auto kernel = &gemma4_flash_fwd_bf16_kernel<KernelTraits, IsLocal, ReturnLse>;
  cudaError_t status = cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
      KernelTraits::kSmemSize);
  if (status == cudaSuccess) initialized = true;
  return status;
}

// Common launcher for sliding and global variants. The grid dimensions map to:
//   x = query row block, y = batch, z = query head.
template <typename KernelTraits, bool IsLocal, bool ReturnLse>
cudaError_t launch_attention(
    Gemma4FlashFwdParams &params,
    int batch_size,
    cudaStream_t stream) {
  cudaError_t status = set_kernel_smem<KernelTraits, IsLocal, ReturnLse>();
  if (status != cudaSuccess) return status;
  const dim3 grid_dim(cuda::ceil_div(params.seqlen_q, KernelTraits::kBlockM),
                      batch_size,
                      GEMMA4_NUM_QUERY_HEADS);
  constexpr dim3 block_dim(KernelTraits::kNThreads);
  gemma4_flash_fwd_bf16_kernel<KernelTraits, IsLocal, ReturnLse>
      <<<grid_dim, block_dim, KernelTraits::kSmemSize, stream>>>(params);
  return cudaGetLastError();
}

template <typename KernelTraits, bool IsLocal>
cudaError_t launch_attention_maybe_lse(
    Gemma4FlashFwdParams &params,
    int batch_size,
    cudaStream_t stream) {
  return params.softmax_lse_ptr == nullptr
             ? launch_attention<KernelTraits, IsLocal, false>(
                   params, batch_size, stream)
             : launch_attention<KernelTraits, IsLocal, true>(
                   params, batch_size, stream);
}

// Gemma 4 sliding layers: head_dim=256, block 64x64, local causal window.
cudaError_t launch_sliding(
    Gemma4FlashFwdParams &params,
    int batch_size,
    cudaStream_t stream) {
  return launch_attention_maybe_lse<Gemma4SlidingFa2KernelTraits, true>(
      params, batch_size, stream);
}

// Gemma 4 global layers: head_dim=512, block 32x32, full causal context.
cudaError_t launch_global(
    Gemma4FlashFwdParams &params,
    int batch_size,
    cudaStream_t stream) {
  return launch_attention_maybe_lse<Gemma4GlobalFa2KernelTraits, false>(
      params, batch_size, stream);
}

constexpr int kSlidingNormRopeThreads = GEMMA4_SLIDING_HEAD_DIM;
static_assert(kSlidingNormRopeThreads % kWarpSize == 0);
static_assert(GEMMA4_SLIDING_HEAD_DIM == 256);
constexpr int kSlidingNormRopeHeadsPerBlock =
    kSlidingNormRopeThreads / kWarpSize;
constexpr int kSlidingNormRopeValuesPerLane =
    GEMMA4_SLIDING_HEAD_DIM / kWarpSize;
constexpr int kSlidingNormRopePairsPerLane =
    kSlidingNormRopeValuesPerLane / 2;
static_assert(GEMMA4_NUM_QUERY_HEADS % kSlidingNormRopeHeadsPerBlock == 0);

// Sum a per-lane value across one warp for the per-head RMSNorm denominator.
__device__ __forceinline__ float sliding_prep_warp_sum(float value) {
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    value += __shfl_xor_sync(0xffffffffu, value, offset);
  }
  return value;
}

// Load the values owned by this lane and return the RMS scale for the whole
// head. Each warp owns exactly one head.
__device__ __forceinline__ float sliding_prep_head_rms_scale(
    const __nv_bfloat16 *__restrict__ in,
    int lane,
    float (&values)[kSlidingNormRopeValuesPerLane]) {
  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < kSlidingNormRopeValuesPerLane; ++i) {
    // The lane-strided layout gives every lane 8 dimensions of a 256-wide head.
    values[i] = __bfloat162float(in[lane + i * kWarpSize]);
    sum = fmaf(values[i], values[i], sum);
  }
  sum = sliding_prep_warp_sum(sum);
  return rsqrtf(sum / float(GEMMA4_SLIDING_HEAD_DIM) + GEMMA4_RMS_NORM_EPS);
}

// Apply learned RMSNorm to one Q/K head and then rotate the normalized values
// with sliding-layer RoPE.
__device__ __forceinline__ void sliding_prep_weighted_rope_head(
    __nv_bfloat16 *__restrict__ out,
    const __nv_bfloat16 *__restrict__ in,
    const __nv_bfloat16 *__restrict__ weight,
    const float *__restrict__ cos_row,
    const float *__restrict__ sin_row,
    int lane) {
  constexpr int kRotaryHalf = GEMMA4_SLIDING_HEAD_DIM / 2;
  float values[kSlidingNormRopeValuesPerLane];
  const float scale = sliding_prep_head_rms_scale(in, lane, values);
#pragma unroll
  for (int i = 0; i < kSlidingNormRopePairsPerLane; ++i) {
    const int dim = lane + i * kWarpSize;
    // RoPE pairs the low half and high half of the head at the same lane-owned
    // offset, after applying the learned Q/K norm weight.
    const float lo = values[i] * scale * __bfloat162float(weight[dim]);
    const float hi = values[i + kSlidingNormRopePairsPerLane] * scale *
                     __bfloat162float(weight[kRotaryHalf + dim]);
#if GEMMA4_FA_USE_SHARED_ROPE_HELPER
    gemma4_rope::store_rotated_pair_bf16(out, cos_row, sin_row, kRotaryHalf,
                                         dim, lo, hi);
#else
    const float c = __ldg(cos_row + dim);
    const float s = __ldg(sin_row + dim);
    out[dim] = __float2bfloat16_rn(fmaf(-hi, s, lo * c));
    out[kRotaryHalf + dim] = __float2bfloat16_rn(fmaf(lo, s, hi * c));
#endif
  }
}

// Apply scale-free RMSNorm to one V head. Gemma 4 sliding V has no learned norm
// weight and no RoPE.
__device__ __forceinline__ void sliding_prep_scale_head(
    __nv_bfloat16 *__restrict__ out,
    const __nv_bfloat16 *__restrict__ in,
    int lane) {
  float values[kSlidingNormRopeValuesPerLane];
  const float scale = sliding_prep_head_rms_scale(in, lane, values);
#pragma unroll
  for (int i = 0; i < kSlidingNormRopeValuesPerLane; ++i) {
    out[lane + i * kWarpSize] = __float2bfloat16_rn(values[i] * scale);
  }
}

// Prepare all sliding-layer Q/K/V heads for prefill attention. Each block owns
// one sequence position, one batch, and a small group of heads.
__global__ __launch_bounds__(kSlidingNormRopeThreads)
void gemma4_sliding_qkv_norm_rope_kernel(
    __nv_bfloat16 *__restrict__ q_prepared,
    __nv_bfloat16 *__restrict__ k_prepared,
    __nv_bfloat16 *__restrict__ v_prepared,
    const __nv_bfloat16 *__restrict__ q,
    const __nv_bfloat16 *__restrict__ k,
    const __nv_bfloat16 *__restrict__ v,
    const __nv_bfloat16 *__restrict__ q_norm_weight,
    const __nv_bfloat16 *__restrict__ k_norm_weight,
    const float *__restrict__ cos,
    const float *__restrict__ sin,
    int seq_len) {
  constexpr int kQHeads = GEMMA4_NUM_QUERY_HEADS;
  constexpr int kKvHeads = GEMMA4_SLIDING_KV_HEADS;
  constexpr int kHeadDim = GEMMA4_SLIDING_HEAD_DIM;
  constexpr int kRotaryHalf = kHeadDim / 2;

  const int seq = blockIdx.x;
  const int batch = blockIdx.z;
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp = threadIdx.x / kWarpSize;
  const int head = blockIdx.y * kSlidingNormRopeHeadsPerBlock + warp;
  // Cos/sin tables are row-major by position and rotary-pair index.
  const float *cos_row = cos + int64_t(seq) * kRotaryHalf;
  const float *sin_row = sin + int64_t(seq) * kRotaryHalf;

  const int64_t row = int64_t(batch) * seq_len + seq;
  const int64_t q_offset = (int64_t(row) * kQHeads + head) * kHeadDim;
  // All 32 query heads exist; every warp in the block prepares one Q head.
  sliding_prep_weighted_rope_head(q_prepared + q_offset, q + q_offset,
                                  q_norm_weight, cos_row, sin_row,
                                  lane);

  // Only the first 16 heads are KV heads. Extra query-head warps skip K/V.
  if (head < kKvHeads) {
    const int64_t kv_offset = (int64_t(row) * kKvHeads + head) * kHeadDim;
    sliding_prep_weighted_rope_head(k_prepared + kv_offset, k + kv_offset,
                                    k_norm_weight, cos_row, sin_row,
                                    lane);
    sliding_prep_scale_head(v_prepared + kv_offset, v + kv_offset, lane);
  }
}

// Convert a decode token position into the Layout-A paged-cache row for one KV
// head. Page allocation stays on the host/runtime side; this kernel only writes
// if the page table already maps the requested position.
__device__ __forceinline__ int64_t sliding_decode_cache_head_offset(
    const Gemma4KvCacheConfig &config,
    const int32_t *__restrict__ page_table,
    int batch,
    int position,
    int cache_layer,
    int head) {
  const int slot = gemma4_kv_cache_page_slot(config, position);
  const int physical_page =
      __ldg(page_table + batch * config.max_pages_per_seq + slot);
  if (physical_page < 0 || physical_page >= config.num_pages) return -1;
  const int page_offset = gemma4_kv_cache_page_offset(config, position);
  return gemma4_kv_cache_offset(
      config, cache_layer, physical_page, page_offset, head, 0);
}

// Decode prep path: one token per batch. Q is prepared into a compact
// [batch, 32, 256] buffer for paged decode attention, while K/V are normalized,
// RoPE'd where needed, and written straight to the Layout-A paged cache.
__global__ __launch_bounds__(kSlidingNormRopeThreads)
void gemma4_sliding_decode_q_paged_kv_norm_rope_kernel(
    __nv_bfloat16 *__restrict__ q_prepared,
    __nv_bfloat16 *__restrict__ cache_k,
    __nv_bfloat16 *__restrict__ cache_v,
    Gemma4KvCacheConfig cache_config,
    const int32_t *__restrict__ page_table,
    const int32_t *__restrict__ token_position,
    int cache_layer,
    const __nv_bfloat16 *__restrict__ q,
    const __nv_bfloat16 *__restrict__ k,
    const __nv_bfloat16 *__restrict__ v,
    const __nv_bfloat16 *__restrict__ q_norm_weight,
    const __nv_bfloat16 *__restrict__ k_norm_weight,
    const float *__restrict__ cos,
    const float *__restrict__ sin) {
  constexpr int kQHeads = GEMMA4_NUM_QUERY_HEADS;
  constexpr int kKvHeads = GEMMA4_SLIDING_KV_HEADS;
  constexpr int kHeadDim = GEMMA4_SLIDING_HEAD_DIM;
  constexpr int kRotaryHalf = kHeadDim / 2;

  const int batch = blockIdx.x;
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp = threadIdx.x / kWarpSize;
  const int head = blockIdx.y * kSlidingNormRopeHeadsPerBlock + warp;
  if (head >= kQHeads) return;

  const int position = __ldg(token_position + batch);
  if (position < 0) return;
  const float *cos_row = cos + int64_t(position) * kRotaryHalf;
  const float *sin_row = sin + int64_t(position) * kRotaryHalf;

  const int64_t q_offset = (int64_t(batch) * kQHeads + head) * kHeadDim;
  sliding_prep_weighted_rope_head(q_prepared + q_offset, q + q_offset,
                                  q_norm_weight, cos_row, sin_row, lane);

  // Only KV heads have cache rows; the remaining query-head warps are done.
  if (head >= kKvHeads) return;
  const int64_t cache_offset = sliding_decode_cache_head_offset(
      cache_config, page_table, batch, position, cache_layer, head);
  if (cache_offset < 0) return;

  const int64_t kv_offset = (int64_t(batch) * kKvHeads + head) * kHeadDim;
  sliding_prep_weighted_rope_head(cache_k + cache_offset, k + kv_offset,
                                  k_norm_weight, cos_row, sin_row, lane);
  sliding_prep_scale_head(cache_v + cache_offset, v + kv_offset, lane);
}

// Launch the prefill Q/K/V preparation kernel after validating the raw and
// prepared buffers.
cudaError_t prepare_sliding_qkv_norm_rope(
    __nv_bfloat16 *__restrict__ d_q_prepared,
    __nv_bfloat16 *__restrict__ d_k_prepared,
    __nv_bfloat16 *__restrict__ d_v_prepared,
    const __nv_bfloat16 *__restrict__ d_q,
    const __nv_bfloat16 *__restrict__ d_k,
    const __nv_bfloat16 *__restrict__ d_v,
    const __nv_bfloat16 *__restrict__ d_q_norm_weight,
    const __nv_bfloat16 *__restrict__ d_k_norm_weight,
    const float *__restrict__ d_cos,
    const float *__restrict__ d_sin,
    int batch_size,
    int seq_len,
    cudaStream_t stream) {
  if (d_q_prepared == nullptr || d_k_prepared == nullptr ||
      d_v_prepared == nullptr || d_q == nullptr || d_k == nullptr ||
      d_v == nullptr || d_q_norm_weight == nullptr ||
      d_k_norm_weight == nullptr || d_cos == nullptr || d_sin == nullptr ||
      batch_size <= 0 || seq_len <= 0) {
    return cudaErrorInvalidValue;
  }

  constexpr int kHeadGroups =
      (GEMMA4_NUM_QUERY_HEADS + kSlidingNormRopeHeadsPerBlock - 1) /
      kSlidingNormRopeHeadsPerBlock;
  // grid.x = sequence row, grid.y = group of 8 heads, grid.z = batch.
  const dim3 grid_dim(seq_len, kHeadGroups, batch_size);
  constexpr dim3 block_dim(kSlidingNormRopeThreads);
  gemma4_sliding_qkv_norm_rope_kernel<<<grid_dim, block_dim, 0, stream>>>(
      d_q_prepared, d_k_prepared, d_v_prepared, d_q, d_k, d_v,
      d_q_norm_weight, d_k_norm_weight, d_cos, d_sin, seq_len);
  return cudaGetLastError();
}

// Launch the decode prep-cache kernel. This is intentionally only the current
// token path: prefill still keeps contiguous K/V for FA, while decode consumes
// the paged cache through split-KV attention.
cudaError_t prepare_sliding_decode_q_paged_kv_norm_rope(
    __nv_bfloat16 *__restrict__ d_q_prepared,
    __nv_bfloat16 *__restrict__ d_cache_k,
    __nv_bfloat16 *__restrict__ d_cache_v,
    Gemma4KvCacheConfig cache_config,
    const int32_t *__restrict__ d_page_table,
    const int32_t *__restrict__ d_token_position,
    int32_t batch_size,
    int32_t cache_layer,
    const __nv_bfloat16 *__restrict__ d_q,
    const __nv_bfloat16 *__restrict__ d_k,
    const __nv_bfloat16 *__restrict__ d_v,
    const __nv_bfloat16 *__restrict__ d_q_norm_weight,
    const __nv_bfloat16 *__restrict__ d_k_norm_weight,
    const float *__restrict__ d_cos,
    const float *__restrict__ d_sin,
    cudaStream_t stream) {
  if (d_q_prepared == nullptr || d_cache_k == nullptr ||
      d_cache_v == nullptr || d_page_table == nullptr ||
      d_token_position == nullptr || d_q == nullptr || d_k == nullptr ||
      d_v == nullptr || d_q_norm_weight == nullptr ||
      d_k_norm_weight == nullptr || d_cos == nullptr || d_sin == nullptr ||
      batch_size <= 0 ||
      !gemma4_fa_valid_sliding_cache_config(cache_config, cache_layer)) {
    return cudaErrorInvalidValue;
  }

  constexpr int kHeadGroups =
      (GEMMA4_NUM_QUERY_HEADS + kSlidingNormRopeHeadsPerBlock - 1) /
      kSlidingNormRopeHeadsPerBlock;
  // grid.x = batch, grid.y = group of 8 heads. Each warp owns one head.
  const dim3 grid_dim(batch_size, kHeadGroups);
  constexpr dim3 block_dim(kSlidingNormRopeThreads);
  gemma4_sliding_decode_q_paged_kv_norm_rope_kernel<<<grid_dim, block_dim, 0, stream>>>(
      d_q_prepared, d_cache_k, d_cache_v, cache_config, d_page_table,
      d_token_position, cache_layer, d_q, d_k, d_v, d_q_norm_weight,
      d_k_norm_weight, d_cos, d_sin);
  return cudaGetLastError();
}

// Small diagnostic helper for tests/benchmarks that want register count,
// dynamic shared memory, binary target, and related CUDA function attributes.
template <typename KernelTraits, bool IsLocal>
cudaError_t selected_kernel_attributes(long long *out, int len) {
  if (out == nullptr || len < 10) return cudaErrorInvalidValue;
  cudaError_t status = set_kernel_smem<KernelTraits, IsLocal, true>();
  if (status != cudaSuccess) return status;
  auto kernel = &gemma4_flash_fwd_bf16_kernel<KernelTraits, IsLocal, true>;
  cudaFuncAttributes attr{};
  status = cudaFuncGetAttributes(&attr, kernel);
  if (status != cudaSuccess) return status;
  out[0] = static_cast<long long>(attr.sharedSizeBytes);
  out[1] = static_cast<long long>(attr.constSizeBytes);
  out[2] = static_cast<long long>(attr.localSizeBytes);
  out[3] = attr.maxThreadsPerBlock;
  out[4] = attr.numRegs;
  out[5] = attr.ptxVersion;
  out[6] = attr.binaryVersion;
  out[7] = attr.cacheModeCA;
  out[8] = attr.maxDynamicSharedSizeBytes;
  out[9] = attr.preferredShmemCarveout;
  return cudaSuccess;
}

}  // namespace gemma4_flash_attention

// C ABI wrapper for Gemma 4 sliding-window attention. Inputs and output are BF16
// contiguous batch-major tensors in the layout documented by make_params().
extern "C" cudaError_t gemma4_flash_attention_sliding_fwd_bf16(
    __nv_bfloat16 *__restrict__ d_out,
    float *__restrict__ d_softmax_lse,
    const __nv_bfloat16 *__restrict__ d_q,
    const __nv_bfloat16 *__restrict__ d_k,
    const __nv_bfloat16 *__restrict__ d_v,
    int batch_size,
    int seqlen_q,
    int seqlen_k,
    int window_left,
    float softmax_scale,
    cudaStream_t stream) {
  // Public C ABI boundary: reject invalid pointers and dimensions before the
  // kernel can touch device memory.
  if (!gemma4_flash_attention::gemma4_fa_valid_sliding_args(
          d_out, d_q, d_k, d_v, batch_size, seqlen_q, seqlen_k,
          window_left)) {
    return cudaErrorInvalidValue;
  }

  gemma4_flash_attention::Gemma4FlashFwdParams params =
      gemma4_flash_attention::make_params(d_out, d_softmax_lse, d_q, d_k, d_v,
                                          seqlen_q, seqlen_k, window_left,
                                          softmax_scale);
  return gemma4_flash_attention::launch_sliding(params, batch_size, stream);
}

// Gemma sliding prefill helper: Q/K get learned RMSNorm then RoPE; V gets
// scale-free RMSNorm. The prepared tensors keep the normal FA layout:
//   Q: [batch, seq, 32, 256], K/V: [batch, seq, 16, 256].
extern "C" cudaError_t gemma4_flash_attention_sliding_fwd_bf16_norm_rope(
    __nv_bfloat16 *__restrict__ d_out,
    float *__restrict__ d_softmax_lse,
    __nv_bfloat16 *__restrict__ d_q_prepared,
    __nv_bfloat16 *__restrict__ d_k_prepared,
    __nv_bfloat16 *__restrict__ d_v_prepared,
    const __nv_bfloat16 *__restrict__ d_q,
    const __nv_bfloat16 *__restrict__ d_k,
    const __nv_bfloat16 *__restrict__ d_v,
    const __nv_bfloat16 *__restrict__ d_q_norm_weight,
    const __nv_bfloat16 *__restrict__ d_k_norm_weight,
    const float *__restrict__ d_cos,
    const float *__restrict__ d_sin,
    int batch_size,
    int seqlen_q,
    int seqlen_k,
    int window_left,
    float softmax_scale,
    cudaStream_t stream) {
  // Public C ABI boundary: all raw inputs, scratch outputs, norm weights, and
  // RoPE tables are required for the fused prefill helper.
  if (!gemma4_flash_attention::gemma4_fa_valid_sliding_args(
          d_out, d_q, d_k, d_v, batch_size, seqlen_q, seqlen_k,
          window_left) ||
      d_q_prepared == nullptr || d_k_prepared == nullptr ||
      d_v_prepared == nullptr || d_q_norm_weight == nullptr ||
      d_k_norm_weight == nullptr || d_cos == nullptr || d_sin == nullptr) {
    return cudaErrorInvalidValue;
  }
  // ponytail: prefill-only until the paged decode cache lands; decode should
  // prep just the appended token and write Layout-A cache pages once.
  if (seqlen_q != seqlen_k) {
    return cudaErrorInvalidValue;
  }

  cudaError_t status = gemma4_flash_attention::prepare_sliding_qkv_norm_rope(
      d_q_prepared, d_k_prepared, d_v_prepared, d_q, d_k, d_v,
      d_q_norm_weight, d_k_norm_weight, d_cos, d_sin, batch_size, seqlen_q,
      stream);
  if (status != cudaSuccess) return status;

  gemma4_flash_attention::Gemma4FlashFwdParams params =
      gemma4_flash_attention::make_params(
          d_out, d_softmax_lse, d_q_prepared, d_k_prepared, d_v_prepared,
          seqlen_q, seqlen_k, window_left, softmax_scale);
  return gemma4_flash_attention::launch_sliding(params, batch_size, stream);
}

// Gemma sliding decode prep-cache helper: Q gets learned RMSNorm+RoPE into the
// current-token attention buffer; K gets learned RMSNorm+RoPE into the paged
// cache; V gets scale-free RMSNorm into the same paged cache page.
extern "C" cudaError_t gemma4_flash_attention_sliding_decode_prepare_q_paged_kv_bf16(
    __nv_bfloat16 *__restrict__ d_q_prepared,
    __nv_bfloat16 *__restrict__ d_cache_k,
    __nv_bfloat16 *__restrict__ d_cache_v,
    Gemma4KvCacheConfig cache_config,
    const int32_t *__restrict__ d_page_table,
    const int32_t *__restrict__ d_token_position,
    int32_t batch_size,
    int32_t cache_layer,
    const __nv_bfloat16 *__restrict__ d_q,
    const __nv_bfloat16 *__restrict__ d_k,
    const __nv_bfloat16 *__restrict__ d_v,
    const __nv_bfloat16 *__restrict__ d_q_norm_weight,
    const __nv_bfloat16 *__restrict__ d_k_norm_weight,
    const float *__restrict__ d_cos,
    const float *__restrict__ d_sin,
    cudaStream_t stream) {
  return gemma4_flash_attention::prepare_sliding_decode_q_paged_kv_norm_rope(
      d_q_prepared, d_cache_k, d_cache_v, cache_config, d_page_table,
      d_token_position, batch_size, cache_layer, d_q, d_k, d_v,
      d_q_norm_weight, d_k_norm_weight, d_cos, d_sin, stream);
}

// Expose launch metadata so tests and benchmarks can sanity-check occupancy and
// shared-memory requirements without duplicating template internals.
// Return the dynamic shared-memory bytes required by sliding attention.
extern "C" size_t gemma4_flash_attention_sliding_smem_bytes() {
  return gemma4_flash_attention::Gemma4SlidingFa2KernelTraits::kSmemSize;
}

// Return the CUDA threadblock size used by sliding attention.
extern "C" int gemma4_flash_attention_sliding_threads_per_block() {
  return gemma4_flash_attention::Gemma4SlidingFa2KernelTraits::kNThreads;
}

// Fill a small attribute array for the compiled sliding attention kernel.
extern "C" cudaError_t gemma4_flash_attention_sliding_kernel_attributes(
    long long *out,
    int len) {
  return gemma4_flash_attention::selected_kernel_attributes<
      gemma4_flash_attention::Gemma4SlidingFa2KernelTraits, true>(out, len);
}

// C ABI wrapper for Gemma 4 global attention. It shares the same implementation
// but selects the global compile-time traits and disables the local window mask.
extern "C" cudaError_t gemma4_flash_attention_global_fwd_bf16(
    __nv_bfloat16 *__restrict__ d_out,
    float *__restrict__ d_softmax_lse,
    const __nv_bfloat16 *__restrict__ d_q,
    const __nv_bfloat16 *__restrict__ d_k,
    const __nv_bfloat16 *__restrict__ d_v,
    int batch_size,
    int seqlen_q,
    int seqlen_k,
    float softmax_scale,
    cudaStream_t stream) {
  // Public C ABI boundary: global attention expects already-prepared Q/K/V
  // tensors in the fixed Gemma batch-major layout.
  if (!gemma4_flash_attention::gemma4_fa_valid_qkv_args(
          d_out, d_q, d_k, d_v, batch_size, seqlen_q, seqlen_k)) {
    return cudaErrorInvalidValue;
  }

  gemma4_flash_attention::Gemma4FlashFwdParams params =
      gemma4_flash_attention::make_params(d_out, d_softmax_lse, d_q, d_k, d_v,
                                          seqlen_q, seqlen_k, seqlen_k,
                                          softmax_scale);
  return gemma4_flash_attention::launch_global(params, batch_size, stream);
}

// Return the dynamic shared-memory bytes required by global attention.
extern "C" size_t gemma4_flash_attention_global_smem_bytes() {
  return gemma4_flash_attention::Gemma4GlobalFa2KernelTraits::kSmemSize;
}

// Return the CUDA threadblock size used by global attention.
extern "C" int gemma4_flash_attention_global_threads_per_block() {
  return gemma4_flash_attention::Gemma4GlobalFa2KernelTraits::kNThreads;
}

// Fill a small attribute array for the compiled global attention kernel.
extern "C" cudaError_t gemma4_flash_attention_global_kernel_attributes(
    long long *out,
    int len) {
  return gemma4_flash_attention::selected_kernel_attributes<
      gemma4_flash_attention::Gemma4GlobalFa2KernelTraits, false>(out, len);
}
