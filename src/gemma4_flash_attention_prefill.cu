#include <cuda_bf16.h>
#include <cuda/cmath>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <type_traits>

#include <cute/algorithm/gemm.hpp>
#include <cute/algorithm/tensor_reduce.hpp>
#include <cute/tensor.hpp>
#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/layout/layout.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

#include "gemma4_flash_attention.cuh"
#include "gemma4_cuda_utils.cuh"
#include "gemma4_rmsnorm.cuh"
#include "gemma4_rope.cuh"
#include "gemma4.h"

namespace gemma4_flash_attention {

using namespace cute;

using Element = cutlass::bfloat16_t;

constexpr int kWarpSize = 32;

struct Gemma4FlashFwdParams {
  using index_t = int64_t;

  const Element *__restrict__ q_ptr;
  const Element *__restrict__ k_ptr;
  const Element *__restrict__ v_ptr;
  Element *__restrict__ o_ptr;

  int seqlen_q;
  int seqlen_k;

  float scale_softmax_log2;
  int window_size;
};

template <bool kIsLocal_, int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_>
struct Gemma4FlashFwdKernelTraits {
  // This repo targets RTX A6000/sm86; restore a trait base if pre-SM80 grows back.
  using Element = ::gemma4_flash_attention::Element;

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
  static constexpr bool kIsLocal = kIsLocal_;
  static constexpr int kKvHeads = kIsLocal ? GEMMA4_SLIDING_KV_HEADS : GEMMA4_GLOBAL_KV_HEADS;

  // Global memory is loaded as 128-bit vectors. kBlockKSmem controls the K tile width.
  using BlockKSmemInt = std::conditional_t<(kHeadDim % 64 == 0), Int<64>, Int<32>>;
  static constexpr int kBlockKSmem = BlockKSmemInt::value;
  static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
  static constexpr int kGmemThreadsPerRow = kBlockKSmem / kGmemElemsPerLoad;
  using SmemSwizzle = std::conditional_t<(kBlockKSmem == 32), Swizzle<2, 3, 3>, Swizzle<3, 3, 3>>;

  using TiledMmaQK = TiledMMA<
      MMA_Atom_Arch,
      Layout<Shape<Int<kNWarps>, _1, _1>>,
      Tile<Int<16 * kNWarps>, _16, _16>>;
  using TiledMmaPV = TiledMMA<
      MMA_Atom_Arch,
      Layout<Shape<Int<kNWarps>, _1, _1>>,
      Tile<Int<16 * kNWarps>, _16, _16>>;

  using SmemLayoutAtomQ =
      decltype(composition(
          SmemSwizzle{},
          Layout<Shape<_8, BlockKSmemInt>, Stride<BlockKSmemInt, _1>>{}));
  using SmemLayoutQ = decltype(tile_to_shape(SmemLayoutAtomQ{}, Shape<Int<kBlockM>, Int<kHeadDim>>{}));

  using SmemLayoutKV = decltype(tile_to_shape(SmemLayoutAtomQ{}, Shape<Int<kBlockN>, Int<kHeadDim>>{}));
  using SmemLayoutVtransposed = decltype(composition(SmemLayoutKV{}, make_layout(Shape<Int<kHeadDim>, Int<kBlockN>>{}, GenRowMajor{})));
  using SmemLayoutVtransposedNoSwizzle = decltype(get_nonswizzle_portion(SmemLayoutVtransposed{}));

  using SmemLayoutAtomO = decltype(composition(SmemSwizzle{}, Layout<Shape<_8, BlockKSmemInt>, Stride<BlockKSmemInt, _1>>{}));
  using SmemLayoutO = decltype(tile_to_shape(SmemLayoutAtomO{}, Shape<Int<kBlockM>, Int<kHeadDim>>{}));
  using SmemCopyAtomO = Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>;

  static constexpr int kSmemQSize = size(SmemLayoutQ{}) * sizeof(Element);
  static constexpr int kSmemKVSize = size(SmemLayoutKV{}) * 2 * sizeof(Element);
  static constexpr int kSmemSize = kSmemQSize + kSmemKVSize;

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
    Gemma4FlashFwdKernelTraits<true, GEMMA4_SLIDING_HEAD_DIM, 64, 64, 4>;
using Gemma4GlobalFa2KernelTraits =
    Gemma4FlashFwdKernelTraits<false, GEMMA4_GLOBAL_HEAD_DIM, 32, 32, 2>;

template <bool IsGlobal>
struct Gemma4AttentionTraits;

template <>
struct Gemma4AttentionTraits<false> {
  static constexpr bool kIsGlobal = false;
  static constexpr bool kHasVProjection = true;
  static constexpr int kKvHeads = GEMMA4_SLIDING_KV_HEADS;
  static constexpr int kHeadDim = GEMMA4_SLIDING_HEAD_DIM;
  static constexpr int kRotaryDim = GEMMA4_SLIDING_HEAD_DIM;
};

template <>
struct Gemma4AttentionTraits<true> {
  static constexpr bool kIsGlobal = true;
  static constexpr bool kHasVProjection = false;
  static constexpr int kKvHeads = GEMMA4_GLOBAL_KV_HEADS;
  static constexpr int kHeadDim = GEMMA4_GLOBAL_HEAD_DIM;
  static constexpr int kRotaryDim = GEMMA4_GLOBAL_HEAD_DIM / 4;
};

template <typename Traits>
struct Gemma4AttentionDerived {
  static constexpr int kPrepThreads = Traits::kHeadDim;
  static constexpr int kHeadsPerBlock = kPrepThreads / kWarpSize;
  static constexpr int kValuesPerLane = Traits::kHeadDim / kWarpSize;
  static constexpr int kRotaryHalf = Traits::kRotaryDim / 2;
  static constexpr int kRotaryPairsPerLane = kRotaryHalf / kWarpSize;
  static constexpr int kGqaRatio = GEMMA4_NUM_QUERY_HEADS / Traits::kKvHeads;

  static_assert(kHeadsPerBlock >= kGqaRatio);
};

// Return the maximum value across the 4 lanes that jointly own one score row.
__device__ __forceinline__ float gemma4_fa_quad_reduce_max(float x) {
  x = std::max(x, __shfl_xor_sync(uint32_t(-1), x, 2));
  x = std::max(x, __shfl_xor_sync(uint32_t(-1), x, 1));
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
template <typename Tensor0, typename Tensor1, typename Tensor2,
          typename Tensor3, typename Tensor4, typename TiledMma,
          typename TiledCopyA, typename TiledCopyB, typename ThrCopyA,
          typename ThrCopyB>
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
  cute::copy(smem_tiled_copy_A, tCsA(_, _, _0{}), tCrA_copy_view(_, _, _0{}));
  cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{}));
#pragma unroll
  for (int i = 0; i < size<2>(tCrA); ++i) {
    if (i < size<2>(tCrA) - 1) {
      cute::copy(smem_tiled_copy_A, tCsA(_, _, i + 1), tCrA_copy_view(_, _, i + 1));
      cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1));
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

__device__ __forceinline__ int32_t gemma4_warp_uniform_ldg_i32(
    const int32_t *__restrict__ ptr,
    int lane) {
  int32_t value = 0;
  if (lane == 0) value = __ldg(ptr);
  return __shfl_sync(0xffffffffu, value, 0);
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
__device__ __forceinline__ void gemma4_fa_reduce_rows_max(
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
__device__ __forceinline__ void gemma4_fa_reduce_rows_sum(
    Tensor<Engine0, Layout0> &dst,
    Tensor<Engine1, Layout1> const &src) {
#pragma unroll
  for (int i = 0; i < size(dst); i++) {
    dst(i) = gemma4_fa_quad_reduce_sum(src(i));
  }
}

// Apply exp2(score * scale - row_max * scale). Using exp2 lets the kernel keep
// the softmax scale in log2 units, matching upstream FlashAttention.
template <typename Engine0, typename Layout0, typename Engine1,
          typename Layout1>
__forceinline__ __device__ void gemma4_fa_scale_apply_exp2(
    Tensor<Engine0, Layout0> &tensor,
    Tensor<Engine1, Layout1> const &max_values,
    const float scale) {
#pragma unroll
  for (int mi = 0; mi < size<0>(tensor); ++mi) {
    const float row_max = max_values(mi);
    const float max_scaled = row_max == -INFINITY ? 0.0f : row_max * scale;
#pragma unroll
    for (int ni = 0; ni < size<1>(tensor); ++ni) {
      tensor(mi, ni) = exp2f(tensor(mi, ni) * scale - max_scaled);
    }
  }
}

// Initializes online-softmax row sums while applying exp2 in one pass.
template <typename Engine0, typename Layout0, typename Engine1,
          typename Layout1, typename Engine2, typename Layout2>
__forceinline__ __device__ void gemma4_fa_scale_apply_exp2_sum_first(
    Tensor<Engine0, Layout0> &tensor,
    Tensor<Engine1, Layout1> const &max_values,
    Tensor<Engine2, Layout2> &row_sum,
    const float scale) {
#pragma unroll
  for (int mi = 0; mi < size<0>(tensor); ++mi) {
    const float row_max = max_values(mi);
    const float max_scaled = row_max == -INFINITY ? 0.0f : row_max * scale;
    float sum = 0.0f;
#pragma unroll
    for (int ni = 0; ni < size<1>(tensor); ++ni) {
      const float score = exp2f(tensor(mi, ni) * scale - max_scaled);
      tensor(mi, ni) = score;
      sum += score;
    }
    row_sum(mi) = sum;
  }
}

// Updates existing online-softmax row sums while applying exp2 in one pass.
template <typename Engine0, typename Layout0, typename Engine1,
          typename Layout1, typename Engine2, typename Layout2>
__forceinline__ __device__ void gemma4_fa_scale_apply_exp2_sum_next(
    Tensor<Engine0, Layout0> &tensor,
    Tensor<Engine1, Layout1> const &max_values,
    Tensor<Engine2, Layout2> &row_sum,
    const float scale) {
#pragma unroll
  for (int mi = 0; mi < size<0>(tensor); ++mi) {
    const float row_max = max_values(mi);
    const float max_scaled = row_max == -INFINITY ? 0.0f : row_max * scale;
    float sum = row_sum(mi);
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
template <int Rows, typename KernelTraits>
struct Gemma4FlashSoftmax {
  using TensorT = decltype(make_tensor<float>(Shape<Int<Rows>>{}));
  TensorT row_max;
  TensorT row_sum;

  // Initializes online-softmax state for the first visible K/V block.
  template <typename Tensor0>
  __forceinline__ __device__ void softmax_rescale_first_block(
      Tensor0 &acc_s,
      float softmax_scale_log2) {
    Tensor scores = make_tensor(acc_s.data(), gemma4_fa_acc_rowcol(acc_s.layout()));
    cute::fill(row_max, -INFINITY);
#pragma unroll
    for (int mi = 0; mi < size<0>(scores); ++mi) {
      row_max(mi) = cute::reduce(scores(mi, _), row_max(mi), cute::max_fn{});
    }
    gemma4_fa_reduce_rows_max(row_max, row_max);

    if constexpr (KernelTraits::kIsLocal) {
      // Sliding attention measured best when exp2 and local sum share a pass.
      gemma4_fa_scale_apply_exp2_sum_first(
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
  }

  // Updates online-softmax state after the first block and rescales O into the
  // new max frame when the current score tile changes the running maximum.
  template <bool MayBeMasked, typename Tensor0, typename Tensor1>
  __forceinline__ __device__ void softmax_rescale_next_block(
      Tensor0 &acc_s,
      Tensor1 &acc_o,
      float softmax_scale_log2,
      bool score_block_fully_visible) {
    Tensor scores = make_tensor(acc_s.data(), gemma4_fa_acc_rowcol(acc_s.layout()));
    Tensor scores_max_prev = make_fragment_like(row_max);
    cute::copy(row_max, scores_max_prev);
#pragma unroll
    for (int mi = 0; mi < size<0>(scores); ++mi) {
      row_max(mi) = cute::reduce(scores(mi, _), row_max(mi), cute::max_fn{});
    }
    gemma4_fa_reduce_rows_max(row_max, row_max);

    Tensor acc_o_rowcol = make_tensor(acc_o.data(), gemma4_fa_acc_rowcol(acc_o.layout()));
#pragma unroll
    for (int mi = 0; mi < size(row_max); ++mi) {
      float scores_max_cur = row_max(mi);
      if constexpr (MayBeMasked) {
        if (!score_block_fully_visible && scores_max_cur == -INFINITY) {
          scores_max_cur = 0.0f;
        }
      }
      const float scores_scale =
          exp2f((scores_max_prev(mi) - scores_max_cur) * softmax_scale_log2);
      // Bring the old denominator and O accumulator into the new max frame.
      row_sum(mi) *= scores_scale;
#pragma unroll
      for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) {
        acc_o_rowcol(mi, ni) *= scores_scale;
      }
    }

    if constexpr (KernelTraits::kIsLocal) {
      // Sliding attention measured best when exp2 and local sum share a pass.
      gemma4_fa_scale_apply_exp2_sum_next(
          scores, row_max, row_sum, softmax_scale_log2);
    } else {
      // Global attention keeps the older two-pass path to avoid extra spills.
      gemma4_fa_scale_apply_exp2(scores, row_max, softmax_scale_log2);
#pragma unroll
      for (int mi = 0; mi < size<0>(scores); ++mi) {
        row_sum(mi) = cute::reduce(scores(mi, _), row_sum(mi), cute::plus{});
      }
    }
  }

  // Finish the softmax denominator and normalize O.
  template <typename Tensor0>
  __forceinline__ __device__ void normalize_softmax(Tensor0 &acc_o) {
    // Complete the row-sum reduction before dividing the accumulated output.
    gemma4_fa_reduce_rows_sum(row_sum, row_sum);
    Tensor acc_o_rowcol = make_tensor(acc_o.data(), gemma4_fa_acc_rowcol(acc_o.layout()));
#pragma unroll
    for (int mi = 0; mi < size<0>(acc_o_rowcol); ++mi) {
      const float sum = row_sum(mi);
      // Empty or fully masked rows produce a harmless scale.
      const bool invalid_sum = sum == 0.0f || sum != sum;
      const float inv_sum = invalid_sum ? 1.0f : 1.0f / sum;
#pragma unroll
      for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) {
        acc_o_rowcol(mi, ni) *= inv_sum;
      }
    }
  }
};

// Score mask for Gemma's causal attention variants. Local layers additionally
// clamp the visible keys to the sliding window.
template <typename KernelTraits, typename Engine, typename Layout,
          typename CoordEngine, typename CoordLayout>
__forceinline__ __device__ void gemma4_apply_score_mask(
    Tensor<Engine, Layout> &tensor_,
    Tensor<CoordEngine, CoordLayout> const &coords_,
    const int col_idx_offset,
    const int row_idx_offset,
    const int max_seqlen_k,
    const int max_seqlen_q,
    const int window_size) {
  Tensor tensor = make_tensor(tensor_.data(), gemma4_fa_acc_rowcol(tensor_.layout()));
  Tensor coords = make_tensor(coords_.data(), gemma4_fa_acc_rowcol(coords_.layout()));
#pragma unroll
  for (int mi = 0; mi < size<0, 1>(tensor); ++mi) {
#pragma unroll
    for (int i = 0; i < size<0, 0>(tensor); ++i) {
      const auto row_coord = make_coord(i, mi);
#pragma unroll
      for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
#pragma unroll
        for (int j = 0; j < size<1, 0>(tensor); ++j) {
          const auto col_coord = make_coord(j, nj);
          const auto score_coord = coords(row_coord, col_coord);
          const int row_idx = row_idx_offset + int(get<0>(score_coord));
          const int col_idx = col_idx_offset + int(get<1>(score_coord));
          const int key_row = row_idx + max_seqlen_k - max_seqlen_q;
          const int right = std::min(max_seqlen_k, key_row + 1);
          bool masked = col_idx >= right;
          if constexpr (KernelTraits::kIsLocal) {
            const int left = std::max(0, key_row - window_size + 1);
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

// Returns whether a score tile is already inside the causal/window bounds.
template <typename KernelTraits>
__device__ __forceinline__ bool gemma4_score_block_fully_visible(
    int col_idx_offset,
    int block_n,
    int row_idx_offset,
    int valid_rows,
    int seqlen_delta,
    int max_seqlen_k,
    int window_size) {
  // Partial K tiles still need masks for columns beyond seqlen_k.
  if (col_idx_offset + block_n > max_seqlen_k) return false;

  // The last key column in this K tile must be no later than the earliest
  // allowed causal key for the first valid query row.
  const int block_end = std::min(max_seqlen_k, col_idx_offset + block_n) - 1;
  const int earliest_right = row_idx_offset + seqlen_delta;
  if (block_end > earliest_right) return false;

  if constexpr (KernelTraits::kIsLocal) {
    // The latest valid query row has the strictest left-window boundary.
    const int latest_key_row = row_idx_offset + valid_rows - 1 + seqlen_delta;
    const int latest_left = std::max(0, latest_key_row - window_size + 1);
    return col_idx_offset >= latest_left;
  }

  return true;
}

template <typename Pointer>
__forceinline__ __device__ auto gemma4_make_bshd_view(
    Pointer ptr,
    int seqlen,
    int heads,
    int head_dim) {
  return make_tensor(
      make_gmem_ptr(ptr),
      make_shape(seqlen, heads, head_dim),
      make_stride(heads * head_dim, head_dim, _1{}));
}

// Store one prefill O tile for one batch and query-head row block.
template <typename KernelTraits, typename TensorAccO, typename TiledMmaPV>
__forceinline__ __device__ void gemma4_store_o_tile(
    const Gemma4FlashFwdParams &params,
    Gemma4FlashFwdParams::index_t q_batch_offset,
    int bidh,
    int m_block,
    int q_tile_remaining,
    TensorAccO &acc_o,
    TiledMmaPV tiled_mma_pv,
    int tidx,
    char *smem) {
  using Element = typename KernelTraits::Element;
  constexpr int kBlockM = KernelTraits::kBlockM;
  constexpr int kHeadDim = KernelTraits::kHeadDim;

  using AccOElement = typename TensorAccO::value_type;
  constexpr int kAccONumel = decltype(size(acc_o))::value;
  cutlass::NumericArrayConverter<Element, AccOElement, kAccONumel>
      convert_acc_o;
  const auto *acc_o_array =
      reinterpret_cast<const cutlass::Array<AccOElement, kAccONumel> *>(
          acc_o.data());
  auto acc_o_frag = convert_acc_o(*acc_o_array);
  Tensor rO = make_tensor(make_rmem_ptr<Element>(&acc_o_frag), acc_o.layout());
  Tensor sO = make_tensor(
      make_smem_ptr(reinterpret_cast<Element *>(smem)),
      typename KernelTraits::SmemLayoutO{});
  auto smem_tiled_copy_O =
      make_tiled_copy_C(typename KernelTraits::SmemCopyAtomO{}, tiled_mma_pv);
  auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
  Tensor taccOrO = smem_thr_copy_O.retile_S(rO);
  Tensor taccOsO = smem_thr_copy_O.partition_D(sO);
  cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);

  Tensor mO = gemma4_make_bshd_view(
      params.o_ptr + q_batch_offset, params.seqlen_q,
      GEMMA4_NUM_QUERY_HEADS, kHeadDim);
  Tensor gO = local_tile(
      mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
      make_coord(m_block, 0));
  typename KernelTraits::GmemTiledCopyO gmem_tiled_copy_O;
  auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
  Tensor tOsO = gmem_thr_copy_O.partition_S(sO);
  Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
  __syncthreads();

  Tensor tOrO = make_tensor<Element>(shape(tOgO));
  cute::copy(gmem_tiled_copy_O, tOsO, tOrO);

  Tensor cO = make_identity_tensor(make_shape(size<0>(sO), size<1>(sO)));
  Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
  gemma4_fa_copy_mn_guarded(
      gmem_tiled_copy_O, tOrO, tOgO, tOcO, q_tile_remaining);
}

enum class Gemma4KvBlockPhase {
  kFirst,
  kBoundary,
  kSteady,
};

// Processes one streamed K/V block in the prefill FlashAttention loop.
template <
    typename KernelTraits, Gemma4KvBlockPhase Phase, typename TensorAccO,
    typename Softmax,
    typename TiledMmaQK, typename TiledMmaPV, typename GmemTiledCopyQKV,
    typename TensorVgV, typename TensorVsV, typename TensorKVcKV,
    typename TensorKgK, typename TensorKsK, typename TensorSrQ,
    typename TensorSrK, typename TensorOrVt, typename TensorSsQ,
    typename TensorSsK, typename TensorOsVt, typename SmemTiledCopyQ,
    typename SmemTiledCopyK, typename SmemTiledCopyV,
    typename SmemThrCopyQ, typename SmemThrCopyK, typename SmemThrCopyV>
__forceinline__ __device__ void gemma4_process_kv_block(
    const Gemma4FlashFwdParams &params,
    int n_block,
    int n_block_min,
    int q_tile_start,
    int valid_q_rows,
    int seqlen_delta,
    TensorAccO &acc_o,
    Softmax &softmax,
    TiledMmaQK tiled_mma_qk,
    TiledMmaPV tiled_mma_pv,
    GmemTiledCopyQKV gmem_tiled_copy_QKV,
    TensorVgV &tVgV,
    TensorVsV &tVsV,
    TensorKVcKV &tKVcKV,
    TensorKgK &tKgK,
    TensorKsK &tKsK,
    TensorSrQ &tSrQ,
    TensorSrK &tSrK,
    TensorOrVt &tOrVt,
    TensorSsQ &tSsQ,
    TensorSsK &tSsK,
    TensorOsVt &tOsVt,
    SmemTiledCopyQ smem_tiled_copy_Q,
    SmemTiledCopyK smem_tiled_copy_K,
    SmemTiledCopyV smem_tiled_copy_V,
    SmemThrCopyQ smem_thr_copy_Q,
    SmemThrCopyK smem_thr_copy_K,
    SmemThrCopyV smem_thr_copy_V) {
  using Element = typename KernelTraits::Element;
  constexpr int kBlockM = KernelTraits::kBlockM;
  constexpr int kBlockN = KernelTraits::kBlockN;
  constexpr bool kFirstBlock = Phase == Gemma4KvBlockPhase::kFirst;
  constexpr bool kMayNeedMask =
      Phase != Gemma4KvBlockPhase::kSteady || KernelTraits::kIsLocal;

  Tensor acc_s = partition_fragment_C(tiled_mma_qk, Shape<Int<kBlockM>, Int<kBlockN>>{});
  auto thr_mma_qk = tiled_mma_qk.get_thread_slice(threadIdx.x);
  Tensor cS = make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});
  Tensor tScS = thr_mma_qk.partition_C(cS);
  clear(acc_s);
  asm volatile("cp.async.wait_group 0;\n" ::);
  __syncthreads();

  if constexpr (kFirstBlock) {
    gemma4_fa_copy_mn_guarded_clear(
        gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVsV, tKVcKV,
        params.seqlen_k - n_block * kBlockN);
  } else {
    cute::copy(gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVsV);
  }
  cute::cp_async_fence();

  gemma4_fa_gemm(
      acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma_qk, smem_tiled_copy_Q,
      smem_tiled_copy_K, smem_thr_copy_Q, smem_thr_copy_K);

  bool score_block_fully_visible = true;
  if constexpr (kMayNeedMask) {
    const int score_col_offset = n_block * kBlockN;
    score_block_fully_visible =
        gemma4_score_block_fully_visible<KernelTraits>(
            score_col_offset, kBlockN, q_tile_start, valid_q_rows,
            seqlen_delta, params.seqlen_k, params.window_size);
    if (!score_block_fully_visible) {
      gemma4_apply_score_mask<KernelTraits>(
          acc_s, tScS, score_col_offset, q_tile_start, params.seqlen_k,
          params.seqlen_q, params.window_size);
    }
  }

  asm volatile("cp.async.wait_group 0;\n" ::);
  __syncthreads();
  if (n_block > n_block_min) {
    cute::copy(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block - 1), tKsK);
    cute::cp_async_fence();
  }

  if constexpr (kFirstBlock) {
    softmax.softmax_rescale_first_block(acc_s, params.scale_softmax_log2);
  } else {
    softmax.template softmax_rescale_next_block<kMayNeedMask>(
        acc_s, acc_o, params.scale_softmax_log2, score_block_fully_visible);
  }

  using AccSElement = typename decltype(acc_s)::value_type;
  constexpr int kAccSNumel = decltype(size(acc_s))::value;
  cutlass::NumericArrayConverter<Element, AccSElement, kAccSNumel>
      convert_acc_s;
  const auto *acc_s_array =
      reinterpret_cast<const cutlass::Array<AccSElement, kAccSNumel> *>(
          acc_s.data());
  auto acc_s_frag = convert_acc_s(*acc_s_array);
  Tensor rP = make_tensor(make_rmem_ptr<Element>(&acc_s_frag), acc_s.layout());
  Tensor tOrP = make_tensor(
      rP.data(),
      gemma4_fa_acc_Aregs<typename KernelTraits::TiledMmaPV>(rP.layout()));
  gemma4_fa_gemm_rs(
      acc_o, tOrP, tOrVt, tOsVt, tiled_mma_pv, smem_tiled_copy_V,
      smem_thr_copy_V);
}

// Compute one block of query rows for one batch and one query head. This is the
// heart of the kernel:
//   1. stage Q once;
//   2. stream K/V blocks from right to left;
//   3. compute QK^T, mask, online softmax, and P*V;
//   4. normalize and write O.
template <typename KernelTraits>
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
  constexpr int kHeadRatio = GEMMA4_NUM_QUERY_HEADS / KernelTraits::kKvHeads;
  constexpr int kQRowStride = GEMMA4_NUM_QUERY_HEADS * kHeadDim;
  constexpr int kKVRowStride = KernelTraits::kKvHeads * kHeadDim;
  // Fixed batch-major tensors keep batch offset math direct.
  const index_t batch = index_t(bidb);
  const int q_tile_start = m_block * kBlockM;
  const int q_tile_remaining = params.seqlen_q - q_tile_start;
  const int valid_q_rows = std::min(kBlockM, q_tile_remaining);

  const int seqlen_delta = params.seqlen_k - params.seqlen_q;
  const index_t q_batch_offset = batch * index_t(params.seqlen_q) * kQRowStride;
  const index_t kv_batch_offset = batch * index_t(params.seqlen_k) * kKVRowStride;

  // Work backward over only the visible K blocks. Local layers clamp the left
  // side to the sliding window; global layers start at block zero.
  int n_block_min = 0;
  if constexpr (KernelTraits::kIsLocal) {
    n_block_min = std::max(
        0, (q_tile_start + seqlen_delta - params.window_size + 1) / kBlockN);
  }
  // The right edge is causal: queries in this tile never need blocks wholly in
  // the future, even when seqlen_q != seqlen_k during decode.
  const int n_block_max = std::min(
      cute::ceil_div(params.seqlen_k, kBlockN),
      cute::ceil_div(q_tile_start + kBlockM + seqlen_delta, kBlockN));

  typename KernelTraits::TiledMmaPV tiled_mma_pv;
  auto thr_mma_pv = tiled_mma_pv.get_thread_slice(tidx);
  Tensor acc_o = partition_fragment_C(
      tiled_mma_pv, Shape<Int<kBlockM>, Int<kHeadDim>>{});

  // If a local tile has no visible keys, write zero output so edge cases stay
  // defined instead of relying on stale output memory.
  if (n_block_max <= n_block_min) {
    clear(acc_o);
    gemma4_store_o_tile<KernelTraits>(
        params, q_batch_offset, bidh, m_block, q_tile_remaining, acc_o,
        tiled_mma_pv, tidx, smem_);
    return;
  }

  // Create global-memory tensor views, then select the tile for this query
  // block and the KV head mapped by Gemma's grouped-query attention ratio.
  Tensor mQ = gemma4_make_bshd_view(
      params.q_ptr + q_batch_offset, params.seqlen_q,
      GEMMA4_NUM_QUERY_HEADS, kHeadDim);
  Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_coord(m_block, 0));
  const int kv_head = bidh / kHeadRatio;
  Tensor mK = gemma4_make_bshd_view(
      params.k_ptr + kv_batch_offset, params.seqlen_k,
      KernelTraits::kKvHeads, kHeadDim);
  Tensor gK = local_tile(mK(_, kv_head, _), Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_coord(_, 0));
  Tensor mV = gemma4_make_bshd_view(
      params.v_ptr + kv_batch_offset, params.seqlen_k,
      KernelTraits::kKvHeads, kHeadDim);
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
  auto thr_mma_qk = tiled_mma_qk.get_thread_slice(tidx);
  Tensor tSrQ = thr_mma_qk.partition_fragment_A(sQ);
  Tensor tSrK = thr_mma_qk.partition_fragment_B(sK);
  Tensor tOrVt = thr_mma_pv.partition_fragment_B(sVtNoSwizzle);

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
  Gemma4FlashSoftmax<2 * size<1>(acc_o), KernelTraits> softmax;

  // The first few blocks may touch the causal/window boundary and need masks.
  // After that, global blocks are fully visible; local blocks still check the
  // left edge of the sliding window as part of the steady phase.
  constexpr int kMaskingSteps = cute::ceil_div(kBlockM, kBlockN) + 1;
  gemma4_process_kv_block<
      KernelTraits, Gemma4KvBlockPhase::kFirst>(
      params, n_block, n_block_min, q_tile_start, valid_q_rows,
      seqlen_delta, acc_o, softmax, tiled_mma_qk, tiled_mma_pv,
      gmem_tiled_copy_QKV, tVgV, tVsV, tKVcKV, tKgK, tKsK, tSrQ,
      tSrK, tOrVt, tSsQ, tSsK, tOsVt, smem_tiled_copy_Q,
      smem_tiled_copy_K, smem_tiled_copy_V, smem_thr_copy_Q,
      smem_thr_copy_K, smem_thr_copy_V);
  --n_block;
#pragma unroll
  for (int masking_step = 1;
       masking_step < kMaskingSteps && n_block >= n_block_min;
       ++masking_step, --n_block) {
    gemma4_process_kv_block<
        KernelTraits, Gemma4KvBlockPhase::kBoundary>(
        params, n_block, n_block_min, q_tile_start, valid_q_rows,
        seqlen_delta, acc_o, softmax, tiled_mma_qk, tiled_mma_pv,
        gmem_tiled_copy_QKV, tVgV, tVsV, tKVcKV, tKgK, tKsK, tSrQ,
        tSrK, tOrVt, tSsQ, tSsK, tOsVt, smem_tiled_copy_Q,
        smem_tiled_copy_K, smem_tiled_copy_V, smem_thr_copy_Q,
        smem_thr_copy_K, smem_thr_copy_V);
  }
  for (; n_block >= n_block_min; --n_block) {
    gemma4_process_kv_block<
        KernelTraits, Gemma4KvBlockPhase::kSteady>(
        params, n_block, n_block_min, q_tile_start, valid_q_rows,
        seqlen_delta, acc_o, softmax, tiled_mma_qk, tiled_mma_pv,
        gmem_tiled_copy_QKV, tVgV, tVsV, tKVcKV, tKgK, tKsK, tSrQ,
        tSrK, tOrVt, tSsQ, tSsK, tOsVt, smem_tiled_copy_Q,
        smem_tiled_copy_K, smem_tiled_copy_V, smem_thr_copy_Q,
        smem_thr_copy_K, smem_thr_copy_V);
  }

  // Finalize O = softmax(QK^T)V and store through shared memory so the MMA
  // accumulator fragment can be rearranged into coalesced global writes.
  softmax.normalize_softmax(acc_o);
  gemma4_store_o_tile<KernelTraits>(
      params, q_batch_offset, bidh, m_block, q_tile_remaining, acc_o,
      tiled_mma_pv, tidx, smem_);
}

// One CTA computes one query row-block for one batch and one query head.
template <typename KernelTraits>
__global__ void gemma4_flash_fwd_bf16_kernel(
    __grid_constant__ const Gemma4FlashFwdParams params) {
  const int m_block = blockIdx.x;
  const int bidb = blockIdx.y;
  const int bidh = blockIdx.z;
  gemma4_compute_attn_1rowblock<KernelTraits>(params, bidb, bidh, m_block);
}

template <typename Traits>
__device__ __forceinline__ void prep_load_head_values(
    const __nv_bfloat16 *__restrict__ in,
    int lane,
    float (&values)[Gemma4AttentionDerived<Traits>::kValuesPerLane]) {
  using Derived = Gemma4AttentionDerived<Traits>;
#pragma unroll
  for (int i = 0; i < Derived::kValuesPerLane; ++i) {
    // Lane-strided loads make one warp own one complete attention head.
    values[i] = __bfloat162float(in[lane + i * kWarpSize]);
  }
}

// Apply learned RMSNorm to one Q/K head and then rotate the configured rotary
// prefix. Global attention leaves the non-rotary tail as normalized channels.
template <typename Traits>
__device__ __forceinline__ void prep_weighted_rope_head_values(
    __nv_bfloat16 *__restrict__ out,
    const float (&values)[Gemma4AttentionDerived<Traits>::kValuesPerLane],
    const __nv_bfloat16 *__restrict__ weight,
    const float *__restrict__ cos_row,
    const float *__restrict__ sin_row,
    int lane) {
  using Derived = Gemma4AttentionDerived<Traits>;
  const float scale = gemma4_rmsnorm_warp_scale_f32_device(
      values, Derived::kValuesPerLane, Traits::kHeadDim, GEMMA4_RMS_NORM_EPS);
#pragma unroll
  for (int i = 0; i < Derived::kRotaryPairsPerLane; ++i) {
    const int dim = lane + i * kWarpSize;
    const float lo =
        values[i] * scale * __bfloat162float(loadg(weight + dim));
    const int hi_index = i + Derived::kRotaryPairsPerLane;
    const int hi_dim = Derived::kRotaryHalf + dim;
    const float hi = values[hi_index] * scale *
                     __bfloat162float(loadg(weight + hi_dim));
    gemma4_rope::store_rotated_pair_bf16(
        out, cos_row, sin_row, Derived::kRotaryHalf, dim, lo, hi);
  }

#pragma unroll
  for (int i = 0; i < Derived::kValuesPerLane; ++i) {
    const int dim = lane + i * kWarpSize;
    if (dim >= Traits::kRotaryDim) {
      const float value =
          values[i] * scale * __bfloat162float(loadg(weight + dim));
      out[dim] = __float2bfloat16_rn(value);
    }
  }
}

// Apply scale-free RMSNorm to one V head. Global attention derives V from the
// raw K projection and then applies this same scale-free normalization.
template <typename Traits>
__device__ __forceinline__ void prep_scale_free_head_values(
    __nv_bfloat16 *__restrict__ out,
    const float (&values)[Gemma4AttentionDerived<Traits>::kValuesPerLane],
    int lane) {
  using Derived = Gemma4AttentionDerived<Traits>;
  const float scale = gemma4_rmsnorm_warp_scale_f32_device(
      values, Derived::kValuesPerLane, Traits::kHeadDim, GEMMA4_RMS_NORM_EPS);
#pragma unroll
  for (int i = 0; i < Derived::kValuesPerLane; ++i) {
    out[lane + i * kWarpSize] = __float2bfloat16_rn(values[i] * scale);
  }
}

// Prepare one prefill Q/K/V head group for the caller-provided CTA coordinates.
template <typename Traits>
__device__ __forceinline__ void phase_qkv_norm_rope(
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
    const int32_t *__restrict__ token_position,
    int seq_len,
    int seq,
    int head_group,
    int batch,
    int thread_idx) {
  using Derived = Gemma4AttentionDerived<Traits>;
  constexpr int kQHeads = GEMMA4_NUM_QUERY_HEADS;
  constexpr int kKvHeads = Traits::kKvHeads;
  constexpr int kHeadDim = Traits::kHeadDim;

  const int lane = thread_idx & (kWarpSize - 1);
  const int warp = thread_idx / kWarpSize;
  const int head = head_group * Derived::kHeadsPerBlock + warp;
  const int64_t row = int64_t(batch) * seq_len + seq;
  const int position =
      gemma4_warp_uniform_ldg_i32(token_position + row, lane);
  if (position < 0) return;
  // Cos/sin tables are row-major by position and rotary-pair index.
  const float *cos_row = cos + int64_t(position) * Derived::kRotaryHalf;
  const float *sin_row = sin + int64_t(position) * Derived::kRotaryHalf;

  const int64_t q_offset = (int64_t(row) * kQHeads + head) * kHeadDim;
  float q_values[Derived::kValuesPerLane];
  prep_load_head_values<Traits>(q + q_offset, lane, q_values);
  prep_weighted_rope_head_values<Traits>(
      q_prepared + q_offset, q_values, q_norm_weight, cos_row, sin_row, lane);

  // Only KV heads have K/V rows. Extra query-head warps prepare Q and stop.
  if (head < kKvHeads) {
    const int64_t kv_offset = (int64_t(row) * kKvHeads + head) * kHeadDim;
    float k_values[Derived::kValuesPerLane];
    prep_load_head_values<Traits>(k + kv_offset, lane, k_values);
    prep_weighted_rope_head_values<Traits>(
        k_prepared + kv_offset, k_values, k_norm_weight, cos_row, sin_row,
        lane);
    if constexpr (Traits::kHasVProjection) {
      float v_values[Derived::kValuesPerLane];
      prep_load_head_values<Traits>(v + kv_offset, lane, v_values);
      prep_scale_free_head_values<Traits>(
          v_prepared + kv_offset, v_values, lane);
    } else {
      prep_scale_free_head_values<Traits>(
          v_prepared + kv_offset, k_values, lane);
    }
  }
}

// Wrapper: one CTA prepares one sequence position, batch, and head group.
template <typename Traits>
__global__ __launch_bounds__(Gemma4AttentionDerived<Traits>::kPrepThreads)
void gemma4_qkv_norm_rope_kernel(
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
    const int32_t *__restrict__ token_position,
    int seq_len) {
  phase_qkv_norm_rope<Traits>(
      q_prepared, k_prepared, v_prepared, q, k, v, q_norm_weight,
      k_norm_weight, cos, sin, token_position, seq_len, int(blockIdx.x),
      int(blockIdx.y), int(blockIdx.z), int(threadIdx.x));
}

}  // namespace gemma4_flash_attention

// Gemma sliding prefill helper: Q/K get learned RMSNorm then RoPE; V gets
// scale-free RMSNorm. The prepared tensors keep the normal FA layout:
//   Q: [batch, seq, 16, 256], K/V: [batch, seq, 8, 256].
cudaError_t gemma4_flash_attention_sliding_fwd_bf16_norm_rope(
    __nv_bfloat16 *__restrict__ d_out,
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
    const int32_t *__restrict__ d_token_position,
    int batch_size,
    int seqlen_q,
    int seqlen_k,
    int window_size,
    float softmax_scale,
    cudaStream_t stream) {
  // This prefill helper assumes contiguous self-attention rows. Decode appends
  // use paged-cache prep instead.
  if (batch_size <= 0 || seqlen_q <= 0 || seqlen_q != seqlen_k ||
      window_size <= 0) {
    return cudaErrorInvalidValue;
  }
  if (d_out == nullptr || d_q_prepared == nullptr ||
      d_k_prepared == nullptr || d_v_prepared == nullptr ||
      d_token_position == nullptr) {
    return cudaErrorInvalidValue;
  }

  using PrepTraits = gemma4_flash_attention::Gemma4AttentionTraits<false>;
  using PrepDerived = gemma4_flash_attention::Gemma4AttentionDerived<PrepTraits>;
  constexpr int kHeadGroups =
      div_up(GEMMA4_NUM_QUERY_HEADS, PrepDerived::kHeadsPerBlock);
  const dim3 prep_grid_dim(seqlen_q, kHeadGroups, batch_size);
  constexpr dim3 prep_block_dim(PrepDerived::kPrepThreads);
  gemma4_flash_attention::gemma4_qkv_norm_rope_kernel<PrepTraits>
      <<<prep_grid_dim, prep_block_dim, 0, stream>>>(
          d_q_prepared, d_k_prepared, d_v_prepared, d_q, d_k, d_v,
          d_q_norm_weight, d_k_norm_weight, d_cos, d_sin, d_token_position,
          seqlen_q);
  GEMMA4_RETURN_IF_CUDA_ERROR(cudaGetLastError());

  gemma4_flash_attention::Gemma4FlashFwdParams params{};
  params.q_ptr =
      reinterpret_cast<const gemma4_flash_attention::Element *>(d_q_prepared);
  params.k_ptr =
      reinterpret_cast<const gemma4_flash_attention::Element *>(d_k_prepared);
  params.v_ptr =
      reinterpret_cast<const gemma4_flash_attention::Element *>(d_v_prepared);
  params.o_ptr = reinterpret_cast<gemma4_flash_attention::Element *>(d_out);
  params.seqlen_q = seqlen_q;
  params.seqlen_k = seqlen_k;
  params.scale_softmax_log2 = softmax_scale * float(M_LOG2E);
  params.window_size = window_size;

  using KernelTraits = gemma4_flash_attention::Gemma4SlidingFa2KernelTraits;
  static bool smem_initialized = false;
  if (!smem_initialized) {
    auto kernel =
        &gemma4_flash_attention::gemma4_flash_fwd_bf16_kernel<KernelTraits>;
    GEMMA4_RETURN_IF_CUDA_ERROR(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
        KernelTraits::kSmemSize));
    smem_initialized = true;
  }

  const dim3 grid_dim(cute::ceil_div(seqlen_q, KernelTraits::kBlockM),
                      batch_size, GEMMA4_NUM_QUERY_HEADS);
  constexpr dim3 block_dim(KernelTraits::kNThreads);
  gemma4_flash_attention::gemma4_flash_fwd_bf16_kernel<KernelTraits>
      <<<grid_dim, block_dim, KernelTraits::kSmemSize, stream>>>(params);
  return cudaGetLastError();
}

// Gemma global prefill helper: Q/K get learned RMSNorm then p-RoPE; V is
// derived from K with scale-free RMSNorm because global layers have no V GEMM.
cudaError_t gemma4_flash_attention_global_fwd_bf16_norm_rope(
    __nv_bfloat16 *__restrict__ d_out,
    __nv_bfloat16 *__restrict__ d_q_prepared,
    __nv_bfloat16 *__restrict__ d_k_prepared,
    __nv_bfloat16 *__restrict__ d_v_prepared,
    const __nv_bfloat16 *__restrict__ d_q,
    const __nv_bfloat16 *__restrict__ d_k,
    const __nv_bfloat16 *__restrict__ d_q_norm_weight,
    const __nv_bfloat16 *__restrict__ d_k_norm_weight,
    const float *__restrict__ d_cos,
    const float *__restrict__ d_sin,
    const int32_t *__restrict__ d_token_position,
    int batch_size,
    int seqlen_q,
    int seqlen_k,
    float softmax_scale,
    cudaStream_t stream) {
  if (batch_size <= 0 || seqlen_q <= 0 || seqlen_q != seqlen_k) {
    return cudaErrorInvalidValue;
  }
  if (d_out == nullptr || d_q_prepared == nullptr ||
      d_k_prepared == nullptr || d_v_prepared == nullptr ||
      d_token_position == nullptr) {
    return cudaErrorInvalidValue;
  }

  using PrepTraits = gemma4_flash_attention::Gemma4AttentionTraits<true>;
  using PrepDerived = gemma4_flash_attention::Gemma4AttentionDerived<PrepTraits>;
  constexpr int kHeadGroups =
      div_up(GEMMA4_NUM_QUERY_HEADS, PrepDerived::kHeadsPerBlock);
  const dim3 prep_grid_dim(seqlen_q, kHeadGroups, batch_size);
  constexpr dim3 prep_block_dim(PrepDerived::kPrepThreads);
  gemma4_flash_attention::gemma4_qkv_norm_rope_kernel<PrepTraits>
      <<<prep_grid_dim, prep_block_dim, 0, stream>>>(
          d_q_prepared, d_k_prepared, d_v_prepared, d_q, d_k, d_k,
          d_q_norm_weight, d_k_norm_weight, d_cos, d_sin, d_token_position,
          seqlen_q);
  GEMMA4_RETURN_IF_CUDA_ERROR(cudaGetLastError());

  gemma4_flash_attention::Gemma4FlashFwdParams params{};
  params.q_ptr =
      reinterpret_cast<const gemma4_flash_attention::Element *>(d_q_prepared);
  params.k_ptr =
      reinterpret_cast<const gemma4_flash_attention::Element *>(d_k_prepared);
  params.v_ptr =
      reinterpret_cast<const gemma4_flash_attention::Element *>(d_v_prepared);
  params.o_ptr = reinterpret_cast<gemma4_flash_attention::Element *>(d_out);
  params.seqlen_q = seqlen_q;
  params.seqlen_k = seqlen_k;
  params.scale_softmax_log2 = softmax_scale * float(M_LOG2E);
  params.window_size = 0;

  using KernelTraits = gemma4_flash_attention::Gemma4GlobalFa2KernelTraits;
  static bool smem_initialized = false;
  if (!smem_initialized) {
    auto kernel =
        &gemma4_flash_attention::gemma4_flash_fwd_bf16_kernel<KernelTraits>;
    GEMMA4_RETURN_IF_CUDA_ERROR(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
        KernelTraits::kSmemSize));
    smem_initialized = true;
  }

  const dim3 grid_dim(cute::ceil_div(seqlen_q, KernelTraits::kBlockM),
                      batch_size, GEMMA4_NUM_QUERY_HEADS);
  constexpr dim3 block_dim(KernelTraits::kNThreads);
  gemma4_flash_attention::gemma4_flash_fwd_bf16_kernel<KernelTraits>
      <<<grid_dim, block_dim, KernelTraits::kSmemSize, stream>>>(params);
  return cudaGetLastError();
}
