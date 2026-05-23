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
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <type_traits>

#include <cute/tensor.hpp>
#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/layout/layout.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

#include "gemma4_flash_attention.cuh"
#include "gemma4.h"

namespace gemma4_flash_attention {

using namespace cute;

constexpr int kWarpSize = 32;
using Gemma4Fa2Element = cutlass::bfloat16_t;

struct Gemma4FlashFwdParams {
  using index_t = int64_t;

  void *__restrict__ q_ptr;
  void *__restrict__ k_ptr;
  void *__restrict__ v_ptr;
  void *__restrict__ o_ptr;
  void *__restrict__ softmax_lse_ptr;

  index_t q_batch_stride;
  index_t k_batch_stride;
  index_t v_batch_stride;
  index_t o_batch_stride;
  index_t q_row_stride;
  index_t k_row_stride;
  index_t v_row_stride;
  index_t o_row_stride;
  index_t q_head_stride;
  index_t k_head_stride;
  index_t v_head_stride;
  index_t o_head_stride;

  int b;
  int h;
  int h_k;
  int h_h_k_ratio;
  int seqlen_q;
  int seqlen_k;
  int seqlen_q_rounded;
  int seqlen_k_rounded;
  int d;
  int total_q;

  float scale_softmax;
  float scale_softmax_log2;
  int window_size_left;
};

template <typename T>
constexpr T round_up(T value, T multiple) {
  return ((value + multiple - 1) / multiple) * multiple;
}

template <int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_,
          typename elem_type = cutlass::half_t>
struct Gemma4FlashKernelTraits {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  using Element = elem_type;
  static constexpr bool Has_cp_async = true;
#else
  using Element = cutlass::half_t;
  static constexpr bool Has_cp_async = false;
#endif

  using ElementAccum = float;
  using index_t = int64_t;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  using MMA_Atom_Arch = std::conditional_t<
      std::is_same_v<elem_type, cutlass::half_t>,
      MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
      MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>>;
#else
  using MMA_Atom_Arch = MMA_Atom<SM75_16x8x8_F32F16F16F32_TN>;
#endif

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
  using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, elem_type>;
  using SmemCopyAtomTransposed = Copy_Atom<SM75_U16x8_LDSM_T, elem_type>;
#else
  using SmemCopyAtom = Copy_Atom<DefaultCopy, elem_type>;
  using SmemCopyAtomTransposed = Copy_Atom<DefaultCopy, elem_type>;
#endif
};

template <int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_,
          typename elem_type = cutlass::half_t,
          typename Base = Gemma4FlashKernelTraits<kHeadDim_, kBlockM_,
                                                  kBlockN_, kNWarps_,
                                                  elem_type>>
struct Gemma4FlashFwdKernelTraits : public Base {
  using Element = typename Base::Element;
  using ElementAccum = typename Base::ElementAccum;
  using index_t = typename Base::index_t;
  using SmemCopyAtom = typename Base::SmemCopyAtom;
  using SmemCopyAtomTransposed = typename Base::SmemCopyAtomTransposed;

  static constexpr bool Has_cp_async = Base::Has_cp_async;
  static constexpr bool Share_Q_K_smem = false;
  static constexpr bool Is_Q_in_regs = false;

  static constexpr int kNWarps = kNWarps_;
  static constexpr int kNThreads = kNWarps * kWarpSize;
  static constexpr int kBlockM = kBlockM_;
  static constexpr int kBlockN = kBlockN_;
  static constexpr int kHeadDim = kHeadDim_;
  static_assert(kHeadDim % 32 == 0, "FA head dim must be a multiple of 32");

  static constexpr int kBlockKSmem = kHeadDim % 64 == 0 ? 64 : 32;
  static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
  static constexpr int kGmemThreadsPerRow = kBlockKSmem / kGmemElemsPerLoad;
  static constexpr int kSwizzle = kBlockKSmem == 32 ? 2 : 3;
  static_assert(kHeadDim % kGmemElemsPerLoad == 0,
                "FA head dim must be a multiple of vector load width");
  static_assert(kNThreads % kGmemThreadsPerRow == 0,
                "FA thread count must divide row load layout");

  using TiledMma = TiledMMA<
      typename Base::MMA_Atom_Arch,
      Layout<Shape<Int<kNWarps>, _1, _1>>,
      Tile<Int<16 * kNWarps>, _16, _16>>;

  using SmemLayoutAtomQ = decltype(
      composition(Swizzle<kSwizzle, 3, 3>{},
                  Layout<Shape<_8, Int<kBlockKSmem>>,
                         Stride<Int<kBlockKSmem>, _1>>{}));
  using SmemLayoutQ = decltype(
      tile_to_shape(SmemLayoutAtomQ{},
                    Shape<Int<kBlockM>, Int<kHeadDim>>{}));
  using SmemLayoutKV = decltype(
      tile_to_shape(SmemLayoutAtomQ{},
                    Shape<Int<kBlockN>, Int<kHeadDim>>{}));
  using SmemLayoutVtransposed = decltype(
      composition(SmemLayoutKV{},
                  make_layout(Shape<Int<kHeadDim>, Int<kBlockN>>{},
                              GenRowMajor{})));
  using SmemLayoutVtransposedNoSwizzle =
      decltype(get_nonswizzle_portion(SmemLayoutVtransposed{}));

  using SmemLayoutAtomO = decltype(
      composition(Swizzle<kSwizzle, 3, 3>{},
                  Layout<Shape<Int<8>, Int<kBlockKSmem>>,
                         Stride<Int<kBlockKSmem>, _1>>{}));
  using SmemLayoutO = decltype(
      tile_to_shape(SmemLayoutAtomO{},
                    Shape<Int<kBlockM>, Int<kHeadDim>>{}));
  using SmemCopyAtomO =
      Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>;

  static constexpr int kSmemQSize = size(SmemLayoutQ{}) * sizeof(Element);
  static constexpr int kSmemKVSize = size(SmemLayoutKV{}) * 2 * sizeof(Element);
  static constexpr int kSmemSize = kSmemQSize + kSmemKVSize;

  using GmemLayoutAtom =
      Layout<Shape<Int<kNThreads / kGmemThreadsPerRow>,
                   Int<kGmemThreadsPerRow>>,
             Stride<Int<kGmemThreadsPerRow>, _1>>;
  using GmemCopyStruct = std::conditional_t<
      Has_cp_async,
      SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>,
      AutoVectorizingCopyWithAssumedAlignment<128>>;
  using GmemTiledCopyQKV = decltype(
      make_tiled_copy(Copy_Atom<GmemCopyStruct, Element>{},
                      GmemLayoutAtom{},
                      Layout<Shape<_1, _8>>{}));
  using GmemTiledCopyO = decltype(
      make_tiled_copy(
          Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>{},
          GmemLayoutAtom{},
          Layout<Shape<_1, _8>>{}));
};

using Gemma4SlidingFa2KernelTraits =
    Gemma4FlashFwdKernelTraits<GEMMA4_SLIDING_HEAD_DIM, 64, 64, 4,
                               Gemma4Fa2Element>;
using Gemma4GlobalFa2KernelTraits =
    Gemma4FlashFwdKernelTraits<GEMMA4_GLOBAL_HEAD_DIM, 32, 32, 2,
                               Gemma4Fa2Element>;

template <typename T>
struct Gemma4MaxOp {
  __device__ __forceinline__ T operator()(T const &x, T const &y) {
    return x > y ? x : y;
  }
};

template <>
struct Gemma4MaxOp<float> {
  __device__ __forceinline__ float operator()(float const &x, float const &y) {
    return max(x, y);
  }
};

template <typename T>
struct Gemma4SumOp {
  __device__ __forceinline__ T operator()(T const &x, T const &y) {
    return x + y;
  }
};

template <int Threads>
struct Gemma4Allreduce {
  static_assert(Threads == 32 || Threads == 16 || Threads == 8 || Threads == 4,
                "Unsupported warp reduction width");
  template <typename T, typename Operator>
  static __device__ __forceinline__ T run(T x, Operator &op) {
    constexpr int kOffset = Threads / 2;
    x = op(x, __shfl_xor_sync(uint32_t(-1), x, kOffset));
    return Gemma4Allreduce<kOffset>::run(x, op);
  }
};

template <>
struct Gemma4Allreduce<2> {
  template <typename T, typename Operator>
  static __device__ __forceinline__ T run(T x, Operator &op) {
    x = op(x, __shfl_xor_sync(uint32_t(-1), x, 1));
    return x;
  }
};

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
  if (!AInRegs) {
    cute::copy(smem_tiled_copy_A, tCsA(_, _, _0{}), tCrA_copy_view(_, _, _0{}));
  }
  if (!BInRegs) {
    cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{}));
  }
#pragma unroll
  for (int i = 0; i < size<2>(tCrA); ++i) {
    if (i < size<2>(tCrA) - 1) {
      if (!AInRegs) {
        cute::copy(smem_tiled_copy_A, tCsA(_, _, i + 1), tCrA_copy_view(_, _, i + 1));
      }
      if (!BInRegs) {
        cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1));
      }
    }
    cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
  }
}

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

template <typename Layout>
__forceinline__ __device__ auto gemma4_fa_acc_rowcol(Layout acc_layout) {
  static_assert(decltype(size<0>(acc_layout))::value == 4);
  auto l = logical_divide(acc_layout, Shape<_2>{});
  return make_layout(make_layout(get<0, 1>(l), get<1>(l)),
                     make_layout(get<0, 0>(l), get<2>(l)));
}

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
    return make_layout(make_layout(get<0>(l), get<2, 0>(l)),
                       get<1>(l), get<2, 1>(l));
  }
}

template <typename ToType, typename Engine, typename Layout>
__forceinline__ __device__ auto gemma4_fa_convert_type(
    Tensor<Engine, Layout> const &tensor) {
  using FromType = typename Engine::value_type;
  constexpr int numel = decltype(size(tensor))::value;
  cutlass::NumericArrayConverter<ToType, FromType, numel> convert_op;
  auto frag = convert_op(
      *reinterpret_cast<const cutlass::Array<FromType, numel> *>(tensor.data()));
  return make_tensor(make_rmem_ptr<ToType>(&frag), tensor.layout());
}

template <int N>
CUTE_HOST_DEVICE void gemma4_fa_cp_async_wait() {
#if defined(CUTE_ARCH_CP_ASYNC_SM80_ENABLED)
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
#endif
}

template <bool IsEvenMN = true, bool IsEvenK = true,
          bool ClearOobMN = false, bool ClearOobK = true,
          typename TiledCopy, typename Engine0, typename Layout0,
          typename Engine1, typename Layout1, typename Engine2,
          typename Layout2, typename Engine3, typename Layout3>
__forceinline__ __device__ void gemma4_fa_copy(
    TiledCopy tiled_copy,
    Tensor<Engine0, Layout0> const &src,
    Tensor<Engine1, Layout1> &dst,
    Tensor<Engine2, Layout2> const &identity_mn,
    Tensor<Engine3, Layout3> const &predicate_k,
    const int max_mn = 0) {
  static_assert(!(ClearOobMN && !ClearOobK),
                "Cannot clear only MN without clearing K");
#pragma unroll
  for (int m = 0; m < size<1>(src); ++m) {
    if (IsEvenMN || get<0>(identity_mn(0, m, 0)) < max_mn) {
#pragma unroll
      for (int k = 0; k < size<2>(src); ++k) {
        if (IsEvenK || predicate_k(k)) {
          cute::copy(tiled_copy, src(_, m, k), dst(_, m, k));
        } else if (ClearOobK) {
          cute::clear(dst(_, m, k));
        }
      }
    } else if (ClearOobMN) {
      cute::clear(dst(_, m, _));
    }
  }
}

template <bool zero_init = true, typename Engine0, typename Layout0,
          typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void gemma4_fa_thread_reduce(
    Tensor<Engine0, Layout0> const &tensor,
    Tensor<Engine1, Layout1> &summary,
    Operator &op) {
#pragma unroll
  for (int mi = 0; mi < size<0>(tensor); mi++) {
    summary(mi) = zero_init ? tensor(mi, 0) : op(summary(mi), tensor(mi, 0));
#pragma unroll
    for (int ni = 1; ni < size<1>(tensor); ni++) {
      summary(mi) = op(summary(mi), tensor(mi, ni));
    }
  }
}

template <typename Engine0, typename Layout0, typename Engine1,
          typename Layout1, typename Operator>
__device__ __forceinline__ void gemma4_fa_quad_allreduce(
    Tensor<Engine0, Layout0> &dst,
    Tensor<Engine1, Layout1> &src,
    Operator &op) {
#pragma unroll
  for (int i = 0; i < size(dst); i++) {
    dst(i) = Gemma4Allreduce<4>::run(src(i), op);
  }
}

template <bool zero_init = true, typename Engine0, typename Layout0,
          typename Engine1, typename Layout1>
__device__ __forceinline__ void gemma4_fa_reduce_max(
    Tensor<Engine0, Layout0> const &tensor,
    Tensor<Engine1, Layout1> &max_values) {
  Gemma4MaxOp<float> max_op;
  gemma4_fa_thread_reduce<zero_init>(tensor, max_values, max_op);
  gemma4_fa_quad_allreduce(max_values, max_values, max_op);
}

template <bool zero_init = true, typename Engine0, typename Layout0,
          typename Engine1, typename Layout1>
__device__ __forceinline__ void gemma4_fa_reduce_sum(
    Tensor<Engine0, Layout0> const &tensor,
    Tensor<Engine1, Layout1> &sum_values) {
  Gemma4SumOp<float> sum_op;
  gemma4_fa_thread_reduce<zero_init>(tensor, sum_values, sum_op);
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
        max_values(mi) == -INFINITY ? 0.0f : max_values(mi) * (ScaleMax ? scale : float(M_LOG2E));
#pragma unroll
    for (int ni = 0; ni < size<1>(tensor); ++ni) {
      tensor(mi, ni) = exp2f(tensor(mi, ni) * scale - max_scaled);
    }
  }
}

template <int Rows>
struct Gemma4FlashSoftmax {
  using TensorT = decltype(make_tensor<float>(Shape<Int<Rows>>{}));
  TensorT row_max;
  TensorT row_sum;

  template <bool IsFirst, bool CheckInf = false, typename Tensor0, typename Tensor1>
  __forceinline__ __device__ void softmax_rescale_o(
      Tensor0 &acc_s,
      Tensor1 &acc_o,
      float softmax_scale_log2) {
    Tensor scores = make_tensor(acc_s.data(), gemma4_fa_acc_rowcol(acc_s.layout()));
    if (IsFirst) {
      gemma4_fa_reduce_max</*zero_init=*/true>(scores, row_max);
      gemma4_fa_scale_apply_exp2(scores, row_max, softmax_scale_log2);
      gemma4_fa_reduce_sum</*zero_init=*/true>(scores, row_sum);
    } else {
      Tensor scores_max_prev = make_fragment_like(row_max);
      cute::copy(row_max, scores_max_prev);
      gemma4_fa_reduce_max</*zero_init=*/false>(scores, row_max);
      Tensor acc_o_rowcol =
          make_tensor(acc_o.data(), gemma4_fa_acc_rowcol(acc_o.layout()));
#pragma unroll
      for (int mi = 0; mi < size(row_max); ++mi) {
        float scores_max_cur =
            !CheckInf ? row_max(mi) : (row_max(mi) == -INFINITY ? 0.0f : row_max(mi));
        float scores_scale =
            exp2f((scores_max_prev(mi) - scores_max_cur) * softmax_scale_log2);
        row_sum(mi) *= scores_scale;
#pragma unroll
        for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) {
          acc_o_rowcol(mi, ni) *= scores_scale;
        }
      }
      gemma4_fa_scale_apply_exp2(scores, row_max, softmax_scale_log2);
      gemma4_fa_reduce_sum</*zero_init=*/false>(scores, row_sum);
    }
  }

  template <typename Tensor0>
  __forceinline__ __device__ TensorT normalize_softmax_lse(
      Tensor0 &acc_o,
      float softmax_scale) {
    Gemma4SumOp<float> sum_op;
    gemma4_fa_quad_allreduce(row_sum, row_sum, sum_op);
    TensorT lse = make_fragment_like(row_sum);
    Tensor acc_o_rowcol = make_tensor(acc_o.data(), gemma4_fa_acc_rowcol(acc_o.layout()));
#pragma unroll
    for (int mi = 0; mi < size<0>(acc_o_rowcol); ++mi) {
      float sum = row_sum(mi);
      float inv_sum = (sum == 0.0f || sum != sum) ? 1.0f : 1.0f / sum;
      lse(mi) = (sum == 0.0f || sum != sum) ? INFINITY : row_max(mi) * softmax_scale + __logf(sum);
#pragma unroll
      for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) {
        acc_o_rowcol(mi, ni) *= inv_sum;
      }
    }
    return lse;
  }
};

template <typename Engine, typename Layout>
__forceinline__ __device__ void gemma4_apply_local_sliding_mask(
    Tensor<Engine, Layout> &tensor_,
    const int col_idx_offset_,
    const int row_idx_offset,
    const int warp_row_stride,
    const int max_seqlen_k,
    const int max_seqlen_q,
    const int window_size_left) {
  Tensor tensor = make_tensor(tensor_.data(), gemma4_fa_acc_rowcol(tensor_.layout()));
  const int lane_id = threadIdx.x % kWarpSize;
  const int col_idx_offset = col_idx_offset_ + (lane_id % 4) * 2;
#pragma unroll
  for (int mi = 0; mi < size<0, 1>(tensor); ++mi) {
    const int row_idx_base = row_idx_offset + mi * warp_row_stride;
#pragma unroll
    for (int i = 0; i < size<0, 0>(tensor); ++i) {
      const int row_idx = row_idx_base + i * 8;
      const int left = std::max(0, row_idx + max_seqlen_k - max_seqlen_q - window_size_left);
      const int right = std::min(max_seqlen_k, row_idx + 1 + max_seqlen_k - max_seqlen_q);
#pragma unroll
      for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
        const int col_idx_base = col_idx_offset + nj * 8;
#pragma unroll
        for (int j = 0; j < size<1, 0>(tensor); ++j) {
          const int col_idx = col_idx_base + j;
          if (col_idx >= right || col_idx < left) {
            tensor(make_coord(i, mi), make_coord(j, nj)) = -INFINITY;
          }
        }
      }
    }
  }
}

template <typename Engine, typename Layout>
__forceinline__ __device__ void gemma4_apply_causal_mask(
    Tensor<Engine, Layout> &tensor_,
    const int col_idx_offset_,
    const int row_idx_offset,
    const int warp_row_stride,
    const int max_seqlen_k,
    const int max_seqlen_q) {
  Tensor tensor = make_tensor(tensor_.data(), gemma4_fa_acc_rowcol(tensor_.layout()));
  const int lane_id = threadIdx.x % kWarpSize;
  const int col_idx_offset = col_idx_offset_ + (lane_id % 4) * 2;
#pragma unroll
  for (int mi = 0; mi < size<0, 1>(tensor); ++mi) {
    const int row_idx_base = row_idx_offset + mi * warp_row_stride;
#pragma unroll
    for (int i = 0; i < size<0, 0>(tensor); ++i) {
      const int row_idx = row_idx_base + i * 8;
      const int right = std::min(max_seqlen_k, row_idx + 1 + max_seqlen_k - max_seqlen_q);
#pragma unroll
      for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
        const int col_idx_base = col_idx_offset + nj * 8;
#pragma unroll
        for (int j = 0; j < size<1, 0>(tensor); ++j) {
          const int col_idx = col_idx_base + j;
          if (col_idx >= right) {
            tensor(make_coord(i, mi), make_coord(j, nj)) = -INFINITY;
          }
        }
      }
    }
  }
}

template <bool IsLocal, typename Engine, typename Layout>
__forceinline__ __device__ void gemma4_apply_attention_mask(
    Tensor<Engine, Layout> &tensor,
    const int col_idx_offset,
    const int row_idx_offset,
    const int warp_row_stride,
    const int max_seqlen_k,
    const int max_seqlen_q,
    const int window_size_left) {
  if constexpr (IsLocal) {
    gemma4_apply_local_sliding_mask(tensor, col_idx_offset, row_idx_offset,
                                    warp_row_stride, max_seqlen_k, max_seqlen_q,
                                    window_size_left);
  } else {
    gemma4_apply_causal_mask(tensor, col_idx_offset, row_idx_offset,
                             warp_row_stride, max_seqlen_k, max_seqlen_q);
  }
}

struct Gemma4FixedBlockInfo {
  template <typename Params>
  __device__ explicit Gemma4FixedBlockInfo(const Params &params)
      : actual_seqlen_q(params.seqlen_q),
        actual_seqlen_k(params.seqlen_k) {}

  template <typename index_t>
  __forceinline__ __device__ index_t q_offset(
      const index_t batch_stride,
      const index_t /*row_stride*/,
      const int bidb) const {
    return index_t(bidb) * batch_stride;
  }

  template <typename index_t>
  __forceinline__ __device__ index_t k_offset(
      const index_t batch_stride,
      const index_t /*row_stride*/,
      const int bidb) const {
    return index_t(bidb) * batch_stride;
  }

  const int actual_seqlen_q;
  const int actual_seqlen_k;
};

template <typename ElementAccum, typename Params, int BlockM>
__forceinline__ __device__ auto gemma4_get_lse_tile(
    const Params &params,
    const int bidb,
    const int bidh,
    const int m_block) {
  auto gmem_ptr_lse =
      make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr));
  Tensor mLSE = make_tensor(
      gmem_ptr_lse,
      make_shape(params.b, params.h, params.seqlen_q),
      make_stride(params.h * params.seqlen_q, params.seqlen_q, 1));
  return local_tile(mLSE(bidb, bidh, _), Shape<Int<BlockM>>{},
                    make_coord(m_block));
}

template <typename KernelTraits, typename Params, bool IsLocal>
inline __device__ void gemma4_compute_attn_1rowblock(
    const Params &params,
    const int bidb,
    const int bidh,
    const int m_block) {
  using Element = typename KernelTraits::Element;
  using ElementAccum = typename KernelTraits::ElementAccum;
  using index_t = typename KernelTraits::index_t;

  extern __shared__ char smem_[];

  const int tidx = threadIdx.x;
  constexpr int kBlockM = KernelTraits::kBlockM;
  constexpr int kBlockN = KernelTraits::kBlockN;
  constexpr int kHeadDim = KernelTraits::kHeadDim;
  constexpr int kNWarps = KernelTraits::kNWarps;

  const Gemma4FixedBlockInfo binfo(params);
  if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

  const int n_block_min = IsLocal
                              ? std::max(0,
                                         (m_block * kBlockM +
                                          binfo.actual_seqlen_k -
                                          binfo.actual_seqlen_q -
                                          params.window_size_left) /
                                             kBlockN)
                              : 0;
  int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);
  n_block_max = std::min(
      n_block_max,
      cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k -
                         binfo.actual_seqlen_q,
                     kBlockN));

  if (n_block_max <= n_block_min) {
    Tensor mO = make_tensor(
        make_gmem_ptr(reinterpret_cast<Element *>(params.o_ptr) +
                      binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)),
        make_shape(binfo.actual_seqlen_q, params.h, params.d),
        make_stride(params.o_row_stride, params.o_head_stride, _1{}));
    Tensor gO = local_tile(mO(_, bidh, _),
                           Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(m_block, 0));
    Tensor gLSE =
        gemma4_get_lse_tile<ElementAccum, Params, kBlockM>(params, bidb, bidh, m_block);

    typename KernelTraits::GmemTiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
    Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
    Tensor tOrO = make_tensor<Element>(shape(tOgO));
    clear(tOrO);
    Tensor cO = make_identity_tensor(make_shape(size<0>(gO), size<1>(gO)));
    Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
    Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
    gemma4_fa_copy</*IsEvenMN=*/false, /*IsEvenK=*/true,
                   /*ClearOobMN=*/false, /*ClearOobK=*/false>(
        gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO,
        binfo.actual_seqlen_q - m_block * kBlockM);
#pragma unroll
    for (int m = 0; m < size<1>(tOgO); ++m) {
      const int row = get<0>(tOcO(0, m, 0));
      if (row < binfo.actual_seqlen_q - m_block * kBlockM &&
          get<1>(tOcO(0, m, 0)) == 0) {
        gLSE(row) = INFINITY;
      }
    }
    return;
  }

  Tensor mQ = make_tensor(
      make_gmem_ptr(reinterpret_cast<Element *>(params.q_ptr) +
                    binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)),
      make_shape(binfo.actual_seqlen_q, params.h, params.d),
      make_stride(params.q_row_stride, params.q_head_stride, _1{}));
  Tensor gQ = local_tile(mQ(_, bidh, _),
                         Shape<Int<kBlockM>, Int<kHeadDim>>{},
                         make_coord(m_block, 0));
  Tensor mK = make_tensor(
      make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) +
                    binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb)),
      make_shape(binfo.actual_seqlen_k, params.h_k, params.d),
      make_stride(params.k_row_stride, params.k_head_stride, _1{}));
  Tensor gK = local_tile(mK(_, bidh / params.h_h_k_ratio, _),
                         Shape<Int<kBlockN>, Int<kHeadDim>>{},
                         make_coord(_, 0));
  Tensor mV = make_tensor(
      make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) +
                    binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb)),
      make_shape(binfo.actual_seqlen_k, params.h_k, params.d),
      make_stride(params.v_row_stride, params.v_head_stride, _1{}));
  Tensor gV = local_tile(mV(_, bidh / params.h_h_k_ratio, _),
                         Shape<Int<kBlockN>, Int<kHeadDim>>{},
                         make_coord(_, 0));

  Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                          typename KernelTraits::SmemLayoutQ{});
  Tensor sK = make_tensor(sQ.data() + size(sQ),
                          typename KernelTraits::SmemLayoutKV{});
  Tensor sV = make_tensor(sK.data() + size(sK),
                          typename KernelTraits::SmemLayoutKV{});
  Tensor sVt = make_tensor(sV.data(),
                           typename KernelTraits::SmemLayoutVtransposed{});
  Tensor sVtNoSwizzle =
      make_tensor(sV.data().get(),
                  typename KernelTraits::SmemLayoutVtransposedNoSwizzle{});

  typename KernelTraits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
  auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);
  Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
  Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
  Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);
  Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
  Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);
  Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);

  typename KernelTraits::TiledMma tiled_mma;
  auto thr_mma = tiled_mma.get_thread_slice(tidx);
  Tensor tSrQ = thr_mma.partition_fragment_A(sQ);
  Tensor tSrK = thr_mma.partition_fragment_B(sK);
  Tensor tOrVt = thr_mma.partition_fragment_B(sVtNoSwizzle);
  Tensor acc_o =
      partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});

  auto smem_tiled_copy_Q =
      make_tiled_copy_A(typename KernelTraits::SmemCopyAtom{}, tiled_mma);
  auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
  Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);

  auto smem_tiled_copy_K =
      make_tiled_copy_B(typename KernelTraits::SmemCopyAtom{}, tiled_mma);
  auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
  Tensor tSsK = smem_thr_copy_K.partition_S(sK);

  auto smem_tiled_copy_V =
      make_tiled_copy_B(typename KernelTraits::SmemCopyAtomTransposed{}, tiled_mma);
  auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
  Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);

  Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));
  Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));
  Tensor tQcQ = gmem_thr_copy_QKV.partition_S(cQ);
  Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(cKV);
  Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
  Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));

  gemma4_fa_copy</*IsEvenMN=*/false, /*IsEvenK=*/true>(
      gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ,
      binfo.actual_seqlen_q - m_block * kBlockM);

  int n_block = n_block_max - 1;
  gemma4_fa_copy</*IsEvenMN=*/false, /*IsEvenK=*/true>(
      gmem_tiled_copy_QKV, tKgK(_, _, _, n_block), tKsK, tKVcKV, tKVpKV,
      binfo.actual_seqlen_k - n_block * kBlockN);
  cute::cp_async_fence();

  clear(acc_o);
  Gemma4FlashSoftmax<2 * size<1>(acc_o)> softmax;

  constexpr int kMaskingSteps = cute::ceil_div(kBlockM, kBlockN) + 1;
#pragma unroll
  for (int masking_step = 0;
       masking_step < kMaskingSteps;
       ++masking_step, --n_block) {
    Tensor acc_s =
        partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
    clear(acc_s);
    gemma4_fa_cp_async_wait<0>();
    __syncthreads();

    if (masking_step > 0) {
      gemma4_fa_copy</*IsEvenMN=*/true, /*IsEvenK=*/true>(
          gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVsV, tKVcKV, tKVpKV);
    } else {
      gemma4_fa_copy</*IsEvenMN=*/false, /*IsEvenK=*/true,
                     /*ClearOobMN=*/true>(
          gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVsV, tKVcKV, tKVpKV,
          binfo.actual_seqlen_k - n_block * kBlockN);
    }
    cute::cp_async_fence();

    gemma4_fa_gemm</*AInRegs=*/false>(
        acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q,
        smem_tiled_copy_K, smem_thr_copy_Q, smem_thr_copy_K);

    gemma4_apply_attention_mask<IsLocal>(
        acc_s, n_block * kBlockN,
        m_block * kBlockM + (tidx / kWarpSize) * 16 + (tidx % kWarpSize) / 4,
        kNWarps * 16, binfo.actual_seqlen_k, binfo.actual_seqlen_q,
        params.window_size_left);

    gemma4_fa_cp_async_wait<0>();
    __syncthreads();
    if (n_block > n_block_min) {
      gemma4_fa_copy</*IsEvenMN=*/true, /*IsEvenK=*/true>(
          gmem_tiled_copy_QKV, tKgK(_, _, _, n_block - 1), tKsK, tKVcKV, tKVpKV);
      cute::cp_async_fence();
    }

    masking_step == 0
        ? softmax.template softmax_rescale_o</*IsFirst=*/true, /*CheckInf=*/true>(
              acc_s, acc_o, params.scale_softmax_log2)
        : softmax.template softmax_rescale_o</*IsFirst=*/false, /*CheckInf=*/true>(
              acc_s, acc_o, params.scale_softmax_log2);

    Tensor rP = gemma4_fa_convert_type<Element>(acc_s);
    Tensor tOrP = make_tensor(
        rP.data(),
        gemma4_fa_acc_Aregs<typename KernelTraits::TiledMma>(rP.layout()));
    gemma4_fa_gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma,
                      smem_tiled_copy_V, smem_thr_copy_V);

    if (kMaskingSteps > 1 && n_block <= n_block_min) {
      --n_block;
      break;
    }
  }

  for (; n_block >= n_block_min; --n_block) {
    Tensor acc_s =
        partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
    clear(acc_s);
    gemma4_fa_cp_async_wait<0>();
    __syncthreads();
    gemma4_fa_copy</*IsEvenMN=*/true, /*IsEvenK=*/true>(
        gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVsV, tKVcKV, tKVpKV);
    cute::cp_async_fence();

    gemma4_fa_gemm</*AInRegs=*/false>(
        acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q,
        smem_tiled_copy_K, smem_thr_copy_Q, smem_thr_copy_K);

    gemma4_fa_cp_async_wait<0>();
    __syncthreads();
    if (n_block > n_block_min) {
      gemma4_fa_copy</*IsEvenMN=*/true, /*IsEvenK=*/true>(
          gmem_tiled_copy_QKV, tKgK(_, _, _, n_block - 1), tKsK, tKVcKV, tKVpKV);
      cute::cp_async_fence();
    }

    if constexpr (IsLocal) {
      gemma4_apply_attention_mask<true>(
          acc_s, n_block * kBlockN,
          m_block * kBlockM + (tidx / kWarpSize) * 16 + (tidx % kWarpSize) / 4,
          kNWarps * 16, binfo.actual_seqlen_k, binfo.actual_seqlen_q,
          params.window_size_left);
    }
    softmax.template softmax_rescale_o</*IsFirst=*/false, IsLocal>(
        acc_s, acc_o, params.scale_softmax_log2);

    Tensor rP = gemma4_fa_convert_type<Element>(acc_s);
    Tensor tOrP = make_tensor(
        rP.data(),
        gemma4_fa_acc_Aregs<typename KernelTraits::TiledMma>(rP.layout()));
    gemma4_fa_gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma,
                      smem_tiled_copy_V, smem_thr_copy_V);
  }

  Tensor lse = softmax.normalize_softmax_lse(acc_o, params.scale_softmax);
  Tensor rO = gemma4_fa_convert_type<Element>(acc_o);
  Tensor sO = make_tensor(sQ.data(), typename KernelTraits::SmemLayoutO{});
  auto smem_tiled_copy_O =
      make_tiled_copy_C(typename KernelTraits::SmemCopyAtomO{}, tiled_mma);
  auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
  Tensor taccOrO = smem_thr_copy_O.retile_S(rO);
  Tensor taccOsO = smem_thr_copy_O.partition_D(sO);
  cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);

  Tensor mO = make_tensor(
      make_gmem_ptr(reinterpret_cast<Element *>(params.o_ptr) +
                    binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)),
      make_shape(binfo.actual_seqlen_q, params.h, params.d),
      make_stride(params.o_row_stride, params.o_head_stride, _1{}));
  Tensor gO = local_tile(mO(_, bidh, _),
                         Shape<Int<kBlockM>, Int<kHeadDim>>{},
                         make_coord(m_block, 0));
  Tensor gLSE =
      gemma4_get_lse_tile<ElementAccum, Params, kBlockM>(params, bidb, bidh, m_block);

  typename KernelTraits::GmemTiledCopyO gmem_tiled_copy_O;
  auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
  Tensor tOsO = gmem_thr_copy_O.partition_S(sO);
  Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
  __syncthreads();

  Tensor tOrO = make_tensor<Element>(shape(tOgO));
  cute::copy(gmem_tiled_copy_O, tOsO, tOrO);

  Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});
  Tensor taccOcO = thr_mma.partition_C(caccO);
  Tensor taccOcO_row = logical_divide(taccOcO, Shape<_2>{})(make_coord(0, _), _, 0);
  if (get<1>(taccOcO_row(0)) == 0) {
#pragma unroll
    for (int mi = 0; mi < size(lse); ++mi) {
      const int row = get<0>(taccOcO_row(mi));
      if (row < binfo.actual_seqlen_q - m_block * kBlockM) {
        gLSE(row) = lse(mi);
      }
    }
  }

  Tensor cO = make_identity_tensor(make_shape(size<0>(sO), size<1>(sO)));
  Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
  Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
  gemma4_fa_copy</*IsEvenMN=*/false, /*IsEvenK=*/true,
                 /*ClearOobMN=*/false, /*ClearOobK=*/false>(
      gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO,
      binfo.actual_seqlen_q - m_block * kBlockM);
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#define GEMMA4_FA_KERNEL_PARAM_MODIFIER __grid_constant__
#else
#define GEMMA4_FA_KERNEL_PARAM_MODIFIER
#endif

template <typename KernelTraits, bool IsLocal>
__global__ void gemma4_flash_fwd_bf16_kernel(
    GEMMA4_FA_KERNEL_PARAM_MODIFIER const Gemma4FlashFwdParams params) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  const int m_block = blockIdx.x;
  const int bidb = blockIdx.y;
  const int bidh = blockIdx.z;
  gemma4_compute_attn_1rowblock<KernelTraits, Gemma4FlashFwdParams, IsLocal>(
      params, bidb, bidh, m_block);
#else
  if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
    printf("FATAL: Gemma 4 sliding FlashAttention requires sm80 or newer.\\n");
  }
#endif
}

Gemma4FlashFwdParams make_params(
    __nv_bfloat16 *__restrict__ d_out,
    float *__restrict__ d_softmax_lse,
    const __nv_bfloat16 *__restrict__ d_q,
    const __nv_bfloat16 *__restrict__ d_k,
    const __nv_bfloat16 *__restrict__ d_v,
    int batch_size,
    int seqlen_q,
    int seqlen_k,
    int q_heads,
    int kv_heads,
    int head_dim,
    int window_left,
    float softmax_scale) {
  Gemma4FlashFwdParams params{};
  params.q_ptr = const_cast<__nv_bfloat16 *>(d_q);
  params.k_ptr = const_cast<__nv_bfloat16 *>(d_k);
  params.v_ptr = const_cast<__nv_bfloat16 *>(d_v);
  params.o_ptr = d_out;
  params.softmax_lse_ptr = d_softmax_lse;

  params.b = batch_size;
  params.h = q_heads;
  params.h_k = kv_heads;
  params.h_h_k_ratio = q_heads / kv_heads;
  params.seqlen_q = seqlen_q;
  params.seqlen_k = seqlen_k;
  params.seqlen_q_rounded = round_up(seqlen_q, 128);
  params.seqlen_k_rounded = round_up(seqlen_k, 128);
  params.d = head_dim;
  params.total_q = batch_size * seqlen_q;

  params.q_head_stride = head_dim;
  params.k_head_stride = head_dim;
  params.v_head_stride = head_dim;
  params.o_head_stride = head_dim;

  params.q_row_stride = q_heads * head_dim;
  params.k_row_stride = kv_heads * head_dim;
  params.v_row_stride = kv_heads * head_dim;
  params.o_row_stride = q_heads * head_dim;

  params.q_batch_stride = int64_t(seqlen_q) * params.q_row_stride;
  params.k_batch_stride = int64_t(seqlen_k) * params.k_row_stride;
  params.v_batch_stride = int64_t(seqlen_k) * params.v_row_stride;
  params.o_batch_stride = int64_t(seqlen_q) * params.o_row_stride;

  params.scale_softmax = softmax_scale;
  params.scale_softmax_log2 = softmax_scale * float(M_LOG2E);
  params.window_size_left = window_left;
  return params;
}

template <typename KernelTraits, bool IsLocal>
cudaError_t set_kernel_smem() {
  auto kernel = &gemma4_flash_fwd_bf16_kernel<KernelTraits, IsLocal>;
  if (KernelTraits::kSmemSize < 48 * 1024) return cudaSuccess;
  return cudaFuncSetAttribute(kernel,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              KernelTraits::kSmemSize);
}

template <typename KernelTraits, bool IsLocal>
cudaError_t launch_attention(Gemma4FlashFwdParams &params, cudaStream_t stream) {
  cudaError_t status = set_kernel_smem<KernelTraits, IsLocal>();
  if (status != cudaSuccess) return status;
  const dim3 grid_dim(cute::ceil_div(params.seqlen_q, KernelTraits::kBlockM),
                      params.b,
                      params.h);
  constexpr dim3 block_dim(KernelTraits::kNThreads);
  gemma4_flash_fwd_bf16_kernel<KernelTraits, IsLocal>
      <<<grid_dim, block_dim, KernelTraits::kSmemSize, stream>>>(params);
  return cudaGetLastError();
}

cudaError_t launch_sliding(Gemma4FlashFwdParams &params, cudaStream_t stream) {
  return launch_attention<Gemma4SlidingFa2KernelTraits, true>(params, stream);
}

cudaError_t launch_global(Gemma4FlashFwdParams &params, cudaStream_t stream) {
  return launch_attention<Gemma4GlobalFa2KernelTraits, false>(params, stream);
}

template <typename KernelTraits, bool IsLocal>
cudaError_t selected_kernel_attributes(long long *out, int len) {
  if (out == nullptr || len < 16) return cudaErrorInvalidValue;
  cudaError_t status = set_kernel_smem<KernelTraits, IsLocal>();
  if (status != cudaSuccess) return status;
  auto kernel = &gemma4_flash_fwd_bf16_kernel<KernelTraits, IsLocal>;
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
  out[10] = attr.clusterDimMustBeSet;
  out[11] = attr.requiredClusterWidth;
  out[12] = attr.requiredClusterHeight;
  out[13] = attr.requiredClusterDepth;
  out[14] = attr.clusterSchedulingPolicyPreference;
  out[15] = attr.nonPortableClusterSizeAllowed;
  return cudaSuccess;
}

}  // namespace gemma4_flash_attention

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
  if (d_out == nullptr || d_softmax_lse == nullptr || d_q == nullptr ||
      d_k == nullptr || d_v == nullptr) {
    return cudaErrorInvalidValue;
  }
  if (batch_size <= 0 || seqlen_q <= 0 || seqlen_k <= 0) {
    return cudaErrorInvalidValue;
  }
  if (window_left < 0) {
    return cudaErrorInvalidValue;
  }

  gemma4_flash_attention::Gemma4FlashFwdParams params =
      gemma4_flash_attention::make_params(d_out, d_softmax_lse, d_q, d_k, d_v,
                                          batch_size, seqlen_q, seqlen_k,
                                          GEMMA4_NUM_QUERY_HEADS,
                                          GEMMA4_SLIDING_KV_HEADS,
                                          GEMMA4_SLIDING_HEAD_DIM,
                                          window_left, softmax_scale);
  return gemma4_flash_attention::launch_sliding(params, stream);
}

extern "C" size_t gemma4_flash_attention_sliding_smem_bytes() {
  return gemma4_flash_attention::Gemma4SlidingFa2KernelTraits::kSmemSize;
}

extern "C" int gemma4_flash_attention_sliding_threads_per_block() {
  return gemma4_flash_attention::Gemma4SlidingFa2KernelTraits::kNThreads;
}

extern "C" cudaError_t gemma4_flash_attention_sliding_kernel_attributes(
    long long *out,
    int len) {
  return gemma4_flash_attention::selected_kernel_attributes<
      gemma4_flash_attention::Gemma4SlidingFa2KernelTraits, true>(out, len);
}

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
  if (d_out == nullptr || d_softmax_lse == nullptr || d_q == nullptr ||
      d_k == nullptr || d_v == nullptr) {
    return cudaErrorInvalidValue;
  }
  if (batch_size <= 0 || seqlen_q <= 0 || seqlen_k <= 0) {
    return cudaErrorInvalidValue;
  }

  gemma4_flash_attention::Gemma4FlashFwdParams params =
      gemma4_flash_attention::make_params(d_out, d_softmax_lse, d_q, d_k, d_v,
                                          batch_size, seqlen_q, seqlen_k,
                                          GEMMA4_NUM_QUERY_HEADS,
                                          GEMMA4_GLOBAL_KV_HEADS,
                                          GEMMA4_GLOBAL_HEAD_DIM,
                                          seqlen_k, softmax_scale);
  return gemma4_flash_attention::launch_global(params, stream);
}

extern "C" size_t gemma4_flash_attention_global_smem_bytes() {
  return gemma4_flash_attention::Gemma4GlobalFa2KernelTraits::kSmemSize;
}

extern "C" int gemma4_flash_attention_global_threads_per_block() {
  return gemma4_flash_attention::Gemma4GlobalFa2KernelTraits::kNThreads;
}

extern "C" cudaError_t gemma4_flash_attention_global_kernel_attributes(
    long long *out,
    int len) {
  return gemma4_flash_attention::selected_kernel_attributes<
      gemma4_flash_attention::Gemma4GlobalFa2KernelTraits, false>(out, len);
}
