/***************************************************************************************************
 * Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/
/*! \file
    \brief No-op synchronization event hooks for the trimmed Gemma forward-only build.
*/

#pragma once

#include "cutlass/detail/helper_macros.hpp"

namespace cutlass {
namespace arch {

template <typename... Args>
CUTLASS_HOST_DEVICE void synclog_emit_named_barrier_arrive_and_wait(Args&&...) {}
template <typename... Args>
CUTLASS_HOST_DEVICE void synclog_emit_named_barrier_arrive(Args&&...) {}
template <typename... Args>
CUTLASS_HOST_DEVICE void synclog_emit_cluster_barrier_init(Args&&...) {}
template <typename... Args>
CUTLASS_HOST_DEVICE void synclog_emit_cluster_barrier_wait(Args&&...) {}
template <typename... Args>
CUTLASS_HOST_DEVICE void synclog_emit_cluster_barrier_test_wait(Args&&...) {}
template <typename... Args>
CUTLASS_HOST_DEVICE void synclog_emit_cluster_barrier_try_wait(Args&&...) {}
template <typename... Args>
CUTLASS_HOST_DEVICE void synclog_emit_cluster_barrier_arrive_cluster(Args&&...) {}
template <typename... Args>
CUTLASS_HOST_DEVICE void synclog_emit_cluster_barrier_arrive(Args&&...) {}
template <typename... Args>
CUTLASS_HOST_DEVICE void synclog_emit_cluster_transaction_barrier_arrive_and_expect_tx(Args&&...) {}
template <typename... Args>
CUTLASS_HOST_DEVICE void synclog_emit_cluster_transaction_barrier_expect_transaction(Args&&...) {}
template <typename... Args>
CUTLASS_HOST_DEVICE void synclog_emit_cluster_transaction_barrier_complete_transaction(Args&&...) {}
template <typename... Args>
CUTLASS_HOST_DEVICE void synclog_emit_fence_barrier_init(Args&&...) {}
template <typename... Args>
CUTLASS_HOST_DEVICE void synclog_emit_fence_view_async_shared(Args&&...) {}
template <typename... Args>
CUTLASS_HOST_DEVICE void synclog_emit_cpasync_barrier_arrive(Args&&...) {}
template <typename... Args>
CUTLASS_HOST_DEVICE void synclog_emit_cp_async(Args&&...) {}
template <typename... Args>
CUTLASS_HOST_DEVICE void synclog_emit_cp_async_zfill(Args&&...) {}
template <typename... Args>
CUTLASS_HOST_DEVICE void synclog_emit_cp_async_nan(Args&&...) {}
template <typename... Args>
CUTLASS_HOST_DEVICE void synclog_emit_cp_async_fence(Args&&...) {}
template <typename... Args>
CUTLASS_HOST_DEVICE void synclog_emit_cp_async_wait(Args&&...) {}
template <typename... Args>
CUTLASS_HOST_DEVICE void synclog_emit_cp_async_wait_all(Args&&...) {}
template <typename... Args>
CUTLASS_HOST_DEVICE void synclog_emit_tma_load(Args&&...) {}
template <typename... Args>
CUTLASS_HOST_DEVICE void synclog_emit_tma_store(Args&&...) {}
template <typename... Args>
CUTLASS_HOST_DEVICE void synclog_emit_tma_store_arrive(Args&&...) {}
template <typename... Args>
CUTLASS_HOST_DEVICE void synclog_emit_tma_store_wait(Args&&...) {}
template <typename... Args>
CUTLASS_HOST_DEVICE void synclog_emit_warpgroup_arrive(Args&&...) {}
template <typename... Args>
CUTLASS_HOST_DEVICE void synclog_emit_warpgroup_wait(Args&&...) {}
template <typename... Args>
CUTLASS_HOST_DEVICE void synclog_emit_warpgroup_commit_batch(Args&&...) {}
template <typename... Args>
CUTLASS_HOST_DEVICE void synclog_emit_wgmma_reg_smem(Args&&...) {}
template <typename... Args>
CUTLASS_HOST_DEVICE void synclog_emit_wgmma_smem_smem(Args&&...) {}

}  // namespace arch
}  // namespace cutlass
