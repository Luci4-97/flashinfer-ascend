// Migrated from CATLASS example 99_ascend950_general_indexer.
// See LICENSES/CANN-Open-Software-License-2.0.txt at the repository root.
#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif
#include "kernel_operator.h"
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#include "arch35/general_indexer.h"
#else
#error "general_indexer requires arch35 (__CCE_AICORE__ == 310)"
#endif

extern "C" __global__ __aicore__ void fi_general_indexer(
    GM_ADDR query, GM_ADDR key, GM_ADDR weights, GM_ADDR block_table,
    GM_ADDR cu_seqlens_q, GM_ADDR cu_seqlens_k, GM_ADDR seqused_q,
    GM_ADDR seqused_k, GM_ADDR cmp_residual_k, GM_ADDR output_idx_offset,
    GM_ADDR zero_page, GM_ADDR indices, GM_ADDR values, GM_ADDR scores,
    GM_ADDR workspace, GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    REGISTER_TILING_DEFAULT(GeneralIndexerTilingData);
    AscendC::SetSysWorkspace(workspace);

    using InputType = DTYPE_QUERY;
    using WeightType = DTYPE_WEIGHTS;
    using Contiguous = GeneralIndexerKernelDetail::ContiguousKeyLoader<InputType>;
    using Paged = GeneralIndexerKernelDetail::PagedKeyLoader<InputType>;

    if (TILING_KEY_IS(0)) {
        GeneralIndexerKernelDetail::RunKernel<InputType, WeightType, Contiguous>(
            query, key, weights, indices, values, scores, block_table, zero_page,
            output_idx_offset, tiling);
    } else if (TILING_KEY_IS(1)) {
        GeneralIndexerKernelDetail::RunKernel<InputType, WeightType, Paged>(
            query, key, weights, indices, values, scores, block_table, zero_page,
            output_idx_offset, tiling);
    }
    (void)cu_seqlens_q;
    (void)cu_seqlens_k;
    (void)seqused_q;
    (void)seqused_k;
    (void)cmp_residual_k;
}
