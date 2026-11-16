/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLASHINFER_GENERAL_INDEXER_ARCH35_H
#define FLASHINFER_GENERAL_INDEXER_ARCH35_H

/**
 * Strategy axes (template = documentation):
 * - KeyLoader: ContiguousKeyLoader (dense TND/BSND) | PagedKeyLoader (PA_BBND + block_table)
 * - Mask / seqused / cmp: tiling attrs → BlockReduceHead::ApplyMaskToUb (mask_mode 0|3)
 * - Output: BlockTopK indices (+ in-kernel sparse values / debug return_score)
 */

#include "kernel_operator.h"

#include "general_indexer_kernel.h"
#include "general_indexer_source.h"
#include "block/block_scheduler.hpp"
#include "block/block_mmad.hpp"
#include "block/block_reduce_head.hpp"
#include "block/block_topk.hpp"
#include "block/key_loader.hpp"

namespace GeneralIndexerKernelDetail {

using RequestSource = GeneralIndexerRequestSource;
using KeyCoord = SparseIndexer::Common::Block::KeyCoord;
using BlockScheduler = SparseIndexer::Common::Block::BlockScheduler<
    RequestSource, SparseIndexer::Common::Block::RequestBlockDiagonalPolicy>;

template <class InputType>
using ContiguousKeyLoader = SparseIndexer::Common::Block::ContiguousKeyLoader<
    Catlass::Arch::Ascend950, Catlass::Gemm::GemmType<InputType, Catlass::layout::ColumnMajor>>;

template <class InputType>
using PagedKeyLoader = SparseIndexer::Common::Block::PagedKeyLoader<
    Catlass::Arch::Ascend950, Catlass::Gemm::GemmType<InputType, Catlass::layout::ColumnMajor>>;

template <class InputType, class KeyLoader>
using BlockMmad = SparseIndexer::Common::Block::BlockMmad<InputType, InputType, KeyLoader>;

template <class WeightType, class MaskPolicy>
using BlockReduceHead = SparseIndexer::Common::Block::BlockReduceHead<float, WeightType, float, MaskPolicy>;

using BlockTopK = SparseIndexer::Common::Block::BlockTopK<float, int32_t>;

template <class InputType, class WeightType, class KeyLoader, class MaskPolicy>
using Kernel = GeneralIndexerKernel<
    BlockScheduler,
    BlockMmad<InputType, KeyLoader>,
    BlockReduceHead<WeightType, MaskPolicy>,
    BlockTopK>;

template <typename InputType, typename WeightType, class KeyLoader, class MaskPolicy>
CATLASS_DEVICE void RunKernel(
    GM_ADDR query, GM_ADDR key, GM_ADDR weights, GM_ADDR indices, GM_ADDR values,
    GM_ADDR scores, GM_ADDR block_table, GM_ADDR zero_page, GM_ADDR output_idx_offset,
    GM_ADDR cu_seqlens_q, GM_ADDR cu_seqlens_k, GM_ADDR seqused_q, GM_ADDR seqused_k,
    GM_ADDR cmp_residual_k, GM_ADDR tiling)
{
    __gm__ GeneralIndexerTilingData const* tilingData = reinterpret_cast<__gm__ GeneralIndexerTilingData*>(tiling);
    AscendC::GlobalTensor<InputType> gmQ;
    AscendC::GlobalTensor<InputType> gmK;
    AscendC::GlobalTensor<WeightType> gmW;
    AscendC::GlobalTensor<int32_t> gmOut;
    AscendC::GlobalTensor<float> gmScores;
    AscendC::GlobalTensor<float> gmValues;
    AscendC::GlobalTensor<int32_t> gmBlockTable;
    AscendC::GlobalTensor<InputType> gmZeroPage;
    AscendC::GlobalTensor<int32_t> gmOutputIdxOffset;
    gmQ.SetGlobalBuffer(reinterpret_cast<__gm__ InputType*>(query));
    gmK.SetGlobalBuffer(reinterpret_cast<__gm__ InputType*>(key));
    gmW.SetGlobalBuffer(reinterpret_cast<__gm__ WeightType*>(weights));
    gmOut.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(indices));
    gmValues.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(values));
    gmScores.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(scores));
    if (block_table != nullptr) {
        gmBlockTable.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(block_table));
    }
    if (zero_page != nullptr) {
        gmZeroPage.SetGlobalBuffer(reinterpret_cast<__gm__ InputType*>(zero_page));
    }
    if (output_idx_offset != nullptr) {
        gmOutputIdxOffset.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(output_idx_offset));
    }

    Kernel<InputType, WeightType, KeyLoader, MaskPolicy> kernel;
    kernel(gmQ, gmK, gmW, gmOut, gmValues, gmScores, tilingData, cu_seqlens_q, cu_seqlens_k,
           seqused_q, seqused_k, cmp_residual_k, gmBlockTable, gmZeroPage, gmOutputIdxOffset);
}

template <typename InputType, typename WeightType, class KeyLoader>
CATLASS_DEVICE void RunKernelWithMaskPolicy(
    GM_ADDR query, GM_ADDR key, GM_ADDR weights, GM_ADDR indices, GM_ADDR values,
    GM_ADDR scores, GM_ADDR block_table, GM_ADDR zero_page, GM_ADDR output_idx_offset,
    GM_ADDR cu_seqlens_q, GM_ADDR cu_seqlens_k, GM_ADDR seqused_q, GM_ADDR seqused_k,
    GM_ADDR cmp_residual_k, bool needMask, GM_ADDR tiling)
{
    if (needMask) {
        RunKernel<InputType, WeightType, KeyLoader, SparseIndexer::Common::Block::LiV2MaskPolicy<float>>(
            query, key, weights, indices, values, scores, block_table, zero_page, output_idx_offset,
            cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, cmp_residual_k, tiling);
    } else {
        RunKernel<InputType, WeightType, KeyLoader, SparseIndexer::Common::Block::NoMaskPolicy<float>>(
            query, key, weights, indices, values, scores, block_table, zero_page, output_idx_offset,
            cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, cmp_residual_k, tiling);
    }
}

} // namespace GeneralIndexerKernelDetail

#endif // FLASHINFER_GENERAL_INDEXER_ARCH35_H
