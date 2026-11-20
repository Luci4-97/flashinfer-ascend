/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#ifndef FLASHINFER_SPARSE_INDEXER_BLOCK_SELECTION_PREFIX_SELECT_HPP
#define FLASHINFER_SPARSE_INDEXER_BLOCK_SELECTION_PREFIX_SELECT_HPP

#include "catlass/arch/resource.hpp"
#include "catlass/detail/macros.hpp"
#include "sparse_indexer/block/selection/runtime.hpp"

namespace SparseIndexer::Common::Block {

/**
 * Inline TopK used after one gS1 has finished all S2 work on the same AIV.
 * LocalRows disables the generic TopK block's subblock row split, so each AIV
 * consumes exactly the rows it owns.
 */
template <class VectorBlock_>
struct InlinePrefixLiTopKVFPolicy {
    using VectorBlock = VectorBlock_;

    CATLASS_DEVICE static void RunLocal(Catlass::Arch::Resource<Catlass::Arch::Ascend950>& resource,
        SelectionRuntimeParams const& selection, AscendC::GlobalTensor<float>& scores,
        AscendC::GlobalTensor<int32_t> const& outIndices, uint32_t s1Start, uint32_t rows,
        uint32_t kStart, uint32_t kLen, uint32_t localQStart, uint32_t qLen, int32_t residual)
    {
        if (rows == 0) {
            return;
        }
        AscendC::GlobalTensor<int32_t> dummyOffset;
        typename VectorBlock::Params params{};
        params.topK = selection.topK;
        params.s2Stride = selection.s2Stride;
        params.s2Tile = selection.s2Tile;
        params.hasOutputIdxOffset = selection.hasOutputIdxOffset;
        VectorBlock topk(resource, scores, outIndices, params, dummyOffset);
        topk(s1Start, rows, kStart, kLen, localQStart, qLen, kLen, residual);
    }
};

} // namespace SparseIndexer::Common::Block

#endif // FLASHINFER_SPARSE_INDEXER_BLOCK_SELECTION_PREFIX_SELECT_HPP
