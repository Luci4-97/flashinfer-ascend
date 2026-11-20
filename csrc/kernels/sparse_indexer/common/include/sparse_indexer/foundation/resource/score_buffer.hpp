/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#ifndef FLASHINFER_SPARSE_INDEXER_BLOCK_SUPPORT_SCORE_BUFFER_HPP
#define FLASHINFER_SPARSE_INDEXER_BLOCK_SUPPORT_SCORE_BUFFER_HPP

#include "catlass/arch/arch.hpp"
#include "catlass/detail/macros.hpp"
#include "sparse_indexer/foundation/contracts/common.hpp"

namespace SparseIndexer::Common::Block {

template <class Tiling>
CATLASS_DEVICE inline uint32_t ScoreElementsPerAiv(Tiling const* tiling)
{
    const uint32_t workRows = CeilDiv(tiling->s1Tile, 2U);
    const uint32_t colPerAiv = AlignUp(tiling->s2Tile, 8U);
    return workRows * tiling->n1 * colPerAiv;
}

template <class Tiling>
CATLASS_DEVICE inline uint32_t ScoreBufferCount(Tiling const* tiling)
{
    constexpr uint64_t kUbSize = Catlass::Arch::Ascend950::UB_SIZE;
    constexpr uint64_t kScratchReserve = 64U * 1024U;
    const uint64_t twoScores = 2ULL * ScoreElementsPerAiv(tiling) * sizeof(float);
    return twoScores + kScratchReserve <= kUbSize ? 2U : 1U;
}

} // namespace SparseIndexer::Common::Block

#endif // FLASHINFER_SPARSE_INDEXER_BLOCK_SUPPORT_SCORE_BUFFER_HPP
