/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLASHINFER_SPARSE_INDEXER_BLOCK_REDUCE_DIRECT_WORKSPACE_STORE_HPP
#define FLASHINFER_SPARSE_INDEXER_BLOCK_REDUCE_DIRECT_WORKSPACE_STORE_HPP

#include <cstdint>
#include <type_traits>

#include "catlass/catlass.hpp"
#include "kernel_operator.h"

namespace SparseIndexer::Common::Block {

/// Copy a UB tile to GM. A different destination type supplies its own store.
template <class DtypeDst_, class DtypeScore_>
struct DirectWorkspaceStore {
    static_assert(std::is_same_v<DtypeDst_, DtypeScore_>, "DirectWorkspaceStore requires matching element types");

    using DtypeDst = DtypeDst_;
    using DtypeScore = DtypeScore_;

    /// Write rows of a packed UB tile. srcRowStride / dstRowStride are element counts.
    CATLASS_DEVICE static void StoreTile(
        AscendC::GlobalTensor<DtypeDst> gmDst, AscendC::LocalTensor<DtypeScore>& ubTile, uint32_t rows, uint32_t cols,
        uint32_t srcRowStride, uint32_t dstRowStride, uint32_t eventId = EVENT_ID0)
    {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventId);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventId);
        uint32_t copyBytes = cols * sizeof(DtypeDst);
        uint32_t srcRowBytes = srcRowStride * sizeof(DtypeDst);
        uint32_t copyAligned = (copyBytes + Catlass::BYTE_PER_BLK - 1) / Catlass::BYTE_PER_BLK * Catlass::BYTE_PER_BLK;
        uint32_t srcStrideBlocks = (srcRowBytes - copyAligned) / Catlass::BYTE_PER_BLK;
        uint32_t dstStrideBytes = (dstRowStride - cols) * sizeof(DtypeDst);
        AscendC::DataCopyExtParams params(rows, copyBytes, srcStrideBlocks, dstStrideBytes, 0);
        AscendC::DataCopyPad(gmDst, ubTile, params);
        // The caller defers this event until the same ping-pong buffer is
        // reused (or until TopK takes over UB).
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventId);
    }
};

} // namespace SparseIndexer::Common::Block

#endif // FLASHINFER_SPARSE_INDEXER_BLOCK_REDUCE_DIRECT_WORKSPACE_STORE_HPP
