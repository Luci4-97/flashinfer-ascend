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

#ifndef FLASHINFER_GENERAL_INDEXER_BLOCK_BLOCK_SCHEDULER_HPP
#define FLASHINFER_GENERAL_INDEXER_BLOCK_BLOCK_SCHEDULER_HPP

#include <cstdint>
#include "catlass/detail/alignment.hpp"
#include "kernel_operator.h"

#include "sparse_indexer_align.hpp"

namespace GeneralIndexer::Block {

/**
 * Request-aware mmad/reduce/TopK scheduler.
 * cu_seqlens are embedded in tiling; ctor copies them into on-chip arrays.
 */
class BlockScheduler {
public:
    static constexpr uint32_t kMaxBatch = SparseIndexer::Common::kMaxBatch;

    struct Params {
        uint32_t s1{0};
        uint32_t s2{0};
        uint32_t s1Tile{0};
        uint32_t s2Tile{0};
        uint32_t batch{1};
        int32_t cuQ[kMaxBatch + 1]{0};
        int32_t cuK[kMaxBatch + 1]{0};
    };

    CATLASS_DEVICE
    explicit BlockScheduler(Params const& params)
        : s1(params.s1),
          s2(params.s2),
          s1Tile(params.s1Tile),
          s2Tile(params.s2Tile),
          batch(params.batch == 0 ? 1U : params.batch)
    {
        if (batch > kMaxBatch) {
            batch = 1;
        }
        for (uint32_t i = 0; i <= batch; ++i) {
            cuQ_[i] = params.cuQ[i];
            cuK_[i] = params.cuK[i];
        }
        if (batch == 1 && cuQ_[1] == 0 && cuK_[1] == 0) {
            cuQ_[0] = 0;
            cuQ_[1] = static_cast<int32_t>(s1);
            cuK_[0] = 0;
            cuK_[1] = static_cast<int32_t>(s2);
        }

        totalMmadTasks = 0;
        totalQTiles = 0;
        for (uint32_t b = 0; b < batch; ++b) {
            const uint32_t qLen = static_cast<uint32_t>(cuQ_[b + 1] - cuQ_[b]);
            const uint32_t kLen = static_cast<uint32_t>(cuK_[b + 1] - cuK_[b]);
            if (qLen == 0) {
                continue;
            }
            const uint32_t qDim = CeilDiv(qLen, s1Tile);
            totalQTiles += qDim;
            if (kLen == 0) {
                continue;
            }
            const uint32_t kDim = CeilDiv(kLen, s2Tile);
            totalMmadTasks += static_cast<uint64_t>(qDim) * static_cast<uint64_t>(kDim);
        }
        nextMmad = block_idx;
        nextQTile = block_idx;
    }

    /// Next mmad/reduce tile: request × qTile × kTile (block-diagonal only).
    CATLASS_DEVICE
    bool GetNextCoord(
        uint32_t& qStart, uint32_t& actualQ, uint32_t& kStart, uint32_t& actualK, uint32_t& batchIdx,
        uint32_t& localQStart)
    {
        if (nextMmad >= totalMmadTasks) {
            return false;
        }
        uint64_t tileId = nextMmad;
        nextMmad += block_num;
        uint64_t acc = 0;
        for (uint32_t b = 0; b < batch; ++b) {
            const uint32_t reqQStart = static_cast<uint32_t>(cuQ_[b]);
            const uint32_t qLen = static_cast<uint32_t>(cuQ_[b + 1] - cuQ_[b]);
            const uint32_t reqKStart = static_cast<uint32_t>(cuK_[b]);
            const uint32_t reqKLen = static_cast<uint32_t>(cuK_[b + 1] - cuK_[b]);
            if (qLen == 0 || reqKLen == 0) {
                continue;
            }
            const uint32_t qDim = CeilDiv(qLen, s1Tile);
            const uint32_t kDim = CeilDiv(reqKLen, s2Tile);
            const uint64_t tasks = static_cast<uint64_t>(qDim) * static_cast<uint64_t>(kDim);
            if (tileId < acc + tasks) {
                const uint64_t local = tileId - acc;
                const uint32_t qBlockIdx = static_cast<uint32_t>(local / kDim);
                const uint32_t kBlockIdx = static_cast<uint32_t>(local % kDim);
                batchIdx = b;
                localQStart = qBlockIdx * s1Tile;
                qStart = reqQStart + localQStart;
                actualQ = (qBlockIdx + 1U == qDim) ? (qLen - localQStart) : s1Tile;
                kStart = reqKStart + kBlockIdx * s2Tile;
                actualK = (kBlockIdx + 1U == kDim) ? (reqKLen - kBlockIdx * s2Tile) : s2Tile;
                return true;
            }
            acc += tasks;
        }
        return false;
    }

    /// Next TopK q-tile: scans only this request's K interval [kStart, kStart+kLen).
    CATLASS_DEVICE
    bool GetNextTopKCoord(uint32_t& qStart, uint32_t& actualQ, uint32_t& kStart, uint32_t& kLen)
    {
        if (nextQTile >= totalQTiles) {
            return false;
        }
        uint64_t tileId = nextQTile;
        nextQTile += block_num;
        uint64_t acc = 0;
        for (uint32_t b = 0; b < batch; ++b) {
            const uint32_t reqQStart = static_cast<uint32_t>(cuQ_[b]);
            const uint32_t qLen = static_cast<uint32_t>(cuQ_[b + 1] - cuQ_[b]);
            const uint32_t reqKStart = static_cast<uint32_t>(cuK_[b]);
            const uint32_t reqKLen = static_cast<uint32_t>(cuK_[b + 1] - cuK_[b]);
            const uint32_t qDim = qLen > 0 ? CeilDiv(qLen, s1Tile) : 0U;
            if (tileId < acc + qDim) {
                const uint32_t qBlockIdx = static_cast<uint32_t>(tileId - acc);
                const uint32_t localQStart = qBlockIdx * s1Tile;
                qStart = reqQStart + localQStart;
                actualQ = (qBlockIdx + 1U == qDim) ? (qLen - localQStart) : s1Tile;
                kStart = reqKStart;
                kLen = reqKLen;
                return true;
            }
            acc += qDim;
        }
        return false;
    }

private:
    uint32_t s1{0};
    uint32_t s2{0};
    uint32_t s1Tile{0};
    uint32_t s2Tile{0};
    uint32_t batch{1};
    int32_t cuQ_[kMaxBatch + 1]{0};
    int32_t cuK_[kMaxBatch + 1]{0};
    uint64_t totalMmadTasks{0};
    uint64_t totalQTiles{0};
    uint64_t nextMmad{0};
    uint64_t nextQTile{0};
};

} // namespace GeneralIndexer::Block

#endif // FLASHINFER_GENERAL_INDEXER_BLOCK_BLOCK_SCHEDULER_HPP
