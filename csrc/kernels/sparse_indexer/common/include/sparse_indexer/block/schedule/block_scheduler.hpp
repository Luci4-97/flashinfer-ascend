/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
 * an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
 * specific language governing permissions and limitations under the License.
 */

#ifndef FLASHINFER_SPARSE_INDEXER_COMMON_BLOCK_BLOCK_SCHEDULER_HPP
#define FLASHINFER_SPARSE_INDEXER_COMMON_BLOCK_BLOCK_SCHEDULER_HPP

#include <cstdint>

#include "catlass/detail/macros.hpp"
#include "sparse_indexer/foundation/contracts/problem.hpp"

namespace SparseIndexer::Common::Block {

using ScheduleCoord = SparseIndexer::Common::WorkItem;

namespace detail {

template <class T, class U>
CATLASS_DEVICE constexpr uint32_t CeilDiv(const T& value, const U& divisor)
{
    return divisor == 0 ? 0 : static_cast<uint32_t>((value + divisor - 1) / divisor);
}

} // namespace detail

/**
 * Request × qTile × kTile schedule. The request ranges come from a variant-specific
 * RequestRangeSource; this policy does not know any concrete metadata schema.
 */
class RequestBlockDiagonalPolicy {
public:
    struct Params {
        uint32_t s1Tile{1};
        uint32_t s2Tile{1};
        uint32_t topkRowsPerTask{1};
    };

    template <class RequestRangeSource>
    CATLASS_DEVICE static uint64_t CountMmad(RequestRangeSource& source, Params const& params)
    {
        uint64_t total = 0;
        for (uint32_t b = 0; b < source.Batch(); ++b) {
            const uint32_t qLen = source.QLen(b);
            const uint32_t kLen = source.KLen(b);
            if (qLen == 0 || kLen == 0) {
                continue;
            }
            const uint64_t qDim = detail::CeilDiv(qLen, params.s1Tile);
            const uint64_t kDim = detail::CeilDiv(kLen, params.s2Tile);
            total += qDim * kDim;
        }
        return total;
    }

    template <class RequestRangeSource>
    CATLASS_DEVICE static uint64_t CountTopK(RequestRangeSource& source, Params const& params)
    {
        uint64_t total = 0;
        const uint32_t rowsPerTask = params.topkRowsPerTask == 0 ? 1U : params.topkRowsPerTask;
        for (uint32_t b = 0; b < source.Batch(); ++b) {
            total += detail::CeilDiv(source.QLen(b), rowsPerTask);
        }
        return total;
    }

    template <class RequestRangeSource>
    CATLASS_DEVICE static bool FindMmad(
        RequestRangeSource& source, Params const& params, uint64_t tileId, ScheduleCoord& coord)
    {
        uint64_t accumulated = 0;
        for (uint32_t b = 0; b < source.Batch(); ++b) {
            const uint32_t requestQStart = source.QBegin(b);
            const uint32_t qLen = source.QLen(b);
            const uint32_t requestKStart = source.KBegin(b);
            const uint32_t kLen = source.KLen(b);
            if (qLen == 0 || kLen == 0) {
                continue;
            }
            const uint32_t qDim = detail::CeilDiv(qLen, params.s1Tile);
            const uint32_t kDim = detail::CeilDiv(kLen, params.s2Tile);
            const uint64_t tasks = static_cast<uint64_t>(qDim) * kDim;
            if (tileId >= accumulated + tasks) {
                accumulated += tasks;
                continue;
            }
            const uint64_t local = tileId - accumulated;
            const uint32_t qBlock = static_cast<uint32_t>(local / kDim);
            const uint32_t kBlock = static_cast<uint32_t>(local % kDim);
            coord.requestId = b;
            coord.localQBegin = qBlock * params.s1Tile;
            coord.qStart = requestQStart + coord.localQBegin;
            coord.qRows = (qBlock + 1U == qDim) ? (qLen - coord.localQBegin) : params.s1Tile;
            coord.kStart = requestKStart + kBlock * params.s2Tile;
            coord.kRows = (kBlock + 1U == kDim) ? (kLen - kBlock * params.s2Tile) : params.s2Tile;
            return true;
        }
        return false;
    }

    CATLASS_DEVICE static uint32_t CountS2Tiles(ScheduleCoord const&, Params const&)
    {
        return 1U;
    }

    CATLASS_DEVICE static ScheduleCoord MakeS2Coord(ScheduleCoord const& coord, uint32_t, Params const&)
    {
        return coord;
    }

    template <class RequestRangeSource>
    CATLASS_DEVICE static bool FindTopK(
        RequestRangeSource& source, Params const& params, uint64_t tileId, ScheduleCoord& coord)
    {
        uint64_t accumulated = 0;
        const uint32_t rowsPerTask = params.topkRowsPerTask == 0 ? 1U : params.topkRowsPerTask;
        for (uint32_t b = 0; b < source.Batch(); ++b) {
            const uint32_t requestQStart = source.QBegin(b);
            const uint32_t qLen = source.QLen(b);
            const uint32_t requestKStart = source.KBegin(b);
            const uint32_t kLen = source.KLen(b);
            const uint32_t qDim = detail::CeilDiv(qLen, rowsPerTask);
            if (tileId >= accumulated + qDim) {
                accumulated += qDim;
                continue;
            }
            const uint32_t qBlock = static_cast<uint32_t>(tileId - accumulated);
            coord.requestId = b;
            coord.localQBegin = qBlock * rowsPerTask;
            coord.qStart = requestQStart + coord.localQBegin;
            coord.qRows = (qBlock + 1U == qDim) ? (qLen - coord.localQBegin) : rowsPerTask;
            coord.kStart = requestKStart;
            coord.kRows = kLen;
            return true;
        }
        return false;
    }
};

/**
 * Common scheduler. New indexer variants provide their own RequestRangeSource; a variant with
 * different ordering can provide a new SchedulePolicy without changing this class.
 */
template <class RequestRangeSource_, class SchedulePolicy_>
class BlockScheduler {
public:
    using RequestRangeSource = RequestRangeSource_;
    using SchedulePolicy = SchedulePolicy_;
    using Coord = ScheduleCoord;

    struct Params {
        typename SchedulePolicy::Params policy{};
        RequestRangeSource source{};
    };

    CATLASS_DEVICE
    explicit BlockScheduler(Params const& params)
        : source_(params.source), policy_(params.policy)
    {
        totalMmadTasks_ = SchedulePolicy::CountMmad(source_, policy_);
        totalTopKTasks_ = SchedulePolicy::CountTopK(source_, policy_);
        nextMmadTask_ = block_idx;
        nextTopKTask_ = block_idx;
    }

    CATLASS_DEVICE
    bool Next(WorkItem& coord)
    {
        if (nextMmadTask_ >= totalMmadTasks_) {
            return false;
        }
        const uint64_t taskId = nextMmadTask_;
        nextMmadTask_ += block_num;
        return SchedulePolicy::FindMmad(source_, policy_, taskId, coord);
    }

    CATLASS_DEVICE
    RequestRangeSource const& source() const
    {
        return source_;
    }

    CATLASS_DEVICE
    typename SchedulePolicy::Params const& policy() const
    {
        return policy_;
    }

private:
    RequestRangeSource source_{};
    typename SchedulePolicy::Params policy_{};
    uint64_t totalMmadTasks_{0};
    uint64_t totalTopKTasks_{0};
    uint64_t nextMmadTask_{0};
    uint64_t nextTopKTask_{0};
};

} // namespace SparseIndexer::Common::Block

#endif // FLASHINFER_SPARSE_INDEXER_COMMON_BLOCK_BLOCK_SCHEDULER_HPP
