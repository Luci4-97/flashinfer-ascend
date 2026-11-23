/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#ifndef FLASHINFER_SPARSE_INDEXER_BLOCK_BLOCK_SCHEDULE_HPP
#define FLASHINFER_SPARSE_INDEXER_BLOCK_BLOCK_SCHEDULE_HPP

#include <cstdint>

#include "catlass/detail/macros.hpp"
#include "sparse_indexer/foundation/contracts/problem.hpp"
#include "sparse_indexer/foundation/primitive/mask.hpp"

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
class BlockSchedule {
public:
    using RequestRangeSource = RequestRangeSource_;
    using SchedulePolicy = SchedulePolicy_;
    using Coord = ScheduleCoord;

    struct Params {
        typename SchedulePolicy::Params policy{};
        RequestRangeSource source{};
    };

    CATLASS_DEVICE
    explicit BlockSchedule(Params const& params)
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

/**
 * One work item per bN2/gS1 block; all S2 tiles execute inside that core.
 * The policy is parameterized by a variant request source and task factory.
 */
class StaticGSA1BlockPolicy : public RequestBlockDiagonalPolicy {
public:
    using Base = RequestBlockDiagonalPolicy;
    using Params = typename Base::Params;

    template <class RequestRangeSource>
    CATLASS_DEVICE static uint64_t CountMmad(RequestRangeSource& source, Params const& params)
    {
        uint64_t total = 0;
        for (uint32_t b = 0; b < source.Batch(); ++b) {
            const uint32_t qLen = source.QLen(b);
            if (qLen == 0 || source.KLen(b) == 0) {
                continue;
            }
            total += detail::CeilDiv(qLen, params.s1Tile);
        }
        return total;
    }

    template <class RequestRangeSource>
    CATLASS_DEVICE static bool FindMmad(
        RequestRangeSource& source, Params const& params, uint64_t tileId, WorkItem& coord)
    {
        uint64_t accumulated = 0;
        for (uint32_t b = 0; b < source.Batch(); ++b) {
            const uint32_t requestQStart = source.QBegin(b);
            const uint32_t qLen = source.QLen(b);
            const uint32_t kLen = source.KLen(b);
            if (qLen == 0 || kLen == 0) {
                continue;
            }
            const uint32_t qDim = detail::CeilDiv(qLen, params.s1Tile);
            if (tileId >= accumulated + qDim) {
                accumulated += qDim;
                continue;
            }
            const uint32_t qBlock = static_cast<uint32_t>(tileId - accumulated);
            coord.requestId = b;
            coord.localQBegin = qBlock * params.s1Tile;
            coord.qStart = requestQStart + coord.localQBegin;
            coord.qRows = (qBlock + 1U == qDim) ? (qLen - coord.localQBegin) : params.s1Tile;
            coord.kStart = source.KBegin(b);
            coord.kRows = kLen;
            return true;
        }
        return false;
    }

    CATLASS_DEVICE static uint32_t CountS2Tiles(WorkItem const& coord, Params const& params)
    {
        return detail::CeilDiv(coord.kRows, params.s2Tile);
    }

    CATLASS_DEVICE static WorkItem MakeS2Coord(WorkItem const& coord, uint32_t s2Idx, Params const& params)
    {
        auto tile = coord;
        tile.kStart = coord.kStart + s2Idx * params.s2Tile;
        tile.kRows = AscendC::Std::min(params.s2Tile, coord.kRows - s2Idx * params.s2Tile);
        return tile;
    }
};

template <class RequestSource_, class TaskFactory_>
struct StaticGSA1SchedulePolicy {
    using RequestSource = RequestSource_;
    using TaskFactory = TaskFactory_;
    using Scheduler = BlockSchedule<RequestSource, StaticGSA1BlockPolicy>;
    using Params = typename Scheduler::Params;
    using Coord = WorkItem;
    using Task = typename TaskFactory::Task;

    template <class Tiling>
    CATLASS_DEVICE static Params MakeParams(
        Tiling const* tiling, GM_ADDR cuSeqlensQ, GM_ADDR sequsedK, GM_ADDR cmpResidualK)
    {
        Params params{};
        params.policy.s1Tile = tiling->s1Tile;
        params.policy.s2Tile = tiling->s2Tile;
        params.policy.topkRowsPerTask = tiling->topkRowsPerTask;
        params.source.InitFromGm(tiling, cuSeqlensQ, sequsedK, cmpResidualK);
        return params;
    }

    CATLASS_DEVICE static uint32_t CountS2Tiles(Coord const& coord, StaticGSA1BlockPolicy::Params const& params)
    {
        return StaticGSA1BlockPolicy::CountS2Tiles(coord, params);
    }

    CATLASS_DEVICE static Coord MakeS2Coord(
        Coord const& coord, uint32_t s2Idx, StaticGSA1BlockPolicy::Params const& params)
    {
        return StaticGSA1BlockPolicy::MakeS2Coord(coord, s2Idx, params);
    }

    CATLASS_DEVICE static Task MakeTask(Coord const& coord, RequestSource const& source)
    {
        return TaskFactory::Make(coord, source);
    }
};


} // namespace SparseIndexer::Common::Block


#endif // FLASHINFER_SPARSE_INDEXER_BLOCK_BLOCK_SCHEDULE_HPP
