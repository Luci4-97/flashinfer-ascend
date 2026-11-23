/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#ifndef FLASHINFER_SPARSE_INDEXER_PIPELINE_REDUCE_SELECT_EPILOGUE_HPP
#define FLASHINFER_SPARSE_INDEXER_PIPELINE_REDUCE_SELECT_EPILOGUE_HPP

#include "catlass/arch/resource.hpp"
#include "catlass/detail/macros.hpp"
#include "sparse_indexer/block/mask/visible_coord.hpp"
#include "sparse_indexer/block/reduce/runtime.hpp"
#include "sparse_indexer/block/selection/runtime.hpp"

namespace SparseIndexer::Common::Pipeline {

/**
 * AIV epilogue: consume cube score buffers, reduce quantized QK heads, and run
 * TopK on the rows owned by this AIV before releasing the score buffer.
 */
template <class ReducePolicy_, class SelectionPolicy_, class SchedulePolicy_, class SyncPolicy_>
struct ReduceSelectEpilogue {
    using ReducePolicy = ReducePolicy_;
    using SelectionPolicy = SelectionPolicy_;
    using SchedulePolicy = SchedulePolicy_;
    using SyncPolicy = SyncPolicy_;
    using Block = typename ReducePolicy::Block;

    template <class Scheduler_>
    CATLASS_DEVICE static void Run(Catlass::Arch::Resource<Catlass::Arch::Ascend950>& resource,
        AscendC::LocalTensor<typename Block::DtypeScore>& ubC,
        AscendC::GlobalTensor<float> const& gmW, AscendC::GlobalTensor<float> const& gmQScale,
        AscendC::GlobalTensor<float> const& gmKScale, AscendC::GlobalTensor<int32_t> const& gmBlockTable,
        AscendC::GlobalTensor<float>& scores,
        SparseIndexer::Common::Block::ReduceRuntimeParams<Block> const& runtime,
        SparseIndexer::Common::Block::SelectionRuntimeParams const& selection, Scheduler_& scheduler,
        AscendC::GlobalTensor<int32_t> const& outIndices)
    {
        Block reducer(ubC, resource, gmW, gmQScale, gmKScale, gmBlockTable, scores, runtime.block);
        const bool doubleBuffered = runtime.block.scoreBuffers > 1U;
        SyncPolicy::AivArriveVecReady(0);
        if (doubleBuffered) {
            SyncPolicy::AivArriveVecReady(1);
        }
        typename SchedulePolicy::Coord coord{};
        while (scheduler.Next(coord)) {
            const auto task = SchedulePolicy::MakeTask(coord, scheduler.source());
            const auto visibleCoord = SparseIndexer::Common::Block::VisibleCoord(
                runtime.block.mask, coord, task.activeQ, task.activeK, task.residual);
            const uint32_t s2Tiles = SchedulePolicy::CountS2Tiles(visibleCoord, scheduler.policy());
            for (uint32_t s2Idx = 0; s2Idx < s2Tiles; ++s2Idx) {
                const auto tile = SchedulePolicy::MakeS2Coord(visibleCoord, s2Idx, scheduler.policy());
                const uint32_t pingpong = doubleBuffered ? (s2Idx & 1U) : 0U;
                SyncPolicy::AivWaitCubeDone(pingpong);
                reducer(
                    tile.qStart, tile.qRows, tile.kStart, tile.kRows,
                    tile.localQBegin, task.activeQ, task.activeK, task.requestKStart, task.residual,
                    task.maskActive, tile.requestId, pingpong);
                // The last tile also carries the score hand-off for this gS1 task.
                // Keep the cube/vector ready flag pending until the owning AIV has
                // finished TopK, otherwise the next gS1 task can overwrite UB.
                if (s2Idx + 1U < s2Tiles) {
                    SyncPolicy::AivArriveVecReady(pingpong);
                }
            }

            reducer.WaitAllOutputs();
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            const uint32_t sub = AscendC::GetSubBlockIdx();
            const uint32_t rowsPerAiv = SparseIndexer::Common::CeilDiv(task.coord.qRows, 2U);
            const uint32_t rowBegin = sub * rowsPerAiv;
            if (rowBegin < task.coord.qRows) {
                const uint32_t rows = AscendC::Std::min(rowsPerAiv, task.coord.qRows - rowBegin);
                SelectionPolicy::RunLocal(resource, selection, scores, outIndices,
                    task.coord.qStart + rowBegin, rows, task.requestKStart, task.activeK,
                    task.coord.localQBegin + rowBegin, task.activeQ, task.residual);
            }
            const uint32_t lastPingpong = doubleBuffered ? ((s2Tiles - 1U) & 1U) : 0U;
            SyncPolicy::AivArriveVecReady(lastPingpong);
        }
    }
};

} // namespace SparseIndexer::Common::Pipeline

#endif // FLASHINFER_SPARSE_INDEXER_PIPELINE_REDUCE_SELECT_EPILOGUE_HPP
