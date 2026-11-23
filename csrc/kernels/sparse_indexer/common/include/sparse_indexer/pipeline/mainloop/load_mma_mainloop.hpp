/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#ifndef FLASHINFER_SPARSE_INDEXER_PIPELINE_LOAD_MMA_MAINLOOP_HPP
#define FLASHINFER_SPARSE_INDEXER_PIPELINE_LOAD_MMA_MAINLOOP_HPP

#include "catlass/arch/resource.hpp"
#include "catlass/detail/macros.hpp"
#include "sparse_indexer/foundation/primitive/mask.hpp"
#include "sparse_indexer/block/block_qk_mmad.hpp"

namespace SparseIndexer::Common::Pipeline {

/**
 * AIC mainloop: schedule gS1 work, prefetch paged K tiles, issue MMAD, and
 * publish score buffers to the vector epilogue. The block policy only supplies
 * the MMA implementation and its key-loader composition.
 */
template <class MmadPolicy_, class SchedulePolicy_, class SyncPolicy_>
struct LoadMmaMainloop {
    using MmadPolicy = MmadPolicy_;
    using SchedulePolicy = SchedulePolicy_;
    using SyncPolicy = SyncPolicy_;
    using Block = typename MmadPolicy::Block;
    using InputType = typename MmadPolicy::InputType;

    template <class Scheduler_>
    CATLASS_DEVICE static void Run(Catlass::Arch::Resource<Catlass::Arch::Ascend950>& resource,
        AscendC::LocalTensor<typename Block::DtypeC> const& ubC,
        AscendC::GlobalTensor<InputType> const& gmQ, AscendC::GlobalTensor<InputType> const& gmK,
        SparseIndexer::Common::Block::MmaRuntimeParams<Block> const& runtime,
        AscendC::GlobalTensor<int32_t> const& gmBlockTable,
        AscendC::GlobalTensor<InputType> const& gmZeroPage, Scheduler_& scheduler)
    {
        Block blockMmad(
            ubC, gmQ, gmK, runtime.problemShape, runtime.tileShape, resource, runtime.block,
            gmBlockTable, gmZeroPage);
        const bool doubleBuffered = runtime.scoreBuffers > 1U;
        const uint32_t scoreElems = runtime.scoreElemsPerAiv;
        uint32_t lastPingpong = 0;
        typename SchedulePolicy::Coord coord{};
        while (scheduler.Next(coord)) {
            const auto task = SchedulePolicy::MakeTask(coord, scheduler.source());
            const auto visibleCoord = SparseIndexer::Common::Block::VisibleCoord(
                runtime.mask, coord, task.activeQ, task.activeK, task.residual);
            const uint32_t s2Tiles = SchedulePolicy::CountS2Tiles(visibleCoord, scheduler.policy());
            blockMmad.BeginTask(coord.qStart * runtime.heads, coord.qRows * runtime.heads);
            for (uint32_t s2Idx = 0; s2Idx < s2Tiles; ++s2Idx) {
                const auto tile = SchedulePolicy::MakeS2Coord(visibleCoord, s2Idx, scheduler.policy());
                const SparseIndexer::Common::Block::KeyCoord keyCoord{
                    tile.kStart, tile.kStart - task.requestKStart, tile.requestId};
                if (s2Idx == 0U) {
                    blockMmad.PrefetchKey(0, keyCoord, tile.kRows);
                }
                if (s2Idx + 1U < s2Tiles) {
                    const auto nextTile = SchedulePolicy::MakeS2Coord(visibleCoord, s2Idx + 1U, scheduler.policy());
                    const SparseIndexer::Common::Block::KeyCoord nextKeyCoord{
                        nextTile.kStart, nextTile.kStart - task.requestKStart, nextTile.requestId};
                    blockMmad.PrefetchKey((s2Idx + 1U) % 3U, nextKeyCoord, nextTile.kRows);
                }
                const uint32_t keyBufferIdx = s2Idx % 3U;
                const uint32_t pingpong = doubleBuffered ? (s2Idx & 1U) : 0U;
                if (doubleBuffered) {
                    blockMmad.Compute(
                        tile.qStart * runtime.heads, tile.qRows * runtime.heads, keyBufferIdx, tile.kRows);
                    SyncPolicy::AicWaitCubeReady(pingpong);
                    blockMmad.SetOutputOffset(pingpong * scoreElems);
                    blockMmad.Store(tile.qRows * runtime.heads, tile.kRows);
                } else {
                    SyncPolicy::AicWaitCubeReady(0);
                    blockMmad.Compute(
                        tile.qStart * runtime.heads, tile.qRows * runtime.heads, keyBufferIdx, tile.kRows);
                    blockMmad.Store(tile.qRows * runtime.heads, tile.kRows);
                }
                SyncPolicy::AicArriveCubeDone(pingpong);
                lastPingpong = pingpong;
            }
        }
        SyncPolicy::AicFinalWait(lastPingpong);
    }
};

} // namespace SparseIndexer::Common::Pipeline

#endif // FLASHINFER_SPARSE_INDEXER_PIPELINE_LOAD_MMA_MAINLOOP_HPP
