/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#ifndef FLASHINFER_QUANT_LIGHTNING_INDEXER_CONFIG_HPP
#define FLASHINFER_QUANT_LIGHTNING_INDEXER_CONFIG_HPP

#include "arch35/quant_lightning_indexer_source.h"
#include "arch35/quant_lightning_indexer_task.h"
#include "quant_lightning_indexer_tiling_data.h"
#include "sparse_indexer/block/load/paged_key.hpp"
#include "sparse_indexer/block/mask/runtime.hpp"
#include "sparse_indexer/block/mma/runtime.hpp"
#include "sparse_indexer/block/mma/split_m.hpp"
#include "sparse_indexer/block/reduce/inline_row_split.hpp"
#include "sparse_indexer/block/reduce/quant_head.hpp"
#include "sparse_indexer/block/reduce/runtime.hpp"
#include "sparse_indexer/block/schedule/static_gsa1.hpp"
#include "sparse_indexer/block/selection/prefix_select.hpp"
#include "sparse_indexer/block/selection/runtime.hpp"
#include "sparse_indexer/block/selection/topk_vf.hpp"
#include "sparse_indexer/foundation/resource/score_buffer.hpp"
#include "sparse_indexer/foundation/contracts/common.hpp"
#include "sparse_indexer/foundation/contracts/policy.hpp"
#include "sparse_indexer/foundation/resource/current_policy.hpp"
#include "sparse_indexer/foundation/sync/subblock_barrier.hpp"

namespace SparseIndexer::QuantLightningIndexer::Config {

using QkDtype = fp8_e4m3fn_t;
using KeyLoad = SparseIndexer::Common::Block::PagedKeyLoadPolicy<QkDtype>;
using Sync = SparseIndexer::Common::SubblockPairBlockBarrierSyncPolicy;
using Resource = SparseIndexer::Common::CurrentResourcePolicy<Catlass::Arch::Ascend950>;

using Schedule = SparseIndexer::Common::Block::StaticGSA1SchedulePolicy<
    QuantLightningIndexerKernelDetail::QuantLightningIndexerRequestSource,
    QuantLightningIndexerKernelDetail::QuantLightningIndexerTaskFactory>;

using Mmad = SparseIndexer::Common::Block::SplitMMmadPolicy<QkDtype, KeyLoad>;
using QuantHead = SparseIndexer::Common::Block::BlockReduceQuantHead<
    SparseIndexer::Common::Block::LiV2MaskPolicy<float>, true>;
using Reduce = SparseIndexer::Common::Block::InlineRowSplitReducePolicy<QuantHead>;
using Selection = SparseIndexer::Common::Block::InlinePrefixLiTopKVFPolicy<
    SparseIndexer::Common::Block::BlockTopKVector<float, int32_t, true>>;

using ScheduleParams = typename Schedule::Params;
using MmaParams = SparseIndexer::Common::Block::MmaRuntimeParams<typename Mmad::Block>;
using ReduceParams = SparseIndexer::Common::Block::ReduceRuntimeParams<QuantHead>;
using SelectionParams = SparseIndexer::Common::Block::SelectionRuntimeParams;

struct CurrentPolicy : SparseIndexer::Common::PolicyBundle<
    Schedule, KeyLoad, Mmad, Reduce, Selection, Sync, Resource> {
    using ScheduleRuntimeParams = ScheduleParams;
    using MmaRuntimeParams = MmaParams;
    using ReduceRuntimeParams = ReduceParams;
    using SelectionRuntimeParams = SelectionParams;

    CATLASS_DEVICE static SparseIndexer::Common::ScoreBufferGeometry ScoreGeometry(
        __gm__ QuantLightningIndexerTilingData const* tiling)
    {
        return {
            SparseIndexer::Common::CeilDiv(tiling->s1Tile, 2U),
            tiling->n1,
            SparseIndexer::Common::AlignUp(tiling->s2Tile, 8U)};
    }

    CATLASS_DEVICE static uint32_t ScoreElements(__gm__ QuantLightningIndexerTilingData const* tiling)
    {
        return SparseIndexer::Common::ScoreElementsPerAiv(ScoreGeometry(tiling));
    }

    CATLASS_DEVICE static uint32_t ScoreBuffers(__gm__ QuantLightningIndexerTilingData const* tiling)
    {
        return SparseIndexer::Common::ScoreBufferCount(
            ScoreElements(tiling), Catlass::Arch::Ascend950::UB_SIZE);
    }

    CATLASS_DEVICE __attribute__((always_inline)) static ScheduleParams MakeScheduleParams(
        __gm__ QuantLightningIndexerTilingData const* tiling, GM_ADDR cuSeqlensQ, GM_ADDR sequsedK,
        GM_ADDR cmpResidualK)
    {
        return Schedule::MakeParams(tiling, cuSeqlensQ, sequsedK, cmpResidualK);
    }

    CATLASS_DEVICE __attribute__((always_inline)) static MmaParams MakeMmaParams(
        __gm__ QuantLightningIndexerTilingData const* tiling)
    {
        MmaParams runtime;
        runtime.block.keyLoader.d = tiling->d;
        runtime.block.keyLoader.blockSize = tiling->blockSize;
        runtime.block.keyLoader.maxBlocksPerBatch = tiling->maxBlocksPerBatch;
        runtime.block.keyLoader.numPages = tiling->numPages;
        runtime.block.keyLoader.keyStride0 = tiling->keyStride0;
        runtime.block.mSplitAlignment = 2U * tiling->n1;
        runtime.problemShape = Catlass::GemmCoord(
            tiling->tq * tiling->n1, tiling->maxSeqLenK, tiling->d);
        runtime.tileShape = Catlass::GemmCoord(
            SparseIndexer::Common::AlignUp(tiling->s1Tile * tiling->n1, 16U), tiling->s2Tile, tiling->d);
        runtime.heads = tiling->n1;
        runtime.scoreBuffers = ScoreBuffers(tiling);
        runtime.scoreElemsPerAiv = ScoreElements(tiling);
        runtime.mask = SparseIndexer::Common::Block::MaskRuntimeParams{
            tiling->maskMode, static_cast<uint32_t>(tiling->cmpRatio), tiling->s2Tile};
        return runtime;
    }

    CATLASS_DEVICE __attribute__((always_inline)) static ReduceParams MakeReduceParams(
        __gm__ QuantLightningIndexerTilingData const* tiling)
    {
        ReduceParams runtime;
        runtime.block.n1 = tiling->n1;
        runtime.block.s2 = tiling->maxSeqLenK;
        runtime.block.s1Tile = tiling->s1Tile;
        runtime.block.s2Tile = tiling->s2Tile;
        runtime.block.maskMode = tiling->maskMode;
        runtime.block.cmpRatio = tiling->cmpRatio;
        runtime.block.blockSize = tiling->blockSize;
        runtime.block.maxBlocksPerBatch = tiling->maxBlocksPerBatch;
        runtime.block.numPages = tiling->numPages;
        runtime.block.kScaleStride0 = tiling->kScaleStride0;
        runtime.block.scoreBuffers = ScoreBuffers(tiling);
        runtime.scoreBuffers = runtime.block.scoreBuffers;
        runtime.mask = SparseIndexer::Common::Block::MaskRuntimeParams{
            tiling->maskMode, static_cast<uint32_t>(tiling->cmpRatio), tiling->s2Tile};
        return runtime;
    }

    CATLASS_DEVICE __attribute__((always_inline)) static SelectionParams MakeSelectionParams(
        __gm__ QuantLightningIndexerTilingData const* tiling)
    {
        return SelectionParams{tiling->topk, tiling->maxSeqLenK, tiling->topkS2Tile, false};
    }
};

} // namespace SparseIndexer::QuantLightningIndexer::Config

#endif // FLASHINFER_QUANT_LIGHTNING_INDEXER_CONFIG_HPP
