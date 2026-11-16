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

#ifndef FLASHINFER_GENERAL_INDEXER_KERNEL_HPP
#define FLASHINFER_GENERAL_INDEXER_KERNEL_HPP

#include "catlass/arch/resource.hpp"
#include "catlass/detail/alignment.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "kernel_operator.h"
#include "c_api/sync/sync.h"

#include "general_indexer_tiling_data.h"

/**
 * out = TopK(Sum(ReLU(Q @ K^T) * W, dim=1), dim=-1)
 * Q: [S1, N1, D]; (N1 <= 256)
 * K: [S2, N2, D]; (N2 = 1)
 * W: [S1, N1]
 * out: [S1, k]
 *
 * Multi-request TND: Scheduler issues request × qTile × kTile mmad/reduce;
 * AIV applies mask_mode/seqused onto workspace before TopK.
 */
template <class BlockScheduler_, class BlockMmad_, class BlockReduceHead_, class BlockTopK_>
class GeneralIndexerKernel {
    using BlockScheduler = BlockScheduler_;
    using BlockMmad = BlockMmad_;
    using BlockReduceHead = BlockReduceHead_;
    using BlockTopK = BlockTopK_;
    using DtypeQ = typename BlockMmad::DtypeA;
    using DtypeK = typename BlockMmad::DtypeB;
    using DtypeW = typename BlockReduceHead::DtypeW;
    using DtypeS = typename BlockTopK::DtypeS;
    using DtypeOut = typename BlockTopK::DtypeOut;
    using KeyLoader = typename BlockMmad::KeyLoader;
    using KeyLoaderParams = typename KeyLoader::Params;

    CATLASS_DEVICE static typename BlockScheduler::Params MakeSchedulerParams(
        __gm__ GeneralIndexerTilingData const* tiling)
    {
        typename BlockScheduler::Params params{};
        params.s1 = tiling->s1;
        params.s2 = tiling->s2;
        params.s1Tile = tiling->s1Tile;
        params.s2Tile = tiling->s2Tile;
        params.batch = tiling->batch == 0 ? 1U : tiling->batch;
        for (uint32_t i = 0; i <= BlockScheduler::kMaxBatch; ++i) {
            params.cuQ[i] = tiling->cuSeqlensQ[i];
            params.cuK[i] = tiling->cuSeqlensK[i];
        }
        return params;
    }

    CATLASS_DEVICE static KeyLoaderParams MakeKeyLoaderParams(
        __gm__ GeneralIndexerTilingData const* tiling)
    {
        KeyLoaderParams params{};
        if constexpr (KeyLoader::kPaged) {
            params.d = tiling->d;
            params.blockSize = tiling->blockSize;
            params.maxBlocksPerBatch = tiling->maxBlocksPerBatch;
            params.numPages = tiling->numPages;
            params.batch = tiling->batch == 0 ? 1U : tiling->batch;
            for (uint32_t i = 0; i <= KeyLoader::kMaxBatch; ++i) {
                params.cuK[i] = tiling->cuSeqlensK[i];
            }
        }
        return params;
    }

    CATLASS_DEVICE static typename BlockReduceHead::Params MakeReduceParams(
        __gm__ GeneralIndexerTilingData const* tiling)
    {
        typename BlockReduceHead::Params params{};
        params.n1 = tiling->n1;
        params.s2 = tiling->s2;
        params.s1Tile = tiling->s1Tile;
        params.s2Tile = tiling->s2Tile;
        params.maskMode = tiling->maskMode;
        params.cmpRatio = tiling->cmpRatio;
        params.hasSequsedQ = tiling->hasSequsedQ;
        params.hasSequsedK = tiling->hasSequsedK;
        for (uint32_t i = 0; i < BlockReduceHead::kMaxBatch; ++i) {
            params.sequsedQ[i] = tiling->sequsedQ[i];
            params.sequsedK[i] = tiling->sequsedK[i];
            params.residual[i] = tiling->cmpResidualK[i];
        }
        for (uint32_t i = 0; i <= BlockReduceHead::kMaxBatch; ++i) {
            params.cuK[i] = tiling->cuSeqlensK[i];
        }
        return params;
    }

    CATLASS_DEVICE static typename BlockTopK::Params MakeTopKParams(
        __gm__ GeneralIndexerTilingData const* tiling)
    {
        typename BlockTopK::Params params{};
        params.topK = tiling->k;
        params.s2Stride = tiling->s2;
        params.s2Tile = tiling->s2Tile;
        params.hasOutputIdxOffset = tiling->hasOutputIdxOffset != 0;
        return params;
    }

public:
    CATLASS_DEVICE
    GeneralIndexerKernel() : ubC_(resource_.ubBuf.template GetBufferByByte<typename BlockMmad::DtypeC>(0))
    {}

    CATLASS_DEVICE void operator()(
        AscendC::GlobalTensor<DtypeQ> const& gmQ, AscendC::GlobalTensor<DtypeK> const& gmK,
        AscendC::GlobalTensor<DtypeW> const& gmW, AscendC::GlobalTensor<DtypeOut> const& outIndices,
        AscendC::GlobalTensor<DtypeS> const& outValues, AscendC::GlobalTensor<DtypeS>& scores,
        __gm__ GeneralIndexerTilingData const* tiling,
        AscendC::GlobalTensor<int32_t> const& gmBlockTable,
        AscendC::GlobalTensor<DtypeK> const& gmZeroPage,
        AscendC::GlobalTensor<int32_t> const& gmOutputIdxOffset)
    {
        typename BlockScheduler::Params schedulerParams = MakeSchedulerParams(tiling);
        BlockScheduler blockScheduler(schedulerParams);

        uint32_t qStart = 0;
        uint32_t actualQ = 0;
        uint32_t kStart = 0;
        uint32_t actualK = 0;
        uint32_t batchIdx = 0;
        uint32_t localQStart = 0;
        if ASCEND_IS_AIC {
            typename BlockMmad::Params mmadParams{MakeKeyLoaderParams(tiling)};
            BlockMmad blockMmad(
                ubC_, gmQ, gmK, Catlass::GemmCoord(tiling->s1 * tiling->n1, tiling->s2, tiling->d),
                Catlass::GemmCoord(
                    SparseIndexer::Common::AlignUp(tiling->s1Tile * tiling->n1, Catlass::C0_NUM_PER_FRACTAL), tiling->s2Tile, tiling->d),
                resource_, mmadParams, gmBlockTable, gmZeroPage);
            while (blockScheduler.GetNextCoord(qStart, actualQ, kStart, actualK, batchIdx, localQStart)) {
                asc_sync_block_wait(PIPE_FIX, FLAG_VEC_TO_CUBE);
                blockMmad(qStart * tiling->n1, actualQ * tiling->n1, kStart, actualK, batchIdx);
                asc_sync_block_arrive(PIPE_FIX, FLAG_CUBE_TO_VEC);
            }
            asc_sync_block_wait(PIPE_FIX, FLAG_VEC_TO_CUBE);
        } else {
            typename BlockReduceHead::Params reduceParams = MakeReduceParams(tiling);
            BlockReduceHead blockReduceHead(ubC_, resource_, gmW, scores, reduceParams);
            asc_sync_block_arrive(PIPE_MTE3, FLAG_VEC_TO_CUBE);
            while (blockScheduler.GetNextCoord(qStart, actualQ, kStart, actualK, batchIdx, localQStart)) {
                asc_sync_block_wait(PIPE_V, FLAG_CUBE_TO_VEC);
                blockReduceHead(qStart, actualQ, kStart, actualK, batchIdx, localQStart);
                asc_sync_block_arrive(PIPE_MTE3, FLAG_VEC_TO_CUBE);
            }
        }

        AscendC::SyncAll<false>();
        if ASCEND_IS_AIV {
            typename BlockTopK::Params topkParams = MakeTopKParams(tiling);
            BlockTopK blockTopK(resource_, scores, outIndices, topkParams, gmOutputIdxOffset, outValues);
            while (blockScheduler.GetNextTopKCoord(qStart, actualQ, kStart, actualK)) {
                blockTopK(qStart, actualQ, kStart, actualK);
            }
        }
    }

private:
    static constexpr uint16_t FLAG_VEC_TO_CUBE = 0;
    static constexpr uint16_t FLAG_CUBE_TO_VEC = 1;

    Catlass::Arch::Resource<Catlass::Arch::Ascend950> resource_;
    AscendC::LocalTensor<typename BlockMmad::DtypeC> ubC_;
};

#endif // FLASHINFER_GENERAL_INDEXER_KERNEL_HPP
