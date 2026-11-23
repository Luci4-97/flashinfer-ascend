/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 *
 * Vector reduce for QLI quant_mode=1:
 *   score[s2] = k_scale[s2] * sum_h (w[h] * q_scale[h]) * ReLU(Q @ K^T)[h, s2]
 * Cube produces raw FP8 matmul into shared UB; this block owns scales + ReLU +
 * head reduce + mask3, matching QLIv2's vector ownership of float scales.
 */

#ifndef FLASHINFER_SPARSE_INDEXER_BLOCK_REDUCE_QUANT_HEAD_HPP
#define FLASHINFER_SPARSE_INDEXER_BLOCK_REDUCE_QUANT_HEAD_HPP

#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "kernel_operator.h"

#include "sparse_indexer/block/reduce/direct_workspace_store.hpp"
#include "sparse_indexer/block/mask/mask_policy.hpp"
#include "sparse_indexer/block/mask/runtime.hpp"
#include "sparse_indexer/foundation/contracts/paged_kv.hpp"
#include "sparse_indexer/foundation/resource/linear_arena.hpp"
#include "sparse_indexer/foundation/contracts/common.hpp"

namespace SparseIndexer::Common::Block {

namespace Reg = AscendC::Reg;

__simd_vf__ void ScaleHeadRowVF(__ubuf__ float *out, __ubuf__ float *score, __ubuf__ float *weight,
                                __ubuf__ float *kScale, uint32_t n1, uint32_t headStride, uint32_t cols)
{
    Reg::MaskReg mask = Reg::UpdateMask<float>(cols);
    Reg::RegTensor<float> sum;
    Reg::RegTensor<float> qk;
    Reg::RegTensor<float> w;
    Reg::RegTensor<float> scale;

    Reg::Duplicate(sum, 0.0f, mask);
    for (uint32_t head = 0; head < n1; ++head) {
        // score/headStride are 32-byte aligned in the QLI UB layout.
        Reg::LoadAlign<float>(qk, score + head * headStride);
        Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(w, weight + head);
        Reg::Relu(qk, qk, mask);
        Reg::MulAddDst(sum, qk, w, mask);
    }
    Reg::LoadAlign<float>(scale, kScale);
    Reg::Mul(sum, sum, scale, mask);
    Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM>(out, sum, mask);
}

// H4-only fast path: one row x two 64-lane segments. Keeping the register
// footprint to two accumulators avoids the pressure that made a 2-row port
// slower on this UB layout.
__simd_vf__ void ScaleHeadH4Row2SegVF(__ubuf__ float *out, __ubuf__ float *score,
                                      __ubuf__ float *weight, __ubuf__ float *kScale,
                                      uint32_t headStride)
{
    Reg::MaskReg mask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    Reg::RegTensor<float> acc0;
    Reg::RegTensor<float> acc1;
    Reg::RegTensor<float> qk;
    Reg::RegTensor<float> w;
    Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(w, weight);
    Reg::LoadAlign<float>(qk, score);
    Reg::Relu(qk, qk, mask);
    Reg::Mul(acc0, qk, w, mask);
    Reg::LoadAlign<float>(qk, score + 64U);
    Reg::Relu(qk, qk, mask);
    Reg::Mul(acc1, qk, w, mask);
    Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(w, weight + 1U);
    Reg::LoadAlign<float>(qk, score + headStride);
    Reg::Relu(qk, qk, mask);
    Reg::MulAddDst(acc0, qk, w, mask);
    Reg::LoadAlign<float>(qk, score + headStride + 64U);
    Reg::Relu(qk, qk, mask);
    Reg::MulAddDst(acc1, qk, w, mask);
    Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(w, weight + 2U);
    Reg::LoadAlign<float>(qk, score + 2U * headStride);
    Reg::Relu(qk, qk, mask);
    Reg::MulAddDst(acc0, qk, w, mask);
    Reg::LoadAlign<float>(qk, score + 2U * headStride + 64U);
    Reg::Relu(qk, qk, mask);
    Reg::MulAddDst(acc1, qk, w, mask);
    Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(w, weight + 3U);
    Reg::LoadAlign<float>(qk, score + 3U * headStride);
    Reg::Relu(qk, qk, mask);
    Reg::MulAddDst(acc0, qk, w, mask);
    Reg::LoadAlign<float>(qk, score + 3U * headStride + 64U);
    Reg::Relu(qk, qk, mask);
    Reg::MulAddDst(acc1, qk, w, mask);
    Reg::RegTensor<float> scale0;
    Reg::RegTensor<float> scale1;
    Reg::LoadAlign<float>(scale0, kScale);
    Reg::LoadAlign<float>(scale1, kScale + 64U);
    Reg::Mul(acc0, acc0, scale0, mask);
    Reg::Mul(acc1, acc1, scale1, mask);
    Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM>(out, acc0, mask);
    Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM>(out + 64U, acc1, mask);
}

template <class MaskPolicy_ = SparseIndexer::Common::Block::LiV2MaskPolicy<float>, bool RowSplit_ = false>
class BlockReduceQuantHead {
public:
    using DtypeScore = float;
    using DtypeW = float;
    using DtypeDst = float;
    using ArchTag = Catlass::Arch::Ascend950;
    using MaskPolicy = MaskPolicy_;
    using Block = BlockReduceQuantHead;

    static constexpr uint32_t kFloatBlock = Catlass::BYTE_PER_BLK / sizeof(float);
    static constexpr uint32_t kFloatVec = 256 / sizeof(float);
    static constexpr uint32_t kMaxRepeat = 255;
    static constexpr uint32_t kKScaleCacheTiles = 16;
    static constexpr bool kRowSplit = RowSplit_;

    struct Params {
        uint32_t n1{0};
        uint32_t s2{0};
        uint32_t s1Tile{0};
        uint32_t s2Tile{0};
        uint32_t blockSize{0};
        uint32_t maxBlocksPerBatch{0};
        uint32_t numPages{0};
        uint64_t kScaleStride0{0};
        SparseIndexer::Common::Block::MaskRuntimeParams mask{};
        uint32_t scoreBuffers{1};
    };

    CATLASS_DEVICE
    BlockReduceQuantHead(
        AscendC::LocalTensor<float>& ubScore, Catlass::Arch::Resource<ArchTag>& resource,
        AscendC::GlobalTensor<float> const& gmW, AscendC::GlobalTensor<float> const& gmQScale,
        AscendC::GlobalTensor<float> const& gmKScale, AscendC::GlobalTensor<int32_t> const& gmBlockTable,
        AscendC::GlobalTensor<float> const& gmDst, Params const& params)
        : ubScore_(ubScore),
          gmW_(gmW),
          gmQScale_(gmQScale),
          gmKScale_(gmKScale),
          gmBlockTable_(gmBlockTable),
          gmDst_(gmDst),
          params_(params),
          n1_(params.n1),
          s2_(params.s2),
          s1Tile_(params.s1Tile),
          s2Tile_(params.s2Tile),
          maskPolicy_(ubOut_, ubWBrcb_, ubTmp_)
    {
        wStride_ = SparseIndexer::Common::AlignUp(n1_, kFloatBlock);
        // ScaleHeadVF processes at least one 256-byte row vector. With a
        // four-row tile, actualS1 can round up to eight rows, so scratch must
        // follow the compute-aligned row count rather than s1Tile alone.
        const uint32_t workRows = kRowSplit ? SparseIndexer::Common::CeilDiv(s1Tile_, 2U) : s1Tile_;
        const uint32_t vectorRows = AscendC::Std::max(workRows, kFloatBlock);
        const uint32_t aivPerAic = AscendC::GetSubBlockNum();
        const uint32_t colPerAiv = kRowSplit
            ? SparseIndexer::Common::AlignUp(s2Tile_, 8U)
            : SparseIndexer::Common::AlignUp(s2Tile_, 16U * aivPerAic) / aivPerAic;
        const uint32_t outStride = kRowSplit ? SparseIndexer::Common::AlignUp(s2Tile_, 64U) : colPerAiv;
        scoreElems_ = workRows * n1_ * colPerAiv;
        outElems_ = workRows * outStride;
        SparseIndexer::Common::LinearArena<decltype(resource.ubBuf)> arena(
            resource.ubBuf, params.scoreBuffers * scoreElems_ * sizeof(float));
        ubOutBase_ = arena.Take<float>(params.scoreBuffers * outElems_);
        ubOut_ = ubOutBase_;
        ubW_ = arena.Take<float>(vectorRows * wStride_);
        ubWIn_ = arena.Take<float>(vectorRows * wStride_);
        ubQScaleIn_ = arena.Take<float>(vectorRows * wStride_);
        ubWBrcb_ = arena.Take<float>(vectorRows * kFloatBlock, 32U);
        // MaskPolicy may build a full 256-byte vector per row.
        ubTmp_ = arena.Take<float>(vectorRows * kFloatVec, 32U);
        kScaleCacheCols_ = kKScaleCacheTiles * s2Tile_;
        ubKScale_ = arena.Take<float>(kScaleCacheCols_, 256U);
    }

    CATLASS_DEVICE
    void WaitOutput(uint32_t bufferIdx)
    {
        if (outStorePending_[bufferIdx]) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0 + bufferIdx);
            outStorePending_[bufferIdx] = false;
        }
    }

    CATLASS_DEVICE
    void WaitAllOutputs()
    {
        for (uint32_t i = 0; i < params_.scoreBuffers; ++i) {
            WaitOutput(i);
        }
    }

    CATLASS_DEVICE
    void ScaleHeadRow(uint32_t row, uint32_t sourceStride, uint32_t outStride, uint32_t cols,
                      uint32_t kScaleOffset, uint32_t scoreElemOffset)
    {
        constexpr uint32_t kBlockFloats = 64;
        constexpr uint32_t kPairFloats = 2U * kBlockFloats;
        uint32_t off = 0;
        if (n1_ == 4U) {
            for (; off + kPairFloats <= cols; off += kPairFloats) {
                ScaleHeadH4Row2SegVF(
                    (__ubuf__ float *)ubOut_[row * outStride + off].GetPhyAddr(),
                    (__ubuf__ float *)ubScore_[scoreElemOffset + (row * n1_) * sourceStride + off].GetPhyAddr(),
                    (__ubuf__ float *)ubW_[row * wStride_].GetPhyAddr(),
                    (__ubuf__ float *)ubKScale_[kScaleOffset + off].GetPhyAddr(), sourceStride);
            }
        }
        for (; off < cols; off += kBlockFloats) {
            const uint32_t blockCols = AscendC::Std::min(kBlockFloats, cols - off);
            ScaleHeadRowVF((__ubuf__ float *)ubOut_[row * outStride + off].GetPhyAddr(),
                (__ubuf__ float *)ubScore_[scoreElemOffset + (row * n1_) * sourceStride + off].GetPhyAddr(),
                (__ubuf__ float *)ubW_[row * wStride_].GetPhyAddr(),
                (__ubuf__ float *)ubKScale_[kScaleOffset + off].GetPhyAddr(), n1_, sourceStride, blockCols);
        }
    }

    CATLASS_DEVICE
    void ScaleHeadVF(uint32_t rows, uint32_t sourceStride, uint32_t outStride, uint32_t cols,
                    uint32_t kScaleOffset, uint32_t scoreElemOffset)
    {
        for (uint32_t row = 0; row < rows; ++row) {
            ScaleHeadRow(row, sourceStride, outStride, cols, kScaleOffset, scoreElemOffset);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    CATLASS_DEVICE
    void LoadKScaleRange(uint32_t start, uint32_t cols, uint32_t batchIdx, uint32_t dstOffset)
    {
        uint32_t localCol = 0;
        while (localCol < cols) {
            const uint32_t tok = start + localCol;
            const uint32_t block = tok / params_.blockSize;
            const uint32_t off = tok % params_.blockSize;
            const uint32_t segCols = AscendC::Std::min(cols - localCol, params_.blockSize - off);
            int32_t page = -1;
            if (block < params_.maxBlocksPerBatch) {
                page = gmBlockTable_.GetValue(
                    static_cast<int64_t>(batchIdx) * params_.maxBlocksPerBatch + block);
            }
            if (page >= 0 && static_cast<uint32_t>(page) < params_.numPages) {
                AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID1);
                AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID1);
                AscendC::DataCopyExtParams copyIn{
                    1, static_cast<uint32_t>(segCols * sizeof(float)), 0, 0, 0};
                AscendC::DataCopyPadExtParams<float> pad{false, 0, 0, 0.0f};
                AscendC::DataCopyPad(
                    ubKScale_[dstOffset + localCol],
                    gmKScale_[static_cast<uint32_t>(page) * params_.kScaleStride0 + off], copyIn, pad);
            } else {
                AscendC::Duplicate(ubKScale_[dstOffset + localCol], 0.0f, segCols);
            }
            localCol += segCols;
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
        AscendC::PipeBarrier<PIPE_V>();
    }

    CATLASS_DEVICE
    void LoadKScaleCache(uint32_t cacheStart, uint32_t fillCols, uint32_t batchIdx)
    {
        LoadKScaleRange(cacheStart, fillCols, batchIdx, 0);
    }

    CATLASS_DEVICE void operator()(
        uint32_t s1Start, uint32_t actualS1, uint32_t s2Start, uint32_t actualS2, uint32_t localQStart,
        uint32_t qAct, uint32_t kAct, uint32_t reqKStart, int32_t residual, bool maskActive, uint32_t batchIdx,
        uint32_t scoreBufferIdx)
    {
        const uint32_t sub = AscendC::GetSubBlockIdx();
        uint32_t rowBegin = 0;
        uint32_t rows = 0;
        uint32_t colBegin = 0;
        uint32_t cols = 0;
        uint32_t sourceStride = 0;
        uint32_t outStride = 0;
        if constexpr (kRowSplit) {
            const uint32_t rowsPerAiv = SparseIndexer::Common::CeilDiv(actualS1, 2U);
            rowBegin = sub * rowsPerAiv;
            if (rowBegin >= actualS1) {
                return;
            }
            rows = AscendC::Std::min(rowsPerAiv, actualS1 - rowBegin);
            cols = actualS2;
            sourceStride = SparseIndexer::Common::AlignUp(actualS2, 8U);
            outStride = SparseIndexer::Common::AlignUp(actualS2, 64U);
        } else {
            const uint32_t aivPerAic = AscendC::GetSubBlockNum();
            outStride = SparseIndexer::Common::AlignUp(actualS2, 16U * aivPerAic) / aivPerAic;
            colBegin = sub * outStride;
            if (colBegin >= actualS2) {
                return;
            }
            rows = actualS1;
            cols = AscendC::Std::min(outStride, actualS2 - colBegin);
            sourceStride = outStride;
        }
        const uint32_t outBufferIdx = scoreBufferIdx % params_.scoreBuffers;
        ubOut_ = ubOutBase_[outBufferIdx * outElems_];
        WaitOutput(outBufferIdx);

        const uint32_t alignedRows = SparseIndexer::Common::AlignUp(rows, kFloatBlock);
        const uint32_t outputRowStart = s1Start + rowBegin;
        const uint32_t requestQStart = localQStart + rowBegin;

        // W and QScale are invariant across every S2 tile owned by this row
        // group. Fuse them once per gS1 task instead of repeating two tiny
        // GM->UB copies and a Mul for each 128-column tile.
        const bool weightCacheHit =
            weightCacheValid_ && weightCacheRowStart_ == outputRowStart && weightCacheRows_ == rows;
        if (!weightCacheHit) {
            const uint32_t rightPad = wStride_ - n1_;
            AscendC::DataCopyExtParams load{
                static_cast<uint16_t>(rows), static_cast<uint32_t>(n1_ * sizeof(float)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<float> pad{rightPad != 0, 0, static_cast<uint8_t>(rightPad), 0.0f};
            AscendC::DataCopyPad(ubWIn_, gmW_[outputRowStart * n1_], load, pad);
            AscendC::DataCopyPad(ubQScaleIn_, gmQScale_[outputRowStart * n1_], load, pad);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            if (alignedRows > rows) {
                AscendC::Duplicate(ubWIn_[rows * wStride_], 0.0f, (alignedRows - rows) * wStride_);
                AscendC::Duplicate(ubQScaleIn_[rows * wStride_], 0.0f, (alignedRows - rows) * wStride_);
                AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::Mul(ubW_, ubWIn_, ubQScaleIn_, alignedRows * wStride_);
            AscendC::PipeBarrier<PIPE_V>();
            weightCacheValid_ = true;
            weightCacheRowStart_ = outputRowStart;
            weightCacheRows_ = rows;
        }

        // QLIv2 caches 16 S2 tiles of PA key scales. Reuse the same chunk so
        // only page-id lookups and one MTE2 transfer are paid per 2048 tokens.
        uint32_t kScaleOffset = 0;
        if constexpr (kRowSplit) {
            const uint32_t tileStart = s2Start + colBegin;
            const uint32_t cacheStart = tileStart / kScaleCacheCols_ * kScaleCacheCols_;
            const bool cacheHit = kScaleCacheValid_ && kScaleCacheBatch_ == batchIdx &&
                kScaleCacheStart_ == cacheStart;
            if (!cacheHit) {
                const uint32_t fillCols = cacheStart < kAct
                    ? AscendC::Std::min(kScaleCacheCols_, kAct - cacheStart)
                    : 0U;
                LoadKScaleCache(cacheStart, fillCols, batchIdx);
                kScaleCacheValid_ = true;
                kScaleCacheBatch_ = batchIdx;
                kScaleCacheStart_ = cacheStart;
            }
            kScaleOffset = tileStart - cacheStart;
        } else {
            LoadKScaleRange(s2Start + colBegin, cols, batchIdx, 0);
        }
        const uint32_t scoreElemOffset = scoreBufferIdx * scoreElems_;
        bool integratedRowMask = false;
        if constexpr (kRowSplit) {
            if (maskActive && params_.mask.maskMode == 3) {
                integratedRowMask = true;
                uint32_t rowCols[4] = {cols, cols, cols, cols};
                const uint32_t tileBase = s2Start + colBegin - reqKStart;
                bool needsFill = false;
                for (uint32_t row = 0; row < rows; ++row) {
                    const uint32_t localQ = localQStart + rowBegin + row;
                    if (localQ >= qAct) {
                        rowCols[row] = 0U;
                    } else {
                        const uint32_t visible = SparseIndexer::Common::Block::Mask3VisibleLen(
                            localQ, qAct, kAct, params_.mask.cmpRatio, residual);
                        rowCols[row] = visible > tileBase
                            ? AscendC::Std::min(cols, visible - tileBase)
                            : 0U;
                    }
                    needsFill = needsFill || rowCols[row] < cols;
                }
                if (needsFill) {
                    for (uint32_t row = 0; row < rows; ++row) {
                        if (rowCols[row] < cols) {
                            auto rowOut = ubOut_[row * outStride];
                            AscendC::Duplicate(rowOut.ReinterpretCast<uint32_t>(),
                                SparseIndexer::Common::Block::kFloatNegInfBits, outStride);
                        }
                    }
                    AscendC::PipeBarrier<PIPE_V>();
                }
                for (uint32_t row = 0; row < rows; ++row) {
                    if (rowCols[row] != 0U) {
                        ScaleHeadRow(row, sourceStride, outStride, rowCols[row], kScaleOffset, scoreElemOffset);
                    }
                }
                AscendC::PipeBarrier<PIPE_V>();
            }
        }
        if (!integratedRowMask) {
            ScaleHeadVF(rows, sourceStride, outStride, cols, kScaleOffset, scoreElemOffset);
            maskPolicy_.Apply(
                rows, cols, outStride, colBegin, s2Start, requestQStart, qAct, kAct, reqKStart, residual,
                maskActive, params_.mask.maskMode, params_.mask.cmpRatio);
        }
        SparseIndexer::Common::Block::DirectWorkspaceStore<float, float>::StoreTile(
            gmDst_[outputRowStart * s2_ + s2Start + colBegin], ubOut_, rows, cols, outStride, s2_,
            EVENT_ID0 + outBufferIdx);
        outStorePending_[outBufferIdx] = true;
    }

private:
    AscendC::LocalTensor<float>& ubScore_;
    AscendC::LocalTensor<float> ubOutBase_;
    AscendC::LocalTensor<float> ubOut_;
    AscendC::LocalTensor<float> ubW_;
    AscendC::LocalTensor<float> ubWIn_;
    AscendC::LocalTensor<float> ubQScaleIn_;
    AscendC::LocalTensor<float> ubWBrcb_;
    AscendC::LocalTensor<float> ubTmp_;
    AscendC::LocalTensor<float> ubKScale_;
    AscendC::GlobalTensor<float> const& gmW_;
    AscendC::GlobalTensor<float> const& gmQScale_;
    AscendC::GlobalTensor<float> const& gmKScale_;
    AscendC::GlobalTensor<int32_t> const& gmBlockTable_;
    AscendC::GlobalTensor<float> const& gmDst_;
    Params params_{};
    uint32_t n1_{0};
    uint32_t s2_{0};
    uint32_t s1Tile_{0};
    uint32_t s2Tile_{0};
    uint32_t wStride_{0};
    uint32_t scoreElems_{0};
    uint32_t outElems_{0};
    bool outStorePending_[2]{false, false};
    bool weightCacheValid_{false};
    uint32_t weightCacheRowStart_{0};
    uint32_t weightCacheRows_{0};
    uint32_t kScaleCacheCols_{0};
    uint32_t kScaleCacheStart_{0};
    uint32_t kScaleCacheBatch_{0};
    bool kScaleCacheValid_{false};
    MaskPolicy maskPolicy_;
};

} // namespace SparseIndexer::Common::Block

#endif // FLASHINFER_SPARSE_INDEXER_BLOCK_REDUCE_QUANT_HEAD_HPP
