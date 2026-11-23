/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#ifndef FLASHINFER_SPARSE_INDEXER_BLOCK_BLOCK_TOPK_HPP
#define FLASHINFER_SPARSE_INDEXER_BLOCK_BLOCK_TOPK_HPP

#include "kernel_operator.h"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"

#include "sparse_indexer/foundation/contracts/common.hpp"
#include "sparse_indexer/foundation/primitive/mask.hpp"
#include "sparse_indexer/foundation/primitive/topk_vf.hpp"
#include "sparse_indexer/foundation/resource/linear_arena.hpp"

namespace SparseIndexer::Common::Block {

struct SelectionRuntimeParams {
    uint32_t topK{0};
    uint32_t s2Stride{0};
    uint32_t s2Tile{0};
    bool hasOutputIdxOffset{false};
};

namespace Reg = AscendC::Reg;
using AscendC::CMPMODE;
using AscendC::RoundMode;

__simd_vf__ void Bf16KeysToSortableVF(__ubuf__ uint16_t *keys, uint32_t visibleLen, uint32_t paddedLen)
{
    constexpr uint32_t kKeysPerBlock = 128;
    Reg::MaskReg maskAllB16 = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg validMask;
    Reg::MaskReg invalidMask;
    Reg::RegTensor<uint16_t> bf16Bits;
    Reg::RegTensor<uint16_t> keyReg;
    Reg::RegTensor<uint16_t> zeroReg;
    Reg::RegTensor<int16_t> laneIdx;
    liV2TopkCommon::Bf16SortConstCtx ctx;
    liV2TopkCommon::InitBf16SortConstCtx(ctx, maskAllB16);
    Reg::Duplicate(zeroReg, static_cast<uint16_t>(0), maskAllB16);

    for (uint32_t off = 0; off < paddedLen; off += kKeysPerBlock) {
        Reg::LoadAlign<uint16_t, Reg::LoadDist::DIST_NORM>(bf16Bits, keys + off);
        liV2TopkCommon::Bf16ToSortableKey(keyReg, (Reg::RegTensor<bfloat16_t> &)bf16Bits, ctx, maskAllB16);
        Reg::Arange(laneIdx, static_cast<int16_t>(off));
        Reg::Compares<int16_t, CMPMODE::LT>(validMask, laneIdx, static_cast<int16_t>(visibleLen), maskAllB16);
        Reg::Not(invalidMask, validMask, maskAllB16);
        Reg::Select(keyReg, zeroReg, keyReg, invalidMask);
        Reg::StoreAlign<uint16_t, Reg::StoreDist::DIST_NORM>(keys + off, keyReg, maskAllB16);
    }
}

__simd_vf__ void BuildPrefixIndexVF(__ubuf__ uint16_t *indices, uint32_t visibleLen, uint32_t topK)
{
    constexpr uint32_t kU16Lanes = 128;
    Reg::MaskReg maskAllB16 = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg validMask;
    Reg::RegTensor<int16_t> laneIdx;
    Reg::RegTensor<uint16_t> indexReg;
    Reg::RegTensor<uint16_t> sentinel;
    Reg::Duplicate(sentinel, static_cast<uint16_t>(0xffff), maskAllB16);

    const uint32_t paddedTopK = SparseIndexer::Common::AlignUp(topK, kU16Lanes);
    for (uint32_t off = 0; off < paddedTopK; off += kU16Lanes) {
        Reg::Arange(laneIdx, static_cast<int16_t>(off));
        Reg::Compares<int16_t, CMPMODE::LT>(validMask, laneIdx, static_cast<int16_t>(visibleLen), maskAllB16);
        Reg::Select(indexReg, (Reg::RegTensor<uint16_t> &)laneIdx, sentinel, validMask);
        Reg::StoreAlign<uint16_t, Reg::StoreDist::DIST_NORM>(indices + off, indexReg, maskAllB16);
    }
}

__simd_vf__ void FinalizeIndexVF(__ubuf__ int32_t *out, __ubuf__ uint16_t *indices, uint32_t validLen,
                                 uint32_t topK, int32_t base)
{
    Reg::MaskReg maskAllB32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg validMask;
    Reg::RegTensor<uint16_t> index16;
    Reg::RegTensor<uint32_t> index32;
    Reg::RegTensor<int32_t> result;
    Reg::RegTensor<int32_t> negativeOne;
    Reg::Duplicate(negativeOne, static_cast<int32_t>(-1), maskAllB32);

    const uint32_t topKBlocks = SparseIndexer::Common::CeilDiv(topK, 64U);
    for (uint32_t block = 0; block < topKBlocks; ++block) {
        const uint32_t off = block * 64U;
        Reg::LoadAlign<uint16_t, Reg::LoadDist::DIST_UNPACK_B16>(index16, indices + off);
        index32 = (Reg::RegTensor<uint32_t> &)index16;
        Reg::Compares<uint32_t, CMPMODE::LT>(validMask, index32, validLen, maskAllB32);
        Reg::Adds(index32, index32, static_cast<uint32_t>(base), maskAllB32);
        Reg::Select(result, (Reg::RegTensor<int32_t> &)index32, negativeOne, validMask);
        Reg::StoreAlign<int32_t, Reg::StoreDist::DIST_NORM>(out + off, result, maskAllB32);
    }
}

template <class DtypeScore_, class DtypeOut_, bool LocalRows_ = false>
class BlockTopK {
public:
    struct Params {
        uint32_t topK{0};
        uint32_t s2Stride{0};
        uint32_t s2Tile{0};
        bool hasOutputIdxOffset{false};
    };

    using DtypeScore = DtypeScore_;
    using DtypeOut = DtypeOut_;
    using ArchTag = Catlass::Arch::Ascend950;
    using VectorBlock = BlockTopK;

    static constexpr uint32_t kSplitS1 = LocalRows_ ? 1U : 2U;
    static constexpr uint32_t kFloatBlock = Catlass::BYTE_PER_BLK / sizeof(DtypeScore);

    CATLASS_DEVICE BlockTopK(Catlass::Arch::Resource<ArchTag>& resource,
        AscendC::GlobalTensor<DtypeScore> const& workspace, AscendC::GlobalTensor<DtypeOut> const& outIndices,
        Params const& params, AscendC::GlobalTensor<int32_t> const& outputIdxOffset)
        : gmScores_(workspace),
          gmOut_(outIndices),
          gmOutputIdxOffset_(outputIdxOffset),
          topK_(params.topK),
          s2Stride_(params.s2Stride),
          s2Tile_(params.s2Tile),
          hasOutputIdxOffset_(params.hasOutputIdxOffset),
          tileColAlign_(SparseIndexer::Common::AlignUp(params.s2Tile, 256U)),
          topKAlign_(SparseIndexer::Common::AlignUp(params.topK, 256U))
    {
        const uint32_t outWords = topKAlign_ + 64U;
        const uint32_t tmpWords = SparseIndexer::Common::AlignUp(sharedBufferBytes(params.topK, tileColAlign_), 4U) / 4U;
        SparseIndexer::Common::LinearArena<decltype(resource.ubBuf)> arena(resource.ubBuf);
        scoreUb_ = arena.Take<DtypeScore>(tileColAlign_);
        const uint32_t inputCols = AscendC::Std::max(tileColAlign_, topKAlign_);
        inputUb_ = arena.Take<uint16_t>(inputCols, 256U);
        outIdxUb_ = arena.Take<uint32_t>(outWords);
        outValueUb_ = arena.Take<uint16_t>(topKAlign_ + 64U, 256U);
        topkTmpUb_ = arena.Take<uint32_t>(tmpWords);

        hisIndex1_ = outIdxUb_;
        hisIndex2_ = topkTmpUb_[0];
        hist_ = hisIndex2_[topKAlign_];
        idxHigh_ = hist_[256];
        idxLow_ = idxHigh_[256];
        nk_ = idxLow_[256];
        tmpIndex_ = nk_[64].template ReinterpretCast<uint16_t>();
    }

    CATLASS_DEVICE static uint32_t VisibleLen(uint32_t localQ, uint32_t qLen, uint32_t kLen, int32_t residual)
    {
        const int64_t raw = static_cast<int64_t>(kLen) * 4 + residual - static_cast<int64_t>(qLen) + localQ + 1;
        if (raw <= 0) {
            return 0;
        }
        return AscendC::Std::min(kLen, static_cast<uint32_t>(raw / 4));
    }

    CATLASS_DEVICE static void RunLocal(Catlass::Arch::Resource<ArchTag>& resource,
        SelectionRuntimeParams const& selection, AscendC::GlobalTensor<float>& scores,
        AscendC::GlobalTensor<int32_t> const& outIndices, uint32_t s1Start, uint32_t rows,
        uint32_t kStart, uint32_t kLen, uint32_t localQStart, uint32_t qLen, int32_t residual)
    {
        if (rows == 0) {
            return;
        }
        AscendC::GlobalTensor<int32_t> dummyOffset;
        typename BlockTopK::Params params{};
        params.topK = selection.topK;
        params.s2Stride = selection.s2Stride;
        params.s2Tile = selection.s2Tile;
        params.hasOutputIdxOffset = selection.hasOutputIdxOffset;
        BlockTopK topk(resource, scores, outIndices, params, dummyOffset);
        topk(s1Start, rows, kStart, kLen, localQStart, qLen, kLen, residual);
    }

    CATLASS_DEVICE void operator()(uint32_t s1Start, uint32_t actualS1, uint32_t kStart, uint32_t kLen,
        uint32_t localQStart, uint32_t qLen, uint32_t requestKLen, int32_t residual)
    {
        if (actualS1 == 0) {
            return;
        }
        uint32_t rowBegin = 0;
        uint32_t rowCount = actualS1;
        if constexpr (!LocalRows_) {
            const uint32_t rowsPerSub = SparseIndexer::Common::CeilDiv(actualS1, kSplitS1);
            const uint32_t sub = AscendC::GetSubBlockIdx();
            rowBegin = sub * rowsPerSub;
            if (rowBegin >= actualS1) {
                return;
            }
            rowCount = AscendC::Std::min(rowsPerSub, actualS1 - rowBegin);
        }

        AscendC::DataCopyExtParams copyIn{1, 0, 0, 0, 0};
        AscendC::DataCopyExtParams copyOut{1, static_cast<uint32_t>(topK_ * sizeof(DtypeOut)), 0, 0, 0};
        AscendC::LocalTensor<DtypeOut> outIdx = outIdxUb_.template ReinterpretCast<DtypeOut>();

        for (uint32_t ri = 0; ri < rowCount; ++ri) {
            const uint32_t s1Row = s1Start + rowBegin + ri;
            const uint32_t visibleLen = VisibleLen(localQStart + rowBegin + ri, qLen, requestKLen, residual);
            if (visibleLen < topK_) {
                BuildPrefixIndexVF((__ubuf__ uint16_t *)tmpIndex_.GetPhyAddr(), visibleLen, topK_);
            } else {
                copyIn.blockLen = kLen * sizeof(DtypeScore);
                AscendC::DataCopyPadExtParams<DtypeScore> padParams{false, 0, 0, static_cast<DtypeScore>(0)};
                AscendC::DataCopyPad(scoreUb_, gmScores_[s1Row * s2Stride_ + kStart], copyIn, padParams);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                AscendC::PipeBarrier<PIPE_V>();
                const uint32_t paddedLen =
                    SparseIndexer::Common::AlignUp(AscendC::Std::max(visibleLen, topK_), 256U);
                AscendC::Cast(inputUb_.template ReinterpretCast<bfloat16_t>(), scoreUb_, RoundMode::CAST_RINT,
                    paddedLen);
                AscendC::PipeBarrier<PIPE_V>();
                Bf16KeysToSortableVF((__ubuf__ uint16_t *)inputUb_.GetPhyAddr(), visibleLen, paddedLen);
                AscendC::PipeBarrier<PIPE_V>();
                topkb16gather::LiTopKVF<false>(tmpIndex_, outValueUb_, inputUb_, hist_, idxHigh_, idxLow_, nk_, topK_,
                    paddedLen);
            }

            int32_t rowOff = hasOutputIdxOffset_ ? gmOutputIdxOffset_.GetValue(s1Row) : 0;
            AscendC::PipeBarrier<PIPE_V>();
            FinalizeIndexVF((__ubuf__ int32_t *)outIdx.GetPhyAddr(), (__ubuf__ uint16_t *)tmpIndex_.GetPhyAddr(),
                visibleLen, topK_, static_cast<int32_t>(kStart) + rowOff);
            // LocalRows processes multiple rows on the same AIV. The VF
            // finalize store must be visible before the MTE3 output copy.
            if constexpr (LocalRows_) {
                AscendC::PipeBarrier<PIPE_ALL>();
            }
            storeRow(outIdx, s1Row, copyOut);
        }
    }

private:
    CATLASS_DEVICE static uint32_t sharedBufferBytes(uint32_t topK, uint32_t trunkLen)
    {
        const uint32_t alignedTopK = SparseIndexer::Common::AlignUp(topK, 256U);
        const uint32_t bytes1 = (2U * alignedTopK + 3U * 256U + 64U) * sizeof(uint32_t);
        const uint32_t bytes2 = (alignedTopK + trunkLen) * sizeof(uint16_t);
        const uint32_t reuse = alignedTopK * sizeof(uint32_t);
        return bytes1 + bytes2 - reuse;
    }

    CATLASS_DEVICE void storeRow(AscendC::LocalTensor<DtypeOut> const& outIdx,
        uint32_t s1Row, AscendC::DataCopyExtParams const& copyOut)
    {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::DataCopyPad(gmOut_[s1Row * topK_], outIdx, copyOut);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
    }

    AscendC::GlobalTensor<DtypeScore> const& gmScores_;
    AscendC::GlobalTensor<DtypeOut> const& gmOut_;
    AscendC::GlobalTensor<int32_t> const& gmOutputIdxOffset_;
    AscendC::LocalTensor<DtypeScore> scoreUb_;
    AscendC::LocalTensor<uint16_t> inputUb_;
    AscendC::LocalTensor<uint32_t> outIdxUb_;
    AscendC::LocalTensor<uint16_t> outValueUb_;
    AscendC::LocalTensor<uint32_t> topkTmpUb_;
    AscendC::LocalTensor<uint32_t> hisIndex1_;
    AscendC::LocalTensor<uint32_t> hisIndex2_;
    AscendC::LocalTensor<uint32_t> hist_;
    AscendC::LocalTensor<uint32_t> idxHigh_;
    AscendC::LocalTensor<uint32_t> idxLow_;
    AscendC::LocalTensor<uint32_t> nk_;
    AscendC::LocalTensor<uint16_t> tmpIndex_;
    uint32_t topK_{0};
    uint32_t s2Stride_{0};
    uint32_t s2Tile_{0};
    bool hasOutputIdxOffset_{false};
    uint32_t tileColAlign_{0};
    uint32_t topKAlign_{0};
};


} // namespace SparseIndexer::Common::Block


#endif // FLASHINFER_SPARSE_INDEXER_BLOCK_BLOCK_TOPK_HPP
