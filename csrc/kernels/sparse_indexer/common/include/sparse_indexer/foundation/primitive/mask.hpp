/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#ifndef FLASHINFER_SPARSE_INDEXER_FOUNDATION_PRIMITIVE_MASK_HPP
#define FLASHINFER_SPARSE_INDEXER_FOUNDATION_PRIMITIVE_MASK_HPP

#include <cstdint>

#include "kernel_operator.h"
#include "catlass/catlass.hpp"
#include "catlass/detail/macros.hpp"
#include "sparse_indexer/foundation/contracts/common.hpp"
#include "sparse_indexer/foundation/contracts/problem.hpp"

namespace SparseIndexer::Common::Block {

struct MaskRuntimeParams {
    int32_t maskMode{0};
    uint32_t cmpRatio{1};
    uint32_t s2Tile{1};
};

/// IEEE-754 float -inf bit pattern (shared by Reduce mask writes and TopK filters).
inline constexpr uint32_t kFloatNegInfBits = 0xFF800000U;

CATLASS_DEVICE bool IsNegInfFloat(float v)
{
    union {
        float f;
        uint32_t u;
    } bits;
    bits.f = v;
    return bits.u == kFloatNegInfBits;
}

/**
 * LI V2 create_mask visible column count for local Q row ``localQ``.
 * ``qAct`` / ``kAct`` are seqused lengths (or full request lengths when seqused absent).
 */
CATLASS_DEVICE uint32_t Mask3VisibleLen(
    uint32_t localQ, uint32_t qAct, uint32_t kAct, int32_t cmpRatio, int32_t residual)
{
    if (kAct == 0) {
        return 0;
    }
    const int32_t ratio = cmpRatio <= 0 ? 1 : cmpRatio;
    const int32_t tmp = static_cast<int32_t>(kAct) * ratio + residual - static_cast<int32_t>(qAct);
    const int32_t numer = tmp + static_cast<int32_t>(localQ) + 1;
    if (numer < 0) {
        return 0;
    }
    const uint32_t vis = static_cast<uint32_t>(numer / ratio);
    return vis < kAct ? vis : kAct;
}

template <class DtypeScore>
class NoMaskPolicy {
public:
    CATLASS_DEVICE
    NoMaskPolicy(
        AscendC::LocalTensor<DtypeScore> const&, AscendC::LocalTensor<DtypeScore> const&,
        AscendC::LocalTensor<DtypeScore> const&)
    {}

    CATLASS_DEVICE
    void Apply(
        uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, int32_t,
        bool, int32_t, int32_t)
    {}
};

/// LI V2 causal/seqused/cmp mask applied on the score tile in UB.
template <class DtypeScore_>
class LiV2MaskPolicy {
public:
    using DtypeScore = DtypeScore_;
    static constexpr uint32_t kFloatBlock = Catlass::BYTE_PER_BLK / sizeof(DtypeScore);

    CATLASS_DEVICE
    LiV2MaskPolicy(
        AscendC::LocalTensor<DtypeScore> const& ubOut, AscendC::LocalTensor<DtypeScore> const& ubMaskScratch,
        AscendC::LocalTensor<DtypeScore> const& ubTmp)
        : ubOut_(ubOut), ubMaskScratch_(ubMaskScratch), ubTmp_(ubTmp)
    {}

    CATLASS_DEVICE
    void Apply(
        uint32_t actualS1, uint32_t cols, uint32_t colPerAiv, uint32_t colBegin, uint32_t s2Start,
        uint32_t localQStart, uint32_t qAct, uint32_t kAct, uint32_t reqKStart, int32_t residual,
        bool maskActive, int32_t maskMode, int32_t cmpRatio)
    {
        if (!maskActive) {
            return;
        }

        const uint32_t kBase = s2Start + colBegin - reqKStart;
        static_assert(sizeof(DtypeScore) == sizeof(uint32_t), "mask VF assumes FP32 scores");
        uint32_t calcLen = SparseIndexer::Common::AlignUp(cols, kFloatBlock);
        if (calcLen > colPerAiv) {
            calcLen = colPerAiv;
        }
        AscendC::LocalTensor<uint8_t> maskUb = ubMaskScratch_.template ReinterpretCast<uint8_t>();
        AscendC::LocalTensor<uint32_t> ubTmpU32 = ubTmp_.template ReinterpretCast<uint32_t>();

        AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t ri = 0; ri < actualS1; ++ri) {
            const uint32_t localQ = localQStart + ri;
            uint32_t vis = kAct;
            if (localQ >= qAct) {
                vis = 0;
            } else if (maskMode == 3) {
                vis = Mask3VisibleLen(localQ, qAct, kAct, cmpRatio, residual);
            }
            if (vis >= kBase + cols) {
                continue;
            }

            AscendC::CreateVecIndex(ubTmp_, static_cast<DtypeScore>(0), calcLen);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Adds(ubTmp_, ubTmp_, static_cast<DtypeScore>(kBase), calcLen);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::CompareScalar(
                maskUb, ubTmp_, static_cast<DtypeScore>(vis), AscendC::CMPMODE::GE, calcLen);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Duplicate(ubTmpU32, kFloatNegInfBits, static_cast<int32_t>(calcLen));
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Select(
                ubOut_[ri * colPerAiv], maskUb, ubTmp_, ubOut_[ri * colPerAiv],
                AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, calcLen);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

private:
    AscendC::LocalTensor<DtypeScore> const& ubOut_;
    AscendC::LocalTensor<DtypeScore> const& ubMaskScratch_;
    AscendC::LocalTensor<DtypeScore> const& ubTmp_;
};

/**
 * Causal mask3 only needs the prefix visible to the last row of a gS1 tile.
 * Keep one tile for an empty-visible request so the CV hand-off stays paired.
 */
CATLASS_DEVICE inline WorkItem VisibleCoord(MaskRuntimeParams const& mask, WorkItem const& coord,
    uint32_t activeQ, uint32_t activeK, int32_t residual)
{
    auto visible = coord;
    if (mask.maskMode != 3 || coord.qRows == 0U) {
        return visible;
    }
    const uint32_t lastLocalQ = coord.localQBegin + coord.qRows - 1U;
    const uint32_t visibleK = Mask3VisibleLen(lastLocalQ, activeQ, activeK, mask.cmpRatio, residual);
    visible.kRows = AscendC::Std::max(visibleK, AscendC::Std::min(coord.kRows, mask.s2Tile));
    return visible;
}


} // namespace SparseIndexer::Common::Block

#endif // FLASHINFER_SPARSE_INDEXER_FOUNDATION_PRIMITIVE_MASK_HPP
