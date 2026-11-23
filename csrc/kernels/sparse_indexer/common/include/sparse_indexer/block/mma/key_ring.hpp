/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#ifndef FLASHINFER_SPARSE_INDEXER_BLOCK_MMA_KEY_RING_HPP
#define FLASHINFER_SPARSE_INDEXER_BLOCK_MMA_KEY_RING_HPP

#include "catlass/detail/macros.hpp"
#include "kernel_operator.h"
#include "sparse_indexer/block/load/key_loader.hpp"
#include "sparse_indexer/foundation/contracts/common.hpp"

namespace SparseIndexer::Common::Block {

/**
 * Private L1B ring for one BlockMmad. It owns only the key loader and the
 * three key-buffer events; the Q/L0/L0C lifecycle remains in BlockMmad.
 */
template <class DtypeB_, class KeyLoader_>
class KeyRing {
public:
    using DtypeB = DtypeB_;
    using KeyLoader = KeyLoader_;
    using LayoutB = typename KeyLoader::LayoutB;
    using LoaderParams = typename KeyLoader::Params;
    using TensorB = AscendC::LocalTensor<DtypeB>;

    CATLASS_DEVICE KeyRing() = default;

    CATLASS_DEVICE void Setup(
        TensorB const& l1B, AscendC::GlobalTensor<DtypeB> const& gmB, LayoutB const& layoutB,
        uint32_t problemK, uint32_t kPart, uint32_t keyPartElems, uint32_t keyBufElems,
        LoaderParams const& params, AscendC::GlobalTensor<int32_t> const& gmTable,
        AscendC::GlobalTensor<DtypeB> const& gmZero)
    {
        l1B_ = l1B;
        problemK_ = problemK;
        kPart_ = kPart;
        keyPartElems_ = keyPartElems;
        keyBufElems_ = keyBufElems;
        keyLoader_.Setup(gmB, layoutB, params, gmTable, gmZero);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
        initialized_ = true;
    }

    CATLASS_DEVICE ~KeyRing()
    {
        if (!initialized_) {
            return;
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
    }

    CATLASS_DEVICE static constexpr uint32_t Event(uint32_t bufferIdx)
    {
        return bufferIdx == 0U ? EVENT_ID0 : (bufferIdx == 1U ? EVENT_ID2 : EVENT_ID3);
    }

    CATLASS_DEVICE TensorB Buffer(uint32_t bufferIdx, uint32_t kIdx) const
    {
        return l1B_[bufferIdx * keyBufElems_ + kIdx * keyPartElems_];
    }

    CATLASS_DEVICE void WaitReady(uint32_t bufferIdx) const
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(Event(bufferIdx));
    }

    CATLASS_DEVICE void Release(uint32_t bufferIdx) const
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(Event(bufferIdx));
    }

    CATLASS_DEVICE void Prefetch(
        uint32_t bufferIdx, KeyCoord const& keyCoord, uint32_t actualN, uint32_t alignedN)
    {
        const uint32_t kTileCount = SparseIndexer::Common::CeilDiv(problemK_, kPart_);
        auto layoutBInL1 = Catlass::layout::nZ::template MakeLayout<DtypeB>(kPart_, alignedN);
        const uint32_t eventId = Event(bufferIdx);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(eventId);
        for (uint32_t kIdx = 0; kIdx < kTileCount; ++kIdx) {
            const uint32_t kStart = kIdx * kPart_;
            const uint32_t actualK = AscendC::Std::min(kPart_, problemK_ - kStart);
            keyLoader_.Load(Buffer(bufferIdx, kIdx), layoutBInL1, kStart, actualK, keyCoord, actualN);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(eventId);
    }

private:
    TensorB l1B_;
    KeyLoader keyLoader_;
    uint32_t problemK_{0};
    uint32_t kPart_{0};
    uint32_t keyPartElems_{0};
    uint32_t keyBufElems_{0};
    bool initialized_{false};
};

} // namespace SparseIndexer::Common::Block

#endif // FLASHINFER_SPARSE_INDEXER_BLOCK_MMA_KEY_RING_HPP
