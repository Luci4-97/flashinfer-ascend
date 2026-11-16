/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#ifndef FLASHINFER_GENERAL_INDEXER_ARCH35_SOURCE_H
#define FLASHINFER_GENERAL_INDEXER_ARCH35_SOURCE_H

#include <cstdint>

#include "kernel_operator.h"

#include "general_indexer_tiling_data.h"
#include "sparse_indexer_common.hpp"

namespace GeneralIndexerKernelDetail {

/**
 * Variant-specific adapter between tiling/runtime metadata and the common scheduler.
 * The common BlockScheduler only sees normalized request ranges.
 */
class GeneralIndexerRequestSource {
public:
    static constexpr uint32_t kMaxBatch = SparseIndexer::Common::kMaxBatch;

    GeneralIndexerRequestSource() = default;

    CATLASS_DEVICE
    void InitFromTiling(__gm__ GeneralIndexerTilingData const* tiling)
    {
        s1_ = tiling->s1;
        s2_ = tiling->s2;
        batch_ = tiling->batch == 0 ? 1U : tiling->batch;
        if (batch_ > kMaxBatch) {
            batch_ = 1U;
        }
        maskMode_ = tiling->maskMode;
        hasSequsedQ_ = tiling->hasSequsedQ != 0;
        hasSequsedK_ = tiling->hasSequsedK != 0;
        for (uint32_t i = 0; i <= batch_; ++i) {
            cuQ_[i] = tiling->cuSeqlensQ[i];
            cuK_[i] = tiling->cuSeqlensK[i];
        }
        for (uint32_t i = 0; i < batch_; ++i) {
            sequsedQ_[i] = tiling->sequsedQ[i];
            sequsedK_[i] = tiling->sequsedK[i];
            cmpResidualK_[i] = tiling->cmpResidualK[i];
        }
    }

    CATLASS_DEVICE uint32_t Batch() const { return batch_; }
    CATLASS_DEVICE uint32_t QBegin(uint32_t b) const { return static_cast<uint32_t>(cuQ_[b]); }
    CATLASS_DEVICE uint32_t QLen(uint32_t b) const { return static_cast<uint32_t>(cuQ_[b + 1] - cuQ_[b]); }
    CATLASS_DEVICE uint32_t KBegin(uint32_t b) const { return static_cast<uint32_t>(cuK_[b]); }
    CATLASS_DEVICE uint32_t KLen(uint32_t b) const { return static_cast<uint32_t>(cuK_[b + 1] - cuK_[b]); }
    CATLASS_DEVICE uint32_t ActiveQ(uint32_t b) const { return hasSequsedQ_ ? static_cast<uint32_t>(sequsedQ_[b]) : QLen(b); }
    CATLASS_DEVICE uint32_t ActiveK(uint32_t b) const { return hasSequsedK_ ? static_cast<uint32_t>(sequsedK_[b]) : KLen(b); }
    CATLASS_DEVICE int32_t Residual(uint32_t b) const { return cmpResidualK_[b]; }
    CATLASS_DEVICE bool MaskActive() const { return maskMode_ == 3 || hasSequsedQ_ || hasSequsedK_; }

private:
    uint32_t s1_{0};
    uint32_t s2_{0};
    uint32_t batch_{1};
    int32_t maskMode_{0};
    bool hasSequsedQ_{false};
    bool hasSequsedK_{false};
    int32_t cuQ_[kMaxBatch + 1]{0};
    int32_t cuK_[kMaxBatch + 1]{0};
    int32_t sequsedQ_[kMaxBatch]{0};
    int32_t sequsedK_[kMaxBatch]{0};
    int32_t cmpResidualK_[kMaxBatch]{0};
};

} // namespace GeneralIndexerKernelDetail

#endif // FLASHINFER_GENERAL_INDEXER_ARCH35_SOURCE_H
