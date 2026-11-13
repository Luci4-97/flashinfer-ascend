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

#ifndef FLASHINFER_GENERAL_INDEXER_TILING_DATA_H
#define FLASHINFER_GENERAL_INDEXER_TILING_DATA_H

#include <cstdint>

#include "catlass/detail/macros.hpp"

namespace GeneralIndexerAlign {
template <class T, class U>
CATLASS_HOST_DEVICE constexpr uint32_t AlignUp(const T& value, const U& alignment)
{
    return static_cast<uint32_t>((value + alignment - 1) / alignment * alignment);
}
} // namespace GeneralIndexerAlign

/**
 * @brief Device-visible tiling for the general indexer.
 *
 * Host embeds cu_seqlens / seqused / cmp residual so AIC/AIV can schedule and
 * mask without GetValue on separate GM buffers. Runtime Tensor addresses are
 * passed explicitly to the kernel entry and are never cached in tiling data.
 */
struct GeneralIndexerTilingData {
    static constexpr uint32_t kMaxBatch = 16;

    uint32_t s1{0};
    uint32_t n1{0};
    uint32_t s2{0};
    uint32_t d{0};
    uint32_t s1Tile{0};
    uint32_t s2Tile{0};
    uint32_t k{0};
    uint32_t batch{1};
    uint32_t hasCuSeqlens{0};
    uint32_t hasSequsedQ{0};
    uint32_t hasSequsedK{0};
    int32_t maskMode{0};
    int32_t cmpRatio{1};
    int32_t cuSeqlensQ[kMaxBatch + 1]{0};
    int32_t cuSeqlensK[kMaxBatch + 1]{0};
    int32_t sequsedQ[kMaxBatch]{0};
    int32_t sequsedK[kMaxBatch]{0};
    int32_t cmpResidualK[kMaxBatch]{0};
    uint32_t hasOutputIdxOffset{0};

    // PA_BBND in-kernel paging metadata. Tensor pointers are kernel arguments.
    uint32_t isPaged{0};
    uint32_t blockSize{0};
    uint32_t maxBlocksPerBatch{0};
    uint32_t numPages{0};
};

#endif // FLASHINFER_GENERAL_INDEXER_TILING_DATA_H
