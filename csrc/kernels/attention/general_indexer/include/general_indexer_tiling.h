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

#ifndef FLASHINFER_GENERAL_INDEXER_TILING_H
#define FLASHINFER_GENERAL_INDEXER_TILING_H

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include <acl/acl.h>

#include "general_indexer_tiling_data.h"
#include "sparse_indexer_align.hpp"

namespace GeneralIndexerTiling {

struct Params {
    uint32_t s1{0};
    uint32_t n1{0};
    uint32_t s2{0};
    uint32_t d{0};
    uint32_t k{0};
    uint32_t batch{1};
    aclDataType dataType{ACL_FLOAT16};
    aclDataType weightsDataType{ACL_FLOAT16};
    const void* cuSeqlensQ{nullptr};
    const void* cuSeqlensK{nullptr};
    const void* sequsedQ{nullptr};
    const void* sequsedK{nullptr};
    const void* cmpResidualK{nullptr};
    bool hasBlockTable{false};
    bool hasOutputIdxOffset{false};
    int32_t maskMode{0};
    int32_t cmpRatio{1};
    uint32_t isPaged{0};
    uint32_t blockSize{0};
    uint32_t maxBlocksPerBatch{0};
    uint32_t numPages{0};
};


inline void GetTiling(const Params& params, GeneralIndexerTilingData& tilingData)
{
    if (params.s1 == 0 || params.n1 == 0 || params.s2 == 0 || params.d == 0) {
        throw std::invalid_argument("GeneralIndexerTiling: s1, n1, s2 and d must be positive");
    }
    if (params.k == 0 || params.k > params.s2) {
        throw std::invalid_argument("GeneralIndexerTiling: k must be in [1, s2]");
    }
    // LI V2 950: k ∈ [1, 8192]. S2 is streamed over s2Tile; no single-segment S2 cap (M3).
    constexpr uint32_t kTopKCap = 8192;
    if (params.k > kTopKCap) {
        throw std::invalid_argument("GeneralIndexerTiling: k must be <= 8192");
    }

    const bool hasCuQ = params.cuSeqlensQ != nullptr;
    const bool hasCuK = params.cuSeqlensK != nullptr;
    if (hasCuQ != hasCuK) {
        throw std::invalid_argument("GeneralIndexerTiling: cuSeqlensQ and cuSeqlensK must be provided together");
    }

    uint32_t batch = params.batch == 0 ? 1U : params.batch;
    tilingData.hasCuSeqlens = 0;
    if (!hasCuQ) {
        if (batch != 1) {
            throw std::invalid_argument("GeneralIndexerTiling: batch must be 1 without cu_seqlens");
        }
        tilingData.cuSeqlensQ[0] = 0;
        tilingData.cuSeqlensQ[1] = static_cast<int32_t>(params.s1);
        tilingData.cuSeqlensK[0] = 0;
        tilingData.cuSeqlensK[1] = static_cast<int32_t>(params.s2);
    } else {
        if (batch < 1 || batch > GeneralIndexerTilingData::kMaxBatch) {
            throw std::invalid_argument("GeneralIndexerTiling: batch must be in [1, kMaxBatch] with cu_seqlens");
        }
        std::vector<int32_t> cuQ(batch + 1);
        std::vector<int32_t> cuK(batch + 1);
        const size_t bytes = static_cast<size_t>(batch + 1) * sizeof(int32_t);
        if (aclrtMemcpy(cuQ.data(), bytes, params.cuSeqlensQ, bytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_ERROR_NONE) {
            throw std::runtime_error("GeneralIndexerTiling: failed to copy cuSeqlensQ");
        }
        if (aclrtMemcpy(cuK.data(), bytes, params.cuSeqlensK, bytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_ERROR_NONE) {
            throw std::runtime_error("GeneralIndexerTiling: failed to copy cuSeqlensK");
        }
        if (cuQ[0] != 0 || cuK[0] != 0) {
            throw std::invalid_argument("GeneralIndexerTiling: cu_seqlens must start with 0");
        }
        if (static_cast<uint32_t>(cuQ[batch]) != params.s1 || static_cast<uint32_t>(cuK[batch]) != params.s2) {
            throw std::invalid_argument("GeneralIndexerTiling: cu_seqlens end must equal s1/s2");
        }
        for (uint32_t b = 0; b < batch; ++b) {
            if (cuQ[b + 1] < cuQ[b] || cuK[b + 1] < cuK[b]) {
                throw std::invalid_argument("GeneralIndexerTiling: cu_seqlens must be non-decreasing");
            }
            const uint32_t kLen = static_cast<uint32_t>(cuK[b + 1] - cuK[b]);
            if (kLen > 0 && params.k > kLen) {
                throw std::invalid_argument("GeneralIndexerTiling: k must be <= each request's key length");
            }
        }
        tilingData.hasCuSeqlens = 1;
        for (uint32_t i = 0; i <= batch; ++i) {
            tilingData.cuSeqlensQ[i] = cuQ[i];
            tilingData.cuSeqlensK[i] = cuK[i];
        }
    }

    tilingData.s1 = params.s1;
    tilingData.n1 = params.n1;
    tilingData.s2 = params.s2;
    tilingData.d = params.d;
    tilingData.k = params.k;
    tilingData.batch = batch;
    tilingData.maskMode = params.maskMode;
    tilingData.cmpRatio = params.cmpRatio <= 0 ? 1 : params.cmpRatio;
    tilingData.hasOutputIdxOffset = params.hasOutputIdxOffset ? 1U : 0U;
    tilingData.isPaged = params.isPaged;
    tilingData.blockSize = params.blockSize;
    tilingData.maxBlocksPerBatch = params.maxBlocksPerBatch;
    tilingData.numPages = params.numPages;
    if (params.isPaged) {
        if (!params.hasBlockTable || params.blockSize == 0 || params.numPages == 0 ||
            params.maxBlocksPerBatch == 0) {
            throw std::invalid_argument(
                "GeneralIndexerTiling: paged Key requires blockTable, blockSize, numPages, maxBlocks");
        }
        if (params.sequsedK == nullptr) {
            throw std::invalid_argument("GeneralIndexerTiling: paged Key requires sequsedK");
        }
    }
    tilingData.hasSequsedQ = 0;
    tilingData.hasSequsedK = 0;
    for (uint32_t b = 0; b < batch; ++b) {
        const uint32_t qLen = static_cast<uint32_t>(tilingData.cuSeqlensQ[b + 1] - tilingData.cuSeqlensQ[b]);
        const uint32_t kLen = static_cast<uint32_t>(tilingData.cuSeqlensK[b + 1] - tilingData.cuSeqlensK[b]);
        tilingData.sequsedQ[b] = static_cast<int32_t>(qLen);
        tilingData.sequsedK[b] = static_cast<int32_t>(kLen);
        tilingData.cmpResidualK[b] = 0;
    }
    if (params.sequsedQ != nullptr) {
        std::vector<int32_t> seqQ(batch);
        const size_t bytes = static_cast<size_t>(batch) * sizeof(int32_t);
        if (aclrtMemcpy(seqQ.data(), bytes, params.sequsedQ, bytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_ERROR_NONE) {
            throw std::runtime_error("GeneralIndexerTiling: failed to copy sequsedQ");
        }
        for (uint32_t b = 0; b < batch; ++b) {
            tilingData.sequsedQ[b] = seqQ[b];
        }
        tilingData.hasSequsedQ = 1;
    }
    if (params.sequsedK != nullptr) {
        std::vector<int32_t> seqK(batch);
        const size_t bytes = static_cast<size_t>(batch) * sizeof(int32_t);
        if (aclrtMemcpy(seqK.data(), bytes, params.sequsedK, bytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_ERROR_NONE) {
            throw std::runtime_error("GeneralIndexerTiling: failed to copy sequsedK");
        }
        for (uint32_t b = 0; b < batch; ++b) {
            tilingData.sequsedK[b] = seqK[b];
        }
        tilingData.hasSequsedK = 1;
    }
    if (params.cmpResidualK != nullptr) {
        std::vector<int32_t> residual(batch);
        const size_t bytes = static_cast<size_t>(batch) * sizeof(int32_t);
        if (aclrtMemcpy(residual.data(), bytes, params.cmpResidualK, bytes, ACL_MEMCPY_DEVICE_TO_HOST) !=
            ACL_ERROR_NONE) {
            throw std::runtime_error("GeneralIndexerTiling: failed to copy cmpResidualK");
        }
        for (uint32_t b = 0; b < batch; ++b) {
            tilingData.cmpResidualK[b] = residual[b];
        }
    }

    uint32_t wElemBytes = 2;
    if (params.weightsDataType == ACL_FLOAT) {
        wElemBytes = 4;
    } else if (params.weightsDataType == ACL_FLOAT16 || params.weightsDataType == ACL_BF16) {
        wElemBytes = 2;
    } else {
        throw std::invalid_argument("GeneralIndexerTiling: unsupported weightsDataType");
    }

    constexpr uint32_t kS2TileCap = 256;
    constexpr uint32_t kUbSize = 248 * 1024;
    constexpr uint32_t kL0cSize = 256 * 1024;
    constexpr uint32_t aivPerAic = 2;
    constexpr uint32_t cubeFractalN = 16;
    constexpr uint32_t cubeFractalM = 16;
    constexpr uint32_t l0cFractalN = 16;
    constexpr uint32_t splitNAlign = l0cFractalN * aivPerAic;
    const uint32_t wStride = SparseIndexer::Common::AlignUp(params.n1, 32 / wElemBytes);
    // Keep s2Tile cube-fractal aligned (may exceed s2); actualK is clamped by the scheduler.
    const uint32_t s2Tile =
        params.s2 <= kS2TileCap ? SparseIndexer::Common::AlignUp(params.s2, cubeFractalN) : SparseIndexer::Common::AlignUp(kS2TileCap, cubeFractalN);
    const uint32_t s2Split = SparseIndexer::Common::AlignUp(s2Tile, splitNAlign);
    const uint64_t ubBytesPerS1 = sizeof(float) * (s2Split / aivPerAic) * (params.n1 + 1) +
                                  (sizeof(float) + wElemBytes) * wStride + sizeof(float) + 32 +
                                  256;
    const uint32_t l0cRows = kL0cSize / (s2Split * static_cast<uint32_t>(sizeof(float)));
    tilingData.s2Tile = s2Tile;
    // Capacity may exceed params.s1 so Reduce can pad actualS1 to 8 (D1).
    uint32_t s1Tile = static_cast<uint32_t>(kUbSize / ubBytesPerS1);
    while (s1Tile > 0 && SparseIndexer::Common::AlignUp(s1Tile * params.n1, cubeFractalM) > l0cRows) {
        --s1Tile;
    }
    // BlockReduceHead pads actualS1 to FP32 block (8) for UB DataCopy/Cast/Relu.
    // Keep s1Tile a multiple of 8 and at least 8 whenever UB/L0C allow.
    constexpr uint32_t kS1Align = 8;
    if (s1Tile >= kS1Align) {
        s1Tile = s1Tile / kS1Align * kS1Align;
    } else if (
        s1Tile > 0 && kS1Align * ubBytesPerS1 <= kUbSize &&
        SparseIndexer::Common::AlignUp(kS1Align * params.n1, cubeFractalM) <= l0cRows) {
        s1Tile = kS1Align;
    }
    tilingData.s1Tile = s1Tile;
    if (tilingData.s1Tile == 0) {
        throw std::invalid_argument("GeneralIndexerTiling: s1Tile is zero under UB/L0C limits");
    }

}

} // namespace GeneralIndexerTiling

#endif // FLASHINFER_GENERAL_INDEXER_TILING_H
