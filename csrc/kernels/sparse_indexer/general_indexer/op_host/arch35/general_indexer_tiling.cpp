/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#include <stdexcept>

#include "general_indexer_host.h"
#include "general_indexer_tiling.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
aclDataType AclDtype(ge::DataType dtype)
{
    switch (dtype) {
        case ge::DT_FLOAT16: return ACL_FLOAT16;
        case ge::DT_BF16: return ACL_BF16;
        case ge::DT_FLOAT: return ACL_FLOAT;
        case ge::DT_INT32: return ACL_INT32;
        default: throw std::invalid_argument("unsupported GeneralIndexer dtype");
    }
}

const void* DataOf(const gert::TilingContext* context, size_t index)
{
    const auto* tensor = context->GetInputTensor(index);
    return tensor == nullptr ? nullptr : tensor->template GetData<uint8_t>();
}
} // namespace

ge::graphStatus FiGeneralIndexerTiling(gert::TilingContext* context)
{
    using namespace general_indexer_host;
    try {
        const auto* queryTensor = context->GetInputTensor(QUERY);
        const auto* keyTensor = context->GetInputTensor(KEY);
        const auto* weightsTensor = context->GetInputTensor(WEIGHTS);
        if (queryTensor == nullptr || keyTensor == nullptr || weightsTensor == nullptr) {
            return ge::GRAPH_FAILED;
        }
        const auto& qs = queryTensor->GetStorageShape();
        const auto& ws = weightsTensor->GetStorageShape();
        const auto& ks = keyTensor->GetStorageShape();
        if (qs.GetDimNum() != 3 || ws.GetDimNum() != 2 ||
            (ks.GetDimNum() != 2 && ks.GetDimNum() != 3)) {
            return ge::GRAPH_FAILED;
        }
        const int64_t tq = qs.GetDim(0);
        const int64_t n1 = qs.GetDim(1);
        const int64_t d = qs.GetDim(2);
        if (tq <= 0 || n1 <= 0 || d <= 0 || ws.GetDim(0) != tq || ws.GetDim(1) != n1 ||
            ks.GetDim(ks.GetDimNum() - 1) != d) {
            return ge::GRAPH_FAILED;
        }

        const int64_t isPaged = Attr<int64_t>(context, 1, 0);
        const int64_t k = Attr<int64_t>(context, 0, 0);
        const int64_t maxSeq = Attr<int64_t>(context, 4, 0);
        const auto* cuQTensor = context->GetInputTensor(CU_Q);
        int64_t batch = 1;
        if (cuQTensor != nullptr) batch = cuQTensor->GetStorageShape().GetDim(0) - 1;
        int64_t tk = 0;
        int64_t pages = 0;
        int64_t blockSize = 0;
        int64_t maxBlocks = 0;
        if (isPaged != 0) {
            if (ks.GetDimNum() != 3 || maxSeq <= 0) return ge::GRAPH_FAILED;
            pages = ks.GetDim(0);
            blockSize = ks.GetDim(1);
            const auto* table = context->GetInputTensor(BLOCK_TABLE);
            if (table == nullptr) return ge::GRAPH_FAILED;
            maxBlocks = table->GetStorageShape().GetDim(1);
            tk = batch * maxSeq;
        } else {
            if (ks.GetDimNum() != 2) return ge::GRAPH_FAILED;
            tk = ks.GetDim(0);
        }

        GeneralIndexerTiling::Params params;
        params.s1 = static_cast<uint32_t>(tq);
        params.n1 = static_cast<uint32_t>(n1);
        params.s2 = static_cast<uint32_t>(tk);
        params.d = static_cast<uint32_t>(d);
        params.k = static_cast<uint32_t>(k);
        params.batch = static_cast<uint32_t>(batch);
        params.dataType = AclDtype(context->GetInputDesc(QUERY)->GetDataType());
        params.weightsDataType = AclDtype(context->GetInputDesc(WEIGHTS)->GetDataType());
        params.cuSeqlensQ = DataOf(context, CU_Q);
        params.cuSeqlensK = DataOf(context, CU_K);
        params.sequsedQ = DataOf(context, SEQ_Q);
        params.sequsedK = DataOf(context, SEQ_K);
        params.cmpResidualK = DataOf(context, CMP_RESIDUAL);
        params.hasBlockTable = context->GetInputTensor(BLOCK_TABLE) != nullptr;
        params.hasOutputIdxOffset = context->GetInputTensor(IDX_OFFSET) != nullptr;
        params.maskMode = static_cast<int32_t>(Attr<int64_t>(context, 2, 0));
        params.cmpRatio = static_cast<int32_t>(Attr<int64_t>(context, 3, 1));
        params.isPaged = static_cast<uint32_t>(isPaged);
        params.blockSize = static_cast<uint32_t>(blockSize);
        params.maxBlocksPerBatch = static_cast<uint32_t>(maxBlocks);
        params.numPages = static_cast<uint32_t>(pages);

        auto* data = context->GetTilingData<GeneralIndexerTilingData>();
        if (data == nullptr) return ge::GRAPH_FAILED;
        *data = {};
        GeneralIndexerTiling::GetTiling(params, *data);

        platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
        const uint32_t cores = platform.GetCoreNumAic();
        if (cores == 0) return ge::GRAPH_FAILED;
        context->SetBlockDim(cores);
        context->SetTilingKey(isPaged != 0 ? 1 : 0);
        context->GetWorkspaceSizes(1)[0] = platform.GetLibApiWorkSpaceSize();
        return ge::GRAPH_SUCCESS;
    } catch (const std::exception&) {
        return ge::GRAPH_FAILED;
    }
}
