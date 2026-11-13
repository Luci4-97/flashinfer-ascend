/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#include <cstdint>
#include <stdexcept>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "general_indexer_tiling.h"

#define FI_STRING_INNER(value) #value
#define FI_STRING(value) FI_STRING_INNER(value)
#define FI_REGISTER_OP(op) OP_ADD(op)

namespace {
constexpr size_t QUERY = 0;
constexpr size_t KEY = 1;
constexpr size_t WEIGHTS = 2;
constexpr size_t BLOCK_TABLE = 3;
constexpr size_t CU_Q = 4;
constexpr size_t CU_K = 5;
constexpr size_t SEQ_Q = 6;
constexpr size_t SEQ_K = 7;
constexpr size_t CMP_RESIDUAL = 8;
constexpr size_t IDX_OFFSET = 9;
constexpr size_t ZERO_PAGE = 10;

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

template <typename T, typename Context>
T Attr(const Context* context, size_t index, T fallback)
{
    const auto* attrs = context->GetAttrs();
    if (attrs == nullptr) return fallback;
    const T* value = attrs->template GetAttrPointer<T>(index);
    return value == nullptr ? fallback : *value;
}

ge::graphStatus Infer(gert::InferShapeContext* context)
{
    const auto* query = context->GetInputShape(QUERY);
    const auto* key = context->GetInputShape(KEY);
    if (query == nullptr || key == nullptr || query->GetDimNum() != 3 ||
        (key->GetDimNum() != 2 && key->GetDimNum() != 3)) {
        return ge::GRAPH_FAILED;
    }
    const auto& q = *query;
    const auto& kshape = *key;
    const int64_t tq = q.GetDim(0);
    const int64_t d = q.GetDim(2);
    int64_t blockSize = 0;
    int64_t pages = 0;
    int64_t tk = 0;
    const int64_t isPaged = Attr<int64_t>(context, 1, 0);
    const int64_t topk = Attr<int64_t>(context, 0, 0);
    const int64_t maxSeq = Attr<int64_t>(context, 4, 0);
    if (isPaged != 0) {
        if (kshape.GetDimNum() != 3 || kshape.GetDim(2) != d) return ge::GRAPH_FAILED;
        pages = kshape.GetDim(0);
        blockSize = kshape.GetDim(1);
        const auto* cuQ = context->GetInputShape(CU_Q);
        int64_t batch = 1;
        if (cuQ != nullptr && cuQ->GetDimNum() == 1) batch = cuQ->GetDim(0) - 1;
        if (batch < 1 || maxSeq <= 0) return ge::GRAPH_FAILED;
        tk = batch * maxSeq;
    } else {
        if (kshape.GetDimNum() != 2 || kshape.GetDim(1) != d) return ge::GRAPH_FAILED;
        tk = kshape.GetDim(0);
    }
    (void)pages;
    (void)blockSize;
    if (tq <= 0 || tk <= 0 || topk <= 0) return ge::GRAPH_FAILED;
    auto* indices = context->GetOutputShape(0);
    auto* values = context->GetOutputShape(1);
    auto* scores = context->GetOutputShape(2);
    for (auto* out : {indices, values}) {
        out->SetDimNum(2);
        out->SetDim(0, tq);
        out->SetDim(1, topk);
    }
    scores->SetDimNum(2);
    scores->SetDim(0, tq);
    scores->SetDim(1, tk);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferType(gert::InferDataTypeContext* context)
{
    (void)context;
    context->SetOutputDataType(0, ge::DT_INT32);
    context->SetOutputDataType(1, ge::DT_FLOAT);
    context->SetOutputDataType(2, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}
} // namespace

ge::graphStatus FiGeneralIndexerTiling(gert::TilingContext* context)
{
    auto fail = [](int) { return ge::GRAPH_FAILED; };
    try {
        const auto* queryTensor = context->GetInputTensor(QUERY);
        const auto* keyTensor = context->GetInputTensor(KEY);
        const auto* weightsTensor = context->GetInputTensor(WEIGHTS);
        if (queryTensor == nullptr || keyTensor == nullptr || weightsTensor == nullptr) {
            return fail(1);
        }
        const auto& qs = queryTensor->GetStorageShape();
        const auto& ws = weightsTensor->GetStorageShape();
        const auto& ks = keyTensor->GetStorageShape();
        if (qs.GetDimNum() != 3 || ws.GetDimNum() != 2 ||
            (ks.GetDimNum() != 2 && ks.GetDimNum() != 3)) {
            return fail(2);
        }
        const int64_t tq = qs.GetDim(0);
        const int64_t n1 = qs.GetDim(1);
        const int64_t d = qs.GetDim(2);
        if (tq <= 0 || n1 <= 0 || d <= 0 || ws.GetDim(0) != tq || ws.GetDim(1) != n1 ||
            ks.GetDim(ks.GetDimNum() - 1) != d) {
            return fail(3);
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
            if (ks.GetDimNum() != 3 || maxSeq <= 0) return fail(4);
            pages = ks.GetDim(0);
            blockSize = ks.GetDim(1);
            const auto* table = context->GetInputTensor(BLOCK_TABLE);
            if (table == nullptr) return fail(5);
            maxBlocks = table->GetStorageShape().GetDim(1);
            tk = batch * maxSeq;
        } else {
            if (ks.GetDimNum() != 2) return fail(6);
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
        params.blockTable = DataOf(context, BLOCK_TABLE);
        params.outputIdxOffset = DataOf(context, IDX_OFFSET);
        params.zeroPage = DataOf(context, ZERO_PAGE);
        params.maskMode = static_cast<int32_t>(Attr<int64_t>(context, 2, 0));
        params.cmpRatio = static_cast<int32_t>(Attr<int64_t>(context, 3, 1));
        params.isPaged = static_cast<uint32_t>(isPaged);
        params.blockSize = static_cast<uint32_t>(blockSize);
        params.maxBlocksPerBatch = static_cast<uint32_t>(maxBlocks);
        params.numPages = static_cast<uint32_t>(pages);

        auto* data = context->GetTilingData<GeneralIndexerTilingData>();
        if (data == nullptr) return fail(7);
        *data = {};
        uint64_t scoreWorkspaceBytes = 0;
        GeneralIndexerTiling::GetTiling(params, *data, scoreWorkspaceBytes);
        (void)scoreWorkspaceBytes;

        platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
        const uint32_t cores = platform.GetCoreNumAic();
        if (cores == 0) return fail(8);
        context->SetBlockDim(cores);
        context->SetTilingKey(isPaged != 0 ? 1 : 0);
        context->GetWorkspaceSizes(1)[0] = platform.GetLibApiWorkSpaceSize();
        return ge::GRAPH_SUCCESS;
    } catch (const std::exception&) {
        return ge::GRAPH_FAILED;
    }
}

namespace ops {
class FI_PACKAGE_OP_TYPE : public OpDef {
public:
    explicit FI_PACKAGE_OP_TYPE(const char* name) : OpDef(name)
    {
        Input("query").ParamType(REQUIRED).DataType({ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
            ge::DT_BF16, ge::DT_BF16, ge::DT_BF16}).Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
            ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        Input("key").ParamType(REQUIRED).DataType({ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
            ge::DT_BF16, ge::DT_BF16, ge::DT_BF16}).Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
            ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        Input("weights").ParamType(REQUIRED).DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT,
            ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT}).Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
            ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        Input("block_table").ParamType(OPTIONAL).DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
            ge::DT_INT32, ge::DT_INT32, ge::DT_INT32}).Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
            ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        for (const char* name : {"cu_seqlens_q", "cu_seqlens_k", "seqused_q", "seqused_k",
                                 "cmp_residual_k", "output_idx_offset"}) {
            Input(name).ParamType(OPTIONAL).DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
                ge::DT_INT32, ge::DT_INT32, ge::DT_INT32}).Format({ge::FORMAT_ND, ge::FORMAT_ND,
                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        }
        Input("score_zero_page").ParamType(OPTIONAL).DataType({ge::DT_FLOAT16, ge::DT_FLOAT16,
            ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16}).Format({ge::FORMAT_ND, ge::FORMAT_ND,
            ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        Output("indices").ParamType(REQUIRED).DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
            ge::DT_INT32, ge::DT_INT32, ge::DT_INT32}).Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
            ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        Output("values").ParamType(REQUIRED).DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
            ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT}).Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
            ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        Output("scores").ParamType(REQUIRED).DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
            ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT}).Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
            ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        Attr("k").AttrType(REQUIRED).Int(0);
        Attr("is_paged").AttrType(OPTIONAL).Int(0);
        Attr("mask_mode").AttrType(OPTIONAL).Int(0);
        Attr("cmp_ratio").AttrType(OPTIONAL).Int(1);
        Attr("max_seq_len_k").AttrType(OPTIONAL).Int(0);
        SetInferShape(Infer).SetInferDataType(InferType);
        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true).DynamicFormatFlag(true).DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true).NeedCheckSupportFlag(false).PrecisionReduceFlag(true);
        AICore().SetTiling(FiGeneralIndexerTiling).AddConfig(FI_STRING(FI_COMPUTE_UNIT), config);
    }
};
FI_REGISTER_OP(FI_PACKAGE_OP_TYPE);
} // namespace ops
