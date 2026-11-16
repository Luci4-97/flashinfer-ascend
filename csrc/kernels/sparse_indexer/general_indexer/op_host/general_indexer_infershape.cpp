/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#include "general_indexer_host.h"

ge::graphStatus InferGeneralIndexerShape(gert::InferShapeContext* context)
{
    using namespace general_indexer_host;
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
    int64_t tk = 0;
    const int64_t isPaged = Attr<int64_t>(context, 1, 0);
    const int64_t topk = Attr<int64_t>(context, 0, 0);
    const int64_t maxSeq = Attr<int64_t>(context, 4, 0);
    if (isPaged != 0) {
        if (kshape.GetDimNum() != 3 || kshape.GetDim(2) != d) return ge::GRAPH_FAILED;
        const auto* cuQ = context->GetInputShape(CU_Q);
        int64_t batch = 1;
        if (cuQ != nullptr && cuQ->GetDimNum() == 1) batch = cuQ->GetDim(0) - 1;
        if (batch < 1 || maxSeq <= 0) return ge::GRAPH_FAILED;
        tk = batch * maxSeq;
    } else {
        if (kshape.GetDimNum() != 2 || kshape.GetDim(1) != d) return ge::GRAPH_FAILED;
        tk = kshape.GetDim(0);
    }
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

ge::graphStatus InferGeneralIndexerType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, ge::DT_INT32);
    context->SetOutputDataType(1, ge::DT_FLOAT);
    context->SetOutputDataType(2, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}
