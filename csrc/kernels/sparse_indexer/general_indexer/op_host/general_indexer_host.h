/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#ifndef FLASHINFER_GENERAL_INDEXER_HOST_H
#define FLASHINFER_GENERAL_INDEXER_HOST_H

#include <cstddef>
#include <cstdint>

#include "register/op_def_registry.h"

namespace general_indexer_host {
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

template <typename T, typename Context>
T Attr(const Context* context, size_t index, T fallback)
{
    const auto* attrs = context->GetAttrs();
    if (attrs == nullptr) return fallback;
    const T* value = attrs->template GetAttrPointer<T>(index);
    return value == nullptr ? fallback : *value;
}
} // namespace general_indexer_host

ge::graphStatus InferGeneralIndexerShape(gert::InferShapeContext* context);
ge::graphStatus InferGeneralIndexerType(gert::InferDataTypeContext* context);
ge::graphStatus FiGeneralIndexerTiling(gert::TilingContext* context);

#endif // FLASHINFER_GENERAL_INDEXER_HOST_H
