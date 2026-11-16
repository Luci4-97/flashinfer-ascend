/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#include "general_indexer_host.h"
#include "register/op_def_registry.h"

#define FI_STRING_INNER(value) #value
#define FI_STRING(value) FI_STRING_INNER(value)
#define FI_REGISTER_OP(op) OP_ADD(op)

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
        SetInferShape(InferGeneralIndexerShape).SetInferDataType(InferGeneralIndexerType);
        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true).DynamicFormatFlag(true).DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true).NeedCheckSupportFlag(false).PrecisionReduceFlag(true);
        AICore().SetTiling(FiGeneralIndexerTiling).AddConfig(FI_STRING(FI_COMPUTE_UNIT), config);
    }
};
FI_REGISTER_OP(FI_PACKAGE_OP_TYPE);
} // namespace ops
