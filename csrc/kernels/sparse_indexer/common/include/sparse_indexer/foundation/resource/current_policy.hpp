/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#ifndef FLASHINFER_SPARSE_INDEXER_FOUNDATION_RESOURCE_CURRENT_POLICY_HPP
#define FLASHINFER_SPARSE_INDEXER_FOUNDATION_RESOURCE_CURRENT_POLICY_HPP

#include "catlass/arch/resource.hpp"
#include "catlass/detail/macros.hpp"
#include "kernel_operator.h"

namespace SparseIndexer::Common {

/** C0 resource policy: the existing blocks allocate from one Catlass UB arena. */
template <class ArchTag_>
struct CurrentResourcePolicy {
    using ArchTag = ArchTag_;
    using Resource = Catlass::Arch::Resource<ArchTag>;

    template <class Dtype_>
    CATLASS_DEVICE static AscendC::LocalTensor<Dtype_> GetUbC(Resource& resource)
    {
        return resource.ubBuf.template GetBufferByByte<Dtype_>(0);
    }
};

} // namespace SparseIndexer::Common

#endif // FLASHINFER_SPARSE_INDEXER_FOUNDATION_RESOURCE_CURRENT_POLICY_HPP
