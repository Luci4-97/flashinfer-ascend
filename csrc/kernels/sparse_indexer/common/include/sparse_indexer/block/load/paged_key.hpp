/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#ifndef FLASHINFER_SPARSE_INDEXER_BLOCK_LOAD_PAGED_KEY_HPP
#define FLASHINFER_SPARSE_INDEXER_BLOCK_LOAD_PAGED_KEY_HPP

#include "sparse_indexer/block/load/key_loader.hpp"

namespace SparseIndexer::Common::Block {

/**
 * PA_BBND key loader policy. The load policy owns the key address mapping and
 * is composed into the mmad policy instead of being selected inside the kernel.
 */
template <class Dtype_>
struct PagedKeyLoadPolicy {
    using Dtype = Dtype_;
    using Loader = SparseIndexer::Common::Block::PagedKeyLoader<
        Catlass::Arch::Ascend950, Catlass::Gemm::GemmType<Dtype, Catlass::layout::ColumnMajor>>;
};

} // namespace SparseIndexer::Common::Block

#endif // FLASHINFER_SPARSE_INDEXER_BLOCK_LOAD_PAGED_KEY_HPP
