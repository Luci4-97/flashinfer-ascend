/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#ifndef FLASHINFER_SPARSE_INDEXER_BLOCK_REDUCE_RUNTIME_HPP
#define FLASHINFER_SPARSE_INDEXER_BLOCK_REDUCE_RUNTIME_HPP

#include <cstdint>

#include "sparse_indexer/block/mask/runtime.hpp"

namespace SparseIndexer::Common::Block {

template <class Block_>
struct ReduceRuntimeParams {
    using Block = Block_;
    typename Block::Params block{};
    MaskRuntimeParams mask{};
};

} // namespace SparseIndexer::Common::Block

#endif // FLASHINFER_SPARSE_INDEXER_BLOCK_REDUCE_RUNTIME_HPP
