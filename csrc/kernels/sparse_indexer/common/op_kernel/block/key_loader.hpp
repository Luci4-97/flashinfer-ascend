/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 */

#ifndef FLASHINFER_GENERAL_INDEXER_BLOCK_KEY_LOADER_HPP
#define FLASHINFER_GENERAL_INDEXER_BLOCK_KEY_LOADER_HPP

#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/matrix_coord.hpp"
#include "kernel_operator.h"

#include "sparse_indexer_align.hpp"

namespace SparseIndexer::Common::Block {

/**
 * Contiguous Key → L1B (default path).
 * Load uses logical (kPart, nStart) into ColumnMajor GM [D, S2].
 */
template <class ArchTag_, class BType_>
class ContiguousKeyLoader {
public:
    static constexpr uint32_t kMaxBatch = SparseIndexer::Common::kMaxBatch;

    struct Params {};

    static constexpr bool kPaged = false;
    using ArchTag = ArchTag_;
    using BType = BType_;
    using DtypeB = typename BType::Element;
    using LayoutB = typename BType::Layout;
    using CopyGmToL1B = Catlass::Gemm::Tile::CopyGmToL1DynamicOptimized<ArchTag, BType>;

    CATLASS_DEVICE
    ContiguousKeyLoader() = default;

    CATLASS_DEVICE
    void Setup(
        AscendC::GlobalTensor<DtypeB> const& gmB, LayoutB const& layoutB,
        Params const& /*params*/,
        AscendC::GlobalTensor<int32_t> const& /*gmTable*/,
        AscendC::GlobalTensor<DtypeB> const& /*gmZero*/)
    {
        gmB_ = gmB;
        layoutB_ = layoutB;
    }

    template <class TensorL1, class LayoutL1>
    CATLASS_DEVICE void Load(
        TensorL1 const& l1B, LayoutL1 const& layoutL1, uint32_t kPartStart, uint32_t actualK, uint32_t nStart,
        uint32_t actualN, uint32_t /*batchIdx*/)
    {
        int64_t offsetB = layoutB_.GetOffset(Catlass::MatrixCoord(kPartStart, nStart));
        copyGmToL1B_(
            l1B, gmB_[offsetB], layoutL1, layoutB_.GetTileLayout(Catlass::MatrixCoord(actualK, actualN)));
    }

private:
    AscendC::GlobalTensor<DtypeB> gmB_;
    LayoutB layoutB_{};
    CopyGmToL1B copyGmToL1B_;
};

/**
 * PA_BBND Key → L1B via block_table (M4.1 / D3).
 *
 * Geometry:
 * - Pages GM is [numPages, blockSize, D] (row-major); logical n is request-local
 *   via ``nStart - cuSeqlensK[batchIdx]``.
 * - Each Load splits ``actualN`` into page-aligned runs
 *   ``[offInPage, blockSize - offInPage]`` so a single s2Tile may cross pages.
 * - Illegal / OOB page ids (``<0`` or ``>= numPages``) copy from a host-zeroed
 *   page buffer (ColumnMajor(d, blockSize) view starting at token 0).
 *
 * Note: production gather quality (batched page copy / less zeroing) is optional
 * M5 work; this loader prioritizes correct semantics over peak PA bandwidth.
 */
template <class ArchTag_, class BType_>
class PagedKeyLoader {
public:
    static constexpr uint32_t kMaxBatch = SparseIndexer::Common::kMaxBatch;

    struct Params {
        uint32_t d{0};
        uint32_t blockSize{0};
        uint32_t maxBlocksPerBatch{0};
        uint32_t numPages{0};
        uint32_t batch{1};
        int32_t cuK[kMaxBatch + 1]{0};
    };

    static constexpr bool kPaged = true;
    using ArchTag = ArchTag_;
    using BType = BType_;
    using DtypeB = typename BType::Element;
    using LayoutB = typename BType::Layout;
    using CopyGmToL1B = Catlass::Gemm::Tile::CopyGmToL1DynamicOptimized<ArchTag, BType>;

    CATLASS_DEVICE
    PagedKeyLoader() = default;

    CATLASS_DEVICE
    void Setup(
        AscendC::GlobalTensor<DtypeB> const& gmPages, LayoutB const& /*layoutB*/,
        Params const& params,
        AscendC::GlobalTensor<int32_t> const& gmTable,
        AscendC::GlobalTensor<DtypeB> const& gmZero)
    {
        gmPages_ = gmPages;
        blockSize_ = params.blockSize;
        maxBlocks_ = params.maxBlocksPerBatch;
        numPages_ = params.numPages;
        d_ = params.d;
        batch_ = params.batch == 0 ? 1U : params.batch;
        gmTable_ = gmTable;
        gmZero_ = gmZero;
        for (uint32_t i = 0; i <= batch_ && i <= kMaxBatch; ++i) {
            cuK_[i] = params.cuK[i];
        }
    }

    template <class TensorL1, class LayoutL1>
    CATLASS_DEVICE void Load(
        TensorL1 const& l1B, LayoutL1 const& layoutL1, uint32_t kPartStart, uint32_t actualK, uint32_t nStart,
        uint32_t actualN, uint32_t batchIdx)
    {
        uint32_t localN = nStart;
        if (batchIdx < batch_) {
            localN = nStart - static_cast<uint32_t>(cuK_[batchIdx]);
        }

        uint32_t copyCol = 0;
        while (copyCol < actualN) {
            uint32_t tok = localN + copyCol;
            uint32_t logicalBlock = (blockSize_ == 0) ? 0U : (tok / blockSize_);
            uint32_t offInPage = (blockSize_ == 0) ? 0U : (tok % blockSize_);
            uint32_t run = actualN - copyCol;
            if (blockSize_ > offInPage) {
                run = AscendC::Std::min(run, blockSize_ - offInPage);
            }

            int32_t pageId = -1;
            if (batchIdx < batch_ && logicalBlock < maxBlocks_) {
                pageId = gmTable_.GetValue(static_cast<int64_t>(batchIdx) * maxBlocks_ + logicalBlock);
            }

            AscendC::GlobalTensor<DtypeB> src;
            if (pageId >= 0 && static_cast<uint32_t>(pageId) < numPages_) {
                int64_t srcOff = (static_cast<int64_t>(pageId) * blockSize_ + offInPage) * d_ + kPartStart;
                src = gmPages_[srcOff];
            } else {
                // Illegal page → zeros (zero page is ColumnMajor(d, blockSize) starting at 0).
                src = gmZero_[static_cast<int64_t>(kPartStart)];
            }

            int64_t dstOff = layoutL1.GetOffset(Catlass::MatrixCoord(0, copyCol));
            auto layoutSrc = LayoutB::template MakeLayout<DtypeB>(d_, run);
            copyGmToL1B_(
                l1B[dstOff], src, layoutL1, layoutSrc.GetTileLayout(Catlass::MatrixCoord(actualK, run)));
            copyCol += run;
        }
    }

private:
    AscendC::GlobalTensor<DtypeB> gmPages_;
    AscendC::GlobalTensor<int32_t> gmTable_;
    AscendC::GlobalTensor<DtypeB> gmZero_;
    LayoutB layoutB_{};
    CopyGmToL1B copyGmToL1B_;
    uint32_t blockSize_{0};
    uint32_t maxBlocks_{0};
    uint32_t numPages_{0};
    uint32_t d_{0};
    uint32_t batch_{1};
    int32_t cuK_[kMaxBatch + 1]{0};
};

} // namespace SparseIndexer::Common::Block

#endif // FLASHINFER_GENERAL_INDEXER_BLOCK_KEY_LOADER_HPP
