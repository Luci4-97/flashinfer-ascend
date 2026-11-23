/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 *
 * Compile-only IRS (indexer_region_score) capability fixture. It proves that
 * the public indexer contracts can describe the required variant matrix
 * without importing IRS sources or sharing QLI tiling/public ABI types.
 */

#ifndef FLASHINFER_TESTS_SPARSE_INDEXER_IRS_MOCK_CONFIG_HPP
#define FLASHINFER_TESTS_SPARSE_INDEXER_IRS_MOCK_CONFIG_HPP

#include <type_traits>

#include "sparse_indexer/foundation/contracts/paged_kv.hpp"
#include "sparse_indexer/foundation/contracts/policy.hpp"
#include "sparse_indexer/foundation/contracts/problem.hpp"

namespace SparseIndexer::Test::IrsMock {

struct ContinuousCache {
    static constexpr Common::IndexerCacheKind kind = Common::IndexerCacheKind::Continuous;
};
struct PagedCache {
    static constexpr Common::IndexerCacheKind kind = Common::IndexerCacheKind::Paged;
};

struct NoMask {
    static constexpr Common::IndexerMaskKind kind = Common::IndexerMaskKind::None;
};
struct Mask3 {
    static constexpr Common::IndexerMaskKind kind = Common::IndexerMaskKind::CompressedMask3;
};
struct RegionCausal {
    static constexpr Common::IndexerMaskKind kind = Common::IndexerMaskKind::RegionCausal;
};

struct TokenSelection {};
struct RegionSelection {};

struct IndicesOnly {
    static constexpr Common::IndexerOutputKind kind = Common::IndexerOutputKind::IndicesOnly;
};
struct ScoreAndIndices {
    static constexpr Common::IndexerOutputKind kind = Common::IndexerOutputKind::ScoreAndIndices;
};

struct ScheduleTag {};
struct MmadTag {};
struct SyncTag {};
struct ResourceTag {};

/// IRS keeps its own config and ABI while composing the public policy slots.
template <class Cache_, class Mask_, class Selection_, class Output_, uint32_t HeadDim_>
struct Variant : Common::PolicyBundle<ScheduleTag, Cache_, MmadTag, Mask_, Selection_, SyncTag, ResourceTag> {
    using Cache = Cache_;
    using Mask = Mask_;
    using Selection = Selection_;
    using Output = Output_;
    static constexpr uint32_t headDim = HeadDim_;
    static constexpr bool emitsScores =
        Output::kind == Common::IndexerOutputKind::ScoreAndIndices;
};

using D128PagedNoneRegionScore = Variant<PagedCache, NoMask, RegionSelection, ScoreAndIndices, 128U>;
using D128PagedMask3TokenIndices = Variant<PagedCache, Mask3, TokenSelection, IndicesOnly, 128U>;
using D128ContinuousRegionScore = Variant<ContinuousCache, RegionCausal, RegionSelection, ScoreAndIndices, 128U>;
using D256PagedMask3RegionScore = Variant<PagedCache, Mask3, RegionSelection, ScoreAndIndices, 256U>;
using D256ContinuousNoneTokenIndices = Variant<ContinuousCache, NoMask, TokenSelection, IndicesOnly, 256U>;

template <class Variant_>
struct Contract {
    static_assert(Common::IsPolicyBundle<Variant_>::value, "IRS variant must compose the public policy slots");
    static_assert(Variant_::headDim == 128U || Variant_::headDim == 256U,
        "IRS mock must cover both supported head dimensions");
    static_assert(std::is_same_v<typename Variant_::Cache, PagedCache> ||
        std::is_same_v<typename Variant_::Cache, ContinuousCache>, "cache policy must be explicit");
    static_assert(std::is_same_v<typename Variant_::Mask, NoMask> ||
        std::is_same_v<typename Variant_::Mask, Mask3> ||
        std::is_same_v<typename Variant_::Mask, RegionCausal>, "mask policy must be explicit");
    static_assert(std::is_same_v<typename Variant_::Selection, TokenSelection> ||
        std::is_same_v<typename Variant_::Selection, RegionSelection>, "selection policy must be explicit");
    static_assert(std::is_same_v<typename Variant_::Output, IndicesOnly> ||
        std::is_same_v<typename Variant_::Output, ScoreAndIndices>, "output policy must be explicit");
};

using Contracts = std::tuple<
    Contract<D128PagedNoneRegionScore>,
    Contract<D128PagedMask3TokenIndices>,
    Contract<D128ContinuousRegionScore>,
    Contract<D256PagedMask3RegionScore>,
    Contract<D256ContinuousNoneTokenIndices>>;

} // namespace SparseIndexer::Test::IrsMock

#endif // FLASHINFER_TESTS_SPARSE_INDEXER_IRS_MOCK_CONFIG_HPP
