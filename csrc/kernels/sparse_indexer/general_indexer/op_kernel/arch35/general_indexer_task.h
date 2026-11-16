/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#ifndef FLASHINFER_GENERAL_INDEXER_ARCH35_TASK_H
#define FLASHINFER_GENERAL_INDEXER_ARCH35_TASK_H

#include <cstdint>

#include "block/block_scheduler.hpp"
#include "general_indexer_source.h"

namespace GeneralIndexerKernelDetail {

struct GeneralIndexerTask {
    SparseIndexer::Common::Block::ScheduleCoord coord{};
    uint32_t requestKStart{0};
    uint32_t activeQ{0};
    uint32_t activeK{0};
    int32_t residual{0};
    bool maskActive{false};
};

CATLASS_DEVICE inline GeneralIndexerTask MakeGeneralIndexerTask(
    SparseIndexer::Common::Block::ScheduleCoord const& coord, GeneralIndexerRequestSource const& source)
{
    GeneralIndexerTask task{};
    task.coord = coord;
    task.requestKStart = source.KBegin(coord.batchIdx);
    task.activeQ = source.ActiveQ(coord.batchIdx);
    task.activeK = source.ActiveK(coord.batchIdx);
    task.residual = source.Residual(coord.batchIdx);
    task.maskActive = source.MaskActive();
    return task;
}

} // namespace GeneralIndexerKernelDetail

#endif // FLASHINFER_GENERAL_INDEXER_ARCH35_TASK_H
