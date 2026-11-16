#include <cstdint>
#include <cstdio>

// Host-side stand-ins for Ascend C block builtins used by the scheduler.
uint32_t block_idx = 0;
uint32_t block_num = 1;

#include "block/block_scheduler.hpp"

struct FakeMetadataA {
    uint32_t batch = 2;
    int32_t token_offsets_q[3] = {0, 4, 7};
    int32_t token_offsets_k[3] = {0, 6, 8};
};

struct FakeRequestSourceA {
    FakeMetadataA metadata;

    uint32_t Batch() const { return metadata.batch; }
    uint32_t QBegin(uint32_t b) const { return static_cast<uint32_t>(metadata.token_offsets_q[b]); }
    uint32_t QLen(uint32_t b) const { return QBegin(b + 1) - QBegin(b); }
    uint32_t KBegin(uint32_t b) const { return static_cast<uint32_t>(metadata.token_offsets_k[b]); }
    uint32_t KLen(uint32_t b) const { return KBegin(b + 1) - KBegin(b); }
};

struct FakeMetadataB {
    uint32_t requests = 2;
    uint64_t query_prefix[3] = {0, 1, 3};
    uint64_t key_prefix[3] = {0, 8, 9};
};

struct FakeRequestSourceB {
    FakeMetadataB different_schema;

    uint32_t Batch() const { return different_schema.requests; }
    uint32_t QBegin(uint32_t b) const { return static_cast<uint32_t>(different_schema.query_prefix[b]); }
    uint32_t QLen(uint32_t b) const { return QBegin(b + 1) - QBegin(b); }
    uint32_t KBegin(uint32_t b) const { return static_cast<uint32_t>(different_schema.key_prefix[b]); }
    uint32_t KLen(uint32_t b) const { return KBegin(b + 1) - KBegin(b); }
};

using SchedulerA = SparseIndexer::Common::Block::BlockScheduler<
    FakeRequestSourceA, SparseIndexer::Common::Block::RequestBlockDiagonalPolicy>;
using SchedulerB = SparseIndexer::Common::Block::BlockScheduler<
    FakeRequestSourceB, SparseIndexer::Common::Block::RequestBlockDiagonalPolicy>;

static bool CheckA()
{
    SchedulerA::Params params{};
    params.policy.s1Tile = 4;
    params.policy.s2Tile = 4;
    SchedulerA scheduler(params);
    SparseIndexer::Common::Block::ScheduleCoord coord{};
    const uint32_t expected[][5] = {
        {4, 0, 0, 0, 4},
        {4, 0, 0, 4, 2},
        {3, 4, 0, 6, 2},
    };
    uint32_t index = 0;
    while (scheduler.GetNextCoord(coord)) {
        if (index >= 3 || coord.actualQ != expected[index][0] || coord.qStart != expected[index][1] ||
            coord.localQStart != expected[index][2] || coord.kStart != expected[index][3] ||
            coord.actualK != expected[index][4]) {
            return false;
        }
        ++index;
    }
    if (index != 3) {
        return false;
    }
    const uint32_t expectedTopK[][5] = {
        {4, 0, 0, 0, 6},
        {3, 4, 0, 6, 2},
    };
    index = 0;
    while (scheduler.GetNextTopKCoord(coord)) {
        if (index >= 2 || coord.actualQ != expectedTopK[index][0] || coord.qStart != expectedTopK[index][1] ||
            coord.localQStart != expectedTopK[index][2] || coord.kStart != expectedTopK[index][3] ||
            coord.actualK != expectedTopK[index][4]) {
            return false;
        }
        ++index;
    }
    return index == 2;
}

static bool CheckB()
{
    SchedulerB::Params params{};
    params.policy.s1Tile = 1;
    params.policy.s2Tile = 4;
    SchedulerB scheduler(params);
    SparseIndexer::Common::Block::ScheduleCoord coord{};
    const uint32_t expected[][5] = {
        {1, 0, 0, 0, 4},
        {1, 0, 0, 4, 4},
        {1, 1, 0, 8, 1},
        {1, 2, 1, 8, 1},
    };
    uint32_t count = 0;
    while (scheduler.GetNextCoord(coord)) {
        if (count >= 4 || coord.actualQ != expected[count][0] || coord.qStart != expected[count][1] ||
            coord.localQStart != expected[count][2] || coord.kStart != expected[count][3] ||
            coord.actualK != expected[count][4]) {
            return false;
        }
        ++count;
    }
    return count == 4;
}

int main()
{
    if (!CheckA()) {
        return 1;
    }
    if (!CheckB()) {
        return 2;
    }
    SchedulerA::Params params{};
    params.policy.s1Tile = 4;
    params.policy.s2Tile = 4;
    SchedulerA scheduler(params);
    SparseIndexer::Common::Block::ScheduleCoord coord{};
    while (scheduler.GetNextCoord(coord)) {
        std::printf("%u:%u:%u:%u:%u\n", coord.actualQ, coord.qStart, coord.localQStart, coord.kStart, coord.actualK);
    }
    while (scheduler.GetNextTopKCoord(coord)) {
        std::printf("%u:%u:%u:%u:%u\n", coord.actualQ, coord.qStart, coord.localQStart, coord.kStart, coord.actualK);
    }
    return 0;
}
