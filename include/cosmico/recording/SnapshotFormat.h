#pragma once
#include <cstdint>

namespace cosmico {

static constexpr uint64_t SNAPSHOT_MAGIC = 0x50414E53534D4F43ULL; // "COSMSNAP"
static constexpr uint32_t SNAPSHOT_VERSION = 1;

struct SnapshotFileHeader {
    uint64_t magic;
    uint32_t version;
    uint32_t particleCount;
    uint32_t particleStride;   // 48
    uint32_t frameCount;
    uint32_t captureStride;
    uint32_t backend;
    uint8_t  reserved[28];
};
static_assert(sizeof(SnapshotFileHeader) == 64, "Header must be 64 bytes");

struct FrameIndexEntry {
    uint64_t fileOffset;
    uint64_t simStep;
    double   simTime;
};
static_assert(sizeof(FrameIndexEntry) == 24, "FrameIndexEntry must be 24 bytes");

} // namespace cosmico
