#pragma once
#include <cosmico/recording/SnapshotFormat.h>
#include <string>
#include <vector>
#include <fstream>
#include <cstdint>

namespace cosmico {

class SnapshotPlayer {
public:
    bool open(const std::string& path);
    void close();

    bool loadFrame(uint32_t frameIndex);
    void uploadToCuda(void* cudaPtr);

    const void* frameData() const { return m_hostBuffer.data(); }
    bool isOpen() const { return m_open; }
    uint32_t frameCount() const { return m_header.frameCount; }
    uint32_t particleCount() const { return m_header.particleCount; }
    uint32_t currentFrame() const { return m_currentFrame; }
    uint32_t backend() const { return m_header.backend; }
    double frameSimTime(uint32_t i) const;
    uint64_t frameSimStep(uint32_t i) const;

private:
    std::ifstream m_file;
    SnapshotFileHeader m_header{};
    std::vector<FrameIndexEntry> m_frameIndex;
    std::vector<uint8_t> m_hostBuffer;
    bool m_open = false;
    uint32_t m_currentFrame = 0;
};

} // namespace cosmico
