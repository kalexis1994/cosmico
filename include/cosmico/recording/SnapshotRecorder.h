#pragma once
#include <cosmico/recording/SnapshotFormat.h>
#include <string>
#include <vector>
#include <fstream>
#include <cstdint>

namespace cosmico {

class SnapshotRecorder {
public:
    bool start(const std::string& path, uint32_t particleCount,
               uint32_t captureStride, uint32_t backend);
    void stop();
    void captureFrame(uint64_t simStep, double simTime, void* cudaPtr);
    bool isRecording() const { return m_recording; }
    uint32_t framesCaptured() const { return static_cast<uint32_t>(m_frameIndex.size()); }
    uint64_t bytesWritten() const { return m_bytesWritten; }

private:
    std::ofstream m_file;
    std::vector<FrameIndexEntry> m_frameIndex;
    std::vector<uint8_t> m_hostBuffer;
    SnapshotFileHeader m_header{};
    bool m_recording = false;
    uint64_t m_bytesWritten = 0;
};

} // namespace cosmico
