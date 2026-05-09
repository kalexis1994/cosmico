#include <cosmico/recording/SnapshotRecorder.h>
#include <cstdio>
#include <cstring>

namespace cosmico {

// Defined in SnapshotCuda.cu
void snapshotReadbackCuda(void* hostDst, const void* devicePtr, size_t bytes);

bool SnapshotRecorder::start(const std::string& path, uint32_t particleCount,
                             uint32_t captureStride, uint32_t backend) {
    if (m_recording) stop();

    m_file.open(path, std::ios::binary | std::ios::trunc);
    if (!m_file.is_open()) {
        fprintf(stderr, "[SnapshotRecorder] Failed to open %s\n", path.c_str());
        return false;
    }

    m_header = {};
    m_header.magic = SNAPSHOT_MAGIC;
    m_header.version = SNAPSHOT_VERSION;
    m_header.particleCount = particleCount;
    m_header.particleStride = 48;
    m_header.frameCount = 0;
    m_header.captureStride = captureStride;
    m_header.backend = backend;
    std::memset(m_header.reserved, 0, sizeof(m_header.reserved));

    // Write placeholder header
    m_file.write(reinterpret_cast<const char*>(&m_header), sizeof(m_header));

    m_frameIndex.clear();
    m_hostBuffer.resize(static_cast<size_t>(particleCount) * 48);
    m_bytesWritten = sizeof(m_header);
    m_recording = true;

    fprintf(stderr, "[SnapshotRecorder] Recording started: %s (%u particles, stride %u)\n",
            path.c_str(), particleCount, captureStride);
    return true;
}

void SnapshotRecorder::stop() {
    if (!m_recording) return;
    m_recording = false;

    // Write frame index table at current position
    uint64_t indexOffset = m_bytesWritten;
    for (const auto& entry : m_frameIndex) {
        m_file.write(reinterpret_cast<const char*>(&entry), sizeof(entry));
    }

    // Update header with final frame count
    m_header.frameCount = static_cast<uint32_t>(m_frameIndex.size());
    m_file.seekp(0);
    m_file.write(reinterpret_cast<const char*>(&m_header), sizeof(m_header));

    m_file.close();
    fprintf(stderr, "[SnapshotRecorder] Recording stopped: %u frames, %llu bytes\n",
            m_header.frameCount, static_cast<unsigned long long>(m_bytesWritten));
}

void SnapshotRecorder::captureFrame(uint64_t simStep, double simTime, void* cudaPtr) {
    if (!m_recording || !cudaPtr) return;

    // Skip frames based on capture stride
    if (m_header.captureStride > 1 && (simStep % m_header.captureStride) != 0) return;

    size_t frameBytes = static_cast<size_t>(m_header.particleCount) * m_header.particleStride;

    // Readback from GPU
    snapshotReadbackCuda(m_hostBuffer.data(), cudaPtr, frameBytes);

    // Record index entry
    FrameIndexEntry entry{};
    entry.fileOffset = m_bytesWritten;
    entry.simStep = simStep;
    entry.simTime = simTime;
    m_frameIndex.push_back(entry);

    // Write frame data
    m_file.write(reinterpret_cast<const char*>(m_hostBuffer.data()),
                 static_cast<std::streamsize>(frameBytes));
    m_bytesWritten += frameBytes;
}

} // namespace cosmico
