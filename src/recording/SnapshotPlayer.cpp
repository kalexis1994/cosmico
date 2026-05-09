#include <cosmico/recording/SnapshotPlayer.h>
#include <cstdio>

namespace cosmico {

// Defined in SnapshotCuda.cu
void snapshotUploadCuda(void* devicePtr, const void* hostSrc, size_t bytes);

bool SnapshotPlayer::open(const std::string& path) {
    if (m_open) close();

    m_file.open(path, std::ios::binary);
    if (!m_file.is_open()) {
        fprintf(stderr, "[SnapshotPlayer] Failed to open %s\n", path.c_str());
        return false;
    }

    // Read header
    m_file.read(reinterpret_cast<char*>(&m_header), sizeof(m_header));
    if (!m_file || m_header.magic != SNAPSHOT_MAGIC) {
        fprintf(stderr, "[SnapshotPlayer] Invalid snapshot file\n");
        m_file.close();
        return false;
    }
    if (m_header.version != SNAPSHOT_VERSION) {
        fprintf(stderr, "[SnapshotPlayer] Unsupported version %u\n", m_header.version);
        m_file.close();
        return false;
    }
    if (m_header.frameCount == 0) {
        fprintf(stderr, "[SnapshotPlayer] File has 0 frames\n");
        m_file.close();
        return false;
    }

    // Read frame index table (at end of file, after all frame data)
    size_t frameBytes = static_cast<size_t>(m_header.particleCount) * m_header.particleStride;
    uint64_t indexOffset = sizeof(SnapshotFileHeader) +
                           static_cast<uint64_t>(m_header.frameCount) * frameBytes;
    m_file.seekg(static_cast<std::streamoff>(indexOffset));

    m_frameIndex.resize(m_header.frameCount);
    m_file.read(reinterpret_cast<char*>(m_frameIndex.data()),
                static_cast<std::streamsize>(m_header.frameCount * sizeof(FrameIndexEntry)));
    if (!m_file) {
        fprintf(stderr, "[SnapshotPlayer] Failed to read frame index\n");
        m_file.close();
        return false;
    }

    m_hostBuffer.resize(frameBytes);
    m_currentFrame = 0;
    m_open = true;

    // Load first frame
    loadFrame(0);

    fprintf(stderr, "[SnapshotPlayer] Opened: %u frames, %u particles\n",
            m_header.frameCount, m_header.particleCount);
    return true;
}

void SnapshotPlayer::close() {
    if (!m_open) return;
    m_file.close();
    m_frameIndex.clear();
    m_hostBuffer.clear();
    m_open = false;
    m_currentFrame = 0;
    m_header = {};
}

bool SnapshotPlayer::loadFrame(uint32_t frameIndex) {
    if (!m_open || frameIndex >= m_header.frameCount) return false;

    const auto& entry = m_frameIndex[frameIndex];
    m_file.seekg(static_cast<std::streamoff>(entry.fileOffset));

    size_t frameBytes = static_cast<size_t>(m_header.particleCount) * m_header.particleStride;
    m_file.read(reinterpret_cast<char*>(m_hostBuffer.data()),
                static_cast<std::streamsize>(frameBytes));

    if (!m_file) {
        fprintf(stderr, "[SnapshotPlayer] Failed to read frame %u\n", frameIndex);
        m_file.clear(); // Clear error state
        return false;
    }

    m_currentFrame = frameIndex;
    return true;
}

void SnapshotPlayer::uploadToCuda(void* cudaPtr) {
    if (!m_open || !cudaPtr || m_hostBuffer.empty()) return;
    size_t frameBytes = static_cast<size_t>(m_header.particleCount) * m_header.particleStride;
    snapshotUploadCuda(cudaPtr, m_hostBuffer.data(), frameBytes);
}

double SnapshotPlayer::frameSimTime(uint32_t i) const {
    if (i >= m_frameIndex.size()) return 0.0;
    return m_frameIndex[i].simTime;
}

uint64_t SnapshotPlayer::frameSimStep(uint32_t i) const {
    if (i >= m_frameIndex.size()) return 0;
    return m_frameIndex[i].simStep;
}

} // namespace cosmico
