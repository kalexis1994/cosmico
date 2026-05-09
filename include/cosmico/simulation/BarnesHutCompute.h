#pragma once
#include <cosmico/simulation/SimulationParams.h>
#include <cstdint>
#include <cstddef>

namespace cosmico {

class CudaInterop;

struct BarnesHutStats {
    float kernelTimeMs = 0.0f;  // Total kernel execution time
    int nodeCount = 0;          // Number of tree nodes used
};

class BarnesHutCompute {
public:
    void init(int maxParticles);
    void destroy();

    // Set the particle buffer pointer (CUDA device pointer from interop)
    void setParticleBuffer(void* cudaParticlePtr, size_t bufferSize);

    // Execute one simulation step (all 6 kernels)
    void step(const SimulationParams& params, void* cudaStream);

    // Update particle count (called when particle count changes)
    void updateParticleCount(int count);

    const BarnesHutStats& stats() const { return m_stats; }

private:
    void* m_particlePtr = nullptr;     // CUDA device pointer to particle SSBO
    size_t m_particleBufferSize = 0;

    void* m_treeNodes = nullptr;       // CUDA device memory for octree nodes
    void* m_bbox = nullptr;            // CUDA device memory for bounding box
    void* m_nodeCounter = nullptr;     // CUDA device memory for atomic node counter

    void* m_startEvent = nullptr;      // cudaEvent_t for timing
    void* m_stopEvent = nullptr;       // cudaEvent_t for timing

    int m_maxParticles = 0;
    int m_particleCount = 0;
    size_t m_treeMemorySize = 0;

    BarnesHutStats m_stats;
};

} // namespace cosmico
