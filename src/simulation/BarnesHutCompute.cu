#include <cosmico/simulation/BarnesHutCompute.h>
#include <kernels/barnes_hut.cuh>
#include <cuda_runtime.h>
#include <stdexcept>
#include <cstdio>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "[BarnesHut] CUDA Error: %s at %s:%d\n", \
                    cudaGetErrorString(err), __FILE__, __LINE__); \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err)); \
        } \
    } while(0)

namespace cosmico {

void BarnesHutCompute::init(int maxParticles) {
    m_maxParticles = maxParticles;
    m_treeMemorySize = cuda::getTreeMemorySize(maxParticles);

    // Allocate octree node memory
    CUDA_CHECK(cudaMalloc(&m_treeNodes, m_treeMemorySize));

    // Allocate bounding box
    CUDA_CHECK(cudaMalloc(&m_bbox, sizeof(cuda::BoundingBox)));

    // Allocate node counter
    CUDA_CHECK(cudaMalloc(&m_nodeCounter, sizeof(int)));

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    m_startEvent = start;
    m_stopEvent = stop;

    fprintf(stderr, "[BarnesHut] Initialized: max %d particles, tree memory %zu MB\n",
            maxParticles, m_treeMemorySize / (1024 * 1024));
}

void BarnesHutCompute::destroy() {
    if (m_startEvent) {
        cudaEventDestroy(static_cast<cudaEvent_t>(m_startEvent));
        m_startEvent = nullptr;
    }
    if (m_stopEvent) {
        cudaEventDestroy(static_cast<cudaEvent_t>(m_stopEvent));
        m_stopEvent = nullptr;
    }
    if (m_nodeCounter) {
        cudaFree(m_nodeCounter);
        m_nodeCounter = nullptr;
    }
    if (m_bbox) {
        cudaFree(m_bbox);
        m_bbox = nullptr;
    }
    if (m_treeNodes) {
        cudaFree(m_treeNodes);
        m_treeNodes = nullptr;
    }
}

void BarnesHutCompute::setParticleBuffer(void* cudaParticlePtr, size_t bufferSize) {
    m_particlePtr = cudaParticlePtr;
    m_particleBufferSize = bufferSize;
}

void BarnesHutCompute::updateParticleCount(int count) {
    m_particleCount = count;

    // Reallocate tree if needed
    size_t needed = cuda::getTreeMemorySize(count);
    if (needed > m_treeMemorySize) {
        if (m_treeNodes) cudaFree(m_treeNodes);
        m_treeMemorySize = needed;
        CUDA_CHECK(cudaMalloc(&m_treeNodes, m_treeMemorySize));
        fprintf(stderr, "[BarnesHut] Reallocated tree: %zu MB\n", m_treeMemorySize / (1024 * 1024));
    }
}

void BarnesHutCompute::step(const SimulationParams& params, void* cudaStream) {
    if (!m_particlePtr || m_particleCount == 0) return;

    cudaStream_t stream = static_cast<cudaStream_t>(cudaStream);

    // Record start event
    CUDA_CHECK(cudaEventRecord(static_cast<cudaEvent_t>(m_startEvent), stream));

    // Build params
    cuda::BarnesHutParams bhParams;
    bhParams.G = params.G;
    bhParams.dt = params.dt;
    bhParams.softening = params.softening;
    bhParams.particleCount = m_particleCount;
    bhParams.theta = params.theta;
    bhParams.maxNodes = m_particleCount * 3; // Must match getTreeMemorySize

    // Launch the full pipeline
    cuda::launchBarnesHutStep(
        static_cast<cuda::ParticleGpu*>(m_particlePtr),
        static_cast<cuda::OctreeNode*>(m_treeNodes),
        static_cast<cuda::BoundingBox*>(m_bbox),
        static_cast<int*>(m_nodeCounter),
        bhParams,
        stream
    );

    // Record stop event
    CUDA_CHECK(cudaEventRecord(static_cast<cudaEvent_t>(m_stopEvent), stream));

    // Query timing from previous frame (non-blocking)
    float ms = 0.0f;
    cudaError_t err = cudaEventQuery(static_cast<cudaEvent_t>(m_stopEvent));
    if (err == cudaSuccess) {
        cudaEventElapsedTime(&ms, static_cast<cudaEvent_t>(m_startEvent),
                             static_cast<cudaEvent_t>(m_stopEvent));
        m_stats.kernelTimeMs = ms;
    }
    // If not ready yet, keep previous timing

    // Read node counter for stats (async, will be available next frame)
    int nodeCount = 0;
    cudaMemcpyAsync(&nodeCount, m_nodeCounter, sizeof(int), cudaMemcpyDeviceToHost, stream);
    m_stats.nodeCount = nodeCount;
}

} // namespace cosmico
