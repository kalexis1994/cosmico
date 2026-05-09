#include <cosmico/simulation/BarnesHutCompute.h>
#include <kernels/barnes_hut.cuh>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>

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

    // Diagnostics: device accumulator + pinned host mirror + completion event
    CUDA_CHECK(cudaMalloc(&m_diagAccumDev, sizeof(cuda::DiagnosticsAccum)));
    CUDA_CHECK(cudaMallocHost(&m_diagAccumHost, sizeof(cuda::DiagnosticsAccum)));
    cudaEvent_t diagEvent;
    CUDA_CHECK(cudaEventCreate(&diagEvent));
    m_diagEvent = diagEvent;

    // Create CUDA events for kernel timing
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
    if (m_diagEvent) {
        cudaEventDestroy(static_cast<cudaEvent_t>(m_diagEvent));
        m_diagEvent = nullptr;
    }
    if (m_diagAccumHost) {
        cudaFreeHost(m_diagAccumHost);
        m_diagAccumHost = nullptr;
    }
    if (m_diagAccumDev) {
        cudaFree(m_diagAccumDev);
        m_diagAccumDev = nullptr;
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
    m_diagPending = false;
    m_baselineCaptured = false;
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

void BarnesHutCompute::resetDiagnostics() {
    m_stats.stepCount = 0;
    m_stats.simTime = 0.0;
    m_stats.diagValid = false;
    m_stats.kineticEnergy = 0.0;
    m_stats.potentialEnergy = 0.0;
    m_stats.totalEnergy = 0.0;
    m_stats.initialEnergy = 0.0;
    m_stats.energyDrift = 0.0;
    m_stats.momentum[0] = m_stats.momentum[1] = m_stats.momentum[2] = 0.0;
    m_stats.momentumMag = 0.0;
    m_stats.centerOfMass[0] = m_stats.centerOfMass[1] = m_stats.centerOfMass[2] = 0.0;
    m_stats.totalMass = 0.0;
    m_diagPending = false;
    m_baselineCaptured = false;
}

void BarnesHutCompute::step(const SimulationParams& params, void* cudaStream) {
    if (!m_particlePtr || m_particleCount == 0) return;

    cudaStream_t stream = static_cast<cudaStream_t>(cudaStream);

    // ── Consume pending diagnostics readback from a previous step ──
    if (m_diagPending) {
        if (cudaEventQuery(static_cast<cudaEvent_t>(m_diagEvent)) == cudaSuccess) {
            const auto* accum = static_cast<const cuda::DiagnosticsAccum*>(m_diagAccumHost);
            m_stats.kineticEnergy   = accum->ke;
            m_stats.potentialEnergy = accum->pe * 0.5;       // tree walk double-counts pairs
            m_stats.totalEnergy     = m_stats.kineticEnergy + m_stats.potentialEnergy;
            m_stats.momentum[0]     = accum->px;
            m_stats.momentum[1]     = accum->py;
            m_stats.momentum[2]     = accum->pz;
            m_stats.momentumMag     = std::sqrt(accum->px * accum->px +
                                                accum->py * accum->py +
                                                accum->pz * accum->pz);
            m_stats.totalMass       = accum->m;
            if (accum->m > 0.0) {
                m_stats.centerOfMass[0] = accum->mx / accum->m;
                m_stats.centerOfMass[1] = accum->my / accum->m;
                m_stats.centerOfMass[2] = accum->mz / accum->m;
            }
            if (!m_baselineCaptured) {
                m_stats.initialEnergy = m_stats.totalEnergy;
                m_baselineCaptured = true;
            }
            double e0 = std::abs(m_stats.initialEnergy);
            m_stats.energyDrift = (e0 > 1e-30)
                ? (m_stats.totalEnergy - m_stats.initialEnergy) / e0 : 0.0;
            m_stats.diagValid = true;
            m_diagPending = false;
        }
    }

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

    // ── Queue diagnostics on the same stream every diagInterval steps ──
    // The tree built by launchBarnesHutStep is reused; drift is measured
    // post-step so what's reported matches the position/velocity that the
    // renderer just drew.
    if (!m_diagPending && (m_stats.stepCount % m_diagInterval == 0)) {
        cudaMemsetAsync(m_diagAccumDev, 0, sizeof(cuda::DiagnosticsAccum), stream);
        int rootIdx = m_particleCount; // matches launchBarnesHutStep convention
        cuda::launchDiagnostics(
            static_cast<const cuda::ParticleGpu*>(m_particlePtr),
            static_cast<const cuda::OctreeNode*>(m_treeNodes),
            bhParams,
            rootIdx,
            static_cast<cuda::DiagnosticsAccum*>(m_diagAccumDev),
            stream
        );
        cudaMemcpyAsync(m_diagAccumHost, m_diagAccumDev,
                        sizeof(cuda::DiagnosticsAccum),
                        cudaMemcpyDeviceToHost, stream);
        CUDA_CHECK(cudaEventRecord(static_cast<cudaEvent_t>(m_diagEvent), stream));
        m_diagPending = true;
    }

    m_stats.stepCount++;
    m_stats.simTime += static_cast<double>(params.dt);

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
