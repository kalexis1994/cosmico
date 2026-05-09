#pragma once
#include <cosmico/simulation/SimulationParams.h>
#include <cstdint>
#include <cstddef>

namespace cosmico {

class CudaInterop;

struct BarnesHutStats {
    float kernelTimeMs = 0.0f;  // Total kernel execution time
    int nodeCount = 0;          // Number of tree nodes used

    // ── Conservation diagnostics (computed every diagInterval steps) ──
    // Updated with one-frame lag relative to particle state because the
    // device→host copy is queued on the same stream as the simulation.
    bool diagValid = false;
    double kineticEnergy = 0.0;
    double potentialEnergy = 0.0;
    double totalEnergy = 0.0;
    double initialEnergy = 0.0;       // captured on first valid sample
    double energyDrift = 0.0;         // (E - E_init) / |E_init|, unitless
    double momentum[3] = {0.0, 0.0, 0.0};
    double momentumMag = 0.0;
    double centerOfMass[3] = {0.0, 0.0, 0.0};
    double totalMass = 0.0;

    // Step / time bookkeeping (always valid)
    uint64_t stepCount = 0;
    double simTime = 0.0;
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

    // Re-baseline energy drift and reset the step/time counters. Call
    // this after a reset (new IC) so drift is measured from the new
    // initial state rather than the previous run's.
    void resetDiagnostics();

    const BarnesHutStats& stats() const { return m_stats; }

private:
    void* m_particlePtr = nullptr;     // CUDA device pointer to particle SSBO
    size_t m_particleBufferSize = 0;

    void* m_treeNodes = nullptr;       // CUDA device memory for octree nodes
    void* m_bbox = nullptr;            // CUDA device memory for bounding box
    void* m_nodeCounter = nullptr;     // CUDA device memory for atomic node counter

    void* m_startEvent = nullptr;      // cudaEvent_t for kernel timing
    void* m_stopEvent = nullptr;       // cudaEvent_t for kernel timing

    // Diagnostics buffers + completion event for async readback.
    void* m_diagAccumDev = nullptr;    // cuda::DiagnosticsAccum (device)
    void* m_diagAccumHost = nullptr;   // cuda::DiagnosticsAccum (pinned host)
    void* m_diagEvent = nullptr;       // cudaEvent_t recorded after memcpy
    bool m_diagPending = false;        // a readback is in flight
    bool m_baselineCaptured = false;   // initialEnergy populated yet

    int m_diagInterval = 4;            // run diagnostics every Nth step

    int m_maxParticles = 0;
    int m_particleCount = 0;
    size_t m_treeMemorySize = 0;

    BarnesHutStats m_stats;
};

} // namespace cosmico
