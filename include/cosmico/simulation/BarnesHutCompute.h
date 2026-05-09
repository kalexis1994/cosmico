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

    // Increments each time a new diagnostics sample is consumed by the
    // CPU. UI uses this to push history points only when there's a new
    // value, instead of duplicating the last sample every frame.
    uint64_t diagSampleCount = 0;
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

    // Diagnostics runs on its OWN cuda stream so the Vulkan render path
    // (which syncs the simulation stream every frame) is never blocked
    // by the PE tree-walk. Cross-stream dependencies via two events:
    //   m_treeBuiltEvent — sim stream records it after each step;
    //                      diag stream waits on it before walking the tree.
    //   m_diagEvent      — diag stream records it after memcpy completes;
    //                      CPU polls for the result, and the next sim step
    //                      queues a wait on it before clearing the tree
    //                      (so we never overwrite a tree being walked).
    void* m_diagStream = nullptr;      // cudaStream_t (separate from sim)
    void* m_diagAccumDev = nullptr;    // cuda::DiagnosticsAccum (device)
    void* m_diagAccumHost = nullptr;   // cuda::DiagnosticsAccum (pinned host)
    void* m_diagEvent = nullptr;       // cudaEvent_t — diag complete
    void* m_treeBuiltEvent = nullptr;  // cudaEvent_t — sim step done
    bool m_diagPending = false;        // diag launched, not yet consumed
    bool m_baselineCaptured = false;   // initialEnergy populated yet

    // Run diagnostics every Nth step. The PE kernel walks the octree at
    // a cost comparable to forceKernel, so frequent runs cause periodic
    // micro-stutters. 30 gives ~2 samples/sec at 60fps — plenty for a
    // slowly-changing drift indicator without disturbing framerate.
    int m_diagInterval = 30;

    int m_maxParticles = 0;
    int m_particleCount = 0;
    size_t m_treeMemorySize = 0;

    BarnesHutStats m_stats;
};

} // namespace cosmico
