#pragma once
#include <cosmico/simulation/SimulationParams.h>
#include <cosmico/simulation/InitialConditions.h>
#include <cosmico/simulation/InflationParams.h>
#include <cosmico/simulation/PMParams.h>
#include <cosmico/simulation/CDT2DParams.h>
#include <cosmico/simulation/CDT3DParams.h>
#include <volk.h>
#include <memory>
#include <string>
#include <cstdint>

namespace cosmico {

class VkContext;
class ParticleSystem;
class NBodyCompute;
class BarnesHutCompute;
class PMCompute;
class CudaInterop;
class InflationCompute;
class CDT2DCompute;
class CDT3DCompute;
class SimGraph;
class SimGraphCompiler;
class SimGraphExecutor;
class NodeEditorUI;
struct CompiledPlan;
struct InflationStateData;
struct PMStateData;
struct CDT2DStateData;
struct CDT3DStateData;

enum class ComputeBackend {
    BruteForce,   // Vulkan compute O(N²)
    BarnesHut,    // CUDA Barnes-Hut O(N log N)
    PM,           // CUDA Particle-Mesh O(N + M log M)
    Inflation,    // CUDA inflation lattice (cuFFT)
    CDT2D,        // CPU Monte Carlo CDT 2D
    CDT3D,        // CUDA CDT 3D (2+1D)
    NodeGraph,    // CUDA node-based pipeline
    Count
};

const char* computeBackendName(ComputeBackend b);

class Simulation {
public:
    Simulation();
    ~Simulation();

    void init(VkContext& ctx, const std::string& shaderDir);
    void destroy(VkContext& ctx);

    void reset(VkContext& ctx, InitialCondition ic);

    // Brute-force step (records into Vulkan command buffer)
    void step(VkCommandBuffer cmd);

    // Barnes-Hut step (launches CUDA kernels asynchronously)
    void stepCuda();
    // Sync CUDA stream — call once before Vulkan render, not per sub-step
    bool syncCuda();

    // Inflation step (CUDA lattice simulation)
    void stepInflation();
    // Reset inflation field
    void resetInflation();

    ParticleSystem& particleSystem() { return *m_particles; }
    SimulationParams& params() { return m_params; }
    void updateParams(VkContext& ctx);

    InitialCondition currentCondition() const { return m_currentIC; }

    // Backend management
    ComputeBackend backend() const { return m_backend; }
    void setBackend(VkContext& ctx, ComputeBackend backend);
    bool cudaAvailable() const { return m_cudaAvailable; }

    // Timeline semaphore for Vulkan↔CUDA sync
    VkSemaphore timelineSemaphore() const { return m_timelineSemaphore; }
    uint64_t currentSemaphoreValue() const { return m_semaphoreValue; }
    void advanceSemaphoreValue() { m_semaphoreValue++; }

    // Raw CUDA particle pointer (for recording/playback)
    void* cudaParticlePtr() const { return m_cudaParticlePtr; }

    // Get CUDA kernel stats
    float cudaKernelTimeMs() const;
    int cudaTreeNodeCount() const;

    // PM backend
    PMParams& pmParams() { return m_pmParams; }
    void stepPM();
    void resetPM(VkContext& ctx, InitialCondition ic);
    const PMStateData* pmState() const;
    float pmKernelTimeMs() const;

    // Inflation backend
    InflationParams& inflationParams() { return m_inflationParams; }
    const InflationStateData* inflationState() const;
    float inflationKernelTimeMs() const;
    void extractDensityVolume(uint8_t* hostDst, int targetN);
    void extractCMBMap(uint8_t* hostDst, int mapW, int mapH);

    // Zel'dovich structure formation
    void startZeldovich();
    void stepZeldovich();
    void extractZeldovichDensity(uint8_t* hostDst, int targetN);
    bool zeldovichActive() const;
    float zeldovichGrowth() const;
    void stopZeldovich();

    // CDT 2D backend
    CDT2DParams& cdt2dParams() { return m_cdt2dParams; }
    void stepCDT2D();
    void resetCDT2D();
    const CDT2DStateData* cdt2dState() const;

    // CDT 3D backend (CUDA)
    CDT3DParams& cdt3dParams() { return m_cdt3dParams; }
    void stepCDT3D();
    void resetCDT3D();
    const CDT3DStateData* cdt3dState() const;
    float cdt3dKernelTimeMs() const;

    // Node Graph backend
    void stepNodeGraph();
    void resetNodeGraph(VkContext& ctx, InitialCondition ic);
    SimGraph& simGraph();
    SimGraphExecutor& graphExecutor();
    NodeEditorUI& nodeEditorUI();
    const CompiledPlan* compiledPlan() const;
    void buildNodeGraph();
    void populateDefaultPMGraph();
    void populateDefaultNBodyGraph();

private:
    void initCuda(VkContext& ctx);
    void destroyCuda(VkContext& ctx);

    std::unique_ptr<ParticleSystem> m_particles;
    std::unique_ptr<NBodyCompute> m_compute;

    // CUDA Barnes-Hut
    std::unique_ptr<BarnesHutCompute> m_bhCompute;
    std::unique_ptr<CudaInterop> m_cudaInterop;
    void* m_cudaParticlePtr = nullptr;
    bool m_cudaAvailable = false;

    // CUDA PM
    std::unique_ptr<PMCompute> m_pmCompute;
    PMParams m_pmParams;

    // CUDA Inflation
    std::unique_ptr<InflationCompute> m_inflationCompute;
    InflationParams m_inflationParams;

    // CDT 2D (CPU)
    std::unique_ptr<CDT2DCompute> m_cdt2dCompute;
    CDT2DParams m_cdt2dParams;

    // CDT 3D (CUDA)
    std::unique_ptr<CDT3DCompute> m_cdt3dCompute;
    CDT3DParams m_cdt3dParams;

    // Node Graph
    std::unique_ptr<SimGraph> m_simGraph;
    std::unique_ptr<SimGraphCompiler> m_graphCompiler;
    std::unique_ptr<SimGraphExecutor> m_graphExecutor;
    std::unique_ptr<NodeEditorUI> m_nodeEditorUI;
    std::unique_ptr<CompiledPlan> m_compiledPlan;

    // Timeline semaphore for Vulkan↔CUDA synchronization
    VkSemaphore m_timelineSemaphore = VK_NULL_HANDLE;
    uint64_t m_semaphoreValue = 0;

    SimulationParams m_params;
    InitialCondition m_currentIC = InitialCondition::Disk;
    ComputeBackend m_backend = ComputeBackend::BruteForce;
    std::string m_shaderDir;
};

} // namespace cosmico
