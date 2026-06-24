#include <cosmico/simulation/Simulation.h>
#include <cosmico/simulation/ParticleSystem.h>
#include <cosmico/simulation/NBodyCompute.h>
#include <cosmico/simulation/BarnesHutCompute.h>
#include <cosmico/simulation/PMCompute.h>
#include <cosmico/simulation/InflationCompute.h>
#include <cosmico/simulation/CDT2DCompute.h>
#include <cosmico/simulation/CDT3DCompute.h>
#include <cosmico/cuda/CudaInterop.h>
#include <cosmico/vulkan/VkContext.h>
#include <cosmico/vulkan/VkSync.h>
#include <cosmico/nodes/SimGraph.h>
#include <cosmico/nodes/SimGraphCompiler.h>
#include <cosmico/nodes/SimGraphExecutor.h>
#include <cosmico/nodes/NodeEditorUI.h>
#include <cstdio>

namespace cosmico {

const char* computeBackendName(ComputeBackend b) {
    switch (b) {
        case ComputeBackend::BruteForce: return "Brute Force (Vulkan)";
        case ComputeBackend::BarnesHut:  return "Barnes-Hut (CUDA)";
        case ComputeBackend::PM:         return "Particle-Mesh (CUDA)";
        case ComputeBackend::Inflation:  return "Inflation (CUDA)";
        case ComputeBackend::CDT2D:      return "CDT 2D (CPU)";
        case ComputeBackend::CDT3D:      return "CDT 3D (CUDA)";
        case ComputeBackend::NodeGraph:  return "Node Graph (CUDA)";
        default: return "Unknown";
    }
}

Simulation::Simulation() = default;
Simulation::~Simulation() = default;

void Simulation::init(VkContext& ctx, const std::string& shaderDir) {
    m_shaderDir = shaderDir;

    // Initialize with brute-force backend (standard VMA-allocated buffers)
    m_particles = std::make_unique<ParticleSystem>();
    m_particles->init(ctx, 65536);

    m_compute = std::make_unique<NBodyCompute>();
    m_compute->init(ctx, m_particles->computeSetLayout(), shaderDir);

    // Try to initialize CUDA if external memory is supported
    if (ctx.externalMemorySupported()) {
        initCuda(ctx);
    }

    reset(ctx, m_currentIC);
}

void Simulation::initCuda(VkContext& ctx) {
    try {
        m_cudaInterop = std::make_unique<CudaInterop>();
        m_cudaInterop->init();

        m_bhCompute = std::make_unique<BarnesHutCompute>();
        // Allocate for up to 1M particles
        m_bhCompute->init(1048576);

        // Create exportable timeline semaphore
        m_timelineSemaphore = createExportableTimelineSemaphore(ctx.device(), 0);
        m_semaphoreValue = 0;

        // Import timeline semaphore into CUDA
        HANDLE semHandle = getTimelineSemaphoreWin32Handle(ctx.device(), m_timelineSemaphore);
        m_cudaInterop->importTimelineSemaphore(semHandle);

        m_cudaAvailable = true;
        fprintf(stderr, "[Cosmico] CUDA Barnes-Hut backend available\n");
    } catch (const std::exception& e) {
        fprintf(stderr, "[Cosmico] CUDA init failed: %s (brute-force only)\n", e.what());
        m_cudaAvailable = false;
        m_cudaInterop.reset();
        m_bhCompute.reset();
    }
}

void Simulation::destroy(VkContext& ctx) {
    if (m_graphExecutor) {
        m_graphExecutor->destroy();
        m_graphExecutor.reset();
    }
    if (m_nodeEditorUI) {
        m_nodeEditorUI->destroy();
        m_nodeEditorUI.reset();
    }
    m_simGraph.reset();
    m_graphCompiler.reset();
    m_compiledPlan.reset();

    if (m_cdt3dCompute) {
        m_cdt3dCompute->destroy();
        m_cdt3dCompute.reset();
    }
    if (m_pmCompute) {
        m_pmCompute->destroy();
        m_pmCompute.reset();
    }
    if (m_inflationCompute) {
        m_inflationCompute->destroy();
        m_inflationCompute.reset();
    }
    destroyCuda(ctx);
    m_compute->destroy(ctx);
    m_particles->destroy(ctx);
}

void Simulation::destroyCuda(VkContext& ctx) {
    if (m_bhCompute) {
        m_bhCompute->destroy();
        m_bhCompute.reset();
    }
    if (m_cudaInterop) {
        m_cudaInterop->destroy();
        m_cudaInterop.reset();
    }
    if (m_timelineSemaphore) {
        vkDestroySemaphore(ctx.device(), m_timelineSemaphore, nullptr);
        m_timelineSemaphore = VK_NULL_HANDLE;
    }
    m_cudaParticlePtr = nullptr;
    m_cudaAvailable = false;
}

void Simulation::setBackend(VkContext& ctx, ComputeBackend backend) {
    if (backend == m_backend) return;
    if (backend == ComputeBackend::BarnesHut && !m_cudaAvailable) return;
    if (backend == ComputeBackend::PM && !m_cudaAvailable) return;
    if (backend == ComputeBackend::Inflation && !m_cudaAvailable) return;
    if (backend == ComputeBackend::CDT3D && !m_cudaAvailable) return;
    if (backend == ComputeBackend::NodeGraph && !m_cudaAvailable) return;

    vkDeviceWaitIdle(ctx.device());

    ComputeBackend oldBackend = m_backend;
    m_backend = backend;

    // Destroy Node Graph compute if leaving NodeGraph backend
    if (oldBackend == ComputeBackend::NodeGraph) {
        if (m_graphExecutor) {
            m_graphExecutor->destroy();
            m_graphExecutor.reset();
        }
        if (m_nodeEditorUI) {
            m_nodeEditorUI->destroy();
            m_nodeEditorUI.reset();
        }
        m_simGraph.reset();
        m_graphCompiler.reset();
        m_compiledPlan.reset();
    }

    // Destroy PM compute if leaving PM backend
    if (oldBackend == ComputeBackend::PM && m_pmCompute) {
        m_pmCompute->destroy();
        m_pmCompute.reset();
    }

    // Destroy inflation compute if leaving Inflation backend
    if (oldBackend == ComputeBackend::Inflation && m_inflationCompute) {
        m_inflationCompute->destroy();
        m_inflationCompute.reset();
    }

    // Destroy CDT2D compute if leaving CDT2D backend
    if (oldBackend == ComputeBackend::CDT2D) {
        m_cdt2dCompute.reset();
    }

    // Destroy CDT3D compute if leaving CDT3D backend
    if (oldBackend == ComputeBackend::CDT3D && m_cdt3dCompute) {
        m_cdt3dCompute->destroy();
        m_cdt3dCompute.reset();
    }

    // Initialize CDT2D compute if entering CDT2D backend
    if (backend == ComputeBackend::CDT2D) {
        m_cdt2dCompute = std::make_unique<CDT2DCompute>();
        m_cdt2dCompute->init(m_cdt2dParams);
        fprintf(stderr, "[Cosmico] Switched to CDT 2D backend\n");
        return;
    }

    // Initialize CDT3D compute if entering CDT3D backend
    if (backend == ComputeBackend::CDT3D) {
        try {
            m_cdt3dCompute = std::make_unique<CDT3DCompute>();
            m_cdt3dCompute->init(m_cdt3dParams);
            fprintf(stderr, "[Cosmico] Switched to CDT 3D backend\n");
        } catch (const std::exception& e) {
            fprintf(stderr, "[Cosmico] CDT 3D init failed: %s\n", e.what());
            m_cdt3dCompute.reset();
            m_backend = oldBackend;
        }
        return;
    }

    // Initialize PM compute if entering PM backend
    if (backend == ComputeBackend::PM) {
        try {
            // Need exportable buffers for CUDA access (same as Barnes-Hut)
            auto currentParams = m_params;

            m_compute->destroy(ctx);
            m_particles->destroy(ctx);

            m_particles = std::make_unique<ParticleSystem>();
            m_particles->initExportable(ctx, 4194304);  // 4M max

            m_compute = std::make_unique<NBodyCompute>();
            m_compute->init(ctx, m_particles->computeSetLayout(), m_shaderDir);

            HANDLE memHandle = m_particles->getExportableHandle(ctx.device());
            VkDeviceSize bufSize = m_particles->exportableBufferSize();
            m_cudaParticlePtr = m_cudaInterop->importBuffer(memHandle, static_cast<size_t>(bufSize));

            m_pmCompute = std::make_unique<PMCompute>();
            m_pmCompute->init(4194304, m_pmParams);
            m_pmCompute->setParticleBuffer(m_cudaParticlePtr, static_cast<size_t>(bufSize));
            m_pmCompute->updateParticleCount(m_pmParams.particleCount);

            m_params = currentParams;
            m_params.particleCount = m_pmParams.particleCount;
            m_semaphoreValue = 0;

            reset(ctx, m_currentIC);

            fprintf(stderr, "[Cosmico] Switched to PM backend\n");
        } catch (const std::exception& e) {
            fprintf(stderr, "[Cosmico] PM init failed: %s\n", e.what());
            m_pmCompute.reset();
            m_backend = oldBackend;
        }
        return;
    }

    // Initialize Node Graph if entering NodeGraph backend
    if (backend == ComputeBackend::NodeGraph) {
        try {
            // Same exportable buffer setup as PM
            auto currentParams = m_params;

            m_compute->destroy(ctx);
            m_particles->destroy(ctx);

            m_particles = std::make_unique<ParticleSystem>();
            m_particles->initExportable(ctx, 4194304);  // 4M max

            m_compute = std::make_unique<NBodyCompute>();
            m_compute->init(ctx, m_particles->computeSetLayout(), m_shaderDir);

            HANDLE memHandle = m_particles->getExportableHandle(ctx.device());
            VkDeviceSize bufSize = m_particles->exportableBufferSize();
            m_cudaParticlePtr = m_cudaInterop->importBuffer(memHandle, static_cast<size_t>(bufSize));

            m_params = currentParams;
            m_params.particleCount = m_pmParams.particleCount;
            m_semaphoreValue = 0;

            // Create node graph infrastructure
            m_simGraph = std::make_unique<SimGraph>();
            m_graphCompiler = std::make_unique<SimGraphCompiler>();
            m_nodeEditorUI = std::make_unique<NodeEditorUI>();
            m_nodeEditorUI->init();

            m_graphExecutor = std::make_unique<SimGraphExecutor>();
            m_graphExecutor->init(4194304, m_pmParams.gridN, m_pmParams);
            m_graphExecutor->resources().setParticleBuffer(m_cudaParticlePtr, static_cast<size_t>(bufSize));
            m_graphExecutor->resources().setParticleCount(m_pmParams.particleCount);

            // Pre-populate default PM pipeline graph
            populateDefaultPMGraph();

            reset(ctx, m_currentIC);

            // Auto-build the default graph
            buildNodeGraph();

            fprintf(stderr, "[Cosmico] Switched to Node Graph backend\n");
        } catch (const std::exception& e) {
            fprintf(stderr, "[Cosmico] Node Graph init failed: %s\n", e.what());
            m_graphExecutor.reset();
            m_nodeEditorUI.reset();
            m_simGraph.reset();
            m_graphCompiler.reset();
            m_compiledPlan.reset();
            m_backend = oldBackend;
        }
        return;
    }

    // Initialize inflation compute if entering Inflation backend
    if (backend == ComputeBackend::Inflation) {
        try {
            m_inflationCompute = std::make_unique<InflationCompute>();
            m_inflationCompute->init(m_inflationParams);
            fprintf(stderr, "[Cosmico] Switched to Inflation backend\n");
        } catch (const std::exception& e) {
            fprintf(stderr, "[Cosmico] Inflation init failed: %s\n", e.what());
            m_inflationCompute.reset();
            m_backend = oldBackend;
        }
        return;
    }

    if (backend == ComputeBackend::BarnesHut && oldBackend != ComputeBackend::Inflation) {
        // Switch from brute-force to Barnes-Hut: need exportable buffers
        auto currentIC = m_currentIC;
        auto currentParams = m_params;

        m_compute->destroy(ctx);
        m_particles->destroy(ctx);

        m_particles = std::make_unique<ParticleSystem>();
        m_particles->initExportable(ctx, 1048576);

        m_compute = std::make_unique<NBodyCompute>();
        m_compute->init(ctx, m_particles->computeSetLayout(), m_shaderDir);

        HANDLE memHandle = m_particles->getExportableHandle(ctx.device());
        VkDeviceSize bufSize = m_particles->exportableBufferSize();
        m_cudaParticlePtr = m_cudaInterop->importBuffer(memHandle, static_cast<size_t>(bufSize));
        m_bhCompute->setParticleBuffer(m_cudaParticlePtr, static_cast<size_t>(bufSize));

        m_params = currentParams;
        m_semaphoreValue = 0;
        reset(ctx, currentIC);

    } else if (backend == ComputeBackend::BarnesHut && oldBackend == ComputeBackend::Inflation) {
        // From Inflation to Barnes-Hut: recreate particle system as exportable
        m_compute->destroy(ctx);
        m_particles->destroy(ctx);

        m_particles = std::make_unique<ParticleSystem>();
        m_particles->initExportable(ctx, 1048576);

        m_compute = std::make_unique<NBodyCompute>();
        m_compute->init(ctx, m_particles->computeSetLayout(), m_shaderDir);

        HANDLE memHandle = m_particles->getExportableHandle(ctx.device());
        VkDeviceSize bufSize = m_particles->exportableBufferSize();
        m_cudaParticlePtr = m_cudaInterop->importBuffer(memHandle, static_cast<size_t>(bufSize));
        m_bhCompute->setParticleBuffer(m_cudaParticlePtr, static_cast<size_t>(bufSize));

        m_semaphoreValue = 0;
        reset(ctx, m_currentIC);

    } else if (backend == ComputeBackend::BruteForce) {
        // Switch to brute-force: need standard VMA buffers
        auto currentIC = m_currentIC;
        auto currentParams = m_params;

        m_cudaParticlePtr = nullptr;

        m_compute->destroy(ctx);
        m_particles->destroy(ctx);

        m_particles = std::make_unique<ParticleSystem>();
        m_particles->init(ctx, 65536);

        m_compute = std::make_unique<NBodyCompute>();
        m_compute->init(ctx, m_particles->computeSetLayout(), m_shaderDir);

        m_params = currentParams;
        if (m_params.particleCount > 65536) {
            m_params.particleCount = 65536;
        }

        reset(ctx, currentIC);
    }
}

void Simulation::reset(VkContext& ctx, InitialCondition ic) {
    m_currentIC = ic;
    float boxSize = (m_backend == ComputeBackend::PM || m_backend == ComputeBackend::NodeGraph)
                     ? m_pmParams.boxSize : 0.0f;
    auto data = generateInitialConditions(ic, static_cast<uint32_t>(m_params.particleCount), boxSize);

    if ((m_backend == ComputeBackend::PM || m_backend == ComputeBackend::NodeGraph) && m_particles->isExportable()) {
        m_particles->uploadExportable(ctx, data.data(), static_cast<uint32_t>(m_params.particleCount));
        if (m_pmCompute) {
            m_pmCompute->updateParticleCount(m_params.particleCount);
            m_pmCompute->resetState();
            // Replace the uploaded uniform lattice with a Zel'dovich (1LPT)
            // cosmological field — the realistic seed for structure formation.
            // Structure formation lives in the expanding (comoving) frame, which
            // is also what seeds the growing-mode peculiar velocities. Tying
            // expansion to the IC also avoids comoving "leaking" into other ICs.
            m_pmParams.comoving = (ic == InitialCondition::Cosmological);
            if (ic == InitialCondition::Cosmological && m_cudaInterop) {
                m_pmCompute->generateCosmologicalIC(m_pmParams, m_cudaInterop->stream());
            }
        }
        m_semaphoreValue = 0;
    } else if (m_backend == ComputeBackend::BarnesHut && m_particles->isExportable()) {
        m_particles->uploadExportable(ctx, data.data(), static_cast<uint32_t>(m_params.particleCount));
        m_bhCompute->updateParticleCount(m_params.particleCount);
        m_semaphoreValue = 0;
    } else {
        m_particles->upload(ctx, data.data(), static_cast<uint32_t>(m_params.particleCount));
    }
    m_compute->updateParams(ctx, m_params);
}

void Simulation::step(VkCommandBuffer cmd) {
    // Dispatch compute on current read->write pair
    m_compute->dispatch(cmd, m_particles->computeDescriptorSet(),
                        m_particles->particleCount());

    // Barrier: compute write → vertex shader read
    insertComputeToGraphicsBarrier(cmd, m_particles->writeBuffer().buffer(),
                                    m_particles->writeBuffer().size());

    // Swap so the just-written buffer becomes the read buffer
    m_particles->swapBuffers();
}

void Simulation::stepInflation() {
    if (!m_inflationCompute) return;
    m_inflationCompute->step(m_inflationParams);
    m_inflationCompute->computePowerSpectrum(m_inflationParams);
}

void Simulation::resetInflation() {
    if (!m_inflationCompute) return;
    m_inflationCompute->reset(m_inflationParams);
}

void Simulation::stepCuda() {
    if (!m_cudaAvailable || !m_bhCompute || !m_cudaParticlePtr) return;

    // Launch kernels async — no sync here, caller must sync before reading buffer
    m_bhCompute->step(m_params, m_cudaInterop->stream());
}

bool Simulation::syncCuda() {
    if (!m_cudaInterop) return true;
    return m_cudaInterop->syncStream();
}

void Simulation::updateParams(VkContext& ctx) {
    m_compute->updateParams(ctx, m_params);
    if (m_bhCompute) {
        m_bhCompute->updateParticleCount(m_params.particleCount);
    }
}

float Simulation::cudaKernelTimeMs() const {
    return m_bhCompute ? m_bhCompute->stats().kernelTimeMs : 0.0f;
}

int Simulation::cudaTreeNodeCount() const {
    return m_bhCompute ? m_bhCompute->stats().nodeCount : 0;
}

const InflationStateData* Simulation::inflationState() const {
    if (m_inflationCompute) {
        return &m_inflationCompute->state();
    }
    return nullptr;
}

float Simulation::inflationKernelTimeMs() const {
    return m_inflationCompute ? m_inflationCompute->kernelTimeMs() : 0.0f;
}

void Simulation::extractDensityVolume(uint8_t* hostDst, int targetN) {
    if (m_inflationCompute) {
        m_inflationCompute->extractDensityVolume(hostDst, targetN);
    }
}

void Simulation::extractCMBMap(uint8_t* hostDst, int mapW, int mapH) {
    if (m_inflationCompute) {
        m_inflationCompute->extractCMBMap(hostDst, mapW, mapH,
                                           m_inflationParams.cmbRadius,
                                           m_inflationParams.boxSize,
                                           m_inflationParams.Mpl);
    }
}

void Simulation::startZeldovich() {
    if (m_inflationCompute) {
        m_inflationCompute->startZeldovich(m_inflationParams);
    }
}

void Simulation::stepZeldovich() {
    if (m_inflationCompute) {
        m_inflationCompute->stepZeldovich(m_inflationParams);
    }
}

void Simulation::extractZeldovichDensity(uint8_t* hostDst, int targetN) {
    if (m_inflationCompute) {
        m_inflationCompute->extractZeldovichDensity(hostDst, targetN);
    }
}

bool Simulation::zeldovichActive() const {
    return m_inflationCompute && m_inflationCompute->zeldovichActive();
}

float Simulation::zeldovichGrowth() const {
    return m_inflationCompute ? m_inflationCompute->zeldovichGrowth() : 0.0f;
}

void Simulation::stopZeldovich() {
    if (m_inflationCompute) {
        m_inflationCompute->stopZeldovich();
    }
}

// ── PM ────────────────────────────────────────────────────────────────────

void Simulation::stepPM() {
    if (!m_pmCompute || !m_cudaParticlePtr) return;
    m_pmCompute->step(m_pmParams, m_cudaInterop->stream());
}

void Simulation::resetPM(VkContext& ctx, InitialCondition ic) {
    if (!m_pmCompute) return;

    m_currentIC = ic;
    m_params.particleCount = m_pmParams.particleCount;

    auto data = generateInitialConditions(m_currentIC,
                                           static_cast<uint32_t>(m_pmParams.particleCount),
                                           m_pmParams.boxSize);

    if (m_particles->isExportable()) {
        m_particles->uploadExportable(ctx, data.data(),
                                       static_cast<uint32_t>(m_pmParams.particleCount));
    }
    m_pmCompute->updateParticleCount(m_pmParams.particleCount);
    m_pmCompute->resetState();
    // Cosmological IC: replace the uniform lattice with a Zel'dovich (1LPT)
    // field in the expanding (comoving) frame, which seeds growing-mode velocities.
    m_pmParams.comoving = (ic == InitialCondition::Cosmological);
    if (ic == InitialCondition::Cosmological && m_cudaInterop) {
        m_pmCompute->generateCosmologicalIC(m_pmParams, m_cudaInterop->stream());
    }
    m_semaphoreValue = 0;
}

const PMStateData* Simulation::pmState() const {
    if (m_pmCompute) {
        return &m_pmCompute->state();
    }
    return nullptr;
}

float Simulation::pmKernelTimeMs() const {
    return m_pmCompute ? m_pmCompute->kernelTimeMs() : 0.0f;
}

// ── CDT 2D ────────────────────────────────────────────────────────────────

void Simulation::stepCDT2D() {
    if (m_cdt2dCompute) {
        m_cdt2dCompute->step(m_cdt2dParams);
    }
}

void Simulation::resetCDT2D() {
    if (m_cdt2dCompute) {
        m_cdt2dCompute->reset(m_cdt2dParams);
    }
}

const CDT2DStateData* Simulation::cdt2dState() const {
    if (m_cdt2dCompute) {
        return &m_cdt2dCompute->state();
    }
    return nullptr;
}

// ── CDT 3D ────────────────────────────────────────────────────────────────

void Simulation::stepCDT3D() {
    if (m_cdt3dCompute) {
        m_cdt3dCompute->step(m_cdt3dParams);
    }
}

void Simulation::resetCDT3D() {
    if (m_cdt3dCompute) {
        m_cdt3dCompute->reset(m_cdt3dParams);
    }
}

const CDT3DStateData* Simulation::cdt3dState() const {
    if (m_cdt3dCompute) {
        return &m_cdt3dCompute->state();
    }
    return nullptr;
}

float Simulation::cdt3dKernelTimeMs() const {
    return m_cdt3dCompute ? m_cdt3dCompute->kernelTimeMs() : 0.0f;
}

// ── Node Graph ────────────────────────────────────────────────────────────

void Simulation::stepNodeGraph() {
    if (!m_graphExecutor || !m_compiledPlan || !m_compiledPlan->valid) return;
    if (!m_cudaParticlePtr) return;

    // Update pool params each frame
    auto& pool = m_graphExecutor->resources();
    pool.setParams(m_pmParams);
    pool.setParticleCount(m_pmParams.particleCount);

    // Execute for each sub-step
    for (int sub = 0; sub < m_pmParams.stepsPerFrame; sub++) {
        m_graphExecutor->execute(*m_compiledPlan, static_cast<cudaStream_t>(m_cudaInterop->stream()));
    }
}

void Simulation::resetNodeGraph(VkContext& ctx, InitialCondition ic) {
    m_currentIC = ic;
    m_params.particleCount = m_pmParams.particleCount;

    auto data = generateInitialConditions(m_currentIC,
                                           static_cast<uint32_t>(m_pmParams.particleCount),
                                           m_pmParams.boxSize);

    if (m_particles->isExportable()) {
        m_particles->uploadExportable(ctx, data.data(),
                                       static_cast<uint32_t>(m_pmParams.particleCount));
    }

    if (m_graphExecutor) {
        m_graphExecutor->resources().setParticleCount(m_pmParams.particleCount);
    }
    m_semaphoreValue = 0;
}

SimGraph& Simulation::simGraph() {
    return *m_simGraph;
}

SimGraphExecutor& Simulation::graphExecutor() {
    return *m_graphExecutor;
}

NodeEditorUI& Simulation::nodeEditorUI() {
    return *m_nodeEditorUI;
}

const CompiledPlan* Simulation::compiledPlan() const {
    return m_compiledPlan.get();
}

void Simulation::buildNodeGraph() {
    if (!m_simGraph || !m_graphCompiler) return;
    m_compiledPlan = std::make_unique<CompiledPlan>(m_graphCompiler->compile(*m_simGraph));
    if (m_compiledPlan->valid) {
        fprintf(stderr, "[NodeGraph] Build succeeded: %zu nodes in execution order\n",
                m_compiledPlan->executionOrder.size());
    } else {
        fprintf(stderr, "[NodeGraph] Build failed: %s\n", m_compiledPlan->error.c_str());
    }
}

void Simulation::populateDefaultPMGraph() {
    if (!m_simGraph) return;
    auto& g = *m_simGraph;
    g.addNode("ParticleSource");   // [0]
    g.addNode("HalfDrift");        // [1] half-drift 1
    g.addNode("ClearGrid");        // [2]
    g.addNode("CICDeposit");       // [3]
    g.addNode("FFTForward");       // [4]
    g.addNode("GreenMultiply");    // [5]
    g.addNode("FFTInverse");       // [6]
    g.addNode("Gradient");         // [7]
    g.addNode("CICInterpolate");   // [8]
    g.addNode("Kick");             // [9]
    g.addNode("HalfDrift");        // [10] half-drift 2
    g.addNode("Output");           // [11] pipeline endpoint

    auto findOut = [&](size_t idx, const char* name) -> int {
        for (auto& p : g.nodes[idx]->outputs)
            if (p.name == name) return p.id;
        return -1;
    };
    auto findIn = [&](size_t idx, const char* name) -> int {
        for (auto& p : g.nodes[idx]->inputs)
            if (p.name == name) return p.id;
        return -1;
    };

    g.addLink(findOut(0, "particles"), findIn(1, "particles"));
    g.addLink(findOut(1, "particles"), findIn(3, "particles"));
    g.addLink(findOut(2, "grid"), findIn(3, "grid"));
    g.addLink(findOut(3, "densityGrid"), findIn(4, "densityGrid"));
    g.addLink(findOut(4, "complexGrid"), findIn(5, "complexGrid"));
    g.addLink(findOut(5, "complexGrid"), findIn(6, "complexGrid"));
    g.addLink(findOut(6, "potentialGrid"), findIn(7, "potentialGrid"));
    g.addLink(findOut(7, "forceX"), findIn(8, "forceX"));
    g.addLink(findOut(7, "forceY"), findIn(8, "forceY"));
    g.addLink(findOut(7, "forceZ"), findIn(8, "forceZ"));
    g.addLink(findOut(1, "particles"), findIn(8, "particles"));
    g.addLink(findOut(8, "particles"), findIn(9, "particles"));
    g.addLink(findOut(9, "particles"), findIn(10, "particles"));
    g.addLink(findOut(10, "particles"), findIn(11, "particles"));
}

void Simulation::populateDefaultNBodyGraph() {
    if (!m_simGraph) return;
    auto& g = *m_simGraph;
    g.addNode("ParticleSource");   // [0]
    g.addNode("HalfDrift");        // [1] half-drift 1
    g.addNode("ClearAccel");       // [2] zero acceleration fields
    g.addNode("PairwiseGravity");  // [3] O(N^2) all-pairs gravity
    g.addNode("Kick");             // [4] v += a * dt
    g.addNode("HalfDrift");        // [5] half-drift 2
    g.addNode("Output");           // [6] pipeline endpoint

    auto findOut = [&](size_t idx, const char* name) -> int {
        for (auto& p : g.nodes[idx]->outputs)
            if (p.name == name) return p.id;
        return -1;
    };
    auto findIn = [&](size_t idx, const char* name) -> int {
        for (auto& p : g.nodes[idx]->inputs)
            if (p.name == name) return p.id;
        return -1;
    };

    // DKD leapfrog: ParticleSource -> HalfDrift -> ClearAccel -> PairwiseGravity -> Kick -> HalfDrift -> Output
    g.addLink(findOut(0, "particles"), findIn(1, "particles"));
    g.addLink(findOut(1, "particles"), findIn(2, "particles"));
    g.addLink(findOut(2, "particles"), findIn(3, "particles"));
    g.addLink(findOut(3, "particles"), findIn(4, "particles"));
    g.addLink(findOut(4, "particles"), findIn(5, "particles"));
    g.addLink(findOut(5, "particles"), findIn(6, "particles"));
}

} // namespace cosmico
