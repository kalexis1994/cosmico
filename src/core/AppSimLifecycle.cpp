#include <cosmico/core/Application.h>
#include <cosmico/core/PathUtils.h>
#include <cosmico/vulkan/VkContext.h>
#include <cosmico/vulkan/VkImGui.h>
#include <cosmico/renderer/Camera.h>
#include <cosmico/renderer/ParticleRenderer.h>
#include <cosmico/renderer/VolumeRenderer.h>
#include <cosmico/renderer/CMBRenderer.h>
#include <cosmico/renderer/CDT2DRenderer.h>
#include <cosmico/renderer/CDT3DRenderer.h>
#include <cosmico/renderer/OffscreenTarget.h>
#include <cosmico/renderer/PlanetTextures.h>
#include <cosmico/simulation/Simulation.h>
#include <cosmico/simulation/InitialConditions.h>
#include <cosmico/simulation/ParticleSystem.h>
#include <cosmico/ui/DebugUI.h>
#include <cosmico/ui/SciencePanel.h>
#include <cosmico/ui/SimulationCatalog.h>
#include <cosmico/ui/TimelineUI.h>
#include <cosmico/recording/SnapshotRecorder.h>
#include <cosmico/recording/SnapshotPlayer.h>

#include <glm/glm.hpp>
#include <cstdio>
#include <filesystem>

namespace cosmico {

void Application::initSimulation(int catalogIndex, InitialCondition icOverride) {
    const auto& entry = m_catalog->entry(catalogIndex);

    // Determine the actual IC (caller may override for scenarios)
    InitialCondition actualIC = (icOverride != InitialCondition::Count)
        ? icOverride : entry.initialCondition;

    // Simulation
    m_simulation = std::make_unique<Simulation>();
    m_simulation->init(*m_context, m_shaderDir);

    // Apply params from config.json
    m_simulation->params() = entry.defaultParams;

    // Set backend
    m_simulation->setBackend(*m_context, entry.backend);

    // Camera: apply per-simulation config
    {
        const auto& cc = entry.cameraConfig;
        m_camera->reset(cc.distance, cc.pitch, cc.yaw,
                        glm::vec3(cc.target[0], cc.target[1], cc.target[2]),
                        cc.fov, cc.nearPlane, cc.farPlane);
        if (entry.backend == ComputeBackend::CDT2D)
            m_camera->orthographic = true;
    }

    // Planet textures (solar system only)
    if (actualIC == InitialCondition::SolarSystem) {
        std::string texDir = resolvePathNextToExe("resources/planet-textures",
            std::string(COSMICO_RESOURCES_DIR) + "/planet-textures");
        m_planetTextures = std::make_unique<PlanetTextures>();
        if (!m_planetTextures->init(*m_context, texDir)) {
            fprintf(stderr, "[Cosmico] Planet textures failed to load, falling back to flat colors\n");
            m_planetTextures.reset();
        }
    }

    // Particle renderer (graphics pipeline)
    bool usesParticles = (entry.backend != ComputeBackend::Inflation &&
                          entry.backend != ComputeBackend::CDT2D &&
                          entry.backend != ComputeBackend::CDT3D);
    if (usesParticles) {
        m_particleRenderer = std::make_unique<ParticleRenderer>();
        VkDescriptorSetLayout texLayout = m_planetTextures
            ? m_planetTextures->descriptorSetLayout() : VK_NULL_HANDLE;
        m_particleRenderer->init(*m_context, m_offscreenTarget->renderPass(),
                                  m_simulation->particleSystem().graphicsSetLayout(),
                                  m_shaderDir, texLayout);
    }

    // Volume renderer (3D density visualization for inflation)
    m_volumeRenderer = std::make_unique<VolumeRenderer>();
    m_volumeRenderer->init(*m_context, m_offscreenTarget->renderPass(), m_shaderDir);

    // CMB renderer (spherical last scattering surface)
    m_cmbRenderer = std::make_unique<CMBRenderer>();
    m_cmbRenderer->init(*m_context, m_offscreenTarget->renderPass(), m_shaderDir);

    // CDT 2D renderer
    m_cdt2dRenderer = std::make_unique<CDT2DRenderer>();
    m_cdt2dRenderer->init(*m_context, m_offscreenTarget->renderPass(), m_shaderDir);

    // CDT 3D renderer
    m_cdt3dRenderer = std::make_unique<CDT3DRenderer>();
    m_cdt3dRenderer->init(*m_context, m_offscreenTarget->renderPass(), m_shaderDir);

    // Debug UI
    m_debugUI = std::make_unique<DebugUI>();

    // Science panel — each simulation owns its own papers/ folder
    m_sciencePanel = std::make_unique<SciencePanel>();
    std::string papersDir = (!entry.papersDir.empty() && std::filesystem::exists(entry.papersDir))
        ? entry.papersDir : std::string();
    m_sciencePanel->init(m_imguiBackend->fontBody,
                         m_imguiBackend->fontHeading,
                         m_imguiBackend->fontFormula,
                         m_imguiBackend->fontBodyZoom,
                         m_imguiBackend->fontHeadingZoom,
                         m_imguiBackend->fontFormulaZoom,
                         papersDir);

    // Recording / playback
    m_recorder = std::make_unique<SnapshotRecorder>();
    m_player = std::make_unique<SnapshotPlayer>();
    m_timelineUI = std::make_unique<TimelineUI>();
    m_timelineState = {};
    m_simStepCounter = 0;
    m_simTime = 0.0;
    m_playbackMode = false;

    m_pendingIC = actualIC;

    m_paused = true;
    m_accumulator = 0.0f;
    m_pendingBackendChange = false;
    m_pendingReset = false;
    m_backToGalleryRequested = false;

    fprintf(stderr, "[Cosmico] Launched simulation: %s (%s)\n",
            entry.title.c_str(), entry.backendStr.c_str());
}

void Application::cleanupSimulation() {
    if (!m_simulation) return;

    // Stop recording/playback before GPU teardown
    if (m_recorder && m_recorder->isRecording()) m_recorder->stop();
    if (m_player && m_player->isOpen()) m_player->close();
    m_recorder.reset();
    m_player.reset();
    m_timelineUI.reset();
    m_playbackMode = false;

    vkDeviceWaitIdle(m_context->device());

    m_debugUI.reset();
    m_sciencePanel.reset();

    if (m_cdt3dRenderer) {
        m_cdt3dRenderer->destroy(m_context->device(), m_context->allocator());
        m_cdt3dRenderer.reset();
    }
    if (m_cdt2dRenderer) {
        m_cdt2dRenderer->destroy(m_context->device(), m_context->allocator());
        m_cdt2dRenderer.reset();
    }
    if (m_cmbRenderer) {
        m_cmbRenderer->destroy(m_context->device(), m_context->allocator());
        m_cmbRenderer.reset();
    }
    if (m_volumeRenderer) {
        m_volumeRenderer->destroy(m_context->device(), m_context->allocator());
        m_volumeRenderer.reset();
    }
    if (m_planetTextures) {
        m_planetTextures->destroy(m_context->device(), m_context->allocator());
        m_planetTextures.reset();
    }
    if (m_particleRenderer) {
        m_particleRenderer->destroy(m_context->device());
        m_particleRenderer.reset();
    }
    if (m_simulation) {
        m_simulation->destroy(*m_context);
        m_simulation.reset();
    }

    m_camera->orthographic = false;

    fprintf(stderr, "[Cosmico] Returned to gallery\n");
}

} // namespace cosmico
