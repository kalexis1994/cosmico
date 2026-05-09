#pragma once
#include <cosmico/simulation/Simulation.h>
#include <cosmico/renderer/OffscreenTarget.h>
#include <cosmico/ui/TimelineUI.h>
#include <memory>
#include <vector>
#include <string>
#include <cstdint>

namespace cosmico {

class Window;
class Input;
class Timer;
class VkContext;
class VkSwapchain;
class Renderer;
class Camera;
class DebugUI;
class SciencePanel;
class VkImGuiBackend;
class ParticleRenderer;
class VolumeRenderer;
class CMBRenderer;
class CDT2DRenderer;
class CDT3DRenderer;
class SimulationCatalog;
class SnapshotRecorder;
class SnapshotPlayer;
class TimelineUI;
class PlanetTextures;

enum class AppState {
    Gallery,
    Detail,
    Running
};

class Application {
public:
    Application();
    ~Application();

    void run();

private:
    void init();
    void mainLoop();
    void cleanup();

    // State machine phases
    void renderGalleryState();
    void renderDetailState();
    void renderRunningState(VkCommandBuffer cmd);

    // Simulation lifecycle (deferred from init to Launch click)
    void initSimulation(int catalogIndex,
                        InitialCondition icOverride = InitialCondition::Count);
    void cleanupSimulation();

    // Shared infrastructure (always alive)
    std::unique_ptr<Window> m_window;
    std::unique_ptr<Input> m_input;
    std::unique_ptr<Timer> m_timer;
    std::unique_ptr<VkContext> m_context;
    std::unique_ptr<VkSwapchain> m_swapchain;
    std::unique_ptr<Renderer> m_renderer;
    std::unique_ptr<Camera> m_camera;
    std::unique_ptr<VkImGuiBackend> m_imguiBackend;
    std::unique_ptr<SimulationCatalog> m_catalog;

    // Simulation-specific (created on Launch, destroyed on Back)
    std::unique_ptr<Simulation> m_simulation;
    std::unique_ptr<DebugUI> m_debugUI;
    std::unique_ptr<SciencePanel> m_sciencePanel;
    std::unique_ptr<ParticleRenderer> m_particleRenderer;
    std::unique_ptr<VolumeRenderer> m_volumeRenderer;
    std::unique_ptr<CMBRenderer> m_cmbRenderer;
    std::unique_ptr<CDT2DRenderer> m_cdt2dRenderer;
    std::unique_ptr<CDT3DRenderer> m_cdt3dRenderer;
    std::unique_ptr<PlanetTextures> m_planetTextures;

    // Staging buffers for CUDA -> CPU -> Vulkan texture uploads
    static constexpr int VOLUME_TEXTURE_N = 128;
    static constexpr int CMB_MAP_W = 1024;
    static constexpr int CMB_MAP_H = 512;
    std::vector<uint8_t> m_volumeData;
    std::vector<uint8_t> m_cmbData;

    bool m_paused = true;
    float m_accumulator = 0.0f;

    // Gallery state machine
    AppState m_appState = AppState::Gallery;
    int m_selectedSimIndex = -1;

    // Resolved paths
    std::string m_shaderDir;

    // Deferred state changes (applied between frames, not mid-frame)
    bool m_pendingBackendChange = false;
    ComputeBackend m_pendingBackend = ComputeBackend::BruteForce;
    bool m_pendingReset = false;
    InitialCondition m_pendingIC = InitialCondition::Disk;

    // Back-to-gallery signal from DebugUI
    bool m_backToGalleryRequested = false;

    // Offscreen viewport for dockable 3D rendering
    std::unique_ptr<OffscreenTarget> m_offscreenTarget;
    uint32_t m_viewportWidth = 800;
    uint32_t m_viewportHeight = 600;
    bool m_viewportHovered = false;
    float m_viewportScreenX = 0.0f;  // top-left of viewport content in window coords
    float m_viewportScreenY = 0.0f;
    bool m_pendingViewportResize = false;
    float m_viewportResizeTimer = 0.0f;
    bool m_dockLayoutInitialized = false;

    // Recording / playback
    std::unique_ptr<SnapshotRecorder> m_recorder;
    std::unique_ptr<SnapshotPlayer> m_player;
    std::unique_ptr<TimelineUI> m_timelineUI;
    TimelineUI::State m_timelineState;
    uint64_t m_simStepCounter = 0;
    double m_simTime = 0.0;
    bool m_playbackMode = false;
};

} // namespace cosmico
