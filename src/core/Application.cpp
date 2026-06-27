#include <cosmico/core/Application.h>
#include <cosmico/core/PathUtils.h>
#include <cosmico/core/Window.h>
#include <cosmico/core/Input.h>
#include <cosmico/core/Timer.h>
#include <cosmico/vulkan/VkContext.h>
#include <cosmico/vulkan/VkSwapchain.h>
#include <cosmico/vulkan/VkImGui.h>
#include <cosmico/renderer/Renderer.h>
#include <cosmico/renderer/Camera.h>
#include <cosmico/renderer/ParticleRenderer.h>
#include <cosmico/renderer/VolumeRenderer.h>
#include <cosmico/renderer/CMBRenderer.h>
#include <cosmico/renderer/CDT2DRenderer.h>
#include <cosmico/renderer/CDT3DRenderer.h>
#include <cosmico/simulation/Simulation.h>
#include <cosmico/simulation/InitialConditions.h>
#include <cosmico/simulation/ParticleSystem.h>
#include <cosmico/simulation/InflationCompute.h>
#include <cosmico/simulation/CDT2DCompute.h>
#include <cosmico/simulation/CDT3DCompute.h>
#include <cosmico/ui/DebugUI.h>
#include <cosmico/ui/SciencePanel.h>
#include <cosmico/ui/SimulationCatalog.h>
#include <cosmico/nodes/NodeEditorUI.h>
#include <cosmico/nodes/SimGraphCompiler.h>
#include <cosmico/recording/SnapshotRecorder.h>
#include <cosmico/recording/SnapshotPlayer.h>
#include <cosmico/ui/TimelineUI.h>
#include <cosmico/renderer/PlanetTextures.h>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_internal.h>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <sstream>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#include <libgen.h>
#include <climits>
#endif
#include <filesystem>

namespace cosmico {

// Resolve a path relative to the executable directory.
// Falls back to the compile-time path if the relative one doesn't exist.
// Declared in <cosmico/core/PathUtils.h>; defined here so windows.h /
// unistd.h stay out of the public header.
std::string resolvePathNextToExe(const std::string& relativePath,
                                  [[maybe_unused]] const std::string& fallback) {
    std::filesystem::path exeDir;
#ifdef _WIN32
    char buf[MAX_PATH];
    GetModuleFileNameA(nullptr, buf, MAX_PATH);
    exeDir = std::filesystem::path(buf).parent_path();
#else
    char buf[PATH_MAX];
    ssize_t len = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (len > 0) { buf[len] = '\0'; exeDir = std::filesystem::path(buf).parent_path(); }
    else { return fallback; }
#endif
    auto candidate = exeDir / relativePath;
    if (std::filesystem::exists(candidate))
        return candidate.string();
    return fallback;
}

Application::Application() {
    init();
}

Application::~Application() {
    cleanup();
}

void Application::init() {
    m_volumeData.resize(VOLUME_TEXTURE_N * VOLUME_TEXTURE_N * VOLUME_TEXTURE_N);
    m_cmbData.resize(CMB_MAP_W * CMB_MAP_H);

    m_window = std::make_unique<Window>(1600, 900, "Cosmico - Cosmological Simulation Engine");
    m_input = std::make_unique<Input>();
    m_timer = std::make_unique<Timer>();
    m_camera = std::make_unique<Camera>();

    m_input->init(m_window->handle());
    m_camera->init(50.0f);

    // Vulkan init
    m_context = std::make_unique<VkContext>();
    m_context->init(m_window->handle());

    m_swapchain = std::make_unique<VkSwapchain>();
    m_swapchain->init(*m_context, m_window->width(), m_window->height());

    m_renderer = std::make_unique<Renderer>();
    m_renderer->init(*m_context, *m_swapchain);

    // Resolve paths
    m_shaderDir = resolvePathNextToExe("shaders", COSMICO_SHADER_DIR);

    // ImGui (needed for gallery rendering — must init before OffscreenTarget)
    m_imguiBackend = std::make_unique<VkImGuiBackend>();
    m_imguiBackend->init(*m_context, *m_swapchain, m_window->handle(), m_renderer->renderPass());

    // Offscreen render target for dockable 3D viewport (after ImGui, uses AddTexture)
    m_offscreenTarget = std::make_unique<OffscreenTarget>();
    m_offscreenTarget->init(*m_context, m_swapchain->imageFormat(),
                             m_swapchain->depthFormat(), 800, 600);

    // Gallery catalog
    std::string simsDir = resolvePathNextToExe("simulations", COSMICO_SIMULATIONS_DIR);
    m_catalog = std::make_unique<SimulationCatalog>();
    m_catalog->init(*m_context, simsDir);

    m_timer->reset();
    fprintf(stderr, "[Cosmico] Initialization complete (Gallery mode)\n");
}

// initSimulation() and cleanupSimulation() live in AppSimLifecycle.cpp.

// ─── Main loop ──────────────────────────────────────────────────────

void Application::mainLoop() {
    while (!m_window->shouldClose()) {
        m_window->pollEvents();
        m_input->update();
        m_timer->tick();

        float dt = m_timer->deltaTime();

        // Camera update (gate by viewport hover in Running state, but always allow in free cam)
        {
            bool isFreeCamera = (m_camera->mode() == CameraMode::Free);
            bool allowCameraInput = (m_appState != AppState::Running) || m_viewportHovered || isFreeCamera;
            if (allowCameraInput) {
                float vpW, vpH, cursorX, cursorY;
                if (m_appState == AppState::Running) {
                    vpW = static_cast<float>(m_viewportWidth);
                    vpH = static_cast<float>(m_viewportHeight);
                    cursorX = static_cast<float>(m_input->mouseX()) - (m_viewportScreenX + vpW * 0.5f);
                    cursorY = static_cast<float>(m_input->mouseY()) - (m_viewportScreenY + vpH * 0.5f);
                } else {
                    vpW = static_cast<float>(m_window->width());
                    vpH = static_cast<float>(m_window->height());
                    cursorX = static_cast<float>(m_input->mouseX()) - vpW * 0.5f;
                    cursorY = static_cast<float>(m_input->mouseY()) - vpH * 0.5f;
                }
                // Anchored expansion view: the camera is pinned in space, so the
                // wheel does a field-of-view zoom. Stash/restore the FOV across
                // the mode switch.
                bool fovZoom = m_simulation
                    && m_simulation->backend() == ComputeBackend::PM
                    && m_simulation->pmParams().physicalView
                    && !m_simulation->pmParams().isotropicExpansion;
                if (fovZoom && !m_camera->fovZoom)
                    m_savedFov = m_camera->fovDegrees;
                else if (!fovZoom && m_camera->fovZoom && m_savedFov > 0.0f)
                    m_camera->fovDegrees = m_savedFov;
                m_camera->fovZoom = fovZoom;

                m_camera->update(*m_input, dt, vpH, cursorX, cursorY);
            }
        }

        // Tab toggles free camera; click exits free camera
        if (m_appState == AppState::Running) {
            bool isFree = (m_camera->mode() == CameraMode::Free);

            if (m_input->isKeyPressed(GLFW_KEY_TAB)) {
                if (!isFree) {
                    m_camera->setMode(CameraMode::Free);
                    glfwSetInputMode(m_window->handle(), GLFW_CURSOR, GLFW_CURSOR_DISABLED);
                    if (glfwRawMouseMotionSupported())
                        glfwSetInputMode(m_window->handle(), GLFW_RAW_MOUSE_MOTION, GLFW_TRUE);
                } else {
                    m_camera->setMode(CameraMode::Orbit);
                    glfwSetInputMode(m_window->handle(), GLFW_RAW_MOUSE_MOTION, GLFW_FALSE);
                    glfwSetInputMode(m_window->handle(), GLFW_CURSOR, GLFW_CURSOR_NORMAL);
                }
            } else if (isFree && (m_input->isMouseButtonPressed(GLFW_MOUSE_BUTTON_LEFT) ||
                                   m_input->isMouseButtonPressed(GLFW_MOUSE_BUTTON_RIGHT) ||
                                   m_input->isMouseButtonPressed(GLFW_MOUSE_BUTTON_MIDDLE))) {
                m_camera->setMode(CameraMode::Orbit);
                glfwSetInputMode(m_window->handle(), GLFW_RAW_MOUSE_MOTION, GLFW_FALSE);
                glfwSetInputMode(m_window->handle(), GLFW_CURSOR, GLFW_CURSOR_NORMAL);
            }
        }

        // Escape to close
        if (m_input->isKeyPressed(GLFW_KEY_ESCAPE)) {
            if (m_appState == AppState::Running) {
                // Release cursor if free camera was active
                if (m_camera->mode() == CameraMode::Free) {
                    m_camera->setMode(CameraMode::Orbit);
                    glfwSetInputMode(m_window->handle(), GLFW_RAW_MOUSE_MOTION, GLFW_FALSE);
                    glfwSetInputMode(m_window->handle(), GLFW_CURSOR, GLFW_CURSOR_NORMAL);
                }
                // Go back to gallery instead of closing
                cleanupSimulation();
                m_appState = AppState::Gallery;
                m_selectedSimIndex = -1;
                continue;
            } else if (m_appState == AppState::Detail) {
                m_appState = AppState::Gallery;
                m_selectedSimIndex = -1;
                continue;
            } else {
                break;  // Close app from gallery
            }
        }

        // Handle resize
        if (m_window->wasResized()) {
            m_window->resetResizedFlag();
            if (m_window->width() == 0 || m_window->height() == 0) {
                m_window->waitEvents();
                continue;
            }
            m_swapchain->recreate(*m_context, m_window->width(), m_window->height());
            m_renderer->recreateFramebuffers(*m_context, *m_swapchain);
            continue;
        }

        // Viewport resize — immediate so aspect ratio stays correct
        if (m_pendingViewportResize) {
            vkDeviceWaitIdle(m_context->device());
            m_offscreenTarget->resize(*m_context, m_viewportWidth, m_viewportHeight);
            m_pendingViewportResize = false;
        }

        // ── Apply deferred changes BETWEEN frames (Running state only) ──
        if (m_appState == AppState::Running && m_simulation) {
            if (m_pendingBackendChange) {
                vkDeviceWaitIdle(m_context->device());

                bool wasCDT2D = (m_simulation->backend() == ComputeBackend::CDT2D);
                bool goingToInflation = (m_pendingBackend == ComputeBackend::Inflation);
                bool goingToCDT2D = (m_pendingBackend == ComputeBackend::CDT2D);
                bool goingToCDT3D = (m_pendingBackend == ComputeBackend::CDT3D);

                m_simulation->setBackend(*m_context, m_pendingBackend);

                if (goingToCDT2D) {
                    m_camera->orthographic = true;
                } else if (wasCDT2D) {
                    m_camera->orthographic = false;
                }
                if (goingToCDT3D) {
                    m_camera->orthographic = false;
                }

                if (!goingToInflation && !goingToCDT2D && !goingToCDT3D) {
                    if (m_particleRenderer) {
                        m_particleRenderer->destroy(m_context->device());
                    } else {
                        m_particleRenderer = std::make_unique<ParticleRenderer>();
                    }
                    VkDescriptorSetLayout texLayout = m_planetTextures
                        ? m_planetTextures->descriptorSetLayout() : VK_NULL_HANDLE;
                    m_particleRenderer->init(*m_context, m_offscreenTarget->renderPass(),
                                              m_simulation->particleSystem().graphicsSetLayout(),
                                              m_shaderDir, texLayout);
                }
                m_pendingBackendChange = false;
                m_pendingReset = false;
                m_accumulator = 0.0f;
            }

            if (m_pendingReset) {
                vkDeviceWaitIdle(m_context->device());
                if (m_simulation->backend() == ComputeBackend::Inflation) {
                    m_simulation->resetInflation();
                } else if (m_simulation->backend() == ComputeBackend::PM) {
                    m_simulation->resetPM(*m_context, m_pendingIC);
                } else if (m_simulation->backend() == ComputeBackend::CDT2D) {
                    m_simulation->resetCDT2D();
                } else if (m_simulation->backend() == ComputeBackend::CDT3D) {
                    m_simulation->resetCDT3D();
                } else if (m_simulation->backend() == ComputeBackend::NodeGraph) {
                    m_simulation->resetNodeGraph(*m_context, m_pendingIC);
                } else {
                    m_simulation->reset(*m_context, m_pendingIC);
                }
                m_pendingReset = false;
                m_accumulator = 0.0f;
            }

            // Space to toggle pause (only in Running state)
            if (m_input->isKeyPressed(GLFW_KEY_SPACE)) {
                m_paused = !m_paused;
            }
        }

        // ── Pre-frame compute (Running state only) ──
        if (m_appState == AppState::Running && m_simulation) {
            bool shouldStep = !m_paused && !m_playbackMode;

            bool usingCuda = (m_simulation->backend() == ComputeBackend::BarnesHut);
            bool usingPM = (m_simulation->backend() == ComputeBackend::PM);
            bool usingInflation = (m_simulation->backend() == ComputeBackend::Inflation);
            bool usingCDT2D = (m_simulation->backend() == ComputeBackend::CDT2D);
            bool usingCDT3D = (m_simulation->backend() == ComputeBackend::CDT3D);
            bool usingNodeGraph = (m_simulation->backend() == ComputeBackend::NodeGraph);
            bool zeldovichMode = usingInflation && m_simulation->zeldovichActive();

            if (shouldStep && usingCDT2D) m_simulation->stepCDT2D();
            if (shouldStep && usingCDT3D) m_simulation->stepCDT3D();

            if (shouldStep && usingInflation) {
                if (zeldovichMode) m_simulation->stepZeldovich();
                else m_simulation->stepInflation();
            }

            if (shouldStep && usingNodeGraph) {
                m_simulation->stepNodeGraph();
                if (!m_simulation->syncCuda())
                    fprintf(stderr, "[CUDA] Node Graph kernel failed\n");
            }
            if (shouldStep && usingPM) {
                m_simulation->stepPM();
                if (!m_simulation->syncCuda())
                    fprintf(stderr, "[CUDA] PM kernel failed\n");
            }
            if (shouldStep && usingCuda) {
                m_simulation->stepCuda();
                if (!m_simulation->syncCuda())
                    fprintf(stderr, "[CUDA] Barnes-Hut kernel failed\n");
            }

            // Recording: capture frame after sim step
            if (shouldStep) {
                m_simStepCounter++;
                m_simTime += dt;
                if (m_recorder && m_recorder->isRecording())
                    m_recorder->captureFrame(m_simStepCounter, m_simTime,
                                             m_simulation->cudaParticlePtr());
            }

            // Playback: load frame and upload to GPU
            if (m_playbackMode && m_player && m_player->isOpen()) {
                int target = m_timelineState.currentFrame;
                if (target != static_cast<int>(m_player->currentFrame())) {
                    m_player->loadFrame(static_cast<uint32_t>(target));
                    m_player->uploadToCuda(m_simulation->cudaParticlePtr());
                }
            }
        }

        // Begin frame
        uint32_t imageIndex;
        if (!m_renderer->beginFrame(*m_context, *m_swapchain, imageIndex)) {
            m_swapchain->recreate(*m_context, m_window->width(), m_window->height());
            m_renderer->recreateFramebuffers(*m_context, *m_swapchain);
            continue;
        }

        VkCommandBuffer cmd = m_renderer->currentCommandBuffer();

        // ── Running state: Vulkan compute + texture uploads BEFORE render pass ──
        if (m_appState == AppState::Running && m_simulation) {
            bool usingCuda = (m_simulation->backend() == ComputeBackend::BarnesHut);
            bool usingPM = (m_simulation->backend() == ComputeBackend::PM);
            bool usingInflation = (m_simulation->backend() == ComputeBackend::Inflation);
            bool usingCDT2D = (m_simulation->backend() == ComputeBackend::CDT2D);
            bool usingCDT3D = (m_simulation->backend() == ComputeBackend::CDT3D);
            bool usingNodeGraph = (m_simulation->backend() == ComputeBackend::NodeGraph);
            bool zeldovichMode = usingInflation && m_simulation->zeldovichActive();

            // Vulkan compute: brute-force backend
            if (!m_paused && !m_playbackMode && !usingCuda && !usingPM && !usingInflation &&
                !usingCDT2D && !usingCDT3D && !usingNodeGraph) {
                m_accumulator += dt;
                float simDt = m_simulation->params().dt;
                int maxSteps = 4;
                int steps = 0;
                while (m_accumulator >= simDt && steps < maxSteps) {
                    m_simulation->step(cmd);
                    m_accumulator -= simDt;
                    steps++;
                }
                if (m_accumulator > simDt) m_accumulator = 0.0f;
            }

            // Extract and upload volume/CMB data
            bool volumeReady = false;
            if (usingInflation && m_simulation->inflationParams().showVolume) {
                if (zeldovichMode)
                    m_simulation->extractZeldovichDensity(m_volumeData.data(), VOLUME_TEXTURE_N);
                else
                    m_simulation->extractDensityVolume(m_volumeData.data(), VOLUME_TEXTURE_N);
                volumeReady = true;
            }
            bool cmbReady = false;
            if (usingInflation && !zeldovichMode && m_simulation->inflationParams().showCMB) {
                m_simulation->extractCMBMap(m_cmbData.data(), CMB_MAP_W, CMB_MAP_H);
                cmbReady = true;
            }

            if (volumeReady && m_volumeRenderer && m_volumeRenderer->isReady())
                m_volumeRenderer->updateTexture(cmd, m_volumeData.data(), VOLUME_TEXTURE_N);
            if (cmbReady && m_cmbRenderer && m_cmbRenderer->isReady())
                m_cmbRenderer->updateTexture(cmd, m_cmbData.data(), CMB_MAP_W, CMB_MAP_H);
        }

        // ── Offscreen pass: 3D content (Running state only) ──
        if (m_appState == AppState::Running && m_simulation) {
            uint32_t currentFrame = m_renderer->currentFrame();
            m_offscreenTarget->beginRenderPass(cmd, currentFrame);
            renderRunningState(cmd);
            m_offscreenTarget->endRenderPass(cmd);
        }

        // ── Swapchain pass: ImGui only ──
        m_renderer->beginRenderPass(*m_swapchain, imageIndex);

        m_imguiBackend->newFrame();

        switch (m_appState) {
        case AppState::Gallery:
            renderGalleryState();
            break;
        case AppState::Detail:
            renderDetailState();
            break;
        case AppState::Running:
        {
            // ── DockSpace over entire window ──
            ImGuiID dockspaceId = ImGui::DockSpaceOverViewport(
                0, ImGui::GetMainViewport(),
                ImGuiDockNodeFlags_PassthruCentralNode);

            // ── Default dock layout (first time only) ──
            //
            //  ┌─────────┬──────────────────────────┐
            //  │ Config  │                           │
            //  │         │       Viewport            │
            //  ├─────────┤       (large)             │
            //  │  Node   │                           │
            //  │  Props  ├───────────────────────────┤
            //  │         │  Node Editor (timeline)   │
            //  └─────────┴───────────────────────────┘
            //
            if (!m_dockLayoutInitialized) {
                m_dockLayoutInitialized = true;

                ImGui::DockBuilderRemoveNode(dockspaceId);
                ImGui::DockBuilderAddNode(dockspaceId,
                    ImGuiDockNodeFlags_DockSpace);
                ImGui::DockBuilderSetNodeSize(dockspaceId,
                    ImGui::GetMainViewport()->Size);

                // 1. Split left column (20%) from right area
                ImGuiID dockLeft = 0;
                ImGuiID dockRight = 0;
                ImGui::DockBuilderSplitNode(
                    dockspaceId, ImGuiDir_Left, 0.20f, &dockLeft, &dockRight);

                // 2. Split right: viewport top (70%) / node editor bottom (30%)
                ImGuiID dockViewport = 0;
                ImGuiID dockBottom = 0;
                ImGui::DockBuilderSplitNode(
                    dockRight, ImGuiDir_Up, 0.70f, &dockViewport, &dockBottom);

                // 3. Split left column: config top (55%) / node properties bottom (45%)
                ImGuiID dockLeftTop = 0;
                ImGuiID dockLeftBottom = 0;
                ImGui::DockBuilderSplitNode(
                    dockLeft, ImGuiDir_Up, 0.55f, &dockLeftTop, &dockLeftBottom);

                // Dock windows using stable ### IDs
                ImGui::DockBuilderDockWindow("###SimConfig", dockLeftTop);
                ImGui::DockBuilderDockWindow("###SciencePanel", dockLeftTop);
                ImGui::DockBuilderDockWindow("Viewport", dockViewport);
                ImGui::DockBuilderDockWindow("###NodeEditor", dockBottom);
                ImGui::DockBuilderDockWindow("###Timeline", dockBottom);
                ImGui::DockBuilderDockWindow("###NodeProperties", dockLeftBottom);

                ImGui::DockBuilderFinish(dockspaceId);
            }

            // ── Viewport window (3D offscreen content) ──
            if (m_simulation) {
                ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
                ImGui::Begin("Viewport");
                ImGui::PopStyleVar();

                m_viewportHovered = ImGui::IsWindowHovered();

                // Save viewport content top-left for cursor-relative calculations
                ImVec2 vpScreen = ImGui::GetCursorScreenPos();
                m_viewportScreenX = vpScreen.x;
                m_viewportScreenY = vpScreen.y;

                ImVec2 contentSize = ImGui::GetContentRegionAvail();
                uint32_t newW = std::max(static_cast<uint32_t>(contentSize.x), 1u);
                uint32_t newH = std::max(static_cast<uint32_t>(contentSize.y), 1u);

                if (newW != m_offscreenTarget->extent().width ||
                    newH != m_offscreenTarget->extent().height) {
                    m_viewportWidth = newW;
                    m_viewportHeight = newH;
                    m_pendingViewportResize = true;
                    m_viewportResizeTimer = 0.0f;
                }

                uint32_t currentFrame = m_renderer->currentFrame();
                ImGui::Image(
                    reinterpret_cast<ImTextureID>(m_offscreenTarget->imguiTexture(currentFrame)),
                    contentSize);

                // Camera mode overlay (bottom-left of viewport)
                {
                    bool isFree = (m_camera->mode() == CameraMode::Free);
                    const char* label = isFree ? "Free Camera" : "Orbit Camera";
                    ImVec2 textSize = ImGui::CalcTextSize(label);
                    ImVec2 overlayPos(vpScreen.x + 8.0f,
                                      vpScreen.y + contentSize.y - textSize.y - 8.0f);
                    ImDrawList* dl = ImGui::GetWindowDrawList();
                    dl->AddRectFilled(ImVec2(overlayPos.x - 4, overlayPos.y - 2),
                                      ImVec2(overlayPos.x + textSize.x + 4, overlayPos.y + textSize.y + 2),
                                      IM_COL32(0, 0, 0, 140), 4.0f);
                    dl->AddText(overlayPos, IM_COL32(255, 255, 255, 220), label);

                    if (isFree) {
                        char speedBuf[64];
                        snprintf(speedBuf, sizeof(speedBuf), "Speed: %.1f  [Scroll]  Click to exit",
                                 m_camera->freeMoveSpeed);
                        ImVec2 sz2 = ImGui::CalcTextSize(speedBuf);
                        ImVec2 p2(vpScreen.x + 8.0f,
                                  overlayPos.y - sz2.y - 6.0f);
                        dl->AddRectFilled(ImVec2(p2.x - 4, p2.y - 2),
                                          ImVec2(p2.x + sz2.x + 4, p2.y + sz2.y + 2),
                                          IM_COL32(0, 0, 0, 140), 4.0f);
                        dl->AddText(p2, IM_COL32(255, 255, 255, 220), speedBuf);
                    }
                }

                ImGui::End();
            }

            // DebugUI + SciencePanel rendered here
            if (m_simulation) {
                bool resetRequested = false;
                bool zeldovichRequested = false;
                bool zeldovichStopRequested = false;
                auto currentBackend = m_simulation->backend();

                bool usingNodeGraph = (currentBackend == ComputeBackend::NodeGraph);
                bool usingCDT3D = (currentBackend == ComputeBackend::CDT3D);
                bool usingPM = (currentBackend == ComputeBackend::PM);
                bool usingCDT2D = (currentBackend == ComputeBackend::CDT2D);
                bool usingInflation = (currentBackend == ComputeBackend::Inflation);

                if (usingNodeGraph) {
                    m_debugUI->renderNodeGraph(m_simulation->pmParams(),
                                                m_timer->fps(), m_paused,
                                                currentBackend, resetRequested);
                    auto editorResult = m_simulation->nodeEditorUI().render(
                        m_simulation->simGraph(),
                        m_simulation->compiledPlan(),
                        !m_paused,
                        m_simulation->cudaAvailable());
                    if (editorResult.buildRequested) m_simulation->buildNodeGraph();
                    if (editorResult.resetRequested) {
                        m_pendingReset = true;
                        m_pendingIC = m_simulation->currentCondition();
                    }
                } else if (usingCDT3D) {
                    m_debugUI->renderCDT3D(m_simulation->cdt3dParams(),
                                            m_timer->fps(), m_paused,
                                            currentBackend, resetRequested,
                                            m_simulation->cdt3dState(),
                                            m_simulation->cdt3dKernelTimeMs());
                } else if (usingPM) {
                    m_debugUI->renderPM(m_simulation->pmParams(),
                                         m_timer->fps(), m_paused,
                                         currentBackend, resetRequested,
                                         m_simulation->pmState(),
                                         m_simulation->pmKernelTimeMs());
                } else if (usingCDT2D) {
                    m_debugUI->renderCDT2D(m_simulation->cdt2dParams(),
                                            m_timer->fps(), m_paused,
                                            currentBackend, resetRequested,
                                            m_simulation->cdt2dState());
                } else if (usingInflation) {
                    m_debugUI->renderInflation(m_simulation->inflationParams(),
                                                m_timer->fps(), m_paused,
                                                currentBackend, resetRequested,
                                                m_simulation->inflationState(),
                                                m_simulation->inflationKernelTimeMs(),
                                                m_simulation->zeldovichActive(),
                                                m_simulation->zeldovichGrowth(),
                                                zeldovichRequested,
                                                zeldovichStopRequested);
                } else {
                    m_debugUI->render(m_simulation->params(), m_timer->fps(), m_paused,
                                      currentBackend, resetRequested,
                                      m_simulation->cudaKernelTimeMs(),
                                      m_simulation->cudaTreeNodeCount());
                }

                m_sciencePanel->render(currentBackend);

                // Timeline panel
                if (m_timelineUI)
                    m_timelineUI->render(m_recorder.get(), m_player.get(),
                                         m_timelineState, dt);

                if (resetRequested) {
                    m_pendingReset = true;
                    m_pendingIC = m_simulation->currentCondition();
                }
                if (zeldovichRequested) m_simulation->startZeldovich();
                if (zeldovichStopRequested) m_simulation->stopZeldovich();

                // Check if DebugUI requested return to gallery
                if (m_debugUI->backToGalleryRequested) {
                    m_debugUI->backToGalleryRequested = false;
                    m_backToGalleryRequested = true;
                }

                // Process timeline state changes
                if (m_timelineState.wantRecord && m_recorder && m_simulation) {
                    // Generate recording path
                    auto now = std::chrono::system_clock::now();
                    auto t = std::chrono::system_clock::to_time_t(now);
                    std::tm tm{};
#ifdef _WIN32
                    localtime_s(&tm, &t);
#else
                    localtime_r(&t, &tm);
#endif
                    std::ostringstream oss;
                    oss << std::put_time(&tm, "%Y%m%d_%H%M%S");

                    std::string simName = "sim";
                    if (m_selectedSimIndex >= 0)
                        simName = m_catalog->entry(m_selectedSimIndex).title;
                    // Sanitize name for filename
                    for (auto& c : simName)
                        if (c == ' ' || c == '/' || c == '\\') c = '_';

                    std::filesystem::path recDir = std::filesystem::current_path() / "recordings";
                    std::filesystem::create_directories(recDir);
                    std::string recPath = (recDir / (simName + "_" + oss.str() + ".cosmsnap")).string();

                    uint32_t pc = m_simulation->particleSystem().particleCount();
                    m_recorder->start(recPath, pc,
                                      static_cast<uint32_t>(m_timelineState.captureStride),
                                      static_cast<uint32_t>(m_simulation->backend()));
                    m_timelineState.wantRecord = false;
                }
                if (m_timelineState.wantStopRecord && m_recorder) {
                    m_recorder->stop();
                    m_timelineState.wantStopRecord = false;
                }
                if (m_timelineState.wantOpen && m_player) {
                    if (m_player->open(m_timelineState.filePath)) {
                        m_playbackMode = true;
                        m_paused = true;
                        m_timelineState.currentFrame = 0;
                        m_timelineState.playing = false;
                    }
                    m_timelineState.wantOpen = false;
                }
                if (m_timelineState.wantClose && m_player) {
                    m_player->close();
                    m_playbackMode = false;
                    m_timelineState.wantClose = false;
                }
            }

            // Handle back-to-gallery from DebugUI
            if (m_backToGalleryRequested) {
                m_backToGalleryRequested = false;
                if (m_camera->mode() == CameraMode::Free) {
                    m_camera->setMode(CameraMode::Orbit);
                    glfwSetInputMode(m_window->handle(), GLFW_RAW_MOUSE_MOTION, GLFW_FALSE);
                    glfwSetInputMode(m_window->handle(), GLFW_CURSOR, GLFW_CURSOR_NORMAL);
                }
                cleanupSimulation();
                m_appState = AppState::Gallery;
                m_selectedSimIndex = -1;
            }
            break;
        }
        }

        m_imguiBackend->render(cmd);
        m_renderer->endRenderPass();
        m_renderer->endFrame(*m_context, *m_swapchain, imageIndex);

        // Update simulation params if changed via UI
        if (m_appState == AppState::Running && m_simulation) {
            m_simulation->updateParams(*m_context);
        }
    }

    vkDeviceWaitIdle(m_context->device());
}

// ─── Gallery state ──────────────────────────────────────────────────

void Application::renderGalleryState() {
    int clickedIndex = -1;
    if (m_catalog->renderGallery(clickedIndex)) {
        m_selectedSimIndex = clickedIndex;
        m_appState = AppState::Detail;
    }
}

// ─── Detail state ───────────────────────────────────────────────────

void Application::renderDetailState() {
    bool launchRequested = false;
    bool editRequested = false;
    bool backRequested = false;
    int selectedScenarioIndex = -2;
    m_catalog->renderDetail(m_selectedSimIndex, launchRequested, editRequested,
                            backRequested, selectedScenarioIndex);

    // Scenario card clicked (New Scenario = -1, saved = 0+)
    if (selectedScenarioIndex >= -1 && m_selectedSimIndex >= 0) {
        const auto& selEntry = m_catalog->entry(m_selectedSimIndex);
        InitialCondition ic;
        if (selectedScenarioIndex == -1) {
            // "+ New Scenario" — use first availableCondition
            ic = initialConditionFromString(selEntry.availableConditions[0]);
        } else {
            const auto* sc = m_catalog->scenario(m_selectedSimIndex, selectedScenarioIndex);
            ic = sc ? sc->initialCondition : selEntry.initialCondition;
        }
        initSimulation(m_selectedSimIndex, ic);

        // Apply scenario-specific params (overrides parent defaults)
        if (selectedScenarioIndex >= 0) {
            const auto* sc = m_catalog->scenario(m_selectedSimIndex, selectedScenarioIndex);
            if (sc) {
                m_simulation->params() = sc->params;
                if (selEntry.backend == ComputeBackend::PM ||
                    selEntry.backend == ComputeBackend::NodeGraph) {
                    m_simulation->pmParams() = sc->pmParams;
                }
            }
        }

        m_pendingReset = true;  // Force reset with the scenario's IC
        m_appState = AppState::Running;
    }

    if ((launchRequested || editRequested) && m_selectedSimIndex >= 0) {
        const auto& selEntry = m_catalog->entry(m_selectedSimIndex);
        ComputeBackend originalBackend = selEntry.backend;
        initSimulation(m_selectedSimIndex);
        // Edit Pipeline: force NodeGraph backend to open node editor
        if (editRequested && m_simulation->backend() != ComputeBackend::NodeGraph) {
            m_simulation->setBackend(*m_context, ComputeBackend::NodeGraph);
            // Recreate particle renderer for NodeGraph backend
            if (m_particleRenderer) {
                m_particleRenderer->destroy(m_context->device());
            } else {
                m_particleRenderer = std::make_unique<ParticleRenderer>();
            }
            VkDescriptorSetLayout texLayout = m_planetTextures
                ? m_planetTextures->descriptorSetLayout() : VK_NULL_HANDLE;
            m_particleRenderer->init(*m_context, m_offscreenTarget->renderPass(),
                                      m_simulation->particleSystem().graphicsSetLayout(),
                                      m_shaderDir, texLayout);
        }
        // If editing a BruteForce or BarnesHut sim, replace the default PM graph
        // with an N-Body direct graph
        if (editRequested &&
            (originalBackend == ComputeBackend::BruteForce ||
             originalBackend == ComputeBackend::BarnesHut)) {
            // Clear the PM graph that setBackend() populated and replace with N-Body
            m_simulation->simGraph().clear();
            m_simulation->populateDefaultNBodyGraph();
            m_simulation->buildNodeGraph();
        }
        m_appState = AppState::Running;
    }
    if (backRequested) {
        m_appState = AppState::Gallery;
        m_selectedSimIndex = -1;
    }
}

// renderRunningState() lives in AppSceneRender.cpp.

// ─── Cleanup ────────────────────────────────────────────────────────

void Application::cleanup() {
    if (m_context && m_context->device()) {
        vkDeviceWaitIdle(m_context->device());

        // Cleanup simulation if running
        cleanupSimulation();

        // Cleanup catalog
        if (m_catalog) {
            m_catalog->destroy(*m_context);
            m_catalog.reset();
        }

        // Cleanup offscreen target
        if (m_offscreenTarget) {
            m_offscreenTarget->destroy(m_context->device(), m_context->allocator());
            m_offscreenTarget.reset();
        }

        if (m_imguiBackend) m_imguiBackend->destroy(m_context->device());

        if (m_renderer) m_renderer->destroy(m_context->device());
        if (m_swapchain) m_swapchain->destroy(m_context->device(), m_context->allocator());

        m_context->destroy();
    }
}

void Application::run() {
    mainLoop();
}

} // namespace cosmico
