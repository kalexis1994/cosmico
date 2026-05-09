#include <cosmico/ui/DebugUI.h>
#include <cosmico/simulation/Simulation.h>

#include <imgui.h>
#include <cmath>
#include <string>

namespace cosmico {

void DebugUI::render(SimulationParams& params, float fps, bool& paused,
                      ComputeBackend backend,
                      bool& resetRequested,
                      float cudaKernelTimeMs, int cudaTreeNodeCount) {
    ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(340, 0), ImGuiCond_FirstUseEver);

    ImGui::Begin("Cosmico - Debug###SimConfig");

    if (ImGui::SmallButton("<< Gallery")) { backToGalleryRequested = true; }
    ImGui::Separator();

    // Performance
    ImGui::Text("FPS: %.1f (%.2f ms)", fps, fps > 0 ? 1000.0f / fps : 0.0f);
    ImGui::Text("Particles: %s",
        params.particleCount >= 1000000 ? "1M" :
        params.particleCount >= 1000 ?
            (std::to_string(params.particleCount / 1000) + "K").c_str() :
            std::to_string(params.particleCount).c_str());
    ImGui::Separator();

    // Backend label (read-only — selected from gallery)
    ImGui::Text("Backend: %s", computeBackendName(backend));
    ImGui::Separator();

    // CUDA stats (when Barnes-Hut is active)
    if (backend == ComputeBackend::BarnesHut) {
        ImGui::Text("Barnes-Hut Stats");
        ImGui::Text("  Kernel time: %.2f ms", cudaKernelTimeMs);
        ImGui::Text("  Tree nodes: %d", cudaTreeNodeCount);
        ImGui::SliderFloat("Theta", &params.theta, 0.1f, 1.5f, "%.2f");
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Opening angle: lower = more accurate, higher = faster");
        }
        ImGui::Separator();
    }

    // Simulation controls
    ImGui::Text("Simulation");
    ImGui::Checkbox("Paused", &paused);
    ImGui::SameLine();
    resetRequested = ImGui::Button("Reset");

    ImGui::SliderFloat("G (gravity)", &params.G, 0.01f, 10.0f, "%.3f", ImGuiSliderFlags_Logarithmic);
    ImGui::SliderFloat("dt (timestep)", &params.dt, 0.000001f, 0.01f, "%.6f", ImGuiSliderFlags_Logarithmic);
    ImGui::SliderFloat("Softening", &params.softening, 0.01f, 5.0f, "%.2f");
    ImGui::Checkbox("Show Dark Matter", &params.showDarkMatter);
    ImGui::Separator();

    // Particle count — presets + manual input
    bool isBarnesHut = (backend == ComputeBackend::BarnesHut);
    int maxCount = isBarnesHut ? 1048576 : 65536;

    ImGui::Text("Particle Count");

    // Presets combo with more options
    if (isBarnesHut) {
        static const int counts[] = {
            256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536,
            100000, 131072, 200000, 262144, 500000, 524288, 750000, 1048576
        };
        static const char* labels[] = {
            "256", "512", "1K", "2K", "4K", "8K", "16K", "32K", "65K",
            "100K", "131K", "200K", "262K", "500K", "524K", "750K", "1M"
        };
        static constexpr int numCounts = 17;

        int currentIdx = -1;
        for (int i = 0; i < numCounts; i++) {
            if (counts[i] == params.particleCount) { currentIdx = i; break; }
        }
        const char* preview = (currentIdx >= 0) ? labels[currentIdx] : "Custom";
        if (ImGui::BeginCombo("Presets", preview)) {
            for (int i = 0; i < numCounts; i++) {
                bool selected = (counts[i] == params.particleCount);
                if (ImGui::Selectable(labels[i], selected)) {
                    params.particleCount = counts[i];
                    resetRequested = true;
                }
                if (selected) ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }
    } else {
        static const int counts[] = {
            128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536
        };
        static const char* labels[] = {
            "128", "256", "512", "1K", "2K", "4K", "8K", "16K", "32K", "65K"
        };
        static constexpr int numCounts = 10;

        int currentIdx = -1;
        for (int i = 0; i < numCounts; i++) {
            if (counts[i] == params.particleCount) { currentIdx = i; break; }
        }
        const char* preview = (currentIdx >= 0) ? labels[currentIdx] : "Custom";
        if (ImGui::BeginCombo("Presets", preview)) {
            for (int i = 0; i < numCounts; i++) {
                bool selected = (counts[i] == params.particleCount);
                if (ImGui::Selectable(labels[i], selected)) {
                    params.particleCount = counts[i];
                    resetRequested = true;
                }
                if (selected) ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }
    }

    // Manual input
    static int manualCount = 0;
    if (manualCount == 0) manualCount = params.particleCount;
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 70);
    ImGui::InputInt("##manual", &manualCount, 0, 0);
    if (manualCount < 2) manualCount = 2;
    if (manualCount > maxCount) manualCount = maxCount;
    ImGui::SameLine();
    if (ImGui::Button("Apply")) {
        params.particleCount = manualCount;
        resetRequested = true;
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Max: %d", maxCount);
    }
    // Keep manual field in sync when presets change
    if (resetRequested) manualCount = params.particleCount;

    ImGui::End();
}

// renderInflation, renderPM, renderCDT2D, renderCDT3D live in their own
// per-backend TUs (DebugUI_Inflation.cpp etc.).

// ── Node Graph ────────────────────────────────────────────────────────────

void DebugUI::renderNodeGraph(PMParams& params, float fps, bool& paused,
                               ComputeBackend backend, bool& resetRequested) {
    ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(340, 0), ImGuiCond_FirstUseEver);

    ImGui::Begin("Cosmico - Node Graph###SimConfig");

    if (ImGui::SmallButton("<< Gallery")) { backToGalleryRequested = true; }
    ImGui::Separator();

    // Performance
    ImGui::Text("FPS: %.1f (%.2f ms)", fps, fps > 0 ? 1000.0f / fps : 0.0f);
    ImGui::Text("Particles: %s",
        params.particleCount >= 1000000 ?
            (std::to_string(params.particleCount / 1000000) + "M").c_str() :
        params.particleCount >= 1000 ?
            (std::to_string(params.particleCount / 1000) + "K").c_str() :
            std::to_string(params.particleCount).c_str());
    ImGui::Text("Grid: %d^3", params.gridN);
    ImGui::Separator();

    // Backend label (read-only — selected from gallery)
    ImGui::Text("Backend: %s", computeBackendName(backend));
    ImGui::Separator();

    // Pause/Resume
    if (ImGui::Button(paused ? "Resume" : "Pause")) {
        paused = !paused;
    }
    ImGui::SameLine();
    if (ImGui::Button("Reset")) {
        resetRequested = true;
    }
    ImGui::Separator();

    // Simulation parameters (shared with PM)
    ImGui::Text("Pipeline Parameters");
    ImGui::SliderFloat("G", &params.G, 0.01f, 10.0f, "%.3f");
    ImGui::SliderFloat("dt", &params.dt, 0.000001f, 0.01f, "%.6f", ImGuiSliderFlags_Logarithmic);
    ImGui::SliderInt("Steps/Frame", &params.stepsPerFrame, 1, 50);
    ImGui::SliderFloat("Box Size", &params.boxSize, 10.0f, 500.0f, "%.1f");
    ImGui::SliderFloat("Damping", &params.damping, 0.0f, 1.0f, "%.3f");
    ImGui::Separator();

    // Grid size
    ImGui::Text("Grid Size");
    int gridOptions[] = {64, 128, 256, 512};
    const char* gridLabels[] = {"64", "128", "256", "512"};
    int currentGridIdx = 1;
    for (int i = 0; i < 4; i++) {
        if (gridOptions[i] == params.gridN) { currentGridIdx = i; break; }
    }
    if (ImGui::Combo("Grid N", &currentGridIdx, gridLabels, 4)) {
        params.gridN = gridOptions[currentGridIdx];
    }
    ImGui::Separator();

    // Particle count presets
    ImGui::Text("Particle Count");
    int presets[] = {128, 1024, 4096, 16384, 32768, 65536, 131072, 262144, 524288, 1048576};
    const char* presetLabels[] = {"128", "1K", "4K", "16K", "32K", "64K", "128K", "256K", "512K", "1M"};
    int presetCount = 10;
    int currentPreset = -1;
    for (int i = 0; i < presetCount; i++) {
        if (presets[i] == params.particleCount) { currentPreset = i; break; }
    }
    if (ImGui::Combo("Particles", &currentPreset, presetLabels, presetCount)) {
        if (currentPreset >= 0 && currentPreset < presetCount) {
            params.particleCount = presets[currentPreset];
            resetRequested = true;
        }
    }
    ImGui::Separator();
    ImGui::TextDisabled("Use the Node Graph Editor window");
    ImGui::TextDisabled("to wire simulation pipeline nodes.");

    ImGui::End();
}

} // namespace cosmico
