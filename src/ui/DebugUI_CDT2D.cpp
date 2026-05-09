#include <cosmico/ui/DebugUI.h>
#include <cosmico/simulation/CDT2DCompute.h>

#include <imgui.h>
#include <vector>

namespace cosmico {

void DebugUI::renderCDT2D(CDT2DParams& params, float fps, bool& paused,
                            ComputeBackend backend, bool& resetRequested,
                            const CDT2DStateData* state) {
    ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(380, 0), ImGuiCond_FirstUseEver);

    ImGui::Begin("Cosmico - CDT 2D###SimConfig");

    if (ImGui::SmallButton("<< Gallery")) { backToGalleryRequested = true; }
    ImGui::Separator();

    // Performance
    ImGui::Text("FPS: %.1f (%.2f ms)", fps, fps > 0 ? 1000.0f / fps : 0.0f);
    if (state) {
        ImGui::Text("Sweeps: %d", state->totalSweeps);
    }
    ImGui::Separator();

    // Backend label (read-only — selected from gallery)
    ImGui::Text("Backend: %s", computeBackendName(backend));
    ImGui::Separator();

    // Simulation controls
    ImGui::Text("Monte Carlo");
    ImGui::Checkbox("Paused", &paused);
    ImGui::SameLine();
    resetRequested = ImGui::Button("Reset");

    int oldT = params.T;
    ImGui::SliderInt("T (slices)", &params.T, 10, 100);
    if (params.T != oldT) resetRequested = true;

    ImGui::SliderInt("N_target", &params.N_target, 1000, 100000, "%d", ImGuiSliderFlags_Logarithmic);
    ImGui::SliderFloat("lambda", &params.lambda, 0.0f, 2.0f, "%.3f");
    ImGui::SliderFloat("epsilon", &params.epsilon, 0.001f, 0.1f, "%.3f", ImGuiSliderFlags_Logarithmic);
    ImGui::SliderInt("Sweeps/frame", &params.sweepsPerFrame, 1, 1000, "%d", ImGuiSliderFlags_Logarithmic);
    ImGui::Separator();

    // State
    if (state) {
        ImGui::Text("State");
        ImGui::Text("  N2 (triangles): %d", state->totalTriangles);
        ImGui::Text("  Acceptance rate: %.1f%%", state->acceptanceRate * 100.0f);
        ImGui::Separator();

        // Volume profile plot n(t)
        if (!state->volumeProfile.empty()) {
            ImGui::Text("Volume Profile n(t)");
            std::vector<float> profile(state->volumeProfile.size());
            float maxN = 0.0f;
            for (size_t i = 0; i < state->volumeProfile.size(); i++) {
                profile[i] = static_cast<float>(state->volumeProfile[i]);
                if (profile[i] > maxN) maxN = profile[i];
            }
            ImGui::PlotLines("##nt", profile.data(), static_cast<int>(profile.size()),
                0, nullptr, 0.0f, maxN * 1.2f, ImVec2(0, 100));
        }

        // Average volume profile with cos^2 overlay
        if (!state->avgVolumeProfile.empty() && state->totalSweeps > 0) {
            ImGui::Text("Average <n(t)>");
            int T = static_cast<int>(state->avgVolumeProfile.size());
            std::vector<float> avg(T);
            float maxAvg = 0.0f;
            for (int i = 0; i < T; i++) {
                avg[i] = state->avgVolumeProfile[i];
                if (avg[i] > maxAvg) maxAvg = avg[i];
            }
            ImGui::PlotLines("##avgnt", avg.data(), T,
                0, nullptr, 0.0f, maxAvg * 1.2f, ImVec2(0, 100));

            // Theoretical cos^2 overlay info
            ImGui::TextDisabled("Expected: cos^2(pi*t/T)");
        }

        // Observables
        ImGui::Separator();
        ImGui::Text("Observables");
        ImGui::Text("  d_H (Hausdorff): %.2f", state->hausdorffDimension);
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Expected: 2.0 after thermalization");
        }
        ImGui::Text("  d_s (Spectral):  %.2f", state->spectralDimension);
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Expected: 2.0 after thermalization");
        }
    } else {
        ImGui::TextDisabled("No CDT state available");
    }

    ImGui::Separator();

    // Visual controls
    ImGui::Text("Visualization");
    ImGui::Checkbox("Wireframe", &params.showWireframe);
    ImGui::Checkbox("Color by Curvature", &params.colorByCurvature);
    ImGui::SliderFloat("Mesh Scale", &params.meshScale, 0.1f, 5.0f, "%.2f");
    ImGui::SliderFloat("Render Smoothing", &params.renderSmoothing, 0.0f, 0.99f, "%.2f");
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("0 = raw fluctuations, 0.95 = smooth universe shape");
    }

    ImGui::End();
}

} // namespace cosmico
