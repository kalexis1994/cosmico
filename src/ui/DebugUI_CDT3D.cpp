#include <cosmico/ui/DebugUI.h>
#include <cosmico/simulation/CDT3DCompute.h>

#include <imgui.h>
#include <vector>

namespace cosmico {

void DebugUI::renderCDT3D(CDT3DParams& params, float fps, bool& paused,
                           ComputeBackend backend, bool& resetRequested,
                           const CDT3DStateData* state,
                           float kernelTimeMs) {
    ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(380, 0), ImGuiCond_FirstUseEver);

    ImGui::Begin("Cosmico - CDT 3D###SimConfig");

    if (ImGui::SmallButton("<< Gallery")) { backToGalleryRequested = true; }
    ImGui::Separator();

    // Performance
    ImGui::Text("FPS: %.1f (%.2f ms)", fps, fps > 0 ? 1000.0f / fps : 0.0f);
    ImGui::Text("Kernel: %.2f ms", kernelTimeMs);
    if (state) {
        ImGui::Text("Sweeps: %d", state->totalSweeps);
    }
    ImGui::Separator();

    // Backend label (read-only — selected from gallery)
    ImGui::Text("Backend: %s", computeBackendName(backend));
    ImGui::Separator();

    // Monte Carlo Controls
    ImGui::Text("Monte Carlo");
    ImGui::Checkbox("Paused", &paused);
    ImGui::SameLine();
    resetRequested = ImGui::Button("Reset");

    int oldT = params.T;
    ImGui::SliderInt("T (slices)", &params.T, 5, 50);
    if (params.T != oldT) resetRequested = true;

    ImGui::SliderInt("N3 target", &params.N3_target, 1000, 100000, "%d", ImGuiSliderFlags_Logarithmic);
    ImGui::SliderFloat("k0", &params.k0, 0.0f, 5.0f, "%.3f");
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Inverse Newton constant (coefficient of N0)");
    }
    ImGui::SliderFloat("k3", &params.k3, 0.0f, 3.0f, "%.3f");
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Cosmological constant (coefficient of N3)");
    }
    ImGui::SliderFloat("epsilon", &params.epsilon, 0.001f, 0.1f, "%.3f", ImGuiSliderFlags_Logarithmic);
    ImGui::SliderInt("Sweeps/frame", &params.sweepsPerFrame, 1, 200, "%d", ImGuiSliderFlags_Logarithmic);

    // Move probabilities
    if (ImGui::TreeNode("Move Probabilities")) {
        ImGui::SliderFloat("(2,6)/(6,2)", &params.prob26, 0.0f, 1.0f, "%.2f");
        ImGui::SliderFloat("(4,4)", &params.prob44, 0.0f, 1.0f, "%.2f");
        ImGui::SliderFloat("(2,3)/(3,2)", &params.prob23, 0.0f, 1.0f, "%.2f");

        // Normalize
        float total = params.prob26 + params.prob44 + params.prob23;
        if (total > 0.001f) {
            params.prob26 /= total;
            params.prob44 /= total;
            params.prob23 /= total;
        }
        ImGui::TreePop();
    }
    ImGui::Separator();

    // State
    if (state) {
        ImGui::Text("State");
        ImGui::Text("  N3 (tetrahedra): %d", state->totalTetrahedra);
        ImGui::Text("  N0 (vertices): %d", state->totalVertices);
        ImGui::Text("  Acceptance: %.1f%%", state->acceptanceRate * 100.0f);

        if (ImGui::TreeNode("Acceptance by Move")) {
            ImGui::Text("  (2,6)/(6,2): %.1f%%", state->acceptanceRate26 * 100.0f);
            ImGui::Text("  (4,4):       %.1f%%", state->acceptanceRate44 * 100.0f);
            ImGui::Text("  (2,3)/(3,2): %.1f%%", state->acceptanceRate23 * 100.0f);
            ImGui::TreePop();
        }
        ImGui::Separator();

        // Volume profile N2(t)
        if (!state->volumeProfile.empty()) {
            ImGui::Text("Volume Profile N2(t)");
            std::vector<float> profile(state->volumeProfile.size());
            float maxN = 0.0f;
            for (size_t i = 0; i < state->volumeProfile.size(); i++) {
                profile[i] = static_cast<float>(state->volumeProfile[i]);
                if (profile[i] > maxN) maxN = profile[i];
            }
            ImGui::PlotLines("##n2t", profile.data(), static_cast<int>(profile.size()),
                0, nullptr, 0.0f, maxN * 1.2f, ImVec2(0, 100));
        }

        // Average volume profile
        if (!state->avgVolumeProfile.empty() && state->totalSweeps > 0) {
            ImGui::Text("Average <N2(t)>");
            int T = static_cast<int>(state->avgVolumeProfile.size());
            std::vector<float> avg(T);
            float maxAvg = 0.0f;
            for (int i = 0; i < T; i++) {
                avg[i] = state->avgVolumeProfile[i];
                if (avg[i] > maxAvg) maxAvg = avg[i];
            }
            ImGui::PlotLines("##avgn2t", avg.data(), T,
                0, nullptr, 0.0f, maxAvg * 1.2f, ImVec2(0, 100));
            ImGui::TextDisabled("Expected: cos^3(pi*t/T) in phase C");
        }

        // Observables
        ImGui::Separator();
        ImGui::Text("Observables");
        ImGui::Text("  d_H (Hausdorff): %.2f", state->hausdorffDimension);
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Expected: ~3.0 after thermalization");
        }
        ImGui::Text("  d_s (Spectral):  %.2f", state->spectralDimension);
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Expected: ~3.0 (large scale), ~2.0 (small scale)");
        }

        // Phase hint
        ImGui::Separator();
        ImGui::Text("Phase Hint");
        float k0 = params.k0;
        float k3 = params.k3;
        if (k0 < 0.5f) {
            ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f), "  Phase A (crumpled)");
        } else if (k3 > 2.0f) {
            ImGui::TextColored(ImVec4(0.0f, 0.5f, 1.0f, 1.0f), "  Phase B (branched polymer)");
        } else {
            ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "  Phase C (de Sitter)");
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Approximate based on (k0=%.2f, k3=%.2f)\n"
                              "Phase C is the physical phase with semiclassical geometry",
                              k0, k3);
        }
    } else {
        ImGui::TextDisabled("No CDT 3D state available");
    }

    ImGui::Separator();

    // Visual controls
    ImGui::Text("Visualization");
    ImGui::Checkbox("Wireframe", &params.showWireframe);
    ImGui::Checkbox("Color by Curvature", &params.colorByCurvature);
    ImGui::SliderFloat("Mesh Scale", &params.meshScale, 0.1f, 5.0f, "%.2f");
    ImGui::SliderFloat("Render Smoothing", &params.renderSmoothing, 0.0f, 0.99f, "%.2f");
    ImGui::SliderFloat("Sphere Spacing", &params.sphereSpacing, 0.5f, 10.0f, "%.1f");
    ImGui::SliderFloat("Light Intensity", &params.lightIntensity, 0.0f, 2.0f, "%.2f");

    ImGui::End();
}

} // namespace cosmico
