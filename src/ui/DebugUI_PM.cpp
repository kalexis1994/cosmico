#include <cosmico/ui/DebugUI.h>
#include <cosmico/simulation/PMCompute.h>

#include <imgui.h>
#include <vector>
#include <string>
#include <cmath>

namespace cosmico {

void DebugUI::renderPM(PMParams& params, float fps, bool& paused,
                        ComputeBackend backend, bool& resetRequested,
                        const PMStateData* state,
                        float kernelTimeMs) {
    ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(380, 0), ImGuiCond_FirstUseEver);

    ImGui::Begin("Cosmico - PM Gravity###SimConfig");

    if (ImGui::SmallButton("<< Gallery")) { backToGalleryRequested = true; }
    ImGui::Separator();

    // Performance
    ImGui::Text("FPS: %.1f (%.2f ms)", fps, fps > 0 ? 1000.0f / fps : 0.0f);
    ImGui::Text("Kernel: %.2f ms", kernelTimeMs);
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

    // Simulation controls
    ImGui::Text("Simulation");
    ImGui::Checkbox("Paused", &paused);
    ImGui::SameLine();
    resetRequested = ImGui::Button("Reset");

    // Grid size
    static const int gridSizes[] = { 64, 128, 256, 512 };
    static const char* gridLabels[] = { "64", "128", "256", "512" };
    int gridIdx = -1;
    for (int i = 0; i < 4; i++) {
        if (gridSizes[i] == params.gridN) { gridIdx = i; break; }
    }
    const char* gridPreview = (gridIdx >= 0) ? gridLabels[gridIdx] : "Custom";
    if (ImGui::BeginCombo("Grid N", gridPreview)) {
        for (int i = 0; i < 4; i++) {
            bool selected = (gridSizes[i] == params.gridN);
            if (ImGui::Selectable(gridLabels[i], selected)) {
                params.gridN = gridSizes[i];
            }
            if (selected) ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
    }

    if (!params.comoving) {
        ImGui::SliderFloat("G (gravity)", &params.G, 0.01f, 10.0f, "%.3f", ImGuiSliderFlags_Logarithmic);
    }
    ImGui::SliderFloat("dt (timestep)", &params.dt, 0.000001f, 0.01f, "%.6f", ImGuiSliderFlags_Logarithmic);
    ImGui::SliderFloat("Box Size", &params.boxSize, 10.0f, 500.0f, "%.0f");
    ImGui::SliderInt("Steps/frame", &params.stepsPerFrame, 1, 40);
    ImGui::SliderFloat("Force smoothing", &params.forceSmoothing, 0.2f, 3.0f, "%.2f");
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("PM force Gaussian smoothing (x particle spacing)\n"
                          "lower = sharper filaments/halos, higher = smoother/diffuse");
    }
    ImGui::Checkbox("Open Boundary", &params.openBoundary);
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("No periodic wrapping (infinite box)");
    }
    ImGui::Separator();

    // Particle count presets
    ImGui::Text("Particle Count");
    {
        static const int counts[] = {
            32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304
        };
        static const char* labels[] = {
            "32K", "64K", "128K", "256K", "512K", "1M", "2M", "4M"
        };
        static constexpr int numCounts = 8;

        int currentIdx = -1;
        for (int i = 0; i < numCounts; i++) {
            if (counts[i] == params.particleCount) { currentIdx = i; break; }
        }
        const char* preview = (currentIdx >= 0) ? labels[currentIdx] : "Custom";
        if (ImGui::BeginCombo("Presets##pm", preview)) {
            for (int i = 0; i < numCounts; i++) {
                bool selected = (counts[i] == params.particleCount);
                if (ImGui::Selectable(labels[i], selected)) {
                    params.particleCount = counts[i];
                    // Auto-scale boxSize to maintain inter-particle spacing
                    params.boxSize = 1.5625f * std::cbrt(static_cast<float>(counts[i]));
                    resetRequested = true;
                }
                if (selected) ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }
    }
    ImGui::Separator();

    // Cosmological expansion
    ImGui::Text("Cosmological Expansion");
    if (ImGui::Checkbox("Enable Expansion", &params.comoving)) {
        if (params.comoving) {
            resetRequested = true;  // Reset to apply initial scale factor
        }
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Friedmann expansion: H(a) = H0*sqrt(Om/a^3 + OL)\n"
                          "Adds Hubble drag + derived G from H0, Om\n"
                          "Damping acts as pressure support");
    }
    if (params.comoving) {
        ImGui::SliderFloat("H0 (sim units)", &params.H0, 0.001f, 0.1f, "%.4f", ImGuiSliderFlags_Logarithmic);
        ImGui::SliderFloat("Omega_M", &params.OmegaM, 0.1f, 2.0f, "%.2f");
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("1.0 = Einstein-de Sitter (matter only)\n"
                              "0.3 = LCDM (with dark energy)");
        }
        ImGui::SliderFloat("a_init", &params.aInit, 0.01f, 1.0f, "%.3f", ImGuiSliderFlags_Logarithmic);
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Initial scale factor (z_init = 1/a - 1)\n"
                              "0.1 = z~9, 0.5 = z~1, 1.0 = today");
        }
        ImGui::SliderFloat("a_max (freeze)", &params.aMax, 0.5f, 100.0f, "%.1f", ImGuiSliderFlags_Logarithmic);
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Freeze expansion past this scale factor.\n"
                              "High = keep expanding (web freezes in place);\n"
                              "low = stop expanding (keeps collapsing to center)");
        }
        ImGui::SliderFloat("IC amplitude", &params.zeldovichAmplitude, 0.05f, 0.8f, "%.2f");
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Initial Zel'dovich displacement (x spacing).\n"
                              "Higher = more initial structure / faster collapse.\n"
                              "Press Reset to apply.");
        }
        ImGui::SliderFloat("Damping", &params.damping, 0.0f, 20.0f, "%.1f");
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Pressure support / velocity dispersion\n"
                              "Prevents total collapse of nonlinear structures\n"
                              "Combined with Hubble drag");
        }
    } else {
        ImGui::SliderFloat("Damping", &params.damping, 0.0f, 20.0f, "%.1f");
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Artificial velocity friction");
        }
    }
    ImGui::Separator();

    // Momentum correction
    ImGui::Checkbox("Momentum Correction", &params.correctMomentum);
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Subtract mean velocity each step to prevent net drift");
    }
    ImGui::Separator();

    // Sink particles (black hole formation)
    ImGui::Text("Sink Particles (Black Holes)");
    ImGui::BeginDisabled(!paused);
    ImGui::Checkbox("Enable Sink Formation", &params.enableSink);
    ImGui::EndDisabled();
    if (!paused && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
        ImGui::SetTooltip("Pause simulation to toggle sink particles");
    }
    if (params.enableSink) {
        ImGui::SliderFloat("Density Threshold", &params.sinkDensityThreshold,
                           5.0f, 200.0f, "%.0f x mean");
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Density above this multiple of mean triggers sink formation");
        }
        ImGui::SliderFloat("Accretion Radius", &params.sinkRadius,
                           0.5f, 10.0f, "%.1f cells");
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Particles within this radius of a sink get absorbed");
        }
        if (state) {
            ImGui::Text("  Sinks: %d | Absorbed: %d", state->sinkCount, state->absorbedCount);
        }
    }
    ImGui::Separator();

    // State display
    if (state) {
        ImGui::Text("State");
        ImGui::Text("  t = %.4f", state->time);
        ImGui::Text("  step = %d", state->step);
        if (params.comoving) {
            ImGui::Text("  a(t) = %.6f", state->scaleFactor);
            ImGui::Text("  z = %.2f", state->redshift);
            ImGui::Text("  H(a) = %.6f", state->hubble);
            ImGui::Checkbox("Physical view (expand by a)", &params.physicalView);
            if (params.physicalView) {
                float raw = (params.aInit > 1e-6f)
                          ? (float)(state->scaleFactor / params.aInit) : 1.0f;
                ImGui::SameLine();
                if (raw > 2.5f) ImGui::TextDisabled("(x2.5 cap, real x%.0f)", raw);
                else            ImGui::TextDisabled("(x%.1f)", raw);
            }
        }
        ImGui::Separator();

        ImGui::Text("Energy");
        ImGui::Text("  KE = %.6e", state->kineticEnergy);
        ImGui::Text("  PE = %.6e", state->potentialEnergy);
        ImGui::Text("  Total = %.6e", state->totalEnergy);
        ImGui::Text("  |p| = %.6e", state->momentumMag);
        ImGui::Separator();

        // Power spectrum plot
        if (params.showPowerSpectrum && !state->powerSpectrum.empty()) {
            ImGui::Text("Power Spectrum log10 P(k)");

            std::vector<float> logP(state->powerSpectrum.size());
            for (size_t i = 0; i < state->powerSpectrum.size(); i++) {
                logP[i] = (state->powerSpectrum[i] > 0)
                    ? log10f(state->powerSpectrum[i]) : -20.0f;
            }

            static float pkMin = -20.0f, pkMax = 0.0f;
            float frameMin = 1e30f, frameMax = -1e30f;
            for (size_t i = 0; i < logP.size(); i++) {
                if (logP[i] < -19.0f) continue;
                if (logP[i] < frameMin) frameMin = logP[i];
                if (logP[i] > frameMax) frameMax = logP[i];
            }
            if (frameMin < frameMax) {
                pkMin += (frameMin - 1.0f - pkMin) * 0.1f;
                pkMax += (frameMax + 1.0f - pkMax) * 0.1f;
            }

            ImGui::PlotLines("##pkpm", logP.data(), (int)logP.size(),
                0, nullptr, pkMin, pkMax, ImVec2(0, 120));
        }
        ImGui::Checkbox("Show P(k)", &params.showPowerSpectrum);
    } else {
        ImGui::TextDisabled("No PM state available");
    }

    // VRAM estimate
    ImGui::Separator();
    size_t vramBytes = PMCompute::estimateVRAM(params.gridN, params.particleCount);
    ImGui::Text("Est. VRAM: %zu MB", vramBytes / (1024 * 1024));

    ImGui::End();
}

} // namespace cosmico
