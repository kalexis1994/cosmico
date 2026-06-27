#include <cosmico/ui/DebugUI.h>
#include <cosmico/simulation/InflationCompute.h>

#include <imgui.h>
#include <vector>
#include <cmath>

namespace cosmico {

void DebugUI::renderInflation(InflationParams& params, float fps, bool& paused,
                               ComputeBackend backend, bool& resetRequested,
                               const InflationStateData* state,
                               float kernelTimeMs,
                               bool zeldovichActive, float zeldovichGrowth,
                               bool& zeldovichRequested, bool& zeldovichStopRequested) {
    ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(380, 0), ImGuiCond_FirstUseEver);

    ImGui::Begin("Cosmico - Inflation###SimConfig");

    if (ImGui::SmallButton("<< Gallery")) { backToGalleryRequested = true; }
    ImGui::Separator();

    // Performance
    ImGui::Text("FPS: %.1f (%.2f ms)", fps, fps > 0 ? 1000.0f / fps : 0.0f);
    ImGui::Text("Kernel: %.2f ms", kernelTimeMs);
    ImGui::Text("Grid: %d^3 (%d M points)", params.gridN,
        (params.gridN * params.gridN * params.gridN) / 1000000);
    ImGui::Separator();

    // Backend label (read-only — selected from gallery)
    ImGui::Text("Backend: %s", computeBackendName(backend));
    ImGui::Separator();

    // Simulation controls
    ImGui::Text("Simulation");
    ImGui::Checkbox("Paused", &paused);
    ImGui::SameLine();
    resetRequested = ImGui::Button("Reset");

    // Potential selector
    const char* potentialNames[] = { "Quadratic (m^2 phi^2/2)", "Starobinsky", "Hilltop" };
    int pot = static_cast<int>(params.potential);
    if (ImGui::Combo("Potential", &pot, potentialNames, 3)) {
        params.potential = static_cast<InflationParams::Potential>(pot);
        resetRequested = true;
    }

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
                resetRequested = true;
            }
            if (selected) ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
    }

    ImGui::SliderFloat("phi_0", &params.phi0, 0.1f, 25.0f, "%.2f");
    ImGui::SliderFloat("dt", &params.dt, 0.000001f, 0.01f, "%.6f", ImGuiSliderFlags_Logarithmic);
    ImGui::SliderInt("Steps/frame", &params.stepsPerFrame, 1, 50);

    // Potential-specific params
    if (params.potential == InflationParams::Quadratic) {
        ImGui::SliderFloat("m^2", &params.m2, 1e-8f, 1e-4f, "%.2e", ImGuiSliderFlags_Logarithmic);
    } else if (params.potential == InflationParams::Starobinsky) {
        ImGui::SliderFloat("Lambda^4", &params.lambda4, 1e-14f, 1e-8f, "%.2e", ImGuiSliderFlags_Logarithmic);
    } else if (params.potential == InflationParams::Hilltop) {
        ImGui::SliderFloat("Lambda^4", &params.lambda4, 1e-14f, 1e-8f, "%.2e", ImGuiSliderFlags_Logarithmic);
        ImGui::SliderFloat("mu", &params.mu, 1.0f, 30.0f, "%.1f");
    }

    ImGui::Separator();

    // ─── Volume rendering controls ──────────────────────────────────
    ImGui::Text("Volume Rendering");
    ImGui::Checkbox("Show Volume", &params.showVolume);
    if (params.showVolume) {
        ImGui::SliderFloat("Opacity", &params.volumeOpacity, 0.0f, 5.0f, "%.2f");
        ImGui::SliderInt("Ray Steps", &params.volumeSteps, 32, 256);
    }
    ImGui::Separator();

    // ─── CMB Rendering ──────────────────────────────────────────────
    ImGui::Text("CMB Rendering");
    ImGui::Checkbox("Show CMB Sphere", &params.showCMB);
    if (params.showCMB) {
        ImGui::SliderFloat("CMB Radius", &params.cmbRadius, 0.05f, 0.5f, "%.2f");
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Shell radius as fraction of the simulation box");
        }
        ImGui::SliderFloat("CMB Contrast", &params.cmbContrast, 0.1f, 10.0f, "%.2f",
                           ImGuiSliderFlags_Logarithmic);
        ImGui::SliderFloat("CMB Opacity", &params.cmbOpacity, 0.0f, 1.0f, "%.2f");
        ImGui::Checkbox("Immersive sky (skybox around camera)", &params.cmbImmersive);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Centre the CMB sphere on the camera so it surrounds you\n"
                              "in every direction (fly in with Tab). Off = ball at the origin.");
        ImGui::Checkbox("Rigorous CMB (acoustic peaks)", &params.rigorousCMB);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Gaussian a_lm field from a LCDM C_l (real CMB statistics)\n"
                              "instead of a slice of the inflaton field");
        if (params.rigorousCMB) {
            ImGui::SameLine();
            if (ImGui::SmallButton("Regenerate")) params.cmbSeed++;
        }
    }
    ImGui::Separator();

    // ─── Structure Formation (Zel'dovich) ───────────────────────────
    ImGui::Text("Structure Formation");
    if (!zeldovichActive) {
        if (ImGui::Button("Start Zel'dovich")) {
            zeldovichRequested = true;
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Takes current inflation fluctuations and\n"
                              "evolves them into cosmic web structure\n"
                              "using the Zel'dovich approximation");
        }
    } else {
        ImGui::Text("  D(t) = %.4f", zeldovichGrowth);
        ImGui::SliderFloat("Growth Speed", &params.zeldovichSpeed, 0.001f, 0.1f, "%.3f",
                           ImGuiSliderFlags_Logarithmic);
        ImGui::SliderFloat("Log Scale", &params.zeldovichLogScale, 1.0f, 10.0f, "%.1f");
        if (ImGui::Button("Stop Zel'dovich")) {
            zeldovichStopRequested = true;
        }
    }
    ImGui::Separator();

    // ─── State display ─────────────────────────────────────────────
    if (state) {
        ImGui::Text("State");
        ImGui::Text("  t = %.4f", state->time);
        ImGui::Text("  a(t) = %.6f", state->scaleFactor);
        ImGui::Text("  H = %.6e", state->hubble);
        ImGui::Text("  N_e = %.4f", state->efolds);
        ImGui::Text("  <phi> = %.6f", state->phiMean);
        if (state->efolds > 1.0) {
            ImGui::Text("  n_s = %.4f", state->spectralIndex);
        }
        ImGui::Separator();

        ImGui::Text("Energy Densities");
        ImGui::Text("  KE = %.6e", state->kineticEnergy);
        ImGui::Text("  GE = %.6e", state->gradientEnergy);
        ImGui::Text("  PE = %.6e", state->potentialEnergy);
        double total = state->kineticEnergy + state->gradientEnergy + state->potentialEnergy;
        ImGui::Text("  Total = %.6e", total);
        ImGui::Separator();

        // Power spectrum plot
        if (!state->powerSpectrum.empty()) {
            ImGui::Text("Power Spectrum log10 P(k)");

            // Convert to log scale for plotting
            std::vector<float> logP(state->powerSpectrum.size());
            for (size_t i = 0; i < state->powerSpectrum.size(); i++) {
                logP[i] = (state->powerSpectrum[i] > 0)
                    ? log10f(state->powerSpectrum[i]) : -20.0f;
            }

            // Use smoothed bounds to prevent jitter
            static float pkMin = -20.0f, pkMax = 0.0f;
            float frameMin = 1e30f, frameMax = -1e30f;
            for (size_t i = 0; i < logP.size(); i++) {
                if (logP[i] < -19.0f) continue;
                if (logP[i] < frameMin) frameMin = logP[i];
                if (logP[i] > frameMax) frameMax = logP[i];
            }
            if (frameMin < frameMax) {
                // Smooth bounds: slowly adapt (avoids jitter)
                pkMin += (frameMin - 1.0f - pkMin) * 0.1f;
                pkMax += (frameMax + 1.0f - pkMax) * 0.1f;
            }

            ImGui::PlotLines("##pk", logP.data(), (int)logP.size(),
                0, nullptr, pkMin, pkMax, ImVec2(0, 120));
        }

        // Field slice: plot fluctuation delta_phi = phi - <phi>
        if (!state->fieldSlice.empty()) {
            float phiMean = (float)state->phiMean;

            // Subtract mean to show fluctuations
            std::vector<float> deltaPhi(state->fieldSlice.size());
            float maxDev = 0.0f;
            for (size_t i = 0; i < state->fieldSlice.size(); i++) {
                deltaPhi[i] = state->fieldSlice[i] - phiMean;
                float dev = fabsf(deltaPhi[i]);
                if (dev > maxDev) maxDev = dev;
            }

            float range = maxDev * 1.3f + 1e-8f;

            ImGui::Text("delta phi(x)  <phi>=%.4f  max|dphi|=%.2e",
                        phiMean, maxDev);
            ImGui::PlotLines("##slice", deltaPhi.data(),
                (int)deltaPhi.size(),
                0, nullptr, -range, range,
                ImVec2(0, 100));
        }
    } else {
        ImGui::TextDisabled("No inflation state available");
    }

    // VRAM estimate
    ImGui::Separator();
    size_t vramBytes = InflationCompute::estimateVRAM(params.gridN);
    ImGui::Text("Est. VRAM: %zu MB", vramBytes / (1024 * 1024));

    ImGui::End();
}

} // namespace cosmico
