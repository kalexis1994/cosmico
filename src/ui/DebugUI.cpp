#include <cosmico/ui/DebugUI.h>
#include <cosmico/simulation/Simulation.h>
#include <cosmico/simulation/InflationCompute.h>
#include <cosmico/simulation/PMCompute.h>
#include <cosmico/simulation/CDT2DCompute.h>
#include <cosmico/simulation/CDT3DCompute.h>
#include <imgui.h>
#include <cmath>
#include <algorithm>

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
    ImGui::SliderInt("Steps/frame", &params.stepsPerFrame, 1, 20);
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
