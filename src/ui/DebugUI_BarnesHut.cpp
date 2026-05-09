#include <cosmico/ui/DebugUI.h>
#include <cosmico/simulation/Simulation.h>
#include <cosmico/simulation/BarnesHutCompute.h>

#include <imgui.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>

namespace cosmico {

namespace {

// Format a particle count compactly: 65536 -> "64K", 1048576 -> "1M".
std::string formatCount(int n) {
    if (n >= 1000000) {
        char buf[32]; std::snprintf(buf, sizeof(buf), "%dM", n / 1000000); return buf;
    }
    if (n >= 1000) {
        char buf[32]; std::snprintf(buf, sizeof(buf), "%dK", n / 1000); return buf;
    }
    return std::to_string(n);
}

// Color a number red/yellow/green based on |drift| magnitude.
ImVec4 driftColor(double absDrift) {
    if (absDrift < 1e-3) return ImVec4(0.4f, 1.0f, 0.5f, 1.0f);  // green: <0.1%
    if (absDrift < 1e-2) return ImVec4(1.0f, 0.9f, 0.3f, 1.0f);  // yellow: <1%
    return ImVec4(1.0f, 0.4f, 0.3f, 1.0f);                       // red: >=1%
}

// Push then pop a sample on a rolling history buffer of fixed max size.
void pushHistory(std::vector<float>& buf, float sample, int maxSize) {
    if (static_cast<int>(buf.size()) >= maxSize) {
        buf.erase(buf.begin());
    }
    buf.push_back(sample);
}

} // namespace

void DebugUI::renderBarnesHut(SimulationParams& params, float fps, bool& paused,
                               ComputeBackend backend, bool& resetRequested,
                               const BarnesHutStats* stats) {
    ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(380, 0), ImGuiCond_FirstUseEver);

    ImGui::Begin("Cosmico - Barnes-Hut###SimConfig");

    if (ImGui::SmallButton("<< Gallery")) { backToGalleryRequested = true; }
    ImGui::Separator();

    // ─── Performance ───────────────────────────────────────────────
    ImGui::Text("FPS: %.1f (%.2f ms)", fps, fps > 0 ? 1000.0f / fps : 0.0f);
    if (stats) {
        ImGui::Text("Kernel: %.2f ms", stats->kernelTimeMs);
        ImGui::Text("Tree nodes: %d", stats->nodeCount);
        if (params.particleCount > 0) {
            float perPart = static_cast<float>(stats->nodeCount) /
                            static_cast<float>(params.particleCount);
            ImGui::SameLine();
            ImGui::TextDisabled("(%.2f / particle)", perPart);
        }
    }
    ImGui::Text("Particles: %s", formatCount(params.particleCount).c_str());
    if (stats) {
        ImGui::Text("Sim time: %.4f", stats->simTime);
        ImGui::SameLine();
        ImGui::TextDisabled("step %llu",
            static_cast<unsigned long long>(stats->stepCount));
    }
    ImGui::Separator();

    // Backend label (read-only — selected from gallery)
    ImGui::Text("Backend: %s", computeBackendName(backend));
    ImGui::Separator();

    // ─── Simulation controls ───────────────────────────────────────
    ImGui::Text("Simulation");
    ImGui::Checkbox("Paused", &paused);
    ImGui::SameLine();
    resetRequested = ImGui::Button("Reset");

    ImGui::SliderFloat("G (gravity)", &params.G,
        0.01f, 10.0f, "%.3f", ImGuiSliderFlags_Logarithmic);
    ImGui::SliderFloat("dt (timestep)", &params.dt,
        0.000001f, 0.01f, "%.6f", ImGuiSliderFlags_Logarithmic);
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Smaller dt -> better energy conservation, slower sim.\n"
                          "Watch the drift indicator below to gauge stability.");
    }
    ImGui::SliderFloat("Softening", &params.softening, 0.01f, 5.0f, "%.2f");
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Plummer softening: prevents singular forces at\n"
                          "close encounters. Larger -> smoother but less accurate.");
    }
    ImGui::SliderFloat("Theta", &params.theta, 0.0f, 1.5f, "%.2f");
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Barnes-Hut opening angle (s/d criterion):\n"
                          "  0.0 = exact O(N^2), no approximation\n"
                          "  0.5 = standard accuracy/speed balance\n"
                          "  1.0 = aggressive, faster but coarser forces\n"
                          "Lower theta -> more tree nodes visited per step.");
    }
    ImGui::Checkbox("Show Dark Matter", &params.showDarkMatter);
    ImGui::Separator();

    // ─── Particle Count ────────────────────────────────────────────
    ImGui::Text("Particle Count");
    {
        static const int counts[] = {
            256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536,
            100000, 131072, 200000, 262144, 500000, 524288, 750000, 1048576
        };
        static const char* labels[] = {
            "256", "512", "1K", "2K", "4K", "8K", "16K", "32K", "65K",
            "100K", "131K", "200K", "262K", "500K", "524K", "750K", "1M"
        };
        constexpr int numCounts = 17;

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
    constexpr int maxCount = 1048576;
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
    if (resetRequested) manualCount = params.particleCount;
    ImGui::Separator();

    // ─── Conservation diagnostics ──────────────────────────────────
    ImGui::Text("Conservation");
    if (!stats || !stats->diagValid) {
        ImGui::TextDisabled("Computing... (run a few steps)");
    } else {
        // Detect reset of underlying sim — clear plot history.
        if (stats->stepCount < m_bhLastStep) {
            m_bhEnergyHistory.clear();
            m_bhMomentumHistory.clear();
        }
        m_bhLastStep = stats->stepCount;

        pushHistory(m_bhEnergyHistory, static_cast<float>(stats->totalEnergy), kBhHistMax);
        pushHistory(m_bhMomentumHistory, static_cast<float>(stats->momentumMag), kBhHistMax);

        ImGui::Text("Energy");
        ImGui::Text("  KE = %.4e", stats->kineticEnergy);
        ImGui::Text("  PE = %.4e", stats->potentialEnergy);
        ImGui::Text("  E  = %.4e", stats->totalEnergy);
        ImGui::Text("  Drift:");
        ImGui::SameLine();
        double drift = stats->energyDrift;
        ImGui::TextColored(driftColor(std::abs(drift)),
            "%+.3e (%+.3f%%)", drift, drift * 100.0);
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("(E - E_initial) / |E_initial|.\n"
                              "Symplectic leapfrog should keep this bounded.\n"
                              "Sustained growth -> dt too large or theta too coarse.");
        }

        // Energy plot — auto-scaled with smoothed bounds to avoid jitter.
        if (m_bhEnergyHistory.size() >= 2) {
            float emin = *std::min_element(m_bhEnergyHistory.begin(), m_bhEnergyHistory.end());
            float emax = *std::max_element(m_bhEnergyHistory.begin(), m_bhEnergyHistory.end());
            float pad = std::max((emax - emin) * 0.2f, 1e-6f);
            ImGui::PlotLines("##E", m_bhEnergyHistory.data(),
                static_cast<int>(m_bhEnergyHistory.size()),
                0, "E(t)", emin - pad, emax + pad, ImVec2(0, 70));
        }

        ImGui::Spacing();
        ImGui::Text("Momentum");
        ImGui::Text("  |P| = %.4e", stats->momentumMag);
        ImGui::Text("  Px  = %+.4e", stats->momentum[0]);
        ImGui::Text("  Py  = %+.4e", stats->momentum[1]);
        ImGui::Text("  Pz  = %+.4e", stats->momentum[2]);
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Total linear momentum. For a closed system\n"
                              "with no net push, this should stay near zero.");
        }

        if (m_bhMomentumHistory.size() >= 2) {
            float pmax = *std::max_element(m_bhMomentumHistory.begin(), m_bhMomentumHistory.end());
            ImGui::PlotLines("##P", m_bhMomentumHistory.data(),
                static_cast<int>(m_bhMomentumHistory.size()),
                0, "|P|(t)", 0.0f, pmax * 1.2f + 1e-6f, ImVec2(0, 50));
        }

        ImGui::Spacing();
        ImGui::Text("Center of Mass");
        ImGui::Text("  M  = %.4e", stats->totalMass);
        ImGui::Text("  COM = (%+.3f, %+.3f, %+.3f)",
            stats->centerOfMass[0],
            stats->centerOfMass[1],
            stats->centerOfMass[2]);
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Mass-weighted barycenter of all particles.\n"
                              "Should remain stationary (drift = momentum / total mass).");
        }
    }

    ImGui::End();
}

} // namespace cosmico
