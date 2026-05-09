#pragma once
#include <cosmico/simulation/SimulationParams.h>
#include <cosmico/simulation/InitialConditions.h>
#include <cosmico/simulation/InflationParams.h>
#include <cosmico/simulation/PMParams.h>
#include <cosmico/simulation/CDT2DParams.h>
#include <cosmico/simulation/CDT3DParams.h>
#include <cstdint>
#include <vector>

namespace cosmico {

// Forward declarations
enum class ComputeBackend;
const char* computeBackendName(ComputeBackend b);
struct InflationStateData;
struct PMStateData;
struct CDT2DStateData;
struct CDT3DStateData;
struct BarnesHutStats;

class DebugUI {
public:
    void render(SimulationParams& params, float fps, bool& paused,
                ComputeBackend backend,
                bool& resetRequested,
                float cudaKernelTimeMs, int cudaTreeNodeCount);

    void renderBarnesHut(SimulationParams& params, float fps, bool& paused,
                         ComputeBackend backend, bool& resetRequested,
                         const BarnesHutStats* stats);

    void renderInflation(InflationParams& params, float fps, bool& paused,
                         ComputeBackend backend, bool& resetRequested,
                         const InflationStateData* state,
                         float kernelTimeMs,
                         bool zeldovichActive, float zeldovichGrowth,
                         bool& zeldovichRequested, bool& zeldovichStopRequested);

    void renderPM(PMParams& params, float fps, bool& paused,
                  ComputeBackend backend, bool& resetRequested,
                  const PMStateData* state,
                  float kernelTimeMs);

    void renderCDT2D(CDT2DParams& params, float fps, bool& paused,
                     ComputeBackend backend, bool& resetRequested,
                     const CDT2DStateData* state);

    void renderCDT3D(CDT3DParams& params, float fps, bool& paused,
                     ComputeBackend backend, bool& resetRequested,
                     const CDT3DStateData* state,
                     float kernelTimeMs);

    void renderNodeGraph(PMParams& params, float fps, bool& paused,
                         ComputeBackend backend, bool& resetRequested);

    // Set by any render method when user clicks "Back to Gallery"
    bool backToGalleryRequested = false;

private:
    // Rolling history of E_total and momentum magnitude for the
    // Barnes-Hut diagnostics plots. Cleared when stats->stepCount
    // resets to a smaller value (i.e. user reset the simulation).
    std::vector<float> m_bhEnergyHistory;
    std::vector<float> m_bhMomentumHistory;
    uint64_t m_bhLastStep = 0;
    static constexpr int kBhHistMax = 256;
};

} // namespace cosmico
