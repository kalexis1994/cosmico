#pragma once

namespace cosmico {

struct CDT2DParams {
    int T = 40;                   // Number of time slices
    int N_target = 10000;         // Target total triangles N2
    float lambda = 0.693f;        // Cosmological constant (~ln2 = critical)
    float epsilon = 0.02f;        // Volume fixing strength
    int sweepsPerFrame = 100;     // MC sweeps per render frame
    int initialN = 10;            // Initial vertices per slice
    bool showWireframe = true;
    bool colorByCurvature = true;
    float meshScale = 1.0f;
    float renderSmoothing = 0.95f; // EMA factor: 0=raw, 0.99=very smooth
};

} // namespace cosmico
