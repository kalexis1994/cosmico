#pragma once

namespace cosmico {

struct CDT3DParams {
    int T = 20;                    // Time slices (periodic)
    int N3_target = 10000;         // Target tetrahedra
    float k0 = 1.0f;              // ~inverse Newton (coeff of N0)
    float k3 = 0.693f;            // Cosmological constant (coeff of N3)
    float epsilon = 0.01f;        // Volume fixing (N3 - N3_target)^2
    int sweepsPerFrame = 10;
    int initialVerticesPerSlice = 12; // Icosahedron

    // Move probabilities (sum to 1)
    float prob26 = 0.4f;          // (2,6)/(6,2)
    float prob44 = 0.3f;          // (4,4)
    float prob23 = 0.3f;          // (2,3)/(3,2)

    // Rendering
    bool showWireframe = true;
    bool colorByCurvature = true;
    float meshScale = 1.0f;
    float renderSmoothing = 0.90f;
    float sphereSpacing = 3.0f;   // Vertical separation between slices
    float lightIntensity = 0.8f;
};

} // namespace cosmico
