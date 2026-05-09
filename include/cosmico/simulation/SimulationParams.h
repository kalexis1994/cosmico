#pragma once

namespace cosmico {

struct SimulationParams {
    float G = 1.0f;             // Gravitational constant
    float dt = 0.001f;          // Timestep
    float softening = 0.5f;     // Softening length to avoid singularities
    int particleCount = 4096;   // Number of particles
    float theta = 0.5f;        // Barnes-Hut opening angle (0=exact, 1=fast)
    bool showDarkMatter = false; // Render dark matter particles (type < 0)
};

} // namespace cosmico
