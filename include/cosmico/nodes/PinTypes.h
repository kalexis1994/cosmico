#pragma once
#include <cstdint>
#include <imgui.h>

namespace cosmico {

enum class PinType : uint8_t {
    Particles,     // ParticleGpu* buffer
    DensityGrid,   // float[N^3] real grid
    ComplexGrid,   // cufftComplex[N*N*(N/2+1)]
    ForceField,    // float[N^3] single component
    Scalar,        // float (dt, damping, etc.)
    Stats,         // energy/momentum readback
};

inline const char* pinTypeName(PinType t) {
    switch (t) {
        case PinType::Particles:   return "Particles";
        case PinType::DensityGrid: return "DensityGrid";
        case PinType::ComplexGrid: return "ComplexGrid";
        case PinType::ForceField:  return "ForceField";
        case PinType::Scalar:      return "Scalar";
        case PinType::Stats:       return "Stats";
    }
    return "Unknown";
}

inline ImU32 pinTypeColor(PinType t) {
    switch (t) {
        case PinType::Particles:   return IM_COL32(100, 220, 100, 255); // green
        case PinType::DensityGrid: return IM_COL32(80, 150, 230, 255);  // blue
        case PinType::ComplexGrid: return IM_COL32(180, 100, 220, 255); // purple
        case PinType::ForceField:  return IM_COL32(230, 160, 50, 255);  // orange
        case PinType::Scalar:      return IM_COL32(180, 180, 180, 255); // gray
        case PinType::Stats:       return IM_COL32(230, 230, 80, 255);  // yellow
    }
    return IM_COL32(255, 255, 255, 255);
}

} // namespace cosmico
