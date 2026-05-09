#pragma once
#include <vector>
#include <string>
#include <cstdint>

namespace cosmico {

struct ParticleData {
    float position[4];   // xyz + mass
    float velocity[4];   // xyz + temperature
    float attributes[4]; // density, pressure, smoothing_length, type
};

enum class InitialCondition {
    Sphere,
    Disk,
    TwoBody,
    GalaxyCollision,
    Galaxy,
    Cosmological,
    SolarSystem,
    Count
};

const char* initialConditionName(InitialCondition ic);
InitialCondition initialConditionFromString(const std::string& s);

std::vector<ParticleData> generateInitialConditions(InitialCondition type, uint32_t count,
                                                    float boxSize = 0.0f);

// Number of spherical bodies (rendered with ray-sphere shader) for a given IC.
uint32_t sphereBodyCount(InitialCondition type);

} // namespace cosmico
