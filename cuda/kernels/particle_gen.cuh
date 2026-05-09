#pragma once
#include <kernels/barnes_hut.cuh>   // ParticleGpu
#include <cuda_runtime.h>

namespace cosmico {
namespace cuda {

struct ICGenParams {
    int icType;             // 0=Sphere, 1=Disk, 2=TwoBody, 3=GalaxyCollision, 4=Cosmological, 5=SolarSystem
    int particleCount;
    float massPerParticle;
    unsigned int seed;

    // Sphere
    float radius;
    float velocitySpread;

    // Disk
    float diskRadius;
    float centralMass;
    float heightScale;

    // TwoBody
    float separation;
    float bodyMass;

    // GalaxyCollision
    float approachSpeed;

    // Cosmological
    float jitterFraction;
    float boxSize;

    // Advanced profiles
    int densityProfile;     // 0=Uniform, 1=Plummer, 2=Hernquist, 3=NFW
    float profileScale;
    int velocityDist;       // 0=Zero, 1=Uniform, 2=MaxwellBoltzmann
    float temperature;
    int massFunction;       // 0=Equal, 1=PowerLaw
    float powerLawIndex;
    int noiseType;          // 0=None, 1=Perlin, 2=White
    float noiseAmplitude;
};

void launchICGenerate(ParticleGpu* particles, const ICGenParams& params, cudaStream_t stream);

} // namespace cuda
} // namespace cosmico
