#pragma once
#include <kernels/barnes_hut.cuh>   // ParticleGpu
#include <cuda_runtime.h>

namespace cosmico {
namespace cuda {

// Zero acceleration fields (density/pressure/smoothing used as ax/ay/az).
void launchClearAccel(ParticleGpu* particles, int N, cudaStream_t stream);

// O(N^2) tiled all-pairs gravity: computes acceleration and stores in
// density/pressure/smoothing fields (ax/ay/az).
void launchPairwiseGravity(ParticleGpu* particles, int N, float G,
                            float softening, cudaStream_t stream);

} // namespace cuda
} // namespace cosmico
