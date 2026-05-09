#include "kernels/nbody_direct.cuh"

namespace cosmico {
namespace cuda {

static constexpr int NBODY_BLOCK = 256;

// ---- Clear Accel: zero out acceleration scratch fields ----

__global__ void clearAccelKernel(ParticleGpu* __restrict__ particles, int N) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= N) return;
    particles[gid].density  = 0.0f;
    particles[gid].pressure = 0.0f;
    particles[gid].smoothing = 0.0f;
}

void launchClearAccel(ParticleGpu* particles, int N, cudaStream_t stream) {
    int blocks = (N + NBODY_BLOCK - 1) / NBODY_BLOCK;
    clearAccelKernel<<<blocks, NBODY_BLOCK, 0, stream>>>(particles, N);
}

// ---- Pairwise Gravity: O(N^2) tiled all-pairs ----
// Each thread accumulates gravitational acceleration for one particle
// by iterating over all others in tiles of NBODY_BLOCK using shared memory.
// Stores result in density/pressure/smoothing fields (ax/ay/az).

__global__ void pairwiseGravityKernel(ParticleGpu* __restrict__ particles,
                                       int N, float G, float eps2) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float4 sharedPos[NBODY_BLOCK]; // xyz = position, w = mass

    float3 pos = make_float3(0.0f, 0.0f, 0.0f);
    bool active = (gid < N);

    if (active) {
        pos = make_float3(particles[gid].px, particles[gid].py, particles[gid].pz);
    }

    float ax = 0.0f, ay = 0.0f, az = 0.0f;

    int tileCount = (N + NBODY_BLOCK - 1) / NBODY_BLOCK;

    for (int tile = 0; tile < tileCount; tile++) {
        int tileIdx = tile * NBODY_BLOCK + threadIdx.x;
        if (tileIdx < N) {
            sharedPos[threadIdx.x] = make_float4(
                particles[tileIdx].px,
                particles[tileIdx].py,
                particles[tileIdx].pz,
                particles[tileIdx].mass);
        } else {
            sharedPos[threadIdx.x] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
        __syncthreads();

        if (active) {
            for (int j = 0; j < NBODY_BLOCK; j++) {
                int idx = tile * NBODY_BLOCK + j;
                if (idx >= N || idx == gid) continue;

                float rx = sharedPos[j].x - pos.x;
                float ry = sharedPos[j].y - pos.y;
                float rz = sharedPos[j].z - pos.z;

                float distSq = rx * rx + ry * ry + rz * rz + eps2;
                float invDist = rsqrtf(distSq);
                float invDist3 = invDist * invDist * invDist;

                float f = G * sharedPos[j].w * invDist3;
                ax += f * rx;
                ay += f * ry;
                az += f * rz;
            }
        }
        __syncthreads();
    }

    // Store acceleration in scratch fields
    if (active) {
        particles[gid].density  = ax;
        particles[gid].pressure = ay;
        particles[gid].smoothing = az;
    }
}

void launchPairwiseGravity(ParticleGpu* particles, int N, float G,
                            float softening, cudaStream_t stream) {
    float eps2 = softening * softening;
    int blocks = (N + NBODY_BLOCK - 1) / NBODY_BLOCK;
    pairwiseGravityKernel<<<blocks, NBODY_BLOCK, 0, stream>>>(particles, N, G, eps2);
}

} // namespace cuda
} // namespace cosmico
