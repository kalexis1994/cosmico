#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace cosmico {
namespace cuda {

// Must match cosmico::ParticleData layout (48 bytes)
struct ParticleGpu {
    float px, py, pz, mass;   // position + mass
    float vx, vy, vz, temp;   // velocity + temperature
    float density, pressure, smoothing, type; // attributes
};

struct OctreeNode {
    float cx, cy, cz;        // Center of mass
    float mass;               // Total mass of subtree
    int children[8];          // -1=empty, -2=locked, >=0=child index
    int particleCount;        // Particles in subtree
    int startIdx;             // Leaf: particle index; Internal: -1
    float minX, minY, minZ;   // Bounding box min
    float maxX, maxY, maxZ;   // Bounding box max
};

struct BarnesHutParams {
    float G;
    float dt;
    float softening;
    int particleCount;
    float theta;
    int maxNodes;             // = 2 * particleCount
};

// Reduction buffer for bounding box computation
struct BoundingBox {
    float minX, minY, minZ;
    float maxX, maxY, maxZ;
};

// Per-step accumulator for energy / momentum / center-of-mass diagnostics.
// Computed by walking the existing octree, so cost is comparable to the
// force kernel. Doubles for accumulation precision; the host divides
// pe by 2 since pairs (i,j) and (j,i) are both counted by the tree walk.
struct DiagnosticsAccum {
    double ke;          // kinetic energy = 0.5 * sum(m_i * v_i^2)
    double pe;          // potential energy * 2 (host halves it)
    double px, py, pz;  // total momentum components
    double mx, my, mz;  // sum(m_i * r_i) — used to derive COM
    double m;           // total mass
};

// Host-callable wrapper for the entire Barnes-Hut pipeline
void launchBarnesHutStep(
    ParticleGpu* particles,
    OctreeNode* nodes,
    BoundingBox* d_bbox,
    int* d_nodeCounter,
    const BarnesHutParams& params,
    cudaStream_t stream
);

// Compute per-step diagnostics. Reads the freshly-built tree from
// launchBarnesHutStep, so call this AFTER launchBarnesHutStep on the
// same stream to reuse the up-to-date tree.
void launchDiagnostics(
    const ParticleGpu* particles,
    const OctreeNode* nodes,
    const BarnesHutParams& params,
    int rootIdx,
    DiagnosticsAccum* d_accum,
    cudaStream_t stream
);

// Get required tree memory size in bytes for N particles
size_t getTreeMemorySize(int particleCount);

} // namespace cuda
} // namespace cosmico
