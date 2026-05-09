#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace cosmico {
namespace cuda {

// Shared GPU parameters for CDT 3D kernels
struct CDT3DGpuParams {
    int T;
    int numVertices;
    int numTetrahedra;
    int numSpatialTriangles;
    int numEdges;
    float sphereSpacing;
    float meshScale;
    int colorByCurvature;
    float renderSmoothing;
};

// ─── Host-callable kernel wrappers ──────────────────────────────────────

// Force-directed sphere embedding: relax vertex positions on S^2
void launchSphereEmbedKernel(
    float3* positions,           // [numVerts] in/out positions on sphere
    const int2* edges,           // [numEdges] edge list
    int numVerts,
    int numEdges,
    float targetEdgeLen,
    float radius,
    float dt,                    // relaxation step
    cudaStream_t stream
);

// Build render data: per-triangle compute normals + colors, write to SSBO
void launchBuildRenderDataKernel(
    const float3* vertexPos,     // Embedded positions
    const int3* triangles,       // Spatial triangulation
    const int* triSlice,         // Slice assignment per triangle
    const float* curvature,      // Per-vertex curvature
    float* renderOutput,         // [numTris*3*9] -> SSBO (x,y,z,nx,ny,nz,r,g,b)
    int numTriangles,
    float yOffset,               // Vertical spacing between slices
    int colorByCurvature,
    int T,
    cudaStream_t stream
);

// Parallel random walks on dual graph for spectral dimension
void launchRandomWalkKernel(
    const int* tetNeighbors,     // [numTets*4] adjacency flat array
    const int* tetAlive,         // [numTets] alive flags (int for GPU)
    int numTets,
    int numWalks,
    int walkLength,              // sigma max
    float* returnProb,           // [walkLength] output P(sigma)
    unsigned long long seed,
    cudaStream_t stream
);

// Volume profile: count spatial triangles per time slice
void launchVolumeProfileKernel(
    const int* tetVertices,      // [numTets*4] flat vertex indices
    const int* vertexSlice,      // [numVerts] time slice per vertex
    const int* tetAlive,         // [numTets] alive flags
    int numTets,
    int T,
    int* volumeProfile,          // [T] output N2(t) spatial triangles
    cudaStream_t stream
);

// Gaussian curvature at each vertex: 2*pi - sum(incident angles)
void launchCurvatureKernel(
    const float3* positions,
    const int3* triangles,
    const int* triPerVertex,     // CSR start indices [numVerts+1]
    const int* triList,          // CSR triangle indices
    float* curvature,            // [numVerts] output
    int numVerts,
    cudaStream_t stream
);

} // namespace cuda
} // namespace cosmico
