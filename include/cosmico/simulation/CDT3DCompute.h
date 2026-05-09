#pragma once
#include <cosmico/simulation/CDT3DParams.h>
#include <cosmico/simulation/SimplicialComplex3D.h>
#include <vector>
#include <cstdint>

namespace cosmico {

struct CDT3DStateData {
    int totalSweeps = 0;
    int totalTetrahedra = 0;
    int totalVertices = 0;
    float acceptanceRate = 0.0f;
    float acceptanceRate26 = 0.0f;
    float acceptanceRate44 = 0.0f;
    float acceptanceRate23 = 0.0f;
    std::vector<int> volumeProfile;        // N2(t) spatial triangles per slice
    std::vector<float> avgVolumeProfile;   // Running <N2(t)>
    float hausdorffDimension = 0.0f;
    float spectralDimension = 0.0f;
    std::vector<float> vertexData;         // Render data (readback from SSBO)
    int triangleCount = 0;
};

class CDT3DCompute {
public:
    void init(const CDT3DParams& params);
    void destroy();
    void reset(const CDT3DParams& params);
    void step(const CDT3DParams& params);

    const CDT3DStateData& state() const { return m_state; }
    float kernelTimeMs() const { return m_kernelTimeMs; }

    // Vulkan-CUDA interop: set SSBO device pointer for direct write
    void setRenderBuffer(void* cudaPtr, size_t size);

    static size_t estimateVRAM(int maxTets);

private:
    // CPU topology
    SimplicialComplex3D m_complex;
    int m_T = 0;

    // Pachner moves (CPU)
    void metropolisSweep(const CDT3DParams& params);
    bool tryMove26(const CDT3DParams& p);
    bool tryMove62(const CDT3DParams& p);
    bool tryMove44(const CDT3DParams& p);
    bool tryMove23(const CDT3DParams& p);
    bool tryMove32(const CDT3DParams& p);
    float actionChange(int dN0, int dN3, const CDT3DParams& p) const;

    // CUDA device arrays (void* to keep header CUDA-free)
    void* m_d_vertexPos = nullptr;      // float3[] vertex sphere positions
    void* m_d_triIndices = nullptr;     // int3[] spatial triangle indices
    void* m_d_triSlice = nullptr;       // int[] which slice each triangle belongs to
    void* m_d_curvature = nullptr;      // float[] per-vertex curvature
    void* m_d_renderVerts = nullptr;    // float[9][] output render vertices (interop SSBO)
    void* m_d_observables = nullptr;    // Reduction buffers for d_H, d_s
    void* m_d_rwStates = nullptr;       // Random walk return probabilities
    void* m_d_volumeProfile = nullptr;  // int[T] volume profile

    // Topology upload buffers
    void* m_d_tetVertices = nullptr;    // int[numTets*4] flat tet vertex indices
    void* m_d_tetNeighbors = nullptr;   // int[numTets*4] flat neighbor indices
    void* m_d_tetAlive = nullptr;       // int[numTets] alive flags
    void* m_d_vertexSlice = nullptr;    // int[numVerts] time slice per vertex

    // Edge list for embedding
    void* m_d_edges = nullptr;          // int2[] edge list
    int m_numEdges = 0;

    // CSR for curvature
    void* m_d_triPerVertex = nullptr;   // int[numVerts+1] CSR offsets
    void* m_d_triList = nullptr;        // int[] CSR triangle indices

    // CUDA stream + events (timing)
    void* m_stream = nullptr;           // cudaStream_t
    void* m_startEvent = nullptr;
    void* m_stopEvent = nullptr;

    // State
    CDT3DStateData m_state;
    float m_kernelTimeMs = 0.0f;
    std::vector<double> m_avgN2;
    std::vector<float> m_smoothN2;
    int m_measureCount = 0;

    // MC stats
    uint64_t m_rngState = 0;
    int m_attempted = 0, m_accepted = 0;
    int m_attempted26 = 0, m_accepted26 = 0;
    int m_attempted44 = 0, m_accepted44 = 0;
    int m_attempted23 = 0, m_accepted23 = 0;

    // Interop
    bool m_hasRenderBuffer = false;
    size_t m_renderBufferSize = 0;

    // GPU capacity tracking
    int m_gpuMaxVerts = 0;
    int m_gpuMaxTets = 0;
    int m_gpuMaxTris = 0;

    // Persistent sphere embedding positions (survive across frames)
    std::vector<float> m_embedX, m_embedY, m_embedZ;
    std::vector<bool> m_embedInitialized;  // Has this vertex been placed?
    std::vector<bool> m_embedAnchor;       // Is this an original icosahedron vertex?

    // Internal helpers
    void uploadTopologyToGPU();
    void buildEdgeList(const std::vector<SimplicialComplex3D::Triangle>& allTris);
    void buildCSR(const std::vector<SimplicialComplex3D::Triangle>& allTris, int numVerts);
    void computeObservablesCPU(const CDT3DParams& params);
    void buildRenderDataCPU(const CDT3DParams& params);
};

} // namespace cosmico
