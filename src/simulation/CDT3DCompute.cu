#include <cosmico/simulation/CDT3DCompute.h>
#include <kernels/cdt3d.cuh>
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <cstring>
#include <chrono>
#include <unordered_set>
#include <set>
#include <map>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "[CDT3D] CUDA Error: %s at %s:%d\n", \
                    cudaGetErrorString(err), __FILE__, __LINE__); \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err)); \
        } \
    } while(0)

namespace cosmico {

// ── Fast xoshiro256** RNG (same as CDT2D) ──────────────────────────────

static inline uint64_t rotl64(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

struct Xoshiro256ss {
    uint64_t s[4];

    void seed(uint64_t v) {
        for (int i = 0; i < 4; i++) {
            v += 0x9e3779b97f4a7c15ULL;
            uint64_t z = v;
            z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
            z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
            s[i] = z ^ (z >> 31);
        }
    }

    uint64_t next() {
        uint64_t result = rotl64(s[1] * 5, 7) * 9;
        uint64_t t = s[1] << 17;
        s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
        s[2] ^= t; s[3] = rotl64(s[3], 45);
        return result;
    }

    int nextInt(int n) {
        return static_cast<int>(next() % static_cast<uint64_t>(n));
    }

    float nextFloat() {
        return static_cast<float>(next() >> 11) * 0x1.0p-53f;
    }
};

static Xoshiro256ss s_rng3d;

// ── VRAM estimation ─────────────────────────────────────────────────────

size_t CDT3DCompute::estimateVRAM(int maxTets) {
    size_t total = 0;
    int maxVerts = maxTets;
    int maxTris = maxTets * 2;
    total += maxVerts * sizeof(float) * 3;
    total += maxTris * sizeof(int) * 3;
    total += maxTris * sizeof(int);
    total += maxVerts * sizeof(float);
    total += maxTris * 3 * 9 * sizeof(float);
    total += maxTets * 4 * sizeof(int);
    total += maxTets * 4 * sizeof(int);
    total += maxTets * sizeof(int);
    total += maxVerts * sizeof(int);
    total += 1024 * sizeof(float);
    return total;
}

// ── Init / Destroy / Reset ──────────────────────────────────────────────

void CDT3DCompute::init(const CDT3DParams& params) {
    auto now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    s_rng3d.seed(static_cast<uint64_t>(now));
    m_rngState = static_cast<uint64_t>(now);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    m_stream = stream;

    cudaEvent_t startEvt, stopEvt;
    CUDA_CHECK(cudaEventCreate(&startEvt));
    CUDA_CHECK(cudaEventCreate(&stopEvt));
    m_startEvent = startEvt;
    m_stopEvent = stopEvt;

    reset(params);
}

void CDT3DCompute::destroy() {
    if (m_d_vertexPos) { cudaFree(m_d_vertexPos); m_d_vertexPos = nullptr; }
    if (m_d_triIndices) { cudaFree(m_d_triIndices); m_d_triIndices = nullptr; }
    if (m_d_triSlice) { cudaFree(m_d_triSlice); m_d_triSlice = nullptr; }
    if (m_d_curvature) { cudaFree(m_d_curvature); m_d_curvature = nullptr; }
    if (m_d_observables) { cudaFree(m_d_observables); m_d_observables = nullptr; }
    if (m_d_rwStates) { cudaFree(m_d_rwStates); m_d_rwStates = nullptr; }
    if (m_d_volumeProfile) { cudaFree(m_d_volumeProfile); m_d_volumeProfile = nullptr; }
    if (m_d_tetVertices) { cudaFree(m_d_tetVertices); m_d_tetVertices = nullptr; }
    if (m_d_tetNeighbors) { cudaFree(m_d_tetNeighbors); m_d_tetNeighbors = nullptr; }
    if (m_d_tetAlive) { cudaFree(m_d_tetAlive); m_d_tetAlive = nullptr; }
    if (m_d_vertexSlice) { cudaFree(m_d_vertexSlice); m_d_vertexSlice = nullptr; }
    if (m_d_edges) { cudaFree(m_d_edges); m_d_edges = nullptr; }
    if (m_d_triPerVertex) { cudaFree(m_d_triPerVertex); m_d_triPerVertex = nullptr; }
    if (m_d_triList) { cudaFree(m_d_triList); m_d_triList = nullptr; }

    if (!m_hasRenderBuffer && m_d_renderVerts) {
        cudaFree(m_d_renderVerts);
    }
    m_d_renderVerts = nullptr;

    if (m_stopEvent) { cudaEventDestroy((cudaEvent_t)m_stopEvent); m_stopEvent = nullptr; }
    if (m_startEvent) { cudaEventDestroy((cudaEvent_t)m_startEvent); m_startEvent = nullptr; }
    if (m_stream) { cudaStreamDestroy((cudaStream_t)m_stream); m_stream = nullptr; }

    m_gpuMaxVerts = 0;
    m_gpuMaxTets = 0;
    m_gpuMaxTris = 0;
}

void CDT3DCompute::reset(const CDT3DParams& params) {
    m_T = params.T;

    m_complex.initIcosahedronSandwich(params.T, params.initialVerticesPerSlice);

    m_measureCount = 0;
    m_avgN2.assign(m_T, 0.0);
    m_smoothN2.assign(m_T, 0.0f);

    m_attempted = m_accepted = 0;
    m_attempted26 = m_accepted26 = 0;
    m_attempted44 = m_accepted44 = 0;
    m_attempted23 = m_accepted23 = 0;

    m_state = CDT3DStateData{};
    m_state.volumeProfile.resize(m_T, 0);
    m_state.avgVolumeProfile.resize(m_T, 0.0f);
    m_kernelTimeMs = 0.0f;

    // Initialize icosahedron vertex positions on unit sphere.
    // Vertex numbering from initIcosahedronSandwich: 12 per slice, ordered as
    //   0=top, 1-5=upper ring, 6-10=lower ring, 11=bottom
    // Each slice t has vertices at indices [t*12 .. t*12+11].
    static const float icoPos[12][3] = {
        { 0.0f,  1.0f,  0.0f},  // 0: top
        { 0.8944f,  0.4472f,  0.0f},       // 1
        { 0.2764f,  0.4472f,  0.8507f},    // 2
        {-0.7236f,  0.4472f,  0.5257f},    // 3
        {-0.7236f,  0.4472f, -0.5257f},    // 4
        { 0.2764f,  0.4472f, -0.8507f},    // 5
        { 0.7236f, -0.4472f,  0.5257f},    // 6
        {-0.2764f, -0.4472f,  0.8507f},    // 7
        {-0.8944f, -0.4472f,  0.0f},       // 8
        {-0.2764f, -0.4472f, -0.8507f},    // 9
        { 0.7236f, -0.4472f, -0.5257f},    // 10
        { 0.0f, -1.0f,  0.0f},  // 11: bottom
    };

    int maxV = m_complex.maxVertices();
    m_embedX.assign(maxV, 0.0f);
    m_embedY.assign(maxV, 0.0f);
    m_embedZ.assign(maxV, 0.0f);
    m_embedInitialized.assign(maxV, false);
    m_embedAnchor.assign(maxV, false);

    for (int t = 0; t < m_T; t++) {
        for (int i = 0; i < 12; i++) {
            int v = t * 12 + i;
            if (v < maxV && m_complex.vertex(v).alive) {
                m_embedX[v] = icoPos[i][0];
                m_embedY[v] = icoPos[i][1];
                m_embedZ[v] = icoPos[i][2];
                m_embedInitialized[v] = true;
                m_embedAnchor[v] = true;  // Original icosahedron vertex — never move
            }
        }
    }

    computeObservablesCPU(params);
    buildRenderDataCPU(params);

    fprintf(stderr, "[CDT3D] Init: tets=%d verts=%d spatialTris=%d\n",
            m_complex.tetCount(), m_complex.vertexCount(), m_state.triangleCount);
}

// ── Interop ─────────────────────────────────────────────────────────────

void CDT3DCompute::setRenderBuffer(void* cudaPtr, size_t size) {
    m_d_renderVerts = cudaPtr;
    m_renderBufferSize = size;
    m_hasRenderBuffer = (cudaPtr != nullptr);
}

// ── Step ────────────────────────────────────────────────────────────────

void CDT3DCompute::step(const CDT3DParams& params) {
    if (params.T != m_T) {
        reset(params);
        return;
    }

    cudaStream_t stream = (cudaStream_t)m_stream;
    CUDA_CHECK(cudaEventRecord((cudaEvent_t)m_startEvent, stream));

    // 1. CPU: Pachner moves
    m_attempted = m_accepted = 0;
    m_attempted26 = m_accepted26 = 0;
    m_attempted44 = m_accepted44 = 0;
    m_attempted23 = m_accepted23 = 0;

    for (int s = 0; s < params.sweepsPerFrame; s++) {
        metropolisSweep(params);
    }

    // 2. Update running averages
    m_measureCount++;

    // 3. Compute observables and render data
    computeObservablesCPU(params);
    buildRenderDataCPU(params);

    CUDA_CHECK(cudaEventRecord((cudaEvent_t)m_stopEvent, stream));
    CUDA_CHECK(cudaEventSynchronize((cudaEvent_t)m_stopEvent));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, (cudaEvent_t)m_startEvent, (cudaEvent_t)m_stopEvent));
    m_kernelTimeMs = ms;
}

// ── Regge action change ────────────────────────────────────────────────

float CDT3DCompute::actionChange(int dN0, int dN3, const CDT3DParams& p) const {
    int N3 = m_complex.tetCount();
    float dS_action = -p.k0 * static_cast<float>(dN0) + p.k3 * static_cast<float>(dN3);
    float dS_volume = p.epsilon * static_cast<float>(
        (N3 + dN3 - p.N3_target) * (N3 + dN3 - p.N3_target) -
        (N3 - p.N3_target) * (N3 - p.N3_target));
    return dS_action + dS_volume;
}

// ── Metropolis Sweep ───────────────────────────────────────────────────
//
// Adjacency is rebuilt periodically (every N accepted moves or at the
// end of the sweep).  The (2,6)/(6,2) moves use direct neighbor pointers
// with fallback to tetsAroundEdge, while (2,3)/(3,2) use adjacency
// walking.  Rebuilding after every accepted move is too expensive during
// warmup, so we batch the rebuilds.

void CDT3DCompute::metropolisSweep(const CDT3DParams& params) {
    int numAttempts = std::max(50, m_complex.tetCount() / 10);
    int acceptedThisSweep = 0;
    static constexpr int REBUILD_INTERVAL = 50;  // Rebuild adjacency every N accepted moves

    for (int i = 0; i < numAttempts; i++) {
        float r = s_rng3d.nextFloat();
        bool accepted = false;

        if (r < params.prob26) {
            m_attempted26++;
            if (s_rng3d.nextInt(2) == 0) {
                accepted = tryMove26(params);
            } else {
                accepted = tryMove62(params);
            }
            if (accepted) m_accepted26++;
        } else if (r < params.prob26 + params.prob44) {
            m_attempted44++;
            accepted = tryMove44(params);
            if (accepted) m_accepted44++;
        } else {
            m_attempted23++;
            if (s_rng3d.nextInt(2) == 0) {
                accepted = tryMove23(params);
            } else {
                accepted = tryMove32(params);
            }
            if (accepted) m_accepted23++;
        }

        m_attempted++;
        if (accepted) {
            m_accepted++;
            acceptedThisSweep++;
            if (acceptedThisSweep % REBUILD_INTERVAL == 0) {
                m_complex.buildAdjacency();
            }
        }
    }

    // Always rebuild at the end of each sweep to ensure clean state
    if (acceptedThisSweep > 0) {
        m_complex.buildAdjacency();
    }
}

// ── Helper: check that a tet properly spans two adjacent time slices ───
static bool isValidCDTTet(const SimplicialComplex3D& cx, int v0, int v1, int v2, int v3, int T) {
    int s[4] = {
        cx.vertex(v0).timeSlice,
        cx.vertex(v1).timeSlice,
        cx.vertex(v2).timeSlice,
        cx.vertex(v3).timeSlice
    };
    // Must have exactly 2 distinct slices
    int minS = *std::min_element(s, s + 4);
    int maxS = *std::max_element(s, s + 4);
    if (minS == maxS) return false; // All same slice — degenerate!
    // Must be adjacent slices (with periodic wrapping)
    if (maxS - minS != 1 && !(minS == 0 && maxS == T - 1)) return false;
    return true;
}

// ── Pachner Move (2,6): Insert vertex into a spatial triangle ──────────
//
// A spatial triangle has all 3 vertices in the same time slice.
// It is shared by exactly 2 tets: one (3,1) above and one (1,3) below.
// We insert a new vertex in the triangle, splitting 2 tets into 6.
// ΔN₀=+1, ΔN₃=+4

bool CDT3DCompute::tryMove26(const CDT3DParams& p) {
    int nTets = m_complex.maxTets();
    if (nTets == 0) return false;

    // Find a tet with a spatial face
    int tetIdx = -1;
    int spatialFace = -1;
    for (int attempt = 0; attempt < 30; attempt++) {
        int t = s_rng3d.nextInt(nTets);
        if (!m_complex.tet(t).alive) continue;

        const auto& tet = m_complex.tet(t);
        for (int f = 0; f < 4; f++) {
            int fv[3]; int idx = 0;
            for (int i = 0; i < 4; i++) {
                if (i != f) fv[idx++] = tet.vertices[i];
            }
            int s0 = m_complex.vertex(fv[0]).timeSlice;
            int s1 = m_complex.vertex(fv[1]).timeSlice;
            int s2 = m_complex.vertex(fv[2]).timeSlice;
            int sOpp = m_complex.vertex(tet.vertices[f]).timeSlice;
            if (s0 == s1 && s1 == s2 && sOpp != s0) {
                tetIdx = t;
                spatialFace = f;
                break;
            }
        }
        if (tetIdx >= 0) break;
    }
    if (tetIdx < 0) return false;

    // Get face vertices (a,b,c) all in the same slice
    const auto& tetA = m_complex.tet(tetIdx);
    int fv[3]; int idx = 0;
    for (int i = 0; i < 4; i++) {
        if (i != spatialFace) fv[idx++] = tetA.vertices[i];
    }
    int topVert = tetA.vertices[spatialFace]; // Vertex in the adjacent slice
    int sliceFace = m_complex.vertex(fv[0]).timeSlice;

    // Find the neighbor tet sharing this spatial face.
    // First try the adjacency pointer (fast path).
    int nbTet = m_complex.tet(tetIdx).neighbors[spatialFace];

    // Validate: the neighbor must actually share all 3 face vertices
    if (nbTet >= 0 && m_complex.tet(nbTet).alive) {
        const auto& nb = m_complex.tet(nbTet);
        int shared = 0;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 3; j++) {
                if (nb.vertices[i] == fv[j]) { shared++; break; }
            }
        }
        if (shared != 3) nbTet = -1; // Stale pointer — invalidate
    } else {
        nbTet = -1;
    }

    // If the fast path failed, do a brute-force search via tetsAroundEdge
    if (nbTet < 0) {
        auto tetsEdge = m_complex.tetsAroundEdge(fv[0], fv[1]);
        for (int ti : tetsEdge) {
            if (ti == tetIdx) continue;
            // Check if this tet also contains fv[2]
            bool hasFv2 = false;
            for (int k = 0; k < 4; k++) {
                if (m_complex.tet(ti).vertices[k] == fv[2]) { hasFv2 = true; break; }
            }
            if (hasFv2) { nbTet = ti; break; }
        }
    }
    if (nbTet < 0) return false;

    // Find the opposite vertex in the neighbor tet (in the other adjacent slice)
    int botVert = -1;
    const auto& tetB = m_complex.tet(nbTet);
    for (int i = 0; i < 4; i++) {
        int v = tetB.vertices[i];
        if (v != fv[0] && v != fv[1] && v != fv[2]) {
            botVert = v;
            break;
        }
    }
    if (botVert < 0) return false;

    // Verify botVert is in a different slice than face vertices
    if (m_complex.vertex(botVert).timeSlice == sliceFace) return false;

    // Metropolis: ΔN₀=+1, ΔN₃=+4
    float dS = actionChange(1, 4, p);
    if (dS > 0.0f && s_rng3d.nextFloat() >= std::exp(-dS)) return false;

    // Create new vertex in the spatial slice
    int vNew = m_complex.allocVertex(sliceFace);

    TetType typeA = tetA.type;
    TetType typeB = tetB.type;

    m_complex.freeTet(tetIdx);
    m_complex.freeTet(nbTet);

    // 3 tets on the "top" side (with topVert)
    m_complex.allocTet(typeA, fv[0], fv[1], vNew, topVert);
    m_complex.allocTet(typeA, fv[1], fv[2], vNew, topVert);
    m_complex.allocTet(typeA, fv[2], fv[0], vNew, topVert);

    // 3 tets on the "bottom" side (with botVert)
    m_complex.allocTet(typeB, fv[0], fv[1], vNew, botVert);
    m_complex.allocTet(typeB, fv[1], fv[2], vNew, botVert);
    m_complex.allocTet(typeB, fv[2], fv[0], vNew, botVert);

    return true;
}

// ── Pachner Move (6,2): Remove vertex with exactly 6 incident tets ────
//
// Reverse of (2,6). The vertex must have exactly 3 spatial neighbors
// forming a single triangle, with one top and one bottom apex vertex.
// ΔN₀=-1, ΔN₃=-4

bool CDT3DCompute::tryMove62(const CDT3DParams& p) {
    int nVerts = m_complex.maxVertices();
    if (nVerts == 0) return false;

    int vIdx = -1;
    for (int attempt = 0; attempt < 20; attempt++) {
        int v = s_rng3d.nextInt(nVerts);
        if (m_complex.vertex(v).alive) { vIdx = v; break; }
    }
    if (vIdx < 0) return false;

    // Never remove anchor vertices (original icosahedron) — they anchor the embedding
    if (vIdx < static_cast<int>(m_embedAnchor.size()) && m_embedAnchor[vIdx]) return false;

    auto incidentTets = m_complex.tetsAroundVertex(vIdx);
    if (incidentTets.size() != 6) return false; // Must be exactly 6

    int slice = m_complex.vertex(vIdx).timeSlice;

    // Classify tets into above/below and find top/bot apex vertices
    int topVert = -1, botVert = -1;
    int aboveCount = 0, belowCount = 0;

    for (int ti : incidentTets) {
        int otherSliceVert = -1;
        for (int k = 0; k < 4; k++) {
            int v = m_complex.tet(ti).vertices[k];
            if (m_complex.vertex(v).timeSlice != slice) {
                otherSliceVert = v;
                break;
            }
        }
        if (otherSliceVert < 0) return false;

        int otherSlice = m_complex.vertex(otherSliceVert).timeSlice;
        int aboveSlice = (slice + 1) % m_T;
        int belowSlice = (slice - 1 + m_T) % m_T;

        if (otherSlice == aboveSlice) {
            aboveCount++;
            if (topVert < 0) topVert = otherSliceVert;
            else if (topVert != otherSliceVert) return false;
        } else if (otherSlice == belowSlice) {
            belowCount++;
            if (botVert < 0) botVert = otherSliceVert;
            else if (botVert != otherSliceVert) return false;
        } else {
            return false;
        }
    }

    if (aboveCount != 3 || belowCount != 3) return false;
    if (topVert < 0 || botVert < 0) return false;

    // Collect the 3 spatial neighbors (same slice, not vIdx)
    std::set<int> neighborSet;
    for (int ti : incidentTets) {
        for (int k = 0; k < 4; k++) {
            int v = m_complex.tet(ti).vertices[k];
            if (v != vIdx && m_complex.vertex(v).timeSlice == slice) {
                neighborSet.insert(v);
            }
        }
    }
    if (neighborSet.size() != 3) return false;

    std::vector<int> ring(neighborSet.begin(), neighborSet.end());

    // Metropolis: ΔN₀=-1, ΔN₃=-4
    float dS = actionChange(-1, -4, p);
    if (dS > 0.0f && s_rng3d.nextFloat() >= std::exp(-dS)) return false;

    // Determine tet types from existing tets
    TetType typeAbove = TetType::Tet31, typeBelow = TetType::Tet13;
    for (int ti : incidentTets) {
        for (int k = 0; k < 4; k++) {
            int v = m_complex.tet(ti).vertices[k];
            if (v == topVert) { typeAbove = m_complex.tet(ti).type; break; }
            if (v == botVert) { typeBelow = m_complex.tet(ti).type; break; }
        }
    }

    // Free all 6 tets
    for (int ti : incidentTets) {
        m_complex.freeTet(ti);
    }

    // Create 2 new tets
    m_complex.allocTet(typeAbove, ring[0], ring[1], ring[2], topVert);
    m_complex.allocTet(typeBelow, ring[0], ring[1], ring[2], botVert);

    m_complex.freeVertex(vIdx);
    return true;
}

// ── Pachner Move (4,4): Flip a timelike edge ──────────────────────────
//
// Finds 4 tets sharing a timelike edge (v0,v1) where v0 is in slice t
// and v1 is in slice t+1.  The 4 "other" vertices form a cycle around
// the edge.  We flip the edge to the diagonal of this cycle.
// ΔN₀=0, ΔN₃=0.

bool CDT3DCompute::tryMove44(const CDT3DParams& p) {
    int nTets = m_complex.maxTets();
    if (nTets == 0) return false;

    int tetIdx = -1;
    for (int attempt = 0; attempt < 20; attempt++) {
        int t = s_rng3d.nextInt(nTets);
        if (m_complex.tet(t).alive) { tetIdx = t; break; }
    }
    if (tetIdx < 0) return false;

    // Find a timelike edge in this tet
    const auto& tet = m_complex.tet(tetIdx);
    int v0 = -1, v1 = -1;
    static const int edgePairs[6][2] = {{0,1},{0,2},{0,3},{1,2},{1,3},{2,3}};

    // Try all 6 edges in random order
    int startEdge = s_rng3d.nextInt(6);
    for (int off = 0; off < 6; off++) {
        int ei = (startEdge + off) % 6;
        int va = tet.vertices[edgePairs[ei][0]];
        int vb = tet.vertices[edgePairs[ei][1]];
        if (m_complex.vertex(va).timeSlice != m_complex.vertex(vb).timeSlice) {
            v0 = va; v1 = vb;
            break;
        }
    }
    if (v0 < 0) return false;

    // Need exactly 4 tets around this timelike edge
    auto tetsEdge = m_complex.tetsAroundEdge(v0, v1);
    if (tetsEdge.size() != 4) return false;

    // Collect the "other" vertices and classify by slice
    int s0 = m_complex.vertex(v0).timeSlice;
    int s1 = m_complex.vertex(v1).timeSlice;

    std::vector<int> sameAsV0, sameAsV1;
    for (int ti : tetsEdge) {
        for (int k = 0; k < 4; k++) {
            int v = m_complex.tet(ti).vertices[k];
            if (v == v0 || v == v1) continue;
            int sv = m_complex.vertex(v).timeSlice;
            if (sv == s0) {
                if (std::find(sameAsV0.begin(), sameAsV0.end(), v) == sameAsV0.end())
                    sameAsV0.push_back(v);
            } else if (sv == s1) {
                if (std::find(sameAsV1.begin(), sameAsV1.end(), v) == sameAsV1.end())
                    sameAsV1.push_back(v);
            }
        }
    }

    // Standard (4,4): 2 vertices in each slice.  Pick the diagonal.
    // We need: sameAsV0 = {c0, c1} and sameAsV1 = {d0, d1}
    // New edge: (c0, d0) or (c1, d1) — whichever is the "opposite diagonal"
    if (sameAsV0.size() != 2 || sameAsV1.size() != 2) return false;

    // The new edge should connect one vertex from each slice
    // that are NOT already connected through the current 4 tets
    // Try both possible new timelike diagonals
    int candidates[2][2] = {
        {sameAsV0[0], sameAsV1[1]},
        {sameAsV0[1], sameAsV1[0]}
    };

    int newA = -1, newB = -1;
    for (int ci = 0; ci < 2; ci++) {
        int ca = candidates[ci][0], cb = candidates[ci][1];
        // Check that (ca, cb) do NOT share any of the 4 tets
        bool sharesTet = false;
        for (int ti : tetsEdge) {
            bool hasA = false, hasB = false;
            for (int k = 0; k < 4; k++) {
                if (m_complex.tet(ti).vertices[k] == ca) hasA = true;
                if (m_complex.tet(ti).vertices[k] == cb) hasB = true;
            }
            if (hasA && hasB) { sharesTet = true; break; }
        }
        if (!sharesTet) {
            newA = ca; newB = cb;
            break;
        }
    }
    if (newA < 0) return false;

    // The remaining 2 "other" vertices
    int remC = -1, remD = -1;
    for (int v : sameAsV0) { if (v != newA && v != newB) { if (remC < 0) remC = v; else remD = v; } }
    for (int v : sameAsV1) { if (v != newA && v != newB) { if (remC < 0) remC = v; else remD = v; } }
    if (remC < 0 || remD < 0) return false;

    // Validate new tets would be proper CDT
    if (!isValidCDTTet(m_complex, v0, newA, newB, remC, m_T)) return false;
    if (!isValidCDTTet(m_complex, v0, newA, newB, remD, m_T)) return false;
    if (!isValidCDTTet(m_complex, v1, newA, newB, remC, m_T)) return false;
    if (!isValidCDTTet(m_complex, v1, newA, newB, remD, m_T)) return false;

    // Metropolis: ΔN₃=0
    float dS = actionChange(0, 0, p);
    if (dS > 0.0f && s_rng3d.nextFloat() >= std::exp(-dS)) return false;

    TetType type = m_complex.tet(tetsEdge[0]).type;
    for (int ti : tetsEdge) {
        m_complex.freeTet(ti);
    }

    m_complex.allocTet(type, v0, newA, newB, remC);
    m_complex.allocTet(type, v0, newA, newB, remD);
    m_complex.allocTet(type, v1, newA, newB, remC);
    m_complex.allocTet(type, v1, newA, newB, remD);

    return true;
}

// ── Pachner Move (2,3): Two tets sharing a face → three tets sharing an edge
//
// Pick 2 tetrahedra sharing a triangular face (a,b,c), with opposite
// vertices d and e.  Replace them with 3 tets sharing the new edge (d,e):
//   (a,b,d,e), (b,c,d,e), (c,a,d,e)
// Precondition: edge (d,e) must not already exist in the triangulation.
// ΔN₀=0, ΔN₃=+1

bool CDT3DCompute::tryMove23(const CDT3DParams& p) {
    int nTets = m_complex.maxTets();
    if (nTets == 0) return false;

    // Pick a random alive tet
    int tetIdx = -1;
    for (int attempt = 0; attempt < 20; attempt++) {
        int t = s_rng3d.nextInt(nTets);
        if (m_complex.tet(t).alive) { tetIdx = t; break; }
    }
    if (tetIdx < 0) return false;

    // Pick a random face of this tet
    int face = s_rng3d.nextInt(4);

    // The neighbor across this face
    int nbTet = m_complex.tet(tetIdx).neighbors[face];
    if (nbTet < 0 || !m_complex.tet(nbTet).alive) return false;

    // Get the face vertices (a,b,c) and opposite vertices (d,e)
    const auto& tetA = m_complex.tet(tetIdx);
    const auto& tetB = m_complex.tet(nbTet);

    int d = tetA.vertices[face]; // vertex opposite the shared face in tetA

    // Find e = vertex opposite the shared face in tetB
    int e = -1;
    int faceInB = -1;
    for (int i = 0; i < 4; i++) {
        int v = tetB.vertices[i];
        bool inFace = false;
        for (int j = 0; j < 4; j++) {
            if (j != face && tetA.vertices[j] == v) { inFace = true; break; }
        }
        if (!inFace) { e = v; faceInB = i; break; }
    }
    if (e < 0 || d == e) return false;

    // Face vertices (a,b,c): the 3 vertices of tetA excluding d
    int fv[3]; int idx = 0;
    for (int i = 0; i < 4; i++) {
        if (i != face) fv[idx++] = tetA.vertices[i];
    }
    int a = fv[0], b = fv[1], c = fv[2];

    // Check that edge (d,e) does not already exist
    // i.e., no alive tet contains both d and e
    auto tetsD = m_complex.tetsAroundVertex(d);
    for (int ti : tetsD) {
        const auto& t = m_complex.tet(ti);
        for (int k = 0; k < 4; k++) {
            if (t.vertices[k] == e) return false; // edge already exists
        }
    }

    // Validate new tets span exactly 2 adjacent slices
    if (!isValidCDTTet(m_complex, a, b, d, e, m_T)) return false;
    if (!isValidCDTTet(m_complex, b, c, d, e, m_T)) return false;
    if (!isValidCDTTet(m_complex, c, a, d, e, m_T)) return false;

    // Metropolis: ΔN₀=0, ΔN₃=+1
    float dS = actionChange(0, 1, p);
    if (dS > 0.0f && s_rng3d.nextFloat() >= std::exp(-dS)) return false;

    TetType type = tetA.type;
    m_complex.freeTet(tetIdx);
    m_complex.freeTet(nbTet);

    m_complex.allocTet(type, a, b, d, e);
    m_complex.allocTet(type, b, c, d, e);
    m_complex.allocTet(type, c, a, d, e);

    return true;
}

// ── Pachner Move (3,2): Three tets sharing an edge → two tets sharing a face
//
// Pick an edge (d,e) shared by exactly 3 tetrahedra.  The 3 "other" vertices
// (a,b,c) — one per tet — must not form an existing triangle (face).
// Replace 3 tets with 2: (a,b,c,d) and (a,b,c,e).
// ΔN₀=0, ΔN₃=-1

bool CDT3DCompute::tryMove32(const CDT3DParams& p) {
    int nTets = m_complex.maxTets();
    if (nTets == 0) return false;

    // Pick a random alive tet
    int tetIdx = -1;
    for (int attempt = 0; attempt < 20; attempt++) {
        int t = s_rng3d.nextInt(nTets);
        if (m_complex.tet(t).alive) { tetIdx = t; break; }
    }
    if (tetIdx < 0) return false;

    // Pick a random edge of this tet
    const auto& tet0 = m_complex.tet(tetIdx);
    static const int edgePairs[6][2] = {{0,1},{0,2},{0,3},{1,2},{1,3},{2,3}};
    int ei = s_rng3d.nextInt(6);
    int d = tet0.vertices[edgePairs[ei][0]];
    int e = tet0.vertices[edgePairs[ei][1]];

    // Find all tets around edge (d,e)
    auto tetsEdge = m_complex.tetsAroundEdge(d, e);
    if (tetsEdge.size() != 3) return false;

    // Collect the 3 "other" vertices (one per tet, not d or e)
    std::vector<int> others;
    for (int ti : tetsEdge) {
        for (int k = 0; k < 4; k++) {
            int v = m_complex.tet(ti).vertices[k];
            if (v != d && v != e) {
                others.push_back(v);
            }
        }
    }
    // Each tet contributes exactly 2 "other" verts, but we want 3 DISTINCT ones
    std::sort(others.begin(), others.end());
    others.erase(std::unique(others.begin(), others.end()), others.end());
    if (others.size() != 3) return false;

    int a = others[0], b = others[1], c = others[2];

    // Check that the face (a,b,c) does not already exist
    // i.e., no alive tet contains all 3 of (a,b,c)
    auto tetsA = m_complex.tetsAroundVertex(a);
    for (int ti : tetsA) {
        const auto& t = m_complex.tet(ti);
        bool hasB = false, hasC = false;
        for (int k = 0; k < 4; k++) {
            if (t.vertices[k] == b) hasB = true;
            if (t.vertices[k] == c) hasC = true;
        }
        if (hasB && hasC) return false; // face already exists
    }

    // Validate new tets
    if (!isValidCDTTet(m_complex, a, b, c, d, m_T)) return false;
    if (!isValidCDTTet(m_complex, a, b, c, e, m_T)) return false;

    // Metropolis: ΔN₀=0, ΔN₃=-1
    float dS = actionChange(0, -1, p);
    if (dS > 0.0f && s_rng3d.nextFloat() >= std::exp(-dS)) return false;

    TetType type = m_complex.tet(tetsEdge[0]).type;
    for (int ti : tetsEdge) {
        m_complex.freeTet(ti);
    }

    m_complex.allocTet(type, a, b, c, d);
    m_complex.allocTet(type, a, b, c, e);

    return true;
}

// ── CPU Observables ────────────────────────────────────────────────────

void CDT3DCompute::computeObservablesCPU(const CDT3DParams& params) {
    m_state.totalSweeps = m_measureCount * params.sweepsPerFrame;
    m_state.totalTetrahedra = m_complex.tetCount();
    m_state.totalVertices = m_complex.vertexCount();

    // Acceptance rates
    if (m_attempted > 0)
        m_state.acceptanceRate = static_cast<float>(m_accepted) / static_cast<float>(m_attempted);
    if (m_attempted26 > 0)
        m_state.acceptanceRate26 = static_cast<float>(m_accepted26) / static_cast<float>(m_attempted26);
    if (m_attempted44 > 0)
        m_state.acceptanceRate44 = static_cast<float>(m_accepted44) / static_cast<float>(m_attempted44);
    if (m_attempted23 > 0)
        m_state.acceptanceRate23 = static_cast<float>(m_accepted23) / static_cast<float>(m_attempted23);

    // Volume profile: count spatial triangles per slice
    // Use a single pass over all tets instead of T separate passes
    m_state.volumeProfile.assign(m_T, 0);
    int maxTets = m_complex.maxTets();
    for (int ti = 0; ti < maxTets; ti++) {
        if (!m_complex.tet(ti).alive) continue;
        const auto& tet = m_complex.tet(ti);

        for (int f = 0; f < 4; f++) {
            int fv[3]; int idx = 0;
            for (int i = 0; i < 4; i++) {
                if (i != f) fv[idx++] = tet.vertices[i];
            }

            int s0 = m_complex.vertex(fv[0]).timeSlice;
            int s1 = m_complex.vertex(fv[1]).timeSlice;
            int s2 = m_complex.vertex(fv[2]).timeSlice;

            // Spatial face: all 3 vertices in same slice, opposite vertex in different slice
            if (s0 == s1 && s1 == s2 && s0 >= 0 && s0 < m_T) {
                int sOpp = m_complex.vertex(tet.vertices[f]).timeSlice;
                if (sOpp != s0) {
                    m_state.volumeProfile[s0]++;
                }
            }
        }
    }

    // Update running averages
    for (int t = 0; t < m_T; t++) {
        m_avgN2[t] += static_cast<double>(m_state.volumeProfile[t]);
    }

    if (m_measureCount > 10) {
        for (int t = 0; t < m_T; t++) {
            m_state.avgVolumeProfile[t] = static_cast<float>(m_avgN2[t] / m_measureCount);
        }
    }

    // ── Expensive observables: only compute every 10 frames ───────────
    if (m_measureCount % 10 != 1 && m_measureCount > 1) return;

    // ── Hausdorff dimension (geodesic volume on dual graph) ───────────
    {
        int maxR = std::min(m_T, 15);
        if (maxR < 4) maxR = 4;
        int numSamples = 10;

        std::vector<double> logR, logV;

        for (int r = 1; r <= maxR; r++) {
            double totalVol = 0.0;

            for (int sample = 0; sample < numSamples; sample++) {
                int startTet = -1;
                for (int att = 0; att < 20; att++) {
                    int t = s_rng3d.nextInt(maxTets);
                    if (m_complex.tet(t).alive) { startTet = t; break; }
                }
                if (startTet < 0) continue;

                // BFS to distance r
                std::unordered_set<int> visited;
                std::vector<int> current = {startTet};
                visited.insert(startTet);

                for (int d = 0; d < r && !current.empty(); d++) {
                    std::vector<int> next;
                    for (int t : current) {
                        for (int f = 0; f < 4; f++) {
                            int nb = m_complex.tet(t).neighbors[f];
                            if (nb >= 0 && m_complex.tet(nb).alive &&
                                visited.find(nb) == visited.end()) {
                                visited.insert(nb);
                                next.push_back(nb);
                            }
                        }
                    }
                    current = std::move(next);
                }
                totalVol += static_cast<double>(visited.size());
            }
            totalVol /= numSamples;

            if (totalVol > 0) {
                logR.push_back(std::log(static_cast<double>(r)));
                logV.push_back(std::log(totalVol));
            }
        }

        if (logR.size() >= 3) {
            double sumX = 0, sumY = 0, sumXY = 0, sumXX = 0;
            int nn = static_cast<int>(logR.size());
            for (int i = 0; i < nn; i++) {
                sumX += logR[i]; sumY += logV[i];
                sumXY += logR[i] * logV[i]; sumXX += logR[i] * logR[i];
            }
            double denom = nn * sumXX - sumX * sumX;
            if (std::abs(denom) > 1e-12) {
                m_state.hausdorffDimension = static_cast<float>(
                    (nn * sumXY - sumX * sumY) / denom);
            }
        }
    }

    // ── Spectral dimension (random walk on dual graph) ─────────────────
    {
        int numWalks = 50;
        int maxSigma = std::min(100, std::max(20, m_complex.tetCount() / 4));

        std::vector<double> returnProb(maxSigma + 1, 0.0);

        for (int w = 0; w < numWalks; w++) {
            int startTet = -1;
            for (int att = 0; att < 20; att++) {
                int t = s_rng3d.nextInt(maxTets);
                if (m_complex.tet(t).alive) { startTet = t; break; }
            }
            if (startTet < 0) continue;

            int current = startTet;
            for (int sigma = 1; sigma <= maxSigma; sigma++) {
                int dir = s_rng3d.nextInt(4);
                int nb = m_complex.tet(current).neighbors[dir];
                if (nb >= 0 && m_complex.tet(nb).alive) {
                    current = nb;
                }
                if (current == startTet) {
                    returnProb[sigma] += 1.0;
                }
            }
        }

        for (int sigma = 1; sigma <= maxSigma; sigma++) {
            returnProb[sigma] /= numWalks;
        }

        std::vector<double> logSig, logP;
        int fitStart = std::max(5, maxSigma / 10);
        for (int sigma = fitStart; sigma <= maxSigma; sigma++) {
            if (returnProb[sigma] > 1e-8) {
                logSig.push_back(std::log(static_cast<double>(sigma)));
                logP.push_back(std::log(returnProb[sigma]));
            }
        }

        if (logSig.size() >= 5) {
            double sumX = 0, sumY = 0, sumXY = 0, sumXX = 0;
            int nn = static_cast<int>(logSig.size());
            for (int i = 0; i < nn; i++) {
                sumX += logSig[i]; sumY += logP[i];
                sumXY += logSig[i] * logP[i]; sumXX += logSig[i] * logSig[i];
            }
            double denom = nn * sumXX - sumX * sumX;
            if (std::abs(denom) > 1e-12) {
                double slope = (nn * sumXY - sumX * sumY) / denom;
                m_state.spectralDimension = static_cast<float>(-2.0 * slope);
            }
        }
    }
}

// ── CPU Render Data ────────────────────────────────────────────────────

void CDT3DCompute::buildRenderDataCPU(const CDT3DParams& params) {
    int maxTets = m_complex.maxTets();
    int maxV = m_complex.maxVertices();

    // ── 1. Collect spatial triangles (one pass over all tets) ───────────
    struct TriData { int v0, v1, v2, slice; };
    std::vector<TriData> allTris;

    struct FaceKey {
        int v[3];
        bool operator<(const FaceKey& o) const {
            if (v[0] != o.v[0]) return v[0] < o.v[0];
            if (v[1] != o.v[1]) return v[1] < o.v[1];
            return v[2] < o.v[2];
        }
    };
    std::set<FaceKey> seenFaces;

    for (int ti = 0; ti < maxTets; ti++) {
        if (!m_complex.tet(ti).alive) continue;
        const auto& tet = m_complex.tet(ti);

        for (int f = 0; f < 4; f++) {
            int fv[3]; int idx = 0;
            for (int i = 0; i < 4; i++) {
                if (i != f) fv[idx++] = tet.vertices[i];
            }

            int s0 = m_complex.vertex(fv[0]).timeSlice;
            int s1 = m_complex.vertex(fv[1]).timeSlice;
            int s2 = m_complex.vertex(fv[2]).timeSlice;

            if (s0 == s1 && s1 == s2 && s0 >= 0 && s0 < m_T) {
                int sOpp = m_complex.vertex(tet.vertices[f]).timeSlice;
                if (sOpp != s0) {
                    FaceKey key;
                    key.v[0] = fv[0]; key.v[1] = fv[1]; key.v[2] = fv[2];
                    if (key.v[0] > key.v[1]) std::swap(key.v[0], key.v[1]);
                    if (key.v[1] > key.v[2]) std::swap(key.v[1], key.v[2]);
                    if (key.v[0] > key.v[1]) std::swap(key.v[0], key.v[1]);

                    if (seenFaces.insert(key).second) {
                        allTris.push_back({fv[0], fv[1], fv[2], s0});
                    }
                }
            }
        }
    }

    int numTris = static_cast<int>(allTris.size());
    m_state.triangleCount = numTris;



    // ── 2. Build per-vertex neighbor list from spatial triangulation ─────
    // Each vertex's neighbors = other vertices it shares a spatial edge with
    std::vector<std::vector<int>> neighbors(maxV);

    for (const auto& tri : allTris) {
        neighbors[tri.v0].push_back(tri.v1);
        neighbors[tri.v0].push_back(tri.v2);
        neighbors[tri.v1].push_back(tri.v0);
        neighbors[tri.v1].push_back(tri.v2);
        neighbors[tri.v2].push_back(tri.v0);
        neighbors[tri.v2].push_back(tri.v1);
    }
    // Dedup neighbor lists
    for (int v = 0; v < maxV; v++) {
        auto& nb = neighbors[v];
        std::sort(nb.begin(), nb.end());
        nb.erase(std::unique(nb.begin(), nb.end()), nb.end());
    }

    // ── 3. Laplacian sphere embedding (persistent positions) ─────────
    // Grow embedding arrays if needed
    if (static_cast<int>(m_embedX.size()) < maxV) {
        m_embedX.resize(maxV, 0.0f);
        m_embedY.resize(maxV, 0.0f);
        m_embedZ.resize(maxV, 0.0f);
        m_embedInitialized.resize(maxV, false);
    }

    // Initialize any new (uninitialized) vertices from neighbor centroid
    for (int v = 0; v < maxV; v++) {
        if (!m_complex.vertex(v).alive) {
            m_embedInitialized[v] = false;
            continue;
        }
        if (m_embedInitialized[v]) continue;

        // Place at centroid of initialized neighbors, projected to sphere
        float sx = 0, sy = 0, sz = 0;
        int count = 0;
        for (int nb : neighbors[v]) {
            if (m_embedInitialized[nb]) {
                sx += m_embedX[nb];
                sy += m_embedY[nb];
                sz += m_embedZ[nb];
                count++;
            }
        }

        if (count > 0) {
            float len = std::sqrt(sx*sx + sy*sy + sz*sz);
            if (len > 1e-6f) {
                m_embedX[v] = sx / len;
                m_embedY[v] = sy / len;
                m_embedZ[v] = sz / len;
            } else {
                // Neighbors cancel out — add tiny random jitter
                float phi = s_rng3d.nextFloat() * 6.28318f;
                float ct = s_rng3d.nextFloat() * 2.0f - 1.0f;
                float st = std::sqrt(1.0f - ct*ct);
                m_embedX[v] = st * std::cos(phi);
                m_embedY[v] = ct;
                m_embedZ[v] = st * std::sin(phi);
            }
        } else {
            // No initialized neighbors — random on sphere
            float phi = s_rng3d.nextFloat() * 6.28318f;
            float ct = s_rng3d.nextFloat() * 2.0f - 1.0f;
            float st = std::sqrt(1.0f - ct*ct);
            m_embedX[v] = st * std::cos(phi);
            m_embedY[v] = ct;
            m_embedZ[v] = st * std::sin(phi);
        }
        m_embedInitialized[v] = true;
    }

    // Anchored relaxation: the initial icosahedron vertices (indices
    // < T*12) are FIXED as anchors.  Only newer vertices (added by Pachner
    // moves) are relaxed toward their neighbor centroid.  This prevents
    // the embedding from collapsing while still allowing new vertices to
    // find a natural position on the sphere surface.
    //
    // We do multiple iterations with moderate alpha because the anchors
    // prevent collapse.

    // Grow anchor array if needed
    if (static_cast<int>(m_embedAnchor.size()) < maxV) {
        m_embedAnchor.resize(maxV, false);
    }
    // Mark removed anchors as non-anchor (vertex was freed and potentially reused)
    for (int v = 0; v < maxV; v++) {
        if (!m_complex.vertex(v).alive) m_embedAnchor[v] = false;
    }

    // Anchored Tutte relaxation: original icosahedron vertices are FIXED,
    // only new vertices (added by Pachner moves) move toward the centroid
    // of their neighbors.  Because the anchors are fixed and well-distributed
    // on the sphere, the embedding cannot collapse.
    float alpha = 0.5f;
    int numIters = 10;

    std::vector<float> newX(maxV), newY(maxV), newZ(maxV);

    for (int iter = 0; iter < numIters; iter++) {
        for (int v = 0; v < maxV; v++) {
            newX[v] = m_embedX[v];
            newY[v] = m_embedY[v];
            newZ[v] = m_embedZ[v];

            // Skip dead, isolated, or anchor vertices
            if (!m_complex.vertex(v).alive || neighbors[v].empty()) continue;
            if (m_embedAnchor[v]) continue;  // Anchors never move

            // Centroid of neighbors on the sphere surface
            float cx = 0, cy = 0, cz = 0;
            for (int nb : neighbors[v]) {
                cx += m_embedX[nb];
                cy += m_embedY[nb];
                cz += m_embedZ[nb];
            }
            int deg = static_cast<int>(neighbors[v].size());
            cx /= deg; cy /= deg; cz /= deg;

            // Move toward centroid
            float bx = m_embedX[v] * (1.0f - alpha) + cx * alpha;
            float by = m_embedY[v] * (1.0f - alpha) + cy * alpha;
            float bz = m_embedZ[v] * (1.0f - alpha) + cz * alpha;

            // Project back to unit sphere
            float len = std::sqrt(bx*bx + by*by + bz*bz);
            if (len > 1e-8f) {
                newX[v] = bx / len;
                newY[v] = by / len;
                newZ[v] = bz / len;
            }
        }

        for (int v = 0; v < maxV; v++) {
            m_embedX[v] = newX[v];
            m_embedY[v] = newY[v];
            m_embedZ[v] = newZ[v];
        }
    }

    // ── 4. Build final world positions from embedding ──────────────────
    float spacing = params.sphereSpacing;
    float radius = params.meshScale;

    std::vector<float> posX(maxV), posY(maxV), posZ(maxV);
    for (int v = 0; v < maxV; v++) {
        if (!m_complex.vertex(v).alive) continue;
        int t = m_complex.vertex(v).timeSlice;
        float yOff = (static_cast<float>(t) - static_cast<float>(m_T - 1) * 0.5f) * spacing;

        // Sphere in XZ plane, Y = stacking axis
        posX[v] = radius * m_embedX[v];
        posY[v] = yOff + radius * m_embedY[v];
        posZ[v] = radius * m_embedZ[v];
    }

    // ── 5. Curvature from deficit angle ────────────────────────────────
    std::vector<float> curvature(maxV, 0.0f);
    std::vector<int> triCount(maxV, 0);

    // Simple curvature: 2*pi - n*(pi/3) where n=number of incident triangles
    // This is the topological curvature (not dependent on embedding quality)
    for (const auto& tri : allTris) {
        triCount[tri.v0]++;
        triCount[tri.v1]++;
        triCount[tri.v2]++;
    }
    for (int v = 0; v < maxV; v++) {
        if (m_complex.vertex(v).alive && triCount[v] > 0) {
            // For equilateral triangles on a sphere, angle ~ pi/3
            // Gaussian curvature ~ 2*pi - n*(pi/3)
            curvature[v] = 6.28318530718f - static_cast<float>(triCount[v]) * 1.0471975512f;
        }
    }
    float maxAbsCurv = 0.5f;
    for (int v = 0; v < maxV; v++) {
        float ac = std::abs(curvature[v]);
        if (ac > maxAbsCurv) maxAbsCurv = ac;
    }

    // ── 6. Build render output ─────────────────────────────────────────
    m_state.vertexData.resize(numTris * 3 * 9);
    float* out = m_state.vertexData.data();

    for (int i = 0; i < numTris; i++) {
        const auto& tri = allTris[i];

        float px[3] = {posX[tri.v0], posX[tri.v1], posX[tri.v2]};
        float py[3] = {posY[tri.v0], posY[tri.v1], posY[tri.v2]};
        float pz[3] = {posZ[tri.v0], posZ[tri.v1], posZ[tri.v2]};

        // Face normal
        float e1x = px[1] - px[0], e1y = py[1] - py[0], e1z = pz[1] - pz[0];
        float e2x = px[2] - px[0], e2y = py[2] - py[0], e2z = pz[2] - pz[0];
        float nx = e1y * e2z - e1z * e2y;
        float ny = e1z * e2x - e1x * e2z;
        float nz = e1x * e2y - e1y * e2x;
        float nLen = std::sqrt(nx*nx + ny*ny + nz*nz);
        if (nLen > 1e-6f) { nx /= nLen; ny /= nLen; nz /= nLen; }

        // Orient normal outward from sphere center
        float yOff = (static_cast<float>(tri.slice) - static_cast<float>(m_T - 1) * 0.5f) * spacing;
        float cx = (px[0] + px[1] + px[2]) / 3.0f;
        float cy = (py[0] + py[1] + py[2]) / 3.0f - yOff;
        float cz = (pz[0] + pz[1] + pz[2]) / 3.0f;
        float radialDot = nx*cx + ny*cy + nz*cz;
        if (radialDot < 0.0f) { nx = -nx; ny = -ny; nz = -nz; }

        int vts[3] = {tri.v0, tri.v1, tri.v2};
        for (int j = 0; j < 3; j++) {
            *out++ = px[j]; *out++ = py[j]; *out++ = pz[j];
            *out++ = nx;    *out++ = ny;    *out++ = nz;

            if (params.colorByCurvature) {
                float c = curvature[vts[j]] / maxAbsCurv;
                c = std::max(-1.0f, std::min(1.0f, c));
                if (c > 0.0f) {
                    *out++ = 1.0f; *out++ = 1.0f - c * 0.7f; *out++ = 1.0f - c * 0.7f;
                } else {
                    float s = -c;
                    *out++ = 1.0f - s * 0.7f; *out++ = 1.0f - s * 0.7f; *out++ = 1.0f;
                }
            } else {
                float frac = static_cast<float>(tri.slice) / static_cast<float>(std::max(1, m_T - 1));
                *out++ = 0.2f + 0.6f * frac;
                *out++ = 0.6f - 0.3f * frac;
                *out++ = 0.8f;
            }
        }
    }

}

// ── Stubs for future GPU use ───────────────────────────────────────────

void CDT3DCompute::uploadTopologyToGPU() {}
void CDT3DCompute::buildEdgeList(const std::vector<SimplicialComplex3D::Triangle>&) {}
void CDT3DCompute::buildCSR(const std::vector<SimplicialComplex3D::Triangle>&, int) {}

} // namespace cosmico
