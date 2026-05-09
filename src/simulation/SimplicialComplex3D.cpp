#include <cosmico/simulation/SimplicialComplex3D.h>
#include <algorithm>
#include <unordered_set>
#include <unordered_map>
#include <cassert>
#include <cstdio>
#include <cmath>

namespace cosmico {

// ── Free-list allocation ────────────────────────────────────────────────

int SimplicialComplex3D::allocVertex(int timeSlice) {
    int idx;
    if (!m_freeVertices.empty()) {
        idx = m_freeVertices.back();
        m_freeVertices.pop_back();
    } else {
        idx = static_cast<int>(m_vertices.size());
        m_vertices.push_back({});
    }
    m_vertices[idx].timeSlice = timeSlice;
    m_vertices[idx].tetIndex = -1;
    m_vertices[idx].alive = true;
    m_aliveVertices++;
    return idx;
}

void SimplicialComplex3D::freeVertex(int idx) {
    m_vertices[idx].alive = false;
    m_vertices[idx].timeSlice = -1;
    m_vertices[idx].tetIndex = -1;
    m_freeVertices.push_back(idx);
    m_aliveVertices--;
}

int SimplicialComplex3D::allocTet(TetType type, int v0, int v1, int v2, int v3) {
    int idx;
    if (!m_freeTets.empty()) {
        idx = m_freeTets.back();
        m_freeTets.pop_back();
    } else {
        idx = static_cast<int>(m_tets.size());
        m_tets.push_back({});
    }
    auto& t = m_tets[idx];
    t.vertices = {v0, v1, v2, v3};
    t.neighbors = {-1, -1, -1, -1};
    t.type = type;
    t.alive = true;
    m_aliveTets++;

    // Update vertex -> tet references
    if (v0 >= 0) m_vertices[v0].tetIndex = idx;
    if (v1 >= 0) m_vertices[v1].tetIndex = idx;
    if (v2 >= 0) m_vertices[v2].tetIndex = idx;
    if (v3 >= 0) m_vertices[v3].tetIndex = idx;

    return idx;
}

void SimplicialComplex3D::freeTet(int idx) {
    m_tets[idx].alive = false;
    m_tets[idx].vertices = {-1, -1, -1, -1};
    m_tets[idx].neighbors = {-1, -1, -1, -1};
    m_freeTets.push_back(idx);
    m_aliveTets--;
}

// ── Adjacency helpers ──────────────────────────────────────────────────

void SimplicialComplex3D::setNeighbor(int tetIdx, int face, int neighborTet) {
    m_tets[tetIdx].neighbors[face] = neighborTet;
}

int SimplicialComplex3D::findFace(int tetIdx, int oppositeVertex) const {
    const auto& t = m_tets[tetIdx];
    for (int i = 0; i < 4; i++) {
        if (t.vertices[i] == oppositeVertex) return i;
    }
    return -1;
}

int SimplicialComplex3D::sharedFaceIndex(int tetA, int tetB) const {
    // Find which face of tetA is adjacent to tetB
    const auto& a = m_tets[tetA];
    for (int i = 0; i < 4; i++) {
        if (a.neighbors[i] == tetB) return i;
    }
    return -1;
}

void SimplicialComplex3D::glueTetrahedra(int tetA, int faceA, int tetB, int faceB) {
    m_tets[tetA].neighbors[faceA] = tetB;
    m_tets[tetB].neighbors[faceB] = tetA;
}

// ── Topological queries ────────────────────────────────────────────────

std::vector<int> SimplicialComplex3D::tetsAroundVertex(int vertIdx) const {
    std::vector<int> result;
    if (!m_vertices[vertIdx].alive) return result;

    int startTet = m_vertices[vertIdx].tetIndex;
    if (startTet < 0) return result;

    // BFS/DFS traversal through adjacency
    std::unordered_set<int> visited;
    std::vector<int> stack;
    stack.push_back(startTet);

    while (!stack.empty()) {
        int t = stack.back();
        stack.pop_back();

        if (!visited.insert(t).second) continue;
        if (!m_tets[t].alive) continue;

        // Check if this tet contains the vertex
        bool hasVert = false;
        for (int i = 0; i < 4; i++) {
            if (m_tets[t].vertices[i] == vertIdx) {
                hasVert = true;
                break;
            }
        }
        if (!hasVert) continue;

        result.push_back(t);

        // Walk to neighbors that also contain this vertex
        for (int f = 0; f < 4; f++) {
            int nb = m_tets[t].neighbors[f];
            if (nb >= 0 && visited.find(nb) == visited.end()) {
                stack.push_back(nb);
            }
        }
    }
    return result;
}

std::vector<int> SimplicialComplex3D::tetsAroundEdge(int v0, int v1) const {
    std::vector<int> result;

    // Start from any tet incident to v0
    int startTet = m_vertices[v0].tetIndex;
    if (startTet < 0) return result;

    std::unordered_set<int> visited;
    std::vector<int> stack;
    stack.push_back(startTet);

    while (!stack.empty()) {
        int t = stack.back();
        stack.pop_back();

        if (!visited.insert(t).second) continue;
        if (!m_tets[t].alive) continue;

        // Check if this tet contains both vertices
        bool has0 = false, has1 = false;
        for (int i = 0; i < 4; i++) {
            if (m_tets[t].vertices[i] == v0) has0 = true;
            if (m_tets[t].vertices[i] == v1) has1 = true;
        }
        if (!has0 || !has1) {
            // If it contains v0 but not v1, still add neighbors to check
            if (has0) {
                for (int f = 0; f < 4; f++) {
                    int nb = m_tets[t].neighbors[f];
                    if (nb >= 0 && visited.find(nb) == visited.end()) {
                        stack.push_back(nb);
                    }
                }
            }
            continue;
        }

        result.push_back(t);

        for (int f = 0; f < 4; f++) {
            int nb = m_tets[t].neighbors[f];
            if (nb >= 0 && visited.find(nb) == visited.end()) {
                stack.push_back(nb);
            }
        }
    }
    return result;
}

std::vector<SimplicialComplex3D::Triangle> SimplicialComplex3D::spatialTriangulation(int timeSlice) const {
    std::vector<Triangle> result;

    // A spatial triangle is a face of a tetrahedron where all 3 vertices
    // are in the same time slice
    std::unordered_set<uint64_t> seen;

    for (int ti = 0; ti < static_cast<int>(m_tets.size()); ti++) {
        if (!m_tets[ti].alive) continue;

        const auto& tet = m_tets[ti];

        // Check each face (4 faces per tet, each face = 3 vertices)
        for (int f = 0; f < 4; f++) {
            // Face f = the 3 vertices EXCLUDING vertex[f]
            int fv[3];
            int idx = 0;
            for (int i = 0; i < 4; i++) {
                if (i != f) fv[idx++] = tet.vertices[i];
            }

            // Check if all 3 are in the target time slice
            if (m_vertices[fv[0]].timeSlice != timeSlice) continue;
            if (m_vertices[fv[1]].timeSlice != timeSlice) continue;
            if (m_vertices[fv[2]].timeSlice != timeSlice) continue;

            // Sort to create unique key
            int sorted[3] = {fv[0], fv[1], fv[2]};
            if (sorted[0] > sorted[1]) std::swap(sorted[0], sorted[1]);
            if (sorted[1] > sorted[2]) std::swap(sorted[1], sorted[2]);
            if (sorted[0] > sorted[1]) std::swap(sorted[0], sorted[1]);

            uint64_t key = (uint64_t)sorted[0] * 1000000ULL +
                           (uint64_t)sorted[1] * 1000ULL +
                           (uint64_t)sorted[2];

            if (seen.insert(key).second) {
                result.push_back({sorted[0], sorted[1], sorted[2]});
            }
        }
    }
    return result;
}

// ── Validation ──────────────────────────────────────────────────────────

bool SimplicialComplex3D::validateAdjacency() const {
    int errors = 0;
    for (int ti = 0; ti < static_cast<int>(m_tets.size()); ti++) {
        if (!m_tets[ti].alive) continue;

        const auto& tet = m_tets[ti];
        for (int f = 0; f < 4; f++) {
            int nb = tet.neighbors[f];
            if (nb < 0) continue;
            if (!m_tets[nb].alive) {
                fprintf(stderr, "[Validate] Tet %d face %d -> dead neighbor %d\n", ti, f, nb);
                errors++;
                continue;
            }

            // Check reciprocity: nb should have ti as a neighbor
            bool found = false;
            for (int nf = 0; nf < 4; nf++) {
                if (m_tets[nb].neighbors[nf] == ti) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                fprintf(stderr, "[Validate] Tet %d -> %d but %d doesn't point back\n",
                        ti, nb, nb);
                errors++;
            }
        }

        // Verify vertices are alive
        for (int i = 0; i < 4; i++) {
            int v = tet.vertices[i];
            if (v < 0 || !m_vertices[v].alive) {
                fprintf(stderr, "[Validate] Tet %d has dead/invalid vertex[%d]=%d\n", ti, i, v);
                errors++;
            }
        }
    }

    if (errors > 0) {
        fprintf(stderr, "[Validate] %d adjacency errors found!\n", errors);
    }
    return errors == 0;
}

// ── Initialization: Icosahedron × S¹ ──────────────────────────────────

void SimplicialComplex3D::initIcosahedronSandwich(int T, int /*verticesPerSlice*/) {
    m_vertices.clear();
    m_tets.clear();
    m_freeVertices.clear();
    m_freeTets.clear();
    m_aliveVertices = 0;
    m_aliveTets = 0;
    m_T = T;

    // Icosahedron: 12 vertices, 20 triangular faces
    // We place 12 vertices per time slice, connected by the icosahedron faces
    // Between adjacent slices: build tetrahedra from the triangular prism

    // Icosahedron vertex positions (unit sphere, for reference)
    // We don't need positions for topology — just connectivity
    static const int icoFaces[20][3] = {
        {0,1,2}, {0,2,3}, {0,3,4}, {0,4,5}, {0,5,1},
        {1,6,2}, {2,6,7}, {2,7,3}, {3,7,8}, {3,8,4},
        {4,8,9}, {4,9,5}, {5,9,10}, {5,10,1}, {1,10,6},
        {6,11,7}, {7,11,8}, {8,11,9}, {9,11,10}, {10,11,6}
    };

    // Allocate vertices: 12 per slice
    std::vector<std::vector<int>> sliceVerts(T);
    for (int t = 0; t < T; t++) {
        sliceVerts[t].resize(12);
        for (int i = 0; i < 12; i++) {
            sliceVerts[t][i] = allocVertex(t);
        }
    }

    // Between each pair of adjacent slices (t, t+1), build tetrahedra
    // For each triangular face of the icosahedron, we create a prism
    // and split it into 3 tetrahedra (standard prism decomposition)
    for (int t = 0; t < T; t++) {
        int tNext = (t + 1) % T;
        const auto& bot = sliceVerts[t];
        const auto& top = sliceVerts[tNext];

        for (int f = 0; f < 20; f++) {
            int b0 = bot[icoFaces[f][0]];
            int b1 = bot[icoFaces[f][1]];
            int b2 = bot[icoFaces[f][2]];
            int t0 = top[icoFaces[f][0]];
            int t1 = top[icoFaces[f][1]];
            int t2 = top[icoFaces[f][2]];

            // Split prism (b0,b1,b2)-(t0,t1,t2) into 3 tets
            // Standard diagonal decomposition preserving consistent orientation
            // Tet31: 3 vertices from bottom slice, 1 from top
            // Tet13: 1 vertex from bottom slice, 3 from top
            allocTet(TetType::Tet31, b0, b1, b2, t0);  // (3,1)
            allocTet(TetType::Tet13, b1, b2, t0, t1);  // mixed -> Tet13
            allocTet(TetType::Tet13, b2, t0, t1, t2);  // (1,3)
        }
    }

    // Build adjacency: for each pair of tets sharing a face, glue them
    // This is O(N_tet^2) for init but small enough at startup
    buildAdjacency();

    fprintf(stderr, "[SimplicialComplex3D] Init: T=%d, %d vertices, %d tetrahedra\n",
            T, m_aliveVertices, m_aliveTets);
}

// Helper: build adjacency by finding shared faces between all alive tets
// A face is identified by its 3 sorted vertex indices
void SimplicialComplex3D::buildAdjacency() {
    // Hash map: sorted face triple -> (tetIdx, faceIdx)
    struct FaceKey {
        int v[3];
        bool operator==(const FaceKey& o) const {
            return v[0] == o.v[0] && v[1] == o.v[1] && v[2] == o.v[2];
        }
    };
    struct FaceHash {
        size_t operator()(const FaceKey& k) const {
            size_t h = (size_t)k.v[0] * 73856093ULL;
            h ^= (size_t)k.v[1] * 19349663ULL;
            h ^= (size_t)k.v[2] * 83492791ULL;
            return h;
        }
    };

    struct FaceEntry {
        int tetIdx;
        int faceIdx;
    };

    std::unordered_map<FaceKey, FaceEntry, FaceHash> faceMap;

    for (int ti = 0; ti < static_cast<int>(m_tets.size()); ti++) {
        if (!m_tets[ti].alive) continue;
        const auto& tet = m_tets[ti];

        for (int f = 0; f < 4; f++) {
            // Face f = vertices excluding vertex[f]
            FaceKey key;
            int idx = 0;
            for (int i = 0; i < 4; i++) {
                if (i != f) key.v[idx++] = tet.vertices[i];
            }
            // Sort
            if (key.v[0] > key.v[1]) std::swap(key.v[0], key.v[1]);
            if (key.v[1] > key.v[2]) std::swap(key.v[1], key.v[2]);
            if (key.v[0] > key.v[1]) std::swap(key.v[0], key.v[1]);

            auto it = faceMap.find(key);
            if (it != faceMap.end()) {
                // Found matching face: glue
                glueTetrahedra(ti, f, it->second.tetIdx, it->second.faceIdx);
                faceMap.erase(it);
            } else {
                faceMap[key] = {ti, f};
            }
        }
    }
}

} // namespace cosmico
