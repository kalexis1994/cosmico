#pragma once
#include <array>
#include <vector>
#include <cstdint>

namespace cosmico {

enum class TetType : uint8_t { Tet31, Tet13 };

struct Vertex3D {
    int timeSlice = -1;
    int tetIndex = -1;     // Any incident tet (for traversal)
    bool alive = false;
};

struct Tetrahedron {
    std::array<int, 4> vertices = {-1,-1,-1,-1};    // Vertex indices
    std::array<int, 4> neighbors = {-1,-1,-1,-1};   // Neighbor tet opposite vertex[i]
    TetType type = TetType::Tet31;
    bool alive = false;
};

class SimplicialComplex3D {
public:
    // Allocation with O(1) free-list
    int allocVertex(int timeSlice);
    void freeVertex(int idx);
    int allocTet(TetType type, int v0, int v1, int v2, int v3);
    void freeTet(int idx);

    // Topological queries
    std::vector<int> tetsAroundVertex(int vertIdx) const;
    std::vector<int> tetsAroundEdge(int v0, int v1) const;

    struct Triangle { int v0, v1, v2; };
    std::vector<Triangle> spatialTriangulation(int timeSlice) const;

    // Adjacency helpers
    void setNeighbor(int tetIdx, int face, int neighborTet);
    int findFace(int tetIdx, int oppositeVertex) const;
    int sharedFaceIndex(int tetA, int tetB) const;
    void glueTetrahedra(int tetA, int faceA, int tetB, int faceB);

    // Validation (debug)
    bool validateAdjacency() const;

    // Initialization & adjacency rebuild
    void initIcosahedronSandwich(int T, int verticesPerSlice);
    void buildAdjacency();

    // Access
    const Vertex3D& vertex(int idx) const { return m_vertices[idx]; }
    Vertex3D& vertex(int idx) { return m_vertices[idx]; }
    const Tetrahedron& tet(int idx) const { return m_tets[idx]; }
    Tetrahedron& tet(int idx) { return m_tets[idx]; }

    int vertexCount() const { return m_aliveVertices; }
    int tetCount() const { return m_aliveTets; }
    int maxVertices() const { return static_cast<int>(m_vertices.size()); }
    int maxTets() const { return static_cast<int>(m_tets.size()); }
    int timeSlices() const { return m_T; }

    // Bulk data for GPU upload
    const std::vector<Vertex3D>& vertices() const { return m_vertices; }
    const std::vector<Tetrahedron>& tets() const { return m_tets; }

private:
    std::vector<Vertex3D> m_vertices;
    std::vector<Tetrahedron> m_tets;

    // Free lists
    std::vector<int> m_freeVertices;
    std::vector<int> m_freeTets;

    int m_aliveVertices = 0;
    int m_aliveTets = 0;
    int m_T = 0;
};

} // namespace cosmico
