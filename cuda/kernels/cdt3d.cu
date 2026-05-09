#include <kernels/cdt3d.cuh>
#include <cstdio>

namespace cosmico {
namespace cuda {

// ── Sphere Embedding Kernel ─────────────────────────────────────────────

__global__ void sphereEmbedKernelImpl(
    float3* positions,
    const int2* edges,
    int numVerts,
    int numEdges,
    float targetEdgeLen,
    float radius,
    float dt
) {
    int vid = blockIdx.x * blockDim.x + threadIdx.x;
    if (vid >= numVerts) return;

    float3 pos = positions[vid];
    float fx = 0.0f, fy = 0.0f, fz = 0.0f;

    // Accumulate spring forces from incident edges
    for (int e = 0; e < numEdges; e++) {
        int2 edge = edges[e];
        int other = -1;
        if (edge.x == vid) other = edge.y;
        else if (edge.y == vid) other = edge.x;
        else continue;

        float3 otherPos = positions[other];
        float dx = otherPos.x - pos.x;
        float dy = otherPos.y - pos.y;
        float dz = otherPos.z - pos.z;
        float dist = sqrtf(dx*dx + dy*dy + dz*dz);
        if (dist < 1e-6f) continue;

        float force = (dist - targetEdgeLen) / dist;
        fx += force * dx;
        fy += force * dy;
        fz += force * dz;
    }

    // Apply force
    pos.x += fx * dt;
    pos.y += fy * dt;
    pos.z += fz * dt;

    // Project back onto sphere
    float len = sqrtf(pos.x*pos.x + pos.y*pos.y + pos.z*pos.z);
    if (len > 1e-6f) {
        float invLen = radius / len;
        pos.x *= invLen;
        pos.y *= invLen;
        pos.z *= invLen;
    }

    positions[vid] = pos;
}

void launchSphereEmbedKernel(
    float3* positions, const int2* edges,
    int numVerts, int numEdges,
    float targetEdgeLen, float radius, float dt,
    cudaStream_t stream
) {
    if (numVerts == 0) return;
    int threads = 256;
    int blocks = (numVerts + threads - 1) / threads;
    sphereEmbedKernelImpl<<<blocks, threads, 0, stream>>>(
        positions, edges, numVerts, numEdges, targetEdgeLen, radius, dt
    );
}

// ── Build Render Data Kernel ──────────────────────────────────────────

__global__ void buildRenderDataKernelImpl(
    const float3* vertexPos,
    const int3* triangles,
    const int* triSlice,
    const float* curvature,
    float* renderOutput,
    int numTriangles,
    float yOffset,
    int colorByCurvature,
    int T
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numTriangles) return;

    int3 tri = triangles[tid];
    int slice = triSlice[tid];

    float3 p0 = vertexPos[tri.x];
    float3 p1 = vertexPos[tri.y];
    float3 p2 = vertexPos[tri.z];

    // Apply vertical offset for time slice
    float yOff = (float)(slice - T / 2) * yOffset;
    p0.y += yOff;
    p1.y += yOff;
    p2.y += yOff;

    // Compute face normal
    float e1x = p1.x - p0.x, e1y = p1.y - p0.y, e1z = p1.z - p0.z;
    float e2x = p2.x - p0.x, e2y = p2.y - p0.y, e2z = p2.z - p0.z;
    float nx = e1y * e2z - e1z * e2y;
    float ny = e1z * e2x - e1x * e2z;
    float nz = e1x * e2y - e1y * e2x;
    float nLen = sqrtf(nx*nx + ny*ny + nz*nz);
    if (nLen > 1e-6f) {
        nx /= nLen; ny /= nLen; nz /= nLen;
    }

    // Color: curvature-based or uniform
    float r0, g0, b0, r1, g1, b1, r2, g2, b2;
    if (colorByCurvature) {
        // Map curvature to blue-white-red
        auto curv2color = [](float c, float& cr, float& cg, float& cb) {
            float v = fmaxf(-1.0f, fminf(1.0f, c * 5.0f)); // scale up for visibility
            if (v > 0.0f) {
                cr = 1.0f; cg = 1.0f - v; cb = 1.0f - v;
            } else {
                float s = -v;
                cr = 1.0f - s; cg = 1.0f - s; cb = 1.0f;
            }
        };
        curv2color(curvature[tri.x], r0, g0, b0);
        curv2color(curvature[tri.y], r1, g1, b1);
        curv2color(curvature[tri.z], r2, g2, b2);
    } else {
        // Uniform color: light blue tinted by slice
        float t = (float)slice / (float)T;
        r0 = r1 = r2 = 0.3f + 0.2f * t;
        g0 = g1 = g2 = 0.5f + 0.2f * t;
        b0 = b1 = b2 = 0.8f;
    }

    // Write 3 vertices * 9 floats
    float* out = renderOutput + tid * 27; // 3 * 9
    // Vertex 0
    out[0] = p0.x; out[1] = p0.y; out[2] = p0.z;
    out[3] = nx;   out[4] = ny;   out[5] = nz;
    out[6] = r0;   out[7] = g0;   out[8] = b0;
    // Vertex 1
    out[9]  = p1.x; out[10] = p1.y; out[11] = p1.z;
    out[12] = nx;   out[13] = ny;   out[14] = nz;
    out[15] = r1;   out[16] = g1;   out[17] = b1;
    // Vertex 2
    out[18] = p2.x; out[19] = p2.y; out[20] = p2.z;
    out[21] = nx;   out[22] = ny;   out[23] = nz;
    out[24] = r2;   out[25] = g2;   out[26] = b2;
}

void launchBuildRenderDataKernel(
    const float3* vertexPos, const int3* triangles, const int* triSlice,
    const float* curvature, float* renderOutput,
    int numTriangles, float yOffset, int colorByCurvature, int T,
    cudaStream_t stream
) {
    if (numTriangles == 0) return;
    int threads = 256;
    int blocks = (numTriangles + threads - 1) / threads;
    buildRenderDataKernelImpl<<<blocks, threads, 0, stream>>>(
        vertexPos, triangles, triSlice, curvature, renderOutput,
        numTriangles, yOffset, colorByCurvature, T
    );
}

// ── Random Walk Kernel (Spectral Dimension) ───────────────────────────

__global__ void randomWalkKernelImpl(
    const int* tetNeighbors,
    const int* tetAlive,
    int numTets,
    int numWalks,
    int walkLength,
    float* returnProb,
    unsigned long long seed
) {
    int wid = blockIdx.x * blockDim.x + threadIdx.x;
    if (wid >= numWalks) return;

    // Initialize RNG: simple LCG per thread
    unsigned long long rng = seed ^ ((unsigned long long)wid * 6364136223846793005ULL + 1442695040888963407ULL);

    // Pick random starting tet
    auto nextRng = [&]() -> unsigned int {
        rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
        return (unsigned int)(rng >> 33);
    };

    // Find an alive tet to start from
    int startTet = -1;
    for (int attempt = 0; attempt < 100; attempt++) {
        int t = nextRng() % numTets;
        if (tetAlive[t]) {
            startTet = t;
            break;
        }
    }
    if (startTet < 0) return;

    int currentTet = startTet;

    for (int sigma = 0; sigma < walkLength; sigma++) {
        // Pick random neighbor (0-3)
        int dir = nextRng() % 4;
        int nb = tetNeighbors[currentTet * 4 + dir];

        if (nb >= 0 && tetAlive[nb]) {
            currentTet = nb;
        }
        // else stay in place

        // Check return
        if (currentTet == startTet) {
            atomicAdd(&returnProb[sigma], 1.0f);
        }
    }
}

void launchRandomWalkKernel(
    const int* tetNeighbors, const int* tetAlive,
    int numTets, int numWalks, int walkLength,
    float* returnProb, unsigned long long seed,
    cudaStream_t stream
) {
    if (numTets == 0 || numWalks == 0) return;

    // Zero the output
    cudaMemsetAsync(returnProb, 0, walkLength * sizeof(float), stream);

    int threads = 256;
    int blocks = (numWalks + threads - 1) / threads;
    randomWalkKernelImpl<<<blocks, threads, 0, stream>>>(
        tetNeighbors, tetAlive, numTets, numWalks, walkLength, returnProb, seed
    );
}

// ── Volume Profile Kernel ─────────────────────────────────────────────

__global__ void volumeProfileKernelImpl(
    const int* tetVertices,
    const int* vertexSlice,
    const int* tetAlive,
    int numTets,
    int T,
    int* volumeProfile
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numTets) return;
    if (!tetAlive[tid]) return;

    // For each face of this tet, check if all 3 vertices are in the same slice
    int v[4];
    v[0] = tetVertices[tid * 4 + 0];
    v[1] = tetVertices[tid * 4 + 1];
    v[2] = tetVertices[tid * 4 + 2];
    v[3] = tetVertices[tid * 4 + 3];

    int s[4];
    s[0] = vertexSlice[v[0]];
    s[1] = vertexSlice[v[1]];
    s[2] = vertexSlice[v[2]];
    s[3] = vertexSlice[v[3]];

    // Check each face (opposite vertex i)
    // Face 0: v1,v2,v3  Face 1: v0,v2,v3  Face 2: v0,v1,v3  Face 3: v0,v1,v2
    for (int f = 0; f < 4; f++) {
        int fs[3];
        int idx = 0;
        for (int i = 0; i < 4; i++) {
            if (i != f) fs[idx++] = s[i];
        }
        if (fs[0] == fs[1] && fs[1] == fs[2]) {
            int slice = fs[0];
            if (slice >= 0 && slice < T) {
                // Each spatial triangle is shared by 2 tets, so count 1 per face
                // We only count if opposite vertex is in a DIFFERENT slice
                if (s[f] != slice) {
                    atomicAdd(&volumeProfile[slice], 1);
                }
            }
        }
    }
}

void launchVolumeProfileKernel(
    const int* tetVertices, const int* vertexSlice, const int* tetAlive,
    int numTets, int T, int* volumeProfile,
    cudaStream_t stream
) {
    if (numTets == 0) return;

    cudaMemsetAsync(volumeProfile, 0, T * sizeof(int), stream);

    int threads = 256;
    int blocks = (numTets + threads - 1) / threads;
    volumeProfileKernelImpl<<<blocks, threads, 0, stream>>>(
        tetVertices, vertexSlice, tetAlive, numTets, T, volumeProfile
    );
}

// ── Curvature Kernel ──────────────────────────────────────────────────

__global__ void curvatureKernelImpl(
    const float3* positions,
    const int3* triangles,
    const int* triPerVertex,
    const int* triList,
    float* curvature,
    int numVerts
) {
    int vid = blockIdx.x * blockDim.x + threadIdx.x;
    if (vid >= numVerts) return;

    int start = triPerVertex[vid];
    int end = triPerVertex[vid + 1];

    float angleSum = 0.0f;
    float3 pv = positions[vid];

    for (int i = start; i < end; i++) {
        int triIdx = triList[i];
        int3 tri = triangles[triIdx];

        // Find the two other vertices of this triangle
        int other0 = -1, other1 = -1;
        if (tri.x == vid) { other0 = tri.y; other1 = tri.z; }
        else if (tri.y == vid) { other0 = tri.x; other1 = tri.z; }
        else { other0 = tri.x; other1 = tri.y; }

        float3 p0 = positions[other0];
        float3 p1 = positions[other1];

        // Vectors from vid to neighbors
        float e0x = p0.x - pv.x, e0y = p0.y - pv.y, e0z = p0.z - pv.z;
        float e1x = p1.x - pv.x, e1y = p1.y - pv.y, e1z = p1.z - pv.z;

        float len0 = sqrtf(e0x*e0x + e0y*e0y + e0z*e0z);
        float len1 = sqrtf(e1x*e1x + e1y*e1y + e1z*e1z);

        if (len0 > 1e-8f && len1 > 1e-8f) {
            float dot = (e0x*e1x + e0y*e1y + e0z*e1z) / (len0 * len1);
            dot = fmaxf(-1.0f, fminf(1.0f, dot));
            angleSum += acosf(dot);
        }
    }

    // Gaussian curvature: 2*pi - sum of angles
    curvature[vid] = 6.28318530718f - angleSum;
}

void launchCurvatureKernel(
    const float3* positions, const int3* triangles,
    const int* triPerVertex, const int* triList,
    float* curvature, int numVerts,
    cudaStream_t stream
) {
    if (numVerts == 0) return;
    int threads = 256;
    int blocks = (numVerts + threads - 1) / threads;
    curvatureKernelImpl<<<blocks, threads, 0, stream>>>(
        positions, triangles, triPerVertex, triList, curvature, numVerts
    );
}

} // namespace cuda
} // namespace cosmico
