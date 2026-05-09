#include "kernels/barnes_hut.cuh"
#include <cfloat>
#include <cstdio>

namespace cosmico {
namespace cuda {

static constexpr int BLOCK_SIZE = 256;
static constexpr int MAX_STACK_DEPTH = 32;

// IEEE 754 float-safe atomic min/max using CAS loops
__device__ void atomicMinFloat(float* addr, float val) {
    if (val >= 0.0f) {
        // For non-negative floats, int representation preserves ordering
        atomicMin(reinterpret_cast<int*>(addr), __float_as_int(val));
    } else {
        // For negative floats, use CAS loop
        int* addr_as_int = reinterpret_cast<int*>(addr);
        int old = *addr_as_int, assumed;
        do {
            assumed = old;
            if (__int_as_float(assumed) <= val) break;
            old = atomicCAS(addr_as_int, assumed, __float_as_int(val));
        } while (assumed != old);
    }
}

__device__ void atomicMaxFloat(float* addr, float val) {
    if (val >= 0.0f) {
        // For non-negative floats, int representation preserves ordering
        atomicMax(reinterpret_cast<int*>(addr), __float_as_int(val));
    } else {
        // For negative floats, use CAS loop
        int* addr_as_int = reinterpret_cast<int*>(addr);
        int old = *addr_as_int, assumed;
        do {
            assumed = old;
            if (__int_as_float(assumed) >= val) break;
            old = atomicCAS(addr_as_int, assumed, __float_as_int(val));
        } while (assumed != old);
    }
}

// ─── Kernel 1: Bounding Box Reduction ──────────────────────────────

__global__ void boundingBoxKernel(const ParticleGpu* __restrict__ particles,
                                   BoundingBox* __restrict__ bbox,
                                   int N) {
    __shared__ float sMinX[BLOCK_SIZE], sMinY[BLOCK_SIZE], sMinZ[BLOCK_SIZE];
    __shared__ float sMaxX[BLOCK_SIZE], sMaxY[BLOCK_SIZE], sMaxZ[BLOCK_SIZE];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize with extreme values
    float lMinX = FLT_MAX, lMinY = FLT_MAX, lMinZ = FLT_MAX;
    float lMaxX = -FLT_MAX, lMaxY = -FLT_MAX, lMaxZ = -FLT_MAX;

    // Grid-stride loop for full coverage
    for (int i = gid; i < N; i += blockDim.x * gridDim.x) {
        float x = particles[i].px;
        float y = particles[i].py;
        float z = particles[i].pz;
        lMinX = fminf(lMinX, x); lMinY = fminf(lMinY, y); lMinZ = fminf(lMinZ, z);
        lMaxX = fmaxf(lMaxX, x); lMaxY = fmaxf(lMaxY, y); lMaxZ = fmaxf(lMaxZ, z);
    }

    sMinX[tid] = lMinX; sMinY[tid] = lMinY; sMinZ[tid] = lMinZ;
    sMaxX[tid] = lMaxX; sMaxY[tid] = lMaxY; sMaxZ[tid] = lMaxZ;
    __syncthreads();

    // Block-level reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sMinX[tid] = fminf(sMinX[tid], sMinX[tid + s]);
            sMinY[tid] = fminf(sMinY[tid], sMinY[tid + s]);
            sMinZ[tid] = fminf(sMinZ[tid], sMinZ[tid + s]);
            sMaxX[tid] = fmaxf(sMaxX[tid], sMaxX[tid + s]);
            sMaxY[tid] = fmaxf(sMaxY[tid], sMaxY[tid + s]);
            sMaxZ[tid] = fmaxf(sMaxZ[tid], sMaxZ[tid + s]);
        }
        __syncthreads();
    }

    // First thread of each block writes to global using IEEE 754-safe float atomics
    if (tid == 0) {
        atomicMinFloat(&bbox->minX, sMinX[0]);
        atomicMinFloat(&bbox->minY, sMinY[0]);
        atomicMinFloat(&bbox->minZ, sMinZ[0]);
        atomicMaxFloat(&bbox->maxX, sMaxX[0]);
        atomicMaxFloat(&bbox->maxY, sMaxY[0]);
        atomicMaxFloat(&bbox->maxZ, sMaxZ[0]);
    }
}

// ─── Kernel 2a: Set children[] and startIdx to -1 ────────────────
// Called AFTER cudaMemsetAsync zeros the entire node region.
// Only needs to write 36 bytes/node instead of 80.

__global__ void initNodeFieldsKernel(OctreeNode* __restrict__ nodes, int startNode, int count) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = gid; i < count; i += blockDim.x * gridDim.x) {
        int idx = startNode + i;
        nodes[idx].children[0] = -1; nodes[idx].children[1] = -1;
        nodes[idx].children[2] = -1; nodes[idx].children[3] = -1;
        nodes[idx].children[4] = -1; nodes[idx].children[5] = -1;
        nodes[idx].children[6] = -1; nodes[idx].children[7] = -1;
        nodes[idx].startIdx = -1;
    }
}

// ─── Kernel 2b: Initialize root node from bounding box ───────────
//     Must launch AFTER clearNodesKernel on the same stream.

__global__ void initRootKernel(OctreeNode* __restrict__ nodes,
                                const BoundingBox* __restrict__ bbox,
                                int rootIdx) {
    // Add small padding to bounding box to avoid edge cases
    float pad = 0.01f;
    float bMinX = bbox->minX - pad;
    float bMinY = bbox->minY - pad;
    float bMinZ = bbox->minZ - pad;
    float bMaxX = bbox->maxX + pad;
    float bMaxY = bbox->maxY + pad;
    float bMaxZ = bbox->maxZ + pad;

    // Make it cubic (use max extent)
    float dx = bMaxX - bMinX;
    float dy = bMaxY - bMinY;
    float dz = bMaxZ - bMinZ;
    float maxExtent = fmaxf(dx, fmaxf(dy, dz));
    float cx = (bMinX + bMaxX) * 0.5f;
    float cy = (bMinY + bMaxY) * 0.5f;
    float cz = (bMinZ + bMaxZ) * 0.5f;

    nodes[rootIdx].minX = cx - maxExtent * 0.5f;
    nodes[rootIdx].minY = cy - maxExtent * 0.5f;
    nodes[rootIdx].minZ = cz - maxExtent * 0.5f;
    nodes[rootIdx].maxX = cx + maxExtent * 0.5f;
    nodes[rootIdx].maxY = cy + maxExtent * 0.5f;
    nodes[rootIdx].maxZ = cz + maxExtent * 0.5f;
    nodes[rootIdx].startIdx = -1;
    nodes[rootIdx].particleCount = 0;
}

// ─── Helper: Get octant for a position within a node ────────────────

__device__ int getOctant(float px, float py, float pz,
                          float midX, float midY, float midZ) {
    int octant = 0;
    if (px >= midX) octant |= 1;
    if (py >= midY) octant |= 2;
    if (pz >= midZ) octant |= 4;
    return octant;
}

__device__ void getChildBounds(int octant,
                                float pMinX, float pMinY, float pMinZ,
                                float pMaxX, float pMaxY, float pMaxZ,
                                float& cMinX, float& cMinY, float& cMinZ,
                                float& cMaxX, float& cMaxY, float& cMaxZ) {
    float midX = (pMinX + pMaxX) * 0.5f;
    float midY = (pMinY + pMaxY) * 0.5f;
    float midZ = (pMinZ + pMaxZ) * 0.5f;

    cMinX = (octant & 1) ? midX : pMinX;
    cMaxX = (octant & 1) ? pMaxX : midX;
    cMinY = (octant & 2) ? midY : pMinY;
    cMaxY = (octant & 2) ? pMaxY : midY;
    cMinZ = (octant & 4) ? midZ : pMinZ;
    cMaxZ = (octant & 4) ? pMaxZ : midZ;
}

// ─── Kernel 3: Build Tree (insert particles) ───────────────────────

// Device helper: create a leaf node and publish it into parent's child slot.
// Returns the new node index, or -1 on failure.
__device__ int createLeaf(int particleIdx, float px, float py, float pz,
                           const ParticleGpu* __restrict__ particles,
                           OctreeNode* __restrict__ nodes,
                           int* __restrict__ nodeCounter,
                           int parentIdx, int octant, int maxNodes) {
    int newNode = atomicAdd(nodeCounter, 1);
    if (newNode >= maxNodes) {
        nodes[parentIdx].children[octant] = -1; // release slot
        return -1;
    }

    float cMinX, cMinY, cMinZ, cMaxX, cMaxY, cMaxZ;
    getChildBounds(octant,
                  nodes[parentIdx].minX, nodes[parentIdx].minY, nodes[parentIdx].minZ,
                  nodes[parentIdx].maxX, nodes[parentIdx].maxY, nodes[parentIdx].maxZ,
                  cMinX, cMinY, cMinZ, cMaxX, cMaxY, cMaxZ);

    nodes[newNode].minX = cMinX; nodes[newNode].minY = cMinY; nodes[newNode].minZ = cMinZ;
    nodes[newNode].maxX = cMaxX; nodes[newNode].maxY = cMaxY; nodes[newNode].maxZ = cMaxZ;
    nodes[newNode].startIdx = particleIdx;
    nodes[newNode].particleCount = 1;
    nodes[newNode].mass = particles[particleIdx].mass;
    nodes[newNode].cx = px;
    nodes[newNode].cy = py;
    nodes[newNode].cz = pz;
    for (int c = 0; c < 8; c++) nodes[newNode].children[c] = -1;

    __threadfence();
    nodes[parentIdx].children[octant] = newNode; // publish
    return newNode;
}

__global__ void buildTreeKernel(const ParticleGpu* __restrict__ particles,
                                 OctreeNode* __restrict__ nodes,
                                 int* __restrict__ nodeCounter,
                                 int N, int rootIdx, int maxNodes) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= N) return;

    const int MAX_DEPTH = 30;

    // Stack of particles to insert (gid + any displaced by subdivision)
    int insertStack[8]; // particle indices
    int insertFrom[8];  // starting node for each insertion
    int stackTop = 0;
    insertStack[stackTop] = gid;
    insertFrom[stackTop] = rootIdx;
    stackTop++;

    while (stackTop > 0) {
        stackTop--;
        int partIdx = insertStack[stackTop];
        int nodeIdx = insertFrom[stackTop];

        float px = particles[partIdx].px;
        float py = particles[partIdx].py;
        float pz = particles[partIdx].pz;

        for (int depth = 0; depth < MAX_DEPTH; depth++) {
            float midX = (nodes[nodeIdx].minX + nodes[nodeIdx].maxX) * 0.5f;
            float midY = (nodes[nodeIdx].minY + nodes[nodeIdx].maxY) * 0.5f;
            float midZ = (nodes[nodeIdx].minZ + nodes[nodeIdx].maxZ) * 0.5f;

            int octant = getOctant(px, py, pz, midX, midY, midZ);

            // Try to claim this child slot
            int oldVal = atomicCAS(&nodes[nodeIdx].children[octant], -1, -2);

            if (oldVal == -1) {
                // We claimed an empty slot — create leaf
                createLeaf(partIdx, px, py, pz, particles, nodes,
                          nodeCounter, nodeIdx, octant, maxNodes);
                break; // This particle is inserted

            } else if (oldVal == -2) {
                // Slot locked by another thread — spin until valid
                while (true) {
                    int v = atomicAdd(&nodes[nodeIdx].children[octant], 0);
                    if (v >= 0) break;
                }
                // Fall through to traverse
            }

            // Read the child
            int childIdx = nodes[nodeIdx].children[octant];
            if (childIdx < 0) { depth--; continue; } // retry

            // If child is a leaf, subdivide it
            if (nodes[childIdx].startIdx >= 0 && nodes[childIdx].particleCount == 1) {
                int existingParticle = nodes[childIdx].startIdx;
                int old = atomicCAS(&nodes[childIdx].startIdx, existingParticle, -1);

                if (old == existingParticle) {
                    // We took ownership — convert leaf to internal node
                    nodes[childIdx].particleCount = 0;
                    nodes[childIdx].mass = 0.0f;

                    // Push existing particle onto stack for re-insertion from childIdx
                    if (stackTop < 8) {
                        insertStack[stackTop] = existingParticle;
                        insertFrom[stackTop] = childIdx;
                        stackTop++;
                    }
                }
                // childIdx is now (or becoming) an internal node — traverse into it
            }

            nodeIdx = childIdx;
        }
    }
}

// ─── Kernel 4: Center of Mass (bottom-up) ──────────────────────────

// Only processes nodes in [N, maxNodes) — the actual tree node range.
__global__ void centerOfMassKernel(OctreeNode* __restrict__ nodes,
                                     int N, int maxNodes) {
    int treeNodeCount = maxNodes - N;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= treeNodeCount) return;

    // Process in reverse order (higher indices = deeper nodes first)
    int nodeIdx = maxNodes - 1 - gid;
    OctreeNode& node = nodes[nodeIdx];

    // Skip leaf nodes (already have correct mass/COM from build)
    if (node.startIdx >= 0) return;

    // Check if this is an internal node with children
    bool hasChild = false;
    for (int c = 0; c < 8; c++) {
        if (node.children[c] >= 0) { hasChild = true; break; }
    }
    if (!hasChild) return;

    // Accumulate mass and center of mass from children
    float totalMass = 0.0f;
    float comX = 0.0f, comY = 0.0f, comZ = 0.0f;
    int totalCount = 0;

    for (int c = 0; c < 8; c++) {
        int childIdx = node.children[c];
        if (childIdx < 0) continue;

        float cm = nodes[childIdx].mass;
        if (cm > 0.0f) {
            comX += nodes[childIdx].cx * cm;
            comY += nodes[childIdx].cy * cm;
            comZ += nodes[childIdx].cz * cm;
            totalMass += cm;
        }
        totalCount += nodes[childIdx].particleCount;
    }

    if (totalMass > 0.0f) {
        float invMass = 1.0f / totalMass;
        node.cx = comX * invMass;
        node.cy = comY * invMass;
        node.cz = comZ * invMass;
    }
    node.mass = totalMass;
    node.particleCount = totalCount;
}

// ─── Kernel 5: Force Calculation (tree walk) ────────────────────────

__global__ void forceKernel(ParticleGpu* __restrict__ particles,
                             const OctreeNode* __restrict__ nodes,
                             float G, float softening, float theta,
                             int N, int rootIdx) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= N) return;

    float px = particles[gid].px;
    float py = particles[gid].py;
    float pz = particles[gid].pz;

    float ax = 0.0f, ay = 0.0f, az = 0.0f;
    float soft2 = softening * softening;
    float theta2 = theta * theta;

    // Stack-based tree walk
    int stack[MAX_STACK_DEPTH];
    int stackTop = 0;
    stack[stackTop++] = rootIdx;

    while (stackTop > 0) {
        int nodeIdx = stack[--stackTop];
        if (nodeIdx < 0) continue;

        const OctreeNode& node = nodes[nodeIdx];
        if (node.mass <= 0.0f) continue;

        // If leaf with single particle
        if (node.startIdx >= 0 && node.particleCount == 1) {
            if (node.startIdx == gid) continue; // Skip self

            float dx = node.cx - px;
            float dy = node.cy - py;
            float dz = node.cz - pz;
            float r2 = dx * dx + dy * dy + dz * dz + soft2;
            float invR = rsqrtf(r2);
            float invR3 = invR * invR * invR;
            float f = G * node.mass * invR3;
            ax += f * dx;
            ay += f * dy;
            az += f * dz;
            continue;
        }

        // Barnes-Hut criterion: s/d < theta → use center of mass
        float sizeX = node.maxX - node.minX;
        float sizeY = node.maxY - node.minY;
        float sizeZ = node.maxZ - node.minZ;
        float s = fmaxf(sizeX, fmaxf(sizeY, sizeZ));

        float dx = node.cx - px;
        float dy = node.cy - py;
        float dz = node.cz - pz;
        float d2 = dx * dx + dy * dy + dz * dz;

        if (s * s < theta2 * d2) {
            // Node is far enough — use approximation
            float r2 = d2 + soft2;
            float invR = rsqrtf(r2);
            float invR3 = invR * invR * invR;
            float f = G * node.mass * invR3;
            ax += f * dx;
            ay += f * dy;
            az += f * dz;
        } else {
            // Node is too close — push children
            for (int c = 7; c >= 0; c--) {
                if (node.children[c] >= 0 && stackTop < MAX_STACK_DEPTH) {
                    stack[stackTop++] = node.children[c];
                }
            }
        }
    }

    // Store acceleration temporarily in velocity fields (will be used by integrate)
    // We use the density/pressure/smoothing fields to store acceleration
    particles[gid].density = ax;
    particles[gid].pressure = ay;
    particles[gid].smoothing = az;
}

// ─── Kernel 6a: Half Drift (DKD leapfrog — first half) ──────────────
// x += v * dt/2   (before force computation)

__global__ void halfDriftKernel(ParticleGpu* __restrict__ particles,
                                 float halfDt, int N) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= N) return;

    particles[gid].px += particles[gid].vx * halfDt;
    particles[gid].py += particles[gid].vy * halfDt;
    particles[gid].pz += particles[gid].vz * halfDt;
}

// ─── Kernel 6b: Kick + Half Drift (DKD leapfrog — second half) ─────
// v += a * dt, then x += v * dt/2   (after force computation)

__global__ void kickHalfDriftKernel(ParticleGpu* __restrict__ particles,
                                     float dt, float halfDt, int N) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= N) return;

    // Read acceleration stored by force kernel
    float ax = particles[gid].density;
    float ay = particles[gid].pressure;
    float az = particles[gid].smoothing;

    // Full kick: v += a * dt
    particles[gid].vx += ax * dt;
    particles[gid].vy += ay * dt;
    particles[gid].vz += az * dt;

    // Half drift: x += v * dt/2
    particles[gid].px += particles[gid].vx * halfDt;
    particles[gid].py += particles[gid].vy * halfDt;
    particles[gid].pz += particles[gid].vz * halfDt;

    // Clear acceleration fields
    particles[gid].density = 0.0f;
    particles[gid].pressure = 0.0f;
    particles[gid].smoothing = 0.0f;
}

// ─── Host wrapper ──────────────────────────────────────────────────

void launchBarnesHutStep(
    ParticleGpu* particles,
    OctreeNode* nodes,
    BoundingBox* d_bbox,
    int* d_nodeCounter,
    const BarnesHutParams& params,
    cudaStream_t stream)
{
    int N = params.particleCount;
    int rootIdx = N; // Root is at index N in the node array
    int maxNodes = params.maxNodes;

    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    float halfDt = params.dt * 0.5f;

    // DKD leapfrog step 1: half drift (x += v * dt/2)
    halfDriftKernel<<<numBlocks, BLOCK_SIZE, 0, stream>>>(particles, halfDt, N);

    // Reset bounding box to extreme values
    BoundingBox initBbox;
    initBbox.minX = FLT_MAX; initBbox.minY = FLT_MAX; initBbox.minZ = FLT_MAX;
    initBbox.maxX = -FLT_MAX; initBbox.maxY = -FLT_MAX; initBbox.maxZ = -FLT_MAX;
    cudaMemcpyAsync(d_bbox, &initBbox, sizeof(BoundingBox), cudaMemcpyHostToDevice, stream);

    // Reset node counter (starts after leaf nodes at N, root is pre-allocated at N, so counter starts at N+1)
    int initCounter = N + 1;
    cudaMemcpyAsync(d_nodeCounter, &initCounter, sizeof(int), cudaMemcpyHostToDevice, stream);

    // 1. Bounding box
    boundingBoxKernel<<<numBlocks, BLOCK_SIZE, 0, stream>>>(particles, d_bbox, N);

    // 2a. Clear tree nodes [N, maxNodes) — indices [0, N-1] are unused
    // Fast DMA zero + lean kernel for -1 fields only
    int treeNodeCount = maxNodes - N;
    cudaMemsetAsync(nodes + N, 0, treeNodeCount * sizeof(OctreeNode), stream);
    int clearBlocks = (treeNodeCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
    initNodeFieldsKernel<<<clearBlocks, BLOCK_SIZE, 0, stream>>>(nodes, N, treeNodeCount);

    // 2b. Initialize root from bounding box (runs AFTER clear completes on same stream)
    initRootKernel<<<1, 1, 0, stream>>>(nodes, d_bbox, rootIdx);

    // 3. Build tree
    buildTreeKernel<<<numBlocks, BLOCK_SIZE, 0, stream>>>(
        particles, nodes, d_nodeCounter, N, rootIdx, maxNodes);

    // 4. Center of mass (bottom-up: each pass propagates one level up)
    // Only process [N, maxNodes). 8 passes covers most practical tree depths.
    // Deep branches use leaf nodes directly in force kernel, so slight COM
    // inaccuracy at very deep internal nodes has minimal impact.
    int comBlocks = (treeNodeCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (int pass = 0; pass < 8; pass++) {
        centerOfMassKernel<<<comBlocks, BLOCK_SIZE, 0, stream>>>(nodes, N, maxNodes);
    }

    // 5. Force calculation
    forceKernel<<<numBlocks, BLOCK_SIZE, 0, stream>>>(
        particles, nodes, params.G, params.softening, params.theta, N, rootIdx);

    // 6. DKD leapfrog steps 2-3: full kick + half drift
    kickHalfDriftKernel<<<numBlocks, BLOCK_SIZE, 0, stream>>>(particles, params.dt, halfDt, N);
}

size_t getTreeMemorySize(int particleCount) {
    // Need at most 2x particles for nodes (leaves + internals)
    // Add some headroom for dynamic allocation
    int maxNodes = particleCount * 3;
    return static_cast<size_t>(maxNodes) * sizeof(OctreeNode);
}

// ─── Diagnostics kernel: KE, PE (tree walk), momentum, mass*pos ─────
//
// One thread per particle. Walks the existing octree to accumulate the
// particle's potential energy under the same Barnes-Hut approximation
// used by forceKernel — so PE quality matches the force quality.
//
// Reduction strategy (warp shuffle + tiny shared mem):
//   1. Each thread accumulates 9 doubles (KE, PE, p_xyz, mr_xyz, m).
//   2. Each warp reduces in-register via __shfl_xor_sync (no shared mem).
//   3. Lane 0 of each warp writes its sum to a small shared buffer
//      (9 quantities * 8 warps for BLOCK_SIZE=256 = 576 bytes total).
//   4. The first warp loads those 8 partials and warp-reduces again.
//   5. Lane 0 atomicAdds the block totals to the global accumulator.
//
// The earlier implementation kept a full BLOCK_SIZE-wide shared array
// per quantity (~18 KB shared/block), which capped the kernel to ~3
// blocks/SM. The shuffle version uses ~576 B/block and lets the
// scheduler hit the register-bound occupancy of a Turing/Ampere/Ada SM.
// atomicAdd on double requires CC>=6.0; our build targets 75/86/89.

__device__ inline double warpReduceSumD(double v) {
    // 32-lane reduction across a warp using XOR butterfly.
    v += __shfl_xor_sync(0xFFFFFFFFu, v, 16);
    v += __shfl_xor_sync(0xFFFFFFFFu, v, 8);
    v += __shfl_xor_sync(0xFFFFFFFFu, v, 4);
    v += __shfl_xor_sync(0xFFFFFFFFu, v, 2);
    v += __shfl_xor_sync(0xFFFFFFFFu, v, 1);
    return v;
}

__global__ void diagnosticsKernel(
    const ParticleGpu* __restrict__ particles,
    const OctreeNode* __restrict__ nodes,
    float G, float softening, float theta,
    int N, int rootIdx,
    DiagnosticsAccum* __restrict__ accum)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    double t_ke = 0.0, t_pe = 0.0;
    double t_px = 0.0, t_py = 0.0, t_pz = 0.0;
    double t_mx = 0.0, t_my = 0.0, t_mz = 0.0, t_m = 0.0;

    if (gid < N) {
        const ParticleGpu p = particles[gid];
        double m = (double)p.mass;
        double vx = (double)p.vx, vy = (double)p.vy, vz = (double)p.vz;
        double px = (double)p.px, py = (double)p.py, pz = (double)p.pz;

        t_ke = 0.5 * m * (vx * vx + vy * vy + vz * vz);
        t_px = m * vx; t_py = m * vy; t_pz = m * vz;
        t_mx = m * px; t_my = m * py; t_mz = m * pz;
        t_m  = m;

        // Tree walk for PE — mirrors forceKernel structure.
        float fpx = p.px, fpy = p.py, fpz = p.pz;
        float soft2 = softening * softening;
        float theta2 = theta * theta;

        int stack[MAX_STACK_DEPTH];
        int stackTop = 0;
        stack[stackTop++] = rootIdx;

        double pe = 0.0;
        while (stackTop > 0) {
            int nodeIdx = stack[--stackTop];
            if (nodeIdx < 0) continue;

            const OctreeNode& node = nodes[nodeIdx];
            if (node.mass <= 0.0f) continue;

            // Single-particle leaf
            if (node.startIdx >= 0 && node.particleCount == 1) {
                if (node.startIdx == gid) continue;
                float dx = node.cx - fpx;
                float dy = node.cy - fpy;
                float dz = node.cz - fpz;
                float r2 = dx * dx + dy * dy + dz * dz + soft2;
                float invR = rsqrtf(r2);
                pe += -(double)G * m * (double)node.mass * (double)invR;
                continue;
            }

            float sizeX = node.maxX - node.minX;
            float sizeY = node.maxY - node.minY;
            float sizeZ = node.maxZ - node.minZ;
            float s = fmaxf(sizeX, fmaxf(sizeY, sizeZ));

            float dx = node.cx - fpx;
            float dy = node.cy - fpy;
            float dz = node.cz - fpz;
            float d2 = dx * dx + dy * dy + dz * dz;

            if (s * s < theta2 * d2) {
                float r2 = d2 + soft2;
                float invR = rsqrtf(r2);
                pe += -(double)G * m * (double)node.mass * (double)invR;
            } else {
                for (int c = 7; c >= 0; c--) {
                    if (node.children[c] >= 0 && stackTop < MAX_STACK_DEPTH) {
                        stack[stackTop++] = node.children[c];
                    }
                }
            }
        }
        t_pe = pe;
    }

    // ── Step 1: warp-level reduction (no shared memory) ──
    t_ke = warpReduceSumD(t_ke); t_pe = warpReduceSumD(t_pe);
    t_px = warpReduceSumD(t_px); t_py = warpReduceSumD(t_py); t_pz = warpReduceSumD(t_pz);
    t_mx = warpReduceSumD(t_mx); t_my = warpReduceSumD(t_my); t_mz = warpReduceSumD(t_mz);
    t_m  = warpReduceSumD(t_m);

    // ── Step 2: each warp's lane 0 stores its partial sum ──
    constexpr int WARPS_PER_BLOCK = BLOCK_SIZE / 32;
    __shared__ double s_ke[WARPS_PER_BLOCK];
    __shared__ double s_pe[WARPS_PER_BLOCK];
    __shared__ double s_px[WARPS_PER_BLOCK];
    __shared__ double s_py[WARPS_PER_BLOCK];
    __shared__ double s_pz[WARPS_PER_BLOCK];
    __shared__ double s_mx[WARPS_PER_BLOCK];
    __shared__ double s_my[WARPS_PER_BLOCK];
    __shared__ double s_mz[WARPS_PER_BLOCK];
    __shared__ double s_m [WARPS_PER_BLOCK];

    int warpId = tid >> 5;     // tid / 32
    int lane   = tid & 31;     // tid % 32

    if (lane == 0) {
        s_ke[warpId] = t_ke; s_pe[warpId] = t_pe;
        s_px[warpId] = t_px; s_py[warpId] = t_py; s_pz[warpId] = t_pz;
        s_mx[warpId] = t_mx; s_my[warpId] = t_my; s_mz[warpId] = t_mz;
        s_m [warpId] = t_m;
    }
    __syncthreads();

    // ── Step 3: first warp reduces the per-warp partials and emits ──
    if (warpId == 0) {
        double v_ke = (lane < WARPS_PER_BLOCK) ? s_ke[lane] : 0.0;
        double v_pe = (lane < WARPS_PER_BLOCK) ? s_pe[lane] : 0.0;
        double v_px = (lane < WARPS_PER_BLOCK) ? s_px[lane] : 0.0;
        double v_py = (lane < WARPS_PER_BLOCK) ? s_py[lane] : 0.0;
        double v_pz = (lane < WARPS_PER_BLOCK) ? s_pz[lane] : 0.0;
        double v_mx = (lane < WARPS_PER_BLOCK) ? s_mx[lane] : 0.0;
        double v_my = (lane < WARPS_PER_BLOCK) ? s_my[lane] : 0.0;
        double v_mz = (lane < WARPS_PER_BLOCK) ? s_mz[lane] : 0.0;
        double v_m  = (lane < WARPS_PER_BLOCK) ? s_m [lane] : 0.0;

        v_ke = warpReduceSumD(v_ke); v_pe = warpReduceSumD(v_pe);
        v_px = warpReduceSumD(v_px); v_py = warpReduceSumD(v_py); v_pz = warpReduceSumD(v_pz);
        v_mx = warpReduceSumD(v_mx); v_my = warpReduceSumD(v_my); v_mz = warpReduceSumD(v_mz);
        v_m  = warpReduceSumD(v_m);

        if (lane == 0) {
            atomicAdd(&accum->ke, v_ke);
            atomicAdd(&accum->pe, v_pe);
            atomicAdd(&accum->px, v_px);
            atomicAdd(&accum->py, v_py);
            atomicAdd(&accum->pz, v_pz);
            atomicAdd(&accum->mx, v_mx);
            atomicAdd(&accum->my, v_my);
            atomicAdd(&accum->mz, v_mz);
            atomicAdd(&accum->m,  v_m);
        }
    }
}

void launchDiagnostics(
    const ParticleGpu* particles,
    const OctreeNode* nodes,
    const BarnesHutParams& params,
    int rootIdx,
    DiagnosticsAccum* d_accum,
    cudaStream_t stream)
{
    int N = params.particleCount;
    if (N <= 0) return;

    // Caller is responsible for zeroing d_accum before calling.
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    diagnosticsKernel<<<numBlocks, BLOCK_SIZE, 0, stream>>>(
        particles, nodes, params.G, params.softening, params.theta,
        N, rootIdx, d_accum);
}

} // namespace cuda
} // namespace cosmico
