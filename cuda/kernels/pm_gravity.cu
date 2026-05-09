#include "kernels/pm_gravity.cuh"
#include <cstdio>
#include <cfloat>

namespace cosmico {
namespace cuda {

static constexpr int PM_BLOCK = 256;
static constexpr float PI = 3.14159265358979323846f;

// ─── 1. Clear grid ──────────────────────────────────────────────────

__global__ void pmClearGridKernel(float* __restrict__ density, int N3) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N3) density[idx] = 0.0f;
}

void launchPMClearGrid(float* density, int N3, cudaStream_t stream) {
    int blocks = (N3 + PM_BLOCK - 1) / PM_BLOCK;
    pmClearGridKernel<<<blocks, PM_BLOCK, 0, stream>>>(density, N3);
}

// ─── 2. CIC deposit ─────────────────────────────────────────────────

__global__ void pmDepositKernel(const ParticleGpu* __restrict__ particles,
                                 float* __restrict__ density,
                                 int N, int particleCount, float boxSize,
                                 bool isolated) {
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= particleCount) return;
    if (particles[pid].type < 0.0f) return;  // Skip absorbed particles

    float invCell = (float)N / boxSize;
    // Cell volume: (boxSize/N)^3 — divide to convert mass → density
    float cellVol = (boxSize / (float)N);
    cellVol = cellVol * cellVol * cellVol;
    float invCellVol = 1.0f / cellVol;

    float x = particles[pid].px;
    float y = particles[pid].py;
    float z = particles[pid].pz;
    float mass = particles[pid].mass;

    if (isolated) {
        // Offset from origin-centered [-boxSize/2, boxSize/2) to [0, boxSize)
        float half = boxSize * 0.5f;
        x += half;
        y += half;
        z += half;
        // Skip particles outside the grid domain
        if (x < 0.0f || x >= boxSize || y < 0.0f || y >= boxSize ||
            z < 0.0f || z >= boxSize) return;
    } else {
        // Wrap to [0, boxSize)
        x = x - floorf(x / boxSize) * boxSize;
        y = y - floorf(y / boxSize) * boxSize;
        z = z - floorf(z / boxSize) * boxSize;
    }

    // Grid coordinates
    float gx = x * invCell;
    float gy = y * invCell;
    float gz = z * invCell;

    // CIC: base cell and fractional offsets
    int ix0 = (int)floorf(gx - 0.5f);
    int iy0 = (int)floorf(gy - 0.5f);
    int iz0 = (int)floorf(gz - 0.5f);

    float wx = gx - 0.5f - (float)ix0;
    float wy = gy - 0.5f - (float)iy0;
    float wz = gz - 0.5f - (float)iz0;

    // Deposit to 8 surrounding cells (mass/cellVol = density)
    for (int di = 0; di < 2; di++) {
        float fx = (di == 0) ? (1.0f - wx) : wx;
        int cix = ix0 + di;
        if (isolated) {
            if (cix < 0 || cix >= N) continue;  // Clamp: skip out-of-bounds
        } else {
            cix = ((cix % N) + N) % N;          // Periodic wrap
        }
        for (int dj = 0; dj < 2; dj++) {
            float fy = (dj == 0) ? (1.0f - wy) : wy;
            int ciy = iy0 + dj;
            if (isolated) {
                if (ciy < 0 || ciy >= N) continue;
            } else {
                ciy = ((ciy % N) + N) % N;
            }
            for (int dk = 0; dk < 2; dk++) {
                float fz = (dk == 0) ? (1.0f - wz) : wz;
                int ciz = iz0 + dk;
                if (isolated) {
                    if (ciz < 0 || ciz >= N) continue;
                } else {
                    ciz = ((ciz % N) + N) % N;
                }

                float weight = mass * invCellVol * fx * fy * fz;
                int cellIdx = cix * N * N + ciy * N + ciz;
                atomicAdd(&density[cellIdx], weight);
            }
        }
    }
}

void launchPMDeposit(const ParticleGpu* particles, float* density,
                     int particleCount, int gridN, float boxSize,
                     bool isolated, cudaStream_t stream) {
    int blocks = (particleCount + PM_BLOCK - 1) / PM_BLOCK;
    pmDepositKernel<<<blocks, PM_BLOCK, 0, stream>>>(
        particles, density, gridN, particleCount, boxSize, isolated);
}

// ─── 3. Subtract mean density ───────────────────────────────────────

__global__ void pmSubtractMeanKernel(float* __restrict__ density,
                                      int N3, float meanDensity) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N3) {
        density[idx] -= meanDensity;
    }
}

void launchPMSubtractMean(float* density, int N3, float meanDensity,
                          cudaStream_t stream) {
    int blocks = (N3 + PM_BLOCK - 1) / PM_BLOCK;
    pmSubtractMeanKernel<<<blocks, PM_BLOCK, 0, stream>>>(density, N3, meanDensity);
}

// ─── 4. Green's function multiply ───────────────────────────────────
// Discrete Green's function: G(k) = -4πG / k_eff²
// k_eff² = (2/h)² * [sin²(πnx/N) + sin²(πny/N) + sin²(πnz/N)]
// Normalization: divide by N³ for FFT convention

__global__ void pmGreenMultiplyKernel(cufftComplex* __restrict__ rho_hat,
                                       int N, int Ncomplex,
                                       float G_const, float cellSize,
                                       float smoothRadius) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= Ncomplex) return;

    int Nh = N / 2 + 1;
    int iz = idx % Nh;
    int iy = (idx / Nh) % N;
    int ix = idx / (Nh * N);

    // DC mode: zero potential (no self-gravity of mean field)
    if (ix == 0 && iy == 0 && iz == 0) {
        rho_hat[idx].x = 0.0f;
        rho_hat[idx].y = 0.0f;
        return;
    }

    // Discrete k_eff² using sin² for alias-free Green's function
    float invN = 1.0f / (float)N;
    float sx = sinf(PI * (float)ix * invN);
    float sy = sinf(PI * (float)iy * invN);
    float sz = sinf(PI * (float)iz * invN);
    float keff2 = (2.0f / cellSize) * (2.0f / cellSize) *
                  (sx * sx + sy * sy + sz * sz);

    // Φ̂ = -4πG · ρ̂ / (k_eff² · N³)
    float N3 = (float)(N * N * N);
    float factor = -4.0f * PI * G_const / (keff2 * N3);

    // Gaussian smoothing: suppress high-k modes (shot noise damping)
    // exp(-k_eff² · σ² / 2) — σ = smoothRadius in code units
    float smooth = expf(-0.5f * keff2 * smoothRadius * smoothRadius);
    factor *= smooth;

    rho_hat[idx].x *= factor;
    rho_hat[idx].y *= factor;
}

void launchPMGreenMultiply(cufftComplex* density_hat, int gridN, int Ncomplex,
                           float G, float cellSize, float smoothRadius,
                           cudaStream_t stream) {
    int blocks = (Ncomplex + PM_BLOCK - 1) / PM_BLOCK;
    pmGreenMultiplyKernel<<<blocks, PM_BLOCK, 0, stream>>>(
        density_hat, gridN, Ncomplex, G, cellSize, smoothRadius);
}

// ─── 5. Gradient (central finite differences) ───────────────────────

__global__ void pmGradientKernel(const float* __restrict__ potential,
                                  float* __restrict__ forceX,
                                  float* __restrict__ forceY,
                                  float* __restrict__ forceZ,
                                  int N, float inv2h, float invh,
                                  bool isolated) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N3 = N * N * N;
    if (idx >= N3) return;

    int iz = idx % N;
    int iy = (idx / N) % N;
    int ix = idx / (N * N);

    if (isolated) {
        // One-sided differences at boundaries, central elsewhere
        // X direction
        if (ix == 0) {
            forceX[idx] = -(potential[1 * N * N + iy * N + iz] -
                            potential[0 * N * N + iy * N + iz]) * invh;
        } else if (ix == N - 1) {
            forceX[idx] = -(potential[(N-1) * N * N + iy * N + iz] -
                            potential[(N-2) * N * N + iy * N + iz]) * invh;
        } else {
            forceX[idx] = -(potential[(ix+1) * N * N + iy * N + iz] -
                            potential[(ix-1) * N * N + iy * N + iz]) * inv2h;
        }
        // Y direction
        if (iy == 0) {
            forceY[idx] = -(potential[ix * N * N + 1 * N + iz] -
                            potential[ix * N * N + 0 * N + iz]) * invh;
        } else if (iy == N - 1) {
            forceY[idx] = -(potential[ix * N * N + (N-1) * N + iz] -
                            potential[ix * N * N + (N-2) * N + iz]) * invh;
        } else {
            forceY[idx] = -(potential[ix * N * N + (iy+1) * N + iz] -
                            potential[ix * N * N + (iy-1) * N + iz]) * inv2h;
        }
        // Z direction
        if (iz == 0) {
            forceZ[idx] = -(potential[ix * N * N + iy * N + 1] -
                            potential[ix * N * N + iy * N + 0]) * invh;
        } else if (iz == N - 1) {
            forceZ[idx] = -(potential[ix * N * N + iy * N + (N-1)] -
                            potential[ix * N * N + iy * N + (N-2)]) * invh;
        } else {
            forceZ[idx] = -(potential[ix * N * N + iy * N + (iz+1)] -
                            potential[ix * N * N + iy * N + (iz-1)]) * inv2h;
        }
    } else {
        // Periodic neighbor indices
        int ixp = (ix + 1) % N;
        int ixm = (ix - 1 + N) % N;
        int iyp = (iy + 1) % N;
        int iym = (iy - 1 + N) % N;
        int izp = (iz + 1) % N;
        int izm = (iz - 1 + N) % N;

        // Force = -∇Φ using central differences: F = -(Φ[+1] - Φ[-1]) / (2h)
        forceX[idx] = -(potential[ixp * N * N + iy * N + iz] -
                        potential[ixm * N * N + iy * N + iz]) * inv2h;
        forceY[idx] = -(potential[ix * N * N + iyp * N + iz] -
                        potential[ix * N * N + iym * N + iz]) * inv2h;
        forceZ[idx] = -(potential[ix * N * N + iy * N + izp] -
                        potential[ix * N * N + iy * N + izm]) * inv2h;
    }
}

void launchPMGradient(const float* potential, float* forceX, float* forceY,
                      float* forceZ, int gridN, float cellSize,
                      bool isolated, cudaStream_t stream) {
    int N3 = gridN * gridN * gridN;
    int blocks = (N3 + PM_BLOCK - 1) / PM_BLOCK;
    float inv2h = 1.0f / (2.0f * cellSize);
    float invh = 1.0f / cellSize;
    pmGradientKernel<<<blocks, PM_BLOCK, 0, stream>>>(
        potential, forceX, forceY, forceZ, gridN, inv2h, invh, isolated);
}

// ─── 6. CIC interpolation (adjoint of deposit) ─────────────────────
// Interpolates force field values to particle positions using CIC weights

__global__ void pmInterpolateKernel(ParticleGpu* __restrict__ particles,
                                     const float* __restrict__ forceX,
                                     const float* __restrict__ forceY,
                                     const float* __restrict__ forceZ,
                                     int N, int particleCount, float boxSize,
                                     bool isolated) {
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= particleCount) return;
    if (particles[pid].type < 0.0f) return;  // Skip absorbed particles

    float invCell = (float)N / boxSize;

    float x = particles[pid].px;
    float y = particles[pid].py;
    float z = particles[pid].pz;

    if (isolated) {
        // Offset from origin-centered to [0, boxSize)
        float half = boxSize * 0.5f;
        x += half;
        y += half;
        z += half;
        // Out-of-bounds particles: zero acceleration
        if (x < 0.0f || x >= boxSize || y < 0.0f || y >= boxSize ||
            z < 0.0f || z >= boxSize) {
            particles[pid].density = 0.0f;
            particles[pid].pressure = 0.0f;
            particles[pid].smoothing = 0.0f;
            return;
        }
    } else {
        // Wrap to [0, boxSize)
        x = x - floorf(x / boxSize) * boxSize;
        y = y - floorf(y / boxSize) * boxSize;
        z = z - floorf(z / boxSize) * boxSize;
    }

    // Grid coordinates
    float gx = x * invCell;
    float gy = y * invCell;
    float gz = z * invCell;

    // CIC base cell
    int ix0 = (int)floorf(gx - 0.5f);
    int iy0 = (int)floorf(gy - 0.5f);
    int iz0 = (int)floorf(gz - 0.5f);

    float wx = gx - 0.5f - (float)ix0;
    float wy = gy - 0.5f - (float)iy0;
    float wz = gz - 0.5f - (float)iz0;

    // Interpolate from 8 surrounding cells (adjoint of CIC deposit)
    float ax = 0.0f, ay = 0.0f, az = 0.0f;

    for (int di = 0; di < 2; di++) {
        float fx = (di == 0) ? (1.0f - wx) : wx;
        int cix = ix0 + di;
        if (isolated) {
            if (cix < 0 || cix >= N) continue;
        } else {
            cix = ((cix % N) + N) % N;
        }
        for (int dj = 0; dj < 2; dj++) {
            float fy = (dj == 0) ? (1.0f - wy) : wy;
            int ciy = iy0 + dj;
            if (isolated) {
                if (ciy < 0 || ciy >= N) continue;
            } else {
                ciy = ((ciy % N) + N) % N;
            }
            for (int dk = 0; dk < 2; dk++) {
                float fz = (dk == 0) ? (1.0f - wz) : wz;
                int ciz = iz0 + dk;
                if (isolated) {
                    if (ciz < 0 || ciz >= N) continue;
                } else {
                    ciz = ((ciz % N) + N) % N;
                }

                float w = fx * fy * fz;
                int cellIdx = cix * N * N + ciy * N + ciz;

                ax += forceX[cellIdx] * w;
                ay += forceY[cellIdx] * w;
                az += forceZ[cellIdx] * w;
            }
        }
    }

    // Store acceleration in scratch fields (density, pressure, smoothing)
    particles[pid].density = ax;
    particles[pid].pressure = ay;
    particles[pid].smoothing = az;
}

void launchPMInterpolate(ParticleGpu* particles, const float* forceX,
                         const float* forceY, const float* forceZ,
                         int particleCount, int gridN, float boxSize,
                         bool isolated, cudaStream_t stream) {
    int blocks = (particleCount + PM_BLOCK - 1) / PM_BLOCK;
    pmInterpolateKernel<<<blocks, PM_BLOCK, 0, stream>>>(
        particles, forceX, forceY, forceZ, gridN, particleCount, boxSize, isolated);
}

// ─── 7. Kick: v += a * dt/2 ─────────────────────────────────────────

__global__ void pmKickKernel(ParticleGpu* __restrict__ particles,
                              int particleCount, float kickDt,
                              float dampFactor,
                              bool comoving, float H,
                              float maxVel) {
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= particleCount) return;
    if (particles[pid].type < 0.0f) return;  // Skip absorbed particles

    float ax = particles[pid].density;
    float ay = particles[pid].pressure;
    float az = particles[pid].smoothing;

    // Apply velocity damping: v *= exp(-γ·dt) ≈ dampFactor
    float vx = particles[pid].vx * dampFactor;
    float vy = particles[pid].vy * dampFactor;
    float vz = particles[pid].vz * dampFactor;

    if (comoving) {
        float drag = 1.0f - H * kickDt;
        vx *= drag;
        vy *= drag;
        vz *= drag;
    }

    vx = vx + ax * kickDt;
    vy = vy + ay * kickDt;
    vz = vz + az * kickDt;

    // Velocity cap: prevent NaN/Inf and runaway
    float v2 = vx * vx + vy * vy + vz * vz;
    if (v2 > maxVel * maxVel) {
        float scale = maxVel / sqrtf(v2);
        vx *= scale;
        vy *= scale;
        vz *= scale;
    }

    particles[pid].vx = vx;
    particles[pid].vy = vy;
    particles[pid].vz = vz;
}

void launchPMKick(ParticleGpu* particles, int particleCount,
                  float kickDt, float dampFactor,
                  bool comoving, float H,
                  float maxVel, cudaStream_t stream) {
    int blocks = (particleCount + PM_BLOCK - 1) / PM_BLOCK;
    pmKickKernel<<<blocks, PM_BLOCK, 0, stream>>>(
        particles, particleCount, kickDt, dampFactor, comoving, H, maxVel);
}

// ─── 8. Drift: x += v * dt ──────────────────────────────────────────

__global__ void pmDriftKernel(ParticleGpu* __restrict__ particles,
                               int particleCount, float dt, float boxSize) {
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= particleCount) return;
    if (particles[pid].type < 0.0f) return;  // Skip absorbed particles

    float vx = particles[pid].vx;
    float vy = particles[pid].vy;
    float vz = particles[pid].vz;

    // NaN/Inf guard: reset corrupted particles to origin with zero velocity
    if (!isfinite(vx) || !isfinite(vy) || !isfinite(vz)) {
        particles[pid].vx = 0.0f;
        particles[pid].vy = 0.0f;
        particles[pid].vz = 0.0f;
        return;
    }

    float x = particles[pid].px + vx * dt;
    float y = particles[pid].py + vy * dt;
    float z = particles[pid].pz + vz * dt;

    // Periodic wrap to [-boxSize/2, boxSize/2) — centered at origin for camera
    float half = boxSize * 0.5f;
    x = x - floorf((x + half) / boxSize) * boxSize;
    y = y - floorf((y + half) / boxSize) * boxSize;
    z = z - floorf((z + half) / boxSize) * boxSize;

    particles[pid].px = x;
    particles[pid].py = y;
    particles[pid].pz = z;
}

void launchPMDrift(ParticleGpu* particles, int particleCount,
                   float dt, float boxSize, cudaStream_t stream) {
    int blocks = (particleCount + PM_BLOCK - 1) / PM_BLOCK;
    pmDriftKernel<<<blocks, PM_BLOCK, 0, stream>>>(
        particles, particleCount, dt, boxSize);
}

// ─── 9. Energy + momentum reduction ─────────────────────────────────
// energySums[0] = KE, [1] = PE, [2] = px, [3] = py, [4] = pz, [5] = totalMass

__global__ void pmEnergyReductionKernel(const ParticleGpu* __restrict__ particles,
                                         const float* __restrict__ potential,
                                         int particleCount, int gridN,
                                         float boxSize,
                                         double* __restrict__ energySums) {
    __shared__ double sKE[PM_BLOCK];
    __shared__ double sPE[PM_BLOCK];
    __shared__ double sPx[PM_BLOCK];
    __shared__ double sPy[PM_BLOCK];
    __shared__ double sPz[PM_BLOCK];
    __shared__ double sMass[PM_BLOCK];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    double lKE = 0.0, lPE = 0.0, lPx = 0.0, lPy = 0.0, lPz = 0.0, lMass = 0.0;

    for (int i = gid; i < particleCount; i += blockDim.x * gridDim.x) {
        if (particles[i].type < 0.0f) continue;  // Skip absorbed particles
        float m = particles[i].mass;
        float vx = particles[i].vx;
        float vy = particles[i].vy;
        float vz = particles[i].vz;

        lKE += 0.5 * m * (vx * vx + vy * vy + vz * vz);

        // Approximate PE by interpolating potential at particle position (NGP for speed)
        float invCell = (float)gridN / boxSize;
        float x = particles[i].px - floorf(particles[i].px / boxSize) * boxSize;
        float y = particles[i].py - floorf(particles[i].py / boxSize) * boxSize;
        float z = particles[i].pz - floorf(particles[i].pz / boxSize) * boxSize;
        int ix = (int)(x * invCell) % gridN;
        int iy = (int)(y * invCell) % gridN;
        int iz = (int)(z * invCell) % gridN;
        float phi = potential[ix * gridN * gridN + iy * gridN + iz];
        lPE += 0.5 * m * phi;

        lPx += m * vx;
        lPy += m * vy;
        lPz += m * vz;
        lMass += m;
    }

    sKE[tid] = lKE; sPE[tid] = lPE;
    sPx[tid] = lPx; sPy[tid] = lPy; sPz[tid] = lPz;
    sMass[tid] = lMass;
    __syncthreads();

    // Warp reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sKE[tid] += sKE[tid + s];
            sPE[tid] += sPE[tid + s];
            sPx[tid] += sPx[tid + s];
            sPy[tid] += sPy[tid + s];
            sPz[tid] += sPz[tid + s];
            sMass[tid] += sMass[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(&energySums[0], sKE[0]);
        atomicAdd(&energySums[1], sPE[0]);
        atomicAdd(&energySums[2], sPx[0]);
        atomicAdd(&energySums[3], sPy[0]);
        atomicAdd(&energySums[4], sPz[0]);
        atomicAdd(&energySums[5], sMass[0]);
    }
}

void launchPMEnergyReduction(const ParticleGpu* particles,
                             const float* potential, int particleCount,
                             int gridN, float boxSize,
                             double* d_energySums, cudaStream_t stream) {
    // Clear sums first
    cudaMemsetAsync(d_energySums, 0, 6 * sizeof(double), stream);
    int blocks = (particleCount + PM_BLOCK - 1) / PM_BLOCK;
    if (blocks > 256) blocks = 256;  // Cap blocks for reduction
    pmEnergyReductionKernel<<<blocks, PM_BLOCK, 0, stream>>>(
        particles, potential, particleCount, gridN, boxSize, d_energySums);
}

// ─── 10. Momentum correction: v -= <v> ──────────────────────────────

__global__ void pmMomentumCorrectionKernel(ParticleGpu* __restrict__ particles,
                                            int particleCount,
                                            const double* __restrict__ energySums) {
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= particleCount) return;
    if (particles[pid].type < 0.0f) return;  // Skip absorbed particles

    double totalMass = energySums[5];
    if (totalMass < 1e-10) return;

    float dvx = (float)(energySums[2] / totalMass);
    float dvy = (float)(energySums[3] / totalMass);
    float dvz = (float)(energySums[4] / totalMass);

    particles[pid].vx -= dvx;
    particles[pid].vy -= dvy;
    particles[pid].vz -= dvz;
}

void launchPMMomentumCorrection(ParticleGpu* particles, int particleCount,
                                const double* d_energySums,
                                cudaStream_t stream) {
    int blocks = (particleCount + PM_BLOCK - 1) / PM_BLOCK;
    pmMomentumCorrectionKernel<<<blocks, PM_BLOCK, 0, stream>>>(
        particles, particleCount, d_energySums);
}

// ─── 11. Power spectrum ─────────────────────────────────────────────

__global__ void pmPowerSpectrumKernel(const cufftComplex* __restrict__ rho_hat,
                                       int N, int Ncomplex,
                                       float* __restrict__ spectrum,
                                       int* __restrict__ counts,
                                       int nBins, float dk) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= Ncomplex) return;

    int Nh = N / 2 + 1;
    int iz = idx % Nh;
    int iy = (idx / Nh) % N;
    int ix = idx / (Nh * N);

    // Frequency indices
    int nx = (ix <= N / 2) ? ix : ix - N;
    int ny = (iy <= N / 2) ? iy : iy - N;
    int nz = iz;

    float kmag = sqrtf((float)(nx * nx + ny * ny + nz * nz)) * dk;
    if (kmag < 1e-6f) return;

    int bin = (int)(kmag / dk);
    if (bin >= nBins) return;

    float re = rho_hat[idx].x;
    float im = rho_hat[idx].y;
    float power = re * re + im * im;

    // Account for Hermitian symmetry: modes with iz > 0 and iz < N/2 appear twice
    float weight = (iz > 0 && iz < N / 2) ? 2.0f : 1.0f;

    atomicAdd(&spectrum[bin], power * weight);
    atomicAdd(&counts[bin], 1);
}

void launchPMPowerSpectrum(const cufftComplex* density_hat, int gridN,
                           int Ncomplex, float* d_spectrum, int* d_counts,
                           int nBins, float dk, cudaStream_t stream) {
    cudaMemsetAsync(d_spectrum, 0, nBins * sizeof(float), stream);
    cudaMemsetAsync(d_counts, 0, nBins * sizeof(int), stream);
    int blocks = (Ncomplex + PM_BLOCK - 1) / PM_BLOCK;
    pmPowerSpectrumKernel<<<blocks, PM_BLOCK, 0, stream>>>(
        density_hat, gridN, Ncomplex, d_spectrum, d_counts, nBins, dk);
}

// ─── 12. Sink formation: create sinks in high-density cells ──────────
// Each normal particle checks its local density. If ρ > threshold × ρ_mean,
// and it is the most massive particle in its cell, it becomes a sink.
// Nearby particles within sinkRadius are absorbed (type = -1).

__global__ void pmSinkFormationKernel(ParticleGpu* __restrict__ particles,
                                       const float* __restrict__ density,
                                       int particleCount, int gridN,
                                       float boxSize, float threshold,
                                       float sinkRadius,
                                       int* __restrict__ d_sinkCount) {
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= particleCount) return;
    if (particles[pid].type != 0.0f) return;  // Only normal particles

    float invCell = (float)gridN / boxSize;

    float x = particles[pid].px;
    float y = particles[pid].py;
    float z = particles[pid].pz;

    // Wrap to [0, boxSize)
    x = x - floorf(x / boxSize) * boxSize;
    y = y - floorf(y / boxSize) * boxSize;
    z = z - floorf(z / boxSize) * boxSize;

    // NGP cell index
    int ix = (int)(x * invCell) % gridN;
    int iy = (int)(y * invCell) % gridN;
    int iz = (int)(z * invCell) % gridN;
    if (ix < 0) ix += gridN;
    if (iy < 0) iy += gridN;
    if (iz < 0) iz += gridN;

    float localDensity = density[ix * gridN * gridN + iy * gridN + iz];

    // Compute mean density
    float cellVol = boxSize / (float)gridN;
    cellVol = cellVol * cellVol * cellVol;
    float totalMass = (float)particleCount;  // Assume unit mass particles initially
    float meanDensity = totalMass / (boxSize * boxSize * boxSize) * cellVol;

    // Check threshold
    if (localDensity < threshold * meanDensity) return;

    // Use atomicCAS on type to claim this particle as a new sink
    // Only the first particle in a high-density region becomes a sink
    float oldType = atomicExch(&particles[pid].type, 1.0f);
    if (oldType != 0.0f) return;  // Someone else got here first

    atomicAdd(d_sinkCount, 1);

    // Now absorb nearby normal particles
    float r2max = sinkRadius * sinkRadius;

    for (int j = 0; j < particleCount; j++) {
        if (j == pid) continue;
        if (particles[j].type != 0.0f) continue;  // Only absorb normal particles

        float dx = particles[j].px - particles[pid].px;
        float dy = particles[j].py - particles[pid].py;
        float dz = particles[j].pz - particles[pid].pz;

        // Periodic wrap
        float half = boxSize * 0.5f;
        if (dx > half) dx -= boxSize;
        if (dx < -half) dx += boxSize;
        if (dy > half) dy -= boxSize;
        if (dy < -half) dy += boxSize;
        if (dz > half) dz -= boxSize;
        if (dz < -half) dz += boxSize;

        float r2 = dx * dx + dy * dy + dz * dz;
        if (r2 > r2max) continue;

        // Try to absorb this particle
        float neighborType = atomicExch(&particles[j].type, -1.0f);
        if (neighborType != 0.0f) continue;  // Already absorbed or is a sink

        // Merge: conserve mass and momentum
        float mSink = particles[pid].mass;
        float mNeighbor = particles[j].mass;
        float mTotal = mSink + mNeighbor;

        if (mTotal > 0.0f) {
            float invTotal = 1.0f / mTotal;
            particles[pid].vx = (mSink * particles[pid].vx + mNeighbor * particles[j].vx) * invTotal;
            particles[pid].vy = (mSink * particles[pid].vy + mNeighbor * particles[j].vy) * invTotal;
            particles[pid].vz = (mSink * particles[pid].vz + mNeighbor * particles[j].vz) * invTotal;
        }

        particles[pid].mass = mTotal;
        // Update sink type to reflect accreted mass (visual indicator)
        particles[pid].type = fmaxf(1.0f, mTotal);
    }
}

void launchPMSinkFormation(ParticleGpu* particles, const float* density,
                           int particleCount, int gridN, float boxSize,
                           float threshold, float sinkRadius,
                           int* d_sinkCount, int* d_newParticleCount,
                           cudaStream_t stream) {
    cudaMemsetAsync(d_sinkCount, 0, sizeof(int), stream);
    int blocks = (particleCount + PM_BLOCK - 1) / PM_BLOCK;
    pmSinkFormationKernel<<<blocks, PM_BLOCK, 0, stream>>>(
        particles, density, particleCount, gridN, boxSize,
        threshold, sinkRadius, d_sinkCount);
}

// ─── 13. Sink accretion: absorb particles near existing sinks ────────
// Each normal particle checks distance to all sinks. If within sinkRadius,
// it gets absorbed. Momentum is transferred via atomic operations on
// dedicated momentum accumulator fields (density/pressure/smoothing reused).

// Pass 1: Prepare sinks — store current momentum in scratch fields
__global__ void pmSinkPrepareKernel(ParticleGpu* __restrict__ particles,
                                     int particleCount) {
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= particleCount) return;
    if (particles[pid].type <= 0.0f) return;  // Only sinks

    // Store momentum = mass * velocity in scratch fields
    float m = particles[pid].mass;
    particles[pid].density  = m * particles[pid].vx;
    particles[pid].pressure = m * particles[pid].vy;
    particles[pid].smoothing = m * particles[pid].vz;
}

// Pass 2: Each normal particle finds nearest sink within radius and gets absorbed
__global__ void pmSinkAccretionKernel(ParticleGpu* __restrict__ particles,
                                       int particleCount, float sinkRadius,
                                       float boxSize) {
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= particleCount) return;
    if (particles[pid].type != 0.0f) return;  // Only normal particles

    float r2max = sinkRadius * sinkRadius;
    float half = boxSize * 0.5f;

    // Find nearest sink
    int nearestSink = -1;
    float nearestR2 = r2max;

    for (int j = 0; j < particleCount; j++) {
        if (particles[j].type <= 0.0f) continue;  // Only sinks (type > 0)

        float dx = particles[pid].px - particles[j].px;
        float dy = particles[pid].py - particles[j].py;
        float dz = particles[pid].pz - particles[j].pz;

        // Periodic wrap
        if (dx > half) dx -= boxSize;
        if (dx < -half) dx += boxSize;
        if (dy > half) dy -= boxSize;
        if (dy < -half) dy += boxSize;
        if (dz > half) dz -= boxSize;
        if (dz < -half) dz += boxSize;

        float r2 = dx * dx + dy * dy + dz * dz;
        if (r2 < nearestR2) {
            nearestR2 = r2;
            nearestSink = j;
        }
    }

    if (nearestSink < 0) return;

    // Try to mark as absorbed
    float oldType = atomicExch(&particles[pid].type, -1.0f);
    if (oldType != 0.0f) return;  // Race: someone else absorbed us

    // Transfer mass and momentum to sink using atomics on scratch fields
    float m = particles[pid].mass;
    atomicAdd(&particles[nearestSink].mass, m);
    atomicAdd(&particles[nearestSink].density,  m * particles[pid].vx);  // px momentum
    atomicAdd(&particles[nearestSink].pressure, m * particles[pid].vy);  // py momentum
    atomicAdd(&particles[nearestSink].smoothing, m * particles[pid].vz); // pz momentum
}

// Pass 3: Convert accumulated momentum back to velocity for sinks
__global__ void pmSinkFinalizeKernel(ParticleGpu* __restrict__ particles,
                                      int particleCount) {
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= particleCount) return;
    if (particles[pid].type <= 0.0f) return;  // Only sinks

    float m = particles[pid].mass;
    if (m < 1e-10f) return;

    float invM = 1.0f / m;
    particles[pid].vx = particles[pid].density  * invM;
    particles[pid].vy = particles[pid].pressure * invM;
    particles[pid].vz = particles[pid].smoothing * invM;

    // Update type to reflect total accreted mass (visual size indicator)
    particles[pid].type = fmaxf(1.0f, m);
}

// Count sinks and absorbed particles
__global__ void pmSinkCountKernel(const ParticleGpu* __restrict__ particles,
                                   int particleCount,
                                   int* __restrict__ d_sinkCount,
                                   int* __restrict__ d_absorbedCount) {
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= particleCount) return;

    if (particles[pid].type > 0.0f) {
        atomicAdd(d_sinkCount, 1);
    } else if (particles[pid].type < 0.0f) {
        atomicAdd(d_absorbedCount, 1);
    }
}

void launchPMSinkAccretion(ParticleGpu* particles, int particleCount,
                           float sinkRadius, float boxSize,
                           int* d_sinkCount, int* d_newParticleCount,
                           cudaStream_t stream) {
    int blocks = (particleCount + PM_BLOCK - 1) / PM_BLOCK;

    // Pass 1: Store current sink momentum in scratch fields
    pmSinkPrepareKernel<<<blocks, PM_BLOCK, 0, stream>>>(
        particles, particleCount);

    // Pass 2: Accrete normal particles into nearest sinks
    pmSinkAccretionKernel<<<blocks, PM_BLOCK, 0, stream>>>(
        particles, particleCount, sinkRadius, boxSize);

    // Pass 3: Convert accumulated momentum back to velocity
    pmSinkFinalizeKernel<<<blocks, PM_BLOCK, 0, stream>>>(
        particles, particleCount);

    // Count sinks and absorbed
    cudaMemsetAsync(d_sinkCount, 0, sizeof(int), stream);
    cudaMemsetAsync(d_newParticleCount, 0, sizeof(int), stream);
    pmSinkCountKernel<<<blocks, PM_BLOCK, 0, stream>>>(
        particles, particleCount, d_sinkCount, d_newParticleCount);
}

// ─── 14. Compute free-space Green's function on (2N)³ grid ────────────
// G(r) = -1/(4π|r|), DC=0, normalized by cell volume and (2N)³ for FFT

__global__ void pmComputeGreenFunctionKernel(float* __restrict__ green,
                                              int N, float cellSize,
                                              float G_const) {
    int N2 = 2 * N;
    int N2_3 = N2 * N2 * N2;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N2_3) return;

    int iz = idx % N2;
    int iy = (idx / N2) % N2;
    int ix = idx / (N2 * N2);

    // Wrap distances so that index N maps to -N (shortest image on 2N grid)
    int dx = (ix <= N) ? ix : ix - N2;
    int dy = (iy <= N) ? iy : iy - N2;
    int dz = (iz <= N) ? iz : iz - N2;

    // DC component: zero
    if (dx == 0 && dy == 0 && dz == 0) {
        green[idx] = 0.0f;
        return;
    }

    float r = cellSize * sqrtf((float)(dx * dx + dy * dy + dz * dz));
    // G(r) = -G / (4π r), then divide by (2N)³ for FFT normalization
    float N2_3f = (float)N2_3;
    green[idx] = -G_const / (4.0f * PI * r * N2_3f);
}

void launchPMComputeGreenFunction(float* green_out, int N, float cellSize,
                                   float G_const, cudaStream_t stream) {
    int N2 = 2 * N;
    int N2_3 = N2 * N2 * N2;
    int blocks = (N2_3 + PM_BLOCK - 1) / PM_BLOCK;
    pmComputeGreenFunctionKernel<<<blocks, PM_BLOCK, 0, stream>>>(
        green_out, N, cellSize, G_const);
}

// ─── 4b. Isolated Green's multiply: element-wise ρ̂ × Ĝ ───────────────

__global__ void pmGreenMultiplyIsolatedKernel(cufftComplex* __restrict__ rho_hat,
                                               const cufftComplex* __restrict__ green_hat,
                                               int Ncomplex_padded) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= Ncomplex_padded) return;

    float a = rho_hat[idx].x;
    float b = rho_hat[idx].y;
    float c = green_hat[idx].x;
    float d = green_hat[idx].y;

    // Complex multiply: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
    rho_hat[idx].x = a * c - b * d;
    rho_hat[idx].y = a * d + b * c;
}

void launchPMGreenMultiplyIsolated(cufftComplex* rho_hat, const cufftComplex* green_hat,
                                    int Ncomplex_padded, cudaStream_t stream) {
    int blocks = (Ncomplex_padded + PM_BLOCK - 1) / PM_BLOCK;
    pmGreenMultiplyIsolatedKernel<<<blocks, PM_BLOCK, 0, stream>>>(
        rho_hat, green_hat, Ncomplex_padded);
}

// ─── 15. Zero-pad: copy N³ density into corner of (2N)³ padded grid ──

__global__ void pmZeroPadDensityKernel(const float* __restrict__ density_N,
                                        float* __restrict__ padded_2N,
                                        int N) {
    int N3 = N * N * N;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N3) return;

    int iz = idx % N;
    int iy = (idx / N) % N;
    int ix = idx / (N * N);

    int N2 = 2 * N;
    int padIdx = ix * N2 * N2 + iy * N2 + iz;
    padded_2N[padIdx] = density_N[idx];
}

void launchPMZeroPadDensity(const float* density_N, float* padded_2N,
                             int N, cudaStream_t stream) {
    int N3 = N * N * N;
    int blocks = (N3 + PM_BLOCK - 1) / PM_BLOCK;
    pmZeroPadDensityKernel<<<blocks, PM_BLOCK, 0, stream>>>(
        density_N, padded_2N, N);
}

// ─── 16. Extract: copy N³ potential from (2N)³ padded result ──────────

__global__ void pmExtractPotentialKernel(const float* __restrict__ padded_2N,
                                          float* __restrict__ potential_N,
                                          int N) {
    int N3 = N * N * N;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N3) return;

    int iz = idx % N;
    int iy = (idx / N) % N;
    int ix = idx / (N * N);

    int N2 = 2 * N;
    int padIdx = ix * N2 * N2 + iy * N2 + iz;
    potential_N[idx] = padded_2N[padIdx];
}

void launchPMExtractPotential(const float* padded_2N, float* potential_N,
                               int N, cudaStream_t stream) {
    int N3 = N * N * N;
    int blocks = (N3 + PM_BLOCK - 1) / PM_BLOCK;
    pmExtractPotentialKernel<<<blocks, PM_BLOCK, 0, stream>>>(
        padded_2N, potential_N, N);
}

} // namespace cuda
} // namespace cosmico
