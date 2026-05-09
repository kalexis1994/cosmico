#include <kernels/cmb_extract.cuh>
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

namespace cosmico {
namespace cuda {

// ─── Trilinear interpolation with periodic boundary ─────────────────────

__device__ float trilinearSample(const float* __restrict__ phi, int N, float x, float y, float z) {
    // Wrap to [0, N) with periodic boundary
    x = fmodf(x, (float)N);
    y = fmodf(y, (float)N);
    z = fmodf(z, (float)N);
    if (x < 0.0f) x += (float)N;
    if (y < 0.0f) y += (float)N;
    if (z < 0.0f) z += (float)N;

    int x0 = (int)floorf(x);
    int y0 = (int)floorf(y);
    int z0 = (int)floorf(z);
    int x1 = (x0 + 1) % N;
    int y1 = (y0 + 1) % N;
    int z1 = (z0 + 1) % N;
    x0 = x0 % N;
    y0 = y0 % N;
    z0 = z0 % N;

    float fx = x - floorf(x);
    float fy = y - floorf(y);
    float fz = z - floorf(z);

    // 8 corner samples
    float c000 = phi[x0 * N * N + y0 * N + z0];
    float c001 = phi[x0 * N * N + y0 * N + z1];
    float c010 = phi[x0 * N * N + y1 * N + z0];
    float c011 = phi[x0 * N * N + y1 * N + z1];
    float c100 = phi[x1 * N * N + y0 * N + z0];
    float c101 = phi[x1 * N * N + y0 * N + z1];
    float c110 = phi[x1 * N * N + y1 * N + z0];
    float c111 = phi[x1 * N * N + y1 * N + z1];

    // Interpolate along z
    float c00 = c000 * (1.0f - fz) + c001 * fz;
    float c01 = c010 * (1.0f - fz) + c011 * fz;
    float c10 = c100 * (1.0f - fz) + c101 * fz;
    float c11 = c110 * (1.0f - fz) + c111 * fz;

    // Interpolate along y
    float c0 = c00 * (1.0f - fy) + c01 * fy;
    float c1 = c10 * (1.0f - fy) + c11 * fy;

    // Interpolate along x
    return c0 * (1.0f - fx) + c1 * fx;
}

// ─── Stats kernel: compute mean and sigma of dT on the shell ────────────
// Samples every 4th pixel for speed, accumulates sum and sumSq via atomics.
// d_stats[0] = sum(dT), d_stats[1] = sum(dT^2)

__global__ void cmbStatsKernel(const float* __restrict__ phi, int N,
                                float phiMean, float Mpl,
                                float radius, float boxSize,
                                int mapW, int mapH,
                                double* __restrict__ d_stats) {
    extern __shared__ double sdata[];
    double* sSum   = sdata;
    double* sSumSq = sdata + blockDim.x;

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Sample every 4th pixel
    int totalSamples = (mapW / 4) * (mapH / 4);
    unsigned int stride = blockDim.x * gridDim.x;

    float R = radius * boxSize;
    float center = boxSize * 0.5f;
    float invMpl3 = 1.0f / (3.0f * Mpl);

    double localSum = 0.0;
    double localSumSq = 0.0;

    for (unsigned int i = idx; i < (unsigned int)totalSamples; i += stride) {
        int su = (i % (mapW / 4)) * 4;
        int sv = (i / (mapW / 4)) * 4;

        float theta = ((float)sv / (float)mapH) * 3.14159265f;
        float phi_angle = ((float)su / (float)mapW) * 2.0f * 3.14159265f;

        float sinT = sinf(theta);
        float cosT = cosf(theta);
        float sinP = sinf(phi_angle);
        float cosP = cosf(phi_angle);

        float sx = center + R * sinT * cosP;
        float sy = center + R * sinT * sinP;
        float sz = center + R * cosT;

        // Convert to grid coordinates
        float gx = sx / boxSize * (float)N;
        float gy = sy / boxSize * (float)N;
        float gz = sz / boxSize * (float)N;

        float val = trilinearSample(phi, N, gx, gy, gz);
        double dT = -(double)(val - phiMean) * (double)invMpl3;

        localSum += dT;
        localSumSq += dT * dT;
    }

    sSum[tid] = localSum;
    sSumSq[tid] = localSumSq;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sSum[tid] += sSum[tid + s];
            sSumSq[tid] += sSumSq[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(&d_stats[0], sSum[0]);
        atomicAdd(&d_stats[1], sSumSq[0]);
    }
}

// ─── Sample kernel: full equirectangular map with normalization ──────────

__global__ void cmbSampleKernel(const float* __restrict__ phi, int N,
                                 float phiMean, float Mpl,
                                 float radius, float boxSize,
                                 uint8_t* __restrict__ cmbMap,
                                 int mapW, int mapH,
                                 float dtMean, float dtSigma) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = mapW * mapH;
    if (idx >= total) return;

    int u = idx % mapW;
    int v = idx / mapW;

    float theta = ((float)v + 0.5f) / (float)mapH * 3.14159265f;
    float phi_angle = ((float)u + 0.5f) / (float)mapW * 2.0f * 3.14159265f;

    float sinT = sinf(theta);
    float cosT = cosf(theta);
    float sinP = sinf(phi_angle);
    float cosP = cosf(phi_angle);

    float R = radius * boxSize;
    float center = boxSize * 0.5f;

    float sx = center + R * sinT * cosP;
    float sy = center + R * sinT * sinP;
    float sz = center + R * cosT;

    // Convert to grid coordinates
    float gx = sx / boxSize * (float)N;
    float gy = sy / boxSize * (float)N;
    float gz = sz / boxSize * (float)N;

    float val = trilinearSample(phi, N, gx, gy, gz);

    // Sachs-Wolfe: dT/T ~ -delta_phi / (3 * Mpl)
    float dT = -(val - phiMean) / (3.0f * Mpl);

    // Normalize: map (dT - mean) / (3*sigma) to [0, 1]
    float safeSigma = (dtSigma > 1e-20f) ? dtSigma : 1e-20f;
    float normalized = (dT - dtMean) / (3.0f * safeSigma) * 0.5f + 0.5f;

    // Clamp to [0, 1] and convert to uint8
    normalized = fminf(fmaxf(normalized, 0.0f), 1.0f);
    cmbMap[idx] = (uint8_t)(normalized * 255.0f);
}

// ─── Host wrapper ─────────────────────────────────────────────────────────

void launchCMBExtraction(const float* d_phi, int N,
                          float phiMean, float Mpl,
                          float radius, float boxSize,
                          uint8_t* d_cmbMap, int mapW, int mapH,
                          double* d_stats,
                          cudaStream_t stream) {
    // Zero stats
    double zeros[2] = {0.0, 0.0};
    cudaMemcpyAsync(d_stats, zeros, 2 * sizeof(double), cudaMemcpyHostToDevice, stream);

    // Stats kernel (subsampled)
    int totalSamples = (mapW / 4) * (mapH / 4);
    int blockSize = 256;
    int numBlocks = (totalSamples + blockSize - 1) / blockSize;
    if (numBlocks > 512) numBlocks = 512;
    size_t sharedMem = blockSize * 2 * sizeof(double);

    cmbStatsKernel<<<numBlocks, blockSize, sharedMem, stream>>>(
        d_phi, N, phiMean, Mpl, radius, boxSize, mapW, mapH, d_stats);

    // Read back stats
    double hostStats[2];
    cudaMemcpyAsync(hostStats, d_stats, 2 * sizeof(double), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    double mean = hostStats[0] / totalSamples;
    double variance = hostStats[1] / totalSamples - mean * mean;
    double sigma = (variance > 0.0) ? sqrt(variance) : 1e-20;

    // Sample kernel (full resolution)
    int totalPixels = mapW * mapH;
    int blocks2 = (totalPixels + blockSize - 1) / blockSize;

    cmbSampleKernel<<<blocks2, blockSize, 0, stream>>>(
        d_phi, N, phiMean, Mpl, radius, boxSize,
        d_cmbMap, mapW, mapH,
        (float)mean, (float)sigma);
}

} // namespace cuda
} // namespace cosmico
