#include <kernels/density_extract.cuh>
#include <cuda_runtime.h>
#include <cstdio>
#include <cfloat>
#include <cmath>

namespace cosmico {
namespace cuda {

// ─── Sum + SumSq reduction for mean/sigma ─────────────────────────────────
// Computes sum(|δφ|) and sum(|δφ|²) across all N³ field points.
// Output: d_stats[0] = sum, d_stats[1] = sumSq

__global__ void statsReductionKernel(const float* __restrict__ phi,
                                      int N3, float phiMean,
                                      double* __restrict__ d_stats) {
    extern __shared__ double sdata[];
    double* sSum   = sdata;
    double* sSumSq = sdata + blockDim.x;

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    double localSum = 0.0;
    double localSumSq = 0.0;

    for (unsigned int i = idx; i < (unsigned int)N3; i += stride) {
        double dev = (double)fabsf(phi[i] - phiMean);
        localSum += dev;
        localSumSq += dev * dev;
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

// ─── Downsample + Sigma-based normalize ───────────────────────────────────
// Block-average from srcN³ to dstN³, then normalize using:
//   value = clamp((blockAvg_|δφ| - meanDev) / (nSigma * sigma), 0, 1)
// This highlights peaks that are above the average deviation.

__global__ void downsampleSigmaNormalizeKernel(const float* __restrict__ phi,
                                                uint8_t* __restrict__ density,
                                                int srcN, int dstN,
                                                float phiMean,
                                                float meanDev, float sigma) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = dstN * dstN * dstN;
    if (idx >= total) return;

    int dz = idx % dstN;
    int dy = (idx / dstN) % dstN;
    int dx = idx / (dstN * dstN);

    int blockSize = srcN / dstN;
    if (blockSize < 1) blockSize = 1;

    int sx0 = dx * blockSize;
    int sy0 = dy * blockSize;
    int sz0 = dz * blockSize;

    float sum = 0.0f;
    int count = 0;

    for (int bx = 0; bx < blockSize && (sx0 + bx) < srcN; bx++) {
        for (int by = 0; by < blockSize && (sy0 + by) < srcN; by++) {
            for (int bz = 0; bz < blockSize && (sz0 + bz) < srcN; bz++) {
                int si = (sx0 + bx) * srcN * srcN + (sy0 + by) * srcN + (sz0 + bz);
                sum += fabsf(phi[si] - phiMean);
                count++;
            }
        }
    }

    float avg = (count > 0) ? sum / count : 0.0f;

    // Sigma-based normalization: how many sigma above the mean deviation
    // Maps [meanDev, meanDev + nSigma*sigma] → [0, 1]
    float nSigma = 3.0f;
    float normalized = (sigma > 1e-20f)
        ? (avg - meanDev) / (nSigma * sigma)
        : 0.0f;

    // Clamp and apply power curve to spread out the distribution
    normalized = fminf(fmaxf(normalized, 0.0f), 1.0f);

    density[idx] = (uint8_t)(normalized * 255.0f);
}

// ─── Host wrapper ─────────────────────────────────────────────────────────

void launchDensityExtraction(const float* d_phi, int srcN, float phiMean,
                              uint8_t* d_density, int dstN,
                              float* d_minVal, float* d_maxVal,
                              cudaStream_t stream) {
    int N3 = srcN * srcN * srcN;

    // d_minVal/d_maxVal reused as d_stats (2 doubles = 16 bytes, fits in 2 floats * 2 = OK)
    // Actually we need double storage — reinterpret the two float pointers as one double* pair
    // Safer: use d_minVal as base for 2 doubles (16 bytes fits in staging)
    double* d_stats = reinterpret_cast<double*>(d_minVal);

    // Zero out stats
    double zeros[2] = {0.0, 0.0};
    cudaMemcpyAsync(d_stats, zeros, 2 * sizeof(double), cudaMemcpyHostToDevice, stream);

    // Launch stats reduction (sum and sumSq of |δφ|)
    int blockSize = 256;
    int numBlocks = (N3 + blockSize - 1) / blockSize;
    if (numBlocks > 1024) numBlocks = 1024;
    size_t sharedMem = blockSize * 2 * sizeof(double);

    statsReductionKernel<<<numBlocks, blockSize, sharedMem, stream>>>(
        d_phi, N3, phiMean, d_stats);

    // Read back stats
    double hostStats[2];
    cudaMemcpyAsync(hostStats, d_stats, 2 * sizeof(double), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    double meanDev = hostStats[0] / N3;
    double variance = hostStats[1] / N3 - meanDev * meanDev;
    double sigma = (variance > 0.0) ? sqrt(variance) : 1e-20;

    // Launch downsample + sigma-normalize
    int dstN3 = dstN * dstN * dstN;
    int blocks2 = (dstN3 + blockSize - 1) / blockSize;

    downsampleSigmaNormalizeKernel<<<blocks2, blockSize, 0, stream>>>(
        d_phi, d_density, srcN, dstN, phiMean,
        (float)meanDev, (float)sigma);
}

} // namespace cuda
} // namespace cosmico
