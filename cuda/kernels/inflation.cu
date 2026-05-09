#include <kernels/inflation.cuh>
#include <cstdio>
#include <cmath>
#include <curand_kernel.h>

namespace cosmico {
namespace cuda {

// ─── Device helpers ─────────────────────────────────────────────────────

// Compute V'(phi) for selected potential
__device__ float dVdphi(float phi, const InflationGpuParams& p) {
    switch (p.potentialType) {
    case 0: // Quadratic: V' = m^2 * phi
        return p.m2 * phi;
    case 1: { // Starobinsky: V' = 2*Lambda4/Mpl * sqrt(2/3) * exp(-sqrt(2/3)*phi/Mpl) * (1 - exp(-sqrt(2/3)*phi/Mpl))
        float s = sqrtf(2.0f / 3.0f);
        float e = expf(-s * phi / p.Mpl);
        return 2.0f * p.lambda4 / p.Mpl * s * e * (1.0f - e);
    }
    case 2: { // Hilltop: V' = -4*Lambda4*phi*(1-(phi/mu)^2)/mu^2
        float ratio = phi / p.mu;
        return -4.0f * p.lambda4 * phi * (1.0f - ratio * ratio) / (p.mu * p.mu);
    }
    default:
        return p.m2 * phi;
    }
}

// Compute V(phi) for selected potential
__device__ float Vphi(float phi, const InflationGpuParams& p) {
    switch (p.potentialType) {
    case 0: // Quadratic: V = 1/2 m^2 phi^2
        return 0.5f * p.m2 * phi * phi;
    case 1: { // Starobinsky: V = Lambda4 * (1 - exp(-sqrt(2/3)*phi/Mpl))^2
        float s = sqrtf(2.0f / 3.0f);
        float e = expf(-s * phi / p.Mpl);
        float t = 1.0f - e;
        return p.lambda4 * t * t;
    }
    case 2: { // Hilltop: V = Lambda4 * (1 - (phi/mu)^2)^2
        float ratio = phi / p.mu;
        float t = 1.0f - ratio * ratio;
        return p.lambda4 * t * t;
    }
    default:
        return 0.5f * p.m2 * phi * phi;
    }
}

// Convert 3D Fourier index to k^2 value
__device__ float computeKsq(int ix, int iy, int iz, int N, float dk) {
    // Frequencies: 0,1,...,N/2,-N/2+1,...,-1
    int kx = (ix <= N / 2) ? ix : ix - N;
    int ky = (iy <= N / 2) ? iy : iy - N;
    int kz = iz; // R2C: kz in [0, N/2]
    float fkx = kx * dk;
    float fky = ky * dk;
    float fkz = kz * dk;
    return fkx * fkx + fky * fky + fkz * fkz;
}

// ─── Kernel 1: Initialize field in real space ───────────────────────────

__global__ void initFieldRealSpaceKernel(
    float* phi,
    float* pi,
    const InflationGpuParams params,
    float phi0,
    float dphi0,
    unsigned long long seed
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= params.N3) return;

    // Homogeneous background + tiny fluctuations
    curandState rng;
    curand_init(seed, idx, 0, &rng);

    // Fluctuation amplitude ~ H/(2*pi) ~ 1e-5 for typical inflation
    // Keep it small so the background dominates
    float fluctuation = curand_normal(&rng) * 1e-5f;

    phi[idx] = phi0 + fluctuation;
    pi[idx] = params.a3 * dphi0;  // pi = a^3 * dphi/dt
}

void launchInitFieldKernel(
    FieldGpu& field,
    const InflationGpuParams& params,
    float phi0,
    unsigned long long seed,
    cudaStream_t stream
) {
    int blockSize = 256;
    int gridSize = (params.N3 + blockSize - 1) / blockSize;

    // dphi0 = 0 by default (slow-roll start)
    initFieldRealSpaceKernel<<<gridSize, blockSize, 0, stream>>>(
        field.phi, field.pi, params, phi0, 0.0f, seed);
}

// ─── Kernel 2: Laplacian in Fourier space ───────────────────────────────

__global__ void laplacianKernel(
    cufftComplex* phi_hat,
    const InflationGpuParams params
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int Nhalf = params.N / 2 + 1;
    int totalComplex = params.N * params.N * Nhalf;

    if (idx >= totalComplex) return;

    int iz = idx % Nhalf;
    int iy = (idx / Nhalf) % params.N;
    int ix = idx / (Nhalf * params.N);

    float ksq = computeKsq(ix, iy, iz, params.N, params.dk);

    // Laplacian in Fourier: multiply by -k^2 / N^3
    // The 1/N^3 normalizes the cuFFT round-trip (R2C then C2R)
    float factor = -ksq / (float)params.N3;
    phi_hat[idx].x *= factor;
    phi_hat[idx].y *= factor;
}

void launchLaplacianKernel(
    cufftComplex* phi_hat,
    const InflationGpuParams& params,
    cudaStream_t stream
) {
    int Nhalf = params.N / 2 + 1;
    int totalComplex = params.N * params.N * Nhalf;
    int blockSize = 256;
    int gridSize = (totalComplex + blockSize - 1) / blockSize;

    laplacianKernel<<<gridSize, blockSize, 0, stream>>>(phi_hat, params);
}

// ─── Kernel 3: Leapfrog kick ────────────────────────────────────────────

__global__ void leapfrogKickKernel(
    float* pi,
    const float* phi,
    const float* laplacian,
    const InflationGpuParams params,
    float halfDt
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= params.N3) return;

    // pi += halfDt * a^3 * (laplacian/a^2 - V'(phi))
    float force = laplacian[idx] / (params.a * params.a) - dVdphi(phi[idx], params);
    pi[idx] += halfDt * params.a3 * force;
}

void launchLeapfrogKickKernel(
    FieldGpu& field,
    const InflationGpuParams& params,
    float halfDt,
    cudaStream_t stream
) {
    int blockSize = 256;
    int gridSize = (params.N3 + blockSize - 1) / blockSize;

    leapfrogKickKernel<<<gridSize, blockSize, 0, stream>>>(
        field.pi, field.phi, field.laplacian, params, halfDt);
}

// ─── Kernel 4: Leapfrog drift ───────────────────────────────────────────

__global__ void leapfrogDriftKernel(
    float* phi,
    const float* pi,
    const InflationGpuParams params
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= params.N3) return;

    // phi += dt * pi / a^3
    phi[idx] += params.dt * pi[idx] / params.a3;
}

void launchLeapfrogDriftKernel(
    FieldGpu& field,
    const InflationGpuParams& params,
    cudaStream_t stream
) {
    int blockSize = 256;
    int gridSize = (params.N3 + blockSize - 1) / blockSize;

    leapfrogDriftKernel<<<gridSize, blockSize, 0, stream>>>(
        field.phi, field.pi, params);
}

// ─── Kernel 5: Energy reduction ─────────────────────────────────────────

// Block-level reduction using shared memory + warp shuffle
__device__ void warpReduce6(double* vals) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        for (int i = 0; i < 6; i++) {
            vals[i] += __shfl_down_sync(0xffffffff, vals[i], offset);
        }
    }
}

__global__ void energyReductionKernel(
    const float* phi,
    const float* pi,
    const float* laplacian,
    const InflationGpuParams params,
    double* output  // 6 doubles: KE, GE, PE, phiSum, phi2Sum, Vsum
) {
    // Shared memory for block reduction: 6 values * 32 warps
    __shared__ double sdata[6 * 32];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    int warpId = threadIdx.x >> 5;

    double vals[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    if (idx < params.N3) {
        float p = phi[idx];
        float mom = pi[idx];
        float lap = laplacian[idx]; // Note: laplacian already computed for energy

        float phidot = mom / params.a3;
        float v = Vphi(p, params);

        // We approximate gradient energy from the laplacian:
        // <|grad phi|^2> ~ -<phi * laplacian(phi)>
        // This avoids computing gradients separately
        float gradE = -p * lap; // phi * (-nabla^2 phi) ~ |grad phi|^2

        vals[0] = 0.5 * (double)(phidot * phidot);          // KE
        vals[1] = 0.5 * (double)(gradE) / (params.a * params.a); // GE
        vals[2] = (double)v;                                      // PE
        vals[3] = (double)p;                                      // phiSum
        vals[4] = (double)(p * p);                                // phi2Sum
        vals[5] = (double)v;                                      // Vsum
    }

    // Warp-level reduction
    warpReduce6(vals);

    // Write warp result to shared memory
    if (lane == 0) {
        for (int i = 0; i < 6; i++) {
            sdata[warpId * 6 + i] = vals[i];
        }
    }
    __syncthreads();

    // First warp reduces shared memory
    int numWarps = blockDim.x / 32;
    if (warpId == 0) {
        for (int i = 0; i < 6; i++) {
            vals[i] = (lane < numWarps) ? sdata[lane * 6 + i] : 0.0;
        }
        warpReduce6(vals);

        if (lane == 0) {
            for (int i = 0; i < 6; i++) {
                atomicAdd(&output[i], vals[i]);
            }
        }
    }
}

void launchEnergyReductionKernel(
    const FieldGpu& field,
    const InflationGpuParams& params,
    double* d_output,
    cudaStream_t stream
) {
    // Zero output
    cudaMemsetAsync(d_output, 0, 6 * sizeof(double), stream);

    int blockSize = 256;
    int gridSize = (params.N3 + blockSize - 1) / blockSize;

    energyReductionKernel<<<gridSize, blockSize, 0, stream>>>(
        field.phi, field.pi, field.laplacian, params, d_output);
}

// ─── Kernel 6: Power spectrum binning ───────────────────────────────────

__global__ void powerSpectrumKernel(
    const cufftComplex* phi_hat,
    const InflationGpuParams params,
    float* spectrum,
    int* counts,
    int nBins,
    float logKmin,
    float dlogK
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int Nhalf = params.N / 2 + 1;
    int totalComplex = params.N * params.N * Nhalf;

    if (idx >= totalComplex) return;

    int iz = idx % Nhalf;
    int iy = (idx / Nhalf) % params.N;
    int ix = idx / (Nhalf * params.N);

    float ksq = computeKsq(ix, iy, iz, params.N, params.dk);
    if (ksq < 1e-12f) return; // Skip zero mode

    float k = sqrtf(ksq);
    float logK = logf(k);

    int bin = (int)((logK - logKmin) / dlogK);
    if (bin < 0 || bin >= nBins) return;

    float re = phi_hat[idx].x;
    float im = phi_hat[idx].y;
    float power = re * re + im * im;

    // Weight by 4*pi*k^2 / Volume, normalized by N^3^2 for cuFFT
    float weight = 4.0f * 3.14159265f * ksq / (params.L * params.L * params.L);
    float norm = (float)params.N3 * (float)params.N3;
    power = power * weight / norm;

    // R2C: modes with kz != 0 and kz != N/2 appear twice (hermitian symmetry)
    int mult = (iz > 0 && iz < params.N / 2) ? 2 : 1;

    atomicAdd(&spectrum[bin], power * mult);
    atomicAdd(&counts[bin], mult);
}

void launchPowerSpectrumKernel(
    const cufftComplex* phi_hat,
    const InflationGpuParams& params,
    float* d_spectrum,
    int* d_counts,
    int nBins,
    float kMin,
    float kMax,
    cudaStream_t stream
) {
    // Zero output arrays
    cudaMemsetAsync(d_spectrum, 0, nBins * sizeof(float), stream);
    cudaMemsetAsync(d_counts, 0, nBins * sizeof(int), stream);

    float logKmin = logf(kMin);
    float logKmax = logf(kMax);
    float dlogK = (logKmax - logKmin) / (float)nBins;

    int Nhalf = params.N / 2 + 1;
    int totalComplex = params.N * params.N * Nhalf;
    int blockSize = 256;
    int gridSize = (totalComplex + blockSize - 1) / blockSize;

    powerSpectrumKernel<<<gridSize, blockSize, 0, stream>>>(
        phi_hat, params, d_spectrum, d_counts, nBins, logKmin, dlogK);
}

} // namespace cuda
} // namespace cosmico
