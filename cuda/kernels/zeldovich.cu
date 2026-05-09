#include <kernels/zeldovich.cuh>
#include <cuda_runtime.h>
#include <cstdio>

namespace cosmico {
namespace cuda {

// ─── Displacement multiply kernel ────────────────────────────────────────
// In Fourier space, compute the displacement component for direction dir:
//   s_hat_d(k) = i * k_d / k² * T(k) * phi_hat(k) / N³
// T(k) = 1 / (1 + (k/keq)²)  — suppresses small-scale power

__global__ void displacementMultiplyKernel(cufftComplex* __restrict__ phi_hat,
                                            int N, int Ncomplex,
                                            float dk, float L,
                                            int dir, float keq) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= Ncomplex) return;

    int Nh = N / 2 + 1;
    int iz = idx % Nh;
    int iy = (idx / Nh) % N;
    int ix = idx / (Nh * N);

    // Frequency indices (centered)
    int nx = (ix <= N / 2) ? ix : ix - N;
    int ny = (iy <= N / 2) ? iy : iy - N;
    int nz = iz;

    float n2 = (float)(nx * nx + ny * ny + nz * nz);

    // DC mode: zero displacement
    if (n2 < 0.5f) {
        phi_hat[idx].x = 0.0f;
        phi_hat[idx].y = 0.0f;
        return;
    }

    // Wavenumber component for this direction
    float n_d;
    if (dir == 0) n_d = (float)nx;
    else if (dir == 1) n_d = (float)ny;
    else n_d = (float)nz;

    float k_phys2 = dk * dk * n2;

    // Transfer function: suppress small-scale (high-k) modes
    float T = 1.0f / (1.0f + k_phys2 / (keq * keq));

    // Displacement multiplier: i * k_d / k² * T(k)
    // Physical k_d = dk * n_d, physical k² = dk² * n²
    // So: i * dk * n_d / (dk² * n²) * T = i * n_d / (dk * n²) * T
    // But we also need to account for FFT normalization (1/N³ on inverse)
    int N3 = N * N * N;
    float factor = n_d * T / (dk * n2 * (float)N3);

    // Multiply by i: i * (a + bi) = (-b + ai)
    float re = phi_hat[idx].x;
    float im = phi_hat[idx].y;
    phi_hat[idx].x = -factor * im;
    phi_hat[idx].y =  factor * re;
}

void launchDisplacementMultiply(cufftComplex* phi_hat,
                                 int N, int Ncomplex,
                                 float dk, float L,
                                 int dir, float keq,
                                 cudaStream_t stream) {
    int blockSize = 256;
    int numBlocks = (Ncomplex + blockSize - 1) / blockSize;
    displacementMultiplyKernel<<<numBlocks, blockSize, 0, stream>>>(
        phi_hat, N, Ncomplex, dk, L, dir, keq);
}

// ─── Clear grid kernel ──────────────────────────────────────────────────

__global__ void clearGridKernel(float* density, int N3) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N3) density[idx] = 0.0f;
}

void launchClearGrid(float* density, int N3, cudaStream_t stream) {
    int blockSize = 256;
    int numBlocks = (N3 + blockSize - 1) / blockSize;
    clearGridKernel<<<numBlocks, blockSize, 0, stream>>>(density, N3);
}

// ─── Zel'dovich CIC deposit kernel ──────────────────────────────────────

__global__ void zeldovichDepositKernel(const float* __restrict__ sx,
                                        const float* __restrict__ sy,
                                        const float* __restrict__ sz,
                                        float* __restrict__ density,
                                        int srcN, int dstN,
                                        float D, float L) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int srcN3 = srcN * srcN * srcN;
    if (idx >= srcN3) return;

    // Lagrangian 3D index
    int iz_src = idx % srcN;
    int iy_src = (idx / srcN) % srcN;
    int ix_src = idx / (srcN * srcN);

    // Physical Lagrangian position
    float qx = (float)ix_src * L / (float)srcN;
    float qy = (float)iy_src * L / (float)srcN;
    float qz = (float)iz_src * L / (float)srcN;

    // Displaced Eulerian position
    float x = qx + D * sx[idx];
    float y = qy + D * sy[idx];
    float z = qz + D * sz[idx];

    // Periodic boundary: wrap to [0, L)
    x = x - floorf(x / L) * L;
    y = y - floorf(y / L) * L;
    z = z - floorf(z / L) * L;

    // Map to density grid coordinates [0, dstN)
    float gx = x * (float)dstN / L;
    float gy = y * (float)dstN / L;
    float gz = z * (float)dstN / L;

    // CIC: base cell and fractional offsets
    int ix0 = (int)floorf(gx - 0.5f);
    int iy0 = (int)floorf(gy - 0.5f);
    int iz0 = (int)floorf(gz - 0.5f);

    float wx = gx - 0.5f - (float)ix0;
    float wy = gy - 0.5f - (float)iy0;
    float wz = gz - 0.5f - (float)iz0;

    // Deposit to 8 surrounding cells (CIC trilinear interpolation)
    for (int di = 0; di < 2; di++) {
        float fx = (di == 0) ? (1.0f - wx) : wx;
        int cix = ((ix0 + di) % dstN + dstN) % dstN;
        for (int dj = 0; dj < 2; dj++) {
            float fy = (dj == 0) ? (1.0f - wy) : wy;
            int ciy = ((iy0 + dj) % dstN + dstN) % dstN;
            for (int ddk = 0; ddk < 2; ddk++) {
                float fz = (ddk == 0) ? (1.0f - wz) : wz;
                int ciz = ((iz0 + ddk) % dstN + dstN) % dstN;

                float weight = fx * fy * fz;
                int cellIdx = cix * dstN * dstN + ciy * dstN + ciz;
                atomicAdd(&density[cellIdx], weight);
            }
        }
    }
}

void launchZeldovichDeposit(const float* sx, const float* sy, const float* sz,
                             float* density, int srcN, int dstN,
                             float D, float L,
                             cudaStream_t stream) {
    int srcN3 = srcN * srcN * srcN;
    int blockSize = 256;
    int numBlocks = (srcN3 + blockSize - 1) / blockSize;
    zeldovichDepositKernel<<<numBlocks, blockSize, 0, stream>>>(
        sx, sy, sz, density, srcN, dstN, D, L);
}

// ─── Normalize density to uint8 with log scaling ────────────────────────

__global__ void normalizeDensityKernel(const float* __restrict__ density,
                                        uint8_t* __restrict__ output,
                                        int N3, float meanDensity,
                                        float logScale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N3) return;

    // Overdensity: delta = rho / rho_mean - 1
    float delta = (meanDensity > 1e-10f)
        ? density[idx] / meanDensity - 1.0f
        : 0.0f;

    // Log scaling of overdensity for visualization
    // Only show overdense regions (delta > 0)
    float val = 0.0f;
    if (delta > 0.0f) {
        val = logf(1.0f + delta) / logScale;
    }
    val = fminf(fmaxf(val, 0.0f), 1.0f);

    output[idx] = (uint8_t)(val * 255.0f);
}

void launchNormalizeDensity(const float* density, uint8_t* output,
                             int N3, float meanDensity, float logScale,
                             cudaStream_t stream) {
    int blockSize = 256;
    int numBlocks = (N3 + blockSize - 1) / blockSize;
    normalizeDensityKernel<<<numBlocks, blockSize, 0, stream>>>(
        density, output, N3, meanDensity, logScale);
}

} // namespace cuda
} // namespace cosmico
