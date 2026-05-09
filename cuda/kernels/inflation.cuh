#pragma once
#include <cuda_runtime.h>
#include <cufft.h>
#include <cstdint>

namespace cosmico {
namespace cuda {

// GPU-side parameters for inflation kernels
struct InflationGpuParams {
    int N;                  // Grid side length
    int N3;                 // N*N*N total grid points
    int Ncomplex;           // N*N*(N/2+1) complex elements (R2C hermitian)
    float L;                // Box size
    float dk;               // Fourier spacing = 2*pi/L
    float dt;               // Timestep
    float a;                // Scale factor a(t)
    float a3;               // a^3 (precomputed)
    float Mpl;              // Reduced Planck mass

    // Potential type: 0=Quadratic, 1=Starobinsky, 2=Hilltop
    int potentialType;
    float m2;               // Quadratic mass^2
    float lambda4;          // Starobinsky Lambda^4
    float mu;               // Hilltop mu
};

// GPU field arrays (device pointers)
struct FieldGpu {
    float* phi;             // Field values [N^3]
    float* pi;              // Conjugate momentum pi = a^3 * dphi/dt [N^3]
    float* laplacian;       // Temp buffer for nabla^2 phi [N^3]
    cufftComplex* phi_hat;  // Fourier transform [N*N*(N/2+1)] (R2C hermitian)
};

// ─── Host-callable kernel wrappers ──────────────────────────────────────

// Initialize field with phi0 + vacuum fluctuations in Fourier space
void launchInitFieldKernel(
    FieldGpu& field,
    const InflationGpuParams& params,
    float phi0,
    unsigned long long seed,
    cudaStream_t stream
);

// Multiply phi_hat by -k^2/N^3 to compute Laplacian in Fourier space
void launchLaplacianKernel(
    cufftComplex* phi_hat,
    const InflationGpuParams& params,
    cudaStream_t stream
);

// Leapfrog kick: pi += halfDt * a^3 * (laplacian/a^2 - V'(phi))
void launchLeapfrogKickKernel(
    FieldGpu& field,
    const InflationGpuParams& params,
    float halfDt,
    cudaStream_t stream
);

// Leapfrog drift: phi += dt * pi / a^3
void launchLeapfrogDriftKernel(
    FieldGpu& field,
    const InflationGpuParams& params,
    cudaStream_t stream
);

// Reduce energy densities over N^3 grid points
// Output: 6 doubles [KE, GE, PE, phiSum, phi2Sum, Vsum]
void launchEnergyReductionKernel(
    const FieldGpu& field,
    const InflationGpuParams& params,
    double* d_output,
    cudaStream_t stream
);

// Bin |phi_hat(k)|^2 by |k| into logarithmic bins
void launchPowerSpectrumKernel(
    const cufftComplex* phi_hat,
    const InflationGpuParams& params,
    float* d_spectrum,
    int* d_counts,
    int nBins,
    float kMin,
    float kMax,
    cudaStream_t stream
);

} // namespace cuda
} // namespace cosmico
