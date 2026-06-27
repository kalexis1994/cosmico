#pragma once
#include <kernels/barnes_hut.cuh>   // ParticleGpu
#include <cuda_runtime.h>
#include <cufft.h>

namespace cosmico {
namespace cuda {

// PM gravity parameters passed to kernels
struct PMGpuParams {
    int gridN;              // Grid side length
    int gridN3;             // N^3
    int Ncomplex;           // N * N * (N/2+1)
    float boxSize;          // Box size L
    float cellSize;         // L / N
    float G;                // Gravitational constant
    float dt;               // Timestep
    int particleCount;
    bool comoving;
    float H;                // Current Hubble parameter (for drag)
};

// ─── Kernel launch wrappers ────────────────────────────────────────

// 1. Clear density grid to zero
void launchPMClearGrid(float* density, int N3, cudaStream_t stream);

// 2. CIC deposit: particles → density grid
//    isolated=true: no periodic wrap, skip out-of-bounds, offset by boxSize/2
void launchPMDeposit(const ParticleGpu* particles, float* density,
                     int particleCount, int gridN, float boxSize,
                     bool isolated, cudaStream_t stream);

// 3. Subtract mean density (δρ = ρ - ρ_mean)
void launchPMSubtractMean(float* density, int N3, float meanDensity,
                          cudaStream_t stream);

// 4. Green's function multiply in Fourier space: Φ̂(k) = -4πG·ρ̂(k)/k_eff²
//    with Gaussian smoothing exp(-k²σ²/2) to damp shot noise
//    (periodic mode only — isolated mode uses convolution with G_hat)
void launchPMGreenMultiply(cufftComplex* density_hat, int gridN, int Ncomplex,
                           float G, float cellSize, float smoothRadius,
                           cudaStream_t stream);

// 4b. Isolated Green's function: element-wise multiply ρ̂ × Ĝ on padded 2N grid
void launchPMGreenMultiplyIsolated(cufftComplex* rho_hat, const cufftComplex* green_hat,
                                    int Ncomplex_padded, cudaStream_t stream);

// 5. Gradient (central finite differences): Φ → (Fx, Fy, Fz)
//    isolated=true: one-sided differences at boundaries
void launchPMGradient(const float* potential, float* forceX, float* forceY,
                      float* forceZ, int gridN, float cellSize,
                      bool isolated, cudaStream_t stream);

// 6. CIC interpolation (adjoint of deposit): grid forces → particle acceleration
//    isolated=true: no periodic wrap, zero accel for out-of-bounds particles
void launchPMInterpolate(ParticleGpu* particles, const float* forceX,
                         const float* forceY, const float* forceZ,
                         int particleCount, int gridN, float boxSize,
                         bool isolated, cudaStream_t stream);

// 7. Kick: v = v*dampFactor + a*dt (with optional Hubble drag + velocity cap)
void launchPMKick(ParticleGpu* particles, int particleCount,
                  float kickDt, float dampFactor,
                  bool comoving, float H,
                  float maxVel, cudaStream_t stream);

// 8. Drift: x += v * dt with periodic wrap to [0, L)
void launchPMDrift(ParticleGpu* particles, int particleCount,
                   float dt, float boxSize, cudaStream_t stream);

// 9. Energy + momentum reduction
void launchPMEnergyReduction(const ParticleGpu* particles,
                             const float* potential, int particleCount,
                             int gridN, float boxSize,
                             double* d_energySums, cudaStream_t stream);

// 10. Momentum correction: v -= <v>
void launchPMMomentumCorrection(ParticleGpu* particles, int particleCount,
                                const double* d_energySums,
                                cudaStream_t stream);

// 11. Power spectrum: P(k) from density_hat binned by |k|
void launchPMPowerSpectrum(const cufftComplex* density_hat, int gridN,
                           int Ncomplex, float* d_spectrum, int* d_counts,
                           int nBins, float dk, cudaStream_t stream);

// Per-particle local overdensity δ=ρ/ρ̄ → particle.density, for luminosity shading
void launchPMStoreLuminosity(ParticleGpu* particles, const float* density,
                             int particleCount, int gridN, float boxSize,
                             float meanDensity, cudaStream_t stream);

// Density-field Σρ and Σρ² (out[2]) for the σ_δ growth diagnostic
void launchPMGridSums(const float* density, int N3, double* d_out,
                      cudaStream_t stream);

// 12. Sink formation: mark high-density cells as sinks, kept in a compact
//     index list (d_sinkList, size d_sinkCount, capped at maxSinks).
void launchPMSinkFormation(ParticleGpu* particles, const float* density,
                           int particleCount, int gridN, float boxSize,
                           float threshold, int maxSinks,
                           int* d_sinkList, int* d_sinkCount,
                           cudaStream_t stream);

// 13. Sink accretion: absorb nearby particles into the listed sinks (O(N·nSinks))
void launchPMSinkAccretion(ParticleGpu* particles, int particleCount,
                           float sinkRadius, float boxSize, int maxSinks,
                           int* d_sinkList, int* d_sinkCount,
                           int* d_newParticleCount, cudaStream_t stream);

// ─── Isolated (non-periodic) boundary condition helpers ──────────────

// 14. Compute free-space Green's function on 2N grid and store in green_out
//     G(r) = -1/(4π|r|) with DC=0, then apply FFT normalization
void launchPMComputeGreenFunction(float* green_out, int N, float cellSize,
                                   float G_const, cudaStream_t stream);

// 15. Zero-pad: copy N³ density into corner of (2N)³ padded grid
void launchPMZeroPadDensity(const float* density_N, float* padded_2N,
                             int N, cudaStream_t stream);

// 16. Extract: copy N³ potential from (2N)³ padded result
void launchPMExtractPotential(const float* padded_2N, float* potential_N,
                               int N, cudaStream_t stream);

} // namespace cuda
} // namespace cosmico
