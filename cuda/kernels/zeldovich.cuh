#pragma once
#include <cuda_runtime.h>
#include <cufft.h>
#include <cstdint>
#include <kernels/barnes_hut.cuh>   // ParticleGpu

namespace cosmico {
namespace cuda {

// ─── Cosmological IC (Zel'dovich 1LPT) helpers ───────────────────────────
// Fill a grid with unit-variance Gaussian white noise (one sample per cell).
void launchZeldovichWhiteNoise(float* grid, int N3, unsigned int seed,
                               cudaStream_t stream);

// Shape a white-noise spectrum into delta_hat ∝ k^(ns/2)·white(k) by
// multiplying each Fourier mode by |n|^(ns/2) (DC mode zeroed). The transfer
// function T(k) is applied later by launchDisplacementMultiply, so together
// they realise sqrt(P(k)) = k^(ns/2)·T(k).
void launchZeldovichTilt(cufftComplex* deltaHat, int N, int Ncomplex,
                         float ns, float kf, float gamma, cudaStream_t stream);

// Displace particles off their uniform Lagrangian lattice by the displacement
// field (sx,sy,sz) sampled at each particle's lattice point q via CIC:
//   x = q + dispScale·s(q),   v = velFactor·dispScale·s(q)
// perSide is the lattice side (ceil(cbrt(count))); q is reconstructed from the
// particle index so positions need not be pre-seeded.
void launchZeldovichApplyToParticles(ParticleGpu* particles, int count,
                                     const float* sx, const float* sy,
                                     const float* sz,
                                     int N, int perSide, float L,
                                     float dispScale, float velFactor,
                                     cudaStream_t stream);

// Multiply phi_hat by the displacement multiplier for a given direction.
// Computes: phi_hat[k] *= i * k_dir * T(k) / (k² * N³)
// After IFFT, this gives the displacement component s_dir(q).
// dir: 0=x, 1=y, 2=z
void launchDisplacementMultiply(cufftComplex* phi_hat,
                                 int N, int Ncomplex,
                                 float dk, float L,
                                 int dir, float keq,
                                 cudaStream_t stream);

// Zel'dovich CIC deposit: for each Lagrangian point q on srcN grid,
// compute x = q + D*s(q) and deposit mass onto dstN density grid.
void launchZeldovichDeposit(const float* sx, const float* sy, const float* sz,
                             float* density, int srcN, int dstN,
                             float D, float L,
                             cudaStream_t stream);

// Clear density grid to zero
void launchClearGrid(float* density, int N3, cudaStream_t stream);

// Normalize float density grid to uint8 with log scaling.
// overdensity = density / meanDensity - 1
// output = clamp(log(1 + overdensity) / logScale, 0, 1) * 255
void launchNormalizeDensity(const float* density, uint8_t* output,
                             int N3, float meanDensity, float logScale,
                             cudaStream_t stream);

} // namespace cuda
} // namespace cosmico
