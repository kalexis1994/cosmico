#pragma once
#include <cuda_runtime.h>
#include <cufft.h>
#include <cstdint>

namespace cosmico {
namespace cuda {

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
