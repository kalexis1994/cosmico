#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace cosmico {
namespace cuda {

// Extract CMB temperature map from inflation field.
// Samples phi on a spherical shell, computes Sachs-Wolfe dT/T = -dphi/(3*Mpl),
// normalizes by sigma, and outputs an equirectangular uint8_t map [mapW x mapH].
//
// d_phi:     source field [N^3 floats] on device
// N:         source grid side length
// phiMean:   mean field value (for delta_phi = phi - phiMean)
// Mpl:       reduced Planck mass
// radius:    shell radius as fraction of box (0..1), multiplied by boxSize internally
// boxSize:   comoving box size L
// d_cmbMap:  output [mapW * mapH uint8_t] on device
// mapW, mapH: equirectangular map dimensions (e.g. 1024 x 512)
// d_stats:   device scratch for 4 doubles (sum, sumSq, mean, sigma)
void launchCMBExtraction(const float* d_phi, int N,
                          float phiMean, float Mpl,
                          float radius, float boxSize,
                          uint8_t* d_cmbMap, int mapW, int mapH,
                          double* d_stats,
                          cudaStream_t stream);

} // namespace cuda
} // namespace cosmico
