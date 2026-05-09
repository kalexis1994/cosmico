#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace cosmico {
namespace cuda {

// Extract density volume from inflation field.
// Performs min/max reduction of |phi - phiMean|, then downsamples and normalizes
// from srcN^3 to dstN^3 as uint8_t [0-255].
//
// d_phi:     source field [srcN^3 floats] on device
// srcN:      source grid side length
// phiMean:   mean field value
// d_density: output density [dstN^3 uint8_t] on device
// dstN:      output grid side length (e.g. 128)
// d_minVal:  device scratch for min reduction (1 float)
// d_maxVal:  device scratch for max reduction (1 float)
void launchDensityExtraction(const float* d_phi, int srcN, float phiMean,
                              uint8_t* d_density, int dstN,
                              float* d_minVal, float* d_maxVal,
                              cudaStream_t stream);

} // namespace cuda
} // namespace cosmico
