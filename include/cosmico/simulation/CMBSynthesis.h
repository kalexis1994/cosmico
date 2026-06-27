#pragma once
#include <cstdint>

namespace cosmico {

// Synthesize a rigorous CMB temperature map (equirectangular lat-long, row =
// colatitude θ from north pole, col = longitude φ) as a Gaussian random field
// on the sphere drawn from a ΛCDM-like angular power spectrum C_ℓ with acoustic
// peaks:  a_ℓm ~ N(0, C_ℓ)  →  T(θ,φ) = Σ a_ℓm Y_ℓm  via fully-normalised
// associated Legendre recurrence. Output is uint8, fluctuations centred at 128
// (±3σ mapped to [0,255]). One-shot CPU; cache the result.
void synthesizeCMBMap(uint8_t* out, int W, int H, int lMax, unsigned seed);

} // namespace cosmico
