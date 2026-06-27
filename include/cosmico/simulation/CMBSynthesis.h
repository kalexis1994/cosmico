#pragma once
#include <cstdint>

namespace cosmico {

// Synthesize a rigorous CMB temperature map (equirectangular lat-long, row =
// colatitude θ from north pole, col = longitude φ) as a Gaussian random field
// on the sphere drawn from a ΛCDM-like angular power spectrum C_ℓ with acoustic
// peaks:  a_ℓm ~ N(0, C_ℓ)  →  T(θ,φ) = Σ a_ℓm Y_ℓm  via fully-normalised
// associated Legendre recurrence. Output is uint8, fluctuations centred at 128
// (±3σ mapped to [0,255]). One-shot CPU; cache the result.
//
// deltaNs re-tilts the primordial spectrum: C_ℓ ·= (ℓ/200)^deltaNs. Pass
// (n_s − 0.96) (optionally exaggerated) to couple the CMB to the inflaton's
// measured spectral index — the acoustic peaks stay, the slope follows n_s.
void synthesizeCMBMap(uint8_t* out, int W, int H, int lMax, unsigned seed,
                      double deltaNs = 0.0);

} // namespace cosmico
