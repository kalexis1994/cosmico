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
// The angular power spectrum C_ℓ is COMPUTED from the cosmological parameters
// (Ω_m, Ω_b, h, n_s): a semi-analytic photon-baryon acoustic model whose scales
// (sound horizon, damping, distance to last scattering) come from the
// Eisenstein-Hu / Hu-Sugiyama fitting formulas. So the acoustic peaks move and
// re-weight with the parameters — change Ω_b and the odd peaks grow, etc.
void synthesizeCMBMap(uint8_t* out, int W, int H, int lMax, unsigned seed,
                      double omegaM, double omegaB, double hubble, double ns);

} // namespace cosmico
