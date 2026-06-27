#pragma once

namespace cosmico {

struct InflationParams {
    int gridN = 256;                    // Grid side length (64-2048)
    float boxSize = 10.0f;              // Comoving box size L
    float phi0 = 16.0f;                 // Initial field value (Planck units)
    float dphi0 = 0.0f;                 // Initial field velocity
    float dt = 0.001f;                  // Timestep
    int stepsPerFrame = 10;             // Sub-steps per render frame

    float Mpl = 1.0f;                   // Reduced Planck mass

    enum Potential { Quadratic, Starobinsky, Hilltop };
    Potential potential = Quadratic;

    // Potential-specific parameters
    float m2 = 1e-6f;                   // Quadratic: V = 1/2 m^2 phi^2
    float lambda4 = 1e-12f;             // Starobinsky: Lambda^4
    float mu = 15.0f;                   // Hilltop: mu scale

    // Volume rendering
    bool showVolume = true;
    float volumeOpacity = 3.0f;
    int volumeSteps = 128;

    // CMB rendering
    bool showCMB = false;
    float cmbRadius = 0.2f;            // Shell radius as fraction of box (0.05-0.5)
    float cmbContrast = 1.0f;          // Visual contrast amplification (0.1-10.0)
    float cmbOpacity = 0.85f;          // Sphere opacity (0.0-1.0)
    // Rigorous CMB: synthesize the map from a ΛCDM C_ℓ with acoustic peaks
    // (Gaussian a_ℓm field) instead of slicing the inflaton field. cmbSeed bumps
    // to request a fresh realisation.
    bool rigorousCMB = true;
    unsigned cmbSeed = 1;
    // Immersive sky: centre the last-scattering sphere on the camera so it
    // surrounds the observer (a CMB skybox), as we actually see it. Off = a
    // small ball at the origin viewed from outside.
    bool cmbImmersive = true;
    // Recombination timeline: a cosmic clock that advances from the Big Bang to
    // recombination (~380,000 yr). While it runs you're immersed in the opaque
    // plasma glow; at the target the CMB pattern resolves ("snapshot taken") and
    // it stops. cmbAgeYears is advanced by the app each frame.
    bool cmbTimeline = true;
    float cmbAgeYears = 0.0f;
    // Couple the CMB to the inflaton: re-tilt C_ℓ by (n_s − 0.96)·exaggeration,
    // so changing the inflaton model visibly re-tilts the CMB. cmbResync bumps
    // to request a re-synthesis (toggle/exaggeration change) keeping the seed.
    bool cmbCoupleNs = true;
    float cmbTiltExaggerate = 8.0f;
    unsigned cmbResync = 0;
    // Celestial alt-az grid overlaid on the CMB sphere (meridians of azimuth +
    // parallels of altitude, horizon highlighted) to help orient while looking.
    bool cmbGrid = false;

    // Zel'dovich structure formation
    float zeldovichSpeed = 0.02f;      // Growth factor increment per sub-step
    float zeldovichLogScale = 3.0f;    // Log normalization for density display
};

} // namespace cosmico
