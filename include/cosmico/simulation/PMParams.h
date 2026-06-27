#pragma once

namespace cosmico {

struct PMParams {
    int gridN = 128;                    // Grid side length (64/128/256/512)
    float boxSize = 100.0f;             // Comoving box size L
    float G = 1.0f;                      // Gravitational constant (matches BruteForce/BH)
    float dt = 0.001f;                  // Timestep (matches BruteForce/BH)
    int stepsPerFrame = 10;             // Sub-steps per render frame
    int particleCount = 32768;          // Number of particles (default 32K)
    float damping = 0.0f;              // Velocity damping rate (0 = symplectic)
    float softening = 0.05f;            // Gravitational softening (N-Body direct)
    bool openBoundary = false;          // No periodic wrap (infinite box)

    // Cosmological expansion (Friedmann + comoving coordinates)
    bool comoving = false;
    float H0 = 0.02f;                  // Hubble constant (simulation units)
    float OmegaM = 1.0f;               // Matter density parameter (EdS default)
    float aInit = 0.3f;                // Initial scale factor (z_init ≈ 2.3)
    float aMax = 1.5f;                 // Freeze expansion past this a (lets the web finish collapsing)

    // Cosmological initial conditions (Zel'dovich 1LPT — used by the
    // "Cosmological" IC). A Gaussian random field with P(k) ∝ k^ns·T²(k)
    // displaces particles off a uniform lattice and seeds growing-mode
    // peculiar velocities, replacing the old white-noise jitter.
    float ns = 0.96f;                  // Primordial spectral index
    float zeldovichAmplitude = 0.3f;   // Initial RMS displacement / mean particle spacing
    int   zeldovichSeed = 1234;        // Seed for the Gaussian random field
    float forceSmoothing = 0.8f;       // PM force Gaussian smoothing, in mean particle spacings (lower = sharper web)

    // Real CDM transfer function T(k) (BBKS 1986 + Sugiyama 1995 shape),
    // replacing the toy Lorentzian. Needs a physical box size to map grid modes
    // to physical wavenumbers k [h/Mpc]; Γ = Ωm·h·exp(−Ωb(1+√(2h)/Ωm)).
    float omegaB = 0.048f;             // Baryon density parameter
    float hubbleH = 0.68f;             // Dimensionless Hubble (H0 = 100h km/s/Mpc)
    float boxSizeMpc = 200.0f;         // Physical comoving box size [Mpc/h]

    // Corrections
    bool correctMomentum = true;        // Subtract mean velocity each step

    // Sink particles (black hole formation)
    bool enableSink = false;            // Only togglable when paused
    float sinkDensityThreshold = 50.0f; // ρ/ρ_mean threshold for formation
    float sinkRadius = 2.0f;           // Accretion radius (in grid cells)

    // Diagnostics
    bool showPowerSpectrum = true;      // Show P(k) plot in UI

    // Rendering: when true, scale particle positions by a(t)/aInit about the
    // origin so the box visibly inflates and structure recedes ("physical"
    // coordinates). When false, render the comoving frame (expansion hidden).
    bool physicalView = false;

    // Camera "exposure": multiplies particle brightness on output. Lifts faint
    // diffuse structure (and the dispersed physical-view web) out of the dark.
    float renderExposure = 1.0f;
};

} // namespace cosmico
