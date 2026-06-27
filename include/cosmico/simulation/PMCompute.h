#pragma once
#include <cosmico/simulation/PMParams.h>
#include <vector>
#include <cstdint>
#include <fstream>

// Forward declare CUDA/cuFFT types
typedef int cufftHandle;
typedef struct CUstream_st* cudaStream_t;
typedef struct CUevent_st* cudaEvent_t;

namespace cosmico {

struct PMStats {
    float kernelTimeMs = 0.0f;
    double kineticEnergy = 0.0;
    double potentialEnergy = 0.0;
    double totalEnergy = 0.0;
    double momentumMag = 0.0;
};

struct PMStateData {
    double time = 0.0;
    int step = 0;
    double kineticEnergy = 0.0;
    double potentialEnergy = 0.0;
    double totalEnergy = 0.0;
    double momentumMag = 0.0;
    double scaleFactor = 1.0;       // a(t)
    double redshift = 0.0;          // z = 1/a - 1
    double hubble = 0.0;            // H(a)
    int sinkCount = 0;              // Number of active sink particles
    int absorbedCount = 0;          // Total particles absorbed
    std::vector<float> powerSpectrum;
};

class PMCompute {
public:
    void init(int maxParticles, const PMParams& params);
    void destroy();

    // Set the particle buffer pointer (CUDA device pointer from interop)
    void setParticleBuffer(void* cudaParticlePtr, size_t bufferSize);

    // Seed cosmological initial conditions in-place on the particle buffer:
    // a Zel'dovich (1LPT) displacement of a uniform Lagrangian lattice with a
    // P(k) ∝ k^ns·T²(k) Gaussian random field plus growing-mode velocities.
    // Reuses the existing FFT plans and grid buffers. Call once after the
    // lattice has been uploaded to the interop buffer.
    void generateCosmologicalIC(const PMParams& params, void* cudaStream);

    // Execute one simulation step
    void step(const PMParams& params, void* cudaStream);

    // Update particle count
    void updateParticleCount(int count);

    // Change grid size (destroys and recreates FFT plans + grid buffers)
    void updateGrid(int newGridN);

    // Reset simulation state
    void resetState();

    const PMStats& stats() const { return m_stats; }
    const PMStateData& state() const { return m_state; }
    float kernelTimeMs() const { return m_stats.kernelTimeMs; }

    // VRAM usage estimate in bytes
    static size_t estimateVRAM(int gridN, int particleCount);

private:
    void allocateGridBuffers(int N);
    void freeGridBuffers();
    void createFFTPlans(int N);
    void destroyFFTPlans();

    // Isolated BC helpers
    void allocateIsolatedBuffers(int N);
    void freeIsolatedBuffers();
    void precomputeGreenFunction(int N, float cellSize, float G_const, void* cudaStream);

    // Particle buffer (external, from CudaInterop)
    void* m_particlePtr = nullptr;
    size_t m_particleBufferSize = 0;

    // Grid buffers (device pointers stored as void*)
    void* m_d_density = nullptr;       // float[N³] — reused as potential after FFT
    void* m_d_density_hat = nullptr;   // cufftComplex[N*N*(N/2+1)]
    void* m_d_forceX = nullptr;        // float[N³]
    void* m_d_forceY = nullptr;        // float[N³]
    void* m_d_forceZ = nullptr;        // float[N³]

    // Reduction buffers
    void* m_d_energySums = nullptr;    // double[6]

    // Power spectrum buffers
    void* m_d_spectrum = nullptr;      // float[nBins]
    void* m_d_counts = nullptr;        // int[nBins]

    // cuFFT plans (N³ periodic)
    cufftHandle m_planR2C = 0;
    cufftHandle m_planC2R = 0;
    bool m_fftPlansCreated = false;

    // Isolated (non-periodic) boundary condition buffers
    void* m_d_paddedDensity = nullptr;   // float[(2N)³]
    void* m_d_paddedComplex = nullptr;   // cufftComplex[2N*2N*(N+1)]
    void* m_d_greenHat = nullptr;        // cufftComplex[2N*2N*(N+1)]
    cufftHandle m_planR2C_2N = 0;
    cufftHandle m_planC2R_2N = 0;
    bool m_isolatedPlansCreated = false;
    bool m_greenPrecomputed = false;
    int m_isolatedGridN = 0;             // N for which isolated buffers were allocated

    // CUDA events for timing
    void* m_startEvent = nullptr;
    void* m_stopEvent = nullptr;

    // State
    int m_maxParticles = 0;
    int m_particleCount = 0;
    int m_currentGridN = 0;
    int m_nBins = 64;

    float m_scaleFactor = 1.0f;

    PMStats m_stats;
    PMStateData m_state;

    // Sink particle buffers
    static constexpr int MAX_SINKS = 256;  // cap so accretion stays O(N·nSinks)
    void* m_d_sinkCount = nullptr;         // int[1] — sink list size
    void* m_d_newParticleCount = nullptr;  // int[1] — device counter for absorbed particles
    void* m_d_sinkList = nullptr;          // int[MAX_SINKS] — compact sink indices

    // Host staging
    double m_h_energySums[6] = {};
    std::vector<float> m_h_spectrum;
    std::vector<int> m_h_counts;

    // Scientific diagnostic log (CSV): σ_δ growth vs linear theory D(a), virial,
    // energies — written next to the executable as pm_science.csv.
    void* m_d_gridSums = nullptr;       // double[2] — Σρ, Σρ²
    std::ofstream m_sciLog;
    bool m_sciLogOpen = false;
    double m_sigmaInit = -1.0;          // σ_δ at first sample (reference)
    double m_growthInit = -1.0;         // D(a) at first sample (reference)
};

} // namespace cosmico
