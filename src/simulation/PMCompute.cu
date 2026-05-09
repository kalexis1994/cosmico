#include <cosmico/simulation/PMCompute.h>
#include <kernels/pm_gravity.cuh>
#include <cuda_runtime.h>
#include <cufft.h>
#include <stdexcept>
#include <cstdio>
#include <cmath>
#include <string>
#include <algorithm>
#include <cstring>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "[PM] CUDA Error: %s at %s:%d\n", \
                    cudaGetErrorString(err), __FILE__, __LINE__); \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err)); \
        } \
    } while(0)

#define CUFFT_CHECK(call) \
    do { \
        cufftResult err = (call); \
        if (err != CUFFT_SUCCESS) { \
            fprintf(stderr, "[PM] cuFFT Error: %d at %s:%d\n", \
                    (int)err, __FILE__, __LINE__); \
            throw std::runtime_error("cuFFT error: " + std::to_string((int)err)); \
        } \
    } while(0)

namespace cosmico {

// ─── VRAM estimation ────────────────────────────────────────────────────

size_t PMCompute::estimateVRAM(int gridN, int particleCount) {
    size_t N3 = (size_t)gridN * gridN * gridN;
    size_t Ncomplex = (size_t)gridN * gridN * (gridN / 2 + 1);

    size_t total = 0;
    total += N3 * sizeof(float);              // density/potential (reused)
    total += Ncomplex * 2 * sizeof(float);    // density_hat (cufftComplex)
    total += N3 * sizeof(float) * 3;          // forceX, forceY, forceZ
    total += 6 * sizeof(double);              // energy sums
    total += 128 * sizeof(float);             // power spectrum
    total += 128 * sizeof(int);               // power counts
    total += (size_t)particleCount * 48;      // particles (ParticleGpu)
    // Isolated BC adds ~8× FFT grid overhead (padded density, complex, Green hat)
    // Not included here since it's allocated on demand
    return total;
}

// ─── Memory management ──────────────────────────────────────────────────

void PMCompute::allocateGridBuffers(int N) {
    // VRAM guard
    size_t needed = estimateVRAM(N, m_maxParticles);
    size_t freeMem = 0, totalMem = 0;
    cudaMemGetInfo(&freeMem, &totalMem);

    if (needed > freeMem) {
        fprintf(stderr, "[PM] ERROR: N=%d requires %zu MB but only %zu MB free!\n",
                N, needed / (1024 * 1024), freeMem / (1024 * 1024));
        throw std::runtime_error("Insufficient VRAM for PM grid size " + std::to_string(N));
    }
    if (needed > totalMem * 80 / 100) {
        fprintf(stderr, "[PM] WARNING: N=%d uses %zu MB (%.0f%% of %zu MB total VRAM)\n",
                N, needed / (1024 * 1024),
                100.0 * needed / totalMem, totalMem / (1024 * 1024));
    }

    size_t N3 = (size_t)N * N * N;
    size_t Ncomplex = (size_t)N * N * (N / 2 + 1);

    CUDA_CHECK(cudaMalloc(&m_d_density, N3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&m_d_density_hat, Ncomplex * sizeof(cufftComplex)));
    CUDA_CHECK(cudaMalloc(&m_d_forceX, N3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&m_d_forceY, N3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&m_d_forceZ, N3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&m_d_energySums, 6 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&m_d_spectrum, m_nBins * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&m_d_counts, m_nBins * sizeof(int)));

    // Sink particle counters
    CUDA_CHECK(cudaMalloc(&m_d_sinkCount, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&m_d_newParticleCount, sizeof(int)));

    m_currentGridN = N;

    fprintf(stderr, "[PM] Allocated grid buffers: N=%d, ~%zu MB\n",
            N, needed / (1024 * 1024));
}

void PMCompute::freeGridBuffers() {
    if (m_d_density) { cudaFree(m_d_density); m_d_density = nullptr; }
    if (m_d_density_hat) { cudaFree(m_d_density_hat); m_d_density_hat = nullptr; }
    if (m_d_forceX) { cudaFree(m_d_forceX); m_d_forceX = nullptr; }
    if (m_d_forceY) { cudaFree(m_d_forceY); m_d_forceY = nullptr; }
    if (m_d_forceZ) { cudaFree(m_d_forceZ); m_d_forceZ = nullptr; }
    if (m_d_energySums) { cudaFree(m_d_energySums); m_d_energySums = nullptr; }
    if (m_d_spectrum) { cudaFree(m_d_spectrum); m_d_spectrum = nullptr; }
    if (m_d_counts) { cudaFree(m_d_counts); m_d_counts = nullptr; }
    if (m_d_sinkCount) { cudaFree(m_d_sinkCount); m_d_sinkCount = nullptr; }
    if (m_d_newParticleCount) { cudaFree(m_d_newParticleCount); m_d_newParticleCount = nullptr; }
    freeIsolatedBuffers();
    m_currentGridN = 0;
}

// ─── Isolated (non-periodic) BC buffer management ───────────────────────

void PMCompute::allocateIsolatedBuffers(int N) {
    if (m_isolatedGridN == N && m_isolatedPlansCreated) return;

    freeIsolatedBuffers();

    int N2 = 2 * N;
    size_t N2_3 = (size_t)N2 * N2 * N2;
    size_t Ncomplex_2N = (size_t)N2 * N2 * (N + 1);  // 2N * 2N * (2N/2+1) = 2N*2N*(N+1)

    CUDA_CHECK(cudaMalloc(&m_d_paddedDensity, N2_3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&m_d_paddedComplex, Ncomplex_2N * sizeof(cufftComplex)));
    CUDA_CHECK(cudaMalloc(&m_d_greenHat, Ncomplex_2N * sizeof(cufftComplex)));

    CUFFT_CHECK(cufftPlan3d(&m_planR2C_2N, N2, N2, N2, CUFFT_R2C));
    CUFFT_CHECK(cufftPlan3d(&m_planC2R_2N, N2, N2, N2, CUFFT_C2R));
    m_isolatedPlansCreated = true;
    m_isolatedGridN = N;
    m_greenPrecomputed = false;

    fprintf(stderr, "[PM] Allocated isolated BC buffers: 2N=%d, padded ~%zu MB\n",
            N2, (N2_3 * sizeof(float) + 2 * Ncomplex_2N * sizeof(cufftComplex)) / (1024 * 1024));
}

void PMCompute::freeIsolatedBuffers() {
    if (m_d_paddedDensity) { cudaFree(m_d_paddedDensity); m_d_paddedDensity = nullptr; }
    if (m_d_paddedComplex) { cudaFree(m_d_paddedComplex); m_d_paddedComplex = nullptr; }
    if (m_d_greenHat) { cudaFree(m_d_greenHat); m_d_greenHat = nullptr; }
    if (m_isolatedPlansCreated) {
        cufftDestroy(m_planR2C_2N);
        cufftDestroy(m_planC2R_2N);
        m_isolatedPlansCreated = false;
    }
    m_isolatedGridN = 0;
    m_greenPrecomputed = false;
}

void PMCompute::precomputeGreenFunction(int N, float cellSize, float G_const,
                                          void* cudaStream) {
    cudaStream_t stream = static_cast<cudaStream_t>(cudaStream);

    // Compute G(r) on real (2N)³ grid into m_d_paddedDensity (temporary)
    cuda::launchPMComputeGreenFunction(static_cast<float*>(m_d_paddedDensity),
                                        N, cellSize, G_const, stream);

    // FFT G(r) → Ĝ(k), store in m_d_greenHat
    CUFFT_CHECK(cufftSetStream(m_planR2C_2N, stream));
    CUFFT_CHECK(cufftExecR2C(m_planR2C_2N,
                              static_cast<cufftReal*>(m_d_paddedDensity),
                              static_cast<cufftComplex*>(m_d_greenHat)));

    m_greenPrecomputed = true;
    fprintf(stderr, "[PM] Precomputed isolated Green's function: N=%d, G=%.3f\n", N, G_const);
}

void PMCompute::createFFTPlans(int N) {
    CUFFT_CHECK(cufftPlan3d(&m_planR2C, N, N, N, CUFFT_R2C));
    CUFFT_CHECK(cufftPlan3d(&m_planC2R, N, N, N, CUFFT_C2R));
    m_fftPlansCreated = true;
    fprintf(stderr, "[PM] Created FFT plans: N=%d\n", N);
}

void PMCompute::destroyFFTPlans() {
    if (m_fftPlansCreated) {
        cufftDestroy(m_planR2C);
        cufftDestroy(m_planC2R);
        m_fftPlansCreated = false;
    }
}

// ─── Init / Destroy ─────────────────────────────────────────────────────

void PMCompute::init(int maxParticles, const PMParams& params) {
    m_maxParticles = maxParticles;
    m_particleCount = params.particleCount;

    // Allocate grid buffers and FFT plans
    allocateGridBuffers(params.gridN);
    createFFTPlans(params.gridN);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    m_startEvent = start;
    m_stopEvent = stop;

    // Init host staging
    m_h_spectrum.resize(m_nBins, 0.0f);
    m_h_counts.resize(m_nBins, 0);

    resetState();

    fprintf(stderr, "[PM] Initialized: max %d particles, grid %d^3\n",
            maxParticles, params.gridN);
}

void PMCompute::destroy() {
    if (m_startEvent) {
        cudaEventDestroy(static_cast<cudaEvent_t>(m_startEvent));
        m_startEvent = nullptr;
    }
    if (m_stopEvent) {
        cudaEventDestroy(static_cast<cudaEvent_t>(m_stopEvent));
        m_stopEvent = nullptr;
    }
    destroyFFTPlans();
    freeGridBuffers();
}

void PMCompute::setParticleBuffer(void* cudaParticlePtr, size_t bufferSize) {
    m_particlePtr = cudaParticlePtr;
    m_particleBufferSize = bufferSize;
}

void PMCompute::updateParticleCount(int count) {
    m_particleCount = count;
}

void PMCompute::updateGrid(int newGridN) {
    if (newGridN == m_currentGridN) return;

    cudaDeviceSynchronize();
    destroyFFTPlans();
    freeGridBuffers();
    allocateGridBuffers(newGridN);
    createFFTPlans(newGridN);
}

void PMCompute::resetState() {
    m_state.time = 0.0;
    m_state.step = 0;
    m_state.kineticEnergy = 0.0;
    m_state.potentialEnergy = 0.0;
    m_state.totalEnergy = 0.0;
    m_state.momentumMag = 0.0;
    m_state.scaleFactor = 1.0;
    m_state.redshift = 0.0;
    m_state.hubble = 0.0;
    m_state.powerSpectrum.clear();
    m_state.sinkCount = 0;
    m_state.absorbedCount = 0;
    m_scaleFactor = 1.0f;  // Will be set to aInit on first comoving step
    memset(m_h_energySums, 0, sizeof(m_h_energySums));
}

// ─── Step: full PM pipeline ─────────────────────────────────────────────

void PMCompute::step(const PMParams& params, void* cudaStream) {
    if (!m_particlePtr || m_particleCount == 0) return;

    // Update grid if needed
    if (params.gridN != m_currentGridN) {
        updateGrid(params.gridN);
    }

    // Initialize scale factor on first step or when comoving toggled
    if (params.comoving && m_state.step == 0) {
        m_scaleFactor = params.aInit;
    }

    cudaStream_t stream = static_cast<cudaStream_t>(cudaStream);
    auto* particles = static_cast<cuda::ParticleGpu*>(m_particlePtr);

    int N = m_currentGridN;
    int N3 = N * N * N;
    int Ncomplex = N * N * (N / 2 + 1);
    float cellSize = params.boxSize / (float)N;

    // Gaussian smoothing radius: 2x inter-particle spacing to suppress shot noise
    // and prevent small-scale gravitational collapse
    float smoothRadius = 2.0f * params.boxSize / std::cbrt((float)m_particleCount);

    static constexpr float PI_F = 3.14159265358979323846f;

    // Set FFT stream
    CUFFT_CHECK(cufftSetStream(m_planR2C, stream));
    CUFFT_CHECK(cufftSetStream(m_planC2R, stream));

    // Record start event
    CUDA_CHECK(cudaEventRecord(static_cast<cudaEvent_t>(m_startEvent), stream));

    for (int sub = 0; sub < params.stepsPerFrame; sub++) {

        // ── Compute effective parameters ──
        float G_eff, dampFactor, driftDt, halfDriftDt;
        float H_current = 0.0f;

        if (params.comoving) {
            // ── Friedmann equation: H(a) = H₀ √(Ω_m/a³ + Ω_Λ) ──
            float a = m_scaleFactor;
            float OmegaL = std::max(0.0f, 1.0f - params.OmegaM); // flat universe
            float H2 = params.H0 * params.H0 *
                        (params.OmegaM / (a * a * a) + OmegaL);
            H_current = std::sqrt(std::max(H2, 0.0f));

            // ── Peculiar velocity formulation: u = a·ẋ ──
            //   Equation of motion: du/dt = -H·u - ∇φ
            //   Poisson: ∇²φ = (3/2)H₀²Ω_m·δ/a
            //   → G_eff = 3H₀²Ω_m / (8πρ̄·a)  (single 1/a factor)
            //   Drift: dx = u·dt/a
            //   Drag: dampFactor = 1 - H·dt  (Hubble friction)
            float L = params.boxSize;
            float rhoMean = (float)m_particleCount / (L * L * L);
            float G_cosmo = (3.0f * params.H0 * params.H0 * params.OmegaM)
                          / (8.0f * PI_F * rhoMean);

            G_eff = G_cosmo / a;  // Single 1/a from Poisson equation

            // Hubble drag on peculiar velocity: du/dt includes -H·u
            // Combined with pressure support / velocity dispersion (damping)
            // dampFactor = (1 - H·dt) × exp(-γ·dt)
            float hubbleDrag = std::max(0.0f, 1.0f - H_current * params.dt);
            float pressureDamp = std::exp(-params.damping * params.dt);
            dampFactor = hubbleDrag * pressureDamp;

            // Comoving drift: dx = u·dt/a  (u is peculiar velocity)
            driftDt = params.dt / a;
            halfDriftDt = driftDt * 0.5f;
        } else {
            G_eff = params.G;
            dampFactor = std::exp(-params.damping * params.dt);
            driftDt = params.dt;
            halfDriftDt = params.dt * 0.5f;
        }

        // ── Leapfrog DKD: D(dt/2) → forces → K(dt) → D(dt/2) ──
        // Open boundary: pass huge box so drift never wraps
        float driftBox = params.openBoundary ? 1e18f : params.boxSize;

        // ── 1. Half drift ──
        cuda::launchPMDrift(particles, m_particleCount, halfDriftDt,
                             driftBox, stream);

        // ── 2. Clear density grid ──
        cuda::launchPMClearGrid(static_cast<float*>(m_d_density), N3, stream);

        // ── 3. CIC deposit: particles → density grid ──
        cuda::launchPMDeposit(particles, static_cast<float*>(m_d_density),
                              m_particleCount, N, params.boxSize,
                              params.openBoundary, stream);

        // ── 3.5 Sink formation: create sinks in high-density cells ──
        if (params.enableSink) {
            cuda::launchPMSinkFormation(particles,
                                         static_cast<float*>(m_d_density),
                                         m_particleCount, N, params.boxSize,
                                         params.sinkDensityThreshold,
                                         params.sinkRadius * cellSize,
                                         static_cast<int*>(m_d_sinkCount),
                                         static_cast<int*>(m_d_newParticleCount),
                                         stream);
        }

        if (params.openBoundary) {
            // ── Isolated BC pipeline: zero-pad → FFT(2N) → convolve Ĝ → IFFT(2N) → extract ──

            // Ensure isolated buffers and Green's function are ready
            allocateIsolatedBuffers(N);
            if (!m_greenPrecomputed) {
                precomputeGreenFunction(N, cellSize, G_eff, stream);
            }

            int N2 = 2 * N;
            size_t N2_3 = (size_t)N2 * N2 * N2;
            int Ncomplex_2N = N2 * N2 * (N + 1);

            CUFFT_CHECK(cufftSetStream(m_planR2C_2N, stream));
            CUFFT_CHECK(cufftSetStream(m_planC2R_2N, stream));

            // Clear padded grid
            cuda::launchPMClearGrid(static_cast<float*>(m_d_paddedDensity),
                                     static_cast<int>(N2_3), stream);

            // Copy N³ density into corner of (2N)³
            cuda::launchPMZeroPadDensity(static_cast<float*>(m_d_density),
                                          static_cast<float*>(m_d_paddedDensity),
                                          N, stream);

            // FFT forward on (2N)³
            CUFFT_CHECK(cufftExecR2C(m_planR2C_2N,
                                      static_cast<cufftReal*>(m_d_paddedDensity),
                                      static_cast<cufftComplex*>(m_d_paddedComplex)));

            // Multiply ρ̂ × Ĝ (element-wise convolution)
            cuda::launchPMGreenMultiplyIsolated(
                static_cast<cufftComplex*>(m_d_paddedComplex),
                static_cast<cufftComplex*>(m_d_greenHat),
                Ncomplex_2N, stream);

            // FFT inverse on (2N)³
            CUFFT_CHECK(cufftExecC2R(m_planC2R_2N,
                                      static_cast<cufftComplex*>(m_d_paddedComplex),
                                      static_cast<cufftReal*>(m_d_paddedDensity)));

            // Extract N³ potential from padded result back into m_d_density
            cuda::launchPMExtractPotential(static_cast<float*>(m_d_paddedDensity),
                                            static_cast<float*>(m_d_density),
                                            N, stream);
        } else {
            // ── Periodic pipeline ──

            // ── 4. FFT forward: ρ(x) → ρ̂(k) ──
            CUFFT_CHECK(cufftExecR2C(m_planR2C,
                                      static_cast<cufftReal*>(m_d_density),
                                      static_cast<cufftComplex*>(m_d_density_hat)));

            // ── 5. Green's function with G_eff (includes 1/a² in comoving) ──
            cuda::launchPMGreenMultiply(static_cast<cufftComplex*>(m_d_density_hat),
                                         N, Ncomplex, G_eff, cellSize,
                                         smoothRadius, stream);

            // ── 6. FFT inverse: Φ̂(k) → Φ(x) ──
            CUFFT_CHECK(cufftExecC2R(m_planC2R,
                                      static_cast<cufftComplex*>(m_d_density_hat),
                                      static_cast<cufftReal*>(m_d_density)));
        }

        // ── 7. Gradient: Φ → (Fx, Fy, Fz) ──
        cuda::launchPMGradient(static_cast<float*>(m_d_density),
                                static_cast<float*>(m_d_forceX),
                                static_cast<float*>(m_d_forceY),
                                static_cast<float*>(m_d_forceZ),
                                N, cellSize, params.openBoundary, stream);

        // ── 8. CIC interpolation: grid forces → particle acceleration ──
        cuda::launchPMInterpolate(particles,
                                   static_cast<float*>(m_d_forceX),
                                   static_cast<float*>(m_d_forceY),
                                   static_cast<float*>(m_d_forceZ),
                                   m_particleCount, N, params.boxSize,
                                   params.openBoundary, stream);

        // ── 9. Full kick: v = v*dampFactor + F*dt ──
        // Velocity cap: limit displacement per step (disabled for open boundary)
        float maxVel = params.openBoundary ? 1e18f
                     : params.boxSize * 0.05f / std::max(driftDt, 1e-6f);
        cuda::launchPMKick(particles, m_particleCount, params.dt,
                            dampFactor, false, 0.0f, maxVel, stream);

        // ── 9.5 Sink accretion: absorb particles near sinks ──
        if (params.enableSink) {
            cuda::launchPMSinkAccretion(particles, m_particleCount,
                                         params.sinkRadius * cellSize,
                                         params.boxSize,
                                         static_cast<int*>(m_d_sinkCount),
                                         static_cast<int*>(m_d_newParticleCount),
                                         stream);
        }

        // ── 10. Half drift ──
        cuda::launchPMDrift(particles, m_particleCount, halfDriftDt,
                             driftBox, stream);

        // ── 11. Momentum correction ──
        if (params.correctMomentum) {
            cuda::launchPMEnergyReduction(particles,
                                           static_cast<float*>(m_d_density),
                                           m_particleCount, N, params.boxSize,
                                           static_cast<double*>(m_d_energySums),
                                           stream);
            cuda::launchPMMomentumCorrection(particles, m_particleCount,
                                              static_cast<double*>(m_d_energySums),
                                              stream);
        }

        // ── 12. Evolve scale factor (Friedmann): da/dt = a·H(a) ──
        if (params.comoving) {
            float da = m_scaleFactor * H_current * params.dt;
            m_scaleFactor += da;
        }

        m_state.time += params.dt;
        m_state.step++;
    }

    // Update cosmological state
    if (params.comoving) {
        m_state.scaleFactor = m_scaleFactor;
        m_state.redshift = (m_scaleFactor > 1e-6) ? (1.0 / m_scaleFactor - 1.0) : 1e6;
        float OmegaL = std::max(0.0f, 1.0f - params.OmegaM);
        float a = m_scaleFactor;
        m_state.hubble = params.H0 * std::sqrt(
            params.OmegaM / (a * a * a) + OmegaL);
    } else {
        m_state.scaleFactor = 1.0;
        m_state.redshift = 0.0;
        m_state.hubble = 0.0;
    }

    // ── Energy computation (once per frame, after all sub-steps) ──
    cuda::launchPMEnergyReduction(particles,
                                   static_cast<float*>(m_d_density),
                                   m_particleCount, N, params.boxSize,
                                   static_cast<double*>(m_d_energySums),
                                   stream);

    // Async readback of energy sums
    CUDA_CHECK(cudaMemcpyAsync(m_h_energySums, m_d_energySums,
                                6 * sizeof(double), cudaMemcpyDeviceToHost, stream));

    // ── Power spectrum (once per frame) ──
    if (params.showPowerSpectrum) {
        // Need density in Fourier space — redo deposit + FFT for spectrum
        cuda::launchPMClearGrid(static_cast<float*>(m_d_forceX), N3, stream); // reuse forceX
        cuda::launchPMDeposit(particles, static_cast<float*>(m_d_forceX),
                              m_particleCount, N, params.boxSize,
                              params.openBoundary, stream);

        // Use C2R plan's input for spectrum — store in density_hat
        CUFFT_CHECK(cufftExecR2C(m_planR2C,
                                  static_cast<cufftReal*>(m_d_forceX),
                                  static_cast<cufftComplex*>(m_d_density_hat)));

        float dk = 2.0f * 3.14159265f / params.boxSize;
        cuda::launchPMPowerSpectrum(static_cast<cufftComplex*>(m_d_density_hat),
                                     N, Ncomplex,
                                     static_cast<float*>(m_d_spectrum),
                                     static_cast<int*>(m_d_counts),
                                     m_nBins, dk, stream);

        CUDA_CHECK(cudaMemcpyAsync(m_h_spectrum.data(), m_d_spectrum,
                                    m_nBins * sizeof(float), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(m_h_counts.data(), m_d_counts,
                                    m_nBins * sizeof(int), cudaMemcpyDeviceToHost, stream));
    }

    // Record stop event
    CUDA_CHECK(cudaEventRecord(static_cast<cudaEvent_t>(m_stopEvent), stream));

    // Query timing (non-blocking, from previous frame)
    float ms = 0.0f;
    cudaError_t err = cudaEventQuery(static_cast<cudaEvent_t>(m_stopEvent));
    if (err == cudaSuccess) {
        cudaEventElapsedTime(&ms, static_cast<cudaEvent_t>(m_startEvent),
                             static_cast<cudaEvent_t>(m_stopEvent));
        m_stats.kernelTimeMs = ms;
    }

    // Update state from readback (uses previous frame's data — 1 frame latency)
    m_stats.kineticEnergy = m_h_energySums[0];
    m_stats.potentialEnergy = m_h_energySums[1];
    m_stats.totalEnergy = m_h_energySums[0] + m_h_energySums[1];
    double px = m_h_energySums[2], py = m_h_energySums[3], pz = m_h_energySums[4];
    m_stats.momentumMag = std::sqrt(px * px + py * py + pz * pz);

    m_state.kineticEnergy = m_stats.kineticEnergy;
    m_state.potentialEnergy = m_stats.potentialEnergy;
    m_state.totalEnergy = m_stats.totalEnergy;
    m_state.momentumMag = m_stats.momentumMag;

    // Readback sink counters
    if (params.enableSink) {
        int h_sinkCount = 0, h_absorbedCount = 0;
        CUDA_CHECK(cudaMemcpyAsync(&h_sinkCount, m_d_sinkCount,
                                    sizeof(int), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(&h_absorbedCount, m_d_newParticleCount,
                                    sizeof(int), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        m_state.sinkCount = h_sinkCount;
        m_state.absorbedCount = h_absorbedCount;
    }

    // Build averaged power spectrum
    if (params.showPowerSpectrum) {
        m_state.powerSpectrum.resize(m_nBins);
        for (int i = 0; i < m_nBins; i++) {
            m_state.powerSpectrum[i] = (m_h_counts[i] > 0)
                ? m_h_spectrum[i] / (float)m_h_counts[i]
                : 0.0f;
        }
    }
}

} // namespace cosmico
