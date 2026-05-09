#include <cosmico/simulation/InflationCompute.h>
#include <kernels/inflation.cuh>
#include <kernels/density_extract.cuh>
#include <kernels/cmb_extract.cuh>
#include <kernels/zeldovich.cuh>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cstdio>
#include <cmath>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <cstring>
#include <chrono>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "[Inflation] CUDA Error: %s at %s:%d\n", \
                    cudaGetErrorString(err), __FILE__, __LINE__); \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err)); \
        } \
    } while(0)

#define CUFFT_CHECK(call) \
    do { \
        cufftResult err = (call); \
        if (err != CUFFT_SUCCESS) { \
            fprintf(stderr, "[Inflation] cuFFT Error: %d at %s:%d\n", \
                    (int)err, __FILE__, __LINE__); \
            throw std::runtime_error("cuFFT error: " + std::to_string((int)err)); \
        } \
    } while(0)

namespace cosmico {

// ─── Helper: build GPU params from CPU params + current state ───────────

static cuda::InflationGpuParams buildGpuParams(const InflationParams& p, int N, double a) {
    cuda::InflationGpuParams gp{};
    gp.N = N;
    gp.N3 = N * N * N;
    gp.Ncomplex = N * N * (N / 2 + 1);
    gp.L = p.boxSize;
    gp.dk = 2.0f * 3.14159265f / p.boxSize;
    gp.dt = p.dt;
    gp.a = (float)a;
    gp.a3 = (float)(a * a * a);
    gp.Mpl = p.Mpl;
    gp.potentialType = static_cast<int>(p.potential);
    gp.m2 = p.m2;
    gp.lambda4 = p.lambda4;
    gp.mu = p.mu;
    return gp;
}

// ─── VRAM estimation ────────────────────────────────────────────────────

size_t InflationCompute::estimateVRAM(int gridN) {
    size_t N3 = (size_t)gridN * gridN * gridN;
    size_t Ncomplex = (size_t)gridN * gridN * (gridN / 2 + 1);

    size_t total = 0;
    total += N3 * sizeof(float);             // phi
    total += N3 * sizeof(float);             // pi
    total += N3 * sizeof(float);             // laplacian
    total += Ncomplex * 2 * sizeof(float);   // phi_hat (cufftComplex)
    total += 6 * sizeof(double);             // energy sums
    total += 128 * sizeof(float);            // power spectrum
    total += 128 * sizeof(int);              // power counts
    return total;
}

// ─── Memory management ──────────────────────────────────────────────────

void InflationCompute::allocateField(int N) {
    // VRAM guard: check if we have enough memory
    size_t needed = estimateVRAM(N);
    size_t freeMem = 0, totalMem = 0;
    cudaMemGetInfo(&freeMem, &totalMem);

    // Warn if using >80% of VRAM; abort if we'd exceed free memory
    if (needed > freeMem) {
        fprintf(stderr, "[Inflation] ERROR: N=%d requires %zu MB but only %zu MB free!\n",
                N, needed / (1024 * 1024), freeMem / (1024 * 1024));
        throw std::runtime_error("Insufficient VRAM for grid size " + std::to_string(N));
    }
    if (needed > totalMem * 80 / 100) {
        fprintf(stderr, "[Inflation] WARNING: N=%d uses %zu MB (%.0f%% of %zu MB total VRAM)\n",
                N, needed / (1024 * 1024),
                100.0 * needed / totalMem, totalMem / (1024 * 1024));
    }

    size_t N3 = (size_t)N * N * N;
    size_t Ncomplex = (size_t)N * N * (N / 2 + 1);

    CUDA_CHECK(cudaMalloc(&m_d_phi, N3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&m_d_pi, N3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&m_d_laplacian, N3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&m_d_phi_hat, Ncomplex * sizeof(cufftComplex)));

    CUDA_CHECK(cudaMalloc(&m_d_energySums, 6 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&m_d_powerSpectrum, m_nBins * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&m_d_powerCounts, m_nBins * sizeof(int)));

    m_currentN = N;
    fprintf(stderr, "[Inflation] Allocated field: N=%d, VRAM ~%zu MB (free: %zu MB)\n",
            N, needed / (1024 * 1024), freeMem / (1024 * 1024));
}

void InflationCompute::freeField() {
    if (m_d_phi) { cudaFree(m_d_phi); m_d_phi = nullptr; }
    if (m_d_pi) { cudaFree(m_d_pi); m_d_pi = nullptr; }
    if (m_d_laplacian) { cudaFree(m_d_laplacian); m_d_laplacian = nullptr; }
    if (m_d_phi_hat) { cudaFree(m_d_phi_hat); m_d_phi_hat = nullptr; }
    if (m_d_energySums) { cudaFree(m_d_energySums); m_d_energySums = nullptr; }
    if (m_d_powerSpectrum) { cudaFree(m_d_powerSpectrum); m_d_powerSpectrum = nullptr; }
    if (m_d_powerCounts) { cudaFree(m_d_powerCounts); m_d_powerCounts = nullptr; }
    if (m_d_density) { cudaFree(m_d_density); m_d_density = nullptr; }
    if (m_d_minVal) { cudaFree(m_d_minVal); m_d_minVal = nullptr; }
    if (m_d_maxVal) { cudaFree(m_d_maxVal); m_d_maxVal = nullptr; }
    if (m_d_cmbMap) { cudaFree(m_d_cmbMap); m_d_cmbMap = nullptr; }
    if (m_d_cmbStats) { cudaFree(m_d_cmbStats); m_d_cmbStats = nullptr; }
    m_cmbMapW = 0;
    m_cmbMapH = 0;
    if (m_d_sx) { cudaFree(m_d_sx); m_d_sx = nullptr; }
    if (m_d_sy) { cudaFree(m_d_sy); m_d_sy = nullptr; }
    if (m_d_sz) { cudaFree(m_d_sz); m_d_sz = nullptr; }
    if (m_d_zeldovichDensity) { cudaFree(m_d_zeldovichDensity); m_d_zeldovichDensity = nullptr; }
    m_currentN = 0;
    m_densityN = 0;
    m_zeldovichDstN = 0;
    m_zeldovichActive = false;
}

void InflationCompute::createFFTPlans(int N) {
    CUFFT_CHECK(cufftPlan3d(&m_planR2C, N, N, N, CUFFT_R2C));
    CUFFT_CHECK(cufftPlan3d(&m_planC2R, N, N, N, CUFFT_C2R));
    CUFFT_CHECK(cufftSetStream(m_planR2C, m_stream));
    CUFFT_CHECK(cufftSetStream(m_planC2R, m_stream));
    m_fftPlansCreated = true;
}

void InflationCompute::destroyFFTPlans() {
    if (m_fftPlansCreated) {
        cufftDestroy(m_planR2C);
        cufftDestroy(m_planC2R);
        m_fftPlansCreated = false;
    }
}

// ─── Lifecycle ──────────────────────────────────────────────────────────

void InflationCompute::init(const InflationParams& params) {
    // Create stream and events
    CUDA_CHECK(cudaStreamCreate(&m_stream));
    CUDA_CHECK(cudaEventCreate(&m_startEvent));
    CUDA_CHECK(cudaEventCreate(&m_stopEvent));

    // Allocate field and create FFT plans
    allocateField(params.gridN);
    createFFTPlans(params.gridN);

    // Initialize state
    m_state = InflationStateData{};
    m_state.scaleFactor = 1.0;

    // Allocate host staging
    m_h_spectrum.resize(m_nBins);
    m_h_counts.resize(m_nBins);
    m_h_sliceBuffer.resize(params.gridN);

    // Initialize field
    reset(params);

    fprintf(stderr, "[Inflation] Initialized: grid=%d, potential=%d\n",
            params.gridN, (int)params.potential);
}

void InflationCompute::destroy() {
    if (m_stream) {
        cudaStreamSynchronize(m_stream);
    }

    destroyFFTPlans();
    freeField();

    if (m_startEvent) {
        cudaEventDestroy(m_startEvent);
        m_startEvent = nullptr;
    }
    if (m_stopEvent) {
        cudaEventDestroy(m_stopEvent);
        m_stopEvent = nullptr;
    }
    if (m_stream) {
        cudaStreamDestroy(m_stream);
        m_stream = nullptr;
    }
}

void InflationCompute::reset(const InflationParams& params) {
    int N = params.gridN;

    // Reallocate if grid size changed
    if (N != m_currentN) {
        cudaStreamSynchronize(m_stream);
        destroyFFTPlans();
        freeField();
        allocateField(N);
        createFFTPlans(N);
        m_h_sliceBuffer.resize(N);
    }

    // Reset state
    m_state = InflationStateData{};
    m_state.scaleFactor = 1.0;

    auto gp = buildGpuParams(params, N, 1.0);

    // Build FieldGpu struct with device pointers
    cuda::FieldGpu field{};
    field.phi = static_cast<float*>(m_d_phi);
    field.pi = static_cast<float*>(m_d_pi);
    field.laplacian = static_cast<float*>(m_d_laplacian);
    field.phi_hat = static_cast<cufftComplex*>(m_d_phi_hat);

    // Initialize field in real space: phi = phi0 + tiny fluctuations, pi = 0
    auto seed = static_cast<unsigned long long>(
        std::chrono::steady_clock::now().time_since_epoch().count());
    cuda::launchInitFieldKernel(field, gp, params.phi0, seed, m_stream);

    // Compute initial Laplacian for energy calculation
    // Forward FFT: phi(x) -> phi_hat(k)
    CUFFT_CHECK(cufftExecR2C(m_planR2C,
        static_cast<cufftReal*>(m_d_phi),
        static_cast<cufftComplex*>(m_d_phi_hat)));

    // Laplacian in Fourier: phi_hat *= -k^2/N^3
    cuda::launchLaplacianKernel(
        static_cast<cufftComplex*>(m_d_phi_hat), gp, m_stream);

    // Inverse FFT: phi_hat(k) -> laplacian(x)
    CUFFT_CHECK(cufftExecC2R(m_planC2R,
        static_cast<cufftComplex*>(m_d_phi_hat),
        static_cast<cufftReal*>(m_d_laplacian)));

    // Compute energy to get initial Hubble parameter
    cuda::launchEnergyReductionKernel(field, gp,
        static_cast<double*>(m_d_energySums), m_stream);

    // Sync and read back energy
    CUDA_CHECK(cudaStreamSynchronize(m_stream));
    CUDA_CHECK(cudaMemcpy(m_h_energySums, m_d_energySums,
        6 * sizeof(double), cudaMemcpyDeviceToHost));

    int N3 = N * N * N;
    double KE = m_h_energySums[0] / N3;
    double GE = m_h_energySums[1] / N3;
    double PE = m_h_energySums[2] / N3;
    double phiMean = m_h_energySums[3] / N3;

    m_state.kineticEnergy = KE;
    m_state.gradientEnergy = GE;
    m_state.potentialEnergy = PE;
    m_state.phiMean = phiMean;

    double rhoTotal = KE + GE + PE;
    double Mpl = params.Mpl;
    // H^2 = 8*pi / (3*Mpl^2) * rho
    double H2 = 8.0 * 3.14159265358979 / (3.0 * Mpl * Mpl) * rhoTotal;
    m_state.hubble = (H2 > 0) ? sqrt(H2) : 0.0;

    fprintf(stderr, "[Inflation] Reset: <phi>=%.4f, H=%.6e, rho=%.6e\n",
            phiMean, m_state.hubble, rhoTotal);
}

// ─── Physics step ───────────────────────────────────────────────────────

void InflationCompute::step(const InflationParams& params) {
    int N = params.gridN;
    double a = m_state.scaleFactor;

    cuda::FieldGpu field{};
    field.phi = static_cast<float*>(m_d_phi);
    field.pi = static_cast<float*>(m_d_pi);
    field.laplacian = static_cast<float*>(m_d_laplacian);
    field.phi_hat = static_cast<cufftComplex*>(m_d_phi_hat);

    CUDA_CHECK(cudaEventRecord(m_startEvent, m_stream));

    float dt = params.dt;

    for (int s = 0; s < params.stepsPerFrame; s++) {
        auto gp = buildGpuParams(params, N, a);

        // 1. Forward FFT: phi(x) -> phi_hat(k)
        CUFFT_CHECK(cufftExecR2C(m_planR2C,
            static_cast<cufftReal*>(m_d_phi),
            static_cast<cufftComplex*>(m_d_phi_hat)));

        // 2. Laplacian in Fourier: phi_hat *= -k^2/N^3
        cuda::launchLaplacianKernel(
            static_cast<cufftComplex*>(m_d_phi_hat), gp, m_stream);

        // 3. Inverse FFT: phi_hat(k) -> laplacian(x)
        CUFFT_CHECK(cufftExecC2R(m_planC2R,
            static_cast<cufftComplex*>(m_d_phi_hat),
            static_cast<cufftReal*>(m_d_laplacian)));

        // 4. Half kick: pi += (dt/2) * a^3 * (lap/a^2 - V'(phi))
        cuda::launchLeapfrogKickKernel(field, gp, dt * 0.5f, m_stream);

        // 5. Full drift: phi += dt * pi / a^3
        cuda::launchLeapfrogDriftKernel(field, gp, m_stream);

        // 6. Recompute laplacian for second half-kick
        CUFFT_CHECK(cufftExecR2C(m_planR2C,
            static_cast<cufftReal*>(m_d_phi),
            static_cast<cufftComplex*>(m_d_phi_hat)));
        cuda::launchLaplacianKernel(
            static_cast<cufftComplex*>(m_d_phi_hat), gp, m_stream);
        CUFFT_CHECK(cufftExecC2R(m_planC2R,
            static_cast<cufftComplex*>(m_d_phi_hat),
            static_cast<cufftReal*>(m_d_laplacian)));

        // 7. Second half kick
        cuda::launchLeapfrogKickKernel(field, gp, dt * 0.5f, m_stream);

        // Update time
        m_state.time += dt;

        // Update scale factor (a(t+dt) = a(t) * (1 + H * dt))
        a *= (1.0 + m_state.hubble * dt);
        m_state.scaleFactor = a;
        if (a > 0) {
            m_state.efolds = log(a);
        }
    }

    // ─── Energy computation (once per frame) ────────────────────────────

    auto gp = buildGpuParams(params, N, a);

    // Need laplacian in real space for energy computation
    // It's already computed from the last sub-step's second half-kick
    cuda::launchEnergyReductionKernel(field, gp,
        static_cast<double*>(m_d_energySums), m_stream);

    CUDA_CHECK(cudaEventRecord(m_stopEvent, m_stream));

    // Sync and read back
    CUDA_CHECK(cudaStreamSynchronize(m_stream));

    // Timing
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, m_startEvent, m_stopEvent);
    m_kernelTimeMs = ms;

    // Energy readback
    CUDA_CHECK(cudaMemcpy(m_h_energySums, m_d_energySums,
        6 * sizeof(double), cudaMemcpyDeviceToHost));

    int N3 = N * N * N;
    double KE = m_h_energySums[0] / N3;
    double GE = m_h_energySums[1] / N3;
    double PE = m_h_energySums[2] / N3;
    double phiMean = m_h_energySums[3] / N3;

    m_state.kineticEnergy = KE;
    m_state.gradientEnergy = GE;
    m_state.potentialEnergy = PE;
    m_state.phiMean = phiMean;

    // Update Hubble from Friedmann equation
    double rhoTotal = KE + GE + PE;
    updateScaleFactor(rhoTotal, params.Mpl);

    // Extract field slice for visualization
    extractFieldSlice();
}

void InflationCompute::updateScaleFactor(double rhoTotal, double Mpl) {
    // H^2 = 8*pi / (3 * Mpl^2) * rho
    double H2 = 8.0 * 3.14159265358979 / (3.0 * Mpl * Mpl) * rhoTotal;
    m_state.hubble = (H2 > 0) ? sqrt(H2) : 0.0;
}

void InflationCompute::extractFieldSlice() {
    if (m_currentN <= 0) return;

    int N = m_currentN;
    // Extract a 1D slice through the center: phi[N/2, N/2, 0..N-1]
    size_t offset = ((size_t)(N / 2) * N + (N / 2)) * N;
    m_h_sliceBuffer.resize(N);

    cudaMemcpy(m_h_sliceBuffer.data(),
        static_cast<float*>(m_d_phi) + offset,
        N * sizeof(float), cudaMemcpyDeviceToHost);

    m_state.fieldSlice = m_h_sliceBuffer;
}

// ─── Power spectrum ─────────────────────────────────────────────────────

void InflationCompute::computePowerSpectrum(const InflationParams& params) {
    int N = params.gridN;
    auto gp = buildGpuParams(params, N, m_state.scaleFactor);

    // Forward FFT to get phi_hat (needed fresh — step() may have overwritten it)
    CUFFT_CHECK(cufftExecR2C(m_planR2C,
        static_cast<cufftReal*>(m_d_phi),
        static_cast<cufftComplex*>(m_d_phi_hat)));

    float dk = gp.dk;
    float kMin = dk;  // Minimum non-zero k
    float kMax = dk * (N / 2) * 1.732f; // sqrt(3) * kNyquist
    if (kMin <= 0) kMin = 0.01f;

    cuda::launchPowerSpectrumKernel(
        static_cast<cufftComplex*>(m_d_phi_hat), gp,
        static_cast<float*>(m_d_powerSpectrum),
        static_cast<int*>(m_d_powerCounts),
        m_nBins, kMin, kMax, m_stream);

    CUDA_CHECK(cudaStreamSynchronize(m_stream));

    // Read back
    m_h_spectrum.resize(m_nBins);
    m_h_counts.resize(m_nBins);
    CUDA_CHECK(cudaMemcpy(m_h_spectrum.data(), m_d_powerSpectrum,
        m_nBins * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(m_h_counts.data(), m_d_powerCounts,
        m_nBins * sizeof(int), cudaMemcpyDeviceToHost));

    // Normalize by bin count and build k array
    float logKmin = logf(kMin);
    float logKmax = logf(kMax);
    float dlogK = (logKmax - logKmin) / (float)m_nBins;

    m_state.powerSpectrum.resize(m_nBins);
    m_state.kBins.resize(m_nBins);

    for (int i = 0; i < m_nBins; i++) {
        float logK = logKmin + (i + 0.5f) * dlogK;
        m_state.kBins[i] = expf(logK);
        m_state.powerSpectrum[i] = (m_h_counts[i] > 0)
            ? m_h_spectrum[i] / m_h_counts[i]
            : 0.0f;
    }

    // Compute spectral index n_s via linear fit of log P vs log k
    // Use bins in the middle range (avoiding edge effects)
    int fitStart = m_nBins / 4;
    int fitEnd = 3 * m_nBins / 4;
    int fitCount = 0;
    double sumX = 0, sumY = 0, sumXY = 0, sumXX = 0;

    for (int i = fitStart; i < fitEnd; i++) {
        if (m_state.powerSpectrum[i] > 0 && m_state.kBins[i] > 0) {
            double x = log((double)m_state.kBins[i]);
            double y = log((double)m_state.powerSpectrum[i]);
            sumX += x;
            sumY += y;
            sumXY += x * y;
            sumXX += x * x;
            fitCount++;
        }
    }

    if (fitCount > 2) {
        double slope = (fitCount * sumXY - sumX * sumY) /
                       (fitCount * sumXX - sumX * sumX);
        m_state.spectralIndex = (float)(slope + 1.0); // n_s = d ln P / d ln k + 1
    }
}

// ─── Density extraction for volume rendering ────────────────────────────

void InflationCompute::extractDensityVolume(uint8_t* hostDst, int targetN) {
    if (m_currentN <= 0 || !m_d_phi) return;

    // Allocate density device buffers if needed
    if (m_densityN != targetN) {
        if (m_d_density) cudaFree(m_d_density);
        if (m_d_minVal) cudaFree(m_d_minVal);
        if (m_d_maxVal) cudaFree(m_d_maxVal);

        size_t dstN3 = (size_t)targetN * targetN * targetN;
        CUDA_CHECK(cudaMalloc(&m_d_density, dstN3 * sizeof(uint8_t)));
        // d_minVal is reused as double[2] stats buffer inside the kernel (16 bytes)
        CUDA_CHECK(cudaMalloc(&m_d_minVal, 2 * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&m_d_maxVal, sizeof(float))); // unused but kept for cleanup symmetry
        m_densityN = targetN;
    }

    float phiMean = (float)m_state.phiMean;

    cuda::launchDensityExtraction(
        static_cast<const float*>(m_d_phi),
        m_currentN, phiMean,
        static_cast<uint8_t*>(m_d_density), targetN,
        static_cast<float*>(m_d_minVal),
        static_cast<float*>(m_d_maxVal),
        m_stream);

    // Copy result to host
    size_t dstN3 = (size_t)targetN * targetN * targetN;
    CUDA_CHECK(cudaMemcpyAsync(hostDst, m_d_density, dstN3 * sizeof(uint8_t),
                                cudaMemcpyDeviceToHost, m_stream));
    CUDA_CHECK(cudaStreamSynchronize(m_stream));
}

// ─── CMB map extraction ─────────────────────────────────────────────────

void InflationCompute::extractCMBMap(uint8_t* hostDst, int mapW, int mapH,
                                      float radius, float boxSize, float Mpl) {
    if (m_currentN <= 0 || !m_d_phi) return;

    // Allocate CMB device buffers if size changed
    if (m_cmbMapW != mapW || m_cmbMapH != mapH) {
        if (m_d_cmbMap) cudaFree(m_d_cmbMap);
        if (m_d_cmbStats) cudaFree(m_d_cmbStats);

        size_t mapSize = (size_t)mapW * mapH;
        CUDA_CHECK(cudaMalloc(&m_d_cmbMap, mapSize * sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc(&m_d_cmbStats, 2 * sizeof(double)));
        m_cmbMapW = mapW;
        m_cmbMapH = mapH;
    }

    float phiMean = (float)m_state.phiMean;

    cuda::launchCMBExtraction(
        static_cast<const float*>(m_d_phi),
        m_currentN, phiMean, Mpl, radius, boxSize,
        static_cast<uint8_t*>(m_d_cmbMap), mapW, mapH,
        static_cast<double*>(m_d_cmbStats),
        m_stream);

    // Copy result to host
    size_t mapSize = (size_t)mapW * mapH;
    CUDA_CHECK(cudaMemcpyAsync(hostDst, m_d_cmbMap, mapSize * sizeof(uint8_t),
                                cudaMemcpyDeviceToHost, m_stream));
    CUDA_CHECK(cudaStreamSynchronize(m_stream));
}

// ─── Zel'dovich structure formation ─────────────────────────────────────

void InflationCompute::startZeldovich(const InflationParams& params) {
    if (m_currentN <= 0 || !m_d_phi) return;

    int N = m_currentN;
    size_t N3 = (size_t)N * N * N;
    int dstN = 128;

    // Allocate displacement arrays
    if (m_d_sx) cudaFree(m_d_sx);
    if (m_d_sy) cudaFree(m_d_sy);
    if (m_d_sz) cudaFree(m_d_sz);
    if (m_d_zeldovichDensity) cudaFree(m_d_zeldovichDensity);

    CUDA_CHECK(cudaMalloc(&m_d_sx, N3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&m_d_sy, N3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&m_d_sz, N3 * sizeof(float)));

    size_t dstN3 = (size_t)dstN * dstN * dstN;
    CUDA_CHECK(cudaMalloc(&m_d_zeldovichDensity, dstN3 * sizeof(float)));
    m_zeldovichDstN = dstN;

    float dk = 2.0f * 3.14159265f / params.boxSize;
    // Transfer function cutoff: k_eq at ~N/8 modes (suppresses small-scale power)
    float keq = dk * (float)(N / 8);

    fprintf(stderr, "[Zeldovich] Computing displacement field from phi (N=%d)...\n", N);

    // Compute displacement for each direction (x, y, z)
    // Each iteration: FFT phi → multiply by displacement kernel → IFFT → store
    float* outputs[3] = {
        static_cast<float*>(m_d_sx),
        static_cast<float*>(m_d_sy),
        static_cast<float*>(m_d_sz)
    };

    for (int dir = 0; dir < 3; dir++) {
        // Forward FFT: phi(x) → phi_hat(k)
        CUFFT_CHECK(cufftExecR2C(m_planR2C,
            static_cast<cufftReal*>(m_d_phi),
            static_cast<cufftComplex*>(m_d_phi_hat)));

        // Multiply by i * k_d * T(k) / (k² * N³)
        int Ncomplex = N * N * (N / 2 + 1);
        cuda::launchDisplacementMultiply(
            static_cast<cufftComplex*>(m_d_phi_hat),
            N, Ncomplex, dk, params.boxSize, dir, keq, m_stream);

        // Inverse FFT: → displacement component s_d(q)
        CUFFT_CHECK(cufftExecC2R(m_planC2R,
            static_cast<cufftComplex*>(m_d_phi_hat),
            static_cast<cufftReal*>(outputs[dir])));
    }

    CUDA_CHECK(cudaStreamSynchronize(m_stream));

    // Compute RMS displacement for normalization reference
    // Just sample a few values to estimate
    std::vector<float> sample(1000);
    size_t stride = N3 / 1000;
    double sumSq = 0.0;
    // Read back sx to estimate amplitude
    std::vector<float> hostSx(N3);
    CUDA_CHECK(cudaMemcpy(hostSx.data(), m_d_sx, N3 * sizeof(float), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < N3; i += stride) {
        double v = hostSx[i];
        sumSq += v * v;
    }
    m_displacementRms = (float)sqrt(sumSq / (N3 / stride));
    if (m_displacementRms < 1e-20f) m_displacementRms = 1e-20f;

    m_zeldovichD = 0.0f;
    m_zeldovichBoxSize = params.boxSize;
    m_zeldovichActive = true;

    fprintf(stderr, "[Zeldovich] Displacement field computed. RMS=%.6e, keq=%.4f\n",
            m_displacementRms, keq);
}

void InflationCompute::stepZeldovich(const InflationParams& params) {
    if (!m_zeldovichActive) return;

    m_zeldovichD += params.zeldovichSpeed;
    m_zeldovichLogScale = params.zeldovichLogScale;
}

void InflationCompute::extractZeldovichDensity(uint8_t* hostDst, int targetN) {
    (void)targetN; // dstN is fixed at m_zeldovichDstN
    if (!m_zeldovichActive || !m_d_sx) return;

    int N = m_currentN;
    int dstN = targetN;
    size_t dstN3 = (size_t)dstN * dstN * dstN;

    // Clear density grid
    cuda::launchClearGrid(static_cast<float*>(m_d_zeldovichDensity), (int)dstN3, m_stream);

    // CIC deposit with current growth factor
    // Scale D so that D=1 gives ~1 grid cell of displacement
    float effectiveD = m_zeldovichD;

    cuda::launchZeldovichDeposit(
        static_cast<const float*>(m_d_sx),
        static_cast<const float*>(m_d_sy),
        static_cast<const float*>(m_d_sz),
        static_cast<float*>(m_d_zeldovichDensity),
        N, dstN, effectiveD, m_zeldovichBoxSize,
        m_stream);

    // Mean density: srcN³ particles deposited into dstN³ cells
    float meanDensity = (float)(N * N * N) / (float)(dstN * dstN * dstN);

    // Ensure density output buffer exists
    if (m_densityN != targetN) {
        if (m_d_density) cudaFree(m_d_density);
        CUDA_CHECK(cudaMalloc(&m_d_density, dstN3 * sizeof(uint8_t)));
        if (!m_d_minVal) CUDA_CHECK(cudaMalloc(&m_d_minVal, 2 * sizeof(double)));
        if (!m_d_maxVal) CUDA_CHECK(cudaMalloc(&m_d_maxVal, sizeof(float)));
        m_densityN = targetN;
    }

    // Normalize to uint8 with log scaling
    float logScale = m_zeldovichLogScale;
    cuda::launchNormalizeDensity(
        static_cast<const float*>(m_d_zeldovichDensity),
        static_cast<uint8_t*>(m_d_density),
        (int)dstN3, meanDensity, logScale, m_stream);

    // Copy to host
    CUDA_CHECK(cudaMemcpyAsync(hostDst, m_d_density, dstN3 * sizeof(uint8_t),
                                cudaMemcpyDeviceToHost, m_stream));
    CUDA_CHECK(cudaStreamSynchronize(m_stream));
}

void InflationCompute::stopZeldovich() {
    m_zeldovichActive = false;
    m_zeldovichD = 0.0f;
    if (m_d_sx) { cudaFree(m_d_sx); m_d_sx = nullptr; }
    if (m_d_sy) { cudaFree(m_d_sy); m_d_sy = nullptr; }
    if (m_d_sz) { cudaFree(m_d_sz); m_d_sz = nullptr; }
    if (m_d_zeldovichDensity) { cudaFree(m_d_zeldovichDensity); m_d_zeldovichDensity = nullptr; }
    m_zeldovichDstN = 0;
}

} // namespace cosmico
