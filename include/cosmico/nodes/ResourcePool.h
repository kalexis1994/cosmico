#pragma once
#include <cosmico/simulation/PMParams.h>

// Forward declare CUDA/cuFFT types
typedef int cufftHandle;

namespace cosmico {

class ResourcePool {
public:
    void init(int maxParticles, int gridN);
    void destroy();

    // Named resource access
    void* particleBuffer() const { return m_particlePtr; }
    float* densityGrid() const { return m_d_density; }
    void* complexGrid() const { return m_d_density_hat; }
    float* forceGrid(int axis) const;
    cufftHandle fftPlanR2C() const { return m_planR2C; }
    cufftHandle fftPlanC2R() const { return m_planC2R; }

    // Reduction / diagnostic buffers
    double* energySums() const { return m_d_energySums; }
    float* powerSpectrum() const { return m_d_spectrum; }
    int* powerSpectrumCounts() const { return m_d_counts; }
    int* sinkCount() const { return m_d_sinkCount; }
    int* newParticleCount() const { return m_d_newParticleCount; }
    int* sinkList() const { return m_d_sinkList; }
    static constexpr int MAX_SINKS = 256;

    // Params passthrough
    void setParams(const PMParams& p) { m_params = p; }
    const PMParams& params() const { return m_params; }

    int gridN() const { return m_gridN; }
    int gridN3() const { return m_gridN * m_gridN * m_gridN; }
    int nComplex() const { return m_gridN * m_gridN * (m_gridN / 2 + 1); }
    float cellSize() const { return m_params.boxSize / static_cast<float>(m_gridN); }
    int particleCount() const { return m_particleCount; }
    void setParticleCount(int n) { m_particleCount = n; }
    void setParticleBuffer(void* ptr, size_t size);

    int nBins() const { return m_nBins; }

    // Comoving state (shared across nodes within a frame)
    float scaleFactor = 1.0f;
    float hubble = 0.0f;
    bool comoving = false;

private:
    void allocateGridBuffers(int N);
    void freeGridBuffers();
    void createFFTPlans(int N);
    void destroyFFTPlans();

    void* m_particlePtr = nullptr;
    size_t m_particleBufferSize = 0;

    float* m_d_density = nullptr;
    void* m_d_density_hat = nullptr;
    float* m_d_forceX = nullptr;
    float* m_d_forceY = nullptr;
    float* m_d_forceZ = nullptr;

    double* m_d_energySums = nullptr;
    float* m_d_spectrum = nullptr;
    int* m_d_counts = nullptr;
    int* m_d_sinkCount = nullptr;
    int* m_d_newParticleCount = nullptr;
    int* m_d_sinkList = nullptr;

    cufftHandle m_planR2C = 0;
    cufftHandle m_planC2R = 0;
    bool m_fftPlansCreated = false;

    int m_gridN = 0;
    int m_particleCount = 0;
    int m_nBins = 64;

    PMParams m_params;
};

} // namespace cosmico
