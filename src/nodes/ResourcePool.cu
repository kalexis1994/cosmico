#include <cosmico/nodes/ResourcePool.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <stdexcept>
#include <cstdio>
#include <string>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "[NodeGraph] CUDA Error: %s at %s:%d\n", \
                    cudaGetErrorString(err), __FILE__, __LINE__); \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err)); \
        } \
    } while(0)

#define CUFFT_CHECK(call) \
    do { \
        cufftResult err = (call); \
        if (err != CUFFT_SUCCESS) { \
            fprintf(stderr, "[NodeGraph] cuFFT Error: %d at %s:%d\n", \
                    (int)err, __FILE__, __LINE__); \
            throw std::runtime_error("cuFFT error: " + std::to_string((int)err)); \
        } \
    } while(0)

namespace cosmico {

void ResourcePool::init(int maxParticles, int gridN) {
    m_particleCount = maxParticles;
    m_gridN = gridN;
    allocateGridBuffers(gridN);
    createFFTPlans(gridN);
}

void ResourcePool::destroy() {
    destroyFFTPlans();
    freeGridBuffers();
}

float* ResourcePool::forceGrid(int axis) const {
    switch (axis) {
        case 0: return m_d_forceX;
        case 1: return m_d_forceY;
        case 2: return m_d_forceZ;
        default: return nullptr;
    }
}

void ResourcePool::setParticleBuffer(void* ptr, size_t size) {
    m_particlePtr = ptr;
    m_particleBufferSize = size;
}

void ResourcePool::allocateGridBuffers(int N) {
    size_t N3 = static_cast<size_t>(N) * N * N;
    size_t Ncomplex = static_cast<size_t>(N) * N * (N / 2 + 1);

    CUDA_CHECK(cudaMalloc(&m_d_density, N3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&m_d_density_hat, Ncomplex * sizeof(cufftComplex)));
    CUDA_CHECK(cudaMalloc(&m_d_forceX, N3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&m_d_forceY, N3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&m_d_forceZ, N3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&m_d_energySums, 6 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&m_d_spectrum, m_nBins * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&m_d_counts, m_nBins * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&m_d_sinkCount, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&m_d_newParticleCount, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&m_d_sinkList, MAX_SINKS * sizeof(int)));

    fprintf(stderr, "[NodeGraph] Allocated grid buffers: N=%d\n", N);
}

void ResourcePool::freeGridBuffers() {
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
    if (m_d_sinkList) { cudaFree(m_d_sinkList); m_d_sinkList = nullptr; }
    m_gridN = 0;
}

void ResourcePool::createFFTPlans(int N) {
    CUFFT_CHECK(cufftPlan3d(&m_planR2C, N, N, N, CUFFT_R2C));
    CUFFT_CHECK(cufftPlan3d(&m_planC2R, N, N, N, CUFFT_C2R));
    m_fftPlansCreated = true;
    fprintf(stderr, "[NodeGraph] Created FFT plans: N=%d\n", N);
}

void ResourcePool::destroyFFTPlans() {
    if (m_fftPlansCreated) {
        cufftDestroy(m_planR2C);
        cufftDestroy(m_planC2R);
        m_fftPlansCreated = false;
    }
}

} // namespace cosmico
