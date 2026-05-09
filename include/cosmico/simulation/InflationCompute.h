#pragma once
#include <cosmico/simulation/InflationParams.h>
#include <vector>
#include <cstdint>

// Forward declare CUDA/cuFFT types to avoid including cuda headers in C++ header
typedef int cufftHandle;
typedef struct CUstream_st* cudaStream_t;
typedef struct CUevent_st* cudaEvent_t;

namespace cosmico {

namespace cuda {
struct FieldGpu;
struct InflationGpuParams;
struct InflationState;
}

struct InflationStateData {
    double time = 0.0;
    double scaleFactor = 1.0;
    double hubble = 0.0;
    double efolds = 0.0;
    double phiMean = 0.0;
    double kineticEnergy = 0.0;
    double gradientEnergy = 0.0;
    double potentialEnergy = 0.0;
    float spectralIndex = 0.0f;

    std::vector<float> powerSpectrum;
    std::vector<float> kBins;
    std::vector<float> fieldSlice;
};

class InflationCompute {
public:
    void init(const InflationParams& params);
    void destroy();
    void reset(const InflationParams& params);
    void step(const InflationParams& params);
    void computePowerSpectrum(const InflationParams& params);

    const InflationStateData& state() const { return m_state; }
    float kernelTimeMs() const { return m_kernelTimeMs; }

    // Extract density volume for 3D rendering.
    // Downsamples from current grid to targetN^3, normalizes to [0,255].
    void extractDensityVolume(uint8_t* hostDst, int targetN);

    // CMB map extraction
    void extractCMBMap(uint8_t* hostDst, int mapW, int mapH,
                       float radius, float boxSize, float Mpl);

    // Zel'dovich structure formation
    void startZeldovich(const InflationParams& params);
    void stepZeldovich(const InflationParams& params);
    void extractZeldovichDensity(uint8_t* hostDst, int targetN);
    bool zeldovichActive() const { return m_zeldovichActive; }
    float zeldovichGrowth() const { return m_zeldovichD; }
    void stopZeldovich();

    // VRAM usage estimate in bytes
    static size_t estimateVRAM(int gridN);

private:
    void allocateField(int N);
    void freeField();
    void createFFTPlans(int N);
    void destroyFFTPlans();
    void updateScaleFactor(double rhoTotal, double Mpl);
    void extractFieldSlice();

    // Field arrays (device pointers stored as void* to keep header CUDA-free)
    void* m_d_phi = nullptr;
    void* m_d_pi = nullptr;
    void* m_d_laplacian = nullptr;
    void* m_d_phi_hat = nullptr;

    // Reduction / spectrum device buffers
    void* m_d_energySums = nullptr;     // 6 doubles
    void* m_d_powerSpectrum = nullptr;  // nBins floats
    void* m_d_powerCounts = nullptr;    // nBins ints

    // cuFFT plans
    cufftHandle m_planR2C = 0;
    cufftHandle m_planC2R = 0;
    bool m_fftPlansCreated = false;

    // CUDA stream and events
    cudaStream_t m_stream = nullptr;
    cudaEvent_t m_startEvent = nullptr;
    cudaEvent_t m_stopEvent = nullptr;

    // State
    InflationStateData m_state;
    float m_kernelTimeMs = 0.0f;
    int m_currentN = 0;
    int m_nBins = 64;

    // Density extraction device buffers
    void* m_d_density = nullptr;       // uint8_t[targetN^3]
    void* m_d_minVal = nullptr;        // double[2] scratch for stats reduction
    void* m_d_maxVal = nullptr;        // float (unused, kept for cleanup)
    int m_densityN = 0;

    // CMB map extraction
    void* m_d_cmbMap = nullptr;        // uint8_t[mapW*mapH]
    void* m_d_cmbStats = nullptr;      // double[2] scratch for stats reduction
    int m_cmbMapW = 0;
    int m_cmbMapH = 0;

    // Zel'dovich structure formation
    bool m_zeldovichActive = false;
    float m_zeldovichD = 0.0f;         // Growth factor D(t)
    float m_displacementRms = 1.0f;    // RMS displacement for normalization
    float m_zeldovichBoxSize = 10.0f;  // Box size used for deposit
    float m_zeldovichLogScale = 3.0f; // Log normalization for density
    void* m_d_sx = nullptr;            // float[N³] displacement x
    void* m_d_sy = nullptr;            // float[N³] displacement y
    void* m_d_sz = nullptr;            // float[N³] displacement z
    void* m_d_zeldovichDensity = nullptr; // float[dstN³] CIC accumulation
    int m_zeldovichDstN = 0;

    // Host staging for readback
    double m_h_energySums[6] = {};
    std::vector<float> m_h_spectrum;
    std::vector<int> m_h_counts;
    std::vector<float> m_h_sliceBuffer;
};

} // namespace cosmico
