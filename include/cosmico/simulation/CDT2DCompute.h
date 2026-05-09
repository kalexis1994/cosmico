#pragma once
#include <cosmico/simulation/CDT2DParams.h>
#include <vector>

namespace cosmico {

struct CDT2DStateData {
    int totalSweeps = 0;
    int totalTriangles = 0;
    float acceptanceRate = 0.0f;
    std::vector<int> volumeProfile;        // n(t) current
    std::vector<float> avgVolumeProfile;   // Running <n(t)>
    float hausdorffDimension = 0.0f;
    float spectralDimension = 0.0f;
    // Render data: each triangle = 3 vertices x 5 floats (x,y,r,g,b)
    std::vector<float> vertexData;
    int triangleCount = 0;
};

class CDT2DCompute {
public:
    void init(const CDT2DParams& params);
    void reset(const CDT2DParams& params);
    void step(const CDT2DParams& params);

    const CDT2DStateData& state() const { return m_state; }

private:
    void metropolisSweep(const CDT2DParams& params);
    void computeObservables(const CDT2DParams& params);
    void buildRenderData(const CDT2DParams& params);

    // Triangulation state
    int m_T = 0;
    std::vector<int> m_n;   // m_n[t] = vertices in slice t
    int m_N2 = 0;           // Total triangles = 2 * sum(m_n)

    // Running averages
    int m_measureCount = 0;
    std::vector<double> m_avgN;  // Accumulator for <n(t)>

    // Smoothed profile for rendering (exponential moving average)
    std::vector<float> m_smoothN;

    // MC statistics
    int m_attempted = 0;
    int m_accepted = 0;

    // RNG
    uint64_t m_rngState = 0;

    CDT2DStateData m_state;
};

} // namespace cosmico
