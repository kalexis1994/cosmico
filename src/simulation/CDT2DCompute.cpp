#include <cosmico/simulation/CDT2DCompute.h>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <chrono>

namespace cosmico {

// ── Fast xoshiro256** RNG ─────────────────────────────────────────────────

static inline uint64_t rotl(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

struct Xoshiro256 {
    uint64_t s[4];

    void seed(uint64_t v) {
        // SplitMix64 seeding
        for (int i = 0; i < 4; i++) {
            v += 0x9e3779b97f4a7c15ULL;
            uint64_t z = v;
            z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
            z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
            s[i] = z ^ (z >> 31);
        }
    }

    uint64_t next() {
        uint64_t result = rotl(s[1] * 5, 7) * 9;
        uint64_t t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = rotl(s[3], 45);
        return result;
    }

    // Uniform int in [0, n)
    int nextInt(int n) {
        return static_cast<int>(next() % static_cast<uint64_t>(n));
    }

    // Uniform float in [0, 1)
    float nextFloat() {
        return static_cast<float>(next() >> 11) * 0x1.0p-53f;
    }
};

static Xoshiro256 s_rng;

// ── CDT2DCompute ──────────────────────────────────────────────────────────

void CDT2DCompute::init(const CDT2DParams& params) {
    auto now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    s_rng.seed(static_cast<uint64_t>(now));
    m_rngState = static_cast<uint64_t>(now);
    reset(params);
}

void CDT2DCompute::reset(const CDT2DParams& params) {
    m_T = params.T;
    m_n.resize(m_T);

    // Start near equilibrium: n(t) = N_target/(2T) gives N2 ≈ N_target
    // Use cos² modulation so it starts close to the expected thermalized profile
    int avgN = params.N_target / (2 * m_T);
    if (avgN < 4) avgN = 4;
    for (int t = 0; t < m_T; t++) {
        float frac = static_cast<float>(t) / static_cast<float>(m_T);
        float cosVal = std::cos(3.14159265f * (frac - 0.5f));
        int n = static_cast<int>(avgN * (0.3f + 0.7f * cosVal * cosVal));
        if (n < 3) n = 3;
        m_n[t] = n;
    }

    m_N2 = 0;
    for (int t = 0; t < m_T; t++) {
        m_N2 += m_n[t] + m_n[(t + 1) % m_T]; // each slab has n_bot UP + n_top DOWN
    }

    // Reset running averages
    m_measureCount = 0;
    m_avgN.assign(m_T, 0.0);

    // Initialize smoothed profile to current state
    m_smoothN.resize(m_T);
    for (int t = 0; t < m_T; t++) {
        m_smoothN[t] = static_cast<float>(m_n[t]);
    }

    m_attempted = 0;
    m_accepted = 0;

    m_state = CDT2DStateData{};
    m_state.volumeProfile.resize(m_T);
    m_state.avgVolumeProfile.resize(m_T, 0.0f);

    computeObservables(params);
    buildRenderData(params);
}

void CDT2DCompute::step(const CDT2DParams& params) {
    // Handle T change
    if (params.T != m_T) {
        reset(params);
        return;
    }

    // Reset per-frame MC stats
    m_attempted = 0;
    m_accepted = 0;

    for (int s = 0; s < params.sweepsPerFrame; s++) {
        metropolisSweep(params);
    }

    // Update running averages
    m_measureCount++;
    for (int t = 0; t < m_T; t++) {
        m_avgN[t] += static_cast<double>(m_n[t]);
    }

    // Update smoothed profile (exponential moving average for rendering)
    float alpha = params.renderSmoothing; // 0.95 = smooth, 0 = raw
    for (int t = 0; t < m_T; t++) {
        m_smoothN[t] = alpha * m_smoothN[t] + (1.0f - alpha) * static_cast<float>(m_n[t]);
    }

    computeObservables(params);
    buildRenderData(params);
}

void CDT2DCompute::metropolisSweep(const CDT2DParams& params) {
    int nTarget = params.N_target;
    float lambda = params.lambda;
    float eps = params.epsilon;

    // One sweep = N_target attempts
    for (int i = 0; i < nTarget; i++) {
        int t = s_rng.nextInt(m_T);
        bool doAdd = s_rng.nextInt(2) == 0;

        m_attempted++;

        if (doAdd) {
            // ADD: n(t) -> n(t)+1, N2 += 2
            int oldN2 = m_N2;
            int newN2 = oldN2 + 2;

            // dS = lambda * 2 + epsilon * ((newN2 - nTarget)^2 - (oldN2 - nTarget)^2)
            float dS_action = lambda * 2.0f;
            float dS_volume = eps * static_cast<float>(
                (newN2 - nTarget) * (newN2 - nTarget) -
                (oldN2 - nTarget) * (oldN2 - nTarget));
            float dS = dS_action + dS_volume;

            if (dS <= 0.0f || s_rng.nextFloat() < std::exp(-dS)) {
                m_n[t]++;
                m_N2 = newN2;
                m_accepted++;
            }
        } else {
            // REMOVE: n(t) -> n(t)-1, N2 -= 2  (reject if n(t) <= 3)
            if (m_n[t] <= 3) continue;

            int oldN2 = m_N2;
            int newN2 = oldN2 - 2;

            float dS_action = lambda * (-2.0f);
            float dS_volume = eps * static_cast<float>(
                (newN2 - nTarget) * (newN2 - nTarget) -
                (oldN2 - nTarget) * (oldN2 - nTarget));
            float dS = dS_action + dS_volume;

            if (dS <= 0.0f || s_rng.nextFloat() < std::exp(-dS)) {
                m_n[t]--;
                m_N2 = newN2;
                m_accepted++;
            }
        }
    }
}

void CDT2DCompute::computeObservables(const CDT2DParams& params) {
    m_state.totalSweeps = m_measureCount * params.sweepsPerFrame;
    m_state.totalTriangles = m_N2;

    // Acceptance rate
    if (m_attempted > 0) {
        m_state.acceptanceRate = static_cast<float>(m_accepted) / static_cast<float>(m_attempted);
    }

    // Copy volume profile
    for (int t = 0; t < m_T; t++) {
        m_state.volumeProfile[t] = m_n[t];
    }

    // Average volume profile (skip first 10 measurements as thermalization)
    int thermSkip = 10;
    if (m_measureCount > thermSkip) {
        for (int t = 0; t < m_T; t++) {
            m_state.avgVolumeProfile[t] = static_cast<float>(m_avgN[t] / m_measureCount);
        }
    }

    // ── Hausdorff dimension (2D geodesic volume) ──────────────────────
    // V(r) = number of vertices reachable within geodesic distance r
    // From (t0, x0): temporal step costs 1, spatial step costs 1
    // At time distance |dt|, remaining spatial budget = r - |dt|
    // Accessible vertices at slice t0+dt = min(n[t0+dt], 2*(r-|dt|)+1)
    {
        int maxR = m_T / 4;
        if (maxR < 4) maxR = 4;

        std::vector<double> logR, logV;
        for (int r = 1; r <= maxR; r++) {
            double totalVol = 0.0;
            for (int t0 = 0; t0 < m_T; t0++) {
                double vol = 0.0;
                for (int dt = -r; dt <= r; dt++) {
                    int tt = ((t0 + dt) % m_T + m_T) % m_T;
                    int spatialBudget = 2 * (r - std::abs(dt)) + 1;
                    int accessible = m_n[tt] < spatialBudget ? m_n[tt] : spatialBudget;
                    vol += accessible;
                }
                totalVol += vol;
            }
            totalVol /= m_T;
            if (totalVol > 0) {
                logR.push_back(std::log(static_cast<double>(r)));
                logV.push_back(std::log(totalVol));
            }
        }

        // Linear regression: logV = d_H * logR + b
        if (logR.size() >= 3) {
            double sumX = 0, sumY = 0, sumXY = 0, sumXX = 0;
            int nn = static_cast<int>(logR.size());
            for (int i = 0; i < nn; i++) {
                sumX += logR[i];
                sumY += logV[i];
                sumXY += logR[i] * logV[i];
                sumXX += logR[i] * logR[i];
            }
            double denom = nn * sumXX - sumX * sumX;
            if (std::abs(denom) > 1e-12) {
                m_state.hausdorffDimension = static_cast<float>(
                    (nn * sumXY - sumX * sumY) / denom);
            }
        }
    }

    // ── Spectral dimension (2D random walk) ───────────────────────────
    // Walk on the 2D triangulation: at each step pick one of 4 directions
    // (spatial left/right, temporal up/down) with equal probability.
    // Temporal move: t -> t±1, spatial position remapped proportionally.
    // Spatial move: x -> (x±1) mod n(t).
    // P(return after sigma steps) ~ sigma^(-d_s/2)
    {
        int numWalks = 800;
        int maxSigma = m_T * 4;
        if (maxSigma > 400) maxSigma = 400;
        if (maxSigma < 40) maxSigma = 40;

        std::vector<double> returnProb(maxSigma + 1, 0.0);

        for (int w = 0; w < numWalks; w++) {
            int t0 = s_rng.nextInt(m_T);
            int x0 = s_rng.nextInt(m_n[t0]);
            int t = t0, x = x0;

            for (int sigma = 1; sigma <= maxSigma; sigma++) {
                int choice = s_rng.nextInt(4);
                if (choice == 0) {
                    // Temporal: t -> t+1
                    int tNew = (t + 1) % m_T;
                    x = static_cast<int>(static_cast<long long>(x) * m_n[tNew] / m_n[t]);
                    if (x >= m_n[tNew]) x = m_n[tNew] - 1;
                    t = tNew;
                } else if (choice == 1) {
                    // Temporal: t -> t-1
                    int tNew = ((t - 1) % m_T + m_T) % m_T;
                    x = static_cast<int>(static_cast<long long>(x) * m_n[tNew] / m_n[t]);
                    if (x >= m_n[tNew]) x = m_n[tNew] - 1;
                    t = tNew;
                } else if (choice == 2) {
                    // Spatial: x -> x+1
                    x = (x + 1) % m_n[t];
                } else {
                    // Spatial: x -> x-1
                    x = ((x - 1) % m_n[t] + m_n[t]) % m_n[t];
                }

                if (t == t0 && x == x0) {
                    returnProb[sigma] += 1.0;
                }
            }
        }

        for (int sigma = 1; sigma <= maxSigma; sigma++) {
            returnProb[sigma] /= numWalks;
        }

        // Fit log P vs log sigma in the scaling regime
        // d_s = -2 * slope
        std::vector<double> logSig, logP;
        int fitStart = 10; // skip short-distance transients
        for (int sigma = fitStart; sigma <= maxSigma; sigma++) {
            if (returnProb[sigma] > 1e-8) {
                logSig.push_back(std::log(static_cast<double>(sigma)));
                logP.push_back(std::log(returnProb[sigma]));
            }
        }

        if (logSig.size() >= 5) {
            double sumX = 0, sumY = 0, sumXY = 0, sumXX = 0;
            int nn = static_cast<int>(logSig.size());
            for (int i = 0; i < nn; i++) {
                sumX += logSig[i];
                sumY += logP[i];
                sumXY += logSig[i] * logP[i];
                sumXX += logSig[i] * logSig[i];
            }
            double denom = nn * sumXX - sumX * sumX;
            if (std::abs(denom) > 1e-12) {
                double slope = (nn * sumXY - sumX * sumY) / denom;
                m_state.spectralDimension = static_cast<float>(-2.0 * slope);
            }
        }
    }
}

void CDT2DCompute::buildRenderData(const CDT2DParams& params) {
    // Count total triangles for rendering (from real triangulation)
    int totalTris = 0;
    for (int t = 0; t < m_T; t++) {
        int tNext = (t + 1) % m_T;
        totalTris += m_n[t] + m_n[tNext]; // UPs + DOWNs per slab
    }
    m_state.triangleCount = totalTris;

    // Preallocate: each triangle = 3 vertices * 5 floats (x,y,r,g,b)
    m_state.vertexData.resize(totalTris * 3 * 5);
    float* out = m_state.vertexData.data();

    // Compute discrete curvature from smoothed profile (for visual coherence)
    std::vector<float> curvature(m_T);
    float maxAbsCurv = 0.0f;
    for (int t = 0; t < m_T; t++) {
        int tPrev = ((t - 1) % m_T + m_T) % m_T;
        int tNext = (t + 1) % m_T;
        curvature[t] = 2.0f * m_smoothN[t] - m_smoothN[tPrev] - m_smoothN[tNext];
        float ac = std::abs(curvature[t]);
        if (ac > maxAbsCurv) maxAbsCurv = ac;
    }
    if (maxAbsCurv < 1.0f) maxAbsCurv = 1.0f;

    // Color mapping function: curvature -> RGB
    // Blue (negative) -> White (zero) -> Red (positive)
    auto curvColor = [&](float q, float& r, float& g, float& b) {
        float v = q / maxAbsCurv; // [-1, 1]
        if (v > 1.0f) v = 1.0f;
        if (v < -1.0f) v = -1.0f;
        if (!params.colorByCurvature) {
            r = 0.4f; g = 0.6f; b = 0.8f;
            return;
        }
        if (v > 0.0f) {
            r = 1.0f; g = 1.0f - v; b = 1.0f - v;
        } else {
            float s = -v;
            r = 1.0f - s; g = 1.0f - s; b = 1.0f;
        }
    };

    float scale = params.meshScale;
    float ySpacing = scale * 1.0f;
    float yCenter = static_cast<float>(m_T) * 0.5f;

    // Use smoothed profile for vertex positions (shape of the universe)
    // The actual triangulation count m_n determines how many triangles per slab,
    // but the smoothed width determines WHERE vertices are placed.
    float maxSmooth = 3.0f;
    for (int t = 0; t < m_T; t++) {
        if (m_smoothN[t] > maxSmooth) maxSmooth = m_smoothN[t];
    }

    // Adaptive xSpacing: widest slice ≈ 50% of total height
    float totalHeight = static_cast<float>(m_T) * ySpacing;
    float xSpacing = (totalHeight * 0.5f) / maxSmooth;

    // Helper: x position of vertex i out of n, within a slice of smoothed width w
    // Maps [0..n] to [-w/2..+w/2] where w = smoothN * xSpacing
    auto xPos = [&](int i, int n, float smoothW) -> float {
        float frac = static_cast<float>(i) / static_cast<float>(n); // [0, 1]
        return (frac - 0.5f) * smoothW * xSpacing;
    };

    for (int t = 0; t < m_T; t++) {
        int tNext = (t + 1) % m_T;
        int nBot = m_n[t];
        int nTop = m_n[tNext];
        float wBot = m_smoothN[t];
        float wTop = m_smoothN[tNext];

        float yBot = (static_cast<float>(t) - yCenter) * ySpacing;
        float yTop = (static_cast<float>(t + 1) - yCenter) * ySpacing;

        // Curvature colors for this slab
        float cr, cg, cb;
        float avgCurv = (curvature[t] + curvature[tNext]) * 0.5f;
        curvColor(avgCurv, cr, cg, cb);

        // Zipper algorithm: interleave UP and DOWN triangles
        int botIdx = 0, topIdx = 0;
        while (botIdx < nBot || topIdx < nTop) {
            bool doUp;
            if (botIdx >= nBot) {
                doUp = false;
            } else if (topIdx >= nTop) {
                doUp = true;
            } else {
                doUp = (static_cast<float>(botIdx) / nBot <=
                        static_cast<float>(topIdx) / nTop);
            }

            if (doUp && botIdx < nBot) {
                float x0 = xPos(botIdx, nBot, wBot);
                float x1 = xPos(botIdx + 1, nBot, wBot);
                float xT = xPos(topIdx, nTop, wTop);

                *out++ = x0;  *out++ = yBot;
                *out++ = cr; *out++ = cg; *out++ = cb;
                *out++ = x1;  *out++ = yBot;
                *out++ = cr; *out++ = cg; *out++ = cb;
                *out++ = xT;  *out++ = yTop;
                *out++ = cr; *out++ = cg; *out++ = cb;

                botIdx++;
            } else if (topIdx < nTop) {
                float xT0 = xPos(topIdx, nTop, wTop);
                float xT1 = xPos(topIdx + 1, nTop, wTop);
                float xB  = xPos(botIdx, nBot, wBot);

                *out++ = xT0; *out++ = yTop;
                *out++ = cr; *out++ = cg; *out++ = cb;
                *out++ = xT1; *out++ = yTop;
                *out++ = cr; *out++ = cg; *out++ = cb;
                *out++ = xB;  *out++ = yBot;
                *out++ = cr; *out++ = cg; *out++ = cb;

                topIdx++;
            } else {
                break;
            }
        }
    }

    size_t written = static_cast<size_t>(out - m_state.vertexData.data());
    m_state.triangleCount = static_cast<int>(written / 15);
}

} // namespace cosmico
