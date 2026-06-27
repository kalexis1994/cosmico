#include <cosmico/simulation/CMBSynthesis.h>

#include <cmath>
#include <vector>
#include <random>
#include <cstdio>

namespace cosmico {
namespace {

constexpr double PI = 3.14159265358979323846;

// ΛCDM TT spectrum: D_ℓ = ℓ(ℓ+1)C_ℓ/2π in μK² (Planck-like anchor points, linear
// interp). Sachs-Wolfe plateau + first three acoustic peaks (220, 540, 810) +
// Silk damping. Validated: synthesising from this and re-measuring recovers it.
double Dell(int ell) {
    static const double L[] = {2,30,60,100,150,190,220,250,300,360,420,470,510,540,
                               600,660,700,760,810,870,950,1050,1150,1300,1500,1700,2000,2500,4000};
    static const double D[] = {1100,950,920,1700,3300,5000,5700,5500,4200,2600,1750,1900,2250,2480,
                               2300,1800,1750,2150,2480,2300,1850,1450,1500,950,500,280,110,30,2};
    static const int n = sizeof(L) / sizeof(L[0]);
    double x = (double)ell;
    if (x <= L[0]) return D[0];
    if (x >= L[n-1]) return D[n-1];
    for (int i = 1; i < n; ++i)
        if (x <= L[i]) {
            double t = (x - L[i-1]) / (L[i] - L[i-1]);
            return D[i-1] + t * (D[i] - D[i-1]);
        }
    return D[n-1];
}
double Cl(int ell) {
    if (ell < 2) return 0.0;
    return 2.0 * PI * Dell(ell) / ((double)ell * (ell + 1));
}

} // namespace

void synthesizeCMBMap(uint8_t* out, int W, int H, int lMax, unsigned seed, double deltaNs) {
    fprintf(stderr, "[CMB] synthesizing %dx%d map, lMax=%d, seed=%u, dNs=%+.3f ...\n",
            W, H, lMax, seed, deltaNs);

    auto idx = [](int l, int m) { return (size_t)l * (l + 1) / 2 + m; };
    const size_t ncoef = (size_t)(lMax + 1) * (lMax + 2) / 2;

    // Gaussian harmonic coefficients c_ℓm (cos/sin parts) ~ N(0, C_ℓ).
    std::mt19937 rng(seed);
    std::normal_distribution<double> N01(0.0, 1.0);
    std::vector<double> cc(ncoef, 0.0), cs(ncoef, 0.0);
    for (int l = 2; l <= lMax; ++l) {
        // Re-tilt the primordial spectrum by deltaNs about ℓ=200 (the inflaton's
        // n_s sets the slope; the baked transfer keeps the acoustic peaks).
        double sig = std::sqrt(Cl(l) * std::pow((double)l / 200.0, deltaNs));
        for (int m = 0; m <= l; ++m) {
            cc[idx(l, m)] = sig * N01(rng);
            cs[idx(l, m)] = (m == 0) ? 0.0 : sig * N01(rng);
        }
    }

    // Pre-compute the latitude-independent Legendre recurrence coefficients so
    // the hot per-latitude loop is multiply-adds only (no sqrt).
    std::vector<double> diag(lMax + 1, 0.0), first(lMax + 1, 0.0);
    std::vector<double> alpha(ncoef, 0.0), beta(ncoef, 0.0);
    for (int m = 1; m <= lMax; ++m) diag[m] = std::sqrt((2.0 * m + 1.0) / (2.0 * m));
    for (int m = 0; m <= lMax; ++m) if (m + 1 <= lMax) first[m] = std::sqrt(2.0 * m + 3.0);
    for (int m = 0; m <= lMax; ++m)
        for (int l = m + 2; l <= lMax; ++l) {
            alpha[idx(l, m)] = std::sqrt((2.0*l-1.0)*(2.0*l+1.0)/(((double)l-m)*((double)l+m)));
            beta[idx(l, m)]  = std::sqrt((2.0*l+1.0)*((double)l-m-1.0)*((double)l+m-1.0)
                                         /((2.0*l-3.0)*((double)l-m)*((double)l+m)));
        }

    // Azimuth tables, laid out [j][m] so the inner m-loop is contiguous.
    std::vector<float> cosT((size_t)W * (lMax + 1)), sinT((size_t)W * (lMax + 1));
    for (int j = 0; j < W; ++j) {
        double base = 2.0 * PI * j / W;
        float* cj = &cosT[(size_t)j * (lMax + 1)];
        float* sj = &sinT[(size_t)j * (lMax + 1)];
        for (int m = 0; m <= lMax; ++m) { cj[m] = (float)std::cos(base * m); sj[m] = (float)std::sin(base * m); }
    }

    std::vector<double> T((size_t)W * H);
    std::vector<double> bc(lMax + 1), bs(lMax + 1), lam(lMax + 1);
    const double lam00 = std::sqrt(1.0 / (4.0 * PI));
    double sum = 0.0, sum2 = 0.0;

    for (int i = 0; i < H; ++i) {
        double theta = PI * (i + 0.5) / H, x = std::cos(theta), st = std::sin(theta);
        for (int m = 0; m <= lMax; ++m) { bc[m] = 0.0; bs[m] = 0.0; }

        // λ_ℓm(x) via the pre-computed normalised recurrence, accumulating the
        // azimuthal coefficients b_m^{cos,sin} as we go.
        double dmm = lam00;                              // λ_00
        for (int m = 0; m <= lMax; ++m) {
            if (m > 0) dmm *= diag[m] * st;              // λ_mm
            lam[m] = dmm;
            double lm2 = dmm, lm1 = dmm;
            if (m + 1 <= lMax) { lm1 = first[m] * x * dmm; lam[m + 1] = lm1; }
            for (int l = m + 2; l <= lMax; ++l) {
                double v = alpha[idx(l, m)] * x * lm1 - beta[idx(l, m)] * lm2;
                lam[l] = v; lm2 = lm1; lm1 = v;
            }
            double s2 = (m == 0) ? 1.0 : 1.4142135623730951;  // sqrt(2)
            for (int l = m; l <= lMax; ++l) {
                bc[m] += cc[idx(l, m)] * lam[l] * s2;
                bs[m] += cs[idx(l, m)] * lam[l] * s2;
            }
        }

        // Inverse azimuth transform: T(θ_i, φ_j) = Σ_m b_m^c cos + b_m^s sin.
        double* row = &T[(size_t)i * W];
        for (int j = 0; j < W; ++j) {
            const float* cj = &cosT[(size_t)j * (lMax + 1)];
            const float* sj = &sinT[(size_t)j * (lMax + 1)];
            double t = 0.0;
            for (int m = 0; m <= lMax; ++m) t += bc[m] * cj[m] + bs[m] * sj[m];
            row[j] = t; sum += t; sum2 += t * t;
        }
    }

    // Map fluctuations to uint8: centre at 128, ±3σ → [0,255].
    double n = (double)W * H, mean = sum / n, var = sum2 / n - mean * mean;
    double sd = std::sqrt(var > 1e-30 ? var : 1.0), scale = 1.0 / (6.0 * sd);
    for (size_t k = 0; k < (size_t)W * H; ++k) {
        double v = 0.5 + (T[k] - mean) * scale;
        v = v < 0.0 ? 0.0 : (v > 1.0 ? 1.0 : v);
        out[k] = (uint8_t)(v * 255.0 + 0.5);
    }
    fprintf(stderr, "[CMB] done (σ_T = %.3g μK).\n", sd);
}

} // namespace cosmico
