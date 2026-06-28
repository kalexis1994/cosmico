#include <cosmico/simulation/CMBSynthesis.h>

#include <cmath>
#include <vector>
#include <random>
#include <cstdio>

namespace cosmico {
namespace {

constexpr double PI = 3.14159265358979323846;

// Acoustic scales from the cosmological parameters (Eisenstein-Hu 1998 /
// Hu-Sugiyama 1996 fitting formulas): the acoustic multipole ellA (peak
// spacing), the Silk-damping multipole ellD, and the baryon loading R*.
struct Scales { double ellA, ellD, Rstar; };

Scales computeScales(double Om, double Ob, double h) {
    double wm = Om*h*h, wb = Ob*h*h, theta = 1.0;   // T_cmb/2.7 ~ 1
    double g1 = 0.0783*std::pow(wb,-0.238)/(1.0+39.5*std::pow(wb,0.763));
    double g2 = 0.560/(1.0+21.1*std::pow(wb,1.81));
    double zstar = 1048.0*(1.0+0.00124*std::pow(wb,-0.738))*(1.0+g1*std::pow(wm,g2));
    double zeq  = 2.5e4*wm;
    double keq  = 7.46e-2*wm;                                  // Mpc^-1
    double Rstar = 31.5*wb*(1000.0/zstar);
    double Req   = 31.5*wb*(1000.0/zeq);
    double rs = (2.0/(3.0*keq))*std::sqrt(6.0/Req)             // sound horizon, Mpc
              * std::log((std::sqrt(1.0+Rstar)+std::sqrt(Rstar+Req))/(1.0+std::sqrt(Req)));
    double kSilk = 1.6*std::pow(wb,0.52)*std::pow(wm,0.73)*(1.0+std::pow(10.4*wm,-0.95));
    double OL = 1.0-Om; const int N = 2000; double sum = 0.0, dz = zstar/N;  // D_A integral
    for (int i = 0; i <= N; ++i) {
        double z = i*dz, E = std::sqrt(Om*(1.0+z)*(1.0+z)*(1.0+z)+OL);
        sum += ((i==0||i==N)?0.5:1.0)/E;
    }
    double DA = (2997.92458/h)*sum*dz;                         // Mpc
    Scales s; s.ellA = PI*DA/rs; s.ellD = kSilk*DA; s.Rstar = Rstar;
    return s;
}

// D_ℓ = ℓ(ℓ+1)C_ℓ/2π: baryon-loaded photon-baryon acoustic oscillation (odd
// peaks enhanced by the baryon zero-point offset) + Doppler + Silk damping +
// Sachs-Wolfe plateau, tilted by n_s. Validated: first peak ~ ℓ220 for the
// fiducial parameters; peaks shift with Ω_m/h and re-weight with Ω_b.
double Dell(double ell, const Scales& s, double ns) {
    if (ell < 2.0) ell = 2.0;
    double R = s.Rstar;
    double x = PI*(ell/s.ellA + 0.25);                  // phase (first peak near 0.75 ellA)
    double mono = (1.0+R)*std::cos(x) - R;              // monopole, baryon-loaded
    double vel  = std::sin(x)/std::sqrt(3.0*(1.0+R));   // Doppler (out of phase)
    double silk = std::exp(-(ell/s.ellD)*(ell/s.ellD));
    double prim = std::pow(ell/200.0, ns-1.0);          // primordial tilt
    return prim*((mono*mono + vel*vel)*silk + 0.7);     // +0.7 = Sachs-Wolfe plateau
}
double Cl(double ell, const Scales& s, double ns) {
    if (ell < 2.0) return 0.0;
    return 2.0*PI*Dell(ell, s, ns)/(ell*(ell+1.0));
}

} // namespace

void synthesizeCMBMap(uint8_t* out, int W, int H, int lMax, unsigned seed,
                      double omegaM, double omegaB, double hubble, double ns) {
    Scales scales = computeScales(omegaM, omegaB, hubble);
    fprintf(stderr, "[CMB] synth %dx%d lMax=%d seed=%u  Om=%.2f Ob=%.3f h=%.2f ns=%.3f  ellA=%.0f ...\n",
            W, H, lMax, seed, omegaM, omegaB, hubble, ns, scales.ellA);

    auto idx = [](int l, int m) { return (size_t)l * (l + 1) / 2 + m; };
    const size_t ncoef = (size_t)(lMax + 1) * (lMax + 2) / 2;

    // Gaussian harmonic coefficients c_ℓm (cos/sin parts) ~ N(0, C_ℓ).
    std::mt19937 rng(seed);
    std::normal_distribution<double> N01(0.0, 1.0);
    std::vector<double> cc(ncoef, 0.0), cs(ncoef, 0.0);
    for (int l = 2; l <= lMax; ++l) {
        double sig = std::sqrt(Cl((double)l, scales, ns));
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
