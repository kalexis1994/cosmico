#include <cosmico/simulation/InitialConditions.h>
#include <cmath>
#include <random>

namespace cosmico {

const char* initialConditionName(InitialCondition ic) {
    switch (ic) {
        case InitialCondition::Sphere: return "Sphere";
        case InitialCondition::Disk: return "Disk";
        case InitialCondition::TwoBody: return "Two-Body";
        case InitialCondition::GalaxyCollision: return "Galaxy Collision";
        case InitialCondition::Galaxy: return "Galaxy";
        case InitialCondition::Cosmological: return "Cosmological";
        case InitialCondition::SolarSystem: return "Solar System";
        default: return "Unknown";
    }
}

InitialCondition initialConditionFromString(const std::string& s) {
    if (s == "Sphere")          return InitialCondition::Sphere;
    if (s == "Disk")            return InitialCondition::Disk;
    if (s == "TwoBody")         return InitialCondition::TwoBody;
    if (s == "GalaxyCollision") return InitialCondition::GalaxyCollision;
    if (s == "Galaxy")          return InitialCondition::Galaxy;
    if (s == "Cosmological")    return InitialCondition::Cosmological;
    if (s == "SolarSystem")     return InitialCondition::SolarSystem;
    return InitialCondition::Sphere;
}

static std::mt19937& rng() {
    static std::mt19937 gen(42);
    return gen;
}

static std::vector<ParticleData> generateSphere(uint32_t count) {
    std::vector<ParticleData> particles(count);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    float radius = 15.0f;

    for (uint32_t i = 0; i < count; i++) {
        // Uniform distribution in sphere via rejection sampling
        float x, y, z;
        do {
            x = dist(rng());
            y = dist(rng());
            z = dist(rng());
        } while (x*x + y*y + z*z > 1.0f);

        float r = radius * std::cbrt(x*x + y*y + z*z);
        float norm = std::sqrt(x*x + y*y + z*z);
        if (norm > 0.001f) { x /= norm; y /= norm; z /= norm; }

        particles[i].position[0] = x * r;
        particles[i].position[1] = y * r;
        particles[i].position[2] = z * r;
        particles[i].position[3] = 1.0f; // mass

        // Small random velocities
        particles[i].velocity[0] = dist(rng()) * 0.5f;
        particles[i].velocity[1] = dist(rng()) * 0.5f;
        particles[i].velocity[2] = dist(rng()) * 0.5f;
        particles[i].velocity[3] = 0.0f;

        particles[i].attributes[0] = 0.0f;
        particles[i].attributes[1] = 0.0f;
        particles[i].attributes[2] = 0.0f;
        particles[i].attributes[3] = 0.0f;
    }
    return particles;
}

static std::vector<ParticleData> generateDisk(uint32_t count) {
    std::vector<ParticleData> particles(count);
    std::uniform_real_distribution<float> angleDist(0.0f, 2.0f * 3.14159265f);
    std::uniform_real_distribution<float> radiusDist(0.0f, 1.0f);
    std::normal_distribution<float> heightDist(0.0f, 0.3f);

    float diskRadius = 20.0f;
    float centralMass = static_cast<float>(count); // Total mass ~ N particles
    float G = 1.0f;

    // Central massive particle
    particles[0].position[0] = 0.0f;
    particles[0].position[1] = 0.0f;
    particles[0].position[2] = 0.0f;
    particles[0].position[3] = centralMass * 0.5f;
    particles[0].velocity[0] = 0.0f;
    particles[0].velocity[1] = 0.0f;
    particles[0].velocity[2] = 0.0f;
    particles[0].velocity[3] = 0.0f;
    particles[0].attributes[0] = 0.0f;
    particles[0].attributes[1] = 0.0f;
    particles[0].attributes[2] = 0.0f;
    particles[0].attributes[3] = 0.0f;

    for (uint32_t i = 1; i < count; i++) {
        float angle = angleDist(rng());
        float r = diskRadius * std::sqrt(radiusDist(rng()));
        r = std::max(r, 1.0f); // Avoid very close particles

        float x = r * std::cos(angle);
        float z = r * std::sin(angle);
        float y = heightDist(rng());

        particles[i].position[0] = x;
        particles[i].position[1] = y;
        particles[i].position[2] = z;
        particles[i].position[3] = 1.0f;

        // Circular orbital velocity: v = sqrt(G*M/r)
        float v = std::sqrt(G * centralMass * 0.5f / r);
        // Tangential direction (perpendicular to radius in XZ plane)
        float tx = -std::sin(angle);
        float tz = std::cos(angle);

        particles[i].velocity[0] = tx * v;
        particles[i].velocity[1] = 0.0f;
        particles[i].velocity[2] = tz * v;
        particles[i].velocity[3] = 0.0f;

        particles[i].attributes[0] = 0.0f;
        particles[i].attributes[1] = 0.0f;
        particles[i].attributes[2] = 0.0f;
        particles[i].attributes[3] = 0.0f;
    }
    return particles;
}

static std::vector<ParticleData> generateTwoBody(uint32_t count) {
    std::vector<ParticleData> particles(count);
    std::normal_distribution<float> noise(0.0f, 0.1f);

    float mass = 500.0f;
    float separation = 20.0f;
    // Circular orbit: v = sqrt(G*M/(2*r))
    float G = 1.0f;
    float v = std::sqrt(G * mass / (2.0f * separation));

    // Body 1 cluster
    uint32_t half = count / 2;
    for (uint32_t i = 0; i < half; i++) {
        particles[i].position[0] = -separation / 2.0f + noise(rng());
        particles[i].position[1] = noise(rng());
        particles[i].position[2] = noise(rng());
        particles[i].position[3] = (i == 0) ? mass : 1.0f;

        particles[i].velocity[0] = 0.0f;
        particles[i].velocity[1] = (i == 0) ? v : v + noise(rng()) * 0.5f;
        particles[i].velocity[2] = 0.0f;
        particles[i].velocity[3] = 0.0f;

        particles[i].attributes[0] = 0.0f;
        particles[i].attributes[1] = 0.0f;
        particles[i].attributes[2] = 0.0f;
        particles[i].attributes[3] = 0.0f;
    }

    // Body 2 cluster
    for (uint32_t i = half; i < count; i++) {
        particles[i].position[0] = separation / 2.0f + noise(rng());
        particles[i].position[1] = noise(rng());
        particles[i].position[2] = noise(rng());
        particles[i].position[3] = (i == half) ? mass : 1.0f;

        particles[i].velocity[0] = 0.0f;
        particles[i].velocity[1] = (i == half) ? -v : -v + noise(rng()) * 0.5f;
        particles[i].velocity[2] = 0.0f;
        particles[i].velocity[3] = 0.0f;

        particles[i].attributes[0] = 0.0f;
        particles[i].attributes[1] = 0.0f;
        particles[i].attributes[2] = 0.0f;
        particles[i].attributes[3] = 0.0f;
    }
    return particles;
}

static std::vector<ParticleData> generateGalaxyCollision(uint32_t count) {
    std::vector<ParticleData> particles(count);
    std::uniform_real_distribution<float> angleDist(0.0f, 2.0f * 3.14159265f);
    std::uniform_real_distribution<float> radiusDist(0.0f, 1.0f);
    std::normal_distribution<float> heightDist(0.0f, 0.2f);

    float diskRadius = 10.0f;
    float separation = 30.0f;
    float approachSpeed = 2.0f;
    uint32_t half = count / 2;
    float galaxyMass = static_cast<float>(half);
    float G = 1.0f;

    // Galaxy 1
    for (uint32_t i = 0; i < half; i++) {
        float angle = angleDist(rng());
        float r = diskRadius * std::sqrt(radiusDist(rng()));
        r = std::max(r, 0.5f);

        particles[i].position[0] = -separation / 2.0f + r * std::cos(angle);
        particles[i].position[1] = heightDist(rng());
        particles[i].position[2] = r * std::sin(angle);
        particles[i].position[3] = (i == 0) ? galaxyMass * 0.3f : 1.0f;

        float v = std::sqrt(G * galaxyMass * 0.3f / r);
        particles[i].velocity[0] = approachSpeed + (-std::sin(angle)) * v;
        particles[i].velocity[1] = 0.0f;
        particles[i].velocity[2] = std::cos(angle) * v;
        particles[i].velocity[3] = 0.0f;

        particles[i].attributes[0] = 0.0f;
        particles[i].attributes[1] = 0.0f;
        particles[i].attributes[2] = 0.0f;
        particles[i].attributes[3] = 0.0f;
    }

    // Galaxy 2
    for (uint32_t i = half; i < count; i++) {
        float angle = angleDist(rng());
        float r = diskRadius * std::sqrt(radiusDist(rng()));
        r = std::max(r, 0.5f);

        particles[i].position[0] = separation / 2.0f + r * std::cos(angle);
        particles[i].position[1] = heightDist(rng());
        particles[i].position[2] = r * std::sin(angle);
        particles[i].position[3] = (i == half) ? galaxyMass * 0.3f : 1.0f;

        float v = std::sqrt(G * galaxyMass * 0.3f / r);
        particles[i].velocity[0] = -approachSpeed + (-std::sin(angle)) * v;
        particles[i].velocity[1] = 0.0f;
        particles[i].velocity[2] = std::cos(angle) * v;
        particles[i].velocity[3] = 0.0f;

        particles[i].attributes[0] = 0.0f;
        particles[i].attributes[1] = 0.0f;
        particles[i].attributes[2] = 0.0f;
        particles[i].attributes[3] = 0.0f;
    }
    return particles;
}

// ── Spiral Galaxy (Hernquist 1990 / Springel 2005) ────────────────────
//
// Three-component galaxy in virial equilibrium:
//   - Hernquist dark-matter halo (static analytic potential + live particles)
//   - Hernquist stellar bulge    (visible, exact DF velocities)
//   - Exponential stellar disk   (visible, Toomre Q ≈ 1.2)
//
// Halo velocities use the **exact Hernquist (1990) analytic distribution
// function** with rejection sampling — this produces a halo that is in
// perfect equilibrium rather than the Maxwellian/Jeans approximation.
//
// Disk uses proper Toomre Q with epicyclic frequency and asymmetric
// drift correction.
//
// Particle mass ratio m_halo/m_disk ≈ 2 to minimise two-body heating.
//
static std::vector<ParticleData> generateGalaxy(uint32_t count) {
    constexpr double PI  = 3.14159265358979323846;
    constexpr double G   = 1.0;

    std::vector<ParticleData> particles(count);
    std::uniform_real_distribution<double> unitDist(0.0, 1.0);
    std::uniform_real_distribution<double> angleDist(0.0, 2.0 * PI);
    std::normal_distribution<double> gaussDist(0.0, 1.0);

    // ── Structural parameters ────────────────────────────────────────
    // Using N-body units: a_halo = 1 as reference scale
    double ah  = 10.0;    // halo  Hernquist scale radius
    double Rd  = 2.0;     // disk  exponential scale length  (Rd/ah ~ 0.2)
    double z0  = 0.2;     // disk  scale height  (z0 = 0.1 Rd)
    double ab  = 0.5;     // bulge Hernquist scale radius

    // ── Mass budget ──────────────────────────────────────────────────
    double Mtot = static_cast<double>(count);
    double Mh = Mtot * 0.90;    // 90% in halo (dominant)
    double Md = Mtot * 0.08;    // 8%  in disk
    double Mb = Mtot * 0.02;    // 2%  in bulge

    // ── Particle counts ──────────────────────────────────────────────
    // Set for m_halo/m_disk ≈ 2:
    //   N_h/N_d = (M_h/M_d) / (m_h/m_d) = (90/8) / 2 = 5.625
    // Disk 15%, Bulge 5%, Halo 80%
    uint32_t diskCount  = static_cast<uint32_t>(count * 0.15);
    uint32_t bulgeCount = static_cast<uint32_t>(count * 0.05);
    uint32_t haloCount  = count - diskCount - bulgeCount;
    if (diskCount < 100) diskCount = 100;
    if (bulgeCount < 50)  bulgeCount = 50;

    double mDiskP  = Md / static_cast<double>(diskCount);
    double mBulgeP = Mb / static_cast<double>(bulgeCount);
    double mHaloP  = Mh / static_cast<double>(haloCount);
    // m_halo/m_disk ≈ (0.90/0.80) / (0.08/0.15) ≈ 2.1  — good

    // ── Total potential ──────────────────────────────────────────────
    auto Phi_total = [&](double r) -> double {
        double PhiH = -G * Mh / (r + ah);
        double PhiB = -G * Mb / (r + ab);
        // Disk monopole approximation
        double PhiD = -G * Md / std::sqrt(r * r + Rd * Rd);
        return PhiH + PhiB + PhiD;
    };

    auto dPhidr_total = [&](double r) -> double {
        double fH = G * Mh / ((r + ah) * (r + ah));
        double fB = G * Mb / ((r + ab) * (r + ab));
        // Disk enclosed mass (exponential): M_d(<R) = M_d [1 - (1 + R/Rd) e^{-R/Rd}]
        double mDenc = Md * (1.0 - (1.0 + r / Rd) * std::exp(-r / Rd));
        double fD = G * mDenc / std::max(r * r, 0.001);
        return fH + fB + fD;
    };

    auto vCirc = [&](double R) -> double {
        return std::sqrt(std::max(R * dPhidr_total(R), 0.0));
    };

    // ── Hernquist (1990) exact distribution function ─────────────────
    // f(E) for an isotropic Hernquist sphere in its OWN potential only
    // (for the halo component in the combined potential, we use the
    //  halo DF evaluated at binding energy from the TOTAL potential)
    //
    // q = sqrt(E_bind * a / (G*M)),   s = q²
    // f = M / (8√2 π³ a³ (GM/a)^{3/2}) × 1/(1-s)^{5/2}
    //     × [3 arcsin(√s) + √(s(1-s))(1-2s)(8s²-8s-3)]
    //
    // For compound potential: we use the DF of each component evaluated
    // at the binding energy E_bind = -Phi_total(r) - v²/2 but with the
    // DF parameters of that specific component. This is the "adiabatic"
    // approach from Springel et al. (2005).

    auto hernquistDF = [&](double E_bind, double M, double a) -> double {
        double vg = G * M / a;              // energy scale
        double s  = E_bind / vg;            // dimensionless ∈ (0,1)
        if (s <= 1e-10 || s >= 1.0 - 1e-10) return 0.0;

        double sqS  = std::sqrt(s);
        double s1   = 1.0 - s;
        double sqS1 = std::sqrt(s1);

        double term1 = 3.0 * std::asin(sqS);
        double term2 = sqS * sqS1 * (1.0 - 2.0 * s)
                       * (8.0 * s * s - 8.0 * s - 3.0);

        double prefactor = M / (8.0 * std::sqrt(2.0) * PI * PI * PI
                               * a * a * a * std::pow(vg, 1.5));
        return prefactor * (term1 + term2) / std::pow(s1, 2.5);
    };

    // Speed sampling via rejection for a Hernquist component
    // in the combined potential
    auto sampleSpeed = [&](double r, double M_comp, double a_comp) -> double {
        double psi = -Phi_total(r);  // relative potential (positive)
        if (psi <= 0.0) return 0.0;
        double v_esc = std::sqrt(2.0 * psi);

        // g(v|r) = 4π v² f(ψ - v²/2)
        // Find g_max by scanning
        double g_max = 0.0;
        constexpr int NSCAN = 200;
        for (int k = 1; k <= NSCAN; k++) {
            double v = v_esc * k / (NSCAN + 1.0);
            double E_bind = psi - 0.5 * v * v;
            if (E_bind <= 0.0) continue;
            double g = 4.0 * PI * v * v * hernquistDF(E_bind, M_comp, a_comp);
            if (g > g_max) g_max = g;
        }
        g_max *= 1.10; // 10% safety margin

        if (g_max <= 0.0) return 0.0;

        // Rejection sampling
        for (int attempt = 0; attempt < 10000; attempt++) {
            double v = v_esc * unitDist(rng());
            double E_bind = psi - 0.5 * v * v;
            if (E_bind <= 0.0) continue;
            double g = 4.0 * PI * v * v * hernquistDF(E_bind, M_comp, a_comp);
            if (unitDist(rng()) * g_max <= g) return v;
        }
        // Fallback: return low dispersion speed
        return v_esc * 0.1;
    };

    // ── Epicyclic frequency κ (numerical) ────────────────────────────
    auto epicyclicKappa = [&](double R) -> double {
        double vc = vCirc(R);
        double Omega = vc / std::max(R, 0.01);
        // κ² = R dΩ²/dR + 4Ω²
        double dR = R * 0.01;
        double vcP = vCirc(R + dR);
        double vcM = vCirc(std::max(R - dR, 0.01));
        double OmegaP = vcP / (R + dR);
        double OmegaM = vcM / std::max(R - dR, 0.01);
        double dOmega2dR = (OmegaP * OmegaP - OmegaM * OmegaM) / (2.0 * dR);
        double kappa2 = R * dOmega2dR + 4.0 * Omega * Omega;
        return std::sqrt(std::max(kappa2, Omega * Omega));  // κ ≥ Ω always
    };

    // ── Surface density ──────────────────────────────────────────────
    auto surfaceDensity = [&](double R) -> double {
        return Md / (2.0 * PI * Rd * Rd) * std::exp(-R / Rd);
    };

    uint32_t idx = 0;

    // ── 1. Disk particles ────────────────────────────────────────────
    for (uint32_t i = 0; i < diskCount && idx < count; i++, idx++) {
        // Sample exponential disk radius via inverse CDF
        double r;
        do {
            r = -Rd * std::log(std::max(unitDist(rng()), 1e-8));
        } while (r > Rd * 8.0 || r < Rd * 0.1);  // 0.1 Rd to 8 Rd

        double angle = angleDist(rng());
        double height = gaussDist(rng()) * z0;

        particles[idx].position[0] = static_cast<float>(r * std::cos(angle));
        particles[idx].position[1] = static_cast<float>(height);
        particles[idx].position[2] = static_cast<float>(r * std::sin(angle));
        particles[idx].position[3] = static_cast<float>(mDiskP);

        // Circular velocity from combined potential
        double vc = vCirc(r);

        // Epicyclic frequency
        double kappa = epicyclicKappa(r);
        double Omega = vc / std::max(r, 0.01);

        // Toomre Q = σ_R κ / (3.36 G Σ)
        // We set Q = 1.2 → σ_R = Q × 3.36 G Σ / κ
        double Q_toomre = 1.2;
        double Sigma = surfaceDensity(r);
        double sigmaR = Q_toomre * 3.36 * G * Sigma / std::max(kappa, 0.001);

        // Clamp σ_R: not more than 40% of v_c (prevents hot disk)
        // and not less than 2% of v_c (prevents cold fragmentation from noise)
        sigmaR = std::max(sigmaR, 0.02 * vc);
        sigmaR = std::min(sigmaR, 0.40 * vc);

        // Tangential dispersion: σ_φ = (κ / 2Ω) σ_R
        double sigmaRatio = kappa / (2.0 * std::max(Omega, 0.001));
        double sigmaPhi = sigmaRatio * sigmaR;

        // Vertical dispersion: σ_z ≈ 0.5 σ_R (thin disk)
        double sigmaZ = 0.5 * sigmaR;

        // Asymmetric drift correction (Binney & Tremaine 4.228):
        // <v_φ>² = v_c² + σ_R² [σ_φ²/σ_R² - 1 + 2R/Rd]
        // For exponential disk: d ln(ρ σ_R²)/d ln(R) → -R/Rd terms
        double adCorr = sigmaR * sigmaR
                        * (sigmaRatio * sigmaRatio - 1.0 + 2.0 * r / Rd);
        double vPhiMean2 = vc * vc + adCorr;
        // Safety floor: never slower than 30% of v_c
        double vPhiMean = std::sqrt(std::max(vPhiMean2, 0.09 * vc * vc));

        double vPhi = vPhiMean + gaussDist(rng()) * sigmaPhi;
        double vR   = gaussDist(rng()) * sigmaR;
        double vZ   = gaussDist(rng()) * sigmaZ;

        double cosA = std::cos(angle), sinA = std::sin(angle);
        particles[idx].velocity[0] = static_cast<float>(vR * cosA - vPhi * sinA);
        particles[idx].velocity[1] = static_cast<float>(vZ);
        particles[idx].velocity[2] = static_cast<float>(vR * sinA + vPhi * cosA);
        particles[idx].velocity[3] = 0.0f;

        particles[idx].attributes[0] = 0.0f;
        particles[idx].attributes[1] = 0.0f;
        particles[idx].attributes[2] = 0.0f;
        particles[idx].attributes[3] = 0.0f; // visible star (type 0)
    }

    // ── 2. Bulge particles (Hernquist, exact DF velocities) ──────────
    for (uint32_t i = 0; i < bulgeCount && idx < count; i++, idx++) {
        // Sample Hernquist radius: CDF P(<r) = r²/(r+a)²
        // Invert: r = a √u / (1 - √u)
        double u = std::min(unitDist(rng()), 0.995);
        double sqrtU = std::sqrt(u);
        double r = ab * sqrtU / (1.0 - sqrtU);
        r = std::min(r, ab * 30.0);
        r = std::max(r, 0.05);

        double cosTheta = 2.0 * unitDist(rng()) - 1.0;
        double sinTheta = std::sqrt(1.0 - cosTheta * cosTheta);
        double phi = angleDist(rng());

        particles[idx].position[0] = static_cast<float>(r * sinTheta * std::cos(phi));
        particles[idx].position[1] = static_cast<float>(r * cosTheta);
        particles[idx].position[2] = static_cast<float>(r * sinTheta * std::sin(phi));
        particles[idx].position[3] = static_cast<float>(mBulgeP);

        // Exact DF velocity sampling
        double speed = sampleSpeed(r, Mb, ab);

        // Isotropic direction
        double vCosTheta = 2.0 * unitDist(rng()) - 1.0;
        double vSinTheta = std::sqrt(1.0 - vCosTheta * vCosTheta);
        double vPhi = angleDist(rng());

        particles[idx].velocity[0] = static_cast<float>(speed * vSinTheta * std::cos(vPhi));
        particles[idx].velocity[1] = static_cast<float>(speed * vCosTheta);
        particles[idx].velocity[2] = static_cast<float>(speed * vSinTheta * std::sin(vPhi));
        particles[idx].velocity[3] = 0.0f;

        particles[idx].attributes[0] = 0.0f;
        particles[idx].attributes[1] = 0.0f;
        particles[idx].attributes[2] = 0.0f;
        particles[idx].attributes[3] = 0.0f; // visible star
    }

    // ── 3. Halo particles (Hernquist, exact DF velocities) ───────────
    for (uint32_t i = 0; i < haloCount && idx < count; i++, idx++) {
        double u = std::min(unitDist(rng()), 0.998);
        double sqrtU = std::sqrt(u);
        double r = ah * sqrtU / (1.0 - sqrtU);
        r = std::min(r, ah * 30.0);  // ~300 units
        r = std::max(r, 0.2);

        double cosTheta = 2.0 * unitDist(rng()) - 1.0;
        double sinTheta = std::sqrt(1.0 - cosTheta * cosTheta);
        double phi = angleDist(rng());

        particles[idx].position[0] = static_cast<float>(r * sinTheta * std::cos(phi));
        particles[idx].position[1] = static_cast<float>(r * cosTheta);
        particles[idx].position[2] = static_cast<float>(r * sinTheta * std::sin(phi));
        particles[idx].position[3] = static_cast<float>(mHaloP);

        // Exact DF velocity sampling
        double speed = sampleSpeed(r, Mh, ah);

        // Isotropic direction
        double vCosTheta = 2.0 * unitDist(rng()) - 1.0;
        double vSinTheta = std::sqrt(1.0 - vCosTheta * vCosTheta);
        double vPhi = angleDist(rng());

        particles[idx].velocity[0] = static_cast<float>(speed * vSinTheta * std::cos(vPhi));
        particles[idx].velocity[1] = static_cast<float>(speed * vCosTheta);
        particles[idx].velocity[2] = static_cast<float>(speed * vSinTheta * std::sin(vPhi));
        particles[idx].velocity[3] = 0.0f;

        particles[idx].attributes[0] = 0.0f;
        particles[idx].attributes[1] = 0.0f;
        particles[idx].attributes[2] = 0.0f;
        particles[idx].attributes[3] = -1.0f; // dark matter (invisible by default)
    }

    // ── Subtract center-of-mass position AND velocity ────────────────
    double comX = 0, comY = 0, comZ = 0;
    double comVx = 0, comVy = 0, comVz = 0, totalMass = 0;
    for (uint32_t i = 0; i < count; i++) {
        double m = particles[i].position[3];
        comX  += m * particles[i].position[0];
        comY  += m * particles[i].position[1];
        comZ  += m * particles[i].position[2];
        comVx += m * particles[i].velocity[0];
        comVy += m * particles[i].velocity[1];
        comVz += m * particles[i].velocity[2];
        totalMass += m;
    }
    comX /= totalMass; comY /= totalMass; comZ /= totalMass;
    comVx /= totalMass; comVy /= totalMass; comVz /= totalMass;
    for (uint32_t i = 0; i < count; i++) {
        particles[i].position[0] -= static_cast<float>(comX);
        particles[i].position[1] -= static_cast<float>(comY);
        particles[i].position[2] -= static_cast<float>(comZ);
        particles[i].velocity[0] -= static_cast<float>(comVx);
        particles[i].velocity[1] -= static_cast<float>(comVy);
        particles[i].velocity[2] -= static_cast<float>(comVz);
    }

    return particles;
}

static std::vector<ParticleData> generateCosmological(uint32_t count, float boxSize) {
    std::vector<ParticleData> particles(count);
    // Cold start: zero initial velocities (structure grows from density perturbations only)

    // Place particles on a uniform grid with small random jitter
    // Centered at origin: positions in [-boxSize/2, boxSize/2)
    float half = boxSize * 0.5f;

    // Find best cube root for grid placement
    int perSide = static_cast<int>(std::ceil(std::cbrt(static_cast<float>(count))));
    float spacing = boxSize / static_cast<float>(perSide);

    uint32_t idx = 0;
    for (int ix = 0; ix < perSide && idx < count; ix++) {
        for (int iy = 0; iy < perSide && idx < count; iy++) {
            for (int iz = 0; iz < perSide && idx < count; iz++) {
                float jitter = spacing * 0.02f;

                // Center at origin: [-half, half)
                float x = (ix + 0.5f) * spacing - half;
                float y = (iy + 0.5f) * spacing - half;
                float z = (iz + 0.5f) * spacing - half;

                // Small random jitter to break grid symmetry
                x += jitter * (2.0f * std::uniform_real_distribution<float>(0.0f, 1.0f)(rng()) - 1.0f);
                y += jitter * (2.0f * std::uniform_real_distribution<float>(0.0f, 1.0f)(rng()) - 1.0f);
                z += jitter * (2.0f * std::uniform_real_distribution<float>(0.0f, 1.0f)(rng()) - 1.0f);

                // Wrap to [-half, half)
                float hh = half;
                x = x - std::floor((x + hh) / boxSize) * boxSize;
                y = y - std::floor((y + hh) / boxSize) * boxSize;
                z = z - std::floor((z + hh) / boxSize) * boxSize;

                particles[idx].position[0] = x;
                particles[idx].position[1] = y;
                particles[idx].position[2] = z;
                particles[idx].position[3] = 1.0f; // unit mass

                particles[idx].velocity[0] = 0.0f;
                particles[idx].velocity[1] = 0.0f;
                particles[idx].velocity[2] = 0.0f;
                particles[idx].velocity[3] = 0.0f;

                particles[idx].attributes[0] = 0.0f;
                particles[idx].attributes[1] = 0.0f;
                particles[idx].attributes[2] = 0.0f;
                particles[idx].attributes[3] = 0.0f;

                idx++;
            }
        }
    }

    return particles;
}

// ── Solar System with real orbital data ──────────────────────────────
//
// Unit system (natural for Kepler with G = 1):
//   Distance : 1 unit = 1 AU
//   Mass     : 1 unit  chosen so that G*M_sun = 4π² AU³/yr²
//              => M_sun = 4π² ≈ 39.478
//   Velocity : AU/yr   (Earth orbital speed ≈ 2π AU/yr ≈ 6.283)
//   Time     : 1 yr    (with dt = 0.001 → ~1000 steps per year)
//
// Orbital elements: semi-major axis (AU), orbital speed (AU/yr),
// mass (M_sun ratio × 4π²), inclination (rad).
// All planets start at perihelion on the +X axis, velocity along +Z.
// NASA/JPL values (J2000 epoch mean elements).

static std::vector<ParticleData> generateSolarSystem(uint32_t count) {
    constexpr float PI = 3.14159265358979f;
    constexpr float TWOPI = 2.0f * PI;
    constexpr float GM_SUN = 4.0f * PI * PI;  // ≈ 39.478  (G=1 units)

    // Body data: name, semi-major axis (AU), eccentricity,
    //            inclination (deg), mass (kg) → converted below
    struct Body {
        float a;        // semi-major axis (AU)
        float e;        // eccentricity
        float inc;      // inclination to ecliptic (degrees)
        float mass;     // in G=1 units (relative to GM_SUN)
        float type;     // particle type tag
        float visualRadius; // artistically scaled visual radius (sim units)
    };

    // Mass ratios: mass_planet / mass_sun  (×GM_SUN to get sim mass)
    // NASA fact sheets
    constexpr float M_SUN_KG     = 1.989e30f;
    constexpr float M_MERCURY_KG = 3.301e23f;
    constexpr float M_VENUS_KG   = 4.867e24f;
    constexpr float M_EARTH_KG   = 5.972e24f;
    constexpr float M_MARS_KG    = 6.417e23f;
    constexpr float M_JUPITER_KG = 1.898e27f;
    constexpr float M_SATURN_KG  = 5.683e26f;
    constexpr float M_URANUS_KG  = 8.681e25f;
    constexpr float M_NEPTUNE_KG = 1.024e26f;

    // Real equatorial radii (km) — NASA planetary fact sheets
    // Visual radius = real radius × VR_SCALE (proportionally correct)
    // Scale chosen so Jupiter ≈ 0.01 AU visual radius (~21× real)
    constexpr float R_JUPITER_KM = 69911.0f;
    constexpr float VR_SCALE = 0.01f / R_JUPITER_KM;  // ≈ 1.43e-7

    Body bodies[] = {
        // Sun — capped at 0.006 (proportional would be ~0.1, still large vs Mercury orbit)
        { 0.0f,    0.0f,     0.0f,   GM_SUN,                                        0.0f, 0.006f                 },
        // Mercury (real radius 2440 km)
        { 0.3871f, 0.2056f,  7.005f, GM_SUN * (M_MERCURY_KG / M_SUN_KG),            1.0f, 2440.0f  * VR_SCALE   },
        // Venus (real radius 6052 km)
        { 0.7233f, 0.0068f,  3.395f, GM_SUN * (M_VENUS_KG   / M_SUN_KG),            2.0f, 6052.0f  * VR_SCALE   },
        // Earth (real radius 6371 km)
        { 1.0000f, 0.0167f,  0.000f, GM_SUN * (M_EARTH_KG   / M_SUN_KG),            3.0f, 6371.0f  * VR_SCALE   },
        // Mars (real radius 3390 km)
        { 1.5237f, 0.0934f,  1.850f, GM_SUN * (M_MARS_KG    / M_SUN_KG),            4.0f, 3390.0f  * VR_SCALE   },
        // Jupiter (real radius 69911 km)
        { 5.2026f, 0.0485f,  1.303f, GM_SUN * (M_JUPITER_KG / M_SUN_KG),            5.0f, 69911.0f * VR_SCALE   },
        // Saturn (real radius 58232 km)
        { 9.5549f, 0.0556f,  2.489f, GM_SUN * (M_SATURN_KG  / M_SUN_KG),            6.0f, 58232.0f * VR_SCALE   },
        // Uranus (real radius 25362 km)
        { 19.218f, 0.0472f,  0.773f, GM_SUN * (M_URANUS_KG  / M_SUN_KG),            7.0f, 25362.0f * VR_SCALE   },
        // Neptune (real radius 24622 km)
        { 30.110f, 0.0086f,  1.770f, GM_SUN * (M_NEPTUNE_KG / M_SUN_KG),            8.0f, 24622.0f * VR_SCALE   },
    };
    // Major moon orbital data
    struct MoonData {
        int parentIdx;    // index in particles array
        float a;          // semi-major axis around parent (AU)
        float e;          // eccentricity
        float inc;        // inclination (degrees)
        float massKg;     // mass (kg)
        float type;       // particle type tag
        float radiusKm;   // real equatorial radius (km)
    };

    MoonData moons[] = {
        // Moon (Earth)
        { 3, 0.00257f, 0.0549f,   5.16f, 7.342e22f, 11.0f, 1737.0f },
        // Io (Jupiter)
        { 5, 0.00282f, 0.0041f,   0.04f, 8.932e22f, 12.0f, 1822.0f },
        // Europa (Jupiter)
        { 5, 0.00449f, 0.009f,    0.47f, 4.800e22f, 13.0f, 1561.0f },
        // Ganymede (Jupiter)
        { 5, 0.00716f, 0.0013f,   0.18f, 1.482e23f, 14.0f, 2634.0f },
        // Callisto (Jupiter)
        { 5, 0.01258f, 0.0074f,   0.19f, 1.076e23f, 15.0f, 2410.0f },
        // Titan (Saturn)
        { 6, 0.00817f, 0.0288f,   0.33f, 1.345e23f, 16.0f, 2575.0f },
        // Enceladus (Saturn)
        { 6, 0.00159f, 0.0047f,   0.02f, 1.080e20f, 17.0f,  252.0f },
        // Triton (Neptune) — retrograde orbit
        { 8, 0.00237f, 0.0f,    156.90f, 2.140e22f, 18.0f, 1353.0f },
    };
    constexpr int NUM_MOONS = 8;
    constexpr int NUM_BODIES = 9 + NUM_MOONS;  // 17 (Sun + 8 planets + 8 moons)

    std::vector<ParticleData> particles(count);

    // Place the 9 real bodies (Sun + 8 planets)
    for (int b = 0; b < NUM_BODIES && b < static_cast<int>(count); b++) {
        const auto& body = bodies[b];
        float incRad = body.inc * (PI / 180.0f);

        if (b == 0) {
            // Sun at origin, stationary in heliocentric frame
            particles[b].position[0] = 0.0f;
            particles[b].position[1] = 0.0f;
            particles[b].position[2] = 0.0f;
            particles[b].position[3] = body.mass;
            particles[b].velocity[0] = 0.0f;
            particles[b].velocity[1] = 0.0f;
            particles[b].velocity[2] = 0.0f;
            particles[b].velocity[3] = body.visualRadius;
        } else {
            // Start at perihelion: r_peri = a(1-e)
            float rPeri = body.a * (1.0f - body.e);

            // Vis-viva at perihelion: v = sqrt(GM(1+e) / (a(1-e)))
            //                           = sqrt(GM_SUN * (1+e) / (a*(1-e)))
            float vPeri = std::sqrt(GM_SUN * (1.0f + body.e) / (body.a * (1.0f - body.e)));

            // Position at perihelion along +X, velocity along +Z
            // Then rotate by inclination around X axis
            float px = rPeri;
            float py = 0.0f;
            float pz = 0.0f;

            float vx = 0.0f;
            float vy = 0.0f;
            float vz = vPeri;

            // Apply inclination rotation (around X axis)
            float cosI = std::cos(incRad);
            float sinI = std::sin(incRad);
            float py2 = py * cosI - pz * sinI;
            float pz2 = py * sinI + pz * cosI;
            float vy2 = vy * cosI - vz * sinI;
            float vz2 = vy * sinI + vz * cosI;

            particles[b].position[0] = px;
            particles[b].position[1] = py2;
            particles[b].position[2] = pz2;
            particles[b].position[3] = body.mass;

            particles[b].velocity[0] = vx;
            particles[b].velocity[1] = vy2;
            particles[b].velocity[2] = vz2;
            particles[b].velocity[3] = body.visualRadius;
        }

        particles[b].attributes[0] = 0.0f;
        particles[b].attributes[1] = 0.0f;
        particles[b].attributes[2] = 0.0f;
        particles[b].attributes[3] = body.type;
    }

    // Place 8 major moons (heliocentric = parent position + orbital offset)
    // Moon orbital distances are scaled up to match the visual radius magnification,
    // otherwise moons appear to touch their parent planets.
    // VR_SCALE gives ~21× real radii, so we scale orbits by 10× for a natural look.
    constexpr float MOON_ORBIT_SCALE = 10.0f;
    constexpr float GOLDEN_ANGLE = 2.39996322f; // ~137.508° — spreads moons around parent
    for (int m = 0; m < NUM_MOONS && (9 + m) < static_cast<int>(count); m++) {
        const auto& moon = moons[m];
        int idx = 9 + m;
        int pIdx = moon.parentIdx;

        // Parent's heliocentric state (already placed above)
        float ppx = particles[pIdx].position[0];
        float ppy = particles[pIdx].position[1];
        float ppz = particles[pIdx].position[2];
        float pvx = particles[pIdx].velocity[0];
        float pvy = particles[pIdx].velocity[1];
        float pvz = particles[pIdx].velocity[2];

        // GM of parent body (G=1, so GM = mass stored in position[3])
        float gmParent = particles[pIdx].position[3];

        // Perihelion distance and vis-viva speed at perihelion
        float aScaled = moon.a * MOON_ORBIT_SCALE;
        float rPeri = aScaled * (1.0f - moon.e);
        float vPeri = std::sqrt(gmParent * (1.0f + moon.e) / (aScaled * (1.0f - moon.e)));

        // Deterministic angular placement around parent
        float angle = m * GOLDEN_ANGLE;

        // Position offset in orbital plane (XZ)
        float dx  =  rPeri * std::cos(angle);
        float dy  =  0.0f;
        float dz  =  rPeri * std::sin(angle);

        // Velocity offset (tangential to orbit)
        float dvx = -std::sin(angle) * vPeri;
        float dvy =  0.0f;
        float dvz =  std::cos(angle) * vPeri;

        // Apply inclination rotation around X axis
        // (Triton's 156.9° naturally produces retrograde orbit)
        float incRad = moon.inc * (PI / 180.0f);
        float cosI = std::cos(incRad);
        float sinI = std::sin(incRad);

        float dy2  = dy  * cosI - dz  * sinI;
        float dz2  = dy  * sinI + dz  * cosI;
        float dvy2 = dvy * cosI - dvz * sinI;
        float dvz2 = dvy * sinI + dvz * cosI;

        // Moon mass in sim units (G=1)
        float mass = GM_SUN * (moon.massKg / M_SUN_KG);

        // Heliocentric state = parent + offset
        particles[idx].position[0] = ppx + dx;
        particles[idx].position[1] = ppy + dy2;
        particles[idx].position[2] = ppz + dz2;
        particles[idx].position[3] = mass;

        particles[idx].velocity[0] = pvx + dvx;
        particles[idx].velocity[1] = pvy + dvy2;
        particles[idx].velocity[2] = pvz + dvz2;
        // Moons use 1/4 of VR_SCALE — they orbit close to parents so
        // full magnification makes them look disproportionately large
        particles[idx].velocity[3] = moon.radiusKm * VR_SCALE * 0.25f;

        particles[idx].attributes[0] = 0.0f;
        particles[idx].attributes[1] = 0.0f;
        particles[idx].attributes[2] = 0.0f;
        particles[idx].attributes[3] = moon.type;
    }

    // Fill remaining particles as asteroid belt + Kuiper belt test particles
    if (count > static_cast<uint32_t>(NUM_BODIES)) {
        std::uniform_real_distribution<float> angleDist(0.0f, TWOPI);
        std::uniform_real_distribution<float> unitDist(0.0f, 1.0f);
        std::normal_distribution<float> heightDist(0.0f, 0.05f);  // thin disk

        uint32_t remaining = count - static_cast<uint32_t>(NUM_BODIES);
        uint32_t asteroidCount = remaining * 2 / 3;  // 2/3 asteroids, 1/3 Kuiper

        for (uint32_t i = NUM_BODIES; i < count; i++) {
            uint32_t idx = i - NUM_BODIES;
            float angle = angleDist(rng());
            float r;
            float type;

            if (idx < asteroidCount) {
                // Asteroid belt: 2.1 - 3.3 AU
                r = 2.1f + unitDist(rng()) * 1.2f;
                type = 9.0f;  // asteroid
            } else {
                // Kuiper belt: 30 - 50 AU
                r = 30.0f + unitDist(rng()) * 20.0f;
                type = 10.0f; // KBO
            }

            float height = heightDist(rng()) * r;

            float px = r * std::cos(angle);
            float pz = r * std::sin(angle);
            float py = height;

            float v = std::sqrt(GM_SUN / r);
            float vx = -std::sin(angle) * v;
            float vz =  std::cos(angle) * v;

            float mass = GM_SUN * 1.0e-15f;

            particles[i].position[0] = px;
            particles[i].position[1] = py;
            particles[i].position[2] = pz;
            particles[i].position[3] = mass;

            particles[i].velocity[0] = vx;
            particles[i].velocity[1] = 0.0f;
            particles[i].velocity[2] = vz;
            particles[i].velocity[3] = 0.0f;

            particles[i].attributes[0] = 0.0f;
            particles[i].attributes[1] = 0.0f;
            particles[i].attributes[2] = 0.0f;
            particles[i].attributes[3] = type;
        }
    }

    return particles;
}

std::vector<ParticleData> generateInitialConditions(InitialCondition type, uint32_t count,
                                                    float boxSize) {
    rng().seed(42); // Deterministic seed for reproducibility

    // Auto-scale boxSize if not provided (0): maintain spacing ~ 1.5625 * cbrt(count)
    if (boxSize <= 0.0f) {
        boxSize = 1.5625f * std::cbrt(static_cast<float>(count));
    }

    switch (type) {
        case InitialCondition::Sphere: return generateSphere(count);
        case InitialCondition::Disk: return generateDisk(count);
        case InitialCondition::TwoBody: return generateTwoBody(count);
        case InitialCondition::GalaxyCollision: return generateGalaxyCollision(count);
        case InitialCondition::Galaxy: return generateGalaxy(count);
        case InitialCondition::Cosmological: return generateCosmological(count, boxSize);
        case InitialCondition::SolarSystem: return generateSolarSystem(count);
        default: return generateDisk(count);
    }
}

uint32_t sphereBodyCount(InitialCondition type) {
    if (type == InitialCondition::SolarSystem)
        return 17; // Sun + 8 planets + 8 moons
    return 0;
}

} // namespace cosmico
