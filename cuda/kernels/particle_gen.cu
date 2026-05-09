#include <kernels/particle_gen.cuh>
#include <curand_kernel.h>
#include <cstdio>
#include <cmath>

namespace cosmico {
namespace cuda {

// ── Device helpers ──────────────────────────────────────────────────────

__device__ float3 rejectSampleSphere(curandState& state) {
    float x, y, z;
    do {
        x = 2.0f * curand_uniform(&state) - 1.0f;
        y = 2.0f * curand_uniform(&state) - 1.0f;
        z = 2.0f * curand_uniform(&state) - 1.0f;
    } while (x*x + y*y + z*z > 1.0f);
    return make_float3(x, y, z);
}

__device__ float plummerRadius(curandState& state, float a) {
    float u = curand_uniform(&state);
    // r/a = (u^(-2/3) - 1)^(-1/2)
    return a / sqrtf(powf(u, -2.0f/3.0f) - 1.0f);
}

__device__ float hernquistRadius(curandState& state, float a) {
    float u = curand_uniform(&state);
    // M(r)/M = r^2/(r+a)^2 => r = a * sqrt(u) / (1 - sqrt(u))
    float su = sqrtf(u);
    return a * su / fmaxf(1.0f - su, 1e-6f);
}

__device__ float nfwRadius(curandState& state, float rs) {
    // Simplified NFW sampling via rejection
    float u = curand_uniform(&state) * 5.0f; // sample up to 5*rs
    float r = u * rs;
    return r;
}

__device__ float maxwellBoltzmannSpeed(curandState& state, float temp) {
    // Box-Muller for 3 Gaussian components, take magnitude
    float u1 = fmaxf(curand_uniform(&state), 1e-8f);
    float u2 = curand_uniform(&state);
    float u3 = fmaxf(curand_uniform(&state), 1e-8f);
    float u4 = curand_uniform(&state);
    float u5 = fmaxf(curand_uniform(&state), 1e-8f);
    float u6 = curand_uniform(&state);
    float vx = sqrtf(-2.0f * logf(u1) * temp) * cosf(6.283185f * u2);
    float vy = sqrtf(-2.0f * logf(u3) * temp) * cosf(6.283185f * u4);
    float vz = sqrtf(-2.0f * logf(u5) * temp) * cosf(6.283185f * u6);
    return sqrtf(vx*vx + vy*vy + vz*vz);
}

// ── Main IC generation kernel ───────────────────────────────────────────

__global__ void icGenerateKernel(ParticleGpu* particles, ICGenParams params) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= params.particleCount) return;

    curandState state;
    curand_init(params.seed, i, 0, &state);

    float px = 0, py = 0, pz = 0;
    float vx = 0, vy = 0, vz = 0;
    float mass = params.massPerParticle;
    float temp = 0;
    float ptype = 0;

    switch (params.icType) {
    case 0: { // Sphere
        float3 p = rejectSampleSphere(state);
        float r;
        if (params.densityProfile == 1) {
            r = plummerRadius(state, params.profileScale > 0 ? params.profileScale : params.radius * 0.3f);
            r = fminf(r, params.radius);
        } else if (params.densityProfile == 2) {
            r = hernquistRadius(state, params.profileScale > 0 ? params.profileScale : params.radius * 0.3f);
            r = fminf(r, params.radius);
        } else if (params.densityProfile == 3) {
            r = nfwRadius(state, params.profileScale > 0 ? params.profileScale : params.radius * 0.2f);
            r = fminf(r, params.radius);
        } else {
            r = params.radius * cbrtf(p.x*p.x + p.y*p.y + p.z*p.z);
        }
        float norm = sqrtf(p.x*p.x + p.y*p.y + p.z*p.z);
        if (norm > 1e-4f) { p.x /= norm; p.y /= norm; p.z /= norm; }
        px = p.x * r;
        py = p.y * r;
        pz = p.z * r;

        // Velocity
        if (params.velocityDist == 1) {
            vx = (2.0f * curand_uniform(&state) - 1.0f) * params.velocitySpread;
            vy = (2.0f * curand_uniform(&state) - 1.0f) * params.velocitySpread;
            vz = (2.0f * curand_uniform(&state) - 1.0f) * params.velocitySpread;
        } else if (params.velocityDist == 2) {
            float speed = maxwellBoltzmannSpeed(state, params.temperature);
            float3 dir = rejectSampleSphere(state);
            float dnorm = sqrtf(dir.x*dir.x + dir.y*dir.y + dir.z*dir.z);
            if (dnorm > 1e-4f) { dir.x /= dnorm; dir.y /= dnorm; dir.z /= dnorm; }
            vx = dir.x * speed;
            vy = dir.y * speed;
            vz = dir.z * speed;
        }
        break;
    }
    case 1: { // Disk
        if (i == 0) {
            // Central massive particle
            px = py = pz = 0;
            mass = params.centralMass;
            vx = vy = vz = 0;
        } else {
            float angle = curand_uniform(&state) * 6.283185f;
            float r = params.diskRadius * sqrtf(curand_uniform(&state));
            r = fmaxf(r, 1.0f);
            px = r * cosf(angle);
            pz = r * sinf(angle);
            py = curand_normal(&state) * params.heightScale;

            float v = sqrtf(fmaxf(params.centralMass / r, 0.0f));
            vx = -sinf(angle) * v;
            vy = 0;
            vz = cosf(angle) * v;
        }
        break;
    }
    case 2: { // TwoBody
        int half = params.particleCount / 2;
        float noise = curand_normal(&state) * 0.1f;
        if (i < half) {
            px = -params.separation / 2.0f + noise;
            py = curand_normal(&state) * 0.1f;
            pz = curand_normal(&state) * 0.1f;
            float v = sqrtf(params.bodyMass / (2.0f * params.separation));
            vy = (i == 0) ? v : v + curand_normal(&state) * 0.05f;
            mass = (i == 0) ? params.bodyMass : params.massPerParticle;
        } else {
            px = params.separation / 2.0f + noise;
            py = curand_normal(&state) * 0.1f;
            pz = curand_normal(&state) * 0.1f;
            float v = sqrtf(params.bodyMass / (2.0f * params.separation));
            vy = (i == half) ? -v : -v + curand_normal(&state) * 0.05f;
            mass = (i == half) ? params.bodyMass : params.massPerParticle;
        }
        break;
    }
    case 3: { // GalaxyCollision
        int half = params.particleCount / 2;
        float galaxyMass = (float)half;
        float angle = curand_uniform(&state) * 6.283185f;
        float r = params.diskRadius * sqrtf(curand_uniform(&state));
        r = fmaxf(r, 0.5f);
        float h = curand_normal(&state) * 0.2f;

        float orbV = sqrtf(fmaxf(galaxyMass * 0.3f / r, 0.0f));
        float tx = -sinf(angle);
        float tz = cosf(angle);

        if (i < half) {
            px = -params.separation / 2.0f + r * cosf(angle);
            py = h;
            pz = r * sinf(angle);
            mass = (i == 0) ? galaxyMass * 0.3f : params.massPerParticle;
            vx = params.approachSpeed + tx * orbV;
            vy = 0;
            vz = tz * orbV;
        } else {
            px = params.separation / 2.0f + r * cosf(angle);
            py = h;
            pz = r * sinf(angle);
            mass = (i == half) ? galaxyMass * 0.3f : params.massPerParticle;
            vx = -params.approachSpeed + tx * orbV;
            vy = 0;
            vz = tz * orbV;
        }
        break;
    }
    case 4: { // Cosmological
        float L = params.boxSize;
        float halfBox = L * 0.5f;
        int perSide = (int)ceilf(cbrtf((float)params.particleCount));
        float spacing = L / (float)perSide;

        int iz = i % perSide;
        int iy = (i / perSide) % perSide;
        int ix = i / (perSide * perSide);

        px = (ix + 0.5f) * spacing - halfBox;
        py = (iy + 0.5f) * spacing - halfBox;
        pz = (iz + 0.5f) * spacing - halfBox;

        float jitter = spacing * params.jitterFraction;
        px += (2.0f * curand_uniform(&state) - 1.0f) * jitter;
        py += (2.0f * curand_uniform(&state) - 1.0f) * jitter;
        pz += (2.0f * curand_uniform(&state) - 1.0f) * jitter;

        // Wrap to [-half, half)
        px = px - floorf((px + halfBox) / L) * L;
        py = py - floorf((py + halfBox) / L) * L;
        pz = pz - floorf((pz + halfBox) / L) * L;

        vx = vy = vz = 0;
        break;
    }
    case 5: { // SolarSystem
        // Units: distance=AU, G=1, M_sun=4π², velocity=AU/yr
        constexpr float PI_F  = 3.14159265f;
        constexpr float GM_S  = 4.0f * PI_F * PI_F;  // ~39.478

        // Planet data: a(AU), e, inc(rad), mass(sim), type, visualRadius
        // Sun + Mercury..Neptune = 9 bodies
        constexpr int NUM_BODIES = 9;
        const float bodyA[]    = { 0.0f,    0.3871f, 0.7233f, 1.0000f, 1.5237f, 5.2026f, 9.5549f, 19.218f, 30.110f };
        const float bodyE[]    = { 0.0f,    0.2056f, 0.0068f, 0.0167f, 0.0934f, 0.0485f, 0.0556f, 0.0472f, 0.0086f };
        const float bodyInc[]  = { 0.0f,    0.12228f, 0.05926f, 0.0f, 0.03229f, 0.02274f, 0.04344f, 0.01349f, 0.03089f }; // radians
        const float bodyMass[] = {
            GM_S,
            GM_S * 1.660e-7f,   // Mercury
            GM_S * 2.448e-6f,   // Venus
            GM_S * 3.003e-6f,   // Earth
            GM_S * 3.227e-7f,   // Mars
            GM_S * 9.545e-4f,   // Jupiter
            GM_S * 2.858e-4f,   // Saturn
            GM_S * 4.366e-5f,   // Uranus
            GM_S * 5.150e-5f    // Neptune
        };
        const float bodyType[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8 };
        const float bodyVisualRadius[] = { 1.5f, 0.15f, 0.25f, 0.25f, 0.18f, 0.6f, 0.55f, 0.35f, 0.35f };

        if (i < NUM_BODIES) {
            if (i == 0) {
                // Sun
                px = py = pz = 0;
                vx = vy = vz = 0;
            } else {
                float rPeri = bodyA[i] * (1.0f - bodyE[i]);
                float vPeri = sqrtf(GM_S * (1.0f + bodyE[i]) / (bodyA[i] * (1.0f - bodyE[i])));
                float cosI = cosf(bodyInc[i]);
                float sinI = sinf(bodyInc[i]);
                px = rPeri;
                py = 0;
                pz = 0;
                vx = 0;
                vy = -vPeri * sinI;
                vz =  vPeri * cosI;
            }
            mass = bodyMass[i];
            ptype = bodyType[i];
            temp = bodyVisualRadius[i];
        } else {
            // Filler: asteroid belt (2/3) + Kuiper belt (1/3)
            int idx = i - NUM_BODIES;
            int remaining = params.particleCount - NUM_BODIES;
            int asteroidCount = remaining * 2 / 3;

            float angle = curand_uniform(&state) * 6.283185f;
            float r;
            if (idx < asteroidCount) {
                r = 2.1f + curand_uniform(&state) * 1.2f;
                ptype = 9.0f;
                temp = 0.03f;
            } else {
                r = 30.0f + curand_uniform(&state) * 20.0f;
                ptype = 10.0f;
                temp = 0.04f;
            }
            float h = curand_normal(&state) * 0.05f * r;
            px = r * cosf(angle);
            pz = r * sinf(angle);
            py = h;

            float v = sqrtf(GM_S / r);
            vx = -sinf(angle) * v;
            vy = 0;
            vz =  cosf(angle) * v;

            mass = GM_S * 1.0e-15f;
        }
        break;
    }
    default:
        break;
    }

    // Noise perturbation
    if (params.noiseType == 2) { // White noise
        px += (2.0f * curand_uniform(&state) - 1.0f) * params.noiseAmplitude;
        py += (2.0f * curand_uniform(&state) - 1.0f) * params.noiseAmplitude;
        pz += (2.0f * curand_uniform(&state) - 1.0f) * params.noiseAmplitude;
    }

    // Mass function
    if (params.massFunction == 1) { // PowerLaw
        float u = curand_uniform(&state);
        mass = params.massPerParticle * powf(u, 1.0f / params.powerLawIndex);
    }

    particles[i].px = px;
    particles[i].py = py;
    particles[i].pz = pz;
    particles[i].mass = mass;
    particles[i].vx = vx;
    particles[i].vy = vy;
    particles[i].vz = vz;
    particles[i].temp = temp;
    particles[i].density = 0;
    particles[i].pressure = 0;
    particles[i].smoothing = 0;
    particles[i].type = ptype;
}

// ── Normalize kernel: center + remove bulk momentum ────────────────────

__global__ void icNormalizeKernel(ParticleGpu* particles, int N,
                                  float* d_scratch) {
    // d_scratch layout: [sumPx, sumPy, sumPz, sumVx, sumVy, sumVz, totalMass]
    // Phase 1: accumulate (atomics)
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    atomicAdd(&d_scratch[0], particles[i].px * particles[i].mass);
    atomicAdd(&d_scratch[1], particles[i].py * particles[i].mass);
    atomicAdd(&d_scratch[2], particles[i].pz * particles[i].mass);
    atomicAdd(&d_scratch[3], particles[i].vx * particles[i].mass);
    atomicAdd(&d_scratch[4], particles[i].vy * particles[i].mass);
    atomicAdd(&d_scratch[5], particles[i].vz * particles[i].mass);
    atomicAdd(&d_scratch[6], particles[i].mass);
}

__global__ void icApplyNormalization(ParticleGpu* particles, int N,
                                     float* d_scratch) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float invM = 1.0f / fmaxf(d_scratch[6], 1e-8f);
    float cx = d_scratch[0] * invM;
    float cy = d_scratch[1] * invM;
    float cz = d_scratch[2] * invM;
    float cvx = d_scratch[3] * invM;
    float cvy = d_scratch[4] * invM;
    float cvz = d_scratch[5] * invM;

    particles[i].px -= cx;
    particles[i].py -= cy;
    particles[i].pz -= cz;
    particles[i].vx -= cvx;
    particles[i].vy -= cvy;
    particles[i].vz -= cvz;
}

// ── Host launch function ────────────────────────────────────────────────

void launchICGenerate(ParticleGpu* particles, const ICGenParams& params, cudaStream_t stream) {
    int N = params.particleCount;
    if (N <= 0) return;

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    // Copy params to device constant memory via kernel argument
    icGenerateKernel<<<numBlocks, blockSize, 0, stream>>>(particles, params);

    // Normalize: center of mass + bulk velocity removal
    float* d_scratch = nullptr;
    cudaMallocAsync(&d_scratch, 7 * sizeof(float), stream);
    cudaMemsetAsync(d_scratch, 0, 7 * sizeof(float), stream);

    icNormalizeKernel<<<numBlocks, blockSize, 0, stream>>>(particles, N, d_scratch);
    icApplyNormalization<<<numBlocks, blockSize, 0, stream>>>(particles, N, d_scratch);

    cudaFreeAsync(d_scratch, stream);

    fprintf(stderr, "[ICGen] Generated %d particles (type=%d) on GPU\n", N, params.icType);
}

} // namespace cuda
} // namespace cosmico
