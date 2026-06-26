#include <cosmico/nodes/PMNodes.h>
#include <cosmico/nodes/ResourcePool.h>
#include <kernels/pm_gravity.cuh>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cmath>
#include <cstdio>

namespace cosmico {

// Helper to create pins with incremental IDs
static int s_nextPinId = 10000;  // Start high to avoid conflicts with node IDs

static Pin makePin(const char* name, PinType type, bool isOutput) {
    Pin p;
    p.id = s_nextPinId++;
    p.name = name;
    p.type = type;
    p.isOutput = isOutput;
    return p;
}

// Helper to add a param declaration
static void addParam(SimNode& node, const std::string& key, const std::string& label,
                     const ParamValue& defaultVal, const std::string& tooltip = "",
                     ParamDescriptor::UIHint hint = ParamDescriptor::UIHint::Default,
                     bool inlineDisp = false,
                     std::optional<float> minV = std::nullopt,
                     std::optional<float> maxV = std::nullopt,
                     const std::string& fmt = "") {
    NodeParam np;
    np.descriptor.key = key;
    np.descriptor.label = label;
    np.descriptor.tooltip = tooltip;
    np.descriptor.defaultValue = defaultVal;
    np.descriptor.minVal = minV;
    np.descriptor.maxVal = maxV;
    np.descriptor.uiHint = hint;
    np.descriptor.inlineDisplay = inlineDisp;
    np.descriptor.format = fmt;
    np.value = defaultVal;
    np.overridden = false;
    node.params.push_back(std::move(np));
}

static void addBoolParam(SimNode& node, const std::string& key, const std::string& label,
                         bool defaultVal, const std::string& tooltip = "",
                         bool inlineDisp = false) {
    NodeParam np;
    np.descriptor.key = key;
    np.descriptor.label = label;
    np.descriptor.tooltip = tooltip;
    np.descriptor.defaultValue = defaultVal;
    np.descriptor.uiHint = ParamDescriptor::UIHint::Checkbox;
    np.descriptor.inlineDisplay = inlineDisp;
    np.value = defaultVal;
    np.overridden = false;
    node.params.push_back(std::move(np));
}

// ─── HalfDriftNode ───────────────────────────────────────────────────

HalfDriftNode::HalfDriftNode() {
    name = "Half Drift";
    typeName = "HalfDrift";
    inputs.push_back(makePin("particles", PinType::Particles, false));
    outputs.push_back(makePin("particles", PinType::Particles, true));
    declareParams();
}

void HalfDriftNode::declareParams() {
    params.clear();
    addParam(*this, "dt", "Timestep", 0.001f, "Integration timestep",
             ParamDescriptor::UIHint::Drag, true, 0.0001f, 0.1f, "%.4f");
    addParam(*this, "boxSize", "Box Size", 100.0f, "Simulation box size",
             ParamDescriptor::UIHint::Drag, false, 1.0f, 10000.0f, "%.1f");
    addBoolParam(*this, "openBoundary", "Open Boundary", false,
                 "Disable periodic wrapping");
}

void HalfDriftNode::execute(ResourcePool& pool, cudaStream_t stream) {
    auto* particles = static_cast<cuda::ParticleGpu*>(pool.particleBuffer());

    float dt = getFloat("dt", pool);
    float boxSize = getFloat("boxSize", pool);
    bool openBound = getBool("openBoundary", pool);

    float driftDt;
    if (pool.comoving) {
        driftDt = dt / pool.scaleFactor;
    } else {
        driftDt = dt;
    }
    float halfDt = driftDt * 0.5f;
    float driftBox = openBound ? 1e18f : boxSize;

    cuda::launchPMDrift(particles, pool.particleCount(), halfDt, driftBox, stream);
}

// ─── ClearGridNode ───────────────────────────────────────────────────

ClearGridNode::ClearGridNode() {
    name = "Clear Grid";
    typeName = "ClearGrid";
    outputs.push_back(makePin("grid", PinType::DensityGrid, true));
}

void ClearGridNode::execute(ResourcePool& pool, cudaStream_t stream) {
    cuda::launchPMClearGrid(pool.densityGrid(), pool.gridN3(), stream);
}

// ─── CICDepositNode ─────────────────────────────────────────────────

CICDepositNode::CICDepositNode() {
    name = "CIC Deposit";
    typeName = "CICDeposit";
    inputs.push_back(makePin("particles", PinType::Particles, false));
    inputs.push_back(makePin("grid", PinType::DensityGrid, false));
    outputs.push_back(makePin("densityGrid", PinType::DensityGrid, true));
    declareParams();
}

void CICDepositNode::declareParams() {
    params.clear();
    addParam(*this, "boxSize", "Box Size", 100.0f, "Simulation box size",
             ParamDescriptor::UIHint::Drag, false, 1.0f, 10000.0f, "%.1f");
}

void CICDepositNode::execute(ResourcePool& pool, cudaStream_t stream) {
    auto* particles = static_cast<cuda::ParticleGpu*>(pool.particleBuffer());
    float boxSize = getFloat("boxSize", pool);
    cuda::launchPMDeposit(particles, pool.densityGrid(),
                          pool.particleCount(), pool.gridN(),
                          boxSize, pool.params().openBoundary, stream);
}

// ─── FFTForwardNode ─────────────────────────────────────────────────

FFTForwardNode::FFTForwardNode() {
    name = "FFT Forward";
    typeName = "FFTForward";
    inputs.push_back(makePin("densityGrid", PinType::DensityGrid, false));
    outputs.push_back(makePin("complexGrid", PinType::ComplexGrid, true));
}

void FFTForwardNode::execute(ResourcePool& pool, cudaStream_t stream) {
    cufftSetStream(pool.fftPlanR2C(), stream);
    cufftExecR2C(pool.fftPlanR2C(),
                 static_cast<cufftReal*>(static_cast<void*>(pool.densityGrid())),
                 static_cast<cufftComplex*>(pool.complexGrid()));
}

// ─── GreenMultiplyNode ──────────────────────────────────────────────

GreenMultiplyNode::GreenMultiplyNode() {
    name = "Green Multiply";
    typeName = "GreenMultiply";
    inputs.push_back(makePin("complexGrid", PinType::ComplexGrid, false));
    outputs.push_back(makePin("complexGrid", PinType::ComplexGrid, true));
    declareParams();
}

void GreenMultiplyNode::declareParams() {
    params.clear();
    addParam(*this, "G", "Gravity (G)", 1.0f, "Gravitational constant",
             ParamDescriptor::UIHint::Drag, true, 0.0f, 100.0f, "%.3f");
}

void GreenMultiplyNode::execute(ResourcePool& pool, cudaStream_t stream) {
    float G_val = getFloat("G", pool);
    float boxSize = pool.params().boxSize;

    float G_eff;
    if (pool.comoving) {
        static constexpr float PI_F = 3.14159265358979323846f;
        float L = boxSize;
        float rhoMean = static_cast<float>(pool.particleCount()) / (L * L * L);
        float G_cosmo = (3.0f * pool.params().H0 * pool.params().H0 * pool.params().OmegaM)
                       / (8.0f * PI_F * rhoMean);
        G_eff = G_cosmo / pool.scaleFactor;
    } else {
        G_eff = G_val;
    }

    float smoothRadius = 2.0f * boxSize / std::cbrt(static_cast<float>(pool.particleCount()));

    cuda::launchPMGreenMultiply(static_cast<cufftComplex*>(pool.complexGrid()),
                                pool.gridN(), pool.nComplex(),
                                G_eff, pool.cellSize(), smoothRadius, stream);
}

// ─── FFTInverseNode ─────────────────────────────────────────────────

FFTInverseNode::FFTInverseNode() {
    name = "FFT Inverse";
    typeName = "FFTInverse";
    inputs.push_back(makePin("complexGrid", PinType::ComplexGrid, false));
    outputs.push_back(makePin("potentialGrid", PinType::DensityGrid, true));
}

void FFTInverseNode::execute(ResourcePool& pool, cudaStream_t stream) {
    cufftSetStream(pool.fftPlanC2R(), stream);
    cufftExecC2R(pool.fftPlanC2R(),
                 static_cast<cufftComplex*>(pool.complexGrid()),
                 static_cast<cufftReal*>(static_cast<void*>(pool.densityGrid())));
}

// ─── GradientNode ───────────────────────────────────────────────────

GradientNode::GradientNode() {
    name = "Gradient";
    typeName = "Gradient";
    inputs.push_back(makePin("potentialGrid", PinType::DensityGrid, false));
    outputs.push_back(makePin("forceX", PinType::ForceField, true));
    outputs.push_back(makePin("forceY", PinType::ForceField, true));
    outputs.push_back(makePin("forceZ", PinType::ForceField, true));
}

void GradientNode::execute(ResourcePool& pool, cudaStream_t stream) {
    cuda::launchPMGradient(pool.densityGrid(),
                           pool.forceGrid(0), pool.forceGrid(1), pool.forceGrid(2),
                           pool.gridN(), pool.cellSize(),
                           pool.params().openBoundary, stream);
}

// ─── CICInterpolateNode ─────────────────────────────────────────────

CICInterpolateNode::CICInterpolateNode() {
    name = "CIC Interpolate";
    typeName = "CICInterpolate";
    inputs.push_back(makePin("particles", PinType::Particles, false));
    inputs.push_back(makePin("forceX", PinType::ForceField, false));
    inputs.push_back(makePin("forceY", PinType::ForceField, false));
    inputs.push_back(makePin("forceZ", PinType::ForceField, false));
    outputs.push_back(makePin("particles", PinType::Particles, true));
    declareParams();
}

void CICInterpolateNode::declareParams() {
    params.clear();
    addParam(*this, "boxSize", "Box Size", 100.0f, "Simulation box size",
             ParamDescriptor::UIHint::Drag, false, 1.0f, 10000.0f, "%.1f");
}

void CICInterpolateNode::execute(ResourcePool& pool, cudaStream_t stream) {
    auto* particles = static_cast<cuda::ParticleGpu*>(pool.particleBuffer());
    float boxSize = getFloat("boxSize", pool);
    cuda::launchPMInterpolate(particles,
                              pool.forceGrid(0), pool.forceGrid(1), pool.forceGrid(2),
                              pool.particleCount(), pool.gridN(),
                              boxSize, pool.params().openBoundary, stream);
}

// ─── KickNode ───────────────────────────────────────────────────────

KickNode::KickNode() {
    name = "Kick";
    typeName = "Kick";
    inputs.push_back(makePin("particles", PinType::Particles, false));
    outputs.push_back(makePin("particles", PinType::Particles, true));
    declareParams();
}

void KickNode::declareParams() {
    params.clear();
    addParam(*this, "dt", "Timestep", 0.001f, "Integration timestep",
             ParamDescriptor::UIHint::Drag, true, 0.0001f, 0.1f, "%.4f");
    addParam(*this, "damping", "Damping", 0.0f, "Velocity damping rate (0 = symplectic)",
             ParamDescriptor::UIHint::Drag, true, 0.0f, 1.0f, "%.4f");
    addParam(*this, "boxSize", "Box Size", 100.0f, "Simulation box size",
             ParamDescriptor::UIHint::Drag, false, 1.0f, 10000.0f, "%.1f");
    addBoolParam(*this, "openBoundary", "Open Boundary", false,
                 "Disable periodic wrapping");
}

void KickNode::execute(ResourcePool& pool, cudaStream_t stream) {
    auto* particles = static_cast<cuda::ParticleGpu*>(pool.particleBuffer());

    float dt = getFloat("dt", pool);
    float damping = getFloat("damping", pool);
    float boxSize = getFloat("boxSize", pool);
    bool openBound = getBool("openBoundary", pool);

    float dampFactor;
    float driftDt;
    if (pool.comoving) {
        float OmegaL = std::max(0.0f, 1.0f - pool.params().OmegaM);
        float a = pool.scaleFactor;
        float H2 = pool.params().H0 * pool.params().H0 * (pool.params().OmegaM / (a * a * a) + OmegaL);
        float H_current = std::sqrt(std::max(H2, 0.0f));
        float hubbleDrag = std::max(0.0f, 1.0f - H_current * dt);
        float pressureDamp = std::exp(-damping * dt);
        dampFactor = hubbleDrag * pressureDamp;
        driftDt = dt / a;
    } else {
        dampFactor = std::exp(-damping * dt);
        driftDt = dt;
    }

    float maxVel = openBound ? 1e18f
                 : boxSize * 0.05f / std::max(driftDt, 1e-6f);

    cuda::launchPMKick(particles, pool.particleCount(), dt,
                       dampFactor, false, 0.0f, maxVel, stream);
}

// ─── SinkAccretionNode ──────────────────────────────────────────────

SinkAccretionNode::SinkAccretionNode() {
    name = "Sink Accretion";
    typeName = "SinkAccretion";
    inputs.push_back(makePin("particles", PinType::Particles, false));
    outputs.push_back(makePin("particles", PinType::Particles, true));
    declareParams();
}

void SinkAccretionNode::declareParams() {
    params.clear();
    addBoolParam(*this, "enableSink", "Enable Sink", false,
                 "Enable sink particle accretion", true);
    addParam(*this, "sinkRadius", "Sink Radius", 2.0f, "Accretion radius (grid cells)",
             ParamDescriptor::UIHint::Drag, false, 0.1f, 20.0f, "%.1f");
}

void SinkAccretionNode::execute(ResourcePool& pool, cudaStream_t stream) {
    bool enable = getBool("enableSink", pool);
    if (!enable) return;

    auto* particles = static_cast<cuda::ParticleGpu*>(pool.particleBuffer());
    float sinkRadius = getFloat("sinkRadius", pool);
    float boxSize = pool.params().boxSize;
    float cellSize = pool.cellSize();

    cuda::launchPMSinkAccretion(particles, pool.particleCount(),
                                sinkRadius * cellSize, boxSize,
                                ResourcePool::MAX_SINKS,
                                pool.sinkList(), pool.sinkCount(),
                                pool.newParticleCount(), stream);
}

// ─── EnergyReduceNode ───────────────────────────────────────────────

EnergyReduceNode::EnergyReduceNode() {
    name = "Energy Reduce";
    typeName = "EnergyReduce";
    inputs.push_back(makePin("particles", PinType::Particles, false));
    outputs.push_back(makePin("stats", PinType::Stats, true));
    declareParams();
}

void EnergyReduceNode::declareParams() {
    params.clear();
    addParam(*this, "boxSize", "Box Size", 100.0f, "Simulation box size",
             ParamDescriptor::UIHint::Drag, false, 1.0f, 10000.0f, "%.1f");
}

void EnergyReduceNode::execute(ResourcePool& pool, cudaStream_t stream) {
    auto* particles = static_cast<const cuda::ParticleGpu*>(pool.particleBuffer());
    float boxSize = getFloat("boxSize", pool);
    cuda::launchPMEnergyReduction(particles, pool.densityGrid(),
                                  pool.particleCount(), pool.gridN(),
                                  boxSize,
                                  pool.energySums(), stream);
}

// ---- OutputNode ----

OutputNode::OutputNode() {
    name = "Output";
    typeName = "Output";
    inputs.push_back(makePin("particles", PinType::Particles, false));
}

void OutputNode::execute(ResourcePool& /*pool*/, cudaStream_t /*stream*/) {
    // No-op: particles are already in the shared buffer, ready for rendering.
}

} // namespace cosmico
