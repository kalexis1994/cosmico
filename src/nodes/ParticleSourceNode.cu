#include <cosmico/nodes/ParticleSourceNode.h>
#include <cosmico/nodes/ResourcePool.h>
#include <cosmico/nodes/ICNodes.h>
#include <cuda_runtime.h>
#include <cstring>

namespace cosmico {

static int s_psSrcPinId = 40000;

static Pin makePSPin(const char* name, PinType type, bool isOutput) {
    Pin p;
    p.id = s_psSrcPinId++;
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

static void addIntParam(SimNode& node, const std::string& key, const std::string& label,
                        int defaultVal, const std::string& tooltip = "",
                        bool inlineDisp = false,
                        const std::vector<std::string>& enumLabels = {},
                        std::optional<float> minV = std::nullopt,
                        std::optional<float> maxV = std::nullopt) {
    NodeParam np;
    np.descriptor.key = key;
    np.descriptor.label = label;
    np.descriptor.tooltip = tooltip;
    np.descriptor.defaultValue = defaultVal;
    np.descriptor.enumLabels = enumLabels;
    np.descriptor.minVal = minV;
    np.descriptor.maxVal = maxV;
    np.descriptor.uiHint = enumLabels.empty() ? ParamDescriptor::UIHint::Drag : ParamDescriptor::UIHint::Dropdown;
    np.descriptor.inlineDisplay = inlineDisp;
    np.value = defaultVal;
    np.overridden = false;
    node.params.push_back(std::move(np));
}

// ─── ParticleSourceCompound ────────────────────────────────────────────

ParticleSourceCompound::ParticleSourceCompound() {
    name = "Particle Source";
    typeName = "ParticleSource";
    outputs.push_back(makePSPin("particles", PinType::Particles, true));
    declareParams();
    buildInnerGraph();
}

void ParticleSourceCompound::declareParams() {
    params.clear();

    // IC type dropdown (inline)
    addIntParam(*this, "icType", "IC Type", 0,
                "Initial condition distribution", true,
                {"Sphere", "Disk", "TwoBody", "GalaxyCollision", "Cosmological"});

    // Common params (inline)
    addIntParam(*this, "particleCount", "Particles", 32768,
                "Number of particles to generate", true,
                {}, 64.0f, 4194304.0f);
    addParam(*this, "massPerParticle", "Mass/Particle", 1.0f,
             "Default mass per particle",
             ParamDescriptor::UIHint::Drag, false, 0.001f, 10000.0f, "%.3f");
    addIntParam(*this, "seed", "Seed", 42,
                "Random seed for reproducibility", false,
                {}, 0.0f, 999999.0f);

    // Sphere params
    addParam(*this, "radius", "Radius", 15.0f,
             "Sphere radius",
             ParamDescriptor::UIHint::Drag, false, 0.1f, 1000.0f, "%.1f");
    addParam(*this, "velocitySpread", "Velocity Spread", 0.5f,
             "Random velocity amplitude",
             ParamDescriptor::UIHint::Drag, false, 0.0f, 100.0f, "%.2f");

    // Disk params
    addParam(*this, "diskRadius", "Disk Radius", 20.0f,
             "Disk outer radius",
             ParamDescriptor::UIHint::Drag, false, 1.0f, 1000.0f, "%.1f");
    addParam(*this, "centralMass", "Central Mass", 32768.0f,
             "Mass of central particle",
             ParamDescriptor::UIHint::Drag, false, 1.0f, 1e7f, "%.0f");
    addParam(*this, "heightScale", "Height Scale", 0.3f,
             "Vertical scatter of disk",
             ParamDescriptor::UIHint::Drag, false, 0.0f, 10.0f, "%.2f");

    // TwoBody params
    addParam(*this, "separation", "Separation", 20.0f,
             "Distance between two bodies",
             ParamDescriptor::UIHint::Drag, false, 1.0f, 1000.0f, "%.1f");
    addParam(*this, "bodyMass", "Body Mass", 500.0f,
             "Mass of each central body",
             ParamDescriptor::UIHint::Drag, false, 1.0f, 1e6f, "%.0f");

    // GalaxyCollision params
    addParam(*this, "approachSpeed", "Approach Speed", 2.0f,
             "Initial approach velocity",
             ParamDescriptor::UIHint::Drag, false, 0.0f, 100.0f, "%.1f");

    // Cosmological params
    addParam(*this, "jitterFraction", "Jitter Fraction", 0.02f,
             "Grid jitter as fraction of spacing",
             ParamDescriptor::UIHint::Drag, false, 0.0f, 0.5f, "%.3f");
    addParam(*this, "boxSize", "Box Size", 100.0f,
             "Cosmological box size",
             ParamDescriptor::UIHint::Drag, false, 1.0f, 10000.0f, "%.1f");

    // Advanced: density profile
    addIntParam(*this, "densityProfile", "Density Profile", 0,
                "Spatial density distribution", false,
                {"Uniform", "Plummer", "Hernquist", "NFW"});
    addParam(*this, "profileScale", "Profile Scale", 5.0f,
             "Scale length for density profile",
             ParamDescriptor::UIHint::Drag, false, 0.01f, 100.0f, "%.2f");

    // Advanced: velocity distribution
    addIntParam(*this, "velocityDist", "Velocity Distribution", 0,
                "Initial velocity prescription", false,
                {"Zero", "Uniform", "Maxwell-Boltzmann"});
    addParam(*this, "temperature", "Temperature", 1.0f,
             "Thermal velocity scale",
             ParamDescriptor::UIHint::Drag, false, 0.0f, 1000.0f, "%.2f");

    // Advanced: mass function
    addIntParam(*this, "massFunction", "Mass Function", 0,
                "Particle mass distribution", false,
                {"Equal", "Power Law"});
    addParam(*this, "powerLawIndex", "PL Index", -2.35f,
             "Power law index (Salpeter = -2.35)",
             ParamDescriptor::UIHint::Drag, false, -5.0f, 0.0f, "%.2f");

    // Advanced: noise
    addIntParam(*this, "noiseType", "Noise Type", 0,
                "Position perturbation type", false,
                {"None", "Perlin", "White"});
    addParam(*this, "noiseAmplitude", "Noise Amplitude", 0.1f,
             "Perturbation amplitude",
             ParamDescriptor::UIHint::Drag, false, 0.0f, 10.0f, "%.3f");
}

void ParticleSourceCompound::buildInnerGraph() {
    auto& g = m_innerGraph;
    g.clear();

    g.addNode("ICDistribution");   // [0]
    g.addNode("ICVelocity");       // [1]
    g.addNode("ICMassAssign");     // [2]
    g.addNode("ICNoise");          // [3]
    g.addNode("ICNormalize");      // [4]

    // Chain: Distribution -> Velocity -> MassAssign -> Noise -> Normalize
    auto findOut = [&](size_t idx, const char* n) -> int {
        for (auto& p : g.nodes[idx]->outputs)
            if (p.name == n) return p.id;
        return -1;
    };
    auto findIn = [&](size_t idx, const char* n) -> int {
        for (auto& p : g.nodes[idx]->inputs)
            if (p.name == n) return p.id;
        return -1;
    };

    g.addLink(findOut(0, "particles"), findIn(1, "particles"));
    g.addLink(findOut(1, "particles"), findIn(2, "particles"));
    g.addLink(findOut(2, "particles"), findIn(3, "particles"));
    g.addLink(findOut(3, "particles"), findIn(4, "particles"));
}

cuda::ICGenParams ParticleSourceCompound::buildICGenParams(const ResourcePool& pool) const {
    cuda::ICGenParams gp{};

    gp.icType          = getInt("icType", pool);
    gp.particleCount   = getInt("particleCount", pool);
    gp.massPerParticle = getFloat("massPerParticle");
    gp.seed            = static_cast<unsigned int>(getInt("seed"));

    // Sphere
    gp.radius          = getFloat("radius");
    gp.velocitySpread  = getFloat("velocitySpread");

    // Disk
    gp.diskRadius      = getFloat("diskRadius");
    gp.centralMass     = getFloat("centralMass");
    gp.heightScale     = getFloat("heightScale");

    // TwoBody
    gp.separation      = getFloat("separation");
    gp.bodyMass        = getFloat("bodyMass");

    // GalaxyCollision
    gp.approachSpeed   = getFloat("approachSpeed");

    // Cosmological
    gp.jitterFraction  = getFloat("jitterFraction");
    gp.boxSize         = getFloat("boxSize", pool);

    // Advanced
    gp.densityProfile  = getInt("densityProfile");
    gp.profileScale    = getFloat("profileScale");
    gp.velocityDist    = getInt("velocityDist");
    gp.temperature     = getFloat("temperature");
    gp.massFunction    = getInt("massFunction");
    gp.powerLawIndex   = getFloat("powerLawIndex");
    gp.noiseType       = getInt("noiseType");
    gp.noiseAmplitude  = getFloat("noiseAmplitude");

    return gp;
}

bool ParticleSourceCompound::paramsChanged(const cuda::ICGenParams& a,
                                            const cuda::ICGenParams& b) const {
    return memcmp(&a, &b, sizeof(cuda::ICGenParams)) != 0;
}

void ParticleSourceCompound::execute(ResourcePool& pool, cudaStream_t stream) {
    auto gp = buildICGenParams(pool);

    // Only regenerate on first frame or when params change
    if (!m_generated || paramsChanged(gp, m_lastGenParams)) {
        auto* particles = static_cast<cuda::ParticleGpu*>(pool.particleBuffer());

        // Update particle count in pool if it changed
        pool.setParticleCount(gp.particleCount);

        cuda::launchICGenerate(particles, gp, stream);

        m_lastGenParams = gp;
        m_generated = true;
    }
}

} // namespace cosmico
