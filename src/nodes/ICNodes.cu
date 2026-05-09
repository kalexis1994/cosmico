#include <cosmico/nodes/ICNodes.h>
#include <cosmico/nodes/ResourcePool.h>
#include <cuda_runtime.h>

namespace cosmico {

// These IC sub-nodes are conceptual pipeline stages inside the ParticleSource
// compound node. The actual GPU work is done by the compound's execute() which
// calls launchICGenerate. These nodes exist so the inner graph is visible and
// navigable, but individually they are no-ops (the compound handles everything).

static int s_icNextPinId = 30000;

static Pin makeICPin(const char* name, PinType type, bool isOutput) {
    Pin p;
    p.id = s_icNextPinId++;
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

// ─── ICDistributionNode ─────────────────────────────────────────────

ICDistributionNode::ICDistributionNode() {
    name = "IC Distribution";
    typeName = "ICDistribution";
    outputs.push_back(makeICPin("particles", PinType::Particles, true));
    declareParams();
}

void ICDistributionNode::declareParams() {
    params.clear();
    addIntParam(*this, "densityProfile", "Density Profile", 0,
                "Spatial distribution of particles", true,
                {"Uniform", "Plummer", "Hernquist", "NFW"});
    addParam(*this, "profileScale", "Scale Length", 5.0f,
             "Scale radius for density profile",
             ParamDescriptor::UIHint::Drag, false, 0.1f, 100.0f, "%.1f");
}

void ICDistributionNode::execute(ResourcePool& /*pool*/, cudaStream_t /*stream*/) {
    // No-op: compound parent handles actual generation
}

// ─── ICVelocityNode ─────────────────────────────────────────────────

ICVelocityNode::ICVelocityNode() {
    name = "IC Velocity";
    typeName = "ICVelocity";
    inputs.push_back(makeICPin("particles", PinType::Particles, false));
    outputs.push_back(makeICPin("particles", PinType::Particles, true));
    declareParams();
}

void ICVelocityNode::declareParams() {
    params.clear();
    addIntParam(*this, "velocityDist", "Velocity Distribution", 0,
                "How initial velocities are assigned", true,
                {"Zero", "Uniform", "Maxwell-Boltzmann"});
    addParam(*this, "temperature", "Temperature", 1.0f,
             "Thermal velocity scale (MB distribution)",
             ParamDescriptor::UIHint::Drag, false, 0.0f, 100.0f, "%.2f");
}

void ICVelocityNode::execute(ResourcePool& /*pool*/, cudaStream_t /*stream*/) {
    // No-op: compound parent handles actual generation
}

// ─── ICMassAssignNode ───────────────────────────────────────────────

ICMassAssignNode::ICMassAssignNode() {
    name = "IC Mass Assign";
    typeName = "ICMassAssign";
    inputs.push_back(makeICPin("particles", PinType::Particles, false));
    outputs.push_back(makeICPin("particles", PinType::Particles, true));
    declareParams();
}

void ICMassAssignNode::declareParams() {
    params.clear();
    addIntParam(*this, "massFunction", "Mass Function", 0,
                "How particle masses are assigned", true,
                {"Equal", "Power Law"});
    addParam(*this, "powerLawIndex", "Power Law Index", -2.35f,
             "Salpeter-like index for mass function",
             ParamDescriptor::UIHint::Drag, false, -5.0f, 0.0f, "%.2f");
}

void ICMassAssignNode::execute(ResourcePool& /*pool*/, cudaStream_t /*stream*/) {
    // No-op: compound parent handles actual generation
}

// ─── ICNoiseNode ────────────────────────────────────────────────────

ICNoiseNode::ICNoiseNode() {
    name = "IC Noise";
    typeName = "ICNoise";
    inputs.push_back(makeICPin("particles", PinType::Particles, false));
    outputs.push_back(makeICPin("particles", PinType::Particles, true));
    declareParams();
}

void ICNoiseNode::declareParams() {
    params.clear();
    addIntParam(*this, "noiseType", "Noise Type", 0,
                "Position perturbation type", true,
                {"None", "Perlin", "White"});
    addParam(*this, "noiseAmplitude", "Amplitude", 0.1f,
             "Noise displacement amplitude",
             ParamDescriptor::UIHint::Drag, false, 0.0f, 10.0f, "%.3f");
}

void ICNoiseNode::execute(ResourcePool& /*pool*/, cudaStream_t /*stream*/) {
    // No-op: compound parent handles actual generation
}

// ─── ICNormalizeNode ────────────────────────────────────────────────

ICNormalizeNode::ICNormalizeNode() {
    name = "IC Normalize";
    typeName = "ICNormalize";
    inputs.push_back(makeICPin("particles", PinType::Particles, false));
    outputs.push_back(makeICPin("particles", PinType::Particles, true));
}

void ICNormalizeNode::execute(ResourcePool& /*pool*/, cudaStream_t /*stream*/) {
    // No-op: normalization is handled by launchICGenerate
}

} // namespace cosmico
