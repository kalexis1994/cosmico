#include <cosmico/nodes/NBodyNodes.h>
#include <cosmico/nodes/ResourcePool.h>
#include <kernels/nbody_direct.cuh>
#include <cuda_runtime.h>

namespace cosmico {

// Pin-id helper
static int s_nbodyNextPinId = 20000;

static Pin makeNBodyPin(const char* name, PinType type, bool isOutput) {
    Pin p;
    p.id = s_nbodyNextPinId++;
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

// ---- ClearAccelNode ----

ClearAccelNode::ClearAccelNode() {
    name = "Clear Accel";
    typeName = "ClearAccel";
    inputs.push_back(makeNBodyPin("particles", PinType::Particles, false));
    outputs.push_back(makeNBodyPin("particles", PinType::Particles, true));
}

void ClearAccelNode::execute(ResourcePool& pool, cudaStream_t stream) {
    auto* particles = static_cast<cuda::ParticleGpu*>(pool.particleBuffer());
    cuda::launchClearAccel(particles, pool.particleCount(), stream);
}

// ---- PairwiseGravityNode ----

PairwiseGravityNode::PairwiseGravityNode() {
    name = "Pairwise Gravity";
    typeName = "PairwiseGravity";
    inputs.push_back(makeNBodyPin("particles", PinType::Particles, false));
    outputs.push_back(makeNBodyPin("particles", PinType::Particles, true));
    declareParams();
}

void PairwiseGravityNode::declareParams() {
    params.clear();
    addParam(*this, "G", "Gravity (G)", 1.0f, "Gravitational constant",
             ParamDescriptor::UIHint::Drag, true, 0.0f, 100.0f, "%.3f");
    addParam(*this, "softening", "Softening", 0.05f, "Gravitational softening length",
             ParamDescriptor::UIHint::Drag, true, 0.001f, 10.0f, "%.4f");
}

void PairwiseGravityNode::execute(ResourcePool& pool, cudaStream_t stream) {
    auto* particles = static_cast<cuda::ParticleGpu*>(pool.particleBuffer());

    float G = getFloat("G", pool);
    float softening = getFloat("softening", pool);

    cuda::launchPairwiseGravity(particles, pool.particleCount(),
                                 G, softening, stream);
}

} // namespace cosmico
