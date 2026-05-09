#include <cosmico/nodes/SimNode.h>
#include <cosmico/nodes/ResourcePool.h>

namespace cosmico {

// ── Fallback: map param keys to PMParams fields ──────────────────────────

static float pmParamFloat(const std::string& key, const PMParams& p) {
    if (key == "G")          return p.G;
    if (key == "dt")         return p.dt;
    if (key == "damping")    return p.damping;
    if (key == "softening")  return p.softening;
    if (key == "boxSize")    return p.boxSize;
    if (key == "sinkRadius") return p.sinkRadius;
    if (key == "sinkDensityThreshold") return p.sinkDensityThreshold;
    if (key == "H0")         return p.H0;
    if (key == "OmegaM")     return p.OmegaM;
    if (key == "aInit")      return p.aInit;
    return 0.0f;
}

static int pmParamInt(const std::string& key, const PMParams& p) {
    if (key == "gridN")         return p.gridN;
    if (key == "stepsPerFrame") return p.stepsPerFrame;
    if (key == "particleCount") return p.particleCount;
    return 0;
}

static bool pmParamBool(const std::string& key, const PMParams& p) {
    if (key == "openBoundary")    return p.openBoundary;
    if (key == "enableSink")      return p.enableSink;
    if (key == "comoving")        return p.comoving;
    if (key == "correctMomentum") return p.correctMomentum;
    if (key == "showPowerSpectrum") return p.showPowerSpectrum;
    return false;
}

// ── Typed getters with PMParams fallback ─────────────────────────────────

float SimNode::getFloat(const std::string& key, const ResourcePool& pool) const {
    const NodeParam* p = findParam(key);
    if (p && p->overridden) return std::get<float>(p->value);
    return pmParamFloat(key, pool.params());
}

int SimNode::getInt(const std::string& key, const ResourcePool& pool) const {
    const NodeParam* p = findParam(key);
    if (p && p->overridden) return std::get<int>(p->value);
    return pmParamInt(key, pool.params());
}

bool SimNode::getBool(const std::string& key, const ResourcePool& pool) const {
    const NodeParam* p = findParam(key);
    if (p && p->overridden) return std::get<bool>(p->value);
    return pmParamBool(key, pool.params());
}

// ── Direct getters (no fallback) ─────────────────────────────────────────

float SimNode::getFloat(const std::string& key) const {
    const NodeParam* p = findParam(key);
    if (p) return std::get<float>(p->overridden ? p->value : p->descriptor.defaultValue);
    return 0.0f;
}

int SimNode::getInt(const std::string& key) const {
    const NodeParam* p = findParam(key);
    if (p) return std::get<int>(p->overridden ? p->value : p->descriptor.defaultValue);
    return 0;
}

bool SimNode::getBool(const std::string& key) const {
    const NodeParam* p = findParam(key);
    if (p) return std::get<bool>(p->overridden ? p->value : p->descriptor.defaultValue);
    return false;
}

std::string SimNode::getString(const std::string& key) const {
    const NodeParam* p = findParam(key);
    if (p) return std::get<std::string>(p->overridden ? p->value : p->descriptor.defaultValue);
    return "";
}

// ── Param lookup / mutation ──────────────────────────────────────────────

NodeParam* SimNode::findParam(const std::string& key) {
    for (auto& p : params)
        if (p.descriptor.key == key) return &p;
    return nullptr;
}

const NodeParam* SimNode::findParam(const std::string& key) const {
    for (auto& p : params)
        if (p.descriptor.key == key) return &p;
    return nullptr;
}

void SimNode::setParam(const std::string& key, const ParamValue& value) {
    NodeParam* p = findParam(key);
    if (p) {
        p->value = value;
        p->overridden = true;
    }
}

void SimNode::resetParam(const std::string& key) {
    NodeParam* p = findParam(key);
    if (p) {
        p->value = p->descriptor.defaultValue;
        p->overridden = false;
    }
}

} // namespace cosmico
