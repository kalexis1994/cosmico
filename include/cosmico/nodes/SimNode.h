#pragma once
#include <cosmico/nodes/PinTypes.h>
#include <cosmico/nodes/NodeParam.h>
#include <string>
#include <vector>
#include <imgui.h>

typedef struct CUstream_st* cudaStream_t;

namespace cosmico {

class ResourcePool;

struct Pin {
    int id = -1;
    std::string name;
    PinType type = PinType::Scalar;
    int linkedPinId = -1;  // Connected pin (-1 = unconnected)
    bool isOutput = false;
};

class SimNode {
public:
    int id = -1;
    std::string name;
    std::string typeName;  // Registry key (e.g. "CICDeposit")
    ImVec2 position{0, 0};
    std::vector<Pin> inputs;
    std::vector<Pin> outputs;
    std::vector<NodeParam> params;

    virtual ~SimNode() = default;
    virtual void execute(ResourcePool& pool, cudaStream_t stream) = 0;
    virtual void declareParams() {}

    // Typed getters with PMParams fallback (implemented in SimNode.cpp)
    float getFloat(const std::string& key, const ResourcePool& pool) const;
    int   getInt(const std::string& key, const ResourcePool& pool) const;
    bool  getBool(const std::string& key, const ResourcePool& pool) const;

    // Direct getters (no fallback — returns local value or descriptor default)
    float       getFloat(const std::string& key) const;
    int         getInt(const std::string& key) const;
    bool        getBool(const std::string& key) const;
    std::string getString(const std::string& key) const;

    // Param mutation
    NodeParam*       findParam(const std::string& key);
    const NodeParam* findParam(const std::string& key) const;
    void setParam(const std::string& key, const ParamValue& value);
    void resetParam(const std::string& key);

    Pin* findPin(int pinId) {
        for (auto& p : inputs) if (p.id == pinId) return &p;
        for (auto& p : outputs) if (p.id == pinId) return &p;
        return nullptr;
    }

    const Pin* findPin(int pinId) const {
        for (auto& p : inputs) if (p.id == pinId) return &p;
        for (auto& p : outputs) if (p.id == pinId) return &p;
        return nullptr;
    }
};

} // namespace cosmico
