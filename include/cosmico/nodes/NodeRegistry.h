#pragma once
#include <cosmico/nodes/SimNode.h>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace cosmico {

class NodeRegistry {
public:
    static NodeRegistry& instance() {
        static NodeRegistry reg;
        return reg;
    }

    void registerNode(const std::string& typeName,
                      std::function<std::unique_ptr<SimNode>()> factory) {
        m_factories[typeName] = std::move(factory);
        if (std::find(m_typeNames.begin(), m_typeNames.end(), typeName) == m_typeNames.end()) {
            m_typeNames.push_back(typeName);
        }
    }

    std::unique_ptr<SimNode> create(const std::string& typeName) const {
        auto it = m_factories.find(typeName);
        if (it == m_factories.end()) return nullptr;
        return it->second();
    }

    const std::vector<std::string>& typeNames() const { return m_typeNames; }

private:
    NodeRegistry() = default;
    std::unordered_map<std::string, std::function<std::unique_ptr<SimNode>()>> m_factories;
    std::vector<std::string> m_typeNames;
};

// Auto-registration helper
template<typename T>
struct NodeRegistrar {
    explicit NodeRegistrar(const char* name) {
        NodeRegistry::instance().registerNode(name, []() {
            return std::make_unique<T>();
        });
    }
};

#define REGISTER_SIM_NODE(TypeName, Class) \
    static cosmico::NodeRegistrar<Class> s_reg_##Class(TypeName)

} // namespace cosmico
