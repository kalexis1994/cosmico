#include <cosmico/nodes/SimGraph.h>
#include <cosmico/nodes/CompoundNode.h>
#include <nlohmann/json.hpp>
#include <fstream>
#include <cstdio>

using json = nlohmann::json;

namespace cosmico {

void SimGraph::saveToJson(const std::string& path) const {
    json j;

    // Serialize nodes
    json jNodes = json::array();
    for (const auto& n : nodes) {
        json jNode;
        jNode["id"] = n->id;
        jNode["type"] = n->typeName;
        jNode["x"] = n->position.x;
        jNode["y"] = n->position.y;

        // Serialize overridden params only
        json jParams = json::object();
        for (const auto& p : n->params) {
            if (!p.overridden) continue;
            const auto& key = p.descriptor.key;
            std::visit([&](const auto& v) {
                jParams[key] = v;
            }, p.value);
        }
        if (!jParams.empty()) {
            jNode["params"] = jParams;
        }

        jNodes.push_back(std::move(jNode));
    }
    j["nodes"] = std::move(jNodes);

    // Serialize links
    json jLinks = json::array();
    for (const auto& l : links) {
        json jLink;
        jLink["id"] = l.id;
        jLink["start"] = l.startPinId;
        jLink["end"] = l.endPinId;
        jLinks.push_back(std::move(jLink));
    }
    j["links"] = std::move(jLinks);

    std::ofstream out(path);
    if (!out.is_open()) {
        fprintf(stderr, "[NodeGraph] Failed to save graph to %s\n", path.c_str());
        return;
    }
    out << j.dump(2) << "\n";

    fprintf(stderr, "[NodeGraph] Saved graph: %zu nodes, %zu links to %s\n",
            nodes.size(), links.size(), path.c_str());
}

void SimGraph::loadFromJson(const std::string& path) {
    std::ifstream in(path);
    if (!in.is_open()) {
        fprintf(stderr, "[NodeGraph] Failed to load graph from %s\n", path.c_str());
        return;
    }

    json j;
    try {
        in >> j;
    } catch (const json::parse_error& e) {
        fprintf(stderr, "[NodeGraph] JSON parse error: %s\n", e.what());
        return;
    }

    clear();

    int maxNodeId = 0;
    int maxPinId = 9999;

    // Load nodes
    if (j.contains("nodes") && j["nodes"].is_array()) {
        for (const auto& jNode : j["nodes"]) {
            std::string type = jNode.value("type", "");
            int id = jNode.value("id", 0);
            float x = jNode.value("x", 0.0f);
            float y = jNode.value("y", 0.0f);

            auto node = NodeRegistry::instance().create(type);
            if (!node) continue;
            node->id = id;
            node->position = ImVec2(x, y);

            // Assign pin IDs
            for (auto& pin : node->inputs)  pin.id = ++maxPinId;
            for (auto& pin : node->outputs) pin.id = ++maxPinId;

            // Restore overridden params
            if (jNode.contains("params") && jNode["params"].is_object()) {
                for (auto& [key, val] : jNode["params"].items()) {
                    NodeParam* p = node->findParam(key);
                    if (!p) continue;

                    // Match the type from the descriptor's default value
                    std::visit([&](const auto& defVal) {
                        using T = std::decay_t<decltype(defVal)>;
                        if constexpr (std::is_same_v<T, float>) {
                            p->value = val.get<float>();
                        } else if constexpr (std::is_same_v<T, int>) {
                            p->value = val.get<int>();
                        } else if constexpr (std::is_same_v<T, bool>) {
                            p->value = val.get<bool>();
                        } else if constexpr (std::is_same_v<T, std::string>) {
                            p->value = val.get<std::string>();
                        }
                    }, p->descriptor.defaultValue);
                    p->overridden = true;
                }
            }

            if (id >= maxNodeId) maxNodeId = id + 1;
            nodes.push_back(std::move(node));
        }
    }

    // Load links
    int maxLinkId = 19999;
    if (j.contains("links") && j["links"].is_array()) {
        for (const auto& jLink : j["links"]) {
            int linkId   = jLink.value("id", 0);
            int startPin = jLink.value("start", 0);
            int endPin   = jLink.value("end", 0);
            links.push_back({linkId, startPin, endPin});
            if (linkId >= maxLinkId) maxLinkId = linkId + 1;

            // Update linked pin reference
            for (auto& n : nodes) {
                Pin* ep = n->findPin(endPin);
                if (ep) ep->linkedPinId = startPin;
            }
        }
    }

    m_nextNodeId = maxNodeId;
    m_nextPinId = maxPinId + 1;
    m_nextLinkId = maxLinkId + 1;

    fprintf(stderr, "[NodeGraph] Loaded graph: %zu nodes, %zu links from %s\n",
            nodes.size(), links.size(), path.c_str());
}

} // namespace cosmico
