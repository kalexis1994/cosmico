#pragma once
#include <cosmico/nodes/SimNode.h>
#include <cosmico/nodes/NodeRegistry.h>
#include <nlohmann/json_fwd.hpp>
#include <memory>
#include <vector>
#include <string>

namespace cosmico {

class SimGraph {
public:
    struct Link {
        int id;
        int startPinId;  // Output pin
        int endPinId;    // Input pin
    };

    std::vector<std::unique_ptr<SimNode>> nodes;
    std::vector<Link> links;

    // Add a node by type name, returns node ID
    int addNode(const std::string& typeName) {
        auto node = NodeRegistry::instance().create(typeName);
        if (!node) return -1;
        int nodeId = m_nextNodeId++;
        node->id = nodeId;
        // Assign unique pin IDs
        for (auto& pin : node->inputs) {
            pin.id = m_nextPinId++;
        }
        for (auto& pin : node->outputs) {
            pin.id = m_nextPinId++;
        }
        nodes.push_back(std::move(node));
        return nodeId;
    }

    void removeNode(int nodeId) {
        // Remove all links connected to this node's pins
        auto* node = findNode(nodeId);
        if (!node) return;
        for (auto& pin : node->inputs) removeLinksForPin(pin.id);
        for (auto& pin : node->outputs) removeLinksForPin(pin.id);
        nodes.erase(
            std::remove_if(nodes.begin(), nodes.end(),
                [nodeId](const auto& n) { return n->id == nodeId; }),
            nodes.end());
    }

    int addLink(int startPinId, int endPinId) {
        // Find the pins and validate type compatibility
        Pin* startPin = nullptr;
        Pin* endPin = nullptr;
        for (auto& n : nodes) {
            if (!startPin) startPin = n->findPin(startPinId);
            if (!endPin) endPin = n->findPin(endPinId);
        }
        if (!startPin || !endPin) return -1;
        if (startPin->type != endPin->type) return -1;
        if (!startPin->isOutput || endPin->isOutput) return -1;

        // Remove existing link to this input (only one connection per input)
        removeLinksForPin(endPinId);

        int linkId = m_nextLinkId++;
        links.push_back({linkId, startPinId, endPinId});
        endPin->linkedPinId = startPinId;
        return linkId;
    }

    void removeLink(int linkId) {
        auto it = std::find_if(links.begin(), links.end(),
            [linkId](const Link& l) { return l.id == linkId; });
        if (it != links.end()) {
            // Clear the input pin's linked reference
            for (auto& n : nodes) {
                Pin* pin = n->findPin(it->endPinId);
                if (pin) pin->linkedPinId = -1;
            }
            links.erase(it);
        }
    }

    SimNode* findNode(int nodeId) {
        for (auto& n : nodes) if (n->id == nodeId) return n.get();
        return nullptr;
    }

    const SimNode* findNode(int nodeId) const {
        for (auto& n : nodes) if (n->id == nodeId) return n.get();
        return nullptr;
    }

    // Find which node owns a given pin
    SimNode* findNodeByPin(int pinId) {
        for (auto& n : nodes) {
            if (n->findPin(pinId)) return n.get();
        }
        return nullptr;
    }

    void clear() {
        nodes.clear();
        links.clear();
    }

    // Serialization
    void saveToJson(const std::string& path) const;
    void loadFromJson(const std::string& path);

private:
    void removeLinksForPin(int pinId) {
        for (auto it = links.begin(); it != links.end(); ) {
            if (it->startPinId == pinId || it->endPinId == pinId) {
                // Clear linked reference
                for (auto& n : nodes) {
                    Pin* ep = n->findPin(it->endPinId);
                    if (ep) ep->linkedPinId = -1;
                }
                it = links.erase(it);
            } else {
                ++it;
            }
        }
    }

    int m_nextNodeId = 1;
    int m_nextPinId = 10000;
    int m_nextLinkId = 20000;
};

} // namespace cosmico
