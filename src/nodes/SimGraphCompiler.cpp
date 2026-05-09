#include <cosmico/nodes/SimGraphCompiler.h>
#include <cosmico/nodes/CompoundNode.h>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <algorithm>

namespace cosmico {

CompiledPlan SimGraphCompiler::compile(SimGraph& graph) {
    CompiledPlan plan;

    std::string error;
    if (!validate(graph, error)) {
        plan.valid = false;
        plan.error = error;
        return plan;
    }

    plan.executionOrder = topologicalSort(graph);
    if (plan.executionOrder.empty() && !graph.nodes.empty()) {
        plan.valid = false;
        plan.error = "Cycle detected in graph";
        return plan;
    }

    plan.valid = true;
    return plan;
}

bool SimGraphCompiler::validate(SimGraph& graph, std::string& error) {
    if (graph.nodes.empty()) {
        error = "Graph is empty";
        return false;
    }

    // Check for at least one ParticleSource
    bool hasSource = false;
    for (const auto& node : graph.nodes) {
        if (node->typeName == "ParticleSource") {
            hasSource = true;
            break;
        }
    }
    if (!hasSource) {
        error = "Graph must contain at least one ParticleSource node";
        return false;
    }

    // Check that all required inputs are connected.
    // Required = all inputs on a node (optional nodes like SinkAccretion
    // handle missing features internally via params).
    for (const auto& node : graph.nodes) {
        for (const auto& pin : node->inputs) {
            bool connected = false;
            for (const auto& link : graph.links) {
                if (link.endPinId == pin.id) {
                    connected = true;
                    break;
                }
            }
            if (!connected) {
                error = "Node '" + node->name + "' has unconnected input '" + pin.name + "'";
                return false;
            }
        }
    }

    // Check pin type compatibility on all links
    for (const auto& link : graph.links) {
        Pin* startPin = nullptr;
        Pin* endPin = nullptr;
        for (auto& n : graph.nodes) {
            if (!startPin) startPin = n->findPin(link.startPinId);
            if (!endPin) endPin = n->findPin(link.endPinId);
        }
        if (!startPin || !endPin) {
            error = "Link references non-existent pin";
            return false;
        }
        if (startPin->type != endPin->type) {
            error = "Type mismatch: " + std::string(pinTypeName(startPin->type)) +
                    " -> " + std::string(pinTypeName(endPin->type));
            return false;
        }
    }

    // Validate compound nodes' inner graphs
    for (const auto& node : graph.nodes) {
        auto* compound = dynamic_cast<CompoundNode*>(node.get());
        if (compound) {
            if (!compound->compile()) {
                error = "Compound node '" + node->name + "' has invalid inner graph";
                return false;
            }
        }
    }

    // Check for cycles
    if (hasCycle(graph)) {
        error = "Cycle detected in graph (must be a DAG)";
        return false;
    }

    return true;
}

bool SimGraphCompiler::hasCycle(SimGraph& graph) {
    // Build adjacency: node -> set of downstream node IDs
    std::unordered_map<int, std::vector<int>> adj;
    std::unordered_map<int, int> inDegree;

    for (const auto& node : graph.nodes) {
        inDegree[node->id] = 0;
    }

    for (const auto& link : graph.links) {
        SimNode* srcNode = graph.findNodeByPin(link.startPinId);
        SimNode* dstNode = graph.findNodeByPin(link.endPinId);
        if (srcNode && dstNode && srcNode != dstNode) {
            adj[srcNode->id].push_back(dstNode->id);
            inDegree[dstNode->id]++;
        }
    }

    // Kahn's algorithm
    std::queue<int> q;
    for (auto& [id, deg] : inDegree) {
        if (deg == 0) q.push(id);
    }

    int visited = 0;
    while (!q.empty()) {
        int cur = q.front();
        q.pop();
        visited++;
        for (int next : adj[cur]) {
            if (--inDegree[next] == 0) q.push(next);
        }
    }

    return visited != static_cast<int>(graph.nodes.size());
}

std::vector<SimNode*> SimGraphCompiler::topologicalSort(SimGraph& graph) {
    // Build adjacency
    std::unordered_map<int, std::vector<int>> adj;
    std::unordered_map<int, int> inDegree;

    for (const auto& node : graph.nodes) {
        inDegree[node->id] = 0;
    }

    for (const auto& link : graph.links) {
        SimNode* srcNode = graph.findNodeByPin(link.startPinId);
        SimNode* dstNode = graph.findNodeByPin(link.endPinId);
        if (srcNode && dstNode && srcNode != dstNode) {
            adj[srcNode->id].push_back(dstNode->id);
            inDegree[dstNode->id]++;
        }
    }

    // Kahn's algorithm — stable sort by node ID for deterministic order
    std::priority_queue<int, std::vector<int>, std::greater<int>> q;
    for (auto& [id, deg] : inDegree) {
        if (deg == 0) q.push(id);
    }

    std::vector<SimNode*> order;
    order.reserve(graph.nodes.size());

    while (!q.empty()) {
        int cur = q.top();
        q.pop();
        SimNode* node = graph.findNode(cur);
        if (node) order.push_back(node);

        for (int next : adj[cur]) {
            if (--inDegree[next] == 0) q.push(next);
        }
    }

    if (order.size() != graph.nodes.size()) {
        return {}; // Cycle detected
    }
    return order;
}

} // namespace cosmico
