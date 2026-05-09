#include <cosmico/nodes/NodeEditorUI.h>
#include <cosmico/nodes/CompoundNode.h>

#include <imgui.h>
#include <imnodes.h>
#include <vector>

namespace cosmico {

void NodeEditorUI::updateSelection() {
    int numSelected = ImNodes::NumSelectedNodes();
    if (numSelected == 1) {
        int selected = -1;
        ImNodes::GetSelectedNodes(&selected);
        m_selectedNodeId = selected;
    } else if (numSelected == 0) {
        m_selectedNodeId = -1;
    }
}

void NodeEditorUI::renderPropertyPanel(SimGraph& graph) {
    ImGui::SetNextWindowSize(ImVec2(300, 400), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(1270, 10), ImGuiCond_FirstUseEver);

    if (!ImGui::Begin("Node Properties###NodeProperties")) {
        ImGui::End();
        return;
    }

    SimGraph* activeGraph = currentGraph();
    if (!activeGraph) activeGraph = &graph;

    if (m_selectedNodeId < 0) {
        ImGui::TextDisabled("Select a node to edit its properties");
        ImGui::End();
        return;
    }

    SimNode* node = activeGraph->findNode(m_selectedNodeId);
    if (!node) {
        ImGui::TextDisabled("Node not found");
        m_selectedNodeId = -1;
        ImGui::End();
        return;
    }

    // Node header
    ImGui::Text("%s", node->name.c_str());
    ImGui::TextDisabled("Type: %s  |  ID: %d", node->typeName.c_str(), node->id);

    // Compound node indicator
    auto* compound = dynamic_cast<CompoundNode*>(node);
    if (compound) {
        ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "Compound node (double-click to enter)");
    }

    ImGui::Separator();

    if (node->params.empty()) {
        ImGui::TextDisabled("No configurable parameters");
        ImGui::End();
        return;
    }

    // Render all params with full controls
    for (auto& param : node->params) {
        renderParamWidget(param);
    }

    ImGui::Separator();
    if (ImGui::Button("Reset All")) {
        for (auto& param : node->params) {
            param.value = param.descriptor.defaultValue;
            param.overridden = false;
        }
    }

    ImGui::End();
}

void NodeEditorUI::handleNewLinks(SimGraph& graph) {
    int startPinId, endPinId;
    if (ImNodes::IsLinkCreated(&startPinId, &endPinId)) {
        Pin* startPin = nullptr;
        Pin* endPin = nullptr;
        for (auto& n : graph.nodes) {
            if (!startPin) startPin = n->findPin(startPinId);
            if (!endPin) endPin = n->findPin(endPinId);
        }

        if (startPin && endPin) {
            if (startPin->isOutput && !endPin->isOutput) {
                graph.addLink(startPinId, endPinId);
            } else if (!startPin->isOutput && endPin->isOutput) {
                graph.addLink(endPinId, startPinId);
            }
        }
    }
}

void NodeEditorUI::handleDeletedLinks(SimGraph& graph) {
    int linkId;
    if (ImNodes::IsLinkDestroyed(&linkId)) {
        graph.removeLink(linkId);
    }

    int numSelected = ImNodes::NumSelectedNodes();
    if (numSelected > 0 && ImGui::IsKeyPressed(ImGuiKey_Delete)) {
        std::vector<int> selectedNodes(numSelected);
        ImNodes::GetSelectedNodes(selectedNodes.data());
        for (int nodeId : selectedNodes) {
            if (nodeId == m_selectedNodeId) m_selectedNodeId = -1;
            graph.removeNode(nodeId);
        }
        ImNodes::ClearNodeSelection();
    }

    int numSelectedLinks = ImNodes::NumSelectedLinks();
    if (numSelectedLinks > 0 && ImGui::IsKeyPressed(ImGuiKey_Delete)) {
        std::vector<int> selectedLinks(numSelectedLinks);
        ImNodes::GetSelectedLinks(selectedLinks.data());
        for (int lid : selectedLinks) {
            graph.removeLink(lid);
        }
        ImNodes::ClearLinkSelection();
    }
}

} // namespace cosmico
