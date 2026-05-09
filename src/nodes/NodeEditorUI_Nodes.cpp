#include <cosmico/nodes/NodeEditorUI.h>
#include <cosmico/nodes/CompoundNode.h>
#include <cosmico/nodes/PinTypes.h>

#include <imgui.h>
#include <imnodes.h>

namespace cosmico {

void NodeEditorUI::renderNodes(SimGraph& graph) {
    for (auto& node : graph.nodes) {
        // Highlight selected node
        bool isSelected = (node->id == m_selectedNodeId);
        bool isCompound = (dynamic_cast<CompoundNode*>(node.get()) != nullptr);

        if (isSelected) {
            ImNodes::PushColorStyle(ImNodesCol_NodeOutline, IM_COL32(255, 200, 50, 255));
        } else if (isCompound) {
            ImNodes::PushColorStyle(ImNodesCol_NodeOutline, IM_COL32(100, 180, 255, 200));
        }

        ImNodes::BeginNode(node->id);

        // Title bar
        ImNodes::BeginNodeTitleBar();
        if (isCompound) {
            ImGui::TextUnformatted("[+] ");
            ImGui::SameLine();
        }
        ImGui::TextUnformatted(node->name.c_str());
        ImNodes::EndNodeTitleBar();

        // Input pins
        for (auto& pin : node->inputs) {
            ImNodes::PushColorStyle(ImNodesCol_Pin, pinTypeColor(pin.type));
            ImNodes::PushColorStyle(ImNodesCol_PinHovered, pinTypeColor(pin.type));
            ImNodes::BeginInputAttribute(pin.id);
            ImGui::TextUnformatted(pin.name.c_str());
            ImNodes::EndInputAttribute();
            ImNodes::PopColorStyle();
            ImNodes::PopColorStyle();
        }

        // Inline params (between inputs and outputs)
        for (auto& param : node->params) {
            if (param.descriptor.inlineDisplay) {
                renderInlineParam(param, node->id);
            }
        }

        // Output pins
        for (auto& pin : node->outputs) {
            ImNodes::PushColorStyle(ImNodesCol_Pin, pinTypeColor(pin.type));
            ImNodes::PushColorStyle(ImNodesCol_PinHovered, pinTypeColor(pin.type));
            ImNodes::BeginOutputAttribute(pin.id);
            float textWidth = ImGui::CalcTextSize(pin.name.c_str()).x;
            float z = m_editorStack.empty() ? 1.0f : m_editorStack.back().zoomLevel;
            ImGui::Indent(120.0f * z - textWidth);
            ImGui::TextUnformatted(pin.name.c_str());
            ImNodes::EndOutputAttribute();
            ImNodes::PopColorStyle();
            ImNodes::PopColorStyle();
        }

        ImNodes::EndNode();

        if (isSelected || isCompound) {
            ImNodes::PopColorStyle();
        }
    }
}

void NodeEditorUI::renderLinks(SimGraph& graph) {
    for (const auto& link : graph.links) {
        ImU32 color = IM_COL32(200, 200, 200, 255);
        for (auto& node : graph.nodes) {
            Pin* pin = node->findPin(link.startPinId);
            if (pin) {
                color = pinTypeColor(pin->type);
                break;
            }
        }
        ImNodes::PushColorStyle(ImNodesCol_Link, color);
        ImNodes::PushColorStyle(ImNodesCol_LinkHovered, color);
        ImNodes::PushColorStyle(ImNodesCol_LinkSelected, color);
        ImNodes::Link(link.id, link.startPinId, link.endPinId);
        ImNodes::PopColorStyle();
        ImNodes::PopColorStyle();
        ImNodes::PopColorStyle();
    }
}

} // namespace cosmico
