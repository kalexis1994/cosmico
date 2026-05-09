#include <cosmico/nodes/NodeEditorUI.h>
#include <cosmico/nodes/NodeRegistry.h>
#include <cosmico/nodes/CompoundNode.h>
#include <imgui.h>
#include <imnodes.h>
#include <cstdio>
#include <algorithm>
#include <cmath>

namespace cosmico {

void NodeEditorUI::init() {
    ImNodes::CreateContext();
    // Root level created in first render() call when we have the graph pointer
}

void NodeEditorUI::destroy() {
    for (auto& level : m_editorStack) {
        if (level.context) {
            ImNodes::EditorContextFree(level.context);
        }
    }
    m_editorStack.clear();
    ImNodes::DestroyContext();
}

SimGraph* NodeEditorUI::currentGraph() {
    if (m_editorStack.empty()) return nullptr;
    return m_editorStack.back().graph;
}

void NodeEditorUI::pushGraph(SimGraph* graph, const std::string& name) {
    EditorLevel level;
    level.graph = graph;
    level.context = ImNodes::EditorContextCreate();
    level.name = name;
    level.firstRender = true;
    m_editorStack.push_back(level);
    m_selectedNodeId = -1;
}

void NodeEditorUI::popGraph() {
    if (m_editorStack.size() <= 1) return; // Can't pop root

    auto& top = m_editorStack.back();
    if (top.context) {
        ImNodes::EditorContextFree(top.context);
    }
    m_editorStack.pop_back();
    m_selectedNodeId = -1;
}

NodeEditorUI::Result NodeEditorUI::render(SimGraph& graph,
                                           const CompiledPlan* currentPlan,
                                           bool isRunning,
                                           bool cudaAvailable) {
    Result result;

    // Initialize root level if needed
    if (m_editorStack.empty()) {
        EditorLevel root;
        root.graph = &graph;
        root.context = ImNodes::EditorContextCreate();
        root.name = "Root";
        root.firstRender = true;
        m_editorStack.push_back(root);

        // Apply style
        ImNodes::EditorContextSet(root.context);
        ImNodes::StyleColorsDark();
        auto& style = ImNodes::GetStyle();
        style.NodePadding = ImVec2(8.0f, 4.0f);
        style.NodeBorderThickness = 1.5f;
        style.LinkThickness = 2.5f;
        style.PinCircleRadius = 4.0f;
        style.Flags |= ImNodesStyleFlags_GridLines;
    }

    // Always keep root pointing to the actual graph (it may change between calls)
    m_editorStack[0].graph = &graph;

    SimGraph* activeGraph = currentGraph();
    if (!activeGraph) return result;

    ImGui::SetNextWindowSize(ImVec2(900, 600), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(360, 10), ImGuiCond_FirstUseEver);

    if (!ImGui::Begin("Node Editor###NodeEditor")) {
        ImGui::End();
        return result;
    }

    auto& currentLevel = m_editorStack.back();
    ImNodes::EditorContextSet(currentLevel.context);

    // Toolbar
    {
        if (!cudaAvailable) ImGui::BeginDisabled();

        if (ImGui::Button("Build")) {
            result.buildRequested = true;
        }
        ImGui::SameLine();

        if (ImGui::Button("Reset")) {
            result.resetRequested = true;
        }

        if (!cudaAvailable) {
            ImGui::EndDisabled();
            ImGui::SameLine();
            ImGui::TextDisabled("(CUDA required)");
        }

        // Show compilation status
        ImGui::SameLine();
        ImGui::Spacing();
        ImGui::SameLine();
        if (currentPlan) {
            if (currentPlan->valid) {
                ImGui::TextColored(ImVec4(0.3f, 0.9f, 0.3f, 1.0f),
                    "Compiled: %d nodes", (int)currentPlan->executionOrder.size());
            } else {
                ImGui::TextColored(ImVec4(0.9f, 0.3f, 0.3f, 1.0f),
                    "Error: %s", currentPlan->error.c_str());
            }
        } else {
            ImGui::TextDisabled("Not compiled");
        }

        if (isRunning) {
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(0.3f, 0.8f, 1.0f, 1.0f), " [Running]");
        }
    }

    // Breadcrumb bar
    renderBreadcrumbs();

    float zoom = currentLevel.zoomLevel;

    // Zoom indicator
    if (zoom != 1.0f) {
        ImGui::Text("%.0f%%", zoom * 100.0f);
    }

    // Apply zoom: scale font and imnodes style
    auto& style = ImNodes::GetStyle();
    float origNodePadX = style.NodePadding.x;
    float origNodePadY = style.NodePadding.y;
    float origBorder = style.NodeBorderThickness;
    float origLink = style.LinkThickness;
    float origPin = style.PinCircleRadius;
    float origHoverDist = style.LinkHoverDistance;
    float origPinHoverRad = style.PinHoverRadius;
    float origPinOffset = style.PinOffset;

    style.NodePadding.x *= zoom;
    style.NodePadding.y *= zoom;
    style.NodeBorderThickness *= zoom;
    style.LinkThickness *= zoom;
    style.PinCircleRadius *= zoom;
    style.LinkHoverDistance *= zoom;
    style.PinHoverRadius *= zoom;
    style.PinOffset *= zoom;

    ImGui::SetWindowFontScale(zoom);

    // Node editor canvas
    ImNodes::BeginNodeEditor();

    // Right-click context menu
    renderMenuBar(*activeGraph);

    // Render nodes
    renderNodes(*activeGraph);

    // Render links
    renderLinks(*activeGraph);

    // Minimap
    ImNodes::MiniMap(0.15f, ImNodesMiniMapLocation_BottomRight);

    // Capture zoom input INSIDE the editor child window (IsEditorHovered checks ImGui::IsWindowHovered)
    bool editorHovered = ImNodes::IsEditorHovered();
    float pendingWheel = 0.0f;
    if (editorHovered) {
        pendingWheel = ImGui::GetIO().MouseWheel;
    }

    ImNodes::EndNodeEditor();

    // Restore zoom state
    ImGui::SetWindowFontScale(1.0f);

    style.NodePadding.x = origNodePadX;
    style.NodePadding.y = origNodePadY;
    style.NodeBorderThickness = origBorder;
    style.LinkThickness = origLink;
    style.PinCircleRadius = origPin;
    style.LinkHoverDistance = origHoverDist;
    style.PinHoverRadius = origPinHoverRad;
    style.PinOffset = origPinOffset;

    // Apply zoom with mouse wheel
    if (pendingWheel != 0.0f) {
        float oldZoom = zoom;
        zoom *= (pendingWheel > 0.0f) ? 1.1f : (1.0f / 1.1f);
        zoom = std::clamp(zoom, ZOOM_MIN, ZOOM_MAX);
        currentLevel.zoomLevel = zoom;

        // Scale node positions and panning so the point under cursor stays fixed
        float zoomFactor = zoom / oldZoom;
        ImVec2 panning = ImNodes::EditorContextGetPanning();

        // Scale all node positions around origin
        for (auto& node : activeGraph->nodes) {
            ImVec2 pos = node->position;
            pos.x *= zoomFactor;
            pos.y *= zoomFactor;
            ImNodes::SetNodeEditorSpacePos(node->id, pos);
            node->position = pos;
        }

        // Scale panning and adjust so cursor point stays fixed
        ImVec2 mousePos = ImGui::GetIO().MousePos;
        ImVec2 canvasOrigin = ImGui::GetItemRectMin();
        float mx = mousePos.x - canvasOrigin.x;
        float my = mousePos.y - canvasOrigin.y;

        ImVec2 newPanning;
        newPanning.x = mx - zoomFactor * (mx - panning.x);
        newPanning.y = my - zoomFactor * (my - panning.y);
        ImNodes::EditorContextResetPanning(newPanning);
    }

    // Handle interactions
    handleNewLinks(*activeGraph);
    handleDeletedLinks(*activeGraph);

    // Track selection
    updateSelection();

    // Double-click detection for compound nodes
    if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
        int hoveredNode = -1;
        if (ImNodes::IsNodeHovered(&hoveredNode)) {
            SimNode* node = activeGraph->findNode(hoveredNode);
            if (node) {
                auto* compound = dynamic_cast<CompoundNode*>(node);
                if (compound) {
                    pushGraph(&compound->innerGraph(), node->name);
                }
            }
        }
    }

    // Set initial positions
    if (currentLevel.firstRender && !activeGraph->nodes.empty()) {
        float x = 50.0f;
        for (auto& node : activeGraph->nodes) {
            ImNodes::SetNodeEditorSpacePos(node->id, ImVec2(x, 50.0f));
            node->position = ImVec2(x, 50.0f);
            x += 180.0f;
        }
        currentLevel.firstRender = false;
    }

    // Update node positions from editor
    for (auto& node : activeGraph->nodes) {
        node->position = ImNodes::GetNodeEditorSpacePos(node->id);
    }

    ImGui::End();

    // Property panel (separate window)
    renderPropertyPanel(*activeGraph);

    return result;
}

void NodeEditorUI::renderBreadcrumbs() {
    if (m_editorStack.size() <= 1) return; // Don't show if only root

    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.2f, 0.3f, 1.0f));

    for (size_t i = 0; i < m_editorStack.size(); i++) {
        if (i > 0) {
            ImGui::SameLine();
            ImGui::TextDisabled(">");
            ImGui::SameLine();
        }

        bool isCurrent = (i == m_editorStack.size() - 1);
        if (isCurrent) {
            ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "%s",
                               m_editorStack[i].name.c_str());
        } else {
            if (ImGui::SmallButton(m_editorStack[i].name.c_str())) {
                // Pop back to this level
                while (m_editorStack.size() > i + 1) {
                    popGraph();
                }
            }
        }
    }

    // Back button
    ImGui::SameLine();
    ImGui::Spacing();
    ImGui::SameLine();
    if (ImGui::SmallButton("Back")) {
        popGraph();
    }

    ImGui::PopStyleColor();
    ImGui::Separator();
}

void NodeEditorUI::renderMenuBar(SimGraph& graph) {
    // Right-click context menu to add nodes
    if (ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows) &&
        ImNodes::IsEditorHovered() &&
        ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
        ImGui::OpenPopup("add_node_popup");
    }

    if (ImGui::BeginPopup("add_node_popup")) {
        ImGui::TextDisabled("Add Node");
        ImGui::Separator();

        const auto& types = NodeRegistry::instance().typeNames();
        for (const auto& typeName : types) {
            if (ImGui::MenuItem(typeName.c_str())) {
                int nodeId = graph.addNode(typeName);
                if (nodeId >= 0) {
                    ImVec2 mousePos = ImGui::GetMousePosOnOpeningCurrentPopup();
                    ImNodes::SetNodeScreenSpacePos(nodeId, mousePos);
                }
            }
        }
        ImGui::EndPopup();
    }
}

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

void NodeEditorUI::renderInlineParam(NodeParam& param, int nodeId) {
    float z = m_editorStack.empty() ? 1.0f : m_editorStack.back().zoomLevel;
    ImGui::PushItemWidth(100.0f * z);
    ImGui::PushID(&param);

    // Override indicator dot
    if (param.overridden) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.2f, 0.8f, 1.0f, 1.0f));
        ImGui::Bullet();
        ImGui::PopStyleColor();
        ImGui::SameLine();
    }

    const auto& d = param.descriptor;
    const char* fmt = d.format.empty() ? nullptr : d.format.c_str();

    // Render based on type
    std::visit([&](auto& val) {
        using T = std::decay_t<decltype(val)>;
        if constexpr (std::is_same_v<T, float>) {
            float v = val;
            bool changed = false;
            if (d.uiHint == ParamDescriptor::UIHint::Slider && d.minVal && d.maxVal) {
                changed = ImGui::SliderFloat(d.label.c_str(), &v,
                                             *d.minVal, *d.maxVal, fmt ? fmt : "%.3f");
            } else {
                float speed = (d.maxVal && d.minVal) ? (*d.maxVal - *d.minVal) * 0.002f : 0.01f;
                changed = ImGui::DragFloat(d.label.c_str(), &v, speed,
                                           d.minVal.value_or(0.0f),
                                           d.maxVal.value_or(0.0f),
                                           fmt ? fmt : "%.3f");
            }
            if (changed) {
                val = v;
                param.overridden = true;
            }
        } else if constexpr (std::is_same_v<T, int>) {
            int v = val;
            bool changed = false;
            if (!d.enumLabels.empty()) {
                if (ImGui::BeginCombo(d.label.c_str(),
                    (v >= 0 && v < (int)d.enumLabels.size()) ? d.enumLabels[v].c_str() : "?")) {
                    for (int i = 0; i < (int)d.enumLabels.size(); i++) {
                        if (ImGui::Selectable(d.enumLabels[i].c_str(), v == i)) {
                            v = i;
                            changed = true;
                        }
                    }
                    ImGui::EndCombo();
                }
            } else {
                changed = ImGui::DragInt(d.label.c_str(), &v, 1.0f,
                                         d.minVal ? (int)*d.minVal : 0,
                                         d.maxVal ? (int)*d.maxVal : 0);
            }
            if (changed) {
                val = v;
                param.overridden = true;
            }
        } else if constexpr (std::is_same_v<T, bool>) {
            bool v = val;
            if (ImGui::Checkbox(d.label.c_str(), &v)) {
                val = v;
                param.overridden = true;
            }
        }
    }, param.value);

    if (!d.tooltip.empty() && ImGui::IsItemHovered()) {
        ImGui::SetTooltip("%s", d.tooltip.c_str());
    }

    ImGui::PopID();
    ImGui::PopItemWidth();
}

void NodeEditorUI::renderParamWidget(NodeParam& param) {
    const auto& d = param.descriptor;
    const char* fmt = d.format.empty() ? nullptr : d.format.c_str();

    ImGui::PushID(&param);

    // Override toggle
    bool ov = param.overridden;
    if (ImGui::Checkbox("##override", &ov)) {
        if (ov) {
            param.overridden = true;
        } else {
            param.value = param.descriptor.defaultValue;
            param.overridden = false;
        }
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Override global value");
    }
    ImGui::SameLine();

    if (!param.overridden) ImGui::BeginDisabled();

    ImGui::PushItemWidth(-60.0f);

    std::visit([&](auto& val) {
        using T = std::decay_t<decltype(val)>;
        if constexpr (std::is_same_v<T, float>) {
            float v = val;
            bool changed = false;
            if (d.uiHint == ParamDescriptor::UIHint::Slider && d.minVal && d.maxVal) {
                changed = ImGui::SliderFloat(d.label.c_str(), &v,
                                             *d.minVal, *d.maxVal, fmt ? fmt : "%.3f");
            } else if (d.uiHint == ParamDescriptor::UIHint::Input) {
                changed = ImGui::InputFloat(d.label.c_str(), &v, 0.0f, 0.0f, fmt ? fmt : "%.3f");
            } else {
                float speed = (d.maxVal && d.minVal) ? (*d.maxVal - *d.minVal) * 0.002f : 0.01f;
                changed = ImGui::DragFloat(d.label.c_str(), &v, speed,
                                           d.minVal.value_or(0.0f),
                                           d.maxVal.value_or(0.0f),
                                           fmt ? fmt : "%.3f");
            }
            if (changed) {
                val = v;
                param.overridden = true;
            }
        } else if constexpr (std::is_same_v<T, int>) {
            int v = val;
            bool changed = false;
            if (!d.enumLabels.empty()) {
                if (ImGui::BeginCombo(d.label.c_str(),
                    (v >= 0 && v < (int)d.enumLabels.size()) ? d.enumLabels[v].c_str() : "?")) {
                    for (int i = 0; i < (int)d.enumLabels.size(); i++) {
                        if (ImGui::Selectable(d.enumLabels[i].c_str(), v == i)) {
                            v = i;
                            changed = true;
                        }
                    }
                    ImGui::EndCombo();
                }
            } else {
                changed = ImGui::DragInt(d.label.c_str(), &v, 1.0f,
                                         d.minVal ? (int)*d.minVal : 0,
                                         d.maxVal ? (int)*d.maxVal : 0);
            }
            if (changed) {
                val = v;
                param.overridden = true;
            }
        } else if constexpr (std::is_same_v<T, bool>) {
            bool v = val;
            if (ImGui::Checkbox(d.label.c_str(), &v)) {
                val = v;
                param.overridden = true;
            }
        } else if constexpr (std::is_same_v<T, std::string>) {
            char buf[256];
            strncpy(buf, val.c_str(), sizeof(buf) - 1);
            buf[sizeof(buf) - 1] = '\0';
            if (ImGui::InputText(d.label.c_str(), buf, sizeof(buf))) {
                val = std::string(buf);
                param.overridden = true;
            }
        }
    }, param.value);

    ImGui::PopItemWidth();

    if (!param.overridden) ImGui::EndDisabled();

    // Reset button
    ImGui::SameLine();
    if (ImGui::SmallButton("R")) {
        param.value = param.descriptor.defaultValue;
        param.overridden = false;
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Reset to default");
    }

    ImGui::PopID();
}

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
