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

// renderNodes/renderLinks live in NodeEditorUI_Nodes.cpp.
// renderInlineParam/renderParamWidget live in NodeEditorUI_Params.cpp.
// updateSelection/renderPropertyPanel/handleNewLinks/handleDeletedLinks
// live in NodeEditorUI_Interaction.cpp.

} // namespace cosmico

