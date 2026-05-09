#pragma once
#include <cosmico/nodes/SimGraph.h>
#include <cosmico/nodes/SimGraphCompiler.h>
#include <string>
#include <vector>

// Forward declaration — imnodes uses ImNodesEditorContext*
struct ImNodesEditorContext;

namespace cosmico {

class NodeEditorUI {
public:
    void init();
    void destroy();

    struct Result {
        bool buildRequested = false;
        bool resetRequested = false;
    };

    Result render(SimGraph& graph, const CompiledPlan* currentPlan,
                  bool isRunning, bool cudaAvailable);

private:
    void renderMenuBar(SimGraph& graph);
    void renderNodes(SimGraph& graph);
    void renderLinks(SimGraph& graph);
    void handleNewLinks(SimGraph& graph);
    void handleDeletedLinks(SimGraph& graph);
    void updateSelection();
    void renderPropertyPanel(SimGraph& graph);
    void renderInlineParam(NodeParam& param, int nodeId);
    void renderParamWidget(NodeParam& param);
    void renderBreadcrumbs();

    // Sub-graph navigation
    void pushGraph(SimGraph* graph, const std::string& name);
    void popGraph();
    SimGraph* currentGraph();

    struct EditorLevel {
        SimGraph* graph = nullptr;
        ImNodesEditorContext* context = nullptr;
        std::string name;
        bool firstRender = true;
        float zoomLevel = 1.0f;
    };
    std::vector<EditorLevel> m_editorStack;

    int m_selectedNodeId = -1;
    static constexpr float ZOOM_MIN = 0.25f;
    static constexpr float ZOOM_MAX = 2.0f;
};

} // namespace cosmico
