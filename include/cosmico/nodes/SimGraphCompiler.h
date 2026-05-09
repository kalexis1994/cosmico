#pragma once
#include <cosmico/nodes/SimGraph.h>
#include <vector>
#include <string>

namespace cosmico {

struct CompiledPlan {
    std::vector<SimNode*> executionOrder;  // Topologically sorted
    bool valid = false;
    std::string error;
};

class SimGraphCompiler {
public:
    CompiledPlan compile(SimGraph& graph);

private:
    bool validate(SimGraph& graph, std::string& error);
    std::vector<SimNode*> topologicalSort(SimGraph& graph);
    bool hasCycle(SimGraph& graph);
};

} // namespace cosmico
