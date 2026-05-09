#pragma once
#include <cosmico/nodes/SimNode.h>
#include <cosmico/nodes/SimGraph.h>
#include <cosmico/nodes/SimGraphCompiler.h>

namespace cosmico {

class CompoundNode : public SimNode {
public:
    void execute(ResourcePool& pool, cudaStream_t stream) override;

    SimGraph& innerGraph() { return m_innerGraph; }
    const SimGraph& innerGraph() const { return m_innerGraph; }

    bool compile();

    bool expanded = false;

protected:
    virtual void buildInnerGraph() = 0;

    SimGraph m_innerGraph;
    SimGraphCompiler m_innerCompiler;
    CompiledPlan m_innerPlan;
};

} // namespace cosmico
