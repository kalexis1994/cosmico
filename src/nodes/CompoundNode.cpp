#include <cosmico/nodes/CompoundNode.h>
#include <cosmico/nodes/ResourcePool.h>

namespace cosmico {

void CompoundNode::execute(ResourcePool& pool, cudaStream_t stream) {
    if (!m_innerPlan.valid) return;

    for (SimNode* node : m_innerPlan.executionOrder) {
        node->execute(pool, stream);
    }
}

bool CompoundNode::compile() {
    m_innerPlan = m_innerCompiler.compile(m_innerGraph);
    return m_innerPlan.valid;
}

} // namespace cosmico
