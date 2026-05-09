#include <cosmico/nodes/SimGraphExecutor.h>
#include <cstdio>

namespace cosmico {

void SimGraphExecutor::init(int maxParticles, int gridN, const PMParams& params) {
    m_pool.init(maxParticles, gridN);
    m_pool.setParams(params);
    fprintf(stderr, "[NodeGraph] Executor initialized: %d particles, %d^3 grid\n",
            maxParticles, gridN);
}

void SimGraphExecutor::destroy() {
    m_pool.destroy();
}

void SimGraphExecutor::execute(const CompiledPlan& plan, cudaStream_t stream) {
    if (!plan.valid) return;

    for (SimNode* node : plan.executionOrder) {
        node->execute(m_pool, stream);
    }
}

} // namespace cosmico
