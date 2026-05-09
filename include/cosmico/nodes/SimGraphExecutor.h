#pragma once
#include <cosmico/nodes/ResourcePool.h>
#include <cosmico/nodes/SimGraphCompiler.h>

typedef struct CUstream_st* cudaStream_t;

namespace cosmico {

class SimGraphExecutor {
public:
    void init(int maxParticles, int gridN, const PMParams& params);
    void destroy();

    void execute(const CompiledPlan& plan, cudaStream_t stream);

    ResourcePool& resources() { return m_pool; }
    const ResourcePool& resources() const { return m_pool; }

private:
    ResourcePool m_pool;
};

} // namespace cosmico
