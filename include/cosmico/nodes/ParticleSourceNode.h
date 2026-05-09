#pragma once
#include <cosmico/nodes/CompoundNode.h>
#include <kernels/particle_gen.cuh>

namespace cosmico {

class ParticleSourceCompound : public CompoundNode {
public:
    ParticleSourceCompound();
    void declareParams() override;
    void execute(ResourcePool& pool, cudaStream_t stream) override;

protected:
    void buildInnerGraph() override;

private:
    bool m_generated = false;
    cuda::ICGenParams m_lastGenParams{};

    // Build ICGenParams from current node params
    cuda::ICGenParams buildICGenParams(const ResourcePool& pool) const;
    bool paramsChanged(const cuda::ICGenParams& a, const cuda::ICGenParams& b) const;
};

} // namespace cosmico
