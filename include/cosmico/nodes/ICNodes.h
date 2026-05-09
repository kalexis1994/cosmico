#pragma once
#include <cosmico/nodes/SimNode.h>

namespace cosmico {

// IC sub-nodes for the ParticleSource compound node's inner graph.
// These are conceptual pipeline stages that the compound executes internally.

// Position generation (profile-dependent)
class ICDistributionNode : public SimNode {
public:
    ICDistributionNode();
    void declareParams() override;
    void execute(ResourcePool& pool, cudaStream_t stream) override;
};

// Velocity assignment (orbital/thermal/zero)
class ICVelocityNode : public SimNode {
public:
    ICVelocityNode();
    void declareParams() override;
    void execute(ResourcePool& pool, cudaStream_t stream) override;
};

// Mass function (equal/power-law)
class ICMassAssignNode : public SimNode {
public:
    ICMassAssignNode();
    void declareParams() override;
    void execute(ResourcePool& pool, cudaStream_t stream) override;
};

// Perturbations (none/perlin/white)
class ICNoiseNode : public SimNode {
public:
    ICNoiseNode();
    void declareParams() override;
    void execute(ResourcePool& pool, cudaStream_t stream) override;
};

// Center + remove bulk momentum
class ICNormalizeNode : public SimNode {
public:
    ICNormalizeNode();
    void execute(ResourcePool& pool, cudaStream_t stream) override;
};

} // namespace cosmico
