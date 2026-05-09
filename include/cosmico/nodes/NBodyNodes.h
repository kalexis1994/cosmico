#pragma once
#include <cosmico/nodes/SimNode.h>

namespace cosmico {

// Zero acceleration scratch fields (density/pressure/smoothing = ax/ay/az).
class ClearAccelNode : public SimNode {
public:
    ClearAccelNode();
    void execute(ResourcePool& pool, cudaStream_t stream) override;
};

// O(N^2) direct all-pairs gravity: computes pairwise gravitational
// acceleration, stores in density/pressure/smoothing fields (ax/ay/az).
class PairwiseGravityNode : public SimNode {
public:
    PairwiseGravityNode();
    void declareParams() override;
    void execute(ResourcePool& pool, cudaStream_t stream) override;
};

} // namespace cosmico
