#pragma once
#include <cosmico/nodes/SimNode.h>

namespace cosmico {

// Half drift: x += v * (dt/2)
class HalfDriftNode : public SimNode {
public:
    HalfDriftNode();
    void declareParams() override;
    void execute(ResourcePool& pool, cudaStream_t stream) override;
};

// Clear density grid to zero
class ClearGridNode : public SimNode {
public:
    ClearGridNode();
    void execute(ResourcePool& pool, cudaStream_t stream) override;
};

// CIC deposit: particles -> density grid
class CICDepositNode : public SimNode {
public:
    CICDepositNode();
    void declareParams() override;
    void execute(ResourcePool& pool, cudaStream_t stream) override;
};

// FFT forward: density(x) -> density_hat(k)
class FFTForwardNode : public SimNode {
public:
    FFTForwardNode();
    void execute(ResourcePool& pool, cudaStream_t stream) override;
};

// Green's function multiply in Fourier space
class GreenMultiplyNode : public SimNode {
public:
    GreenMultiplyNode();
    void declareParams() override;
    void execute(ResourcePool& pool, cudaStream_t stream) override;
};

// FFT inverse: potential_hat(k) -> potential(x)
class FFTInverseNode : public SimNode {
public:
    FFTInverseNode();
    void execute(ResourcePool& pool, cudaStream_t stream) override;
};

// Gradient: potential -> (forceX, forceY, forceZ)
class GradientNode : public SimNode {
public:
    GradientNode();
    void execute(ResourcePool& pool, cudaStream_t stream) override;
};

// CIC interpolation: grid forces -> particle accelerations
class CICInterpolateNode : public SimNode {
public:
    CICInterpolateNode();
    void declareParams() override;
    void execute(ResourcePool& pool, cudaStream_t stream) override;
};

// Kick: v = v * dampFactor + a * dt
class KickNode : public SimNode {
public:
    KickNode();
    void declareParams() override;
    void execute(ResourcePool& pool, cudaStream_t stream) override;
};

// Sink accretion: absorb particles near existing sinks (optional)
class SinkAccretionNode : public SimNode {
public:
    SinkAccretionNode();
    void declareParams() override;
    void execute(ResourcePool& pool, cudaStream_t stream) override;
};

// Energy reduction: compute kinetic + potential energy
class EnergyReduceNode : public SimNode {
public:
    EnergyReduceNode();
    void declareParams() override;
    void execute(ResourcePool& pool, cudaStream_t stream) override;
};

// Output: terminal sink node marking the pipeline endpoint for rendering.
// No computation — the particles are already in the shared buffer.
class OutputNode : public SimNode {
public:
    OutputNode();
    void execute(ResourcePool& pool, cudaStream_t stream) override;
};

} // namespace cosmico
