#include <cosmico/nodes/NodeRegistry.h>
#include <cosmico/nodes/PMNodes.h>
#include <cosmico/nodes/NBodyNodes.h>
#include <cosmico/nodes/ParticleSourceNode.h>
#include <cosmico/nodes/ICNodes.h>

namespace cosmico {

// Register PM pipeline nodes (ParticleSource replaced by compound version)
REGISTER_SIM_NODE("ParticleSource", ParticleSourceCompound);
REGISTER_SIM_NODE("HalfDrift",     HalfDriftNode);
REGISTER_SIM_NODE("ClearGrid",     ClearGridNode);
REGISTER_SIM_NODE("CICDeposit",    CICDepositNode);
REGISTER_SIM_NODE("FFTForward",    FFTForwardNode);
REGISTER_SIM_NODE("GreenMultiply", GreenMultiplyNode);
REGISTER_SIM_NODE("FFTInverse",    FFTInverseNode);
REGISTER_SIM_NODE("Gradient",      GradientNode);
REGISTER_SIM_NODE("CICInterpolate",CICInterpolateNode);
REGISTER_SIM_NODE("Kick",          KickNode);
REGISTER_SIM_NODE("SinkAccretion", SinkAccretionNode);
REGISTER_SIM_NODE("EnergyReduce",  EnergyReduceNode);
REGISTER_SIM_NODE("Output",        OutputNode);

// Register N-Body direct pipeline nodes (granular: reuses HalfDrift + Kick from PM)
REGISTER_SIM_NODE("ClearAccel",       ClearAccelNode);
REGISTER_SIM_NODE("PairwiseGravity",  PairwiseGravityNode);

// Register IC sub-nodes (used inside ParticleSource compound)
REGISTER_SIM_NODE("ICDistribution", ICDistributionNode);
REGISTER_SIM_NODE("ICVelocity",     ICVelocityNode);
REGISTER_SIM_NODE("ICMassAssign",   ICMassAssignNode);
REGISTER_SIM_NODE("ICNoise",        ICNoiseNode);
REGISTER_SIM_NODE("ICNormalize",    ICNormalizeNode);

} // namespace cosmico
