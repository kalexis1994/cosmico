#pragma once
#include <cosmico/vulkan/VkBuffer.h>
#include <cosmico/vulkan/VkDescriptors.h>
#include <volk.h>
#include <string>

namespace cosmico {

class VkContext;
struct SimulationParams;

class NBodyCompute {
public:
    void init(VkContext& ctx, VkDescriptorSetLayout particleSetLayout,
              const std::string& shaderDir);
    void destroy(VkContext& ctx);

    void updateParams(VkContext& ctx, const SimulationParams& params);
    void dispatch(VkCommandBuffer cmd, VkDescriptorSet particleSet, uint32_t particleCount);

private:
    VkPipeline m_pipeline = VK_NULL_HANDLE;
    VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_paramsSetLayout = VK_NULL_HANDLE;
    VkDescriptorSet m_paramsSet = VK_NULL_HANDLE;
    DescriptorPool m_descriptorPool;
    GpuBuffer m_paramsBuffer;

    static constexpr uint32_t WORKGROUP_SIZE = 256;
};

} // namespace cosmico
