#include <cosmico/simulation/NBodyCompute.h>
#include <cosmico/vulkan/VkContext.h>
#include <cosmico/vulkan/VkPipeline.h>
#include <cosmico/vulkan/VkDescriptors.h>
#include <cosmico/simulation/SimulationParams.h>

namespace cosmico {

void NBodyCompute::init(VkContext& ctx, VkDescriptorSetLayout particleSetLayout,
                         const std::string& shaderDir) {
    // UBO layout for simulation params
    m_paramsSetLayout = DescriptorLayoutBuilder()
        .addBinding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT)
        .build(ctx.device());

    // Params UBO buffer
    m_paramsBuffer.create(ctx.allocator(), sizeof(SimulationParams),
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        VMA_MEMORY_USAGE_AUTO,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
        VMA_ALLOCATION_CREATE_MAPPED_BIT);

    // Descriptor pool and set for params
    std::vector<VkDescriptorPoolSize> poolSizes = {
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 }
    };
    m_descriptorPool.init(ctx.device(), 1, poolSizes);
    m_paramsSet = m_descriptorPool.allocate(ctx.device(), m_paramsSetLayout);

    auto bufInfo = m_paramsBuffer.descriptorInfo();
    DescriptorWriter()
        .writeBuffer(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, &bufInfo)
        .update(ctx.device(), m_paramsSet);

    // Pipeline layout: set 0 = particles (SSBO in/out), set 1 = params (UBO)
    VkDescriptorSetLayout layouts[] = { particleSetLayout, m_paramsSetLayout };
    VkPipelineLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.setLayoutCount = 2;
    layoutInfo.pSetLayouts = layouts;

    vkCreatePipelineLayout(ctx.device(), &layoutInfo, nullptr, &m_pipelineLayout);

    // Load compute shader and build pipeline
    auto shader = loadShaderModule(ctx.device(), shaderDir + "/nbody_gravity.comp.spv");
    m_pipeline = ComputePipelineBuilder()
        .setShader(shader)
        .setPipelineLayout(m_pipelineLayout)
        .build(ctx.device());

    vkDestroyShaderModule(ctx.device(), shader, nullptr);
}

void NBodyCompute::destroy(VkContext& ctx) {
    vkDestroyPipeline(ctx.device(), m_pipeline, nullptr);
    vkDestroyPipelineLayout(ctx.device(), m_pipelineLayout, nullptr);
    m_descriptorPool.destroy(ctx.device());
    vkDestroyDescriptorSetLayout(ctx.device(), m_paramsSetLayout, nullptr);
    m_paramsBuffer.destroy(ctx.allocator());
}

void NBodyCompute::updateParams(VkContext& ctx, const SimulationParams& params) {
    m_paramsBuffer.upload(ctx.allocator(), &params, sizeof(SimulationParams));
}

void NBodyCompute::dispatch(VkCommandBuffer cmd, VkDescriptorSet particleSet, uint32_t particleCount) {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline);

    VkDescriptorSet sets[] = { particleSet, m_paramsSet };
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout,
                            0, 2, sets, 0, nullptr);

    uint32_t groupCount = (particleCount + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
    vkCmdDispatch(cmd, groupCount, 1, 1);
}

} // namespace cosmico
