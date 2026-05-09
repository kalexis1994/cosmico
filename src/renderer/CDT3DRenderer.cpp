#include <cosmico/renderer/CDT3DRenderer.h>
#include <cosmico/vulkan/VkContext.h>
#include <cosmico/vulkan/VkPipeline.h>
#include <cosmico/vulkan/VkDescriptors.h>
#include <stdexcept>
#include <cstring>
#include <cstdio>
#include <algorithm>

namespace cosmico {

// Max 300K triangles * 3 verts * 9 floats = 8.1M floats = 32.4MB
static constexpr int MAX_TRIANGLES = 300000;
static constexpr VkDeviceSize MAX_SSBO_SIZE = MAX_TRIANGLES * 3 * 9 * sizeof(float);

void CDT3DRenderer::init(VkContext& ctx, VkRenderPass renderPass, const std::string& shaderDir) {
    VkDevice device = ctx.device();
    m_allocator = ctx.allocator();

    // ── Host-visible SSBO (CPU_TO_GPU) ────────────────────────────────────
    VkBufferCreateInfo bufInfo{};
    bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufInfo.size = MAX_SSBO_SIZE;
    bufInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

    VmaAllocationCreateInfo allocCI{};
    allocCI.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;
    allocCI.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VmaAllocationInfo allocInfo{};
    if (vmaCreateBuffer(m_allocator, &bufInfo, &allocCI,
                        &m_ssbo, &m_ssboAlloc, &allocInfo) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create CDT3D SSBO");
    }
    m_ssboMapped = allocInfo.pMappedData;
    m_ssboSize = MAX_SSBO_SIZE;

    // ── Descriptor Set Layout: binding 0 = SSBO, vertex stage ─────────────
    m_setLayout = DescriptorLayoutBuilder()
        .addBinding(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_VERTEX_BIT)
        .build(device);

    // ── Descriptor Pool + Set ─────────────────────────────────────────────
    VkDescriptorPoolSize poolSize{};
    poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSize.descriptorCount = 1;

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets = 1;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;

    if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &m_descriptorPool) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create CDT3D descriptor pool");
    }

    VkDescriptorSetAllocateInfo setAllocInfo{};
    setAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    setAllocInfo.descriptorPool = m_descriptorPool;
    setAllocInfo.descriptorSetCount = 1;
    setAllocInfo.pSetLayouts = &m_setLayout;

    if (vkAllocateDescriptorSets(device, &setAllocInfo, &m_descriptorSet) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate CDT3D descriptor set");
    }

    // Write descriptor
    VkDescriptorBufferInfo descBufInfo{};
    descBufInfo.buffer = m_ssbo;
    descBufInfo.offset = 0;
    descBufInfo.range = MAX_SSBO_SIZE;

    DescriptorWriter()
        .writeBuffer(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &descBufInfo)
        .update(device, m_descriptorSet);

    // ── Pipeline Layout ───────────────────────────────────────────────────
    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    pushRange.offset = 0;
    pushRange.size = sizeof(CDT3DRenderPushConstants);

    VkPipelineLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.setLayoutCount = 1;
    layoutInfo.pSetLayouts = &m_setLayout;
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges = &pushRange;

    if (vkCreatePipelineLayout(device, &layoutInfo, nullptr, &m_pipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create CDT3D pipeline layout");
    }

    // ── Graphics Pipeline ─────────────────────────────────────────────────
    auto vertShader = loadShaderModule(device, shaderDir + "/cdt3d.vert.spv");
    auto fragShader = loadShaderModule(device, shaderDir + "/cdt3d.frag.spv");

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vertShader;
    stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fragShader;
    stages[1].pName = "main";

    // No vertex input (reading from SSBO via gl_VertexIndex)
    VkPipelineVertexInputStateCreateInfo vertexInput{};
    vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_NONE;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    // Depth test ON (3D rendering needs depth)
    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

    // Opaque (no blending)
    VkPipelineColorBlendAttachmentState colorBlend{};
    colorBlend.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlend.blendEnable = VK_FALSE;

    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlend;

    VkDynamicState dynamicStates[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = 2;
    dynamicState.pDynamicStates = dynamicStates;

    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = stages;
    pipelineInfo.pVertexInputState = &vertexInput;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pDepthStencilState = &depthStencil;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.pDynamicState = &dynamicState;
    pipelineInfo.layout = m_pipelineLayout;
    pipelineInfo.renderPass = renderPass;
    pipelineInfo.subpass = 0;

    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_pipeline) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create CDT3D graphics pipeline");
    }

    vkDestroyShaderModule(device, vertShader, nullptr);
    vkDestroyShaderModule(device, fragShader, nullptr);

    fprintf(stderr, "[CDT3DRenderer] Initialized: max %d triangles, SSBO %.1f MB\n",
            MAX_TRIANGLES, MAX_SSBO_SIZE / (1024.0f * 1024.0f));
}

void CDT3DRenderer::destroy(VkDevice device, VmaAllocator allocator) {
    if (m_pipeline) {
        vkDestroyPipeline(device, m_pipeline, nullptr);
        m_pipeline = VK_NULL_HANDLE;
    }
    if (m_pipelineLayout) {
        vkDestroyPipelineLayout(device, m_pipelineLayout, nullptr);
        m_pipelineLayout = VK_NULL_HANDLE;
    }
    if (m_descriptorPool) {
        vkDestroyDescriptorPool(device, m_descriptorPool, nullptr);
        m_descriptorPool = VK_NULL_HANDLE;
    }
    if (m_setLayout) {
        vkDestroyDescriptorSetLayout(device, m_setLayout, nullptr);
        m_setLayout = VK_NULL_HANDLE;
    }
    if (m_ssbo) {
        vmaDestroyBuffer(allocator, m_ssbo, m_ssboAlloc);
        m_ssbo = VK_NULL_HANDLE;
        m_ssboMapped = nullptr;
    }
    m_triangleCount = 0;
}

void CDT3DRenderer::updateMesh(const float* vertexData, int triangleCount) {
    if (!m_ssboMapped) return;

    int clampedCount = (triangleCount < MAX_TRIANGLES) ? triangleCount : MAX_TRIANGLES;
    VkDeviceSize dataSize = static_cast<VkDeviceSize>(clampedCount) * 3 * 9 * sizeof(float);
    memcpy(m_ssboMapped, vertexData, dataSize);
    m_triangleCount = clampedCount;
}

void CDT3DRenderer::draw(VkCommandBuffer cmd, const CDT3DRenderPushConstants& pc) {
    if (!m_pipeline || m_triangleCount == 0) return;

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout,
                            0, 1, &m_descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmd, m_pipelineLayout,
                       VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, sizeof(CDT3DRenderPushConstants), &pc);
    vkCmdDraw(cmd, static_cast<uint32_t>(3 * m_triangleCount), 1, 0, 0);
}

} // namespace cosmico
