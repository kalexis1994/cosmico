#include <cosmico/renderer/ParticleRenderer.h>
#include <cosmico/vulkan/VkContext.h>
#include <cosmico/vulkan/VkPipeline.h>
#include <stdexcept>

namespace cosmico {

// Helper: create a graphics pipeline with the given fragment shader
static VkPipeline createParticlePipeline(
    VkDevice device, VkRenderPass renderPass, VkPipelineLayout layout,
    VkShaderModule vertShader, VkShaderModule fragShader,
    bool additive = false)
{
    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vertShader;
    stages[0].pName  = "main";
    stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fragShader;
    stages[1].pName  = "main";

    VkPipelineVertexInputStateCreateInfo vertexInput{};
    vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount  = 1;

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType       = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth   = 1.0f;
    rasterizer.cullMode    = VK_CULL_MODE_NONE;
    rasterizer.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable  = additive ? VK_FALSE : VK_TRUE;
    depthStencil.depthWriteEnable = additive ? VK_FALSE : VK_TRUE;
    depthStencil.depthCompareOp   = VK_COMPARE_OP_LESS_OR_EQUAL;

    VkPipelineColorBlendAttachmentState colorBlend{};
    colorBlend.colorWriteMask     = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                    VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlend.blendEnable        = VK_TRUE;
    if (additive) {
        // Additive blending: overlapping particles sum their brightness
        colorBlend.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        colorBlend.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlend.colorBlendOp        = VK_BLEND_OP_ADD;
        colorBlend.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlend.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlend.alphaBlendOp        = VK_BLEND_OP_ADD;
    } else {
        // Standard alpha blending
        colorBlend.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        colorBlend.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        colorBlend.colorBlendOp        = VK_BLEND_OP_ADD;
        colorBlend.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlend.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        colorBlend.alphaBlendOp        = VK_BLEND_OP_ADD;
    }

    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments    = &colorBlend;

    VkDynamicState dynamicStates[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = 2;
    dynamicState.pDynamicStates    = dynamicStates;

    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount          = 2;
    pipelineInfo.pStages             = stages;
    pipelineInfo.pVertexInputState   = &vertexInput;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState      = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState   = &multisampling;
    pipelineInfo.pDepthStencilState  = &depthStencil;
    pipelineInfo.pColorBlendState    = &colorBlending;
    pipelineInfo.pDynamicState       = &dynamicState;
    pipelineInfo.layout              = layout;
    pipelineInfo.renderPass          = renderPass;
    pipelineInfo.subpass             = 0;

    VkPipeline pipeline = VK_NULL_HANDLE;
    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo,
                                  nullptr, &pipeline) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create particle graphics pipeline");
    }
    return pipeline;
}

void ParticleRenderer::init(VkContext& ctx, VkRenderPass renderPass,
                             VkDescriptorSetLayout particleSetLayout,
                             const std::string& shaderDir,
                             VkDescriptorSetLayout textureSetLayout) {
    // Pipeline layout with particle SSBO descriptor set + push constants
    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    pushRange.offset     = 0;
    pushRange.size       = sizeof(ParticleRenderPushConstants);

    VkDescriptorSetLayout setLayouts[2] = { particleSetLayout, textureSetLayout };
    uint32_t setLayoutCount = (textureSetLayout != VK_NULL_HANDLE) ? 2 : 1;

    VkPipelineLayoutCreateInfo layoutInfo{};
    layoutInfo.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.setLayoutCount         = setLayoutCount;
    layoutInfo.pSetLayouts            = setLayouts;
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges    = &pushRange;

    vkCreatePipelineLayout(ctx.device(), &layoutInfo, nullptr, &m_pipelineLayout);

    // Load shaders
    auto vertShader       = loadShaderModule(ctx.device(), shaderDir + "/particle.vert.spv");
    auto fragShader       = loadShaderModule(ctx.device(), shaderDir + "/particle.frag.spv");
    auto fragShaderSimple = loadShaderModule(ctx.device(), shaderDir + "/particle_simple.frag.spv");

    // Sphere pipeline: uses gl_FragDepth for correct sphere depth
    m_pipeline = createParticlePipeline(
        ctx.device(), renderPass, m_pipelineLayout,
        vertShader, fragShader);

    // Simple pipeline: additive blending, no depth write → glowing particles
    m_simplePipeline = createParticlePipeline(
        ctx.device(), renderPass, m_pipelineLayout,
        vertShader, fragShaderSimple, true);

    vkDestroyShaderModule(ctx.device(), vertShader, nullptr);
    vkDestroyShaderModule(ctx.device(), fragShader, nullptr);
    vkDestroyShaderModule(ctx.device(), fragShaderSimple, nullptr);
}

void ParticleRenderer::destroy(VkDevice device) {
    vkDestroyPipeline(device, m_pipeline, nullptr);
    vkDestroyPipeline(device, m_simplePipeline, nullptr);
    vkDestroyPipelineLayout(device, m_pipelineLayout, nullptr);
}

void ParticleRenderer::draw(VkCommandBuffer cmd, VkDescriptorSet particleSet,
                             uint32_t particleCount,
                             const ParticleRenderPushConstants& pushConstants,
                             VkDescriptorSet textureSet,
                             uint32_t sphereCount) {
    // Bind shared resources
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout,
                            0, 1, &particleSet, 0, nullptr);
    if (textureSet != VK_NULL_HANDLE) {
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout,
                                1, 1, &textureSet, 0, nullptr);
    }
    vkCmdPushConstants(cmd, m_pipelineLayout,
                       VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, sizeof(ParticleRenderPushConstants), &pushConstants);

    // Draw 1: Spherical bodies (first sphereCount instances) — ray-sphere + gl_FragDepth
    if (sphereCount > 0) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline);
        vkCmdDraw(cmd, 6, sphereCount, 0, 0);
    }

    // Draw 2: Regular particles (remaining) — simple shader, early-Z enabled
    uint32_t regularCount = particleCount - sphereCount;
    if (regularCount > 0) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_simplePipeline);
        vkCmdDraw(cmd, 6, regularCount, 0, sphereCount);
    }
}

} // namespace cosmico
