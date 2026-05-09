#include <cosmico/vulkan/VkPipeline.h>
#include <fstream>
#include <stdexcept>

namespace cosmico {

VkShaderModule loadShaderModule(VkDevice device, const std::string& filepath) {
    std::ifstream file(filepath, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open shader file: " + filepath);
    }

    size_t fileSize = static_cast<size_t>(file.tellg());
    std::vector<uint32_t> buffer(fileSize / sizeof(uint32_t));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(fileSize));

    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = fileSize;
    createInfo.pCode = buffer.data();

    VkShaderModule module;
    if (vkCreateShaderModule(device, &createInfo, nullptr, &module) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create shader module: " + filepath);
    }
    return module;
}

GraphicsPipelineBuilder& GraphicsPipelineBuilder::setShaders(VkShaderModule vert, VkShaderModule frag) {
    m_vertShader = vert;
    m_fragShader = frag;
    return *this;
}

GraphicsPipelineBuilder& GraphicsPipelineBuilder::setInputTopology(VkPrimitiveTopology topology) {
    m_topology = topology;
    return *this;
}

GraphicsPipelineBuilder& GraphicsPipelineBuilder::setPolygonMode(VkPolygonMode mode) {
    m_polygonMode = mode;
    return *this;
}

GraphicsPipelineBuilder& GraphicsPipelineBuilder::setCullMode(VkCullModeFlags cull, VkFrontFace front) {
    m_cullMode = cull;
    m_frontFace = front;
    return *this;
}

GraphicsPipelineBuilder& GraphicsPipelineBuilder::setDepthTest(bool enable, bool write, VkCompareOp op) {
    m_depthTest = enable;
    m_depthWrite = write;
    m_depthOp = op;
    return *this;
}

GraphicsPipelineBuilder& GraphicsPipelineBuilder::setColorFormat(VkFormat format) {
    m_colorFormat = format;
    return *this;
}

GraphicsPipelineBuilder& GraphicsPipelineBuilder::setDepthFormat(VkFormat format) {
    m_depthFormat = format;
    return *this;
}

GraphicsPipelineBuilder& GraphicsPipelineBuilder::enableAdditiveBlending() {
    m_additiveBlend = true;
    return *this;
}

GraphicsPipelineBuilder& GraphicsPipelineBuilder::setPipelineLayout(VkPipelineLayout layout) {
    m_pipelineLayout = layout;
    return *this;
}

VkPipeline GraphicsPipelineBuilder::build(VkDevice device) {
    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = m_vertShader;
    stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = m_fragShader;
    stages[1].pName = "main";

    // No vertex input — we read from SSBO via gl_VertexIndex
    VkPipelineVertexInputStateCreateInfo vertexInput{};
    vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = m_topology;
    inputAssembly.primitiveRestartEnable = VK_FALSE;

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode = m_polygonMode;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = m_cullMode;
    rasterizer.frontFace = m_frontFace;
    rasterizer.depthBiasEnable = VK_FALSE;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = m_depthTest ? VK_TRUE : VK_FALSE;
    depthStencil.depthWriteEnable = m_depthWrite ? VK_TRUE : VK_FALSE;
    depthStencil.depthCompareOp = m_depthOp;
    depthStencil.depthBoundsTestEnable = VK_FALSE;
    depthStencil.stencilTestEnable = VK_FALSE;

    VkPipelineColorBlendAttachmentState colorBlend{};
    colorBlend.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    if (m_additiveBlend) {
        colorBlend.blendEnable = VK_TRUE;
        colorBlend.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        colorBlend.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlend.colorBlendOp = VK_BLEND_OP_ADD;
        colorBlend.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlend.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlend.alphaBlendOp = VK_BLEND_OP_ADD;
    } else {
        colorBlend.blendEnable = VK_FALSE;
    }

    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.logicOpEnable = VK_FALSE;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlend;

    VkDynamicState dynamicStates[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = 2;
    dynamicState.pDynamicStates = dynamicStates;

    // We need a render pass — use dynamic rendering (VK 1.3) or create a compatible render pass.
    // For Vulkan 1.2 compatibility, we pass VK_NULL_HANDLE and use VkPipelineRenderingCreateInfo.
    // Actually, we'll use a traditional render pass passed externally. For now, build with
    // VK_NULL_HANDLE and let the caller create pipeline with a render pass.

    // We use dynamic rendering via VkPipelineRenderingCreateInfoKHR for simplicity...
    // Actually, let's just require a render pass from outside. We'll add it.
    // For now, the pipeline is built without a render pass (the Renderer will create one).

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
    pipelineInfo.renderPass = VK_NULL_HANDLE; // Set by caller via subpass

    // Use VkPipelineRenderingCreateInfo for dynamic rendering (Vulkan 1.2 + extension or 1.3)
    VkPipelineRenderingCreateInfo renderingInfo{};
    renderingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachmentFormats = &m_colorFormat;
    renderingInfo.depthAttachmentFormat = m_depthFormat;

    pipelineInfo.pNext = &renderingInfo;

    VkPipeline pipeline;
    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create graphics pipeline");
    }
    return pipeline;
}

ComputePipelineBuilder& ComputePipelineBuilder::setShader(VkShaderModule shader) {
    m_shader = shader;
    return *this;
}

ComputePipelineBuilder& ComputePipelineBuilder::setPipelineLayout(VkPipelineLayout layout) {
    m_layout = layout;
    return *this;
}

VkPipeline ComputePipelineBuilder::build(VkDevice device) {
    VkPipelineShaderStageCreateInfo stage{};
    stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = m_shader;
    stage.pName = "main";

    VkComputePipelineCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    info.stage = stage;
    info.layout = m_layout;

    VkPipeline pipeline;
    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create compute pipeline");
    }
    return pipeline;
}

} // namespace cosmico
