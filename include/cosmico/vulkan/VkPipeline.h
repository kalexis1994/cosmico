#pragma once
#include <volk.h>
#include <string>
#include <vector>

namespace cosmico {

VkShaderModule loadShaderModule(VkDevice device, const std::string& filepath);

class GraphicsPipelineBuilder {
public:
    GraphicsPipelineBuilder& setShaders(VkShaderModule vert, VkShaderModule frag);
    GraphicsPipelineBuilder& setInputTopology(VkPrimitiveTopology topology);
    GraphicsPipelineBuilder& setPolygonMode(VkPolygonMode mode);
    GraphicsPipelineBuilder& setCullMode(VkCullModeFlags cull, VkFrontFace front);
    GraphicsPipelineBuilder& setDepthTest(bool enable, bool write, VkCompareOp op);
    GraphicsPipelineBuilder& setColorFormat(VkFormat format);
    GraphicsPipelineBuilder& setDepthFormat(VkFormat format);
    GraphicsPipelineBuilder& enableAdditiveBlending();
    GraphicsPipelineBuilder& setPipelineLayout(VkPipelineLayout layout);

    VkPipeline build(VkDevice device);

private:
    VkShaderModule m_vertShader = VK_NULL_HANDLE;
    VkShaderModule m_fragShader = VK_NULL_HANDLE;
    VkPrimitiveTopology m_topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
    VkPolygonMode m_polygonMode = VK_POLYGON_MODE_FILL;
    VkCullModeFlags m_cullMode = VK_CULL_MODE_NONE;
    VkFrontFace m_frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    bool m_depthTest = false;
    bool m_depthWrite = false;
    VkCompareOp m_depthOp = VK_COMPARE_OP_LESS_OR_EQUAL;
    VkFormat m_colorFormat = VK_FORMAT_B8G8R8A8_SRGB;
    VkFormat m_depthFormat = VK_FORMAT_UNDEFINED;
    bool m_additiveBlend = false;
    VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
};

class ComputePipelineBuilder {
public:
    ComputePipelineBuilder& setShader(VkShaderModule shader);
    ComputePipelineBuilder& setPipelineLayout(VkPipelineLayout layout);
    VkPipeline build(VkDevice device);

private:
    VkShaderModule m_shader = VK_NULL_HANDLE;
    VkPipelineLayout m_layout = VK_NULL_HANDLE;
};

} // namespace cosmico
