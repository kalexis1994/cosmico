#pragma once
#include <volk.h>
#include <vk_mem_alloc.h>
#include <string>
#include <cstdint>

namespace cosmico {

class VkContext;

struct CMBRenderPushConstants {
    alignas(16) float invViewProj[16]; // 64 bytes
    alignas(16) float cameraPos[4];    // 16 bytes (xyz + padding)
    alignas(16) float sphereCenter[4]; // 16 bytes (xyz + radius in w)
    float contrastScale;               // 4 bytes
    float opacity;                     // 4 bytes
    float padding[2];                  // 8 bytes -> total 112
};

class CMBRenderer {
public:
    void init(VkContext& ctx, VkRenderPass renderPass, const std::string& shaderDir);
    void destroy(VkDevice device, VmaAllocator allocator);
    void updateTexture(VkCommandBuffer cmd, const uint8_t* hostData, int width, int height);
    void draw(VkCommandBuffer cmd, const CMBRenderPushConstants& pc);
    bool isReady() const { return m_textureW > 0; }

private:
    // 2D texture (equirectangular CMB map)
    VkImage m_cmbImage = VK_NULL_HANDLE;
    VmaAllocation m_cmbAlloc = VK_NULL_HANDLE;
    VkImageView m_cmbView = VK_NULL_HANDLE;
    VkSampler m_sampler = VK_NULL_HANDLE;

    // Staging buffer
    VkBuffer m_stagingBuffer = VK_NULL_HANDLE;
    VmaAllocation m_stagingAlloc = VK_NULL_HANDLE;

    // Descriptors
    VkDescriptorSetLayout m_setLayout = VK_NULL_HANDLE;
    VkDescriptorPool m_descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet m_descriptorSet = VK_NULL_HANDLE;

    // Pipeline
    VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_pipeline = VK_NULL_HANDLE;

    // State
    int m_textureW = 0;
    int m_textureH = 0;
    VmaAllocator m_allocator = VK_NULL_HANDLE;
};

} // namespace cosmico
