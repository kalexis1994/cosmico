#pragma once
#include <volk.h>
#include <vk_mem_alloc.h>
#include <string>
#include <cstdint>

namespace cosmico {

class VkContext;

struct VolumeRenderPushConstants {
    alignas(16) float invViewProj[16]; // 64 bytes
    alignas(16) float cameraPos[4];    // 16 bytes (xyz + padding)
    alignas(16) float volumeMin[4];    // 16 bytes (xyz = -boxHalf)
    alignas(16) float volumeMax[4];    // 16 bytes (xyz = +boxHalf)
    float opacityScale;                // 4 bytes
    float stepSize;                    // 4 bytes
    int numSteps;                      // 4 bytes
    float padding;                     // 4 bytes → total 128
};

class VolumeRenderer {
public:
    void init(VkContext& ctx, VkRenderPass renderPass, const std::string& shaderDir);
    void destroy(VkDevice device, VmaAllocator allocator);
    void updateTexture(VkCommandBuffer cmd, const uint8_t* hostData, int N);
    void draw(VkCommandBuffer cmd, const VolumeRenderPushConstants& pc);
    bool isReady() const { return m_textureN > 0; }

private:
    // 3D texture
    VkImage m_volumeImage = VK_NULL_HANDLE;
    VmaAllocation m_volumeAlloc = VK_NULL_HANDLE;
    VkImageView m_volumeView = VK_NULL_HANDLE;
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
    int m_textureN = 0;
    VmaAllocator m_allocator = VK_NULL_HANDLE;
};

} // namespace cosmico
