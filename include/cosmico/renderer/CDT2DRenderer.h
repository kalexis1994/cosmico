#pragma once
#include <volk.h>
#include <vk_mem_alloc.h>
#include <string>
#include <cstdint>

namespace cosmico {

class VkContext;

struct CDT2DRenderPushConstants {
    alignas(16) float viewProj[16]; // 64 bytes
    float meshScale;                // 4 bytes
    int showWireframe;              // 4 bytes
    float padding[2];              // 8 bytes -> total 80
};

class CDT2DRenderer {
public:
    void init(VkContext& ctx, VkRenderPass renderPass, const std::string& shaderDir);
    void destroy(VkDevice device, VmaAllocator allocator);

    // Upload vertex data (CPU -> mapped SSBO)
    void updateMesh(const float* vertexData, int triangleCount);

    void draw(VkCommandBuffer cmd, const CDT2DRenderPushConstants& pc);

private:
    // Host-visible SSBO (CPU_TO_GPU)
    VkBuffer m_ssbo = VK_NULL_HANDLE;
    VmaAllocation m_ssboAlloc = VK_NULL_HANDLE;
    void* m_ssboMapped = nullptr;
    VkDeviceSize m_ssboSize = 0;

    // Descriptors
    VkDescriptorSetLayout m_setLayout = VK_NULL_HANDLE;
    VkDescriptorPool m_descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet m_descriptorSet = VK_NULL_HANDLE;

    // Pipeline
    VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_pipeline = VK_NULL_HANDLE;

    int m_triangleCount = 0;
    VmaAllocator m_allocator = VK_NULL_HANDLE;
};

} // namespace cosmico
