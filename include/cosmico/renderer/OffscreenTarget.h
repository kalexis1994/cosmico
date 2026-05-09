#pragma once
#include <volk.h>
#include <vk_mem_alloc.h>
#include <cstdint>

namespace cosmico {

class VkContext;

class OffscreenTarget {
public:
    void init(VkContext& ctx, VkFormat colorFmt, VkFormat depthFmt, uint32_t w, uint32_t h);
    void destroy(VkDevice device, VmaAllocator allocator);
    void resize(VkContext& ctx, uint32_t w, uint32_t h);

    VkRenderPass renderPass() const { return m_renderPass; }
    VkExtent2D extent() const { return m_extent; }
    VkFramebuffer framebuffer(uint32_t frameIndex) const { return m_frames[frameIndex].framebuffer; }
    VkDescriptorSet imguiTexture(uint32_t frameIndex) const { return m_frames[frameIndex].imguiDescriptorSet; }

    void beginRenderPass(VkCommandBuffer cmd, uint32_t frameIndex);
    void endRenderPass(VkCommandBuffer cmd);

private:
    void createResources(VkDevice device, VmaAllocator allocator);
    void destroyResources(VkDevice device, VmaAllocator allocator);

    VkRenderPass m_renderPass = VK_NULL_HANDLE;
    VkSampler m_sampler = VK_NULL_HANDLE;
    VkFormat m_colorFormat{};
    VkFormat m_depthFormat{};
    VkExtent2D m_extent{};

    struct PerFrame {
        VkImage colorImage = VK_NULL_HANDLE;
        VmaAllocation colorAlloc = VK_NULL_HANDLE;
        VkImageView colorView = VK_NULL_HANDLE;
        VkImage depthImage = VK_NULL_HANDLE;
        VmaAllocation depthAlloc = VK_NULL_HANDLE;
        VkImageView depthView = VK_NULL_HANDLE;
        VkFramebuffer framebuffer = VK_NULL_HANDLE;
        VkDescriptorSet imguiDescriptorSet = VK_NULL_HANDLE;
    };
    static constexpr uint32_t FRAME_COUNT = 2;
    PerFrame m_frames[FRAME_COUNT]{};
};

} // namespace cosmico
