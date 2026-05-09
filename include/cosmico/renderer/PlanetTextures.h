#pragma once
#include <volk.h>
#include <vk_mem_alloc.h>
#include <string>

namespace cosmico {

class VkContext;

class PlanetTextures {
public:
    bool init(VkContext& ctx, const std::string& textureDir);
    void destroy(VkDevice device, VmaAllocator allocator);

    VkDescriptorSetLayout descriptorSetLayout() const { return m_setLayout; }
    VkDescriptorSet descriptorSet() const { return m_descriptorSet; }

    static constexpr uint32_t LAYER_COUNT = 20;  // 19 planets + 1 ring texture
    static constexpr uint32_t TEX_WIDTH = 2048;
    static constexpr uint32_t TEX_HEIGHT = 1024;

private:
    VkImage m_image = VK_NULL_HANDLE;
    VmaAllocation m_alloc = VK_NULL_HANDLE;
    VkImageView m_view = VK_NULL_HANDLE;
    VkSampler m_sampler = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_setLayout = VK_NULL_HANDLE;
    VkDescriptorPool m_descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet m_descriptorSet = VK_NULL_HANDLE;
};

} // namespace cosmico
