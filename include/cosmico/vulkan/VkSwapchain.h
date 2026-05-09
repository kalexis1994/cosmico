#pragma once
#include <volk.h>
#include <vk_mem_alloc.h>
#include <vector>
#include <cstdint>

namespace cosmico {

class VkContext;

class VkSwapchain {
public:
    void init(VkContext& ctx, uint32_t width, uint32_t height);
    void destroy(VkDevice device, VmaAllocator allocator);
    void recreate(VkContext& ctx, uint32_t width, uint32_t height);

    VkSwapchainKHR handle() const { return m_swapchain; }
    VkFormat imageFormat() const { return m_format; }
    VkExtent2D extent() const { return m_extent; }
    uint32_t imageCount() const { return static_cast<uint32_t>(m_imageViews.size()); }
    VkImageView imageView(uint32_t i) const { return m_imageViews[i]; }

    VkFormat depthFormat() const { return m_depthFormat; }
    VkImageView depthImageView() const { return m_depthImageView; }

private:
    void create(VkContext& ctx, uint32_t width, uint32_t height);
    void createImageViews(VkDevice device);
    void createDepthResources(VkContext& ctx);
    void cleanup(VkDevice device, VmaAllocator allocator);

    VkSurfaceFormatKHR chooseSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& formats);
    VkPresentModeKHR choosePresentMode(const std::vector<VkPresentModeKHR>& modes);
    VkExtent2D chooseExtent(const VkSurfaceCapabilitiesKHR& caps, uint32_t w, uint32_t h);
    VkFormat findDepthFormat(VkPhysicalDevice physDevice);

    VkSwapchainKHR m_swapchain = VK_NULL_HANDLE;
    VkFormat m_format{};
    VkExtent2D m_extent{};
    std::vector<VkImage> m_images;
    std::vector<VkImageView> m_imageViews;

    VkFormat m_depthFormat{};
    VkImage m_depthImage = VK_NULL_HANDLE;
    VkImageView m_depthImageView = VK_NULL_HANDLE;
    VmaAllocation m_depthAllocation = VK_NULL_HANDLE;
};

} // namespace cosmico
