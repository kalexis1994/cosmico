#include <cosmico/vulkan/VkSwapchain.h>
#include <cosmico/vulkan/VkContext.h>
#include <vk_mem_alloc.h>
#include <algorithm>
#include <stdexcept>

namespace cosmico {

void VkSwapchain::init(VkContext& ctx, uint32_t width, uint32_t height) {
    create(ctx, width, height);
}

void VkSwapchain::destroy(VkDevice device, VmaAllocator allocator) {
    cleanup(device, allocator);
}

void VkSwapchain::recreate(VkContext& ctx, uint32_t width, uint32_t height) {
    vkDeviceWaitIdle(ctx.device());
    cleanup(ctx.device(), ctx.allocator());
    create(ctx, width, height);
}

void VkSwapchain::create(VkContext& ctx, uint32_t width, uint32_t height) {
    VkSurfaceCapabilitiesKHR caps;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(ctx.physicalDevice(), ctx.surface(), &caps);

    uint32_t formatCount;
    vkGetPhysicalDeviceSurfaceFormatsKHR(ctx.physicalDevice(), ctx.surface(), &formatCount, nullptr);
    std::vector<VkSurfaceFormatKHR> formats(formatCount);
    vkGetPhysicalDeviceSurfaceFormatsKHR(ctx.physicalDevice(), ctx.surface(), &formatCount, formats.data());

    uint32_t modeCount;
    vkGetPhysicalDeviceSurfacePresentModesKHR(ctx.physicalDevice(), ctx.surface(), &modeCount, nullptr);
    std::vector<VkPresentModeKHR> modes(modeCount);
    vkGetPhysicalDeviceSurfacePresentModesKHR(ctx.physicalDevice(), ctx.surface(), &modeCount, modes.data());

    auto surfaceFormat = chooseSurfaceFormat(formats);
    auto presentMode = choosePresentMode(modes);
    m_extent = chooseExtent(caps, width, height);
    m_format = surfaceFormat.format;

    uint32_t imageCount = caps.minImageCount + 1;
    if (caps.maxImageCount > 0 && imageCount > caps.maxImageCount) {
        imageCount = caps.maxImageCount;
    }

    VkSwapchainCreateInfoKHR createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    createInfo.surface = ctx.surface();
    createInfo.minImageCount = imageCount;
    createInfo.imageFormat = m_format;
    createInfo.imageColorSpace = surfaceFormat.colorSpace;
    createInfo.imageExtent = m_extent;
    createInfo.imageArrayLayers = 1;
    createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    auto families = ctx.queueFamilies();
    uint32_t familyIndices[] = { families.graphicsFamily, families.presentFamily };
    if (families.graphicsFamily != families.presentFamily) {
        createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        createInfo.queueFamilyIndexCount = 2;
        createInfo.pQueueFamilyIndices = familyIndices;
    } else {
        createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }

    createInfo.preTransform = caps.currentTransform;
    createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    createInfo.presentMode = presentMode;
    createInfo.clipped = VK_TRUE;

    if (vkCreateSwapchainKHR(ctx.device(), &createInfo, nullptr, &m_swapchain) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create swapchain");
    }

    uint32_t swapImageCount;
    vkGetSwapchainImagesKHR(ctx.device(), m_swapchain, &swapImageCount, nullptr);
    m_images.resize(swapImageCount);
    vkGetSwapchainImagesKHR(ctx.device(), m_swapchain, &swapImageCount, m_images.data());

    createImageViews(ctx.device());
    createDepthResources(ctx);
}

void VkSwapchain::createImageViews(VkDevice device) {
    m_imageViews.resize(m_images.size());
    for (size_t i = 0; i < m_images.size(); i++) {
        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = m_images[i];
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = m_format;
        viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = 1;

        if (vkCreateImageView(device, &viewInfo, nullptr, &m_imageViews[i]) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create swapchain image view");
        }
    }
}

void VkSwapchain::createDepthResources(VkContext& ctx) {
    m_depthFormat = findDepthFormat(ctx.physicalDevice());

    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.format = m_depthFormat;
    imageInfo.extent = { m_extent.width, m_extent.height, 1 };
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;

    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

    if (vmaCreateImage(ctx.allocator(), &imageInfo, &allocInfo,
                       &m_depthImage, &m_depthAllocation, nullptr) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create depth image");
    }

    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = m_depthImage;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = m_depthFormat;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    if (vkCreateImageView(ctx.device(), &viewInfo, nullptr, &m_depthImageView) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create depth image view");
    }
}

void VkSwapchain::cleanup(VkDevice device, VmaAllocator allocator) {
    if (m_depthImageView) {
        vkDestroyImageView(device, m_depthImageView, nullptr);
        m_depthImageView = VK_NULL_HANDLE;
    }
    if (m_depthImage && allocator) {
        vmaDestroyImage(allocator, m_depthImage, m_depthAllocation);
        m_depthImage = VK_NULL_HANDLE;
        m_depthAllocation = VK_NULL_HANDLE;
    }

    for (auto view : m_imageViews) {
        vkDestroyImageView(device, view, nullptr);
    }
    m_imageViews.clear();

    if (m_swapchain) {
        vkDestroySwapchainKHR(device, m_swapchain, nullptr);
        m_swapchain = VK_NULL_HANDLE;
    }
}

VkSurfaceFormatKHR VkSwapchain::chooseSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& formats) {
    for (const auto& f : formats) {
        if (f.format == VK_FORMAT_B8G8R8A8_SRGB && f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            return f;
        }
    }
    return formats[0];
}

VkPresentModeKHR VkSwapchain::choosePresentMode(const std::vector<VkPresentModeKHR>& modes) {
    for (auto mode : modes) {
        if (mode == VK_PRESENT_MODE_MAILBOX_KHR) return mode;
    }
    return VK_PRESENT_MODE_FIFO_KHR;
}

VkExtent2D VkSwapchain::chooseExtent(const VkSurfaceCapabilitiesKHR& caps, uint32_t w, uint32_t h) {
    if (caps.currentExtent.width != UINT32_MAX) {
        return caps.currentExtent;
    }
    VkExtent2D ext = { w, h };
    ext.width = std::clamp(ext.width, caps.minImageExtent.width, caps.maxImageExtent.width);
    ext.height = std::clamp(ext.height, caps.minImageExtent.height, caps.maxImageExtent.height);
    return ext;
}

VkFormat VkSwapchain::findDepthFormat(VkPhysicalDevice physDevice) {
    VkFormat candidates[] = { VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT };
    for (auto format : candidates) {
        VkFormatProperties props;
        vkGetPhysicalDeviceFormatProperties(physDevice, format, &props);
        if (props.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) {
            return format;
        }
    }
    throw std::runtime_error("Failed to find suitable depth format");
}

} // namespace cosmico
