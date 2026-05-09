#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <cosmico/renderer/OffscreenTarget.h>
#include <cosmico/vulkan/VkContext.h>

#include <stdexcept>
#include <array>
#include <algorithm>

// Forward-declare ImGui Vulkan helpers to avoid header conflicts
struct ImDrawData;
extern "C++" {
    VkDescriptorSet ImGui_ImplVulkan_AddTexture(VkSampler sampler, VkImageView image_view, VkImageLayout image_layout);
    void            ImGui_ImplVulkan_RemoveTexture(VkDescriptorSet descriptor_set);
}

namespace cosmico {

void OffscreenTarget::init(VkContext& ctx, VkFormat colorFmt, VkFormat depthFmt,
                            uint32_t w, uint32_t h) {
    m_colorFormat = colorFmt;
    m_depthFormat = depthFmt;
    m_extent = { std::max(w, 1u), std::max(h, 1u) };

    VkDevice device = ctx.device();

    // Render pass (color -> SHADER_READ_ONLY for ImGui sampling)
    VkAttachmentDescription colorAttachment{};
    colorAttachment.format = m_colorFormat;
    colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkAttachmentDescription depthAttachment{};
    depthAttachment.format = m_depthFormat;
    depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference colorRef{};
    colorRef.attachment = 0;
    colorRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkAttachmentReference depthRef{};
    depthRef.attachment = 1;
    depthRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorRef;
    subpass.pDepthStencilAttachment = &depthRef;

    // Two dependencies: rendering writes -> fragment shader reads
    std::array<VkSubpassDependency, 2> dependencies{};

    // External -> subpass 0: wait for previous frame's fragment sampling to finish
    dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[0].dstSubpass = 0;
    dependencies[0].srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    dependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                                    VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependencies[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    dependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                                     VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

    // Subpass 0 -> external: make color writes visible for fragment shader sampling
    dependencies[1].srcSubpass = 0;
    dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependencies[1].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    dependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    dependencies[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

    VkAttachmentDescription attachments[] = { colorAttachment, depthAttachment };

    VkRenderPassCreateInfo rpInfo{};
    rpInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rpInfo.attachmentCount = 2;
    rpInfo.pAttachments = attachments;
    rpInfo.subpassCount = 1;
    rpInfo.pSubpasses = &subpass;
    rpInfo.dependencyCount = static_cast<uint32_t>(dependencies.size());
    rpInfo.pDependencies = dependencies.data();

    if (vkCreateRenderPass(device, &rpInfo, nullptr, &m_renderPass) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create offscreen render pass");
    }

    // Sampler (shared across frames)
    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.maxLod = 1.0f;

    if (vkCreateSampler(device, &samplerInfo, nullptr, &m_sampler) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create offscreen sampler");
    }

    // Per-frame images, views, framebuffers, ImGui textures
    createResources(device, ctx.allocator());
}

void OffscreenTarget::createResources(VkDevice device, VmaAllocator allocator) {
    for (uint32_t i = 0; i < FRAME_COUNT; i++) {
        auto& f = m_frames[i];

        // Color image
        VkImageCreateInfo colorCI{};
        colorCI.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        colorCI.imageType = VK_IMAGE_TYPE_2D;
        colorCI.format = m_colorFormat;
        colorCI.extent = { m_extent.width, m_extent.height, 1 };
        colorCI.mipLevels = 1;
        colorCI.arrayLayers = 1;
        colorCI.samples = VK_SAMPLE_COUNT_1_BIT;
        colorCI.tiling = VK_IMAGE_TILING_OPTIMAL;
        colorCI.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        colorCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        colorCI.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        VmaAllocationCreateInfo allocCI{};
        allocCI.usage = VMA_MEMORY_USAGE_GPU_ONLY;

        if (vmaCreateImage(allocator, &colorCI, &allocCI,
                           &f.colorImage, &f.colorAlloc, nullptr) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create offscreen color image");
        }

        // Color view
        VkImageViewCreateInfo colorViewCI{};
        colorViewCI.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        colorViewCI.image = f.colorImage;
        colorViewCI.viewType = VK_IMAGE_VIEW_TYPE_2D;
        colorViewCI.format = m_colorFormat;
        colorViewCI.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        colorViewCI.subresourceRange.baseMipLevel = 0;
        colorViewCI.subresourceRange.levelCount = 1;
        colorViewCI.subresourceRange.baseArrayLayer = 0;
        colorViewCI.subresourceRange.layerCount = 1;

        if (vkCreateImageView(device, &colorViewCI, nullptr, &f.colorView) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create offscreen color view");
        }

        // Depth image
        VkImageCreateInfo depthCI{};
        depthCI.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        depthCI.imageType = VK_IMAGE_TYPE_2D;
        depthCI.format = m_depthFormat;
        depthCI.extent = { m_extent.width, m_extent.height, 1 };
        depthCI.mipLevels = 1;
        depthCI.arrayLayers = 1;
        depthCI.samples = VK_SAMPLE_COUNT_1_BIT;
        depthCI.tiling = VK_IMAGE_TILING_OPTIMAL;
        depthCI.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
        depthCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        depthCI.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        if (vmaCreateImage(allocator, &depthCI, &allocCI,
                           &f.depthImage, &f.depthAlloc, nullptr) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create offscreen depth image");
        }

        // Depth view
        VkImageViewCreateInfo depthViewCI{};
        depthViewCI.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        depthViewCI.image = f.depthImage;
        depthViewCI.viewType = VK_IMAGE_VIEW_TYPE_2D;
        depthViewCI.format = m_depthFormat;
        depthViewCI.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        depthViewCI.subresourceRange.baseMipLevel = 0;
        depthViewCI.subresourceRange.levelCount = 1;
        depthViewCI.subresourceRange.baseArrayLayer = 0;
        depthViewCI.subresourceRange.layerCount = 1;

        if (vkCreateImageView(device, &depthViewCI, nullptr, &f.depthView) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create offscreen depth view");
        }

        // Framebuffer
        VkImageView attachments[] = { f.colorView, f.depthView };
        VkFramebufferCreateInfo fbCI{};
        fbCI.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fbCI.renderPass = m_renderPass;
        fbCI.attachmentCount = 2;
        fbCI.pAttachments = attachments;
        fbCI.width = m_extent.width;
        fbCI.height = m_extent.height;
        fbCI.layers = 1;

        if (vkCreateFramebuffer(device, &fbCI, nullptr, &f.framebuffer) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create offscreen framebuffer");
        }

        // Register as ImGui texture
        f.imguiDescriptorSet = ImGui_ImplVulkan_AddTexture(
            m_sampler, f.colorView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }
}

void OffscreenTarget::destroyResources(VkDevice device, VmaAllocator allocator) {
    for (uint32_t i = 0; i < FRAME_COUNT; i++) {
        auto& f = m_frames[i];
        if (f.imguiDescriptorSet) {
            ImGui_ImplVulkan_RemoveTexture(f.imguiDescriptorSet);
            f.imguiDescriptorSet = VK_NULL_HANDLE;
        }
        if (f.framebuffer) {
            vkDestroyFramebuffer(device, f.framebuffer, nullptr);
            f.framebuffer = VK_NULL_HANDLE;
        }
        if (f.depthView) {
            vkDestroyImageView(device, f.depthView, nullptr);
            f.depthView = VK_NULL_HANDLE;
        }
        if (f.depthImage) {
            vmaDestroyImage(allocator, f.depthImage, f.depthAlloc);
            f.depthImage = VK_NULL_HANDLE;
            f.depthAlloc = VK_NULL_HANDLE;
        }
        if (f.colorView) {
            vkDestroyImageView(device, f.colorView, nullptr);
            f.colorView = VK_NULL_HANDLE;
        }
        if (f.colorImage) {
            vmaDestroyImage(allocator, f.colorImage, f.colorAlloc);
            f.colorImage = VK_NULL_HANDLE;
            f.colorAlloc = VK_NULL_HANDLE;
        }
    }
}

void OffscreenTarget::destroy(VkDevice device, VmaAllocator allocator) {
    destroyResources(device, allocator);

    if (m_sampler) {
        vkDestroySampler(device, m_sampler, nullptr);
        m_sampler = VK_NULL_HANDLE;
    }
    if (m_renderPass) {
        vkDestroyRenderPass(device, m_renderPass, nullptr);
        m_renderPass = VK_NULL_HANDLE;
    }
}

void OffscreenTarget::resize(VkContext& ctx, uint32_t w, uint32_t h) {
    m_extent = { std::max(w, 1u), std::max(h, 1u) };
    destroyResources(ctx.device(), ctx.allocator());
    createResources(ctx.device(), ctx.allocator());
}

void OffscreenTarget::beginRenderPass(VkCommandBuffer cmd, uint32_t frameIndex) {
    std::array<VkClearValue, 2> clearValues{};
    clearValues[0].color = {{0.01f, 0.01f, 0.02f, 1.0f}};
    clearValues[1].depthStencil = {1.0f, 0};

    VkRenderPassBeginInfo rpInfo{};
    rpInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpInfo.renderPass = m_renderPass;
    rpInfo.framebuffer = m_frames[frameIndex].framebuffer;
    rpInfo.renderArea.offset = {0, 0};
    rpInfo.renderArea.extent = m_extent;
    rpInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
    rpInfo.pClearValues = clearValues.data();

    vkCmdBeginRenderPass(cmd, &rpInfo, VK_SUBPASS_CONTENTS_INLINE);

    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(m_extent.width);
    viewport.height = static_cast<float>(m_extent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &viewport);

    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = m_extent;
    vkCmdSetScissor(cmd, 0, 1, &scissor);
}

void OffscreenTarget::endRenderPass(VkCommandBuffer cmd) {
    vkCmdEndRenderPass(cmd);
}

} // namespace cosmico
