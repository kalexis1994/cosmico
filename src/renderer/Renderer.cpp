#include <cosmico/renderer/Renderer.h>
#include <cosmico/vulkan/VkContext.h>
#include <cosmico/vulkan/VkSwapchain.h>
#include <cosmico/vulkan/VkSync.h>
#include <stdexcept>
#include <array>

namespace cosmico {

void Renderer::init(VkContext& ctx, VkSwapchain& swapchain) {
    createRenderPass(ctx, swapchain);
    createFramebuffers(swapchain, ctx.device());
    createCommandResources(ctx);
    createSyncObjects(ctx.device());
}

void Renderer::destroy(VkDevice device) {
    for (auto sem : m_imageAvailableSemaphores) {
        vkDestroySemaphore(device, sem, nullptr);
    }
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        vkDestroySemaphore(device, m_renderFinishedSemaphores[i], nullptr);
        vkDestroyFence(device, m_inFlightFences[i], nullptr);
    }
    vkDestroyCommandPool(device, m_commandPool, nullptr);
    for (auto fb : m_framebuffers) {
        vkDestroyFramebuffer(device, fb, nullptr);
    }
    vkDestroyRenderPass(device, m_renderPass, nullptr);
}

void Renderer::recreateFramebuffers(VkContext& ctx, VkSwapchain& swapchain) {
    for (auto fb : m_framebuffers) {
        vkDestroyFramebuffer(ctx.device(), fb, nullptr);
    }
    vkDestroyRenderPass(ctx.device(), m_renderPass, nullptr);
    createRenderPass(ctx, swapchain);
    createFramebuffers(swapchain, ctx.device());
}

bool Renderer::beginFrame(VkContext& ctx, VkSwapchain& swapchain, uint32_t& imageIndex) {
    vkWaitForFences(ctx.device(), 1, &m_inFlightFences[m_currentFrame], VK_TRUE, UINT64_MAX);

    // Cycle through acquire semaphores (one per swapchain image)
    m_currentAcquireSemaphore = m_imageAvailableSemaphores[m_acquireSemaphoreIndex];
    VkResult result = vkAcquireNextImageKHR(ctx.device(), swapchain.handle(), UINT64_MAX,
        m_currentAcquireSemaphore, VK_NULL_HANDLE, &imageIndex);
    m_acquireSemaphoreIndex = (m_acquireSemaphoreIndex + 1) %
        static_cast<uint32_t>(m_imageAvailableSemaphores.size());

    if (result == VK_ERROR_OUT_OF_DATE_KHR) return false;
    if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
        throw std::runtime_error("Failed to acquire swapchain image");
    }

    vkResetFences(ctx.device(), 1, &m_inFlightFences[m_currentFrame]);

    VkCommandBuffer cmd = m_commandBuffers[m_currentFrame];
    vkResetCommandBuffer(cmd, 0);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &beginInfo);

    return true;
}

void Renderer::endFrame(VkContext& ctx, VkSwapchain& swapchain, uint32_t imageIndex) {
    VkCommandBuffer cmd = m_commandBuffers[m_currentFrame];
    vkEndCommandBuffer(cmd);

    VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = &m_currentAcquireSemaphore;
    submitInfo.pWaitDstStageMask = &waitStage;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = &m_renderFinishedSemaphores[m_currentFrame];

    if (vkQueueSubmit(ctx.graphicsQueue(), 1, &submitInfo, m_inFlightFences[m_currentFrame]) != VK_SUCCESS) {
        throw std::runtime_error("Failed to submit draw command buffer");
    }

    VkPresentInfoKHR presentInfo{};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = &m_renderFinishedSemaphores[m_currentFrame];
    VkSwapchainKHR swapchains[] = { swapchain.handle() };
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = swapchains;
    presentInfo.pImageIndices = &imageIndex;

    vkQueuePresentKHR(ctx.presentQueue(), &presentInfo);

    m_currentFrame = (m_currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
}

void Renderer::endFrameWithTimelineWait(VkContext& ctx, VkSwapchain& swapchain, uint32_t imageIndex,
                                          VkSemaphore timelineSem, uint64_t waitValue, uint64_t signalValue) {
    VkCommandBuffer cmd = m_commandBuffers[m_currentFrame];
    vkEndCommandBuffer(cmd);

    // We need to wait on both:
    // 1. Image available (binary semaphore from acquire)
    // 2. CUDA compute done (timeline semaphore)
    VkSemaphore waitSemaphores[] = { m_currentAcquireSemaphore, timelineSem };
    VkPipelineStageFlags waitStages[] = {
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_PIPELINE_STAGE_VERTEX_INPUT_BIT  // Wait for CUDA to finish before reading vertices
    };
    uint64_t waitValues[] = { 0, waitValue }; // Binary semaphore value is ignored

    // Signal both the render-finished binary semaphore AND the timeline semaphore
    VkSemaphore signalSemaphores[] = { m_renderFinishedSemaphores[m_currentFrame], timelineSem };
    uint64_t signalValues[] = { 0, signalValue }; // Binary semaphore value is ignored

    VkTimelineSemaphoreSubmitInfo timelineInfo{};
    timelineInfo.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
    timelineInfo.waitSemaphoreValueCount = 2;
    timelineInfo.pWaitSemaphoreValues = waitValues;
    timelineInfo.signalSemaphoreValueCount = 2;
    timelineInfo.pSignalSemaphoreValues = signalValues;

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.pNext = &timelineInfo;
    submitInfo.waitSemaphoreCount = 2;
    submitInfo.pWaitSemaphores = waitSemaphores;
    submitInfo.pWaitDstStageMask = waitStages;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;
    submitInfo.signalSemaphoreCount = 2;
    submitInfo.pSignalSemaphores = signalSemaphores;

    if (vkQueueSubmit(ctx.graphicsQueue(), 1, &submitInfo, m_inFlightFences[m_currentFrame]) != VK_SUCCESS) {
        throw std::runtime_error("Failed to submit draw command buffer with timeline wait");
    }

    VkPresentInfoKHR presentInfo{};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = &m_renderFinishedSemaphores[m_currentFrame];
    VkSwapchainKHR swapchains[] = { swapchain.handle() };
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = swapchains;
    presentInfo.pImageIndices = &imageIndex;

    vkQueuePresentKHR(ctx.presentQueue(), &presentInfo);

    m_currentFrame = (m_currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
}

void Renderer::beginRenderPass(VkSwapchain& swapchain, uint32_t imageIndex) {
    VkCommandBuffer cmd = m_commandBuffers[m_currentFrame];

    std::array<VkClearValue, 2> clearValues{};
    clearValues[0].color = {{0.01f, 0.01f, 0.02f, 1.0f}}; // Dark space background
    clearValues[1].depthStencil = {1.0f, 0};

    VkRenderPassBeginInfo rpInfo{};
    rpInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpInfo.renderPass = m_renderPass;
    rpInfo.framebuffer = m_framebuffers[imageIndex];
    rpInfo.renderArea.offset = {0, 0};
    rpInfo.renderArea.extent = swapchain.extent();
    rpInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
    rpInfo.pClearValues = clearValues.data();

    vkCmdBeginRenderPass(cmd, &rpInfo, VK_SUBPASS_CONTENTS_INLINE);

    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(swapchain.extent().width);
    viewport.height = static_cast<float>(swapchain.extent().height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &viewport);

    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = swapchain.extent();
    vkCmdSetScissor(cmd, 0, 1, &scissor);
}

void Renderer::endRenderPass() {
    vkCmdEndRenderPass(m_commandBuffers[m_currentFrame]);
}

void Renderer::createRenderPass(VkContext& ctx, VkSwapchain& swapchain) {
    VkAttachmentDescription colorAttachment{};
    colorAttachment.format = swapchain.imageFormat();
    colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentDescription depthAttachment{};
    depthAttachment.format = swapchain.depthFormat();
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

    VkSubpassDependency dependency{};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                              VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependency.srcAccessMask = 0;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                              VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                               VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

    VkAttachmentDescription attachments[] = { colorAttachment, depthAttachment };

    VkRenderPassCreateInfo rpInfo{};
    rpInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rpInfo.attachmentCount = 2;
    rpInfo.pAttachments = attachments;
    rpInfo.subpassCount = 1;
    rpInfo.pSubpasses = &subpass;
    rpInfo.dependencyCount = 1;
    rpInfo.pDependencies = &dependency;

    if (vkCreateRenderPass(ctx.device(), &rpInfo, nullptr, &m_renderPass) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create render pass");
    }
}

void Renderer::createFramebuffers(VkSwapchain& swapchain, VkDevice device) {
    m_framebuffers.resize(swapchain.imageCount());
    for (uint32_t i = 0; i < swapchain.imageCount(); i++) {
        VkImageView attachments[] = { swapchain.imageView(i), swapchain.depthImageView() };

        VkFramebufferCreateInfo fbInfo{};
        fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fbInfo.renderPass = m_renderPass;
        fbInfo.attachmentCount = 2;
        fbInfo.pAttachments = attachments;
        fbInfo.width = swapchain.extent().width;
        fbInfo.height = swapchain.extent().height;
        fbInfo.layers = 1;

        if (vkCreateFramebuffer(device, &fbInfo, nullptr, &m_framebuffers[i]) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create framebuffer");
        }
    }
}

void Renderer::createCommandResources(VkContext& ctx) {
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = ctx.queueFamilies().graphicsFamily;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    if (vkCreateCommandPool(ctx.device(), &poolInfo, nullptr, &m_commandPool) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create command pool");
    }

    m_commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = m_commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = MAX_FRAMES_IN_FLIGHT;

    if (vkAllocateCommandBuffers(ctx.device(), &allocInfo, m_commandBuffers.data()) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate command buffers");
    }
}

void Renderer::createSyncObjects(VkDevice device) {
    // One image-available semaphore per swapchain image to avoid reuse conflicts
    uint32_t semCount = static_cast<uint32_t>(m_framebuffers.size());
    m_imageAvailableSemaphores.resize(semCount);
    for (uint32_t i = 0; i < semCount; i++) {
        m_imageAvailableSemaphores[i] = createSemaphore(device);
    }
    m_acquireSemaphoreIndex = 0;

    m_renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
    m_inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        m_renderFinishedSemaphores[i] = createSemaphore(device);
        m_inFlightFences[i] = createFence(device, true);
    }
}

} // namespace cosmico
