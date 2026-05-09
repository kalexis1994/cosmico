#pragma once
#include <volk.h>
#include <vector>
#include <cstdint>
#include <functional>

namespace cosmico {

class VkContext;
class VkSwapchain;

static constexpr uint32_t MAX_FRAMES_IN_FLIGHT = 2;

class Renderer {
public:
    void init(VkContext& ctx, VkSwapchain& swapchain);
    void destroy(VkDevice device);
    void recreateFramebuffers(VkContext& ctx, VkSwapchain& swapchain);

    // Returns false if swapchain needs recreation
    bool beginFrame(VkContext& ctx, VkSwapchain& swapchain, uint32_t& imageIndex);
    void endFrame(VkContext& ctx, VkSwapchain& swapchain, uint32_t imageIndex);

    // endFrame variant that waits on a timeline semaphore (for CUDA interop)
    void endFrameWithTimelineWait(VkContext& ctx, VkSwapchain& swapchain, uint32_t imageIndex,
                                   VkSemaphore timelineSem, uint64_t waitValue, uint64_t signalValue);

    VkRenderPass renderPass() const { return m_renderPass; }
    VkCommandBuffer currentCommandBuffer() const { return m_commandBuffers[m_currentFrame]; }
    uint32_t currentFrame() const { return m_currentFrame; }

    void beginRenderPass(VkSwapchain& swapchain, uint32_t imageIndex);
    void endRenderPass();

private:
    void createRenderPass(VkContext& ctx, VkSwapchain& swapchain);
    void createFramebuffers(VkSwapchain& swapchain, VkDevice device);
    void createCommandResources(VkContext& ctx);
    void createSyncObjects(VkDevice device);

    VkRenderPass m_renderPass = VK_NULL_HANDLE;
    std::vector<VkFramebuffer> m_framebuffers;

    VkCommandPool m_commandPool = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> m_commandBuffers;

    std::vector<VkSemaphore> m_imageAvailableSemaphores; // Per swapchain image
    std::vector<VkSemaphore> m_renderFinishedSemaphores; // Per frame in flight
    std::vector<VkFence> m_inFlightFences;               // Per frame in flight

    uint32_t m_currentFrame = 0;
    uint32_t m_acquireSemaphoreIndex = 0;
    VkSemaphore m_currentAcquireSemaphore = VK_NULL_HANDLE;
};

} // namespace cosmico
