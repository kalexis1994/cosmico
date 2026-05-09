#include <cosmico/vulkan/VkCommands.h>
#include <stdexcept>

namespace cosmico {

void CommandPool::init(VkDevice device, uint32_t queueFamily, VkCommandPoolCreateFlags flags) {
    VkCommandPoolCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    info.queueFamilyIndex = queueFamily;
    info.flags = flags;

    if (vkCreateCommandPool(device, &info, nullptr, &m_pool) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create command pool");
    }
}

void CommandPool::destroy(VkDevice device) {
    if (m_pool) {
        vkDestroyCommandPool(device, m_pool, nullptr);
        m_pool = VK_NULL_HANDLE;
    }
}

VkCommandBuffer CommandPool::allocate(VkDevice device) {
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = m_pool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer cmd;
    if (vkAllocateCommandBuffers(device, &allocInfo, &cmd) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate command buffer");
    }
    return cmd;
}

void CommandPool::reset(VkDevice device) {
    vkResetCommandPool(device, m_pool, 0);
}

VkCommandBuffer beginSingleTimeCommands(VkDevice device, VkCommandPool pool) {
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = pool;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(device, &allocInfo, &cmd);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &beginInfo);

    return cmd;
}

void endSingleTimeCommands(VkDevice device, VkCommandPool pool, VkQueue queue, VkCommandBuffer cmd) {
    vkEndCommandBuffer(cmd);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;

    vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(queue);

    vkFreeCommandBuffers(device, pool, 1, &cmd);
}

} // namespace cosmico
