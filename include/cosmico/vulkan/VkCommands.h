#pragma once
#include <volk.h>
#include <cstdint>

namespace cosmico {

class CommandPool {
public:
    void init(VkDevice device, uint32_t queueFamily, VkCommandPoolCreateFlags flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);
    void destroy(VkDevice device);

    VkCommandBuffer allocate(VkDevice device);
    void reset(VkDevice device);

    VkCommandPool handle() const { return m_pool; }

private:
    VkCommandPool m_pool = VK_NULL_HANDLE;
};

VkCommandBuffer beginSingleTimeCommands(VkDevice device, VkCommandPool pool);
void endSingleTimeCommands(VkDevice device, VkCommandPool pool, VkQueue queue, VkCommandBuffer cmd);

} // namespace cosmico
