#include <cosmico/vulkan/VkSync.h>
#include <stdexcept>

#ifdef _WIN32
#ifndef DXGI_SHARED_RESOURCE_READ
#define DXGI_SHARED_RESOURCE_READ  0x80000000L
#define DXGI_SHARED_RESOURCE_WRITE 0x00000001L
#endif
#endif

namespace cosmico {

VkFence createFence(VkDevice device, bool signaled) {
    VkFenceCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    if (signaled) info.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    VkFence fence;
    if (vkCreateFence(device, &info, nullptr, &fence) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create fence");
    }
    return fence;
}

VkSemaphore createSemaphore(VkDevice device) {
    VkSemaphoreCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkSemaphore semaphore;
    if (vkCreateSemaphore(device, &info, nullptr, &semaphore) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create semaphore");
    }
    return semaphore;
}

VkSemaphore createTimelineSemaphore(VkDevice device, uint64_t initialValue) {
    VkSemaphoreTypeCreateInfo typeInfo{};
    typeInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
    typeInfo.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    typeInfo.initialValue = initialValue;

    VkSemaphoreCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    info.pNext = &typeInfo;

    VkSemaphore semaphore;
    if (vkCreateSemaphore(device, &info, nullptr, &semaphore) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create timeline semaphore");
    }
    return semaphore;
}

VkSemaphore createExportableTimelineSemaphore(VkDevice device, uint64_t initialValue) {
    VkSemaphoreTypeCreateInfo typeInfo{};
    typeInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
    typeInfo.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    typeInfo.initialValue = initialValue;

    VkExportSemaphoreCreateInfo exportInfo{};
    exportInfo.sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO;
    exportInfo.pNext = &typeInfo;
    exportInfo.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;

    VkExportSemaphoreWin32HandleInfoKHR exportWin32Info{};
    exportWin32Info.sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_WIN32_HANDLE_INFO_KHR;
    exportWin32Info.pNext = &exportInfo;
    exportWin32Info.dwAccess = DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE;
    exportWin32Info.pAttributes = nullptr;
    exportWin32Info.name = nullptr;

    VkSemaphoreCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    info.pNext = &exportWin32Info;

    VkSemaphore semaphore;
    if (vkCreateSemaphore(device, &info, nullptr, &semaphore) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create exportable timeline semaphore");
    }
    return semaphore;
}

#ifdef _WIN32
HANDLE getTimelineSemaphoreWin32Handle(VkDevice device, VkSemaphore semaphore) {
    VkSemaphoreGetWin32HandleInfoKHR handleInfo{};
    handleInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR;
    handleInfo.semaphore = semaphore;
    handleInfo.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;

    HANDLE handle = nullptr;
    if (vkGetSemaphoreWin32HandleKHR(device, &handleInfo, &handle) != VK_SUCCESS) {
        throw std::runtime_error("Failed to get Win32 handle for timeline semaphore");
    }
    return handle;
}
#endif

void insertComputeToGraphicsBarrier(VkCommandBuffer cmd, VkBuffer buffer, VkDeviceSize size) {
    VkBufferMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = buffer;
    barrier.offset = 0;
    barrier.size = size;

    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
        0,
        0, nullptr,
        1, &barrier,
        0, nullptr);
}

} // namespace cosmico
