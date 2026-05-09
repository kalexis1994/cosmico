#pragma once
#include <volk.h>

#ifdef _WIN32
#include <windows.h>
#endif

namespace cosmico {

VkFence createFence(VkDevice device, bool signaled = true);
VkSemaphore createSemaphore(VkDevice device);

// Create a timeline semaphore with an initial value
VkSemaphore createTimelineSemaphore(VkDevice device, uint64_t initialValue = 0);

// Create a timeline semaphore exportable via Win32 handle (for CUDA interop)
VkSemaphore createExportableTimelineSemaphore(VkDevice device, uint64_t initialValue = 0);

#ifdef _WIN32
HANDLE getTimelineSemaphoreWin32Handle(VkDevice device, VkSemaphore semaphore);
#endif

void insertComputeToGraphicsBarrier(VkCommandBuffer cmd, VkBuffer buffer, VkDeviceSize size);

} // namespace cosmico
