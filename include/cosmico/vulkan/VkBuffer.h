#pragma once
#include <volk.h>
#include <vk_mem_alloc.h>
#include <cstdint>

#ifdef _WIN32
#include <windows.h>
#endif

namespace cosmico {

class VkContext;

class GpuBuffer {
public:
    void create(VmaAllocator allocator, VkDeviceSize size, VkBufferUsageFlags usage,
                VmaMemoryUsage memUsage, VmaAllocationCreateFlags flags = 0);

    // Create a buffer with exportable memory (for CUDA interop via Win32 handles)
    void createExportable(VkDevice device, VkPhysicalDevice physicalDevice,
                          VkDeviceSize size, VkBufferUsageFlags usage);

    void destroy(VmaAllocator allocator);
    void destroyExportable(VkDevice device);

    void upload(VmaAllocator allocator, const void* data, VkDeviceSize size);

    VkBuffer buffer() const { return m_buffer; }
    VkDeviceSize size() const { return m_size; }
    VkDeviceMemory memory() const { return m_memory; }
    bool isExportable() const { return m_exportable; }

    VkDescriptorBufferInfo descriptorInfo(VkDeviceSize offset = 0) const;

#ifdef _WIN32
    HANDLE getWin32Handle(VkDevice device) const;
#endif

private:
    uint32_t findMemoryType(VkPhysicalDevice physicalDevice, uint32_t typeFilter,
                             VkMemoryPropertyFlags properties);

    VkBuffer m_buffer = VK_NULL_HANDLE;
    VmaAllocation m_allocation = VK_NULL_HANDLE;  // Used for non-exportable
    VkDeviceMemory m_memory = VK_NULL_HANDLE;     // Used for exportable (manual alloc)
    VkDeviceSize m_size = 0;
    bool m_exportable = false;
};

} // namespace cosmico
