#include <cosmico/vulkan/VkBuffer.h>
#include <stdexcept>
#include <cstring>

#ifdef _WIN32
#ifndef DXGI_SHARED_RESOURCE_READ
#define DXGI_SHARED_RESOURCE_READ  0x80000000L
#define DXGI_SHARED_RESOURCE_WRITE 0x00000001L
#endif
#endif

namespace cosmico {

void GpuBuffer::create(VmaAllocator allocator, VkDeviceSize size, VkBufferUsageFlags usage,
                        VmaMemoryUsage memUsage, VmaAllocationCreateFlags flags) {
    m_size = size;
    m_exportable = false;

    VkBufferCreateInfo bufInfo{};
    bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufInfo.size = size;
    bufInfo.usage = usage;
    bufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = memUsage;
    allocInfo.flags = flags;

    if (vmaCreateBuffer(allocator, &bufInfo, &allocInfo, &m_buffer, &m_allocation, nullptr) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create GPU buffer");
    }
}

void GpuBuffer::createExportable(VkDevice device, VkPhysicalDevice physicalDevice,
                                   VkDeviceSize size, VkBufferUsageFlags usage) {
    m_size = size;
    m_exportable = true;

    // Create buffer with external memory info
    VkExternalMemoryBufferCreateInfo extMemBufInfo{};
    extMemBufInfo.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
    extMemBufInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;

    VkBufferCreateInfo bufInfo{};
    bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufInfo.pNext = &extMemBufInfo;
    bufInfo.size = size;
    bufInfo.usage = usage;
    bufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(device, &bufInfo, nullptr, &m_buffer) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create exportable buffer");
    }

    // Get memory requirements
    VkMemoryRequirements memReqs;
    vkGetBufferMemoryRequirements(device, m_buffer, &memReqs);

    // Dedicated allocation is required for CUDA interop
    VkMemoryDedicatedAllocateInfo dedicatedInfo{};
    dedicatedInfo.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
    dedicatedInfo.buffer = m_buffer;

    // Export memory allocate info (Win32 handle)
    VkExportMemoryAllocateInfo exportInfo{};
    exportInfo.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
    exportInfo.pNext = &dedicatedInfo;
    exportInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;

    // Win32 export info for named handle (needed for CUDA)
    VkExportMemoryWin32HandleInfoKHR exportWin32Info{};
    exportWin32Info.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_WIN32_HANDLE_INFO_KHR;
    exportWin32Info.pNext = &exportInfo;
    exportWin32Info.dwAccess = DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE;
    exportWin32Info.pAttributes = nullptr;
    exportWin32Info.name = nullptr;

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.pNext = &exportWin32Info;
    allocInfo.allocationSize = memReqs.size;
    allocInfo.memoryTypeIndex = findMemoryType(physicalDevice, memReqs.memoryTypeBits,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    if (vkAllocateMemory(device, &allocInfo, nullptr, &m_memory) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate exportable memory");
    }

    if (vkBindBufferMemory(device, m_buffer, m_memory, 0) != VK_SUCCESS) {
        throw std::runtime_error("Failed to bind exportable buffer memory");
    }
}

void GpuBuffer::destroy(VmaAllocator allocator) {
    if (m_buffer && !m_exportable) {
        vmaDestroyBuffer(allocator, m_buffer, m_allocation);
        m_buffer = VK_NULL_HANDLE;
        m_allocation = VK_NULL_HANDLE;
    }
}

void GpuBuffer::destroyExportable(VkDevice device) {
    if (m_buffer && m_exportable) {
        vkDestroyBuffer(device, m_buffer, nullptr);
        vkFreeMemory(device, m_memory, nullptr);
        m_buffer = VK_NULL_HANDLE;
        m_memory = VK_NULL_HANDLE;
    }
}

void GpuBuffer::upload(VmaAllocator allocator, const void* data, VkDeviceSize size) {
    void* mapped = nullptr;
    vmaMapMemory(allocator, m_allocation, &mapped);
    memcpy(mapped, data, static_cast<size_t>(size));
    vmaUnmapMemory(allocator, m_allocation);
    vmaFlushAllocation(allocator, m_allocation, 0, size);
}

VkDescriptorBufferInfo GpuBuffer::descriptorInfo(VkDeviceSize offset) const {
    VkDescriptorBufferInfo info{};
    info.buffer = m_buffer;
    info.offset = offset;
    info.range = m_size;
    return info;
}

uint32_t GpuBuffer::findMemoryType(VkPhysicalDevice physicalDevice, uint32_t typeFilter,
                                     VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProps);

    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) &&
            (memProps.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    throw std::runtime_error("Failed to find suitable memory type for exportable buffer");
}

#ifdef _WIN32
HANDLE GpuBuffer::getWin32Handle(VkDevice device) const {
    VkMemoryGetWin32HandleInfoKHR handleInfo{};
    handleInfo.sType = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
    handleInfo.memory = m_memory;
    handleInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;

    HANDLE handle = nullptr;
    if (vkGetMemoryWin32HandleKHR(device, &handleInfo, &handle) != VK_SUCCESS) {
        throw std::runtime_error("Failed to get Win32 handle for buffer memory");
    }
    return handle;
}
#endif

} // namespace cosmico
