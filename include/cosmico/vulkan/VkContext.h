#pragma once
#include <volk.h>
#include <vk_mem_alloc.h>
#include <cstdint>

struct GLFWwindow;

namespace cosmico {

struct QueueFamilyIndices {
    uint32_t graphicsFamily = UINT32_MAX;
    uint32_t presentFamily = UINT32_MAX;
    bool isComplete() const { return graphicsFamily != UINT32_MAX && presentFamily != UINT32_MAX; }
};

class VkContext {
public:
    void init(GLFWwindow* window);
    void destroy();

    VkInstance instance() const { return m_instance; }
    VkPhysicalDevice physicalDevice() const { return m_physicalDevice; }
    VkDevice device() const { return m_device; }
    VkQueue graphicsQueue() const { return m_graphicsQueue; }
    VkQueue presentQueue() const { return m_presentQueue; }
    VkSurfaceKHR surface() const { return m_surface; }
    VmaAllocator allocator() const { return m_allocator; }
    QueueFamilyIndices queueFamilies() const { return m_queueFamilies; }
    bool externalMemorySupported() const { return m_externalMemorySupported; }

private:
    void createInstance();
    void setupDebugMessenger();
    void createSurface(GLFWwindow* window);
    void pickPhysicalDevice();
    void createLogicalDevice();
    void createAllocator();

    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device);
    bool isDeviceSuitable(VkPhysicalDevice device);
    bool checkExternalMemorySupport(VkPhysicalDevice device);

    VkInstance m_instance = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT m_debugMessenger = VK_NULL_HANDLE;
    VkSurfaceKHR m_surface = VK_NULL_HANDLE;
    VkPhysicalDevice m_physicalDevice = VK_NULL_HANDLE;
    VkDevice m_device = VK_NULL_HANDLE;
    VkQueue m_graphicsQueue = VK_NULL_HANDLE;
    VkQueue m_presentQueue = VK_NULL_HANDLE;
    VmaAllocator m_allocator = VK_NULL_HANDLE;
    QueueFamilyIndices m_queueFamilies;
    bool m_externalMemorySupported = false;

#ifdef NDEBUG
    static constexpr bool ENABLE_VALIDATION = false;
#else
    static constexpr bool ENABLE_VALIDATION = true;
#endif
};

} // namespace cosmico
