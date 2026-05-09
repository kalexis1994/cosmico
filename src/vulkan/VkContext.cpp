#include <cosmico/vulkan/VkContext.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <stdexcept>
#include <vector>
#include <cstring>
#include <cstdio>

namespace cosmico {

static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT /*type*/,
    const VkDebugUtilsMessengerCallbackDataEXT* callbackData,
    void* /*userData*/) {
    if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        fprintf(stderr, "[Vulkan] %s\n", callbackData->pMessage);
    }
    return VK_FALSE;
}

void VkContext::init(GLFWwindow* window) {
    VkResult res = volkInitialize();
    if (res != VK_SUCCESS) {
        throw std::runtime_error("Failed to initialize Volk (Vulkan loader)");
    }

    createInstance();
    if (ENABLE_VALIDATION) setupDebugMessenger();
    createSurface(window);
    pickPhysicalDevice();
    createLogicalDevice();
    createAllocator();
}

void VkContext::destroy() {
    if (m_allocator) vmaDestroyAllocator(m_allocator);
    if (m_device) vkDestroyDevice(m_device, nullptr);
    if (ENABLE_VALIDATION && m_debugMessenger) {
        vkDestroyDebugUtilsMessengerEXT(m_instance, m_debugMessenger, nullptr);
    }
    if (m_surface) vkDestroySurfaceKHR(m_instance, m_surface, nullptr);
    if (m_instance) vkDestroyInstance(m_instance, nullptr);
}

void VkContext::createInstance() {
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Cosmico";
    appInfo.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
    appInfo.pEngineName = "Cosmico Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(0, 1, 0);
    appInfo.apiVersion = VK_API_VERSION_1_2;

    // Required extensions from GLFW
    uint32_t glfwExtCount = 0;
    const char** glfwExts = glfwGetRequiredInstanceExtensions(&glfwExtCount);
    std::vector<const char*> extensions(glfwExts, glfwExts + glfwExtCount);

    // External memory/semaphore support requires these instance extensions
    extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
    extensions.push_back(VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME);
    extensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME);

    std::vector<const char*> layers;
    if (ENABLE_VALIDATION) {
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        layers.push_back("VK_LAYER_KHRONOS_validation");
    }

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;
    createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();
    createInfo.enabledLayerCount = static_cast<uint32_t>(layers.size());
    createInfo.ppEnabledLayerNames = layers.data();

    if (vkCreateInstance(&createInfo, nullptr, &m_instance) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create Vulkan instance");
    }

    volkLoadInstance(m_instance);
}

void VkContext::setupDebugMessenger() {
    VkDebugUtilsMessengerCreateInfoEXT createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                 VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                             VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                             VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    createInfo.pfnUserCallback = debugCallback;

    if (vkCreateDebugUtilsMessengerEXT(m_instance, &createInfo, nullptr, &m_debugMessenger) != VK_SUCCESS) {
        throw std::runtime_error("Failed to set up debug messenger");
    }
}

void VkContext::createSurface(GLFWwindow* window) {
    if (glfwCreateWindowSurface(m_instance, window, nullptr, &m_surface) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create window surface");
    }
}

QueueFamilyIndices VkContext::findQueueFamilies(VkPhysicalDevice device) {
    QueueFamilyIndices indices;
    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &count, nullptr);
    std::vector<VkQueueFamilyProperties> families(count);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &count, families.data());

    for (uint32_t i = 0; i < count; i++) {
        // Prefer a queue that supports both graphics and compute
        if ((families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) &&
            (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT)) {
            indices.graphicsFamily = i;
        }

        VkBool32 presentSupport = VK_FALSE;
        vkGetPhysicalDeviceSurfaceSupportKHR(device, i, m_surface, &presentSupport);
        if (presentSupport) {
            indices.presentFamily = i;
        }

        if (indices.isComplete()) break;
    }
    return indices;
}

bool VkContext::isDeviceSuitable(VkPhysicalDevice device) {
    QueueFamilyIndices indices = findQueueFamilies(device);
    if (!indices.isComplete()) return false;

    // Check swapchain extension
    uint32_t extCount = 0;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extCount, nullptr);
    std::vector<VkExtensionProperties> exts(extCount);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extCount, exts.data());

    bool hasSwapchain = false;
    for (auto& ext : exts) {
        if (strcmp(ext.extensionName, VK_KHR_SWAPCHAIN_EXTENSION_NAME) == 0) {
            hasSwapchain = true;
            break;
        }
    }
    return hasSwapchain;
}

bool VkContext::checkExternalMemorySupport(VkPhysicalDevice device) {
    uint32_t extCount = 0;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extCount, nullptr);
    std::vector<VkExtensionProperties> exts(extCount);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extCount, exts.data());

    bool hasExtMem = false, hasExtMemWin32 = false;
    bool hasExtSem = false, hasExtSemWin32 = false;
    bool hasTimelineSem = false;

    for (auto& ext : exts) {
        if (strcmp(ext.extensionName, VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME) == 0) hasExtMem = true;
        if (strcmp(ext.extensionName, VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME) == 0) hasExtMemWin32 = true;
        if (strcmp(ext.extensionName, VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME) == 0) hasExtSem = true;
        if (strcmp(ext.extensionName, VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME) == 0) hasExtSemWin32 = true;
        if (strcmp(ext.extensionName, VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME) == 0) hasTimelineSem = true;
    }

    return hasExtMem && hasExtMemWin32 && hasExtSem && hasExtSemWin32 && hasTimelineSem;
}

void VkContext::pickPhysicalDevice() {
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(m_instance, &count, nullptr);
    if (count == 0) {
        throw std::runtime_error("No Vulkan-capable GPU found");
    }
    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(m_instance, &count, devices.data());

    // Prefer discrete GPU
    for (auto& dev : devices) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(dev, &props);
        if (isDeviceSuitable(dev) && props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            m_physicalDevice = dev;
            m_queueFamilies = findQueueFamilies(dev);
            m_externalMemorySupported = checkExternalMemorySupport(dev);
            fprintf(stderr, "[Cosmico] GPU: %s (external memory: %s)\n",
                    props.deviceName, m_externalMemorySupported ? "yes" : "no");
            return;
        }
    }

    // Fallback to any suitable device
    for (auto& dev : devices) {
        if (isDeviceSuitable(dev)) {
            m_physicalDevice = dev;
            m_queueFamilies = findQueueFamilies(dev);
            m_externalMemorySupported = checkExternalMemorySupport(dev);
            VkPhysicalDeviceProperties props;
            vkGetPhysicalDeviceProperties(dev, &props);
            fprintf(stderr, "[Cosmico] GPU: %s (external memory: %s)\n",
                    props.deviceName, m_externalMemorySupported ? "yes" : "no");
            return;
        }
    }

    throw std::runtime_error("No suitable GPU found");
}

void VkContext::createLogicalDevice() {
    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
    float priority = 1.0f;

    // Graphics/compute queue
    VkDeviceQueueCreateInfo qci{};
    qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = m_queueFamilies.graphicsFamily;
    qci.queueCount = 1;
    qci.pQueuePriorities = &priority;
    queueCreateInfos.push_back(qci);

    if (m_queueFamilies.presentFamily != m_queueFamilies.graphicsFamily) {
        qci.queueFamilyIndex = m_queueFamilies.presentFamily;
        queueCreateInfos.push_back(qci);
    }

    VkPhysicalDeviceFeatures features{};
    features.largePoints = VK_TRUE; // For gl_PointSize in vertex shader

    // Device extensions
    std::vector<const char*> extensions = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };

    if (m_externalMemorySupported) {
        extensions.push_back(VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME);
        extensions.push_back(VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME);
        extensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME);
        extensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME);
        extensions.push_back(VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME);
    }

    // Timeline semaphore feature (Vulkan 1.2 core but needs explicit enable)
    VkPhysicalDeviceTimelineSemaphoreFeatures timelineFeatures{};
    timelineFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES;
    timelineFeatures.timelineSemaphore = m_externalMemorySupported ? VK_TRUE : VK_FALSE;

    VkDeviceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.pNext = m_externalMemorySupported ? &timelineFeatures : nullptr;
    createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
    createInfo.pQueueCreateInfos = queueCreateInfos.data();
    createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();
    createInfo.pEnabledFeatures = &features;

    if (vkCreateDevice(m_physicalDevice, &createInfo, nullptr, &m_device) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create logical device");
    }

    volkLoadDevice(m_device);

    vkGetDeviceQueue(m_device, m_queueFamilies.graphicsFamily, 0, &m_graphicsQueue);
    vkGetDeviceQueue(m_device, m_queueFamilies.presentFamily, 0, &m_presentQueue);
}

void VkContext::createAllocator() {
    VmaVulkanFunctions vulkanFunctions{};
    vulkanFunctions.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
    vulkanFunctions.vkGetDeviceProcAddr = vkGetDeviceProcAddr;

    VmaAllocatorCreateInfo allocInfo{};
    allocInfo.physicalDevice = m_physicalDevice;
    allocInfo.device = m_device;
    allocInfo.instance = m_instance;
    allocInfo.vulkanApiVersion = VK_API_VERSION_1_2;
    allocInfo.pVulkanFunctions = &vulkanFunctions;

    if (vmaCreateAllocator(&allocInfo, &m_allocator) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create VMA allocator");
    }
}

} // namespace cosmico
