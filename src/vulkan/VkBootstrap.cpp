// This file must be compiled as a single translation unit for Volk and VMA implementations.
// It defines the function pointers for Vulkan and the VMA allocator implementation.

#define VOLK_IMPLEMENTATION
#include <volk.h>

#define VMA_IMPLEMENTATION
#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#include <vk_mem_alloc.h>
