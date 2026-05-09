#include <cosmico/renderer/PlanetTextures.h>
#include <cosmico/vulkan/VkContext.h>
#include <cosmico/vulkan/VkDescriptors.h>

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb_image_resize2.h>
#include <stb_image.h>

#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <array>
#include <vector>

namespace cosmico {

static const std::array<const char*, PlanetTextures::LAYER_COUNT> s_textureFiles = {
    "sun.jpg",
    "mercury.jpg",
    "venus_atmosphere.jpg",
    "earth_daymap.jpg",
    "mars.jpg",
    "jupiter.jpg",
    "saturn.jpg",
    "uranus.jpg",
    "neptune.jpg",
    "ceres_fictional.jpg",
    "eris_fictional.jpg",
    "moon.jpg",
    "io.jpg",
    "europa.jpg",
    "ganymede.jpg",
    "callisto.jpg",
    "titan.jpg",
    "enceladus.jpg",
    "triton.jpg",
    "saturn_ring_alpha.png"   // layer 19: Saturn ring radial profile
};

bool PlanetTextures::init(VkContext& ctx, const std::string& textureDir) {
    VkDevice device = ctx.device();
    VmaAllocator allocator = ctx.allocator();

    const VkDeviceSize layerSize = TEX_WIDTH * TEX_HEIGHT * 4;
    const VkDeviceSize totalSize = layerSize * LAYER_COUNT;

    // Load all textures into a contiguous host buffer
    std::vector<uint8_t> hostData(totalSize);

    for (uint32_t i = 0; i < LAYER_COUNT; i++) {
        std::string path = textureDir + "/" + s_textureFiles[i];
        int w, h, channels;
        unsigned char* pixels = stbi_load(path.c_str(), &w, &h, &channels, 4);
        if (!pixels) {
            fprintf(stderr, "[PlanetTextures] Failed to load: %s\n", path.c_str());
            return false;
        }

        uint8_t* dst = hostData.data() + i * layerSize;

        if (static_cast<uint32_t>(w) == TEX_WIDTH && static_cast<uint32_t>(h) == TEX_HEIGHT) {
            memcpy(dst, pixels, layerSize);
        } else {
            // Resize to target dimensions
            stbir_resize_uint8_linear(
                pixels, w, h, 0,
                dst, TEX_WIDTH, TEX_HEIGHT, 0,
                STBIR_RGBA);
        }

        stbi_image_free(pixels);
    }

    // Create staging buffer
    VkBufferCreateInfo stagingBufInfo{};
    stagingBufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    stagingBufInfo.size = totalSize;
    stagingBufInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

    VmaAllocationCreateInfo stagingAllocInfo{};
    stagingAllocInfo.usage = VMA_MEMORY_USAGE_CPU_ONLY;

    VkBuffer stagingBuffer;
    VmaAllocation stagingAlloc;
    vmaCreateBuffer(allocator, &stagingBufInfo, &stagingAllocInfo,
                    &stagingBuffer, &stagingAlloc, nullptr);

    void* mapped;
    vmaMapMemory(allocator, stagingAlloc, &mapped);
    memcpy(mapped, hostData.data(), totalSize);
    vmaUnmapMemory(allocator, stagingAlloc);

    // Create 2D array image
    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    imageInfo.extent = { TEX_WIDTH, TEX_HEIGHT, 1 };
    uint32_t mipCount = static_cast<uint32_t>(std::floor(std::log2(
        static_cast<double>((std::max)(TEX_WIDTH, TEX_HEIGHT))))) + 1;
    imageInfo.mipLevels = mipCount;
    imageInfo.arrayLayers = LAYER_COUNT;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT
                    | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo imgAllocInfo{};
    imgAllocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

    if (vmaCreateImage(allocator, &imageInfo, &imgAllocInfo,
                       &m_image, &m_alloc, nullptr) != VK_SUCCESS) {
        fprintf(stderr, "[PlanetTextures] Failed to create 2D array image\n");
        vmaDestroyBuffer(allocator, stagingBuffer, stagingAlloc);
        return false;
    }

    // Record upload commands
    VkCommandPoolCreateInfo poolCI{};
    poolCI.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolCI.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    poolCI.queueFamilyIndex = ctx.queueFamilies().graphicsFamily;

    VkCommandPool tempPool;
    vkCreateCommandPool(device, &poolCI, nullptr, &tempPool);

    VkCommandBufferAllocateInfo cmdAllocInfo{};
    cmdAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdAllocInfo.commandPool = tempPool;
    cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAllocInfo.commandBufferCount = 1;

    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(device, &cmdAllocInfo, &cmd);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &beginInfo);

    // Transition: UNDEFINED -> TRANSFER_DST (all layers)
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = m_image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.levelCount = mipCount;
    barrier.subresourceRange.layerCount = LAYER_COUNT;
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                         0, nullptr, 0, nullptr, 1, &barrier);

    // Copy each layer from staging buffer
    for (uint32_t i = 0; i < LAYER_COUNT; i++) {
        VkBufferImageCopy region{};
        region.bufferOffset = i * layerSize;
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.mipLevel = 0;
        region.imageSubresource.baseArrayLayer = i;
        region.imageSubresource.layerCount = 1;
        region.imageExtent = { TEX_WIDTH, TEX_HEIGHT, 1 };

        vkCmdCopyBufferToImage(cmd, stagingBuffer, m_image,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    }

    // ── Generate mipmaps via blit chain ──
    {
        int32_t mipW = static_cast<int32_t>(TEX_WIDTH);
        int32_t mipH = static_cast<int32_t>(TEX_HEIGHT);

        VkImageMemoryBarrier mipBarrier{};
        mipBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        mipBarrier.image = m_image;
        mipBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        mipBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        mipBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        mipBarrier.subresourceRange.levelCount = 1;
        mipBarrier.subresourceRange.baseArrayLayer = 0;
        mipBarrier.subresourceRange.layerCount = LAYER_COUNT;

        for (uint32_t i = 1; i < mipCount; i++) {
            // Level i-1: TRANSFER_DST → TRANSFER_SRC
            mipBarrier.subresourceRange.baseMipLevel = i - 1;
            mipBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            mipBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            mipBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            mipBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                                 0, nullptr, 0, nullptr, 1, &mipBarrier);

            int32_t nextW = mipW > 1 ? mipW / 2 : 1;
            int32_t nextH = mipH > 1 ? mipH / 2 : 1;

            VkImageBlit blit{};
            blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            blit.srcSubresource.mipLevel = i - 1;
            blit.srcSubresource.baseArrayLayer = 0;
            blit.srcSubresource.layerCount = LAYER_COUNT;
            blit.srcOffsets[0] = {0, 0, 0};
            blit.srcOffsets[1] = {mipW, mipH, 1};
            blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            blit.dstSubresource.mipLevel = i;
            blit.dstSubresource.baseArrayLayer = 0;
            blit.dstSubresource.layerCount = LAYER_COUNT;
            blit.dstOffsets[0] = {0, 0, 0};
            blit.dstOffsets[1] = {nextW, nextH, 1};

            vkCmdBlitImage(cmd,
                           m_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           m_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                           1, &blit, VK_FILTER_LINEAR);

            // Level i-1: TRANSFER_SRC → SHADER_READ_ONLY
            mipBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            mipBarrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            mipBarrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            mipBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
                                 0, nullptr, 0, nullptr, 1, &mipBarrier);

            mipW = nextW;
            mipH = nextH;
        }

        // Last mip level: TRANSFER_DST → SHADER_READ_ONLY
        mipBarrier.subresourceRange.baseMipLevel = mipCount - 1;
        mipBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        mipBarrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        mipBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        mipBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
                             0, nullptr, 0, nullptr, 1, &mipBarrier);
    }

    vkEndCommandBuffer(cmd);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;

    vkQueueSubmit(ctx.graphicsQueue(), 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(ctx.graphicsQueue());

    vkDestroyCommandPool(device, tempPool, nullptr);
    vmaDestroyBuffer(allocator, stagingBuffer, stagingAlloc);

    // Create 2D array image view
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = m_image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    viewInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = mipCount;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = LAYER_COUNT;

    if (vkCreateImageView(device, &viewInfo, nullptr, &m_view) != VK_SUCCESS) {
        fprintf(stderr, "[PlanetTextures] Failed to create image view\n");
        return false;
    }

    // Sampler: repeat U (longitude wrap), clamp V (poles), trilinear + anisotropic
    VkPhysicalDeviceProperties devProps;
    vkGetPhysicalDeviceProperties(ctx.physicalDevice(), &devProps);

    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.anisotropyEnable = VK_TRUE;
    samplerInfo.maxAnisotropy = devProps.limits.maxSamplerAnisotropy;
    samplerInfo.maxLod = static_cast<float>(mipCount);

    if (vkCreateSampler(device, &samplerInfo, nullptr, &m_sampler) != VK_SUCCESS) {
        fprintf(stderr, "[PlanetTextures] Failed to create sampler\n");
        return false;
    }

    // Descriptor set layout
    m_setLayout = DescriptorLayoutBuilder()
        .addBinding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    VK_SHADER_STAGE_FRAGMENT_BIT)
        .build(device);

    // Descriptor pool
    VkDescriptorPoolSize poolSize{};
    poolSize.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSize.descriptorCount = 1;

    VkDescriptorPoolCreateInfo descPoolInfo{};
    descPoolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    descPoolInfo.maxSets = 1;
    descPoolInfo.poolSizeCount = 1;
    descPoolInfo.pPoolSizes = &poolSize;

    if (vkCreateDescriptorPool(device, &descPoolInfo, nullptr, &m_descriptorPool) != VK_SUCCESS) {
        fprintf(stderr, "[PlanetTextures] Failed to create descriptor pool\n");
        return false;
    }

    // Allocate descriptor set
    VkDescriptorSetAllocateInfo setAllocInfo{};
    setAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    setAllocInfo.descriptorPool = m_descriptorPool;
    setAllocInfo.descriptorSetCount = 1;
    setAllocInfo.pSetLayouts = &m_setLayout;

    if (vkAllocateDescriptorSets(device, &setAllocInfo, &m_descriptorSet) != VK_SUCCESS) {
        fprintf(stderr, "[PlanetTextures] Failed to allocate descriptor set\n");
        return false;
    }

    // Write descriptor
    VkDescriptorImageInfo descImageInfo{};
    descImageInfo.sampler = m_sampler;
    descImageInfo.imageView = m_view;
    descImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    DescriptorWriter()
        .writeImage(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &descImageInfo)
        .update(device, m_descriptorSet);

    fprintf(stderr, "[PlanetTextures] Loaded %u planet textures (%ux%u)\n",
            LAYER_COUNT, TEX_WIDTH, TEX_HEIGHT);
    return true;
}

void PlanetTextures::destroy(VkDevice device, VmaAllocator allocator) {
    if (m_descriptorPool) {
        vkDestroyDescriptorPool(device, m_descriptorPool, nullptr);
        m_descriptorPool = VK_NULL_HANDLE;
        m_descriptorSet = VK_NULL_HANDLE;
    }
    if (m_setLayout) {
        vkDestroyDescriptorSetLayout(device, m_setLayout, nullptr);
        m_setLayout = VK_NULL_HANDLE;
    }
    if (m_sampler) {
        vkDestroySampler(device, m_sampler, nullptr);
        m_sampler = VK_NULL_HANDLE;
    }
    if (m_view) {
        vkDestroyImageView(device, m_view, nullptr);
        m_view = VK_NULL_HANDLE;
    }
    if (m_image) {
        vmaDestroyImage(allocator, m_image, m_alloc);
        m_image = VK_NULL_HANDLE;
        m_alloc = VK_NULL_HANDLE;
    }
}

} // namespace cosmico
