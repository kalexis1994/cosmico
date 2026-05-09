#include <cosmico/simulation/ParticleSystem.h>
#include <cosmico/vulkan/VkContext.h>
#include <cosmico/vulkan/VkCommands.h>

namespace cosmico {

void ParticleSystem::init(VkContext& ctx, uint32_t maxParticles) {
    m_maxParticles = maxParticles;
    m_exportable = false;
    VkDeviceSize bufferSize = static_cast<VkDeviceSize>(maxParticles) * 48; // 48 bytes per particle

    for (int i = 0; i < 2; i++) {
        m_buffers[i].create(ctx.allocator(), bufferSize,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VMA_MEMORY_USAGE_AUTO,
            VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
    }

    // Create descriptor layouts
    // Compute: set 0 = { binding 0: read SSBO, binding 1: write SSBO }
    m_computeSetLayout = DescriptorLayoutBuilder()
        .addBinding(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT)
        .addBinding(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT)
        .build(ctx.device());

    // Graphics: set 0 = { binding 0: read SSBO }
    m_graphicsSetLayout = DescriptorLayoutBuilder()
        .addBinding(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_VERTEX_BIT)
        .build(ctx.device());

    // Descriptor pool: 4 compute sets (2 ping-pong x 2 bindings) + 2 graphics sets
    std::vector<VkDescriptorPoolSize> poolSizes = {
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 6 }
    };
    m_descriptorPool.init(ctx.device(), 4, poolSizes);

    // Allocate and write descriptor sets
    for (int i = 0; i < 2; i++) {
        m_computeSets[i] = m_descriptorPool.allocate(ctx.device(), m_computeSetLayout);
        m_graphicsSets[i] = m_descriptorPool.allocate(ctx.device(), m_graphicsSetLayout);
    }

    // Write compute sets: [0] reads A writes B, [1] reads B writes A
    for (int i = 0; i < 2; i++) {
        auto readInfo = m_buffers[i].descriptorInfo();
        auto writeInfo = m_buffers[1 - i].descriptorInfo();

        DescriptorWriter()
            .writeBuffer(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &readInfo)
            .writeBuffer(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &writeInfo)
            .update(ctx.device(), m_computeSets[i]);
    }

    // Write graphics sets: [i] reads buffer[i]
    for (int i = 0; i < 2; i++) {
        auto info = m_buffers[i].descriptorInfo();
        DescriptorWriter()
            .writeBuffer(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &info)
            .update(ctx.device(), m_graphicsSets[i]);
    }
}

void ParticleSystem::initExportable(VkContext& ctx, uint32_t maxParticles) {
    m_maxParticles = maxParticles;
    m_exportable = true;
    VkDeviceSize bufferSize = static_cast<VkDeviceSize>(maxParticles) * 48;

    // For CUDA backend: buffer[0] is the exportable shared buffer (CUDA reads/writes directly)
    // buffer[1] is unused (no ping-pong needed — CUDA does in-place update)
    m_buffers[0].createExportable(ctx.device(), ctx.physicalDevice(), bufferSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);

    // Create a staging buffer for initial uploads (host-visible)
    m_stagingBuffer.create(ctx.allocator(), bufferSize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VMA_MEMORY_USAGE_AUTO,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);

    // Descriptor layouts (same as non-exportable for graphics compatibility)
    m_computeSetLayout = DescriptorLayoutBuilder()
        .addBinding(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT)
        .addBinding(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT)
        .build(ctx.device());

    m_graphicsSetLayout = DescriptorLayoutBuilder()
        .addBinding(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_VERTEX_BIT)
        .build(ctx.device());

    // Only need 1 graphics set (single buffer) + 1 compute set (unused but needed for layout)
    std::vector<VkDescriptorPoolSize> poolSizes = {
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4 }
    };
    m_descriptorPool.init(ctx.device(), 3, poolSizes);

    // Graphics set: always reads buffer[0]
    m_graphicsSets[0] = m_descriptorPool.allocate(ctx.device(), m_graphicsSetLayout);
    m_graphicsSets[1] = m_graphicsSets[0]; // Both point to same buffer

    auto info = m_buffers[0].descriptorInfo();
    DescriptorWriter()
        .writeBuffer(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &info)
        .update(ctx.device(), m_graphicsSets[0]);

    // Compute sets (for brute-force fallback — points buffer[0] to both read/write)
    m_computeSets[0] = m_descriptorPool.allocate(ctx.device(), m_computeSetLayout);
    m_computeSets[1] = m_computeSets[0];
    auto selfInfo = m_buffers[0].descriptorInfo();
    DescriptorWriter()
        .writeBuffer(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &selfInfo)
        .writeBuffer(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &selfInfo)
        .update(ctx.device(), m_computeSets[0]);
}

void ParticleSystem::destroy(VkContext& ctx) {
    m_descriptorPool.destroy(ctx.device());
    if (m_computeSetLayout) vkDestroyDescriptorSetLayout(ctx.device(), m_computeSetLayout, nullptr);
    if (m_graphicsSetLayout) vkDestroyDescriptorSetLayout(ctx.device(), m_graphicsSetLayout, nullptr);

    if (m_exportable) {
        m_buffers[0].destroyExportable(ctx.device());
        m_stagingBuffer.destroy(ctx.allocator());
    } else {
        for (int i = 0; i < 2; i++) {
            m_buffers[i].destroy(ctx.allocator());
        }
    }

    m_computeSetLayout = VK_NULL_HANDLE;
    m_graphicsSetLayout = VK_NULL_HANDLE;
}

void ParticleSystem::upload(VkContext& ctx, const void* data, uint32_t particleCount) {
    m_particleCount = particleCount;
    VkDeviceSize size = static_cast<VkDeviceSize>(particleCount) * 48;
    // Upload to both buffers so either can be read first frame
    m_buffers[0].upload(ctx.allocator(), data, size);
    m_buffers[1].upload(ctx.allocator(), data, size);
    m_readIdx = 0;
}

void ParticleSystem::uploadExportable(VkContext& ctx, const void* data, uint32_t particleCount) {
    m_particleCount = particleCount;
    VkDeviceSize size = static_cast<VkDeviceSize>(particleCount) * 48;

    // Upload via staging buffer + copy command
    m_stagingBuffer.upload(ctx.allocator(), data, size);

    // Create a temporary command pool for the copy
    CommandPool tempPool;
    tempPool.init(ctx.device(), ctx.queueFamilies().graphicsFamily,
                  VK_COMMAND_POOL_CREATE_TRANSIENT_BIT);

    auto cmd = beginSingleTimeCommands(ctx.device(), tempPool.handle());

    VkBufferCopy copyRegion{};
    copyRegion.srcOffset = 0;
    copyRegion.dstOffset = 0;
    copyRegion.size = size;
    vkCmdCopyBuffer(cmd, m_stagingBuffer.buffer(), m_buffers[0].buffer(), 1, &copyRegion);

    endSingleTimeCommands(ctx.device(), tempPool.handle(), ctx.graphicsQueue(), cmd);
    tempPool.destroy(ctx.device());

    m_readIdx = 0;
}

void ParticleSystem::swapBuffers() {
    if (!m_exportable) {
        // After compute writes to writeBuffer, it becomes the new readBuffer
        m_readIdx = 1 - m_readIdx;
    }
    // For exportable: no swap needed, CUDA updates buffer[0] in-place
}

#ifdef _WIN32
HANDLE ParticleSystem::getExportableHandle(VkDevice device) const {
    return m_buffers[0].getWin32Handle(device);
}
#endif

VkDeviceSize ParticleSystem::exportableBufferSize() const {
    return m_buffers[0].size();
}

} // namespace cosmico
