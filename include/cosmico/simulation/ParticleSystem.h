#pragma once
#include <cosmico/vulkan/VkBuffer.h>
#include <cosmico/vulkan/VkDescriptors.h>
#include <volk.h>
#include <cstdint>

#ifdef _WIN32
#include <windows.h>
#endif

namespace cosmico {

class VkContext;

class ParticleSystem {
public:
    void init(VkContext& ctx, uint32_t maxParticles);
    void initExportable(VkContext& ctx, uint32_t maxParticles);
    void destroy(VkContext& ctx);

    void upload(VkContext& ctx, const void* data, uint32_t particleCount);
    void uploadExportable(VkContext& ctx, const void* data, uint32_t particleCount);
    void swapBuffers();

    // Current read buffer index (the one compute just wrote to)
    uint32_t readIndex() const { return m_readIdx; }
    uint32_t writeIndex() const { return 1 - m_readIdx; }

    GpuBuffer& readBuffer() { return m_buffers[m_readIdx]; }
    GpuBuffer& writeBuffer() { return m_buffers[1 - m_readIdx]; }

    VkDescriptorSet computeDescriptorSet() const { return m_computeSets[m_readIdx]; }
    VkDescriptorSet graphicsDescriptorSet() const { return m_graphicsSets[m_readIdx]; }

    VkDescriptorSetLayout computeSetLayout() const { return m_computeSetLayout; }
    VkDescriptorSetLayout graphicsSetLayout() const { return m_graphicsSetLayout; }

    uint32_t particleCount() const { return m_particleCount; }
    uint32_t maxParticles() const { return m_maxParticles; }
    bool isExportable() const { return m_exportable; }

    // For CUDA interop: get Win32 handle and buffer size for the single SSBO
    // (Barnes-Hut uses a single buffer, not ping-pong)
#ifdef _WIN32
    HANDLE getExportableHandle(VkDevice device) const;
#endif
    VkDeviceSize exportableBufferSize() const;

private:
    GpuBuffer m_buffers[2];
    uint32_t m_readIdx = 0;
    uint32_t m_particleCount = 0;
    uint32_t m_maxParticles = 0;
    bool m_exportable = false;

    // Staging buffer for initial data upload to exportable (device-local) buffers
    GpuBuffer m_stagingBuffer;

    // Compute: binding 0 = read SSBO, binding 1 = write SSBO
    VkDescriptorSetLayout m_computeSetLayout = VK_NULL_HANDLE;
    VkDescriptorSet m_computeSets[2]; // [0]: read A write B, [1]: read B write A

    // Graphics: binding 0 = read SSBO
    VkDescriptorSetLayout m_graphicsSetLayout = VK_NULL_HANDLE;
    VkDescriptorSet m_graphicsSets[2]; // [0]: read A, [1]: read B

    DescriptorPool m_descriptorPool;
};

} // namespace cosmico
