#pragma once
#include <volk.h>
#include <vector>

namespace cosmico {

class DescriptorLayoutBuilder {
public:
    DescriptorLayoutBuilder& addBinding(uint32_t binding, VkDescriptorType type,
                                         VkShaderStageFlags stages, uint32_t count = 1);
    VkDescriptorSetLayout build(VkDevice device);

private:
    std::vector<VkDescriptorSetLayoutBinding> m_bindings;
};

class DescriptorPool {
public:
    void init(VkDevice device, uint32_t maxSets,
              const std::vector<VkDescriptorPoolSize>& poolSizes);
    void destroy(VkDevice device);
    void reset(VkDevice device);

    VkDescriptorSet allocate(VkDevice device, VkDescriptorSetLayout layout);

private:
    VkDescriptorPool m_pool = VK_NULL_HANDLE;
};

class DescriptorWriter {
public:
    DescriptorWriter& writeBuffer(uint32_t binding, VkDescriptorType type,
                                   const VkDescriptorBufferInfo* bufInfo);
    DescriptorWriter& writeImage(uint32_t binding, VkDescriptorType type,
                                  const VkDescriptorImageInfo* imageInfo);
    void update(VkDevice device, VkDescriptorSet set);

private:
    std::vector<VkWriteDescriptorSet> m_writes;
};

} // namespace cosmico
