#include <cosmico/vulkan/VkDescriptors.h>
#include <stdexcept>

namespace cosmico {

DescriptorLayoutBuilder& DescriptorLayoutBuilder::addBinding(
    uint32_t binding, VkDescriptorType type, VkShaderStageFlags stages, uint32_t count) {
    VkDescriptorSetLayoutBinding b{};
    b.binding = binding;
    b.descriptorType = type;
    b.descriptorCount = count;
    b.stageFlags = stages;
    m_bindings.push_back(b);
    return *this;
}

VkDescriptorSetLayout DescriptorLayoutBuilder::build(VkDevice device) {
    VkDescriptorSetLayoutCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    info.bindingCount = static_cast<uint32_t>(m_bindings.size());
    info.pBindings = m_bindings.data();

    VkDescriptorSetLayout layout;
    if (vkCreateDescriptorSetLayout(device, &info, nullptr, &layout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create descriptor set layout");
    }
    return layout;
}

void DescriptorPool::init(VkDevice device, uint32_t maxSets,
                           const std::vector<VkDescriptorPoolSize>& poolSizes) {
    VkDescriptorPoolCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    info.maxSets = maxSets;
    info.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    info.pPoolSizes = poolSizes.data();
    info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;

    if (vkCreateDescriptorPool(device, &info, nullptr, &m_pool) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create descriptor pool");
    }
}

void DescriptorPool::destroy(VkDevice device) {
    if (m_pool) {
        vkDestroyDescriptorPool(device, m_pool, nullptr);
        m_pool = VK_NULL_HANDLE;
    }
}

void DescriptorPool::reset(VkDevice device) {
    vkResetDescriptorPool(device, m_pool, 0);
}

VkDescriptorSet DescriptorPool::allocate(VkDevice device, VkDescriptorSetLayout layout) {
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = m_pool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &layout;

    VkDescriptorSet set;
    if (vkAllocateDescriptorSets(device, &allocInfo, &set) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate descriptor set");
    }
    return set;
}

DescriptorWriter& DescriptorWriter::writeBuffer(uint32_t binding, VkDescriptorType type,
                                                  const VkDescriptorBufferInfo* bufInfo) {
    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstBinding = binding;
    write.descriptorCount = 1;
    write.descriptorType = type;
    write.pBufferInfo = bufInfo;
    m_writes.push_back(write);
    return *this;
}

DescriptorWriter& DescriptorWriter::writeImage(uint32_t binding, VkDescriptorType type,
                                                  const VkDescriptorImageInfo* imageInfo) {
    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstBinding = binding;
    write.descriptorCount = 1;
    write.descriptorType = type;
    write.pImageInfo = imageInfo;
    m_writes.push_back(write);
    return *this;
}

void DescriptorWriter::update(VkDevice device, VkDescriptorSet set) {
    for (auto& w : m_writes) {
        w.dstSet = set;
    }
    vkUpdateDescriptorSets(device, static_cast<uint32_t>(m_writes.size()), m_writes.data(), 0, nullptr);
    m_writes.clear();
}

} // namespace cosmico
