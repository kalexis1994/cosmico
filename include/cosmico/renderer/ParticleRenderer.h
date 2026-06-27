#pragma once
#include <volk.h>
#include <string>

namespace cosmico {

class VkContext;

struct ParticleRenderPushConstants {
    alignas(16) float viewProj[16];   // 64B (vertex)
    float pointScale;                  // 4B  (vertex)
    float hasPlanetTextures;           // 4B  (fragment flag: 0.0 or 1.0)
    float tanHalfFov;                  // 4B  tan(fov/2) for billboard sizing
    float aspectRatio;                 // 4B  viewport width / height
    alignas(16) float camRight[4];    // 16B (xyz=right, w=camPos.x)
    alignas(16) float camUp[4];       // 16B (xyz=up,    w=camPos.y)
    alignas(16) float camForward[4];  // 16B (xyz=fwd,   w=camPos.z)
    float simTime;                     // 4B  (cosmetic rotation)
    float showDarkMatter;              // 4B  (0.0 = hide, 1.0 = show type<0 particles)
    float lumStrength;                 // 4B  (0 = speed color, 1 = density luminosity)
    float coordScale;                  // 4B  position scale about origin (1 = comoving, a/aInit = physical)
};  // = 144 bytes total

class ParticleRenderer {
public:
    void init(VkContext& ctx, VkRenderPass renderPass, VkDescriptorSetLayout particleSetLayout,
              const std::string& shaderDir,
              VkDescriptorSetLayout textureSetLayout = VK_NULL_HANDLE);
    void destroy(VkDevice device);

    void draw(VkCommandBuffer cmd, VkDescriptorSet particleSet, uint32_t particleCount,
              const ParticleRenderPushConstants& pushConstants,
              VkDescriptorSet textureSet = VK_NULL_HANDLE,
              uint32_t sphereCount = 0);

    VkPipelineLayout pipelineLayout() const { return m_pipelineLayout; }

private:
    VkPipeline m_pipeline       = VK_NULL_HANDLE;  // sphere pipeline (gl_FragDepth)
    VkPipeline m_simplePipeline = VK_NULL_HANDLE;  // particle pipeline (early-Z)
    VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
};

} // namespace cosmico
