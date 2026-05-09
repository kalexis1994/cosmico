#pragma once
#include <volk.h>

struct ImFont;
struct GLFWwindow;

namespace cosmico {

class VkContext;
class VkSwapchain;

class VkImGuiBackend {
public:
    void init(VkContext& ctx, VkSwapchain& swapchain, GLFWwindow* window, VkRenderPass renderPass);
    void destroy(VkDevice device);

    void newFrame();
    void render(VkCommandBuffer cmd);

    // Fonts loaded with extended Unicode math ranges
    // Base sizes (100% zoom)
    ImFont* fontBody    = nullptr;   // 15px Segoe UI
    ImFont* fontHeading = nullptr;   // 20px Segoe UI Bold
    ImFont* fontFormula = nullptr;   // 16px Consolas

    // Multiple sizes for zoom levels (75%, 100%, 125%, 150%, 175%)
    static constexpr int ZOOM_LEVELS = 5;
    static constexpr float ZOOM_SCALES[ZOOM_LEVELS] = {0.75f, 1.0f, 1.25f, 1.5f, 1.75f};
    ImFont* fontBodyZoom[ZOOM_LEVELS]    = {};
    ImFont* fontHeadingZoom[ZOOM_LEVELS] = {};
    ImFont* fontFormulaZoom[ZOOM_LEVELS] = {};

private:
    VkDescriptorPool m_descriptorPool = VK_NULL_HANDLE;
};

} // namespace cosmico
