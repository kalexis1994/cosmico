#include <cosmico/vulkan/VkImGui.h>
#include <cosmico/vulkan/VkContext.h>
#include <cosmico/vulkan/VkSwapchain.h>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>

#include <stdexcept>

namespace cosmico {

static void checkVkResult(VkResult err) {
    if (err != VK_SUCCESS) {
        throw std::runtime_error("ImGui Vulkan error");
    }
}

void VkImGuiBackend::init(VkContext& ctx, VkSwapchain& swapchain, GLFWwindow* window,
                            VkRenderPass renderPass) {
    // Create dedicated descriptor pool for ImGui
    // Needs enough room for font atlas (multiple font sizes) + any user textures
    VkDescriptorPoolSize poolSizes[] = {
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 128 }
    };

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    poolInfo.maxSets = 128;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = poolSizes;

    if (vkCreateDescriptorPool(ctx.device(), &poolInfo, nullptr, &m_descriptorPool) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create ImGui descriptor pool");
    }

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

    // ── Load fonts with extended Unicode (Greek, math, sub/superscripts) ──
    {
        static const ImWchar mathRanges[] = {
            0x0020, 0x00FF,  // Basic Latin + Latin Supplement
            0x0370, 0x03FF,  // Greek and Coptic
            0x2000, 0x206F,  // General Punctuation
            0x2070, 0x209F,  // Superscripts and Subscripts
            0x2100, 0x214F,  // Letterlike Symbols (ℏ ℓ etc.)
            0x2150, 0x218F,  // Number Forms (vulgar fractions ½ ⅓ ¼ etc.)
            0x2190, 0x21FF,  // Arrows
            0x2200, 0x22FF,  // Mathematical Operators
            0x2300, 0x23FF,  // Miscellaneous Technical
            0x27E8, 0x27EF,  // Angle brackets ⟨⟩
            0,
        };

        ImFontConfig cfg;
        cfg.OversampleH = 2;
        cfg.OversampleV = 1;

        // Base sizes
        constexpr float bodyBase    = 15.0f;
        constexpr float headingBase = 20.0f;
        constexpr float formulaBase = 16.0f;

        // Load fonts at each zoom level
        for (int z = 0; z < ZOOM_LEVELS; z++) {
            float s = ZOOM_SCALES[z];
            fontBodyZoom[z] = io.Fonts->AddFontFromFileTTF(
                "C:/Windows/Fonts/segoeui.ttf", bodyBase * s, &cfg, mathRanges);
            fontHeadingZoom[z] = io.Fonts->AddFontFromFileTTF(
                "C:/Windows/Fonts/segoeuib.ttf", headingBase * s, &cfg, mathRanges);
            fontFormulaZoom[z] = io.Fonts->AddFontFromFileTTF(
                "C:/Windows/Fonts/consola.ttf", formulaBase * s, &cfg, mathRanges);
        }

        // Base fonts = 100% zoom (index 1)
        fontBody    = fontBodyZoom[1];
        fontHeading = fontHeadingZoom[1];
        fontFormula = fontFormulaZoom[1];

        // Fallback: if any font failed, use the default
        if (!fontBody)    fontBody    = io.Fonts->AddFontDefault();
        if (!fontHeading) fontHeading = fontBody;
        if (!fontFormula) fontFormula = fontBody;

        // Ensure zoom arrays have valid fallbacks
        for (int z = 0; z < ZOOM_LEVELS; z++) {
            if (!fontBodyZoom[z])    fontBodyZoom[z]    = fontBody;
            if (!fontHeadingZoom[z]) fontHeadingZoom[z] = fontHeading;
            if (!fontFormulaZoom[z]) fontFormulaZoom[z] = fontFormula;
        }
    }

    ImGui::StyleColorsDark();
    ImGuiStyle& style = ImGui::GetStyle();

    // Shape
    style.WindowRounding = 10.0f;
    style.ChildRounding = 6.0f;
    style.FrameRounding = 4.0f;
    style.PopupRounding = 6.0f;
    style.GrabRounding = 4.0f;
    style.TabRounding = 6.0f;
    style.ScrollbarRounding = 6.0f;

    // Spacing
    style.WindowPadding = ImVec2(14.0f, 14.0f);
    style.FramePadding = ImVec2(8.0f, 4.0f);
    style.ItemSpacing = ImVec2(8.0f, 6.0f);
    style.WindowBorderSize = 1.0f;
    style.FrameBorderSize = 0.0f;
    style.ScrollbarSize = 12.0f;

    // Transparency and colors — dark translucent glass look
    ImVec4* c = style.Colors;
    c[ImGuiCol_WindowBg]          = ImVec4(0.06f, 0.06f, 0.10f, 0.90f);
    c[ImGuiCol_ChildBg]           = ImVec4(0.07f, 0.07f, 0.12f, 0.40f);
    c[ImGuiCol_PopupBg]           = ImVec4(0.08f, 0.08f, 0.14f, 0.92f);
    c[ImGuiCol_Border]            = ImVec4(0.30f, 0.30f, 0.45f, 0.35f);
    c[ImGuiCol_FrameBg]           = ImVec4(0.12f, 0.12f, 0.20f, 0.60f);
    c[ImGuiCol_FrameBgHovered]    = ImVec4(0.20f, 0.20f, 0.35f, 0.70f);
    c[ImGuiCol_FrameBgActive]     = ImVec4(0.25f, 0.25f, 0.45f, 0.80f);
    c[ImGuiCol_TitleBg]           = ImVec4(0.06f, 0.06f, 0.12f, 0.90f);
    c[ImGuiCol_TitleBgActive]     = ImVec4(0.10f, 0.10f, 0.22f, 0.95f);
    c[ImGuiCol_TitleBgCollapsed]  = ImVec4(0.04f, 0.04f, 0.08f, 0.60f);
    c[ImGuiCol_ScrollbarBg]       = ImVec4(0.05f, 0.05f, 0.10f, 0.40f);
    c[ImGuiCol_ScrollbarGrab]     = ImVec4(0.30f, 0.30f, 0.45f, 0.50f);
    c[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.40f, 0.40f, 0.55f, 0.70f);
    c[ImGuiCol_ScrollbarGrabActive]  = ImVec4(0.45f, 0.45f, 0.65f, 0.85f);
    c[ImGuiCol_CheckMark]         = ImVec4(0.45f, 0.70f, 1.00f, 1.00f);
    c[ImGuiCol_SliderGrab]        = ImVec4(0.35f, 0.55f, 0.90f, 0.80f);
    c[ImGuiCol_SliderGrabActive]  = ImVec4(0.45f, 0.65f, 1.00f, 1.00f);
    c[ImGuiCol_Button]            = ImVec4(0.18f, 0.18f, 0.32f, 0.65f);
    c[ImGuiCol_ButtonHovered]     = ImVec4(0.28f, 0.28f, 0.50f, 0.80f);
    c[ImGuiCol_ButtonActive]      = ImVec4(0.32f, 0.32f, 0.60f, 0.95f);
    c[ImGuiCol_Header]            = ImVec4(0.18f, 0.18f, 0.32f, 0.55f);
    c[ImGuiCol_HeaderHovered]     = ImVec4(0.26f, 0.26f, 0.48f, 0.70f);
    c[ImGuiCol_HeaderActive]      = ImVec4(0.30f, 0.30f, 0.55f, 0.85f);
    c[ImGuiCol_Separator]         = ImVec4(0.30f, 0.30f, 0.50f, 0.30f);
    c[ImGuiCol_SeparatorHovered]  = ImVec4(0.40f, 0.40f, 0.65f, 0.60f);
    c[ImGuiCol_SeparatorActive]   = ImVec4(0.45f, 0.45f, 0.75f, 0.85f);
    c[ImGuiCol_ResizeGrip]        = ImVec4(0.30f, 0.30f, 0.55f, 0.30f);
    c[ImGuiCol_ResizeGripHovered] = ImVec4(0.40f, 0.40f, 0.65f, 0.60f);
    c[ImGuiCol_ResizeGripActive]  = ImVec4(0.50f, 0.50f, 0.80f, 0.90f);
    c[ImGuiCol_PlotLines]         = ImVec4(0.45f, 0.70f, 1.00f, 0.85f);
    c[ImGuiCol_PlotHistogram]     = ImVec4(0.45f, 0.70f, 1.00f, 0.85f);
    c[ImGuiCol_Text]              = ImVec4(0.90f, 0.90f, 0.95f, 1.00f);
    c[ImGuiCol_TextDisabled]      = ImVec4(0.45f, 0.45f, 0.55f, 1.00f);
    c[ImGuiCol_Tab]               = ImVec4(0.14f, 0.14f, 0.25f, 0.70f);
    c[ImGuiCol_TabHovered]        = ImVec4(0.28f, 0.28f, 0.50f, 0.85f);

    ImGui_ImplGlfw_InitForVulkan(window, true);

    ImGui_ImplVulkan_InitInfo initInfo{};
    initInfo.ApiVersion = VK_API_VERSION_1_2;
    initInfo.Instance = ctx.instance();
    initInfo.PhysicalDevice = ctx.physicalDevice();
    initInfo.Device = ctx.device();
    initInfo.QueueFamily = ctx.queueFamilies().graphicsFamily;
    initInfo.Queue = ctx.graphicsQueue();
    initInfo.DescriptorPool = m_descriptorPool;
    initInfo.MinImageCount = swapchain.imageCount();
    initInfo.ImageCount = swapchain.imageCount();
    initInfo.PipelineInfoMain.RenderPass = renderPass;
    initInfo.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    initInfo.CheckVkResultFn = checkVkResult;

    ImGui_ImplVulkan_Init(&initInfo);
}

void VkImGuiBackend::destroy(VkDevice device) {
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    if (m_descriptorPool) {
        vkDestroyDescriptorPool(device, m_descriptorPool, nullptr);
        m_descriptorPool = VK_NULL_HANDLE;
    }
}

void VkImGuiBackend::newFrame() {
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

void VkImGuiBackend::render(VkCommandBuffer cmd) {
    ImGui::Render();
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
}

} // namespace cosmico
