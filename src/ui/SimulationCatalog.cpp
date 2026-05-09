#ifndef NOMINMAX
#define NOMINMAX
#endif

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <cosmico/ui/SimulationCatalog.h>
#include <cosmico/vulkan/VkContext.h>

#include <nlohmann/json.hpp>
#include <imgui.h>
#include <imgui_impl_vulkan.h>

#include <filesystem>
#include <fstream>
#include <algorithm>
#include <cstdio>

namespace cosmico {

namespace {

// Truncate a string with an ellipsis so it fits within maxWidth at the
// given font size. Returns the original string if it already fits.
// Falls back to an empty string if even "..." doesn't fit.
std::string fitTextWithEllipsis(const std::string& text, float maxWidth, float fontSize) {
    ImFont* font = ImGui::GetFont();
    if (!font || text.empty()) return text;

    auto measure = [&](const char* begin, const char* end) {
        return font->CalcTextSizeA(fontSize, FLT_MAX, 0.0f, begin, end).x;
    };

    if (measure(text.c_str(), text.c_str() + text.size()) <= maxWidth) {
        return text;
    }

    static const char* ellipsis = "...";
    float ellipsisW = measure(ellipsis, ellipsis + 3);
    if (ellipsisW > maxWidth) return std::string();

    // Binary search for the longest prefix that fits with ellipsis appended.
    int lo = 0, hi = static_cast<int>(text.size());
    while (lo < hi) {
        int mid = (lo + hi + 1) / 2;
        float w = measure(text.c_str(), text.c_str() + mid);
        if (w + ellipsisW <= maxWidth) lo = mid;
        else hi = mid - 1;
    }
    return text.substr(0, lo) + ellipsis;
}

} // namespace

// ─── Backend string mapping ─────────────────────────────────────────

ComputeBackend SimulationCatalog::computeBackendFromString(const std::string& s) {
    if (s == "BruteForce") return ComputeBackend::BruteForce;
    if (s == "BarnesHut")  return ComputeBackend::BarnesHut;
    if (s == "PM")         return ComputeBackend::PM;
    if (s == "Inflation")  return ComputeBackend::Inflation;
    if (s == "CDT2D")      return ComputeBackend::CDT2D;
    if (s == "CDT3D")      return ComputeBackend::CDT3D;
    if (s == "NodeGraph")  return ComputeBackend::NodeGraph;
    return ComputeBackend::BruteForce;
}

// ─── Init / Destroy ─────────────────────────────────────────────────

void SimulationCatalog::init(VkContext& ctx, const std::string& simulationsDir) {
    m_simulationsDir = simulationsDir;
    m_ctx = &ctx;

    // Create a shared sampler for all preview textures
    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.maxAnisotropy = 1.0f;
    samplerInfo.maxLod = 1.0f;

    vkCreateSampler(ctx.device(), &samplerInfo, nullptr, &m_sampler);

    scanFolder();

    // Load preview textures for all entries and their scenarios
    for (auto& entry : m_entries) {
        loadPreviewTexture(ctx, entry);
        for (auto& sc : entry.scenarios) {
            loadPreviewTexture(ctx, sc.previewTexture, sc.loaded, sc.previewPath);
        }
    }
}

void SimulationCatalog::destroy(VkContext& ctx) {
    vkDeviceWaitIdle(ctx.device());

    // Free ImGui texture descriptors and Vulkan images
    for (auto& tex : m_textures) {
        if (tex.view) vkDestroyImageView(ctx.device(), tex.view, nullptr);
        if (tex.image) vmaDestroyImage(ctx.allocator(), tex.image, tex.alloc);
    }
    m_textures.clear();

    if (m_sampler) {
        vkDestroySampler(ctx.device(), m_sampler, nullptr);
        m_sampler = VK_NULL_HANDLE;
    }

    m_entries.clear();
    m_ctx = nullptr;
}

// ─── Scan simulation folders ────────────────────────────────────────

void SimulationCatalog::scanFolder() {
    m_entries.clear();
    namespace fs = std::filesystem;

    if (!fs::exists(m_simulationsDir) || !fs::is_directory(m_simulationsDir)) {
        fprintf(stderr, "[SimulationCatalog] simulations dir not found: %s\n",
                m_simulationsDir.c_str());
        return;
    }

    for (const auto& dirEntry : fs::directory_iterator(m_simulationsDir)) {
        if (!dirEntry.is_directory()) continue;

        auto configPath = dirEntry.path() / "config.json";
        if (!fs::exists(configPath)) continue;

        // Parse config.json
        std::ifstream ifs(configPath);
        if (!ifs.is_open()) continue;

        nlohmann::json config;
        try {
            ifs >> config;
        } catch (const nlohmann::json::exception& e) {
            fprintf(stderr, "[SimulationCatalog] Failed to parse %s: %s\n",
                    configPath.string().c_str(), e.what());
            continue;
        }

        SimulationEntry entry;
        entry.folderName = dirEntry.path().filename().string();
        entry.title = config.value("title", entry.folderName);
        entry.description = config.value("description", "");
        entry.backendStr = config.value("backend", "BruteForce");
        entry.backend = computeBackendFromString(entry.backendStr);
        std::string icStr = config.value("initialCondition", "Sphere");
        entry.initialCondition = initialConditionFromString(icStr);
        entry.configPath = configPath.string();

        auto previewPath = dirEntry.path() / "preview.png";
        entry.previewPath = previewPath.string();

        auto graphPath = dirEntry.path() / "graph.json";
        if (fs::exists(graphPath)) {
            entry.graphPath = graphPath.string();
        }

        auto papersPath = dirEntry.path() / "papers";
        if (fs::exists(papersPath)) {
            entry.papersDir = papersPath.string();
        }

        // Parse simulation params from config
        if (config.contains("params") && config["params"].is_object()) {
            auto& p = config["params"];
            if (p.contains("dt"))            entry.defaultParams.dt = p["dt"].get<float>();
            if (p.contains("G"))             entry.defaultParams.G = p["G"].get<float>();
            if (p.contains("softening"))     entry.defaultParams.softening = p["softening"].get<float>();
            if (p.contains("theta"))         entry.defaultParams.theta = p["theta"].get<float>();
            if (p.contains("particleCount")) entry.defaultParams.particleCount = p["particleCount"].get<int>();
        }

        // Parse camera configuration
        if (config.contains("camera") && config["camera"].is_object()) {
            auto& c = config["camera"];
            if (c.contains("distance"))  entry.cameraConfig.distance  = c["distance"].get<float>();
            if (c.contains("pitch"))     entry.cameraConfig.pitch     = c["pitch"].get<float>();
            if (c.contains("yaw"))       entry.cameraConfig.yaw       = c["yaw"].get<float>();
            if (c.contains("fov"))       entry.cameraConfig.fov       = c["fov"].get<float>();
            if (c.contains("near"))      entry.cameraConfig.nearPlane = c["near"].get<float>();
            if (c.contains("far"))       entry.cameraConfig.farPlane  = c["far"].get<float>();
            if (c.contains("target") && c["target"].is_array() && c["target"].size() >= 3) {
                entry.cameraConfig.target[0] = c["target"][0].get<float>();
                entry.cameraConfig.target[1] = c["target"][1].get<float>();
                entry.cameraConfig.target[2] = c["target"][2].get<float>();
            }
        }

        // Parse available initial conditions for scenario support
        if (config.contains("availableConditions") && config["availableConditions"].is_array()) {
            for (const auto& ic : config["availableConditions"])
                entry.availableConditions.push_back(ic.get<std::string>());
        }

        // Scan scenario sub-folders
        auto scenariosPath = dirEntry.path() / "scenarios";
        if (fs::exists(scenariosPath)) {
            scanScenarios(entry, scenariosPath.string());
        }

        m_entries.push_back(std::move(entry));
    }

    // Sort by backend order for consistent display
    std::sort(m_entries.begin(), m_entries.end(),
              [](const SimulationEntry& a, const SimulationEntry& b) {
                  return static_cast<int>(a.backend) < static_cast<int>(b.backend);
              });

    fprintf(stderr, "[SimulationCatalog] Found %zu simulations\n", m_entries.size());
}

// ─── Scan scenario sub-folders ──────────────────────────────────────

void SimulationCatalog::scanScenarios(SimulationEntry& entry, const std::string& scenariosDir) {
    namespace fs = std::filesystem;
    if (!fs::exists(scenariosDir) || !fs::is_directory(scenariosDir)) return;

    for (const auto& dirEntry : fs::directory_iterator(scenariosDir)) {
        if (!dirEntry.is_directory()) continue;

        auto configPath = dirEntry.path() / "config.json";
        if (!fs::exists(configPath)) continue;

        std::ifstream ifs(configPath);
        if (!ifs.is_open()) continue;

        nlohmann::json config;
        try {
            ifs >> config;
        } catch (const nlohmann::json::exception& e) {
            fprintf(stderr, "[SimulationCatalog] Failed to parse scenario %s: %s\n",
                    configPath.string().c_str(), e.what());
            continue;
        }

        ScenarioEntry sc;
        sc.folderName = dirEntry.path().filename().string();
        sc.title = config.value("title", sc.folderName);
        sc.description = config.value("description", "");
        std::string icStr = config.value("initialCondition", "Sphere");
        sc.initialCondition = initialConditionFromString(icStr);

        // Start with parent defaults, then override with scenario params
        sc.params = entry.defaultParams;
        if (config.contains("params") && config["params"].is_object()) {
            auto& p = config["params"];
            if (p.contains("dt"))            sc.params.dt = p["dt"].get<float>();
            if (p.contains("G"))             sc.params.G = p["G"].get<float>();
            if (p.contains("softening"))     sc.params.softening = p["softening"].get<float>();
            if (p.contains("theta"))         sc.params.theta = p["theta"].get<float>();
            if (p.contains("particleCount")) sc.params.particleCount = p["particleCount"].get<int>();
        }

        auto previewPath = dirEntry.path() / "preview.png";
        sc.previewPath = previewPath.string();

        entry.scenarios.push_back(std::move(sc));
    }

    // Sort scenarios by title for consistent display
    std::sort(entry.scenarios.begin(), entry.scenarios.end(),
              [](const ScenarioEntry& a, const ScenarioEntry& b) {
                  return a.title < b.title;
              });
}

// ─── Preview texture loading ────────────────────────────────────────

void SimulationCatalog::loadPreviewTexture(VkContext& ctx, SimulationEntry& entry) {
    loadPreviewTexture(ctx, entry.previewTexture, entry.loaded, entry.previewPath);
}

void SimulationCatalog::loadPreviewTexture(VkContext& ctx, VkDescriptorSet& outTex,
                                            bool& outLoaded, const std::string& path) {
    int w, h, channels;
    unsigned char* pixels = stbi_load(path.c_str(), &w, &h, &channels, 4);
    if (!pixels) {
        fprintf(stderr, "[SimulationCatalog] No preview: %s\n", path.c_str());
        outLoaded = false;
        return;
    }

    VkDeviceSize imageSize = static_cast<VkDeviceSize>(w) * h * 4;

    // Create staging buffer
    VkBufferCreateInfo stagingBufInfo{};
    stagingBufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    stagingBufInfo.size = imageSize;
    stagingBufInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

    VmaAllocationCreateInfo stagingAllocInfo{};
    stagingAllocInfo.usage = VMA_MEMORY_USAGE_CPU_ONLY;

    VkBuffer stagingBuffer;
    VmaAllocation stagingAlloc;
    vmaCreateBuffer(ctx.allocator(), &stagingBufInfo, &stagingAllocInfo,
                    &stagingBuffer, &stagingAlloc, nullptr);

    void* mapped;
    vmaMapMemory(ctx.allocator(), stagingAlloc, &mapped);
    memcpy(mapped, pixels, imageSize);
    vmaUnmapMemory(ctx.allocator(), stagingAlloc);
    stbi_image_free(pixels);

    // Create Vulkan image
    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    imageInfo.extent = { static_cast<uint32_t>(w), static_cast<uint32_t>(h), 1 };
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo imgAllocInfo{};
    imgAllocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

    TextureData tex{};
    vmaCreateImage(ctx.allocator(), &imageInfo, &imgAllocInfo,
                   &tex.image, &tex.alloc, nullptr);

    // Record commands: transition + copy + transition
    VkCommandBuffer cmd;
    VkCommandBufferAllocateInfo cmdAllocInfo{};
    cmdAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAllocInfo.commandBufferCount = 1;

    // We need a command pool — use a temporary one
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    poolInfo.queueFamilyIndex = ctx.queueFamilies().graphicsFamily;

    VkCommandPool tempPool;
    vkCreateCommandPool(ctx.device(), &poolInfo, nullptr, &tempPool);

    cmdAllocInfo.commandPool = tempPool;
    vkAllocateCommandBuffers(ctx.device(), &cmdAllocInfo, &cmd);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &beginInfo);

    // Transition: UNDEFINED → TRANSFER_DST
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = tex.image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.layerCount = 1;
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                         0, nullptr, 0, nullptr, 1, &barrier);

    // Copy buffer to image
    VkBufferImageCopy region{};
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.layerCount = 1;
    region.imageExtent = { static_cast<uint32_t>(w), static_cast<uint32_t>(h), 1 };

    vkCmdCopyBufferToImage(cmd, stagingBuffer, tex.image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    // Transition: TRANSFER_DST → SHADER_READ_ONLY
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
                         0, nullptr, 0, nullptr, 1, &barrier);

    vkEndCommandBuffer(cmd);

    // Submit and wait
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;

    vkQueueSubmit(ctx.graphicsQueue(), 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(ctx.graphicsQueue());

    vkDestroyCommandPool(ctx.device(), tempPool, nullptr);
    vmaDestroyBuffer(ctx.allocator(), stagingBuffer, stagingAlloc);

    // Create image view
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = tex.image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.layerCount = 1;

    vkCreateImageView(ctx.device(), &viewInfo, nullptr, &tex.view);

    // Register with ImGui
    outTex = ImGui_ImplVulkan_AddTexture(
        m_sampler, tex.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    outLoaded = true;

    m_textures.push_back(tex);
}

// ─── Gallery rendering ──────────────────────────────────────────────

bool SimulationCatalog::renderGallery(int& selectedIndex) {
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->Pos);
    ImGui::SetNextWindowSize(viewport->Size);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(30.0f, 30.0f));

    bool clicked = false;

    if (ImGui::Begin("##Gallery", nullptr,
            ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
            ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse |
            ImGuiWindowFlags_NoBringToFrontOnFocus)) {

        // Title
        ImGui::PushFont(nullptr); // Use default for now
        float titleScale = 1.5f;
        ImGui::SetWindowFontScale(titleScale);
        ImGui::TextColored(ImVec4(0.45f, 0.70f, 1.0f, 1.0f), "Cosmico");
        ImGui::SetWindowFontScale(1.0f);
        ImGui::PopFont();
        ImGui::SameLine();
        ImGui::TextDisabled("Cosmological Simulation Engine");
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();
        ImGui::Spacing();

        // Card grid
        float avail = ImGui::GetContentRegionAvail().x;
        constexpr float cardW = 260.0f;
        constexpr float cardH = 300.0f;
        constexpr float cardSpacing = 16.0f;
        constexpr float cardPadX = 14.0f;       // inner horizontal padding
        constexpr float imgW = cardW - 20.0f;
        constexpr float imgH = 160.0f;
        constexpr float titleSize = 16.0f;
        constexpr float descSize = 13.0f;
        constexpr float badgeReserve = 36.0f;   // bottom strip reserved for badge
        int cols = std::max(1, static_cast<int>((avail + cardSpacing) / (cardW + cardSpacing)));

        // Center the grid horizontally within the available area.
        float gridWidth = cols * cardW + (cols - 1) * cardSpacing;
        float rowLeftPad = std::max(0.0f, (avail - gridWidth) * 0.5f);
        float rowStartX = ImGui::GetCursorPosX();

        for (int i = 0; i < static_cast<int>(m_entries.size()); i++) {
            if (i % cols == 0) {
                ImGui::SetCursorPosX(rowStartX + rowLeftPad);
            } else {
                ImGui::SameLine(0.0f, cardSpacing);
            }

            const auto& e = m_entries[i];

            ImGui::PushID(i);
            ImGui::BeginGroup();

            // Card background
            ImVec2 cardStart = ImGui::GetCursorScreenPos();
            ImVec2 cardSize(cardW, cardH);
            ImVec2 cardEnd(cardStart.x + cardW, cardStart.y + cardH);

            // Invisible button for click detection over the whole card
            if (ImGui::InvisibleButton("##card", cardSize)) {
                selectedIndex = i;
                clicked = true;
            }

            bool hovered = ImGui::IsItemHovered();

            // Draw card background
            ImDrawList* dl = ImGui::GetWindowDrawList();
            ImU32 bgColor = hovered ? IM_COL32(40, 40, 70, 230) : IM_COL32(25, 25, 45, 200);
            ImU32 borderColor = hovered ? IM_COL32(100, 130, 255, 200) : IM_COL32(60, 60, 100, 150);
            dl->AddRectFilled(cardStart, cardEnd, bgColor, 8.0f);
            dl->AddRect(cardStart, cardEnd, borderColor, 8.0f, 0, hovered ? 2.0f : 1.0f);

            // Hard clip: nothing drawn after this can escape the card bounds.
            dl->PushClipRect(cardStart, cardEnd, true);

            // Preview image
            ImVec2 imgPos(cardStart.x + 10.0f, cardStart.y + 10.0f);
            ImVec2 imgEnd(imgPos.x + imgW, imgPos.y + imgH);

            if (e.loaded && e.previewTexture) {
                dl->AddImage(reinterpret_cast<ImTextureID>(e.previewTexture),
                             imgPos, imgEnd,
                             ImVec2(0, 0), ImVec2(1, 1));
            } else {
                // Placeholder gradient
                ImU32 gradTop = IM_COL32(40, 60, 100, 255);
                ImU32 gradBot = IM_COL32(20, 30, 50, 255);
                dl->AddRectFilledMultiColor(imgPos, imgEnd,
                    gradTop, gradTop, gradBot, gradBot);
                // Centered text
                const char* noPreview = "No Preview";
                ImVec2 textSize = ImGui::CalcTextSize(noPreview);
                dl->AddText(ImVec2(imgPos.x + (imgW - textSize.x) * 0.5f,
                                   imgPos.y + (imgH - textSize.y) * 0.5f),
                            IM_COL32(120, 120, 150, 200), noPreview);
            }

            // Title — single line with ellipsis if it would overflow.
            float textX = cardStart.x + cardPadX;
            float textY = cardStart.y + imgH + 20.0f;
            float textWidth = cardW - 2.0f * cardPadX;
            std::string fittedTitle = fitTextWithEllipsis(e.title, textWidth, titleSize);
            dl->AddText(nullptr, titleSize,
                        ImVec2(textX, textY),
                        IM_COL32(220, 220, 240, 255),
                        fittedTitle.c_str());

            // Description — wraps within textWidth and is clipped vertically
            // before the badge area, so long copy can never collide.
            float descY = textY + 24.0f;
            float descMaxY = cardEnd.y - badgeReserve;
            ImVec4 descClip(textX, descY, textX + textWidth, descMaxY);
            dl->AddText(nullptr, descSize,
                        ImVec2(textX, descY),
                        IM_COL32(160, 160, 180, 200),
                        e.description.c_str(),
                        nullptr,    // text_end NULL -> render until '\0'
                        textWidth,  // wrap_width
                        &descClip);

            // Backend badge
            float badgeY = cardEnd.y - 28.0f;
            const char* backendLabel = computeBackendName(e.backend);
            ImVec2 badgeSize = ImGui::CalcTextSize(backendLabel);
            float badgeX = cardEnd.x - badgeSize.x - 18.0f;
            dl->AddRectFilled(ImVec2(badgeX - 4.0f, badgeY - 2.0f),
                              ImVec2(badgeX + badgeSize.x + 4.0f, badgeY + badgeSize.y + 2.0f),
                              IM_COL32(50, 80, 130, 200), 4.0f);
            dl->AddText(ImVec2(badgeX, badgeY),
                        IM_COL32(140, 180, 255, 255), backendLabel);

            dl->PopClipRect();
            ImGui::EndGroup();
            ImGui::PopID();
        }
    }
    ImGui::End();
    ImGui::PopStyleVar(2);

    return clicked;
}

// ─── Detail panel ───────────────────────────────────────────────────

void SimulationCatalog::renderDetail(int selectedIndex, bool& launchRequested,
                                      bool& editRequested, bool& backRequested,
                                      int& selectedScenarioIndex) {
    selectedScenarioIndex = -2; // nothing clicked by default
    if (selectedIndex < 0 || selectedIndex >= static_cast<int>(m_entries.size())) return;

    const auto& e = m_entries[selectedIndex];
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->Pos);
    ImGui::SetNextWindowSize(viewport->Size);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(40.0f, 30.0f));

    if (ImGui::Begin("##Detail", nullptr,
            ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
            ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse |
            ImGuiWindowFlags_NoBringToFrontOnFocus)) {

        // Back button
        if (ImGui::ArrowButton("##back", ImGuiDir_Left)) {
            backRequested = true;
        }
        ImGui::SameLine();
        ImGui::Text("Back to Gallery");

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        // Layout: image left, info right
        float contentW = ImGui::GetContentRegionAvail().x;
        float imgW = std::min(400.0f, contentW * 0.4f);
        float imgH = imgW * 0.625f; // 16:10 aspect

        // Preview image
        if (e.loaded && e.previewTexture) {
            ImGui::Image(reinterpret_cast<ImTextureID>(e.previewTexture),
                         ImVec2(imgW, imgH));
        } else {
            ImGui::Dummy(ImVec2(imgW, imgH));
        }

        ImGui::SameLine(0.0f, 30.0f);

        // Info column
        ImGui::BeginGroup();

        ImGui::SetWindowFontScale(1.4f);
        ImGui::TextColored(ImVec4(0.45f, 0.70f, 1.0f, 1.0f), "%s", e.title.c_str());
        ImGui::SetWindowFontScale(1.0f);

        ImGui::Spacing();

        // Backend badge
        ImGui::TextDisabled("Backend:");
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(0.55f, 0.80f, 1.0f, 1.0f), "%s",
                           computeBackendName(e.backend));

        ImGui::Spacing();
        ImGui::Spacing();

        // Description
        ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + contentW - imgW - 60.0f);
        ImGui::TextWrapped("%s", e.description.c_str());
        ImGui::PopTextWrapPos();

        ImGui::EndGroup();

        ImGui::Spacing();
        ImGui::Spacing();
        ImGui::Spacing();

        if (e.hasScenarios()) {
            // ── Sims WITH scenarios: Edit Pipeline button + scenario grid ──

            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.55f, 0.35f, 0.15f, 0.85f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.65f, 0.45f, 0.25f, 0.95f));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.45f, 0.25f, 0.10f, 1.0f));
            if (ImGui::Button("Edit Pipeline", ImVec2(160.0f, 40.0f))) {
                editRequested = true;
            }
            ImGui::PopStyleColor(3);

            ImGui::Spacing();
            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            ImGui::SetWindowFontScale(1.15f);
            ImGui::TextColored(ImVec4(0.65f, 0.80f, 1.0f, 1.0f), "Scenarios");
            ImGui::SetWindowFontScale(1.0f);
            ImGui::Spacing();

            renderScenarioGrid(e, selectedScenarioIndex);
        } else {
            // ── Sims WITHOUT scenarios: Launch + Edit Pipeline ──

            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.15f, 0.45f, 0.85f, 0.85f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.25f, 0.55f, 0.95f, 0.95f));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.10f, 0.35f, 0.75f, 1.0f));
            if (ImGui::Button("Launch Simulation", ImVec2(200.0f, 40.0f))) {
                launchRequested = true;
            }
            ImGui::PopStyleColor(3);

            ImGui::SameLine(0.0f, 12.0f);
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.55f, 0.35f, 0.15f, 0.85f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.65f, 0.45f, 0.25f, 0.95f));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.45f, 0.25f, 0.10f, 1.0f));
            if (ImGui::Button("Edit Pipeline", ImVec2(160.0f, 40.0f))) {
                editRequested = true;
            }
            ImGui::PopStyleColor(3);
        }

        // Additional info below
        ImGui::Spacing();
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        if (!e.graphPath.empty()) {
            ImGui::TextDisabled("Node Graph: %s", e.graphPath.c_str());
        }
        if (!e.papersDir.empty()) {
            ImGui::TextDisabled("Papers: %s", e.papersDir.c_str());
        }
    }
    ImGui::End();
    ImGui::PopStyleVar(2);
}

// ─── Scenario grid ──────────────────────────────────────────────────

void SimulationCatalog::renderScenarioGrid(const SimulationEntry& entry,
                                            int& selectedScenarioIndex) {
    float avail = ImGui::GetContentRegionAvail().x;
    constexpr float cardW = 260.0f;
    constexpr float cardH = 300.0f;
    constexpr float cardSpacing = 16.0f;
    constexpr float cardPadX = 14.0f;
    constexpr float imgW = cardW - 20.0f;
    constexpr float imgH = 160.0f;
    constexpr float titleSize = 16.0f;
    constexpr float descSize = 13.0f;
    constexpr float badgeReserve = 36.0f;
    int cols = std::max(1, static_cast<int>((avail + cardSpacing) / (cardW + cardSpacing)));

    // Total cards: 1 ("+ New Scenario") + saved scenarios
    int totalCards = 1 + static_cast<int>(entry.scenarios.size());

    // Center the grid horizontally — match renderGallery layout.
    float gridWidth = cols * cardW + (cols - 1) * cardSpacing;
    float rowLeftPad = std::max(0.0f, (avail - gridWidth) * 0.5f);
    float rowStartX = ImGui::GetCursorPosX();

    for (int i = 0; i < totalCards; i++) {
        if (i % cols == 0) {
            ImGui::SetCursorPosX(rowStartX + rowLeftPad);
        } else {
            ImGui::SameLine(0.0f, cardSpacing);
        }

        ImGui::PushID(10000 + i); // Offset to avoid ID collision with gallery cards
        ImGui::BeginGroup();

        ImVec2 cardStart = ImGui::GetCursorScreenPos();
        ImVec2 cardSize(cardW, cardH);
        ImVec2 cardEnd(cardStart.x + cardW, cardStart.y + cardH);

        if (ImGui::InvisibleButton("##scenarioCard", cardSize)) {
            selectedScenarioIndex = i - 1; // -1 = New Scenario, 0+ = saved
        }
        bool hovered = ImGui::IsItemHovered();
        ImDrawList* dl = ImGui::GetWindowDrawList();

        ImVec2 imgPos(cardStart.x + 10.0f, cardStart.y + 10.0f);
        ImVec2 imgEnd(imgPos.x + imgW, imgPos.y + imgH);

        float textX = cardStart.x + cardPadX;
        float textY = cardStart.y + imgH + 20.0f;
        float textWidth = cardW - 2.0f * cardPadX;
        float descY = textY + 24.0f;
        float descMaxY = cardEnd.y - badgeReserve;
        ImVec4 descClip(textX, descY, textX + textWidth, descMaxY);

        if (i == 0) {
            // ── "+ New Scenario" card ──
            ImU32 bgColor = hovered ? IM_COL32(30, 55, 35, 230) : IM_COL32(20, 40, 25, 200);
            ImU32 borderColor = hovered ? IM_COL32(80, 200, 100, 200) : IM_COL32(50, 130, 70, 150);
            dl->AddRectFilled(cardStart, cardEnd, bgColor, 8.0f);
            dl->AddRect(cardStart, cardEnd, borderColor, 8.0f, 0, hovered ? 2.0f : 1.0f);

            dl->PushClipRect(cardStart, cardEnd, true);

            // Dashed border effect in image area
            ImU32 gradTop = IM_COL32(30, 60, 35, 255);
            ImU32 gradBot = IM_COL32(15, 35, 20, 255);
            dl->AddRectFilledMultiColor(imgPos, imgEnd,
                gradTop, gradTop, gradBot, gradBot);

            // Large "+" centered in image area
            const char* plusSign = "+";
            ImVec2 plusSize = ImGui::CalcTextSize(plusSign);
            float plusScale = 3.0f;
            ImVec2 scaledSize(plusSize.x * plusScale, plusSize.y * plusScale);
            dl->AddText(nullptr, 14.0f * plusScale,
                        ImVec2(imgPos.x + (imgW - scaledSize.x) * 0.5f,
                               imgPos.y + (imgH - scaledSize.y) * 0.5f),
                        IM_COL32(80, 200, 100, 220), plusSign);

            // Title (single line, ellipsis if needed)
            std::string fittedTitle = fitTextWithEllipsis("New Scenario", textWidth, titleSize);
            dl->AddText(nullptr, titleSize,
                        ImVec2(textX, textY),
                        IM_COL32(140, 230, 160, 255),
                        fittedTitle.c_str());

            // Description: default IC (wraps + clipped)
            std::string defaultIC = entry.availableConditions.empty()
                ? "default" : entry.availableConditions[0];
            std::string desc = "Launch with " + defaultIC + " initial conditions";
            dl->AddText(nullptr, descSize,
                        ImVec2(textX, descY),
                        IM_COL32(120, 180, 140, 200),
                        desc.c_str(),
                        nullptr,
                        textWidth,
                        &descClip);

            dl->PopClipRect();
        } else {
            // ── Saved scenario card ──
            const auto& sc = entry.scenarios[i - 1];

            ImU32 bgColor = hovered ? IM_COL32(40, 40, 70, 230) : IM_COL32(25, 25, 45, 200);
            ImU32 borderColor = hovered ? IM_COL32(100, 130, 255, 200) : IM_COL32(60, 60, 100, 150);
            dl->AddRectFilled(cardStart, cardEnd, bgColor, 8.0f);
            dl->AddRect(cardStart, cardEnd, borderColor, 8.0f, 0, hovered ? 2.0f : 1.0f);

            dl->PushClipRect(cardStart, cardEnd, true);

            // Preview image or gradient fallback
            if (sc.loaded && sc.previewTexture) {
                dl->AddImage(reinterpret_cast<ImTextureID>(sc.previewTexture),
                             imgPos, imgEnd,
                             ImVec2(0, 0), ImVec2(1, 1));
            } else {
                ImU32 gradTop = IM_COL32(40, 60, 100, 255);
                ImU32 gradBot = IM_COL32(20, 30, 50, 255);
                dl->AddRectFilledMultiColor(imgPos, imgEnd,
                    gradTop, gradTop, gradBot, gradBot);
                const char* noPreview = "No Preview";
                ImVec2 textSize = ImGui::CalcTextSize(noPreview);
                dl->AddText(ImVec2(imgPos.x + (imgW - textSize.x) * 0.5f,
                                   imgPos.y + (imgH - textSize.y) * 0.5f),
                            IM_COL32(120, 120, 150, 200), noPreview);
            }

            // Title
            std::string fittedTitle = fitTextWithEllipsis(sc.title, textWidth, titleSize);
            dl->AddText(nullptr, titleSize,
                        ImVec2(textX, textY),
                        IM_COL32(220, 220, 240, 255),
                        fittedTitle.c_str());

            // Description (wraps + clipped)
            dl->AddText(nullptr, descSize,
                        ImVec2(textX, descY),
                        IM_COL32(160, 160, 180, 200),
                        sc.description.c_str(),
                        nullptr,
                        textWidth,
                        &descClip);

            // IC badge (purple tint)
            float badgeY = cardEnd.y - 28.0f;
            const char* icLabel = initialConditionName(sc.initialCondition);
            ImVec2 badgeSize = ImGui::CalcTextSize(icLabel);
            float badgeX = cardEnd.x - badgeSize.x - 18.0f;
            dl->AddRectFilled(ImVec2(badgeX - 4.0f, badgeY - 2.0f),
                              ImVec2(badgeX + badgeSize.x + 4.0f, badgeY + badgeSize.y + 2.0f),
                              IM_COL32(80, 50, 130, 200), 4.0f);
            dl->AddText(ImVec2(badgeX, badgeY),
                        IM_COL32(180, 140, 255, 255), icLabel);

            dl->PopClipRect();
        }

        ImGui::EndGroup();
        ImGui::PopID();
    }
}

// ─── Accessors ──────────────────────────────────────────────────────

const SimulationEntry& SimulationCatalog::entry(int index) const {
    return m_entries[index];
}

int SimulationCatalog::entryCount() const {
    return static_cast<int>(m_entries.size());
}

const ScenarioEntry* SimulationCatalog::scenario(int simIdx, int scIdx) const {
    if (simIdx < 0 || simIdx >= static_cast<int>(m_entries.size())) return nullptr;
    const auto& scs = m_entries[simIdx].scenarios;
    if (scIdx < 0 || scIdx >= static_cast<int>(scs.size())) return nullptr;
    return &scs[scIdx];
}

int SimulationCatalog::scenarioCount(int simIdx) const {
    if (simIdx < 0 || simIdx >= static_cast<int>(m_entries.size())) return 0;
    return static_cast<int>(m_entries[simIdx].scenarios.size());
}

} // namespace cosmico
