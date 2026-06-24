#pragma once
#include <cosmico/simulation/Simulation.h>
#include <cosmico/simulation/InitialConditions.h>
#include <cosmico/simulation/PMParams.h>
#include <volk.h>
#include <vk_mem_alloc.h>
#include <string>
#include <vector>

namespace cosmico {

class VkContext;

struct CameraConfig {
    float distance = 50.0f;
    float pitch    = 0.3f;
    float yaw      = 0.0f;
    float target[3] = {0.0f, 0.0f, 0.0f};
    float fov      = 60.0f;
    float nearPlane = 0.001f;
    float farPlane  = 10000.0f;
};

struct ScenarioEntry {
    std::string folderName;       // "cosmological"
    std::string title;            // "Cosmological Web"
    std::string description;
    InitialCondition initialCondition = InitialCondition::Sphere;
    std::string previewPath;
    VkDescriptorSet previewTexture = VK_NULL_HANDLE;
    bool loaded = false;
    SimulationParams params;      // merged: parent defaults + scenario overrides
    PMParams pmParams;            // PM/cosmology params (grid, box, comoving, …)
};

struct SimulationEntry {
    std::string folderName;       // "nbody-brute-force"
    std::string title;            // "N-Body Brute Force"
    std::string description;      // Full text
    std::string backendStr;       // "BruteForce"
    ComputeBackend backend;
    std::string configPath;       // Full path to config.json
    std::string previewPath;      // Full path to preview.png
    std::string graphPath;        // Full path to graph.json (may not exist)
    std::string papersDir;        // Full path to papers/ subfolder
    InitialCondition initialCondition = InitialCondition::Sphere;
    VkDescriptorSet previewTexture = VK_NULL_HANDLE;  // ImGui texture
    bool loaded = false;          // Preview texture loaded

    SimulationParams defaultParams;                // from config.json "params"
    PMParams pmDefaults;                            // PM/cosmology params from config.json
    CameraConfig cameraConfig;                     // from config.json "camera"
    std::vector<std::string> availableConditions;  // ["Cosmological", "Sphere"]
    std::vector<ScenarioEntry> scenarios;          // from scenarios/ subfolder
    bool hasScenarios() const { return !availableConditions.empty(); }
};

class SimulationCatalog {
public:
    void init(VkContext& ctx, const std::string& simulationsDir);
    void destroy(VkContext& ctx);
    void scanFolder();

    // Gallery rendering — returns true if a card was clicked
    bool renderGallery(int& selectedIndex);
    // Detail panel — sets launchRequested, editRequested, or backRequested
    // selectedScenarioIndex: -2 = nothing, -1 = New Scenario, >= 0 = saved scenario
    void renderDetail(int selectedIndex, bool& launchRequested,
                      bool& editRequested, bool& backRequested,
                      int& selectedScenarioIndex);

    const SimulationEntry& entry(int index) const;
    int entryCount() const;
    const ScenarioEntry* scenario(int simIdx, int scIdx) const;
    int scenarioCount(int simIdx) const;

private:
    void loadPreviewTexture(VkContext& ctx, SimulationEntry& entry);
    void loadPreviewTexture(VkContext& ctx, VkDescriptorSet& outTex, bool& outLoaded,
                            const std::string& path);
    void scanScenarios(SimulationEntry& entry, const std::string& scenariosDir);
    void renderScenarioGrid(const SimulationEntry& entry, int& selectedScenarioIndex);
    static ComputeBackend computeBackendFromString(const std::string& s);

    std::string m_simulationsDir;
    std::vector<SimulationEntry> m_entries;

    // Vulkan resources for preview textures
    VkSampler m_sampler = VK_NULL_HANDLE;
    VkContext* m_ctx = nullptr;

    struct TextureData {
        VkImage image = VK_NULL_HANDLE;
        VmaAllocation alloc = VK_NULL_HANDLE;
        VkImageView view = VK_NULL_HANDLE;
    };
    std::vector<TextureData> m_textures;
};

} // namespace cosmico
