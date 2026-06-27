#include <cosmico/core/Application.h>
#include <cosmico/renderer/Camera.h>
#include <cosmico/renderer/ParticleRenderer.h>
#include <cosmico/renderer/VolumeRenderer.h>
#include <cosmico/renderer/CMBRenderer.h>
#include <cosmico/renderer/CDT2DRenderer.h>
#include <cosmico/renderer/CDT3DRenderer.h>
#include <cosmico/renderer/OffscreenTarget.h>
#include <cosmico/renderer/PlanetTextures.h>
#include <cosmico/simulation/Simulation.h>
#include <cosmico/simulation/PMCompute.h>
#include <cosmico/simulation/InitialConditions.h>
#include <cosmico/simulation/ParticleSystem.h>
#include <cosmico/simulation/InflationCompute.h>
#include <cosmico/simulation/CDT2DCompute.h>
#include <cosmico/simulation/CDT3DCompute.h>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <cmath>
#include <cstring>

namespace cosmico {

void Application::renderRunningState(VkCommandBuffer cmd) {
    if (!m_simulation) return;

    bool usingInflation = (m_simulation->backend() == ComputeBackend::Inflation);
    bool usingCDT2D = (m_simulation->backend() == ComputeBackend::CDT2D);
    bool usingCDT3D = (m_simulation->backend() == ComputeBackend::CDT3D);
    bool zeldovichMode = usingInflation && m_simulation->zeldovichActive();

    // Compute camera matrices using actual offscreen render target size
    float aspect = static_cast<float>(m_offscreenTarget->extent().width) /
                   static_cast<float>(m_offscreenTarget->extent().height);
    glm::mat4 view = m_camera->viewMatrix();
    glm::mat4 proj = m_camera->projectionMatrix(aspect);
    glm::mat4 viewProj = proj * view;
    glm::mat4 invViewProj = glm::inverse(viewProj);
    glm::vec3 camPos = m_camera->position();

    float visualScale = 1.0f;
    if (usingInflation && !zeldovichMode) {
        const InflationStateData* infState = m_simulation->inflationState();
        float scaleFactor = infState ? (float)infState->scaleFactor : 1.0f;
        visualScale = std::pow(scaleFactor, 1.0f / 3.0f);
        float maxVisualScale = m_camera->farPlane * 0.3f /
                               (m_simulation->inflationParams().boxSize * 0.5f);
        if (visualScale > maxVisualScale) visualScale = maxVisualScale;
    }

    float boxHalf = m_simulation->inflationParams().boxSize * 0.5f * visualScale;

    // Volume rendering
    bool volumeReady = usingInflation && m_simulation->inflationParams().showVolume;
    if (volumeReady && m_volumeRenderer && m_volumeRenderer->isReady()) {
        VolumeRenderPushConstants vpc{};
        memcpy(vpc.invViewProj, glm::value_ptr(invViewProj), sizeof(float) * 16);
        vpc.cameraPos[0] = camPos.x;
        vpc.cameraPos[1] = camPos.y;
        vpc.cameraPos[2] = camPos.z;
        vpc.cameraPos[3] = 0.0f;
        vpc.volumeMin[0] = -boxHalf;
        vpc.volumeMin[1] = -boxHalf;
        vpc.volumeMin[2] = -boxHalf;
        vpc.volumeMin[3] = 0.0f;
        vpc.volumeMax[0] = boxHalf;
        vpc.volumeMax[1] = boxHalf;
        vpc.volumeMax[2] = boxHalf;
        vpc.volumeMax[3] = 0.0f;
        vpc.opacityScale = m_simulation->inflationParams().volumeOpacity /
                           (visualScale * visualScale);
        int numSteps = m_simulation->inflationParams().volumeSteps;
        float totalDist = boxHalf * 2.0f * 1.732f;
        vpc.stepSize = totalDist / (float)numSteps;
        vpc.numSteps = numSteps;
        vpc.padding = 0.0f;
        m_volumeRenderer->draw(cmd, vpc);
    }

    // CMB sphere
    bool cmbReady = usingInflation && !zeldovichMode && m_simulation->inflationParams().showCMB;
    if (cmbReady && m_cmbRenderer && m_cmbRenderer->isReady()) {
        const auto& iparams = m_simulation->inflationParams();
        float sphereRadius = iparams.cmbRadius * iparams.boxSize * visualScale;
        CMBRenderPushConstants cpc{};
        memcpy(cpc.invViewProj, glm::value_ptr(invViewProj), sizeof(float) * 16);
        cpc.cameraPos[0] = camPos.x;
        cpc.cameraPos[1] = camPos.y;
        cpc.cameraPos[2] = camPos.z;
        cpc.cameraPos[3] = 0.0f;
        cpc.sphereCenter[0] = 0.0f;
        cpc.sphereCenter[1] = 0.0f;
        cpc.sphereCenter[2] = 0.0f;
        cpc.sphereCenter[3] = sphereRadius;
        cpc.contrastScale = iparams.cmbContrast;
        cpc.opacity = iparams.cmbOpacity;
        cpc.padding[0] = 0.0f;
        cpc.padding[1] = 0.0f;
        m_cmbRenderer->draw(cmd, cpc);
    }

    // CDT2D mesh
    if (usingCDT2D) {
        const CDT2DStateData* cdtState = m_simulation->cdt2dState();
        if (cdtState && !cdtState->vertexData.empty()) {
            m_cdt2dRenderer->updateMesh(cdtState->vertexData.data(), cdtState->triangleCount);
            CDT2DRenderPushConstants cpc{};
            memcpy(cpc.viewProj, glm::value_ptr(viewProj), sizeof(float) * 16);
            cpc.meshScale = m_simulation->cdt2dParams().meshScale;
            cpc.showWireframe = m_simulation->cdt2dParams().showWireframe ? 1 : 0;
            m_cdt2dRenderer->draw(cmd, cpc);
        }
    }

    // CDT3D mesh
    if (usingCDT3D) {
        const CDT3DStateData* cdt3dState = m_simulation->cdt3dState();
        if (cdt3dState && !cdt3dState->vertexData.empty()) {
            m_cdt3dRenderer->updateMesh(cdt3dState->vertexData.data(), cdt3dState->triangleCount);
            CDT3DRenderPushConstants cpc{};
            memcpy(cpc.viewProj, glm::value_ptr(viewProj), sizeof(float) * 16);
            cpc.lightDir[0] = 0.577f;
            cpc.lightDir[1] = 0.577f;
            cpc.lightDir[2] = 0.577f;
            cpc.lightDir[3] = m_simulation->cdt3dParams().lightIntensity;
            cpc.meshScale = m_simulation->cdt3dParams().meshScale;
            cpc.showWireframe = m_simulation->cdt3dParams().showWireframe ? 1 : 0;
            m_cdt3dRenderer->draw(cmd, cpc);
        }
    }

    // Particles
    if (!usingInflation && !usingCDT2D && !usingCDT3D && m_particleRenderer) {
        ParticleRenderPushConstants pc{};
        memcpy(pc.viewProj, glm::value_ptr(viewProj), sizeof(float) * 16);
        pc.pointScale = static_cast<float>(m_offscreenTarget->extent().height) / 2.0f;
        pc.hasPlanetTextures = m_planetTextures ? 1.0f : 0.0f;
        pc.tanHalfFov = std::tan(glm::radians(m_camera->fovDegrees) * 0.5f);
        pc.aspectRatio = static_cast<float>(m_offscreenTarget->extent().width)
                       / static_cast<float>(m_offscreenTarget->extent().height);

        // Camera basis vectors for world-space normal reconstruction
        glm::vec3 fwd = m_camera->forward();
        glm::vec3 right = glm::normalize(glm::cross(fwd, glm::vec3(0.0f, 1.0f, 0.0f)));
        glm::vec3 up = glm::cross(right, fwd);
        pc.camRight[0] = right.x; pc.camRight[1] = right.y;
        pc.camRight[2] = right.z; pc.camRight[3] = camPos.x;
        pc.camUp[0] = up.x; pc.camUp[1] = up.y;
        pc.camUp[2] = up.z; pc.camUp[3] = camPos.y;
        pc.camForward[0] = fwd.x; pc.camForward[1] = fwd.y;
        pc.camForward[2] = fwd.z; pc.camForward[3] = camPos.z;
        pc.simTime = static_cast<float>(m_simTime);
        pc.showDarkMatter = m_simulation->params().showDarkMatter ? 1.0f : 0.0f;
        // Density-luminosity shading: only the PM backend writes the per-particle
        // overdensity the shader needs; others keep the legacy speed coloring.
        pc.lumStrength = (m_simulation->backend() == ComputeBackend::PM) ? 1.0f : 0.0f;

        // Physical-coordinate view (PM only): expand positions by a(t)/aInit so
        // the box inflates and structure recedes; otherwise keep the comoving frame.
        float coordScale = 1.0f;
        if (m_simulation->backend() == ComputeBackend::PM) {
            const PMParams& pmp = m_simulation->pmParams();
            const PMStateData* pmSt = m_simulation->pmState();
            if (pmp.physicalView && pmSt && pmSt->scaleFactor > 0.0 && pmp.aInit > 1e-6f) {
                coordScale = static_cast<float>(pmSt->scaleFactor / pmp.aInit);
                // The real expansion (up to ~×500) flings the web past the far
                // plane into the dark. Cap the on-screen scaling so the voids
                // visibly open up but the structure stays framed and lit.
                if (coordScale > 2.5f) coordScale = 2.5f;
            }
        }
        pc.coordScale = coordScale;

        VkDescriptorSet texSet = m_planetTextures
            ? m_planetTextures->descriptorSet() : VK_NULL_HANDLE;
        m_particleRenderer->draw(cmd,
            m_simulation->particleSystem().graphicsDescriptorSet(),
            m_simulation->particleSystem().particleCount(),
            pc, texSet, sphereBodyCount(m_pendingIC));
    }
}

} // namespace cosmico
