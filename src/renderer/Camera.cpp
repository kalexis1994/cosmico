#include <cosmico/renderer/Camera.h>
#include <cosmico/core/Input.h>
#include <glm/gtc/matrix_transform.hpp>
#include <GLFW/glfw3.h>
#include <algorithm>
#include <cmath>

namespace cosmico {

void Camera::init(float distance) {
    m_distance = distance;
    m_target = glm::vec3(0.0f);
    m_yaw = 0.0f;
    m_pitch = 0.3f;
}

void Camera::reset(float distance, float pitch, float yaw, glm::vec3 target,
                   float fov, float near, float far) {
    m_mode = CameraMode::Orbit;
    m_distance = distance;
    m_pitch = pitch;
    m_yaw = yaw;
    m_target = target;
    fovDegrees = fov;
    nearPlane = near;
    farPlane = far;
    orthographic = false;
}

void Camera::setMode(CameraMode m) {
    if (m == m_mode) return;

    if (m == CameraMode::Free) {
        // Transition orbit → free: place free camera at current orbit position
        m_freePos = position();
        // Orbit looks from pos toward target — derive free yaw/pitch
        glm::vec3 dir = glm::normalize(m_target - m_freePos);
        m_freeYaw   = std::atan2(dir.x, dir.z);
        m_freePitch = std::asin(std::clamp(dir.y, -1.0f, 1.0f));
    } else {
        // Transition free → orbit: place target ahead of free camera
        glm::vec3 dir(std::sin(m_freeYaw) * std::cos(m_freePitch),
                      std::sin(m_freePitch),
                      std::cos(m_freeYaw) * std::cos(m_freePitch));
        m_target = m_freePos + dir * m_distance;
        // Derive orbit yaw/pitch (camera looks from pos toward target)
        glm::vec3 offset = m_freePos - m_target; // points from target to camera
        float d = glm::length(offset);
        if (d > 0.001f) {
            m_distance = d;
            m_yaw   = std::atan2(offset.x, offset.z);
            m_pitch = std::asin(std::clamp(offset.y / d, -1.0f, 1.0f));
        }
    }
    m_mode = m;
}

void Camera::update(const Input& input, float dt, float viewportHeight,
                    float cursorFromCenterX, float cursorFromCenterY) {
    if (m_mode == CameraMode::Free)
        updateFree(input, dt);
    else
        updateOrbit(input, dt, viewportHeight, cursorFromCenterX, cursorFromCenterY);
}

// ── Orbit mode (unchanged) ──────────────────────────────────────────
void Camera::updateOrbit(const Input& input, float /*dt*/, float viewportHeight,
                         float cursorFromCenterX, float cursorFromCenterY) {
    // Orbit with right mouse button
    if (input.isMouseButtonDown(GLFW_MOUSE_BUTTON_RIGHT)) {
        glm::vec3 fixedPos = position();  // capture before rotating (anchored mode)
        m_yaw -= static_cast<float>(input.mouseDeltaX()) * orbitSpeed;
        m_pitch -= static_cast<float>(input.mouseDeltaY()) * orbitSpeed;
        m_pitch = std::clamp(m_pitch, -1.5f, 1.5f);
        if (fovZoom) {
            // Anchored view: rotate in place (look around) instead of orbiting
            // the target — move the target around the fixed camera position so
            // position() stays put while the view direction turns.
            float x = m_distance * std::cos(m_pitch) * std::sin(m_yaw);
            float y = m_distance * std::sin(m_pitch);
            float z = m_distance * std::cos(m_pitch) * std::cos(m_yaw);
            m_target = fixedPos - glm::vec3(x, y, z);
        }
    }

    // Camera basis vectors (shared by pan and zoom-towards-cursor)
    glm::vec3 pos = position();
    glm::vec3 forward = glm::normalize(m_target - pos);
    glm::vec3 right = glm::normalize(glm::cross(forward, glm::vec3(0, 1, 0)));
    glm::vec3 up = glm::cross(right, forward);

    // Pan with middle mouse button (disabled when translation is locked)
    if (!lockTranslate && input.isMouseButtonDown(GLFW_MOUSE_BUTTON_MIDDLE)) {
        float worldPerPixel;
        if (viewportHeight > 0.0f) {
            if (orthographic)
                worldPerPixel = orthoHeight / viewportHeight;
            else
                worldPerPixel = 2.0f * m_distance * std::tan(glm::radians(fovDegrees) * 0.5f) / viewportHeight;
        } else {
            worldPerPixel = panSpeed * m_distance;
        }

        m_target -= right * static_cast<float>(input.mouseDeltaX()) * worldPerPixel;
        m_target -= up    * static_cast<float>(input.mouseDeltaY()) * worldPerPixel;
    }

    // Zoom with scroll wheel — towards cursor
    float scroll = static_cast<float>(input.scrollDelta());
    if (scroll != 0.0f) {
        if (fovZoom) {
            // Field-of-view zoom: magnify without moving the camera (the
            // anchored expansion view keeps the camera fixed in space).
            fovDegrees -= scroll * zoomSpeed * fovDegrees * 0.05f;
            fovDegrees = std::clamp(fovDegrees, 2.0f, 110.0f);
        } else if (orthographic) {
            float oldOrtho = orthoHeight;
            orthoHeight -= scroll * zoomSpeed * (orthoHeight * 0.1f);
            orthoHeight = std::max(orthoHeight, 0.5f);

            if (viewportHeight > 0.0f) {
                float factor = 1.0f - orthoHeight / oldOrtho;
                float wpp = oldOrtho / viewportHeight;
                m_target += (right * cursorFromCenterX - up * cursorFromCenterY) * wpp * factor;
            }
        } else {
            float oldDist = m_distance;
            m_distance -= scroll * zoomSpeed * (m_distance * 0.1f);
            m_distance = std::max(m_distance, 0.1f);

            if (viewportHeight > 0.0f) {
                float factor = 1.0f - m_distance / oldDist;
                float wpp = 2.0f * oldDist * std::tan(glm::radians(fovDegrees) * 0.5f) / viewportHeight;
                m_target += (right * cursorFromCenterX - up * cursorFromCenterY) * wpp * factor;
            }
        }
    }
}

// ── Free (FPS) mode ─────────────────────────────────────────────────
void Camera::updateFree(const Input& input, float dt) {
    // Mouselook always active (cursor captured by Application)
    m_freeYaw   -= static_cast<float>(input.mouseDeltaX()) * freeLookSpeed;
    m_freePitch += static_cast<float>(input.mouseDeltaY()) * freeLookSpeed;
    m_freePitch  = std::clamp(m_freePitch, -1.5f, 1.5f);

    // Scroll wheel adjusts movement speed
    float scroll = static_cast<float>(input.scrollDelta());
    if (scroll != 0.0f) {
        freeMoveSpeed *= (scroll > 0.0f) ? 1.2f : (1.0f / 1.2f);
        freeMoveSpeed = std::clamp(freeMoveSpeed, 0.01f, 100000.0f);
    }

    // Forward follows where the camera is looking (includes pitch)
    glm::vec3 forward(std::sin(m_freeYaw) * std::cos(m_freePitch),
                      std::sin(m_freePitch),
                      std::cos(m_freeYaw) * std::cos(m_freePitch));
    glm::vec3 right = glm::normalize(glm::cross(forward, glm::vec3(0, 1, 0)));
    glm::vec3 worldUp(0.0f, 1.0f, 0.0f);

    float speed = freeMoveSpeed * dt;

    // Arrow keys + WASD
    if (input.isKeyDown(GLFW_KEY_W) || input.isKeyDown(GLFW_KEY_UP))
        m_freePos += forward * speed;
    if (input.isKeyDown(GLFW_KEY_S) || input.isKeyDown(GLFW_KEY_DOWN))
        m_freePos -= forward * speed;
    if (input.isKeyDown(GLFW_KEY_A) || input.isKeyDown(GLFW_KEY_LEFT))
        m_freePos -= right * speed;
    if (input.isKeyDown(GLFW_KEY_D) || input.isKeyDown(GLFW_KEY_RIGHT))
        m_freePos += right * speed;
    if (input.isKeyDown(GLFW_KEY_E))
        m_freePos -= worldUp * speed;
    if (input.isKeyDown(GLFW_KEY_Q))
        m_freePos += worldUp * speed;
}

// ── Shared queries ──────────────────────────────────────────────────
glm::vec3 Camera::position() const {
    if (m_mode == CameraMode::Free)
        return m_freePos;

    float x = m_distance * std::cos(m_pitch) * std::sin(m_yaw);
    float y = m_distance * std::sin(m_pitch);
    float z = m_distance * std::cos(m_pitch) * std::cos(m_yaw);
    return m_target + glm::vec3(x, y, z);
}

glm::vec3 Camera::forward() const {
    if (m_mode == CameraMode::Free)
        return glm::vec3(std::sin(m_freeYaw) * std::cos(m_freePitch),
                         std::sin(m_freePitch),
                         std::cos(m_freeYaw) * std::cos(m_freePitch));
    return glm::normalize(m_target - position());
}

glm::mat4 Camera::viewMatrix() const {
    if (m_mode == CameraMode::Free) {
        glm::vec3 dir(std::sin(m_freeYaw) * std::cos(m_freePitch),
                      std::sin(m_freePitch),
                      std::cos(m_freeYaw) * std::cos(m_freePitch));
        return glm::lookAt(m_freePos, m_freePos + dir, glm::vec3(0.0f, 1.0f, 0.0f));
    }
    return glm::lookAt(position(), m_target, glm::vec3(0.0f, 1.0f, 0.0f));
}

glm::mat4 Camera::projectionMatrix(float aspect) const {
    if (orthographic) {
        float halfH = orthoHeight * 0.5f;
        float halfW = halfH * aspect;
        return glm::ortho(-halfW, halfW, -halfH, halfH, nearPlane, farPlane);
    }
    return glm::perspective(glm::radians(fovDegrees), aspect, nearPlane, farPlane);
}

} // namespace cosmico
