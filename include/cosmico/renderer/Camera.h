#pragma once
#include <glm/glm.hpp>

namespace cosmico {

class Input;

enum class CameraMode { Orbit, Free };

class Camera {
public:
    void init(float distance = 50.0f);
    void reset(float distance, float pitch, float yaw, glm::vec3 target,
               float fov, float near, float far);
    void update(const Input& input, float dt, float viewportHeight = 0.0f,
                float cursorFromCenterX = 0.0f, float cursorFromCenterY = 0.0f);

    glm::mat4 viewMatrix() const;
    glm::mat4 projectionMatrix(float aspect) const;

    float nearPlane = 0.001f;
    float farPlane = 10000.0f;
    float fovDegrees = 60.0f;
    float orbitSpeed = 0.005f;
    float zoomSpeed = 2.0f;
    float panSpeed = 0.01f;
    float freeMoveSpeed = 10.0f;
    float freeLookSpeed = 0.003f;

    bool orthographic = false;
    float orthoHeight = 50.0f;

    // When true, the scroll wheel changes the field of view (visual zoom)
    // instead of the dolly distance — used by the anchored expansion view,
    // where the camera must stay fixed in space.
    bool fovZoom = false;

    CameraMode mode() const { return m_mode; }

    // Carry the free-fly camera along with the cosmological expansion: scale its
    // position about the origin by the per-frame ratio a_now/a_prev so the flow
    // moves it from wherever it currently is.
    void rideExpansion(float ratio) { if (m_mode == CameraMode::Free) m_freePos *= ratio; }
    void setMode(CameraMode m);

    glm::vec3 position() const;
    glm::vec3 forward() const;
    glm::vec3 target() const { return m_target; }

private:
    void updateOrbit(const Input& input, float dt, float viewportHeight,
                     float cursorFromCenterX, float cursorFromCenterY);
    void updateFree(const Input& input, float dt);

    CameraMode m_mode = CameraMode::Orbit;

    // Orbit state
    glm::vec3 m_target{0.0f};
    float m_distance = 50.0f;
    float m_yaw = 0.0f;
    float m_pitch = 0.3f;

    // Free state
    glm::vec3 m_freePos{0.0f, 0.0f, 50.0f};
    float m_freeYaw = 3.14159265f;   // looking toward -Z (toward origin)
    float m_freePitch = 0.0f;
};

} // namespace cosmico
