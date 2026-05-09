#include <cosmico/core/Input.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

namespace cosmico {

double Input::s_scrollAccum = 0.0;

void Input::init(GLFWwindow* window) {
    m_window = window;
    glfwSetWindowUserPointer(window, this);
    glfwSetScrollCallback(window, scrollCallback);
    glfwGetCursorPos(window, &m_mouseX, &m_mouseY);
    m_prevMouseX = m_mouseX;
    m_prevMouseY = m_mouseY;
}

void Input::update() {
    m_keyPrevious = m_keyCurrent;
    m_buttonPrevious = m_buttonCurrent;

    for (int i = 0; i < MAX_KEYS; i++) {
        m_keyCurrent[static_cast<size_t>(i)] = glfwGetKey(m_window, i) == GLFW_PRESS;
    }
    for (int i = 0; i < MAX_BUTTONS; i++) {
        m_buttonCurrent[static_cast<size_t>(i)] = glfwGetMouseButton(m_window, i) == GLFW_PRESS;
    }

    m_prevMouseX = m_mouseX;
    m_prevMouseY = m_mouseY;
    glfwGetCursorPos(m_window, &m_mouseX, &m_mouseY);
    m_mouseDX = m_mouseX - m_prevMouseX;
    m_mouseDY = m_mouseY - m_prevMouseY;

    m_scrollDY = s_scrollAccum;
    s_scrollAccum = 0.0;
}

bool Input::isKeyDown(int key) const {
    return key >= 0 && key < MAX_KEYS && m_keyCurrent[static_cast<size_t>(key)];
}

bool Input::isKeyPressed(int key) const {
    return key >= 0 && key < MAX_KEYS &&
           m_keyCurrent[static_cast<size_t>(key)] && !m_keyPrevious[static_cast<size_t>(key)];
}

bool Input::isMouseButtonDown(int button) const {
    return button >= 0 && button < MAX_BUTTONS && m_buttonCurrent[static_cast<size_t>(button)];
}

bool Input::isMouseButtonPressed(int button) const {
    return button >= 0 && button < MAX_BUTTONS &&
           m_buttonCurrent[static_cast<size_t>(button)] && !m_buttonPrevious[static_cast<size_t>(button)];
}

void Input::scrollCallback(GLFWwindow* /*window*/, double /*xoffset*/, double yoffset) {
    s_scrollAccum += yoffset;
}

} // namespace cosmico
