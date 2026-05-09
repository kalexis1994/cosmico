#pragma once
#include <array>

struct GLFWwindow;

namespace cosmico {

class Input {
public:
    void init(GLFWwindow* window);
    void update();

    bool isKeyDown(int key) const;
    bool isKeyPressed(int key) const;

    bool isMouseButtonDown(int button) const;
    bool isMouseButtonPressed(int button) const;

    double mouseX() const { return m_mouseX; }
    double mouseY() const { return m_mouseY; }
    double mouseDeltaX() const { return m_mouseDX; }
    double mouseDeltaY() const { return m_mouseDY; }
    double scrollDelta() const { return m_scrollDY; }

private:
    static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);

    GLFWwindow* m_window = nullptr;

    static constexpr int MAX_KEYS = 512;
    static constexpr int MAX_BUTTONS = 8;

    std::array<bool, MAX_KEYS> m_keyCurrent{};
    std::array<bool, MAX_KEYS> m_keyPrevious{};
    std::array<bool, MAX_BUTTONS> m_buttonCurrent{};
    std::array<bool, MAX_BUTTONS> m_buttonPrevious{};

    double m_mouseX = 0, m_mouseY = 0;
    double m_mouseDX = 0, m_mouseDY = 0;
    double m_prevMouseX = 0, m_prevMouseY = 0;
    double m_scrollDY = 0;
    static double s_scrollAccum;
};

} // namespace cosmico
