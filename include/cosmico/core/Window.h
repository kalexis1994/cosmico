#pragma once
#include <cstdint>
#include <string>
#include <functional>

struct GLFWwindow;

namespace cosmico {

class Window {
public:
    Window(uint32_t width, uint32_t height, const std::string& title);
    ~Window();

    Window(const Window&) = delete;
    Window& operator=(const Window&) = delete;

    bool shouldClose() const;
    void pollEvents();
    void waitEvents();

    GLFWwindow* handle() const { return m_window; }
    uint32_t width() const { return m_width; }
    uint32_t height() const { return m_height; }
    bool wasResized() const { return m_resized; }
    void resetResizedFlag() { m_resized = false; }

    std::function<void(uint32_t, uint32_t)> onResize;

private:
    static void framebufferResizeCallback(GLFWwindow* window, int width, int height);

    GLFWwindow* m_window = nullptr;
    uint32_t m_width;
    uint32_t m_height;
    bool m_resized = false;
};

} // namespace cosmico
