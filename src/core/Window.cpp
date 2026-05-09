#include <cosmico/core/Window.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <stdexcept>

namespace cosmico {

Window::Window(uint32_t width, uint32_t height, const std::string& title)
    : m_width(width), m_height(height) {
    if (!glfwInit()) {
        throw std::runtime_error("Failed to initialize GLFW");
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API); // Vulkan — no OpenGL context
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

    m_window = glfwCreateWindow(static_cast<int>(width), static_cast<int>(height),
                                title.c_str(), nullptr, nullptr);
    if (!m_window) {
        glfwTerminate();
        throw std::runtime_error("Failed to create GLFW window");
    }

    glfwSetWindowUserPointer(m_window, this);
    glfwSetFramebufferSizeCallback(m_window, framebufferResizeCallback);
}

Window::~Window() {
    if (m_window) {
        glfwDestroyWindow(m_window);
    }
    glfwTerminate();
}

bool Window::shouldClose() const {
    return glfwWindowShouldClose(m_window);
}

void Window::pollEvents() {
    glfwPollEvents();
}

void Window::waitEvents() {
    glfwWaitEvents();
}

void Window::framebufferResizeCallback(GLFWwindow* window, int width, int height) {
    auto* self = static_cast<Window*>(glfwGetWindowUserPointer(window));
    self->m_width = static_cast<uint32_t>(width);
    self->m_height = static_cast<uint32_t>(height);
    self->m_resized = true;
    if (self->onResize) {
        self->onResize(self->m_width, self->m_height);
    }
}

} // namespace cosmico
