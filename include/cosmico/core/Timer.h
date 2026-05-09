#pragma once
#include <chrono>

namespace cosmico {

class Timer {
public:
    void reset();
    void tick();

    float deltaTime() const { return m_dt; }
    float elapsedTime() const { return m_elapsed; }
    float fps() const { return m_dt > 0.0f ? 1.0f / m_dt : 0.0f; }

private:
    using Clock = std::chrono::high_resolution_clock;
    Clock::time_point m_last = Clock::now();
    float m_dt = 0.0f;
    float m_elapsed = 0.0f;
};

} // namespace cosmico
