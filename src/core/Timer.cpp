#include <cosmico/core/Timer.h>

namespace cosmico {

void Timer::reset() {
    m_last = Clock::now();
    m_dt = 0.0f;
    m_elapsed = 0.0f;
}

void Timer::tick() {
    auto now = Clock::now();
    m_dt = std::chrono::duration<float>(now - m_last).count();
    m_last = now;
    m_elapsed += m_dt;

    // Clamp to avoid spiral of death after breakpoints/pauses
    if (m_dt > 0.1f) m_dt = 0.1f;
}

} // namespace cosmico
