#include <cosmico/core/Application.h>
#include <cstdio>
#include <stdexcept>

int main() {
    try {
        cosmico::Application app;
        app.run();
    } catch (const std::exception& e) {
        fprintf(stderr, "[Cosmico] Fatal error: %s\n", e.what());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
