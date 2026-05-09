#pragma once
#include <string>

namespace cosmico {

// Resolve a path relative to the executable directory.
// Falls back to the compile-time path if the relative one doesn't exist.
// Definition lives in src/core/Application.cpp.
std::string resolvePathNextToExe(const std::string& relativePath,
                                  const std::string& fallback);

} // namespace cosmico
