cmake_minimum_required(VERSION 3.25)
project(cosmico VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ─── CUDA configuration ──────────────────────────────────────────────
set(CMAKE_CUDA_ARCHITECTURES 75 86 89)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# CUDA 12.1 + MSVC 14.44: inject compat flags via env so the CMake compiler
# ID step also gets them (VS AddCudaCompileDeps runs cl.exe separately).
set(ENV{NVCC_PREPEND_FLAGS} "-allow-unsupported-compiler -Xcompiler /D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH")
set(CMAKE_CUDA_FLAGS "-allow-unsupported-compiler" CACHE STRING "" FORCE)
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)
message(STATUS "CUDA Toolkit: ${CUDAToolkit_VERSION} at ${CUDAToolkit_TARGET_DIR}")

# ─── Vulkan SDK paths ───────────────────────────────────────────────
# Detect VULKAN_SDK from environment or standard locations
if(NOT DEFINED ENV{VULKAN_SDK})
    file(GLOB VULKAN_SDK_CANDIDATES "C:/VulkanSDK/*")
    list(SORT VULKAN_SDK_CANDIDATES ORDER DESCENDING)
    if(VULKAN_SDK_CANDIDATES)
        list(GET VULKAN_SDK_CANDIDATES 0 VULKAN_SDK_PATH)
        set(ENV{VULKAN_SDK} "${VULKAN_SDK_PATH}")
    endif()
endif()
set(VULKAN_SDK_PATH "$ENV{VULKAN_SDK}")
message(STATUS "Vulkan SDK: ${VULKAN_SDK_PATH}")

# ─── GLFW via FetchContent ──────────────────────────────────────────
include(FetchContent)

FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG        3.4
    GIT_SHALLOW    TRUE
)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(glfw)

# ─── Dear ImGui (docking branch) ────────────────────────────────────
FetchContent_Declare(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG        docking
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(imgui)

# Build ImGui as a library with Vulkan + GLFW backends
add_library(imgui_lib STATIC
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_demo.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_vulkan.cpp
)
target_include_directories(imgui_lib PUBLIC
    ${imgui_SOURCE_DIR}
    ${imgui_SOURCE_DIR}/backends
)
# ImGui Vulkan backend uses Volk for dynamic loading
# Use 32-bit draw indices to avoid 64K vertex limit crashes with font scaling
target_compile_definitions(imgui_lib PUBLIC
    IMGUI_IMPL_VULKAN_USE_VOLK
    IMGUI_IMPL_VULKAN_VOLK_FILENAME=<volk.h>
    "ImDrawIdx=unsigned int"
)
target_include_directories(imgui_lib PUBLIC
    "${VULKAN_SDK_PATH}/Include"
    "${VULKAN_SDK_PATH}/Include/Volk"
)
target_link_libraries(imgui_lib PUBLIC glfw)

# ─── nlohmann/json (JSON parsing for simulation configs) ─────────────
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
    GIT_SHALLOW    TRUE
)
set(JSON_BuildTests OFF CACHE BOOL "" FORCE)
set(JSON_Install OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(nlohmann_json)

# ─── stb (stb_image for preview texture loading) ────────────────────
# Header-only with no CMakeLists.txt — MakeAvailable just populates.
FetchContent_Declare(
    stb
    GIT_REPOSITORY https://github.com/nothings/stb.git
    GIT_TAG        master
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(stb)

# ─── imnodes (node editor extension for ImGui) ───────────────────────
# Its CMakeLists.txt calls find_package(imgui) which we don't satisfy —
# point SOURCE_SUBDIR at a path that does not exist so MakeAvailable
# populates without running add_subdirectory.
FetchContent_Declare(
    imnodes
    GIT_REPOSITORY https://github.com/Nelarius/imnodes.git
    GIT_TAG        master
    GIT_SHALLOW    TRUE
    SOURCE_SUBDIR  _do_not_configure_
)
FetchContent_MakeAvailable(imnodes)

add_library(imnodes_lib STATIC
    ${imnodes_SOURCE_DIR}/imnodes.cpp
)
target_include_directories(imnodes_lib PUBLIC
    ${imnodes_SOURCE_DIR}
)
target_link_libraries(imnodes_lib PUBLIC imgui_lib)

# ─── Cosmico C++ sources ────────────────────────────────────────────
file(GLOB_RECURSE COSMICO_SOURCES
    "src/*.cpp"
)

# ─── Cosmico CUDA sources ───────────────────────────────────────────
file(GLOB_RECURSE COSMICO_CUDA_SOURCES
    "src/*.cu"
    "cuda/*.cu"
)

add_executable(${PROJECT_NAME} ${COSMICO_SOURCES} ${COSMICO_CUDA_SOURCES})

# Allow CUDA files to include project headers
set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_include_directories(${PROJECT_NAME} PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/cuda"
    "${VULKAN_SDK_PATH}/Include"
    "${VULKAN_SDK_PATH}/Include/Volk"
    "${VULKAN_SDK_PATH}/Include/vma"
    "${VULKAN_SDK_PATH}/Include/glm"
    "${stb_SOURCE_DIR}"
)

target_compile_definitions(${PROJECT_NAME} PRIVATE
    VK_NO_PROTOTYPES                         # Volk loads everything dynamically
    VK_USE_PLATFORM_WIN32_KHR                # Win32 Vulkan extensions (external memory/semaphore)
    GLM_FORCE_RADIANS
    GLM_FORCE_DEPTH_ZERO_TO_ONE              # Vulkan depth range
    COSMICO_SHADER_DIR="${CMAKE_CURRENT_BINARY_DIR}/shaders"
    COSMICO_SIMULATIONS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/simulations"
    COSMICO_RESOURCES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/resources"
    _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH # CUDA 12.1 + MSVC 14.44 compat
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    glfw
    imgui_lib
    imnodes_lib
    nlohmann_json::nlohmann_json
    CUDA::cuda_driver
    CUDA::cudart
    CUDA::cufft
    CUDA::curand
)

# MSVC specific
if(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/W4 /permissive- /utf-8>
        $<$<COMPILE_LANGUAGE:CXX>:/external:anglebrackets /external:W0>
    )
else()
    target_compile_options(${PROJECT_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
    )
endif()

# ─── Copy CUDA runtime DLLs next to executable ────────────────────────
# Ensures the app runs on machines without CUDA Toolkit installed.
if(WIN32)
    set(CUDA_BIN_DIR "${CUDAToolkit_BIN_DIR}")
    set(CUDA_DLLS
        "${CUDA_BIN_DIR}/cudart64_12.dll"
        "${CUDA_BIN_DIR}/cufft64_11.dll"
        "${CUDA_BIN_DIR}/curand64_10.dll"
    )
    foreach(DLL ${CUDA_DLLS})
        if(EXISTS "${DLL}")
            add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${DLL}" "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
                COMMENT "Copying ${DLL}"
            )
        endif()
    endforeach()
endif()

# ─── Copy runtime data next to executable (portable distribution) ──────
# Shaders (compiled SPV files)
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_CURRENT_BINARY_DIR}/shaders"
        "$<TARGET_FILE_DIR:${PROJECT_NAME}>/shaders"
    COMMENT "Copying shaders/"
)

# Simulations (gallery configs, preview images, per-simulation papers)
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_CURRENT_SOURCE_DIR}/simulations"
        "$<TARGET_FILE_DIR:${PROJECT_NAME}>/simulations"
    COMMENT "Copying simulations/"
)

# Resources (planet textures, etc.)
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_CURRENT_SOURCE_DIR}/resources"
        "$<TARGET_FILE_DIR:${PROJECT_NAME}>/resources"
    COMMENT "Copying resources/"
)

# ─── Shader compilation ─────────────────────────────────────────────
include(cmake/CompileShaders.cmake)
compile_shaders(${PROJECT_NAME}
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders"
    "${CMAKE_CURRENT_BINARY_DIR}/shaders"
)
