function(compile_shaders TARGET SHADER_DIR OUTPUT_DIR)
    file(GLOB_RECURSE SHADER_FILES
        "${SHADER_DIR}/*.vert"
        "${SHADER_DIR}/*.frag"
        "${SHADER_DIR}/*.comp"
    )

    find_program(GLSLC glslc HINTS "$ENV{VULKAN_SDK}/Bin" "${Vulkan_GLSLC_EXECUTABLE}")
    if(NOT GLSLC)
        message(FATAL_ERROR "glslc not found. Install the Vulkan SDK.")
    endif()

    set(SPIRV_FILES "")

    foreach(SHADER ${SHADER_FILES})
        get_filename_component(SHADER_NAME ${SHADER} NAME)
        set(SPIRV_FILE "${OUTPUT_DIR}/${SHADER_NAME}.spv")

        add_custom_command(
            OUTPUT ${SPIRV_FILE}
            COMMAND ${CMAKE_COMMAND} -E make_directory "${OUTPUT_DIR}"
            COMMAND ${GLSLC} --target-env=vulkan1.2 -O ${SHADER} -o ${SPIRV_FILE}
            DEPENDS ${SHADER}
            COMMENT "Compiling shader: ${SHADER_NAME}"
            VERBATIM
        )

        list(APPEND SPIRV_FILES ${SPIRV_FILE})
    endforeach()

    add_custom_target(${TARGET}_shaders ALL DEPENDS ${SPIRV_FILES})
    add_dependencies(${TARGET} ${TARGET}_shaders)
endfunction()
