cmake_minimum_required(VERSION 3.25)
project(AVX512_Heterogeneous_Engine CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

# ================================================================================================
# CYPRESS COVE (i5-11400) ARCHITECTURAL COMPILER FLAGS
# ================================================================================================
set(AVX512_FLAGS
    -march=rocketlake
    -mavx512f
    -mavx512vl
    -mavx512bw
    -mavx512dq
    -mavx512vnni
    -mavx512vbmi
    -mavx512vbmi2
    -mavx512bitalg
    -mavx512vpopcntdq
    -mgfni
    -mfma
    -O3
    -flto=auto
    -fno-rtti
    -fno-math-errno
    -fno-trapping-math
    -Wall
    -Wextra
    -Wno-unknown-pragmas
    -Wno-unused-parameter
    -Wno-missing-field-initializers
)

add_compile_options(${AVX512_FLAGS})
add_link_options(-flto=auto -pthread)

include_directories(include)

# ================================================================================================
# DEPENDENCY RESOLUTION (Vulkan, Threads, Windowing, X11 Compositing)
# ================================================================================================
find_package(Vulkan REQUIRED)
find_package(Threads REQUIRED)

# ================================================================================================
# AUTOMATED SPIR-V SHADER COMPILATION
# ================================================================================================
find_program(GLSLC_EXECUTABLE glslc)
if (GLSLC_EXECUTABLE)
    file(GLOB SHADER_SOURCES "shaders/*.vert" "shaders/*.frag")
    foreach(SHADER ${SHADER_SOURCES})
        get_filename_component(SHADER_NAME ${SHADER} NAME)
        set(SPV_OUTPUT "${CMAKE_BINARY_DIR}/${SHADER_NAME}.spv")
        add_custom_command(
            OUTPUT ${SPV_OUTPUT}
            COMMAND ${GLSLC_EXECUTABLE} ${SHADER} -o ${SPV_OUTPUT}
            DEPENDS ${SHADER}
            COMMENT "Compiling SPIR-V: ${SHADER_NAME} -> ${SHADER_NAME}.spv"
        )
        list(APPEND SPV_TARGETS ${SPV_OUTPUT})
    endforeach()
    add_custom_target(CompileShaders ALL DEPENDS ${SPV_TARGETS})
endif()

# ================================================================================================
# DYNAMIC SOURCE FILE DISCOVERY (Tracks newly added .cpp files automatically)
# ================================================================================================
file(GLOB ALL_SOURCES CONFIGURE_DEPENDS "src/*.cpp")

# Exclude standalone main entry points from core library set
set(CORE_ENGINE_SOURCES ${ALL_SOURCES})
list(FILTER CORE_ENGINE_SOURCES EXCLUDE REGEX ".*/main\\.cpp$")
list(FILTER CORE_ENGINE_SOURCES EXCLUDE REGEX ".*/test_system2_suite\\.cpp$")

# ================================================================================================
# TARGET 1: MAIN PRESENTATION & NON-EUCLIDEAN BENCHMARK RUNNER
# ================================================================================================
add_executable(avx512_benchmark_runner
    src/main.cpp
    ${CORE_ENGINE_SOURCES}
)
target_link_libraries(avx512_benchmark_runner PRIVATE
    Vulkan::Vulkan
    Threads::Threads
    glfw
    X11
    Xext
    Xcomposite
    Xdamage
    Xfixes
    Xtst
)
if (GLSLC_EXECUTABLE)
    add_dependencies(avx512_benchmark_runner CompileShaders)
endif()

# ================================================================================================
# TARGET 2: SYSTEM 2 SILICON VALIDATION TEST SUITE
# ================================================================================================
add_executable(test_system2_suite
    src/test_system2_suite.cpp
    ${CORE_ENGINE_SOURCES}
)
target_link_libraries(test_system2_suite PRIVATE
    Vulkan::Vulkan
    Threads::Threads
    glfw
    X11
    Xext
    Xcomposite
    Xdamage
    Xfixes
    Xtst
)

# ================================================================================================
# TARGET 3: IN-PROCESS VULKAN LAYER (PROTON / STEAM OFF-LOAD HOOK)
# ================================================================================================
add_library(VkLayer_AVX512_SuperRes SHARED
    layer/src/libVkLayer_AVX512_SuperRes.cpp
    ${CORE_ENGINE_SOURCES}
)
target_link_libraries(VkLayer_AVX512_SuperRes PRIVATE
    Vulkan::Vulkan
    Threads::Threads
)
