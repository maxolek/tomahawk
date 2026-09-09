cmake_minimum_required(VERSION 3.16)
project(tomahawk LANGUAGES CXX)

# -------------------
# Version
# -------------------
# Usage:
#   cmake -S . -B build -G "MinGW Makefiles" -DCMAKE_C_COMPILER=g++ -DCMAKE_CXX_COMPILER=g++ -DCMAKE_BUILD_TYPE=Release -DVERSION=1.2.3
#   cmake --build build
#
if (NOT DEFINED VERSION)
    set(VERSION dev)
endif()

add_compile_definitions(ENGINE_ID="${VERSION}")
add_compile_definitions(PROJECT_ROOT="${CMAKE_SOURCE_DIR}")

# -------------------
# C++ standard
# -------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# -------------------
# Directories
# -------------------
set(SRC_DIR ${CMAKE_SOURCE_DIR}/src)
set(INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)
set(NNUE_DIR ${CMAKE_SOURCE_DIR}/nnue)
set(ENGINES_DIR ${CMAKE_SOURCE_DIR}/engines)
set(DEV_ENGINES_DIR ${ENGINES_DIR}/dev)
set(PROD_ENGINES_DIR ${ENGINES_DIR}/prod)

# -------------------
# Source files
# -------------------
set(SOURCES
    ${SRC_DIR}/board.cpp
    ${SRC_DIR}/book.cpp
    ${SRC_DIR}/engine.cpp
    ${SRC_DIR}/gamestate.cpp
    ${SRC_DIR}/helpers.cpp
    ${SRC_DIR}/magics.cpp
    ${SRC_DIR}/moveGenerator.cpp
    ${NNUE_DIR}/nnue.cpp
    ${SRC_DIR}/PrecomputedMoveData.cpp
    ${SRC_DIR}/searcher.cpp
    ${SRC_DIR}/tomahawk.cpp
    ${SRC_DIR}/UCI.cpp
)

add_executable(tomahawk ${SOURCES})

target_include_directories(tomahawk PRIVATE 
    ${INCLUDE_DIR}
    ${NNUE_DIR}
)

# -------------------
# Warnings
# -------------------
if (MSVC)
    target_compile_options(tomahawk PRIVATE /W4)
else()
    target_compile_options(tomahawk PRIVATE
        -Wall
        -Wextra
        -Wshadow
        -Wuninitialized
        -Wconversion
        -Wpedantic
    )
endif()

# -------------------
# Build type logic
# -------------------
# Default to Release
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# BMI2/AVX2 are x86-only; Apple Silicon uses NEON by default.
set(SIMD_DEBUG_FLAGS)
if (CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64|i[3-6]86)$")
    set(SIMD_DEBUG_FLAGS -mbmi2 -mavx2)
endif()
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# DEBUG
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(tomahawk PRIVATE
        -O1
        -g
        ${SIMD_DEBUG_FLAGS}
        -fno-optimize-sibling-calls
    )
    target_compile_definitions(tomahawk PRIVATE DEBUG)

# PROFILE
elseif (CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    target_compile_options(tomahawk PRIVATE
        -O1
        -g
        ${SIMD_DEBUG_FLAGS}
    )

# ASAN
elseif (CMAKE_BUILD_TYPE STREQUAL "Asan")
    target_compile_options(tomahawk PRIVATE 
        -O1
        -g
        ${SIMD_DEBUG_FLAGS}
        -fsanitize=address,undefined
        -fno-omit-frame-pointer
    )
    target_link_options(tomahawk PRIVATE
        -fsanitize=address,undefined
    )
    target_compile_definitions(tomahawk PRIVATE DEBUG)

# profile with full optimization
elseif (CMAKE_BUILD_TYPE STREQUAL "Profile")
    target_compile_options(tomahawk PRIVATE
        -O3
        -ffast-math
        -march=native
        -flto
        -g
        -fno-omit-frame-pointer
    )
    target_compile_definitions(tomahawk PRIVATE DEBUG)

# analysis (full stats) config
elseif (CMAKE_BUILD_TYPE STREQUAL "DEV")
    target_compile_options(tomahawk PRIVATE 
        -O3
        -ffast-math
        -march=native
        -flto
    )
    target_compile_definitions(tomahawk PRIVATE DEV)

# RELEASE
else()
    target_compile_options(tomahawk PRIVATE
        -O3
        -ffast-math
        -march=native
        -flto
    )
endif()

# -------------------
# Threads (portable)
# -------------------
find_package(Threads REQUIRED)
target_link_libraries(tomahawk PRIVATE Threads::Threads)

# -------------------
# Platform-specific linking
# -------------------
if (WIN32)
    target_link_libraries(tomahawk PRIVATE ole32)
endif()

# -------------------
# Output name & location
# -------------------
if ((CMAKE_BUILD_TYPE STREQUAL "Release") OR (CMAKE_BUILD_TYPE STREQUAL "Prod") OR (CMAKE_BUILD_TYPE STREQUAL "PROD") OR (CMAKE_BUILD_TYPE STREQUAL "RELEASE"))
    set_target_properties(tomahawk PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${PROD_ENGINES_DIR}
    )
else() 
    set_target_properties(tomahawk PROPERTIES 
        RUNTIME_OUTPUT_DIRECTORY ${DEV_ENGINES_DIR}
    )
endif()

# Handle VERSION-based executable naming
if (VERSION AND NOT VERSION STREQUAL "dev")
    if (NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
        set_target_properties(tomahawk PROPERTIES
            OUTPUT_NAME ${VERSION}
        )
    endif()
endif()

# Optional standalone SIMD regression checks (no network weights required).
option(TOMAHAWK_SIMD_TESTS "Build SIMD scalar-reference tests" OFF)
if (TOMAHAWK_SIMD_TESTS)
    enable_testing()
    add_executable(simd_test tests/simd_test.cpp)
    target_include_directories(simd_test PRIVATE ${INCLUDE_DIR} ${NNUE_DIR})
    get_target_property(SIMD_TEST_OPTIONS tomahawk COMPILE_OPTIONS)
    target_compile_options(simd_test PRIVATE ${SIMD_TEST_OPTIONS})
    if (CMAKE_BUILD_TYPE STREQUAL "Asan")
        target_link_options(simd_test PRIVATE -fsanitize=address,undefined)
    endif()
    add_test(NAME simd_scalar_reference COMMAND simd_test)
endif()
