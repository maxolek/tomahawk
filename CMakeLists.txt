cmake_minimum_required(VERSION 3.16)
project(tomahawk LANGUAGES CXX)

# -------------------
# Version
# -------------------
# Usage:
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DVERSION=1.2.3
#   cmake -S . -B build  -DCMAKE_BUILD_TYPE=Release -DVERSION=1.2.3
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
#set(CMAKE_C_COMPILER=g++)
set(CMAKE_CXX_COMPILER=g++)

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
    ${SRC_DIR}/movegen.cpp
    ${SRC_DIR}/nnue.cpp
    ${SRC_DIR}/precomp.cpp
    ${SRC_DIR}/searcher.cpp
    ${SRC_DIR}/uci.cpp
    ${SRC_DIR}/main.cpp
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

# system
if (WIN32)
    message(STATUS "Building Tomahawk for Windows")
elseif (APPLE)
    message(STATUS "Building Tomahawk for macOS")
elseif (UNIX)
    message(STATUS "Building Tomahawk for Linux/Unix")
endif()

# flags
set(TOMAHAWK_LTO_FLAG -flto)
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(TOMAHAWK_LTO_FLAG -flto=auto)
endif()

# type - Default to Release
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# SIMD - These configurations need explicit x86 flags; Apple Silicon uses NEON.
if (APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64)$")
    message(STATUS "Apple Silicon detected — using ARM/NEON")
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64|i[3-6]86)$")
    message(STATUS "x86 detected — enabling AVX2/BMI2 where appropriate")
endif()

#######################
#### COMPILE TYPES
######################

# DEBUG
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(tomahawk PRIVATE
        -O1
        -g
        -march=native
        #-mbmi2
        #-mavx2
        -fno-optimize-sibling-calls
    )
    target_compile_definitions(tomahawk PRIVATE DEBUG)

# profile with full optimization
elseif (CMAKE_BUILD_TYPE STREQUAL "Profile")
    target_compile_options(tomahawk PRIVATE
        -O3
        -ffast-math
        -march=native
        #-mbmi2
        #-mavx2
        ${TOMAHAWK_LTO_FLAG}
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
        #-mbmi2
        #-mavx2
        ${TOMAHAWK_LTO_FLAG}
    )
    target_compile_definitions(tomahawk PRIVATE DEV)

# RELEASE
else()
    target_compile_options(tomahawk PRIVATE
        -O3
        -ffast-math
        -march=native
        #-mbmi2
        #-mavx2
        ${TOMAHAWK_LTO_FLAG}
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
