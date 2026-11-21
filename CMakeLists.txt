cmake_minimum_required(VERSION 3.16)
project(tomahawk LANGUAGES CXX)

option(SMALL_NET "Build with the direct-output small NNUE network" OFF)
set(NNUE_FILE "" CACHE FILEPATH "NNUE weights to bundle with the engine")

# -------------------
# Version
# -------------------
# Usage:
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DVERSION=1.2.3
#     -DSMALL_NET=ON -DNNUE_FILE=nnue/weights/output_buckets_25wdl_1000.bin
#   cmake --build build
#
if (NOT DEFINED VERSION)
    set(VERSION dev)
endif()

if (NOT NNUE_FILE)
    if (SMALL_NET)
        set(NNUE_FILE "${CMAKE_SOURCE_DIR}/nnue/weights/output_buckets_25wdl_1000.bin")
    else()
        set(NNUE_FILE "${CMAKE_SOURCE_DIR}/nnue/weights/bignet_farseer_0wdl_lr001_00001_800batches.bin")
    endif()
endif()
get_filename_component(NNUE_FILE "${NNUE_FILE}" REALPATH BASE_DIR "${CMAKE_SOURCE_DIR}")
if (NOT EXISTS "${NNUE_FILE}")
    message(FATAL_ERROR "NNUE_FILE does not exist: ${NNUE_FILE}")
endif()
set(NNUE_FILE_EFFECTIVE "${NNUE_FILE}" CACHE INTERNAL "Resolved NNUE file used by this build" FORCE)

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

if ((CMAKE_BUILD_TYPE STREQUAL "Release") OR (CMAKE_BUILD_TYPE STREQUAL "Prod") OR (CMAKE_BUILD_TYPE STREQUAL "PROD") OR (CMAKE_BUILD_TYPE STREQUAL "RELEASE"))
    set(ENGINE_OUTPUT_DIR ${PROD_ENGINES_DIR})
else()
    set(ENGINE_OUTPUT_DIR ${DEV_ENGINES_DIR})
endif()
if (VERSION AND NOT VERSION STREQUAL "dev" AND NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(ENGINE_OUTPUT_NAME ${VERSION})
else()
    set(ENGINE_OUTPUT_NAME tomahawk)
endif()
file(TO_CMAKE_PATH "${ENGINE_OUTPUT_DIR}/nnue/${ENGINE_OUTPUT_NAME}.nnue.bin" NNUE_BUNDLE_PATH)

# -------------------
# Source files
# -------------------
set(SOURCES
    ${SRC_DIR}/affinity.cpp
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
    ${SRC_DIR}/threads.cpp
    ${SRC_DIR}/uci.cpp
    ${SRC_DIR}/main.cpp
)

add_executable(tomahawk ${SOURCES})

if (SMALL_NET)
    target_compile_definitions(tomahawk PRIVATE SMALL_NET)
endif()
target_compile_definitions(tomahawk PRIVATE "NNUE_WEIGHT_FILE=\"${NNUE_BUNDLE_PATH}\"")
add_custom_command(TARGET tomahawk POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${ENGINE_OUTPUT_DIR}/nnue"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${NNUE_FILE}" "${NNUE_BUNDLE_PATH}"
    VERBATIM
)

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
set(LTO_FLAG -flto)
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(LTO_FLAG -flto=auto)
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
if (CMAKE_BUILD_TYPE STREQUAL "DEBUG")
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
elseif (CMAKE_BUILD_TYPE STREQUAL "PROFILE")
    target_compile_options(tomahawk PRIVATE
        -O3
        -ffast-math
        -march=native
        #-mbmi2
        #-mavx2
        ${LTO_FLAG}
        -g
        -fno-omit-frame-pointer
    )
    target_compile_definitions(tomahawk PRIVATE DEBUG)

# analysis (full stats + timing) config
elseif (CMAKE_BUILD_TYPE STREQUAL "DEV")
    target_compile_options(tomahawk PRIVATE 
        -O3
        -ffast-math
        -march=native
        #-mbmi2
        #-mavx2
        ${LTO_FLAG}
    )
    target_compile_definitions(tomahawk PRIVATE DEV)

# stats only (non timing) config
elseif (CMAKE_BUILD_TYPE STREQUAL "STATS")
    target_compile_options(tomahawk PRIVATE 
        -O3 
        -ffast-math 
        -march=native 
        #-mbmi2
        #-mavx2
        ${LTO_FLAG}
    )
    target_compile_definitions(tomahawk PRIVATE STATS)

# timing only (no extra stats) config
elseif (CMAKE_BUILD_TYPE STREQUAL "TIMING")
    target_compile_options(tomahawk PRIVATE 
        -O3 
        -ffast-math 
        -march=native 
        #-mbmi2
        #-mavx2
        ${LTO_FLAG}
    )
    target_compile_definitions(tomahawk PRIVATE TIMING)

# RELEASE
else()
    target_compile_options(tomahawk PRIVATE
        -O3
        -ffast-math
        -march=native
        #-mbmi2
        #-mavx2
        ${LTO_FLAG}
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
set_target_properties(tomahawk PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${ENGINE_OUTPUT_DIR}"
    OUTPUT_NAME "${ENGINE_OUTPUT_NAME}"
)
