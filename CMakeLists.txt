cmake_minimum_required(VERSION 3.22...3.30)

include(FetchContent)

# Arm64?
if((CMAKE_GENERATOR_PLATFORM MATCHES "ARM64") OR (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64"))
    set(IS_ARM64 TRUE)
else()
    set(IS_ARM64 FALSE)
endif()

# Download sse2neon for ARM
if(IS_ARM64)
    FetchContent_Declare(
        sse2neon
        GIT_REPOSITORY https://github.com/DLTcollab/sse2neon.git
        GIT_TAG master
        GIT_SHALLOW 1
    )

    message("MathLib: Downloading sse2neon...")
    FetchContent_MakeAvailable(sse2neon)
endif()

# Sources
file(GLOB MATHLIB_H
    "ml.h"
    "ml.hlsli"
)
source_group("MathLib" FILES ${MATHLIB_H})

file(GLOB MATHLIB_GUTS "Guts/*")
source_group("MathLib/Guts" FILES ${MATHLIB_GUTS})

set_property(SOURCE "ml.hlsli" PROPERTY VS_SETTINGS "ExcludedFromBuild=true")

# Library
add_library(MathLib INTERFACE)

if(WIN32) # TODO: is MSVC?
    target_sources(MathLib PRIVATE ${MATHLIB_H} ${MATHLIB_GUTS})
    set_target_properties(MathLib PROPERTIES FOLDER "MathLib")
endif()

if(IS_ARM64 AND MSVC)
    target_compile_options(MathLib INTERFACE
        # ARM64 builds require modern preprocessor
        /Zc:preprocessor
        # Suppress known warnings:
        /wd4310 # cast truncates constant value
        /wd4127 # conditional expression is constant
    )
endif()

target_include_directories(MathLib INTERFACE
    .
    $<$<BOOL:${IS_ARM64}>:${sse2neon_SOURCE_DIR}>
)
