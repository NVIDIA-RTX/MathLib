cmake_minimum_required(VERSION 3.24...4.4.4)

include(FetchContent)

# Arm64?
if((CMAKE_GENERATOR_PLATFORM MATCHES "ARM64") OR (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64"))
    set(IS_ARM64 TRUE)
else()
    set(IS_ARM64 FALSE)
endif()

# Download sse2neon for ARM
if(IS_ARM64)
    fetchcontent_declare(
        sse2neon
        DOWNLOAD_EXTRACT_TIMESTAMP 1
        DOWNLOAD_NO_PROGRESS 1
        SOURCE_SUBDIR "ignore CMakeLists.txt"
        URL "https://github.com/DLTcollab/sse2neon/archive/refs/heads/master.zip"
    )

    message("MathLib: Downloading sse2neon...")
    fetchcontent_makeavailable(sse2neon)
endif()

# Sources
set(MATHLIB_H
    "ml.h"
    "ml.hlsli"
)
source_group("MathLib"
    FILES
        ${MATHLIB_H}
)

set(MATHLIB_GUTS
    "Guts/bool1.h"
    "Guts/conversion.h"
    "Guts/emulation.h"
    "Guts/f32.h"
    "Guts/f64.h"
    "Guts/fsmall.h"
    "Guts/i32.h"
    "Guts/math.h"
    "Guts/other.h"
    "Guts/packing.h"
    "Guts/sorting.h"
    "Guts/swizzle.h"
    "Guts/tests.h"
    "Guts/u32.h"
)
source_group("MathLib/Guts"
    FILES
        ${MATHLIB_GUTS}
)

set_source_files_properties("ml.hlsli"
    PROPERTIES
        VS_SETTINGS "ExcludedFromBuild=true"
)

# Library
add_library(MathLib INTERFACE)

if(WIN32) # TODO: is MSVC?
    target_sources(MathLib
        PRIVATE
            ${MATHLIB_H}
            ${MATHLIB_GUTS}
    )
    set_target_properties(MathLib
        PROPERTIES
            FOLDER "MathLib"
    )
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

target_include_directories(MathLib
    INTERFACE
        "."
        $<$<BOOL:${IS_ARM64}>:${sse2neon_SOURCE_DIR}>
)
