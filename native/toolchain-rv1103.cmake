# CMake Toolchain file for Rockchip RV1103 (Luckfox Pico / Pico Mini / Pico Plus)
# Target: ARM Cortex-A7 @ 1.2 GHz with NEON Advanced SIMD and VFPv4 (armhf)
#
# Usage:
#   cmake -S native -B native/build-rv1103 \
#     -DCMAKE_TOOLCHAIN_FILE=native/toolchain-rv1103.cmake \
#     [-DRUVI_TOOLCHAIN_PREFIX=/path/to/arm-rockchip830-linux-uclibcgnueabihf-]
#   cmake --build native/build-rv1103 -j$(nproc)

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR arm)

# Cross-compiler prefix resolution
if(NOT DEFINED RUVI_TOOLCHAIN_PREFIX)
    if(DEFINED ENV{RUVI_ARM_PREFIX})
        set(RUVI_TOOLCHAIN_PREFIX $ENV{RUVI_ARM_PREFIX})
    elseif(DEFINED ENV{RUVI_ARM_CXX})
        get_filename_component(RUVI_ARM_DIR "$ENV{RUVI_ARM_CXX}" DIRECTORY)
        get_filename_component(RUVI_ARM_NAME "$ENV{RUVI_ARM_CXX}" NAME)
        string(REGEX REPLACE "g\\+\\+$" "" RUVI_PREFIX_BASE "${RUVI_ARM_NAME}")
        if(RUVI_ARM_DIR)
            set(RUVI_TOOLCHAIN_PREFIX "${RUVI_ARM_DIR}/${RUVI_PREFIX_BASE}")
        else()
            set(RUVI_TOOLCHAIN_PREFIX "${RUVI_PREFIX_BASE}")
        endif()
    else()
        set(RUVI_TOOLCHAIN_PREFIX "arm-rockchip830-linux-uclibcgnueabihf-")
    endif()
endif()

set(CMAKE_C_COMPILER   "${RUVI_TOOLCHAIN_PREFIX}gcc")
set(CMAKE_CXX_COMPILER "${RUVI_TOOLCHAIN_PREFIX}g++")
set(CMAKE_ASM_COMPILER "${RUVI_TOOLCHAIN_PREFIX}gcc")

# Cortex-A7 tuning flags: NEON vector engine, VFPv4 hardware float, hard float ABI
set(ARM_FLAGS "-mcpu=cortex-a7 -mfpu=neon-vfpv4 -mfloat-abi=hard")
set(CMAKE_C_FLAGS_INIT   "${ARM_FLAGS}")
set(CMAKE_CXX_FLAGS_INIT "${ARM_FLAGS}")
set(CMAKE_ASM_FLAGS_INIT "${ARM_FLAGS}")

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
