cmake_minimum_required(VERSION 3.16)
project(ruvi_native LANGUAGES CXX ASM)

set(RUVI_SOURCES
    ruvi.cpp
    parallel.cpp
    tokenizer.cpp
    tokenizer_data.cpp
    kernels/kernel.cpp
    kernels/kernel_s3.cpp
    kernels/kernel_rv1103.cpp
    kernels/kernel_rv1103.S
)

add_library(ruvi STATIC ${RUVI_SOURCES})
target_include_directories(ruvi PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(ruvi PUBLIC cxx_std_17)
target_compile_options(ruvi PRIVATE -O3 -Wall -Wextra -Wpedantic -ffp-contract=off)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm" OR CMAKE_SYSTEM_PROCESSOR MATCHES "ARM")
    target_compile_options(ruvi PRIVATE -mcpu=cortex-a7 -mfpu=neon-vfpv4 -mfloat-abi=hard)
endif()

add_executable(ruvi-native main.cpp)
target_link_libraries(ruvi-native PRIVATE ruvi)
target_compile_options(ruvi-native PRIVATE -Wall -Wextra -Wpedantic)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm" OR CMAKE_SYSTEM_PROCESSOR MATCHES "ARM")
    target_compile_options(ruvi-native PRIVATE -mcpu=cortex-a7 -mfpu=neon-vfpv4 -mfloat-abi=hard)
endif()
