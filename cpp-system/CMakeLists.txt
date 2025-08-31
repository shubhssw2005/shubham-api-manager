cmake_minimum_required(VERSION 3.20)
project(UltraLowLatencySystem VERSION 1.0.0 LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags for performance
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -mtune=native -flto")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -fsanitize=address -fsanitize=thread")

# Find required packages
find_package(PkgConfig REQUIRED)
find_package(Threads REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(CURL REQUIRED)

# DPDK
pkg_check_modules(DPDK REQUIRED libdpdk)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${DPDK_INCLUDE_DIRS})

# Add subdirectories for different components
add_subdirectory(src/memory)
add_subdirectory(src/lockfree)
add_subdirectory(src/api-gateway)
add_subdirectory(src/cache)
add_subdirectory(src/stream-processor)
add_subdirectory(src/gpu-compute)
add_subdirectory(src/performance-monitor)
add_subdirectory(src/database)
add_subdirectory(src/network)
add_subdirectory(src/common)
add_subdirectory(src/security)
add_subdirectory(src/integration)
add_subdirectory(src/data-generator)
add_subdirectory(examples)
add_subdirectory(tests)

# Global compile definitions
add_compile_definitions(
    $<$<CONFIG:Release>:ULTRA_PERFORMANCE_MODE>
    $<$<CONFIG:Debug>:ULTRA_DEBUG_MODE>
)

# Installation
install(DIRECTORY config/ DESTINATION etc/ultra-cpp)
install(DIRECTORY scripts/ DESTINATION bin 
        FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                        GROUP_READ GROUP_EXECUTE
                        WORLD_READ WORLD_EXECUTE)