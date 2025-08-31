# Memory Management Components
cmake_minimum_required(VERSION 3.20)

# Find required packages
find_package(PkgConfig REQUIRED)

# Try to find NUMA library (optional on non-Linux systems)
pkg_check_modules(NUMA numa)
if(NOT NUMA_FOUND)
    message(WARNING "NUMA library not found, using fallback implementation")
    set(NUMA_LIBRARIES "")
    set(NUMA_INCLUDE_DIRS "")
    set(NUMA_CFLAGS_OTHER "")
endif()

# Memory management library
add_library(ultra_memory STATIC
    lock_free_allocator.cpp
    numa_allocator.cpp
    rcu_smart_ptr.cpp
    mmap_allocator.cpp
)

target_include_directories(ultra_memory
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
    PRIVATE
        ${NUMA_INCLUDE_DIRS}
)

target_link_libraries(ultra_memory
    PUBLIC
        ${NUMA_LIBRARIES}
        pthread
)

target_compile_features(ultra_memory PUBLIC cxx_std_20)

target_compile_options(ultra_memory PRIVATE
    -Wall -Wextra -Wpedantic
    -O3 -march=native
    -fno-omit-frame-pointer
    ${NUMA_CFLAGS_OTHER}
)

# Add compiler-specific optimizations
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(ultra_memory PRIVATE
        -flto
        -fuse-linker-plugin
    )
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(ultra_memory PRIVATE
        -flto=thin
    )
endif()

# Enable huge pages support
target_compile_definitions(ultra_memory PRIVATE
    _GNU_SOURCE
    ULTRA_ENABLE_HUGE_PAGES=1
)

# Install targets
install(TARGETS ultra_memory
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/memory
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp"
)