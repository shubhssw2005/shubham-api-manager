# Lock-Free Data Structures Library
cmake_minimum_required(VERSION 3.20)

# Create the lockfree library
add_library(lockfree STATIC
    hazard_pointer_system.cpp
)

# Set target properties
set_target_properties(lockfree PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

# Include directories
target_include_directories(lockfree
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# Compiler-specific optimizations
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(lockfree PRIVATE
        -O3
        -march=native
        -mtune=native
        -flto
        -fno-exceptions
        -fno-rtti
        -Wall
        -Wextra
        -Wpedantic
        -Wno-unused-parameter
    )
    
    # Enable specific optimizations for lock-free code
    target_compile_options(lockfree PRIVATE
        -fomit-frame-pointer
        -funroll-loops
        -fprefetch-loop-arrays
        -ffast-math
    )
    
    # Link-time optimizations
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        target_link_options(lockfree PRIVATE -flto)
    endif()
endif()

# MSVC optimizations
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(lockfree PRIVATE
        /O2
        /Oi
        /Ot
        /GL
        /arch:AVX2
    )
    
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        target_link_options(lockfree PRIVATE /LTCG)
    endif()
endif()

# Threading support
find_package(Threads REQUIRED)
target_link_libraries(lockfree PRIVATE Threads::Threads)

# Atomic operations support
if(NOT WIN32)
    target_link_libraries(lockfree PRIVATE atomic)
endif()

# Platform-specific definitions
if(WIN32)
    target_compile_definitions(lockfree PRIVATE
        WIN32_LEAN_AND_MEAN
        NOMINMAX
        _WIN32_WINNT=0x0601
    )
endif()

if(UNIX AND NOT APPLE)
    target_compile_definitions(lockfree PRIVATE
        _GNU_SOURCE
        _POSIX_C_SOURCE=200809L
    )
endif()

# Debug configurations
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(lockfree PRIVATE
        LOCKFREE_DEBUG=1
        LOCKFREE_STATS=1
    )
endif()

# Release configurations
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_definitions(lockfree PRIVATE
        NDEBUG
        LOCKFREE_OPTIMIZED=1
    )
endif()

# Memory sanitizer support
option(ENABLE_ASAN "Enable AddressSanitizer" OFF)
option(ENABLE_TSAN "Enable ThreadSanitizer" OFF)
option(ENABLE_MSAN "Enable MemorySanitizer" OFF)

if(ENABLE_ASAN)
    target_compile_options(lockfree PRIVATE -fsanitize=address -fno-omit-frame-pointer)
    target_link_options(lockfree PRIVATE -fsanitize=address)
endif()

if(ENABLE_TSAN)
    target_compile_options(lockfree PRIVATE -fsanitize=thread -fno-omit-frame-pointer)
    target_link_options(lockfree PRIVATE -fsanitize=thread)
endif()

if(ENABLE_MSAN)
    target_compile_options(lockfree PRIVATE -fsanitize=memory -fno-omit-frame-pointer)
    target_link_options(lockfree PRIVATE -fsanitize=memory)
endif()

# Install targets
install(TARGETS lockfree
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

# Install headers
install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/lockfree
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp"
)