# Memory Management Tests
cmake_minimum_required(VERSION 3.20)

# Find Google Test
find_package(GTest REQUIRED)
find_package(Threads REQUIRED)

# Test executable
add_executable(memory_tests
    test_lock_free_allocator.cpp
    test_numa_allocator.cpp
    test_rcu_smart_ptr.cpp
    test_mmap_allocator.cpp
    test_memory_manager.cpp
)

target_link_libraries(memory_tests
    PRIVATE
        ultra_memory
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
)

target_include_directories(memory_tests
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
)

target_compile_features(memory_tests PRIVATE cxx_std_20)

target_compile_options(memory_tests PRIVATE
    -Wall -Wextra -Wpedantic
    -O2 -g
)

# Enable AddressSanitizer for memory tests
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(memory_tests PRIVATE
        -fsanitize=address
        -fsanitize=thread
    )
    target_link_options(memory_tests PRIVATE
        -fsanitize=address
        -fsanitize=thread
    )
endif()

# Register tests with CTest
include(GoogleTest)
gtest_discover_tests(memory_tests)