# Lock-Free Data Structures Tests
cmake_minimum_required(VERSION 3.20)

# Find required packages
find_package(GTest REQUIRED)
find_package(Threads REQUIRED)

# Test executables
set(LOCKFREE_TESTS
    test_lockfree_hash_table
    test_mpmc_ring_buffer
    test_lockfree_lru_cache
    test_atomic_ref_count
)

# Create test executables
foreach(TEST_NAME ${LOCKFREE_TESTS})
    add_executable(${TEST_NAME} ${TEST_NAME}.cpp)
    
    # Set C++ standard
    set_target_properties(${TEST_NAME} PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
    )
    
    # Include directories
    target_include_directories(${TEST_NAME} PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${GTEST_INCLUDE_DIRS}
    )
    
    # Link libraries
    target_link_libraries(${TEST_NAME}
        lockfree
        GTest::GTest
        GTest::Main
        Threads::Threads
    )
    
    # Compiler optimizations for tests
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        target_compile_options(${TEST_NAME} PRIVATE
            -O2
            -g
            -Wall
            -Wextra
            -Wpedantic
        )
    endif()
    
    # Add to test suite
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
    
    # Set test properties
    set_tests_properties(${TEST_NAME} PROPERTIES
        TIMEOUT 300  # 5 minutes timeout
        LABELS "lockfree"
    )
endforeach()

# Comprehensive test executable that runs all tests
add_executable(lockfree_all_tests
    test_lockfree_hash_table.cpp
    test_mpmc_ring_buffer.cpp
    test_lockfree_lru_cache.cpp
    test_atomic_ref_count.cpp
)

set_target_properties(lockfree_all_tests PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

target_include_directories(lockfree_all_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${GTEST_INCLUDE_DIRS}
)

target_link_libraries(lockfree_all_tests
    lockfree
    GTest::GTest
    GTest::Main
    Threads::Threads
)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(lockfree_all_tests PRIVATE
        -O2
        -g
        -Wall
        -Wextra
        -Wpedantic
    )
endif()

add_test(NAME lockfree_all_tests COMMAND lockfree_all_tests)
set_tests_properties(lockfree_all_tests PROPERTIES
    TIMEOUT 600  # 10 minutes timeout for comprehensive tests
    LABELS "lockfree;comprehensive"
)

# Performance benchmark executable
add_executable(lockfree_benchmarks
    benchmark_lockfree.cpp
)

set_target_properties(lockfree_benchmarks PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

target_include_directories(lockfree_benchmarks PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(lockfree_benchmarks
    lockfree
    Threads::Threads
)

# Aggressive optimizations for benchmarks
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(lockfree_benchmarks PRIVATE
        -O3
        -march=native
        -mtune=native
        -flto
        -fno-exceptions
        -fno-rtti
        -DNDEBUG
    )
    
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        target_link_options(lockfree_benchmarks PRIVATE -flto)
    endif()
endif()

# Memory sanitizer tests (optional)
if(ENABLE_TSAN)
    add_executable(lockfree_tsan_tests
        test_lockfree_hash_table.cpp
        test_mpmc_ring_buffer.cpp
        test_lockfree_lru_cache.cpp
        test_atomic_ref_count.cpp
    )
    
    set_target_properties(lockfree_tsan_tests PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
    )
    
    target_include_directories(lockfree_tsan_tests PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${GTEST_INCLUDE_DIRS}
    )
    
    target_link_libraries(lockfree_tsan_tests
        lockfree
        GTest::GTest
        GTest::Main
        Threads::Threads
    )
    
    target_compile_options(lockfree_tsan_tests PRIVATE
        -fsanitize=thread
        -fno-omit-frame-pointer
        -g
        -O1
    )
    
    target_link_options(lockfree_tsan_tests PRIVATE
        -fsanitize=thread
    )
    
    add_test(NAME lockfree_tsan_tests COMMAND lockfree_tsan_tests)
    set_tests_properties(lockfree_tsan_tests PROPERTIES
        TIMEOUT 1200  # 20 minutes for sanitizer tests
        LABELS "lockfree;sanitizer;tsan"
    )
endif()

# Custom test targets
add_custom_target(test_lockfree
    COMMAND ${CMAKE_CTEST_COMMAND} -L lockfree --output-on-failure
    DEPENDS ${LOCKFREE_TESTS}
    COMMENT "Running lock-free data structure tests"
)

add_custom_target(benchmark_lockfree
    COMMAND lockfree_benchmarks
    DEPENDS lockfree_benchmarks
    COMMENT "Running lock-free data structure benchmarks"
)