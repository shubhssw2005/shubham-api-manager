# Stream processor tests
find_package(GTest REQUIRED)

add_executable(test_stream_processor
    test_stream_processor.cpp
)

target_link_libraries(test_stream_processor
    ultra_stream
    ultra_lockfree
    ultra_common
    GTest::gtest
    GTest::gtest_main
    pthread
)

target_include_directories(test_stream_processor PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

# Enable SIMD optimizations for tests
target_compile_options(test_stream_processor PRIVATE
    -mavx2
    -mfma
    -march=native
)

# Add test to CTest
add_test(NAME StreamProcessorTests COMMAND test_stream_processor)

# Benchmark executable
add_executable(stream_processor_benchmark
    benchmark_stream_processor.cpp
)

target_link_libraries(stream_processor_benchmark
    ultra_stream
    ultra_lockfree
    ultra_common
    pthread
)

target_include_directories(stream_processor_benchmark PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_compile_options(stream_processor_benchmark PRIVATE
    -mavx2
    -mfma
    -march=native
    -O3
)

install(TARGETS test_stream_processor stream_processor_benchmark
        RUNTIME DESTINATION bin/tests)