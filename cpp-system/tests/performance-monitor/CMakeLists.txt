# Performance monitor tests
find_package(GTest REQUIRED)

add_executable(test_performance_monitor
    test_performance_monitor.cpp
)

target_link_libraries(test_performance_monitor
    ultra_monitor
    ultra_common
    GTest::gtest
    GTest::gtest_main
    pthread
)

target_include_directories(test_performance_monitor PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

# Add test to CTest
add_test(NAME PerformanceMonitorTests COMMAND test_performance_monitor)

# Performance benchmark executable
add_executable(benchmark_performance_monitor
    test_performance_monitor.cpp
)

target_link_libraries(benchmark_performance_monitor
    ultra_monitor
    ultra_common
    GTest::gtest
    GTest::gtest_main
    pthread
)

target_compile_definitions(benchmark_performance_monitor PRIVATE
    BENCHMARK_MODE=1
)

target_include_directories(benchmark_performance_monitor PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)