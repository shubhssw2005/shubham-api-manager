# Test suite for ultra-low-latency system
enable_testing()

# Find required testing libraries
find_package(GTest REQUIRED)
find_package(benchmark REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(CURL REQUIRED libcurl)
find_package(nlohmann_json REQUIRED)

# Add memory tests subdirectory
add_subdirectory(memory)

# Add lockfree tests subdirectory
add_subdirectory(lockfree)

# Add cache tests subdirectory
add_subdirectory(cache)

# Add GPU compute tests subdirectory
add_subdirectory(gpu-compute)

# Add network tests subdirectory
add_subdirectory(network)

# Add security tests subdirectory
add_subdirectory(security)

# Testing framework library
add_library(ultra_test_framework
    framework/test_framework.cpp
    load/load_test_suite.cpp
    chaos/chaos_testing_framework.cpp
)

target_include_directories(ultra_test_framework PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CURL_INCLUDE_DIRS}
)

target_link_libraries(ultra_test_framework
    GTest::gtest
    GTest::gtest_main
    benchmark::benchmark
    ${CURL_LIBRARIES}
    nlohmann_json::nlohmann_json
    pthread
)

target_compile_definitions(ultra_test_framework PRIVATE
    ${CURL_CFLAGS_OTHER}
)

# Comprehensive unit tests
add_executable(comprehensive_unit_tests
    unit/test_comprehensive_suite.cpp
)

target_include_directories(comprehensive_unit_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(comprehensive_unit_tests
    ultra_test_framework
    ultra_common
    ultra_cache
    ultra_stream
    ultra_monitor
    ultra_memory
    ultra_lockfree
    GTest::gtest
    GTest::gtest_main
)

# Comprehensive benchmarks
add_executable(comprehensive_benchmarks
    benchmark/benchmark_comprehensive.cpp
)

target_include_directories(comprehensive_benchmarks PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(comprehensive_benchmarks
    ultra_test_framework
    ultra_common
    ultra_cache
    ultra_stream
    ultra_monitor
    ultra_memory
    ultra_lockfree
    benchmark::benchmark
)

# Load testing executable
add_executable(load_tests
    load/load_test_main.cpp
)

target_include_directories(load_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(load_tests
    ultra_test_framework
    ${CURL_LIBRARIES}
    nlohmann_json::nlohmann_json
)

# Chaos testing executable
add_executable(chaos_tests
    chaos/chaos_test_main.cpp
)

target_include_directories(chaos_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(chaos_tests
    ultra_test_framework
    nlohmann_json::nlohmann_json
)

# Main test runner
add_executable(ultra_test_runner
    test_runner.cpp
)

target_include_directories(ultra_test_runner PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(ultra_test_runner
    ultra_test_framework
    ultra_common
    ultra_cache
    ultra_stream
    ultra_monitor
    ultra_memory
    ultra_lockfree
    GTest::gtest
    GTest::gtest_main
    benchmark::benchmark
    ${CURL_LIBRARIES}
    nlohmann_json::nlohmann_json
)

# Legacy unit tests (for compatibility)
add_executable(unit_tests
    unit/test_memory_pool.cpp
    unit/test_lock_free_queue.cpp
    unit/test_ultra_cache.cpp
    unit/test_stream_processor.cpp
    unit/test_performance_monitor.cpp
    common/test_config_service_discovery.cpp
)

target_include_directories(unit_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(unit_tests
    ultra_common
    ultra_cache
    ultra_stream
    ultra_monitor
    GTest::gtest
    GTest::gtest_main
)

# Legacy benchmark tests (for compatibility)
add_executable(benchmark_tests
    benchmark/bench_cache_operations.cpp
    benchmark/bench_stream_processing.cpp
    benchmark/bench_api_gateway.cpp
)

target_link_libraries(benchmark_tests
    ultra_common
    ultra_cache
    ultra_stream
    benchmark::benchmark
)

# Integration tests
add_executable(integration_tests
    integration/test_full_system.cpp
    integration/test_node_js_integration.cpp
)

target_link_libraries(integration_tests
    ultra_common
    ultra_cache
    ultra_stream
    GTest::gtest
    GTest::gtest_main
)

# Register tests with CTest
add_test(NAME ComprehensiveUnitTests COMMAND comprehensive_unit_tests)
add_test(NAME ComprehensiveBenchmarks COMMAND comprehensive_benchmarks --benchmark_min_time=0.1)
add_test(NAME LoadTests COMMAND load_tests --test-type constant --duration 10)
add_test(NAME ChaosTests COMMAND chaos_tests --experiment network_latency --duration 30)

# Legacy test registration (for compatibility)
add_test(NAME UnitTests COMMAND unit_tests)
add_test(NAME BenchmarkTests COMMAND benchmark_tests)
add_test(NAME IntegrationTests COMMAND integration_tests)

# Performance regression tests
add_test(NAME PerformanceRegression COMMAND ultra_test_runner --test-type benchmark --duration 30)

# System integration test
add_test(NAME SystemIntegration COMMAND ultra_test_runner --test-type all --duration 60)

# Install test executables
install(TARGETS ultra_test_runner comprehensive_unit_tests comprehensive_benchmarks
        load_tests chaos_tests
        DESTINATION bin)

# Install test scripts
install(FILES 
        ${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_performance_tests.sh
        ${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_load_tests.sh
        ${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_chaos_tests.sh
        DESTINATION bin
        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                   GROUP_READ GROUP_EXECUTE
                   WORLD_READ WORLD_EXECUTE)