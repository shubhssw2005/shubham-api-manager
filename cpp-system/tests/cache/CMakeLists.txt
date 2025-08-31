# Cache tests
add_executable(test_ultra_cache
    test_ultra_cache.cpp
)

target_include_directories(test_ultra_cache PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/cache
    ${CMAKE_SOURCE_DIR}/include/common
    ${CMAKE_SOURCE_DIR}/include/lockfree
)

target_link_libraries(test_ultra_cache
    ultra_cache
    ultra_common
    ultra_lockfree
    gtest
    gtest_main
    pthread
)

# Cache performance tests
add_executable(test_cache_performance
    test_cache_performance.cpp
)

target_link_libraries(test_cache_performance
    ultra_cache
    ultra_common
    ultra_lockfree
    gtest
    gtest_main
    pthread
)

# Register tests with CTest
add_test(NAME UltraCacheTests COMMAND test_ultra_cache)
add_test(NAME CachePerformanceTests COMMAND test_cache_performance)