# Database tests CMakeLists.txt

# Find required packages for testing
find_package(GTest REQUIRED)
find_package(GMock REQUIRED)

# Database connector tests
add_executable(test_database_connector
    test_database_connector.cpp
)

target_link_libraries(test_database_connector
    PRIVATE
        ultra_database
        ultra_common
        GTest::gtest
        GTest::gtest_main
        GMock::gmock
        GMock::gmock_main
        pthread
)

target_include_directories(test_database_connector
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
)

# Connection pool tests
add_executable(test_connection_pool
    test_connection_pool.cpp
)

target_link_libraries(test_connection_pool
    PRIVATE
        ultra_database
        ultra_common
        GTest::gtest
        GTest::gtest_main
        GMock::gmock
        GMock::gmock_main
        pthread
)

target_include_directories(test_connection_pool
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
)

# Query cache tests
add_executable(test_query_cache
    test_query_cache.cpp
)

target_link_libraries(test_query_cache
    PRIVATE
        ultra_database
        ultra_common
        GTest::gtest
        GTest::gtest_main
        GMock::gmock
        GMock::gmock_main
        pthread
)

target_include_directories(test_query_cache
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
)

# Set test properties
set_target_properties(test_database_connector test_connection_pool test_query_cache
    PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests/database
)

# Compiler options for tests
target_compile_options(test_database_connector
    PRIVATE
        -Wall -Wextra -Wpedantic
        $<$<CONFIG:Debug>:-g3 -O0 -DDEBUG>
        $<$<CONFIG:Release>:-O2 -DNDEBUG>
)

target_compile_options(test_connection_pool
    PRIVATE
        -Wall -Wextra -Wpedantic
        $<$<CONFIG:Debug>:-g3 -O0 -DDEBUG>
        $<$<CONFIG:Release>:-O2 -DNDEBUG>
)

target_compile_options(test_query_cache
    PRIVATE
        -Wall -Wextra -Wpedantic
        $<$<CONFIG:Debug>:-g3 -O0 -DDEBUG>
        $<$<CONFIG:Release>:-O2 -DNDEBUG>
)

# Add tests to CTest
add_test(NAME DatabaseConnectorTest COMMAND test_database_connector)
add_test(NAME ConnectionPoolTest COMMAND test_connection_pool)
add_test(NAME QueryCacheTest COMMAND test_query_cache)

# Set test environment variables
set_tests_properties(DatabaseConnectorTest PROPERTIES
    ENVIRONMENT "GTEST_COLOR=1"
    TIMEOUT 60
)

set_tests_properties(ConnectionPoolTest PROPERTIES
    ENVIRONMENT "GTEST_COLOR=1"
    TIMEOUT 120
)

set_tests_properties(QueryCacheTest PROPERTIES
    ENVIRONMENT "GTEST_COLOR=1"
    TIMEOUT 60
)

# Integration tests (require database)
if(ENABLE_INTEGRATION_TESTS)
    # These tests require a real PostgreSQL database
    set_tests_properties(DatabaseConnectorTest PROPERTIES
        ENVIRONMENT "TEST_DATABASE_URL=postgresql://test_user:test_password@localhost:5432/test_ultra_cpp;GTEST_COLOR=1"
    )
    
    set_tests_properties(ConnectionPoolTest PROPERTIES
        ENVIRONMENT "TEST_DATABASE_URL=postgresql://test_user:test_password@localhost:5432/test_ultra_cpp;GTEST_COLOR=1"
    )
endif()

# Memory debugging for tests
if(ENABLE_ASAN)
    target_compile_options(test_database_connector PRIVATE -fsanitize=address -fno-omit-frame-pointer)
    target_link_options(test_database_connector PRIVATE -fsanitize=address)
    
    target_compile_options(test_connection_pool PRIVATE -fsanitize=address -fno-omit-frame-pointer)
    target_link_options(test_connection_pool PRIVATE -fsanitize=address)
    
    target_compile_options(test_query_cache PRIVATE -fsanitize=address -fno-omit-frame-pointer)
    target_link_options(test_query_cache PRIVATE -fsanitize=address)
endif()

if(ENABLE_TSAN)
    target_compile_options(test_database_connector PRIVATE -fsanitize=thread)
    target_link_options(test_database_connector PRIVATE -fsanitize=thread)
    
    target_compile_options(test_connection_pool PRIVATE -fsanitize=thread)
    target_link_options(test_connection_pool PRIVATE -fsanitize=thread)
    
    target_compile_options(test_query_cache PRIVATE -fsanitize=thread)
    target_link_options(test_query_cache PRIVATE -fsanitize=thread)
endif()

# Benchmark tests (optional)
if(BUILD_BENCHMARKS)
    find_package(benchmark REQUIRED)
    
    add_executable(benchmark_database
        benchmark_database.cpp
    )
    
    target_link_libraries(benchmark_database
        PRIVATE
            ultra_database
            ultra_common
            benchmark::benchmark
            benchmark::benchmark_main
            pthread
    )
    
    target_include_directories(benchmark_database
        PRIVATE
            ${CMAKE_SOURCE_DIR}/include
            ${CMAKE_SOURCE_DIR}/src
    )
    
    set_target_properties(benchmark_database
        PROPERTIES
            CXX_STANDARD 20
            CXX_STANDARD_REQUIRED ON
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/benchmarks/database
    )
endif()

# Coverage reporting
if(ENABLE_COVERAGE)
    target_compile_options(test_database_connector PRIVATE --coverage)
    target_link_options(test_database_connector PRIVATE --coverage)
    
    target_compile_options(test_connection_pool PRIVATE --coverage)
    target_link_options(test_connection_pool PRIVATE --coverage)
    
    target_compile_options(test_query_cache PRIVATE --coverage)
    target_link_options(test_query_cache PRIVATE --coverage)
endif()

# Custom test targets
add_custom_target(test_database
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -R "Database.*Test"
    DEPENDS test_database_connector test_connection_pool test_query_cache
    COMMENT "Running database tests"
)

add_custom_target(test_database_verbose
    COMMAND ${CMAKE_CTEST_COMMAND} --verbose -R "Database.*Test"
    DEPENDS test_database_connector test_connection_pool test_query_cache
    COMMENT "Running database tests with verbose output"
)

# Performance test target
add_custom_target(test_database_performance
    COMMAND test_database_connector --gtest_filter="*Performance*"
    COMMAND test_connection_pool --gtest_filter="*Performance*"
    COMMAND test_query_cache --gtest_filter="*Performance*"
    DEPENDS test_database_connector test_connection_pool test_query_cache
    COMMENT "Running database performance tests"
)

# Integration test target
add_custom_target(test_database_integration
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -R "Database.*Test" -E "DISABLED"
    DEPENDS test_database_connector test_connection_pool test_query_cache
    COMMENT "Running database integration tests (requires PostgreSQL)"
)

message(STATUS "Database tests configured:")
message(STATUS "  Unit tests: test_database_connector, test_connection_pool, test_query_cache")
message(STATUS "  Integration tests: ${ENABLE_INTEGRATION_TESTS}")
message(STATUS "  Benchmarks: ${BUILD_BENCHMARKS}")
message(STATUS "  Coverage: ${ENABLE_COVERAGE}")
message(STATUS "  AddressSanitizer: ${ENABLE_ASAN}")
message(STATUS "  ThreadSanitizer: ${ENABLE_TSAN}")