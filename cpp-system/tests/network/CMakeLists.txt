# Network Protocol Optimization Tests

# Find required packages
find_package(GTest REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(DPDK REQUIRED libdpdk)

# Test sources
set(NETWORK_TEST_SOURCES
    test_http_protocol.cpp
    test_tcp_connection.cpp
    test_load_balancer.cpp
    test_dpdk_network.cpp
)

# Create test executable
add_executable(test_network ${NETWORK_TEST_SOURCES})

# Set target properties
set_target_properties(test_network PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

# Include directories
target_include_directories(test_network
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${DPDK_INCLUDE_DIRS}
        ${GTEST_INCLUDE_DIRS}
)

# Compile definitions
target_compile_definitions(test_network
    PRIVATE
        ULTRA_DPDK_ENABLED
        _GNU_SOURCE
)

# Compiler flags
target_compile_options(test_network
    PRIVATE
        -O2
        -g
        -Wall
        -Wextra
        ${DPDK_CFLAGS}
)

# Link libraries
target_link_libraries(test_network
    PRIVATE
        ultra_network
        ultra_common
        ultra_memory
        ultra_lockfree
        ${DPDK_LIBRARIES}
        ${GTEST_LIBRARIES}
        pthread
        numa
        dl
)

# Link directories
target_link_directories(test_network
    PRIVATE
        ${DPDK_LIBRARY_DIRS}
)

# Add to test suite
add_test(NAME NetworkProtocolTests COMMAND test_network)

# Set test properties
set_tests_properties(NetworkProtocolTests PROPERTIES
    TIMEOUT 300
    LABELS "network;protocol;performance"
)