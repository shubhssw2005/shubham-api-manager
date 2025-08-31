# Network Protocol Optimization Library

# Find required packages
find_package(PkgConfig REQUIRED)

# Find DPDK
pkg_check_modules(DPDK REQUIRED libdpdk)

# Create network library
add_library(ultra_network STATIC
    http_protocol.cpp
    tcp_connection.cpp
    load_balancer.cpp
    dpdk_network.cpp
)

# Set target properties
set_target_properties(ultra_network PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
)

# Include directories
target_include_directories(ultra_network
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
    PRIVATE
        ${DPDK_INCLUDE_DIRS}
)

# Compile definitions
target_compile_definitions(ultra_network
    PRIVATE
        ULTRA_DPDK_ENABLED
        _GNU_SOURCE
)

# Compiler flags for optimization
target_compile_options(ultra_network
    PRIVATE
        -O3
        -march=native
        -mtune=native
        -mavx2
        -msse4.2
        -ffast-math
        -funroll-loops
        -finline-functions
        -fomit-frame-pointer
        ${DPDK_CFLAGS}
)

# Link libraries
target_link_libraries(ultra_network
    PUBLIC
        ultra_common
        ultra_memory
        ultra_lockfree
    PRIVATE
        ${DPDK_LIBRARIES}
        pthread
        numa
        dl
)

# Link directories
target_link_directories(ultra_network
    PRIVATE
        ${DPDK_LIBRARY_DIRS}
)

# Install targets
install(TARGETS ultra_network
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

# Install headers
install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/network
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp"
)