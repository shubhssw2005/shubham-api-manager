# Examples for Ultra Low-Latency Memory Management
cmake_minimum_required(VERSION 3.20)

# Memory management demo
add_executable(memory_demo
    memory_demo.cpp
)

target_link_libraries(memory_demo
    PRIVATE
        ultra_memory
        Threads::Threads
)

target_include_directories(memory_demo
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
)

target_compile_features(memory_demo PRIVATE cxx_std_20)

target_compile_options(memory_demo PRIVATE
    -Wall -Wextra -Wpedantic
    -O2 -march=native
)

# Lock-free data structures demo
add_executable(lockfree_demo
    lockfree_demo.cpp
)

target_link_libraries(lockfree_demo
    PRIVATE
        lockfree
        Threads::Threads
)

target_include_directories(lockfree_demo
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
)

target_compile_features(lockfree_demo PRIVATE cxx_std_20)

target_compile_options(lockfree_demo PRIVATE
    -Wall -Wextra -Wpedantic
    -O2 -march=native
)

# Database connectivity demo
add_executable(database_demo
    database_demo.cpp
)

target_link_libraries(database_demo
    PRIVATE
        ultra_database
        ultra_common
        Threads::Threads
)

target_include_directories(database_demo
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
)

target_compile_features(database_demo PRIVATE cxx_std_20)

target_compile_options(database_demo PRIVATE
    -Wall -Wextra -Wpedantic
    -O2 -march=native
)

# Performance monitor demo
add_executable(performance_monitor_demo
    performance_monitor_demo.cpp
)

target_link_libraries(performance_monitor_demo
    PRIVATE
        ultra_monitor
        ultra_common
        Threads::Threads
)

target_include_directories(performance_monitor_demo
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
)

target_compile_features(performance_monitor_demo PRIVATE cxx_std_20)

target_compile_options(performance_monitor_demo PRIVATE
    -Wall -Wextra -Wpedantic
    -O2 -march=native
)

# Network Protocol Optimization demo
add_executable(network_protocol_demo
    network_protocol_demo.cpp
)

target_link_libraries(network_protocol_demo
    PRIVATE
        ultra_network
        ultra_common
        Threads::Threads
        ${DPDK_LIBRARIES}
)

target_include_directories(network_protocol_demo
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${DPDK_INCLUDE_DIRS}
)

target_compile_features(network_protocol_demo PRIVATE cxx_std_20)

target_compile_options(network_protocol_demo PRIVATE
    -Wall -Wextra -Wpedantic
    -O2 -march=native
    ${DPDK_CFLAGS}
)

target_compile_definitions(network_protocol_demo
    PRIVATE
        ULTRA_DPDK_ENABLED
        _GNU_SOURCE
)

# Security demo
add_executable(security_demo
    security_demo.cpp
)

target_link_libraries(security_demo
    PRIVATE
        ultra_security
        ultra_common
        Threads::Threads
)

target_include_directories(security_demo
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
)

target_compile_features(security_demo PRIVATE cxx_std_20)

target_compile_options(security_demo PRIVATE
    -Wall -Wextra -Wpedantic
    -O2 -march=native -mavx2 -msse4.2
)

# Configuration and Service Discovery demo
add_executable(config_service_discovery_demo
    config_service_discovery_demo.cpp
)

target_link_libraries(config_service_discovery_demo
    PRIVATE
        ultra_common
        Threads::Threads
        nlohmann_json::nlohmann_json
        CURL::libcurl
)

target_include_directories(config_service_discovery_demo
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
)

target_compile_features(config_service_discovery_demo PRIVATE cxx_std_20)

target_compile_options(config_service_discovery_demo PRIVATE
    -Wall -Wextra -Wpedantic
    -O2 -march=native
)

# Performance Optimization demo
add_executable(performance_optimization_demo
    performance_optimization_demo.cpp
)

target_link_libraries(performance_optimization_demo
    PRIVATE
        ultra_monitor
        ultra_common
        Threads::Threads
        numa
)

target_include_directories(performance_optimization_demo
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
)

target_compile_features(performance_optimization_demo PRIVATE cxx_std_20)

target_compile_options(performance_optimization_demo PRIVATE
    -Wall -Wextra -Wpedantic
    -O2 -march=native
)

# GPU Compute demo (only if CUDA is available)
if(CUDAToolkit_FOUND)
    add_executable(gpu_compute_demo
        gpu_compute_demo.cpp
    )

    target_link_libraries(gpu_compute_demo
        PRIVATE
            ultra_gpu
            ultra_common
            Threads::Threads
            CUDA::cudart
    )

    target_include_directories(gpu_compute_demo
        PRIVATE
            ${CMAKE_SOURCE_DIR}/include
    )

    target_compile_features(gpu_compute_demo PRIVATE cxx_std_20)

    target_compile_options(gpu_compute_demo PRIVATE
        -Wall -Wextra -Wpedantic
        -O2 -march=native
    )

    # Set CUDA architectures
    set_property(TARGET gpu_compute_demo PROPERTY CUDA_ARCHITECTURES 75 80 86 89)

    # Install GPU demo
    install(TARGETS memory_demo lockfree_demo database_demo performance_monitor_demo network_protocol_demo security_demo gpu_compute_demo config_service_discovery_demo performance_optimization_demo
        RUNTIME DESTINATION bin/examples
    )
else()
    # Install examples without GPU demo
    install(TARGETS memory_demo lockfree_demo database_demo performance_monitor_demo network_protocol_demo security_demo config_service_discovery_demo performance_optimization_demo
        RUNTIME DESTINATION bin/examples
    )
endif()