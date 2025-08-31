# Common utilities library
add_library(ultra_common STATIC
    memory_pool.cpp
    lock_free_queue.cpp
    performance_counter.cpp
    config_manager.cpp
    logger.cpp
    error_handling.cpp
    circuit_breaker.cpp
    degradation_manager.cpp
    fallback_manager.cpp
    resilience_system.cpp
    service_registry.cpp
    dynamic_routing.cpp
    feature_flags.cpp
)

target_include_directories(ultra_common PUBLIC
    ${CMAKE_SOURCE_DIR}/include/common
)

target_link_libraries(ultra_common
    ${DPDK_LIBRARIES}
    Threads::Threads
    curl
    nlohmann_json::nlohmann_json
)

# Install headers
install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/common/
        DESTINATION include/ultra-cpp/common)