# Ultra-fast cache library
add_library(ultra_cache STATIC
    ultra_cache.cpp
    lock_free_hash_table.cpp
    lru_eviction.cpp
    rdma_replication.cpp
)

target_include_directories(ultra_cache PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/cache
    ${CMAKE_SOURCE_DIR}/include/common
    ${CMAKE_SOURCE_DIR}/include/lockfree
)

target_link_libraries(ultra_cache
    ultra_common
    ultra_lockfree
    pthread
)

# Add DPDK libraries if available
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(DPDK QUIET libdpdk)
    if(DPDK_FOUND)
        target_link_libraries(ultra_cache ${DPDK_LIBRARIES})
        target_include_directories(ultra_cache PRIVATE ${DPDK_INCLUDE_DIRS})
        target_compile_definitions(ultra_cache PRIVATE ULTRA_DPDK_ENABLED)
    endif()
endif()

# Cache service executable
add_executable(ultra-cache-service
    cache_service_main.cpp
)

target_link_libraries(ultra-cache-service
    ultra_cache
    ultra_common
)

# Cache benchmarking tool
add_executable(ultra-cache-benchmark
    cache_benchmark.cpp
)

target_link_libraries(ultra-cache-benchmark
    ultra_cache
    ultra_common
)

install(TARGETS ultra_cache ultra-cache-service ultra-cache-benchmark
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
        ARCHIVE DESTINATION lib)