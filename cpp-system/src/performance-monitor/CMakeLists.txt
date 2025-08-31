# Performance monitoring system
add_library(ultra_monitor STATIC
    performance_monitor.cpp
    hardware_counters.cpp
    metrics_collector.cpp
    prometheus_exporter.cpp
    slo_monitor.cpp
    cpu_affinity.cpp
    cache_optimization.cpp
    compiler_optimization.cpp
    adaptive_tuning.cpp
)

target_include_directories(ultra_monitor PUBLIC
    ${CMAKE_SOURCE_DIR}/include/performance-monitor
)

# Find required packages
find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBCURL REQUIRED libcurl)

target_link_libraries(ultra_monitor
    ultra_common
    ${LIBCURL_LIBRARIES}
    z  # zlib for compression
    pthread
    numa  # NUMA support
)

target_include_directories(ultra_monitor PRIVATE
    ${LIBCURL_INCLUDE_DIRS}
)

target_compile_options(ultra_monitor PRIVATE
    ${LIBCURL_CFLAGS_OTHER}
)

# Performance monitor service
add_executable(ultra-monitor-service
    monitor_service_main.cpp
)

target_link_libraries(ultra-monitor-service
    ultra_monitor
    ultra_common
)

install(TARGETS ultra_monitor ultra-monitor-service
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)