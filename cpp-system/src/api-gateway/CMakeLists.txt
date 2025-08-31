# Ultra-fast API Gateway
add_executable(ultra-api-gateway
    main.cpp
    fast_api_gateway.cpp
    http_parser.cpp
    request_router.cpp
    connection_manager.cpp
)

target_include_directories(ultra-api-gateway PRIVATE
    ${CMAKE_SOURCE_DIR}/include/api-gateway
)

target_link_libraries(ultra-api-gateway
    ultra_common
    ${DPDK_LIBRARIES}
    Threads::Threads
)

# Compiler optimizations for API gateway
target_compile_options(ultra-api-gateway PRIVATE
    -ffast-math
    -funroll-loops
    -finline-functions
)

install(TARGETS ultra-api-gateway DESTINATION bin)