# Real-time stream processor
add_library(ultra_stream STATIC
    stream_processor.cpp
    event_handler.cpp
    simd_aggregator.cpp
    windowed_operations.cpp
)

target_include_directories(ultra_stream PUBLIC
    ${CMAKE_SOURCE_DIR}/include/stream-processor
)

target_link_libraries(ultra_stream
    ultra_common
)

# Enable SIMD optimizations
target_compile_options(ultra_stream PRIVATE
    -mavx2
    -mfma
    -march=native
)

# Stream processor service
add_executable(ultra-stream-processor
    stream_processor_main.cpp
)

target_link_libraries(ultra-stream-processor
    ultra_stream
    ultra_common
)

install(TARGETS ultra_stream ultra-stream-processor
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)