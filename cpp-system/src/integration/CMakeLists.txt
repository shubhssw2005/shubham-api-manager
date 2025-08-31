# Integration module CMakeLists.txt

# Find required packages
find_package(PkgConfig REQUIRED)

# Find libcurl
find_package(CURL REQUIRED)

# Find hiredis (Redis client)
pkg_check_modules(HIREDIS REQUIRED hiredis)

# Find jsoncpp
find_package(PkgConfig REQUIRED)
pkg_check_modules(JSONCPP REQUIRED jsoncpp)

# Find UUID library
find_library(UUID_LIBRARY uuid REQUIRED)

# Find OpenSSL for encryption
find_package(OpenSSL REQUIRED)

# Find zlib for compression
find_package(ZLIB REQUIRED)

# Integration library sources
set(INTEGRATION_SOURCES
    http_proxy.cpp
    session_manager.cpp
    event_bridge.cpp
    monitoring_integration.cpp
)

# Create integration library
add_library(integration STATIC ${INTEGRATION_SOURCES})

# Set target properties
set_target_properties(integration PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
)

# Include directories
target_include_directories(integration
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
    PRIVATE
        ${HIREDIS_INCLUDE_DIRS}
        ${JSONCPP_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(integration
    PUBLIC
        common
        performance-monitor
        CURL::libcurl
        ${HIREDIS_LIBRARIES}
        ${JSONCPP_LIBRARIES}
        ${UUID_LIBRARY}
        OpenSSL::SSL
        OpenSSL::Crypto
        ZLIB::ZLIB
    PRIVATE
        pthread
)

# Compiler definitions
target_compile_definitions(integration
    PRIVATE
        ${HIREDIS_CFLAGS_OTHER}
        ${JSONCPP_CFLAGS_OTHER}
)

# Compiler options
target_compile_options(integration
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -O3
        -march=native
        -mtune=native
        -ffast-math
        -funroll-loops
        -fomit-frame-pointer
        ${HIREDIS_CFLAGS_OTHER}
        ${JSONCPP_CFLAGS_OTHER}
)

# Install targets
install(TARGETS integration
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

# Install headers
install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/integration
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp"
)