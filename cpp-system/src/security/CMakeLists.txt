# Security module CMakeLists.txt

# Find required packages
find_package(OpenSSL REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(JSONCPP jsoncpp)

# Security library sources
set(SECURITY_SOURCES
    jwt_validator.cpp
    rate_limiter.cpp
    request_sanitizer.cpp
    audit_logger.cpp
)

# Create security library
add_library(ultra_security STATIC ${SECURITY_SOURCES})

# Include directories
target_include_directories(ultra_security
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link libraries
target_link_libraries(ultra_security
    PUBLIC
        OpenSSL::SSL
        OpenSSL::Crypto
        ${JSONCPP_LIBRARIES}
        ultra_common
    PRIVATE
        pthread
)

# Compiler definitions
target_compile_definitions(ultra_security
    PRIVATE
        -DOPENSSL_API_COMPAT=0x10100000L
)

# Compiler options for performance
target_compile_options(ultra_security
    PRIVATE
        -O3
        -march=native
        -mtune=native
        -mavx2
        -msse4.2
        -Wall
        -Wextra
        -Werror
)

# Set C++ standard
set_target_properties(ultra_security PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

# Install targets
install(TARGETS ultra_security
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

# Install headers
install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/security
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp"
)