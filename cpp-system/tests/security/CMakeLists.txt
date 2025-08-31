# Security tests CMakeLists.txt

# Find required packages
find_package(GTest REQUIRED)
find_package(GMock REQUIRED)

# Test sources
set(SECURITY_TEST_SOURCES
    test_security_integration.cpp
)

# Create test executable
add_executable(test_security_integration ${SECURITY_TEST_SOURCES})

# Include directories
target_include_directories(test_security_integration
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
        ${GTEST_INCLUDE_DIRS}
        ${GMOCK_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(test_security_integration
    PRIVATE
        ultra_security
        ultra_common
        GTest::GTest
        GTest::Main
        GMock::GMock
        GMock::Main
        pthread
)

# Compiler options
target_compile_options(test_security_integration
    PRIVATE
        -O2
        -march=native
        -mtune=native
        -mavx2
        -msse4.2
        -Wall
        -Wextra
)

# Set C++ standard
set_target_properties(test_security_integration PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

# Add test to CTest
add_test(NAME SecurityIntegrationTest COMMAND test_security_integration)

# Set test properties
set_tests_properties(SecurityIntegrationTest PROPERTIES
    TIMEOUT 300
    LABELS "security;integration"
)