# Database connectivity layer CMakeLists.txt

# Find required packages
find_package(PkgConfig REQUIRED)

# Find PostgreSQL
find_package(PostgreSQL REQUIRED)
if(NOT PostgreSQL_FOUND)
    message(FATAL_ERROR "PostgreSQL development libraries not found")
endif()

# Find liburing
pkg_check_modules(LIBURING REQUIRED liburing)
if(NOT LIBURING_FOUND)
    message(FATAL_ERROR "liburing not found. Install with: sudo apt-get install liburing-dev")
endif()

# Database connectivity library
add_library(ultra_database STATIC
    database_connector.cpp
    connection_pool.cpp
    query_cache.cpp
    io_uring_manager.cpp
)

# Set target properties
set_target_properties(ultra_database PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
)

# Include directories
target_include_directories(ultra_database
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
        ${PostgreSQL_INCLUDE_DIRS}
        ${LIBURING_INCLUDE_DIRS}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link libraries
target_link_libraries(ultra_database
    PUBLIC
        ${PostgreSQL_LIBRARIES}
        ${LIBURING_LIBRARIES}
        ultra_common
        pthread
    PRIVATE
        ${CMAKE_DL_LIBS}
)

# Compiler definitions
target_compile_definitions(ultra_database
    PRIVATE
        _GNU_SOURCE
        LIBURING_AVAILABLE=1
)

# Compiler options for performance
target_compile_options(ultra_database
    PRIVATE
        -O3
        -march=native
        -mtune=native
        -flto
        -ffast-math
        -funroll-loops
        -Wall
        -Wextra
        -Wpedantic
        -Werror
        $<$<CONFIG:Debug>:-g3 -O0 -DDEBUG>
        $<$<CONFIG:Release>:-DNDEBUG>
)

# Link-time optimizations
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set_target_properties(ultra_database PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION TRUE
    )
endif()

# Install targets
install(TARGETS ultra_database
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

# Install headers
install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/database
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp"
)

# Add subdirectory for tests if building tests
if(BUILD_TESTING)
    add_subdirectory(${CMAKE_SOURCE_DIR}/tests/database)
endif()

# Performance profiling support
if(ENABLE_PROFILING)
    target_compile_options(ultra_database PRIVATE -pg)
    target_link_options(ultra_database PRIVATE -pg)
endif()

# Memory debugging support
if(ENABLE_ASAN)
    target_compile_options(ultra_database PRIVATE -fsanitize=address -fno-omit-frame-pointer)
    target_link_options(ultra_database PRIVATE -fsanitize=address)
endif()

if(ENABLE_TSAN)
    target_compile_options(ultra_database PRIVATE -fsanitize=thread)
    target_link_options(ultra_database PRIVATE -fsanitize=thread)
endif()

# Documentation
if(BUILD_DOCS)
    find_package(Doxygen)
    if(DOXYGEN_FOUND)
        set(DOXYGEN_INPUT_DIR ${CMAKE_SOURCE_DIR}/include/database)
        set(DOXYGEN_OUTPUT_DIR ${CMAKE_BINARY_DIR}/docs/database)
        
        configure_file(${CMAKE_SOURCE_DIR}/docs/Doxyfile.in 
                      ${CMAKE_BINARY_DIR}/Doxyfile.database @ONLY)
        
        add_custom_target(docs_database
            COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_BINARY_DIR}/Doxyfile.database
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            COMMENT "Generating database API documentation with Doxygen"
            VERBATIM
        )
    endif()
endif()

# Print configuration summary
message(STATUS "Database module configuration:")
message(STATUS "  PostgreSQL version: ${PostgreSQL_VERSION_STRING}")
message(STATUS "  PostgreSQL libraries: ${PostgreSQL_LIBRARIES}")
message(STATUS "  liburing version: ${LIBURING_VERSION}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Compiler optimizations: ${CMAKE_CXX_FLAGS_RELEASE}")