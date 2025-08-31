cmake_minimum_required(VERSION 3.20)

# Find required packages
find_package(PkgConfig REQUIRED)
find_package(mongocxx REQUIRED)
find_package(bsoncxx REQUIRED)

# Create data generator executable
add_executable(massive_data_generator
    main.cpp
    massive_data_generator.cpp
)

# Link libraries
target_link_libraries(massive_data_generator
    PRIVATE
    common_lib
    mongo::mongocxx_shared
    mongo::bsoncxx_shared
    pthread
)

# Include directories
target_include_directories(massive_data_generator
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
)

# Compiler flags for maximum performance
target_compile_options(massive_data_generator PRIVATE
    -O3
    -march=native
    -mtune=native
    -flto
    -ffast-math
    -DNDEBUG
)

# Link-time optimization
set_target_properties(massive_data_generator PROPERTIES
    INTERPROCEDURAL_OPTIMIZATION TRUE
)

# Install target
install(TARGETS massive_data_generator
    RUNTIME DESTINATION bin
)