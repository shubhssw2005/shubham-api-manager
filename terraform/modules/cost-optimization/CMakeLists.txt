cmake_minimum_required(VERSION 3.13)
set(CMAKE_CXX_STANDARD 17)
project(cost-optimizer LANGUAGES CXX)

# Find required packages
find_package(AWSSDK REQUIRED COMPONENTS core ce sns s3 ec2)
find_package(aws-lambda-runtime REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(JSONCPP jsoncpp)

# Add executable
add_executable(${PROJECT_NAME} cost_optimizer.cpp)

# Link libraries
target_link_libraries(${PROJECT_NAME} 
    AWS::aws-lambda-runtime
    ${AWSSDK_LINK_LIBRARIES}
    ${JSONCPP_LIBRARIES}
)

target_include_directories(${PROJECT_NAME} PRIVATE ${JSONCPP_INCLUDE_DIRS})
target_compile_options(${PROJECT_NAME} PRIVATE ${JSONCPP_CFLAGS_OTHER})

# AWS Lambda requires the executable to be named 'bootstrap'
set_target_properties(${PROJECT_NAME} PROPERTIES OUTPUT_NAME bootstrap)

# Strip the binary to reduce size
if(CMAKE_BUILD_TYPE STREQUAL Release)
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_STRIP} $<TARGET_FILE:${PROJECT_NAME}>)
endif()