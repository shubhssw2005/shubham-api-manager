# GPU Compute Tests
if(CUDAToolkit_FOUND)
    # Find Google Test
    find_package(GTest REQUIRED)
    
    # Create test executable
    add_executable(test_gpu_compute
        test_gpu_compute.cpp
    )
    
    target_include_directories(test_gpu_compute PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
    )
    
    target_link_libraries(test_gpu_compute
        ultra_gpu
        ultra_common
        GTest::gtest
        GTest::gtest_main
        CUDA::cudart
    )
    
    # Set CUDA architectures
    set_property(TARGET test_gpu_compute PROPERTY CUDA_ARCHITECTURES 75 80 86 89)
    
    # Add test to CTest
    add_test(NAME GPUComputeTests COMMAND test_gpu_compute)
    
    # Set test properties
    set_tests_properties(GPUComputeTests PROPERTIES
        TIMEOUT 300
        LABELS "gpu;compute;cuda"
    )
    
    # Install test executable
    install(TARGETS test_gpu_compute
            RUNTIME DESTINATION bin/tests)
            
else()
    message(STATUS "CUDA not found, GPU compute tests will not be built")
endif()