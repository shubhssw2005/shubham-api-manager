# GPU compute engine with CUDA support
if(CUDAToolkit_FOUND)
    add_library(ultra_gpu STATIC
        gpu_compute_engine.cpp
        cuda_memory_pool.cu
        ml_inference.cu
        image_processor.cu
        crypto_accelerator.cu
    )

    target_include_directories(ultra_gpu PUBLIC
        ${CMAKE_SOURCE_DIR}/include/gpu-compute
    )

    target_link_libraries(ultra_gpu
        ultra_common
        CUDA::cudart
        CUDA::cublas
        CUDA::curand
        CUDA::cufft
    )

    # CUDA compilation flags
    set_property(TARGET ultra_gpu PROPERTY CUDA_ARCHITECTURES 75 80 86 89)
    
    # GPU compute service
    add_executable(ultra-gpu-service
        gpu_service_main.cpp
    )

    target_link_libraries(ultra-gpu-service
        ultra_gpu
        ultra_common
    )

    install(TARGETS ultra_gpu ultra-gpu-service
            LIBRARY DESTINATION lib
            RUNTIME DESTINATION bin)
else()
    message(WARNING "CUDA not found, GPU compute engine will not be built")
endif()