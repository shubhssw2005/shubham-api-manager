#include "gpu-compute/gpu_compute_engine.hpp"
#include "common/logger.hpp"
#include "common/error_handling.hpp"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <cufft.h>
#include <memory>
#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>

namespace ultra {
namespace gpu {

// CUDA error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        LOG_ERROR("CUDA error at {}:{}: {}", __FILE__, __LINE__, cudaGetErrorString(error)); \
        throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(error))); \
    } \
} while(0)

class GPUComputeEngine::Impl {
public:
    Config config_;
    cudaDeviceProp device_props_;
    cudaStream_t compute_stream_;
    cudaStream_t memory_stream_;
    cublasHandle_t cublas_handle_;
    curandGenerator_t curand_gen_;
    
    std::unique_ptr<CUDAMemoryPool> memory_pool_;
    std::unique_ptr<MLModelCache> model_cache_;
    
    mutable Stats stats_;
    mutable std::mutex stats_mutex_;
    
    bool initialized_ = false;

    explicit Impl(const Config& config) : config_(config) {}

    ~Impl() {
        if (initialized_) {
            shutdown();
        }
    }

    bool initialize() {
        try {
            // Set CUDA device
            CUDA_CHECK(cudaSetDevice(config_.device_id));
            
            // Get device properties
            CUDA_CHECK(cudaGetDeviceProperties(&device_props_, config_.device_id));
            
            LOG_INFO("Initializing GPU Compute Engine on device: {}", device_props_.name);
            LOG_INFO("Compute capability: {}.{}", device_props_.major, device_props_.minor);
            LOG_INFO("Total memory: {} MB", device_props_.totalGlobalMem / (1024 * 1024));
            
            // Create CUDA streams
            CUDA_CHECK(cudaStreamCreate(&compute_stream_));
            CUDA_CHECK(cudaStreamCreate(&memory_stream_));
            
            // Initialize cuBLAS
            if (cublasCreate(&cublas_handle_) != CUBLAS_STATUS_SUCCESS) {
                LOG_ERROR("Failed to create cuBLAS handle");
                return false;
            }
            cublasSetStream(cublas_handle_, compute_stream_);
            
            // Initialize cuRAND
            if (curandCreateGenerator(&curand_gen_, CURAND_RNG_PSEUDO_DEFAULT) != CURAND_STATUS_SUCCESS) {
                LOG_ERROR("Failed to create cuRAND generator");
                return false;
            }
            curandSetStream(curand_gen_, compute_stream_);
            
            // Initialize memory pool
            memory_pool_ = std::make_unique<CUDAMemoryPool>(config_.memory_pool_size);
            
            // Initialize model cache
            model_cache_ = std::make_unique<MLModelCache>(config_.model_cache_dir);
            
            initialized_ = true;
            LOG_INFO("GPU Compute Engine initialized successfully");
            return true;
            
        } catch (const std::exception& e) {
            LOG_ERROR("Failed to initialize GPU Compute Engine: {}", e.what());
            return false;
        }
    }

    void shutdown() {
        if (!initialized_) return;
        
        LOG_INFO("Shutting down GPU Compute Engine");
        
        // Synchronize all streams
        cudaStreamSynchronize(compute_stream_);
        cudaStreamSynchronize(memory_stream_);
        
        // Cleanup cuRAND
        if (curand_gen_) {
            curandDestroyGenerator(curand_gen_);
        }
        
        // Cleanup cuBLAS
        if (cublas_handle_) {
            cublasDestroy(cublas_handle_);
        }
        
        // Destroy streams
        cudaStreamDestroy(compute_stream_);
        cudaStreamDestroy(memory_stream_);
        
        // Reset device
        cudaDeviceReset();
        
        initialized_ = false;
        LOG_INFO("GPU Compute Engine shutdown complete");
    }

    std::vector<float> infer(const std::string& model_name, const std::vector<float>& input) {
        if (!initialized_) {
            throw std::runtime_error("GPU Compute Engine not initialized");
        }

        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Get model from cache
        auto model_info = model_cache_->get_model(model_name);
        if (!model_info) {
            throw std::runtime_error("Model not found: " + model_name);
        }

        // Allocate GPU memory for input and output
        float* d_input = static_cast<float*>(memory_pool_->allocate(input.size() * sizeof(float)));
        float* d_output = static_cast<float*>(memory_pool_->allocate(model_info->output_size * sizeof(float)));

        try {
            // Copy input to GPU
            CUDA_CHECK(cudaMemcpyAsync(d_input, input.data(), input.size() * sizeof(float),
                                     cudaMemcpyHostToDevice, memory_stream_));
            
            // Wait for memory transfer
            CUDA_CHECK(cudaStreamSynchronize(memory_stream_));
            
            // Run inference (placeholder - would use TensorRT here)
            // For now, just copy input to output as a placeholder
            CUDA_CHECK(cudaMemcpyAsync(d_output, d_input, 
                                     std::min(input.size(), model_info->output_size) * sizeof(float),
                                     cudaMemcpyDeviceToDevice, compute_stream_));
            
            // Wait for computation
            CUDA_CHECK(cudaStreamSynchronize(compute_stream_));
            
            // Copy result back to host
            std::vector<float> output(model_info->output_size);
            CUDA_CHECK(cudaMemcpyAsync(output.data(), d_output, output.size() * sizeof(float),
                                     cudaMemcpyDeviceToHost, memory_stream_));
            
            CUDA_CHECK(cudaStreamSynchronize(memory_stream_));
            
            // Update statistics
            {
                std::lock_guard<std::mutex> lock(stats_mutex_);
                stats_.operations_completed++;
                stats_.inference_count++;
                
                auto end_time = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
                stats_.total_processing_time_ns += duration.count();
            }
            
            return output;
            
        } catch (...) {
            memory_pool_->deallocate(d_input);
            memory_pool_->deallocate(d_output);
            throw;
        }
        
        memory_pool_->deallocate(d_input);
        memory_pool_->deallocate(d_output);
    }

    std::vector<std::vector<float>> infer_batch(const std::string& model_name,
                                              const std::vector<std::vector<float>>& inputs) {
        std::vector<std::vector<float>> results;
        results.reserve(inputs.size());
        
        for (const auto& input : inputs) {
            results.push_back(infer(model_name, input));
        }
        
        return results;
    }

    void* allocate_gpu_memory(size_t size) {
        if (!memory_pool_) {
            throw std::runtime_error("Memory pool not initialized");
        }
        return memory_pool_->allocate(size);
    }

    void free_gpu_memory(void* ptr) {
        if (memory_pool_) {
            memory_pool_->deallocate(ptr);
        }
    }

    GPUComputeEngine::DeviceInfo get_device_info() const {
        DeviceInfo info;
        info.name = device_props_.name;
        info.total_memory = device_props_.totalGlobalMem;
        
        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        info.free_memory = free_mem;
        
        info.compute_capability_major = device_props_.major;
        info.compute_capability_minor = device_props_.minor;
        info.multiprocessor_count = device_props_.multiProcessorCount;
        info.max_threads_per_block = device_props_.maxThreadsPerBlock;
        
        return info;
    }
};

// GPUComputeEngine implementation
GPUComputeEngine::GPUComputeEngine(const Config& config) 
    : pimpl_(std::make_unique<Impl>(config)) {
}

GPUComputeEngine::~GPUComputeEngine() = default;

bool GPUComputeEngine::initialize() {
    return pimpl_->initialize();
}

void GPUComputeEngine::shutdown() {
    pimpl_->shutdown();
}

std::vector<float> GPUComputeEngine::infer(const std::string& model_name, 
                                         const std::vector<float>& input) {
    return pimpl_->infer(model_name, input);
}

std::vector<std::vector<float>> GPUComputeEngine::infer_batch(const std::string& model_name,
                                                            const std::vector<std::vector<float>>& inputs) {
    return pimpl_->infer_batch(model_name, inputs);
}

bool GPUComputeEngine::resize_image_batch(const std::vector<ImageData>& inputs,
                                        std::vector<ImageData>& outputs,
                                        int target_width, int target_height) {
    // Implementation will be in image_processor.cu
    return true; // Placeholder
}

bool GPUComputeEngine::apply_filter_batch(const std::vector<ImageData>& inputs,
                                        std::vector<ImageData>& outputs,
                                        const std::string& filter_type) {
    // Implementation will be in image_processor.cu
    return true; // Placeholder
}

std::vector<uint8_t> GPUComputeEngine::compute_hash_batch(const std::vector<std::vector<uint8_t>>& data) {
    // Implementation will be in crypto_accelerator.cu
    return {}; // Placeholder
}

bool GPUComputeEngine::encrypt_batch(const std::vector<std::vector<uint8_t>>& plaintexts,
                                    std::vector<std::vector<uint8_t>>& ciphertexts,
                                    const std::vector<uint8_t>& key) {
    // Implementation will be in crypto_accelerator.cu
    return true; // Placeholder
}

void* GPUComputeEngine::allocate_gpu_memory(size_t size) {
    return pimpl_->allocate_gpu_memory(size);
}

void GPUComputeEngine::free_gpu_memory(void* ptr) {
    pimpl_->free_gpu_memory(ptr);
}

GPUComputeEngine::Stats GPUComputeEngine::get_stats() const {
    std::lock_guard<std::mutex> lock(pimpl_->stats_mutex_);
    return pimpl_->stats_;
}

void GPUComputeEngine::reset_stats() {
    std::lock_guard<std::mutex> lock(pimpl_->stats_mutex_);
    pimpl_->stats_ = Stats{};
}

GPUComputeEngine::DeviceInfo GPUComputeEngine::get_device_info() const {
    return pimpl_->get_device_info();
}

} // namespace gpu
} // namespace ultra