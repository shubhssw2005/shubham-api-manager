#pragma once

#include <memory>
#include <vector>
#include <string>
#include <functional>
#include <atomic>
#include <optional>
#include <unordered_map>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <cufft.h>

namespace ultra {
namespace gpu {

struct ImageData {
    uint8_t* data;
    int width;
    int height;
    int channels;
    size_t pitch;
};

class GPUComputeEngine {
public:
    struct Config {
        int device_id = 0;
        size_t memory_pool_size = 512 * 1024 * 1024; // 512MB
        bool enable_tensorrt = true;
        size_t max_batch_size = 32;
        std::string model_cache_dir = "/tmp/ultra_models";
    };

    explicit GPUComputeEngine(const Config& config);
    ~GPUComputeEngine();

    // Initialization and cleanup
    bool initialize();
    void shutdown();

    // ML Inference
    std::vector<float> infer(const std::string& model_name, 
                           const std::vector<float>& input);
    
    std::vector<std::vector<float>> infer_batch(const std::string& model_name,
                                              const std::vector<std::vector<float>>& inputs);

    // Image Processing
    bool resize_image_batch(const std::vector<ImageData>& inputs,
                          std::vector<ImageData>& outputs,
                          int target_width, int target_height);
    
    bool apply_filter_batch(const std::vector<ImageData>& inputs,
                          std::vector<ImageData>& outputs,
                          const std::string& filter_type);

    // Cryptographic Operations
    std::vector<uint8_t> compute_hash_batch(const std::vector<std::vector<uint8_t>>& data);
    
    bool encrypt_batch(const std::vector<std::vector<uint8_t>>& plaintexts,
                      std::vector<std::vector<uint8_t>>& ciphertexts,
                      const std::vector<uint8_t>& key);

    // Memory management
    void* allocate_gpu_memory(size_t size);
    void free_gpu_memory(void* ptr);
    
    // Statistics and monitoring
    struct Stats {
        std::atomic<uint64_t> operations_completed{0};
        std::atomic<uint64_t> gpu_utilization_percent{0};
        std::atomic<uint64_t> memory_used_bytes{0};
        std::atomic<uint64_t> inference_count{0};
        std::atomic<uint64_t> image_ops_count{0};
        std::atomic<uint64_t> crypto_ops_count{0};
        std::atomic<uint64_t> total_processing_time_ns{0};
    };
    
    Stats get_stats() const;
    void reset_stats();

    // Device information
    struct DeviceInfo {
        std::string name;
        size_t total_memory;
        size_t free_memory;
        int compute_capability_major;
        int compute_capability_minor;
        int multiprocessor_count;
        int max_threads_per_block;
    };
    
    DeviceInfo get_device_info() const;

private:
    class Impl;
    std::unique_ptr<Impl> pimpl_;
};

// GPU Memory Pool for efficient allocation
class CUDAMemoryPool {
public:
    explicit CUDAMemoryPool(size_t pool_size);
    ~CUDAMemoryPool();

    void* allocate(size_t size, size_t alignment = 256);
    void deallocate(void* ptr);
    
    size_t get_total_size() const;
    size_t get_used_size() const;
    size_t get_free_size() const;

private:
    class Impl;
    std::unique_ptr<Impl> pimpl_;
};

// ML Model Cache for TensorRT models
class MLModelCache {
public:
    struct ModelInfo {
        std::string name;
        std::string path;
        size_t input_size;
        size_t output_size;
        void* engine_ptr;
        void* context_ptr;
    };

    explicit MLModelCache(const std::string& cache_dir);
    ~MLModelCache();

    bool load_model(const std::string& name, const std::string& model_path);
    bool unload_model(const std::string& name);
    
    std::optional<ModelInfo> get_model(const std::string& name);
    std::vector<std::string> list_models() const;

private:
    class Impl;
    std::unique_ptr<Impl> pimpl_;
};

// Image processing kernels
namespace kernels {
    // Resize operations
    cudaError_t launch_resize_kernel(const ImageData& input, ImageData& output,
                                   int target_width, int target_height,
                                   cudaStream_t stream = nullptr);
    
    // Filter operations
    cudaError_t launch_gaussian_blur(const ImageData& input, ImageData& output,
                                   float sigma, cudaStream_t stream = nullptr);
    
    cudaError_t launch_edge_detection(const ImageData& input, ImageData& output,
                                    cudaStream_t stream = nullptr);
    
    // Color space conversions
    cudaError_t launch_rgb_to_yuv(const ImageData& input, ImageData& output,
                                cudaStream_t stream = nullptr);
}

// Crypto acceleration kernels
namespace crypto {
    cudaError_t launch_sha256_batch(const std::vector<std::vector<uint8_t>>& inputs,
                                  std::vector<std::vector<uint8_t>>& outputs,
                                  cudaStream_t stream = nullptr);
    
    cudaError_t launch_aes_encrypt_batch(const std::vector<std::vector<uint8_t>>& plaintexts,
                                       std::vector<std::vector<uint8_t>>& ciphertexts,
                                       const std::vector<uint8_t>& key,
                                       cudaStream_t stream = nullptr);
}

} // namespace gpu
} // namespace ultra