#include "gpu-compute/gpu_compute_engine.hpp"
#include "common/logger.hpp"
#include "common/config_manager.hpp"
#include <iostream>
#include <memory>
#include <signal.h>
#include <thread>
#include <chrono>

namespace ultra {
namespace gpu {

class GPUService {
public:
    GPUService() = default;
    ~GPUService() = default;

    bool initialize() {
        LOG_INFO("Initializing GPU Compute Service");
        
        // Load configuration
        GPUComputeEngine::Config config;
        config.device_id = 0;
        config.memory_pool_size = 512 * 1024 * 1024; // 512MB
        config.enable_tensorrt = true;
        config.max_batch_size = 32;
        config.model_cache_dir = "/tmp/ultra_models";
        
        // Initialize GPU compute engine
        gpu_engine_ = std::make_unique<GPUComputeEngine>(config);
        
        if (!gpu_engine_->initialize()) {
            LOG_ERROR("Failed to initialize GPU Compute Engine");
            return false;
        }
        
        // Print device information
        auto device_info = gpu_engine_->get_device_info();
        LOG_INFO("GPU Device: {}", device_info.name);
        LOG_INFO("Total Memory: {} MB", device_info.total_memory / (1024 * 1024));
        LOG_INFO("Free Memory: {} MB", device_info.free_memory / (1024 * 1024));
        LOG_INFO("Compute Capability: {}.{}", 
                device_info.compute_capability_major, 
                device_info.compute_capability_minor);
        LOG_INFO("Multiprocessors: {}", device_info.multiprocessor_count);
        
        running_ = true;
        LOG_INFO("GPU Compute Service initialized successfully");
        return true;
    }
    
    void run() {
        LOG_INFO("Starting GPU Compute Service");
        
        // Start statistics reporting thread
        std::thread stats_thread([this]() {
            while (running_) {
                std::this_thread::sleep_for(std::chrono::seconds(10));
                report_statistics();
            }
        });
        
        // Main service loop
        while (running_) {
            // In a real implementation, this would handle incoming requests
            // For now, just demonstrate some GPU operations
            
            try {
                // Example ML inference
                std::vector<float> test_input(1024, 1.0f);
                auto result = gpu_engine_->infer("test_model", test_input);
                
                // Example image processing
                std::vector<ImageData> test_images;
                std::vector<ImageData> processed_images;
                gpu_engine_->resize_image_batch(test_images, processed_images, 256, 256);
                
                // Example crypto operations
                std::vector<std::vector<uint8_t>> test_data = {
                    {0x01, 0x02, 0x03, 0x04},
                    {0x05, 0x06, 0x07, 0x08}
                };
                auto hashes = gpu_engine_->compute_hash_batch(test_data);
                
            } catch (const std::exception& e) {
                LOG_WARNING("GPU operation failed: {}", e.what());
            }
            
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        
        stats_thread.join();
        LOG_INFO("GPU Compute Service stopped");
    }
    
    void shutdown() {
        LOG_INFO("Shutting down GPU Compute Service");
        running_ = false;
        
        if (gpu_engine_) {
            gpu_engine_->shutdown();
        }
    }

private:
    void report_statistics() {
        if (!gpu_engine_) return;
        
        auto stats = gpu_engine_->get_stats();
        auto device_info = gpu_engine_->get_device_info();
        
        LOG_INFO("=== GPU Statistics ===");
        LOG_INFO("Operations completed: {}", stats.operations_completed.load());
        LOG_INFO("Inference operations: {}", stats.inference_count.load());
        LOG_INFO("Image operations: {}", stats.image_ops_count.load());
        LOG_INFO("Crypto operations: {}", stats.crypto_ops_count.load());
        LOG_INFO("GPU utilization: {}%", stats.gpu_utilization_percent.load());
        LOG_INFO("Memory used: {} MB", stats.memory_used_bytes.load() / (1024 * 1024));
        LOG_INFO("Free memory: {} MB", device_info.free_memory / (1024 * 1024));
        
        if (stats.operations_completed.load() > 0) {
            uint64_t avg_time = stats.total_processing_time_ns.load() / stats.operations_completed.load();
            LOG_INFO("Average processing time: {} ns", avg_time);
        }
        LOG_INFO("=====================");
    }

    std::unique_ptr<GPUComputeEngine> gpu_engine_;
    std::atomic<bool> running_{false};
};

} // namespace gpu
} // namespace ultra

// Global service instance
std::unique_ptr<ultra::gpu::GPUService> g_service;

// Signal handler for graceful shutdown
void signal_handler(int signal) {
    LOG_INFO("Received signal {}, shutting down...", signal);
    if (g_service) {
        g_service->shutdown();
    }
}

int main(int argc, char* argv[]) {
    // Initialize logging
    ultra::common::Logger::initialize("gpu-service", ultra::common::LogLevel::INFO);
    
    LOG_INFO("Ultra Low-Latency GPU Compute Service starting...");
    
    // Set up signal handlers
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    try {
        // Create and initialize service
        g_service = std::make_unique<ultra::gpu::GPUService>();
        
        if (!g_service->initialize()) {
            LOG_ERROR("Failed to initialize GPU service");
            return 1;
        }
        
        // Run service
        g_service->run();
        
    } catch (const std::exception& e) {
        LOG_ERROR("GPU service error: {}", e.what());
        return 1;
    }
    
    LOG_INFO("GPU Compute Service terminated");
    return 0;
}