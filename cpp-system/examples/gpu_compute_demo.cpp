#include "gpu-compute/gpu_compute_engine.hpp"
#include "common/logger.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <random>

using namespace ultra::gpu;

void demonstrate_ml_inference(GPUComputeEngine& engine) {
    std::cout << "\n=== ML Inference Demo ===" << std::endl;
    
    // Create sample input data
    std::vector<float> input_data(1024);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (auto& val : input_data) {
        val = dis(gen);
    }
    
    try {
        auto start = std::chrono::high_resolution_clock::now();
        
        // Note: This will fail since no model is loaded, but demonstrates the API
        auto result = engine.infer("demo_model", input_data);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        std::cout << "Inference completed in " << duration.count() << " μs" << std::endl;
        std::cout << "Output size: " << result.size() << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "Inference failed (expected): " << e.what() << std::endl;
    }
    
    // Demonstrate batch inference
    std::vector<std::vector<float>> batch_inputs;
    for (int i = 0; i < 4; ++i) {
        std::vector<float> batch_input(512);
        for (auto& val : batch_input) {
            val = dis(gen);
        }
        batch_inputs.push_back(batch_input);
    }
    
    try {
        auto start = std::chrono::high_resolution_clock::now();
        
        auto batch_results = engine.infer_batch("demo_model", batch_inputs);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        std::cout << "Batch inference (" << batch_inputs.size() << " items) completed in " 
                  << duration.count() << " μs" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "Batch inference failed (expected): " << e.what() << std::endl;
    }
}

void demonstrate_image_processing(GPUComputeEngine& engine) {
    std::cout << "\n=== Image Processing Demo ===" << std::endl;
    
    // Create sample image data
    const int width = 256;
    const int height = 256;
    const int channels = 3;
    
    std::vector<ImageData> input_images;
    std::vector<ImageData> output_images;
    
    // Allocate GPU memory for test image
    ImageData test_image;
    test_image.width = width;
    test_image.height = height;
    test_image.channels = channels;
    test_image.pitch = width * channels;
    
    size_t image_size = height * test_image.pitch;
    test_image.data = static_cast<uint8_t*>(engine.allocate_gpu_memory(image_size));
    
    if (test_image.data) {
        input_images.push_back(test_image);
        
        // Demonstrate image resizing
        auto start = std::chrono::high_resolution_clock::now();
        
        bool success = engine.resize_image_batch(input_images, output_images, 128, 128);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        std::cout << "Image resize (" << width << "x" << height << " -> 128x128) ";
        if (success) {
            std::cout << "completed in " << duration.count() << " μs" << std::endl;
        } else {
            std::cout << "failed" << std::endl;
        }
        
        // Demonstrate filter application
        start = std::chrono::high_resolution_clock::now();
        
        success = engine.apply_filter_batch(input_images, output_images, "gaussian_blur");
        
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        std::cout << "Gaussian blur filter ";
        if (success) {
            std::cout << "completed in " << duration.count() << " μs" << std::endl;
        } else {
            std::cout << "failed" << std::endl;
        }
        
        // Cleanup
        engine.free_gpu_memory(test_image.data);
    } else {
        std::cout << "Failed to allocate GPU memory for image processing demo" << std::endl;
    }
}

void demonstrate_crypto_operations(GPUComputeEngine& engine) {
    std::cout << "\n=== Cryptographic Operations Demo ===" << std::endl;
    
    // Create sample data for hashing
    std::vector<std::vector<uint8_t>> hash_data = {
        {'H', 'e', 'l', 'l', 'o', ' ', 'W', 'o', 'r', 'l', 'd'},
        {'G', 'P', 'U', ' ', 'A', 'c', 'c', 'e', 'l', 'e', 'r', 'a', 't', 'e', 'd'},
        {'C', 'r', 'y', 'p', 't', 'o', 'g', 'r', 'a', 'p', 'h', 'y'}
    };
    
    auto start = std::chrono::high_resolution_clock::now();
    
    auto hashes = engine.compute_hash_batch(hash_data);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "Batch hashing (" << hash_data.size() << " items) completed in " 
              << duration.count() << " μs" << std::endl;
    std::cout << "Hash result size: " << hashes.size() << " bytes" << std::endl;
    
    // Demonstrate encryption
    std::vector<std::vector<uint8_t>> plaintexts = hash_data;
    std::vector<std::vector<uint8_t>> ciphertexts;
    std::vector<uint8_t> key(16, 0x42); // Simple test key
    
    start = std::chrono::high_resolution_clock::now();
    
    bool success = engine.encrypt_batch(plaintexts, ciphertexts, key);
    
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "Batch encryption (" << plaintexts.size() << " items) ";
    if (success) {
        std::cout << "completed in " << duration.count() << " μs" << std::endl;
    } else {
        std::cout << "failed" << std::endl;
    }
}

void demonstrate_memory_management(GPUComputeEngine& engine) {
    std::cout << "\n=== Memory Management Demo ===" << std::endl;
    
    // Test various allocation sizes
    std::vector<void*> allocations;
    std::vector<size_t> sizes = {1024, 4096, 16384, 65536, 262144}; // 1KB to 256KB
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Allocate memory blocks
    for (size_t size : sizes) {
        void* ptr = engine.allocate_gpu_memory(size);
        if (ptr) {
            allocations.push_back(ptr);
            std::cout << "Allocated " << size << " bytes at " << ptr << std::endl;
        } else {
            std::cout << "Failed to allocate " << size << " bytes" << std::endl;
        }
    }
    
    auto mid = std::chrono::high_resolution_clock::now();
    
    // Free all allocations
    for (void* ptr : allocations) {
        engine.free_gpu_memory(ptr);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    
    auto alloc_duration = std::chrono::duration_cast<std::chrono::microseconds>(mid - start);
    auto free_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - mid);
    
    std::cout << "Allocation time: " << alloc_duration.count() << " μs" << std::endl;
    std::cout << "Deallocation time: " << free_duration.count() << " μs" << std::endl;
}

void print_device_info(const GPUComputeEngine& engine) {
    std::cout << "\n=== GPU Device Information ===" << std::endl;
    
    auto device_info = engine.get_device_info();
    
    std::cout << "Device Name: " << device_info.name << std::endl;
    std::cout << "Total Memory: " << device_info.total_memory / (1024 * 1024) << " MB" << std::endl;
    std::cout << "Free Memory: " << device_info.free_memory / (1024 * 1024) << " MB" << std::endl;
    std::cout << "Compute Capability: " << device_info.compute_capability_major 
              << "." << device_info.compute_capability_minor << std::endl;
    std::cout << "Multiprocessors: " << device_info.multiprocessor_count << std::endl;
    std::cout << "Max Threads per Block: " << device_info.max_threads_per_block << std::endl;
}

void print_statistics(const GPUComputeEngine& engine) {
    std::cout << "\n=== GPU Statistics ===" << std::endl;
    
    auto stats = engine.get_stats();
    
    std::cout << "Operations Completed: " << stats.operations_completed.load() << std::endl;
    std::cout << "Inference Operations: " << stats.inference_count.load() << std::endl;
    std::cout << "Image Operations: " << stats.image_ops_count.load() << std::endl;
    std::cout << "Crypto Operations: " << stats.crypto_ops_count.load() << std::endl;
    std::cout << "GPU Utilization: " << stats.gpu_utilization_percent.load() << "%" << std::endl;
    std::cout << "Memory Used: " << stats.memory_used_bytes.load() / (1024 * 1024) << " MB" << std::endl;
    
    if (stats.operations_completed.load() > 0) {
        uint64_t avg_time = stats.total_processing_time_ns.load() / stats.operations_completed.load();
        std::cout << "Average Processing Time: " << avg_time << " ns" << std::endl;
    }
}

int main() {
    // Initialize logging
    ultra::common::Logger::initialize("gpu-demo", ultra::common::LogLevel::INFO);
    
    std::cout << "Ultra Low-Latency GPU Compute Engine Demo" << std::endl;
    std::cout << "=========================================" << std::endl;
    
    try {
        // Configure GPU engine
        GPUComputeEngine::Config config;
        config.device_id = 0;
        config.memory_pool_size = 256 * 1024 * 1024; // 256MB
        config.enable_tensorrt = false; // Disable for demo
        config.max_batch_size = 16;
        config.model_cache_dir = "/tmp/demo_models";
        
        // Create and initialize GPU engine
        GPUComputeEngine gpu_engine(config);
        
        if (!gpu_engine.initialize()) {
            std::cerr << "Failed to initialize GPU Compute Engine" << std::endl;
            return 1;
        }
        
        // Print device information
        print_device_info(gpu_engine);
        
        // Reset statistics for clean demo
        gpu_engine.reset_stats();
        
        // Run demonstrations
        demonstrate_memory_management(gpu_engine);
        demonstrate_ml_inference(gpu_engine);
        demonstrate_image_processing(gpu_engine);
        demonstrate_crypto_operations(gpu_engine);
        
        // Print final statistics
        print_statistics(gpu_engine);
        
        // Shutdown
        gpu_engine.shutdown();
        
        std::cout << "\nDemo completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Demo failed: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}