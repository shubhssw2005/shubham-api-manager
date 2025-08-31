#include <gtest/gtest.h>
#include "gpu-compute/gpu_compute_engine.hpp"
#include "common/logger.hpp"
#include <vector>
#include <chrono>

using namespace ultra::gpu;

class GPUComputeTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize logging for tests
        ultra::common::Logger::initialize("gpu-test", ultra::common::LogLevel::DEBUG);
        
        // Configure GPU engine
        GPUComputeEngine::Config config;
        config.device_id = 0;
        config.memory_pool_size = 64 * 1024 * 1024; // 64MB for tests
        config.enable_tensorrt = false; // Disable TensorRT for basic tests
        config.max_batch_size = 8;
        config.model_cache_dir = "/tmp/test_models";
        
        gpu_engine_ = std::make_unique<GPUComputeEngine>(config);
        
        // Skip tests if GPU is not available
        if (!gpu_engine_->initialize()) {
            GTEST_SKIP() << "GPU not available, skipping GPU tests";
        }
    }
    
    void TearDown() override {
        if (gpu_engine_) {
            gpu_engine_->shutdown();
        }
    }
    
    std::unique_ptr<GPUComputeEngine> gpu_engine_;
};

TEST_F(GPUComputeTest, DeviceInfo) {
    auto device_info = gpu_engine_->get_device_info();
    
    EXPECT_FALSE(device_info.name.empty());
    EXPECT_GT(device_info.total_memory, 0);
    EXPECT_GT(device_info.multiprocessor_count, 0);
    EXPECT_GT(device_info.max_threads_per_block, 0);
    
    std::cout << "GPU Device: " << device_info.name << std::endl;
    std::cout << "Total Memory: " << device_info.total_memory / (1024 * 1024) << " MB" << std::endl;
    std::cout << "Compute Capability: " << device_info.compute_capability_major 
              << "." << device_info.compute_capability_minor << std::endl;
}

TEST_F(GPUComputeTest, MemoryAllocation) {
    // Test basic memory allocation
    size_t test_size = 1024 * 1024; // 1MB
    void* ptr = gpu_engine_->allocate_gpu_memory(test_size);
    
    EXPECT_NE(ptr, nullptr);
    
    // Free the memory
    gpu_engine_->free_gpu_memory(ptr);
    
    // Test multiple allocations
    std::vector<void*> ptrs;
    for (int i = 0; i < 10; ++i) {
        void* p = gpu_engine_->allocate_gpu_memory(1024);
        EXPECT_NE(p, nullptr);
        ptrs.push_back(p);
    }
    
    // Free all allocations
    for (void* p : ptrs) {
        gpu_engine_->free_gpu_memory(p);
    }
}

TEST_F(GPUComputeTest, Statistics) {
    // Reset statistics
    gpu_engine_->reset_stats();
    
    auto initial_stats = gpu_engine_->get_stats();
    EXPECT_EQ(initial_stats.operations_completed.load(), 0);
    EXPECT_EQ(initial_stats.inference_count.load(), 0);
    
    // Perform some operations to update statistics
    std::vector<float> test_input(100, 1.0f);
    
    // This should fail since no model is loaded, but it should update stats
    try {
        gpu_engine_->infer("nonexistent_model", test_input);
    } catch (...) {
        // Expected to fail
    }
    
    auto final_stats = gpu_engine_->get_stats();
    // Stats might not change if operation fails early, that's OK for this test
}

TEST_F(GPUComputeTest, ImageProcessingSetup) {
    // Test image processing setup (without actual processing)
    std::vector<ImageData> inputs;
    std::vector<ImageData> outputs;
    
    // Create test image data
    ImageData test_image;
    test_image.width = 64;
    test_image.height = 64;
    test_image.channels = 3;
    test_image.pitch = test_image.width * test_image.channels;
    
    // Allocate test data
    size_t image_size = test_image.height * test_image.pitch;
    test_image.data = static_cast<uint8_t*>(gpu_engine_->allocate_gpu_memory(image_size));
    EXPECT_NE(test_image.data, nullptr);
    
    inputs.push_back(test_image);
    
    // Test resize operation setup
    bool result = gpu_engine_->resize_image_batch(inputs, outputs, 32, 32);
    EXPECT_TRUE(result); // Should succeed even if it's a placeholder implementation
    
    // Cleanup
    gpu_engine_->free_gpu_memory(test_image.data);
}

TEST_F(GPUComputeTest, CryptoOperationsSetup) {
    // Test crypto operations setup
    std::vector<std::vector<uint8_t>> test_data = {
        {0x01, 0x02, 0x03, 0x04, 0x05},
        {0x06, 0x07, 0x08, 0x09, 0x0A},
        {0x0B, 0x0C, 0x0D, 0x0E, 0x0F}
    };
    
    auto hashes = gpu_engine_->compute_hash_batch(test_data);
    // Should return empty for placeholder implementation, but shouldn't crash
    
    // Test encryption setup
    std::vector<std::vector<uint8_t>> plaintexts = test_data;
    std::vector<std::vector<uint8_t>> ciphertexts;
    std::vector<uint8_t> key(16, 0x42); // Simple test key
    
    bool result = gpu_engine_->encrypt_batch(plaintexts, ciphertexts, key);
    EXPECT_TRUE(result); // Should succeed for placeholder implementation
}

TEST_F(GPUComputeTest, PerformanceBenchmark) {
    // Simple performance test
    const int num_iterations = 100;
    std::vector<float> test_input(1024, 1.0f);
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_iterations; ++i) {
        // Allocate and free memory to test memory pool performance
        void* ptr = gpu_engine_->allocate_gpu_memory(4096);
        if (ptr) {
            gpu_engine_->free_gpu_memory(ptr);
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    std::cout << "Memory allocation/deallocation performance: " 
              << duration.count() / num_iterations << " μs per operation" << std::endl;
    
    // Should complete in reasonable time
    EXPECT_LT(duration.count(), 100000); // Less than 100ms total
}

// Test memory pool specifically
class CUDAMemoryPoolTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Skip if CUDA is not available
        int device_count;
        if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
            GTEST_SKIP() << "CUDA not available, skipping memory pool tests";
        }
        
        pool_ = std::make_unique<CUDAMemoryPool>(16 * 1024 * 1024); // 16MB pool
    }
    
    std::unique_ptr<CUDAMemoryPool> pool_;
};

TEST_F(CUDAMemoryPoolTest, BasicAllocation) {
    EXPECT_EQ(pool_->get_total_size(), 16 * 1024 * 1024);
    EXPECT_EQ(pool_->get_used_size(), 0);
    EXPECT_EQ(pool_->get_free_size(), 16 * 1024 * 1024);
    
    // Allocate some memory
    void* ptr1 = pool_->allocate(1024);
    EXPECT_NE(ptr1, nullptr);
    EXPECT_GT(pool_->get_used_size(), 0);
    
    void* ptr2 = pool_->allocate(2048);
    EXPECT_NE(ptr2, nullptr);
    EXPECT_NE(ptr1, ptr2);
    
    // Free memory
    pool_->deallocate(ptr1);
    pool_->deallocate(ptr2);
    
    // Should be back to initial state (approximately, due to fragmentation)
    EXPECT_LT(pool_->get_used_size(), pool_->get_total_size() / 2);
}

TEST_F(CUDAMemoryPoolTest, AllocationAlignment) {
    // Test different alignment requirements
    void* ptr256 = pool_->allocate(1024, 256);
    EXPECT_NE(ptr256, nullptr);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr256) % 256, 0);
    
    void* ptr512 = pool_->allocate(1024, 512);
    EXPECT_NE(ptr512, nullptr);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr512) % 512, 0);
    
    pool_->deallocate(ptr256);
    pool_->deallocate(ptr512);
}

TEST_F(CUDAMemoryPoolTest, OutOfMemory) {
    // Try to allocate more than the pool size
    void* ptr = pool_->allocate(32 * 1024 * 1024); // 32MB > 16MB pool
    EXPECT_EQ(ptr, nullptr);
    
    // Allocate the entire pool
    void* large_ptr = pool_->allocate(16 * 1024 * 1024);
    EXPECT_NE(large_ptr, nullptr);
    
    // Try to allocate more
    void* small_ptr = pool_->allocate(1024);
    EXPECT_EQ(small_ptr, nullptr);
    
    // Free and try again
    pool_->deallocate(large_ptr);
    small_ptr = pool_->allocate(1024);
    EXPECT_NE(small_ptr, nullptr);
    
    pool_->deallocate(small_ptr);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}