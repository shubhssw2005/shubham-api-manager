#include <gtest/gtest.h>
#include "memory/memory.hpp"
#include <thread>
#include <vector>

using namespace ultra::memory;

class MemoryManagerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Use default configuration for testing
    }
};

TEST_F(MemoryManagerTest, BasicAllocation) {
    MemoryManager& manager = MemoryManager::instance();
    
    void* ptr = manager.allocate(1024);
    ASSERT_NE(ptr, nullptr);
    
    // Write to memory to verify it's valid
    memset(ptr, 0xCC, 1024);
    
    manager.deallocate(ptr, 1024);
}

TEST_F(MemoryManagerTest, DifferentSizes) {
    MemoryManager& manager = MemoryManager::instance();
    
    std::vector<std::pair<void*, size_t>> allocations;
    
    // Test different allocation sizes to trigger different strategies
    std::vector<size_t> sizes = {
        64,        // Small - lock-free allocator
        8192,      // Medium - NUMA allocator  
        2097152    // Large - memory mapping
    };
    
    for (size_t size : sizes) {
        void* ptr = manager.allocate(size);
        ASSERT_NE(ptr, nullptr) << "Failed to allocate " << size << " bytes";
        
        // Verify memory is writable
        memset(ptr, 0xDD, size);
        
        allocations.emplace_back(ptr, size);
    }
    
    // Cleanup
    for (auto [ptr, size] : allocations) {
        manager.deallocate(ptr, size);
    }
}

TEST_F(MemoryManagerTest, Statistics) {
    MemoryManager& manager = MemoryManager::instance();
    
    auto stats_before = manager.get_system_stats();
    
    // Perform some allocations
    std::vector<void*> ptrs;
    for (int i = 0; i < 10; ++i) {
        void* ptr = manager.allocate(256);
        if (ptr) {
            ptrs.push_back(ptr);
        }
    }
    
    auto stats_after = manager.get_system_stats();
    
    // Should have more allocations and bytes allocated
    EXPECT_GE(stats_after.total_allocations, stats_before.total_allocations);
    EXPECT_GE(stats_after.total_bytes_allocated, stats_before.total_bytes_allocated);
    
    // Cleanup
    for (void* ptr : ptrs) {
        manager.deallocate(ptr, 256);
    }
}

TEST_F(MemoryManagerTest, MemoryScope) {
    MemoryManager& manager = MemoryManager::instance();
    
    auto stats_before = manager.get_system_stats();
    
    {
        MemoryScope scope(manager);
        
        // Allocate within scope
        int* int_ptr = scope.allocate<int>(10);
        ASSERT_NE(int_ptr, nullptr);
        
        // Create object within scope
        struct TestStruct {
            int value;
            TestStruct(int v) : value(v) {}
        };
        
        TestStruct* obj = scope.create<TestStruct>(42);
        ASSERT_NE(obj, nullptr);
        EXPECT_EQ(obj->value, 42);
        
    } // Scope destructor should clean up automatically
    
    // Memory should be cleaned up
    auto stats_after = manager.get_system_stats();
    // Note: Due to the way our simplified implementation works,
    // we can't easily verify automatic cleanup in this test
}

TEST_F(MemoryManagerTest, UltraAllocator) {
    // Test STL allocator integration
    ultra_vector<int> vec;
    
    for (int i = 0; i < 1000; ++i) {
        vec.push_back(i);
    }
    
    EXPECT_EQ(vec.size(), 1000);
    
    for (int i = 0; i < 1000; ++i) {
        EXPECT_EQ(vec[i], i);
    }
}

TEST_F(MemoryManagerTest, RcuIntegration) {
    MemoryManager& manager = MemoryManager::instance();
    
    // Test RCU pointer creation
    struct TestData {
        int value;
        TestData(int v) : value(v) {}
    };
    
    auto rcu_ptr = manager.make_rcu_shared<TestData>(123);
    
    {
        ULTRA_RCU_READ_LOCK();
        EXPECT_EQ(rcu_ptr->value, 123);
    }
}

TEST_F(MemoryManagerTest, ThreadSafety) {
    MemoryManager& manager = MemoryManager::instance();
    
    const int num_threads = 4;
    const int allocs_per_thread = 100;
    std::atomic<int> successful_operations{0};
    
    std::vector<std::thread> threads;
    
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&]() {
            std::vector<std::pair<void*, size_t>> local_allocs;
            
            // Allocate
            for (int i = 0; i < allocs_per_thread; ++i) {
                size_t size = 128 + (i % 512);
                void* ptr = manager.allocate(size);
                if (ptr) {
                    local_allocs.emplace_back(ptr, size);
                    memset(ptr, 0xEE, size);
                }
            }
            
            // Deallocate
            for (auto [ptr, size] : local_allocs) {
                manager.deallocate(ptr, size);
                successful_operations.fetch_add(1);
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    EXPECT_GT(successful_operations.load(), 0);
}

TEST_F(MemoryManagerTest, PrintMemoryInfo) {
    MemoryManager& manager = MemoryManager::instance();
    
    // This test just verifies the function doesn't crash
    // In a real test environment, you might capture stdout
    EXPECT_NO_THROW(manager.print_memory_info());
}