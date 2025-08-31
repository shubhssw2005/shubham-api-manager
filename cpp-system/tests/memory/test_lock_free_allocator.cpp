#include <gtest/gtest.h>
#include "memory/lock_free_allocator.hpp"
#include <thread>
#include <vector>
#include <atomic>
#include <chrono>

using namespace ultra::memory;

class LockFreeAllocatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        config_.pool_size = 1024 * 1024; // 1MB
        config_.min_block_size = 64;
        config_.max_block_size = 4096;
        config_.numa_aware = false; // Disable for testing
    }
    
    LockFreeAllocator::Config config_;
};

TEST_F(LockFreeAllocatorTest, BasicAllocation) {
    LockFreeAllocator allocator(config_);
    
    void* ptr = allocator.allocate(128);
    ASSERT_NE(ptr, nullptr);
    
    // Write to memory to ensure it's valid
    memset(ptr, 0xAA, 128);
    
    allocator.deallocate(ptr, 128);
}

TEST_F(LockFreeAllocatorTest, MultipleAllocations) {
    LockFreeAllocator allocator(config_);
    
    std::vector<void*> ptrs;
    const size_t num_allocs = 100;
    
    // Allocate multiple blocks
    for (size_t i = 0; i < num_allocs; ++i) {
        void* ptr = allocator.allocate(64 + (i % 256));
        ASSERT_NE(ptr, nullptr);
        ptrs.push_back(ptr);
    }
    
    // Deallocate all blocks
    for (size_t i = 0; i < num_allocs; ++i) {
        allocator.deallocate(ptrs[i], 64 + (i % 256));
    }
}

TEST_F(LockFreeAllocatorTest, ZeroSizeAllocation) {
    LockFreeAllocator allocator(config_);
    
    void* ptr = allocator.allocate(0);
    EXPECT_EQ(ptr, nullptr);
}

TEST_F(LockFreeAllocatorTest, LargeAllocation) {
    LockFreeAllocator allocator(config_);
    
    // Allocate larger than max_block_size
    size_t large_size = config_.max_block_size * 2;
    void* ptr = allocator.allocate(large_size);
    ASSERT_NE(ptr, nullptr);
    
    // Write to memory
    memset(ptr, 0xBB, large_size);
    
    allocator.deallocate(ptr, large_size);
}

TEST_F(LockFreeAllocatorTest, ThreadSafety) {
    LockFreeAllocator allocator(config_);
    
    const int num_threads = 4;
    const int allocs_per_thread = 1000;
    std::atomic<int> successful_allocs{0};
    std::atomic<int> successful_deallocs{0};
    
    std::vector<std::thread> threads;
    
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            std::vector<std::pair<void*, size_t>> local_ptrs;
            
            // Allocate
            for (int i = 0; i < allocs_per_thread; ++i) {
                size_t size = 64 + (i % 512);
                void* ptr = allocator.allocate(size);
                if (ptr) {
                    local_ptrs.emplace_back(ptr, size);
                    successful_allocs.fetch_add(1);
                    
                    // Write pattern to verify memory integrity
                    memset(ptr, t + 1, size);
                }
            }
            
            // Verify and deallocate
            for (auto& [ptr, size] : local_ptrs) {
                // Verify pattern
                uint8_t* bytes = static_cast<uint8_t*>(ptr);
                for (size_t i = 0; i < size; ++i) {
                    EXPECT_EQ(bytes[i], t + 1);
                }
                
                allocator.deallocate(ptr, size);
                successful_deallocs.fetch_add(1);
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    EXPECT_EQ(successful_allocs.load(), successful_deallocs.load());
    EXPECT_GT(successful_allocs.load(), 0);
}

TEST_F(LockFreeAllocatorTest, Statistics) {
    LockFreeAllocator allocator(config_);
    
    const auto& stats_before = allocator.get_stats();
    uint64_t allocs_before = stats_before.allocations.load();
    uint64_t bytes_before = stats_before.bytes_allocated.load();
    
    // Perform some allocations
    std::vector<void*> ptrs;
    size_t total_size = 0;
    
    for (int i = 0; i < 10; ++i) {
        size_t size = 128;
        void* ptr = allocator.allocate(size);
        if (ptr) {
            ptrs.push_back(ptr);
            total_size += size;
        }
    }
    
    const auto& stats_after = allocator.get_stats();
    EXPECT_GE(stats_after.allocations.load(), allocs_before + ptrs.size());
    EXPECT_GE(stats_after.bytes_allocated.load(), bytes_before + total_size);
    
    // Cleanup
    for (size_t i = 0; i < ptrs.size(); ++i) {
        allocator.deallocate(ptrs[i], 128);
    }
}

TEST_F(LockFreeAllocatorTest, AlignmentRequirements) {
    LockFreeAllocator allocator(config_);
    
    // Test various sizes to ensure proper alignment
    std::vector<size_t> test_sizes = {64, 128, 256, 512, 1024};
    
    for (size_t size : test_sizes) {
        void* ptr = allocator.allocate(size);
        ASSERT_NE(ptr, nullptr);
        
        // Check alignment (should be at least cache line aligned)
        uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
        EXPECT_EQ(addr % 64, 0) << "Allocation not cache-line aligned for size " << size;
        
        allocator.deallocate(ptr, size);
    }
}

TEST_F(LockFreeAllocatorTest, PerformanceBenchmark) {
    LockFreeAllocator allocator(config_);
    
    const int num_iterations = 10000;
    const size_t alloc_size = 128;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_iterations; ++i) {
        void* ptr = allocator.allocate(alloc_size);
        if (ptr) {
            allocator.deallocate(ptr, alloc_size);
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    
    double ns_per_op = static_cast<double>(duration.count()) / (num_iterations * 2); // alloc + dealloc
    
    std::cout << "Lock-free allocator performance: " << ns_per_op << " ns per operation\n";
    
    // Should be faster than 1000ns per operation for small allocations
    EXPECT_LT(ns_per_op, 1000.0);
}

TEST_F(LockFreeAllocatorTest, UniquePointerIntegration) {
    LockFreeAllocator allocator(config_);
    
    // Test unique_ptr integration
    {
        auto ptr = make_unique<int>(allocator, 42);
        ASSERT_NE(ptr.get(), nullptr);
        EXPECT_EQ(*ptr, 42);
    } // Should automatically deallocate
    
    // Test with custom class
    struct TestClass {
        int value;
        TestClass(int v) : value(v) {}
    };
    
    {
        auto ptr = make_unique<TestClass>(allocator, 123);
        ASSERT_NE(ptr.get(), nullptr);
        EXPECT_EQ(ptr->value, 123);
    } // Should automatically deallocate
}

TEST_F(LockFreeAllocatorTest, StressTest) {
    LockFreeAllocator allocator(config_);
    
    const int num_threads = 8;
    const int operations_per_thread = 5000;
    std::atomic<bool> start_flag{false};
    std::atomic<int> completed_threads{0};
    
    std::vector<std::thread> threads;
    
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&]() {
            // Wait for start signal
            while (!start_flag.load()) {
                std::this_thread::yield();
            }
            
            std::vector<std::pair<void*, size_t>> allocations;
            
            for (int i = 0; i < operations_per_thread; ++i) {
                if (i % 3 == 0 && !allocations.empty()) {
                    // Deallocate random allocation
                    size_t idx = i % allocations.size();
                    auto [ptr, size] = allocations[idx];
                    allocator.deallocate(ptr, size);
                    allocations.erase(allocations.begin() + idx);
                } else {
                    // Allocate new block
                    size_t size = 64 + (i % 1024);
                    void* ptr = allocator.allocate(size);
                    if (ptr) {
                        allocations.emplace_back(ptr, size);
                    }
                }
            }
            
            // Cleanup remaining allocations
            for (auto [ptr, size] : allocations) {
                allocator.deallocate(ptr, size);
            }
            
            completed_threads.fetch_add(1);
        });
    }
    
    // Start all threads simultaneously
    start_flag.store(true);
    
    // Wait for completion
    for (auto& thread : threads) {
        thread.join();
    }
    
    EXPECT_EQ(completed_threads.load(), num_threads);
}