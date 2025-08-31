#include "../framework/test_framework.hpp"
#include "common/logger.hpp"
#include "common/error_handling.hpp"
#include "memory/numa_allocator.hpp"
#include "lockfree/atomic_ref_count.hpp"
#include "cache/ultra_cache.hpp"
#include <gtest/gtest.h>
#include <thread>
#include <vector>
#include <chrono>

using namespace ultra::testing;

class ComprehensiveUnitTests : public UltraTestFixture {
protected:
    void SetUp() override {
        UltraTestFixture::SetUp();
        // Additional setup for comprehensive tests
    }
};

// Test Logger functionality
TEST_F(ComprehensiveUnitTests, LoggerPerformanceTest) {
    auto logger = ultra::common::Logger::instance();
    
    // Test that logging doesn't exceed latency requirements
    expect_latency_under([&]() {
        logger->info("Test log message with parameter: {}", 42);
    }, 1000); // 1 microsecond max
    
    // Test high-frequency logging
    expect_throughput_over([&]() {
        logger->debug("High frequency log message");
    }, 100000); // 100k logs per second minimum
}

TEST_F(ComprehensiveUnitTests, LoggerThreadSafety) {
    auto logger = ultra::common::Logger::instance();
    std::atomic<int> counter{0};
    
    run_concurrent_test([&]() {
        int value = counter.fetch_add(1);
        logger->info("Thread {} logging message {}", std::this_thread::get_id(), value);
    }, 10, std::chrono::seconds(2));
    
    // Verify no crashes or data corruption occurred
    EXPECT_GT(counter.load(), 0);
}

// Test Error Handling
TEST_F(ComprehensiveUnitTests, ErrorHandlingLatency) {
    using namespace ultra::common;
    
    expect_latency_under([&]() {
        try {
            throw UltraException("Test exception", ErrorCode::PERFORMANCE_DEGRADATION);
        } catch (const UltraException& e) {
            // Exception handling should be fast
            EXPECT_EQ(e.error_code(), ErrorCode::PERFORMANCE_DEGRADATION);
        }
    }, 500); // 500ns max for exception handling
}

TEST_F(ComprehensiveUnitTests, ErrorHandlingRecovery) {
    using namespace ultra::common;
    
    ErrorHandler handler;
    bool recovery_called = false;
    
    handler.register_recovery_action(ErrorCode::MEMORY_EXHAUSTION, [&]() {
        recovery_called = true;
    });
    
    handler.handle_error(ErrorCode::MEMORY_EXHAUSTION, "Test memory exhaustion");
    EXPECT_TRUE(recovery_called);
}

// Test NUMA Allocator
TEST_F(ComprehensiveUnitTests, NUMAAllocatorPerformance) {
    using namespace ultra::memory;
    
    NUMAAllocator allocator(0); // Node 0
    
    // Test allocation performance
    expect_latency_under([&]() {
        void* ptr = allocator.allocate(1024);
        allocator.deallocate(ptr, 1024);
    }, 100); // 100ns max for small allocations
    
    // Test large allocation performance
    expect_latency_under([&]() {
        void* ptr = allocator.allocate(1024 * 1024); // 1MB
        allocator.deallocate(ptr, 1024 * 1024);
    }, 10000); // 10 microseconds max for large allocations
}

TEST_F(ComprehensiveUnitTests, NUMAAllocatorAlignment) {
    using namespace ultra::memory;
    
    NUMAAllocator allocator(0);
    
    // Test various alignment requirements
    for (size_t alignment : {8, 16, 32, 64, 128, 256}) {
        void* ptr = allocator.allocate_aligned(1024, alignment);
        EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr) % alignment, 0)
            << "Allocation not aligned to " << alignment << " bytes";
        allocator.deallocate(ptr, 1024);
    }
}

// Test Atomic Reference Counting
TEST_F(ComprehensiveUnitTests, AtomicRefCountPerformance) {
    using namespace ultra::lockfree;
    
    AtomicRefCount ref_count(1);
    
    // Test increment/decrement performance
    expect_latency_under([&]() {
        ref_count.increment();
        ref_count.decrement();
    }, 50); // 50ns max for ref count operations
    
    // Test high-frequency operations
    expect_throughput_over([&]() {
        ref_count.increment();
        ref_count.decrement();
    }, 10000000); // 10M operations per second
}

TEST_F(ComprehensiveUnitTests, AtomicRefCountConcurrency) {
    using namespace ultra::lockfree;
    
    AtomicRefCount ref_count(1);
    std::atomic<int> increment_count{0};
    std::atomic<int> decrement_count{0};
    
    run_concurrent_test([&]() {
        if (increment_count.load() < 1000) {
            ref_count.increment();
            increment_count++;
        } else if (decrement_count.load() < 1000) {
            if (ref_count.decrement()) {
                decrement_count++;
            }
        }
    }, 8, std::chrono::seconds(1));
    
    // Verify final state is consistent
    EXPECT_GE(ref_count.get_count(), 0);
}

// Test Ultra Cache
TEST_F(ComprehensiveUnitTests, UltraCacheLatency) {
    using namespace ultra::cache;
    
    UltraCache<std::string, std::string> cache(1000);
    
    // Test put operation latency
    expect_latency_under([&]() {
        cache.put("test_key", "test_value");
    }, 200); // 200ns max for cache put
    
    // Test get operation latency
    cache.put("lookup_key", "lookup_value");
    expect_latency_under([&]() {
        auto result = cache.get("lookup_key");
        EXPECT_TRUE(result.has_value());
    }, 100); // 100ns max for cache get
}

TEST_F(ComprehensiveUnitTests, UltraCacheThroughput) {
    using namespace ultra::cache;
    
    UltraCache<int, int> cache(10000);
    
    // Pre-populate cache
    for (int i = 0; i < 1000; ++i) {
        cache.put(i, i * 2);
    }
    
    // Test read throughput
    std::atomic<int> key_counter{0};
    expect_throughput_over([&]() {
        int key = key_counter.fetch_add(1) % 1000;
        auto result = cache.get(key);
        EXPECT_TRUE(result.has_value());
    }, 1000000); // 1M gets per second minimum
}

TEST_F(ComprehensiveUnitTests, UltraCacheEviction) {
    using namespace ultra::cache;
    
    UltraCache<int, std::string> cache(100); // Small cache for testing eviction
    
    // Fill cache beyond capacity
    for (int i = 0; i < 150; ++i) {
        cache.put(i, "value_" + std::to_string(i));
    }
    
    // Verify cache size is maintained
    auto stats = cache.get_stats();
    EXPECT_LE(stats.current_size, 100);
    EXPECT_GT(stats.evictions, 0);
}

// Memory leak detection tests
TEST_F(ComprehensiveUnitTests, MemoryLeakDetection) {
    // This test should pass without memory leaks
    std::vector<void*> allocations;
    
    for (int i = 0; i < 100; ++i) {
        void* ptr = malloc(1024);
        allocations.push_back(ptr);
    }
    
    // Free all allocations
    for (void* ptr : allocations) {
        free(ptr);
    }
    
    // TearDown will check for leaks automatically
}

// Stress tests
TEST_F(ComprehensiveUnitTests, StressTestAllComponents) {
    using namespace ultra::cache;
    using namespace ultra::lockfree;
    using namespace ultra::memory;
    
    // Create components
    UltraCache<int, int> cache(1000);
    AtomicRefCount ref_count(1);
    NUMAAllocator allocator(0);
    
    // Run stress test
    run_concurrent_test([&]() {
        // Cache operations
        int key = rand() % 100;
        cache.put(key, key * 2);
        auto result = cache.get(key);
        
        // Reference counting
        ref_count.increment();
        ref_count.decrement();
        
        // Memory allocation
        void* ptr = allocator.allocate(64);
        allocator.deallocate(ptr, 64);
    }, 4, std::chrono::seconds(5));
    
    // Verify system is still functional
    cache.put(999, 1998);
    auto result = cache.get(999);
    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, 1998);
}

// Edge case tests
TEST_F(ComprehensiveUnitTests, EdgeCaseHandling) {
    using namespace ultra::cache;
    
    UltraCache<std::string, std::string> cache(10);
    
    // Test empty key
    cache.put("", "empty_key_value");
    auto result = cache.get("");
    EXPECT_TRUE(result.has_value());
    
    // Test very long key
    std::string long_key(1000, 'x');
    cache.put(long_key, "long_key_value");
    result = cache.get(long_key);
    EXPECT_TRUE(result.has_value());
    
    // Test null-like values
    cache.put("null_test", "");
    result = cache.get("null_test");
    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, "");
}

// Performance regression tests
TEST_F(ComprehensiveUnitTests, PerformanceRegressionTest) {
    // This test ensures performance doesn't degrade over time
    const uint64_t MAX_CACHE_GET_LATENCY_NS = 100;
    const uint64_t MAX_CACHE_PUT_LATENCY_NS = 200;
    const size_t MIN_CACHE_THROUGHPUT_OPS = 1000000;
    
    using namespace ultra::cache;
    UltraCache<int, int> cache(1000);
    
    // Warm up cache
    for (int i = 0; i < 100; ++i) {
        cache.put(i, i);
    }
    
    // Test get latency
    LatencyMeasurement get_latency;
    for (int i = 0; i < 1000; ++i) {
        PerformanceTimer timer;
        auto result = cache.get(i % 100);
        get_latency.record_latency(timer.elapsed_ns());
    }
    
    auto get_stats = get_latency.calculate_stats();
    EXPECT_LT(get_stats.p99_ns, MAX_CACHE_GET_LATENCY_NS)
        << "Cache get P99 latency regression: " << get_stats.p99_ns << "ns";
    
    // Test put latency
    LatencyMeasurement put_latency;
    for (int i = 1000; i < 2000; ++i) {
        PerformanceTimer timer;
        cache.put(i, i);
        put_latency.record_latency(timer.elapsed_ns());
    }
    
    auto put_stats = put_latency.calculate_stats();
    EXPECT_LT(put_stats.p99_ns, MAX_CACHE_PUT_LATENCY_NS)
        << "Cache put P99 latency regression: " << put_stats.p99_ns << "ns";
}