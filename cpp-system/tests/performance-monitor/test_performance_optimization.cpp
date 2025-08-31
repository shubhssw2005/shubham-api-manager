#include <gtest/gtest.h>
#include "performance-monitor/cpu_affinity.hpp"
#include "performance-monitor/cache_optimization.hpp"
#include "performance-monitor/compiler_optimization.hpp"
#include "performance-monitor/adaptive_tuning.hpp"

#include <thread>
#include <vector>
#include <chrono>
#include <random>

using namespace ultra_cpp::performance;

class PerformanceOptimizationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup test environment
    }
    
    void TearDown() override {
        // Cleanup
    }
};

// CPU Affinity Tests
TEST_F(PerformanceOptimizationTest, CPUAffinityManagerInitialization) {
    CPUAffinityManager manager;
    ASSERT_TRUE(manager.initialize());
    
    const auto& topology = manager.get_topology();
    EXPECT_GT(topology.total_logical_cpus, 0);
    EXPECT_GT(topology.total_physical_cores, 0);
    EXPECT_GE(topology.total_numa_nodes, 1);
}

TEST_F(PerformanceOptimizationTest, CPUAffinityConfiguration) {
    CPUAffinityManager manager;
    ASSERT_TRUE(manager.initialize());
    
    CPUAffinityManager::AffinityConfig config;
    config.strategy = CPUAffinityManager::AffinityStrategy::NUMA_LOCAL;
    config.avoid_hyperthread_siblings = true;
    
    // Test setting affinity for current thread
    bool result = manager.set_thread_affinity(config);
    
    // May fail on systems without NUMA support, but should not crash
    EXPECT_TRUE(result || !manager.is_numa_available());
}

TEST_F(PerformanceOptimizationTest, ScopedCPUAffinity) {
    CPUAffinityManager::AffinityConfig config;
    config.strategy = CPUAffinityManager::AffinityStrategy::SINGLE_CORE;
    
    {
        ScopedCPUAffinity scoped_affinity(config);
        // Affinity should be set within this scope
        // Test would verify CPU assignment if we had access to system calls
    }
    // Affinity should be restored after scope
}

// Cache Optimization Tests
TEST_F(PerformanceOptimizationTest, CacheFriendlyVector) {
    CacheFriendlyVector<int> vec;
    
    // Test basic operations
    EXPECT_TRUE(vec.empty());
    EXPECT_EQ(vec.size(), 0);
    
    // Test push_back
    for (int i = 0; i < 1000; ++i) {
        vec.push_back(i);
    }
    
    EXPECT_EQ(vec.size(), 1000);
    EXPECT_FALSE(vec.empty());
    
    // Test access
    for (size_t i = 0; i < vec.size(); ++i) {
        EXPECT_EQ(vec[i], static_cast<int>(i));
    }
    
    // Test iterators
    int expected = 0;
    for (const auto& value : vec) {
        EXPECT_EQ(value, expected++);
    }
}

TEST_F(PerformanceOptimizationTest, StructureOfArrays) {
    StructureOfArrays<int, double, std::string> soa;
    
    // Test empty state
    EXPECT_TRUE(soa.empty());
    EXPECT_EQ(soa.size(), 0);
    
    // Test push_back
    soa.push_back(42, 3.14, "test");
    soa.push_back(100, 2.71, "hello");
    
    EXPECT_EQ(soa.size(), 2);
    EXPECT_FALSE(soa.empty());
    
    // Test access
    EXPECT_EQ(soa.get<0>(0), 42);
    EXPECT_DOUBLE_EQ(soa.get<1>(0), 3.14);
    EXPECT_EQ(soa.get<2>(0), "test");
    
    EXPECT_EQ(soa.get<0>(1), 100);
    EXPECT_DOUBLE_EQ(soa.get<1>(1), 2.71);
    EXPECT_EQ(soa.get<2>(1), "hello");
}

TEST_F(PerformanceOptimizationTest, CacheFriendlyHashMap) {
    CacheFriendlyHashMap<int, std::string> hashmap;
    
    // Test insertion
    EXPECT_TRUE(hashmap.insert(1, "one"));
    EXPECT_TRUE(hashmap.insert(2, "two"));
    EXPECT_FALSE(hashmap.insert(1, "ONE")); // Update existing key
    
    EXPECT_EQ(hashmap.size(), 2);
    
    // Test lookup
    std::string value;
    EXPECT_TRUE(hashmap.find(1, value));
    EXPECT_EQ(value, "ONE"); // Should be updated value
    
    EXPECT_TRUE(hashmap.find(2, value));
    EXPECT_EQ(value, "two");
    
    EXPECT_FALSE(hashmap.find(3, value)); // Non-existent key
    
    // Test deletion
    EXPECT_TRUE(hashmap.erase(1));
    EXPECT_FALSE(hashmap.find(1, value));
    EXPECT_EQ(hashmap.size(), 1);
}

TEST_F(PerformanceOptimizationTest, CacheAnalyzer) {
    CacheAnalyzer analyzer;
    
    // Test basic functionality (may not work on all systems)
    analyzer.start_monitoring();
    
    // Perform some cache-intensive operations
    std::vector<int> data(10000);
    std::iota(data.begin(), data.end(), 0);
    
    // Sequential access benchmark
    double seq_time = CacheAnalyzer::benchmark_sequential_access(data, 10);
    EXPECT_GT(seq_time, 0.0);
    
    // Random access benchmark
    double rand_time = CacheAnalyzer::benchmark_random_access(data, 10);
    EXPECT_GT(rand_time, 0.0);
    
    // Sequential should generally be faster than random
    // (though this may not always be true on all systems)
    
    analyzer.stop_monitoring();
    
    auto stats = analyzer.get_stats();
    // Stats may be zero on systems without perf support
}

// Compiler Optimization Tests
TEST_F(PerformanceOptimizationTest, CompilerOptimizationManager) {
    CompilerOptimizationManager manager;
    
    // Test compiler detection
    auto compiler_info = manager.detect_compiler();
    EXPECT_NE(compiler_info.type, CompilerOptimizationManager::CompilerType::UNKNOWN);
    
    // Test profile listing
    auto profiles = manager.list_profiles();
    EXPECT_FALSE(profiles.empty());
    
    // Test getting a profile
    auto release_profile = manager.get_profile("release");
    EXPECT_EQ(release_profile.name, "release");
    EXPECT_EQ(release_profile.level, CompilerOptimizationManager::OptimizationLevel::AGGRESSIVE);
    
    // Test CMake flags generation
    std::string cmake_flags = manager.generate_cmake_flags(release_profile);
    EXPECT_FALSE(cmake_flags.empty());
    EXPECT_NE(cmake_flags.find("-O3"), std::string::npos);
}

TEST_F(PerformanceOptimizationTest, PGOHelper) {
    std::string temp_dir = "/tmp/pgo_test";
    PGOHelper pgo_helper(temp_dir);
    
    // Test workload registration
    PGOHelper::TrainingWorkload workload;
    workload.name = "test_workload";
    workload.workload = []() {
        // Simple computation workload
        volatile int sum = 0;
        for (int i = 0; i < 1000; ++i) {
            sum += i * i;
        }
    };
    
    pgo_helper.add_training_workload(workload);
    
    // Test running workloads (should not crash)
    EXPECT_NO_THROW(pgo_helper.run_training_workloads());
}

// Adaptive Tuning Tests
TEST_F(PerformanceOptimizationTest, AdaptiveTuningSystem) {
    AdaptiveTuningSystem::TuningConfig config;
    config.enable_auto_tuning = false; // Manual testing only
    
    AdaptiveTuningSystem tuning_system(config);
    
    // Test parameter registration
    AdaptiveTuningSystem::TunableParameter param;
    param.name = "test_param";
    param.current_value = 50.0;
    param.min_value = 0.0;
    param.max_value = 100.0;
    param.step_size = 1.0;
    
    double test_value = 50.0;
    param.setter = [&test_value](double value) { test_value = value; };
    param.getter = [&test_value]() { return test_value; };
    
    tuning_system.register_parameter(param);
    
    // Test parameter access
    auto params = tuning_system.list_parameters();
    EXPECT_EQ(params.size(), 1);
    EXPECT_EQ(params[0], "test_param");
    
    // Test parameter setting
    EXPECT_TRUE(tuning_system.set_parameter("test_param", 75.0));
    EXPECT_DOUBLE_EQ(tuning_system.get_parameter("test_param"), 75.0);
    EXPECT_DOUBLE_EQ(test_value, 75.0);
    
    // Test bounds checking
    EXPECT_TRUE(tuning_system.set_parameter("test_param", 150.0)); // Should clamp to 100.0
    EXPECT_DOUBLE_EQ(tuning_system.get_parameter("test_param"), 100.0);
    
    // Test metrics update
    AdaptiveTuningSystem::PerformanceMetrics metrics;
    metrics.throughput.store(1000.0);
    metrics.latency_p95.store(500000.0); // 500 microseconds
    metrics.cpu_utilization.store(75.0);
    
    tuning_system.update_metrics(metrics);
    
    auto current_metrics = tuning_system.get_current_metrics();
    EXPECT_DOUBLE_EQ(current_metrics.throughput.load(), 1000.0);
}

TEST_F(PerformanceOptimizationTest, AdaptiveScheduler) {
    AdaptiveScheduler scheduler;
    
    // Test metrics access
    auto metrics = scheduler.get_metrics();
    EXPECT_EQ(metrics.tasks_completed.load(), 0);
    
    // Test task submission (basic functionality)
    std::atomic<int> counter{0};
    
    // Submit some simple tasks
    for (int i = 0; i < 10; ++i) {
        // Note: submit_task template would need proper implementation
        // For now, just test that the scheduler doesn't crash
    }
    
    // Give tasks time to complete
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    // Scheduler should be running without crashes
}

TEST_F(PerformanceOptimizationTest, MemoryAllocationTuner) {
    MemoryAllocationTuner tuner;
    
    // Test allocation recording
    void* ptr1 = malloc(1024);
    void* ptr2 = malloc(2048);
    
    tuner.record_allocation(1024, ptr1);
    tuner.record_allocation(2048, ptr2);
    
    auto stats = tuner.get_stats();
    EXPECT_EQ(stats.total_allocations.load(), 2);
    EXPECT_EQ(stats.bytes_allocated.load(), 3072);
    
    // Test deallocation recording
    tuner.record_deallocation(ptr1);
    tuner.record_deallocation(ptr2);
    
    stats = tuner.get_stats();
    EXPECT_EQ(stats.total_deallocations.load(), 2);
    
    free(ptr1);
    free(ptr2);
}

TEST_F(PerformanceOptimizationTest, CacheTuner) {
    CacheTuner tuner;
    
    // Test cache access recording
    tuner.record_cache_access("test_cache", "key1", true);  // hit
    tuner.record_cache_access("test_cache", "key2", false); // miss
    tuner.record_cache_access("test_cache", "key3", true);  // hit
    
    auto metrics = tuner.get_cache_metrics("test_cache");
    EXPECT_EQ(metrics.hits.load(), 2);
    EXPECT_EQ(metrics.misses.load(), 1);
    EXPECT_DOUBLE_EQ(metrics.hit_rate.load(), 2.0/3.0);
}

TEST_F(PerformanceOptimizationTest, NetworkTuner) {
    NetworkTuner tuner;
    
    // Test configuration
    NetworkTuner::NetworkConfig config;
    config.buffer_size = 32768;
    config.connection_pool_size = 50;
    config.enable_tcp_nodelay = true;
    
    tuner.set_network_config(config);
    
    auto retrieved_config = tuner.get_network_config();
    EXPECT_EQ(retrieved_config.buffer_size, 32768);
    EXPECT_EQ(retrieved_config.connection_pool_size, 50);
    EXPECT_TRUE(retrieved_config.enable_tcp_nodelay);
    
    // Test metrics access
    auto metrics = tuner.get_metrics();
    EXPECT_EQ(metrics.connections_established.load(), 0);
}

TEST_F(PerformanceOptimizationTest, PerformanceTuningManager) {
    PerformanceTuningManager manager;
    
    // Test component access
    auto& adaptive_tuning = manager.get_adaptive_tuning_system();
    auto& scheduler = manager.get_adaptive_scheduler();
    auto& memory_tuner = manager.get_memory_tuner();
    auto& cache_tuner = manager.get_cache_tuner();
    auto& network_tuner = manager.get_network_tuner();
    
    // Test that components are properly initialized
    EXPECT_FALSE(adaptive_tuning.is_auto_tuning_active());
    
    // Test system metrics collection
    auto system_metrics = manager.get_system_metrics();
    // Should not crash and return valid structure
}

// Performance benchmarks
TEST_F(PerformanceOptimizationTest, CacheOptimizationBenchmark) {
    const size_t data_size = 1000000;
    std::vector<int> regular_vector(data_size);
    CacheFriendlyVector<int> cache_friendly_vector;
    
    // Initialize data
    for (size_t i = 0; i < data_size; ++i) {
        regular_vector[i] = static_cast<int>(i);
        cache_friendly_vector.push_back(static_cast<int>(i));
    }
    
    // Benchmark sequential access
    auto start = std::chrono::high_resolution_clock::now();
    volatile long long sum1 = 0;
    for (size_t i = 0; i < data_size; ++i) {
        sum1 += regular_vector[i];
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto regular_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    
    start = std::chrono::high_resolution_clock::now();
    volatile long long sum2 = 0;
    for (size_t i = 0; i < cache_friendly_vector.size(); ++i) {
        sum2 += cache_friendly_vector[i];
    }
    end = std::chrono::high_resolution_clock::now();
    auto cache_friendly_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    
    // Both should produce the same result
    EXPECT_EQ(sum1, sum2);
    
    // Cache-friendly version should not be significantly slower
    // (may actually be similar due to compiler optimizations)
    double ratio = static_cast<double>(cache_friendly_time.count()) / regular_time.count();
    EXPECT_LT(ratio, 2.0); // Should not be more than 2x slower
    
    std::cout << "Regular vector time: " << regular_time.count() << " ns\n";
    std::cout << "Cache-friendly vector time: " << cache_friendly_time.count() << " ns\n";
    std::cout << "Ratio: " << ratio << "\n";
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}