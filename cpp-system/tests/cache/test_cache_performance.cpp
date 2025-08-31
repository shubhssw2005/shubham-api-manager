#include <gtest/gtest.h>
#include "cache/ultra_cache.hpp"
#include "common/types.hpp"
#include <chrono>
#include <thread>
#include <vector>
#include <random>
#include <string>

using namespace ultra::cache;

class CachePerformanceTest : public ::testing::Test {
protected:
    void SetUp() override {
        config_.capacity = 100000;
        config_.shard_count = 16;
        config_.enable_rdma = false;
        config_.enable_predictive_loading = false;
        config_.eviction_policy = UltraCache<std::string, std::string>::Config::EvictionPolicy::LRU;
    }
    
    UltraCache<std::string, std::string>::Config config_;
};

TEST_F(CachePerformanceTest, SingleThreadedThroughput) {
    UltraCache<std::string, std::string> cache(config_);
    
    const int num_operations = 100000;
    
    // Measure PUT throughput
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_operations; ++i) {
        std::string key = "perf_key_" + std::to_string(i);
        std::string value = "perf_value_" + std::to_string(i) + "_content_data";
        cache.put(key, value);
    }
    
    auto put_end_time = std::chrono::high_resolution_clock::now();
    
    // Measure GET throughput
    for (int i = 0; i < num_operations; ++i) {
        std::string key = "perf_key_" + std::to_string(i);
        auto result = cache.get(key);
        EXPECT_TRUE(result.has_value());
    }
    
    auto get_end_time = std::chrono::high_resolution_clock::now();
    
    // Calculate throughput
    auto put_duration_us = std::chrono::duration_cast<std::chrono::microseconds>(
        put_end_time - start_time).count();
    auto get_duration_us = std::chrono::duration_cast<std::chrono::microseconds>(
        get_end_time - put_end_time).count();
    
    double put_ops_per_sec = (num_operations * 1000000.0) / put_duration_us;
    double get_ops_per_sec = (num_operations * 1000000.0) / get_duration_us;
    
    std::cout << "Single-threaded PUT throughput: " << put_ops_per_sec << " ops/sec" << std::endl;
    std::cout << "Single-threaded GET throughput: " << get_ops_per_sec << " ops/sec" << std::endl;
    
    // Performance expectations (conservative)
    EXPECT_GT(put_ops_per_sec, 50000);  // At least 50K PUT ops/sec
    EXPECT_GT(get_ops_per_sec, 100000); // At least 100K GET ops/sec
}

TEST_F(CachePerformanceTest, MultiThreadedThroughput) {
    UltraCache<std::string, std::string> cache(config_);
    
    const int num_threads = 8;
    const int operations_per_thread = 50000;
    
    // Pre-populate cache
    for (int i = 0; i < operations_per_thread; ++i) {
        std::string key = "mt_key_" + std::to_string(i);
        std::string value = "mt_value_" + std::to_string(i);
        cache.put(key, value);
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::vector<std::thread> threads;
    std::vector<uint64_t> thread_operations(num_threads, 0);
    
    // Launch threads
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&cache, t, operations_per_thread, &thread_operations]() {
            std::random_device rd;
            std::mt19937 gen(rd() + t);
            std::uniform_int_distribution<> dis(0, operations_per_thread - 1);
            
            for (int i = 0; i < operations_per_thread; ++i) {
                int key_num = dis(gen);
                std::string key = "mt_key_" + std::to_string(key_num);
                
                if (i % 4 == 0) {
                    // 25% writes
                    std::string value = "updated_mt_value_" + std::to_string(t) + "_" + std::to_string(i);
                    cache.put(key, value);
                } else {
                    // 75% reads
                    cache.get(key);
                }
                
                thread_operations[t]++;
            }
        });
    }
    
    // Wait for completion
    for (auto& thread : threads) {
        thread.join();
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    
    // Calculate total throughput
    auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - start_time).count();
    
    uint64_t total_operations = 0;
    for (auto ops : thread_operations) {
        total_operations += ops;
    }
    
    double total_ops_per_sec = (total_operations * 1000000.0) / duration_us;
    
    std::cout << "Multi-threaded throughput (" << num_threads << " threads): " 
              << total_ops_per_sec << " ops/sec" << std::endl;
    
    // Should scale reasonably with multiple threads
    EXPECT_GT(total_ops_per_sec, 200000); // At least 200K ops/sec with 8 threads
}

TEST_F(CachePerformanceTest, LatencyMeasurement) {
    UltraCache<std::string, std::string> cache(config_);
    
    const int num_operations = 10000;
    std::vector<uint64_t> get_latencies;
    std::vector<uint64_t> put_latencies;
    
    get_latencies.reserve(num_operations);
    put_latencies.reserve(num_operations);
    
    // Pre-populate for GET latency measurement
    for (int i = 0; i < num_operations; ++i) {
        std::string key = "lat_key_" + std::to_string(i);
        std::string value = "lat_value_" + std::to_string(i);
        cache.put(key, value);
    }
    
    // Measure GET latencies
    for (int i = 0; i < num_operations; ++i) {
        std::string key = "lat_key_" + std::to_string(i);
        
        auto start = std::chrono::high_resolution_clock::now();
        auto result = cache.get(key);
        auto end = std::chrono::high_resolution_clock::now();
        
        EXPECT_TRUE(result.has_value());
        
        auto latency_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            end - start).count();
        get_latencies.push_back(latency_ns);
    }
    
    // Measure PUT latencies
    for (int i = 0; i < num_operations; ++i) {
        std::string key = "lat_put_key_" + std::to_string(i);
        std::string value = "lat_put_value_" + std::to_string(i);
        
        auto start = std::chrono::high_resolution_clock::now();
        cache.put(key, value);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto latency_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            end - start).count();
        put_latencies.push_back(latency_ns);
    }
    
    // Calculate statistics
    std::sort(get_latencies.begin(), get_latencies.end());
    std::sort(put_latencies.begin(), put_latencies.end());
    
    auto get_p50 = get_latencies[num_operations / 2];
    auto get_p95 = get_latencies[static_cast<size_t>(num_operations * 0.95)];
    auto get_p99 = get_latencies[static_cast<size_t>(num_operations * 0.99)];
    
    auto put_p50 = put_latencies[num_operations / 2];
    auto put_p95 = put_latencies[static_cast<size_t>(num_operations * 0.95)];
    auto put_p99 = put_latencies[static_cast<size_t>(num_operations * 0.99)];
    
    std::cout << "GET Latency Statistics:" << std::endl;
    std::cout << "  P50: " << get_p50 << " ns (" << (get_p50 / 1000.0) << " μs)" << std::endl;
    std::cout << "  P95: " << get_p95 << " ns (" << (get_p95 / 1000.0) << " μs)" << std::endl;
    std::cout << "  P99: " << get_p99 << " ns (" << (get_p99 / 1000.0) << " μs)" << std::endl;
    
    std::cout << "PUT Latency Statistics:" << std::endl;
    std::cout << "  P50: " << put_p50 << " ns (" << (put_p50 / 1000.0) << " μs)" << std::endl;
    std::cout << "  P95: " << put_p95 << " ns (" << (put_p95 / 1000.0) << " μs)" << std::endl;
    std::cout << "  P99: " << put_p99 << " ns (" << (put_p99 / 1000.0) << " μs)" << std::endl;
    
    // Performance expectations for ultra-low latency
    EXPECT_LT(get_p99, 50000);  // P99 GET latency under 50μs
    EXPECT_LT(put_p99, 100000); // P99 PUT latency under 100μs
}

TEST_F(CachePerformanceTest, CacheEvictionPerformance) {
    // Set small capacity to trigger frequent evictions
    config_.capacity = 1000;
    UltraCache<std::string, std::string> cache(config_);
    
    const int num_operations = 10000; // Much larger than capacity
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_operations; ++i) {
        std::string key = "evict_key_" + std::to_string(i);
        std::string value = "evict_value_" + std::to_string(i);
        cache.put(key, value);
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    
    auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - start_time).count();
    double ops_per_sec = (num_operations * 1000000.0) / duration_us;
    
    auto stats = cache.get_stats();
    
    std::cout << "Eviction performance test:" << std::endl;
    std::cout << "  Operations: " << num_operations << std::endl;
    std::cout << "  Evictions: " << stats.evictions.load() << std::endl;
    std::cout << "  Throughput: " << ops_per_sec << " ops/sec" << std::endl;
    
    // Should handle evictions efficiently
    EXPECT_GT(stats.evictions.load(), 0);
    EXPECT_GT(ops_per_sec, 10000); // At least 10K ops/sec even with evictions
}

TEST_F(CachePerformanceTest, MemoryUsageEfficiency) {
    UltraCache<std::string, std::string> cache(config_);
    
    const int num_entries = 10000;
    const size_t value_size = 100; // bytes
    
    // Calculate expected memory usage
    size_t expected_data_size = num_entries * (20 + value_size); // Rough estimate
    
    // Populate cache
    for (int i = 0; i < num_entries; ++i) {
        std::string key = "mem_key_" + std::to_string(i);
        std::string value(value_size, 'x');
        cache.put(key, value);
    }
    
    auto stats = cache.get_stats();
    size_t actual_memory = stats.memory_usage_bytes.load();
    
    std::cout << "Memory usage test:" << std::endl;
    std::cout << "  Entries: " << num_entries << std::endl;
    std::cout << "  Expected data size: ~" << expected_data_size << " bytes" << std::endl;
    std::cout << "  Actual memory usage: " << actual_memory << " bytes" << std::endl;
    
    if (actual_memory > 0) {
        double overhead_ratio = static_cast<double>(actual_memory) / expected_data_size;
        std::cout << "  Memory overhead ratio: " << overhead_ratio << "x" << std::endl;
        
        // Memory overhead should be reasonable (less than 3x the data size)
        EXPECT_LT(overhead_ratio, 3.0);
    }
}

TEST_F(CachePerformanceTest, ScalabilityWithShards) {
    std::vector<size_t> shard_counts = {1, 4, 16, 64};
    const int operations_per_test = 50000;
    
    for (size_t shard_count : shard_counts) {
        config_.shard_count = shard_count;
        UltraCache<std::string, std::string> cache(config_);
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Mixed workload
        for (int i = 0; i < operations_per_test; ++i) {
            std::string key = "shard_key_" + std::to_string(i % 1000);
            
            if (i % 4 == 0) {
                std::string value = "shard_value_" + std::to_string(i);
                cache.put(key, value);
            } else {
                cache.get(key);
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        
        auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - start_time).count();
        double ops_per_sec = (operations_per_test * 1000000.0) / duration_us;
        
        std::cout << "Shards: " << shard_count << ", Throughput: " << ops_per_sec << " ops/sec" << std::endl;
        
        // More shards should generally improve performance (up to a point)
        EXPECT_GT(ops_per_sec, 10000);
    }
}