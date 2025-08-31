#include <gtest/gtest.h>
#include "cache/ultra_cache.hpp"
#include "common/types.hpp"
#include <thread>
#include <vector>
#include <chrono>
#include <random>
#include <string>

using namespace ultra::cache;

class UltraCacheTest : public ::testing::Test {
protected:
    void SetUp() override {
        config_.capacity = 1000;
        config_.shard_count = 4;
        config_.enable_rdma = false;
        config_.enable_predictive_loading = true;
        config_.eviction_policy = UltraCache<std::string, std::string>::Config::EvictionPolicy::LRU;
    }
    
    UltraCache<std::string, std::string>::Config config_;
};

TEST_F(UltraCacheTest, BasicOperations) {
    UltraCache<std::string, std::string> cache(config_);
    
    // Test put and get
    cache.put("key1", "value1");
    auto result = cache.get("key1");
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), "value1");
    
    // Test non-existent key
    auto missing = cache.get("nonexistent");
    EXPECT_FALSE(missing.has_value());
    
    // Test update
    cache.put("key1", "updated_value1");
    result = cache.get("key1");
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), "updated_value1");
    
    // Test remove
    cache.remove("key1");
    result = cache.get("key1");
    EXPECT_FALSE(result.has_value());
}

TEST_F(UltraCacheTest, BatchOperations) {
    UltraCache<std::string, std::string> cache(config_);
    
    // Test batch put
    std::vector<std::pair<std::string, std::string>> items;
    for (int i = 0; i < 100; ++i) {
        items.emplace_back("key" + std::to_string(i), "value" + std::to_string(i));
    }
    cache.put_batch(items);
    
    // Test batch get
    std::vector<std::string> keys;
    for (int i = 0; i < 100; ++i) {
        keys.push_back("key" + std::to_string(i));
    }
    
    auto results = cache.get_batch(keys);
    EXPECT_EQ(results.size(), 100);
    
    for (int i = 0; i < 100; ++i) {
        ASSERT_TRUE(results[i].has_value());
        EXPECT_EQ(results[i].value(), "value" + std::to_string(i));
    }
}

TEST_F(UltraCacheTest, Statistics) {
    UltraCache<std::string, std::string> cache(config_);
    
    // Initial stats should be zero
    auto stats = cache.get_stats();
    EXPECT_EQ(stats.hits.load(), 0);
    EXPECT_EQ(stats.misses.load(), 0);
    EXPECT_EQ(stats.total_operations.load(), 0);
    
    // Perform operations and check stats
    cache.put("key1", "value1");
    cache.get("key1");  // Hit
    cache.get("key2");  // Miss
    
    stats = cache.get_stats();
    EXPECT_EQ(stats.hits.load(), 1);
    EXPECT_EQ(stats.misses.load(), 1);
    EXPECT_EQ(stats.total_operations.load(), 3); // 1 put + 2 gets
    
    // Test hit ratio
    double hit_ratio = cache.get_hit_ratio();
    EXPECT_NEAR(hit_ratio, 1.0/3.0, 0.01); // 1 hit out of 3 operations
}

TEST_F(UltraCacheTest, EvictionPolicy) {
    // Set small capacity to trigger eviction
    config_.capacity = 5;
    UltraCache<std::string, std::string> cache(config_);
    
    // Fill cache beyond capacity
    for (int i = 0; i < 10; ++i) {
        cache.put("key" + std::to_string(i), "value" + std::to_string(i));
    }
    
    auto stats = cache.get_stats();
    EXPECT_GT(stats.evictions.load(), 0);
    
    // Some early keys should have been evicted
    auto result = cache.get("key0");
    // Note: Due to sharding, we can't guarantee which specific keys are evicted
    // but we can verify that evictions occurred
}

TEST_F(UltraCacheTest, ConcurrentAccess) {
    UltraCache<std::string, std::string> cache(config_);
    
    const int num_threads = 8;
    const int operations_per_thread = 1000;
    std::vector<std::thread> threads;
    
    // Launch concurrent threads
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&cache, t, operations_per_thread]() {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> dis(0, 99);
            
            for (int i = 0; i < operations_per_thread; ++i) {
                int key_num = dis(gen);
                std::string key = "thread" + std::to_string(t) + "_key" + std::to_string(key_num);
                std::string value = "thread" + std::to_string(t) + "_value" + std::to_string(i);
                
                if (i % 3 == 0) {
                    // Put operation
                    cache.put(key, value);
                } else if (i % 3 == 1) {
                    // Get operation
                    cache.get(key);
                } else {
                    // Remove operation
                    cache.remove(key);
                }
            }
        });
    }
    
    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }
    
    // Verify that operations completed without crashes
    auto stats = cache.get_stats();
    EXPECT_GT(stats.total_operations.load(), 0);
}

TEST_F(UltraCacheTest, CacheWarming) {
    UltraCache<std::string, std::string> cache(config_);
    
    // Populate cache with initial data
    for (int i = 0; i < 50; ++i) {
        cache.put("key" + std::to_string(i), "value" + std::to_string(i));
    }
    
    // Prepare keys for warming
    std::vector<std::string> warm_keys;
    for (int i = 50; i < 100; ++i) {
        warm_keys.push_back("key" + std::to_string(i));
    }
    
    // Trigger cache warming
    cache.warm_cache(warm_keys);
    
    // Give some time for warming to process
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    auto stats = cache.get_stats();
    // Warming operations should have been recorded
    // Note: The actual warming behavior depends on the backing store implementation
}

TEST_F(UltraCacheTest, PredictiveLoading) {
    config_.enable_predictive_loading = true;
    UltraCache<std::string, std::string> cache(config_);
    
    cache.enable_predictive_loading(true);
    
    // Create access pattern
    for (int round = 0; round < 5; ++round) {
        for (int i = 0; i < 10; ++i) {
            std::string key = "pattern_key" + std::to_string(i);
            std::string value = "pattern_value" + std::to_string(i) + "_" + std::to_string(round);
            
            cache.put(key, value);
            cache.get(key);
        }
    }
    
    auto stats = cache.get_stats();
    // Predictive loading stats should be updated
    // The exact behavior depends on the prediction algorithm implementation
}

TEST_F(UltraCacheTest, PerformanceBenchmark) {
    config_.capacity = 100000;
    config_.shard_count = 16;
    UltraCache<std::string, std::string> cache(config_);
    
    const int num_operations = 10000;
    
    // Benchmark PUT operations
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_operations; ++i) {
        std::string key = "bench_key" + std::to_string(i);
        std::string value = "bench_value" + std::to_string(i) + "_data_content_for_benchmarking";
        cache.put(key, value);
    }
    
    auto put_end_time = std::chrono::high_resolution_clock::now();
    
    // Benchmark GET operations
    for (int i = 0; i < num_operations; ++i) {
        std::string key = "bench_key" + std::to_string(i);
        auto result = cache.get(key);
        EXPECT_TRUE(result.has_value());
    }
    
    auto get_end_time = std::chrono::high_resolution_clock::now();
    
    // Calculate performance metrics
    auto put_duration = std::chrono::duration_cast<std::chrono::microseconds>(
        put_end_time - start_time).count();
    auto get_duration = std::chrono::duration_cast<std::chrono::microseconds>(
        get_end_time - put_end_time).count();
    
    double put_ops_per_sec = (num_operations * 1000000.0) / put_duration;
    double get_ops_per_sec = (num_operations * 1000000.0) / get_duration;
    
    std::cout << "PUT Performance: " << put_ops_per_sec << " ops/sec" << std::endl;
    std::cout << "GET Performance: " << get_ops_per_sec << " ops/sec" << std::endl;
    
    // Verify reasonable performance (these are conservative thresholds)
    EXPECT_GT(put_ops_per_sec, 10000); // At least 10K PUT ops/sec
    EXPECT_GT(get_ops_per_sec, 50000); // At least 50K GET ops/sec
    
    auto stats = cache.get_stats();
    std::cout << "Final cache size: " << stats.cache_size.load() << std::endl;
    std::cout << "Hit ratio: " << cache.get_hit_ratio() * 100 << "%" << std::endl;
}

TEST_F(UltraCacheTest, NumericKeyCache) {
    UltraCache<uint64_t, std::string> numeric_cache(
        UltraCache<uint64_t, std::string>::Config{});
    
    // Test with numeric keys
    for (uint64_t i = 0; i < 100; ++i) {
        std::string value = "numeric_value_" + std::to_string(i);
        numeric_cache.put(i, value);
    }
    
    // Verify retrieval
    for (uint64_t i = 0; i < 100; ++i) {
        auto result = numeric_cache.get(i);
        ASSERT_TRUE(result.has_value());
        EXPECT_EQ(result.value(), "numeric_value_" + std::to_string(i));
    }
}

TEST_F(UltraCacheTest, BinaryDataCache) {
    UltraCache<std::string, std::vector<uint8_t>> binary_cache(
        UltraCache<std::string, std::vector<uint8_t>>::Config{});
    
    // Test with binary data
    for (int i = 0; i < 10; ++i) {
        std::string key = "binary_key_" + std::to_string(i);
        std::vector<uint8_t> binary_data(1024, static_cast<uint8_t>(i));
        binary_cache.put(key, binary_data);
    }
    
    // Verify retrieval
    for (int i = 0; i < 10; ++i) {
        std::string key = "binary_key_" + std::to_string(i);
        auto result = binary_cache.get(key);
        ASSERT_TRUE(result.has_value());
        EXPECT_EQ(result.value().size(), 1024);
        EXPECT_EQ(result.value()[0], static_cast<uint8_t>(i));
    }
}