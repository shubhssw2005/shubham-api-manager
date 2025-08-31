#include <gtest/gtest.h>
#include <thread>
#include <vector>
#include <atomic>
#include <chrono>
#include <random>
#include "lockfree/lockfree_lru_cache.hpp"

using namespace ultra_cpp::lockfree;

class LockFreeLRUCacheTest : public ::testing::Test {
protected:
    void SetUp() override {
        cache = std::make_unique<LRUCache<int, std::string, 8>>();
    }
    
    void TearDown() override {
        cache.reset();
    }
    
    std::unique_ptr<LRUCache<int, std::string, 8>> cache;
};

TEST_F(LockFreeLRUCacheTest, BasicOperations) {
    // Test empty cache
    EXPECT_TRUE(cache->empty());
    EXPECT_FALSE(cache->full());
    EXPECT_EQ(cache->size(), 0);
    
    // Test put and get
    EXPECT_TRUE(cache->put(1, "value1"));
    EXPECT_TRUE(cache->put(2, "value2"));
    
    auto result1 = cache->get(1);
    ASSERT_TRUE(result1.has_value());
    EXPECT_EQ(result1.value(), "value1");
    
    auto result2 = cache->get(2);
    ASSERT_TRUE(result2.has_value());
    EXPECT_EQ(result2.value(), "value2");
    
    // Test non-existent key
    auto result3 = cache->get(3);
    EXPECT_FALSE(result3.has_value());
    
    EXPECT_EQ(cache->size(), 2);
}

TEST_F(LockFreeLRUCacheTest, UpdateExistingKey) {
    // Insert initial value
    EXPECT_TRUE(cache->put(1, "initial"));
    
    auto result1 = cache->get(1);
    ASSERT_TRUE(result1.has_value());
    EXPECT_EQ(result1.value(), "initial");
    
    // Update value
    EXPECT_TRUE(cache->put(1, "updated"));
    
    auto result2 = cache->get(1);
    ASSERT_TRUE(result2.has_value());
    EXPECT_EQ(result2.value(), "updated");
    
    // Size should remain the same
    EXPECT_EQ(cache->size(), 1);
}

TEST_F(LockFreeLRUCacheTest, RemoveOperations) {
    // Insert and remove
    EXPECT_TRUE(cache->put(1, "value1"));
    EXPECT_TRUE(cache->put(2, "value2"));
    
    EXPECT_TRUE(cache->remove(1));
    
    auto result1 = cache->get(1);
    EXPECT_FALSE(result1.has_value());
    
    auto result2 = cache->get(2);
    ASSERT_TRUE(result2.has_value());
    EXPECT_EQ(result2.value(), "value2");
    
    EXPECT_EQ(cache->size(), 1);
    
    // Remove non-existent key
    EXPECT_FALSE(cache->remove(3));
}

TEST_F(LockFreeLRUCacheTest, LRUEviction) {
    // Fill cache to capacity
    for (int i = 0; i < 8; ++i) {
        EXPECT_TRUE(cache->put(i, "value" + std::to_string(i)));
    }
    
    EXPECT_TRUE(cache->full());
    EXPECT_EQ(cache->size(), 8);
    
    // Access some items to change their order
    cache->get(0);  // Make 0 most recently used
    cache->get(1);  // Make 1 most recently used
    
    // Add new item, should evict least recently used (which should be 2)
    EXPECT_TRUE(cache->put(8, "value8"));
    
    // Item 2 should be evicted
    auto result2 = cache->get(2);
    EXPECT_FALSE(result2.has_value());
    
    // Items 0, 1, and 8 should still be present
    auto result0 = cache->get(0);
    EXPECT_TRUE(result0.has_value());
    
    auto result1 = cache->get(1);
    EXPECT_TRUE(result1.has_value());
    
    auto result8 = cache->get(8);
    EXPECT_TRUE(result8.has_value());
}

TEST_F(LockFreeLRUCacheTest, Statistics) {
    const auto& stats = cache->get_stats();
    
    // Initial state
    EXPECT_EQ(stats.hits.load(), 0);
    EXPECT_EQ(stats.misses.load(), 0);
    EXPECT_EQ(stats.insertions.load(), 0);
    EXPECT_EQ(stats.updates.load(), 0);
    EXPECT_EQ(stats.evictions.load(), 0);
    
    // Perform operations
    cache->put(1, "value1");  // insertion
    cache->get(1);            // hit
    cache->get(2);            // miss
    cache->put(1, "updated"); // update
    
    EXPECT_EQ(stats.hits.load(), 1);
    EXPECT_EQ(stats.misses.load(), 1);
    EXPECT_EQ(stats.insertions.load(), 1);
    EXPECT_EQ(stats.updates.load(), 1);
    EXPECT_EQ(stats.evictions.load(), 0);
    
    // Test hit rate
    EXPECT_DOUBLE_EQ(cache->hit_rate(), 0.5); // 1 hit out of 2 total accesses
    
    // Reset stats
    cache->reset_stats();
    EXPECT_EQ(stats.hits.load(), 0);
    EXPECT_EQ(stats.misses.load(), 0);
    EXPECT_DOUBLE_EQ(cache->hit_rate(), 0.0);
}

TEST_F(LockFreeLRUCacheTest, ConcurrentOperations) {
    const int num_threads = 4;
    const int operations_per_thread = 1000;
    std::vector<std::thread> threads;
    std::atomic<int> total_operations{0};
    
    // Concurrent mixed operations
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> op_dist(0, 2);  // 0=put, 1=get, 2=remove
            std::uniform_int_distribution<int> key_dist(1, 100);
            
            for (int i = 0; i < operations_per_thread; ++i) {
                int key = key_dist(gen);
                int operation = op_dist(gen);
                
                switch (operation) {
                    case 0: // put
                        cache->put(key, "thread" + std::to_string(t) + "_value" + std::to_string(i));
                        break;
                    case 1: // get
                        cache->get(key);
                        break;
                    case 2: // remove
                        cache->remove(key);
                        break;
                }
                
                total_operations.fetch_add(1);
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    EXPECT_EQ(total_operations.load(), num_threads * operations_per_thread);
    
    // Cache should still be functional
    EXPECT_TRUE(cache->put(999, "test"));
    auto result = cache->get(999);
    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), "test");
}

TEST_F(LockFreeLRUCacheTest, ConcurrentProducerConsumer) {
    const int num_producers = 2;
    const int num_consumers = 2;
    const int items_per_producer = 500;
    
    std::vector<std::thread> producers;
    std::vector<std::thread> consumers;
    std::atomic<bool> producers_done{false};
    std::atomic<int> items_produced{0};
    std::atomic<int> cache_hits{0};
    
    // Start producers
    for (int p = 0; p < num_producers; ++p) {
        producers.emplace_back([&, p]() {
            for (int i = 0; i < items_per_producer; ++i) {
                int key = p * items_per_producer + i;
                std::string value = "producer" + std::to_string(p) + "_item" + std::to_string(i);
                
                cache->put(key, value);
                items_produced.fetch_add(1);
                
                // Small delay to allow consumers to work
                if (i % 100 == 0) {
                    std::this_thread::sleep_for(std::chrono::microseconds(1));
                }
            }
        });
    }
    
    // Start consumers
    for (int c = 0; c < num_consumers; ++c) {
        consumers.emplace_back([&]() {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<int> key_dist(0, num_producers * items_per_producer - 1);
            
            while (!producers_done.load()) {
                int key = key_dist(gen);
                auto result = cache->get(key);
                
                if (result.has_value()) {
                    cache_hits.fetch_add(1);
                }
                
                std::this_thread::sleep_for(std::chrono::microseconds(1));
            }
        });
    }
    
    // Wait for producers to finish
    for (auto& producer : producers) {
        producer.join();
    }
    producers_done.store(true);
    
    // Wait for consumers to finish
    for (auto& consumer : consumers) {
        consumer.join();
    }
    
    EXPECT_EQ(items_produced.load(), num_producers * items_per_producer);
    EXPECT_GT(cache_hits.load(), 0); // Should have some cache hits
}

TEST_F(LockFreeLRUCacheTest, PerformanceBenchmark) {
    const int num_operations = 100000;
    
    // Pre-populate cache
    for (int i = 0; i < 8; ++i) {
        cache->put(i, "value" + std::to_string(i));
    }
    
    // Benchmark cache hits
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_operations; ++i) {
        int key = i % 8; // Ensure cache hits
        auto result = cache->get(key);
        EXPECT_TRUE(result.has_value());
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    double ops_per_second = (num_operations * 1000000.0) / duration.count();
    
    // Should achieve at least 500K ops/sec for cache hits
    EXPECT_GT(ops_per_second, 500000.0);
    
    std::cout << "LRU cache hit performance: " << ops_per_second << " ops/sec" << std::endl;
    
    // Benchmark mixed operations
    start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_operations; ++i) {
        if (i % 3 == 0) {
            cache->put(i % 20, "value" + std::to_string(i));
        } else {
            cache->get(i % 20);
        }
    }
    
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    ops_per_second = (num_operations * 1000000.0) / duration.count();
    
    // Mixed operations should still be fast
    EXPECT_GT(ops_per_second, 200000.0);
    
    std::cout << "LRU cache mixed operations performance: " << ops_per_second << " ops/sec" << std::endl;
}