#include <gtest/gtest.h>
#include <thread>
#include <vector>
#include <atomic>
#include <random>
#include <chrono>
#include "lockfree/lockfree_hash_table.hpp"

using namespace ultra_cpp::lockfree;

class LockFreeHashTableTest : public ::testing::Test {
protected:
    void SetUp() override {
        hash_table = std::make_unique<HashTable<uint64_t, std::string, 1024>>();
    }
    
    void TearDown() override {
        hash_table.reset();
    }
    
    std::unique_ptr<HashTable<uint64_t, std::string, 1024>> hash_table;
};

TEST_F(LockFreeHashTableTest, BasicOperations) {
    // Test put and get
    EXPECT_TRUE(hash_table->put(1, "value1"));
    EXPECT_TRUE(hash_table->put(2, "value2"));
    
    auto result1 = hash_table->get(1);
    ASSERT_TRUE(result1.has_value());
    EXPECT_EQ(result1.value(), "value1");
    
    auto result2 = hash_table->get(2);
    ASSERT_TRUE(result2.has_value());
    EXPECT_EQ(result2.value(), "value2");
    
    // Test non-existent key
    auto result3 = hash_table->get(3);
    EXPECT_FALSE(result3.has_value());
}

TEST_F(LockFreeHashTableTest, UpdateExistingKey) {
    // Insert initial value
    EXPECT_TRUE(hash_table->put(1, "initial"));
    
    auto result1 = hash_table->get(1);
    ASSERT_TRUE(result1.has_value());
    EXPECT_EQ(result1.value(), "initial");
    
    // Update value
    EXPECT_TRUE(hash_table->put(1, "updated"));
    
    auto result2 = hash_table->get(1);
    ASSERT_TRUE(result2.has_value());
    EXPECT_EQ(result2.value(), "updated");
}

TEST_F(LockFreeHashTableTest, RemoveOperations) {
    // Insert and remove
    EXPECT_TRUE(hash_table->put(1, "value1"));
    EXPECT_TRUE(hash_table->put(2, "value2"));
    
    EXPECT_TRUE(hash_table->remove(1));
    
    auto result1 = hash_table->get(1);
    EXPECT_FALSE(result1.has_value());
    
    auto result2 = hash_table->get(2);
    ASSERT_TRUE(result2.has_value());
    EXPECT_EQ(result2.value(), "value2");
    
    // Remove non-existent key
    EXPECT_FALSE(hash_table->remove(3));
}

TEST_F(LockFreeHashTableTest, Statistics) {
    const auto& stats = hash_table->get_stats();
    
    // Initial state
    EXPECT_EQ(stats.size.load(), 0);
    
    // Add some entries
    hash_table->put(1, "value1");
    hash_table->put(2, "value2");
    hash_table->put(3, "value3");
    
    EXPECT_EQ(stats.size.load(), 3);
    
    // Remove one entry
    hash_table->remove(2);
    EXPECT_EQ(stats.size.load(), 2);
}

TEST_F(LockFreeHashTableTest, ConcurrentOperations) {
    const int num_threads = 8;
    const int operations_per_thread = 1000;
    std::vector<std::thread> threads;
    std::atomic<int> success_count{0};
    
    // Concurrent insertions
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            for (int i = 0; i < operations_per_thread; ++i) {
                uint64_t key = t * operations_per_thread + i;
                std::string value = "thread" + std::to_string(t) + "_value" + std::to_string(i);
                
                if (hash_table->put(key, value)) {
                    success_count.fetch_add(1);
                }
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    // Verify all insertions succeeded
    EXPECT_EQ(success_count.load(), num_threads * operations_per_thread);
    
    threads.clear();
    success_count.store(0);
    
    // Concurrent reads
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            for (int i = 0; i < operations_per_thread; ++i) {
                uint64_t key = t * operations_per_thread + i;
                auto result = hash_table->get(key);
                
                if (result.has_value()) {
                    success_count.fetch_add(1);
                }
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    // Verify all reads succeeded
    EXPECT_EQ(success_count.load(), num_threads * operations_per_thread);
}

TEST_F(LockFreeHashTableTest, ConcurrentMixedOperations) {
    const int num_threads = 4;
    const int operations_per_thread = 500;
    std::vector<std::thread> threads;
    std::atomic<int> total_operations{0};
    
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> op_dist(0, 2);  // 0=put, 1=get, 2=remove
            std::uniform_int_distribution<uint64_t> key_dist(1, 1000);
            
            for (int i = 0; i < operations_per_thread; ++i) {
                uint64_t key = key_dist(gen);
                int operation = op_dist(gen);
                
                switch (operation) {
                    case 0: // put
                        hash_table->put(key, "value" + std::to_string(key));
                        break;
                    case 1: // get
                        hash_table->get(key);
                        break;
                    case 2: // remove
                        hash_table->remove(key);
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
}

TEST_F(LockFreeHashTableTest, LoadFactor) {
    // Test load factor calculation
    EXPECT_DOUBLE_EQ(hash_table->load_factor(), 0.0);
    
    // Add entries to increase load factor
    for (int i = 0; i < 100; ++i) {
        hash_table->put(i, "value" + std::to_string(i));
    }
    
    double expected_load_factor = 100.0 / 1024.0;
    EXPECT_DOUBLE_EQ(hash_table->load_factor(), expected_load_factor);
}

TEST_F(LockFreeHashTableTest, PerformanceBenchmark) {
    const int num_operations = 100000;
    
    // Benchmark insertions
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_operations; ++i) {
        hash_table->put(i, "value" + std::to_string(i));
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    double ops_per_second = (num_operations * 1000000.0) / duration.count();
    
    // Should achieve at least 100K ops/sec (very conservative)
    EXPECT_GT(ops_per_second, 100000.0);
    
    std::cout << "Hash table insertion performance: " << ops_per_second << " ops/sec" << std::endl;
    
    // Benchmark lookups
    start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_operations; ++i) {
        auto result = hash_table->get(i);
        EXPECT_TRUE(result.has_value());
    }
    
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    ops_per_second = (num_operations * 1000000.0) / duration.count();
    
    // Lookups should be even faster
    EXPECT_GT(ops_per_second, 200000.0);
    
    std::cout << "Hash table lookup performance: " << ops_per_second << " ops/sec" << std::endl;
}