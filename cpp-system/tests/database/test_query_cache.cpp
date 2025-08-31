#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "database/database_connector.hpp"
#include "common/logger.hpp"
#include <thread>
#include <chrono>

using namespace ultra_cpp::database;
using namespace testing;

class QueryCacheTest : public ::testing::Test {
protected:
    void SetUp() override {
        config_.max_entries = 1000;
        config_.default_ttl_seconds = 300; // 5 minutes
        config_.enable_compression = false; // Disable for testing
        config_.max_result_size_bytes = 1024 * 1024; // 1MB
    }
    
    void TearDown() override {
        // Cleanup
    }
    
    QueryCache::Config config_;
    
    DatabaseConnector::QueryResult create_test_result(const std::string& value) {
        DatabaseConnector::QueryResult result;
        result.success = true;
        result.affected_rows = 1;
        result.execution_time_ns = 1000000; // 1ms
        result.rows = {{value}};
        return result;
    }
};

TEST_F(QueryCacheTest, ConstructorInitializesCorrectly) {
    QueryCache cache(config_);
    
    auto stats = cache.get_stats();
    EXPECT_EQ(stats.hits.load(), 0);
    EXPECT_EQ(stats.misses.load(), 0);
    EXPECT_EQ(stats.evictions.load(), 0);
    EXPECT_EQ(stats.invalidations.load(), 0);
    EXPECT_EQ(stats.current_entries.load(), 0);
    EXPECT_EQ(stats.total_size_bytes.load(), 0);
}

TEST_F(QueryCacheTest, BasicCacheOperations) {
    QueryCache cache(config_);
    
    std::string query = "SELECT * FROM users WHERE id = ?";
    std::vector<std::string> params = {"123"};
    auto result = create_test_result("test_value");
    
    // Cache miss on first access
    auto cached_result = cache.get(query, params);
    EXPECT_FALSE(cached_result.has_value());
    
    auto stats = cache.get_stats();
    EXPECT_EQ(stats.misses.load(), 1);
    EXPECT_EQ(stats.hits.load(), 0);
    
    // Put result in cache
    cache.put(query, params, result);
    
    stats = cache.get_stats();
    EXPECT_EQ(stats.current_entries.load(), 1);
    EXPECT_GT(stats.total_size_bytes.load(), 0);
    
    // Cache hit on second access
    cached_result = cache.get(query, params);
    ASSERT_TRUE(cached_result.has_value());
    EXPECT_TRUE(cached_result->success);
    EXPECT_EQ(cached_result->rows.size(), 1);
    EXPECT_EQ(cached_result->rows[0][0], "test_value");
    
    stats = cache.get_stats();
    EXPECT_EQ(stats.hits.load(), 1);
    EXPECT_EQ(stats.misses.load(), 1);
}

TEST_F(QueryCacheTest, DifferentParametersCreateDifferentKeys) {
    QueryCache cache(config_);
    
    std::string query = "SELECT * FROM users WHERE id = ?";
    auto result1 = create_test_result("user1");
    auto result2 = create_test_result("user2");
    
    // Cache results with different parameters
    cache.put(query, {"1"}, result1);
    cache.put(query, {"2"}, result2);
    
    // Verify different results are cached
    auto cached1 = cache.get(query, {"1"});
    auto cached2 = cache.get(query, {"2"});
    
    ASSERT_TRUE(cached1.has_value());
    ASSERT_TRUE(cached2.has_value());
    EXPECT_EQ(cached1->rows[0][0], "user1");
    EXPECT_EQ(cached2->rows[0][0], "user2");
    
    auto stats = cache.get_stats();
    EXPECT_EQ(stats.current_entries.load(), 2);
    EXPECT_EQ(stats.hits.load(), 2);
}

TEST_F(QueryCacheTest, FailedQueriesNotCached) {
    QueryCache cache(config_);
    
    DatabaseConnector::QueryResult failed_result;
    failed_result.success = false;
    failed_result.error_message = "Query failed";
    
    std::string query = "INVALID SQL";
    cache.put(query, {}, failed_result);
    
    // Failed results should not be cached
    auto stats = cache.get_stats();
    EXPECT_EQ(stats.current_entries.load(), 0);
}

TEST_F(QueryCacheTest, TTLExpiration) {
    config_.default_ttl_seconds = 1; // 1 second TTL
    QueryCache cache(config_);
    
    std::string query = "SELECT NOW()";
    auto result = create_test_result("timestamp");
    
    // Cache with short TTL
    cache.put(query, {}, result, 1); // 1 second TTL
    
    // Should be available immediately
    auto cached = cache.get(query, {});
    ASSERT_TRUE(cached.has_value());
    
    auto stats = cache.get_stats();
    EXPECT_EQ(stats.hits.load(), 1);
    
    // Wait for expiration
    std::this_thread::sleep_for(std::chrono::seconds(2));
    
    // Should be expired now
    cached = cache.get(query, {});
    EXPECT_FALSE(cached.has_value());
    
    stats = cache.get_stats();
    EXPECT_EQ(stats.misses.load(), 1);
    EXPECT_EQ(stats.current_entries.load(), 0); // Entry should be removed
}

TEST_F(QueryCacheTest, CacheInvalidation) {
    QueryCache cache(config_);
    
    // Cache multiple entries
    cache.put("SELECT * FROM users", {}, create_test_result("users"));
    cache.put("SELECT * FROM orders", {}, create_test_result("orders"));
    cache.put("SELECT * FROM products", {}, create_test_result("products"));
    
    auto stats = cache.get_stats();
    EXPECT_EQ(stats.current_entries.load(), 3);
    
    // Invalidate entries matching pattern
    cache.invalidate(".*users.*");
    
    stats = cache.get_stats();
    EXPECT_EQ(stats.current_entries.load(), 2);
    EXPECT_EQ(stats.invalidations.load(), 1);
    
    // Verify specific entries
    auto users_result = cache.get("SELECT * FROM users", {});
    auto orders_result = cache.get("SELECT * FROM orders", {});
    
    EXPECT_FALSE(users_result.has_value());
    EXPECT_TRUE(orders_result.has_value());
}

TEST_F(QueryCacheTest, InvalidateAll) {
    QueryCache cache(config_);
    
    // Cache multiple entries
    for (int i = 0; i < 10; ++i) {
        std::string query = "SELECT " + std::to_string(i);
        cache.put(query, {}, create_test_result(std::to_string(i)));
    }
    
    auto stats = cache.get_stats();
    EXPECT_EQ(stats.current_entries.load(), 10);
    
    // Invalidate all
    cache.invalidate_all();
    
    stats = cache.get_stats();
    EXPECT_EQ(stats.current_entries.load(), 0);
    EXPECT_EQ(stats.total_size_bytes.load(), 0);
    EXPECT_EQ(stats.invalidations.load(), 10);
}

TEST_F(QueryCacheTest, MaxEntriesEviction) {
    config_.max_entries = 5; // Small cache for testing
    QueryCache cache(config_);
    
    // Fill cache beyond capacity
    for (int i = 0; i < 10; ++i) {
        std::string query = "SELECT " + std::to_string(i);
        cache.put(query, {}, create_test_result(std::to_string(i)));
    }
    
    auto stats = cache.get_stats();
    EXPECT_LE(stats.current_entries.load(), config_.max_entries);
    EXPECT_GT(stats.evictions.load(), 0);
}

TEST_F(QueryCacheTest, LargeResultNotCached) {
    config_.max_result_size_bytes = 100; // Very small limit
    QueryCache cache(config_);
    
    // Create a large result
    DatabaseConnector::QueryResult large_result;
    large_result.success = true;
    large_result.rows.resize(100);
    for (auto& row : large_result.rows) {
        row.resize(10, "large_data_value_that_exceeds_limit");
    }
    
    std::string query = "SELECT * FROM large_table";
    cache.put(query, {}, large_result);
    
    // Large result should not be cached
    auto stats = cache.get_stats();
    EXPECT_EQ(stats.current_entries.load(), 0);
}

TEST_F(QueryCacheTest, CacheKeyGeneration) {
    QueryCache cache(config_);
    
    // Same query with different parameters should have different keys
    std::string query = "SELECT * FROM users WHERE id = ? AND name = ?";
    
    cache.put(query, {"1", "Alice"}, create_test_result("Alice"));
    cache.put(query, {"2", "Bob"}, create_test_result("Bob"));
    cache.put(query, {"1", "Alice"}, create_test_result("Alice_Updated")); // Same key, should update
    
    auto stats = cache.get_stats();
    EXPECT_EQ(stats.current_entries.load(), 2); // Only 2 unique keys
    
    auto alice_result = cache.get(query, {"1", "Alice"});
    ASSERT_TRUE(alice_result.has_value());
    EXPECT_EQ(alice_result->rows[0][0], "Alice_Updated"); // Should be updated value
}

TEST_F(QueryCacheTest, StatisticsTracking) {
    QueryCache cache(config_);
    
    std::string query = "SELECT test";
    auto result = create_test_result("test");
    
    // Initial stats
    auto stats = cache.get_stats();
    EXPECT_EQ(stats.hits.load(), 0);
    EXPECT_EQ(stats.misses.load(), 0);
    
    // Cache miss
    cache.get(query, {});
    stats = cache.get_stats();
    EXPECT_EQ(stats.misses.load(), 1);
    
    // Cache put
    cache.put(query, {}, result);
    stats = cache.get_stats();
    EXPECT_EQ(stats.current_entries.load(), 1);
    EXPECT_GT(stats.total_size_bytes.load(), 0);
    
    // Cache hit
    cache.get(query, {});
    stats = cache.get_stats();
    EXPECT_EQ(stats.hits.load(), 1);
    
    // Reset stats
    cache.reset_stats();
    stats = cache.get_stats();
    EXPECT_EQ(stats.hits.load(), 0);
    EXPECT_EQ(stats.misses.load(), 0);
    // current_entries and total_size_bytes should not be reset
    EXPECT_EQ(stats.current_entries.load(), 1);
    EXPECT_GT(stats.total_size_bytes.load(), 0);
}

// Performance tests
class QueryCachePerformanceTest : public ::testing::Test {
protected:
    void SetUp() override {
        config_.max_entries = 10000;
        config_.default_ttl_seconds = 3600; // 1 hour
        config_.enable_compression = false;
        config_.max_result_size_bytes = 10 * 1024 * 1024; // 10MB
    }
    
    QueryCache::Config config_;
    
    DatabaseConnector::QueryResult create_test_result(int size) {
        DatabaseConnector::QueryResult result;
        result.success = true;
        result.affected_rows = size;
        result.execution_time_ns = 1000000;
        
        result.rows.resize(size);
        for (int i = 0; i < size; ++i) {
            result.rows[i] = {"col1_" + std::to_string(i), "col2_" + std::to_string(i)};
        }
        
        return result;
    }
};

TEST_F(QueryCachePerformanceTest, HighVolumeCacheOperations) {
    QueryCache cache(config_);
    
    const int num_operations = 10000;
    const int result_size = 10; // 10 rows per result
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Fill cache
    for (int i = 0; i < num_operations; ++i) {
        std::string query = "SELECT * FROM table" + std::to_string(i % 1000); // Some overlap
        std::vector<std::string> params = {std::to_string(i)};
        auto result = create_test_result(result_size);
        
        cache.put(query, params, result);
    }
    
    auto fill_time = std::chrono::high_resolution_clock::now();
    
    // Perform lookups
    int hits = 0;
    int misses = 0;
    
    for (int i = 0; i < num_operations; ++i) {
        std::string query = "SELECT * FROM table" + std::to_string(i % 1000);
        std::vector<std::string> params = {std::to_string(i)};
        
        auto result = cache.get(query, params);
        if (result.has_value()) {
            hits++;
        } else {
            misses++;
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    
    auto fill_duration = std::chrono::duration_cast<std::chrono::milliseconds>(fill_time - start_time);
    auto lookup_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - fill_time);
    
    std::cout << "Cache performance test results:" << std::endl;
    std::cout << "  Fill time: " << fill_duration.count() << "ms" << std::endl;
    std::cout << "  Lookup time: " << lookup_duration.count() << "ms" << std::endl;
    std::cout << "  Cache hits: " << hits << std::endl;
    std::cout << "  Cache misses: " << misses << std::endl;
    
    auto stats = cache.get_stats();
    std::cout << "  Final entries: " << stats.current_entries.load() << std::endl;
    std::cout << "  Total size: " << stats.total_size_bytes.load() << " bytes" << std::endl;
    std::cout << "  Evictions: " << stats.evictions.load() << std::endl;
    
    if (fill_duration.count() > 0) {
        double fill_ops_per_ms = static_cast<double>(num_operations) / fill_duration.count();
        std::cout << "  Fill operations per ms: " << fill_ops_per_ms << std::endl;
    }
    
    if (lookup_duration.count() > 0) {
        double lookup_ops_per_ms = static_cast<double>(num_operations) / lookup_duration.count();
        std::cout << "  Lookup operations per ms: " << lookup_ops_per_ms << std::endl;
    }
}

TEST_F(QueryCachePerformanceTest, ConcurrentCacheAccess) {
    QueryCache cache(config_);
    
    const int num_threads = 8;
    const int operations_per_thread = 1000;
    
    // Pre-populate cache
    for (int i = 0; i < 500; ++i) {
        std::string query = "SELECT * FROM shared_table WHERE id = ?";
        std::vector<std::string> params = {std::to_string(i)};
        cache.put(query, params, create_test_result(5));
    }
    
    std::vector<std::thread> threads;
    std::atomic<int> total_hits{0};
    std::atomic<int> total_misses{0};
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&cache, &total_hits, &total_misses, operations_per_thread, t]() {
            int local_hits = 0;
            int local_misses = 0;
            
            for (int i = 0; i < operations_per_thread; ++i) {
                std::string query = "SELECT * FROM shared_table WHERE id = ?";
                std::vector<std::string> params = {std::to_string((t * operations_per_thread + i) % 1000)};
                
                auto result = cache.get(query, params);
                if (result.has_value()) {
                    local_hits++;
                } else {
                    local_misses++;
                    // Cache the result for future hits
                    cache.put(query, params, create_test_result(5));
                }
            }
            
            total_hits.fetch_add(local_hits);
            total_misses.fetch_add(local_misses);
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    int total_operations = num_threads * operations_per_thread;
    
    std::cout << "Concurrent cache access test results:" << std::endl;
    std::cout << "  Duration: " << duration.count() << "ms" << std::endl;
    std::cout << "  Total operations: " << total_operations << std::endl;
    std::cout << "  Total hits: " << total_hits.load() << std::endl;
    std::cout << "  Total misses: " << total_misses.load() << std::endl;
    std::cout << "  Hit ratio: " << (static_cast<double>(total_hits.load()) / total_operations * 100.0) << "%" << std::endl;
    
    if (duration.count() > 0) {
        double ops_per_ms = static_cast<double>(total_operations) / duration.count();
        std::cout << "  Operations per ms: " << ops_per_ms << std::endl;
    }
    
    auto stats = cache.get_stats();
    std::cout << "  Final cache entries: " << stats.current_entries.load() << std::endl;
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    
    // Initialize logging for tests
    ultra_cpp::common::Logger::initialize(ultra_cpp::common::LogLevel::DEBUG);
    
    return RUN_ALL_TESTS();
}