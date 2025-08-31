#include "database/database_connector.hpp"
#include "common/logger.hpp"
#include <iostream>
#include <thread>
#include <chrono>
#include <vector>
#include <future>

using namespace ultra_cpp::database;
using namespace ultra_cpp::common;

void demonstrate_basic_connector() {
    std::cout << "\n=== Basic Database Connector Demo ===" << std::endl;
    
    // Configure database connection
    DatabaseConnector::Config config;
    config.host = "localhost";
    config.port = 5432;
    config.database = "ultra_cpp_demo";
    config.username = "demo_user";
    config.password = "demo_password";
    config.connection_timeout_ms = 5000;
    config.query_timeout_ms = 30000;
    config.enable_ssl = false; // For demo purposes
    
    DatabaseConnector connector(config);
    
    // Attempt to connect
    std::cout << "Connecting to database..." << std::endl;
    if (!connector.connect()) {
        std::cout << "Failed to connect to database (this is expected in demo environment)" << std::endl;
        return;
    }
    
    std::cout << "Connected successfully!" << std::endl;
    
    // Test basic query
    auto result = connector.execute_query("SELECT 'Hello, World!' as greeting, NOW() as current_time");
    if (result.success) {
        std::cout << "Query executed successfully!" << std::endl;
        std::cout << "Execution time: " << result.execution_time_ns / 1000000.0 << "ms" << std::endl;
        
        for (const auto& row : result.rows) {
            for (const auto& cell : row) {
                std::cout << cell << " ";
            }
            std::cout << std::endl;
        }
    } else {
        std::cout << "Query failed: " << result.error_message << std::endl;
    }
    
    // Test prepared statements
    std::cout << "\nTesting prepared statements..." << std::endl;
    bool prepared = connector.prepare_statement("demo_stmt", 
                                               "SELECT $1::text as input, $2::int as number", 
                                               {TEXTOID, INT4OID});
    
    if (prepared) {
        auto prep_result = connector.execute_prepared("demo_stmt", {"test_string", "42"});
        if (prep_result.success) {
            std::cout << "Prepared statement executed successfully!" << std::endl;
            for (const auto& row : prep_result.rows) {
                for (const auto& cell : row) {
                    std::cout << cell << " ";
                }
                std::cout << std::endl;
            }
        }
    }
    
    // Test transactions
    std::cout << "\nTesting transactions..." << std::endl;
    if (connector.begin_transaction()) {
        std::cout << "Transaction started" << std::endl;
        
        // Execute some queries in transaction
        auto tx_result = connector.execute_query("SELECT 1");
        
        if (connector.commit_transaction()) {
            std::cout << "Transaction committed" << std::endl;
        }
    }
    
    // Display metrics
    auto metrics = connector.get_metrics();
    std::cout << "\nConnection Metrics:" << std::endl;
    std::cout << "  Queries executed: " << metrics.queries_executed.load() << std::endl;
    std::cout << "  Queries failed: " << metrics.queries_failed.load() << std::endl;
    std::cout << "  Total execution time: " << metrics.total_execution_time_ns.load() / 1000000.0 << "ms" << std::endl;
    std::cout << "  Connections created: " << metrics.connections_created.load() << std::endl;
    
    connector.disconnect();
    std::cout << "Disconnected from database" << std::endl;
}

void demonstrate_connection_pool() {
    std::cout << "\n=== Connection Pool Demo ===" << std::endl;
    
    // Configure connection pool
    ConnectionPool::Config config;
    config.db_config.host = "localhost";
    config.db_config.port = 5432;
    config.db_config.database = "ultra_cpp_demo";
    config.db_config.username = "demo_user";
    config.db_config.password = "demo_password";
    config.db_config.enable_ssl = false;
    
    config.min_connections = 3;
    config.max_connections = 10;
    config.connection_idle_timeout_ms = 60000;
    config.health_check_interval_ms = 30000;
    config.enable_load_balancing = true;
    config.enable_failover = true;
    
    ConnectionPool pool(config);
    
    std::cout << "Initializing connection pool..." << std::endl;
    if (!pool.initialize()) {
        std::cout << "Failed to initialize connection pool (expected in demo environment)" << std::endl;
        return;
    }
    
    std::cout << "Connection pool initialized successfully!" << std::endl;
    std::cout << "Pool is healthy: " << (pool.is_healthy() ? "Yes" : "No") << std::endl;
    
    // Test connection acquisition
    std::cout << "\nTesting connection acquisition..." << std::endl;
    {
        auto connection = pool.acquire_connection(5000);
        if (connection.has_value()) {
            std::cout << "Connection acquired successfully!" << std::endl;
            std::cout << "Connection is valid: " << (connection->is_valid() ? "Yes" : "No") << std::endl;
            
            // Use the connection
            auto result = connection->execute_query("SELECT 'Pool connection test' as message");
            if (result.success) {
                std::cout << "Query through pool connection successful!" << std::endl;
            }
            
            // Connection is automatically released when it goes out of scope
        } else {
            std::cout << "Failed to acquire connection" << std::endl;
        }
    }
    
    // Test high-level pool query interface
    std::cout << "\nTesting pool query interface..." << std::endl;
    auto pool_result = pool.execute_query("SELECT COUNT(*) as connection_test");
    if (pool_result.success) {
        std::cout << "Pool query executed successfully!" << std::endl;
    } else {
        std::cout << "Pool query failed: " << pool_result.error_message << std::endl;
    }
    
    // Test concurrent access
    std::cout << "\nTesting concurrent pool access..." << std::endl;
    const int num_threads = 5;
    const int queries_per_thread = 10;
    
    std::vector<std::thread> threads;
    std::atomic<int> successful_queries{0};
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&pool, &successful_queries, queries_per_thread, i]() {
            for (int j = 0; j < queries_per_thread; ++j) {
                std::string query = "SELECT " + std::to_string(i) + " as thread_id, " + 
                                  std::to_string(j) + " as query_id";
                auto result = pool.execute_query(query, 1000);
                if (result.success) {
                    successful_queries.fetch_add(1);
                }
                
                // Small delay to simulate work
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    std::cout << "Concurrent test completed in " << duration.count() << "ms" << std::endl;
    std::cout << "Successful queries: " << successful_queries.load() << "/" << (num_threads * queries_per_thread) << std::endl;
    
    // Display pool statistics
    auto stats = pool.get_stats();
    std::cout << "\nPool Statistics:" << std::endl;
    std::cout << "  Total connections: " << stats.total_connections << std::endl;
    std::cout << "  Active connections: " << stats.active_connections << std::endl;
    std::cout << "  Idle connections: " << stats.idle_connections << std::endl;
    std::cout << "  Failed connections: " << stats.failed_connections << std::endl;
    std::cout << "  Total queries: " << stats.total_queries << std::endl;
    std::cout << "  Failed queries: " << stats.failed_queries << std::endl;
    std::cout << "  Average query time: " << stats.average_query_time_ms << "ms" << std::endl;
    
    pool.shutdown();
    std::cout << "Connection pool shut down" << std::endl;
}

void demonstrate_query_cache() {
    std::cout << "\n=== Query Cache Demo ===" << std::endl;
    
    // Configure query cache
    QueryCache::Config config;
    config.max_entries = 1000;
    config.default_ttl_seconds = 300; // 5 minutes
    config.enable_compression = false;
    config.max_result_size_bytes = 1024 * 1024; // 1MB
    
    QueryCache cache(config);
    
    // Create some test results
    auto create_result = [](const std::string& value) {
        DatabaseConnector::QueryResult result;
        result.success = true;
        result.affected_rows = 1;
        result.execution_time_ns = 1000000; // 1ms
        result.rows = {{value, "additional_data"}};
        return result;
    };
    
    std::cout << "Testing cache operations..." << std::endl;
    
    // Test cache miss
    std::string query = "SELECT * FROM users WHERE id = ?";
    std::vector<std::string> params = {"123"};
    
    auto cached_result = cache.get(query, params);
    std::cout << "Initial cache lookup: " << (cached_result.has_value() ? "HIT" : "MISS") << std::endl;
    
    // Cache the result
    auto result = create_result("user_123_data");
    cache.put(query, params, result);
    std::cout << "Result cached" << std::endl;
    
    // Test cache hit
    cached_result = cache.get(query, params);
    std::cout << "Second cache lookup: " << (cached_result.has_value() ? "HIT" : "MISS") << std::endl;
    
    if (cached_result.has_value()) {
        std::cout << "Cached data: " << cached_result->rows[0][0] << std::endl;
    }
    
    // Test different parameters
    std::vector<std::string> different_params = {"456"};
    auto different_result = cache.get(query, different_params);
    std::cout << "Different params lookup: " << (different_result.has_value() ? "HIT" : "MISS") << std::endl;
    
    // Cache multiple entries
    std::cout << "\nCaching multiple entries..." << std::endl;
    for (int i = 0; i < 10; ++i) {
        std::string user_query = "SELECT * FROM users WHERE id = ?";
        std::vector<std::string> user_params = {std::to_string(i)};
        auto user_result = create_result("user_" + std::to_string(i) + "_data");
        cache.put(user_query, user_params, user_result);
    }
    
    // Test cache performance
    std::cout << "\nTesting cache performance..." << std::endl;
    const int num_lookups = 10000;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    int hits = 0;
    int misses = 0;
    
    for (int i = 0; i < num_lookups; ++i) {
        std::string perf_query = "SELECT * FROM users WHERE id = ?";
        std::vector<std::string> perf_params = {std::to_string(i % 20)}; // Some will hit, some will miss
        
        auto perf_result = cache.get(perf_query, perf_params);
        if (perf_result.has_value()) {
            hits++;
        } else {
            misses++;
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    std::cout << "Performance test completed in " << duration.count() << "μs" << std::endl;
    std::cout << "Cache hits: " << hits << std::endl;
    std::cout << "Cache misses: " << misses << std::endl;
    std::cout << "Average lookup time: " << (duration.count() / static_cast<double>(num_lookups)) << "μs" << std::endl;
    
    // Test cache invalidation
    std::cout << "\nTesting cache invalidation..." << std::endl;
    cache.invalidate(".*users.*");
    
    auto invalidated_result = cache.get(query, params);
    std::cout << "After invalidation lookup: " << (invalidated_result.has_value() ? "HIT" : "MISS") << std::endl;
    
    // Display cache statistics
    auto stats = cache.get_stats();
    std::cout << "\nCache Statistics:" << std::endl;
    std::cout << "  Hits: " << stats.hits.load() << std::endl;
    std::cout << "  Misses: " << stats.misses.load() << std::endl;
    std::cout << "  Evictions: " << stats.evictions.load() << std::endl;
    std::cout << "  Invalidations: " << stats.invalidations.load() << std::endl;
    std::cout << "  Current entries: " << stats.current_entries.load() << std::endl;
    std::cout << "  Total size: " << stats.total_size_bytes.load() << " bytes" << std::endl;
    
    if (stats.hits.load() + stats.misses.load() > 0) {
        double hit_ratio = static_cast<double>(stats.hits.load()) / 
                          (stats.hits.load() + stats.misses.load()) * 100.0;
        std::cout << "  Hit ratio: " << hit_ratio << "%" << std::endl;
    }
}

void demonstrate_async_operations() {
    std::cout << "\n=== Async Operations Demo ===" << std::endl;
    
    DatabaseConnector::Config config;
    config.host = "localhost";
    config.port = 5432;
    config.database = "ultra_cpp_demo";
    config.username = "demo_user";
    config.password = "demo_password";
    config.enable_ssl = false;
    
    DatabaseConnector connector(config);
    
    std::cout << "Testing asynchronous query execution..." << std::endl;
    
    // Submit multiple async queries
    std::vector<std::future<DatabaseConnector::QueryResult>> futures;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < 5; ++i) {
        std::string query = "SELECT " + std::to_string(i) + " as query_number, 'async_test' as test_type";
        futures.push_back(connector.execute_query_async(query));
    }
    
    // Wait for all queries to complete
    std::cout << "Waiting for async queries to complete..." << std::endl;
    
    int successful = 0;
    int failed = 0;
    
    for (auto& future : futures) {
        try {
            auto result = future.get();
            if (result.success) {
                successful++;
                std::cout << "Async query completed successfully" << std::endl;
            } else {
                failed++;
                std::cout << "Async query failed: " << result.error_message << std::endl;
            }
        } catch (const std::exception& e) {
            failed++;
            std::cout << "Async query exception: " << e.what() << std::endl;
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    std::cout << "Async operations completed in " << duration.count() << "ms" << std::endl;
    std::cout << "Successful: " << successful << ", Failed: " << failed << std::endl;
}

int main() {
    // Initialize logging
    Logger::initialize(LogLevel::INFO);
    
    std::cout << "Ultra Low-Latency C++ Database Connectivity Demo" << std::endl;
    std::cout << "================================================" << std::endl;
    
    try {
        // Run demonstrations
        demonstrate_basic_connector();
        demonstrate_connection_pool();
        demonstrate_query_cache();
        demonstrate_async_operations();
        
        std::cout << "\n=== Demo Complete ===" << std::endl;
        std::cout << "Note: Some operations may fail if PostgreSQL is not available," << std::endl;
        std::cout << "but this demonstrates the API usage and error handling." << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Demo failed with exception: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}