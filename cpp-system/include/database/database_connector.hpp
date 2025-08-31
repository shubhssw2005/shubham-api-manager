#pragma once

#include <memory>
#include <string>
#include <vector>
#include <functional>
#include <optional>
#include <atomic>
#include <chrono>
#include <unordered_map>
#include <future>
#include <libpq-fe.h>
#include <liburing.h>

namespace ultra_cpp {
namespace database {

// Forward declarations
class ConnectionPool;
class QueryCache;
class IOUringManager;

/**
 * High-performance PostgreSQL connector with prepared statements
 * Supports asynchronous I/O using io_uring for maximum performance
 */
class DatabaseConnector {
public:
    struct Config {
        std::string host = "localhost";
        uint16_t port = 5432;
        std::string database;
        std::string username;
        std::string password;
        uint32_t connection_timeout_ms = 5000;
        uint32_t query_timeout_ms = 30000;
        bool enable_ssl = true;
        std::string ssl_mode = "require";
    };

    struct QueryResult {
        bool success = false;
        std::string error_message;
        std::vector<std::vector<std::string>> rows;
        uint64_t affected_rows = 0;
        uint64_t execution_time_ns = 0;
    };

    struct PreparedStatement {
        std::string name;
        std::string query;
        std::vector<Oid> param_types;
        uint32_t param_count = 0;
    };

    explicit DatabaseConnector(const Config& config);
    ~DatabaseConnector();

    // Connection management
    bool connect();
    void disconnect();
    bool is_connected() const noexcept;
    bool ping();

    // Prepared statement management
    bool prepare_statement(const std::string& name, const std::string& query, 
                          const std::vector<Oid>& param_types = {});
    bool deallocate_statement(const std::string& name);

    // Synchronous query execution
    QueryResult execute_query(const std::string& query);
    QueryResult execute_prepared(const std::string& statement_name, 
                                const std::vector<std::string>& params = {});

    // Asynchronous query execution
    std::future<QueryResult> execute_query_async(const std::string& query);
    std::future<QueryResult> execute_prepared_async(const std::string& statement_name,
                                                   const std::vector<std::string>& params = {});

    // Transaction support
    bool begin_transaction();
    bool commit_transaction();
    bool rollback_transaction();

    // Performance metrics
    struct Metrics {
        std::atomic<uint64_t> queries_executed{0};
        std::atomic<uint64_t> queries_failed{0};
        std::atomic<uint64_t> total_execution_time_ns{0};
        std::atomic<uint64_t> cache_hits{0};
        std::atomic<uint64_t> cache_misses{0};
        std::atomic<uint64_t> connections_created{0};
        std::atomic<uint64_t> connections_failed{0};
    };

    Metrics get_metrics() const noexcept;
    void reset_metrics() noexcept;

private:
    Config config_;
    PGconn* connection_;
    std::unique_ptr<IOUringManager> io_manager_;
    std::unordered_map<std::string, PreparedStatement> prepared_statements_;
    mutable Metrics metrics_;
    std::atomic<bool> connected_{false};
    std::atomic<bool> in_transaction_{false};

    // Internal helper methods
    QueryResult process_result(PGresult* result, uint64_t start_time_ns);
    std::string build_connection_string() const;
    bool validate_connection();
    void handle_connection_error();
};

/**
 * High-performance connection pool with load balancing and failover
 */
class ConnectionPool {
public:
    struct Config {
        DatabaseConnector::Config db_config;
        uint32_t min_connections = 5;
        uint32_t max_connections = 50;
        uint32_t connection_idle_timeout_ms = 300000; // 5 minutes
        uint32_t health_check_interval_ms = 30000;    // 30 seconds
        bool enable_load_balancing = true;
        bool enable_failover = true;
        std::vector<std::string> replica_hosts;
    };

    enum class LoadBalancingStrategy {
        ROUND_ROBIN,
        LEAST_CONNECTIONS,
        RANDOM,
        WEIGHTED_ROUND_ROBIN
    };

    explicit ConnectionPool(const Config& config);
    ~ConnectionPool();

    // Pool management
    bool initialize();
    void shutdown();
    bool is_healthy() const noexcept;

    // Connection acquisition
    class ConnectionHandle {
    public:
        ConnectionHandle(std::shared_ptr<DatabaseConnector> conn, ConnectionPool* pool);
        ~ConnectionHandle();
        
        DatabaseConnector* operator->() const noexcept { return connection_.get(); }
        DatabaseConnector& operator*() const noexcept { return *connection_; }
        bool is_valid() const noexcept { return connection_ && connection_->is_connected(); }

    private:
        std::shared_ptr<DatabaseConnector> connection_;
        ConnectionPool* pool_;
    };

    std::optional<ConnectionHandle> acquire_connection(uint32_t timeout_ms = 5000);
    void release_connection(std::shared_ptr<DatabaseConnector> connection);

    // High-level query interface
    DatabaseConnector::QueryResult execute_query(const std::string& query, uint32_t timeout_ms = 30000);
    DatabaseConnector::QueryResult execute_prepared(const std::string& statement_name,
                                                   const std::vector<std::string>& params = {},
                                                   uint32_t timeout_ms = 30000);

    // Async query interface
    std::future<DatabaseConnector::QueryResult> execute_query_async(const std::string& query);
    std::future<DatabaseConnector::QueryResult> execute_prepared_async(const std::string& statement_name,
                                                                       const std::vector<std::string>& params = {});

    // Pool statistics
    struct PoolStats {
        uint32_t total_connections = 0;
        uint32_t active_connections = 0;
        uint32_t idle_connections = 0;
        uint32_t failed_connections = 0;
        uint64_t total_queries = 0;
        uint64_t failed_queries = 0;
        double average_query_time_ms = 0.0;
        uint32_t queue_depth = 0;
    };

    PoolStats get_stats() const noexcept;
    void set_load_balancing_strategy(LoadBalancingStrategy strategy);

private:
    Config config_;
    std::vector<std::shared_ptr<DatabaseConnector>> connections_;
    std::vector<std::shared_ptr<DatabaseConnector>> idle_connections_;
    std::atomic<uint32_t> next_connection_index_{0};
    LoadBalancingStrategy lb_strategy_{LoadBalancingStrategy::ROUND_ROBIN};
    
    mutable std::mutex pool_mutex_;
    std::condition_variable connection_available_;
    std::atomic<bool> shutdown_requested_{false};
    std::thread health_check_thread_;
    
    mutable PoolStats stats_;

    // Internal methods
    std::shared_ptr<DatabaseConnector> create_connection();
    void health_check_worker();
    std::shared_ptr<DatabaseConnector> select_connection();
    void cleanup_idle_connections();
    bool validate_connection(std::shared_ptr<DatabaseConnector> conn);
};

/**
 * Query result cache with automatic invalidation
 */
class QueryCache {
public:
    struct Config {
        size_t max_entries = 10000;
        uint32_t default_ttl_seconds = 300; // 5 minutes
        bool enable_compression = true;
        size_t max_result_size_bytes = 1024 * 1024; // 1MB
    };

    struct CacheEntry {
        std::string query_hash;
        DatabaseConnector::QueryResult result;
        std::chrono::steady_clock::time_point created_at;
        std::chrono::steady_clock::time_point expires_at;
        uint32_t access_count = 0;
        size_t size_bytes = 0;
    };

    explicit QueryCache(const Config& config);
    ~QueryCache();

    // Cache operations
    std::optional<DatabaseConnector::QueryResult> get(const std::string& query, 
                                                     const std::vector<std::string>& params = {});
    void put(const std::string& query, const std::vector<std::string>& params,
             const DatabaseConnector::QueryResult& result, uint32_t ttl_seconds = 0);
    void invalidate(const std::string& pattern);
    void invalidate_all();

    // Cache management
    void cleanup_expired();
    void set_ttl(const std::string& query, uint32_t ttl_seconds);

    // Statistics
    struct CacheStats {
        std::atomic<uint64_t> hits{0};
        std::atomic<uint64_t> misses{0};
        std::atomic<uint64_t> evictions{0};
        std::atomic<uint64_t> invalidations{0};
        std::atomic<size_t> current_entries{0};
        std::atomic<size_t> total_size_bytes{0};
    };

    CacheStats get_stats() const noexcept;
    void reset_stats() noexcept;

private:
    Config config_;
    std::unordered_map<std::string, CacheEntry> cache_;
    mutable std::shared_mutex cache_mutex_;
    std::thread cleanup_thread_;
    std::atomic<bool> shutdown_requested_{false};
    mutable CacheStats stats_;

    // Internal methods
    std::string generate_cache_key(const std::string& query, const std::vector<std::string>& params);
    void cleanup_worker();
    void evict_lru_entries(size_t count);
    size_t calculate_entry_size(const CacheEntry& entry);
};

/**
 * io_uring based asynchronous I/O manager for high-performance database operations
 */
class IOUringManager {
public:
    struct Config {
        uint32_t queue_depth = 256;
        uint32_t worker_threads = 4;
        bool enable_sqpoll = true;
        bool enable_iopoll = false;
    };

    explicit IOUringManager(const Config& config);
    ~IOUringManager();

    bool initialize();
    void shutdown();

    // Async operation submission
    std::future<DatabaseConnector::QueryResult> submit_query(PGconn* connection, const std::string& query);
    std::future<DatabaseConnector::QueryResult> submit_prepared_query(PGconn* connection, 
                                                                      const std::string& statement_name,
                                                                      const std::vector<std::string>& params);

    // Statistics
    struct IOStats {
        std::atomic<uint64_t> operations_submitted{0};
        std::atomic<uint64_t> operations_completed{0};
        std::atomic<uint64_t> operations_failed{0};
        std::atomic<uint64_t> total_latency_ns{0};
        std::atomic<uint32_t> queue_depth{0};
    };

    IOStats get_stats() const noexcept;

private:
    Config config_;
    struct io_uring ring_;
    std::vector<std::thread> worker_threads_;
    std::atomic<bool> shutdown_requested_{false};
    mutable IOStats stats_;

    // Internal structures
    struct AsyncOperation {
        enum Type { QUERY, PREPARED_QUERY };
        Type type;
        PGconn* connection;
        std::string query_or_statement;
        std::vector<std::string> params;
        std::promise<DatabaseConnector::QueryResult> promise;
        std::chrono::steady_clock::time_point start_time;
    };

    std::queue<std::unique_ptr<AsyncOperation>> pending_operations_;
    std::mutex operations_mutex_;
    std::condition_variable operations_cv_;

    // Worker methods
    void worker_thread();
    void process_completions();
    DatabaseConnector::QueryResult execute_operation(const AsyncOperation& op);
};

} // namespace database
} // namespace ultra_cpp