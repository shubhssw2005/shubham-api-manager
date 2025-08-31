# Ultra Low-Latency Database Connectivity Layer

This module provides high-performance database connectivity for PostgreSQL with sub-millisecond latency requirements. It implements advanced features like connection pooling, query result caching, and asynchronous I/O using io_uring.

## Features

### 🚀 High-Performance Database Connector
- **Sub-millisecond query execution** for cached and simple queries
- **Prepared statement support** with parameter binding
- **Connection management** with automatic reconnection
- **Transaction support** with ACID guarantees
- **Comprehensive error handling** and recovery

### 🔄 Advanced Connection Pooling
- **Dynamic pool sizing** (min/max connections)
- **Load balancing strategies**: Round-robin, random, least connections
- **Health monitoring** with automatic failover
- **Connection lifecycle management** with idle timeout
- **Thread-safe connection acquisition** and release

### ⚡ Query Result Caching
- **Automatic query result caching** with configurable TTL
- **LRU eviction policy** for memory management
- **Pattern-based invalidation** for cache consistency
- **Compression support** for large result sets
- **Thread-safe cache operations** with shared locks

### 🔧 Asynchronous I/O with io_uring
- **Kernel bypass I/O** for maximum performance
- **Batch operation processing** for efficiency
- **Zero-copy operations** where possible
- **Configurable queue depth** and worker threads
- **Automatic fallback** to synchronous operations

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Application   │    │  Connection     │    │  Query Cache    │
│     Layer       │◄──►│     Pool        │◄──►│                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Database      │    │   io_uring      │    │   Performance   │
│   Connector     │◄──►│   Manager       │◄──►│   Monitor       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │
         ▼
┌─────────────────┐
│   PostgreSQL    │
│   Database      │
└─────────────────┘
```

## Usage Examples

### Basic Database Connection

```cpp
#include "database/database_connector.hpp"

using namespace ultra_cpp::database;

// Configure connection
DatabaseConnector::Config config;
config.host = "localhost";
config.port = 5432;
config.database = "myapp";
config.username = "user";
config.password = "password";

// Create connector
DatabaseConnector connector(config);

// Connect to database
if (connector.connect()) {
    // Execute query
    auto result = connector.execute_query("SELECT * FROM users LIMIT 10");
    
    if (result.success) {
        std::cout << "Query executed in " << result.execution_time_ns / 1000000.0 << "ms" << std::endl;
        
        for (const auto& row : result.rows) {
            for (const auto& cell : row) {
                std::cout << cell << " ";
            }
            std::cout << std::endl;
        }
    }
}
```

### Connection Pool Usage

```cpp
#include "database/database_connector.hpp"

// Configure connection pool
ConnectionPool::Config pool_config;
pool_config.db_config = config; // Use database config from above
pool_config.min_connections = 5;
pool_config.max_connections = 20;

// Create and initialize pool
ConnectionPool pool(pool_config);
if (pool.initialize()) {
    // Execute query through pool
    auto result = pool.execute_query("SELECT COUNT(*) FROM orders");
    
    if (result.success) {
        std::cout << "Result: " << result.rows[0][0] << std::endl;
    }
    
    // Get pool statistics
    auto stats = pool.get_stats();
    std::cout << "Active connections: " << stats.active_connections << std::endl;
    std::cout << "Total queries: " << stats.total_queries << std::endl;
}
```

### Prepared Statements

```cpp
// Prepare statement
bool prepared = connector.prepare_statement(
    "get_user_by_id",
    "SELECT id, name, email FROM users WHERE id = $1",
    {INT4OID}
);

if (prepared) {
    // Execute prepared statement
    auto result = connector.execute_prepared("get_user_by_id", {"123"});
    
    if (result.success && !result.rows.empty()) {
        std::cout << "User: " << result.rows[0][1] << " (" << result.rows[0][2] << ")" << std::endl;
    }
}
```

### Query Caching

```cpp
// Configure cache
QueryCache::Config cache_config;
cache_config.max_entries = 10000;
cache_config.default_ttl_seconds = 300; // 5 minutes

QueryCache cache(cache_config);

// Cache will automatically store and retrieve results
std::string query = "SELECT * FROM products WHERE category = ?";
std::vector<std::string> params = {"electronics"};

// First call - cache miss, executes query
auto result1 = cache.get(query, params);
if (!result1.has_value()) {
    auto db_result = connector.execute_query("SELECT * FROM products WHERE category = 'electronics'");
    cache.put(query, params, db_result);
}

// Second call - cache hit, returns cached result
auto result2 = cache.get(query, params);
if (result2.has_value()) {
    std::cout << "Cache hit! Retrieved " << result2->rows.size() << " rows" << std::endl;
}
```

### Asynchronous Operations

```cpp
// Submit multiple async queries
std::vector<std::future<DatabaseConnector::QueryResult>> futures;

for (int i = 0; i < 10; ++i) {
    std::string query = "SELECT * FROM logs WHERE id = " + std::to_string(i);
    futures.push_back(connector.execute_query_async(query));
}

// Process results as they complete
for (auto& future : futures) {
    auto result = future.get();
    if (result.success) {
        std::cout << "Async query completed in " << result.execution_time_ns / 1000000.0 << "ms" << std::endl;
    }
}
```

## Performance Characteristics

### Latency Targets
- **Cached queries**: < 100 nanoseconds
- **Simple queries**: < 500 microseconds  
- **Complex queries**: < 10 milliseconds
- **Connection acquisition**: < 1 millisecond

### Throughput Targets
- **Queries per second**: 100,000+ QPS per connection
- **Concurrent connections**: 1,000+ per pool
- **Cache hit ratio**: > 90% for typical workloads
- **Memory efficiency**: < 1KB overhead per connection

### Scalability
- **Linear scaling** with CPU cores
- **NUMA-aware** memory allocation
- **Lock-free data structures** for contention-free access
- **Zero-copy operations** where possible

## Configuration

### Database Connection Settings
```ini
[database]
host = localhost
port = 5432
database = myapp
username = dbuser
password = ${DB_PASSWORD}
connection_timeout_ms = 5000
query_timeout_ms = 30000
enable_ssl = true
ssl_mode = require
```

### Connection Pool Settings
```ini
[connection_pool]
min_connections = 5
max_connections = 50
connection_idle_timeout_ms = 300000
health_check_interval_ms = 30000
enable_load_balancing = true
load_balancing_strategy = round_robin
```

### Query Cache Settings
```ini
[query_cache]
max_entries = 10000
default_ttl_seconds = 300
enable_compression = true
max_result_size_bytes = 1048576
```

### io_uring Settings
```ini
[io_uring]
queue_depth = 256
worker_threads = 4
enable_sqpoll = true
enable_iopoll = false
```

## Dependencies

### Required Libraries
- **libpq-dev**: PostgreSQL client library
- **liburing-dev**: Linux io_uring library
- **pthread**: POSIX threads
- **C++20 compiler**: GCC 10+ or Clang 12+

### Installation (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install -y \
    libpq-dev \
    liburing-dev \
    build-essential \
    cmake \
    pkg-config
```

### Installation (CentOS/RHEL)
```bash
sudo yum install -y \
    postgresql-devel \
    liburing-devel \
    gcc-c++ \
    cmake \
    pkgconfig
```

## Building

### CMake Build
```bash
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Build Options
- `BUILD_TESTING=ON`: Enable unit tests
- `ENABLE_ASAN=ON`: Enable AddressSanitizer
- `ENABLE_TSAN=ON`: Enable ThreadSanitizer
- `ENABLE_PROFILING=ON`: Enable performance profiling
- `BUILD_BENCHMARKS=ON`: Build benchmark executables

## Testing

### Unit Tests
```bash
# Run all database tests
make test_database

# Run specific test suites
./tests/database/test_database_connector
./tests/database/test_connection_pool
./tests/database/test_query_cache
```

### Integration Tests
```bash
# Requires PostgreSQL database
export TEST_DATABASE_URL="postgresql://user:pass@localhost:5432/test_db"
make test_database_integration
```

### Performance Tests
```bash
# Run performance benchmarks
make test_database_performance
./benchmarks/database/benchmark_database
```

## Monitoring and Metrics

### Performance Metrics
- Query execution latency (P50, P95, P99, P999)
- Connection pool utilization
- Cache hit/miss ratios
- I/O operation statistics
- Error rates and types

### Health Checks
- Database connectivity status
- Connection pool health
- Cache memory usage
- Query performance degradation

### Prometheus Integration
```cpp
// Metrics are automatically exported in Prometheus format
// Access via HTTP endpoint: http://localhost:9090/metrics

// Example metrics:
// ultra_cpp_db_queries_total{status="success"} 12345
// ultra_cpp_db_query_duration_seconds{quantile="0.95"} 0.001
// ultra_cpp_db_connections_active 15
// ultra_cpp_db_cache_hit_ratio 0.92
```

## Troubleshooting

### Common Issues

#### Connection Failures
```
Error: Failed to connect to database
Solution: Check host, port, credentials, and network connectivity
```

#### High Latency
```
Issue: Query latency > 10ms
Solutions:
- Check database server performance
- Verify network latency
- Review query complexity
- Check connection pool saturation
```

#### Memory Usage
```
Issue: High memory consumption
Solutions:
- Reduce cache size limits
- Check for connection leaks
- Monitor query result sizes
- Enable compression
```

#### Cache Misses
```
Issue: Low cache hit ratio
Solutions:
- Increase cache size
- Adjust TTL settings
- Review query patterns
- Check invalidation logic
```

### Debug Logging
```cpp
// Enable debug logging
Logger::initialize(LogLevel::DEBUG);

// Check connection status
if (!connector.is_connected()) {
    LOG_ERROR("Database connection lost");
}

// Monitor query performance
auto metrics = connector.get_metrics();
LOG_INFO("Average query time: {}ms", 
         metrics.total_execution_time_ns.load() / metrics.queries_executed.load() / 1000000.0);
```

## Security Considerations

### Connection Security
- **SSL/TLS encryption** for data in transit
- **Certificate validation** for server authentication
- **Connection string protection** (avoid hardcoded passwords)
- **Network security** (firewall rules, VPN)

### Query Security
- **Prepared statements** to prevent SQL injection
- **Input validation** and sanitization
- **Query complexity limits** to prevent DoS
- **Audit logging** for security events

### Access Control
- **Database user permissions** (principle of least privilege)
- **Connection limits** per user/application
- **Query timeout enforcement**
- **Resource usage monitoring**

## Performance Tuning

### Database Server Tuning
```sql
-- PostgreSQL configuration for high performance
shared_buffers = 256MB
effective_cache_size = 1GB
work_mem = 4MB
maintenance_work_mem = 64MB
checkpoint_completion_target = 0.9
wal_buffers = 16MB
default_statistics_target = 100
```

### Application Tuning
```cpp
// Optimize connection pool size
config.min_connections = std::thread::hardware_concurrency();
config.max_connections = std::thread::hardware_concurrency() * 4;

// Tune cache settings
cache_config.max_entries = available_memory_mb * 100; // ~10KB per entry
cache_config.default_ttl_seconds = query_frequency_seconds * 10;

// Optimize io_uring settings
io_config.queue_depth = 256; // Balance latency vs throughput
io_config.worker_threads = std::thread::hardware_concurrency() / 2;
```

### System Tuning
```bash
# Linux kernel parameters for high performance
echo 'net.core.rmem_max = 16777216' >> /etc/sysctl.conf
echo 'net.core.wmem_max = 16777216' >> /etc/sysctl.conf
echo 'net.ipv4.tcp_rmem = 4096 87380 16777216' >> /etc/sysctl.conf
echo 'net.ipv4.tcp_wmem = 4096 65536 16777216' >> /etc/sysctl.conf
sysctl -p
```

## License

This database connectivity layer is part of the Ultra Low-Latency C++ System and is licensed under the same terms as the parent project.

## Contributing

Please refer to the main project's contributing guidelines for information on how to contribute to this module.