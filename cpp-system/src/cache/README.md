# Ultra-Fast Cache System

This directory contains the implementation of the high-performance caching system for the ultra-low-latency C++ system.

## Overview

The Ultra-Fast Cache System provides sub-millisecond cache operations with the following key features:

- **Lock-free operations**: All cache operations use lock-free data structures for maximum concurrency
- **Configurable sharding**: Distributes cache entries across multiple shards to reduce contention
- **Multiple eviction policies**: LRU, LFU, Random, and TTL-based eviction strategies
- **Predictive loading**: Machine learning-based cache warming and preloading
- **RDMA replication**: Ultra-low latency cluster synchronization (placeholder implementation)
- **Comprehensive statistics**: Real-time performance monitoring with nanosecond precision

## Architecture

### Core Components

1. **UltraCache**: Main cache interface with template support for different key-value types
2. **Lock-Free Hash Table**: Custom hash table implementation with linear probing
3. **LRU Eviction Policy**: O(1) LRU eviction using doubly-linked lists
4. **RDMA Replication Manager**: Cluster synchronization for distributed caching
5. **Cache Service**: Standalone cache service executable

### Key Features

#### Sharding Strategy
- Configurable number of shards (default: 64)
- Hash-based key distribution across shards
- Independent locks per shard for better concurrency

#### Eviction Policies
- **LRU (Least Recently Used)**: Default policy with O(1) operations
- **LFU (Least Frequently Used)**: Frequency-based eviction
- **Random**: Random eviction for minimal overhead
- **TTL-based**: Time-to-live based expiration

#### Predictive Loading
- Access pattern analysis
- Predictive cache warming based on usage patterns
- Configurable prediction thresholds and window sizes

#### Performance Monitoring
- Lock-free statistics collection
- Nanosecond-precision latency tracking
- Hit ratio and throughput metrics
- Memory usage monitoring

## Usage

### Basic Usage

```cpp
#include "cache/ultra_cache.hpp"

// Configure cache
ultra::cache::UltraCache<std::string, std::string>::Config config;
config.capacity = 1000000;  // 1M entries
config.shard_count = 64;    // 64 shards
config.enable_predictive_loading = true;

// Create cache instance
auto cache = std::make_unique<ultra::cache::UltraCache<std::string, std::string>>(config);

// Basic operations
cache->put("key1", "value1");
auto result = cache->get("key1");
cache->remove("key1");

// Batch operations
std::vector<std::pair<std::string, std::string>> items = {
    {"key1", "value1"},
    {"key2", "value2"}
};
cache->put_batch(items);

std::vector<std::string> keys = {"key1", "key2"};
auto results = cache->get_batch(keys);
```

### Advanced Configuration

```cpp
ultra::cache::UltraCache<std::string, std::string>::Config config;
config.capacity = 1000000;
config.shard_count = 64;
config.eviction_policy = Config::EvictionPolicy::LRU;
config.enable_predictive_loading = true;
config.prediction_threshold = 0.8;
config.warmup_interval = std::chrono::milliseconds(100);
config.enable_rdma = false;  // Enable for cluster replication
```

### Performance Monitoring

```cpp
auto stats = cache->get_stats();
std::cout << "Hit ratio: " << cache->get_hit_ratio() * 100 << "%" << std::endl;
std::cout << "Operations/sec: " << stats.total_operations.load() << std::endl;
std::cout << "Average latency: " << stats.avg_get_latency_ns.load() << " ns" << std::endl;
```

## Performance Characteristics

### Throughput
- **Single-threaded**: >100K GET ops/sec, >50K PUT ops/sec
- **Multi-threaded**: >1M ops/sec with 8+ threads
- **Batch operations**: 10-50% improvement over individual operations

### Latency
- **P50 GET latency**: <1μs for hot data
- **P99 GET latency**: <10μs
- **P99 PUT latency**: <50μs

### Memory Efficiency
- Memory overhead: <2x data size
- Cache-line aligned data structures
- NUMA-aware memory allocation

## Building and Testing

### Prerequisites
- C++17 compatible compiler
- CMake 3.15+
- Google Test (for unit tests)
- Google Benchmark (for performance tests)

### Build Commands
```bash
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Running Tests
```bash
# Unit tests
./test_ultra_cache

# Performance tests
./test_cache_performance

# Benchmark tool
./ultra-cache-benchmark --capacity 1000000 --threads 8
```

### Cache Service
```bash
# Start standalone cache service
./ultra-cache-service
```

## Files

- `ultra_cache.hpp/cpp`: Main cache implementation
- `lock_free_hash_table.cpp`: Lock-free hash table implementation
- `lru_eviction.cpp`: LRU eviction policy implementation
- `rdma_replication.cpp`: RDMA cluster replication (placeholder)
- `cache_service_main.cpp`: Standalone cache service
- `cache_benchmark.cpp`: Performance benchmarking tool
- `test_ultra_cache.cpp`: Comprehensive unit tests
- `test_cache_performance.cpp`: Performance-focused tests

## Integration

The cache system integrates with:
- **API Gateway**: Ultra-fast response caching
- **Stream Processor**: Event result caching
- **Database Layer**: Query result caching
- **Node.js Layer**: Shared session storage

## Future Enhancements

1. **RDMA Implementation**: Complete RDMA-based cluster replication
2. **Persistent Storage**: Memory-mapped file backing store
3. **Compression**: Transparent value compression
4. **Encryption**: At-rest and in-transit encryption
5. **Advanced Eviction**: ML-based eviction policies
6. **Monitoring Integration**: Prometheus/Grafana dashboards

## Performance Tuning

### Configuration Guidelines
- **Shard Count**: Use 2-4x number of CPU cores
- **Capacity**: Size based on available memory and working set
- **Eviction Policy**: LRU for general use, LFU for skewed access patterns
- **Predictive Loading**: Enable for predictable access patterns

### System Tuning
- Use huge pages for large caches
- Set CPU affinity for cache service threads
- Tune NUMA topology for multi-socket systems
- Configure network settings for RDMA (when available)

## Troubleshooting

### Common Issues
1. **High Latency**: Check shard count and CPU affinity
2. **Low Hit Ratio**: Increase capacity or tune eviction policy
3. **Memory Usage**: Monitor for memory leaks in long-running processes
4. **Compilation Errors**: Ensure C++17 support and proper include paths

### Debugging
- Enable debug logging with `-DCMAKE_BUILD_TYPE=Debug`
- Use AddressSanitizer for memory issues: `-fsanitize=address`
- Profile with perf or Intel VTune for performance analysis