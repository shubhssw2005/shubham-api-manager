# Troubleshooting Guide

## Overview

This guide provides solutions to common issues encountered when deploying and operating the Ultra Low-Latency C++ System. It covers performance problems, configuration issues, integration challenges, and operational concerns.

## Table of Contents

1. [Performance Issues](#performance-issues)
2. [Memory Problems](#memory-problems)
3. [Network and DPDK Issues](#network-and-dpdk-issues)
4. [Cache Problems](#cache-problems)
5. [Database Connectivity Issues](#database-connectivity-issues)
6. [GPU Compute Problems](#gpu-compute-problems)
7. [Integration Issues](#integration-issues)
8. [Monitoring and Debugging](#monitoring-and-debugging)
9. [Common Error Messages](#common-error-messages)
10. [Performance Regression Analysis](#performance-regression-analysis)

## Performance Issues

### High Latency (P99 > 1ms)

**Symptoms:**
- Response times consistently above target thresholds
- High tail latencies in monitoring dashboards
- Client timeouts and poor user experience

**Diagnostic Steps:**
```bash
# Check CPU affinity and isolation
cat /proc/cmdline | grep isolcpus
taskset -p <pid>

# Monitor CPU utilization per core
htop -d 1

# Check for CPU throttling
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq

# Monitor hardware performance counters
perf stat -e cycles,instructions,cache-references,cache-misses,branch-misses ./ultra-api-gateway

# Check for memory bandwidth saturation
perf stat -e uncore_imc/data_reads/,uncore_imc/data_writes/ ./ultra-api-gateway
```

**Common Causes and Solutions:**

1. **CPU Power Management Enabled**
   ```bash
   # Disable CPU power management
   echo performance > /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
   
   # Disable CPU idle states
   for i in /sys/devices/system/cpu/cpu*/cpuidle/state*/disable; do
       echo 1 > $i
   done
   ```

2. **Incorrect CPU Affinity**
   ```cpp
   // Verify thread affinity in code
   void check_thread_affinity() {
       cpu_set_t cpuset;
       CPU_ZERO(&cpuset);
       
       if (pthread_getaffinity_np(pthread_self(), sizeof(cpuset), &cpuset) == 0) {
           for (int i = 0; i < CPU_SETSIZE; ++i) {
               if (CPU_ISSET(i, &cpuset)) {
                   std::cout << "Thread bound to CPU " << i << std::endl;
               }
           }
       }
   }
   ```

3. **Memory Allocation Issues**
   ```cpp
   // Check for memory allocation hotspots
   class AllocationProfiler {
   public:
       static void* tracked_malloc(size_t size) {
           auto start = std::chrono::high_resolution_clock::now();
           void* ptr = malloc(size);
           auto end = std::chrono::high_resolution_clock::now();
           
           auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
           if (duration.count() > 1000) { // > 1μs
               std::cout << "Slow allocation: " << size << " bytes took " 
                        << duration.count() << "ns" << std::endl;
           }
           
           return ptr;
       }
   };
   ```

### Low Throughput (< 100K QPS)

**Symptoms:**
- Requests per second below expected capacity
- High CPU utilization with low throughput
- Queue buildup in monitoring

**Diagnostic Steps:**
```bash
# Check network interface utilization
sar -n DEV 1

# Monitor network packet drops
netstat -i

# Check for lock contention
perf record -g ./ultra-api-gateway
perf report --stdio

# Monitor context switches
vmstat 1
```

**Solutions:**

1. **Optimize Lock-Free Data Structures**
   ```cpp
   // Replace mutex with atomic operations
   class OptimizedCounter {
   private:
       std::atomic<uint64_t> count_{0};
       
   public:
       void increment() {
           count_.fetch_add(1, std::memory_order_relaxed);
       }
       
       uint64_t get() const {
           return count_.load(std::memory_order_relaxed);
       }
   };
   ```

2. **Reduce Memory Allocations**
   ```cpp
   // Use object pools instead of frequent allocation
   template<typename T>
   class ObjectPool {
   private:
       std::queue<std::unique_ptr<T>> pool_;
       std::mutex mutex_;
       
   public:
       std::unique_ptr<T> acquire() {
           std::lock_guard<std::mutex> lock(mutex_);
           if (pool_.empty()) {
               return std::make_unique<T>();
           }
           
           auto obj = std::move(pool_.front());
           pool_.pop();
           return obj;
       }
       
       void release(std::unique_ptr<T> obj) {
           std::lock_guard<std::mutex> lock(mutex_);
           pool_.push(std::move(obj));
       }
   };
   ```

## Memory Problems

### Memory Leaks

**Symptoms:**
- Continuously increasing memory usage
- Out of memory errors after extended operation
- System becoming unresponsive

**Diagnostic Tools:**
```bash
# Use Valgrind for leak detection
valgrind --tool=memcheck --leak-check=full --show-leak-kinds=all ./ultra-api-gateway

# AddressSanitizer for runtime detection
g++ -fsanitize=address -g -o ultra-api-gateway src/*.cpp

# Monitor memory usage over time
while true; do
    ps -p <pid> -o pid,vsz,rss,pmem,comm
    sleep 10
done
```

**Common Leak Sources:**
```cpp
// Bad: Missing delete
void bad_example() {
    char* buffer = new char[1024];
    // Missing: delete[] buffer;
}

// Good: RAII with smart pointers
void good_example() {
    auto buffer = std::make_unique<char[]>(1024);
    // Automatically cleaned up
}

// Bad: Circular references with shared_ptr
struct Node {
    std::shared_ptr<Node> next;
    std::shared_ptr<Node> prev; // Creates cycle
};

// Good: Break cycles with weak_ptr
struct Node {
    std::shared_ptr<Node> next;
    std::weak_ptr<Node> prev; // Breaks cycle
};
```

### Memory Fragmentation

**Symptoms:**
- Allocation failures despite available memory
- Increasing memory usage without proportional data growth
- Performance degradation over time

**Solutions:**
```cpp
// Use memory pools to reduce fragmentation
class MemoryPool {
private:
    struct Block {
        Block* next;
    };
    
    Block* free_list_;
    size_t block_size_;
    std::vector<std::unique_ptr<char[]>> chunks_;
    
public:
    explicit MemoryPool(size_t block_size, size_t initial_blocks = 1000)
        : free_list_(nullptr), block_size_(block_size) {
        allocate_chunk(initial_blocks);
    }
    
    void* allocate() {
        if (!free_list_) {
            allocate_chunk(1000); // Allocate more blocks
        }
        
        Block* block = free_list_;
        free_list_ = free_list_->next;
        return block;
    }
    
    void deallocate(void* ptr) {
        Block* block = static_cast<Block*>(ptr);
        block->next = free_list_;
        free_list_ = block;
    }
    
private:
    void allocate_chunk(size_t num_blocks) {
        size_t chunk_size = block_size_ * num_blocks;
        auto chunk = std::make_unique<char[]>(chunk_size);
        
        // Link blocks in free list
        for (size_t i = 0; i < num_blocks; ++i) {
            Block* block = reinterpret_cast<Block*>(chunk.get() + i * block_size_);
            block->next = free_list_;
            free_list_ = block;
        }
        
        chunks_.push_back(std::move(chunk));
    }
};
```

### NUMA Issues

**Symptoms:**
- Inconsistent performance across CPU cores
- High memory access latency
- Poor scaling with multiple threads

**Diagnostic and Solutions:**
```bash
# Check NUMA topology
numactl --hardware

# Monitor NUMA memory usage
numastat

# Check for remote memory access
perf stat -e node-loads,node-load-misses,node-stores,node-store-misses ./ultra-api-gateway
```

```cpp
// NUMA-aware memory allocation
class NumaAwareAllocator {
public:
    static void* allocate_local(size_t size) {
        int node = numa_node_of_cpu(sched_getcpu());
        return numa_alloc_onnode(size, node);
    }
    
    static void deallocate(void* ptr, size_t size) {
        numa_free(ptr, size);
    }
    
    // Bind thread to specific NUMA node
    static void bind_to_node(int node) {
        numa_run_on_node(node);
        numa_set_preferred(node);
    }
};
```

## Network and DPDK Issues

### DPDK Initialization Failures

**Symptoms:**
- Application fails to start with DPDK errors
- Network interfaces not detected
- Memory allocation failures

**Common Issues and Solutions:**

1. **Insufficient Huge Pages**
   ```bash
   # Check current huge pages
   cat /proc/meminfo | grep Huge
   
   # Allocate more huge pages
   echo 1024 > /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages
   
   # Mount huge pages filesystem
   mkdir -p /mnt/huge
   mount -t hugetlbfs nodev /mnt/huge
   ```

2. **Network Interface Not Bound**
   ```bash
   # Check current bindings
   dpdk-devbind.py --status
   
   # Bind interface to DPDK
   modprobe uio_pci_generic
   dpdk-devbind.py --bind=uio_pci_generic 0000:01:00.0
   ```

3. **Insufficient Permissions**
   ```bash
   # Run with appropriate permissions
   sudo ./ultra-api-gateway
   
   # Or set up proper user permissions
   sudo usermod -a -G hugetlb $USER
   sudo chown :hugetlb /dev/hugepages
   sudo chmod g+w /dev/hugepages
   ```

### Packet Loss

**Symptoms:**
- High packet drop rates in network statistics
- Incomplete requests or responses
- Client connection timeouts

**Diagnostic Steps:**
```bash
# Check interface statistics
ethtool -S eth0 | grep -i drop

# Monitor ring buffer usage
cat /proc/net/softnet_stat

# Check for buffer overruns
netstat -i
```

**Solutions:**
```cpp
// Increase ring buffer sizes
struct rte_eth_conf port_conf = {};
port_conf.rxmode.max_rx_pkt_len = RTE_ETHER_MAX_LEN;

// Setup larger RX/TX queues
const uint16_t RX_RING_SIZE = 2048; // Increased from 1024
const uint16_t TX_RING_SIZE = 2048;

// Use multiple queues for better performance
const uint16_t nb_rx_queues = 4;
const uint16_t nb_tx_queues = 4;

ret = rte_eth_dev_configure(port_id, nb_rx_queues, nb_tx_queues, &port_conf);
```

## Cache Problems

### Cache Thrashing

**Symptoms:**
- Low cache hit ratios despite warm cache
- High eviction rates
- Inconsistent response times

**Diagnostic Code:**
```cpp
class CacheAnalyzer {
public:
    struct CacheStats {
        uint64_t hits;
        uint64_t misses;
        uint64_t evictions;
        uint64_t size;
        double hit_ratio;
        double eviction_rate;
    };
    
    static void analyze_cache_performance(const UltraCache<std::string, std::string>& cache) {
        auto stats = cache.get_stats();
        
        double hit_ratio = static_cast<double>(stats.hits) / (stats.hits + stats.misses);
        
        std::cout << "Cache Analysis:\n"
                  << "  Hit Ratio: " << (hit_ratio * 100) << "%\n"
                  << "  Evictions: " << stats.evictions << "\n"
                  << "  Size: " << stats.size << "\n";
        
        if (hit_ratio < 0.8) {
            std::cout << "WARNING: Low hit ratio indicates cache thrashing\n";
        }
        
        if (stats.evictions > stats.hits * 0.1) {
            std::cout << "WARNING: High eviction rate\n";
        }
    }
};
```

**Solutions:**
```cpp
// Implement better cache partitioning
template<typename Key, typename Value>
class PartitionedCache {
private:
    std::vector<UltraCache<Key, Value>> partitions_;
    
    size_t get_partition(const Key& key) const {
        return std::hash<Key>{}(key) % partitions_.size();
    }
    
public:
    explicit PartitionedCache(size_t num_partitions, size_t capacity_per_partition) {
        partitions_.reserve(num_partitions);
        for (size_t i = 0; i < num_partitions; ++i) {
            partitions_.emplace_back(UltraCache<Key, Value>({
                .capacity = capacity_per_partition,
                .shard_count = 16
            }));
        }
    }
    
    std::optional<Value> get(const Key& key) {
        return partitions_[get_partition(key)].get(key);
    }
    
    void put(const Key& key, const Value& value) {
        partitions_[get_partition(key)].put(key, value);
    }
};
```

### Cache Corruption

**Symptoms:**
- Incorrect data returned from cache
- Application crashes when accessing cached data
- Data inconsistencies between cache and database

**Detection and Prevention:**
```cpp
class SafeCache {
private:
    UltraCache<std::string, std::string> cache_;
    
    uint32_t calculate_checksum(const std::string& data) const {
        uint32_t checksum = 0;
        for (char c : data) {
            checksum = checksum * 31 + static_cast<uint32_t>(c);
        }
        return checksum;
    }
    
    struct CacheEntry {
        std::string data;
        uint32_t checksum;
        std::chrono::system_clock::time_point timestamp;
    };
    
public:
    void put(const std::string& key, const std::string& value) {
        CacheEntry entry;
        entry.data = value;
        entry.checksum = calculate_checksum(value);
        entry.timestamp = std::chrono::system_clock::now();
        
        // Serialize entry
        std::string serialized = serialize_entry(entry);
        cache_.put(key, serialized);
    }
    
    std::optional<std::string> get(const std::string& key) {
        auto cached = cache_.get(key);
        if (!cached) {
            return std::nullopt;
        }
        
        auto entry = deserialize_entry(*cached);
        
        // Verify checksum
        if (calculate_checksum(entry.data) != entry.checksum) {
            std::cerr << "Cache corruption detected for key: " << key << std::endl;
            cache_.remove(key); // Remove corrupted entry
            return std::nullopt;
        }
        
        return entry.data;
    }
};
```

## Database Connectivity Issues

### Connection Pool Exhaustion

**Symptoms:**
- "No available connections" errors
- Increasing connection wait times
- Database connection timeouts

**Diagnostic Code:**
```cpp
class ConnectionPoolMonitor {
public:
    static void monitor_pool_health(const DatabaseConnector& connector) {
        auto stats = connector.get_pool_stats();
        
        double utilization = static_cast<double>(stats.active_connections) / stats.pool_size;
        
        std::cout << "Connection Pool Stats:\n"
                  << "  Active: " << stats.active_connections << "/" << stats.pool_size << "\n"
                  << "  Utilization: " << (utilization * 100) << "%\n"
                  << "  Wait Queue: " << stats.waiting_requests << "\n";
        
        if (utilization > 0.9) {
            std::cout << "WARNING: Connection pool near exhaustion\n";
        }
        
        if (stats.waiting_requests > 0) {
            std::cout << "WARNING: Requests waiting for connections\n";
        }
    }
};
```

**Solutions:**
```cpp
// Implement connection pool with better management
class ImprovedConnectionPool {
private:
    struct Connection {
        std::unique_ptr<DatabaseConnection> conn;
        std::chrono::steady_clock::time_point last_used;
        bool is_healthy;
    };
    
    std::queue<std::unique_ptr<Connection>> available_connections_;
    std::set<std::unique_ptr<Connection>> active_connections_;
    std::mutex pool_mutex_;
    std::condition_variable pool_cv_;
    
    size_t max_connections_;
    std::chrono::seconds connection_timeout_;
    
public:
    std::unique_ptr<Connection> acquire_connection(std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(pool_mutex_);
        
        // Wait for available connection
        if (!pool_cv_.wait_for(lock, timeout, [this] {
            return !available_connections_.empty() || 
                   active_connections_.size() < max_connections_;
        })) {
            throw std::runtime_error("Connection pool timeout");
        }
        
        if (!available_connections_.empty()) {
            auto conn = std::move(available_connections_.front());
            available_connections_.pop();
            
            // Validate connection health
            if (!conn->is_healthy || is_connection_stale(conn.get())) {
                conn = create_new_connection();
            }
            
            active_connections_.insert(conn.get());
            return conn;
        }
        
        // Create new connection if under limit
        if (active_connections_.size() < max_connections_) {
            auto conn = create_new_connection();
            active_connections_.insert(conn.get());
            return conn;
        }
        
        throw std::runtime_error("Connection pool exhausted");
    }
    
    void release_connection(std::unique_ptr<Connection> conn) {
        std::lock_guard<std::mutex> lock(pool_mutex_);
        
        active_connections_.erase(conn.get());
        
        if (conn->is_healthy) {
            conn->last_used = std::chrono::steady_clock::now();
            available_connections_.push(std::move(conn));
        }
        
        pool_cv_.notify_one();
    }
};
```

### Query Performance Issues

**Symptoms:**
- Slow database query execution
- High database CPU utilization
- Query timeouts

**Diagnostic Steps:**
```sql
-- PostgreSQL query analysis
EXPLAIN (ANALYZE, BUFFERS) SELECT * FROM posts WHERE author_id = $1;

-- Check for missing indexes
SELECT schemaname, tablename, attname, n_distinct, correlation 
FROM pg_stats 
WHERE tablename = 'posts';

-- Monitor slow queries
SELECT query, mean_time, calls, total_time 
FROM pg_stat_statements 
ORDER BY mean_time DESC 
LIMIT 10;
```

**Solutions:**
```cpp
// Implement query optimization
class QueryOptimizer {
public:
    static std::string optimize_user_posts_query(uint32_t user_id, uint32_t limit, uint32_t offset) {
        // Use covering index to avoid table lookup
        return R"(
            SELECT id, title, created_at, 
                   (SELECT content FROM posts p2 WHERE p2.id = p1.id) as content
            FROM posts_index p1 
            WHERE author_id = $1 
            ORDER BY created_at DESC 
            LIMIT $2 OFFSET $3
        )";
    }
    
    // Batch queries to reduce round trips
    static std::string build_batch_query(const std::vector<uint32_t>& post_ids) {
        std::ostringstream query;
        query << "SELECT id, title, content, author_id, created_at FROM posts WHERE id IN (";
        
        for (size_t i = 0; i < post_ids.size(); ++i) {
            if (i > 0) query << ",";
            query << "$" << (i + 1);
        }
        
        query << ") ORDER BY created_at DESC";
        return query.str();
    }
};
```

## GPU Compute Problems

### CUDA Initialization Failures

**Symptoms:**
- "CUDA driver not found" errors
- GPU memory allocation failures
- Kernel launch failures

**Diagnostic Steps:**
```bash
# Check CUDA installation
nvidia-smi

# Verify CUDA driver version
cat /proc/driver/nvidia/version

# Check GPU memory usage
nvidia-smi --query-gpu=memory.used,memory.free --format=csv

# Monitor GPU utilization
nvidia-smi dmon -s pucvmet
```

**Solutions:**
```cpp
class GPUDiagnostics {
public:
    static bool check_cuda_availability() {
        int device_count;
        cudaError_t error = cudaGetDeviceCount(&device_count);
        
        if (error != cudaSuccess) {
            std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
            return false;
        }
        
        if (device_count == 0) {
            std::cerr << "No CUDA devices found" << std::endl;
            return false;
        }
        
        // Check device properties
        for (int i = 0; i < device_count; ++i) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, i);
            
            std::cout << "GPU " << i << ": " << prop.name << "\n"
                      << "  Compute Capability: " << prop.major << "." << prop.minor << "\n"
                      << "  Global Memory: " << (prop.totalGlobalMem / 1024 / 1024) << " MB\n"
                      << "  Multiprocessors: " << prop.multiProcessorCount << "\n";
        }
        
        return true;
    }
    
    static void check_memory_usage() {
        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        
        size_t used_mem = total_mem - free_mem;
        double usage_percent = static_cast<double>(used_mem) / total_mem * 100;
        
        std::cout << "GPU Memory Usage: " << (used_mem / 1024 / 1024) << " MB / "
                  << (total_mem / 1024 / 1024) << " MB (" << usage_percent << "%)\n";
        
        if (usage_percent > 90) {
            std::cout << "WARNING: GPU memory usage is high\n";
        }
    }
};
```

### GPU Memory Leaks

**Symptoms:**
- Gradually increasing GPU memory usage
- "Out of memory" errors after extended operation
- GPU memory not released after operations

**Detection and Prevention:**
```cpp
class GPUMemoryTracker {
private:
    std::unordered_map<void*, size_t> allocations_;
    std::mutex tracker_mutex_;
    std::atomic<size_t> total_allocated_{0};
    
public:
    void* tracked_malloc(size_t size) {
        void* ptr;
        cudaError_t error = cudaMalloc(&ptr, size);
        
        if (error != cudaSuccess) {
            throw std::runtime_error("GPU memory allocation failed: " + 
                                   std::string(cudaGetErrorString(error)));
        }
        
        {
            std::lock_guard<std::mutex> lock(tracker_mutex_);
            allocations_[ptr] = size;
        }
        
        total_allocated_.fetch_add(size);
        return ptr;
    }
    
    void tracked_free(void* ptr) {
        if (!ptr) return;
        
        size_t size = 0;
        {
            std::lock_guard<std::mutex> lock(tracker_mutex_);
            auto it = allocations_.find(ptr);
            if (it != allocations_.end()) {
                size = it->second;
                allocations_.erase(it);
            }
        }
        
        cudaFree(ptr);
        total_allocated_.fetch_sub(size);
    }
    
    void print_leak_report() const {
        std::lock_guard<std::mutex> lock(tracker_mutex_);
        
        if (!allocations_.empty()) {
            std::cout << "GPU Memory Leaks Detected:\n";
            for (const auto& [ptr, size] : allocations_) {
                std::cout << "  " << ptr << ": " << size << " bytes\n";
            }
        }
        
        std::cout << "Total GPU memory allocated: " << total_allocated_.load() << " bytes\n";
    }
};
```

## Integration Issues

### Node.js Communication Failures

**Symptoms:**
- HTTP requests to C++ services timing out
- Connection refused errors
- Inconsistent response formats

**Diagnostic Steps:**
```bash
# Check if C++ service is listening
netstat -tlnp | grep :8080

# Test connectivity
curl -v http://localhost:8080/health

# Monitor network traffic
tcpdump -i lo port 8080
```

**Solutions:**
```cpp
// Implement robust health check endpoint
class HealthCheckHandler {
public:
    struct HealthStatus {
        bool overall_healthy;
        std::map<std::string, bool> component_status;
        std::map<std::string, std::string> metrics;
        std::chrono::system_clock::time_point timestamp;
    };
    
    HealthStatus check_system_health() {
        HealthStatus status;
        status.timestamp = std::chrono::system_clock::now();
        
        // Check cache health
        status.component_status["cache"] = check_cache_health();
        
        // Check database connectivity
        status.component_status["database"] = check_database_health();
        
        // Check GPU availability
        status.component_status["gpu"] = check_gpu_health();
        
        // Check memory usage
        status.component_status["memory"] = check_memory_health();
        
        // Overall health
        status.overall_healthy = std::all_of(
            status.component_status.begin(),
            status.component_status.end(),
            [](const auto& pair) { return pair.second; }
        );
        
        // Add metrics
        status.metrics["uptime_seconds"] = std::to_string(get_uptime_seconds());
        status.metrics["requests_processed"] = std::to_string(get_total_requests());
        status.metrics["avg_latency_us"] = std::to_string(get_avg_latency_us());
        
        return status;
    }
    
private:
    bool check_cache_health() {
        try {
            // Test cache operation
            UltraCache<std::string, std::string> test_cache({.capacity = 10});
            test_cache.put("health_check", "ok");
            auto result = test_cache.get("health_check");
            return result && *result == "ok";
        } catch (...) {
            return false;
        }
    }
    
    bool check_database_health() {
        try {
            // Test database connection
            DatabaseConnector connector;
            auto result = connector.execute_query("SELECT 1", {});
            return !result.rows.empty();
        } catch (...) {
            return false;
        }
    }
    
    bool check_gpu_health() {
        int device_count;
        return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
    }
    
    bool check_memory_health() {
        // Check if memory usage is within acceptable limits
        auto memory_info = get_memory_info();
        return memory_info.usage_percent < 90.0;
    }
};
```

### Data Format Inconsistencies

**Symptoms:**
- JSON parsing errors between C++ and Node.js
- Type conversion failures
- Unexpected null values

**Solutions:**
```cpp
// Implement strict data validation
class DataValidator {
public:
    static bool validate_post_data(const rapidjson::Document& doc) {
        // Required fields
        if (!doc.HasMember("id") || !doc["id"].IsString()) {
            return false;
        }
        
        if (!doc.HasMember("title") || !doc["title"].IsString()) {
            return false;
        }
        
        if (!doc.HasMember("content") || !doc["content"].IsString()) {
            return false;
        }
        
        if (!doc.HasMember("author_id") || !doc["author_id"].IsUint()) {
            return false;
        }
        
        // Optional fields validation
        if (doc.HasMember("created_at") && !doc["created_at"].IsString()) {
            return false;
        }
        
        if (doc.HasMember("tags") && !doc["tags"].IsArray()) {
            return false;
        }
        
        return true;
    }
    
    static std::string sanitize_json_string(const std::string& input) {
        std::string output;
        output.reserve(input.size() * 2); // Reserve space for escaping
        
        for (char c : input) {
            switch (c) {
                case '"':  output += "\\\""; break;
                case '\\': output += "\\\\"; break;
                case '\b': output += "\\b"; break;
                case '\f': output += "\\f"; break;
                case '\n': output += "\\n"; break;
                case '\r': output += "\\r"; break;
                case '\t': output += "\\t"; break;
                default:
                    if (c < 0x20) {
                        output += "\\u" + to_hex(c);
                    } else {
                        output += c;
                    }
                    break;
            }
        }
        
        return output;
    }
};
```

## Monitoring and Debugging

### Performance Regression Detection

**Implementation:**
```cpp
class PerformanceRegression {
private:
    struct Baseline {
        double avg_latency_us;
        double p95_latency_us;
        double p99_latency_us;
        uint64_t throughput_qps;
        std::chrono::system_clock::time_point timestamp;
    };
    
    std::vector<Baseline> baselines_;
    
public:
    void record_baseline(const PerformanceMetrics& metrics) {
        Baseline baseline;
        baseline.avg_latency_us = metrics.avg_latency_us;
        baseline.p95_latency_us = metrics.p95_latency_us;
        baseline.p99_latency_us = metrics.p99_latency_us;
        baseline.throughput_qps = metrics.throughput_qps;
        baseline.timestamp = std::chrono::system_clock::now();
        
        baselines_.push_back(baseline);
        
        // Keep only recent baselines
        if (baselines_.size() > 100) {
            baselines_.erase(baselines_.begin());
        }
    }
    
    bool detect_regression(const PerformanceMetrics& current) {
        if (baselines_.empty()) {
            return false;
        }
        
        // Calculate average of recent baselines
        double avg_baseline_latency = 0;
        uint64_t avg_baseline_throughput = 0;
        
        size_t recent_count = std::min(baselines_.size(), size_t(10));
        for (size_t i = baselines_.size() - recent_count; i < baselines_.size(); ++i) {
            avg_baseline_latency += baselines_[i].avg_latency_us;
            avg_baseline_throughput += baselines_[i].throughput_qps;
        }
        
        avg_baseline_latency /= recent_count;
        avg_baseline_throughput /= recent_count;
        
        // Check for regression (>20% degradation)
        bool latency_regression = current.avg_latency_us > avg_baseline_latency * 1.2;
        bool throughput_regression = current.throughput_qps < avg_baseline_throughput * 0.8;
        
        if (latency_regression || throughput_regression) {
            std::cout << "PERFORMANCE REGRESSION DETECTED:\n"
                      << "  Current latency: " << current.avg_latency_us << "μs "
                      << "(baseline: " << avg_baseline_latency << "μs)\n"
                      << "  Current throughput: " << current.throughput_qps << " QPS "
                      << "(baseline: " << avg_baseline_throughput << " QPS)\n";
            return true;
        }
        
        return false;
    }
};
```

### Debug Logging

**Implementation:**
```cpp
class DebugLogger {
public:
    enum class LogLevel {
        TRACE = 0,
        DEBUG = 1,
        INFO = 2,
        WARN = 3,
        ERROR = 4
    };
    
private:
    LogLevel current_level_;
    std::ofstream log_file_;
    std::mutex log_mutex_;
    
public:
    explicit DebugLogger(LogLevel level = LogLevel::INFO) 
        : current_level_(level) {
        log_file_.open("ultra_cpp_debug.log", std::ios::app);
    }
    
    template<typename... Args>
    void log(LogLevel level, const std::string& format, Args&&... args) {
        if (level < current_level_) {
            return;
        }
        
        std::lock_guard<std::mutex> lock(log_mutex_);
        
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        
        log_file_ << "[" << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S") << "] "
                  << "[" << level_to_string(level) << "] "
                  << format_string(format, std::forward<Args>(args)...) << std::endl;
    }
    
    void trace(const std::string& msg) { log(LogLevel::TRACE, msg); }
    void debug(const std::string& msg) { log(LogLevel::DEBUG, msg); }
    void info(const std::string& msg) { log(LogLevel::INFO, msg); }
    void warn(const std::string& msg) { log(LogLevel::WARN, msg); }
    void error(const std::string& msg) { log(LogLevel::ERROR, msg); }
};

// Global logger instance
extern DebugLogger g_logger;

// Convenience macros
#define LOG_TRACE(msg) g_logger.trace(msg)
#define LOG_DEBUG(msg) g_logger.debug(msg)
#define LOG_INFO(msg) g_logger.info(msg)
#define LOG_WARN(msg) g_logger.warn(msg)
#define LOG_ERROR(msg) g_logger.error(msg)
```

This comprehensive troubleshooting guide provides systematic approaches to diagnosing and resolving common issues in ultra-low latency C++ systems, helping maintain optimal performance and reliability in production environments.