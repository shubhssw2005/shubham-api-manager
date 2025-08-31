# Performance Tuning Guide

## Overview

This guide provides comprehensive best practices for optimizing the Ultra Low-Latency C++ System to achieve sub-millisecond response times. The recommendations are based on modern hardware capabilities and proven high-frequency trading techniques.

## Table of Contents

1. [Hardware Configuration](#hardware-configuration)
2. [Operating System Tuning](#operating-system-tuning)
3. [Memory Optimization](#memory-optimization)
4. [CPU and Threading](#cpu-and-threading)
5. [Network Optimization](#network-optimization)
6. [Cache Optimization](#cache-optimization)
7. [Compiler Optimizations](#compiler-optimizations)
8. [Profiling and Monitoring](#profiling-and-monitoring)
9. [Common Performance Pitfalls](#common-performance-pitfalls)

## Hardware Configuration

### CPU Selection and Configuration

**Recommended CPU Features:**
- High single-thread performance (>3.5GHz base clock)
- Large L3 cache (>20MB)
- Support for AVX-512 instructions
- Multiple NUMA nodes for scaling

**BIOS Settings:**
```bash
# Disable CPU power management
echo performance > /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Disable CPU idle states for consistent latency
for i in /sys/devices/system/cpu/cpu*/cpuidle/state*/disable; do
    echo 1 > $i
done

# Set CPU affinity for critical processes
taskset -c 0-3 ./ultra-api-gateway
```

**CPU Isolation:**
```bash
# Add to kernel boot parameters
isolcpus=0-7 nohz_full=0-7 rcu_nocbs=0-7

# Verify isolation
cat /proc/cmdline
```

### Memory Configuration

**NUMA Optimization:**
```bash
# Check NUMA topology
numactl --hardware

# Bind process to specific NUMA node
numactl --cpunodebind=0 --membind=0 ./ultra-api-gateway

# Configure huge pages (2MB pages)
echo 1024 > /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages

# Mount huge pages
mkdir -p /mnt/hugepages
mount -t hugetlbfs nodev /mnt/hugepages
```

**Memory Allocation Strategy:**
```cpp
// Use NUMA-aware allocation
class NumaAllocator {
public:
    static void* allocate_on_node(size_t size, int node) {
        void* ptr = numa_alloc_onnode(size, node);
        if (!ptr) {
            throw std::bad_alloc();
        }
        return ptr;
    }
    
    static void deallocate(void* ptr, size_t size) {
        numa_free(ptr, size);
    }
};

// Example usage in cache implementation
template<typename T>
class NumaAwareCache {
private:
    struct alignas(64) CacheLine {
        T data;
        std::atomic<uint64_t> version{0};
        char padding[64 - sizeof(T) - sizeof(std::atomic<uint64_t>)];
    };
    
    std::vector<CacheLine*> numa_shards_;
    
public:
    NumaAwareCache(size_t capacity) {
        int num_nodes = numa_num_configured_nodes();
        numa_shards_.resize(num_nodes);
        
        size_t per_node_capacity = capacity / num_nodes;
        for (int node = 0; node < num_nodes; ++node) {
            numa_shards_[node] = static_cast<CacheLine*>(
                NumaAllocator::allocate_on_node(
                    per_node_capacity * sizeof(CacheLine), node));
        }
    }
};
```

### Storage Configuration

**NVMe Optimization:**
```bash
# Set I/O scheduler for NVMe
echo none > /sys/block/nvme0n1/queue/scheduler

# Increase queue depth
echo 32 > /sys/block/nvme0n1/queue/nr_requests

# Disable write barriers for performance (use with caution)
mount -o nobarrier,noatime /dev/nvme0n1p1 /data
```

## Operating System Tuning

### Kernel Parameters

**Critical Kernel Settings:**
```bash
# /etc/sysctl.conf optimizations

# Network performance
net.core.rmem_max = 268435456
net.core.wmem_max = 268435456
net.core.rmem_default = 65536
net.core.wmem_default = 65536
net.ipv4.tcp_rmem = 4096 65536 268435456
net.ipv4.tcp_wmem = 4096 65536 268435456
net.core.netdev_max_backlog = 30000
net.core.netdev_budget = 600

# Memory management
vm.swappiness = 1
vm.dirty_ratio = 15
vm.dirty_background_ratio = 5
vm.vfs_cache_pressure = 50

# Process scheduling
kernel.sched_migration_cost_ns = 5000000
kernel.sched_autogroup_enabled = 0

# Apply settings
sysctl -p
```

**Real-time Scheduling:**
```cpp
#include <sched.h>
#include <sys/mlock.h>

void configure_realtime_thread() {
    // Set real-time priority
    struct sched_param param;
    param.sched_priority = 99;
    
    if (sched_setscheduler(0, SCHED_FIFO, &param) != 0) {
        perror("sched_setscheduler failed");
    }
    
    // Lock memory to prevent swapping
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
        perror("mlockall failed");
    }
    
    // Set CPU affinity
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);  // Bind to CPU 0
    
    if (pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset) != 0) {
        perror("pthread_setaffinity_np failed");
    }
}
```

### Interrupt Handling

**IRQ Affinity Optimization:**
```bash
#!/bin/bash
# Distribute network interrupts across CPUs

# Find network interface IRQs
INTERFACE="eth0"
IRQS=$(grep $INTERFACE /proc/interrupts | awk '{print $1}' | sed 's/://')

# Distribute IRQs across CPUs 4-7 (leaving 0-3 for application)
CPU=4
for IRQ in $IRQS; do
    echo $((1 << $CPU)) > /proc/irq/$IRQ/smp_affinity
    CPU=$(((CPU + 1) % 4 + 4))  # Cycle through CPUs 4-7
done
```

## Memory Optimization

### Cache-Friendly Data Structures

**Memory Layout Optimization:**
```cpp
// Bad: Poor cache locality
struct BadNode {
    int data;
    BadNode* next;
    char padding[52];  // Wastes cache line space
};

// Good: Cache-optimized structure
struct alignas(64) CacheOptimizedNode {
    int data[15];      // Fill most of cache line
    std::atomic<CacheOptimizedNode*> next;
    char padding[4];   // Align to cache line boundary
};

// Array of Structures vs Structure of Arrays
class ParticleSystemAoS {
    struct Particle {
        float x, y, z;     // Position
        float vx, vy, vz;  // Velocity
        float mass;
        int id;
    };
    std::vector<Particle> particles_;
};

// Better for SIMD processing
class ParticleSystemSoA {
    std::vector<float> x_, y_, z_;        // Positions
    std::vector<float> vx_, vy_, vz_;     // Velocities
    std::vector<float> mass_;
    std::vector<int> id_;
};
```

**Memory Pool Implementation:**
```cpp
template<typename T, size_t BlockSize = 4096>
class MemoryPool {
private:
    struct Block {
        alignas(T) char data[BlockSize * sizeof(T)];
        Block* next;
    };
    
    Block* current_block_;
    size_t current_offset_;
    std::vector<std::unique_ptr<Block>> blocks_;
    
public:
    T* allocate() {
        if (current_offset_ >= BlockSize) {
            allocate_new_block();
        }
        
        T* ptr = reinterpret_cast<T*>(
            current_block_->data + current_offset_ * sizeof(T));
        ++current_offset_;
        return ptr;
    }
    
private:
    void allocate_new_block() {
        auto block = std::make_unique<Block>();
        current_block_ = block.get();
        current_offset_ = 0;
        blocks_.push_back(std::move(block));
    }
};
```

### Lock-Free Programming

**Atomic Operations Best Practices:**
```cpp
// Use appropriate memory ordering
class LockFreeCounter {
private:
    std::atomic<uint64_t> count_{0};
    
public:
    // Relaxed ordering for simple counters
    uint64_t increment() {
        return count_.fetch_add(1, std::memory_order_relaxed);
    }
    
    uint64_t get() const {
        return count_.load(std::memory_order_relaxed);
    }
};

// RCU-style lock-free data structure
template<typename T>
class LockFreeList {
private:
    struct Node {
        std::atomic<T> data;
        std::atomic<Node*> next;
        std::atomic<bool> marked_for_deletion{false};
    };
    
    std::atomic<Node*> head_{nullptr};
    
public:
    void insert(const T& value) {
        auto new_node = new Node;
        new_node->data.store(value, std::memory_order_relaxed);
        
        Node* current_head = head_.load(std::memory_order_acquire);
        do {
            new_node->next.store(current_head, std::memory_order_relaxed);
        } while (!head_.compare_exchange_weak(
            current_head, new_node, 
            std::memory_order_release, 
            std::memory_order_relaxed));
    }
    
    bool find(const T& value) {
        Node* current = head_.load(std::memory_order_acquire);
        while (current != nullptr) {
            if (!current->marked_for_deletion.load(std::memory_order_acquire) &&
                current->data.load(std::memory_order_acquire) == value) {
                return true;
            }
            current = current->next.load(std::memory_order_acquire);
        }
        return false;
    }
};
```

## CPU and Threading

### Thread Affinity and Isolation

**CPU Binding Strategy:**
```cpp
class ThreadManager {
public:
    enum class ThreadType {
        NETWORK_IO,
        COMPUTE,
        BACKGROUND
    };
    
    static void bind_thread_to_cpu(ThreadType type) {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        
        switch (type) {
            case ThreadType::NETWORK_IO:
                // Bind network threads to CPUs 0-1
                CPU_SET(get_next_network_cpu(), &cpuset);
                break;
            case ThreadType::COMPUTE:
                // Bind compute threads to CPUs 2-5
                CPU_SET(get_next_compute_cpu(), &cpuset);
                break;
            case ThreadType::BACKGROUND:
                // Bind background threads to CPUs 6-7
                CPU_SET(get_next_background_cpu(), &cpuset);
                break;
        }
        
        pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
    }
    
private:
    static std::atomic<int> network_cpu_counter_;
    static std::atomic<int> compute_cpu_counter_;
    static std::atomic<int> background_cpu_counter_;
    
    static int get_next_network_cpu() {
        return network_cpu_counter_.fetch_add(1) % 2;  // CPUs 0-1
    }
    
    static int get_next_compute_cpu() {
        return 2 + (compute_cpu_counter_.fetch_add(1) % 4);  // CPUs 2-5
    }
    
    static int get_next_background_cpu() {
        return 6 + (background_cpu_counter_.fetch_add(1) % 2);  // CPUs 6-7
    }
};
```

### SIMD Optimization

**Vectorized Operations:**
```cpp
#include <immintrin.h>

class SIMDOperations {
public:
    // Vectorized array addition (AVX-512)
    static void add_arrays_avx512(const float* a, const float* b, 
                                 float* result, size_t size) {
        const size_t simd_size = 16;  // 512 bits / 32 bits per float
        size_t simd_iterations = size / simd_size;
        
        for (size_t i = 0; i < simd_iterations; ++i) {
            __m512 va = _mm512_load_ps(&a[i * simd_size]);
            __m512 vb = _mm512_load_ps(&b[i * simd_size]);
            __m512 vresult = _mm512_add_ps(va, vb);
            _mm512_store_ps(&result[i * simd_size], vresult);
        }
        
        // Handle remaining elements
        for (size_t i = simd_iterations * simd_size; i < size; ++i) {
            result[i] = a[i] + b[i];
        }
    }
    
    // Vectorized string comparison
    static bool compare_strings_simd(const char* str1, const char* str2, size_t len) {
        const size_t simd_size = 32;  // 256 bits
        size_t simd_iterations = len / simd_size;
        
        for (size_t i = 0; i < simd_iterations; ++i) {
            __m256i v1 = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(&str1[i * simd_size]));
            __m256i v2 = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(&str2[i * simd_size]));
            
            __m256i cmp = _mm256_cmpeq_epi8(v1, v2);
            int mask = _mm256_movemask_epi8(cmp);
            
            if (mask != 0xFFFFFFFF) {
                return false;
            }
        }
        
        // Compare remaining bytes
        return std::memcmp(&str1[simd_iterations * simd_size], 
                          &str2[simd_iterations * simd_size], 
                          len - simd_iterations * simd_size) == 0;
    }
};
```

## Network Optimization

### DPDK Configuration

**DPDK Setup and Tuning:**
```bash
#!/bin/bash
# DPDK environment setup

# Bind network interface to DPDK
modprobe uio_pci_generic
echo "8086 1572" > /sys/bus/pci/drivers/uio_pci_generic/new_id

# Find network device PCI address
lspci | grep Ethernet

# Bind device to DPDK (replace with actual PCI address)
dpdk-devbind.py --bind=uio_pci_generic 0000:01:00.0

# Configure huge pages for DPDK
echo 1024 > /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages
mkdir -p /mnt/huge
mount -t hugetlbfs nodev /mnt/huge
```

**DPDK Application Configuration:**
```cpp
#include <rte_eal.h>
#include <rte_ethdev.h>
#include <rte_mbuf.h>

class DPDKNetworkEngine {
private:
    static constexpr uint16_t RX_RING_SIZE = 1024;
    static constexpr uint16_t TX_RING_SIZE = 1024;
    static constexpr uint16_t NUM_MBUFS = 8191;
    static constexpr uint16_t MBUF_CACHE_SIZE = 250;
    
public:
    int initialize_dpdk(int argc, char* argv[]) {
        // Initialize DPDK EAL
        int ret = rte_eal_init(argc, argv);
        if (ret < 0) {
            return -1;
        }
        
        // Create memory pool for packets
        mbuf_pool_ = rte_pktmbuf_pool_create("MBUF_POOL", NUM_MBUFS,
            MBUF_CACHE_SIZE, 0, RTE_MBUF_DEFAULT_BUF_SIZE, rte_socket_id());
        
        if (mbuf_pool_ == nullptr) {
            return -1;
        }
        
        // Configure network port
        return configure_port(0);  // Port 0
    }
    
private:
    struct rte_mempool* mbuf_pool_;
    
    int configure_port(uint16_t port) {
        struct rte_eth_conf port_conf = {};
        port_conf.rxmode.mq_mode = ETH_MQ_RX_RSS;
        port_conf.rx_adv_conf.rss_conf.rss_key = nullptr;
        port_conf.rx_adv_conf.rss_conf.rss_hf = ETH_RSS_IP | ETH_RSS_TCP;
        
        // Configure the Ethernet device
        int ret = rte_eth_dev_configure(port, 1, 1, &port_conf);
        if (ret != 0) {
            return ret;
        }
        
        // Setup RX queue
        ret = rte_eth_rx_queue_setup(port, 0, RX_RING_SIZE,
            rte_eth_dev_socket_id(port), nullptr, mbuf_pool_);
        if (ret < 0) {
            return ret;
        }
        
        // Setup TX queue
        ret = rte_eth_tx_queue_setup(port, 0, TX_RING_SIZE,
            rte_eth_dev_socket_id(port), nullptr);
        if (ret < 0) {
            return ret;
        }
        
        // Start the Ethernet port
        ret = rte_eth_dev_start(port);
        if (ret < 0) {
            return ret;
        }
        
        // Enable promiscuous mode
        rte_eth_promiscuous_enable(port);
        
        return 0;
    }
};
```

### Zero-Copy Networking

**Efficient Packet Processing:**
```cpp
class ZeroCopyHTTPParser {
private:
    struct HTTPRequest {
        const char* method;
        size_t method_len;
        const char* path;
        size_t path_len;
        const char* headers;
        size_t headers_len;
        const char* body;
        size_t body_len;
    };
    
public:
    // Parse HTTP request without copying data
    bool parse_request(const char* buffer, size_t buffer_len, HTTPRequest& req) {
        const char* current = buffer;
        const char* end = buffer + buffer_len;
        
        // Parse method (GET, POST, etc.)
        req.method = current;
        while (current < end && *current != ' ') {
            ++current;
        }
        req.method_len = current - req.method;
        
        if (current >= end || *current != ' ') {
            return false;
        }
        ++current;  // Skip space
        
        // Parse path
        req.path = current;
        while (current < end && *current != ' ') {
            ++current;
        }
        req.path_len = current - req.path;
        
        // Continue parsing headers and body...
        // Implementation details omitted for brevity
        
        return true;
    }
    
    // SIMD-accelerated header parsing
    const char* find_header_simd(const char* headers, size_t len, 
                                const char* header_name) {
        // Use SIMD instructions to quickly scan for header names
        // Implementation would use _mm256_* intrinsics
        return nullptr;  // Placeholder
    }
};
```

## Cache Optimization

### CPU Cache Optimization

**Cache-Aware Algorithms:**
```cpp
// Cache-oblivious matrix multiplication
template<typename T>
void cache_oblivious_multiply(const T* A, const T* B, T* C, 
                             size_t n, size_t m, size_t p) {
    if (n <= 64 && m <= 64 && p <= 64) {
        // Base case: use simple multiplication for small matrices
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < p; ++j) {
                T sum = 0;
                for (size_t k = 0; k < m; ++k) {
                    sum += A[i * m + k] * B[k * p + j];
                }
                C[i * p + j] = sum;
            }
        }
    } else {
        // Recursive case: divide and conquer
        if (n >= m && n >= p) {
            size_t n1 = n / 2;
            cache_oblivious_multiply(A, B, C, n1, m, p);
            cache_oblivious_multiply(A + n1 * m, B, C + n1 * p, n - n1, m, p);
        } else if (m >= p) {
            size_t m1 = m / 2;
            cache_oblivious_multiply(A, B, C, n, m1, p);
            cache_oblivious_multiply(A + m1, B + m1 * p, C, n, m - m1, p);
        } else {
            size_t p1 = p / 2;
            cache_oblivious_multiply(A, B, C, n, m, p1);
            cache_oblivious_multiply(A, B + p1, C + p1, n, m, p - p1);
        }
    }
}
```

**Prefetching Strategies:**
```cpp
class PrefetchOptimizer {
public:
    // Software prefetching for linked list traversal
    template<typename Node>
    static Node* traverse_with_prefetch(Node* head, size_t steps) {
        Node* current = head;
        Node* prefetch_target = head;
        
        // Prefetch several nodes ahead
        for (int i = 0; i < 3 && prefetch_target; ++i) {
            prefetch_target = prefetch_target->next;
        }
        
        for (size_t i = 0; i < steps && current; ++i) {
            // Prefetch next node
            if (prefetch_target) {
                __builtin_prefetch(prefetch_target, 0, 3);  // Read, high locality
                prefetch_target = prefetch_target->next;
            }
            
            current = current->next;
        }
        
        return current;
    }
    
    // Prefetch for array processing
    template<typename T>
    static void process_array_with_prefetch(T* array, size_t size, 
                                          std::function<void(T&)> processor) {
        constexpr size_t PREFETCH_DISTANCE = 64;  // Cache lines ahead
        
        for (size_t i = 0; i < size; ++i) {
            // Prefetch future elements
            if (i + PREFETCH_DISTANCE < size) {
                __builtin_prefetch(&array[i + PREFETCH_DISTANCE], 1, 3);
            }
            
            processor(array[i]);
        }
    }
};
```#
# Compiler Optimizations

### GCC/Clang Optimization Flags

**Production Build Configuration:**
```cmake
# CMakeLists.txt optimizations
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Release build optimizations
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -mtune=native")

# Additional performance flags
set(PERFORMANCE_FLAGS 
    "-ffast-math"           # Aggressive floating-point optimizations
    "-funroll-loops"        # Unroll loops for better performance
    "-finline-functions"    # Inline function calls
    "-fomit-frame-pointer"  # Omit frame pointer for more registers
    "-flto"                 # Link-time optimization
    "-fno-exceptions"       # Disable exceptions if not needed
    "-fno-rtti"            # Disable RTTI if not needed
)

target_compile_options(ultra_cpp_system PRIVATE ${PERFORMANCE_FLAGS})

# Profile-Guided Optimization (PGO)
if(ENABLE_PGO)
    set(PGO_FLAGS "-fprofile-generate")
    target_compile_options(ultra_cpp_system PRIVATE ${PGO_FLAGS})
    target_link_options(ultra_cpp_system PRIVATE ${PGO_FLAGS})
endif()
```

**Profile-Guided Optimization Workflow:**
```bash
#!/bin/bash
# PGO build process

# Step 1: Build with profiling instrumentation
cmake -DCMAKE_BUILD_TYPE=Release -DENABLE_PGO=ON ..
make -j$(nproc)

# Step 2: Run representative workload to collect profile data
./ultra_cpp_system --benchmark-mode --duration=300s

# Step 3: Rebuild with profile data
cmake -DCMAKE_BUILD_TYPE=Release -DENABLE_PGO_USE=ON ..
make -j$(nproc)
```

### Template Optimization

**Compile-Time Computation:**
```cpp
// Constexpr for compile-time calculations
template<size_t N>
constexpr size_t fibonacci() {
    if constexpr (N <= 1) {
        return N;
    } else {
        return fibonacci<N-1>() + fibonacci<N-2>();
    }
}

// Template specialization for performance-critical paths
template<typename T>
class FastHash {
public:
    uint64_t hash(const T& value) const {
        // Generic implementation
        return std::hash<T>{}(value);
    }
};

// Specialized for strings using SIMD
template<>
class FastHash<std::string> {
public:
    uint64_t hash(const std::string& value) const {
        return hash_string_simd(value.data(), value.size());
    }
    
private:
    uint64_t hash_string_simd(const char* data, size_t len) const {
        // SIMD-optimized string hashing
        uint64_t hash = 0x9e3779b9;
        const size_t simd_chunks = len / 32;
        
        for (size_t i = 0; i < simd_chunks; ++i) {
            __m256i chunk = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(data + i * 32));
            
            // Process 32 bytes at once
            hash ^= _mm256_extract_epi64(chunk, 0);
            hash *= 0x9e3779b9;
            hash ^= _mm256_extract_epi64(chunk, 1);
            hash *= 0x9e3779b9;
            hash ^= _mm256_extract_epi64(chunk, 2);
            hash *= 0x9e3779b9;
            hash ^= _mm256_extract_epi64(chunk, 3);
            hash *= 0x9e3779b9;
        }
        
        // Handle remaining bytes
        for (size_t i = simd_chunks * 32; i < len; ++i) {
            hash ^= data[i];
            hash *= 0x9e3779b9;
        }
        
        return hash;
    }
};
```

## Profiling and Monitoring

### Hardware Performance Counters

**PMU (Performance Monitoring Unit) Integration:**
```cpp
#include <linux/perf_event.h>
#include <sys/syscall.h>
#include <unistd.h>

class HardwareProfiler {
private:
    struct PerfCounter {
        int fd;
        uint64_t config;
        std::string name;
    };
    
    std::vector<PerfCounter> counters_;
    
public:
    void setup_counters() {
        add_counter(PERF_COUNT_HW_CPU_CYCLES, "cpu_cycles");
        add_counter(PERF_COUNT_HW_INSTRUCTIONS, "instructions");
        add_counter(PERF_COUNT_HW_CACHE_REFERENCES, "cache_references");
        add_counter(PERF_COUNT_HW_CACHE_MISSES, "cache_misses");
        add_counter(PERF_COUNT_HW_BRANCH_INSTRUCTIONS, "branch_instructions");
        add_counter(PERF_COUNT_HW_BRANCH_MISSES, "branch_misses");
    }
    
    void start_profiling() {
        for (auto& counter : counters_) {
            ioctl(counter.fd, PERF_EVENT_IOC_RESET, 0);
            ioctl(counter.fd, PERF_EVENT_IOC_ENABLE, 0);
        }
    }
    
    void stop_profiling() {
        for (auto& counter : counters_) {
            ioctl(counter.fd, PERF_EVENT_IOC_DISABLE, 0);
        }
    }
    
    std::map<std::string, uint64_t> get_results() {
        std::map<std::string, uint64_t> results;
        
        for (const auto& counter : counters_) {
            uint64_t value;
            read(counter.fd, &value, sizeof(value));
            results[counter.name] = value;
        }
        
        return results;
    }
    
private:
    void add_counter(uint64_t config, const std::string& name) {
        struct perf_event_attr pe = {};
        pe.type = PERF_TYPE_HARDWARE;
        pe.size = sizeof(pe);
        pe.config = config;
        pe.disabled = 1;
        pe.exclude_kernel = 1;
        pe.exclude_hv = 1;
        
        int fd = syscall(__NR_perf_event_open, &pe, 0, -1, -1, 0);
        if (fd >= 0) {
            counters_.push_back({fd, config, name});
        }
    }
};
```

### Latency Measurement

**High-Precision Timing:**
```cpp
class LatencyMeasurement {
private:
    static uint64_t get_cpu_cycles() {
        uint32_t lo, hi;
        __asm__ __volatile__ ("rdtsc" : "=a" (lo), "=d" (hi));
        return ((uint64_t)hi << 32) | lo;
    }
    
    static double get_cpu_frequency() {
        // Calibrate CPU frequency
        auto start_time = std::chrono::high_resolution_clock::now();
        uint64_t start_cycles = get_cpu_cycles();
        
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        auto end_time = std::chrono::high_resolution_clock::now();
        uint64_t end_cycles = get_cpu_cycles();
        
        auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            end_time - start_time).count();
        
        return static_cast<double>(end_cycles - start_cycles) / duration_ns;
    }
    
    static inline double cpu_freq_ghz_;
    
public:
    static void initialize() {
        cpu_freq_ghz_ = get_cpu_frequency();
    }
    
    class Timer {
    private:
        uint64_t start_cycles_;
        
    public:
        Timer() : start_cycles_(get_cpu_cycles()) {}
        
        uint64_t elapsed_ns() const {
            uint64_t end_cycles = get_cpu_cycles();
            return static_cast<uint64_t>(
                (end_cycles - start_cycles_) / cpu_freq_ghz_);
        }
        
        void reset() {
            start_cycles_ = get_cpu_cycles();
        }
    };
    
    // Histogram for latency distribution
    class LatencyHistogram {
    private:
        std::vector<std::atomic<uint64_t>> buckets_;
        std::vector<uint64_t> bucket_boundaries_;
        
    public:
        LatencyHistogram() {
            // Create logarithmic buckets: 1ns, 10ns, 100ns, 1μs, 10μs, etc.
            for (int i = 0; i <= 9; ++i) {  // Up to 1 second
                uint64_t boundary = static_cast<uint64_t>(std::pow(10, i));
                bucket_boundaries_.push_back(boundary);
                buckets_.emplace_back(0);
            }
        }
        
        void record(uint64_t latency_ns) {
            for (size_t i = 0; i < bucket_boundaries_.size(); ++i) {
                if (latency_ns <= bucket_boundaries_[i]) {
                    buckets_[i].fetch_add(1, std::memory_order_relaxed);
                    return;
                }
            }
            // Overflow bucket
            buckets_.back().fetch_add(1, std::memory_order_relaxed);
        }
        
        void print_distribution() const {
            std::cout << "Latency Distribution:\n";
            for (size_t i = 0; i < buckets_.size(); ++i) {
                uint64_t count = buckets_[i].load(std::memory_order_relaxed);
                if (count > 0) {
                    std::cout << "  <= " << bucket_boundaries_[i] << "ns: " 
                              << count << " samples\n";
                }
            }
        }
    };
};
```

### Memory Profiling

**Custom Memory Allocator with Tracking:**
```cpp
class ProfilingAllocator {
private:
    struct AllocationInfo {
        size_t size;
        std::chrono::time_point<std::chrono::high_resolution_clock> timestamp;
        const char* file;
        int line;
    };
    
    std::unordered_map<void*, AllocationInfo> allocations_;
    std::mutex allocations_mutex_;
    std::atomic<size_t> total_allocated_{0};
    std::atomic<size_t> peak_allocated_{0};
    std::atomic<size_t> allocation_count_{0};
    
public:
    void* allocate(size_t size, const char* file = __FILE__, int line = __LINE__) {
        void* ptr = std::aligned_alloc(64, size);  // 64-byte aligned
        if (!ptr) {
            throw std::bad_alloc();
        }
        
        {
            std::lock_guard<std::mutex> lock(allocations_mutex_);
            allocations_[ptr] = {
                size, 
                std::chrono::high_resolution_clock::now(),
                file,
                line
            };
        }
        
        size_t current_total = total_allocated_.fetch_add(size) + size;
        
        // Update peak if necessary
        size_t current_peak = peak_allocated_.load();
        while (current_total > current_peak && 
               !peak_allocated_.compare_exchange_weak(current_peak, current_total)) {
            // Retry if another thread updated peak
        }
        
        allocation_count_.fetch_add(1);
        return ptr;
    }
    
    void deallocate(void* ptr) {
        if (!ptr) return;
        
        size_t size = 0;
        {
            std::lock_guard<std::mutex> lock(allocations_mutex_);
            auto it = allocations_.find(ptr);
            if (it != allocations_.end()) {
                size = it->second.size;
                allocations_.erase(it);
            }
        }
        
        total_allocated_.fetch_sub(size);
        std::free(ptr);
    }
    
    void print_memory_report() const {
        std::lock_guard<std::mutex> lock(allocations_mutex_);
        
        std::cout << "Memory Usage Report:\n";
        std::cout << "  Current allocated: " << total_allocated_.load() << " bytes\n";
        std::cout << "  Peak allocated: " << peak_allocated_.load() << " bytes\n";
        std::cout << "  Total allocations: " << allocation_count_.load() << "\n";
        std::cout << "  Active allocations: " << allocations_.size() << "\n";
        
        if (!allocations_.empty()) {
            std::cout << "\nActive Allocations:\n";
            for (const auto& [ptr, info] : allocations_) {
                auto duration = std::chrono::high_resolution_clock::now() - info.timestamp;
                auto age_ms = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
                
                std::cout << "  " << ptr << ": " << info.size << " bytes, "
                          << "age: " << age_ms << "ms, "
                          << "location: " << info.file << ":" << info.line << "\n";
            }
        }
    }
};

// Global allocator instance
extern ProfilingAllocator g_profiling_allocator;

// Macro for tracked allocation
#define TRACKED_ALLOC(size) g_profiling_allocator.allocate(size, __FILE__, __LINE__)
#define TRACKED_FREE(ptr) g_profiling_allocator.deallocate(ptr)
```

## Common Performance Pitfalls

### Memory-Related Issues

**False Sharing Prevention:**
```cpp
// Bad: False sharing between threads
struct BadCounters {
    std::atomic<uint64_t> counter1{0};
    std::atomic<uint64_t> counter2{0};  // Same cache line as counter1
};

// Good: Prevent false sharing with alignment
struct alignas(64) GoodCounters {
    std::atomic<uint64_t> counter1{0};
    char padding1[64 - sizeof(std::atomic<uint64_t>)];
    std::atomic<uint64_t> counter2{0};
    char padding2[64 - sizeof(std::atomic<uint64_t>)];
};

// Alternative: Use thread-local storage
thread_local uint64_t tls_counter = 0;

void increment_counter() {
    ++tls_counter;  // No contention between threads
}
```

**Memory Access Patterns:**
```cpp
// Bad: Random memory access
void bad_matrix_multiply(const float* A, const float* B, float* C, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            for (size_t k = 0; k < n; ++k) {
                C[i * n + j] += A[i * n + k] * B[k * n + j];  // Poor cache locality for B
            }
        }
    }
}

// Good: Cache-friendly access pattern
void good_matrix_multiply(const float* A, const float* B, float* C, size_t n) {
    // Transpose B for better cache locality
    std::vector<float> B_transposed(n * n);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            B_transposed[j * n + i] = B[i * n + j];
        }
    }
    
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            float sum = 0;
            for (size_t k = 0; k < n; ++k) {
                sum += A[i * n + k] * B_transposed[j * n + k];  // Sequential access
            }
            C[i * n + j] = sum;
        }
    }
}
```

### Synchronization Issues

**Lock Contention Avoidance:**
```cpp
// Bad: High contention on single mutex
class BadCounter {
private:
    std::mutex mutex_;
    uint64_t count_ = 0;
    
public:
    void increment() {
        std::lock_guard<std::mutex> lock(mutex_);
        ++count_;
    }
    
    uint64_t get() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return count_;
    }
};

// Good: Sharded counters to reduce contention
class GoodCounter {
private:
    static constexpr size_t NUM_SHARDS = 64;
    
    struct alignas(64) Shard {
        std::atomic<uint64_t> count{0};
    };
    
    std::array<Shard, NUM_SHARDS> shards_;
    
    size_t get_shard_index() const {
        // Use thread ID to distribute across shards
        static thread_local size_t shard_index = 
            std::hash<std::thread::id>{}(std::this_thread::get_id()) % NUM_SHARDS;
        return shard_index;
    }
    
public:
    void increment() {
        shards_[get_shard_index()].count.fetch_add(1, std::memory_order_relaxed);
    }
    
    uint64_t get() const {
        uint64_t total = 0;
        for (const auto& shard : shards_) {
            total += shard.count.load(std::memory_order_relaxed);
        }
        return total;
    }
};
```

### Branch Prediction Issues

**Branch-Free Programming:**
```cpp
// Bad: Unpredictable branches
int bad_max(int a, int b) {
    if (a > b) {
        return a;
    } else {
        return b;
    }
}

// Good: Branch-free implementation
int good_max(int a, int b) {
    return a ^ ((a ^ b) & -(a < b));
}

// Conditional moves for better performance
template<typename T>
T conditional_move(bool condition, T true_value, T false_value) {
    // Modern compilers will generate conditional move instructions
    return condition ? true_value : false_value;
}

// Lookup table for complex conditions
class BranchFreeLookup {
private:
    static constexpr std::array<int, 256> lookup_table = []() {
        std::array<int, 256> table{};
        for (int i = 0; i < 256; ++i) {
            // Precompute complex function
            table[i] = complex_function(i);
        }
        return table;
    }();
    
    static constexpr int complex_function(int x) {
        // Some complex computation that would involve branches
        return (x * x + 3 * x + 7) % 17;
    }
    
public:
    static int fast_lookup(uint8_t index) {
        return lookup_table[index];
    }
};
```

## Performance Testing and Validation

### Benchmark Framework

**Comprehensive Benchmarking:**
```cpp
#include <benchmark/benchmark.h>

class PerformanceBenchmark {
public:
    // Latency benchmark
    static void BM_CacheGet(benchmark::State& state) {
        UltraCache<std::string, std::string> cache({.capacity = 100000});
        
        // Warm up cache
        for (int i = 0; i < 10000; ++i) {
            cache.put("key" + std::to_string(i), "value" + std::to_string(i));
        }
        
        for (auto _ : state) {
            auto result = cache.get("key5000");
            benchmark::DoNotOptimize(result);
        }
        
        state.SetItemsProcessed(state.iterations());
    }
    
    // Throughput benchmark
    static void BM_StreamProcessing(benchmark::State& state) {
        StreamProcessor processor({});
        processor.start_processing();
        
        std::atomic<uint64_t> processed_count{0};
        processor.subscribe(1, [&](const StreamProcessor::Event& event) {
            processed_count.fetch_add(1, std::memory_order_relaxed);
            benchmark::DoNotOptimize(event);
        });
        
        for (auto _ : state) {
            StreamProcessor::Event event;
            event.timestamp_ns = std::chrono::high_resolution_clock::now()
                               .time_since_epoch().count();
            event.type = 1;
            event.size = 64;
            
            processor.publish(event);
        }
        
        processor.stop_processing();
        state.SetItemsProcessed(processed_count.load());
    }
};

// Register benchmarks
BENCHMARK(PerformanceBenchmark::BM_CacheGet)->Threads(1)->Threads(4)->Threads(8);
BENCHMARK(PerformanceBenchmark::BM_StreamProcessing)->Range(1000, 1000000);

BENCHMARK_MAIN();
```

### Load Testing

**Realistic Load Simulation:**
```cpp
class LoadTester {
private:
    struct TestConfig {
        size_t num_threads = 8;
        std::chrono::seconds duration{60};
        size_t requests_per_second = 100000;
        std::string target_endpoint = "http://localhost:8080/api/posts";
    };
    
public:
    struct LoadTestResults {
        uint64_t total_requests;
        uint64_t successful_requests;
        uint64_t failed_requests;
        double avg_latency_ms;
        double p95_latency_ms;
        double p99_latency_ms;
        double requests_per_second;
    };
    
    LoadTestResults run_load_test(const TestConfig& config) {
        std::vector<std::thread> workers;
        std::atomic<bool> stop_flag{false};
        std::vector<uint64_t> latencies;
        std::mutex latencies_mutex;
        
        std::atomic<uint64_t> total_requests{0};
        std::atomic<uint64_t> successful_requests{0};
        
        // Start worker threads
        for (size_t i = 0; i < config.num_threads; ++i) {
            workers.emplace_back([&, i]() {
                ThreadManager::bind_thread_to_cpu(ThreadManager::ThreadType::COMPUTE);
                
                auto requests_per_thread = config.requests_per_second / config.num_threads;
                auto interval = std::chrono::microseconds(1000000 / requests_per_thread);
                
                while (!stop_flag.load(std::memory_order_relaxed)) {
                    auto start = std::chrono::high_resolution_clock::now();
                    
                    // Simulate HTTP request
                    bool success = simulate_http_request(config.target_endpoint);
                    
                    auto end = std::chrono::high_resolution_clock::now();
                    auto latency_us = std::chrono::duration_cast<std::chrono::microseconds>(
                        end - start).count();
                    
                    {
                        std::lock_guard<std::mutex> lock(latencies_mutex);
                        latencies.push_back(latency_us);
                    }
                    
                    total_requests.fetch_add(1, std::memory_order_relaxed);
                    if (success) {
                        successful_requests.fetch_add(1, std::memory_order_relaxed);
                    }
                    
                    std::this_thread::sleep_for(interval);
                }
            });
        }
        
        // Run for specified duration
        std::this_thread::sleep_for(config.duration);
        stop_flag.store(true, std::memory_order_relaxed);
        
        // Wait for workers to finish
        for (auto& worker : workers) {
            worker.join();
        }
        
        // Calculate statistics
        std::sort(latencies.begin(), latencies.end());
        
        LoadTestResults results;
        results.total_requests = total_requests.load();
        results.successful_requests = successful_requests.load();
        results.failed_requests = results.total_requests - results.successful_requests;
        
        if (!latencies.empty()) {
            results.avg_latency_ms = std::accumulate(latencies.begin(), latencies.end(), 0.0) 
                                   / latencies.size() / 1000.0;
            results.p95_latency_ms = latencies[latencies.size() * 0.95] / 1000.0;
            results.p99_latency_ms = latencies[latencies.size() * 0.99] / 1000.0;
        }
        
        results.requests_per_second = static_cast<double>(results.total_requests) 
                                    / config.duration.count();
        
        return results;
    }
    
private:
    bool simulate_http_request(const std::string& endpoint) {
        // Simulate network request with realistic timing
        std::this_thread::sleep_for(std::chrono::microseconds(100 + rand() % 400));
        return rand() % 100 < 95;  // 95% success rate
    }
};
```

This comprehensive performance tuning guide provides the foundation for achieving ultra-low latency performance in C++ systems. Regular profiling, measurement, and optimization using these techniques will help maintain sub-millisecond response times even under high load conditions.