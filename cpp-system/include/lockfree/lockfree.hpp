#pragma once

/**
 * Ultra Low-Latency Lock-Free Data Structures Library
 * 
 * This library provides high-performance, lock-free data structures
 * designed for ultra-low latency applications with sub-millisecond
 * response time requirements.
 * 
 * Features:
 * - Lock-free hash table with linear probing and RCU semantics
 * - MPMC (Multi-Producer Multi-Consumer) ring buffers
 * - Lock-free LRU cache with O(1) operations
 * - Atomic reference counting for safe memory reclamation
 * - Hazard pointer system for epoch-based memory management
 * 
 * All data structures are designed to be:
 * - Thread-safe without locks
 * - Cache-friendly with proper alignment
 * - NUMA-aware where applicable
 * - High-performance with minimal contention
 */

#include "lockfree_hash_table.hpp"
#include "mpmc_ring_buffer.hpp"
#include "lockfree_lru_cache.hpp"
#include "atomic_ref_count.hpp"

namespace ultra_cpp {
namespace lockfree {

/**
 * Library version information
 */
struct Version {
    static constexpr int MAJOR = 1;
    static constexpr int MINOR = 0;
    static constexpr int PATCH = 0;
    
    static constexpr const char* STRING = "1.0.0";
};

/**
 * Performance configuration constants
 */
struct Config {
    // Cache line size for alignment
    static constexpr size_t CACHE_LINE_SIZE = 64;
    
    // Default capacities (all powers of 2)
    static constexpr size_t DEFAULT_HASH_TABLE_CAPACITY = 1024 * 1024;  // 1M entries
    static constexpr size_t DEFAULT_RING_BUFFER_CAPACITY = 1024;        // 1K entries
    static constexpr size_t DEFAULT_LRU_CACHE_CAPACITY = 1024;          // 1K entries
    
    // Memory ordering preferences
    static constexpr std::memory_order DEFAULT_ACQUIRE_ORDER = std::memory_order_acquire;
    static constexpr std::memory_order DEFAULT_RELEASE_ORDER = std::memory_order_release;
    static constexpr std::memory_order DEFAULT_RELAXED_ORDER = std::memory_order_relaxed;
    static constexpr std::memory_order DEFAULT_ACQ_REL_ORDER = std::memory_order_acq_rel;
    
    // Performance tuning
    static constexpr size_t MAX_PROBE_DISTANCE = 64;        // Hash table linear probing limit
    static constexpr size_t SPIN_WAIT_ITERATIONS = 1000;    // Spin before yielding
    static constexpr size_t BACKOFF_MAX_DELAY_US = 100;     // Maximum backoff delay
};

/**
 * Utility functions for performance optimization
 */
namespace utils {

/**
 * CPU pause instruction for spin loops
 */
inline void cpu_pause() noexcept {
#if defined(__x86_64__) || defined(_M_X64)
    __builtin_ia32_pause();
#elif defined(__aarch64__)
    __asm__ volatile("yield" ::: "memory");
#else
    std::this_thread::yield();
#endif
}

/**
 * Exponential backoff for contention handling
 */
class ExponentialBackoff {
public:
    ExponentialBackoff() noexcept : delay_(1) {}
    
    void wait() noexcept {
        for (size_t i = 0; i < delay_; ++i) {
            cpu_pause();
        }
        
        delay_ = std::min(delay_ * 2, Config::BACKOFF_MAX_DELAY_US);
    }
    
    void reset() noexcept {
        delay_ = 1;
    }
    
private:
    size_t delay_;
};

/**
 * Check if a number is a power of 2
 */
constexpr bool is_power_of_2(size_t n) noexcept {
    return n > 0 && (n & (n - 1)) == 0;
}

/**
 * Round up to next power of 2
 */
constexpr size_t next_power_of_2(size_t n) noexcept {
    if (n <= 1) return 1;
    
    size_t result = 1;
    while (result < n) {
        result <<= 1;
    }
    return result;
}

/**
 * Fast modulo for power-of-2 divisors
 */
constexpr size_t fast_modulo(size_t value, size_t divisor) noexcept {
    return value & (divisor - 1);
}

} // namespace utils

/**
 * Type aliases for common use cases
 */
using StringHashTable = HashTable<std::string, std::string>;
using IntHashTable = HashTable<uint64_t, uint64_t>;

using StringRingBuffer = MPMCRingBuffer<std::string>;
using IntRingBuffer = MPMCRingBuffer<uint64_t>;

using StringLRUCache = LRUCache<std::string, std::string>;
using IntLRUCache = LRUCache<uint64_t, uint64_t>;

template<typename T>
using AtomicPtr = AtomicRefCount<T>;

/**
 * Factory functions for creating data structures with optimal configurations
 * Note: Capacity must be specified at compile time as template parameter
 */
namespace factory {

template<typename Key, typename Value, size_t Capacity = Config::DEFAULT_HASH_TABLE_CAPACITY>
auto make_hash_table() {
    static_assert(utils::is_power_of_2(Capacity), "Capacity must be power of 2");
    return std::make_unique<HashTable<Key, Value, Capacity>>();
}

template<typename T, size_t Capacity = Config::DEFAULT_RING_BUFFER_CAPACITY>
auto make_ring_buffer() {
    static_assert(utils::is_power_of_2(Capacity), "Capacity must be power of 2");
    return std::make_unique<MPMCRingBuffer<T, Capacity>>();
}

template<typename Key, typename Value, size_t Capacity = Config::DEFAULT_LRU_CACHE_CAPACITY>
auto make_lru_cache() {
    static_assert(utils::is_power_of_2(Capacity), "Capacity must be power of 2");
    return std::make_unique<LRUCache<Key, Value, Capacity>>();
}

} // namespace factory

} // namespace lockfree
} // namespace ultra_cpp