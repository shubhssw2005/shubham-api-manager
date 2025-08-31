#pragma once

/**
 * Ultra Low-Latency Memory Management Library
 * 
 * This header provides access to all memory management components:
 * - Lock-free allocators with thread-local pools
 * - NUMA-aware memory allocation strategies
 * - RCU smart pointers for safe concurrent access
 * - Memory-mapped file I/O with huge pages support
 */

#include "memory/lock_free_allocator.hpp"
#include "memory/numa_allocator.hpp"
#include "memory/rcu_smart_ptr.hpp"
#include "memory/mmap_allocator.hpp"

namespace ultra {
namespace memory {

/**
 * Memory management configuration for the entire system
 */
struct SystemMemoryConfig {
    // Lock-free allocator configuration
    LockFreeAllocator::Config lock_free_config;
    
    // NUMA allocator configuration
    NumaAllocator::Config numa_config;
    
    // Memory mapping configuration
    MmapAllocator::Config mmap_config;
    
    // Global settings
    bool enable_huge_pages = true;
    bool enable_numa_awareness = true;
    bool enable_rcu = true;
    size_t default_alignment = 64; // Cache line alignment
};

/**
 * Global memory manager that coordinates all memory subsystems
 */
class MemoryManager {
public:
    explicit MemoryManager(const SystemMemoryConfig& config = {});
    ~MemoryManager();
    
    // Non-copyable, non-movable
    MemoryManager(const MemoryManager&) = delete;
    MemoryManager& operator=(const MemoryManager&) = delete;
    
    /**
     * Get allocator instances
     */
    LockFreeAllocator& get_lock_free_allocator() { return lock_free_allocator_; }
    NumaAllocator& get_numa_allocator() { return numa_allocator_; }
    RcuManager& get_rcu_manager() { return rcu_manager_; }
    MmapAllocator& get_mmap_allocator() { return mmap_allocator_; }
    
    /**
     * High-level allocation functions that choose the best allocator
     */
    void* allocate(size_t size, size_t alignment = 0) noexcept;
    void deallocate(void* ptr, size_t size) noexcept;
    
    /**
     * Allocate with specific strategy
     */
    void* allocate_numa_local(size_t size) noexcept;
    void* allocate_numa_interleaved(size_t size) noexcept;
    void* allocate_lock_free(size_t size) noexcept;
    
    /**
     * Memory-mapped allocations
     */
    MmapAllocator::MappedFile map_file(const std::string& filename) {
        return mmap_allocator_.map_file(filename);
    }
    
    MmapAllocator::MappedFile create_file(const std::string& filename, size_t size) {
        return mmap_allocator_.create_file(filename, size);
    }
    
    /**
     * RCU operations
     */
    template<typename T>
    RcuPtr<T> make_rcu_ptr(T* ptr) {
        return RcuPtr<T>(ptr);
    }
    
    template<typename T, typename... Args>
    RcuSharedPtr<T> make_rcu_shared(Args&&... args) {
        return ultra::memory::make_rcu_shared<T>(std::forward<Args>(args)...);
    }
    
    /**
     * System-wide memory statistics
     */
    struct SystemStats {
        LockFreeAllocator::Stats lock_free_stats;
        NumaAllocator::Stats numa_stats;
        RcuManager::Stats rcu_stats;
        MmapAllocator::Stats mmap_stats;
        
        uint64_t total_allocations = 0;
        uint64_t total_bytes_allocated = 0;
        uint64_t peak_memory_usage = 0;
    };
    
    SystemStats get_system_stats() const;
    
    /**
     * Memory optimization and maintenance
     */
    void optimize_memory_layout();
    void compact_memory();
    void prefault_memory_pools();
    
    /**
     * Get the global instance
     */
    static MemoryManager& instance();
    
    /**
     * Print comprehensive memory topology and usage information
     */
    void print_memory_info() const;
    
private:
    SystemMemoryConfig config_;
    
    LockFreeAllocator lock_free_allocator_;
    NumaAllocator numa_allocator_;
    RcuManager& rcu_manager_;
    MmapAllocator mmap_allocator_;
    
    // Statistics tracking
    mutable std::atomic<uint64_t> total_allocations_{0};
    mutable std::atomic<uint64_t> total_bytes_allocated_{0};
    mutable std::atomic<uint64_t> peak_memory_usage_{0};
    
    void update_peak_usage(uint64_t current_usage) const;
    size_t choose_allocation_strategy(size_t size, size_t alignment) const;
};

/**
 * RAII memory scope for automatic cleanup
 */
class MemoryScope {
public:
    explicit MemoryScope(MemoryManager& manager = MemoryManager::instance())
        : manager_(manager), rcu_guard_() {}
    
    ~MemoryScope() = default;
    
    // Non-copyable, non-movable
    MemoryScope(const MemoryScope&) = delete;
    MemoryScope& operator=(const MemoryScope&) = delete;
    
    /**
     * Allocate memory within this scope
     */
    template<typename T>
    T* allocate(size_t count = 1) {
        void* ptr = manager_.allocate(sizeof(T) * count, alignof(T));
        if (!ptr) throw std::bad_alloc();
        
        allocated_blocks_.emplace_back(ptr, sizeof(T) * count);
        return static_cast<T*>(ptr);
    }
    
    /**
     * Create object within this scope
     */
    template<typename T, typename... Args>
    T* create(Args&&... args) {
        T* ptr = allocate<T>();
        try {
            new(ptr) T(std::forward<Args>(args)...);
            return ptr;
        } catch (...) {
            // Constructor failed, deallocate
            manager_.deallocate(ptr, sizeof(T));
            allocated_blocks_.pop_back();
            throw;
        }
    }
    
    /**
     * Manual cleanup (automatic on destruction)
     */
    void cleanup() {
        for (auto& [ptr, size] : allocated_blocks_) {
            manager_.deallocate(ptr, size);
        }
        allocated_blocks_.clear();
    }
    
private:
    MemoryManager& manager_;
    RcuReadGuard rcu_guard_;
    std::vector<std::pair<void*, size_t>> allocated_blocks_;
};

/**
 * Convenience macros for memory operations
 */
#define ULTRA_ALLOC(size) ultra::memory::MemoryManager::instance().allocate(size)
#define ULTRA_FREE(ptr, size) ultra::memory::MemoryManager::instance().deallocate(ptr, size)

#define ULTRA_NUMA_LOCAL(size) ultra::memory::MemoryManager::instance().allocate_numa_local(size)
#define ULTRA_NUMA_INTERLEAVED(size) ultra::memory::MemoryManager::instance().allocate_numa_interleaved(size)

#define ULTRA_RCU_READ_LOCK() ultra::memory::RcuReadGuard _rcu_guard
#define ULTRA_RCU_DEFER_DELETE(deleter) ultra::memory::RcuManager::instance().defer_delete(deleter)

/**
 * STL-compatible allocators using ultra memory management
 */
template<typename T>
class UltraAllocator {
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    
    template<typename U>
    struct rebind {
        using other = UltraAllocator<U>;
    };
    
    UltraAllocator() noexcept = default;
    
    template<typename U>
    UltraAllocator(const UltraAllocator<U>&) noexcept {}
    
    pointer allocate(size_type n) {
        void* ptr = MemoryManager::instance().allocate(n * sizeof(T), alignof(T));
        if (!ptr) throw std::bad_alloc();
        return static_cast<pointer>(ptr);
    }
    
    void deallocate(pointer ptr, size_type n) noexcept {
        MemoryManager::instance().deallocate(ptr, n * sizeof(T));
    }
    
    template<typename U>
    bool operator==(const UltraAllocator<U>&) const noexcept { return true; }
    
    template<typename U>
    bool operator!=(const UltraAllocator<U>&) const noexcept { return false; }
};

/**
 * Ultra-optimized containers
 */
template<typename T>
using ultra_vector = std::vector<T, UltraAllocator<T>>;

template<typename Key, typename Value>
using ultra_map = std::map<Key, Value, std::less<Key>, UltraAllocator<std::pair<const Key, Value>>>;

template<typename Key, typename Value>
using ultra_unordered_map = std::unordered_map<Key, Value, std::hash<Key>, std::equal_to<Key>, 
                                              UltraAllocator<std::pair<const Key, Value>>>;

/**
 * Factory functions for ultra containers
 */
template<typename T>
ultra_vector<T> make_ultra_vector() {
    return ultra_vector<T>();
}

template<typename Key, typename Value>
ultra_map<Key, Value> make_ultra_map() {
    return ultra_map<Key, Value>();
}

template<typename Key, typename Value>
ultra_unordered_map<Key, Value> make_ultra_unordered_map() {
    return ultra_unordered_map<Key, Value>();
}

} // namespace memory
} // namespace ultra