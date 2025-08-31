#pragma once

#include <atomic>
#include <memory>
#include <thread>
#include <vector>
#include <cstddef>
#include <cstdint>
#ifdef __linux__
#include <numa.h>
#else
// Mock NUMA functions for non-Linux systems
#define numa_available() (-1)
#define numa_alloc_onnode(size, node) malloc(size)
#define numa_node_of_cpu(cpu) (0)
#endif

namespace ultra {
namespace memory {

/**
 * Lock-free memory allocator with thread-local pools
 * Provides high-performance allocation for ultra-low latency scenarios
 */
class LockFreeAllocator {
public:
    static constexpr size_t DEFAULT_POOL_SIZE = 64 * 1024 * 1024; // 64MB
    static constexpr size_t DEFAULT_BLOCK_SIZE = 64; // Cache line aligned
    static constexpr size_t MAX_BLOCK_SIZE = 4096;
    
    struct Config {
        size_t pool_size = DEFAULT_POOL_SIZE;
        size_t min_block_size = DEFAULT_BLOCK_SIZE;
        size_t max_block_size = MAX_BLOCK_SIZE;
        bool numa_aware = true;
        int numa_node = -1; // -1 for auto-detect
    };
    
    explicit LockFreeAllocator(const Config& config = {});
    ~LockFreeAllocator();
    
    // Non-copyable, non-movable
    LockFreeAllocator(const LockFreeAllocator&) = delete;
    LockFreeAllocator& operator=(const LockFreeAllocator&) = delete;
    LockFreeAllocator(LockFreeAllocator&&) = delete;
    LockFreeAllocator& operator=(LockFreeAllocator&&) = delete;
    
    /**
     * Allocate memory block of specified size
     * @param size Size in bytes (will be rounded up to nearest power of 2)
     * @return Pointer to allocated memory or nullptr on failure
     */
    void* allocate(size_t size) noexcept;
    
    /**
     * Deallocate previously allocated memory block
     * @param ptr Pointer to memory block
     * @param size Size of the block (must match allocation size)
     */
    void deallocate(void* ptr, size_t size) noexcept;
    
    /**
     * Get allocation statistics
     */
    struct Stats {
        std::atomic<uint64_t> allocations{0};
        std::atomic<uint64_t> deallocations{0};
        std::atomic<uint64_t> bytes_allocated{0};
        std::atomic<uint64_t> bytes_deallocated{0};
        std::atomic<uint64_t> pool_misses{0};
        std::atomic<uint64_t> numa_local_allocations{0};
    };
    
    const Stats& get_stats() const noexcept { return stats_; }
    
    /**
     * Get the global instance for the current thread
     */
    static LockFreeAllocator& instance();
    
private:
    struct alignas(64) FreeBlock {
        std::atomic<FreeBlock*> next{nullptr};
        size_t size;
    };
    
    struct alignas(64) ThreadLocalPool {
        std::atomic<FreeBlock*> free_lists[16]; // For different size classes
        char* pool_start;
        char* pool_current;
        char* pool_end;
        int numa_node;
        std::atomic<bool> initialized{false};
        
        ThreadLocalPool() {
            for (auto& list : free_lists) {
                list.store(nullptr);
            }
        }
    };
    
    Config config_;
    Stats stats_;
    
    // Thread-local storage for pools
    static thread_local ThreadLocalPool* tl_pool_;
    
    void initialize_thread_pool();
    ThreadLocalPool* get_thread_pool();
    size_t get_size_class(size_t size) const noexcept;
    void* allocate_from_pool(ThreadLocalPool* pool, size_t size) noexcept;
    void* allocate_large(size_t size) noexcept;
    void deallocate_to_pool(ThreadLocalPool* pool, void* ptr, size_t size) noexcept;
    int get_numa_node() const noexcept;
};

/**
 * RAII wrapper for lock-free allocator
 */
template<typename T>
class LockFreeDeleter {
public:
    explicit LockFreeDeleter(LockFreeAllocator& allocator) 
        : allocator_(&allocator) {}
    
    void operator()(T* ptr) const noexcept {
        if (ptr) {
            ptr->~T();
            allocator_->deallocate(ptr, sizeof(T));
        }
    }
    
private:
    LockFreeAllocator* allocator_;
};

/**
 * Unique pointer using lock-free allocator
 */
template<typename T>
using unique_ptr = std::unique_ptr<T, LockFreeDeleter<T>>;

/**
 * Factory function for creating objects with lock-free allocator
 */
template<typename T, typename... Args>
unique_ptr<T> make_unique(LockFreeAllocator& allocator, Args&&... args) {
    void* ptr = allocator.allocate(sizeof(T));
    if (!ptr) {
        throw std::bad_alloc();
    }
    
    try {
        T* obj = new(ptr) T(std::forward<Args>(args)...);
        return unique_ptr<T>(obj, LockFreeDeleter<T>(allocator));
    } catch (...) {
        allocator.deallocate(ptr, sizeof(T));
        throw;
    }
}

} // namespace memory
} // namespace ultra