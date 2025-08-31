#include "memory/lock_free_allocator.hpp"
#include <sys/mman.h>
#include <unistd.h>
#include <cstring>
#include <algorithm>
#include <new>
#ifdef __linux__
#include <sched.h>
#else
// Mock sched_getcpu for non-Linux systems
static inline int sched_getcpu() { return 0; }
#endif

namespace ultra {
namespace memory {

// Thread-local storage for pools
thread_local LockFreeAllocator::ThreadLocalPool* LockFreeAllocator::tl_pool_ = nullptr;

LockFreeAllocator::LockFreeAllocator(const Config& config) 
    : config_(config) {
    // Validate configuration
    if (config_.pool_size < config_.max_block_size * 16) {
        throw std::invalid_argument("Pool size too small for block size");
    }
    
    if (config_.min_block_size < sizeof(FreeBlock)) {
        throw std::invalid_argument("Block size too small for free list management");
    }
    
    // Ensure block sizes are powers of 2
    if ((config_.min_block_size & (config_.min_block_size - 1)) != 0) {
        throw std::invalid_argument("Min block size must be power of 2");
    }
    
    if ((config_.max_block_size & (config_.max_block_size - 1)) != 0) {
        throw std::invalid_argument("Max block size must be power of 2");
    }
}

LockFreeAllocator::~LockFreeAllocator() {
    // Cleanup is handled by thread-local destructors
}

void* LockFreeAllocator::allocate(size_t size) noexcept {
    if (size == 0) return nullptr;
    
    // Round up to nearest power of 2, minimum block size
    size = std::max(size, config_.min_block_size);
    if (size > config_.max_block_size) {
        return allocate_large(size);
    }
    
    // Get or create thread-local pool
    ThreadLocalPool* pool = get_thread_pool();
    if (!pool) return nullptr;
    
    void* ptr = allocate_from_pool(pool, size);
    if (ptr) {
        stats_.allocations.fetch_add(1, std::memory_order_relaxed);
        stats_.bytes_allocated.fetch_add(size, std::memory_order_relaxed);
        
        // Check if allocation is NUMA-local
        if (config_.numa_aware && numa_available() >= 0) {
            int current_node = numa_node_of_cpu(sched_getcpu());
            if (current_node == pool->numa_node) {
                stats_.numa_local_allocations.fetch_add(1, std::memory_order_relaxed);
            }
        }
    } else {
        stats_.pool_misses.fetch_add(1, std::memory_order_relaxed);
    }
    
    return ptr;
}

void LockFreeAllocator::deallocate(void* ptr, size_t size) noexcept {
    if (!ptr || size == 0) return;
    
    // Round up size to match allocation
    size = std::max(size, config_.min_block_size);
    if (size > config_.max_block_size) {
        munmap(ptr, size);
        stats_.deallocations.fetch_add(1, std::memory_order_relaxed);
        stats_.bytes_deallocated.fetch_add(size, std::memory_order_relaxed);
        return;
    }
    
    ThreadLocalPool* pool = get_thread_pool();
    if (pool) {
        deallocate_to_pool(pool, ptr, size);
        stats_.deallocations.fetch_add(1, std::memory_order_relaxed);
        stats_.bytes_deallocated.fetch_add(size, std::memory_order_relaxed);
    }
}

LockFreeAllocator& LockFreeAllocator::instance() {
    static LockFreeAllocator instance;
    return instance;
}

void LockFreeAllocator::initialize_thread_pool() {
    if (tl_pool_) return;
    
    // Allocate thread-local pool
    tl_pool_ = new ThreadLocalPool();
    
    // Determine NUMA node
    int numa_node = get_numa_node();
    tl_pool_->numa_node = numa_node;
    
    // Allocate pool memory
    void* pool_mem = nullptr;
    if (config_.numa_aware && numa_available() >= 0 && numa_node >= 0) {
        // NUMA-aware allocation
        pool_mem = numa_alloc_onnode(config_.pool_size, numa_node);
    } else {
        // Regular mmap allocation
        pool_mem = mmap(nullptr, config_.pool_size, 
                       PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    }
    
    if (pool_mem == MAP_FAILED || !pool_mem) {
        delete tl_pool_;
        tl_pool_ = nullptr;
        return;
    }
    
    tl_pool_->pool_start = static_cast<char*>(pool_mem);
    tl_pool_->pool_current = tl_pool_->pool_start;
    tl_pool_->pool_end = tl_pool_->pool_start + config_.pool_size;
    tl_pool_->initialized.store(true, std::memory_order_release);
}

LockFreeAllocator::ThreadLocalPool* LockFreeAllocator::get_thread_pool() {
    if (!tl_pool_ || !tl_pool_->initialized.load(std::memory_order_acquire)) {
        initialize_thread_pool();
    }
    return tl_pool_;
}

size_t LockFreeAllocator::get_size_class(size_t size) const noexcept {
    // Find the size class (power of 2 index)
    size_t class_size = config_.min_block_size;
    size_t class_index = 0;
    
    while (class_size < size && class_index < 15) {
        class_size <<= 1;
        ++class_index;
    }
    
    return class_index;
}

void* LockFreeAllocator::allocate_from_pool(ThreadLocalPool* pool, size_t size) noexcept {
    size_t class_index = get_size_class(size);
    if (class_index >= 16) return nullptr;
    
    // Try to get from free list first
    FreeBlock* block = pool->free_lists[class_index].load(std::memory_order_acquire);
    while (block) {
        FreeBlock* next = block->next.load(std::memory_order_acquire);
        if (pool->free_lists[class_index].compare_exchange_weak(
                block, next, std::memory_order_acq_rel, std::memory_order_acquire)) {
            return block;
        }
    }
    
    // Allocate from pool
    size_t actual_size = config_.min_block_size << class_index;
    char* current = pool->pool_current;
    char* new_current = current + actual_size;
    
    if (new_current > pool->pool_end) {
        return nullptr; // Pool exhausted
    }
    
    // Atomic update of pool current pointer
    if (__sync_bool_compare_and_swap(&pool->pool_current, current, new_current)) {
        return current;
    }
    
    // Retry if another thread updated the pointer
    return allocate_from_pool(pool, size);
}

void* LockFreeAllocator::allocate_large(size_t size) noexcept {
    // Use mmap for large allocations
    void* ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    
    if (ptr == MAP_FAILED) {
        return nullptr;
    }
    
    return ptr;
}

void LockFreeAllocator::deallocate_to_pool(ThreadLocalPool* pool, void* ptr, size_t size) noexcept {
    size_t class_index = get_size_class(size);
    if (class_index >= 16) return;
    
    // Add to free list
    FreeBlock* block = static_cast<FreeBlock*>(ptr);
    block->size = size;
    
    FreeBlock* head = pool->free_lists[class_index].load(std::memory_order_acquire);
    do {
        block->next.store(head, std::memory_order_relaxed);
    } while (!pool->free_lists[class_index].compare_exchange_weak(
                head, block, std::memory_order_acq_rel, std::memory_order_acquire));
}

int LockFreeAllocator::get_numa_node() const noexcept {
    if (!config_.numa_aware) return -1;
    
    if (config_.numa_node >= 0) {
        return config_.numa_node;
    }
    
    // Auto-detect current CPU's NUMA node
    if (numa_available() >= 0) {
        int cpu = sched_getcpu();
        if (cpu >= 0) {
            return numa_node_of_cpu(cpu);
        }
    }
    
    return -1;
}

} // namespace memory
} // namespace ultra