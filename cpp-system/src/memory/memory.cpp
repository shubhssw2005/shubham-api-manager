#include "memory/memory.hpp"
#include <iostream>
#include <iomanip>

namespace ultra {
namespace memory {

MemoryManager::MemoryManager(const SystemMemoryConfig& config)
    : config_(config)
    , lock_free_allocator_(config.lock_free_config)
    , numa_allocator_(config.numa_config)
    , rcu_manager_(RcuManager::instance())
    , mmap_allocator_(config.mmap_config) {
    
    // Initialize subsystems
    if (config_.enable_numa_awareness) {
        numa_allocator_.print_topology();
    }
}

MemoryManager::~MemoryManager() {
    // Cleanup is handled by individual allocator destructors
}

void* MemoryManager::allocate(size_t size, size_t alignment) noexcept {
    if (size == 0) return nullptr;
    
    // Choose allocation strategy based on size and requirements
    size_t strategy = choose_allocation_strategy(size, alignment);
    void* ptr = nullptr;
    
    switch (strategy) {
        case 0: // Lock-free allocator for small, frequent allocations
            ptr = lock_free_allocator_.allocate(size);
            break;
            
        case 1: // NUMA-aware allocator for medium allocations
            ptr = numa_allocator_.allocate(size);
            break;
            
        case 2: // Memory-mapped allocator for large allocations
            {
                auto mapped = mmap_allocator_.map_anonymous(size);
                ptr = mapped.is_valid() ? mapped.data() : nullptr;
                // Note: In a real implementation, we'd need to track the MappedFile
                // This is simplified for demonstration
            }
            break;
            
        default:
            ptr = lock_free_allocator_.allocate(size);
            break;
    }
    
    if (ptr) {
        total_allocations_.fetch_add(1, std::memory_order_relaxed);
        uint64_t new_total = total_bytes_allocated_.fetch_add(size, std::memory_order_relaxed) + size;
        update_peak_usage(new_total);
    }
    
    return ptr;
}

void MemoryManager::deallocate(void* ptr, size_t size) noexcept {
    if (!ptr || size == 0) return;
    
    // For simplicity, try each allocator
    // In a real implementation, we'd track which allocator was used
    
    // Try lock-free allocator first (most common case)
    if (size <= LockFreeAllocator::MAX_BLOCK_SIZE) {
        lock_free_allocator_.deallocate(ptr, size);
    } else {
        // Try NUMA allocator
        numa_allocator_.deallocate(ptr, size);
    }
    
    total_bytes_allocated_.fetch_sub(size, std::memory_order_relaxed);
}

void* MemoryManager::allocate_numa_local(size_t size) noexcept {
    void* ptr = numa_allocator_.allocate(size, NumaAllocator::Policy::LOCAL);
    if (ptr) {
        total_allocations_.fetch_add(1, std::memory_order_relaxed);
        uint64_t new_total = total_bytes_allocated_.fetch_add(size, std::memory_order_relaxed) + size;
        update_peak_usage(new_total);
    }
    return ptr;
}

void* MemoryManager::allocate_numa_interleaved(size_t size) noexcept {
    void* ptr = numa_allocator_.allocate(size, NumaAllocator::Policy::INTERLEAVE);
    if (ptr) {
        total_allocations_.fetch_add(1, std::memory_order_relaxed);
        uint64_t new_total = total_bytes_allocated_.fetch_add(size, std::memory_order_relaxed) + size;
        update_peak_usage(new_total);
    }
    return ptr;
}

void* MemoryManager::allocate_lock_free(size_t size) noexcept {
    void* ptr = lock_free_allocator_.allocate(size);
    if (ptr) {
        total_allocations_.fetch_add(1, std::memory_order_relaxed);
        uint64_t new_total = total_bytes_allocated_.fetch_add(size, std::memory_order_relaxed) + size;
        update_peak_usage(new_total);
    }
    return ptr;
}

MemoryManager::SystemStats MemoryManager::get_system_stats() const {
    SystemStats stats;
    
    stats.lock_free_stats = lock_free_allocator_.get_stats();
    stats.numa_stats = numa_allocator_.get_stats();
    stats.rcu_stats = rcu_manager_.get_stats();
    stats.mmap_stats = mmap_allocator_.get_stats();
    
    stats.total_allocations = total_allocations_.load(std::memory_order_relaxed);
    stats.total_bytes_allocated = total_bytes_allocated_.load(std::memory_order_relaxed);
    stats.peak_memory_usage = peak_memory_usage_.load(std::memory_order_relaxed);
    
    return stats;
}

void MemoryManager::optimize_memory_layout() {
    // Trigger RCU synchronization to clean up deferred deletions
    rcu_manager_.synchronize();
    
    // Advise kernel about memory usage patterns
    // This would involve calling madvise on mapped regions
    // Implementation depends on specific use cases
}

void MemoryManager::compact_memory() {
    // Force cleanup of free lists and pools
    // This is a simplified implementation
    optimize_memory_layout();
}

void MemoryManager::prefault_memory_pools() {
    // Pre-fault thread-local pools to avoid page faults during allocation
    // This would involve touching pages in the allocator pools
    // Implementation is allocator-specific
}

MemoryManager& MemoryManager::instance() {
    static MemoryManager instance;
    return instance;
}

void MemoryManager::print_memory_info() const {
    auto stats = get_system_stats();
    
    std::cout << "\n=== Ultra Memory Manager Statistics ===\n";
    
    // System-wide stats
    std::cout << "System-wide:\n";
    std::cout << "  Total allocations: " << stats.total_allocations << "\n";
    std::cout << "  Total bytes allocated: " << (stats.total_bytes_allocated / (1024 * 1024)) << " MB\n";
    std::cout << "  Peak memory usage: " << (stats.peak_memory_usage / (1024 * 1024)) << " MB\n";
    
    // Lock-free allocator stats
    std::cout << "\nLock-free Allocator:\n";
    std::cout << "  Allocations: " << stats.lock_free_stats.allocations << "\n";
    std::cout << "  Deallocations: " << stats.lock_free_stats.deallocations << "\n";
    std::cout << "  Bytes allocated: " << (stats.lock_free_stats.bytes_allocated / 1024) << " KB\n";
    std::cout << "  Pool misses: " << stats.lock_free_stats.pool_misses << "\n";
    std::cout << "  NUMA local allocations: " << stats.lock_free_stats.numa_local_allocations << "\n";
    
    // NUMA allocator stats
    std::cout << "\nNUMA Allocator:\n";
    std::cout << "  Local allocations: " << stats.numa_stats.local_allocations << "\n";
    std::cout << "  Remote allocations: " << stats.numa_stats.remote_allocations << "\n";
    std::cout << "  Interleaved allocations: " << stats.numa_stats.interleaved_allocations << "\n";
    std::cout << "  Migrations: " << stats.numa_stats.migrations << "\n";
    std::cout << "  Bytes allocated: " << (stats.numa_stats.bytes_allocated / 1024) << " KB\n";
    
    // RCU stats
    std::cout << "\nRCU Manager:\n";
    std::cout << "  Read sections: " << stats.rcu_stats.read_sections << "\n";
    std::cout << "  Deferred deletions: " << stats.rcu_stats.deferred_deletions << "\n";
    std::cout << "  Completed deletions: " << stats.rcu_stats.completed_deletions << "\n";
    std::cout << "  Synchronizations: " << stats.rcu_stats.synchronizations << "\n";
    std::cout << "  Active readers: " << stats.rcu_stats.active_readers << "\n";
    
    // Memory mapping stats
    std::cout << "\nMemory Mapping:\n";
    std::cout << "  Files mapped: " << stats.mmap_stats.files_mapped << "\n";
    std::cout << "  Files unmapped: " << stats.mmap_stats.files_unmapped << "\n";
    std::cout << "  Bytes mapped: " << (stats.mmap_stats.bytes_mapped / (1024 * 1024)) << " MB\n";
    std::cout << "  Huge page allocations: " << stats.mmap_stats.huge_page_allocations << "\n";
    std::cout << "  Sync operations: " << stats.mmap_stats.sync_operations << "\n";
    
    std::cout << "\n";
    
    // Print NUMA topology if available
    if (config_.enable_numa_awareness) {
        numa_allocator_.print_topology();
    }
}

void MemoryManager::update_peak_usage(uint64_t current_usage) const {
    uint64_t current_peak = peak_memory_usage_.load(std::memory_order_relaxed);
    while (current_usage > current_peak) {
        if (peak_memory_usage_.compare_exchange_weak(current_peak, current_usage, 
                                                   std::memory_order_relaxed)) {
            break;
        }
    }
}

size_t MemoryManager::choose_allocation_strategy(size_t size, size_t alignment) const {
    // Strategy selection based on size and alignment requirements
    
    // Large allocations (>1MB) use memory mapping
    if (size > 1024 * 1024) {
        return 2; // Memory mapping
    }
    
    // Medium allocations (4KB-1MB) use NUMA-aware allocator
    if (size > 4096) {
        return 1; // NUMA allocator
    }
    
    // Small allocations (<4KB) use lock-free allocator
    return 0; // Lock-free allocator
}

} // namespace memory
} // namespace ultra