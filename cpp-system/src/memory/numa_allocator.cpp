#include "memory/numa_allocator.hpp"
#include <sys/mman.h>
#include <unistd.h>
#include <iostream>
#include <iomanip>
#include <algorithm>
#ifdef __linux__
#include <sched.h>
#else
// Mock sched_getcpu for non-Linux systems
static inline int sched_getcpu() { return 0; }
#endif

namespace ultra {
namespace memory {

NumaAllocator::NumaAllocator(const Config& config) : config_(config) {
    if (!is_numa_available()) {
        std::cerr << "Warning: NUMA not available on this system\n";
    }
}

NumaAllocator::~NumaAllocator() {
    // Clean up any tracked allocations
    std::lock_guard<std::mutex> lock(allocations_mutex_);
    for (const auto& [ptr, info] : allocations_) {
        if (info.size >= config_.large_page_threshold) {
            munmap(ptr, info.size);
        } else {
            numa_free(ptr, info.size);
        }
    }
}

void* NumaAllocator::allocate(size_t size, Policy policy, int node) noexcept {
    if (size == 0) return nullptr;
    
    void* ptr = nullptr;
    int actual_node = -1;
    
    // Use specified policy or default
    Policy actual_policy = (policy == Policy::LOCAL) ? config_.default_policy : policy;
    
    try {
        switch (actual_policy) {
            case Policy::LOCAL:
                ptr = allocate_local(size);
                actual_node = get_current_cpu_node();
                stats_.local_allocations.fetch_add(1, std::memory_order_relaxed);
                break;
                
            case Policy::INTERLEAVE:
                ptr = allocate_interleaved(size);
                actual_node = -1; // Interleaved across nodes
                stats_.interleaved_allocations.fetch_add(1, std::memory_order_relaxed);
                break;
                
            case Policy::BIND:
                if (node < 0) node = config_.preferred_node;
                if (node < 0) node = get_current_cpu_node();
                ptr = allocate_on_node(size, node);
                actual_node = node;
                stats_.remote_allocations.fetch_add(1, std::memory_order_relaxed);
                break;
                
            case Policy::PREFERRED:
                if (node < 0) node = config_.preferred_node;
                if (node < 0) node = get_current_cpu_node();
                ptr = allocate_preferred(size, node);
                actual_node = node;
                break;
        }
        
        if (ptr && config_.track_allocations) {
            track_allocation(ptr, size, actual_node, actual_policy);
        }
        
        if (ptr) {
            stats_.bytes_allocated.fetch_add(size, std::memory_order_relaxed);
        }
        
    } catch (const std::exception& e) {
        std::cerr << "NUMA allocation failed: " << e.what() << std::endl;
        return nullptr;
    }
    
    return ptr;
}

void NumaAllocator::deallocate(void* ptr, size_t size) noexcept {
    if (!ptr || size == 0) return;
    
    if (config_.track_allocations) {
        untrack_allocation(ptr);
    }
    
    if (size >= config_.large_page_threshold) {
        munmap(ptr, size);
    } else {
        numa_free(ptr, size);
    }
    
    stats_.bytes_deallocated.fetch_add(size, std::memory_order_relaxed);
}

bool NumaAllocator::migrate_memory(void* ptr, size_t size, int target_node) noexcept {
    if (!ptr || size == 0 || target_node < 0) return false;
    
    if (!is_numa_available()) return false;
    
    // Create node mask for target node
    struct bitmask* target_mask = numa_allocate_nodemask();
    if (!target_mask) return false;
    
    numa_bitmask_clearall(target_mask);
    numa_bitmask_setbit(target_mask, target_node);
    
    // Migrate pages
    long result = mbind(ptr, size, MPOL_BIND, target_mask->maskp, target_mask->size, MPOL_MF_MOVE);
    
    numa_free_nodemask(target_mask);
    
    if (result == 0) {
        stats_.migrations.fetch_add(1, std::memory_order_relaxed);
        return true;
    }
    
    return false;
}

int NumaAllocator::get_memory_node(void* ptr, size_t size) const noexcept {
    if (!ptr || size == 0 || !is_numa_available()) return -1;
    
    int node = -1;
    if (get_mempolicy(&node, nullptr, 0, ptr, MPOL_F_NODE | MPOL_F_ADDR) == 0) {
        return node;
    }
    
    return -1;
}

int NumaAllocator::get_current_cpu_node() noexcept {
    if (!is_numa_available()) return -1;
    
    int cpu = sched_getcpu();
    if (cpu < 0) return -1;
    
    return numa_node_of_cpu(cpu);
}

int NumaAllocator::get_numa_node_count() noexcept {
    if (!is_numa_available()) return 1;
    return numa_max_node() + 1;
}

std::vector<int> NumaAllocator::get_node_cpus(int node) noexcept {
    std::vector<int> cpus;
    
    if (!is_numa_available() || node < 0) return cpus;
    
    struct bitmask* cpu_mask = numa_allocate_cpumask();
    if (!cpu_mask) return cpus;
    
    if (numa_node_to_cpus(node, cpu_mask) == 0) {
        for (int cpu = 0; cpu < numa_num_possible_cpus(); ++cpu) {
            if (numa_bitmask_isbitset(cpu_mask, cpu)) {
                cpus.push_back(cpu);
            }
        }
    }
    
    numa_free_cpumask(cpu_mask);
    return cpus;
}

bool NumaAllocator::bind_to_node(int node) noexcept {
    if (!is_numa_available() || node < 0) return false;
    
    struct bitmask* node_mask = numa_allocate_nodemask();
    if (!node_mask) return false;
    
    numa_bitmask_clearall(node_mask);
    numa_bitmask_setbit(node_mask, node);
    
    numa_bind(node_mask);
    numa_free_nodemask(node_mask);
    
    return true;
}

void NumaAllocator::print_topology() const {
    if (!is_numa_available()) {
        std::cout << "NUMA not available on this system\n";
        return;
    }
    
    int num_nodes = get_numa_node_count();
    std::cout << "NUMA Topology:\n";
    std::cout << "Number of NUMA nodes: " << num_nodes << "\n";
    
    for (int node = 0; node < num_nodes; ++node) {
        auto cpus = get_node_cpus(node);
        long long free_memory = numa_node_size64(node, nullptr);
        
        std::cout << "Node " << node << ":\n";
        std::cout << "  CPUs: ";
        for (size_t i = 0; i < cpus.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << cpus[i];
        }
        std::cout << "\n";
        std::cout << "  Memory: " << (free_memory / (1024 * 1024)) << " MB\n";
    }
    
    // Print distance matrix
    std::cout << "\nDistance Matrix:\n";
    std::cout << std::setw(6) << "";
    for (int i = 0; i < num_nodes; ++i) {
        std::cout << std::setw(6) << i;
    }
    std::cout << "\n";
    
    for (int i = 0; i < num_nodes; ++i) {
        std::cout << std::setw(6) << i;
        for (int j = 0; j < num_nodes; ++j) {
            std::cout << std::setw(6) << numa_distance(i, j);
        }
        std::cout << "\n";
    }
}

void* NumaAllocator::allocate_local(size_t size) noexcept {
    if (size >= config_.large_page_threshold) {
        // Use mmap with MAP_HUGETLB for large allocations
        void* ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
        return (ptr == MAP_FAILED) ? nullptr : ptr;
    }
    
    return numa_alloc_local(size);
}

void* NumaAllocator::allocate_interleaved(size_t size) noexcept {
    if (size >= config_.large_page_threshold) {
        void* ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
        if (ptr == MAP_FAILED) return nullptr;
        
        // Set interleave policy for the allocated region
        struct bitmask* all_nodes = numa_get_mems_allowed();
        if (all_nodes) {
            mbind(ptr, size, MPOL_INTERLEAVE, all_nodes->maskp, all_nodes->size, 0);
            numa_bitmask_free(all_nodes);
        }
        
        return ptr;
    }
    
    return numa_alloc_interleaved(size);
}

void* NumaAllocator::allocate_on_node(size_t size, int node) noexcept {
    if (node < 0 || node >= get_numa_node_count()) return nullptr;
    
    if (size >= config_.large_page_threshold) {
        void* ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
        if (ptr == MAP_FAILED) return nullptr;
        
        // Bind to specific node
        struct bitmask* node_mask = numa_allocate_nodemask();
        if (node_mask) {
            numa_bitmask_clearall(node_mask);
            numa_bitmask_setbit(node_mask, node);
            mbind(ptr, size, MPOL_BIND, node_mask->maskp, node_mask->size, 0);
            numa_free_nodemask(node_mask);
        }
        
        return ptr;
    }
    
    return numa_alloc_onnode(size, node);
}

void* NumaAllocator::allocate_preferred(size_t size, int preferred_node) noexcept {
    if (preferred_node < 0 || preferred_node >= get_numa_node_count()) {
        return allocate_local(size);
    }
    
    if (size >= config_.large_page_threshold) {
        void* ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
        if (ptr == MAP_FAILED) return nullptr;
        
        // Set preferred policy
        struct bitmask* node_mask = numa_allocate_nodemask();
        if (node_mask) {
            numa_bitmask_clearall(node_mask);
            numa_bitmask_setbit(node_mask, preferred_node);
            mbind(ptr, size, MPOL_PREFERRED, node_mask->maskp, node_mask->size, 0);
            numa_free_nodemask(node_mask);
        }
        
        return ptr;
    }
    
    // Try preferred node first, fallback to local
    void* ptr = numa_alloc_onnode(size, preferred_node);
    if (!ptr) {
        ptr = numa_alloc_local(size);
    }
    
    return ptr;
}

void NumaAllocator::track_allocation(void* ptr, size_t size, int node, Policy policy) {
    if (!ptr) return;
    
    std::lock_guard<std::mutex> lock(allocations_mutex_);
    allocations_[ptr] = {size, node, policy};
    
    if (node >= 0) {
        stats_.per_node_allocations[node] += size;
    }
}

void NumaAllocator::untrack_allocation(void* ptr) {
    if (!ptr) return;
    
    std::lock_guard<std::mutex> lock(allocations_mutex_);
    auto it = allocations_.find(ptr);
    if (it != allocations_.end()) {
        if (it->second.node >= 0) {
            stats_.per_node_allocations[it->second.node] -= it->second.size;
        }
        allocations_.erase(it);
    }
}

bool NumaAllocator::is_numa_available() noexcept {
    static int available = numa_available();
    return available >= 0;
}

} // namespace memory
} // namespace ultra