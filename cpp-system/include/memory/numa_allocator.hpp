#pragma once

#include <atomic>
#include <memory>
#include <vector>
#include <unordered_map>
#include <mutex>
#ifdef __linux__
#include <numa.h>
#include <numaif.h>
#else
// Mock NUMA functions for non-Linux systems
#define numa_available() (-1)
#define numa_alloc_local(size) malloc(size)
#define numa_alloc_onnode(size, node) malloc(size)
#define numa_alloc_interleaved(size) malloc(size)
#define numa_free(ptr, size) free(ptr)
#define numa_node_of_cpu(cpu) (0)
#define numa_max_node() (0)
#define numa_num_possible_cpus() (1)
#define MPOL_BIND 0
#define MPOL_INTERLEAVE 0
#define MPOL_PREFERRED 0
#define MPOL_F_NODE 0
#define MPOL_F_ADDR 0
#define MPOL_MF_MOVE 0
struct bitmask { unsigned long *maskp; unsigned int size; };
static inline struct bitmask* numa_allocate_nodemask() { return nullptr; }
static inline struct bitmask* numa_allocate_cpumask() { return nullptr; }
static inline void numa_free_nodemask(struct bitmask*) {}
static inline void numa_free_cpumask(struct bitmask*) {}
static inline void numa_bitmask_clearall(struct bitmask*) {}
static inline void numa_bitmask_setbit(struct bitmask*, int) {}
static inline int numa_bitmask_isbitset(struct bitmask*, int) { return 0; }
static inline void numa_bind(struct bitmask*) {}
static inline long long numa_node_size64(int, long long*) { return 1024*1024*1024; }
static inline int numa_distance(int, int) { return 10; }
static inline struct bitmask* numa_get_mems_allowed() { return nullptr; }
static inline void numa_bitmask_free(struct bitmask*) {}
static inline int numa_node_to_cpus(int, struct bitmask*) { return -1; }
static inline long mbind(void*, unsigned long, int, const unsigned long*, unsigned long, unsigned) { return -1; }
static inline int get_mempolicy(int*, unsigned long*, unsigned long, void*, unsigned long) { return -1; }
#endif

namespace ultra {
namespace memory {

/**
 * NUMA-aware memory allocation strategies for multi-socket systems
 * Optimizes memory placement based on CPU topology and access patterns
 */
class NumaAllocator {
public:
    enum class Policy {
        LOCAL,      // Allocate on local NUMA node
        INTERLEAVE, // Interleave across all nodes
        BIND,       // Bind to specific node
        PREFERRED   // Prefer specific node, fallback to others
    };
    
    struct Config {
        Policy default_policy = Policy::LOCAL;
        int preferred_node = -1;
        bool track_allocations = true;
        size_t large_page_threshold = 2 * 1024 * 1024; // 2MB
    };
    
    explicit NumaAllocator(const Config& config = {});
    ~NumaAllocator();
    
    // Non-copyable, non-movable
    NumaAllocator(const NumaAllocator&) = delete;
    NumaAllocator& operator=(const NumaAllocator&) = delete;
    
    /**
     * Allocate NUMA-aware memory
     * @param size Size in bytes
     * @param policy Allocation policy (optional, uses default if not specified)
     * @param node Specific NUMA node for BIND/PREFERRED policies
     * @return Pointer to allocated memory or nullptr on failure
     */
    void* allocate(size_t size, Policy policy = Policy::LOCAL, int node = -1) noexcept;
    
    /**
     * Deallocate NUMA-aware memory
     * @param ptr Pointer to memory block
     * @param size Size of the block
     */
    void deallocate(void* ptr, size_t size) noexcept;
    
    /**
     * Migrate memory to different NUMA node
     * @param ptr Pointer to memory block
     * @param size Size of the block
     * @param target_node Target NUMA node
     * @return true on success, false on failure
     */
    bool migrate_memory(void* ptr, size_t size, int target_node) noexcept;
    
    /**
     * Get memory location information
     * @param ptr Pointer to memory block
     * @param size Size of the block
     * @return NUMA node where memory is located, -1 on error
     */
    int get_memory_node(void* ptr, size_t size) const noexcept;
    
    /**
     * Get current CPU's NUMA node
     */
    static int get_current_cpu_node() noexcept;
    
    /**
     * Get total number of NUMA nodes
     */
    static int get_numa_node_count() noexcept;
    
    /**
     * Get CPUs belonging to a NUMA node
     */
    static std::vector<int> get_node_cpus(int node) noexcept;
    
    /**
     * Bind current thread to specific NUMA node
     */
    static bool bind_to_node(int node) noexcept;
    
    /**
     * NUMA allocation statistics
     */
    struct Stats {
        std::atomic<uint64_t> local_allocations{0};
        std::atomic<uint64_t> remote_allocations{0};
        std::atomic<uint64_t> interleaved_allocations{0};
        std::atomic<uint64_t> migrations{0};
        std::atomic<uint64_t> bytes_allocated{0};
        std::atomic<uint64_t> bytes_deallocated{0};
        std::unordered_map<int, uint64_t> per_node_allocations;
    };
    
    const Stats& get_stats() const noexcept { return stats_; }
    
    /**
     * Print NUMA topology information
     */
    void print_topology() const;
    
private:
    struct AllocationInfo {
        size_t size;
        int node;
        Policy policy;
    };
    
    Config config_;
    mutable Stats stats_;
    std::mutex allocations_mutex_;
    std::unordered_map<void*, AllocationInfo> allocations_;
    
    void* allocate_local(size_t size) noexcept;
    void* allocate_interleaved(size_t size) noexcept;
    void* allocate_on_node(size_t size, int node) noexcept;
    void* allocate_preferred(size_t size, int preferred_node) noexcept;
    
    void track_allocation(void* ptr, size_t size, int node, Policy policy);
    void untrack_allocation(void* ptr);
    
    static bool is_numa_available() noexcept;
};

/**
 * NUMA-aware STL allocator
 */
template<typename T>
class NumaStlAllocator {
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
        using other = NumaStlAllocator<U>;
    };
    
    explicit NumaStlAllocator(NumaAllocator& numa_allocator, 
                             NumaAllocator::Policy policy = NumaAllocator::Policy::LOCAL,
                             int node = -1) noexcept
        : numa_allocator_(&numa_allocator), policy_(policy), node_(node) {}
    
    template<typename U>
    NumaStlAllocator(const NumaStlAllocator<U>& other) noexcept
        : numa_allocator_(other.numa_allocator_), policy_(other.policy_), node_(other.node_) {}
    
    pointer allocate(size_type n) {
        void* ptr = numa_allocator_->allocate(n * sizeof(T), policy_, node_);
        if (!ptr) {
            throw std::bad_alloc();
        }
        return static_cast<pointer>(ptr);
    }
    
    void deallocate(pointer ptr, size_type n) noexcept {
        numa_allocator_->deallocate(ptr, n * sizeof(T));
    }
    
    template<typename U>
    bool operator==(const NumaStlAllocator<U>& other) const noexcept {
        return numa_allocator_ == other.numa_allocator_ && 
               policy_ == other.policy_ && 
               node_ == other.node_;
    }
    
    template<typename U>
    bool operator!=(const NumaStlAllocator<U>& other) const noexcept {
        return !(*this == other);
    }
    
private:
    template<typename U> friend class NumaStlAllocator;
    
    NumaAllocator* numa_allocator_;
    NumaAllocator::Policy policy_;
    int node_;
};

/**
 * NUMA-aware vector
 */
template<typename T>
using numa_vector = std::vector<T, NumaStlAllocator<T>>;

/**
 * Factory functions for NUMA-aware containers
 */
template<typename T>
numa_vector<T> make_numa_vector(NumaAllocator& allocator, 
                               NumaAllocator::Policy policy = NumaAllocator::Policy::LOCAL,
                               int node = -1) {
    return numa_vector<T>(NumaStlAllocator<T>(allocator, policy, node));
}

} // namespace memory
} // namespace ultra