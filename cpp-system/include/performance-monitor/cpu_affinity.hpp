#pragma once

#include <vector>
#include <string>
#include <memory>
#include <thread>
#include <atomic>
#include <unordered_map>

#ifdef __linux__
#include <sched.h>
#include <numa.h>
#include <numaif.h>
#endif

namespace ultra_cpp {
namespace performance {

/**
 * CPU topology information
 */
struct CPUTopology {
    struct Core {
        int physical_id;
        int core_id;
        int numa_node;
        std::vector<int> logical_cpus;
        bool is_hyperthread_sibling;
    };
    
    struct NUMANode {
        int node_id;
        std::vector<int> cpus;
        size_t memory_size_mb;
        std::vector<int> distances; // Distance to other NUMA nodes
    };
    
    std::vector<Core> cores;
    std::vector<NUMANode> numa_nodes;
    int total_logical_cpus;
    int total_physical_cores;
    int total_numa_nodes;
};

/**
 * CPU affinity and NUMA optimization manager
 */
class CPUAffinityManager {
public:
    enum class AffinityStrategy {
        NONE,                    // No affinity setting
        SINGLE_CORE,            // Pin to single core
        NUMA_LOCAL,             // Pin to cores within same NUMA node
        PERFORMANCE_CORES,      // Pin to performance cores (avoid SMT siblings)
        ISOLATED_CORES,         // Pin to isolated cores (if available)
        CUSTOM                  // Custom CPU set
    };
    
    struct AffinityConfig {
        AffinityStrategy strategy = AffinityStrategy::NUMA_LOCAL;
        int preferred_numa_node = -1;  // -1 for auto-detect
        std::vector<int> custom_cpus;
        bool avoid_hyperthread_siblings = true;
        bool enable_numa_interleaving = false;
    };
    
    CPUAffinityManager();
    ~CPUAffinityManager();
    
    // Initialize and discover topology
    bool initialize();
    
    // Get system topology information
    const CPUTopology& get_topology() const { return topology_; }
    
    // Set CPU affinity for current thread
    bool set_thread_affinity(const AffinityConfig& config);
    bool set_thread_affinity(std::thread& thread, const AffinityConfig& config);
    bool set_thread_affinity(pthread_t thread, const AffinityConfig& config);
    
    // NUMA memory policy
    bool set_numa_memory_policy(int numa_node, bool strict = false);
    bool set_numa_interleaved_policy(const std::vector<int>& nodes);
    
    // Get optimal CPU set for given strategy
    std::vector<int> get_optimal_cpus(const AffinityConfig& config) const;
    
    // Performance monitoring
    struct AffinityStats {
        std::atomic<uint64_t> context_switches{0};
        std::atomic<uint64_t> numa_faults{0};
        std::atomic<uint64_t> cache_misses{0};
        std::atomic<uint64_t> migrations{0};
    };
    
    const AffinityStats& get_stats() const { return stats_; }
    void reset_stats();
    
    // Utility functions
    int get_current_cpu() const;
    int get_current_numa_node() const;
    bool is_numa_available() const;
    
    // Thread pool optimization
    struct ThreadPoolConfig {
        size_t num_threads;
        AffinityStrategy strategy;
        bool dedicate_numa_nodes = false;
        bool isolate_interrupt_cpus = true;
    };
    
    std::vector<AffinityConfig> generate_thread_pool_affinities(
        const ThreadPoolConfig& config) const;

private:
    CPUTopology topology_;
    mutable AffinityStats stats_;
    bool numa_available_;
    
    // Topology discovery
    bool discover_cpu_topology();
    bool discover_numa_topology();
    void parse_proc_cpuinfo();
    void parse_sys_devices();
    
    // Helper functions
    std::vector<int> get_performance_cores() const;
    std::vector<int> get_isolated_cores() const;
    std::vector<int> filter_hyperthread_siblings(const std::vector<int>& cpus) const;
    int find_best_numa_node() const;
    
#ifdef __linux__
    cpu_set_t create_cpu_set(const std::vector<int>& cpus) const;
#endif
};

/**
 * RAII CPU affinity setter
 */
class ScopedCPUAffinity {
public:
    explicit ScopedCPUAffinity(const CPUAffinityManager::AffinityConfig& config);
    ~ScopedCPUAffinity();
    
    bool is_valid() const { return valid_; }

private:
    bool valid_;
#ifdef __linux__
    cpu_set_t original_affinity_;
#endif
};

/**
 * Performance-aware thread launcher
 */
class PerformanceThread {
public:
    template<typename Function, typename... Args>
    static std::thread launch_with_affinity(
        const CPUAffinityManager::AffinityConfig& config,
        Function&& f, Args&&... args) {
        
        return std::thread([config, f = std::forward<Function>(f), 
                          args = std::make_tuple(std::forward<Args>(args)...)]() mutable {
            CPUAffinityManager manager;
            manager.initialize();
            manager.set_thread_affinity(config);
            
            std::apply(f, std::move(args));
        });
    }
};

} // namespace performance
} // namespace ultra_cpp