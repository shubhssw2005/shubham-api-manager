#pragma once

#include <memory>
#include <vector>
#include <unordered_map>
#include <functional>
#include <atomic>
#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>

namespace ultra_cpp {
namespace performance {

/**
 * Adaptive performance tuning system
 */
class AdaptiveTuningSystem {
public:
    // Performance metrics
    struct PerformanceMetrics {
        std::atomic<double> throughput{0.0};           // Operations per second
        std::atomic<double> latency_p50{0.0};          // 50th percentile latency (ns)
        std::atomic<double> latency_p95{0.0};          // 95th percentile latency (ns)
        std::atomic<double> latency_p99{0.0};          // 99th percentile latency (ns)
        std::atomic<double> cpu_utilization{0.0};      // CPU usage percentage
        std::atomic<double> memory_utilization{0.0};   // Memory usage percentage
        std::atomic<double> cache_hit_rate{0.0};       // Cache hit rate percentage
        std::atomic<uint64_t> error_rate{0};           // Errors per second
        std::atomic<double> queue_depth{0.0};          // Average queue depth
        std::chrono::steady_clock::time_point timestamp;
    };
    
    // Tunable parameters
    struct TunableParameter {
        std::string name;
        double current_value;
        double min_value;
        double max_value;
        double step_size;
        std::function<void(double)> setter;
        std::function<double()> getter;
        double sensitivity = 1.0;  // How much this parameter affects performance
    };
    
    // Tuning objectives
    enum class OptimizationObjective {
        MAXIMIZE_THROUGHPUT,
        MINIMIZE_LATENCY,
        MINIMIZE_CPU_USAGE,
        MINIMIZE_MEMORY_USAGE,
        MAXIMIZE_CACHE_HIT_RATE,
        MINIMIZE_ERROR_RATE,
        BALANCED_PERFORMANCE
    };
    
    struct TuningConfig {
        OptimizationObjective objective = OptimizationObjective::BALANCED_PERFORMANCE;
        std::chrono::milliseconds measurement_interval{1000};
        std::chrono::milliseconds tuning_interval{10000};
        size_t history_size = 100;
        double learning_rate = 0.1;
        double exploration_rate = 0.1;  // For exploration vs exploitation
        bool enable_auto_tuning = true;
        std::vector<std::string> parameters_to_tune;
    };
    
    AdaptiveTuningSystem(const TuningConfig& config = TuningConfig{});
    ~AdaptiveTuningSystem();
    
    // Parameter management
    void register_parameter(const TunableParameter& param);
    void unregister_parameter(const std::string& name);
    std::vector<std::string> list_parameters() const;
    
    // Manual parameter control
    bool set_parameter(const std::string& name, double value);
    double get_parameter(const std::string& name) const;
    
    // Metrics collection
    void update_metrics(const PerformanceMetrics& metrics);
    PerformanceMetrics get_current_metrics() const;
    std::vector<PerformanceMetrics> get_metrics_history() const;
    
    // Tuning control
    void start_auto_tuning();
    void stop_auto_tuning();
    bool is_auto_tuning_active() const;
    
    // Manual tuning trigger
    void trigger_tuning_cycle();
    
    // Performance analysis
    struct TuningResult {
        std::string parameter_name;
        double old_value;
        double new_value;
        double performance_improvement;
        std::chrono::steady_clock::time_point timestamp;
    };
    
    std::vector<TuningResult> get_tuning_history() const;
    
    // Tuning algorithms
    enum class TuningAlgorithm {
        GRADIENT_DESCENT,
        SIMULATED_ANNEALING,
        GENETIC_ALGORITHM,
        BAYESIAN_OPTIMIZATION,
        REINFORCEMENT_LEARNING
    };
    
    void set_tuning_algorithm(TuningAlgorithm algorithm);
    TuningAlgorithm get_tuning_algorithm() const;

private:
    TuningConfig config_;
    std::unordered_map<std::string, TunableParameter> parameters_;
    std::vector<PerformanceMetrics> metrics_history_;
    std::vector<TuningResult> tuning_history_;
    
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::atomic<bool> auto_tuning_active_{false};
    std::thread tuning_thread_;
    
    TuningAlgorithm current_algorithm_ = TuningAlgorithm::GRADIENT_DESCENT;
    
    // Tuning algorithms
    void tuning_loop();
    void run_gradient_descent();
    void run_simulated_annealing();
    void run_genetic_algorithm();
    void run_bayesian_optimization();
    void run_reinforcement_learning();
    
    // Utility functions
    double calculate_fitness(const PerformanceMetrics& metrics) const;
    double calculate_gradient(const std::string& param_name);
    bool should_explore() const;
    void apply_parameter_change(const std::string& name, double new_value);
    void record_tuning_result(const std::string& param_name, 
                             double old_value, double new_value, 
                             double improvement);
};

/**
 * Workload-aware adaptive scheduler
 */
class AdaptiveScheduler {
public:
    enum class WorkloadType {
        CPU_INTENSIVE,
        IO_INTENSIVE,
        MEMORY_INTENSIVE,
        NETWORK_INTENSIVE,
        MIXED,
        UNKNOWN
    };
    
    struct WorkloadCharacteristics {
        WorkloadType type = WorkloadType::UNKNOWN;
        double cpu_intensity = 0.0;      // 0.0 to 1.0
        double io_intensity = 0.0;       // 0.0 to 1.0
        double memory_intensity = 0.0;   // 0.0 to 1.0
        double network_intensity = 0.0;  // 0.0 to 1.0
        size_t typical_batch_size = 1;
        std::chrono::nanoseconds typical_duration{0};
    };
    
    struct SchedulingPolicy {
        std::string name;
        size_t thread_pool_size;
        size_t queue_capacity;
        std::chrono::milliseconds time_slice{10};
        int priority_levels = 3;
        bool enable_work_stealing = true;
        bool enable_numa_awareness = true;
    };
    
    AdaptiveScheduler();
    ~AdaptiveScheduler();
    
    // Workload classification
    void register_workload_type(WorkloadType type, const WorkloadCharacteristics& characteristics);
    WorkloadType classify_workload(const std::function<void()>& task);
    
    // Scheduling policies
    void register_policy(const SchedulingPolicy& policy);
    void set_active_policy(const std::string& policy_name);
    SchedulingPolicy get_active_policy() const;
    
    // Task submission
    template<typename F, typename... Args>
    auto submit_task(F&& f, Args&&... args) -> std::future<std::invoke_result_t<F, Args...>>;
    
    void submit_batch(const std::vector<std::function<void()>>& tasks);
    
    // Adaptive behavior
    void enable_adaptive_scheduling(bool enable = true);
    void set_adaptation_interval(std::chrono::milliseconds interval);
    
    // Performance monitoring
    struct SchedulerMetrics {
        std::atomic<uint64_t> tasks_completed{0};
        std::atomic<uint64_t> tasks_queued{0};
        std::atomic<double> average_wait_time{0.0};
        std::atomic<double> average_execution_time{0.0};
        std::atomic<double> thread_utilization{0.0};
        std::atomic<uint64_t> work_stealing_events{0};
    };
    
    SchedulerMetrics get_metrics() const;
    void reset_metrics();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/**
 * Memory allocation tuner
 */
class MemoryAllocationTuner {
public:
    struct AllocationPattern {
        size_t typical_size;
        size_t alignment;
        std::chrono::nanoseconds lifetime;
        double frequency;  // Allocations per second
        bool is_hot_path;
    };
    
    struct PoolConfig {
        size_t block_size;
        size_t initial_blocks;
        size_t max_blocks;
        bool thread_local = false;
        bool numa_aware = false;
    };
    
    MemoryAllocationTuner();
    ~MemoryAllocationTuner();
    
    // Pattern analysis
    void record_allocation(size_t size, void* ptr);
    void record_deallocation(void* ptr);
    
    // Pool management
    void create_pool(const std::string& name, const PoolConfig& config);
    void* allocate_from_pool(const std::string& pool_name, size_t size);
    void deallocate_to_pool(const std::string& pool_name, void* ptr);
    
    // Adaptive tuning
    void enable_adaptive_pools(bool enable = true);
    void tune_pool_sizes();
    
    // Statistics
    struct AllocationStats {
        std::atomic<uint64_t> total_allocations{0};
        std::atomic<uint64_t> total_deallocations{0};
        std::atomic<uint64_t> bytes_allocated{0};
        std::atomic<uint64_t> bytes_deallocated{0};
        std::atomic<uint64_t> pool_hits{0};
        std::atomic<uint64_t> pool_misses{0};
        std::atomic<double> fragmentation_ratio{0.0};
    };
    
    AllocationStats get_stats() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/**
 * Cache tuning system
 */
class CacheTuner {
public:
    struct CacheConfig {
        size_t capacity;
        size_t shard_count;
        double eviction_threshold = 0.8;
        std::chrono::milliseconds ttl{0};  // 0 = no TTL
        bool enable_prefetching = false;
        size_t prefetch_distance = 1;
    };
    
    enum class EvictionPolicy {
        LRU,
        LFU,
        RANDOM,
        TTL_BASED,
        ADAPTIVE
    };
    
    CacheTuner();
    ~CacheTuner();
    
    // Cache management
    void register_cache(const std::string& name, const CacheConfig& config);
    void set_eviction_policy(const std::string& cache_name, EvictionPolicy policy);
    
    // Access pattern analysis
    void record_cache_access(const std::string& cache_name, const std::string& key, bool hit);
    void analyze_access_patterns();
    
    // Adaptive tuning
    void enable_adaptive_sizing(const std::string& cache_name, bool enable = true);
    void tune_cache_parameters();
    
    // Performance metrics
    struct CacheMetrics {
        std::atomic<uint64_t> hits{0};
        std::atomic<uint64_t> misses{0};
        std::atomic<uint64_t> evictions{0};
        std::atomic<double> hit_rate{0.0};
        std::atomic<double> load_factor{0.0};
        std::atomic<uint64_t> memory_usage{0};
    };
    
    CacheMetrics get_cache_metrics(const std::string& cache_name) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/**
 * Network tuning system
 */
class NetworkTuner {
public:
    struct NetworkConfig {
        size_t buffer_size = 65536;
        size_t connection_pool_size = 100;
        std::chrono::milliseconds connection_timeout{5000};
        std::chrono::milliseconds read_timeout{30000};
        bool enable_tcp_nodelay = true;
        bool enable_tcp_keepalive = true;
        size_t max_concurrent_connections = 1000;
    };
    
    NetworkTuner();
    ~NetworkTuner();
    
    // Configuration management
    void set_network_config(const NetworkConfig& config);
    NetworkConfig get_network_config() const;
    
    // Performance monitoring
    void record_connection_event(const std::string& event_type, 
                                std::chrono::nanoseconds duration);
    void record_throughput(size_t bytes_sent, size_t bytes_received);
    
    // Adaptive tuning
    void enable_adaptive_tuning(bool enable = true);
    void tune_buffer_sizes();
    void tune_connection_limits();
    
    // Metrics
    struct NetworkMetrics {
        std::atomic<uint64_t> connections_established{0};
        std::atomic<uint64_t> connections_failed{0};
        std::atomic<uint64_t> bytes_sent{0};
        std::atomic<uint64_t> bytes_received{0};
        std::atomic<double> average_latency{0.0};
        std::atomic<double> throughput_mbps{0.0};
    };
    
    NetworkMetrics get_metrics() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/**
 * Integrated performance tuning manager
 */
class PerformanceTuningManager {
public:
    PerformanceTuningManager();
    ~PerformanceTuningManager();
    
    // Component access
    AdaptiveTuningSystem& get_adaptive_tuning_system();
    AdaptiveScheduler& get_adaptive_scheduler();
    MemoryAllocationTuner& get_memory_tuner();
    CacheTuner& get_cache_tuner();
    NetworkTuner& get_network_tuner();
    
    // Global tuning control
    void start_all_tuning();
    void stop_all_tuning();
    
    // Performance profiles
    void save_performance_profile(const std::string& name);
    void load_performance_profile(const std::string& name);
    std::vector<std::string> list_performance_profiles() const;
    
    // System-wide metrics
    struct SystemMetrics {
        AdaptiveTuningSystem::PerformanceMetrics adaptive_metrics;
        AdaptiveScheduler::SchedulerMetrics scheduler_metrics;
        MemoryAllocationTuner::AllocationStats memory_stats;
        CacheTuner::CacheMetrics cache_metrics;
        NetworkTuner::NetworkMetrics network_metrics;
    };
    
    SystemMetrics get_system_metrics() const;

private:
    std::unique_ptr<AdaptiveTuningSystem> adaptive_tuning_;
    std::unique_ptr<AdaptiveScheduler> adaptive_scheduler_;
    std::unique_ptr<MemoryAllocationTuner> memory_tuner_;
    std::unique_ptr<CacheTuner> cache_tuner_;
    std::unique_ptr<NetworkTuner> network_tuner_;
};

} // namespace performance
} // namespace ultra_cpp