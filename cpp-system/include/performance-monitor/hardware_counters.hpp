#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ultra {
namespace monitor {

/**
 * Hardware Performance Counter (PMU) integration
 * Provides access to CPU performance monitoring units for detailed performance analysis
 */
class HardwareCounters {
public:
    enum class CounterType {
        CPU_CYCLES,
        INSTRUCTIONS,
        CACHE_REFERENCES,
        CACHE_MISSES,
        BRANCH_INSTRUCTIONS,
        BRANCH_MISSES,
        PAGE_FAULTS,
        CONTEXT_SWITCHES,
        CPU_MIGRATIONS,
        L1D_CACHE_MISSES,
        L1I_CACHE_MISSES,
        LLC_CACHE_MISSES,
        DTLB_MISSES,
        ITLB_MISSES
    };

    struct CounterConfig {
        CounterType type;
        bool enabled = true;
        std::string name;
    };

    struct CounterValue {
        CounterType type;
        uint64_t value;
        uint64_t time_enabled_ns;
        uint64_t time_running_ns;
        double scaling_factor;
    };

    struct CounterGroup {
        std::vector<CounterValue> counters;
        uint64_t timestamp_ns;
        uint32_t cpu_id;
    };

    explicit HardwareCounters();
    ~HardwareCounters();

    // Configuration
    bool initialize(const std::vector<CounterConfig>& configs);
    void shutdown();
    bool is_initialized() const noexcept { return initialized_; }

    // Counter operations
    bool start_counting();
    bool stop_counting();
    bool reset_counters();
    
    // Data collection
    CounterGroup read_counters();
    std::vector<CounterGroup> read_all_cpus();
    
    // Derived metrics calculation
    double calculate_ipc(const CounterGroup& group) const;
    double calculate_cache_hit_rate(const CounterGroup& group) const;
    double calculate_branch_prediction_rate(const CounterGroup& group) const;
    double calculate_memory_bandwidth_gbps(const CounterGroup& group, 
                                         uint64_t duration_ns) const;

    // System capabilities
    static bool is_pmu_available();
    static std::vector<CounterType> get_available_counters();
    static std::string counter_type_to_string(CounterType type);
    
    // Error handling
    std::string get_last_error() const { return last_error_; }

private:
    bool initialized_ = false;
    std::string last_error_;
    
    // Platform-specific implementation
    class PlatformImpl;
    std::unique_ptr<PlatformImpl> impl_;
    
    // Counter management
    struct CounterDescriptor {
        int fd = -1;
        CounterType type;
        uint32_t cpu_id;
        bool active = false;
    };
    
    std::vector<CounterDescriptor> counter_fds_;
    std::atomic<bool> counting_{false};
    
    // Helper methods
    bool setup_counter(CounterType type, uint32_t cpu_id);
    void cleanup_counters();
    uint64_t read_counter_value(int fd) const;
    
    // Platform-specific counter setup
    int create_perf_event(CounterType type, uint32_t cpu_id);
    uint32_t get_perf_event_type(CounterType type) const;
    uint64_t get_perf_event_config(CounterType type) const;
};

/**
 * RAII wrapper for hardware counter collection
 */
class ScopedCounterCollection {
public:
    explicit ScopedCounterCollection(HardwareCounters& counters);
    ~ScopedCounterCollection();
    
    HardwareCounters::CounterGroup get_results();
    
private:
    HardwareCounters& counters_;
    bool started_ = false;
    HardwareCounters::CounterGroup start_values_;
};

} // namespace monitor
} // namespace ultra