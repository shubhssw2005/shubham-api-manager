#pragma once

#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <thread>

namespace ultra {
namespace monitor {

/**
 * High-performance monitoring system with nanosecond precision
 * Implements lock-free metrics collection and hardware performance counters
 */
class PerformanceMonitor {
public:
    struct Config {
        std::chrono::milliseconds collection_interval{100};
        bool enable_hardware_counters = true;
        std::string prometheus_endpoint = "/metrics";
        uint16_t prometheus_port = 9090;
        size_t histogram_buckets = 64;
        bool enable_slo_monitoring = true;
    };

    explicit PerformanceMonitor(const Config& config);
    ~PerformanceMonitor();

    // Lifecycle management
    void start_collection();
    void stop_collection();
    bool is_running() const noexcept { return running_.load(std::memory_order_acquire); }

    // Timer for automatic scope-based timing
    class Timer {
    public:
        Timer(PerformanceMonitor& monitor, const std::string& name);
        ~Timer();
        
        // Manual timing control
        void stop();
        uint64_t elapsed_ns() const noexcept;
        
    private:
        PerformanceMonitor& monitor_;
        std::string name_;
        uint64_t start_cycles_;
        bool stopped_ = false;
    };

    // Metrics operations (lock-free)
    void increment_counter(const std::string& name, uint64_t value = 1) noexcept;
    void set_gauge(const std::string& name, double value) noexcept;
    void observe_histogram(const std::string& name, double value) noexcept;
    void record_timing(const std::string& name, uint64_t duration_ns) noexcept;

    // SLO monitoring
    struct SLOConfig {
        std::string name;
        double target_percentile = 0.99;  // P99
        uint64_t target_latency_ns = 1000000;  // 1ms
        std::chrono::seconds evaluation_window{60};
        std::function<void(const std::string&)> alert_callback;
    };
    
    void register_slo(const SLOConfig& slo_config);
    void check_slo_violations();

    // Prometheus export
    std::string export_prometheus_metrics();
    void start_prometheus_server();
    void stop_prometheus_server();

    // Hardware performance counters
    struct HardwareMetrics {
        uint64_t cpu_cycles = 0;
        uint64_t instructions = 0;
        uint64_t cache_misses = 0;
        uint64_t branch_mispredictions = 0;
        uint64_t page_faults = 0;
        uint64_t context_switches = 0;
        double ipc = 0.0;  // Instructions per cycle
        double cache_hit_rate = 0.0;
    };
    
    HardwareMetrics get_hardware_metrics() const;

    // Performance statistics
    struct PerformanceStats {
        uint64_t total_metrics_collected = 0;
        uint64_t collection_overhead_ns = 0;
        uint64_t export_overhead_ns = 0;
        double metrics_per_second = 0.0;
    };
    
    PerformanceStats get_performance_stats() const;

private:
    Config config_;
    std::atomic<bool> running_{false};
    std::atomic<bool> prometheus_running_{false};
    
    // Collection thread
    std::unique_ptr<std::thread> collection_thread_;
    std::unique_ptr<std::thread> prometheus_thread_;
    
    // Forward declarations for implementation details
    class MetricsCollector;
    class HardwareCounters;
    class PrometheusExporter;
    class SLOMonitor;
    
    std::unique_ptr<MetricsCollector> metrics_collector_;
    std::unique_ptr<HardwareCounters> hardware_counters_;
    std::unique_ptr<PrometheusExporter> prometheus_exporter_;
    std::unique_ptr<SLOMonitor> slo_monitor_;
    
    void collection_loop();
    void prometheus_server_loop();
    
    // High-resolution timing
    static uint64_t get_cpu_cycles() noexcept;
    static uint64_t cycles_to_nanoseconds(uint64_t cycles) noexcept;
};

// Convenience macros for timing
#define ULTRA_TIMER(monitor, name) \
    ultra::monitor::PerformanceMonitor::Timer timer_##__LINE__(monitor, name)

#define ULTRA_TIME_SCOPE(monitor, name) \
    auto timer_##__LINE__ = ultra::monitor::PerformanceMonitor::Timer(monitor, name)

} // namespace monitor
} // namespace ultra