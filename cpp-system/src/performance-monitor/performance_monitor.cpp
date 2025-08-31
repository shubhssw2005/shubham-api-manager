#include "performance-monitor/performance_monitor.hpp"
#include "performance-monitor/metrics_collector.hpp"
#include "performance-monitor/hardware_counters.hpp"
#include "performance-monitor/prometheus_exporter.hpp"
#include "performance-monitor/slo_monitor.hpp"
#include "common/logger.hpp"

#include <chrono>
#include <thread>
#include <x86intrin.h>

namespace ultra {
namespace monitor {

PerformanceMonitor::PerformanceMonitor(const Config& config)
    : config_(config)
    , metrics_collector_(std::make_unique<MetricsCollector>(MetricsCollector::Config{}))
    , hardware_counters_(std::make_unique<HardwareCounters>())
    , prometheus_exporter_(std::make_unique<PrometheusExporter>(
        PrometheusExporter::Config{.port = config.prometheus_port, .endpoint = config.prometheus_endpoint},
        *metrics_collector_))
    , slo_monitor_(std::make_unique<SLOMonitor>(SLOMonitor::Config{}, *metrics_collector_)) {
    
    LOG_INFO("PerformanceMonitor initialized with {} buckets, hardware counters: {}", 
             config_.histogram_buckets, config_.enable_hardware_counters);
}

PerformanceMonitor::~PerformanceMonitor() {
    stop_collection();
    stop_prometheus_server();
}

void PerformanceMonitor::start_collection() {
    if (running_.exchange(true, std::memory_order_acq_rel)) {
        LOG_WARN("PerformanceMonitor already running");
        return;
    }

    // Initialize hardware counters if enabled
    if (config_.enable_hardware_counters) {
        std::vector<HardwareCounters::CounterConfig> counter_configs = {
            {HardwareCounters::CounterType::CPU_CYCLES, true, "cpu_cycles"},
            {HardwareCounters::CounterType::INSTRUCTIONS, true, "instructions"},
            {HardwareCounters::CounterType::CACHE_MISSES, true, "cache_misses"},
            {HardwareCounters::CounterType::BRANCH_MISSES, true, "branch_misses"},
            {HardwareCounters::CounterType::PAGE_FAULTS, true, "page_faults"}
        };
        
        if (!hardware_counters_->initialize(counter_configs)) {
            LOG_WARN("Failed to initialize hardware counters: {}", hardware_counters_->get_last_error());
            config_.enable_hardware_counters = false;
        } else {
            hardware_counters_->start_counting();
            LOG_INFO("Hardware performance counters initialized successfully");
        }
    }

    // Start SLO monitoring if enabled
    if (config_.enable_slo_monitoring) {
        slo_monitor_->start_monitoring();
    }

    // Start collection thread
    collection_thread_ = std::make_unique<std::thread>(&PerformanceMonitor::collection_loop, this);
    
    LOG_INFO("PerformanceMonitor collection started");
}

void PerformanceMonitor::stop_collection() {
    if (!running_.exchange(false, std::memory_order_acq_rel)) {
        return;
    }

    // Stop collection thread
    if (collection_thread_ && collection_thread_->joinable()) {
        collection_thread_->join();
        collection_thread_.reset();
    }

    // Stop hardware counters
    if (config_.enable_hardware_counters && hardware_counters_->is_initialized()) {
        hardware_counters_->stop_counting();
        hardware_counters_->shutdown();
    }

    // Stop SLO monitoring
    if (slo_monitor_) {
        slo_monitor_->stop_monitoring();
    }

    LOG_INFO("PerformanceMonitor collection stopped");
}

void PerformanceMonitor::start_prometheus_server() {
    if (prometheus_running_.exchange(true, std::memory_order_acq_rel)) {
        LOG_WARN("Prometheus server already running");
        return;
    }

    prometheus_thread_ = std::make_unique<std::thread>(&PerformanceMonitor::prometheus_server_loop, this);
    LOG_INFO("Prometheus server started on port {}", config_.prometheus_port);
}

void PerformanceMonitor::stop_prometheus_server() {
    if (!prometheus_running_.exchange(false, std::memory_order_acq_rel)) {
        return;
    }

    if (prometheus_exporter_) {
        prometheus_exporter_->stop_server();
    }

    if (prometheus_thread_ && prometheus_thread_->joinable()) {
        prometheus_thread_->join();
        prometheus_thread_.reset();
    }

    LOG_INFO("Prometheus server stopped");
}

void PerformanceMonitor::increment_counter(const std::string& name, uint64_t value) noexcept {
    metrics_collector_->increment_counter(name, value);
}

void PerformanceMonitor::set_gauge(const std::string& name, double value) noexcept {
    metrics_collector_->set_gauge(name, value);
}

void PerformanceMonitor::observe_histogram(const std::string& name, double value) noexcept {
    metrics_collector_->observe_histogram(name, value);
}

void PerformanceMonitor::record_timing(const std::string& name, uint64_t duration_ns) noexcept {
    metrics_collector_->record_timing(name, duration_ns);
}

void PerformanceMonitor::register_slo(const SLOConfig& slo_config) {
    SLOMonitor::SLODefinition slo_def;
    slo_def.name = slo_config.name;
    slo_def.target_percentile = slo_config.target_percentile;
    slo_def.target_latency_ns = slo_config.target_latency_ns;
    slo_def.evaluation_window = slo_config.evaluation_window;
    
    slo_monitor_->register_slo(slo_def);
    
    LOG_INFO("Registered SLO: {} (P{} < {}ns)", 
             slo_config.name, slo_config.target_percentile * 100, slo_config.target_latency_ns);
}

void PerformanceMonitor::check_slo_violations() {
    // SLO violations are checked automatically by the SLO monitor thread
    // This method can be used for manual checks if needed
    auto all_status = slo_monitor_->get_all_slo_status();
    for (const auto& status : all_status) {
        if (!status.is_healthy) {
            LOG_WARN("SLO violation detected: {} - {}", status.name, status.status_message);
        }
    }
}

std::string PerformanceMonitor::export_prometheus_metrics() {
    return prometheus_exporter_->export_metrics();
}

PerformanceMonitor::HardwareMetrics PerformanceMonitor::get_hardware_metrics() const {
    HardwareMetrics metrics{};
    
    if (!config_.enable_hardware_counters || !hardware_counters_->is_initialized()) {
        return metrics;
    }

    auto counter_group = hardware_counters_->read_counters();
    
    for (const auto& counter : counter_group.counters) {
        switch (counter.type) {
            case HardwareCounters::CounterType::CPU_CYCLES:
                metrics.cpu_cycles = counter.value;
                break;
            case HardwareCounters::CounterType::INSTRUCTIONS:
                metrics.instructions = counter.value;
                break;
            case HardwareCounters::CounterType::CACHE_MISSES:
                metrics.cache_misses = counter.value;
                break;
            case HardwareCounters::CounterType::BRANCH_MISSES:
                metrics.branch_mispredictions = counter.value;
                break;
            case HardwareCounters::CounterType::PAGE_FAULTS:
                metrics.page_faults = counter.value;
                break;
            default:
                break;
        }
    }

    // Calculate derived metrics
    if (metrics.cpu_cycles > 0) {
        metrics.ipc = static_cast<double>(metrics.instructions) / metrics.cpu_cycles;
    }

    // Calculate cache hit rate (assuming we have cache references and misses)
    if (metrics.cache_misses > 0) {
        // This is a simplified calculation - in practice you'd need cache references too
        metrics.cache_hit_rate = 1.0 - (static_cast<double>(metrics.cache_misses) / (metrics.cache_misses + 1000000));
    }

    return metrics;
}

PerformanceMonitor::PerformanceStats PerformanceMonitor::get_performance_stats() const {
    PerformanceStats stats{};
    
    if (metrics_collector_) {
        const auto& collector_stats = metrics_collector_->get_stats();
        stats.total_metrics_collected = collector_stats.total_operations.load();
        stats.collection_overhead_ns = collector_stats.collection_overhead_ns.load();
        
        // Calculate metrics per second
        auto now = std::chrono::steady_clock::now();
        static auto start_time = now;
        auto elapsed_seconds = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();
        if (elapsed_seconds > 0) {
            stats.metrics_per_second = static_cast<double>(stats.total_metrics_collected) / elapsed_seconds;
        }
    }

    if (prometheus_exporter_) {
        const auto& exporter_stats = prometheus_exporter_->get_stats();
        stats.export_overhead_ns = exporter_stats.export_duration_ns.load();
    }

    return stats;
}

void PerformanceMonitor::collection_loop() {
    LOG_INFO("Performance monitor collection loop started");
    
    while (running_.load(std::memory_order_acquire)) {
        auto start_time = get_cpu_cycles();
        
        // Collect hardware metrics if enabled
        if (config_.enable_hardware_counters && hardware_counters_->is_initialized()) {
            auto hw_metrics = get_hardware_metrics();
            
            // Update internal metrics
            set_gauge("hardware.cpu_cycles_per_second", hw_metrics.cpu_cycles);
            set_gauge("hardware.instructions_per_second", hw_metrics.instructions);
            set_gauge("hardware.cache_misses_per_second", hw_metrics.cache_misses);
            set_gauge("hardware.branch_mispredictions_per_second", hw_metrics.branch_mispredictions);
            set_gauge("hardware.ipc", hw_metrics.ipc);
            set_gauge("hardware.cache_hit_rate", hw_metrics.cache_hit_rate);
        }

        // Update performance statistics
        auto perf_stats = get_performance_stats();
        set_gauge("monitor.metrics_per_second", perf_stats.metrics_per_second);
        set_gauge("monitor.collection_overhead_ns", perf_stats.collection_overhead_ns);
        set_gauge("monitor.export_overhead_ns", perf_stats.export_overhead_ns);

        auto end_time = get_cpu_cycles();
        auto collection_duration_ns = cycles_to_nanoseconds(end_time - start_time);
        record_timing("monitor.collection_duration", collection_duration_ns);

        // Sleep until next collection interval
        std::this_thread::sleep_for(config_.collection_interval);
    }
    
    LOG_INFO("Performance monitor collection loop stopped");
}

void PerformanceMonitor::prometheus_server_loop() {
    if (!prometheus_exporter_->start_server()) {
        LOG_ERROR("Failed to start Prometheus server");
        prometheus_running_.store(false, std::memory_order_release);
        return;
    }

    LOG_INFO("Prometheus server loop started");
    
    while (prometheus_running_.load(std::memory_order_acquire)) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    
    LOG_INFO("Prometheus server loop stopped");
}

uint64_t PerformanceMonitor::get_cpu_cycles() noexcept {
    return __rdtsc();
}

uint64_t PerformanceMonitor::cycles_to_nanoseconds(uint64_t cycles) noexcept {
    // Approximate conversion - in practice you'd calibrate this
    // Assuming 3GHz CPU: 1 cycle = 1/3 nanosecond
    static constexpr double CYCLES_PER_NS = 3.0;
    return static_cast<uint64_t>(cycles / CYCLES_PER_NS);
}

// Timer implementation
PerformanceMonitor::Timer::Timer(PerformanceMonitor& monitor, const std::string& name)
    : monitor_(monitor), name_(name), start_cycles_(get_cpu_cycles()) {
}

PerformanceMonitor::Timer::~Timer() {
    if (!stopped_) {
        stop();
    }
}

void PerformanceMonitor::Timer::stop() {
    if (stopped_) return;
    
    auto end_cycles = get_cpu_cycles();
    auto duration_ns = cycles_to_nanoseconds(end_cycles - start_cycles_);
    monitor_.record_timing(name_, duration_ns);
    stopped_ = true;
}

uint64_t PerformanceMonitor::Timer::elapsed_ns() const noexcept {
    auto current_cycles = stopped_ ? start_cycles_ : get_cpu_cycles();
    return cycles_to_nanoseconds(current_cycles - start_cycles_);
}

} // namespace monitor
} // namespace ultra