#pragma once

#include <string>
#include <memory>
#include <unordered_map>
#include <vector>
#include <atomic>
#include <mutex>
#include <thread>
#include <chrono>
#include <functional>

namespace integration {

/**
 * Monitoring integration with existing Prometheus/Grafana setup
 */
class MonitoringIntegration {
public:
    struct Config {
        std::string service_name = "ultra-cpp-system";
        std::string service_version = "1.0.0";
        std::string instance_id;
        std::string prometheus_endpoint = "/metrics";
        uint16_t metrics_port = 9090;
        std::string metrics_bind_address = "0.0.0.0";
        uint32_t collection_interval_ms = 1000;
        uint32_t push_interval_ms = 15000;
        std::string pushgateway_url;
        std::string job_name = "ultra-cpp-system";
        bool enable_push_gateway = false;
        bool enable_http_server = true;
        std::unordered_map<std::string, std::string> static_labels;
    };

    enum class MetricType {
        COUNTER,
        GAUGE,
        HISTOGRAM,
        SUMMARY
    };

    struct MetricDefinition {
        std::string name;
        std::string help;
        MetricType type;
        std::vector<std::string> label_names;
        std::vector<double> histogram_buckets; // For histogram metrics
    };

    struct MetricValue {
        double value;
        std::unordered_map<std::string, std::string> labels;
        std::chrono::system_clock::time_point timestamp;
    };

    struct HistogramValue {
        std::unordered_map<double, uint64_t> buckets; // bucket -> count
        uint64_t count;
        double sum;
        std::unordered_map<std::string, std::string> labels;
        std::chrono::system_clock::time_point timestamp;
    };

    explicit MonitoringIntegration(const Config& config);
    ~MonitoringIntegration();

    // Lifecycle management
    bool initialize();
    void shutdown();
    bool is_running() const { return running_.load(); }

    // Metric registration
    bool register_counter(const std::string& name, 
                         const std::string& help,
                         const std::vector<std::string>& label_names = {});
    
    bool register_gauge(const std::string& name, 
                       const std::string& help,
                       const std::vector<std::string>& label_names = {});
    
    bool register_histogram(const std::string& name, 
                           const std::string& help,
                           const std::vector<double>& buckets,
                           const std::vector<std::string>& label_names = {});
    
    bool register_summary(const std::string& name, 
                         const std::string& help,
                         const std::vector<std::string>& label_names = {});

    // Metric operations
    void increment_counter(const std::string& name, 
                          double value = 1.0,
                          const std::unordered_map<std::string, std::string>& labels = {});
    
    void set_gauge(const std::string& name, 
                   double value,
                   const std::unordered_map<std::string, std::string>& labels = {});
    
    void observe_histogram(const std::string& name, 
                          double value,
                          const std::unordered_map<std::string, std::string>& labels = {});
    
    void observe_summary(const std::string& name, 
                        double value,
                        const std::unordered_map<std::string, std::string>& labels = {});

    // Batch operations for high performance
    void batch_increment_counter(const std::string& name, 
                               const std::vector<std::pair<double, std::unordered_map<std::string, std::string>>>& values);
    
    void batch_set_gauge(const std::string& name, 
                        const std::vector<std::pair<double, std::unordered_map<std::string, std::string>>>& values);

    // System metrics collection
    void collect_system_metrics();
    void collect_performance_metrics();
    void collect_application_metrics();

    // Custom metric collectors
    using MetricCollector = std::function<std::vector<std::pair<std::string, MetricValue>>()>;
    void register_custom_collector(const std::string& name, MetricCollector collector);
    void unregister_custom_collector(const std::string& name);

    // Prometheus format export
    std::string export_prometheus_format();
    std::string export_openmetrics_format();

    // Push to Pushgateway
    bool push_to_gateway();

    // Health and status
    struct MonitoringStats {
        std::atomic<uint64_t> metrics_collected{0};
        std::atomic<uint64_t> metrics_exported{0};
        std::atomic<uint64_t> push_attempts{0};
        std::atomic<uint64_t> push_successes{0};
        std::atomic<uint64_t> push_failures{0};
        std::atomic<uint64_t> http_requests{0};
        std::atomic<double> last_collection_duration_ms{0.0};
        std::atomic<double> last_export_duration_ms{0.0};
    };

    MonitoringStats get_stats() const;
    bool health_check();

private:
    Config config_;
    std::atomic<bool> running_{false};
    MonitoringStats stats_;

    // Metric storage
    std::unordered_map<std::string, MetricDefinition> metric_definitions_;
    std::unordered_map<std::string, std::vector<MetricValue>> counter_values_;
    std::unordered_map<std::string, std::vector<MetricValue>> gauge_values_;
    std::unordered_map<std::string, std::vector<HistogramValue>> histogram_values_;
    std::unordered_map<std::string, std::vector<MetricValue>> summary_values_;
    
    mutable std::shared_mutex metrics_mutex_;

    // Custom collectors
    std::unordered_map<std::string, MetricCollector> custom_collectors_;
    std::mutex collectors_mutex_;

    // Background threads
    std::thread collection_thread_;
    std::thread push_thread_;

    // HTTP server for metrics endpoint
    class MetricsHttpServer;
    std::unique_ptr<MetricsHttpServer> http_server_;

    // HTTP client for push gateway
    class HttpClient;
    std::unique_ptr<HttpClient> http_client_;

    // Internal methods
    void collection_loop();
    void push_loop();
    
    std::string format_metric_name(const std::string& name);
    std::string format_labels(const std::unordered_map<std::string, std::string>& labels);
    std::string escape_label_value(const std::string& value);
    
    // System metrics collectors
    void collect_cpu_metrics();
    void collect_memory_metrics();
    void collect_network_metrics();
    void collect_disk_metrics();
    void collect_process_metrics();
    
    // Performance metrics collectors
    void collect_latency_metrics();
    void collect_throughput_metrics();
    void collect_error_metrics();
    void collect_cache_metrics();
    
    // Application metrics collectors
    void collect_session_metrics();
    void collect_event_metrics();
    void collect_proxy_metrics();
};

/**
 * RAII Timer for automatic latency measurement
 */
class MetricTimer {
public:
    MetricTimer(MonitoringIntegration* monitoring, 
               const std::string& metric_name,
               const std::unordered_map<std::string, std::string>& labels = {});
    ~MetricTimer();

    // Manual control
    void stop();
    double get_elapsed_ms() const;

private:
    MonitoringIntegration* monitoring_;
    std::string metric_name_;
    std::unordered_map<std::string, std::string> labels_;
    std::chrono::steady_clock::time_point start_time_;
    bool stopped_;
};

/**
 * Metric helper macros for convenience
 */
#define METRIC_COUNTER_INC(monitoring, name, ...) \
    (monitoring)->increment_counter(name, 1.0, ##__VA_ARGS__)

#define METRIC_GAUGE_SET(monitoring, name, value, ...) \
    (monitoring)->set_gauge(name, value, ##__VA_ARGS__)

#define METRIC_HISTOGRAM_OBSERVE(monitoring, name, value, ...) \
    (monitoring)->observe_histogram(name, value, ##__VA_ARGS__)

#define METRIC_TIMER(monitoring, name, ...) \
    MetricTimer _timer(monitoring, name, ##__VA_ARGS__)

/**
 * Predefined metric definitions for common use cases
 */
class StandardMetrics {
public:
    static void register_http_metrics(MonitoringIntegration* monitoring);
    static void register_cache_metrics(MonitoringIntegration* monitoring);
    static void register_database_metrics(MonitoringIntegration* monitoring);
    static void register_system_metrics(MonitoringIntegration* monitoring);
    static void register_session_metrics(MonitoringIntegration* monitoring);
    static void register_event_metrics(MonitoringIntegration* monitoring);
    static void register_performance_metrics(MonitoringIntegration* monitoring);

    // Metric name constants
    static constexpr const char* HTTP_REQUESTS_TOTAL = "http_requests_total";
    static constexpr const char* HTTP_REQUEST_DURATION = "http_request_duration_seconds";
    static constexpr const char* HTTP_REQUEST_SIZE = "http_request_size_bytes";
    static constexpr const char* HTTP_RESPONSE_SIZE = "http_response_size_bytes";
    
    static constexpr const char* CACHE_OPERATIONS_TOTAL = "cache_operations_total";
    static constexpr const char* CACHE_HIT_RATIO = "cache_hit_ratio";
    static constexpr const char* CACHE_SIZE_BYTES = "cache_size_bytes";
    static constexpr const char* CACHE_EVICTIONS_TOTAL = "cache_evictions_total";
    
    static constexpr const char* DB_CONNECTIONS_ACTIVE = "db_connections_active";
    static constexpr const char* DB_QUERY_DURATION = "db_query_duration_seconds";
    static constexpr const char* DB_QUERIES_TOTAL = "db_queries_total";
    
    static constexpr const char* SYSTEM_CPU_USAGE = "system_cpu_usage_percent";
    static constexpr const char* SYSTEM_MEMORY_USAGE = "system_memory_usage_bytes";
    static constexpr const char* SYSTEM_NETWORK_BYTES = "system_network_bytes_total";
    
    static constexpr const char* SESSIONS_ACTIVE = "sessions_active_total";
    static constexpr const char* SESSION_DURATION = "session_duration_seconds";
    static constexpr const char* SESSION_OPERATIONS_TOTAL = "session_operations_total";
    
    static constexpr const char* EVENTS_PUBLISHED_TOTAL = "events_published_total";
    static constexpr const char* EVENTS_PROCESSED_TOTAL = "events_processed_total";
    static constexpr const char* EVENT_PROCESSING_DURATION = "event_processing_duration_seconds";
    
    static constexpr const char* PERFORMANCE_LATENCY_P99 = "performance_latency_p99_seconds";
    static constexpr const char* PERFORMANCE_THROUGHPUT = "performance_throughput_ops_per_second";
    static constexpr const char* PERFORMANCE_ERRORS_TOTAL = "performance_errors_total";
};

} // namespace integration