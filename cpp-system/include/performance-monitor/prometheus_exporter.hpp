#pragma once

#include <atomic>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <chrono>

namespace ultra {
namespace monitor {

class MetricsCollector;

/**
 * Zero-copy Prometheus metrics exporter
 * Optimized for minimal serialization overhead and high throughput
 */
class PrometheusExporter {
public:
    struct Config {
        uint16_t port = 9090;
        std::string endpoint = "/metrics";
        std::string job_name = "ultra-cpp-system";
        std::string instance_id;
        bool enable_compression = true;
        size_t buffer_size = 1024 * 1024; // 1MB
        std::chrono::seconds cache_ttl{5};
    };

    explicit PrometheusExporter(const Config& config, MetricsCollector& collector);
    ~PrometheusExporter();

    // Server lifecycle
    bool start_server();
    void stop_server();
    bool is_running() const noexcept { return running_.load(std::memory_order_acquire); }

    // Metrics export
    std::string export_metrics();
    std::string export_metrics_compressed();
    
    // Zero-copy export to buffer
    size_t export_metrics_to_buffer(char* buffer, size_t buffer_size);
    size_t export_compressed_to_buffer(char* buffer, size_t buffer_size);

    // Custom labels and metadata
    void add_global_label(const std::string& key, const std::string& value);
    void remove_global_label(const std::string& key);
    void set_instance_metadata(const std::string& version, const std::string& build_info);

    // Performance statistics
    struct ExporterStats {
        std::atomic<uint64_t> requests_served{0};
        std::atomic<uint64_t> bytes_exported{0};
        std::atomic<uint64_t> export_duration_ns{0};
        std::atomic<uint64_t> compression_ratio_percent{0};
        std::atomic<uint64_t> cache_hits{0};
        std::atomic<uint64_t> cache_misses{0};
    };

    const ExporterStats& get_stats() const noexcept { return stats_; }

private:
    Config config_;
    MetricsCollector& collector_;
    ExporterStats stats_;
    std::atomic<bool> running_{false};

    // HTTP server implementation
    class HttpServer;
    std::unique_ptr<HttpServer> server_;

    // Metrics formatting
    class MetricsFormatter;
    std::unique_ptr<MetricsFormatter> formatter_;

    // Caching system
    struct CacheEntry {
        std::string data;
        std::chrono::steady_clock::time_point timestamp;
        size_t compressed_size;
        std::unique_ptr<char[]> compressed_data;
    };

    mutable std::atomic<CacheEntry*> cached_metrics_{nullptr};
    mutable std::atomic<uint64_t> cache_version_{0};

    // Global labels
    std::unordered_map<std::string, std::string> global_labels_;
    std::string instance_metadata_;

    // Helper methods
    bool is_cache_valid(const CacheEntry* entry) const;
    void update_cache(std::unique_ptr<CacheEntry> new_entry) const;
    std::string format_prometheus_metrics();
    size_t compress_data(const char* input, size_t input_size, char* output, size_t output_size);
};

/**
 * High-performance Prometheus metrics formatter
 * Uses SIMD instructions and zero-copy techniques for fast serialization
 */
class PrometheusFormatter {
public:
    explicit PrometheusFormatter(size_t initial_buffer_size = 64 * 1024);
    ~PrometheusFormatter();

    // Formatting methods
    void reset();
    void add_help(const std::string& metric_name, const std::string& help_text);
    void add_type(const std::string& metric_name, const std::string& type);
    
    // Counter formatting
    void add_counter(const std::string& name, uint64_t value, 
                    const std::unordered_map<std::string, std::string>& labels = {});
    
    // Gauge formatting
    void add_gauge(const std::string& name, double value,
                  const std::unordered_map<std::string, std::string>& labels = {});
    
    // Histogram formatting
    void add_histogram(const std::string& name, 
                      const std::vector<std::pair<double, uint64_t>>& buckets,
                      uint64_t count, double sum,
                      const std::unordered_map<std::string, std::string>& labels = {});

    // Get formatted output
    std::string get_output() const;
    const char* get_buffer() const noexcept { return buffer_.get(); }
    size_t get_size() const noexcept { return size_; }

    // Performance optimization
    void reserve(size_t capacity);
    void set_global_labels(const std::unordered_map<std::string, std::string>& labels);

private:
    std::unique_ptr<char[]> buffer_;
    size_t capacity_;
    size_t size_;
    std::unordered_map<std::string, std::string> global_labels_;

    // Fast string operations
    void append_string(const std::string& str);
    void append_cstring(const char* str, size_t len);
    void append_number(uint64_t value);
    void append_double(double value, int precision = 6);
    void append_labels(const std::unordered_map<std::string, std::string>& labels);
    
    // Buffer management
    void ensure_capacity(size_t additional_size);
    void grow_buffer(size_t new_capacity);
    
    // SIMD-optimized operations
    void simd_append_digits(uint64_t value);
    void simd_escape_string(const char* input, size_t input_len);
};

/**
 * Lightweight HTTP server for metrics endpoint
 * Optimized for high throughput with minimal allocations
 */
class MetricsHttpServer {
public:
    struct Config {
        uint16_t port;
        std::string endpoint;
        size_t thread_pool_size = 4;
        size_t max_connections = 1000;
        std::chrono::seconds keep_alive_timeout{30};
    };

    explicit MetricsHttpServer(const Config& config);
    ~MetricsHttpServer();

    using RequestHandler = std::function<std::string(const std::string& path, 
                                                   const std::unordered_map<std::string, std::string>& headers)>;

    bool start(RequestHandler handler);
    void stop();
    bool is_running() const noexcept { return running_; }

    struct ServerStats {
        std::atomic<uint64_t> requests_handled{0};
        std::atomic<uint64_t> bytes_sent{0};
        std::atomic<uint64_t> active_connections{0};
        std::atomic<uint64_t> total_connections{0};
        std::atomic<uint64_t> request_duration_ns{0};
    };

    const ServerStats& get_stats() const noexcept { return stats_; }

private:
    Config config_;
    ServerStats stats_;
    std::atomic<bool> running_{false};

    // Platform-specific implementation
    class ServerImpl;
    std::unique_ptr<ServerImpl> impl_;
};

} // namespace monitor
} // namespace ultra