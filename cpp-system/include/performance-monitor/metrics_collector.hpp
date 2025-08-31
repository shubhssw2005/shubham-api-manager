#pragma once

#include <atomic>
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace ultra {
namespace monitor {

/**
 * Lock-free metrics collection system with histogram and percentile support
 * Optimized for ultra-low latency with minimal collection overhead
 */
class MetricsCollector {
public:
    struct Config {
        size_t max_metrics = 10000;
        size_t histogram_buckets = 64;
        bool enable_percentiles = true;
        double percentiles[5] = {0.5, 0.95, 0.99, 0.999, 0.9999};
    };

    explicit MetricsCollector(const Config& config);
    ~MetricsCollector();

    // Metric types
    enum class MetricType {
        COUNTER,
        GAUGE,
        HISTOGRAM,
        TIMER
    };

    // Lock-free counter operations
    void increment_counter(const std::string& name, uint64_t value = 1) noexcept;
    uint64_t get_counter_value(const std::string& name) const noexcept;

    // Lock-free gauge operations
    void set_gauge(const std::string& name, double value) noexcept;
    double get_gauge_value(const std::string& name) const noexcept;

    // Lock-free histogram operations
    void observe_histogram(const std::string& name, double value) noexcept;
    void record_timing(const std::string& name, uint64_t duration_ns) noexcept;

    // Percentile calculations
    struct PercentileData {
        double p50 = 0.0;
        double p95 = 0.0;
        double p99 = 0.0;
        double p999 = 0.0;
        double p9999 = 0.0;
        double min = 0.0;
        double max = 0.0;
        double mean = 0.0;
        double stddev = 0.0;
        uint64_t count = 0;
        uint64_t sum = 0;
    };

    PercentileData calculate_percentiles(const std::string& name) const;

    // Histogram data
    struct HistogramBucket {
        double upper_bound;
        std::atomic<uint64_t> count{0};
    };

    struct HistogramData {
        std::vector<HistogramBucket> buckets;
        std::atomic<uint64_t> total_count{0};
        std::atomic<double> sum{0.0};
        std::atomic<double> sum_squares{0.0};
        std::atomic<double> min_value{std::numeric_limits<double>::max()};
        std::atomic<double> max_value{std::numeric_limits<double>::lowest()};
    };

    const HistogramData* get_histogram_data(const std::string& name) const;

    // Metric enumeration
    std::vector<std::string> get_counter_names() const;
    std::vector<std::string> get_gauge_names() const;
    std::vector<std::string> get_histogram_names() const;

    // Performance statistics
    struct CollectorStats {
        std::atomic<uint64_t> total_operations{0};
        std::atomic<uint64_t> counter_operations{0};
        std::atomic<uint64_t> gauge_operations{0};
        std::atomic<uint64_t> histogram_operations{0};
        std::atomic<uint64_t> collection_overhead_ns{0};
        std::atomic<uint64_t> memory_usage_bytes{0};
    };

    const CollectorStats& get_stats() const noexcept { return stats_; }

    // Memory management
    void reset_all_metrics();
    void cleanup_unused_metrics();
    size_t get_memory_usage() const noexcept;

private:
    Config config_;
    CollectorStats stats_;

    // Lock-free counter storage
    struct alignas(64) CounterEntry {
        std::atomic<uint64_t> value{0};
        std::atomic<uint64_t> last_access{0};
        char padding[64 - 2 * sizeof(std::atomic<uint64_t>)];
    };

    // Lock-free gauge storage
    struct alignas(64) GaugeEntry {
        std::atomic<double> value{0.0};
        std::atomic<uint64_t> last_access{0};
        char padding[64 - sizeof(std::atomic<double>) - sizeof(std::atomic<uint64_t>)];
    };

    // Hash table for metric lookup (lock-free)
    template<typename T>
    class LockFreeHashMap {
    public:
        explicit LockFreeHashMap(size_t capacity);
        ~LockFreeHashMap();

        T* get_or_create(const std::string& key);
        T* get(const std::string& key) const;
        std::vector<std::string> get_all_keys() const;
        void clear();

    private:
        struct Entry {
            std::atomic<uint64_t> key_hash{0};
            std::atomic<T*> value{nullptr};
            std::string key;
            std::atomic<bool> valid{false};
        };

        std::vector<Entry> entries_;
        size_t capacity_;
        std::atomic<size_t> size_{0};

        uint64_t hash_string(const std::string& str) const noexcept;
        size_t find_slot(uint64_t hash) const noexcept;
    };

    LockFreeHashMap<CounterEntry> counters_;
    LockFreeHashMap<GaugeEntry> gauges_;
    LockFreeHashMap<HistogramData> histograms_;

    // Histogram bucket generation
    std::vector<double> generate_exponential_buckets(double start, double factor, size_t count) const;
    std::vector<double> generate_linear_buckets(double start, double width, size_t count) const;

    // Utility functions
    uint64_t get_timestamp_ns() const noexcept;
    void update_stats(MetricType type) noexcept;
};

/**
 * SIMD-accelerated percentile calculation
 * Uses vectorized operations for fast histogram processing
 */
class SIMDPercentileCalculator {
public:
    static MetricsCollector::PercentileData calculate_percentiles(
        const std::vector<MetricsCollector::HistogramBucket>& buckets,
        const double* percentiles, size_t percentile_count);

private:
    static void vectorized_cumsum(const uint64_t* counts, uint64_t* cumulative, size_t size);
    static double interpolate_percentile(const std::vector<MetricsCollector::HistogramBucket>& buckets,
                                       const uint64_t* cumulative, uint64_t total_count, double percentile);
};

} // namespace monitor
} // namespace ultra