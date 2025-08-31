#pragma once

#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>
#include <immintrin.h>
#include <limits>
#include "stream_processor.hpp"

namespace ultra_cpp {
namespace stream {

/**
 * Window types for aggregation
 */
enum class WindowType {
    SLIDING,    // Continuous sliding window
    TUMBLING,   // Non-overlapping fixed windows
    SESSION     // Session-based windows with timeout
};

/**
 * Aggregation functions
 */
enum class AggregationType {
    COUNT,
    SUM,
    AVERAGE,
    MIN,
    MAX,
    PERCENTILE_95,
    PERCENTILE_99,
    STANDARD_DEVIATION
};

/**
 * Window configuration
 */
struct WindowConfig {
    WindowType type = WindowType::SLIDING;
    std::chrono::milliseconds window_size{1000};  // 1 second
    std::chrono::milliseconds slide_interval{100}; // 100ms for sliding windows
    std::chrono::milliseconds session_timeout{5000}; // 5 seconds for session windows
    size_t max_elements = 10000;  // Maximum elements per window
    bool enable_simd = true;
};

/**
 * Aggregation result
 */
struct AggregationResult {
    uint64_t window_start_ns;
    uint64_t window_end_ns;
    uint32_t event_type;
    uint32_t tenant_id;
    AggregationType aggregation_type;
    double value;
    uint64_t count;
    
    // Additional statistics
    double min_value;
    double max_value;
    double sum_value;
    double sum_squares;  // For variance calculation
};

/**
 * Callback for aggregation results
 */
using AggregationCallback = std::function<void(const AggregationResult&)>;

/**
 * Window state for efficient aggregation
 */
struct alignas(64) WindowState {
    std::atomic<uint64_t> count{0};
    std::atomic<uint64_t> sum_int{0};  // For integer sums
    std::atomic<double> sum_double{0.0};  // For floating point sums
    std::atomic<double> sum_squares{0.0};
    std::atomic<double> min_value{std::numeric_limits<double>::max()};
    std::atomic<double> max_value{std::numeric_limits<double>::lowest()};
    std::atomic<uint64_t> window_start_ns{0};
    std::atomic<uint64_t> window_end_ns{0};
    
    // Percentile calculation (lock-free histogram)
    static constexpr size_t HISTOGRAM_BUCKETS = 1024;
    std::atomic<uint64_t> histogram[HISTOGRAM_BUCKETS];
    
    WindowState() {
        for (size_t i = 0; i < HISTOGRAM_BUCKETS; ++i) {
            histogram[i].store(0);
        }
    }
    
    void reset() {
        count.store(0);
        sum_int.store(0);
        sum_double.store(0.0);
        sum_squares.store(0.0);
        min_value.store(std::numeric_limits<double>::max());
        max_value.store(std::numeric_limits<double>::lowest());
        window_start_ns.store(0);
        window_end_ns.store(0);
        
        for (size_t i = 0; i < HISTOGRAM_BUCKETS; ++i) {
            histogram[i].store(0);
        }
    }
};

/**
 * High-performance windowed aggregation engine with SIMD acceleration
 */
class WindowedAggregator {
public:
    explicit WindowedAggregator(const WindowConfig& config = {});
    ~WindowedAggregator();
    
    // Non-copyable, non-movable
    WindowedAggregator(const WindowedAggregator&) = delete;
    WindowedAggregator& operator=(const WindowedAggregator&) = delete;
    
    /**
     * Start the aggregation engine
     */
    void start();
    
    /**
     * Stop the aggregation engine
     */
    void stop();
    
    /**
     * Add aggregation for specific event type and tenant
     */
    void add_aggregation(uint32_t event_type, uint32_t tenant_id,
                        AggregationType agg_type, AggregationCallback callback);
    
    /**
     * Process an event for aggregation
     */
    void process_event(const StreamEvent& event, double value);
    
    /**
     * Process a batch of events with SIMD acceleration
     */
    void process_event_batch(const std::vector<const StreamEvent*>& events,
                           const std::vector<double>& values);
    
    /**
     * Force window computation (useful for testing)
     */
    void flush_windows();
    
    /**
     * Get current window statistics
     */
    std::vector<AggregationResult> get_current_results();
    
private:
    WindowConfig config_;
    std::atomic<bool> running_{false};
    
    // Window management
    struct WindowKey {
        uint32_t event_type;
        uint32_t tenant_id;
        uint64_t window_id;
        
        bool operator==(const WindowKey& other) const {
            return event_type == other.event_type && 
                   tenant_id == other.tenant_id && 
                   window_id == other.window_id;
        }
    };
    
    struct WindowKeyHash {
        size_t operator()(const WindowKey& key) const {
            return std::hash<uint64_t>{}(
                (static_cast<uint64_t>(key.event_type) << 32) | 
                key.tenant_id) ^ std::hash<uint64_t>{}(key.window_id);
        }
    };
    
    // Window states indexed by key
    std::unordered_map<WindowKey, std::unique_ptr<WindowState>, WindowKeyHash> windows_;
    
    // Aggregation configurations
    struct AggregationConfig {
        AggregationType type;
        AggregationCallback callback;
    };
    
    std::unordered_map<uint64_t, std::vector<AggregationConfig>> aggregations_;
    
    // Background thread for window management
    std::thread window_manager_thread_;
    
    // Private methods
    void window_manager_loop();
    uint64_t get_window_id(uint64_t timestamp_ns) const;
    WindowKey create_window_key(uint32_t event_type, uint32_t tenant_id, uint64_t timestamp_ns);
    WindowState* get_or_create_window(const WindowKey& key);
    void compute_window_results(const WindowKey& key, WindowState* state);
    void cleanup_expired_windows();
    
    // SIMD-accelerated aggregation functions
    void simd_sum_batch(const double* values, size_t count, double& sum);
    void simd_min_max_batch(const double* values, size_t count, double& min_val, double& max_val);
    void simd_variance_batch(const double* values, size_t count, double mean, double& variance);
    
    // Percentile calculation
    double calculate_percentile(const WindowState* state, double percentile) const;
    size_t get_histogram_bucket(double value) const;
    
    // Utility functions
    uint64_t get_current_time_ns() const;
    uint64_t make_aggregation_key(uint32_t event_type, uint32_t tenant_id) const;
};

} // namespace stream
} // namespace ultra_cpp