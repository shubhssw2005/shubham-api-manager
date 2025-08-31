#pragma once

#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <vector>
#include <cmath>
#include <immintrin.h>
#include <unordered_map>
#include <limits>
#include <string>
#include "stream_processor.hpp"

namespace ultra_cpp {
namespace stream {

/**
 * Anomaly detection algorithms
 */
enum class AnomalyAlgorithm {
    Z_SCORE,           // Statistical Z-score based detection
    MODIFIED_Z_SCORE,  // Modified Z-score using median
    IQR,              // Interquartile Range method
    ISOLATION_FOREST, // Simplified isolation forest
    EXPONENTIAL_SMOOTHING, // Exponential smoothing with confidence intervals
    SEASONAL_DECOMPOSITION  // Seasonal trend decomposition
};

/**
 * Anomaly severity levels
 */
enum class AnomalySeverity {
    LOW = 1,
    MEDIUM = 2,
    HIGH = 3,
    CRITICAL = 4
};

/**
 * Anomaly detection result
 */
struct AnomalyResult {
    uint64_t timestamp_ns;
    uint32_t event_type;
    uint32_t tenant_id;
    uint32_t user_id;
    AnomalyAlgorithm algorithm;
    AnomalySeverity severity;
    double anomaly_score;
    double threshold;
    double actual_value;
    double expected_value;
    std::string description;
    
    // Additional context
    double confidence_interval_lower;
    double confidence_interval_upper;
    uint64_t detection_latency_ns;
};

/**
 * Anomaly detection callback
 */
using AnomalyCallback = std::function<void(const AnomalyResult&)>;

/**
 * Configuration for anomaly detection
 */
struct AnomalyConfig {
    AnomalyAlgorithm algorithm = AnomalyAlgorithm::Z_SCORE;
    double threshold = 3.0;  // Standard deviations for Z-score
    size_t window_size = 1000;  // Number of samples for baseline
    std::chrono::milliseconds update_interval{100};  // Model update frequency
    bool enable_simd = true;
    bool enable_adaptive_threshold = true;
    double false_positive_rate = 0.01;  // Target false positive rate
    
    // Algorithm-specific parameters
    double iqr_multiplier = 1.5;  // For IQR method
    double smoothing_alpha = 0.3;  // For exponential smoothing
    size_t seasonal_period = 60;  // For seasonal decomposition (in samples)
};

/**
 * Statistical model for anomaly detection
 */
struct alignas(64) StatisticalModel {
    std::atomic<double> mean{0.0};
    std::atomic<double> variance{0.0};
    std::atomic<double> median{0.0};
    std::atomic<double> q1{0.0};  // First quartile
    std::atomic<double> q3{0.0};  // Third quartile
    std::atomic<uint64_t> sample_count{0};
    
    // Exponential smoothing state
    std::atomic<double> smoothed_value{0.0};
    std::atomic<double> trend{0.0};
    std::atomic<double> seasonal_component{0.0};
    
    // Adaptive threshold
    std::atomic<double> adaptive_threshold{0.0};
    std::atomic<double> false_positive_count{0.0};
    std::atomic<double> total_predictions{0.0};
    
    // Circular buffer for recent samples (lock-free)
    static constexpr size_t BUFFER_SIZE = 2048;
    std::atomic<double> sample_buffer[BUFFER_SIZE];
    std::atomic<size_t> buffer_head{0};
    std::atomic<bool> buffer_full{false};
    
    StatisticalModel() {
        for (size_t i = 0; i < BUFFER_SIZE; ++i) {
            sample_buffer[i].store(0.0);
        }
    }
    
    void add_sample(double value) {
        size_t head = buffer_head.load();
        sample_buffer[head].store(value);
        
        size_t next_head = (head + 1) % BUFFER_SIZE;
        buffer_head.store(next_head);
        
        if (next_head == 0) {
            buffer_full.store(true);
        }
        
        sample_count.fetch_add(1);
    }
    
    std::vector<double> get_recent_samples() const {
        std::vector<double> samples;
        size_t count = buffer_full.load() ? BUFFER_SIZE : buffer_head.load();
        samples.reserve(count);
        
        for (size_t i = 0; i < count; ++i) {
            samples.push_back(sample_buffer[i].load());
        }
        
        return samples;
    }
};

/**
 * Ultra-fast anomaly detector with microsecond detection latency
 */
class AnomalyDetector {
public:
    explicit AnomalyDetector(const AnomalyConfig& config = {});
    ~AnomalyDetector();
    
    // Non-copyable, non-movable
    AnomalyDetector(const AnomalyDetector&) = delete;
    AnomalyDetector& operator=(const AnomalyDetector&) = delete;
    
    /**
     * Start the anomaly detection engine
     */
    void start();
    
    /**
     * Stop the anomaly detection engine
     */
    void stop();
    
    /**
     * Register anomaly detection for specific event type and tenant
     */
    void register_detector(uint32_t event_type, uint32_t tenant_id, 
                          AnomalyCallback callback);
    
    /**
     * Process a single event for anomaly detection
     */
    void process_event(const StreamEvent& event, double value);
    
    /**
     * Process a batch of events with SIMD acceleration
     */
    void process_event_batch(const std::vector<const StreamEvent*>& events,
                           const std::vector<double>& values);
    
    /**
     * Update statistical models (called periodically)
     */
    void update_models();
    
    /**
     * Get current model statistics
     */
    std::unordered_map<uint64_t, StatisticalModel> get_model_stats() const;
    
    /**
     * Force model recalculation
     */
    void recalculate_models();
    
private:
    AnomalyConfig config_;
    std::atomic<bool> running_{false};
    
    // Statistical models per (event_type, tenant_id)
    std::unordered_map<uint64_t, std::unique_ptr<StatisticalModel>> models_;
    
    // Anomaly callbacks
    std::unordered_map<uint64_t, std::vector<AnomalyCallback>> callbacks_;
    
    // Background thread for model updates
    std::thread model_update_thread_;
    
    // Performance metrics
    std::atomic<uint64_t> anomalies_detected_{0};
    std::atomic<uint64_t> events_processed_{0};
    std::atomic<uint64_t> detection_latency_sum_ns_{0};
    
    // Private methods
    void model_update_loop();
    uint64_t make_model_key(uint32_t event_type, uint32_t tenant_id) const;
    StatisticalModel* get_or_create_model(uint32_t event_type, uint32_t tenant_id);
    
    // Anomaly detection algorithms
    bool detect_z_score_anomaly(const StatisticalModel* model, double value, 
                               double& anomaly_score) const;
    bool detect_modified_z_score_anomaly(const StatisticalModel* model, double value,
                                        double& anomaly_score) const;
    bool detect_iqr_anomaly(const StatisticalModel* model, double value,
                           double& anomaly_score) const;
    bool detect_exponential_smoothing_anomaly(StatisticalModel* model, double value,
                                             double& anomaly_score);
    
    // Statistical calculations with SIMD acceleration
    void simd_calculate_mean_variance(const std::vector<double>& samples,
                                    double& mean, double& variance) const;
    void simd_calculate_percentiles(std::vector<double>& samples,
                                  double& q1, double& median, double& q3) const;
    
    // Model update functions
    void update_z_score_model(StatisticalModel* model);
    void update_iqr_model(StatisticalModel* model);
    void update_exponential_smoothing_model(StatisticalModel* model, double new_value);
    
    // Adaptive threshold adjustment
    void adjust_adaptive_threshold(StatisticalModel* model, bool was_anomaly, bool actual_anomaly);
    
    // Utility functions
    uint64_t get_current_time_ns() const;
    AnomalySeverity calculate_severity(double anomaly_score, double threshold) const;
    std::string generate_description(AnomalyAlgorithm algorithm, double score, double threshold) const;
};

} // namespace stream
} // namespace ultra_cpp