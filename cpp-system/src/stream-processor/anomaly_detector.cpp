#include "../../include/stream-processor/anomaly_detector.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace ultra_cpp {
namespace stream {

AnomalyDetector::AnomalyDetector(const AnomalyConfig& config)
    : config_(config) {
}

AnomalyDetector::~AnomalyDetector() {
    if (running_.load()) {
        stop();
    }
}

void AnomalyDetector::start() {
    if (running_.exchange(true)) {
        return; // Already running
    }
    
    // Start model update thread
    model_update_thread_ = std::thread(&AnomalyDetector::model_update_loop, this);
}

void AnomalyDetector::stop() {
    running_.store(false);
    
    if (model_update_thread_.joinable()) {
        model_update_thread_.join();
    }
}

void AnomalyDetector::register_detector(uint32_t event_type, uint32_t tenant_id, 
                                       AnomalyCallback callback) {
    uint64_t key = make_model_key(event_type, tenant_id);
    callbacks_[key].push_back(std::move(callback));
}

void AnomalyDetector::process_event(const StreamEvent& event, double value) {
    auto start_time = get_current_time_ns();
    
    StatisticalModel* model = get_or_create_model(event.event_type, event.tenant_id);
    if (!model) {
        return;
    }
    
    // Add sample to model
    model->add_sample(value);
    
    // Check for anomaly if we have enough samples
    if (model->sample_count.load() >= config_.window_size) {
        double anomaly_score = 0.0;
        bool is_anomaly = false;
        
        switch (config_.algorithm) {
            case AnomalyAlgorithm::Z_SCORE:
                is_anomaly = detect_z_score_anomaly(model, value, anomaly_score);
                break;
            case AnomalyAlgorithm::MODIFIED_Z_SCORE:
                is_anomaly = detect_modified_z_score_anomaly(model, value, anomaly_score);
                break;
            case AnomalyAlgorithm::IQR:
                is_anomaly = detect_iqr_anomaly(model, value, anomaly_score);
                break;
            case AnomalyAlgorithm::EXPONENTIAL_SMOOTHING:
                is_anomaly = detect_exponential_smoothing_anomaly(model, value, anomaly_score);
                break;
            default:
                is_anomaly = detect_z_score_anomaly(model, value, anomaly_score);
                break;
        }
        
        if (is_anomaly) {
            auto end_time = get_current_time_ns();
            uint64_t detection_latency = end_time - start_time;
            
            // Create anomaly result
            AnomalyResult result;
            result.timestamp_ns = event.timestamp_ns;
            result.event_type = event.event_type;
            result.tenant_id = event.tenant_id;
            result.user_id = event.user_id;
            result.algorithm = config_.algorithm;
            result.anomaly_score = anomaly_score;
            result.threshold = config_.enable_adaptive_threshold ? 
                              model->adaptive_threshold.load() : config_.threshold;
            result.actual_value = value;
            result.expected_value = model->mean.load();
            result.severity = calculate_severity(anomaly_score, result.threshold);
            result.description = generate_description(config_.algorithm, anomaly_score, result.threshold);
            result.detection_latency_ns = detection_latency;
            
            // Set confidence intervals
            double std_dev = std::sqrt(model->variance.load());
            result.confidence_interval_lower = model->mean.load() - 2 * std_dev;
            result.confidence_interval_upper = model->mean.load() + 2 * std_dev;
            
            // Update metrics
            anomalies_detected_.fetch_add(1);
            detection_latency_sum_ns_.fetch_add(detection_latency);
            
            // Call callbacks
            uint64_t key = make_model_key(event.event_type, event.tenant_id);
            auto callback_it = callbacks_.find(key);
            if (callback_it != callbacks_.end()) {
                for (const auto& callback : callback_it->second) {
                    callback(result);
                }
            }
        }
        
        // Update adaptive threshold if enabled
        if (config_.enable_adaptive_threshold) {
            adjust_adaptive_threshold(model, is_anomaly, false); // Assume no ground truth for now
        }
    }
    
    events_processed_.fetch_add(1);
}

void AnomalyDetector::process_event_batch(const std::vector<const StreamEvent*>& events,
                                         const std::vector<double>& values) {
    if (events.size() != values.size()) {
        return; // Mismatched sizes
    }
    
    // Group events by model for efficient processing
    std::unordered_map<uint64_t, std::vector<std::pair<size_t, double>>> events_by_model;
    
    for (size_t i = 0; i < events.size(); ++i) {
        uint64_t key = make_model_key(events[i]->event_type, events[i]->tenant_id);
        events_by_model[key].emplace_back(i, values[i]);
    }
    
    // Process each model's events
    for (const auto& [model_key, model_events] : events_by_model) {
        // Extract event type and tenant ID from key
        uint32_t event_type = static_cast<uint32_t>(model_key >> 32);
        uint32_t tenant_id = static_cast<uint32_t>(model_key & 0xFFFFFFFF);
        
        StatisticalModel* model = get_or_create_model(event_type, tenant_id);
        if (!model) continue;
        
        // SIMD-accelerated batch processing if enabled and batch is large enough
        if (config_.enable_simd && model_events.size() >= 4) {
            std::vector<double> batch_values;
            batch_values.reserve(model_events.size());
            for (const auto& [idx, value] : model_events) {
                batch_values.push_back(value);
            }
            
            // Update model statistics with SIMD
            double mean = 0.0, variance = 0.0;
            simd_calculate_mean_variance(batch_values, mean, variance);
            
            // Update model atomically
            model->mean.store(mean);
            model->variance.store(variance);
            
            // Add all samples to model
            for (double value : batch_values) {
                model->add_sample(value);
            }
            
            // Check for anomalies in batch
            for (const auto& [idx, value] : model_events) {
                process_event(*events[idx], value);
            }
        } else {
            // Fallback to individual processing
            for (const auto& [idx, value] : model_events) {
                process_event(*events[idx], value);
            }
        }
    }
}

void AnomalyDetector::update_models() {
    for (auto& [key, model] : models_) {
        if (!model) continue;
        
        switch (config_.algorithm) {
            case AnomalyAlgorithm::Z_SCORE:
            case AnomalyAlgorithm::MODIFIED_Z_SCORE:
                update_z_score_model(model.get());
                break;
            case AnomalyAlgorithm::IQR:
                update_iqr_model(model.get());
                break;
            case AnomalyAlgorithm::EXPONENTIAL_SMOOTHING:
                // Exponential smoothing is updated per event
                break;
        }
    }
}

std::unordered_map<uint64_t, StatisticalModel> AnomalyDetector::get_model_stats() const {
    std::unordered_map<uint64_t, StatisticalModel> stats;
    
    for (const auto& [key, model] : models_) {
        if (model) {
            stats[key] = *model; // Copy the model
        }
    }
    
    return stats;
}

void AnomalyDetector::recalculate_models() {
    update_models();
}

void AnomalyDetector::model_update_loop() {
    while (running_.load()) {
        update_models();
        std::this_thread::sleep_for(config_.update_interval);
    }
}

uint64_t AnomalyDetector::make_model_key(uint32_t event_type, uint32_t tenant_id) const {
    return (static_cast<uint64_t>(event_type) << 32) | tenant_id;
}

StatisticalModel* AnomalyDetector::get_or_create_model(uint32_t event_type, uint32_t tenant_id) {
    uint64_t key = make_model_key(event_type, tenant_id);
    
    auto it = models_.find(key);
    if (it != models_.end()) {
        return it->second.get();
    }
    
    // Create new model
    auto new_model = std::make_unique<StatisticalModel>();
    StatisticalModel* ptr = new_model.get();
    models_[key] = std::move(new_model);
    
    return ptr;
}

bool AnomalyDetector::detect_z_score_anomaly(const StatisticalModel* model, double value, 
                                            double& anomaly_score) const {
    double mean = model->mean.load();
    double variance = model->variance.load();
    
    if (variance <= 0.0) {
        anomaly_score = 0.0;
        return false;
    }
    
    double std_dev = std::sqrt(variance);
    anomaly_score = std::abs(value - mean) / std_dev;
    
    double threshold = config_.enable_adaptive_threshold ? 
                      model->adaptive_threshold.load() : config_.threshold;
    
    return anomaly_score > threshold;
}

bool AnomalyDetector::detect_modified_z_score_anomaly(const StatisticalModel* model, double value,
                                                     double& anomaly_score) const {
    double median = model->median.load();
    
    // Calculate MAD (Median Absolute Deviation) from recent samples
    auto samples = model->get_recent_samples();
    if (samples.empty()) {
        anomaly_score = 0.0;
        return false;
    }
    
    std::vector<double> deviations;
    deviations.reserve(samples.size());
    for (double sample : samples) {
        deviations.push_back(std::abs(sample - median));
    }
    
    std::sort(deviations.begin(), deviations.end());
    double mad = deviations[deviations.size() / 2];
    
    if (mad == 0.0) {
        anomaly_score = 0.0;
        return false;
    }
    
    anomaly_score = 0.6745 * (value - median) / mad;
    
    double threshold = config_.enable_adaptive_threshold ? 
                      model->adaptive_threshold.load() : config_.threshold;
    
    return std::abs(anomaly_score) > threshold;
}

bool AnomalyDetector::detect_iqr_anomaly(const StatisticalModel* model, double value,
                                        double& anomaly_score) const {
    double q1 = model->q1.load();
    double q3 = model->q3.load();
    double iqr = q3 - q1;
    
    if (iqr <= 0.0) {
        anomaly_score = 0.0;
        return false;
    }
    
    double lower_bound = q1 - config_.iqr_multiplier * iqr;
    double upper_bound = q3 + config_.iqr_multiplier * iqr;
    
    if (value < lower_bound) {
        anomaly_score = (lower_bound - value) / iqr;
    } else if (value > upper_bound) {
        anomaly_score = (value - upper_bound) / iqr;
    } else {
        anomaly_score = 0.0;
        return false;
    }
    
    return true; // Any value outside IQR bounds is considered anomalous
}

bool AnomalyDetector::detect_exponential_smoothing_anomaly(StatisticalModel* model, double value,
                                                          double& anomaly_score) {
    double smoothed = model->smoothed_value.load();
    double trend = model->trend.load();
    
    // Update exponential smoothing
    update_exponential_smoothing_model(model, value);
    
    // Calculate prediction error
    double predicted = smoothed + trend;
    double error = std::abs(value - predicted);
    
    // Use adaptive threshold based on recent prediction errors
    double threshold = config_.enable_adaptive_threshold ? 
                      model->adaptive_threshold.load() : config_.threshold;
    
    anomaly_score = error / (threshold + 1e-10); // Avoid division by zero
    
    return anomaly_score > 1.0;
}

void AnomalyDetector::simd_calculate_mean_variance(const std::vector<double>& samples,
                                                  double& mean, double& variance) const {
    if (samples.empty()) {
        mean = variance = 0.0;
        return;
    }
    
    if (samples.size() < 4) {
        // Fallback to regular calculation
        mean = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
        
        double sum_sq_diff = 0.0;
        for (double sample : samples) {
            double diff = sample - mean;
            sum_sq_diff += diff * diff;
        }
        variance = samples.size() > 1 ? sum_sq_diff / (samples.size() - 1) : 0.0;
        return;
    }
    
    // SIMD calculation
    size_t simd_count = (samples.size() / 4) * 4;
    __m256d sum_vec = _mm256_setzero_pd();
    
    // Calculate sum with SIMD
    for (size_t i = 0; i < simd_count; i += 4) {
        __m256d vals = _mm256_loadu_pd(&samples[i]);
        sum_vec = _mm256_add_pd(sum_vec, vals);
    }
    
    // Horizontal sum
    double temp[4];
    _mm256_storeu_pd(temp, sum_vec);
    double sum = temp[0] + temp[1] + temp[2] + temp[3];
    
    // Add remaining elements
    for (size_t i = simd_count; i < samples.size(); ++i) {
        sum += samples[i];
    }
    
    mean = sum / samples.size();
    
    // Calculate variance with SIMD
    __m256d mean_vec = _mm256_set1_pd(mean);
    __m256d sum_sq_vec = _mm256_setzero_pd();
    
    for (size_t i = 0; i < simd_count; i += 4) {
        __m256d vals = _mm256_loadu_pd(&samples[i]);
        __m256d diff = _mm256_sub_pd(vals, mean_vec);
        __m256d sq_diff = _mm256_mul_pd(diff, diff);
        sum_sq_vec = _mm256_add_pd(sum_sq_vec, sq_diff);
    }
    
    // Horizontal sum of squares
    _mm256_storeu_pd(temp, sum_sq_vec);
    double sum_sq = temp[0] + temp[1] + temp[2] + temp[3];
    
    // Add remaining elements
    for (size_t i = simd_count; i < samples.size(); ++i) {
        double diff = samples[i] - mean;
        sum_sq += diff * diff;
    }
    
    variance = samples.size() > 1 ? sum_sq / (samples.size() - 1) : 0.0;
}

void AnomalyDetector::simd_calculate_percentiles(std::vector<double>& samples,
                                               double& q1, double& median, double& q3) const {
    if (samples.empty()) {
        q1 = median = q3 = 0.0;
        return;
    }
    
    std::sort(samples.begin(), samples.end());
    
    size_t n = samples.size();
    size_t q1_idx = n / 4;
    size_t median_idx = n / 2;
    size_t q3_idx = 3 * n / 4;
    
    q1 = samples[q1_idx];
    median = samples[median_idx];
    q3 = samples[q3_idx];
}

void AnomalyDetector::update_z_score_model(StatisticalModel* model) {
    auto samples = model->get_recent_samples();
    if (samples.empty()) {
        return;
    }
    
    double mean, variance;
    simd_calculate_mean_variance(samples, mean, variance);
    
    model->mean.store(mean);
    model->variance.store(variance);
}

void AnomalyDetector::update_iqr_model(StatisticalModel* model) {
    auto samples = model->get_recent_samples();
    if (samples.empty()) {
        return;
    }
    
    double q1, median, q3;
    simd_calculate_percentiles(samples, q1, median, q3);
    
    model->q1.store(q1);
    model->median.store(median);
    model->q3.store(q3);
}

void AnomalyDetector::update_exponential_smoothing_model(StatisticalModel* model, double new_value) {
    double alpha = config_.smoothing_alpha;
    double current_smoothed = model->smoothed_value.load();
    double current_trend = model->trend.load();
    
    // Update smoothed value
    double new_smoothed = alpha * new_value + (1 - alpha) * (current_smoothed + current_trend);
    
    // Update trend
    double new_trend = alpha * (new_smoothed - current_smoothed) + (1 - alpha) * current_trend;
    
    model->smoothed_value.store(new_smoothed);
    model->trend.store(new_trend);
}

void AnomalyDetector::adjust_adaptive_threshold(StatisticalModel* model, bool was_anomaly, bool actual_anomaly) {
    // Simple adaptive threshold adjustment based on false positive rate
    double current_threshold = model->adaptive_threshold.load();
    double total_predictions = model->total_predictions.fetch_add(1) + 1;
    
    if (was_anomaly && !actual_anomaly) {
        // False positive
        model->false_positive_count.fetch_add(1);
    }
    
    double false_positive_rate = model->false_positive_count.load() / total_predictions;
    
    // Adjust threshold to maintain target false positive rate
    if (false_positive_rate > config_.false_positive_rate) {
        // Too many false positives, increase threshold
        current_threshold *= 1.01;
    } else if (false_positive_rate < config_.false_positive_rate * 0.5) {
        // Too few detections, decrease threshold
        current_threshold *= 0.99;
    }
    
    // Clamp threshold to reasonable bounds
    current_threshold = std::max(1.0, std::min(10.0, current_threshold));
    model->adaptive_threshold.store(current_threshold);
}

uint64_t AnomalyDetector::get_current_time_ns() const {
    return std::chrono::high_resolution_clock::now().time_since_epoch().count();
}

AnomalySeverity AnomalyDetector::calculate_severity(double anomaly_score, double threshold) const {
    double ratio = anomaly_score / threshold;
    
    if (ratio >= 3.0) return AnomalySeverity::CRITICAL;
    if (ratio >= 2.0) return AnomalySeverity::HIGH;
    if (ratio >= 1.5) return AnomalySeverity::MEDIUM;
    return AnomalySeverity::LOW;
}

std::string AnomalyDetector::generate_description(AnomalyAlgorithm algorithm, double score, double threshold) const {
    std::string algo_name;
    switch (algorithm) {
        case AnomalyAlgorithm::Z_SCORE: algo_name = "Z-Score"; break;
        case AnomalyAlgorithm::MODIFIED_Z_SCORE: algo_name = "Modified Z-Score"; break;
        case AnomalyAlgorithm::IQR: algo_name = "IQR"; break;
        case AnomalyAlgorithm::ISOLATION_FOREST: algo_name = "Isolation Forest"; break;
        case AnomalyAlgorithm::EXPONENTIAL_SMOOTHING: algo_name = "Exponential Smoothing"; break;
        case AnomalyAlgorithm::SEASONAL_DECOMPOSITION: algo_name = "Seasonal Decomposition"; break;
    }
    
    return algo_name + " anomaly detected (score: " + std::to_string(score) + 
           ", threshold: " + std::to_string(threshold) + ")";
}

} // namespace stream
} // namespace ultra_cpp