#include "../../include/stream-processor/windowed_aggregator.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace ultra_cpp {
namespace stream {

WindowedAggregator::WindowedAggregator(const WindowConfig& config)
    : config_(config) {
}

WindowedAggregator::~WindowedAggregator() {
    if (running_.load()) {
        stop();
    }
}

void WindowedAggregator::start() {
    if (running_.exchange(true)) {
        return; // Already running
    }
    
    // Start window manager thread
    window_manager_thread_ = std::thread(&WindowedAggregator::window_manager_loop, this);
}

void WindowedAggregator::stop() {
    running_.store(false);
    
    if (window_manager_thread_.joinable()) {
        window_manager_thread_.join();
    }
}

void WindowedAggregator::add_aggregation(uint32_t event_type, uint32_t tenant_id,
                                        AggregationType agg_type, AggregationCallback callback) {
    uint64_t key = make_aggregation_key(event_type, tenant_id);
    aggregations_[key].push_back({agg_type, std::move(callback)});
}

void WindowedAggregator::process_event(const StreamEvent& event, double value) {
    WindowKey window_key = create_window_key(event.event_type, event.tenant_id, event.timestamp_ns);
    WindowState* state = get_or_create_window(window_key);
    
    if (!state) {
        return; // Failed to create window
    }
    
    // Update window state atomically
    state->count.fetch_add(1);
    
    // Update sum (both integer and double versions)
    if (std::floor(value) == value && value >= 0 && value <= UINT64_MAX) {
        state->sum_int.fetch_add(static_cast<uint64_t>(value));
    }
    
    // Atomic update of double sum using compare-and-swap
    double expected = state->sum_double.load();
    while (!state->sum_double.compare_exchange_weak(expected, expected + value)) {
        // Retry until successful
    }
    
    // Update sum of squares for variance calculation
    double squares_expected = state->sum_squares.load();
    while (!state->sum_squares.compare_exchange_weak(squares_expected, squares_expected + (value * value))) {
        // Retry until successful
    }
    
    // Update min value
    double current_min = state->min_value.load();
    while (value < current_min && !state->min_value.compare_exchange_weak(current_min, value)) {
        // Retry until successful
    }
    
    // Update max value
    double current_max = state->max_value.load();
    while (value > current_max && !state->max_value.compare_exchange_weak(current_max, value)) {
        // Retry until successful
    }
    
    // Update histogram for percentile calculation
    size_t bucket = get_histogram_bucket(value);
    if (bucket < WindowState::HISTOGRAM_BUCKETS) {
        state->histogram[bucket].fetch_add(1);
    }
    
    // Set window boundaries if not set
    uint64_t window_start = get_window_id(event.timestamp_ns) * config_.window_size.count() * 1000000ULL;
    uint64_t window_end = window_start + config_.window_size.count() * 1000000ULL;
    
    state->window_start_ns.compare_exchange_strong(window_start, window_start);
    state->window_end_ns.compare_exchange_strong(window_end, window_end);
}

void WindowedAggregator::process_event_batch(const std::vector<const StreamEvent*>& events,
                                           const std::vector<double>& values) {
    if (events.size() != values.size()) {
        return; // Mismatched sizes
    }
    
    // SIMD-accelerated batch processing
    if (config_.enable_simd && events.size() >= 4) {
        // Group events by window for efficient processing
        std::unordered_map<WindowKey, std::vector<std::pair<size_t, double>>, WindowKeyHash> events_by_window;
        
        for (size_t i = 0; i < events.size(); ++i) {
            WindowKey key = create_window_key(events[i]->event_type, events[i]->tenant_id, events[i]->timestamp_ns);
            events_by_window[key].emplace_back(i, values[i]);
        }
        
        // Process each window's events
        for (const auto& [window_key, window_events] : events_by_window) {
            WindowState* state = get_or_create_window(window_key);
            if (!state) continue;
            
            // Extract values for SIMD processing
            std::vector<double> window_values;
            window_values.reserve(window_events.size());
            for (const auto& [idx, value] : window_events) {
                window_values.push_back(value);
            }
            
            // SIMD aggregation
            double sum = 0.0;
            simd_sum_batch(window_values.data(), window_values.size(), sum);
            
            double min_val = std::numeric_limits<double>::max();
            double max_val = std::numeric_limits<double>::lowest();
            simd_min_max_batch(window_values.data(), window_values.size(), min_val, max_val);
            
            // Update window state
            state->count.fetch_add(window_values.size());
            
            double expected_sum = state->sum_double.load();
            while (!state->sum_double.compare_exchange_weak(expected_sum, expected_sum + sum)) {
                // Retry
            }
            
            // Update min/max atomically
            double current_min = state->min_value.load();
            while (min_val < current_min && !state->min_value.compare_exchange_weak(current_min, min_val)) {
                // Retry
            }
            
            double current_max = state->max_value.load();
            while (max_val > current_max && !state->max_value.compare_exchange_weak(current_max, max_val)) {
                // Retry
            }
            
            // Update histogram
            for (double value : window_values) {
                size_t bucket = get_histogram_bucket(value);
                if (bucket < WindowState::HISTOGRAM_BUCKETS) {
                    state->histogram[bucket].fetch_add(1);
                }
            }
        }
    } else {
        // Fallback to individual processing
        for (size_t i = 0; i < events.size(); ++i) {
            process_event(*events[i], values[i]);
        }
    }
}

void WindowedAggregator::flush_windows() {
    uint64_t current_time = get_current_time_ns();
    
    for (auto& [key, state] : windows_) {
        if (state && state->window_end_ns.load() <= current_time) {
            compute_window_results(key, state.get());
        }
    }
    
    cleanup_expired_windows();
}

std::vector<AggregationResult> WindowedAggregator::get_current_results() {
    std::vector<AggregationResult> results;
    
    for (const auto& [key, state] : windows_) {
        if (!state) continue;
        
        uint64_t agg_key = make_aggregation_key(key.event_type, key.tenant_id);
        auto agg_it = aggregations_.find(agg_key);
        if (agg_it == aggregations_.end()) continue;
        
        for (const auto& agg_config : agg_it->second) {
            AggregationResult result;
            result.window_start_ns = state->window_start_ns.load();
            result.window_end_ns = state->window_end_ns.load();
            result.event_type = key.event_type;
            result.tenant_id = key.tenant_id;
            result.aggregation_type = agg_config.type;
            result.count = state->count.load();
            result.min_value = state->min_value.load();
            result.max_value = state->max_value.load();
            result.sum_value = state->sum_double.load();
            result.sum_squares = state->sum_squares.load();
            
            // Calculate specific aggregation value
            switch (agg_config.type) {
                case AggregationType::COUNT:
                    result.value = static_cast<double>(result.count);
                    break;
                case AggregationType::SUM:
                    result.value = result.sum_value;
                    break;
                case AggregationType::AVERAGE:
                    result.value = result.count > 0 ? result.sum_value / result.count : 0.0;
                    break;
                case AggregationType::MIN:
                    result.value = result.min_value;
                    break;
                case AggregationType::MAX:
                    result.value = result.max_value;
                    break;
                case AggregationType::PERCENTILE_95:
                    result.value = calculate_percentile(state.get(), 0.95);
                    break;
                case AggregationType::PERCENTILE_99:
                    result.value = calculate_percentile(state.get(), 0.99);
                    break;
                case AggregationType::STANDARD_DEVIATION: {
                    double mean = result.count > 0 ? result.sum_value / result.count : 0.0;
                    double variance = result.count > 1 ? 
                        (result.sum_squares - (result.sum_value * result.sum_value / result.count)) / (result.count - 1) : 0.0;
                    result.value = std::sqrt(std::max(0.0, variance));
                    break;
                }
            }
            
            results.push_back(result);
        }
    }
    
    return results;
}

void WindowedAggregator::window_manager_loop() {
    while (running_.load()) {
        flush_windows();
        
        // Sleep based on slide interval
        std::this_thread::sleep_for(config_.slide_interval);
    }
}

uint64_t WindowedAggregator::get_window_id(uint64_t timestamp_ns) const {
    uint64_t window_size_ns = config_.window_size.count() * 1000000ULL; // Convert ms to ns
    
    switch (config_.type) {
        case WindowType::TUMBLING:
            return timestamp_ns / window_size_ns;
        case WindowType::SLIDING: {
            uint64_t slide_ns = config_.slide_interval.count() * 1000000ULL;
            return timestamp_ns / slide_ns;
        }
        case WindowType::SESSION:
            // Session windows are more complex and would need session tracking
            return timestamp_ns / window_size_ns;
    }
    
    return timestamp_ns / window_size_ns;
}

WindowedAggregator::WindowKey WindowedAggregator::create_window_key(uint32_t event_type, uint32_t tenant_id, uint64_t timestamp_ns) {
    return {event_type, tenant_id, get_window_id(timestamp_ns)};
}

WindowState* WindowedAggregator::get_or_create_window(const WindowKey& key) {
    auto it = windows_.find(key);
    if (it != windows_.end()) {
        return it->second.get();
    }
    
    // Create new window
    auto new_window = std::make_unique<WindowState>();
    WindowState* ptr = new_window.get();
    windows_[key] = std::move(new_window);
    
    return ptr;
}

void WindowedAggregator::compute_window_results(const WindowKey& key, WindowState* state) {
    uint64_t agg_key = make_aggregation_key(key.event_type, key.tenant_id);
    auto agg_it = aggregations_.find(agg_key);
    if (agg_it == aggregations_.end()) {
        return;
    }
    
    for (const auto& agg_config : agg_it->second) {
        AggregationResult result;
        result.window_start_ns = state->window_start_ns.load();
        result.window_end_ns = state->window_end_ns.load();
        result.event_type = key.event_type;
        result.tenant_id = key.tenant_id;
        result.aggregation_type = agg_config.type;
        result.count = state->count.load();
        result.min_value = state->min_value.load();
        result.max_value = state->max_value.load();
        result.sum_value = state->sum_double.load();
        result.sum_squares = state->sum_squares.load();
        
        // Calculate specific aggregation value (same logic as get_current_results)
        switch (agg_config.type) {
            case AggregationType::COUNT:
                result.value = static_cast<double>(result.count);
                break;
            case AggregationType::SUM:
                result.value = result.sum_value;
                break;
            case AggregationType::AVERAGE:
                result.value = result.count > 0 ? result.sum_value / result.count : 0.0;
                break;
            case AggregationType::MIN:
                result.value = result.min_value;
                break;
            case AggregationType::MAX:
                result.value = result.max_value;
                break;
            case AggregationType::PERCENTILE_95:
                result.value = calculate_percentile(state, 0.95);
                break;
            case AggregationType::PERCENTILE_99:
                result.value = calculate_percentile(state, 0.99);
                break;
            case AggregationType::STANDARD_DEVIATION: {
                double mean = result.count > 0 ? result.sum_value / result.count : 0.0;
                double variance = result.count > 1 ? 
                    (result.sum_squares - (result.sum_value * result.sum_value / result.count)) / (result.count - 1) : 0.0;
                result.value = std::sqrt(std::max(0.0, variance));
                break;
            }
        }
        
        // Call the callback
        agg_config.callback(result);
    }
}

void WindowedAggregator::cleanup_expired_windows() {
    uint64_t current_time = get_current_time_ns();
    uint64_t retention_time = config_.window_size.count() * 2 * 1000000ULL; // Keep windows for 2x window size
    
    auto it = windows_.begin();
    while (it != windows_.end()) {
        if (it->second && it->second->window_end_ns.load() + retention_time < current_time) {
            it = windows_.erase(it);
        } else {
            ++it;
        }
    }
}

void WindowedAggregator::simd_sum_batch(const double* values, size_t count, double& sum) {
    sum = 0.0;
    
    if (count < 4) {
        for (size_t i = 0; i < count; ++i) {
            sum += values[i];
        }
        return;
    }
    
    // SIMD processing for batches of 4
    size_t simd_count = (count / 4) * 4;
    __m256d sum_vec = _mm256_setzero_pd();
    
    for (size_t i = 0; i < simd_count; i += 4) {
        __m256d vals = _mm256_loadu_pd(&values[i]);
        sum_vec = _mm256_add_pd(sum_vec, vals);
    }
    
    // Horizontal sum of the vector
    double temp[4];
    _mm256_storeu_pd(temp, sum_vec);
    sum = temp[0] + temp[1] + temp[2] + temp[3];
    
    // Process remaining elements
    for (size_t i = simd_count; i < count; ++i) {
        sum += values[i];
    }
}

void WindowedAggregator::simd_min_max_batch(const double* values, size_t count, double& min_val, double& max_val) {
    if (count == 0) {
        min_val = std::numeric_limits<double>::max();
        max_val = std::numeric_limits<double>::lowest();
        return;
    }
    
    if (count < 4) {
        min_val = max_val = values[0];
        for (size_t i = 1; i < count; ++i) {
            min_val = std::min(min_val, values[i]);
            max_val = std::max(max_val, values[i]);
        }
        return;
    }
    
    // SIMD processing
    size_t simd_count = (count / 4) * 4;
    __m256d min_vec = _mm256_set1_pd(std::numeric_limits<double>::max());
    __m256d max_vec = _mm256_set1_pd(std::numeric_limits<double>::lowest());
    
    for (size_t i = 0; i < simd_count; i += 4) {
        __m256d vals = _mm256_loadu_pd(&values[i]);
        min_vec = _mm256_min_pd(min_vec, vals);
        max_vec = _mm256_max_pd(max_vec, vals);
    }
    
    // Extract min/max from vectors
    double min_temp[4], max_temp[4];
    _mm256_storeu_pd(min_temp, min_vec);
    _mm256_storeu_pd(max_temp, max_vec);
    
    min_val = std::min({min_temp[0], min_temp[1], min_temp[2], min_temp[3]});
    max_val = std::max({max_temp[0], max_temp[1], max_temp[2], max_temp[3]});
    
    // Process remaining elements
    for (size_t i = simd_count; i < count; ++i) {
        min_val = std::min(min_val, values[i]);
        max_val = std::max(max_val, values[i]);
    }
}

void WindowedAggregator::simd_variance_batch(const double* values, size_t count, double mean, double& variance) {
    variance = 0.0;
    
    if (count < 2) {
        return;
    }
    
    if (count < 4) {
        for (size_t i = 0; i < count; ++i) {
            double diff = values[i] - mean;
            variance += diff * diff;
        }
        variance /= (count - 1);
        return;
    }
    
    // SIMD processing
    size_t simd_count = (count / 4) * 4;
    __m256d mean_vec = _mm256_set1_pd(mean);
    __m256d sum_sq_vec = _mm256_setzero_pd();
    
    for (size_t i = 0; i < simd_count; i += 4) {
        __m256d vals = _mm256_loadu_pd(&values[i]);
        __m256d diff = _mm256_sub_pd(vals, mean_vec);
        __m256d sq_diff = _mm256_mul_pd(diff, diff);
        sum_sq_vec = _mm256_add_pd(sum_sq_vec, sq_diff);
    }
    
    // Extract sum of squares
    double temp[4];
    _mm256_storeu_pd(temp, sum_sq_vec);
    variance = temp[0] + temp[1] + temp[2] + temp[3];
    
    // Process remaining elements
    for (size_t i = simd_count; i < count; ++i) {
        double diff = values[i] - mean;
        variance += diff * diff;
    }
    
    variance /= (count - 1);
}

double WindowedAggregator::calculate_percentile(const WindowState* state, double percentile) const {
    uint64_t total_count = state->count.load();
    if (total_count == 0) {
        return 0.0;
    }
    
    uint64_t target_count = static_cast<uint64_t>(total_count * percentile);
    uint64_t running_count = 0;
    
    for (size_t i = 0; i < WindowState::HISTOGRAM_BUCKETS; ++i) {
        running_count += state->histogram[i].load();
        if (running_count >= target_count) {
            // Linear interpolation within bucket
            double bucket_min = static_cast<double>(i) / WindowState::HISTOGRAM_BUCKETS * 
                               (state->max_value.load() - state->min_value.load()) + state->min_value.load();
            double bucket_max = static_cast<double>(i + 1) / WindowState::HISTOGRAM_BUCKETS * 
                               (state->max_value.load() - state->min_value.load()) + state->min_value.load();
            
            return bucket_min + (bucket_max - bucket_min) * 0.5; // Midpoint approximation
        }
    }
    
    return state->max_value.load();
}

size_t WindowedAggregator::get_histogram_bucket(double value) const {
    // Simple linear mapping - could be improved with logarithmic or adaptive bucketing
    double min_val = 0.0; // Could be made configurable
    double max_val = 1000.0; // Could be made configurable
    
    if (value <= min_val) return 0;
    if (value >= max_val) return WindowState::HISTOGRAM_BUCKETS - 1;
    
    double normalized = (value - min_val) / (max_val - min_val);
    return static_cast<size_t>(normalized * (WindowState::HISTOGRAM_BUCKETS - 1));
}

uint64_t WindowedAggregator::get_current_time_ns() const {
    return std::chrono::high_resolution_clock::now().time_since_epoch().count();
}

uint64_t WindowedAggregator::make_aggregation_key(uint32_t event_type, uint32_t tenant_id) const {
    return (static_cast<uint64_t>(event_type) << 32) | tenant_id;
}

} // namespace stream
} // namespace ultra_cpp