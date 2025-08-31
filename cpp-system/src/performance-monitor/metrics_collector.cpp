#include "performance-monitor/metrics_collector.hpp"
#include "common/logger.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <immintrin.h>
#include <chrono>
#include <limits>

namespace ultra {
namespace monitor {

MetricsCollector::MetricsCollector(const Config& config)
    : config_(config)
    , counters_(config.max_metrics)
    , gauges_(config.max_metrics)
    , histograms_(config.max_metrics) {
    
    LOG_INFO("MetricsCollector initialized with {} max metrics, {} histogram buckets", 
             config_.max_metrics, config_.histogram_buckets);
}

MetricsCollector::~MetricsCollector() {
    LOG_INFO("MetricsCollector destroyed. Total operations: {}", 
             stats_.total_operations.load());
}

void MetricsCollector::increment_counter(const std::string& name, uint64_t value) noexcept {
    auto start_time = get_timestamp_ns();
    
    CounterEntry* entry = counters_.get_or_create(name);
    if (entry) {
        entry->value.fetch_add(value, std::memory_order_relaxed);
        entry->last_access.store(start_time, std::memory_order_relaxed);
        update_stats(MetricType::COUNTER);
    }
    
    auto end_time = get_timestamp_ns();
    stats_.collection_overhead_ns.fetch_add(end_time - start_time, std::memory_order_relaxed);
}

uint64_t MetricsCollector::get_counter_value(const std::string& name) const noexcept {
    CounterEntry* entry = counters_.get(name);
    return entry ? entry->value.load(std::memory_order_relaxed) : 0;
}

void MetricsCollector::set_gauge(const std::string& name, double value) noexcept {
    auto start_time = get_timestamp_ns();
    
    GaugeEntry* entry = gauges_.get_or_create(name);
    if (entry) {
        entry->value.store(value, std::memory_order_relaxed);
        entry->last_access.store(start_time, std::memory_order_relaxed);
        update_stats(MetricType::GAUGE);
    }
    
    auto end_time = get_timestamp_ns();
    stats_.collection_overhead_ns.fetch_add(end_time - start_time, std::memory_order_relaxed);
}

double MetricsCollector::get_gauge_value(const std::string& name) const noexcept {
    GaugeEntry* entry = gauges_.get(name);
    return entry ? entry->value.load(std::memory_order_relaxed) : 0.0;
}

void MetricsCollector::observe_histogram(const std::string& name, double value) noexcept {
    auto start_time = get_timestamp_ns();
    
    HistogramData* histogram = histograms_.get_or_create(name);
    if (!histogram) {
        auto end_time = get_timestamp_ns();
        stats_.collection_overhead_ns.fetch_add(end_time - start_time, std::memory_order_relaxed);
        return;
    }

    // Initialize histogram buckets if this is the first observation
    if (histogram->buckets.empty()) {
        auto buckets = generate_exponential_buckets(0.001, 2.0, config_.histogram_buckets);
        histogram->buckets.reserve(buckets.size());
        for (double bound : buckets) {
            histogram->buckets.push_back({bound, std::atomic<uint64_t>{0}});
        }
    }

    // Find appropriate bucket and increment
    for (auto& bucket : histogram->buckets) {
        if (value <= bucket.upper_bound) {
            bucket.count.fetch_add(1, std::memory_order_relaxed);
            break;
        }
    }

    // Update histogram statistics
    histogram->total_count.fetch_add(1, std::memory_order_relaxed);
    histogram->sum.fetch_add(value, std::memory_order_relaxed);
    histogram->sum_squares.fetch_add(value * value, std::memory_order_relaxed);

    // Update min/max values
    double current_min = histogram->min_value.load(std::memory_order_relaxed);
    while (value < current_min && 
           !histogram->min_value.compare_exchange_weak(current_min, value, std::memory_order_relaxed)) {
        // Retry if CAS failed
    }

    double current_max = histogram->max_value.load(std::memory_order_relaxed);
    while (value > current_max && 
           !histogram->max_value.compare_exchange_weak(current_max, value, std::memory_order_relaxed)) {
        // Retry if CAS failed
    }

    update_stats(MetricType::HISTOGRAM);
    
    auto end_time = get_timestamp_ns();
    stats_.collection_overhead_ns.fetch_add(end_time - start_time, std::memory_order_relaxed);
}

void MetricsCollector::record_timing(const std::string& name, uint64_t duration_ns) noexcept {
    // Convert nanoseconds to seconds for histogram
    double duration_seconds = static_cast<double>(duration_ns) / 1e9;
    observe_histogram(name, duration_seconds);
    update_stats(MetricType::TIMER);
}

MetricsCollector::PercentileData MetricsCollector::calculate_percentiles(const std::string& name) const {
    const HistogramData* histogram = histograms_.get(name);
    if (!histogram || histogram->buckets.empty()) {
        return PercentileData{};
    }

    if (config_.enable_percentiles) {
        return SIMDPercentileCalculator::calculate_percentiles(
            histogram->buckets, config_.percentiles, 5);
    }

    // Fallback to basic statistics
    PercentileData data;
    data.count = histogram->total_count.load(std::memory_order_relaxed);
    data.sum = histogram->sum.load(std::memory_order_relaxed);
    data.min = histogram->min_value.load(std::memory_order_relaxed);
    data.max = histogram->max_value.load(std::memory_order_relaxed);
    
    if (data.count > 0) {
        data.mean = data.sum / data.count;
        
        // Calculate standard deviation
        double sum_squares = histogram->sum_squares.load(std::memory_order_relaxed);
        double variance = (sum_squares / data.count) - (data.mean * data.mean);
        data.stddev = variance > 0 ? std::sqrt(variance) : 0.0;
    }
    
    return data;
}

const MetricsCollector::HistogramData* MetricsCollector::get_histogram_data(const std::string& name) const {
    return histograms_.get(name);
}

std::vector<std::string> MetricsCollector::get_counter_names() const {
    return counters_.get_all_keys();
}

std::vector<std::string> MetricsCollector::get_gauge_names() const {
    return gauges_.get_all_keys();
}

std::vector<std::string> MetricsCollector::get_histogram_names() const {
    return histograms_.get_all_keys();
}

void MetricsCollector::reset_all_metrics() {
    counters_.clear();
    gauges_.clear();
    histograms_.clear();
    
    // Reset statistics
    stats_.total_operations.store(0, std::memory_order_relaxed);
    stats_.counter_operations.store(0, std::memory_order_relaxed);
    stats_.gauge_operations.store(0, std::memory_order_relaxed);
    stats_.histogram_operations.store(0, std::memory_order_relaxed);
    stats_.collection_overhead_ns.store(0, std::memory_order_relaxed);
    
    LOG_INFO("All metrics reset");
}

void MetricsCollector::cleanup_unused_metrics() {
    // This would implement cleanup of metrics not accessed recently
    // For now, just log the operation
    LOG_INFO("Cleanup unused metrics requested");
}

size_t MetricsCollector::get_memory_usage() const noexcept {
    // Estimate memory usage
    size_t counter_memory = counters_.get_all_keys().size() * sizeof(CounterEntry);
    size_t gauge_memory = gauges_.get_all_keys().size() * sizeof(GaugeEntry);
    size_t histogram_memory = histograms_.get_all_keys().size() * sizeof(HistogramData);
    
    return counter_memory + gauge_memory + histogram_memory;
}

std::vector<double> MetricsCollector::generate_exponential_buckets(double start, double factor, size_t count) const {
    std::vector<double> buckets;
    buckets.reserve(count);
    
    double current = start;
    for (size_t i = 0; i < count; ++i) {
        buckets.push_back(current);
        current *= factor;
    }
    
    // Add infinity bucket
    buckets.push_back(std::numeric_limits<double>::infinity());
    
    return buckets;
}

std::vector<double> MetricsCollector::generate_linear_buckets(double start, double width, size_t count) const {
    std::vector<double> buckets;
    buckets.reserve(count);
    
    for (size_t i = 0; i < count; ++i) {
        buckets.push_back(start + i * width);
    }
    
    // Add infinity bucket
    buckets.push_back(std::numeric_limits<double>::infinity());
    
    return buckets;
}

uint64_t MetricsCollector::get_timestamp_ns() const noexcept {
    return std::chrono::steady_clock::now().time_since_epoch().count();
}

void MetricsCollector::update_stats(MetricType type) noexcept {
    stats_.total_operations.fetch_add(1, std::memory_order_relaxed);
    
    switch (type) {
        case MetricType::COUNTER:
            stats_.counter_operations.fetch_add(1, std::memory_order_relaxed);
            break;
        case MetricType::GAUGE:
            stats_.gauge_operations.fetch_add(1, std::memory_order_relaxed);
            break;
        case MetricType::HISTOGRAM:
        case MetricType::TIMER:
            stats_.histogram_operations.fetch_add(1, std::memory_order_relaxed);
            break;
    }
}

// LockFreeHashMap implementation
template<typename T>
MetricsCollector::LockFreeHashMap<T>::LockFreeHashMap(size_t capacity)
    : capacity_(capacity) {
    entries_.resize(capacity_);
}

template<typename T>
MetricsCollector::LockFreeHashMap<T>::~LockFreeHashMap() {
    clear();
}

template<typename T>
T* MetricsCollector::LockFreeHashMap<T>::get_or_create(const std::string& key) {
    uint64_t hash = hash_string(key);
    size_t slot = find_slot(hash);
    
    // Try to find existing entry
    for (size_t i = 0; i < capacity_; ++i) {
        size_t index = (slot + i) % capacity_;
        Entry& entry = entries_[index];
        
        uint64_t expected_hash = 0;
        if (entry.key_hash.compare_exchange_weak(expected_hash, hash, std::memory_order_acq_rel)) {
            // Successfully claimed empty slot
            entry.key = key;
            entry.value.store(new T{}, std::memory_order_release);
            entry.valid.store(true, std::memory_order_release);
            size_.fetch_add(1, std::memory_order_relaxed);
            return entry.value.load(std::memory_order_acquire);
        } else if (expected_hash == hash && entry.valid.load(std::memory_order_acquire) && entry.key == key) {
            // Found existing entry
            return entry.value.load(std::memory_order_acquire);
        }
    }
    
    // Hash table full
    return nullptr;
}

template<typename T>
T* MetricsCollector::LockFreeHashMap<T>::get(const std::string& key) const {
    uint64_t hash = hash_string(key);
    size_t slot = find_slot(hash);
    
    for (size_t i = 0; i < capacity_; ++i) {
        size_t index = (slot + i) % capacity_;
        const Entry& entry = entries_[index];
        
        if (entry.key_hash.load(std::memory_order_acquire) == hash &&
            entry.valid.load(std::memory_order_acquire) &&
            entry.key == key) {
            return entry.value.load(std::memory_order_acquire);
        }
        
        if (entry.key_hash.load(std::memory_order_acquire) == 0) {
            // Empty slot, key not found
            break;
        }
    }
    
    return nullptr;
}

template<typename T>
std::vector<std::string> MetricsCollector::LockFreeHashMap<T>::get_all_keys() const {
    std::vector<std::string> keys;
    keys.reserve(size_.load(std::memory_order_relaxed));
    
    for (const auto& entry : entries_) {
        if (entry.valid.load(std::memory_order_acquire)) {
            keys.push_back(entry.key);
        }
    }
    
    return keys;
}

template<typename T>
void MetricsCollector::LockFreeHashMap<T>::clear() {
    for (auto& entry : entries_) {
        T* value = entry.value.exchange(nullptr, std::memory_order_acq_rel);
        if (value) {
            delete value;
        }
        entry.key_hash.store(0, std::memory_order_release);
        entry.valid.store(false, std::memory_order_release);
        entry.key.clear();
    }
    size_.store(0, std::memory_order_release);
}

template<typename T>
uint64_t MetricsCollector::LockFreeHashMap<T>::hash_string(const std::string& str) const noexcept {
    // Simple FNV-1a hash
    uint64_t hash = 14695981039346656037ULL;
    for (char c : str) {
        hash ^= static_cast<uint64_t>(c);
        hash *= 1099511628211ULL;
    }
    return hash;
}

template<typename T>
size_t MetricsCollector::LockFreeHashMap<T>::find_slot(uint64_t hash) const noexcept {
    return hash % capacity_;
}

// Explicit template instantiations
template class MetricsCollector::LockFreeHashMap<MetricsCollector::CounterEntry>;
template class MetricsCollector::LockFreeHashMap<MetricsCollector::GaugeEntry>;
template class MetricsCollector::LockFreeHashMap<MetricsCollector::HistogramData>;

// SIMDPercentileCalculator implementation
MetricsCollector::PercentileData SIMDPercentileCalculator::calculate_percentiles(
    const std::vector<MetricsCollector::HistogramBucket>& buckets,
    const double* percentiles, size_t percentile_count) {
    
    MetricsCollector::PercentileData data;
    
    if (buckets.empty()) {
        return data;
    }

    // Calculate total count
    uint64_t total_count = 0;
    for (const auto& bucket : buckets) {
        total_count += bucket.count.load(std::memory_order_relaxed);
    }
    
    if (total_count == 0) {
        return data;
    }

    data.count = total_count;

    // Create cumulative counts array
    std::vector<uint64_t> cumulative(buckets.size());
    vectorized_cumsum(reinterpret_cast<const uint64_t*>(buckets.data()), 
                     cumulative.data(), buckets.size());

    // Calculate percentiles
    if (percentile_count >= 1) data.p50 = interpolate_percentile(buckets, cumulative.data(), total_count, 0.5);
    if (percentile_count >= 2) data.p95 = interpolate_percentile(buckets, cumulative.data(), total_count, 0.95);
    if (percentile_count >= 3) data.p99 = interpolate_percentile(buckets, cumulative.data(), total_count, 0.99);
    if (percentile_count >= 4) data.p999 = interpolate_percentile(buckets, cumulative.data(), total_count, 0.999);
    if (percentile_count >= 5) data.p9999 = interpolate_percentile(buckets, cumulative.data(), total_count, 0.9999);

    // Calculate min/max (first and last non-empty buckets)
    for (const auto& bucket : buckets) {
        if (bucket.count.load(std::memory_order_relaxed) > 0) {
            data.min = bucket.upper_bound;
            break;
        }
    }
    
    for (auto it = buckets.rbegin(); it != buckets.rend(); ++it) {
        if (it->count.load(std::memory_order_relaxed) > 0) {
            data.max = it->upper_bound;
            break;
        }
    }

    return data;
}

void SIMDPercentileCalculator::vectorized_cumsum(const uint64_t* counts, uint64_t* cumulative, size_t size) {
    if (size == 0) return;
    
    cumulative[0] = counts[0];
    
    // Use SIMD for vectorized cumulative sum where possible
    size_t simd_size = size & ~3ULL;  // Round down to multiple of 4
    
    for (size_t i = 1; i < simd_size; i += 4) {
        // Load 4 counts
        __m256i counts_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&counts[i]));
        
        // Add previous cumulative value to each element
        __m256i prev_cum = _mm256_set1_epi64x(cumulative[i-1]);
        __m256i result = _mm256_add_epi64(counts_vec, prev_cum);
        
        // Store result
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(&cumulative[i]), result);
        
        // Update for next iteration
        cumulative[i] += cumulative[i-1];
        cumulative[i+1] += cumulative[i];
        cumulative[i+2] += cumulative[i+1];
        cumulative[i+3] += cumulative[i+2];
    }
    
    // Handle remaining elements
    for (size_t i = simd_size; i < size; ++i) {
        cumulative[i] = cumulative[i-1] + counts[i];
    }
}

double SIMDPercentileCalculator::interpolate_percentile(
    const std::vector<MetricsCollector::HistogramBucket>& buckets,
    const uint64_t* cumulative, uint64_t total_count, double percentile) {
    
    uint64_t target_count = static_cast<uint64_t>(percentile * total_count);
    
    for (size_t i = 0; i < buckets.size(); ++i) {
        if (cumulative[i] >= target_count) {
            if (i == 0) {
                return buckets[i].upper_bound;
            }
            
            // Linear interpolation between buckets
            double prev_bound = (i > 0) ? buckets[i-1].upper_bound : 0.0;
            double curr_bound = buckets[i].upper_bound;
            uint64_t prev_count = (i > 0) ? cumulative[i-1] : 0;
            uint64_t curr_count = cumulative[i];
            
            if (curr_count == prev_count) {
                return curr_bound;
            }
            
            double ratio = static_cast<double>(target_count - prev_count) / (curr_count - prev_count);
            return prev_bound + ratio * (curr_bound - prev_bound);
        }
    }
    
    // Return last bucket if not found
    return buckets.empty() ? 0.0 : buckets.back().upper_bound;
}

} // namespace monitor
} // namespace ultra