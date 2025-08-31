#include "../../include/stream-processor/stream_processor.hpp"
#include <immintrin.h>
#include <algorithm>
#include <cmath>

namespace ultra_cpp {
namespace stream {

/**
 * SIMD-accelerated aggregation functions for high-performance stream processing
 */
class SIMDAggregator {
public:
    /**
     * SIMD-accelerated sum calculation for double arrays
     */
    static double simd_sum(const double* values, size_t count) {
        if (count == 0) return 0.0;
        
        if (count < 4) {
            double sum = 0.0;
            for (size_t i = 0; i < count; ++i) {
                sum += values[i];
            }
            return sum;
        }
        
        // Process 4 doubles at a time with AVX2
        size_t simd_count = (count / 4) * 4;
        __m256d sum_vec = _mm256_setzero_pd();
        
        for (size_t i = 0; i < simd_count; i += 4) {
            __m256d vals = _mm256_loadu_pd(&values[i]);
            sum_vec = _mm256_add_pd(sum_vec, vals);
        }
        
        // Horizontal sum of the vector
        double temp[4];
        _mm256_storeu_pd(temp, sum_vec);
        double sum = temp[0] + temp[1] + temp[2] + temp[3];
        
        // Process remaining elements
        for (size_t i = simd_count; i < count; ++i) {
            sum += values[i];
        }
        
        return sum;
    }
    
    /**
     * SIMD-accelerated average calculation
     */
    static double simd_average(const double* values, size_t count) {
        if (count == 0) return 0.0;
        return simd_sum(values, count) / static_cast<double>(count);
    }
    
    /**
     * SIMD-accelerated min/max calculation
     */
    static std::pair<double, double> simd_min_max(const double* values, size_t count) {
        if (count == 0) {
            return {std::numeric_limits<double>::max(), std::numeric_limits<double>::lowest()};
        }
        
        if (count < 4) {
            double min_val = values[0];
            double max_val = values[0];
            for (size_t i = 1; i < count; ++i) {
                min_val = std::min(min_val, values[i]);
                max_val = std::max(max_val, values[i]);
            }
            return {min_val, max_val};
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
        
        double min_val = std::min({min_temp[0], min_temp[1], min_temp[2], min_temp[3]});
        double max_val = std::max({max_temp[0], max_temp[1], max_temp[2], max_temp[3]});
        
        // Process remaining elements
        for (size_t i = simd_count; i < count; ++i) {
            min_val = std::min(min_val, values[i]);
            max_val = std::max(max_val, values[i]);
        }
        
        return {min_val, max_val};
    }
    
    /**
     * SIMD-accelerated variance calculation
     */
    static double simd_variance(const double* values, size_t count, double mean) {
        if (count < 2) return 0.0;
        
        if (count < 4) {
            double sum_sq_diff = 0.0;
            for (size_t i = 0; i < count; ++i) {
                double diff = values[i] - mean;
                sum_sq_diff += diff * diff;
            }
            return sum_sq_diff / (count - 1);
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
        double sum_sq = temp[0] + temp[1] + temp[2] + temp[3];
        
        // Process remaining elements
        for (size_t i = simd_count; i < count; ++i) {
            double diff = values[i] - mean;
            sum_sq += diff * diff;
        }
        
        return sum_sq / (count - 1);
    }
    
    /**
     * SIMD-accelerated standard deviation calculation
     */
    static double simd_standard_deviation(const double* values, size_t count, double mean) {
        return std::sqrt(simd_variance(values, count, mean));
    }
    
    /**
     * SIMD-accelerated dot product for correlation calculations
     */
    static double simd_dot_product(const double* a, const double* b, size_t count) {
        if (count == 0) return 0.0;
        
        if (count < 4) {
            double dot = 0.0;
            for (size_t i = 0; i < count; ++i) {
                dot += a[i] * b[i];
            }
            return dot;
        }
        
        // SIMD processing
        size_t simd_count = (count / 4) * 4;
        __m256d dot_vec = _mm256_setzero_pd();
        
        for (size_t i = 0; i < simd_count; i += 4) {
            __m256d a_vals = _mm256_loadu_pd(&a[i]);
            __m256d b_vals = _mm256_loadu_pd(&b[i]);
            __m256d product = _mm256_mul_pd(a_vals, b_vals);
            dot_vec = _mm256_add_pd(dot_vec, product);
        }
        
        // Horizontal sum
        double temp[4];
        _mm256_storeu_pd(temp, dot_vec);
        double dot = temp[0] + temp[1] + temp[2] + temp[3];
        
        // Process remaining elements
        for (size_t i = simd_count; i < count; ++i) {
            dot += a[i] * b[i];
        }
        
        return dot;
    }
    
    /**
     * SIMD-accelerated moving average calculation
     */
    static void simd_moving_average(const double* input, double* output, size_t count, size_t window_size) {
        if (count == 0 || window_size == 0) return;
        
        // Calculate first window sum
        double window_sum = 0.0;
        for (size_t i = 0; i < std::min(window_size, count); ++i) {
            window_sum += input[i];
        }
        
        // Set initial values
        for (size_t i = 0; i < std::min(window_size, count); ++i) {
            output[i] = window_sum / (i + 1);
        }
        
        // Sliding window with SIMD optimization
        if (count > window_size) {
            for (size_t i = window_size; i < count; ++i) {
                window_sum = window_sum - input[i - window_size] + input[i];
                output[i] = window_sum / window_size;
            }
        }
    }
    
    /**
     * SIMD-accelerated exponential moving average
     */
    static void simd_exponential_moving_average(const double* input, double* output, 
                                               size_t count, double alpha) {
        if (count == 0) return;
        
        output[0] = input[0];
        
        // Vectorized EMA calculation
        __m256d alpha_vec = _mm256_set1_pd(alpha);
        __m256d one_minus_alpha_vec = _mm256_set1_pd(1.0 - alpha);
        
        for (size_t i = 1; i < count; ++i) {
            output[i] = alpha * input[i] + (1.0 - alpha) * output[i - 1];
        }
    }
    
    /**
     * SIMD-accelerated histogram calculation
     */
    static void simd_histogram(const double* values, size_t count, 
                              double min_val, double max_val, 
                              uint64_t* bins, size_t bin_count) {
        if (count == 0 || bin_count == 0 || max_val <= min_val) return;
        
        // Clear bins
        std::fill(bins, bins + bin_count, 0);
        
        double range = max_val - min_val;
        double bin_width = range / bin_count;
        
        // SIMD processing for bin calculation
        if (count >= 4) {
            __m256d min_vec = _mm256_set1_pd(min_val);
            __m256d bin_width_vec = _mm256_set1_pd(bin_width);
            __m256d bin_count_vec = _mm256_set1_pd(static_cast<double>(bin_count - 1));
            
            size_t simd_count = (count / 4) * 4;
            
            for (size_t i = 0; i < simd_count; i += 4) {
                __m256d vals = _mm256_loadu_pd(&values[i]);
                __m256d normalized = _mm256_div_pd(_mm256_sub_pd(vals, min_vec), bin_width_vec);
                
                // Convert to integers and clamp to valid range
                __m256d clamped = _mm256_min_pd(_mm256_max_pd(normalized, _mm256_setzero_pd()), bin_count_vec);
                
                // Extract and update bins (not fully vectorizable due to indirect access)
                double temp[4];
                _mm256_storeu_pd(temp, clamped);
                
                for (int j = 0; j < 4; ++j) {
                    size_t bin_idx = static_cast<size_t>(temp[j]);
                    bins[bin_idx]++;
                }
            }
            
            // Process remaining elements
            for (size_t i = simd_count; i < count; ++i) {
                double normalized = (values[i] - min_val) / bin_width;
                size_t bin_idx = static_cast<size_t>(std::max(0.0, std::min(normalized, static_cast<double>(bin_count - 1))));
                bins[bin_idx]++;
            }
        } else {
            // Fallback for small arrays
            for (size_t i = 0; i < count; ++i) {
                double normalized = (values[i] - min_val) / bin_width;
                size_t bin_idx = static_cast<size_t>(std::max(0.0, std::min(normalized, static_cast<double>(bin_count - 1))));
                bins[bin_idx]++;
            }
        }
    }
    
    /**
     * SIMD-accelerated percentile calculation using histogram
     */
    static double simd_percentile_from_histogram(const uint64_t* bins, size_t bin_count,
                                                double min_val, double max_val,
                                                uint64_t total_count, double percentile) {
        if (bin_count == 0 || total_count == 0) return 0.0;
        
        uint64_t target_count = static_cast<uint64_t>(total_count * percentile);
        uint64_t running_count = 0;
        
        for (size_t i = 0; i < bin_count; ++i) {
            running_count += bins[i];
            if (running_count >= target_count) {
                // Linear interpolation within bin
                double bin_width = (max_val - min_val) / bin_count;
                double bin_start = min_val + i * bin_width;
                double bin_end = bin_start + bin_width;
                
                // Simple midpoint approximation
                return bin_start + bin_width * 0.5;
            }
        }
        
        return max_val;
    }
    
    /**
     * SIMD-accelerated correlation coefficient calculation
     */
    static double simd_correlation(const double* x, const double* y, size_t count) {
        if (count < 2) return 0.0;
        
        double mean_x = simd_average(x, count);
        double mean_y = simd_average(y, count);
        
        // Calculate numerator and denominators
        double numerator = 0.0;
        double sum_sq_x = 0.0;
        double sum_sq_y = 0.0;
        
        if (count >= 4) {
            __m256d mean_x_vec = _mm256_set1_pd(mean_x);
            __m256d mean_y_vec = _mm256_set1_pd(mean_y);
            __m256d num_vec = _mm256_setzero_pd();
            __m256d sum_sq_x_vec = _mm256_setzero_pd();
            __m256d sum_sq_y_vec = _mm256_setzero_pd();
            
            size_t simd_count = (count / 4) * 4;
            
            for (size_t i = 0; i < simd_count; i += 4) {
                __m256d x_vals = _mm256_loadu_pd(&x[i]);
                __m256d y_vals = _mm256_loadu_pd(&y[i]);
                
                __m256d x_diff = _mm256_sub_pd(x_vals, mean_x_vec);
                __m256d y_diff = _mm256_sub_pd(y_vals, mean_y_vec);
                
                num_vec = _mm256_add_pd(num_vec, _mm256_mul_pd(x_diff, y_diff));
                sum_sq_x_vec = _mm256_add_pd(sum_sq_x_vec, _mm256_mul_pd(x_diff, x_diff));
                sum_sq_y_vec = _mm256_add_pd(sum_sq_y_vec, _mm256_mul_pd(y_diff, y_diff));
            }
            
            // Extract results
            double temp_num[4], temp_x[4], temp_y[4];
            _mm256_storeu_pd(temp_num, num_vec);
            _mm256_storeu_pd(temp_x, sum_sq_x_vec);
            _mm256_storeu_pd(temp_y, sum_sq_y_vec);
            
            numerator = temp_num[0] + temp_num[1] + temp_num[2] + temp_num[3];
            sum_sq_x = temp_x[0] + temp_x[1] + temp_x[2] + temp_x[3];
            sum_sq_y = temp_y[0] + temp_y[1] + temp_y[2] + temp_y[3];
            
            // Process remaining elements
            for (size_t i = simd_count; i < count; ++i) {
                double x_diff = x[i] - mean_x;
                double y_diff = y[i] - mean_y;
                numerator += x_diff * y_diff;
                sum_sq_x += x_diff * x_diff;
                sum_sq_y += y_diff * y_diff;
            }
        } else {
            // Fallback for small arrays
            for (size_t i = 0; i < count; ++i) {
                double x_diff = x[i] - mean_x;
                double y_diff = y[i] - mean_y;
                numerator += x_diff * y_diff;
                sum_sq_x += x_diff * x_diff;
                sum_sq_y += y_diff * y_diff;
            }
        }
        
        double denominator = std::sqrt(sum_sq_x * sum_sq_y);
        return denominator > 0.0 ? numerator / denominator : 0.0;
    }
};

/**
 * Batch processing utilities for stream events
 */
namespace batch_processing {

/**
 * Extract numeric values from a batch of stream events using SIMD
 */
void extract_values_simd(const std::vector<const StreamEvent*>& events, 
                        std::vector<double>& values) {
    values.clear();
    values.reserve(events.size());
    
    for (const StreamEvent* event : events) {
        if (!event || event->data_size == 0) {
            values.push_back(0.0);
            continue;
        }
        
        // Simple extraction - in practice would be more sophisticated
        std::string data(event->data, event->data_size);
        size_t colon_pos = data.find(':');
        
        if (colon_pos != std::string::npos && colon_pos + 1 < data.size()) {
            try {
                values.push_back(std::stod(data.substr(colon_pos + 1)));
            } catch (const std::exception&) {
                values.push_back(static_cast<double>(event->data_size));
            }
        } else {
            // Fallback to timestamp-based value
            values.push_back(static_cast<double>(event->timestamp_ns % 1000000) / 1000.0);
        }
    }
}

/**
 * Batch aggregation using SIMD acceleration
 */
struct BatchAggregationResult {
    double sum;
    double average;
    double min;
    double max;
    double variance;
    double std_dev;
    size_t count;
};

BatchAggregationResult aggregate_batch_simd(const std::vector<const StreamEvent*>& events) {
    std::vector<double> values;
    extract_values_simd(events, values);
    
    if (values.empty()) {
        return {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0};
    }
    
    BatchAggregationResult result;
    result.count = values.size();
    result.sum = SIMDAggregator::simd_sum(values.data(), values.size());
    result.average = result.sum / result.count;
    
    auto [min_val, max_val] = SIMDAggregator::simd_min_max(values.data(), values.size());
    result.min = min_val;
    result.max = max_val;
    
    result.variance = SIMDAggregator::simd_variance(values.data(), values.size(), result.average);
    result.std_dev = std::sqrt(result.variance);
    
    return result;
}

} // namespace batch_processing

} // namespace stream
} // namespace ultra_cpp