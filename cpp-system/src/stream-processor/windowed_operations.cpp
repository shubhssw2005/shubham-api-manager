#include "../../include/stream-processor/windowed_aggregator.hpp"
#include "../../include/stream-processor/stream_processor.hpp"
#include <algorithm>
#include <cmath>

namespace ultra_cpp {
namespace stream {

/**
 * Advanced windowed operations for complex stream processing scenarios
 */
class WindowedOperations {
public:
    /**
     * Sliding window with custom step size
     */
    class SlidingWindow {
    public:
        struct Config {
            std::chrono::milliseconds window_size{1000};
            std::chrono::milliseconds step_size{100};
            size_t max_elements = 10000;
        };
        
        explicit SlidingWindow(const Config& config) : config_(config) {
            buffer_.reserve(config_.max_elements);
        }
        
        void add_value(uint64_t timestamp_ns, double value) {
            // Remove expired elements
            uint64_t window_start = timestamp_ns - (config_.window_size.count() * 1000000ULL);
            
            auto it = std::remove_if(buffer_.begin(), buffer_.end(),
                [window_start](const TimestampedValue& tv) {
                    return tv.timestamp_ns < window_start;
                });
            buffer_.erase(it, buffer_.end());
            
            // Add new value
            buffer_.push_back({timestamp_ns, value});
            
            // Maintain size limit
            if (buffer_.size() > config_.max_elements) {
                buffer_.erase(buffer_.begin(), buffer_.begin() + (buffer_.size() - config_.max_elements));
            }
        }
        
        double calculate_average() const {
            if (buffer_.empty()) return 0.0;
            
            double sum = 0.0;
            for (const auto& tv : buffer_) {
                sum += tv.value;
            }
            return sum / buffer_.size();
        }
        
        double calculate_weighted_average() const {
            if (buffer_.empty()) return 0.0;
            
            uint64_t latest_timestamp = buffer_.back().timestamp_ns;
            double weighted_sum = 0.0;
            double weight_sum = 0.0;
            
            for (const auto& tv : buffer_) {
                // Weight based on recency (more recent = higher weight)
                double age_ms = static_cast<double>(latest_timestamp - tv.timestamp_ns) / 1000000.0;
                double weight = std::exp(-age_ms / config_.window_size.count());
                
                weighted_sum += tv.value * weight;
                weight_sum += weight;
            }
            
            return weight_sum > 0.0 ? weighted_sum / weight_sum : 0.0;
        }
        
        std::pair<double, double> calculate_min_max() const {
            if (buffer_.empty()) {
                return {std::numeric_limits<double>::max(), std::numeric_limits<double>::lowest()};
            }
            
            double min_val = buffer_[0].value;
            double max_val = buffer_[0].value;
            
            for (const auto& tv : buffer_) {
                min_val = std::min(min_val, tv.value);
                max_val = std::max(max_val, tv.value);
            }
            
            return {min_val, max_val};
        }
        
        double calculate_percentile(double percentile) const {
            if (buffer_.empty()) return 0.0;
            
            std::vector<double> values;
            values.reserve(buffer_.size());
            for (const auto& tv : buffer_) {
                values.push_back(tv.value);
            }
            
            std::sort(values.begin(), values.end());
            
            size_t index = static_cast<size_t>(percentile * (values.size() - 1));
            return values[index];
        }
        
        size_t size() const { return buffer_.size(); }
        bool empty() const { return buffer_.empty(); }
        
    private:
        struct TimestampedValue {
            uint64_t timestamp_ns;
            double value;
        };
        
        Config config_;
        std::vector<TimestampedValue> buffer_;
    };
    
    /**
     * Tumbling window with exact boundaries
     */
    class TumblingWindow {
    public:
        struct Config {
            std::chrono::milliseconds window_size{1000};
            bool align_to_epoch = true; // Align windows to epoch boundaries
        };
        
        explicit TumblingWindow(const Config& config) : config_(config) {}
        
        void add_value(uint64_t timestamp_ns, double value) {
            uint64_t window_id = get_window_id(timestamp_ns);
            
            if (current_window_id_ != window_id) {
                // Finalize current window if it exists
                if (current_window_id_ != UINT64_MAX && !current_values_.empty()) {
                    finalize_window();
                }
                
                // Start new window
                current_window_id_ = window_id;
                current_values_.clear();
            }
            
            current_values_.push_back(value);
        }
        
        struct WindowResult {
            uint64_t window_id;
            uint64_t start_time_ns;
            uint64_t end_time_ns;
            double sum;
            double average;
            double min;
            double max;
            size_t count;
        };
        
        std::vector<WindowResult> get_completed_windows() {
            std::vector<WindowResult> results = completed_windows_;
            completed_windows_.clear();
            return results;
        }
        
        void force_finalize() {
            if (current_window_id_ != UINT64_MAX && !current_values_.empty()) {
                finalize_window();
            }
        }
        
    private:
        Config config_;
        uint64_t current_window_id_ = UINT64_MAX;
        std::vector<double> current_values_;
        std::vector<WindowResult> completed_windows_;
        
        uint64_t get_window_id(uint64_t timestamp_ns) const {
            uint64_t window_size_ns = config_.window_size.count() * 1000000ULL;
            
            if (config_.align_to_epoch) {
                return timestamp_ns / window_size_ns;
            } else {
                // Custom alignment logic could go here
                return timestamp_ns / window_size_ns;
            }
        }
        
        void finalize_window() {
            if (current_values_.empty()) return;
            
            WindowResult result;
            result.window_id = current_window_id_;
            result.count = current_values_.size();
            
            // Calculate window boundaries
            uint64_t window_size_ns = config_.window_size.count() * 1000000ULL;
            result.start_time_ns = current_window_id_ * window_size_ns;
            result.end_time_ns = result.start_time_ns + window_size_ns;
            
            // Calculate aggregations
            result.sum = 0.0;
            result.min = current_values_[0];
            result.max = current_values_[0];
            
            for (double value : current_values_) {
                result.sum += value;
                result.min = std::min(result.min, value);
                result.max = std::max(result.max, value);
            }
            
            result.average = result.sum / result.count;
            
            completed_windows_.push_back(result);
        }
    };
    
    /**
     * Session window with timeout-based boundaries
     */
    class SessionWindow {
    public:
        struct Config {
            std::chrono::milliseconds session_timeout{5000};
            std::chrono::milliseconds max_session_duration{3600000}; // 1 hour
        };
        
        explicit SessionWindow(const Config& config) : config_(config) {}
        
        void add_value(uint64_t timestamp_ns, double value, uint32_t session_key = 0) {
            auto& session = sessions_[session_key];
            
            // Check if this starts a new session
            if (session.values.empty() || 
                (timestamp_ns - session.last_timestamp_ns) > (config_.session_timeout.count() * 1000000ULL)) {
                
                // Finalize previous session if it exists
                if (!session.values.empty()) {
                    finalize_session(session_key, session);
                }
                
                // Start new session
                session.start_timestamp_ns = timestamp_ns;
                session.values.clear();
            }
            
            // Check for maximum session duration
            if ((timestamp_ns - session.start_timestamp_ns) > (config_.max_session_duration.count() * 1000000ULL)) {
                finalize_session(session_key, session);
                session.start_timestamp_ns = timestamp_ns;
                session.values.clear();
            }
            
            session.values.push_back(value);
            session.last_timestamp_ns = timestamp_ns;
        }
        
        struct SessionResult {
            uint32_t session_key;
            uint64_t start_time_ns;
            uint64_t end_time_ns;
            std::chrono::milliseconds duration;
            double sum;
            double average;
            double min;
            double max;
            size_t count;
        };
        
        std::vector<SessionResult> get_completed_sessions() {
            std::vector<SessionResult> results = completed_sessions_;
            completed_sessions_.clear();
            return results;
        }
        
        void force_finalize_all() {
            for (auto& [key, session] : sessions_) {
                if (!session.values.empty()) {
                    finalize_session(key, session);
                }
            }
        }
        
        void cleanup_expired_sessions(uint64_t current_time_ns) {
            auto it = sessions_.begin();
            while (it != sessions_.end()) {
                auto& session = it->second;
                if (!session.values.empty() && 
                    (current_time_ns - session.last_timestamp_ns) > (config_.session_timeout.count() * 1000000ULL)) {
                    
                    finalize_session(it->first, session);
                    it = sessions_.erase(it);
                } else {
                    ++it;
                }
            }
        }
        
    private:
        struct Session {
            uint64_t start_timestamp_ns = 0;
            uint64_t last_timestamp_ns = 0;
            std::vector<double> values;
        };
        
        Config config_;
        std::unordered_map<uint32_t, Session> sessions_;
        std::vector<SessionResult> completed_sessions_;
        
        void finalize_session(uint32_t session_key, const Session& session) {
            if (session.values.empty()) return;
            
            SessionResult result;
            result.session_key = session_key;
            result.start_time_ns = session.start_timestamp_ns;
            result.end_time_ns = session.last_timestamp_ns;
            result.duration = std::chrono::milliseconds(
                (session.last_timestamp_ns - session.start_timestamp_ns) / 1000000ULL);
            result.count = session.values.size();
            
            // Calculate aggregations
            result.sum = 0.0;
            result.min = session.values[0];
            result.max = session.values[0];
            
            for (double value : session.values) {
                result.sum += value;
                result.min = std::min(result.min, value);
                result.max = std::max(result.max, value);
            }
            
            result.average = result.sum / result.count;
            
            completed_sessions_.push_back(result);
        }
    };
    
    /**
     * Hopping window (overlapping fixed-size windows)
     */
    class HoppingWindow {
    public:
        struct Config {
            std::chrono::milliseconds window_size{1000};
            std::chrono::milliseconds hop_size{500};
            size_t max_windows = 100;
        };
        
        explicit HoppingWindow(const Config& config) : config_(config) {}
        
        void add_value(uint64_t timestamp_ns, double value) {
            // Add to all active windows
            for (auto& window : active_windows_) {
                if (timestamp_ns >= window.start_time_ns && 
                    timestamp_ns < window.start_time_ns + (config_.window_size.count() * 1000000ULL)) {
                    window.values.push_back(value);
                }
            }
            
            // Create new windows if needed
            uint64_t hop_size_ns = config_.hop_size.count() * 1000000ULL;
            uint64_t window_start = (timestamp_ns / hop_size_ns) * hop_size_ns;
            
            // Check if we need a new window starting at this hop
            bool window_exists = false;
            for (const auto& window : active_windows_) {
                if (window.start_time_ns == window_start) {
                    window_exists = true;
                    break;
                }
            }
            
            if (!window_exists) {
                Window new_window;
                new_window.start_time_ns = window_start;
                new_window.values.push_back(value);
                active_windows_.push_back(new_window);
            }
            
            // Remove expired windows and finalize them
            uint64_t window_size_ns = config_.window_size.count() * 1000000ULL;
            auto it = active_windows_.begin();
            while (it != active_windows_.end()) {
                if (timestamp_ns >= it->start_time_ns + window_size_ns) {
                    finalize_window(*it);
                    it = active_windows_.erase(it);
                } else {
                    ++it;
                }
            }
            
            // Limit number of active windows
            if (active_windows_.size() > config_.max_windows) {
                finalize_window(active_windows_.front());
                active_windows_.erase(active_windows_.begin());
            }
        }
        
        struct HoppingWindowResult {
            uint64_t start_time_ns;
            uint64_t end_time_ns;
            double sum;
            double average;
            double min;
            double max;
            size_t count;
        };
        
        std::vector<HoppingWindowResult> get_completed_windows() {
            std::vector<HoppingWindowResult> results = completed_windows_;
            completed_windows_.clear();
            return results;
        }
        
        void force_finalize_all() {
            for (const auto& window : active_windows_) {
                finalize_window(window);
            }
            active_windows_.clear();
        }
        
    private:
        struct Window {
            uint64_t start_time_ns;
            std::vector<double> values;
        };
        
        Config config_;
        std::vector<Window> active_windows_;
        std::vector<HoppingWindowResult> completed_windows_;
        
        void finalize_window(const Window& window) {
            if (window.values.empty()) return;
            
            HoppingWindowResult result;
            result.start_time_ns = window.start_time_ns;
            result.end_time_ns = window.start_time_ns + (config_.window_size.count() * 1000000ULL);
            result.count = window.values.size();
            
            // Calculate aggregations
            result.sum = 0.0;
            result.min = window.values[0];
            result.max = window.values[0];
            
            for (double value : window.values) {
                result.sum += value;
                result.min = std::min(result.min, value);
                result.max = std::max(result.max, value);
            }
            
            result.average = result.sum / result.count;
            
            completed_windows_.push_back(result);
        }
    };
};

/**
 * Window manager that coordinates multiple window types
 */
class WindowManager {
public:
    struct Config {
        bool enable_sliding = true;
        bool enable_tumbling = true;
        bool enable_session = true;
        bool enable_hopping = true;
        
        WindowedOperations::SlidingWindow::Config sliding_config;
        WindowedOperations::TumblingWindow::Config tumbling_config;
        WindowedOperations::SessionWindow::Config session_config;
        WindowedOperations::HoppingWindow::Config hopping_config;
    };
    
    explicit WindowManager(const Config& config = {}) : config_(config) {
        if (config_.enable_sliding) {
            sliding_window_ = std::make_unique<WindowedOperations::SlidingWindow>(config_.sliding_config);
        }
        if (config_.enable_tumbling) {
            tumbling_window_ = std::make_unique<WindowedOperations::TumblingWindow>(config_.tumbling_config);
        }
        if (config_.enable_session) {
            session_window_ = std::make_unique<WindowedOperations::SessionWindow>(config_.session_config);
        }
        if (config_.enable_hopping) {
            hopping_window_ = std::make_unique<WindowedOperations::HoppingWindow>(config_.hopping_config);
        }
    }
    
    void process_event(const StreamEvent& event, double value) {
        if (sliding_window_) {
            sliding_window_->add_value(event.timestamp_ns, value);
        }
        if (tumbling_window_) {
            tumbling_window_->add_value(event.timestamp_ns, value);
        }
        if (session_window_) {
            session_window_->add_value(event.timestamp_ns, value, event.user_id);
        }
        if (hopping_window_) {
            hopping_window_->add_value(event.timestamp_ns, value);
        }
    }
    
    // Get results from all window types
    struct AllWindowResults {
        std::vector<WindowedOperations::TumblingWindow::WindowResult> tumbling_results;
        std::vector<WindowedOperations::SessionWindow::SessionResult> session_results;
        std::vector<WindowedOperations::HoppingWindow::HoppingWindowResult> hopping_results;
        
        // Sliding window current state
        double sliding_average = 0.0;
        double sliding_weighted_average = 0.0;
        std::pair<double, double> sliding_min_max = {0.0, 0.0};
        double sliding_p95 = 0.0;
        size_t sliding_count = 0;
    };
    
    AllWindowResults get_all_results() {
        AllWindowResults results;
        
        if (tumbling_window_) {
            results.tumbling_results = tumbling_window_->get_completed_windows();
        }
        if (session_window_) {
            results.session_results = session_window_->get_completed_sessions();
        }
        if (hopping_window_) {
            results.hopping_results = hopping_window_->get_completed_windows();
        }
        if (sliding_window_) {
            results.sliding_average = sliding_window_->calculate_average();
            results.sliding_weighted_average = sliding_window_->calculate_weighted_average();
            results.sliding_min_max = sliding_window_->calculate_min_max();
            results.sliding_p95 = sliding_window_->calculate_percentile(0.95);
            results.sliding_count = sliding_window_->size();
        }
        
        return results;
    }
    
    void cleanup_expired(uint64_t current_time_ns) {
        if (session_window_) {
            session_window_->cleanup_expired_sessions(current_time_ns);
        }
    }
    
private:
    Config config_;
    std::unique_ptr<WindowedOperations::SlidingWindow> sliding_window_;
    std::unique_ptr<WindowedOperations::TumblingWindow> tumbling_window_;
    std::unique_ptr<WindowedOperations::SessionWindow> session_window_;
    std::unique_ptr<WindowedOperations::HoppingWindow> hopping_window_;
};

} // namespace stream
} // namespace ultra_cpp