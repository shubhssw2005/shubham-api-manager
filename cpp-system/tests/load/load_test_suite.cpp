#include "load_test_suite.hpp"
#include <curl/curl.h>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <json/json.h>

namespace ultra::testing::load {

// HTTPClient implementation
class HTTPClient::Impl {
public:
    Impl(const std::string& base_url) : base_url_(base_url) {
        curl_global_init(CURL_GLOBAL_DEFAULT);
        curl_handle_ = curl_easy_init();
        if (!curl_handle_) {
            throw std::runtime_error("Failed to initialize CURL");
        }
    }
    
    ~Impl() {
        if (curl_handle_) {
            curl_easy_cleanup(curl_handle_);
        }
        curl_global_cleanup();
    }
    
    HTTPClient::Response send_request(const LoadTestConfig::RequestConfig& config) {
        HTTPClient::Response response;
        auto start_time = std::chrono::high_resolution_clock::now();
        
        std::string url = base_url_ + config.path;
        std::string response_body;
        
        // Setup CURL options
        curl_easy_setopt(curl_handle_, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl_handle_, CURLOPT_WRITEFUNCTION, write_callback);
        curl_easy_setopt(curl_handle_, CURLOPT_WRITEDATA, &response_body);
        curl_easy_setopt(curl_handle_, CURLOPT_TIMEOUT_MS, timeout_ms_);
        curl_easy_setopt(curl_handle_, CURLOPT_FOLLOWLOCATION, 1L);
        
        // Set HTTP method
        if (config.method == "POST") {
            curl_easy_setopt(curl_handle_, CURLOPT_POST, 1L);
            curl_easy_setopt(curl_handle_, CURLOPT_POSTFIELDS, config.body.c_str());
        } else if (config.method == "PUT") {
            curl_easy_setopt(curl_handle_, CURLOPT_CUSTOMREQUEST, "PUT");
            curl_easy_setopt(curl_handle_, CURLOPT_POSTFIELDS, config.body.c_str());
        } else if (config.method == "DELETE") {
            curl_easy_setopt(curl_handle_, CURLOPT_CUSTOMREQUEST, "DELETE");
        }
        
        // Set headers
        struct curl_slist* headers = nullptr;
        for (const auto& [key, value] : config.headers) {
            std::string header = key + ": " + value;
            headers = curl_slist_append(headers, header.c_str());
        }
        if (headers) {
            curl_easy_setopt(curl_handle_, CURLOPT_HTTPHEADER, headers);
        }
        
        // Perform request
        CURLcode res = curl_easy_perform(curl_handle_);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        response.response_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time).count();
        
        if (res == CURLE_OK) {
            long status_code;
            curl_easy_getinfo(curl_handle_, CURLINFO_RESPONSE_CODE, &status_code);
            response.status_code = static_cast<int>(status_code);
            response.body = response_body;
            response.success = true;
        } else {
            response.success = false;
            response.error_message = curl_easy_strerror(res);
        }
        
        // Cleanup
        if (headers) {
            curl_slist_free_all(headers);
        }
        
        return response;
    }
    
    void set_timeout(std::chrono::milliseconds timeout) {
        timeout_ms_ = timeout.count();
    }

private:
    std::string base_url_;
    CURL* curl_handle_;
    long timeout_ms_ = 5000;
    
    static size_t write_callback(void* contents, size_t size, size_t nmemb, std::string* userp) {
        size_t total_size = size * nmemb;
        userp->append(static_cast<char*>(contents), total_size);
        return total_size;
    }
};

HTTPClient::HTTPClient(const std::string& base_url) 
    : impl_(std::make_unique<Impl>(base_url)) {}

HTTPClient::~HTTPClient() = default;

HTTPClient::Response HTTPClient::send_request(const LoadTestConfig::RequestConfig& config) {
    return impl_->send_request(config);
}

void HTTPClient::set_timeout(std::chrono::milliseconds timeout) {
    impl_->set_timeout(timeout);
}

void HTTPClient::set_connection_pool_size(size_t size) {
    // Implementation would configure connection pooling
}

// TrafficGenerator implementation
TrafficGenerator::TrafficGenerator(const LoadTestConfig& config) 
    : config_(config), rng_(std::random_device{}()) {}

double TrafficGenerator::get_target_rps(std::chrono::milliseconds elapsed) const {
    switch (config_.pattern) {
        case TrafficPattern::CONSTANT:
            return calculate_constant_rps(elapsed);
        case TrafficPattern::RAMP_UP:
            return calculate_ramp_up_rps(elapsed);
        case TrafficPattern::RAMP_DOWN:
            return calculate_ramp_down_rps(elapsed);
        case TrafficPattern::SPIKE:
            return calculate_spike_rps(elapsed);
        case TrafficPattern::BURST:
            return calculate_burst_rps(elapsed);
        case TrafficPattern::RANDOM:
            return calculate_random_rps(elapsed);
        case TrafficPattern::REALISTIC_WEB:
            return calculate_realistic_web_rps(elapsed);
        default:
            return config_.base_rps;
    }
}

std::chrono::high_resolution_clock::time_point TrafficGenerator::get_next_request_time(
    int worker_id, std::chrono::high_resolution_clock::time_point current_time) const {
    
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        current_time - std::chrono::high_resolution_clock::time_point{});
    
    double target_rps = get_target_rps(elapsed);
    double worker_rps = target_rps / config_.thread_count;
    
    // Calculate interval between requests for this worker
    auto interval_ns = static_cast<uint64_t>(1e9 / worker_rps);
    
    // Add some jitter to avoid thundering herd
    std::lock_guard<std::mutex> lock(rng_mutex_);
    std::uniform_real_distribution<> jitter_dist(0.8, 1.2);
    interval_ns = static_cast<uint64_t>(interval_ns * jitter_dist(rng_));
    
    return current_time + std::chrono::nanoseconds(interval_ns);
}

double TrafficGenerator::calculate_constant_rps(std::chrono::milliseconds elapsed) const {
    return config_.base_rps;
}

double TrafficGenerator::calculate_ramp_up_rps(std::chrono::milliseconds elapsed) const {
    if (elapsed >= config_.ramp_time) {
        return config_.peak_rps;
    }
    
    double progress = static_cast<double>(elapsed.count()) / config_.ramp_time.count();
    return config_.base_rps + (config_.peak_rps - config_.base_rps) * progress;
}

double TrafficGenerator::calculate_ramp_down_rps(std::chrono::milliseconds elapsed) const {
    if (elapsed >= config_.ramp_time) {
        return config_.base_rps;
    }
    
    double progress = static_cast<double>(elapsed.count()) / config_.ramp_time.count();
    return config_.peak_rps - (config_.peak_rps - config_.base_rps) * progress;
}

double TrafficGenerator::calculate_spike_rps(std::chrono::milliseconds elapsed) const {
    auto spike_start = config_.duration / 2;
    auto spike_duration = std::chrono::milliseconds(5000); // 5 second spike
    
    if (elapsed >= spike_start && elapsed < spike_start + spike_duration) {
        return config_.base_rps * config_.spike_multiplier;
    }
    
    return config_.base_rps;
}

double TrafficGenerator::calculate_burst_rps(std::chrono::milliseconds elapsed) const {
    auto cycle_position = elapsed.count() % config_.burst_interval.count();
    
    if (cycle_position < config_.burst_duration.count()) {
        return config_.peak_rps;
    }
    
    return config_.base_rps;
}

double TrafficGenerator::calculate_random_rps(std::chrono::milliseconds elapsed) const {
    std::lock_guard<std::mutex> lock(rng_mutex_);
    std::uniform_real_distribution<> dist(config_.base_rps * 0.5, config_.peak_rps);
    return dist(rng_);
}

double TrafficGenerator::calculate_realistic_web_rps(std::chrono::milliseconds elapsed) const {
    // Simulate realistic web traffic with daily patterns
    double hours = elapsed.count() / (1000.0 * 3600.0);
    double daily_cycle = std::sin(2 * M_PI * hours / 24.0); // 24-hour cycle
    double weekly_cycle = std::sin(2 * M_PI * hours / (24.0 * 7.0)); // Weekly cycle
    
    // Add some randomness
    std::lock_guard<std::mutex> lock(rng_mutex_);
    std::normal_distribution<> noise_dist(0.0, 0.1);
    double noise = noise_dist(rng_);
    
    double factor = 0.5 + 0.3 * daily_cycle + 0.1 * weekly_cycle + noise;
    factor = std::max(0.1, std::min(1.0, factor)); // Clamp between 0.1 and 1.0
    
    return config_.base_rps + (config_.peak_rps - config_.base_rps) * factor;
}

// LoadTestRunner implementation
LoadTestRunner::LoadTestRunner(const LoadTestConfig& config) 
    : config_(config) {
    traffic_generator_ = std::make_unique<TrafficGenerator>(config);
    http_client_ = std::make_unique<HTTPClient>(config.target_endpoint);
    http_client_->set_timeout(config.request_config.timeout);
}

LoadTestResults LoadTestRunner::run_test() {
    running_ = true;
    stop_requested_ = false;
    
    // Reset counters
    total_requests_ = 0;
    successful_requests_ = 0;
    failed_requests_ = 0;
    timeout_requests_ = 0;
    
    {
        std::lock_guard<std::mutex> lock(latency_mutex_);
        latencies_.clear();
    }
    
    {
        std::lock_guard<std::mutex> lock(error_mutex_);
        status_code_counts_.clear();
        error_messages_.clear();
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Start worker threads
    std::vector<std::thread> workers;
    workers.reserve(config_.thread_count);
    
    for (size_t i = 0; i < config_.thread_count; ++i) {
        workers.emplace_back(&LoadTestRunner::worker_thread, this, static_cast<int>(i));
    }
    
    // Start monitoring thread
    std::thread monitor(&LoadTestRunner::monitoring_thread, this);
    
    // Wait for test duration or stop request
    auto end_time = start_time + config_.duration;
    while (std::chrono::high_resolution_clock::now() < end_time && !stop_requested_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    running_ = false;
    
    // Wait for all threads to complete
    for (auto& worker : workers) {
        worker.join();
    }
    monitor.join();
    
    auto actual_end_time = std::chrono::high_resolution_clock::now();
    auto actual_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        actual_end_time - start_time);
    
    return compile_results(actual_duration);
}

void LoadTestRunner::stop_test() {
    stop_requested_ = true;
}

LoadTestRunner::RealTimeStats LoadTestRunner::get_real_time_stats() const {
    RealTimeStats stats;
    
    // Calculate current RPS (approximate)
    static auto last_check = std::chrono::high_resolution_clock::now();
    static size_t last_request_count = 0;
    
    auto now = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_check);
    
    if (elapsed.count() >= 1000) { // Update every second
        size_t current_requests = total_requests_.load();
        stats.current_rps = (current_requests - last_request_count) / (elapsed.count() / 1000.0);
        last_check = now;
        last_request_count = current_requests;
    }
    
    // Calculate error rate
    size_t total = total_requests_.load();
    size_t failed = failed_requests_.load();
    stats.current_error_rate = total > 0 ? static_cast<double>(failed) / total : 0.0;
    
    // Calculate current P99 latency
    {
        std::lock_guard<std::mutex> lock(latency_mutex_);
        if (!latencies_.empty()) {
            auto sorted_latencies = latencies_;
            std::sort(sorted_latencies.begin(), sorted_latencies.end());
            size_t p99_index = static_cast<size_t>(sorted_latencies.size() * 0.99);
            stats.current_p99_latency_ms = sorted_latencies[p99_index];
        }
    }
    
    return stats;
}

void LoadTestRunner::set_progress_callback(ProgressCallback callback) {
    progress_callback_ = callback;
}

void LoadTestRunner::worker_thread(int worker_id) {
    auto next_request_time = std::chrono::high_resolution_clock::now();
    
    while (running_ && !stop_requested_) {
        auto now = std::chrono::high_resolution_clock::now();
        
        if (now >= next_request_time) {
            // Send request
            auto response = http_client_->send_request(config_.request_config);
            
            // Update statistics
            total_requests_++;
            
            if (response.success) {
                successful_requests_++;
                
                // Record latency
                {
                    std::lock_guard<std::mutex> lock(latency_mutex_);
                    latencies_.push_back(response.response_time_ms);
                }
                
                // Record status code
                {
                    std::lock_guard<std::mutex> lock(error_mutex_);
                    status_code_counts_[response.status_code]++;
                }
                
                // Validate response
                bool valid = std::find(config_.validation_config.expected_status_codes.begin(),
                                     config_.validation_config.expected_status_codes.end(),
                                     response.status_code) != 
                           config_.validation_config.expected_status_codes.end();
                
                if (config_.validation_config.response_validator) {
                    valid = valid && config_.validation_config.response_validator(response.body);
                }
                
                if (!valid) {
                    failed_requests_++;
                }
                
                if (response.response_time_ms > config_.validation_config.max_response_time_ms) {
                    timeout_requests_++;
                }
            } else {
                failed_requests_++;
                
                std::lock_guard<std::mutex> lock(error_mutex_);
                error_messages_.push_back(response.error_message);
            }
            
            // Calculate next request time
            next_request_time = traffic_generator_->get_next_request_time(worker_id, now);
        } else {
            // Sleep until next request time
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    }
}

void LoadTestRunner::monitoring_thread() {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    while (running_ && !stop_requested_) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        
        if (progress_callback_) {
            auto now = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time);
            double progress = static_cast<double>(elapsed.count()) / config_.duration.count();
            
            auto stats = get_real_time_stats();
            progress_callback_(progress, stats);
        }
    }
}

LoadTestResults LoadTestRunner::compile_results(std::chrono::milliseconds actual_duration) {
    LoadTestResults results;
    
    results.total_requests = total_requests_.load();
    results.successful_requests = successful_requests_.load();
    results.failed_requests = failed_requests_.load();
    results.timeout_requests = timeout_requests_.load();
    results.actual_duration = actual_duration;
    
    // Calculate latency statistics
    {
        std::lock_guard<std::mutex> lock(latency_mutex_);
        if (!latencies_.empty()) {
            auto sorted = latencies_;
            std::sort(sorted.begin(), sorted.end());
            
            results.latency_stats.min_ms = sorted.front();
            results.latency_stats.max_ms = sorted.back();
            results.latency_stats.p50_ms = sorted[sorted.size() * 50 / 100];
            results.latency_stats.p95_ms = sorted[sorted.size() * 95 / 100];
            results.latency_stats.p99_ms = sorted[sorted.size() * 99 / 100];
            results.latency_stats.p999_ms = sorted[sorted.size() * 999 / 1000];
            
            uint64_t sum = std::accumulate(sorted.begin(), sorted.end(), 0ULL);
            results.latency_stats.mean_ms = static_cast<double>(sum) / sorted.size();
            
            double variance = 0.0;
            for (uint64_t latency : sorted) {
                double diff = static_cast<double>(latency) - results.latency_stats.mean_ms;
                variance += diff * diff;
            }
            variance /= sorted.size();
            results.latency_stats.stddev_ms = std::sqrt(variance);
        }
    }
    
    // Calculate throughput statistics
    double duration_seconds = actual_duration.count() / 1000.0;
    results.throughput_stats.actual_rps = results.total_requests / duration_seconds;
    
    // Calculate error statistics
    results.error_stats.error_rate = results.total_requests > 0 ? 
        static_cast<double>(results.failed_requests) / results.total_requests : 0.0;
    results.error_stats.timeout_rate = results.total_requests > 0 ?
        static_cast<double>(results.timeout_requests) / results.total_requests : 0.0;
    
    {
        std::lock_guard<std::mutex> lock(error_mutex_);
        results.error_stats.status_code_counts = status_code_counts_;
        results.error_stats.error_messages = error_messages_;
    }
    
    // Check SLA compliance
    results.passed_sla = 
        results.error_stats.error_rate <= config_.validation_config.max_error_rate &&
        results.latency_stats.p99_ms <= config_.validation_config.max_response_time_ms;
    
    return results;
}

// LoadTestScenarios implementation
LoadTestConfig LoadTestScenarios::constant_load(size_t rps, std::chrono::seconds duration) {
    LoadTestConfig config;
    config.pattern = TrafficPattern::CONSTANT;
    config.base_rps = rps;
    config.duration = duration;
    return config;
}

LoadTestConfig LoadTestScenarios::ramp_up_test(size_t start_rps, size_t end_rps, 
                                             std::chrono::seconds duration) {
    LoadTestConfig config;
    config.pattern = TrafficPattern::RAMP_UP;
    config.base_rps = start_rps;
    config.peak_rps = end_rps;
    config.duration = duration;
    config.ramp_time = duration;
    return config;
}

LoadTestConfig LoadTestScenarios::spike_test(size_t base_rps, size_t spike_rps, 
                                           std::chrono::seconds duration) {
    LoadTestConfig config;
    config.pattern = TrafficPattern::SPIKE;
    config.base_rps = base_rps;
    config.peak_rps = spike_rps;
    config.duration = duration;
    return config;
}

LoadTestConfig LoadTestScenarios::stress_test(const std::string& endpoint) {
    LoadTestConfig config;
    config.pattern = TrafficPattern::RAMP_UP;
    config.base_rps = 100;
    config.peak_rps = 10000;
    config.duration = std::chrono::minutes(10);
    config.ramp_time = std::chrono::minutes(5);
    config.target_endpoint = endpoint;
    config.validation_config.max_error_rate = 0.05; // 5% error rate allowed for stress test
    return config;
}

LoadTestConfig LoadTestScenarios::realistic_web_traffic(const std::string& endpoint) {
    LoadTestConfig config;
    config.pattern = TrafficPattern::REALISTIC_WEB;
    config.base_rps = 500;
    config.peak_rps = 2000;
    config.duration = std::chrono::hours(1);
    config.target_endpoint = endpoint;
    return config;
}

} // namespace ultra::testing::load