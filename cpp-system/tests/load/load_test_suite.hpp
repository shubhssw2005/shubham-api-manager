#pragma once

#include "../framework/test_framework.hpp"
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <chrono>
#include <atomic>
#include <thread>
#include <random>

namespace ultra::testing::load {

// Traffic pattern definitions
enum class TrafficPattern {
    CONSTANT,           // Constant rate
    RAMP_UP,           // Gradually increasing
    RAMP_DOWN,         // Gradually decreasing
    SPIKE,             // Sudden spike then return to baseline
    BURST,             // Periodic bursts
    RANDOM,            // Random intervals
    REALISTIC_WEB      // Realistic web traffic pattern
};

// Load test configuration
struct LoadTestConfig {
    TrafficPattern pattern = TrafficPattern::CONSTANT;
    size_t base_rps = 1000;                    // Base requests per second
    size_t peak_rps = 5000;                    // Peak requests per second
    std::chrono::seconds duration{60};          // Test duration
    size_t thread_count = 4;                   // Number of worker threads
    std::chrono::milliseconds ramp_time{30000}; // Time to reach peak
    std::chrono::milliseconds burst_interval{10000}; // Interval between bursts
    std::chrono::milliseconds burst_duration{2000};  // Duration of each burst
    double spike_multiplier = 10.0;            // Spike intensity multiplier
    std::string target_endpoint = "http://localhost:8080";
    
    // Request configuration
    struct RequestConfig {
        std::string method = "GET";
        std::string path = "/";
        std::map<std::string, std::string> headers;
        std::string body;
        std::chrono::milliseconds timeout{5000};
    } request_config;
    
    // Validation configuration
    struct ValidationConfig {
        std::vector<int> expected_status_codes = {200};
        std::function<bool(const std::string&)> response_validator;
        uint64_t max_response_time_ms = 1000;
        double max_error_rate = 0.01; // 1% max error rate
    } validation_config;
};

// Load test results
struct LoadTestResults {
    size_t total_requests = 0;
    size_t successful_requests = 0;
    size_t failed_requests = 0;
    size_t timeout_requests = 0;
    
    struct LatencyStats {
        uint64_t min_ms = 0;
        uint64_t max_ms = 0;
        uint64_t p50_ms = 0;
        uint64_t p95_ms = 0;
        uint64_t p99_ms = 0;
        uint64_t p999_ms = 0;
        double mean_ms = 0.0;
        double stddev_ms = 0.0;
    } latency_stats;
    
    struct ThroughputStats {
        double actual_rps = 0.0;
        double peak_rps = 0.0;
        double min_rps = 0.0;
        std::vector<double> rps_over_time;
    } throughput_stats;
    
    struct ErrorStats {
        double error_rate = 0.0;
        double timeout_rate = 0.0;
        std::map<int, size_t> status_code_counts;
        std::vector<std::string> error_messages;
    } error_stats;
    
    std::chrono::milliseconds actual_duration{0};
    bool passed_sla = false;
};

// HTTP client for load testing
class HTTPClient {
public:
    struct Response {
        int status_code = 0;
        std::string body;
        std::map<std::string, std::string> headers;
        uint64_t response_time_ms = 0;
        bool success = false;
        std::string error_message;
    };
    
    explicit HTTPClient(const std::string& base_url);
    ~HTTPClient();
    
    Response send_request(const LoadTestConfig::RequestConfig& config);
    void set_timeout(std::chrono::milliseconds timeout);
    void set_connection_pool_size(size_t size);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

// Traffic pattern generator
class TrafficGenerator {
public:
    explicit TrafficGenerator(const LoadTestConfig& config);
    
    // Get the target RPS at a given time offset
    double get_target_rps(std::chrono::milliseconds elapsed) const;
    
    // Get the next request time for a worker thread
    std::chrono::high_resolution_clock::time_point get_next_request_time(
        int worker_id, std::chrono::high_resolution_clock::time_point current_time) const;

private:
    LoadTestConfig config_;
    mutable std::mt19937 rng_;
    mutable std::mutex rng_mutex_;
    
    double calculate_constant_rps(std::chrono::milliseconds elapsed) const;
    double calculate_ramp_up_rps(std::chrono::milliseconds elapsed) const;
    double calculate_ramp_down_rps(std::chrono::milliseconds elapsed) const;
    double calculate_spike_rps(std::chrono::milliseconds elapsed) const;
    double calculate_burst_rps(std::chrono::milliseconds elapsed) const;
    double calculate_random_rps(std::chrono::milliseconds elapsed) const;
    double calculate_realistic_web_rps(std::chrono::milliseconds elapsed) const;
};

// Main load test runner
class LoadTestRunner {
public:
    explicit LoadTestRunner(const LoadTestConfig& config);
    
    LoadTestResults run_test();
    void stop_test();
    
    // Real-time monitoring
    struct RealTimeStats {
        double current_rps = 0.0;
        size_t active_requests = 0;
        double current_error_rate = 0.0;
        uint64_t current_p99_latency_ms = 0;
    };
    
    RealTimeStats get_real_time_stats() const;
    
    // Progress callback
    using ProgressCallback = std::function<void(double progress, const RealTimeStats&)>;
    void set_progress_callback(ProgressCallback callback);

private:
    LoadTestConfig config_;
    std::unique_ptr<TrafficGenerator> traffic_generator_;
    std::unique_ptr<HTTPClient> http_client_;
    
    std::atomic<bool> running_{false};
    std::atomic<bool> stop_requested_{false};
    
    // Statistics collection
    std::atomic<size_t> total_requests_{0};
    std::atomic<size_t> successful_requests_{0};
    std::atomic<size_t> failed_requests_{0};
    std::atomic<size_t> timeout_requests_{0};
    
    mutable std::mutex latency_mutex_;
    std::vector<uint64_t> latencies_;
    
    mutable std::mutex error_mutex_;
    std::map<int, size_t> status_code_counts_;
    std::vector<std::string> error_messages_;
    
    ProgressCallback progress_callback_;
    
    void worker_thread(int worker_id);
    void monitoring_thread();
    LoadTestResults compile_results(std::chrono::milliseconds actual_duration);
};

// Predefined load test scenarios
class LoadTestScenarios {
public:
    // Basic scenarios
    static LoadTestConfig constant_load(size_t rps, std::chrono::seconds duration);
    static LoadTestConfig ramp_up_test(size_t start_rps, size_t end_rps, 
                                     std::chrono::seconds duration);
    static LoadTestConfig spike_test(size_t base_rps, size_t spike_rps, 
                                   std::chrono::seconds duration);
    static LoadTestConfig burst_test(size_t base_rps, size_t burst_rps,
                                   std::chrono::milliseconds burst_interval,
                                   std::chrono::milliseconds burst_duration,
                                   std::chrono::seconds total_duration);
    
    // Advanced scenarios
    static LoadTestConfig stress_test(const std::string& endpoint);
    static LoadTestConfig endurance_test(const std::string& endpoint);
    static LoadTestConfig capacity_test(const std::string& endpoint);
    static LoadTestConfig realistic_web_traffic(const std::string& endpoint);
    
    // API-specific scenarios
    static LoadTestConfig cache_heavy_workload(const std::string& endpoint);
    static LoadTestConfig write_heavy_workload(const std::string& endpoint);
    static LoadTestConfig mixed_workload(const std::string& endpoint);
};

// Load test suite for running multiple scenarios
class LoadTestSuite {
public:
    struct TestCase {
        std::string name;
        LoadTestConfig config;
        std::function<bool(const LoadTestResults&)> success_criteria;
    };
    
    void add_test_case(const TestCase& test_case);
    void add_test_case(const std::string& name, const LoadTestConfig& config);
    
    struct SuiteResults {
        std::vector<std::pair<std::string, LoadTestResults>> test_results;
        size_t passed_tests = 0;
        size_t failed_tests = 0;
        bool overall_success = false;
    };
    
    SuiteResults run_suite();
    
    // Report generation
    void generate_html_report(const SuiteResults& results, const std::string& filename);
    void generate_json_report(const SuiteResults& results, const std::string& filename);
    void generate_csv_report(const SuiteResults& results, const std::string& filename);

private:
    std::vector<TestCase> test_cases_;
};

} // namespace ultra::testing::load