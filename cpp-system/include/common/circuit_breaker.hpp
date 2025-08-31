#pragma once

#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace ultra {
namespace common {

/**
 * Exception thrown when circuit breaker is open
 */
class CircuitBreakerOpenException : public std::runtime_error {
public:
    explicit CircuitBreakerOpenException(const std::string& circuit_name)
        : std::runtime_error("Circuit breaker '" + circuit_name + "' is open") {}
};

/**
 * Circuit Breaker Pattern Implementation with Exponential Backoff
 * 
 * Prevents cascading failures by monitoring operation success/failure rates
 * and temporarily blocking requests when failure threshold is exceeded.
 */
class CircuitBreaker {
public:
    enum class State {
        CLOSED,     // Normal operation
        OPEN,       // Blocking requests due to failures
        HALF_OPEN   // Testing if service has recovered
    };

    struct Config {
        // Failure threshold to open circuit (percentage)
        double failure_threshold = 0.5;
        
        // Minimum number of requests before evaluating failure rate
        uint32_t minimum_requests = 10;
        
        // Time window for failure rate calculation (ms)
        uint32_t time_window_ms = 60000;
        
        // Initial timeout when circuit opens (ms)
        uint32_t initial_timeout_ms = 5000;
        
        // Maximum timeout for exponential backoff (ms)
        uint32_t max_timeout_ms = 300000;
        
        // Backoff multiplier for exponential backoff
        double backoff_multiplier = 2.0;
        
        // Number of test requests in half-open state
        uint32_t half_open_max_calls = 3;
        
        // Success threshold to close circuit in half-open state
        double half_open_success_threshold = 0.8;
    };

    struct Stats {
        uint64_t total_requests = 0;
        uint64_t successful_requests = 0;
        uint64_t failed_requests = 0;
        uint64_t rejected_requests = 0;
        uint64_t state_changes = 0;
        uint64_t current_timeout_ms = 0;
    };

    explicit CircuitBreaker(const std::string& name, const Config& config);
    ~CircuitBreaker();

    // Execute operation with circuit breaker protection
    template<typename Func, typename... Args>
    auto execute(Func&& func, Args&&... args) -> decltype(func(args...)) {
        if (!allow_request()) {
            record_rejected();
            throw CircuitBreakerOpenException(name_);
        }

        auto start_time = std::chrono::steady_clock::now();
        
        try {
            auto result = func(std::forward<Args>(args)...);
            record_success(start_time);
            return result;
        } catch (...) {
            record_failure(start_time);
            throw;
        }
    }

    // Check if requests are allowed
    bool allow_request();
    
    // Manual success/failure recording
    void record_success(std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now());
    void record_failure(std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now());
    void record_rejected();

    // State management
    State get_state() const { return state_.load(); }
    std::string get_name() const { return name_; }
    Stats get_stats() const { 
        Stats result;
        result.total_requests = stats_.total_requests.load();
        result.successful_requests = stats_.successful_requests.load();
        result.failed_requests = stats_.failed_requests.load();
        result.rejected_requests = stats_.rejected_requests.load();
        result.state_changes = stats_.state_changes.load();
        result.current_timeout_ms = stats_.current_timeout_ms.load();
        return result;
    }
    
    // Force state changes (for testing)
    void force_open();
    void force_close();
    void reset();

private:
    struct RequestRecord {
        std::chrono::steady_clock::time_point timestamp;
        bool success;
        uint64_t duration_ns;
    };

    const std::string name_;
    const Config config_;
    
    std::atomic<State> state_{State::CLOSED};
    std::atomic<std::chrono::steady_clock::time_point> last_failure_time_;
    std::atomic<uint32_t> current_timeout_ms_;
    std::atomic<uint32_t> consecutive_failures_{0};
    std::atomic<uint32_t> half_open_calls_{0};
    std::atomic<uint32_t> half_open_successes_{0};
    
    mutable std::mutex records_mutex_;
    std::vector<RequestRecord> request_records_;
    
    struct AtomicStats {
        std::atomic<uint64_t> total_requests{0};
        std::atomic<uint64_t> successful_requests{0};
        std::atomic<uint64_t> failed_requests{0};
        std::atomic<uint64_t> rejected_requests{0};
        std::atomic<uint64_t> state_changes{0};
        std::atomic<uint64_t> current_timeout_ms{0};
    };
    
    AtomicStats stats_;

    void transition_to_open();
    void transition_to_half_open();
    void transition_to_closed();
    
    bool should_attempt_reset() const;
    double calculate_failure_rate() const;
    void cleanup_old_records();
    uint32_t calculate_next_timeout() const;
};

/**
 * Circuit Breaker Manager for managing multiple circuit breakers
 */
class CircuitBreakerManager {
public:
    static CircuitBreakerManager& instance();
    
    std::shared_ptr<CircuitBreaker> get_or_create(const std::string& name, 
                                                  const CircuitBreaker::Config& config);
    
    std::shared_ptr<CircuitBreaker> get(const std::string& name);
    
    void remove(const std::string& name);
    
    std::vector<std::string> list_circuit_breakers() const;
    
    // Get aggregated stats for all circuit breakers
    struct AggregatedStats {
        uint64_t total_circuits;
        uint64_t open_circuits;
        uint64_t half_open_circuits;
        uint64_t closed_circuits;
        uint64_t total_requests;
        uint64_t total_failures;
        uint64_t total_rejections;
    };
    
    AggregatedStats get_aggregated_stats() const;

private:
    CircuitBreakerManager() = default;
    
    mutable std::mutex breakers_mutex_;
    std::unordered_map<std::string, std::shared_ptr<CircuitBreaker>> breakers_;
};

} // namespace common
} // namespace ultra