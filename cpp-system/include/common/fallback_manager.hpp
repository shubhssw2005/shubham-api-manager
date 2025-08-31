#pragma once

#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include <future>

namespace ultra {
namespace common {

/**
 * Fallback Manager for Node.js Integration
 * 
 * Provides automatic fallback mechanisms to Node.js layer when
 * C++ components fail or performance degrades beyond acceptable limits.
 */
class FallbackManager {
public:
    enum class FallbackReason {
        CIRCUIT_BREAKER_OPEN,
        PERFORMANCE_DEGRADATION,
        COMPONENT_FAILURE,
        RESOURCE_EXHAUSTION,
        TIMEOUT,
        MANUAL_OVERRIDE
    };

    struct FallbackConfig {
        // Node.js service endpoints
        std::string nodejs_base_url = "http://localhost:3005";
        std::string health_check_endpoint = "/health";
        
        // Timeout settings
        std::chrono::milliseconds request_timeout{5000};
        std::chrono::milliseconds health_check_timeout{1000};
        
        // Retry settings
        uint32_t max_retries = 3;
        std::chrono::milliseconds retry_delay{100};
        double retry_backoff_multiplier = 2.0;
        
        // Health check settings
        std::chrono::milliseconds health_check_interval{30000};
        uint32_t consecutive_failures_threshold = 3;
        
        // Fallback decision settings
        bool enable_automatic_fallback = true;
        double performance_threshold_multiplier = 2.0; // 2x slower than target
        uint64_t max_response_time_ns = 10000000; // 10ms
    };

    struct FallbackStats {
        uint64_t total_fallbacks = 0;
        uint64_t successful_fallbacks = 0;
        uint64_t failed_fallbacks = 0;
        uint64_t nodejs_requests = 0;
        uint64_t nodejs_failures = 0;
        uint64_t health_checks_performed = 0;
        uint64_t health_check_failures = 0;
        bool nodejs_available = true;
    };

    using FallbackHandler = std::function<std::string(const std::string& request_data)>;
    using HealthCheckCallback = std::function<void(bool is_healthy)>;

    explicit FallbackManager(const FallbackConfig& config);
    ~FallbackManager();

    // Start/stop health monitoring
    void start_health_monitoring();
    void stop_health_monitoring();

    // Register fallback handlers for different endpoints
    void register_fallback_handler(const std::string& endpoint, FallbackHandler handler);
    void unregister_fallback_handler(const std::string& endpoint);

    // Execute request with automatic fallback
    template<typename Func, typename... Args>
    auto execute_with_fallback(const std::string& endpoint, Func&& cpp_func, Args&&... args) 
        -> std::future<decltype(cpp_func(args...))> {
        
        using ReturnType = decltype(cpp_func(args...));
        auto promise = std::make_shared<std::promise<ReturnType>>();
        auto future = promise->get_future();

        std::thread([this, endpoint, cpp_func = std::forward<Func>(cpp_func), 
                    promise, args...]() mutable {
            try {
                // Try C++ implementation first
                auto start_time = std::chrono::steady_clock::now();
                auto result = cpp_func(args...);
                auto end_time = std::chrono::steady_clock::now();
                
                auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
                    end_time - start_time).count();
                
                // Check if performance is acceptable
                if (should_fallback_due_to_performance(duration)) {
                    LOG_WARNING("C++ performance degraded ({}ns), falling back to Node.js for {}", 
                               duration, endpoint);
                    
                    auto fallback_result = execute_nodejs_fallback(endpoint, "");
                    if constexpr (std::is_same_v<ReturnType, std::string>) {
                        promise->set_value(fallback_result);
                    } else {
                        // For non-string types, would need proper serialization
                        promise->set_value(result);
                    }
                } else {
                    promise->set_value(result);
                }
                
            } catch (const std::exception& e) {
                LOG_ERROR("C++ implementation failed for {}: {}", endpoint, e.what());
                
                try {
                    auto fallback_result = execute_nodejs_fallback(endpoint, "");
                    record_fallback(endpoint, FallbackReason::COMPONENT_FAILURE, true);
                    
                    if constexpr (std::is_same_v<ReturnType, std::string>) {
                        promise->set_value(fallback_result);
                    } else {
                        promise->set_exception(std::current_exception());
                    }
                } catch (const std::exception& fallback_error) {
                    LOG_ERROR("Fallback also failed for {}: {}", endpoint, fallback_error.what());
                    record_fallback(endpoint, FallbackReason::COMPONENT_FAILURE, false);
                    promise->set_exception(std::current_exception());
                }
            }
        }).detach();

        return future;
    }

    // Manual fallback execution
    std::string execute_nodejs_fallback(const std::string& endpoint, const std::string& request_data);
    
    // Fallback decision methods
    bool should_fallback(const std::string& endpoint, FallbackReason reason) const;
    void force_fallback(const std::string& endpoint, bool enable);
    void reset_fallback_state(const std::string& endpoint);

    // Health check methods
    bool is_nodejs_healthy() const { return stats_.nodejs_available.load(); }
    void perform_health_check();
    void register_health_check_callback(HealthCheckCallback callback);

    // Statistics and monitoring
    FallbackStats get_stats() const { 
        FallbackStats result;
        result.total_fallbacks = stats_.total_fallbacks.load();
        result.successful_fallbacks = stats_.successful_fallbacks.load();
        result.failed_fallbacks = stats_.failed_fallbacks.load();
        result.nodejs_requests = stats_.nodejs_requests.load();
        result.nodejs_failures = stats_.nodejs_failures.load();
        result.health_checks_performed = stats_.health_checks_performed.load();
        result.health_check_failures = stats_.health_check_failures.load();
        result.nodejs_available = stats_.nodejs_available.load();
        return result;
    }
    
    struct EndpointStats {
        uint64_t cpp_requests = 0;
        uint64_t cpp_failures = 0;
        uint64_t fallback_requests = 0;
        uint64_t fallback_failures = 0;
        uint64_t avg_cpp_response_time_ns = 0;
        uint64_t avg_fallback_response_time_ns = 0;
        bool forced_fallback = false;
    };
    
    std::unordered_map<std::string, EndpointStats> get_endpoint_stats() const;

private:
    const FallbackConfig config_;
    
    struct AtomicFallbackStats {
        std::atomic<uint64_t> total_fallbacks{0};
        std::atomic<uint64_t> successful_fallbacks{0};
        std::atomic<uint64_t> failed_fallbacks{0};
        std::atomic<uint64_t> nodejs_requests{0};
        std::atomic<uint64_t> nodejs_failures{0};
        std::atomic<uint64_t> health_checks_performed{0};
        std::atomic<uint64_t> health_check_failures{0};
        std::atomic<bool> nodejs_available{true};
    };
    
    AtomicFallbackStats stats_;
    
    mutable std::mutex handlers_mutex_;
    std::unordered_map<std::string, FallbackHandler> fallback_handlers_;
    
    mutable std::mutex endpoint_stats_mutex_;
    std::unordered_map<std::string, EndpointStats> endpoint_stats_;
    
    mutable std::mutex forced_fallbacks_mutex_;
    std::unordered_set<std::string> forced_fallback_endpoints_;
    
    std::vector<HealthCheckCallback> health_check_callbacks_;
    std::unique_ptr<std::thread> health_monitor_thread_;
    std::atomic<bool> health_monitoring_active_{false};
    
    void health_monitoring_loop();
    bool should_fallback_due_to_performance(uint64_t response_time_ns) const;
    void record_fallback(const std::string& endpoint, FallbackReason reason, bool success);
    void update_endpoint_stats(const std::string& endpoint, bool is_cpp, bool success, uint64_t response_time_ns);
    
    // HTTP client methods (simplified implementation)
    std::string make_http_request(const std::string& url, const std::string& data, 
                                 std::chrono::milliseconds timeout);
    
    static std::string reason_to_string(FallbackReason reason);
};

/**
 * Fallback-aware operation wrapper
 * 
 * RAII class that automatically handles fallback logic for operations
 */
template<typename T>
class FallbackOperation {
public:
    FallbackOperation(const std::string& endpoint, FallbackManager& manager)
        : endpoint_(endpoint), manager_(manager), start_time_(std::chrono::steady_clock::now()) {}
    
    ~FallbackOperation() {
        if (!completed_) {
            // Operation was not completed, record as failure
            manager_.record_fallback(endpoint_, FallbackManager::FallbackReason::COMPONENT_FAILURE, false);
        }
    }
    
    void complete_successfully() {
        completed_ = true;
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - start_time_).count();
        manager_.update_endpoint_stats(endpoint_, true, true, duration);
    }
    
    void complete_with_failure() {
        completed_ = true;
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - start_time_).count();
        manager_.update_endpoint_stats(endpoint_, true, false, duration);
    }

private:
    std::string endpoint_;
    FallbackManager& manager_;
    std::chrono::steady_clock::time_point start_time_;
    bool completed_ = false;
};

/**
 * Node.js HTTP Client
 * 
 * Simple HTTP client for communicating with Node.js services
 */
class NodeJSClient {
public:
    struct Response {
        int status_code;
        std::string body;
        std::unordered_map<std::string, std::string> headers;
        std::chrono::milliseconds response_time;
    };

    explicit NodeJSClient(const std::string& base_url);
    
    Response get(const std::string& path, std::chrono::milliseconds timeout = std::chrono::milliseconds(5000));
    Response post(const std::string& path, const std::string& data, 
                 std::chrono::milliseconds timeout = std::chrono::milliseconds(5000));
    
    bool is_healthy(std::chrono::milliseconds timeout = std::chrono::milliseconds(1000));

private:
    std::string base_url_;
    
    Response make_request(const std::string& method, const std::string& path, 
                         const std::string& data, std::chrono::milliseconds timeout);
};

// Convenience macros
#define FALLBACK_EXECUTE(manager, endpoint, cpp_func, ...) \
    manager.execute_with_fallback(endpoint, cpp_func, __VA_ARGS__)

#define FALLBACK_OPERATION(endpoint, manager) \
    ultra::common::FallbackOperation<void> _fallback_op(endpoint, manager)

#define FALLBACK_COMPLETE_SUCCESS() \
    _fallback_op.complete_successfully()

#define FALLBACK_COMPLETE_FAILURE() \
    _fallback_op.complete_with_failure()

} // namespace common
} // namespace ultra