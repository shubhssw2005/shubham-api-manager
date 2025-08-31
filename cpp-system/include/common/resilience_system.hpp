#pragma once

#include "common/circuit_breaker.hpp"
#include "common/degradation_manager.hpp"
#include "common/fallback_manager.hpp"
#include "common/logger.hpp"
#include <memory>
#include <string>
#include <functional>

namespace ultra {
namespace common {

/**
 * Comprehensive Resilience System
 * 
 * Integrates circuit breakers, graceful degradation, fallback mechanisms,
 * and structured logging into a unified resilience framework.
 */
class ResilienceSystem {
public:
    struct Config {
        CircuitBreaker::Config circuit_breaker_config;
        DegradationManager::DegradationConfig degradation_config;
        FallbackManager::FallbackConfig fallback_config;
        Logger::Config logger_config;
        
        // Integration settings
        bool enable_auto_fallback_on_circuit_open = true;
        bool enable_degradation_on_high_error_rate = true;
        double error_rate_degradation_threshold = 0.1; // 10%
        std::chrono::milliseconds metrics_collection_interval{1000};
    };

    explicit ResilienceSystem(const Config& config = {});
    ~ResilienceSystem();

    // Initialize and start all components
    void initialize();
    void start();
    void stop();
    void shutdown();

    // Get component instances
    std::shared_ptr<CircuitBreaker> get_circuit_breaker(const std::string& name);
    DegradationManager& get_degradation_manager() { return *degradation_manager_; }
    FallbackManager& get_fallback_manager() { return *fallback_manager_; }

    // High-level resilient execution
    template<typename Func, typename... Args>
    auto execute_resilient(const std::string& operation_name, Func&& func, Args&&... args) 
        -> std::future<decltype(func(args...))> {
        
        using ReturnType = decltype(func(args...));
        auto promise = std::make_shared<std::promise<ReturnType>>();
        auto future = promise->get_future();

        // Create performance timer context
        LogContext context;
        context.add("operation", operation_name)
               .add("resilience_enabled", "true");

        std::thread([this, operation_name, func = std::forward<Func>(func), 
                    promise, context, args...]() mutable {
            
            PERFORMANCE_TIMER_WITH_CONTEXT(operation_name, context);
            
            try {
                // Get circuit breaker for this operation
                auto circuit_breaker = get_circuit_breaker(operation_name);
                
                // Execute with circuit breaker protection
                auto result = circuit_breaker->execute([&]() {
                    // Check if operation should be degraded
                    if (!degradation_manager_->is_feature_enabled(operation_name)) {
                        LOG_INFO("Operation '{}' disabled due to degradation", operation_name);
                        throw DegradationException("Operation disabled due to system degradation");
                    }
                    
                    return func(args...);
                });
                
                promise->set_value(result);
                
            } catch (const CircuitBreakerOpenException& e) {
                LOG_WARNING("Circuit breaker open for '{}', attempting fallback", operation_name);
                
                if (config_.enable_auto_fallback_on_circuit_open) {
                    try {
                        auto fallback_result = fallback_manager_->execute_nodejs_fallback(
                            "/" + operation_name, "");
                        
                        if constexpr (std::is_same_v<ReturnType, std::string>) {
                            promise->set_value(fallback_result);
                        } else {
                            promise->set_exception(std::current_exception());
                        }
                    } catch (const std::exception& fallback_error) {
                        LOG_ERROR("Fallback failed for '{}': {}", operation_name, fallback_error.what());
                        promise->set_exception(std::current_exception());
                    }
                } else {
                    promise->set_exception(std::current_exception());
                }
                
            } catch (const std::exception& e) {
                LOG_ERROR_WITH_EXCEPTION("Operation failed", e, context);
                promise->set_exception(std::current_exception());
            }
        }).detach();

        return future;
    }

    // Register operation-specific configurations
    void register_operation(const std::string& operation_name, 
                           const CircuitBreaker::Config& cb_config = {},
                           DegradationManager::DegradationLevel min_degradation_level = DegradationManager::DegradationLevel::NORMAL);

    // Metrics and monitoring
    struct SystemStats {
        CircuitBreakerManager::AggregatedStats circuit_breaker_stats;
        DegradationManager::Stats degradation_stats;
        FallbackManager::FallbackStats fallback_stats;
        uint64_t total_operations = 0;
        uint64_t successful_operations = 0;
        uint64_t failed_operations = 0;
        double overall_success_rate = 0.0;
    };

    SystemStats get_system_stats() const;
    
    // Health check
    bool is_system_healthy() const;
    
    // Export metrics in Prometheus format
    std::string export_prometheus_metrics() const;

private:
    Config config_;
    
    std::unique_ptr<DegradationManager> degradation_manager_;
    std::unique_ptr<FallbackManager> fallback_manager_;
    
    std::unique_ptr<std::thread> metrics_thread_;
    std::atomic<bool> metrics_collection_active_{false};
    
    mutable std::mutex stats_mutex_;
    SystemStats cached_stats_;
    
    void setup_integration_callbacks();
    void metrics_collection_loop();
    void update_system_metrics();
    
    // Integration callbacks
    void on_degradation_change(DegradationManager::DegradationLevel old_level, 
                              DegradationManager::DegradationLevel new_level);
    void on_nodejs_health_change(bool is_healthy);
};

/**
 * Exception thrown when operation is disabled due to degradation
 */
class DegradationException : public std::runtime_error {
public:
    explicit DegradationException(const std::string& message)
        : std::runtime_error(message) {}
};

/**
 * Global resilience system instance
 */
class GlobalResilienceSystem {
public:
    static void initialize(const ResilienceSystem::Config& config = {});
    static ResilienceSystem& instance();
    static void shutdown();

private:
    static std::unique_ptr<ResilienceSystem> instance_;
    static std::mutex instance_mutex_;
};

// Convenience macros for resilient execution
#define EXECUTE_RESILIENT(operation_name, func, ...) \
    ultra::common::GlobalResilienceSystem::instance().execute_resilient(operation_name, func, __VA_ARGS__)

#define RESILIENT_OPERATION(operation_name) \
    auto _resilient_future = EXECUTE_RESILIENT(operation_name, [&]()

#define END_RESILIENT_OPERATION() \
    ); \
    return _resilient_future.get();

// Error handling with automatic logging and metrics
#define HANDLE_RESILIENT_ERROR(operation_name, error_handler) \
    try { \
        /* operation code */ \
    } catch (const std::exception& e) { \
        ultra::common::LogContext context; \
        context.add("operation", operation_name) \
               .add("error_type", typeid(e).name()); \
        LOG_ERROR_WITH_EXCEPTION("Resilient operation failed", e, context); \
        error_handler(e); \
    }

} // namespace common
} // namespace ultra