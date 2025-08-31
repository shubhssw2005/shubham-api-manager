#pragma once

#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace ultra {
namespace common {

/**
 * Graceful Degradation Manager
 * 
 * Monitors system performance and automatically degrades functionality
 * when performance issues are detected to maintain system stability.
 */
class DegradationManager {
public:
    enum class DegradationLevel {
        NORMAL = 0,      // Full functionality
        LIGHT = 1,       // Minor optimizations (reduce logging, etc.)
        MODERATE = 2,    // Disable non-essential features
        HEAVY = 3,       // Minimal functionality only
        EMERGENCY = 4    // Critical operations only
    };

    struct PerformanceMetrics {
        double cpu_usage_percent = 0.0;
        double memory_usage_percent = 0.0;
        uint64_t avg_response_time_ns = 0;
        uint64_t p99_response_time_ns = 0;
        double error_rate_percent = 0.0;
        uint64_t active_connections = 0;
        uint64_t queue_depth = 0;
    };

    struct DegradationConfig {
        // CPU usage thresholds for different degradation levels
        double cpu_light_threshold = 70.0;
        double cpu_moderate_threshold = 80.0;
        double cpu_heavy_threshold = 90.0;
        double cpu_emergency_threshold = 95.0;
        
        // Memory usage thresholds
        double memory_light_threshold = 75.0;
        double memory_moderate_threshold = 85.0;
        double memory_heavy_threshold = 92.0;
        double memory_emergency_threshold = 97.0;
        
        // Response time thresholds (nanoseconds)
        uint64_t response_time_light_threshold = 1000000;    // 1ms
        uint64_t response_time_moderate_threshold = 5000000; // 5ms
        uint64_t response_time_heavy_threshold = 10000000;   // 10ms
        uint64_t response_time_emergency_threshold = 50000000; // 50ms
        
        // Error rate thresholds (percentage)
        double error_rate_light_threshold = 1.0;
        double error_rate_moderate_threshold = 5.0;
        double error_rate_heavy_threshold = 10.0;
        double error_rate_emergency_threshold = 25.0;
        
        // Evaluation interval
        std::chrono::milliseconds evaluation_interval{1000};
        
        // Hysteresis factor to prevent oscillation
        double hysteresis_factor = 0.9;
    };

    using DegradationCallback = std::function<void(DegradationLevel old_level, DegradationLevel new_level)>;

    explicit DegradationManager(const DegradationConfig& config);
    ~DegradationManager();

    // Start/stop monitoring
    void start_monitoring();
    void stop_monitoring();
    
    // Update performance metrics
    void update_metrics(const PerformanceMetrics& metrics);
    
    // Get current degradation level
    DegradationLevel get_current_level() const { return current_level_.load(); }
    
    // Register callbacks for degradation level changes
    void register_callback(const std::string& name, DegradationCallback callback);
    void unregister_callback(const std::string& name);
    
    // Manual degradation control
    void force_degradation_level(DegradationLevel level);
    void reset_to_automatic();
    
    // Feature management
    bool is_feature_enabled(const std::string& feature_name) const;
    void register_feature(const std::string& feature_name, DegradationLevel min_level);
    
    // Statistics
    struct Stats {
        uint64_t evaluations_performed = 0;
        uint64_t level_changes = 0;
        uint64_t time_in_normal = 0;
        uint64_t time_in_light = 0;
        uint64_t time_in_moderate = 0;
        uint64_t time_in_heavy = 0;
        uint64_t time_in_emergency = 0;
        std::chrono::steady_clock::time_point last_level_change;
    };
    
    Stats get_stats() const { 
        Stats result;
        result.evaluations_performed = stats_.evaluations_performed.load();
        result.level_changes = stats_.level_changes.load();
        result.time_in_normal = stats_.time_in_normal.load();
        result.time_in_light = stats_.time_in_light.load();
        result.time_in_moderate = stats_.time_in_moderate.load();
        result.time_in_heavy = stats_.time_in_heavy.load();
        result.time_in_emergency = stats_.time_in_emergency.load();
        result.last_level_change = stats_.last_level_change;
        return result;
    }
    PerformanceMetrics get_current_metrics() const;

private:
    const DegradationConfig config_;
    
    std::atomic<DegradationLevel> current_level_{DegradationLevel::NORMAL};
    std::atomic<bool> monitoring_active_{false};
    std::atomic<bool> manual_override_{false};
    
    mutable std::mutex metrics_mutex_;
    PerformanceMetrics current_metrics_;
    
    mutable std::mutex callbacks_mutex_;
    std::unordered_map<std::string, DegradationCallback> callbacks_;
    
    mutable std::mutex features_mutex_;
    std::unordered_map<std::string, DegradationLevel> feature_requirements_;
    
    std::unique_ptr<std::thread> monitoring_thread_;
    
    struct AtomicStats {
        std::atomic<uint64_t> evaluations_performed{0};
        std::atomic<uint64_t> level_changes{0};
        std::atomic<uint64_t> time_in_normal{0};
        std::atomic<uint64_t> time_in_light{0};
        std::atomic<uint64_t> time_in_moderate{0};
        std::atomic<uint64_t> time_in_heavy{0};
        std::atomic<uint64_t> time_in_emergency{0};
        std::chrono::steady_clock::time_point last_level_change;
    };
    
    AtomicStats stats_;
    
    void monitoring_loop();
    DegradationLevel evaluate_degradation_level(const PerformanceMetrics& metrics) const;
    void change_degradation_level(DegradationLevel new_level);
    void notify_callbacks(DegradationLevel old_level, DegradationLevel new_level);
    void update_time_stats(DegradationLevel level, std::chrono::milliseconds duration);
    
public:
    static std::string level_to_string(DegradationLevel level);
};

/**
 * Degradation-aware feature guard
 * 
 * RAII class that checks if a feature should be executed based on
 * current degradation level.
 */
class FeatureGuard {
public:
    FeatureGuard(const std::string& feature_name, DegradationManager& manager);
    ~FeatureGuard() = default;
    
    // Check if feature should be executed
    bool should_execute() const { return should_execute_; }
    
    // Implicit conversion to bool for easy usage
    operator bool() const { return should_execute_; }

private:
    bool should_execute_;
    std::string feature_name_;
};

/**
 * Degradation Strategy Interface
 * 
 * Base class for implementing specific degradation strategies
 */
class DegradationStrategy {
public:
    virtual ~DegradationStrategy() = default;
    
    virtual void apply_degradation(DegradationManager::DegradationLevel level) = 0;
    virtual void restore_functionality(DegradationManager::DegradationLevel from_level) = 0;
    virtual std::string get_strategy_name() const = 0;
};

/**
 * Common degradation strategies
 */
class LoggingDegradationStrategy : public DegradationStrategy {
public:
    void apply_degradation(DegradationManager::DegradationLevel level) override;
    void restore_functionality(DegradationManager::DegradationLevel from_level) override;
    std::string get_strategy_name() const override { return "LoggingDegradation"; }
};

class CacheDegradationStrategy : public DegradationStrategy {
public:
    void apply_degradation(DegradationManager::DegradationLevel level) override;
    void restore_functionality(DegradationManager::DegradationLevel from_level) override;
    std::string get_strategy_name() const override { return "CacheDegradation"; }
};

class ConnectionPoolDegradationStrategy : public DegradationStrategy {
public:
    void apply_degradation(DegradationManager::DegradationLevel level) override;
    void restore_functionality(DegradationManager::DegradationLevel from_level) override;
    std::string get_strategy_name() const override { return "ConnectionPoolDegradation"; }
};

/**
 * Degradation Strategy Manager
 */
class DegradationStrategyManager {
public:
    static DegradationStrategyManager& instance();
    
    void register_strategy(std::unique_ptr<DegradationStrategy> strategy);
    void apply_all_strategies(DegradationManager::DegradationLevel level);
    void restore_all_strategies(DegradationManager::DegradationLevel from_level);
    
    std::vector<std::string> get_registered_strategies() const;

private:
    DegradationStrategyManager() = default;
    
    mutable std::mutex strategies_mutex_;
    std::vector<std::unique_ptr<DegradationStrategy>> strategies_;
};

// Convenience macros
#define DEGRADATION_GUARD(feature_name, manager) \
    ultra::common::FeatureGuard _guard(feature_name, manager); \
    if (!_guard) return;

#define DEGRADATION_GUARD_RETURN(feature_name, manager, return_value) \
    ultra::common::FeatureGuard _guard(feature_name, manager); \
    if (!_guard) return return_value;

} // namespace common
} // namespace ultra