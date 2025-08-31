#include "common/degradation_manager.hpp"
#include "common/logger.hpp"
#include <algorithm>
#include <thread>

namespace ultra {
namespace common {

DegradationManager::DegradationManager(const DegradationConfig& config)
    : config_(config) {
    
    LOG_INFO("Degradation manager initialized with CPU thresholds: {}/{}/{}/{}%", 
             config_.cpu_light_threshold, config_.cpu_moderate_threshold,
             config_.cpu_heavy_threshold, config_.cpu_emergency_threshold);
}

DegradationManager::~DegradationManager() {
    stop_monitoring();
}

void DegradationManager::start_monitoring() {
    if (monitoring_active_.exchange(true)) {
        LOG_WARNING("Degradation monitoring already active");
        return;
    }
    
    monitoring_thread_ = std::make_unique<std::thread>(&DegradationManager::monitoring_loop, this);
    LOG_INFO("Degradation monitoring started");
}

void DegradationManager::stop_monitoring() {
    if (!monitoring_active_.exchange(false)) {
        return;
    }
    
    if (monitoring_thread_ && monitoring_thread_->joinable()) {
        monitoring_thread_->join();
    }
    
    LOG_INFO("Degradation monitoring stopped");
}

void DegradationManager::update_metrics(const PerformanceMetrics& metrics) {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    current_metrics_ = metrics;
}

void DegradationManager::register_callback(const std::string& name, DegradationCallback callback) {
    std::lock_guard<std::mutex> lock(callbacks_mutex_);
    callbacks_[name] = std::move(callback);
    LOG_DEBUG("Registered degradation callback: {}", name);
}

void DegradationManager::unregister_callback(const std::string& name) {
    std::lock_guard<std::mutex> lock(callbacks_mutex_);
    callbacks_.erase(name);
    LOG_DEBUG("Unregistered degradation callback: {}", name);
}

void DegradationManager::force_degradation_level(DegradationLevel level) {
    manual_override_.store(true);
    change_degradation_level(level);
    LOG_WARNING("Degradation level manually set to: {}", level_to_string(level));
}

void DegradationManager::reset_to_automatic() {
    manual_override_.store(false);
    LOG_INFO("Degradation manager reset to automatic mode");
}

bool DegradationManager::is_feature_enabled(const std::string& feature_name) const {
    std::lock_guard<std::mutex> lock(features_mutex_);
    
    auto it = feature_requirements_.find(feature_name);
    if (it == feature_requirements_.end()) {
        return true; // Feature not registered, assume enabled
    }
    
    return current_level_.load() <= it->second;
}

void DegradationManager::register_feature(const std::string& feature_name, DegradationLevel min_level) {
    std::lock_guard<std::mutex> lock(features_mutex_);
    feature_requirements_[feature_name] = min_level;
    LOG_DEBUG("Registered feature '{}' with minimum level: {}", feature_name, level_to_string(min_level));
}

DegradationManager::PerformanceMetrics DegradationManager::get_current_metrics() const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    return current_metrics_;
}

void DegradationManager::monitoring_loop() {
    auto last_evaluation = std::chrono::steady_clock::now();
    
    while (monitoring_active_.load()) {
        std::this_thread::sleep_for(config_.evaluation_interval);
        
        if (manual_override_.load()) {
            continue;
        }
        
        PerformanceMetrics metrics;
        {
            std::lock_guard<std::mutex> lock(metrics_mutex_);
            metrics = current_metrics_;
        }
        
        DegradationLevel new_level = evaluate_degradation_level(metrics);
        DegradationLevel current = current_level_.load();
        
        if (new_level != current) {
            change_degradation_level(new_level);
        }
        
        // Update time statistics
        auto now = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_evaluation);
        update_time_stats(current, duration);
        last_evaluation = now;
        
        stats_.evaluations_performed.fetch_add(1);
    }
}

DegradationManager::DegradationLevel DegradationManager::evaluate_degradation_level(
    const PerformanceMetrics& metrics) const {
    
    DegradationLevel cpu_level = DegradationLevel::NORMAL;
    DegradationLevel memory_level = DegradationLevel::NORMAL;
    DegradationLevel response_level = DegradationLevel::NORMAL;
    DegradationLevel error_level = DegradationLevel::NORMAL;
    
    // Evaluate CPU usage
    if (metrics.cpu_usage_percent >= config_.cpu_emergency_threshold) {
        cpu_level = DegradationLevel::EMERGENCY;
    } else if (metrics.cpu_usage_percent >= config_.cpu_heavy_threshold) {
        cpu_level = DegradationLevel::HEAVY;
    } else if (metrics.cpu_usage_percent >= config_.cpu_moderate_threshold) {
        cpu_level = DegradationLevel::MODERATE;
    } else if (metrics.cpu_usage_percent >= config_.cpu_light_threshold) {
        cpu_level = DegradationLevel::LIGHT;
    }
    
    // Evaluate memory usage
    if (metrics.memory_usage_percent >= config_.memory_emergency_threshold) {
        memory_level = DegradationLevel::EMERGENCY;
    } else if (metrics.memory_usage_percent >= config_.memory_heavy_threshold) {
        memory_level = DegradationLevel::HEAVY;
    } else if (metrics.memory_usage_percent >= config_.memory_moderate_threshold) {
        memory_level = DegradationLevel::MODERATE;
    } else if (metrics.memory_usage_percent >= config_.memory_light_threshold) {
        memory_level = DegradationLevel::LIGHT;
    }
    
    // Evaluate response time
    if (metrics.p99_response_time_ns >= config_.response_time_emergency_threshold) {
        response_level = DegradationLevel::EMERGENCY;
    } else if (metrics.p99_response_time_ns >= config_.response_time_heavy_threshold) {
        response_level = DegradationLevel::HEAVY;
    } else if (metrics.p99_response_time_ns >= config_.response_time_moderate_threshold) {
        response_level = DegradationLevel::MODERATE;
    } else if (metrics.p99_response_time_ns >= config_.response_time_light_threshold) {
        response_level = DegradationLevel::LIGHT;
    }
    
    // Evaluate error rate
    if (metrics.error_rate_percent >= config_.error_rate_emergency_threshold) {
        error_level = DegradationLevel::EMERGENCY;
    } else if (metrics.error_rate_percent >= config_.error_rate_heavy_threshold) {
        error_level = DegradationLevel::HEAVY;
    } else if (metrics.error_rate_percent >= config_.error_rate_moderate_threshold) {
        error_level = DegradationLevel::MODERATE;
    } else if (metrics.error_rate_percent >= config_.error_rate_light_threshold) {
        error_level = DegradationLevel::LIGHT;
    }
    
    // Take the maximum (most severe) degradation level
    DegradationLevel max_level = std::max({cpu_level, memory_level, response_level, error_level});
    
    // Apply hysteresis to prevent oscillation
    DegradationLevel current = current_level_.load();
    if (max_level < current) {
        // Only improve if metrics are significantly better
        double improvement_factor = config_.hysteresis_factor;
        
        bool should_improve = false;
        switch (current) {
            case DegradationLevel::EMERGENCY:
                should_improve = (metrics.cpu_usage_percent < config_.cpu_heavy_threshold * improvement_factor) &&
                               (metrics.memory_usage_percent < config_.memory_heavy_threshold * improvement_factor);
                break;
            case DegradationLevel::HEAVY:
                should_improve = (metrics.cpu_usage_percent < config_.cpu_moderate_threshold * improvement_factor) &&
                               (metrics.memory_usage_percent < config_.memory_moderate_threshold * improvement_factor);
                break;
            case DegradationLevel::MODERATE:
                should_improve = (metrics.cpu_usage_percent < config_.cpu_light_threshold * improvement_factor) &&
                               (metrics.memory_usage_percent < config_.memory_light_threshold * improvement_factor);
                break;
            case DegradationLevel::LIGHT:
                should_improve = (metrics.cpu_usage_percent < config_.cpu_light_threshold * improvement_factor) &&
                               (metrics.memory_usage_percent < config_.memory_light_threshold * improvement_factor);
                break;
            default:
                should_improve = true;
                break;
        }
        
        if (!should_improve) {
            max_level = current;
        }
    }
    
    return max_level;
}

void DegradationManager::change_degradation_level(DegradationLevel new_level) {
    DegradationLevel old_level = current_level_.exchange(new_level);
    
    if (old_level != new_level) {
        stats_.level_changes.fetch_add(1);
        stats_.last_level_change = std::chrono::steady_clock::now();
        
        LOG_WARNING("Degradation level changed: {} -> {}", 
                   level_to_string(old_level), level_to_string(new_level));
        
        notify_callbacks(old_level, new_level);
        
        // Apply degradation strategies
        if (new_level > old_level) {
            DegradationStrategyManager::instance().apply_all_strategies(new_level);
        } else {
            DegradationStrategyManager::instance().restore_all_strategies(old_level);
        }
    }
}

void DegradationManager::notify_callbacks(DegradationLevel old_level, DegradationLevel new_level) {
    std::lock_guard<std::mutex> lock(callbacks_mutex_);
    
    for (const auto& pair : callbacks_) {
        try {
            pair.second(old_level, new_level);
        } catch (const std::exception& e) {
            LOG_ERROR("Exception in degradation callback '{}': {}", pair.first, e.what());
        }
    }
}

void DegradationManager::update_time_stats(DegradationLevel level, std::chrono::milliseconds duration) {
    uint64_t duration_ms = duration.count();
    
    switch (level) {
        case DegradationLevel::NORMAL:
            stats_.time_in_normal.fetch_add(duration_ms);
            break;
        case DegradationLevel::LIGHT:
            stats_.time_in_light.fetch_add(duration_ms);
            break;
        case DegradationLevel::MODERATE:
            stats_.time_in_moderate.fetch_add(duration_ms);
            break;
        case DegradationLevel::HEAVY:
            stats_.time_in_heavy.fetch_add(duration_ms);
            break;
        case DegradationLevel::EMERGENCY:
            stats_.time_in_emergency.fetch_add(duration_ms);
            break;
    }
}

std::string DegradationManager::level_to_string(DegradationLevel level) {
    switch (level) {
        case DegradationLevel::NORMAL: return "NORMAL";
        case DegradationLevel::LIGHT: return "LIGHT";
        case DegradationLevel::MODERATE: return "MODERATE";
        case DegradationLevel::HEAVY: return "HEAVY";
        case DegradationLevel::EMERGENCY: return "EMERGENCY";
        default: return "UNKNOWN";
    }
}

// FeatureGuard Implementation
FeatureGuard::FeatureGuard(const std::string& feature_name, DegradationManager& manager)
    : feature_name_(feature_name) {
    should_execute_ = manager.is_feature_enabled(feature_name);
    
    if (!should_execute_) {
        LOG_DEBUG("Feature '{}' disabled due to degradation level: {}", 
                 feature_name, DegradationManager::level_to_string(manager.get_current_level()));
    }
}

// Degradation Strategies Implementation
void LoggingDegradationStrategy::apply_degradation(DegradationManager::DegradationLevel level) {
    switch (level) {
        case DegradationManager::DegradationLevel::LIGHT:
            Logger::set_level(LogLevel::INFO);
            break;
        case DegradationManager::DegradationLevel::MODERATE:
            Logger::set_level(LogLevel::WARNING);
            break;
        case DegradationManager::DegradationLevel::HEAVY:
        case DegradationManager::DegradationLevel::EMERGENCY:
            Logger::set_level(LogLevel::ERROR);
            break;
        default:
            break;
    }
    LOG_INFO("Applied logging degradation for level: {}", DegradationManager::level_to_string(level));
}

void LoggingDegradationStrategy::restore_functionality(DegradationManager::DegradationLevel from_level) {
    Logger::set_level(LogLevel::INFO);
    LOG_INFO("Restored logging functionality from level: {}", DegradationManager::level_to_string(from_level));
}

void CacheDegradationStrategy::apply_degradation(DegradationManager::DegradationLevel level) {
    // Implementation would interact with cache system
    // For now, just log the action
    LOG_INFO("Applied cache degradation for level: {}", DegradationManager::level_to_string(level));
}

void CacheDegradationStrategy::restore_functionality(DegradationManager::DegradationLevel from_level) {
    LOG_INFO("Restored cache functionality from level: {}", DegradationManager::level_to_string(from_level));
}

void ConnectionPoolDegradationStrategy::apply_degradation(DegradationManager::DegradationLevel level) {
    // Implementation would interact with connection pool
    LOG_INFO("Applied connection pool degradation for level: {}", DegradationManager::level_to_string(level));
}

void ConnectionPoolDegradationStrategy::restore_functionality(DegradationManager::DegradationLevel from_level) {
    LOG_INFO("Restored connection pool functionality from level: {}", DegradationManager::level_to_string(from_level));
}

// DegradationStrategyManager Implementation
DegradationStrategyManager& DegradationStrategyManager::instance() {
    static DegradationStrategyManager instance;
    return instance;
}

void DegradationStrategyManager::register_strategy(std::unique_ptr<DegradationStrategy> strategy) {
    std::lock_guard<std::mutex> lock(strategies_mutex_);
    LOG_INFO("Registered degradation strategy: {}", strategy->get_strategy_name());
    strategies_.push_back(std::move(strategy));
}

void DegradationStrategyManager::apply_all_strategies(DegradationManager::DegradationLevel level) {
    std::lock_guard<std::mutex> lock(strategies_mutex_);
    
    for (auto& strategy : strategies_) {
        try {
            strategy->apply_degradation(level);
        } catch (const std::exception& e) {
            LOG_ERROR("Exception applying degradation strategy '{}': {}", 
                     strategy->get_strategy_name(), e.what());
        }
    }
}

void DegradationStrategyManager::restore_all_strategies(DegradationManager::DegradationLevel from_level) {
    std::lock_guard<std::mutex> lock(strategies_mutex_);
    
    for (auto& strategy : strategies_) {
        try {
            strategy->restore_functionality(from_level);
        } catch (const std::exception& e) {
            LOG_ERROR("Exception restoring degradation strategy '{}': {}", 
                     strategy->get_strategy_name(), e.what());
        }
    }
}

std::vector<std::string> DegradationStrategyManager::get_registered_strategies() const {
    std::lock_guard<std::mutex> lock(strategies_mutex_);
    
    std::vector<std::string> names;
    names.reserve(strategies_.size());
    
    for (const auto& strategy : strategies_) {
        names.push_back(strategy->get_strategy_name());
    }
    
    return names;
}

} // namespace common
} // namespace ultra