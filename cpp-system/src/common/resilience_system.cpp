#include "common/resilience_system.hpp"
#include <sstream>
#include <iomanip>

namespace ultra {
namespace common {

ResilienceSystem::ResilienceSystem(const Config& config) : config_(config) {
    // Initialize logger first
    Logger::initialize("resilience", config_.logger_config);
    
    LOG_INFO("Initializing resilience system with integrated error handling");
}

ResilienceSystem::~ResilienceSystem() {
    shutdown();
}

void ResilienceSystem::initialize() {
    // Create component instances
    degradation_manager_ = std::make_unique<DegradationManager>(config_.degradation_config);
    fallback_manager_ = std::make_unique<FallbackManager>(config_.fallback_config);
    
    // Register default degradation strategies
    auto& strategy_manager = DegradationStrategyManager::instance();
    strategy_manager.register_strategy(std::make_unique<LoggingDegradationStrategy>());
    strategy_manager.register_strategy(std::make_unique<CacheDegradationStrategy>());
    strategy_manager.register_strategy(std::make_unique<ConnectionPoolDegradationStrategy>());
    
    // Setup integration callbacks
    setup_integration_callbacks();
    
    LOG_INFO("Resilience system components initialized");
}

void ResilienceSystem::start() {
    if (!degradation_manager_ || !fallback_manager_) {
        throw std::runtime_error("Resilience system not initialized. Call initialize() first.");
    }
    
    // Start component monitoring
    degradation_manager_->start_monitoring();
    fallback_manager_->start_health_monitoring();
    
    // Start metrics collection
    metrics_collection_active_.store(true);
    metrics_thread_ = std::make_unique<std::thread>(&ResilienceSystem::metrics_collection_loop, this);
    
    LOG_INFO("Resilience system started");
}

void ResilienceSystem::stop() {
    // Stop metrics collection
    if (metrics_collection_active_.exchange(false)) {
        if (metrics_thread_ && metrics_thread_->joinable()) {
            metrics_thread_->join();
        }
    }
    
    // Stop component monitoring
    if (degradation_manager_) {
        degradation_manager_->stop_monitoring();
    }
    
    if (fallback_manager_) {
        fallback_manager_->stop_health_monitoring();
    }
    
    LOG_INFO("Resilience system stopped");
}

void ResilienceSystem::shutdown() {
    stop();
    
    // Shutdown logger
    Logger::shutdown();
    
    LOG_INFO("Resilience system shutdown complete");
}

std::shared_ptr<CircuitBreaker> ResilienceSystem::get_circuit_breaker(const std::string& name) {
    return CircuitBreakerManager::instance().get_or_create(name, config_.circuit_breaker_config);
}

void ResilienceSystem::register_operation(const std::string& operation_name, 
                                        const CircuitBreaker::Config& cb_config,
                                        DegradationManager::DegradationLevel min_degradation_level) {
    
    // Create circuit breaker for operation
    CircuitBreakerManager::instance().get_or_create(operation_name, cb_config);
    
    // Register feature with degradation manager
    degradation_manager_->register_feature(operation_name, min_degradation_level);
    
    LOG_INFO("Registered operation '{}' with resilience system", operation_name);
}

ResilienceSystem::SystemStats ResilienceSystem::get_system_stats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return cached_stats_;
}

bool ResilienceSystem::is_system_healthy() const {
    auto stats = get_system_stats();
    
    // System is healthy if:
    // 1. No circuit breakers are open
    // 2. Degradation level is not emergency
    // 3. Node.js fallback is available
    // 4. Overall success rate is above threshold
    
    bool circuit_breakers_healthy = stats.circuit_breaker_stats.open_circuits == 0;
    bool degradation_healthy = degradation_manager_->get_current_level() != DegradationManager::DegradationLevel::EMERGENCY;
    bool fallback_healthy = fallback_manager_->is_nodejs_healthy();
    bool success_rate_healthy = stats.overall_success_rate >= 0.95; // 95% success rate
    
    return circuit_breakers_healthy && degradation_healthy && fallback_healthy && success_rate_healthy;
}

std::string ResilienceSystem::export_prometheus_metrics() const {
    auto stats = get_system_stats();
    std::ostringstream oss;
    
    // Circuit breaker metrics
    oss << "# HELP ultra_circuit_breaker_total Total number of circuit breakers\n";
    oss << "# TYPE ultra_circuit_breaker_total gauge\n";
    oss << "ultra_circuit_breaker_total " << stats.circuit_breaker_stats.total_circuits << "\n";
    
    oss << "# HELP ultra_circuit_breaker_open Number of open circuit breakers\n";
    oss << "# TYPE ultra_circuit_breaker_open gauge\n";
    oss << "ultra_circuit_breaker_open " << stats.circuit_breaker_stats.open_circuits << "\n";
    
    oss << "# HELP ultra_circuit_breaker_requests_total Total circuit breaker requests\n";
    oss << "# TYPE ultra_circuit_breaker_requests_total counter\n";
    oss << "ultra_circuit_breaker_requests_total " << stats.circuit_breaker_stats.total_requests << "\n";
    
    // Degradation metrics
    oss << "# HELP ultra_degradation_level Current system degradation level\n";
    oss << "# TYPE ultra_degradation_level gauge\n";
    oss << "ultra_degradation_level " << static_cast<int>(degradation_manager_->get_current_level()) << "\n";
    
    oss << "# HELP ultra_degradation_level_changes_total Total degradation level changes\n";
    oss << "# TYPE ultra_degradation_level_changes_total counter\n";
    oss << "ultra_degradation_level_changes_total " << stats.degradation_stats.level_changes.load() << "\n";
    
    // Fallback metrics
    oss << "# HELP ultra_fallback_requests_total Total fallback requests\n";
    oss << "# TYPE ultra_fallback_requests_total counter\n";
    oss << "ultra_fallback_requests_total " << stats.fallback_stats.total_fallbacks.load() << "\n";
    
    oss << "# HELP ultra_nodejs_available Node.js service availability\n";
    oss << "# TYPE ultra_nodejs_available gauge\n";
    oss << "ultra_nodejs_available " << (stats.fallback_stats.nodejs_available.load() ? 1 : 0) << "\n";
    
    // Overall system metrics
    oss << "# HELP ultra_operations_total Total operations executed\n";
    oss << "# TYPE ultra_operations_total counter\n";
    oss << "ultra_operations_total " << stats.total_operations << "\n";
    
    oss << "# HELP ultra_success_rate Overall system success rate\n";
    oss << "# TYPE ultra_success_rate gauge\n";
    oss << "ultra_success_rate " << std::fixed << std::setprecision(4) << stats.overall_success_rate << "\n";
    
    oss << "# HELP ultra_system_healthy System health status\n";
    oss << "# TYPE ultra_system_healthy gauge\n";
    oss << "ultra_system_healthy " << (is_system_healthy() ? 1 : 0) << "\n";
    
    return oss.str();
}

void ResilienceSystem::setup_integration_callbacks() {
    // Register degradation change callback
    degradation_manager_->register_callback("resilience_system", 
        [this](DegradationManager::DegradationLevel old_level, DegradationManager::DegradationLevel new_level) {
            on_degradation_change(old_level, new_level);
        });
    
    // Register Node.js health change callback
    fallback_manager_->register_health_check_callback(
        [this](bool is_healthy) {
            on_nodejs_health_change(is_healthy);
        });
    
    LOG_DEBUG("Integration callbacks setup complete");
}

void ResilienceSystem::metrics_collection_loop() {
    while (metrics_collection_active_.load()) {
        update_system_metrics();
        std::this_thread::sleep_for(config_.metrics_collection_interval);
    }
}

void ResilienceSystem::update_system_metrics() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    // Collect stats from all components
    cached_stats_.circuit_breaker_stats = CircuitBreakerManager::instance().get_aggregated_stats();
    cached_stats_.degradation_stats = degradation_manager_->get_stats();
    cached_stats_.fallback_stats = fallback_manager_->get_stats();
    
    // Calculate overall metrics
    cached_stats_.total_operations = cached_stats_.circuit_breaker_stats.total_requests;
    cached_stats_.failed_operations = cached_stats_.circuit_breaker_stats.total_failures;
    cached_stats_.successful_operations = cached_stats_.total_operations - cached_stats_.failed_operations;
    
    if (cached_stats_.total_operations > 0) {
        cached_stats_.overall_success_rate = 
            static_cast<double>(cached_stats_.successful_operations) / cached_stats_.total_operations;
    } else {
        cached_stats_.overall_success_rate = 1.0;
    }
    
    // Check if we should trigger degradation based on error rate
    if (config_.enable_degradation_on_high_error_rate && 
        cached_stats_.total_operations > 100 && // Minimum sample size
        (1.0 - cached_stats_.overall_success_rate) > config_.error_rate_degradation_threshold) {
        
        LOG_WARNING("High error rate detected ({}%), considering degradation", 
                   (1.0 - cached_stats_.overall_success_rate) * 100);
        
        // Update degradation manager with current performance metrics
        DegradationManager::PerformanceMetrics perf_metrics;
        perf_metrics.error_rate_percent = (1.0 - cached_stats_.overall_success_rate) * 100;
        degradation_manager_->update_metrics(perf_metrics);
    }
}

void ResilienceSystem::on_degradation_change(DegradationManager::DegradationLevel old_level, 
                                           DegradationManager::DegradationLevel new_level) {
    
    LogContext context;
    context.add("old_level", DegradationManager::level_to_string(old_level))
           .add("new_level", DegradationManager::level_to_string(new_level))
           .add("event_type", "degradation_change");
    
    LOG_STRUCTURED(LogLevel::WARNING, "System degradation level changed", context);
    
    // If degradation is severe, consider forcing circuit breakers open
    if (new_level >= DegradationManager::DegradationLevel::HEAVY) {
        LOG_WARNING("Severe degradation detected, system may need manual intervention");
    }
}

void ResilienceSystem::on_nodejs_health_change(bool is_healthy) {
    LogContext context;
    context.add("nodejs_healthy", is_healthy ? "true" : "false")
           .add("event_type", "nodejs_health_change");
    
    if (is_healthy) {
        LOG_STRUCTURED(LogLevel::INFO, "Node.js service recovered", context);
    } else {
        LOG_STRUCTURED(LogLevel::ERROR, "Node.js service unavailable", context);
    }
}

// Global Resilience System Implementation
std::unique_ptr<ResilienceSystem> GlobalResilienceSystem::instance_;
std::mutex GlobalResilienceSystem::instance_mutex_;

void GlobalResilienceSystem::initialize(const ResilienceSystem::Config& config) {
    std::lock_guard<std::mutex> lock(instance_mutex_);
    
    if (instance_) {
        throw std::runtime_error("Global resilience system already initialized");
    }
    
    instance_ = std::make_unique<ResilienceSystem>(config);
    instance_->initialize();
    instance_->start();
}

ResilienceSystem& GlobalResilienceSystem::instance() {
    std::lock_guard<std::mutex> lock(instance_mutex_);
    
    if (!instance_) {
        throw std::runtime_error("Global resilience system not initialized. Call initialize() first.");
    }
    
    return *instance_;
}

void GlobalResilienceSystem::shutdown() {
    std::lock_guard<std::mutex> lock(instance_mutex_);
    
    if (instance_) {
        instance_->shutdown();
        instance_.reset();
    }
}

} // namespace common
} // namespace ultra