#include "common/resilience_system.hpp"
#include "common/circuit_breaker.hpp"
#include "common/degradation_manager.hpp"
#include "common/fallback_manager.hpp"
#include "common/logger.hpp"
#include <iostream>
#include <thread>
#include <chrono>
#include <random>

using namespace ultra::common;

// Simulated service that can fail
class SimulatedService {
public:
    SimulatedService() : failure_rate_(0.1), response_time_ms_(10) {}
    
    void set_failure_rate(double rate) { failure_rate_ = rate; }
    void set_response_time(int ms) { response_time_ms_ = ms; }
    
    std::string process_request(const std::string& request) {
        // Simulate processing time
        std::this_thread::sleep_for(std::chrono::milliseconds(response_time_ms_));
        
        // Simulate random failures
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_real_distribution<> dis(0.0, 1.0);
        
        if (dis(gen) < failure_rate_) {
            throw std::runtime_error("Service temporarily unavailable");
        }
        
        return "Processed: " + request;
    }

private:
    double failure_rate_;
    int response_time_ms_;
};

// Example of using circuit breaker directly
void demonstrate_circuit_breaker() {
    std::cout << "\n=== Circuit Breaker Demo ===\n";
    
    CircuitBreaker::Config config;
    config.failure_threshold = 0.5;
    config.minimum_requests = 5;
    config.initial_timeout_ms = 1000;
    
    CircuitBreaker breaker("demo_service", config);
    SimulatedService service;
    
    // Start with high failure rate
    service.set_failure_rate(0.8);
    
    for (int i = 0; i < 20; ++i) {
        try {
            std::string result = breaker.execute([&]() {
                return service.process_request("request_" + std::to_string(i));
            });
            
            std::cout << "Success: " << result << std::endl;
            
        } catch (const CircuitBreakerOpenException& e) {
            std::cout << "Circuit breaker open: " << e.what() << std::endl;
            
        } catch (const std::exception& e) {
            std::cout << "Service error: " << e.what() << std::endl;
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    auto stats = breaker.get_stats();
    std::cout << "Circuit breaker stats:\n";
    std::cout << "  Total requests: " << stats.total_requests.load() << "\n";
    std::cout << "  Successful: " << stats.successful_requests.load() << "\n";
    std::cout << "  Failed: " << stats.failed_requests.load() << "\n";
    std::cout << "  Rejected: " << stats.rejected_requests.load() << "\n";
    std::cout << "  State: " << static_cast<int>(breaker.get_state()) << "\n";
}

// Example of using degradation manager
void demonstrate_degradation_manager() {
    std::cout << "\n=== Degradation Manager Demo ===\n";
    
    DegradationManager::DegradationConfig config;
    config.cpu_light_threshold = 30.0;
    config.cpu_moderate_threshold = 50.0;
    config.cpu_heavy_threshold = 70.0;
    config.evaluation_interval = std::chrono::milliseconds(500);
    
    DegradationManager manager(config);
    
    // Register features with different degradation levels
    manager.register_feature("analytics", DegradationManager::DegradationLevel::LIGHT);
    manager.register_feature("caching", DegradationManager::DegradationLevel::MODERATE);
    manager.register_feature("core_api", DegradationManager::DegradationLevel::HEAVY);
    
    manager.start_monitoring();
    
    // Simulate increasing system load
    for (int load = 20; load <= 80; load += 20) {
        DegradationManager::PerformanceMetrics metrics;
        metrics.cpu_usage_percent = load;
        metrics.memory_usage_percent = load * 0.8;
        metrics.error_rate_percent = load * 0.1;
        
        manager.update_metrics(metrics);
        
        std::this_thread::sleep_for(std::chrono::milliseconds(600));
        
        std::cout << "CPU Load: " << load << "%, Degradation Level: " 
                  << static_cast<int>(manager.get_current_level()) << std::endl;
        
        std::cout << "  Analytics enabled: " << manager.is_feature_enabled("analytics") << std::endl;
        std::cout << "  Caching enabled: " << manager.is_feature_enabled("caching") << std::endl;
        std::cout << "  Core API enabled: " << manager.is_feature_enabled("core_api") << std::endl;
    }
    
    manager.stop_monitoring();
}

// Example of using fallback manager
void demonstrate_fallback_manager() {
    std::cout << "\n=== Fallback Manager Demo ===\n";
    
    FallbackManager::FallbackConfig config;
    config.nodejs_base_url = "http://localhost:3005";
    config.enable_automatic_fallback = true;
    
    FallbackManager manager(config);
    
    // Register custom fallback handlers
    manager.register_fallback_handler("/api/users", [](const std::string& data) {
        return R"({"users": [{"id": 1, "name": "fallback_user"}], "source": "fallback"})";
    });
    
    manager.register_fallback_handler("/api/health", [](const std::string& data) {
        return R"({"status": "ok", "source": "fallback"})";
    });
    
    // Test fallback execution
    try {
        std::string result = manager.execute_nodejs_fallback("/api/users", "");
        std::cout << "Fallback result: " << result << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "Fallback failed: " << e.what() << std::endl;
    }
    
    auto stats = manager.get_stats();
    std::cout << "Fallback stats:\n";
    std::cout << "  Total fallbacks: " << stats.total_fallbacks.load() << "\n";
    std::cout << "  Successful: " << stats.successful_fallbacks.load() << "\n";
    std::cout << "  Node.js available: " << stats.nodejs_available.load() << "\n";
}

// Example of structured logging
void demonstrate_structured_logging() {
    std::cout << "\n=== Structured Logging Demo ===\n";
    
    // Configure logger for JSON output
    Logger::Config config;
    config.format = Logger::OutputFormat::JSON;
    config.level = LogLevel::DEBUG;
    config.enable_console_output = true;
    
    Logger::initialize("demo", config);
    
    // Basic structured logging
    LogContext context;
    context.add("user_id", "12345")
           .add("session_id", "abcdef")
           .add("operation", "user_login");
    
    LOG_STRUCTURED(LogLevel::INFO, "User login attempt", context);
    
    // Performance logging
    {
        PERFORMANCE_TIMER_WITH_CONTEXT("database_query", context);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    
    // Error logging with exception
    try {
        throw std::runtime_error("Database connection failed");
    } catch (const std::exception& e) {
        LOG_ERROR_WITH_EXCEPTION("Database operation failed", e, context);
    }
    
    // Security event logging
    LogContext security_context;
    security_context.add("ip_address", "192.168.1.100")
                   .add("user_agent", "Mozilla/5.0")
                   .add("attempted_resource", "/admin/users");
    
    LOG_SECURITY("unauthorized_access_attempt", "User attempted to access admin panel", security_context);
    
    // Audit logging
    LOG_AUDIT("user_created", "admin", "user:12345", context);
    
    Logger::shutdown();
}

// Example of integrated resilience system
void demonstrate_resilience_system() {
    std::cout << "\n=== Integrated Resilience System Demo ===\n";
    
    ResilienceSystem::Config config;
    config.circuit_breaker_config.failure_threshold = 0.4;
    config.circuit_breaker_config.minimum_requests = 3;
    config.degradation_config.cpu_light_threshold = 40.0;
    config.fallback_config.enable_automatic_fallback = true;
    config.logger_config.format = Logger::OutputFormat::PLAIN;
    config.logger_config.level = LogLevel::INFO;
    
    ResilienceSystem system(config);
    system.initialize();
    system.start();
    
    // Register operations
    system.register_operation("user_service", config.circuit_breaker_config, 
                             DegradationManager::DegradationLevel::LIGHT);
    system.register_operation("payment_service", config.circuit_breaker_config,
                             DegradationManager::DegradationLevel::NORMAL);
    
    SimulatedService service;
    
    // Test resilient execution with various scenarios
    std::cout << "Testing resilient execution...\n";
    
    // Scenario 1: Normal operation
    service.set_failure_rate(0.1);
    for (int i = 0; i < 5; ++i) {
        try {
            auto future = system.execute_resilient("user_service", [&]() {
                return service.process_request("user_request_" + std::to_string(i));
            });
            
            std::string result = future.get();
            std::cout << "Success: " << result << std::endl;
            
        } catch (const std::exception& e) {
            std::cout << "Error: " << e.what() << std::endl;
        }
    }
    
    // Scenario 2: High failure rate
    std::cout << "\nSimulating high failure rate...\n";
    service.set_failure_rate(0.8);
    
    for (int i = 0; i < 10; ++i) {
        try {
            auto future = system.execute_resilient("payment_service", [&]() {
                return service.process_request("payment_request_" + std::to_string(i));
            });
            
            std::string result = future.get();
            std::cout << "Success: " << result << std::endl;
            
        } catch (const std::exception& e) {
            std::cout << "Error: " << e.what() << std::endl;
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    // Display system statistics
    auto stats = system.get_system_stats();
    std::cout << "\nSystem Statistics:\n";
    std::cout << "  Total operations: " << stats.total_operations << "\n";
    std::cout << "  Success rate: " << (stats.overall_success_rate * 100) << "%\n";
    std::cout << "  Circuit breakers open: " << stats.circuit_breaker_stats.open_circuits << "\n";
    std::cout << "  System healthy: " << (system.is_system_healthy() ? "Yes" : "No") << "\n";
    
    // Export Prometheus metrics
    std::cout << "\nPrometheus Metrics Sample:\n";
    std::string metrics = system.export_prometheus_metrics();
    std::cout << metrics.substr(0, 500) << "...\n";
    
    system.stop();
}

int main() {
    std::cout << "Ultra Low-Latency C++ System - Error Handling and Resilience Demo\n";
    std::cout << "================================================================\n";
    
    try {
        demonstrate_circuit_breaker();
        demonstrate_degradation_manager();
        demonstrate_fallback_manager();
        demonstrate_structured_logging();
        demonstrate_resilience_system();
        
    } catch (const std::exception& e) {
        std::cerr << "Demo failed: " << e.what() << std::endl;
        return 1;
    }
    
    std::cout << "\nDemo completed successfully!\n";
    return 0;
}