#include <gtest/gtest.h>
#include "common/circuit_breaker.hpp"
#include "common/degradation_manager.hpp"
#include "common/fallback_manager.hpp"
#include "common/resilience_system.hpp"
#include "common/logger.hpp"
#include <thread>
#include <chrono>

using namespace ultra::common;

class ErrorHandlingResilienceTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize logger for tests
        Logger::Config log_config;
        log_config.level = LogLevel::DEBUG;
        log_config.format = Logger::OutputFormat::PLAIN;
        log_config.enable_console_output = false; // Disable for tests
        Logger::initialize("test", log_config);
    }
    
    void TearDown() override {
        Logger::shutdown();
    }
};

// Circuit Breaker Tests
TEST_F(ErrorHandlingResilienceTest, CircuitBreakerBasicFunctionality) {
    CircuitBreaker::Config config;
    config.failure_threshold = 0.5;
    config.minimum_requests = 3;
    config.initial_timeout_ms = 100;
    
    CircuitBreaker breaker("test_service", config);
    
    // Initially closed
    EXPECT_EQ(breaker.get_state(), CircuitBreaker::State::CLOSED);
    EXPECT_TRUE(breaker.allow_request());
    
    // Record some failures
    breaker.record_failure();
    breaker.record_failure();
    breaker.record_success();
    breaker.record_failure();
    
    // Should open after exceeding failure threshold
    EXPECT_EQ(breaker.get_state(), CircuitBreaker::State::OPEN);
    EXPECT_FALSE(breaker.allow_request());
}

TEST_F(ErrorHandlingResilienceTest, CircuitBreakerExecution) {
    CircuitBreaker::Config config;
    config.failure_threshold = 0.5;
    config.minimum_requests = 2;
    
    CircuitBreaker breaker("test_execution", config);
    
    // Successful execution
    int result = breaker.execute([]() { return 42; });
    EXPECT_EQ(result, 42);
    
    // Failing execution
    int failure_count = 0;
    for (int i = 0; i < 5; ++i) {
        try {
            breaker.execute([]() -> int { 
                throw std::runtime_error("Test failure"); 
            });
        } catch (const std::runtime_error&) {
            failure_count++;
        } catch (const CircuitBreakerOpenException&) {
            // Circuit breaker opened
            break;
        }
    }
    
    EXPECT_GT(failure_count, 0);
    EXPECT_EQ(breaker.get_state(), CircuitBreaker::State::OPEN);
}

// Degradation Manager Tests
TEST_F(ErrorHandlingResilienceTest, DegradationManagerBasicFunctionality) {
    DegradationManager::DegradationConfig config;
    config.cpu_light_threshold = 50.0;
    config.cpu_moderate_threshold = 70.0;
    config.evaluation_interval = std::chrono::milliseconds(50);
    
    DegradationManager manager(config);
    
    // Initially normal
    EXPECT_EQ(manager.get_current_level(), DegradationManager::DegradationLevel::NORMAL);
    
    // Register a feature
    manager.register_feature("test_feature", DegradationManager::DegradationLevel::LIGHT);
    EXPECT_TRUE(manager.is_feature_enabled("test_feature"));
    
    // Update metrics to trigger degradation
    DegradationManager::PerformanceMetrics metrics;
    metrics.cpu_usage_percent = 80.0; // Above moderate threshold
    
    manager.start_monitoring();
    manager.update_metrics(metrics);
    
    // Wait for evaluation
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    // Should be degraded
    EXPECT_GE(manager.get_current_level(), DegradationManager::DegradationLevel::MODERATE);
    EXPECT_FALSE(manager.is_feature_enabled("test_feature"));
    
    manager.stop_monitoring();
}

TEST_F(ErrorHandlingResilienceTest, FeatureGuard) {
    DegradationManager::DegradationConfig config;
    DegradationManager manager(config);
    
    manager.register_feature("guarded_feature", DegradationManager::DegradationLevel::LIGHT);
    
    // Feature should be enabled initially
    {
        FeatureGuard guard("guarded_feature", manager);
        EXPECT_TRUE(guard.should_execute());
        EXPECT_TRUE(static_cast<bool>(guard));
    }
    
    // Force degradation
    manager.force_degradation_level(DegradationManager::DegradationLevel::MODERATE);
    
    // Feature should be disabled now
    {
        FeatureGuard guard("guarded_feature", manager);
        EXPECT_FALSE(guard.should_execute());
        EXPECT_FALSE(static_cast<bool>(guard));
    }
}

// Fallback Manager Tests
TEST_F(ErrorHandlingResilienceTest, FallbackManagerBasicFunctionality) {
    FallbackManager::FallbackConfig config;
    config.nodejs_base_url = "http://localhost:3005";
    config.enable_automatic_fallback = true;
    
    FallbackManager manager(config);
    
    // Register a custom fallback handler
    manager.register_fallback_handler("/test", [](const std::string& data) {
        return "fallback_response";
    });
    
    // Test fallback execution
    std::string result = manager.execute_nodejs_fallback("/test", "");
    EXPECT_EQ(result, "fallback_response");
    
    // Test fallback decision
    EXPECT_TRUE(manager.should_fallback("/test", FallbackManager::FallbackReason::COMPONENT_FAILURE));
    
    // Force fallback
    manager.force_fallback("/test", true);
    EXPECT_TRUE(manager.should_fallback("/test", FallbackManager::FallbackReason::PERFORMANCE_DEGRADATION));
}

// Structured Logging Tests
TEST_F(ErrorHandlingResilienceTest, StructuredLogging) {
    // Test log context
    LogContext context;
    context.add("user_id", "12345")
           .add("operation", "test_operation")
           .add("duration_ms", 150);
    
    std::string json = context.to_json();
    EXPECT_NE(json.find("user_id"), std::string::npos);
    EXPECT_NE(json.find("12345"), std::string::npos);
    
    // Test structured logging (would need to capture output for full test)
    Logger::log_structured(LogLevel::INFO, "Test message", context);
    
    // Test performance logging
    Logger::log_performance("test_operation", 1500000, context); // 1.5ms in nanoseconds
    
    // Test error logging with exception
    try {
        throw std::runtime_error("Test exception");
    } catch (const std::exception& e) {
        Logger::log_error("Operation failed", e, context);
    }
}

TEST_F(ErrorHandlingResilienceTest, PerformanceTimer) {
    LogContext context;
    context.add("test_id", "timer_test");
    
    {
        PerformanceTimer timer("test_operation", context);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        // Timer destructor will log performance automatically
    }
    
    // Performance should be logged when timer goes out of scope
}

// Resilience System Integration Tests
TEST_F(ErrorHandlingResilienceTest, ResilienceSystemIntegration) {
    ResilienceSystem::Config config;
    config.circuit_breaker_config.failure_threshold = 0.5;
    config.circuit_breaker_config.minimum_requests = 2;
    config.degradation_config.cpu_light_threshold = 50.0;
    config.fallback_config.enable_automatic_fallback = true;
    config.enable_auto_fallback_on_circuit_open = true;
    
    ResilienceSystem system(config);
    system.initialize();
    system.start();
    
    // Register an operation
    system.register_operation("test_op", config.circuit_breaker_config, 
                             DegradationManager::DegradationLevel::LIGHT);
    
    // Test resilient execution with success
    auto future1 = system.execute_resilient("test_op", []() { return std::string("success"); });
    EXPECT_EQ(future1.get(), "success");
    
    // Test system health
    EXPECT_TRUE(system.is_system_healthy());
    
    // Get system stats
    auto stats = system.get_system_stats();
    EXPECT_GT(stats.total_operations, 0);
    
    // Export Prometheus metrics
    std::string metrics = system.export_prometheus_metrics();
    EXPECT_NE(metrics.find("ultra_operations_total"), std::string::npos);
    
    system.stop();
}

TEST_F(ErrorHandlingResilienceTest, GlobalResilienceSystem) {
    ResilienceSystem::Config config;
    config.logger_config.enable_console_output = false;
    
    // Initialize global system
    GlobalResilienceSystem::initialize(config);
    
    // Get instance
    ResilienceSystem& system = GlobalResilienceSystem::instance();
    
    // Test operation registration
    system.register_operation("global_test_op");
    
    // Test resilient execution
    auto future = system.execute_resilient("global_test_op", []() { return 100; });
    EXPECT_EQ(future.get(), 100);
    
    // Shutdown
    GlobalResilienceSystem::shutdown();
}

// Error Recovery Tests
TEST_F(ErrorHandlingResilienceTest, ErrorRecoveryScenarios) {
    CircuitBreaker::Config cb_config;
    cb_config.failure_threshold = 0.6;
    cb_config.minimum_requests = 3;
    cb_config.initial_timeout_ms = 50;
    
    CircuitBreaker breaker("recovery_test", cb_config);
    
    // Cause failures to open circuit
    for (int i = 0; i < 5; ++i) {
        breaker.record_failure();
    }
    
    EXPECT_EQ(breaker.get_state(), CircuitBreaker::State::OPEN);
    
    // Wait for timeout
    std::this_thread::sleep_for(std::chrono::milliseconds(60));
    
    // Should transition to half-open
    EXPECT_TRUE(breaker.allow_request());
    
    // Record success to close circuit
    breaker.record_success();
    breaker.record_success();
    breaker.record_success();
    
    // Should be closed now
    EXPECT_EQ(breaker.get_state(), CircuitBreaker::State::CLOSED);
}

// Performance and Load Tests
TEST_F(ErrorHandlingResilienceTest, PerformanceUnderLoad) {
    CircuitBreaker::Config config;
    config.failure_threshold = 0.1; // Very low threshold
    config.minimum_requests = 10;
    
    CircuitBreaker breaker("load_test", config);
    
    const int num_operations = 1000;
    std::atomic<int> successes{0};
    std::atomic<int> failures{0};
    std::atomic<int> rejections{0};
    
    std::vector<std::thread> threads;
    
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&]() {
            for (int i = 0; i < num_operations / 4; ++i) {
                try {
                    breaker.execute([&]() {
                        // Simulate some failures
                        if (i % 20 == 0) {
                            throw std::runtime_error("Simulated failure");
                        }
                        successes.fetch_add(1);
                        return i;
                    });
                } catch (const CircuitBreakerOpenException&) {
                    rejections.fetch_add(1);
                } catch (const std::runtime_error&) {
                    failures.fetch_add(1);
                }
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    auto stats = breaker.get_stats();
    EXPECT_GT(stats.total_requests.load(), 0);
    EXPECT_GT(successes.load(), 0);
    
    // Verify that circuit breaker prevented some requests when opened
    if (stats.rejected_requests.load() > 0) {
        EXPECT_GT(rejections.load(), 0);
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}