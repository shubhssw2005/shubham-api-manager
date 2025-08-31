# Error Handling and Resilience System

This document describes the comprehensive error handling and resilience system implemented for the Ultra Low-Latency C++ System. The system provides circuit breakers, graceful degradation, automatic fallback mechanisms, and structured logging to ensure system stability and reliability.

## Overview

The resilience system consists of four main components:

1. **Circuit Breaker Pattern** - Prevents cascading failures by monitoring operation success/failure rates
2. **Graceful Degradation Manager** - Automatically reduces functionality when performance issues are detected
3. **Fallback Manager** - Provides automatic fallback to Node.js layer when C++ components fail
4. **Structured Logging** - Comprehensive logging with structured format for better observability

## Components

### Circuit Breaker

The circuit breaker implementation provides protection against cascading failures with exponential backoff.

#### Features
- Configurable failure thresholds and timeouts
- Three states: CLOSED, OPEN, HALF_OPEN
- Exponential backoff with configurable multipliers
- Thread-safe operation with lock-free counters
- Comprehensive statistics and monitoring

#### Usage Example
```cpp
#include "common/circuit_breaker.hpp"

CircuitBreaker::Config config;
config.failure_threshold = 0.5;        // 50% failure rate
config.minimum_requests = 10;          // Minimum requests before evaluation
config.initial_timeout_ms = 5000;      // 5 second initial timeout

CircuitBreaker breaker("my_service", config);

// Execute operation with circuit breaker protection
try {
    auto result = breaker.execute([&]() {
        return call_external_service();
    });
    // Handle successful result
} catch (const CircuitBreakerOpenException& e) {
    // Circuit breaker is open, handle gracefully
} catch (const std::exception& e) {
    // Operation failed, circuit breaker recorded the failure
}
```

#### Configuration Options
- `failure_threshold`: Percentage of failures that triggers circuit opening (0.0-1.0)
- `minimum_requests`: Minimum number of requests before evaluating failure rate
- `time_window_ms`: Time window for failure rate calculation
- `initial_timeout_ms`: Initial timeout when circuit opens
- `max_timeout_ms`: Maximum timeout for exponential backoff
- `backoff_multiplier`: Multiplier for exponential backoff
- `half_open_max_calls`: Number of test requests in half-open state

### Degradation Manager

The degradation manager monitors system performance and automatically degrades functionality to maintain stability.

#### Features
- Five degradation levels: NORMAL, LIGHT, MODERATE, HEAVY, EMERGENCY
- Configurable thresholds for CPU, memory, response time, and error rate
- Feature-based degradation with per-feature minimum levels
- Hysteresis to prevent oscillation
- Automatic strategy application

#### Usage Example
```cpp
#include "common/degradation_manager.hpp"

DegradationManager::DegradationConfig config;
config.cpu_light_threshold = 70.0;     // 70% CPU usage
config.cpu_moderate_threshold = 80.0;  // 80% CPU usage
config.evaluation_interval = std::chrono::milliseconds(1000);

DegradationManager manager(config);

// Register features with minimum degradation levels
manager.register_feature("analytics", DegradationManager::DegradationLevel::LIGHT);
manager.register_feature("caching", DegradationManager::DegradationLevel::MODERATE);

manager.start_monitoring();

// Check if feature should be executed
if (manager.is_feature_enabled("analytics")) {
    // Execute analytics functionality
}

// Or use RAII guard
{
    DEGRADATION_GUARD("analytics", manager);
    // This code only executes if analytics is enabled
    perform_analytics();
}
```

#### Degradation Levels
- **NORMAL**: Full functionality
- **LIGHT**: Minor optimizations (reduce logging verbosity)
- **MODERATE**: Disable non-essential features
- **HEAVY**: Minimal functionality only
- **EMERGENCY**: Critical operations only

### Fallback Manager

The fallback manager provides automatic fallback to Node.js services when C++ components fail or performance degrades.

#### Features
- Automatic health monitoring of Node.js services
- Custom fallback handlers for different endpoints
- Performance-based fallback decisions
- Retry logic with exponential backoff
- Comprehensive statistics and monitoring

#### Usage Example
```cpp
#include "common/fallback_manager.hpp"

FallbackManager::FallbackConfig config;
config.nodejs_base_url = "http://localhost:3005";
config.enable_automatic_fallback = true;

FallbackManager manager(config);

// Register custom fallback handler
manager.register_fallback_handler("/api/users", [](const std::string& data) {
    return R"({"users": [], "source": "fallback"})";
});

manager.start_health_monitoring();

// Execute with automatic fallback
auto future = manager.execute_with_fallback("/api/users", [&]() {
    return cpp_get_users();  // C++ implementation
});

std::string result = future.get();
```

### Structured Logging

Enhanced logging system with structured format support and performance monitoring.

#### Features
- Multiple output formats: PLAIN, JSON, LOGFMT
- Structured context with key-value pairs
- Performance timing with automatic logging
- Security and audit event logging
- Asynchronous logging support
- File and console output

#### Usage Example
```cpp
#include "common/logger.hpp"

// Configure logger
Logger::Config config;
config.format = Logger::OutputFormat::JSON;
config.level = LogLevel::INFO;
Logger::initialize("my_service", config);

// Basic structured logging
LogContext context;
context.add("user_id", "12345")
       .add("operation", "login")
       .add("ip_address", "192.168.1.100");

LOG_STRUCTURED(LogLevel::INFO, "User login successful", context);

// Performance timing
{
    PERFORMANCE_TIMER_WITH_CONTEXT("database_query", context);
    execute_database_query();
    // Automatically logs performance when scope exits
}

// Error logging with exception
try {
    risky_operation();
} catch (const std::exception& e) {
    LOG_ERROR_WITH_EXCEPTION("Operation failed", e, context);
}

// Security event logging
LOG_SECURITY("unauthorized_access", "Invalid token provided", context);

// Audit logging
LOG_AUDIT("user_created", "admin", "user:12345", context);
```

## Integrated Resilience System

The `ResilienceSystem` class integrates all components into a unified framework.

### Features
- Automatic integration between components
- High-level resilient execution wrapper
- Comprehensive system health monitoring
- Prometheus metrics export
- Global instance management

### Usage Example
```cpp
#include "common/resilience_system.hpp"

// Initialize global resilience system
ResilienceSystem::Config config;
config.circuit_breaker_config.failure_threshold = 0.5;
config.degradation_config.cpu_light_threshold = 70.0;
config.fallback_config.nodejs_base_url = "http://localhost:3005";

GlobalResilienceSystem::initialize(config);

// Register operations
auto& system = GlobalResilienceSystem::instance();
system.register_operation("user_service", config.circuit_breaker_config,
                         DegradationManager::DegradationLevel::LIGHT);

// Execute operations with full resilience
auto future = system.execute_resilient("user_service", [&]() {
    return get_user_data(user_id);
});

try {
    auto result = future.get();
    // Handle successful result
} catch (const std::exception& e) {
    // All resilience mechanisms have been applied
    LOG_ERROR("All resilience mechanisms failed: {}", e.what());
}

// Monitor system health
if (system.is_system_healthy()) {
    // System is operating normally
} else {
    // System is degraded, consider manual intervention
}

// Export metrics for monitoring
std::string metrics = system.export_prometheus_metrics();
```

## Configuration

### Environment Variables
- `ULTRA_LOG_LEVEL`: Set logging level (TRACE, DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `ULTRA_LOG_FORMAT`: Set log format (PLAIN, JSON, LOGFMT)
- `ULTRA_NODEJS_URL`: Base URL for Node.js fallback services
- `ULTRA_CIRCUIT_BREAKER_THRESHOLD`: Default circuit breaker failure threshold

### Configuration Files
The system supports configuration through JSON files:

```json
{
  "circuit_breaker": {
    "failure_threshold": 0.5,
    "minimum_requests": 10,
    "initial_timeout_ms": 5000,
    "max_timeout_ms": 300000,
    "backoff_multiplier": 2.0
  },
  "degradation": {
    "cpu_light_threshold": 70.0,
    "cpu_moderate_threshold": 80.0,
    "cpu_heavy_threshold": 90.0,
    "cpu_emergency_threshold": 95.0,
    "evaluation_interval_ms": 1000
  },
  "fallback": {
    "nodejs_base_url": "http://localhost:3005",
    "request_timeout_ms": 5000,
    "health_check_interval_ms": 30000,
    "enable_automatic_fallback": true
  },
  "logging": {
    "level": "INFO",
    "format": "JSON",
    "enable_file_output": true,
    "log_file_path": "/var/log/ultra-cpp.log",
    "enable_async_logging": true
  }
}
```

## Monitoring and Metrics

### Prometheus Metrics
The system exports comprehensive metrics in Prometheus format:

- `ultra_circuit_breaker_total`: Total number of circuit breakers
- `ultra_circuit_breaker_open`: Number of open circuit breakers
- `ultra_circuit_breaker_requests_total`: Total circuit breaker requests
- `ultra_degradation_level`: Current system degradation level
- `ultra_fallback_requests_total`: Total fallback requests
- `ultra_nodejs_available`: Node.js service availability
- `ultra_operations_total`: Total operations executed
- `ultra_success_rate`: Overall system success rate
- `ultra_system_healthy`: System health status

### Health Checks
The system provides health check endpoints:

```cpp
// Check overall system health
bool healthy = system.is_system_healthy();

// Get detailed health information
auto stats = system.get_system_stats();
```

### Alerting
Configure alerts based on metrics:

- Circuit breaker open rate > 10%
- System degradation level >= MODERATE
- Node.js service unavailable
- Overall success rate < 95%
- System unhealthy for > 5 minutes

## Best Practices

### Circuit Breaker Usage
1. Set appropriate failure thresholds based on service SLAs
2. Use different configurations for different service types
3. Monitor circuit breaker state changes
4. Implement proper fallback strategies

### Degradation Management
1. Register features with appropriate minimum degradation levels
2. Use feature guards for non-critical functionality
3. Monitor degradation level changes
4. Implement custom degradation strategies for specific components

### Fallback Strategies
1. Implement meaningful fallback responses
2. Monitor fallback success rates
3. Test fallback mechanisms regularly
4. Consider data consistency implications

### Logging Best Practices
1. Use structured logging with consistent field names
2. Include correlation IDs for request tracing
3. Log performance metrics for critical operations
4. Use appropriate log levels
5. Avoid logging sensitive information

## Testing

### Unit Tests
Run the comprehensive test suite:

```bash
cd cpp-system
mkdir build && cd build
cmake ..
make test_error_handling_resilience
./tests/common/test_error_handling_resilience
```

### Integration Tests
Test with real Node.js services:

```bash
# Start Node.js service
npm start

# Run integration tests
./tests/integration/test_resilience_integration
```

### Load Testing
Test system behavior under load:

```bash
# Run load test
./tests/load/test_resilience_load --requests=10000 --concurrency=100
```

## Troubleshooting

### Common Issues

1. **Circuit Breaker Stuck Open**
   - Check service health
   - Verify timeout configuration
   - Review failure threshold settings

2. **Excessive Degradation**
   - Monitor system resources
   - Check threshold configurations
   - Review performance metrics

3. **Fallback Failures**
   - Verify Node.js service availability
   - Check network connectivity
   - Review fallback handler implementations

4. **High Memory Usage**
   - Enable async logging
   - Adjust log buffer sizes
   - Monitor log file rotation

### Debug Mode
Enable debug logging for troubleshooting:

```cpp
Logger::set_level(LogLevel::DEBUG);
```

### Performance Profiling
Use performance timers to identify bottlenecks:

```cpp
{
    PERFORMANCE_TIMER("critical_operation");
    critical_operation();
}
```

## Future Enhancements

1. **Adaptive Thresholds**: Machine learning-based threshold adjustment
2. **Distributed Circuit Breakers**: Cluster-wide circuit breaker state
3. **Advanced Fallback Strategies**: Canary deployments and A/B testing
4. **Real-time Dashboards**: Web-based monitoring interface
5. **Automated Recovery**: Self-healing mechanisms

## References

- [Circuit Breaker Pattern](https://martinfowler.com/bliki/CircuitBreaker.html)
- [Bulkhead Pattern](https://docs.microsoft.com/en-us/azure/architecture/patterns/bulkhead)
- [Graceful Degradation](https://en.wikipedia.org/wiki/Graceful_degradation)
- [Structured Logging](https://stackify.com/what-is-structured-logging-and-why-developers-need-it/)
- [Prometheus Metrics](https://prometheus.io/docs/concepts/metric_types/)