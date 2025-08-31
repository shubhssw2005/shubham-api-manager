#include "include/common/circuit_breaker.hpp"
#include "include/common/degradation_manager.hpp"
#include "include/common/logger.hpp"
#include <iostream>

int main() {
    try {
        // Test basic logger functionality
        ultra::common::Logger::Config log_config;
        log_config.level = ultra::common::LogLevel::INFO;
        log_config.enable_console_output = true;
        ultra::common::Logger::initialize("test", log_config);
        
        LOG_INFO("Error handling system compilation test");
        
        // Test circuit breaker creation
        ultra::common::CircuitBreaker::Config cb_config;
        cb_config.failure_threshold = 0.5;
        ultra::common::CircuitBreaker breaker("test", cb_config);
        
        LOG_INFO("Circuit breaker created successfully");
        
        // Test degradation manager creation
        ultra::common::DegradationManager::DegradationConfig deg_config;
        deg_config.cpu_light_threshold = 70.0;
        ultra::common::DegradationManager manager(deg_config);
        
        LOG_INFO("Degradation manager created successfully");
        
        std::cout << "Error handling system compilation successful!" << std::endl;
        
        ultra::common::Logger::shutdown();
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Compilation test failed: " << e.what() << std::endl;
        return 1;
    }
}