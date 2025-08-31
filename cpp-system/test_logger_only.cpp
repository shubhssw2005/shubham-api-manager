#include "include/common/logger.hpp"
#include <iostream>

int main() {
    try {
        // Test basic logger functionality
        ultra::common::Logger::Config log_config;
        log_config.level = ultra::common::LogLevel::INFO;
        log_config.enable_console_output = true;
        ultra::common::Logger::initialize("test", log_config);
        
        LOG_INFO("Logger compilation test successful");
        
        // Test structured logging
        ultra::common::LogContext context;
        context.add("test_id", "12345")
               .add("operation", "compilation_test");
        
        LOG_STRUCTURED(ultra::common::LogLevel::INFO, "Structured logging test", context);
        
        std::cout << "Logger system compilation successful!" << std::endl;
        
        ultra::common::Logger::shutdown();
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Logger test failed: " << e.what() << std::endl;
        return 1;
    }
}