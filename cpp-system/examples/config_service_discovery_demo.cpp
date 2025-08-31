#include <iostream>
#include <thread>
#include <chrono>
#include <unordered_map>

#include "common/config_manager.hpp"
#include "common/service_registry.hpp"
#include "common/dynamic_routing.hpp"
#include "common/feature_flags.hpp"
#include "common/logger.hpp"

using namespace ultra::common;

void demonstrate_config_management() {
    std::cout << "\n=== Configuration Management Demo ===\n";
    
    auto& config = ConfigManager::instance();
    
    // Load configuration from file
    if (config.load_from_file("config/ultra-cpp.conf")) {
        std::cout << "✓ Configuration loaded successfully\n";
        
        // Get various configuration values
        int port = config.get<int>("server.port", 8080);
        bool enable_dpdk = config.get<bool>("server.enable_dpdk", false);
        std::string fallback_service = config.get<std::string>("routing.default_fallback_service", "nodejs");
        
        std::cout << "  Server port: " << port << "\n";
        std::cout << "  DPDK enabled: " << (enable_dpdk ? "yes" : "no") << "\n";
        std::cout << "  Fallback service: " << fallback_service << "\n";
        
        // Demonstrate feature flags
        bool ultra_routing = config.is_feature_enabled("ultra_fast_routing");
        bool gpu_accel = config.is_feature_enabled("gpu_acceleration");
        
        std::cout << "  Ultra fast routing: " << (ultra_routing ? "enabled" : "disabled") << "\n";
        std::cout << "  GPU acceleration: " << (gpu_accel ? "enabled" : "disabled") << "\n";
        
        // Start file watching for hot reload
        config.start_file_watching();
        std::cout << "✓ File watching started for hot reload\n";
        
        // Set up change callback
        config.watch_changes([](const std::string& key, const std::string& value) {
            std::cout << "  Config changed: " << key << " = " << value << "\n";
        });
        
    } else {
        std::cout << "✗ Failed to load configuration\n";
    }
}

void demonstrate_service_registry() {
    std::cout << "\n=== Service Registry Demo ===\n";
    
    auto& registry = ServiceRegistry::instance();
    
    // Register some services
    registry.register_service("ultra-cpp-api", "http://localhost:8080", 
                             {"api", "high-performance"}, "http://localhost:8080/health");
    
    registry.register_service("ultra-cpp-cache", "http://localhost:8081", 
                             {"cache", "memory"}, "http://localhost:8081/health");
    
    registry.register_service("nodejs-backend", "http://localhost:3005", 
                             {"api", "fallback"}, "http://localhost:3005/health");
    
    std::cout << "✓ Registered 3 services\n";
    
    // Set up service change monitoring
    registry.watch_service_changes([](const std::string& service_name, bool available) {
        std::cout << "  Service " << service_name << " is now " 
                  << (available ? "available" : "unavailable") << "\n";
    });
    
    // Start health monitoring
    registry.start_health_monitoring();
    std::cout << "✓ Health monitoring started\n";
    
    // Discover services by capability
    auto api_services = registry.discover_services("api");
    std::cout << "  Found " << api_services.size() << " API services:\n";
    for (const auto& service : api_services) {
        std::cout << "    - " << service.name << " at " << service.endpoint << "\n";
    }
    
    // Simulate heartbeats
    registry.heartbeat("ultra-cpp-api");
    registry.heartbeat("ultra-cpp-cache");
    
    // Set service metadata
    registry.set_service_metadata("ultra-cpp-api", "version", "1.0.0");
    registry.set_service_metadata("ultra-cpp-api", "load", "low");
    
    std::cout << "✓ Service metadata updated\n";
}

void demonstrate_dynamic_routing() {
    std::cout << "\n=== Dynamic Routing Demo ===\n";
    
    auto& router = DynamicRouter::instance();
    
    // Load routing configuration from JSON
    std::ifstream routes_file("config/routes.json");
    if (routes_file.is_open()) {
        std::stringstream buffer;
        buffer << routes_file.rdbuf();
        router.load_routes_from_json(buffer.str());
        std::cout << "✓ Loaded routing rules from JSON\n";
    }
    
    // Load A/B test configuration
    std::ifstream ab_tests_file("config/ab_tests.json");
    if (ab_tests_file.is_open()) {
        std::stringstream buffer;
        buffer << ab_tests_file.rdbuf();
        router.load_ab_tests_from_json(buffer.str());
        std::cout << "✓ Loaded A/B test configuration\n";
    }
    
    // Test routing decisions
    std::vector<std::pair<std::string, std::unordered_map<std::string, std::string>>> test_requests = {
        {"/api/v1/posts", {{"X-User-ID", "user123"}}},
        {"/api/media/upload", {{"X-User-ID", "user456"}}},
        {"/api/cache/get", {}},
        {"/admin/dashboard", {}},
        {"/unknown/path", {}}
    };
    
    std::cout << "  Testing routing decisions:\n";
    for (const auto& [path, headers] : test_requests) {
        auto decision = router.route_request(path, headers);
        
        std::cout << "    " << path << " -> ";
        if (decision.matched) {
            std::cout << decision.target_service;
            if (!decision.ab_test_variant.empty()) {
                std::cout << " (A/B: " << decision.ab_test_variant << ")";
            }
        } else {
            std::cout << "no match";
        }
        std::cout << "\n";
    }
    
    // Show routing statistics
    auto stats = router.get_stats();
    std::cout << "  Routing stats:\n";
    std::cout << "    Total requests: " << stats.total_requests << "\n";
    std::cout << "    Unmatched requests: " << stats.unmatched_requests << "\n";
}

void demonstrate_feature_flags() {
    std::cout << "\n=== Feature Flags Demo ===\n";
    
    auto& flags = FeatureFlagManager::instance();
    
    // Load feature flags from JSON
    flags.load_from_file("config/feature_flags.json");
    std::cout << "✓ Loaded feature flags from JSON\n";
    
    // Test feature flag evaluation
    FeatureFlagContext user_context;
    user_context.user_id = "user123";
    user_context.group_id = "beta_testers";
    
    std::vector<std::string> test_flags = {
        "ultra_fast_api",
        "gpu_acceleration", 
        "advanced_caching",
        "real_time_metrics",
        "ab_testing_framework",
        "dpdk_networking"
    };
    
    std::cout << "  Feature flag evaluation for user123 (beta_testers):\n";
    for (const std::string& flag_name : test_flags) {
        bool enabled = flags.is_enabled(flag_name, user_context);
        std::cout << "    " << flag_name << ": " << (enabled ? "enabled" : "disabled") << "\n";
    }
    
    // Test different user context
    FeatureFlagContext admin_context;
    admin_context.user_id = "admin";
    admin_context.group_id = "infrastructure";
    
    std::cout << "  Feature flag evaluation for admin (infrastructure):\n";
    for (const std::string& flag_name : test_flags) {
        bool enabled = flags.is_enabled(flag_name, admin_context);
        std::cout << "    " << flag_name << ": " << (enabled ? "enabled" : "disabled") << "\n";
    }
    
    // Watch for flag changes
    flags.watch_flag_changes([](const std::string& flag_name, bool old_value, bool new_value) {
        std::cout << "  Flag changed: " << flag_name << " " 
                  << (old_value ? "enabled" : "disabled") << " -> " 
                  << (new_value ? "enabled" : "disabled") << "\n";
    });
    
    // Demonstrate runtime flag updates
    std::cout << "✓ Updating gpu_acceleration flag...\n";
    auto gpu_flag = flags.get_flag_info("gpu_acceleration");
    if (gpu_flag) {
        gpu_flag->enabled = true;
        gpu_flag->rollout_percentage = 10.0;
        flags.update_flag("gpu_acceleration", *gpu_flag);
    }
    
    // Show statistics
    auto stats = flags.get_stats();
    std::cout << "  Flag evaluation stats:\n";
    std::cout << "    Total evaluations: " << stats.total_evaluations << "\n";
    for (const auto& [flag, count] : stats.evaluation_counts) {
        std::cout << "    " << flag << ": " << count << " evaluations\n";
    }
}

void demonstrate_integration() {
    std::cout << "\n=== Integration Demo ===\n";
    
    auto& config = ConfigManager::instance();
    auto& registry = ServiceRegistry::instance();
    auto& router = DynamicRouter::instance();
    auto& flags = FeatureFlagManager::instance();
    
    // Simulate a request processing pipeline
    std::string request_path = "/api/v1/posts/123";
    std::unordered_map<std::string, std::string> request_headers = {
        {"X-User-ID", "user789"},
        {"User-Agent", "UltraClient/1.0"}
    };
    
    std::cout << "  Processing request: " << request_path << "\n";
    
    // 1. Check feature flags
    FeatureFlagContext context;
    context.user_id = request_headers["X-User-ID"];
    
    bool ultra_api_enabled = flags.is_enabled("ultra_fast_api", context);
    bool advanced_caching = flags.is_enabled("advanced_caching", context);
    
    std::cout << "    Ultra API enabled: " << (ultra_api_enabled ? "yes" : "no") << "\n";
    std::cout << "    Advanced caching: " << (advanced_caching ? "yes" : "no") << "\n";
    
    // 2. Route the request
    auto routing_decision = router.route_request(request_path, request_headers);
    
    if (routing_decision.matched) {
        std::cout << "    Routed to: " << routing_decision.target_service << "\n";
        
        if (!routing_decision.ab_test_variant.empty()) {
            std::cout << "    A/B test variant: " << routing_decision.ab_test_variant << "\n";
        }
        
        // 3. Discover the target service
        auto service_info = registry.get_service(routing_decision.target_service);
        if (service_info) {
            std::cout << "    Service endpoint: " << service_info->endpoint << "\n";
            std::cout << "    Service healthy: " << (service_info->is_healthy ? "yes" : "no") << "\n";
        } else {
            std::cout << "    Service not found in registry\n";
        }
    } else {
        std::cout << "    No route matched - using fallback\n";
        std::string fallback = config.get<std::string>("routing.default_fallback_service", "nodejs-backend");
        std::cout << "    Fallback service: " << fallback << "\n";
    }
    
    std::cout << "✓ Request processing complete\n";
}

int main() {
    std::cout << "Ultra Low-Latency C++ System - Configuration and Service Discovery Demo\n";
    std::cout << "=====================================================================\n";
    
    try {
        // Initialize logging
        Logger::instance().set_level(LogLevel::INFO);
        
        // Run demonstrations
        demonstrate_config_management();
        demonstrate_service_registry();
        demonstrate_dynamic_routing();
        demonstrate_feature_flags();
        demonstrate_integration();
        
        std::cout << "\n=== Demo Complete ===\n";
        std::cout << "All systems are running. Press Ctrl+C to exit.\n";
        
        // Keep the demo running to show real-time features
        std::this_thread::sleep_for(std::chrono::seconds(5));
        
        // Cleanup
        ConfigManager::instance().stop_file_watching();
        ServiceRegistry::instance().stop_health_monitoring();
        FeatureFlagManager::instance().stop_real_time_updates();
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}