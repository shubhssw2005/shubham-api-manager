#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <fstream>
#include <filesystem>
#include <thread>
#include <chrono>

#include "common/config_manager.hpp"
#include "common/service_registry.hpp"
#include "common/dynamic_routing.hpp"
#include "common/feature_flags.hpp"

using namespace ultra::common;
using namespace std::chrono_literals;

class ConfigServiceDiscoveryTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create temporary config file
        test_config_file_ = "/tmp/test_config.conf";
        std::ofstream config_file(test_config_file_);
        config_file << "[server]\n";
        config_file << "port=8080\n";
        config_file << "enable_dpdk=true\n";
        config_file << "[features]\n";
        config_file << "test_feature=true\n";
        config_file.close();
        
        // Create temporary feature flags file
        test_flags_file_ = "/tmp/test_flags.json";
        std::ofstream flags_file(test_flags_file_);
        flags_file << R"({
            "test_flag": {
                "enabled": true,
                "description": "Test flag",
                "rollout_percentage": 50.0,
                "allowed_users": ["test_user"],
                "allowed_groups": ["test_group"]
            }
        })";
        flags_file.close();
    }
    
    void TearDown() override {
        std::filesystem::remove(test_config_file_);
        std::filesystem::remove(test_flags_file_);
        
        // Clean up singletons
        ConfigManager::instance().stop_file_watching();
        ServiceRegistry::instance().stop_health_monitoring();
        FeatureFlagManager::instance().stop_real_time_updates();
    }
    
    std::string test_config_file_;
    std::string test_flags_file_;
};

TEST_F(ConfigServiceDiscoveryTest, ConfigManagerBasicOperations) {
    auto& config = ConfigManager::instance();
    
    // Test loading from file
    ASSERT_TRUE(config.load_from_file(test_config_file_));
    
    // Test getting values
    EXPECT_EQ(config.get<int>("server.port", 0), 8080);
    EXPECT_TRUE(config.get<bool>("server.enable_dpdk", false));
    EXPECT_TRUE(config.get<bool>("features.test_feature", false));
    
    // Test setting values
    config.set<std::string>("test.key", "test_value");
    EXPECT_EQ(config.get<std::string>("test.key", ""), "test_value");
    
    // Test key existence
    EXPECT_TRUE(config.has_key("server.port"));
    EXPECT_FALSE(config.has_key("nonexistent.key"));
}

TEST_F(ConfigServiceDiscoveryTest, ConfigManagerFeatureFlags) {
    auto& config = ConfigManager::instance();
    config.load_from_file(test_config_file_);
    
    // Test feature flag operations
    EXPECT_TRUE(config.is_feature_enabled("test_feature"));
    EXPECT_FALSE(config.is_feature_enabled("nonexistent_feature"));
    
    // Test setting feature flags
    config.set_feature_flag("new_feature", true);
    EXPECT_TRUE(config.is_feature_enabled("new_feature"));
    
    config.set_feature_flag("new_feature", false);
    EXPECT_FALSE(config.is_feature_enabled("new_feature"));
    
    // Test getting all feature flags
    auto all_flags = config.get_all_feature_flags();
    EXPECT_TRUE(all_flags["test_feature"]);
    EXPECT_FALSE(all_flags["new_feature"]);
}

TEST_F(ConfigServiceDiscoveryTest, ConfigManagerFileWatching) {
    auto& config = ConfigManager::instance();
    config.load_from_file(test_config_file_);
    
    bool callback_called = false;
    std::string changed_key;
    std::string changed_value;
    
    config.watch_changes([&](const std::string& key, const std::string& value) {
        callback_called = true;
        changed_key = key;
        changed_value = value;
    });
    
    config.start_file_watching();
    EXPECT_TRUE(config.is_watching());
    
    // Modify the config file
    std::this_thread::sleep_for(100ms);
    std::ofstream config_file(test_config_file_, std::ios::app);
    config_file << "new_key=new_value\n";
    config_file.close();
    
    // Wait for file watcher to detect change
    std::this_thread::sleep_for(500ms);
    
    config.stop_file_watching();
    EXPECT_FALSE(config.is_watching());
}

TEST_F(ConfigServiceDiscoveryTest, ServiceRegistryBasicOperations) {
    auto& registry = ServiceRegistry::instance();
    
    // Test service registration
    EXPECT_TRUE(registry.register_service("test-service", "http://localhost:8080", 
                                        {"api", "test"}, "http://localhost:8080/health"));
    
    // Test service discovery
    auto service = registry.get_service("test-service");
    ASSERT_TRUE(service.has_value());
    EXPECT_EQ(service->name, "test-service");
    EXPECT_EQ(service->endpoint, "http://localhost:8080");
    EXPECT_EQ(service->capabilities.size(), 2);
    
    // Test service discovery by capability
    auto api_services = registry.discover_services("api");
    EXPECT_EQ(api_services.size(), 1);
    EXPECT_EQ(api_services[0].name, "test-service");
    
    // Test service metadata
    registry.set_service_metadata("test-service", "version", "1.0.0");
    EXPECT_EQ(registry.get_service_metadata("test-service", "version"), "1.0.0");
    
    // Test heartbeat
    registry.heartbeat("test-service");
    
    // Test deregistration
    EXPECT_TRUE(registry.deregister_service("test-service"));
    EXPECT_FALSE(registry.get_service("test-service").has_value());
}

TEST_F(ConfigServiceDiscoveryTest, ServiceRegistryHealthMonitoring) {
    auto& registry = ServiceRegistry::instance();
    
    bool service_change_called = false;
    std::string changed_service;
    bool service_available;
    
    registry.watch_service_changes([&](const std::string& service_name, bool available) {
        service_change_called = true;
        changed_service = service_name;
        service_available = available;
    });
    
    registry.register_service("health-test-service", "http://localhost:9999", 
                            {"test"}, "http://localhost:9999/health");
    
    registry.start_health_monitoring();
    registry.set_heartbeat_timeout(1s);
    registry.set_health_check_interval(1s);
    
    // Wait for health check to potentially fail
    std::this_thread::sleep_for(2s);
    
    registry.stop_health_monitoring();
}

TEST_F(ConfigServiceDiscoveryTest, DynamicRoutingBasicOperations) {
    auto& router = DynamicRouter::instance();
    
    // Test adding routes
    RouteRule rule;
    rule.pattern = "^/api/v1/.*";
    rule.target_service = "api-service";
    rule.priority = 100;
    rule.enabled = true;
    
    router.add_route("api_v1", rule);
    
    // Test routing decision
    std::unordered_map<std::string, std::string> headers;
    auto decision = router.route_request("/api/v1/users", headers);
    
    EXPECT_TRUE(decision.matched);
    EXPECT_EQ(decision.target_service, "api-service");
    
    // Test unmatched route
    auto no_match = router.route_request("/unknown/path", headers);
    EXPECT_FALSE(no_match.matched);
    
    // Test route removal
    router.remove_route("api_v1");
    auto after_removal = router.route_request("/api/v1/users", headers);
    EXPECT_FALSE(after_removal.matched);
}

TEST_F(ConfigServiceDiscoveryTest, DynamicRoutingABTesting) {
    auto& router = DynamicRouter::instance();
    
    // Set up A/B test
    ABTestConfig ab_config;
    ab_config.test_name = "api_test";
    ab_config.control_service = "old-api";
    ab_config.variant_service = "new-api";
    ab_config.traffic_split = 0.5;
    ab_config.user_id_header = "X-User-ID";
    ab_config.enabled = true;
    ab_config.start_time = std::chrono::steady_clock::now() - 1h;
    ab_config.end_time = std::chrono::steady_clock::now() + 1h;
    
    router.add_ab_test("api_test", ab_config);
    
    // Test A/B routing with user ID
    std::unordered_map<std::string, std::string> headers = {{"X-User-ID", "user123"}};
    auto decision = router.route_request("/api/test", headers);
    
    EXPECT_TRUE(decision.matched);
    EXPECT_TRUE(decision.target_service == "old-api" || decision.target_service == "new-api");
    EXPECT_TRUE(decision.ab_test_variant == "control" || decision.ab_test_variant == "variant");
    
    // Test consistent assignment for same user
    auto decision2 = router.route_request("/api/test", headers);
    EXPECT_EQ(decision.target_service, decision2.target_service);
    EXPECT_EQ(decision.ab_test_variant, decision2.ab_test_variant);
}

TEST_F(ConfigServiceDiscoveryTest, DynamicRoutingJSONConfig) {
    auto& router = DynamicRouter::instance();
    
    std::string routes_json = R"({
        "api_route": {
            "pattern": "^/api/.*",
            "target_service": "api-service",
            "priority": 100,
            "enabled": true,
            "headers": {
                "X-Service": "api"
            }
        }
    })";
    
    router.load_routes_from_json(routes_json);
    
    std::unordered_map<std::string, std::string> headers;
    auto decision = router.route_request("/api/users", headers);
    
    EXPECT_TRUE(decision.matched);
    EXPECT_EQ(decision.target_service, "api-service");
    EXPECT_EQ(decision.additional_headers["X-Service"], "api");
}

TEST_F(ConfigServiceDiscoveryTest, FeatureFlagsBasicOperations) {
    auto& flags = FeatureFlagManager::instance();
    
    // Test creating flags
    FeatureFlag flag;
    flag.enabled = true;
    flag.description = "Test flag";
    flag.rollout_percentage = 50.0;
    
    flags.create_flag("test_flag", flag);
    
    // Test flag evaluation
    EXPECT_TRUE(flags.is_enabled("test_flag"));
    
    // Test with context
    FeatureFlagContext context;
    context.user_id = "user123";
    
    bool enabled = flags.is_enabled("test_flag", context);
    // Result depends on hash, but should be consistent
    bool enabled2 = flags.is_enabled("test_flag", context);
    EXPECT_EQ(enabled, enabled2);
    
    // Test flag update
    flag.enabled = false;
    flags.update_flag("test_flag", flag);
    EXPECT_FALSE(flags.is_enabled("test_flag"));
    
    // Test flag deletion
    flags.delete_flag("test_flag");
    EXPECT_FALSE(flags.is_enabled("test_flag"));
}

TEST_F(ConfigServiceDiscoveryTest, FeatureFlagsContextEvaluation) {
    auto& flags = FeatureFlagManager::instance();
    
    FeatureFlag flag;
    flag.enabled = true;
    flag.allowed_users = {"allowed_user"};
    flag.allowed_groups = {"allowed_group"};
    flag.rollout_percentage = 0.0; // No rollout, only explicit users/groups
    
    flags.create_flag("context_flag", flag);
    
    // Test allowed user
    FeatureFlagContext user_context;
    user_context.user_id = "allowed_user";
    EXPECT_TRUE(flags.is_enabled("context_flag", user_context));
    
    // Test disallowed user
    FeatureFlagContext other_user_context;
    other_user_context.user_id = "other_user";
    EXPECT_FALSE(flags.is_enabled("context_flag", other_user_context));
    
    // Test allowed group
    FeatureFlagContext group_context;
    group_context.user_id = "group_user";
    group_context.group_id = "allowed_group";
    EXPECT_TRUE(flags.is_enabled("context_flag", group_context));
}

TEST_F(ConfigServiceDiscoveryTest, FeatureFlagsJSONConfig) {
    auto& flags = FeatureFlagManager::instance();
    
    flags.load_from_file(test_flags_file_);
    
    // Test loaded flag
    EXPECT_TRUE(flags.is_enabled("test_flag"));
    
    // Test with allowed user
    FeatureFlagContext context;
    context.user_id = "test_user";
    EXPECT_TRUE(flags.is_enabled("test_flag", context));
    
    // Test export
    std::string exported = flags.export_to_json();
    EXPECT_FALSE(exported.empty());
    EXPECT_NE(exported.find("test_flag"), std::string::npos);
}

TEST_F(ConfigServiceDiscoveryTest, IntegrationTest) {
    auto& config = ConfigManager::instance();
    auto& registry = ServiceRegistry::instance();
    auto& router = DynamicRouter::instance();
    auto& flags = FeatureFlagManager::instance();
    
    // Load configurations
    config.load_from_file(test_config_file_);
    flags.load_from_file(test_flags_file_);
    
    // Register services
    registry.register_service("cpp-api", "http://localhost:8080", {"api"});
    registry.register_service("nodejs-api", "http://localhost:3000", {"api", "fallback"});
    
    // Set up routing
    RouteRule rule;
    rule.pattern = "^/api/.*";
    rule.target_service = "cpp-api";
    rule.priority = 100;
    rule.enabled = true;
    
    router.add_route("api_route", rule);
    
    // Test integrated request processing
    FeatureFlagContext context;
    context.user_id = "test_user";
    
    bool feature_enabled = flags.is_enabled("test_flag", context);
    EXPECT_TRUE(feature_enabled);
    
    std::unordered_map<std::string, std::string> headers = {{"X-User-ID", "test_user"}};
    auto routing_decision = router.route_request("/api/users", headers);
    
    EXPECT_TRUE(routing_decision.matched);
    EXPECT_EQ(routing_decision.target_service, "cpp-api");
    
    auto service_info = registry.get_service("cpp-api");
    EXPECT_TRUE(service_info.has_value());
    EXPECT_EQ(service_info->endpoint, "http://localhost:8080");
}

// Performance tests
TEST_F(ConfigServiceDiscoveryTest, PerformanceConfigAccess) {
    auto& config = ConfigManager::instance();
    config.load_from_file(test_config_file_);
    
    const int iterations = 100000;
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; ++i) {
        volatile int port = config.get<int>("server.port", 0);
        (void)port; // Suppress unused variable warning
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    
    double avg_ns = static_cast<double>(duration.count()) / iterations;
    std::cout << "Average config access time: " << avg_ns << " ns\n";
    
    // Should be very fast (sub-microsecond)
    EXPECT_LT(avg_ns, 1000.0); // Less than 1 microsecond
}

TEST_F(ConfigServiceDiscoveryTest, PerformanceRouting) {
    auto& router = DynamicRouter::instance();
    
    // Set up multiple routes
    for (int i = 0; i < 10; ++i) {
        RouteRule rule;
        rule.pattern = "^/api/v" + std::to_string(i) + "/.*";
        rule.target_service = "service-" + std::to_string(i);
        rule.priority = 100 - i;
        rule.enabled = true;
        
        router.add_route("route_" + std::to_string(i), rule);
    }
    
    const int iterations = 10000;
    std::unordered_map<std::string, std::string> headers;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; ++i) {
        std::string path = "/api/v" + std::to_string(i % 10) + "/test";
        auto decision = router.route_request(path, headers);
        EXPECT_TRUE(decision.matched);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    
    double avg_ns = static_cast<double>(duration.count()) / iterations;
    std::cout << "Average routing time: " << avg_ns << " ns\n";
    
    // Should be fast (sub-10 microseconds)
    EXPECT_LT(avg_ns, 10000.0); // Less than 10 microseconds
}

TEST_F(ConfigServiceDiscoveryTest, PerformanceFeatureFlags) {
    auto& flags = FeatureFlagManager::instance();
    flags.load_from_file(test_flags_file_);
    
    FeatureFlagContext context;
    context.user_id = "test_user";
    
    const int iterations = 100000;
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; ++i) {
        volatile bool enabled = flags.is_enabled("test_flag", context);
        (void)enabled; // Suppress unused variable warning
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    
    double avg_ns = static_cast<double>(duration.count()) / iterations;
    std::cout << "Average feature flag evaluation time: " << avg_ns << " ns\n";
    
    // Should be very fast (sub-microsecond)
    EXPECT_LT(avg_ns, 1000.0); // Less than 1 microsecond
}