#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include <memory>
#include <functional>
#include <mutex>
#include <random>
#include <regex>

namespace ultra {
namespace common {

struct RouteRule {
    std::string pattern;
    std::regex compiled_pattern;
    std::string target_service;
    std::unordered_map<std::string, std::string> headers;
    int priority{0};
    double weight{1.0};
    bool enabled{true};
};

struct ABTestConfig {
    std::string test_name;
    std::string control_service;
    std::string variant_service;
    double traffic_split{0.5}; // 0.0 = all control, 1.0 = all variant
    std::string user_id_header{"X-User-ID"};
    std::vector<std::string> include_paths;
    std::vector<std::string> exclude_paths;
    bool enabled{true};
    std::chrono::steady_clock::time_point start_time;
    std::chrono::steady_clock::time_point end_time;
};

struct RoutingDecision {
    std::string target_service;
    std::unordered_map<std::string, std::string> additional_headers;
    std::string ab_test_variant; // "control" or "variant" if A/B test applied
    bool matched{false};
};

class DynamicRouter {
public:
    static DynamicRouter& instance();
    
    // Route management
    void add_route(const std::string& name, const RouteRule& rule);
    void remove_route(const std::string& name);
    void update_route(const std::string& name, const RouteRule& rule);
    std::vector<std::string> get_route_names() const;
    
    // A/B Testing
    void add_ab_test(const std::string& name, const ABTestConfig& config);
    void remove_ab_test(const std::string& name);
    void update_ab_test(const std::string& name, const ABTestConfig& config);
    std::vector<std::string> get_ab_test_names() const;
    
    // Routing decisions
    RoutingDecision route_request(const std::string& path, 
                                const std::unordered_map<std::string, std::string>& headers) const;
    
    // Configuration reload
    void reload_from_config();
    void load_routes_from_json(const std::string& json_config);
    void load_ab_tests_from_json(const std::string& json_config);
    
    // Statistics
    struct RoutingStats {
        std::unordered_map<std::string, uint64_t> route_hits;
        std::unordered_map<std::string, uint64_t> ab_test_assignments;
        uint64_t total_requests{0};
        uint64_t unmatched_requests{0};
    };
    
    RoutingStats get_stats() const;
    void reset_stats();
    
    // Hot reload callback
    using ConfigChangeCallback = std::function<void()>;
    void watch_config_changes(ConfigChangeCallback callback);

private:
    DynamicRouter() = default;
    
    bool matches_path_patterns(const std::string& path, 
                              const std::vector<std::string>& patterns) const;
    
    std::string determine_ab_variant(const ABTestConfig& config, 
                                   const std::string& user_id) const;
    
    uint64_t hash_user_id(const std::string& user_id) const;
    
    mutable std::mutex mutex_;
    std::unordered_map<std::string, RouteRule> routes_;
    std::unordered_map<std::string, ABTestConfig> ab_tests_;
    mutable RoutingStats stats_;
    std::vector<ConfigChangeCallback> config_callbacks_;
    mutable std::mt19937 rng_{std::random_device{}()};
};

} // namespace common
} // namespace ultra