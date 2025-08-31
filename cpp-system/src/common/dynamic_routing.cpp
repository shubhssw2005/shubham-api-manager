#include "common/dynamic_routing.hpp"
#include "common/logger.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <sstream>
#include <iomanip>

namespace ultra {
namespace common {

DynamicRouter& DynamicRouter::instance() {
    static DynamicRouter instance;
    return instance;
}

void DynamicRouter::add_route(const std::string& name, const RouteRule& rule) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    RouteRule compiled_rule = rule;
    try {
        compiled_rule.compiled_pattern = std::regex(rule.pattern);
    } catch (const std::exception& e) {
        LOG_ERROR("Invalid regex pattern for route {}: {}", name, e.what());
        return;
    }
    
    routes_[name] = compiled_rule;
    LOG_INFO("Added route: {} -> {}", name, rule.target_service);
}

void DynamicRouter::remove_route(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = routes_.find(name);
    if (it != routes_.end()) {
        routes_.erase(it);
        LOG_INFO("Removed route: {}", name);
    }
}

void DynamicRouter::update_route(const std::string& name, const RouteRule& rule) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = routes_.find(name);
    if (it != routes_.end()) {
        RouteRule compiled_rule = rule;
        try {
            compiled_rule.compiled_pattern = std::regex(rule.pattern);
        } catch (const std::exception& e) {
            LOG_ERROR("Invalid regex pattern for route {}: {}", name, e.what());
            return;
        }
        
        it->second = compiled_rule;
        LOG_INFO("Updated route: {} -> {}", name, rule.target_service);
    }
}

std::vector<std::string> DynamicRouter::get_route_names() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::vector<std::string> names;
    for (const auto& [name, rule] : routes_) {
        names.push_back(name);
    }
    
    return names;
}

void DynamicRouter::add_ab_test(const std::string& name, const ABTestConfig& config) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    ab_tests_[name] = config;
    LOG_INFO("Added A/B test: {} ({}% traffic to variant)", name, config.traffic_split * 100);
}

void DynamicRouter::remove_ab_test(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = ab_tests_.find(name);
    if (it != ab_tests_.end()) {
        ab_tests_.erase(it);
        LOG_INFO("Removed A/B test: {}", name);
    }
}

void DynamicRouter::update_ab_test(const std::string& name, const ABTestConfig& config) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = ab_tests_.find(name);
    if (it != ab_tests_.end()) {
        it->second = config;
        LOG_INFO("Updated A/B test: {} ({}% traffic to variant)", name, config.traffic_split * 100);
    }
}

std::vector<std::string> DynamicRouter::get_ab_test_names() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::vector<std::string> names;
    for (const auto& [name, config] : ab_tests_) {
        names.push_back(name);
    }
    
    return names;
}

RoutingDecision DynamicRouter::route_request(const std::string& path, 
                                           const std::unordered_map<std::string, std::string>& headers) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    RoutingDecision decision;
    stats_.total_requests++;
    
    // First, check A/B tests
    for (const auto& [test_name, config] : ab_tests_) {
        if (!config.enabled) {
            continue;
        }
        
        // Check if current time is within test period
        auto now = std::chrono::steady_clock::now();
        if (now < config.start_time || now > config.end_time) {
            continue;
        }
        
        // Check if path matches include/exclude patterns
        if (!config.include_paths.empty() && !matches_path_patterns(path, config.include_paths)) {
            continue;
        }
        
        if (!config.exclude_paths.empty() && matches_path_patterns(path, config.exclude_paths)) {
            continue;
        }
        
        // Get user ID for consistent assignment
        std::string user_id;
        auto user_header_it = headers.find(config.user_id_header);
        if (user_header_it != headers.end()) {
            user_id = user_header_it->second;
        }
        
        if (!user_id.empty()) {
            std::string variant = determine_ab_variant(config, user_id);
            
            decision.target_service = (variant == "variant") ? config.variant_service : config.control_service;
            decision.ab_test_variant = variant;
            decision.matched = true;
            
            stats_.ab_test_assignments[test_name + "_" + variant]++;
            
            LOG_DEBUG("A/B test {} assigned user {} to variant {}", test_name, user_id, variant);
            return decision;
        }
    }
    
    // If no A/B test matched, use regular routing rules
    std::vector<std::pair<std::string, RouteRule>> sorted_routes;
    for (const auto& [name, rule] : routes_) {
        if (rule.enabled) {
            sorted_routes.emplace_back(name, rule);
        }
    }
    
    // Sort by priority (higher priority first)
    std::sort(sorted_routes.begin(), sorted_routes.end(),
              [](const auto& a, const auto& b) {
                  return a.second.priority > b.second.priority;
              });
    
    // Find matching route
    for (const auto& [name, rule] : sorted_routes) {
        if (std::regex_match(path, rule.compiled_pattern)) {
            decision.target_service = rule.target_service;
            decision.additional_headers = rule.headers;
            decision.matched = true;
            
            stats_.route_hits[name]++;
            
            LOG_DEBUG("Route {} matched path {}", name, path);
            return decision;
        }
    }
    
    stats_.unmatched_requests++;
    LOG_DEBUG("No route matched path: {}", path);
    
    return decision;
}

void DynamicRouter::reload_from_config() {
    // Trigger config reload callbacks
    for (const auto& callback : config_callbacks_) {
        try {
            callback();
        } catch (const std::exception& e) {
            LOG_ERROR("Error in config reload callback: {}", e.what());
        }
    }
}

void DynamicRouter::load_routes_from_json(const std::string& json_config) {
    try {
        auto json = nlohmann::json::parse(json_config);
        
        std::lock_guard<std::mutex> lock(mutex_);
        routes_.clear();
        
        for (const auto& [name, route_json] : json.items()) {
            RouteRule rule;
            rule.pattern = route_json["pattern"];
            rule.target_service = route_json["target_service"];
            rule.priority = route_json.value("priority", 0);
            rule.weight = route_json.value("weight", 1.0);
            rule.enabled = route_json.value("enabled", true);
            
            if (route_json.contains("headers")) {
                for (const auto& [key, value] : route_json["headers"].items()) {
                    rule.headers[key] = value;
                }
            }
            
            try {
                rule.compiled_pattern = std::regex(rule.pattern);
                routes_[name] = rule;
            } catch (const std::exception& e) {
                LOG_ERROR("Invalid regex pattern for route {}: {}", name, e.what());
            }
        }
        
        LOG_INFO("Loaded {} routes from JSON config", routes_.size());
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to parse routes JSON config: {}", e.what());
    }
}

void DynamicRouter::load_ab_tests_from_json(const std::string& json_config) {
    try {
        auto json = nlohmann::json::parse(json_config);
        
        std::lock_guard<std::mutex> lock(mutex_);
        ab_tests_.clear();
        
        for (const auto& [name, test_json] : json.items()) {
            ABTestConfig config;
            config.test_name = name;
            config.control_service = test_json["control_service"];
            config.variant_service = test_json["variant_service"];
            config.traffic_split = test_json.value("traffic_split", 0.5);
            config.user_id_header = test_json.value("user_id_header", "X-User-ID");
            config.enabled = test_json.value("enabled", true);
            
            if (test_json.contains("include_paths")) {
                config.include_paths = test_json["include_paths"];
            }
            
            if (test_json.contains("exclude_paths")) {
                config.exclude_paths = test_json["exclude_paths"];
            }
            
            // Parse timestamps if provided
            if (test_json.contains("start_time")) {
                // Assume ISO 8601 format or epoch seconds
                config.start_time = std::chrono::steady_clock::now(); // Simplified
            }
            
            if (test_json.contains("end_time")) {
                // Assume ISO 8601 format or epoch seconds  
                config.end_time = std::chrono::steady_clock::now() + std::chrono::hours(24 * 30); // Simplified
            }
            
            ab_tests_[name] = config;
        }
        
        LOG_INFO("Loaded {} A/B tests from JSON config", ab_tests_.size());
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to parse A/B tests JSON config: {}", e.what());
    }
}

DynamicRouter::RoutingStats DynamicRouter::get_stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
}

void DynamicRouter::reset_stats() {
    std::lock_guard<std::mutex> lock(mutex_);
    stats_ = RoutingStats{};
}

void DynamicRouter::watch_config_changes(ConfigChangeCallback callback) {
    std::lock_guard<std::mutex> lock(mutex_);
    config_callbacks_.push_back(callback);
}

bool DynamicRouter::matches_path_patterns(const std::string& path, 
                                        const std::vector<std::string>& patterns) const {
    for (const auto& pattern : patterns) {
        try {
            std::regex regex_pattern(pattern);
            if (std::regex_match(path, regex_pattern)) {
                return true;
            }
        } catch (const std::exception& e) {
            LOG_ERROR("Invalid regex pattern: {}", pattern);
        }
    }
    
    return false;
}

std::string DynamicRouter::determine_ab_variant(const ABTestConfig& config, 
                                              const std::string& user_id) const {
    uint64_t hash = hash_user_id(config.test_name + user_id);
    double normalized = static_cast<double>(hash % 10000) / 10000.0;
    
    return (normalized < config.traffic_split) ? "variant" : "control";
}

uint64_t DynamicRouter::hash_user_id(const std::string& user_id) const {
    std::hash<std::string> hasher;
    return hasher(user_id);
}

} // namespace common
} // namespace ultra