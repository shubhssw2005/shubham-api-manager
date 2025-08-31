#include "common/feature_flags.hpp"
#include "common/logger.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <algorithm>
#include <sstream>

namespace ultra {
namespace common {

FeatureFlagManager& FeatureFlagManager::instance() {
    static FeatureFlagManager instance;
    return instance;
}

FeatureFlagManager::~FeatureFlagManager() {
    stop_real_time_updates();
}

void FeatureFlagManager::create_flag(const std::string& name, const FeatureFlag& flag) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    FeatureFlag new_flag = flag;
    new_flag.name = name;
    new_flag.created_at = std::chrono::steady_clock::now();
    new_flag.updated_at = new_flag.created_at;
    
    flags_[name] = new_flag;
    
    LOG_INFO("Created feature flag: {} (enabled: {})", name, flag.enabled);
}

void FeatureFlagManager::update_flag(const std::string& name, const FeatureFlag& flag) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = flags_.find(name);
    if (it != flags_.end()) {
        bool old_value = it->second.enabled;
        
        FeatureFlag updated_flag = flag;
        updated_flag.name = name;
        updated_flag.created_at = it->second.created_at;
        updated_flag.updated_at = std::chrono::steady_clock::now();
        
        it->second = updated_flag;
        
        LOG_INFO("Updated feature flag: {} (enabled: {})", name, flag.enabled);
        
        if (old_value != flag.enabled) {
            notify_flag_change(name, old_value, flag.enabled);
        }
    }
}

void FeatureFlagManager::delete_flag(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = flags_.find(name);
    if (it != flags_.end()) {
        bool old_value = it->second.enabled;
        flags_.erase(it);
        
        LOG_INFO("Deleted feature flag: {}", name);
        notify_flag_change(name, old_value, false);
    }
}

bool FeatureFlagManager::is_enabled(const std::string& flag_name) const {
    FeatureFlagContext empty_context;
    return is_enabled(flag_name, empty_context);
}

bool FeatureFlagManager::is_enabled(const std::string& flag_name, const FeatureFlagContext& context) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    stats_.total_evaluations++;
    stats_.evaluation_counts[flag_name]++;
    
    auto it = flags_.find(flag_name);
    if (it == flags_.end()) {
        return false;
    }
    
    bool result = evaluate_flag_for_context(it->second, context);
    
    if (result) {
        stats_.enabled_counts[flag_name]++;
    } else {
        stats_.disabled_counts[flag_name]++;
    }
    
    return result;
}

std::unordered_map<std::string, bool> FeatureFlagManager::evaluate_all_flags(const FeatureFlagContext& context) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::unordered_map<std::string, bool> results;
    
    for (const auto& [name, flag] : flags_) {
        results[name] = evaluate_flag_for_context(flag, context);
    }
    
    return results;
}

std::vector<std::string> FeatureFlagManager::get_enabled_flags(const FeatureFlagContext& context) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::vector<std::string> enabled_flags;
    
    for (const auto& [name, flag] : flags_) {
        if (evaluate_flag_for_context(flag, context)) {
            enabled_flags.push_back(name);
        }
    }
    
    return enabled_flags;
}

void FeatureFlagManager::load_from_json(const std::string& json_config) {
    try {
        auto json = nlohmann::json::parse(json_config);
        
        std::lock_guard<std::mutex> lock(mutex_);
        
        for (const auto& [name, flag_json] : json.items()) {
            FeatureFlag flag;
            flag.name = name;
            flag.enabled = flag_json.value("enabled", false);
            flag.description = flag_json.value("description", "");
            flag.rollout_percentage = flag_json.value("rollout_percentage", 0.0);
            
            if (flag_json.contains("allowed_users")) {
                flag.allowed_users = flag_json["allowed_users"];
            }
            
            if (flag_json.contains("allowed_groups")) {
                flag.allowed_groups = flag_json["allowed_groups"];
            }
            
            if (flag_json.contains("metadata")) {
                for (const auto& [key, value] : flag_json["metadata"].items()) {
                    flag.metadata[key] = value;
                }
            }
            
            flag.created_at = std::chrono::steady_clock::now();
            flag.updated_at = flag.created_at;
            
            flags_[name] = flag;
        }
        
        LOG_INFO("Loaded {} feature flags from JSON config", flags_.size());
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to parse feature flags JSON config: {}", e.what());
    }
}

void FeatureFlagManager::load_from_file(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        LOG_ERROR("Failed to open feature flags file: {}", filename);
        return;
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    
    load_from_json(buffer.str());
}

std::string FeatureFlagManager::export_to_json() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    nlohmann::json json;
    
    for (const auto& [name, flag] : flags_) {
        nlohmann::json flag_json;
        flag_json["enabled"] = flag.enabled;
        flag_json["description"] = flag.description;
        flag_json["rollout_percentage"] = flag.rollout_percentage;
        flag_json["allowed_users"] = flag.allowed_users;
        flag_json["allowed_groups"] = flag.allowed_groups;
        flag_json["metadata"] = flag.metadata;
        
        json[name] = flag_json;
    }
    
    return json.dump(2);
}

void FeatureFlagManager::start_real_time_updates(const std::string& config_source) {
    if (updating_.load()) {
        return;
    }
    
    config_source_ = config_source;
    updating_.store(true);
    update_thread_ = std::thread(&FeatureFlagManager::real_time_update_thread, this);
    
    LOG_INFO("Started real-time updates for feature flags from: {}", config_source);
}

void FeatureFlagManager::stop_real_time_updates() {
    if (!updating_.load()) {
        return;
    }
    
    updating_.store(false);
    
    if (update_thread_.joinable()) {
        update_thread_.join();
    }
    
    LOG_INFO("Stopped real-time updates for feature flags");
}

void FeatureFlagManager::watch_flag_changes(FlagChangeCallback callback) {
    std::lock_guard<std::mutex> lock(mutex_);
    change_callbacks_.push_back(callback);
}

FeatureFlagManager::FlagStats FeatureFlagManager::get_stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
}

void FeatureFlagManager::reset_stats() {
    std::lock_guard<std::mutex> lock(mutex_);
    stats_ = FlagStats{};
}

std::vector<std::string> FeatureFlagManager::get_all_flag_names() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::vector<std::string> names;
    for (const auto& [name, flag] : flags_) {
        names.push_back(name);
    }
    
    return names;
}

std::optional<FeatureFlag> FeatureFlagManager::get_flag_info(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = flags_.find(name);
    if (it != flags_.end()) {
        return it->second;
    }
    
    return std::nullopt;
}

bool FeatureFlagManager::evaluate_flag_for_context(const FeatureFlag& flag, const FeatureFlagContext& context) const {
    if (!flag.enabled) {
        return false;
    }
    
    // Check if user is explicitly allowed
    if (!context.user_id.empty() && !flag.allowed_users.empty()) {
        auto it = std::find(flag.allowed_users.begin(), flag.allowed_users.end(), context.user_id);
        if (it != flag.allowed_users.end()) {
            return true;
        }
    }
    
    // Check if user's group is explicitly allowed
    if (!context.group_id.empty() && !flag.allowed_groups.empty()) {
        auto it = std::find(flag.allowed_groups.begin(), flag.allowed_groups.end(), context.group_id);
        if (it != flag.allowed_groups.end()) {
            return true;
        }
    }
    
    // Check rollout percentage
    if (flag.rollout_percentage > 0.0 && !context.user_id.empty()) {
        return is_user_in_rollout(flag.name, context.user_id, flag.rollout_percentage);
    }
    
    // If no specific rules apply, return the flag's enabled state
    return flag.enabled;
}

bool FeatureFlagManager::is_user_in_rollout(const std::string& flag_name, const std::string& user_id, double percentage) const {
    uint64_t hash = hash_string(flag_name + user_id);
    double normalized = static_cast<double>(hash % 10000) / 100.0; // 0.0 to 99.99
    
    return normalized < percentage;
}

uint64_t FeatureFlagManager::hash_string(const std::string& str) const {
    std::hash<std::string> hasher;
    return hasher(str);
}

void FeatureFlagManager::real_time_update_thread() {
    while (updating_.load()) {
        try {
            // In a real implementation, this would poll a remote config service
            // For now, we'll just reload from file if it's a file path
            if (!config_source_.empty() && config_source_.find("http") != 0) {
                load_from_file(config_source_);
            }
        } catch (const std::exception& e) {
            LOG_ERROR("Error during real-time update: {}", e.what());
        }
        
        std::this_thread::sleep_for(update_interval_);
    }
}

void FeatureFlagManager::notify_flag_change(const std::string& flag_name, bool old_value, bool new_value) {
    for (const auto& callback : change_callbacks_) {
        try {
            callback(flag_name, old_value, new_value);
        } catch (const std::exception& e) {
            LOG_ERROR("Error in flag change callback: {}", e.what());
        }
    }
}

} // namespace common
} // namespace ultra