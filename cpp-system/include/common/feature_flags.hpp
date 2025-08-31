#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include <memory>
#include <functional>
#include <mutex>
#include <thread>
#include <atomic>
#include <chrono>
#include <optional>

namespace ultra {
namespace common {

struct FeatureFlag {
    std::string name;
    bool enabled{false};
    std::string description;
    std::vector<std::string> allowed_users;
    std::vector<std::string> allowed_groups;
    double rollout_percentage{0.0}; // 0.0 to 100.0
    std::chrono::steady_clock::time_point created_at;
    std::chrono::steady_clock::time_point updated_at;
    std::unordered_map<std::string, std::string> metadata;
};

struct FeatureFlagContext {
    std::string user_id;
    std::string group_id;
    std::unordered_map<std::string, std::string> attributes;
};

class FeatureFlagManager {
public:
    static FeatureFlagManager& instance();
    
    // Flag management
    void create_flag(const std::string& name, const FeatureFlag& flag);
    void update_flag(const std::string& name, const FeatureFlag& flag);
    void delete_flag(const std::string& name);
    
    // Flag evaluation
    bool is_enabled(const std::string& flag_name) const;
    bool is_enabled(const std::string& flag_name, const FeatureFlagContext& context) const;
    
    // Bulk operations
    std::unordered_map<std::string, bool> evaluate_all_flags(const FeatureFlagContext& context) const;
    std::vector<std::string> get_enabled_flags(const FeatureFlagContext& context) const;
    
    // Configuration
    void load_from_json(const std::string& json_config);
    void load_from_file(const std::string& filename);
    std::string export_to_json() const;
    
    // Real-time updates
    void start_real_time_updates(const std::string& config_source);
    void stop_real_time_updates();
    
    // Callbacks
    using FlagChangeCallback = std::function<void(const std::string& flag_name, bool old_value, bool new_value)>;
    void watch_flag_changes(FlagChangeCallback callback);
    
    // Statistics
    struct FlagStats {
        std::unordered_map<std::string, uint64_t> evaluation_counts;
        std::unordered_map<std::string, uint64_t> enabled_counts;
        std::unordered_map<std::string, uint64_t> disabled_counts;
        uint64_t total_evaluations{0};
    };
    
    FlagStats get_stats() const;
    void reset_stats();
    
    // Administrative
    std::vector<std::string> get_all_flag_names() const;
    std::optional<FeatureFlag> get_flag_info(const std::string& name) const;

private:
    FeatureFlagManager() = default;
    ~FeatureFlagManager();
    
    bool evaluate_flag_for_context(const FeatureFlag& flag, const FeatureFlagContext& context) const;
    bool is_user_in_rollout(const std::string& flag_name, const std::string& user_id, double percentage) const;
    uint64_t hash_string(const std::string& str) const;
    void real_time_update_thread();
    void notify_flag_change(const std::string& flag_name, bool old_value, bool new_value);
    
    mutable std::mutex mutex_;
    std::unordered_map<std::string, FeatureFlag> flags_;
    mutable FlagStats stats_;
    std::vector<FlagChangeCallback> change_callbacks_;
    
    // Real-time updates
    std::atomic<bool> updating_{false};
    std::thread update_thread_;
    std::string config_source_;
    std::chrono::seconds update_interval_{5};
};

} // namespace common
} // namespace ultra