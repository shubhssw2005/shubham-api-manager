#pragma once

#include <string>
#include <unordered_map>
#include <memory>
#include <functional>
#include <mutex>
#include <fstream>
#include <sstream>
#include <thread>
#include <atomic>
#include <chrono>
#ifdef __linux__
#include <sys/inotify.h>
#include <unistd.h>
#elif __APPLE__
#include <CoreServices/CoreServices.h>
#endif

namespace ultra {
namespace common {

class ConfigManager {
public:
    static ConfigManager& instance();
    
    // Load configuration from file
    bool load_from_file(const std::string& filename);
    
    // Get configuration values
    template<typename T>
    T get(const std::string& key, const T& default_value = T{}) const {
        std::lock_guard<std::mutex> lock(mutex_);
        
        auto it = config_map_.find(key);
        if (it == config_map_.end()) {
            return default_value;
        }
        
        return parse_value<T>(it->second);
    }
    
    // Set configuration values
    template<typename T>
    void set(const std::string& key, const T& value) {
        std::lock_guard<std::mutex> lock(mutex_);
        config_map_[key] = to_string(value);
        
        // Notify callbacks
        for (const auto& callback : change_callbacks_) {
            callback(key, to_string(value));
        }
    }
    
    // Check if key exists
    bool has_key(const std::string& key) const;
    
    // Get all keys in a section
    std::vector<std::string> get_section_keys(const std::string& section) const;
    
    // Watch for configuration changes
    using ChangeCallback = std::function<void(const std::string& key, const std::string& value)>;
    void watch_changes(ChangeCallback callback);
    
    // Reload configuration
    void reload();
    
    // Hot-reload functionality
    void start_file_watching();
    void stop_file_watching();
    bool is_watching() const { return watching_.load(); }
    
    // Feature flags support
    bool is_feature_enabled(const std::string& feature_name) const;
    void set_feature_flag(const std::string& feature_name, bool enabled);
    std::unordered_map<std::string, bool> get_all_feature_flags() const;

private:
    ConfigManager() = default;
    ~ConfigManager();
    
    template<typename T>
    T parse_value(const std::string& str) const {
        std::istringstream iss(str);
        T value;
        iss >> value;
        return value;
    }
    
    template<typename T>
    std::string to_string(const T& value) const {
        std::ostringstream oss;
        oss << value;
        return oss.str();
    }
    
    void parse_config_line(const std::string& line, std::string& current_section);
    void file_watcher_thread();
    
    mutable std::mutex mutex_;
    std::unordered_map<std::string, std::string> config_map_;
    std::unordered_map<std::string, bool> feature_flags_;
    std::string config_filename_;
    std::vector<ChangeCallback> change_callbacks_;
    
    // File watching
    std::atomic<bool> watching_{false};
    std::thread watcher_thread_;
#ifdef __linux__
    int inotify_fd_{-1};
    int watch_descriptor_{-1};
#elif __APPLE__
    FSEventStreamRef event_stream_{nullptr};
#endif
};

// Template specializations
template<>
inline std::string ConfigManager::parse_value<std::string>(const std::string& str) const {
    return str;
}

template<>
inline bool ConfigManager::parse_value<bool>(const std::string& str) const {
    std::string lower_str = str;
    std::transform(lower_str.begin(), lower_str.end(), lower_str.begin(), ::tolower);
    return lower_str == "true" || lower_str == "1" || lower_str == "yes" || lower_str == "on";
}

template<>
inline std::string ConfigManager::to_string<bool>(const bool& value) const {
    return value ? "true" : "false";
}

template<>
inline std::string ConfigManager::to_string<std::string>(const std::string& value) const {
    return value;
}

} // namespace common
} // namespace ultra