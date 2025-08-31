#include "common/config_manager.hpp"
#include "common/logger.hpp"
#include <algorithm>
#include <cctype>
#include <filesystem>

#ifdef __linux__
#include <sys/inotify.h>
#include <unistd.h>
#include <poll.h>
#elif __APPLE__
#include <CoreServices/CoreServices.h>
#endif

namespace ultra {
namespace common {

ConfigManager& ConfigManager::instance() {
    static ConfigManager instance;
    return instance;
}

ConfigManager::~ConfigManager() {
    stop_file_watching();
}

bool ConfigManager::load_from_file(const std::string& filename) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::ifstream file(filename);
    if (!file.is_open()) {
        LOG_ERROR("Failed to open config file: {}", filename);
        return false;
    }
    
    config_filename_ = filename;
    config_map_.clear();
    
    std::string line;
    std::string current_section;
    
    while (std::getline(file, line)) {
        parse_config_line(line, current_section);
    }
    
    LOG_INFO("Loaded configuration from: {}", filename);
    return true;
}

bool ConfigManager::has_key(const std::string& key) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return config_map_.find(key) != config_map_.end();
}

std::vector<std::string> ConfigManager::get_section_keys(const std::string& section) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::vector<std::string> keys;
    std::string section_prefix = section + ".";
    
    for (const auto& [key, value] : config_map_) {
        if (key.substr(0, section_prefix.length()) == section_prefix) {
            keys.push_back(key.substr(section_prefix.length()));
        }
    }
    
    return keys;
}

void ConfigManager::watch_changes(ChangeCallback callback) {
    std::lock_guard<std::mutex> lock(mutex_);
    change_callbacks_.push_back(callback);
}

void ConfigManager::reload() {
    if (!config_filename_.empty()) {
        load_from_file(config_filename_);
    }
}

void ConfigManager::start_file_watching() {
    if (watching_.load() || config_filename_.empty()) {
        return;
    }
    
#ifdef __linux__
    inotify_fd_ = inotify_init1(IN_NONBLOCK);
    if (inotify_fd_ == -1) {
        LOG_ERROR("Failed to initialize inotify");
        return;
    }
    
    watch_descriptor_ = inotify_add_watch(inotify_fd_, config_filename_.c_str(), 
                                         IN_MODIFY | IN_MOVE_SELF | IN_DELETE_SELF);
    if (watch_descriptor_ == -1) {
        LOG_ERROR("Failed to add watch for config file: {}", config_filename_);
        close(inotify_fd_);
        inotify_fd_ = -1;
        return;
    }
#elif __APPLE__
    // macOS FSEvents implementation would go here
    // For now, we'll use polling as a fallback
    LOG_WARN("File watching on macOS not fully implemented, using polling fallback");
#endif
    
    watching_.store(true);
    watcher_thread_ = std::thread(&ConfigManager::file_watcher_thread, this);
    
    LOG_INFO("Started file watching for config: {}", config_filename_);
}

void ConfigManager::stop_file_watching() {
    if (!watching_.load()) {
        return;
    }
    
    watching_.store(false);
    
    if (watcher_thread_.joinable()) {
        watcher_thread_.join();
    }
    
#ifdef __linux__
    if (watch_descriptor_ != -1) {
        inotify_rm_watch(inotify_fd_, watch_descriptor_);
        watch_descriptor_ = -1;
    }
    
    if (inotify_fd_ != -1) {
        close(inotify_fd_);
        inotify_fd_ = -1;
    }
#elif __APPLE__
    if (event_stream_) {
        FSEventStreamStop(event_stream_);
        FSEventStreamInvalidate(event_stream_);
        FSEventStreamRelease(event_stream_);
        event_stream_ = nullptr;
    }
#endif
    
    LOG_INFO("Stopped file watching for config");
}

bool ConfigManager::is_feature_enabled(const std::string& feature_name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = feature_flags_.find(feature_name);
    if (it != feature_flags_.end()) {
        return it->second;
    }
    
    // Fallback to config value
    std::string config_key = "features." + feature_name;
    auto config_it = config_map_.find(config_key);
    if (config_it != config_map_.end()) {
        return parse_value<bool>(config_it->second);
    }
    
    return false;
}

void ConfigManager::set_feature_flag(const std::string& feature_name, bool enabled) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    bool old_value = feature_flags_[feature_name];
    feature_flags_[feature_name] = enabled;
    
    // Also update config map
    std::string config_key = "features." + feature_name;
    config_map_[config_key] = enabled ? "true" : "false";
    
    // Notify callbacks if value changed
    if (old_value != enabled) {
        for (const auto& callback : change_callbacks_) {
            callback(config_key, enabled ? "true" : "false");
        }
    }
}

std::unordered_map<std::string, bool> ConfigManager::get_all_feature_flags() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::unordered_map<std::string, bool> all_flags = feature_flags_;
    
    // Also include flags from config
    for (const auto& [key, value] : config_map_) {
        if (key.substr(0, 9) == "features.") {
            std::string flag_name = key.substr(9);
            if (all_flags.find(flag_name) == all_flags.end()) {
                all_flags[flag_name] = parse_value<bool>(value);
            }
        }
    }
    
    return all_flags;
}

void ConfigManager::file_watcher_thread() {
#ifdef __linux__
    constexpr size_t EVENT_SIZE = sizeof(struct inotify_event);
    constexpr size_t BUF_LEN = 1024 * (EVENT_SIZE + 16);
    char buffer[BUF_LEN];
    
    while (watching_.load()) {
        struct pollfd pfd = {inotify_fd_, POLLIN, 0};
        int poll_result = poll(&pfd, 1, 1000); // 1 second timeout
        
        if (poll_result > 0 && (pfd.revents & POLLIN)) {
            ssize_t length = read(inotify_fd_, buffer, BUF_LEN);
            
            if (length > 0) {
                size_t i = 0;
                while (i < static_cast<size_t>(length)) {
                    struct inotify_event* event = reinterpret_cast<struct inotify_event*>(&buffer[i]);
                    
                    if (event->mask & (IN_MODIFY | IN_MOVE_SELF)) {
                        LOG_INFO("Config file modified, reloading...");
                        reload();
                    } else if (event->mask & IN_DELETE_SELF) {
                        LOG_WARN("Config file deleted, stopping file watching");
                        watching_.store(false);
                        break;
                    }
                    
                    i += EVENT_SIZE + event->len;
                }
            }
        }
    }
#else
    // Fallback polling implementation for non-Linux systems
    std::filesystem::file_time_type last_write_time;
    
    try {
        last_write_time = std::filesystem::last_write_time(config_filename_);
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to get file modification time: {}", e.what());
        return;
    }
    
    while (watching_.load()) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        
        try {
            auto current_write_time = std::filesystem::last_write_time(config_filename_);
            if (current_write_time != last_write_time) {
                LOG_INFO("Config file modified, reloading...");
                reload();
                last_write_time = current_write_time;
            }
        } catch (const std::exception& e) {
            LOG_WARN("Config file may have been deleted: {}", e.what());
            watching_.store(false);
            break;
        }
    }
#endif
}

void ConfigManager::parse_config_line(const std::string& line, std::string& current_section) {
    // Remove leading/trailing whitespace
    std::string trimmed = line;
    trimmed.erase(0, trimmed.find_first_not_of(" \t"));
    trimmed.erase(trimmed.find_last_not_of(" \t") + 1);
    
    // Skip empty lines and comments
    if (trimmed.empty() || trimmed[0] == '#') {
        return;
    }
    
    // Check for section header [section]
    if (trimmed[0] == '[' && trimmed.back() == ']') {
        current_section = trimmed.substr(1, trimmed.length() - 2);
        return;
    }
    
    // Parse key=value pairs
    size_t eq_pos = trimmed.find('=');
    if (eq_pos != std::string::npos) {
        std::string key = trimmed.substr(0, eq_pos);
        std::string value = trimmed.substr(eq_pos + 1);
        
        // Trim key and value
        key.erase(0, key.find_first_not_of(" \t"));
        key.erase(key.find_last_not_of(" \t") + 1);
        value.erase(0, value.find_first_not_of(" \t"));
        value.erase(value.find_last_not_of(" \t") + 1);
        
        // Add section prefix if we're in a section
        if (!current_section.empty()) {
            key = current_section + "." + key;
        }
        
        config_map_[key] = value;
        
        // Notify callbacks
        for (const auto& callback : change_callbacks_) {
            callback(key, value);
        }
    }
}

} // namespace common
} // namespace ultra