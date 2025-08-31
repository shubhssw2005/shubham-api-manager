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
#include <future>
#include <optional>

namespace ultra {
namespace common {

struct ServiceInfo {
    std::string name;
    std::string endpoint;
    std::vector<std::string> capabilities;
    std::string health_check_url;
    std::chrono::steady_clock::time_point last_heartbeat;
    bool is_healthy{true};
    std::unordered_map<std::string, std::string> metadata;
};

class ServiceRegistry {
public:
    static ServiceRegistry& instance();
    
    // Service registration
    bool register_service(const std::string& name, 
                         const std::string& endpoint,
                         const std::vector<std::string>& capabilities = {},
                         const std::string& health_check_url = "");
    
    bool deregister_service(const std::string& name);
    
    // Service discovery
    std::vector<ServiceInfo> discover_services(const std::string& capability = "") const;
    std::optional<ServiceInfo> get_service(const std::string& name) const;
    
    // Health management
    void heartbeat(const std::string& service_name);
    void start_health_monitoring();
    void stop_health_monitoring();
    
    // Service metadata
    void set_service_metadata(const std::string& service_name, 
                             const std::string& key, 
                             const std::string& value);
    
    std::string get_service_metadata(const std::string& service_name, 
                                   const std::string& key) const;
    
    // Callbacks for service changes
    using ServiceChangeCallback = std::function<void(const std::string& service_name, bool available)>;
    void watch_service_changes(ServiceChangeCallback callback);
    
    // Configuration
    void set_heartbeat_timeout(std::chrono::seconds timeout) { heartbeat_timeout_ = timeout; }
    void set_health_check_interval(std::chrono::seconds interval) { health_check_interval_ = interval; }

private:
    ServiceRegistry() = default;
    ~ServiceRegistry();
    
    void health_monitor_thread();
    bool perform_health_check(const ServiceInfo& service);
    void notify_service_change(const std::string& service_name, bool available);
    
    mutable std::mutex mutex_;
    std::unordered_map<std::string, ServiceInfo> services_;
    std::vector<ServiceChangeCallback> change_callbacks_;
    
    // Health monitoring
    std::atomic<bool> monitoring_{false};
    std::thread health_monitor_thread_;
    std::chrono::seconds heartbeat_timeout_{30};
    std::chrono::seconds health_check_interval_{10};
};

} // namespace common
} // namespace ultra