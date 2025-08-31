#include "common/service_registry.hpp"
#include "common/logger.hpp"
#include <curl/curl.h>
#include <algorithm>
#include <optional>

namespace ultra {
namespace common {

ServiceRegistry& ServiceRegistry::instance() {
    static ServiceRegistry instance;
    return instance;
}

ServiceRegistry::~ServiceRegistry() {
    stop_health_monitoring();
}

bool ServiceRegistry::register_service(const std::string& name, 
                                     const std::string& endpoint,
                                     const std::vector<std::string>& capabilities,
                                     const std::string& health_check_url) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    ServiceInfo service;
    service.name = name;
    service.endpoint = endpoint;
    service.capabilities = capabilities;
    service.health_check_url = health_check_url.empty() ? endpoint + "/health" : health_check_url;
    service.last_heartbeat = std::chrono::steady_clock::now();
    service.is_healthy = true;
    
    services_[name] = service;
    
    LOG_INFO("Registered service: {} at {}", name, endpoint);
    notify_service_change(name, true);
    
    return true;
}

bool ServiceRegistry::deregister_service(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = services_.find(name);
    if (it == services_.end()) {
        return false;
    }
    
    services_.erase(it);
    
    LOG_INFO("Deregistered service: {}", name);
    notify_service_change(name, false);
    
    return true;
}

std::vector<ServiceInfo> ServiceRegistry::discover_services(const std::string& capability) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::vector<ServiceInfo> result;
    
    for (const auto& [name, service] : services_) {
        if (!service.is_healthy) {
            continue;
        }
        
        if (capability.empty()) {
            result.push_back(service);
        } else {
            auto it = std::find(service.capabilities.begin(), service.capabilities.end(), capability);
            if (it != service.capabilities.end()) {
                result.push_back(service);
            }
        }
    }
    
    return result;
}

std::optional<ServiceInfo> ServiceRegistry::get_service(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = services_.find(name);
    if (it != services_.end() && it->second.is_healthy) {
        return it->second;
    }
    
    return std::nullopt;
}

void ServiceRegistry::heartbeat(const std::string& service_name) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = services_.find(service_name);
    if (it != services_.end()) {
        it->second.last_heartbeat = std::chrono::steady_clock::now();
        
        if (!it->second.is_healthy) {
            it->second.is_healthy = true;
            LOG_INFO("Service {} is now healthy", service_name);
            notify_service_change(service_name, true);
        }
    }
}

void ServiceRegistry::start_health_monitoring() {
    if (monitoring_.load()) {
        return;
    }
    
    monitoring_.store(true);
    health_monitor_thread_ = std::thread(&ServiceRegistry::health_monitor_thread, this);
    
    LOG_INFO("Started health monitoring");
}

void ServiceRegistry::stop_health_monitoring() {
    if (!monitoring_.load()) {
        return;
    }
    
    monitoring_.store(false);
    
    if (health_monitor_thread_.joinable()) {
        health_monitor_thread_.join();
    }
    
    LOG_INFO("Stopped health monitoring");
}

void ServiceRegistry::set_service_metadata(const std::string& service_name, 
                                         const std::string& key, 
                                         const std::string& value) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = services_.find(service_name);
    if (it != services_.end()) {
        it->second.metadata[key] = value;
    }
}

std::string ServiceRegistry::get_service_metadata(const std::string& service_name, 
                                                const std::string& key) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = services_.find(service_name);
    if (it != services_.end()) {
        auto meta_it = it->second.metadata.find(key);
        if (meta_it != it->second.metadata.end()) {
            return meta_it->second;
        }
    }
    
    return "";
}

void ServiceRegistry::watch_service_changes(ServiceChangeCallback callback) {
    std::lock_guard<std::mutex> lock(mutex_);
    change_callbacks_.push_back(callback);
}

void ServiceRegistry::health_monitor_thread() {
    while (monitoring_.load()) {
        std::vector<ServiceInfo> services_to_check;
        
        {
            std::lock_guard<std::mutex> lock(mutex_);
            for (const auto& [name, service] : services_) {
                services_to_check.push_back(service);
            }
        }
        
        auto now = std::chrono::steady_clock::now();
        
        for (const auto& service : services_to_check) {
            // Check heartbeat timeout
            auto time_since_heartbeat = now - service.last_heartbeat;
            bool heartbeat_expired = time_since_heartbeat > heartbeat_timeout_;
            
            // Perform health check if URL is available
            bool health_check_passed = true;
            if (!service.health_check_url.empty()) {
                health_check_passed = perform_health_check(service);
            }
            
            bool should_be_healthy = !heartbeat_expired && health_check_passed;
            
            {
                std::lock_guard<std::mutex> lock(mutex_);
                auto it = services_.find(service.name);
                if (it != services_.end() && it->second.is_healthy != should_be_healthy) {
                    it->second.is_healthy = should_be_healthy;
                    
                    if (should_be_healthy) {
                        LOG_INFO("Service {} is now healthy", service.name);
                    } else {
                        LOG_WARN("Service {} is now unhealthy", service.name);
                    }
                    
                    notify_service_change(service.name, should_be_healthy);
                }
            }
        }
        
        std::this_thread::sleep_for(health_check_interval_);
    }
}

bool ServiceRegistry::perform_health_check(const ServiceInfo& service) {
    CURL* curl = curl_easy_init();
    if (!curl) {
        return false;
    }
    
    curl_easy_setopt(curl, CURLOPT_URL, service.health_check_url.c_str());
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 5L);
    curl_easy_setopt(curl, CURLOPT_NOBODY, 1L); // HEAD request
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    
    CURLcode res = curl_easy_perform(curl);
    
    long response_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &response_code);
    
    curl_easy_cleanup(curl);
    
    return (res == CURLE_OK && response_code >= 200 && response_code < 300);
}

void ServiceRegistry::notify_service_change(const std::string& service_name, bool available) {
    for (const auto& callback : change_callbacks_) {
        try {
            callback(service_name, available);
        } catch (const std::exception& e) {
            LOG_ERROR("Error in service change callback: {}", e.what());
        }
    }
}

} // namespace common
} // namespace ultra