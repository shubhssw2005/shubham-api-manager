#include "common/health_check.hpp"
#include "common/logger.hpp"
#include <thread>
#include <shared_mutex>
#include <sstream>
#include <fstream>
#include <sys/statvfs.h>
#include <sys/sysinfo.h>
#include <unistd.h>
#include <ifaddrs.h>
#include <netinet/in.h>
#include <arpa/inet.h>

namespace ultra_cpp {
namespace common {

HealthCheckManager::HealthCheckManager() 
    : startup_time_(std::chrono::system_clock::now()) {
    
    // Register built-in health checks
    register_component("memory", [this]() { return check_memory_usage(); });
    register_component("cpu", [this]() { return check_cpu_usage(); });
    register_component("disk", [this]() { return check_disk_space(); });
    register_component("network", [this]() { return check_network_connectivity(); });
    register_component("dpdk", [this]() { return check_dpdk_status(); });
    register_component("gpu", [this]() { return check_gpu_status(); });
    
    // Set default critical components
    set_critical_components({"memory", "disk", "network"});
    
    collect_system_info();
}

HealthCheckManager::~HealthCheckManager() {
    stop_background_checks();
}

void HealthCheckManager::register_component(const std::string& name, HealthCheckFunction check_func) {
    std::unique_lock<std::shared_mutex> lock(components_mutex_);
    
    auto component = std::make_unique<ComponentInfo>();
    component->name = name;
    component->check_func = check_func;
    component->last_result = ComponentHealth(name);
    
    components_[name] = std::move(component);
    
    Logger::info("Registered health check component: " + name);
}

void HealthCheckManager::unregister_component(const std::string& name) {
    std::unique_lock<std::shared_mutex> lock(components_mutex_);
    components_.erase(name);
    Logger::info("Unregistered health check component: " + name);
}

void HealthCheckManager::update_component_health(const std::string& name, HealthStatus status, 
                                               const std::string& message) {
    std::shared_lock<std::shared_mutex> lock(components_mutex_);
    
    auto it = components_.find(name);
    if (it != components_.end()) {
        it->second->last_result.status = status;
        it->second->last_result.message = message;
        it->second->last_result.last_check = std::chrono::system_clock::now();
    }
}

SystemHealth HealthCheckManager::get_system_health() {
    std::shared_lock<std::shared_mutex> lock(components_mutex_);
    
    SystemHealth health;
    health.version = "1.0.0";
    health.startup_time = startup_time_;
    health.last_update = std::chrono::system_clock::now();
    
    // Collect all component health
    for (const auto& [name, component] : components_) {
        try {
            ComponentHealth result = component->check_func();
            component->last_result = result;
            health.components.push_back(result);
        } catch (const std::exception& e) {
            ComponentHealth error_result(name);
            error_result.status = HealthStatus::UNHEALTHY;
            error_result.message = "Health check failed: " + std::string(e.what());
            health.components.push_back(error_result);
        }
    }
    
    health.overall_status = determine_overall_status();
    health.system_info = system_health_.system_info;
    
    return health;
}

ComponentHealth HealthCheckManager::get_component_health(const std::string& name) {
    std::shared_lock<std::shared_mutex> lock(components_mutex_);
    
    auto it = components_.find(name);
    if (it != components_.end()) {
        try {
            return it->second->check_func();
        } catch (const std::exception& e) {
            ComponentHealth error_result(name);
            error_result.status = HealthStatus::UNHEALTHY;
            error_result.message = "Health check failed: " + std::string(e.what());
            return error_result;
        }
    }
    
    ComponentHealth not_found(name);
    not_found.status = HealthStatus::UNKNOWN;
    not_found.message = "Component not registered";
    return not_found;
}

bool HealthCheckManager::is_alive() {
    // Basic liveness - just check if the process is running
    return true;
}

bool HealthCheckManager::is_ready() {
    std::shared_lock<std::shared_mutex> lock(components_mutex_);
    
    // Check all critical components
    for (const std::string& critical_name : critical_components_) {
        auto it = components_.find(critical_name);
        if (it != components_.end()) {
            try {
                ComponentHealth health = it->second->check_func();
                if (health.status == HealthStatus::UNHEALTHY) {
                    return false;
                }
            } catch (...) {
                return false;
            }
        }
    }
    
    return startup_complete_.load();
}

bool HealthCheckManager::is_startup_complete() {
    // Check if startup sequence is complete
    auto now = std::chrono::system_clock::now();
    auto startup_duration = std::chrono::duration_cast<std::chrono::seconds>(now - startup_time_);
    
    // Consider startup complete after 30 seconds or when explicitly set
    if (startup_duration.count() > 30) {
        startup_complete_.store(true);
    }
    
    return startup_complete_.load();
}

std::string HealthCheckManager::to_json(const SystemHealth& health) {
    std::ostringstream json;
    json << "{\n";
    json << "  \"status\": \"";
    
    switch (health.overall_status) {
        case HealthStatus::HEALTHY: json << "healthy"; break;
        case HealthStatus::DEGRADED: json << "degraded"; break;
        case HealthStatus::UNHEALTHY: json << "unhealthy"; break;
        case HealthStatus::UNKNOWN: json << "unknown"; break;
    }
    
    json << "\",\n";
    json << "  \"version\": \"" << health.version << "\",\n";
    
    auto startup_time_t = std::chrono::system_clock::to_time_t(health.startup_time);
    auto last_update_t = std::chrono::system_clock::to_time_t(health.last_update);
    
    json << "  \"startup_time\": \"" << std::ctime(&startup_time_t) << "\",\n";
    json << "  \"last_update\": \"" << std::ctime(&last_update_t) << "\",\n";
    
    json << "  \"components\": [\n";
    for (size_t i = 0; i < health.components.size(); ++i) {
        const auto& comp = health.components[i];
        json << "    {\n";
        json << "      \"name\": \"" << comp.name << "\",\n";
        json << "      \"status\": \"";
        
        switch (comp.status) {
            case HealthStatus::HEALTHY: json << "healthy"; break;
            case HealthStatus::DEGRADED: json << "degraded"; break;
            case HealthStatus::UNHEALTHY: json << "unhealthy"; break;
            case HealthStatus::UNKNOWN: json << "unknown"; break;
        }
        
        json << "\",\n";
        json << "      \"message\": \"" << comp.message << "\",\n";
        
        auto check_time_t = std::chrono::system_clock::to_time_t(comp.last_check);
        json << "      \"last_check\": \"" << std::ctime(&check_time_t) << "\",\n";
        
        json << "      \"details\": {\n";
        size_t detail_count = 0;
        for (const auto& [key, value] : comp.details) {
            json << "        \"" << key << "\": \"" << value << "\"";
            if (++detail_count < comp.details.size()) json << ",";
            json << "\n";
        }
        json << "      }\n";
        json << "    }";
        if (i < health.components.size() - 1) json << ",";
        json << "\n";
    }
    json << "  ],\n";
    
    json << "  \"system_info\": {\n";
    size_t info_count = 0;
    for (const auto& [key, value] : health.system_info) {
        json << "    \"" << key << "\": \"" << value << "\"";
        if (++info_count < health.system_info.size()) json << ",";
        json << "\n";
    }
    json << "  }\n";
    json << "}\n";
    
    return json.str();
}

std::string HealthCheckManager::get_health_json() {
    SystemHealth health = get_system_health();
    return to_json(health);
}

std::string HealthCheckManager::get_metrics_json() {
    SystemHealth health = get_system_health();
    
    std::ostringstream metrics;
    metrics << "{\n";
    metrics << "  \"timestamp\": " << std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count() << ",\n";
    
    auto uptime = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now() - health.startup_time);
    metrics << "  \"uptime_seconds\": " << uptime.count() << ",\n";
    
    metrics << "  \"component_counts\": {\n";
    int healthy = 0, degraded = 0, unhealthy = 0, unknown = 0;
    
    for (const auto& comp : health.components) {
        switch (comp.status) {
            case HealthStatus::HEALTHY: healthy++; break;
            case HealthStatus::DEGRADED: degraded++; break;
            case HealthStatus::UNHEALTHY: unhealthy++; break;
            case HealthStatus::UNKNOWN: unknown++; break;
        }
    }
    
    metrics << "    \"healthy\": " << healthy << ",\n";
    metrics << "    \"degraded\": " << degraded << ",\n";
    metrics << "    \"unhealthy\": " << unhealthy << ",\n";
    metrics << "    \"unknown\": " << unknown << "\n";
    metrics << "  }\n";
    metrics << "}\n";
    
    return metrics.str();
}

void HealthCheckManager::start_background_checks(std::chrono::milliseconds interval) {
    if (background_running_.load()) {
        return;
    }
    
    background_running_.store(true);
    background_thread_ = std::make_unique<std::thread>(
        &HealthCheckManager::background_check_loop, this, interval);
    
    Logger::info("Started background health checks with interval: " + 
                std::to_string(interval.count()) + "ms");
}

void HealthCheckManager::stop_background_checks() {
    background_running_.store(false);
    if (background_thread_ && background_thread_->joinable()) {
        background_thread_->join();
    }
    background_thread_.reset();
    
    Logger::info("Stopped background health checks");
}

void HealthCheckManager::set_critical_components(const std::vector<std::string>& components) {
    std::unique_lock<std::shared_mutex> lock(components_mutex_);
    critical_components_ = components;
    
    // Mark components as critical
    for (const auto& name : components) {
        auto it = components_.find(name);
        if (it != components_.end()) {
            it->second->is_critical.store(true);
        }
    }
}

void HealthCheckManager::background_check_loop(std::chrono::milliseconds interval) {
    while (background_running_.load()) {
        try {
            system_health_ = get_system_health();
        } catch (const std::exception& e) {
            Logger::error("Background health check failed: " + std::string(e.what()));
        }
        
        std::this_thread::sleep_for(interval);
    }
}

HealthStatus HealthCheckManager::determine_overall_status() {
    bool has_unhealthy = false;
    bool has_degraded = false;
    
    std::shared_lock<std::shared_mutex> lock(components_mutex_);
    
    for (const auto& [name, component] : components_) {
        HealthStatus status = component->last_result.status;
        
        if (component->is_critical.load() && status == HealthStatus::UNHEALTHY) {
            return HealthStatus::UNHEALTHY;
        }
        
        if (status == HealthStatus::UNHEALTHY) {
            has_unhealthy = true;
        } else if (status == HealthStatus::DEGRADED) {
            has_degraded = true;
        }
    }
    
    if (has_unhealthy) {
        return HealthStatus::DEGRADED;
    } else if (has_degraded) {
        return HealthStatus::DEGRADED;
    } else {
        return HealthStatus::HEALTHY;
    }
}

void HealthCheckManager::collect_system_info() {
    system_health_.system_info["hostname"] = []() {
        char hostname[256];
        if (gethostname(hostname, sizeof(hostname)) == 0) {
            return std::string(hostname);
        }
        return std::string("unknown");
    }();
    
    system_health_.system_info["pid"] = std::to_string(getpid());
    
    // Get CPU info
    std::ifstream cpuinfo("/proc/cpuinfo");
    if (cpuinfo.is_open()) {
        std::string line;
        while (std::getline(cpuinfo, line)) {
            if (line.find("model name") != std::string::npos) {
                size_t pos = line.find(':');
                if (pos != std::string::npos) {
                    system_health_.system_info["cpu_model"] = line.substr(pos + 2);
                    break;
                }
            }
        }
    }
}

ComponentHealth HealthCheckManager::check_memory_usage() {
    ComponentHealth health("memory");
    
    struct sysinfo info;
    if (sysinfo(&info) == 0) {
        double total_mb = info.totalram / (1024.0 * 1024.0);
        double free_mb = info.freeram / (1024.0 * 1024.0);
        double used_percent = ((total_mb - free_mb) / total_mb) * 100.0;
        
        health.details["total_mb"] = std::to_string(static_cast<int>(total_mb));
        health.details["free_mb"] = std::to_string(static_cast<int>(free_mb));
        health.details["used_percent"] = std::to_string(static_cast<int>(used_percent));
        
        if (used_percent > 90) {
            health.status = HealthStatus::UNHEALTHY;
            health.message = "Memory usage critical: " + std::to_string(static_cast<int>(used_percent)) + "%";
        } else if (used_percent > 80) {
            health.status = HealthStatus::DEGRADED;
            health.message = "Memory usage high: " + std::to_string(static_cast<int>(used_percent)) + "%";
        } else {
            health.status = HealthStatus::HEALTHY;
            health.message = "Memory usage normal: " + std::to_string(static_cast<int>(used_percent)) + "%";
        }
    } else {
        health.status = HealthStatus::UNKNOWN;
        health.message = "Unable to read memory information";
    }
    
    return health;
}

ComponentHealth HealthCheckManager::check_cpu_usage() {
    ComponentHealth health("cpu");
    
    // Simple CPU check - in production, this would read /proc/stat
    std::ifstream loadavg("/proc/loadavg");
    if (loadavg.is_open()) {
        double load1, load5, load15;
        loadavg >> load1 >> load5 >> load15;
        
        int cpu_count = std::thread::hardware_concurrency();
        double load_percent = (load1 / cpu_count) * 100.0;
        
        health.details["load_1min"] = std::to_string(load1);
        health.details["load_5min"] = std::to_string(load5);
        health.details["load_15min"] = std::to_string(load15);
        health.details["cpu_count"] = std::to_string(cpu_count);
        
        if (load_percent > 90) {
            health.status = HealthStatus::UNHEALTHY;
            health.message = "CPU load critical: " + std::to_string(static_cast<int>(load_percent)) + "%";
        } else if (load_percent > 70) {
            health.status = HealthStatus::DEGRADED;
            health.message = "CPU load high: " + std::to_string(static_cast<int>(load_percent)) + "%";
        } else {
            health.status = HealthStatus::HEALTHY;
            health.message = "CPU load normal: " + std::to_string(static_cast<int>(load_percent)) + "%";
        }
    } else {
        health.status = HealthStatus::UNKNOWN;
        health.message = "Unable to read CPU load information";
    }
    
    return health;
}

ComponentHealth HealthCheckManager::check_disk_space() {
    ComponentHealth health("disk");
    
    struct statvfs stat;
    if (statvfs("/", &stat) == 0) {
        double total_gb = (stat.f_blocks * stat.f_frsize) / (1024.0 * 1024.0 * 1024.0);
        double free_gb = (stat.f_bavail * stat.f_frsize) / (1024.0 * 1024.0 * 1024.0);
        double used_percent = ((total_gb - free_gb) / total_gb) * 100.0;
        
        health.details["total_gb"] = std::to_string(static_cast<int>(total_gb));
        health.details["free_gb"] = std::to_string(static_cast<int>(free_gb));
        health.details["used_percent"] = std::to_string(static_cast<int>(used_percent));
        
        if (used_percent > 95) {
            health.status = HealthStatus::UNHEALTHY;
            health.message = "Disk space critical: " + std::to_string(static_cast<int>(used_percent)) + "%";
        } else if (used_percent > 85) {
            health.status = HealthStatus::DEGRADED;
            health.message = "Disk space low: " + std::to_string(static_cast<int>(used_percent)) + "%";
        } else {
            health.status = HealthStatus::HEALTHY;
            health.message = "Disk space normal: " + std::to_string(static_cast<int>(used_percent)) + "%";
        }
    } else {
        health.status = HealthStatus::UNKNOWN;
        health.message = "Unable to read disk space information";
    }
    
    return health;
}

ComponentHealth HealthCheckManager::check_network_connectivity() {
    ComponentHealth health("network");
    
    // Check if network interfaces are up
    struct ifaddrs *ifaddr;
    if (getifaddrs(&ifaddr) == 0) {
        int interface_count = 0;
        int up_interfaces = 0;
        
        for (struct ifaddrs *ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next) {
            if (ifa->ifa_addr == nullptr) continue;
            
            if (ifa->ifa_addr->sa_family == AF_INET || ifa->ifa_addr->sa_family == AF_INET6) {
                interface_count++;
                if (ifa->ifa_flags & IFF_UP) {
                    up_interfaces++;
                }
            }
        }
        
        freeifaddrs(ifaddr);
        
        health.details["total_interfaces"] = std::to_string(interface_count);
        health.details["up_interfaces"] = std::to_string(up_interfaces);
        
        if (up_interfaces == 0) {
            health.status = HealthStatus::UNHEALTHY;
            health.message = "No network interfaces are up";
        } else if (up_interfaces < interface_count) {
            health.status = HealthStatus::DEGRADED;
            health.message = "Some network interfaces are down";
        } else {
            health.status = HealthStatus::HEALTHY;
            health.message = "All network interfaces are up";
        }
    } else {
        health.status = HealthStatus::UNKNOWN;
        health.message = "Unable to read network interface information";
    }
    
    return health;
}

ComponentHealth HealthCheckManager::check_dpdk_status() {
    ComponentHealth health("dpdk");
    
    // Check if DPDK is initialized and ports are available
    // This is a simplified check - in production, this would use DPDK APIs
    std::ifstream hugepages("/proc/meminfo");
    if (hugepages.is_open()) {
        std::string line;
        bool found_hugepages = false;
        int hugepages_total = 0;
        int hugepages_free = 0;
        
        while (std::getline(hugepages, line)) {
            if (line.find("HugePages_Total:") != std::string::npos) {
                sscanf(line.c_str(), "HugePages_Total: %d", &hugepages_total);
                found_hugepages = true;
            } else if (line.find("HugePages_Free:") != std::string::npos) {
                sscanf(line.c_str(), "HugePages_Free: %d", &hugepages_free);
            }
        }
        
        if (found_hugepages && hugepages_total > 0) {
            health.details["hugepages_total"] = std::to_string(hugepages_total);
            health.details["hugepages_free"] = std::to_string(hugepages_free);
            health.details["hugepages_used"] = std::to_string(hugepages_total - hugepages_free);
            
            if (hugepages_free == 0) {
                health.status = HealthStatus::DEGRADED;
                health.message = "No free hugepages available";
            } else {
                health.status = HealthStatus::HEALTHY;
                health.message = "DPDK hugepages available";
            }
        } else {
            health.status = HealthStatus::UNHEALTHY;
            health.message = "No hugepages configured for DPDK";
        }
    } else {
        health.status = HealthStatus::UNKNOWN;
        health.message = "Unable to check DPDK status";
    }
    
    return health;
}

ComponentHealth HealthCheckManager::check_gpu_status() {
    ComponentHealth health("gpu");
    
    // Check if NVIDIA GPU is available
    std::ifstream nvidia_smi("/proc/driver/nvidia/version");
    if (nvidia_smi.is_open()) {
        std::string version_line;
        std::getline(nvidia_smi, version_line);
        
        health.details["nvidia_driver"] = version_line;
        health.status = HealthStatus::HEALTHY;
        health.message = "NVIDIA GPU driver available";
    } else {
        health.status = HealthStatus::DEGRADED;
        health.message = "No NVIDIA GPU driver found";
    }
    
    return health;
}

// HealthCheckServer implementation
HealthCheckServer::HealthCheckServer(std::shared_ptr<HealthCheckManager> health_manager, 
                                   const Config& config)
    : health_manager_(health_manager), config_(config) {}

HealthCheckServer::~HealthCheckServer() {
    stop();
}

void HealthCheckServer::start() {
    if (running_.load()) {
        return;
    }
    
    running_.store(true);
    server_thread_ = std::make_unique<std::thread>(&HealthCheckServer::server_loop, this);
    
    Logger::info("Health check server started on " + config_.bind_address + ":" + 
                std::to_string(config_.port));
}

void HealthCheckServer::stop() {
    running_.store(false);
    if (server_thread_ && server_thread_->joinable()) {
        server_thread_->join();
    }
    server_thread_.reset();
    
    Logger::info("Health check server stopped");
}

void HealthCheckServer::server_loop() {
    // Simplified HTTP server implementation
    // In production, this would use a proper HTTP library like cpp-httplib
    
    while (running_.load()) {
        try {
            // This is a placeholder for the actual HTTP server implementation
            // The real implementation would listen on a socket and handle HTTP requests
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        } catch (const std::exception& e) {
            Logger::error("Health check server error: " + std::string(e.what()));
        }
    }
}

std::string HealthCheckServer::handle_request(const std::string& path, const std::string& method) {
    if (method != "GET") {
        return create_http_response(405, "text/plain", "Method Not Allowed");
    }
    
    if (path == "/health/live") {
        bool alive = health_manager_->is_alive();
        int status_code = alive ? 200 : 503;
        std::string body = alive ? "OK" : "Service Unavailable";
        return create_http_response(status_code, "text/plain", body);
    }
    
    if (path == "/health/ready") {
        bool ready = health_manager_->is_ready();
        int status_code = ready ? 200 : 503;
        std::string body = ready ? "Ready" : "Not Ready";
        return create_http_response(status_code, "text/plain", body);
    }
    
    if (path == "/health/startup") {
        bool startup_complete = health_manager_->is_startup_complete();
        int status_code = startup_complete ? 200 : 503;
        std::string body = startup_complete ? "Started" : "Starting";
        return create_http_response(status_code, "text/plain", body);
    }
    
    if (path == "/health") {
        std::string json = health_manager_->get_health_json();
        return create_http_response(200, "application/json", json);
    }
    
    if (path == "/metrics" && config_.enable_detailed_metrics) {
        std::string json = health_manager_->get_metrics_json();
        return create_http_response(200, "application/json", json);
    }
    
    return create_http_response(404, "text/plain", "Not Found");
}

std::string HealthCheckServer::create_http_response(int status_code, const std::string& content_type, 
                                                  const std::string& body) {
    std::ostringstream response;
    response << "HTTP/1.1 " << status_code << " ";
    
    switch (status_code) {
        case 200: response << "OK"; break;
        case 404: response << "Not Found"; break;
        case 405: response << "Method Not Allowed"; break;
        case 503: response << "Service Unavailable"; break;
        default: response << "Unknown"; break;
    }
    
    response << "\r\n";
    response << "Content-Type: " << content_type << "\r\n";
    response << "Content-Length: " << body.length() << "\r\n";
    response << "Connection: close\r\n";
    response << "\r\n";
    response << body;
    
    return response.str();
}

} // namespace common
} // namespace ultra_cpp