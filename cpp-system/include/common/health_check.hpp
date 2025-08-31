#pragma once

#include <string>
#include <unordered_map>
#include <atomic>
#include <chrono>
#include <memory>
#include <functional>
#include <vector>

namespace ultra_cpp {
namespace common {

enum class HealthStatus {
    HEALTHY,
    DEGRADED,
    UNHEALTHY,
    UNKNOWN
};

struct ComponentHealth {
    std::string name;
    HealthStatus status;
    std::string message;
    std::chrono::system_clock::time_point last_check;
    std::unordered_map<std::string, std::string> details;
    
    ComponentHealth(const std::string& component_name)
        : name(component_name)
        , status(HealthStatus::UNKNOWN)
        , last_check(std::chrono::system_clock::now()) {}
};

struct SystemHealth {
    HealthStatus overall_status;
    std::string version;
    std::chrono::system_clock::time_point startup_time;
    std::chrono::system_clock::time_point last_update;
    std::vector<ComponentHealth> components;
    std::unordered_map<std::string, std::string> system_info;
    
    SystemHealth() 
        : overall_status(HealthStatus::UNKNOWN)
        , startup_time(std::chrono::system_clock::now())
        , last_update(std::chrono::system_clock::now()) {}
};

using HealthCheckFunction = std::function<ComponentHealth()>;

class HealthCheckManager {
public:
    HealthCheckManager();
    ~HealthCheckManager();
    
    // Register health check functions for different components
    void register_component(const std::string& name, HealthCheckFunction check_func);
    void unregister_component(const std::string& name);
    
    // Manual health updates
    void update_component_health(const std::string& name, HealthStatus status, 
                               const std::string& message = "");
    
    // Get current system health
    SystemHealth get_system_health();
    ComponentHealth get_component_health(const std::string& name);
    
    // Health check endpoints
    bool is_alive();      // Basic liveness check
    bool is_ready();      // Readiness check (all critical components healthy)
    bool is_startup_complete(); // Startup probe
    
    // JSON serialization for HTTP endpoints
    std::string to_json(const SystemHealth& health);
    std::string get_health_json();
    std::string get_metrics_json();
    
    // Start/stop background health checking
    void start_background_checks(std::chrono::milliseconds interval = std::chrono::milliseconds(5000));
    void stop_background_checks();
    
    // Set critical components (must be healthy for readiness)
    void set_critical_components(const std::vector<std::string>& components);
    
private:
    struct ComponentInfo {
        std::string name;
        HealthCheckFunction check_func;
        ComponentHealth last_result;
        std::atomic<bool> is_critical{false};
    };
    
    std::unordered_map<std::string, std::unique_ptr<ComponentInfo>> components_;
    std::vector<std::string> critical_components_;
    
    std::atomic<bool> background_running_{false};
    std::unique_ptr<std::thread> background_thread_;
    mutable std::shared_mutex components_mutex_;
    
    SystemHealth system_health_;
    std::chrono::system_clock::time_point startup_time_;
    std::atomic<bool> startup_complete_{false};
    
    void background_check_loop(std::chrono::milliseconds interval);
    void update_overall_status();
    HealthStatus determine_overall_status();
    void collect_system_info();
    
    // Built-in health checks
    ComponentHealth check_memory_usage();
    ComponentHealth check_cpu_usage();
    ComponentHealth check_disk_space();
    ComponentHealth check_network_connectivity();
    ComponentHealth check_dpdk_status();
    ComponentHealth check_gpu_status();
};

// HTTP Health Check Server
class HealthCheckServer {
public:
    struct Config {
        uint16_t port = 8081;
        std::string bind_address = "0.0.0.0";
        bool enable_detailed_metrics = true;
    };
    
    explicit HealthCheckServer(std::shared_ptr<HealthCheckManager> health_manager, 
                              const Config& config = {});
    ~HealthCheckServer();
    
    void start();
    void stop();
    
private:
    std::shared_ptr<HealthCheckManager> health_manager_;
    Config config_;
    std::atomic<bool> running_{false};
    std::unique_ptr<std::thread> server_thread_;
    
    void server_loop();
    std::string handle_request(const std::string& path, const std::string& method);
    std::string create_http_response(int status_code, const std::string& content_type, 
                                   const std::string& body);
};

} // namespace common
} // namespace ultra_cpp