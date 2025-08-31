#pragma once

#include <string>
#include <memory>
#include <functional>
#include <unordered_map>
#include <atomic>
#include <chrono>
#include <vector>
#include <mutex>
#include <thread>

namespace integration {

/**
 * HTTP Proxy for seamless fallback to Node.js APIs
 * Provides intelligent routing and automatic failover
 */
class HttpProxy {
public:
    struct Config {
        std::string nodejs_upstream = "http://localhost:3005";
        std::string cpp_bind_address = "0.0.0.0";
        uint16_t cpp_port = 8080;
        uint32_t connection_timeout_ms = 5000;
        uint32_t request_timeout_ms = 30000;
        uint32_t max_connections = 1000;
        uint32_t health_check_interval_ms = 10000;
        bool enable_circuit_breaker = true;
        uint32_t circuit_breaker_threshold = 5;
        uint32_t circuit_breaker_timeout_ms = 60000;
    };

    struct RouteConfig {
        std::string path_pattern;
        std::vector<std::string> methods;
        bool prefer_cpp = true;
        uint32_t timeout_ms = 5000;
        bool cache_enabled = false;
        uint32_t cache_ttl_seconds = 300;
    };

    struct UpstreamHealth {
        std::atomic<bool> is_healthy{true};
        std::atomic<uint64_t> last_check_time{0};
        std::atomic<uint32_t> consecutive_failures{0};
        std::atomic<uint64_t> total_requests{0};
        std::atomic<uint64_t> failed_requests{0};
        std::atomic<double> avg_response_time_ms{0.0};
    };

    explicit HttpProxy(const Config& config);
    ~HttpProxy();

    // Lifecycle management
    bool start();
    void stop();
    bool is_running() const { return running_.load(); }

    // Route configuration
    void add_route(const RouteConfig& route);
    void remove_route(const std::string& path_pattern);
    void update_route(const std::string& path_pattern, const RouteConfig& route);

    // Health monitoring
    UpstreamHealth get_nodejs_health() const;
    UpstreamHealth get_cpp_health() const;
    void force_health_check();

    // Circuit breaker control
    void open_circuit_breaker();
    void close_circuit_breaker();
    bool is_circuit_breaker_open() const;

    // Statistics
    struct Stats {
        uint64_t total_requests = 0;
        uint64_t cpp_requests = 0;
        uint64_t nodejs_requests = 0;
        uint64_t failed_requests = 0;
        uint64_t cache_hits = 0;
        uint64_t cache_misses = 0;
        double avg_response_time_ms = 0.0;
        uint64_t circuit_breaker_trips = 0;
    };
    
    Stats get_stats() const;
    void reset_stats();

private:
    Config config_;
    std::atomic<bool> running_{false};
    std::atomic<bool> circuit_breaker_open_{false};
    std::atomic<uint64_t> circuit_breaker_open_time_{0};
    
    // Route management
    std::unordered_map<std::string, RouteConfig> routes_;
    mutable std::shared_mutex routes_mutex_;
    
    // Health monitoring
    UpstreamHealth nodejs_health_;
    UpstreamHealth cpp_health_;
    std::thread health_check_thread_;
    
    // Statistics
    mutable Stats stats_;
    mutable std::mutex stats_mutex_;
    
    // Internal methods
    void health_check_loop();
    bool check_nodejs_health();
    bool check_cpp_health();
    void update_circuit_breaker_state();
    
    // Request routing
    bool should_route_to_cpp(const std::string& path, const std::string& method);
    std::string route_to_nodejs(const std::string& request);
    std::string route_to_cpp(const std::string& request);
    
    // HTTP client for upstream requests
    class HttpClient;
    std::unique_ptr<HttpClient> http_client_;
};

/**
 * HTTP Request/Response structures for internal processing
 */
struct HttpRequest {
    std::string method;
    std::string path;
    std::string query_string;
    std::unordered_map<std::string, std::string> headers;
    std::string body;
    std::chrono::steady_clock::time_point start_time;
    std::string client_ip;
    std::string request_id;
};

struct HttpResponse {
    uint16_t status_code = 200;
    std::unordered_map<std::string, std::string> headers;
    std::string body;
    std::chrono::steady_clock::time_point end_time;
    bool from_cache = false;
    std::string upstream_server;
};

/**
 * Request handler interface for processing HTTP requests
 */
class RequestHandler {
public:
    virtual ~RequestHandler() = default;
    virtual HttpResponse handle_request(const HttpRequest& request) = 0;
    virtual bool can_handle(const std::string& path, const std::string& method) = 0;
};

/**
 * Fallback handler that routes requests to Node.js
 */
class NodejsFallbackHandler : public RequestHandler {
public:
    explicit NodejsFallbackHandler(const std::string& upstream_url);
    
    HttpResponse handle_request(const HttpRequest& request) override;
    bool can_handle(const std::string& path, const std::string& method) override;
    
private:
    std::string upstream_url_;
    class HttpClient;
    std::unique_ptr<HttpClient> client_;
};

} // namespace integration