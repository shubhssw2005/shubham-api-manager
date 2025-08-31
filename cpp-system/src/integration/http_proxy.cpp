#include "integration/http_proxy.hpp"
#include "common/logger.hpp"
#include "performance-monitor/performance_monitor.hpp"
#include <curl/curl.h>
#include <json/json.h>
#include <regex>
#include <sstream>
#include <iomanip>

namespace integration {

// HTTP Client implementation using libcurl
class HttpProxy::HttpClient {
public:
    HttpClient() {
        curl_global_init(CURL_GLOBAL_DEFAULT);
        curl_ = curl_easy_init();
        if (!curl_) {
            throw std::runtime_error("Failed to initialize libcurl");
        }
    }
    
    ~HttpClient() {
        if (curl_) {
            curl_easy_cleanup(curl_);
        }
        curl_global_cleanup();
    }
    
    struct Response {
        long status_code = 0;
        std::string body;
        std::unordered_map<std::string, std::string> headers;
        double total_time = 0.0;
    };
    
    Response make_request(const std::string& url, 
                         const std::string& method,
                         const std::unordered_map<std::string, std::string>& headers,
                         const std::string& body,
                         uint32_t timeout_ms) {
        Response response;
        
        // Reset curl handle
        curl_easy_reset(curl_);
        
        // Set URL
        curl_easy_setopt(curl_, CURLOPT_URL, url.c_str());
        
        // Set method
        if (method == "POST") {
            curl_easy_setopt(curl_, CURLOPT_POST, 1L);
            curl_easy_setopt(curl_, CURLOPT_POSTFIELDS, body.c_str());
            curl_easy_setopt(curl_, CURLOPT_POSTFIELDSIZE, body.length());
        } else if (method == "PUT") {
            curl_easy_setopt(curl_, CURLOPT_CUSTOMREQUEST, "PUT");
            curl_easy_setopt(curl_, CURLOPT_POSTFIELDS, body.c_str());
            curl_easy_setopt(curl_, CURLOPT_POSTFIELDSIZE, body.length());
        } else if (method == "DELETE") {
            curl_easy_setopt(curl_, CURLOPT_CUSTOMREQUEST, "DELETE");
        } else if (method == "PATCH") {
            curl_easy_setopt(curl_, CURLOPT_CUSTOMREQUEST, "PATCH");
            curl_easy_setopt(curl_, CURLOPT_POSTFIELDS, body.c_str());
            curl_easy_setopt(curl_, CURLOPT_POSTFIELDSIZE, body.length());
        }
        
        // Set headers
        struct curl_slist* header_list = nullptr;
        for (const auto& [key, value] : headers) {
            std::string header = key + ": " + value;
            header_list = curl_slist_append(header_list, header.c_str());
        }
        if (header_list) {
            curl_easy_setopt(curl_, CURLOPT_HTTPHEADER, header_list);
        }
        
        // Set timeout
        curl_easy_setopt(curl_, CURLOPT_TIMEOUT_MS, timeout_ms);
        curl_easy_setopt(curl_, CURLOPT_CONNECTTIMEOUT_MS, timeout_ms / 2);
        
        // Set write callback for response body
        curl_easy_setopt(curl_, CURLOPT_WRITEFUNCTION, write_callback);
        curl_easy_setopt(curl_, CURLOPT_WRITEDATA, &response.body);
        
        // Set header callback
        curl_easy_setopt(curl_, CURLOPT_HEADERFUNCTION, header_callback);
        curl_easy_setopt(curl_, CURLOPT_HEADERDATA, &response.headers);
        
        // Perform request
        CURLcode res = curl_easy_perform(curl_);
        
        // Clean up headers
        if (header_list) {
            curl_slist_free_all(header_list);
        }
        
        if (res != CURLE_OK) {
            throw std::runtime_error("HTTP request failed: " + std::string(curl_easy_strerror(res)));
        }
        
        // Get response info
        curl_easy_getinfo(curl_, CURLINFO_RESPONSE_CODE, &response.status_code);
        curl_easy_getinfo(curl_, CURLINFO_TOTAL_TIME, &response.total_time);
        
        return response;
    }
    
private:
    CURL* curl_;
    
    static size_t write_callback(void* contents, size_t size, size_t nmemb, std::string* userp) {
        size_t total_size = size * nmemb;
        userp->append(static_cast<char*>(contents), total_size);
        return total_size;
    }
    
    static size_t header_callback(char* buffer, size_t size, size_t nitems, 
                                 std::unordered_map<std::string, std::string>* headers) {
        size_t total_size = size * nitems;
        std::string header(buffer, total_size);
        
        // Parse header
        size_t colon_pos = header.find(':');
        if (colon_pos != std::string::npos) {
            std::string key = header.substr(0, colon_pos);
            std::string value = header.substr(colon_pos + 1);
            
            // Trim whitespace
            key.erase(0, key.find_first_not_of(" \t\r\n"));
            key.erase(key.find_last_not_of(" \t\r\n") + 1);
            value.erase(0, value.find_first_not_of(" \t\r\n"));
            value.erase(value.find_last_not_of(" \t\r\n") + 1);
            
            (*headers)[key] = value;
        }
        
        return total_size;
    }
};

HttpProxy::HttpProxy(const Config& config) 
    : config_(config), http_client_(std::make_unique<HttpClient>()) {
    
    // Initialize default routes for common API patterns
    add_route({"/api/auth/*", {"POST", "GET"}, false, 5000, false, 0});
    add_route({"/api/users/*", {"GET", "POST", "PUT", "DELETE"}, false, 5000, true, 300});
    add_route({"/api/media/*", {"GET"}, true, 1000, true, 600});
    add_route({"/api/cache/*", {"GET"}, true, 500, true, 300});
    add_route({"/api/metrics", {"GET"}, true, 1000, false, 0});
    
    LOG_INFO("HttpProxy initialized with upstream: {}", config_.nodejs_upstream);
}

HttpProxy::~HttpProxy() {
    stop();
}

bool HttpProxy::start() {
    if (running_.load()) {
        LOG_WARN("HttpProxy already running");
        return true;
    }
    
    try {
        // Start health check thread
        health_check_thread_ = std::thread(&HttpProxy::health_check_loop, this);
        
        running_.store(true);
        LOG_INFO("HttpProxy started on {}:{}", config_.cpp_bind_address, config_.cpp_port);
        return true;
        
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to start HttpProxy: {}", e.what());
        return false;
    }
}

void HttpProxy::stop() {
    if (!running_.load()) {
        return;
    }
    
    running_.store(false);
    
    if (health_check_thread_.joinable()) {
        health_check_thread_.join();
    }
    
    LOG_INFO("HttpProxy stopped");
}

void HttpProxy::add_route(const RouteConfig& route) {
    std::unique_lock<std::shared_mutex> lock(routes_mutex_);
    routes_[route.path_pattern] = route;
    LOG_INFO("Added route: {} -> {}", route.path_pattern, 
             route.prefer_cpp ? "C++" : "Node.js");
}

void HttpProxy::remove_route(const std::string& path_pattern) {
    std::unique_lock<std::shared_mutex> lock(routes_mutex_);
    routes_.erase(path_pattern);
    LOG_INFO("Removed route: {}", path_pattern);
}

void HttpProxy::update_route(const std::string& path_pattern, const RouteConfig& route) {
    std::unique_lock<std::shared_mutex> lock(routes_mutex_);
    routes_[path_pattern] = route;
    LOG_INFO("Updated route: {}", path_pattern);
}

HttpProxy::UpstreamHealth HttpProxy::get_nodejs_health() const {
    return nodejs_health_;
}

HttpProxy::UpstreamHealth HttpProxy::get_cpp_health() const {
    return cpp_health_;
}

void HttpProxy::force_health_check() {
    check_nodejs_health();
    check_cpp_health();
}

void HttpProxy::open_circuit_breaker() {
    circuit_breaker_open_.store(true);
    circuit_breaker_open_time_.store(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()
        ).count()
    );
    
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_.circuit_breaker_trips++;
    
    LOG_WARN("Circuit breaker opened");
}

void HttpProxy::close_circuit_breaker() {
    circuit_breaker_open_.store(false);
    circuit_breaker_open_time_.store(0);
    LOG_INFO("Circuit breaker closed");
}

bool HttpProxy::is_circuit_breaker_open() const {
    if (!config_.enable_circuit_breaker) {
        return false;
    }
    
    if (!circuit_breaker_open_.load()) {
        return false;
    }
    
    // Check if timeout has passed
    auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()
    ).count();
    
    auto open_time = circuit_breaker_open_time_.load();
    if (now - open_time > config_.circuit_breaker_timeout_ms) {
        // Try to close circuit breaker
        const_cast<HttpProxy*>(this)->close_circuit_breaker();
        return false;
    }
    
    return true;
}

HttpProxy::Stats HttpProxy::get_stats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
}

void HttpProxy::reset_stats() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_ = Stats{};
    LOG_INFO("HttpProxy stats reset");
}

void HttpProxy::health_check_loop() {
    while (running_.load()) {
        try {
            check_nodejs_health();
            check_cpp_health();
            update_circuit_breaker_state();
            
            std::this_thread::sleep_for(
                std::chrono::milliseconds(config_.health_check_interval_ms)
            );
        } catch (const std::exception& e) {
            LOG_ERROR("Health check error: {}", e.what());
        }
    }
}

bool HttpProxy::check_nodejs_health() {
    try {
        auto start_time = std::chrono::steady_clock::now();
        
        // Make health check request to Node.js
        auto response = http_client_->make_request(
            config_.nodejs_upstream + "/api/health",
            "GET",
            {{"User-Agent", "HttpProxy/1.0"}},
            "",
            config_.connection_timeout_ms
        );
        
        auto end_time = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time
        ).count();
        
        bool is_healthy = (response.status_code == 200);
        
        // Update health status
        nodejs_health_.is_healthy.store(is_healthy);
        nodejs_health_.last_check_time.store(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()
            ).count()
        );
        
        if (is_healthy) {
            nodejs_health_.consecutive_failures.store(0);
            
            // Update average response time
            auto current_avg = nodejs_health_.avg_response_time_ms.load();
            auto new_avg = (current_avg * 0.9) + (duration * 0.1);
            nodejs_health_.avg_response_time_ms.store(new_avg);
        } else {
            nodejs_health_.consecutive_failures.fetch_add(1);
        }
        
        nodejs_health_.total_requests.fetch_add(1);
        if (!is_healthy) {
            nodejs_health_.failed_requests.fetch_add(1);
        }
        
        return is_healthy;
        
    } catch (const std::exception& e) {
        LOG_ERROR("Node.js health check failed: {}", e.what());
        nodejs_health_.is_healthy.store(false);
        nodejs_health_.consecutive_failures.fetch_add(1);
        nodejs_health_.failed_requests.fetch_add(1);
        return false;
    }
}

bool HttpProxy::check_cpp_health() {
    // For C++ health, we check internal metrics and resource usage
    try {
        // Check memory usage
        auto memory_usage = performance::PerformanceMonitor::get_memory_usage();
        bool memory_ok = memory_usage.used_percent < 90.0;
        
        // Check CPU usage
        auto cpu_usage = performance::PerformanceMonitor::get_cpu_usage();
        bool cpu_ok = cpu_usage.usage_percent < 95.0;
        
        // Check if we can allocate memory
        bool allocation_ok = true;
        try {
            std::vector<char> test_allocation(1024 * 1024); // 1MB test
        } catch (const std::bad_alloc&) {
            allocation_ok = false;
        }
        
        bool is_healthy = memory_ok && cpu_ok && allocation_ok;
        
        cpp_health_.is_healthy.store(is_healthy);
        cpp_health_.last_check_time.store(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()
            ).count()
        );
        
        if (is_healthy) {
            cpp_health_.consecutive_failures.store(0);
        } else {
            cpp_health_.consecutive_failures.fetch_add(1);
        }
        
        cpp_health_.total_requests.fetch_add(1);
        if (!is_healthy) {
            cpp_health_.failed_requests.fetch_add(1);
        }
        
        return is_healthy;
        
    } catch (const std::exception& e) {
        LOG_ERROR("C++ health check failed: {}", e.what());
        cpp_health_.is_healthy.store(false);
        cpp_health_.consecutive_failures.fetch_add(1);
        cpp_health_.failed_requests.fetch_add(1);
        return false;
    }
}

void HttpProxy::update_circuit_breaker_state() {
    if (!config_.enable_circuit_breaker) {
        return;
    }
    
    // Check if we should open the circuit breaker
    auto nodejs_failures = nodejs_health_.consecutive_failures.load();
    auto cpp_failures = cpp_health_.consecutive_failures.load();
    
    if (!circuit_breaker_open_.load()) {
        if (nodejs_failures >= config_.circuit_breaker_threshold ||
            cpp_failures >= config_.circuit_breaker_threshold) {
            open_circuit_breaker();
        }
    }
}

bool HttpProxy::should_route_to_cpp(const std::string& path, const std::string& method) {
    // If circuit breaker is open, route everything to Node.js
    if (is_circuit_breaker_open()) {
        return false;
    }
    
    // If C++ is not healthy, route to Node.js
    if (!cpp_health_.is_healthy.load()) {
        return false;
    }
    
    // Check route configuration
    std::shared_lock<std::shared_mutex> lock(routes_mutex_);
    
    for (const auto& [pattern, config] : routes_) {
        // Simple pattern matching (could be enhanced with regex)
        if (pattern.back() == '*') {
            std::string prefix = pattern.substr(0, pattern.length() - 1);
            if (path.substr(0, prefix.length()) == prefix) {
                // Check if method is allowed
                if (std::find(config.methods.begin(), config.methods.end(), method) != config.methods.end()) {
                    return config.prefer_cpp;
                }
            }
        } else if (path == pattern) {
            // Check if method is allowed
            if (std::find(config.methods.begin(), config.methods.end(), method) != config.methods.end()) {
                return config.prefer_cpp;
            }
        }
    }
    
    // Default to Node.js for unknown routes
    return false;
}

// NodejsFallbackHandler implementation
NodejsFallbackHandler::NodejsFallbackHandler(const std::string& upstream_url)
    : upstream_url_(upstream_url), client_(std::make_unique<HttpProxy::HttpClient>()) {
}

HttpResponse NodejsFallbackHandler::handle_request(const HttpRequest& request) {
    HttpResponse response;
    
    try {
        // Build full URL
        std::string url = upstream_url_ + request.path;
        if (!request.query_string.empty()) {
            url += "?" + request.query_string;
        }
        
        // Forward headers (excluding hop-by-hop headers)
        auto headers = request.headers;
        headers.erase("connection");
        headers.erase("upgrade");
        headers.erase("proxy-authenticate");
        headers.erase("proxy-authorization");
        headers.erase("te");
        headers.erase("trailers");
        headers.erase("transfer-encoding");
        
        // Add proxy headers
        headers["X-Forwarded-For"] = request.client_ip;
        headers["X-Forwarded-Proto"] = "http";
        headers["X-Request-ID"] = request.request_id;
        
        // Make request to Node.js
        auto client_response = client_->make_request(
            url, request.method, headers, request.body, 30000
        );
        
        // Convert response
        response.status_code = static_cast<uint16_t>(client_response.status_code);
        response.headers = client_response.headers;
        response.body = client_response.body;
        response.upstream_server = "nodejs";
        response.end_time = std::chrono::steady_clock::now();
        
        LOG_DEBUG("Proxied request to Node.js: {} {} -> {}", 
                 request.method, request.path, response.status_code);
        
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to proxy request to Node.js: {}", e.what());
        response.status_code = 502;
        response.body = R"({"error": "Bad Gateway", "message": "Failed to reach upstream server"})";
        response.headers["Content-Type"] = "application/json";
        response.upstream_server = "error";
        response.end_time = std::chrono::steady_clock::now();
    }
    
    return response;
}

bool NodejsFallbackHandler::can_handle(const std::string& path, const std::string& method) {
    // Fallback handler can handle any request
    return true;
}

} // namespace integration