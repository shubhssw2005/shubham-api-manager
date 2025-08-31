#include "common/fallback_manager.hpp"
#include "common/logger.hpp"
#include <curl/curl.h>
#include <thread>
#include <sstream>

namespace ultra {
namespace common {

// Callback for libcurl to write response data
static size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::string* userp) {
    size_t total_size = size * nmemb;
    userp->append(static_cast<char*>(contents), total_size);
    return total_size;
}

FallbackManager::FallbackManager(const FallbackConfig& config) : config_(config) {
    // Initialize libcurl
    curl_global_init(CURL_GLOBAL_DEFAULT);
    
    LOG_INFO("Fallback manager initialized with Node.js base URL: {}", config_.nodejs_base_url);
}

FallbackManager::~FallbackManager() {
    stop_health_monitoring();
    curl_global_cleanup();
}

void FallbackManager::start_health_monitoring() {
    if (health_monitoring_active_.exchange(true)) {
        LOG_WARNING("Health monitoring already active");
        return;
    }
    
    health_monitor_thread_ = std::make_unique<std::thread>(&FallbackManager::health_monitoring_loop, this);
    LOG_INFO("Health monitoring started for Node.js services");
}

void FallbackManager::stop_health_monitoring() {
    if (!health_monitoring_active_.exchange(false)) {
        return;
    }
    
    if (health_monitor_thread_ && health_monitor_thread_->joinable()) {
        health_monitor_thread_->join();
    }
    
    LOG_INFO("Health monitoring stopped");
}

void FallbackManager::register_fallback_handler(const std::string& endpoint, FallbackHandler handler) {
    std::lock_guard<std::mutex> lock(handlers_mutex_);
    fallback_handlers_[endpoint] = std::move(handler);
    LOG_DEBUG("Registered fallback handler for endpoint: {}", endpoint);
}

void FallbackManager::unregister_fallback_handler(const std::string& endpoint) {
    std::lock_guard<std::mutex> lock(handlers_mutex_);
    fallback_handlers_.erase(endpoint);
    LOG_DEBUG("Unregistered fallback handler for endpoint: {}", endpoint);
}

std::string FallbackManager::execute_nodejs_fallback(const std::string& endpoint, const std::string& request_data) {
    auto start_time = std::chrono::steady_clock::now();
    
    try {
        // Check if we have a custom fallback handler
        {
            std::lock_guard<std::mutex> lock(handlers_mutex_);
            auto it = fallback_handlers_.find(endpoint);
            if (it != fallback_handlers_.end()) {
                auto result = it->second(request_data);
                auto end_time = std::chrono::steady_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
                
                update_endpoint_stats(endpoint, false, true, duration);
                record_fallback(endpoint, FallbackReason::MANUAL_OVERRIDE, true);
                return result;
            }
        }
        
        // Use HTTP client to call Node.js service
        NodeJSClient client(config_.nodejs_base_url);
        NodeJSClient::Response response;
        
        if (request_data.empty()) {
            response = client.get(endpoint, config_.request_timeout);
        } else {
            response = client.post(endpoint, request_data, config_.request_timeout);
        }
        
        stats_.nodejs_requests.fetch_add(1);
        
        if (response.status_code >= 200 && response.status_code < 300) {
            auto end_time = std::chrono::steady_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
            
            update_endpoint_stats(endpoint, false, true, duration);
            record_fallback(endpoint, FallbackReason::COMPONENT_FAILURE, true);
            
            LOG_DEBUG("Successful fallback to Node.js for {}: {} ({}ms)", 
                     endpoint, response.status_code, response.response_time.count());
            
            return response.body;
        } else {
            stats_.nodejs_failures.fetch_add(1);
            throw std::runtime_error("Node.js service returned error: " + std::to_string(response.status_code));
        }
        
    } catch (const std::exception& e) {
        auto end_time = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
        
        stats_.nodejs_failures.fetch_add(1);
        update_endpoint_stats(endpoint, false, false, duration);
        record_fallback(endpoint, FallbackReason::COMPONENT_FAILURE, false);
        
        LOG_ERROR("Fallback to Node.js failed for {}: {}", endpoint, e.what());
        throw;
    }
}

bool FallbackManager::should_fallback(const std::string& endpoint, FallbackReason reason) const {
    if (!config_.enable_automatic_fallback) {
        return false;
    }
    
    // Check if fallback is forced for this endpoint
    {
        std::lock_guard<std::mutex> lock(forced_fallbacks_mutex_);
        if (forced_fallback_endpoints_.count(endpoint) > 0) {
            return true;
        }
    }
    
    // Check if Node.js is available
    if (!stats_.nodejs_available.load()) {
        LOG_DEBUG("Node.js not available, cannot fallback for {}", endpoint);
        return false;
    }
    
    switch (reason) {
        case FallbackReason::CIRCUIT_BREAKER_OPEN:
        case FallbackReason::COMPONENT_FAILURE:
        case FallbackReason::RESOURCE_EXHAUSTION:
            return true;
            
        case FallbackReason::PERFORMANCE_DEGRADATION:
        case FallbackReason::TIMEOUT:
            // Check endpoint-specific performance history
            {
                std::lock_guard<std::mutex> lock(endpoint_stats_mutex_);
                auto it = endpoint_stats_.find(endpoint);
                if (it != endpoint_stats_.end()) {
                    const auto& stats = it->second;
                    // Fallback if C++ failure rate is high
                    if (stats.cpp_requests > 10) {
                        double failure_rate = static_cast<double>(stats.cpp_failures) / stats.cpp_requests;
                        return failure_rate > 0.1; // 10% failure rate threshold
                    }
                }
            }
            return true;
            
        case FallbackReason::MANUAL_OVERRIDE:
            return true;
            
        default:
            return false;
    }
}

void FallbackManager::force_fallback(const std::string& endpoint, bool enable) {
    std::lock_guard<std::mutex> lock(forced_fallbacks_mutex_);
    
    if (enable) {
        forced_fallback_endpoints_.insert(endpoint);
        LOG_INFO("Forced fallback enabled for endpoint: {}", endpoint);
    } else {
        forced_fallback_endpoints_.erase(endpoint);
        LOG_INFO("Forced fallback disabled for endpoint: {}", endpoint);
    }
}

void FallbackManager::reset_fallback_state(const std::string& endpoint) {
    {
        std::lock_guard<std::mutex> lock(forced_fallbacks_mutex_);
        forced_fallback_endpoints_.erase(endpoint);
    }
    
    {
        std::lock_guard<std::mutex> lock(endpoint_stats_mutex_);
        endpoint_stats_.erase(endpoint);
    }
    
    LOG_INFO("Reset fallback state for endpoint: {}", endpoint);
}

void FallbackManager::perform_health_check() {
    auto start_time = std::chrono::steady_clock::now();
    
    try {
        NodeJSClient client(config_.nodejs_base_url);
        bool is_healthy = client.is_healthy(config_.health_check_timeout);
        
        stats_.health_checks_performed.fetch_add(1);
        
        bool was_available = stats_.nodejs_available.exchange(is_healthy);
        
        if (was_available != is_healthy) {
            LOG_INFO("Node.js health status changed: {} -> {}", 
                    was_available ? "healthy" : "unhealthy",
                    is_healthy ? "healthy" : "unhealthy");
            
            // Notify callbacks
            for (auto& callback : health_check_callbacks_) {
                try {
                    callback(is_healthy);
                } catch (const std::exception& e) {
                    LOG_ERROR("Exception in health check callback: {}", e.what());
                }
            }
        }
        
        if (!is_healthy) {
            stats_.health_check_failures.fetch_add(1);
        }
        
    } catch (const std::exception& e) {
        stats_.health_checks_performed.fetch_add(1);
        stats_.health_check_failures.fetch_add(1);
        stats_.nodejs_available.store(false);
        
        LOG_ERROR("Health check failed: {}", e.what());
    }
}

void FallbackManager::register_health_check_callback(HealthCheckCallback callback) {
    health_check_callbacks_.push_back(std::move(callback));
}

std::unordered_map<std::string, FallbackManager::EndpointStats> FallbackManager::get_endpoint_stats() const {
    std::lock_guard<std::mutex> lock(endpoint_stats_mutex_);
    return endpoint_stats_;
}

void FallbackManager::health_monitoring_loop() {
    while (health_monitoring_active_.load()) {
        perform_health_check();
        std::this_thread::sleep_for(config_.health_check_interval);
    }
}

bool FallbackManager::should_fallback_due_to_performance(uint64_t response_time_ns) const {
    return response_time_ns > config_.max_response_time_ns;
}

void FallbackManager::record_fallback(const std::string& endpoint, FallbackReason reason, bool success) {
    stats_.total_fallbacks.fetch_add(1);
    
    if (success) {
        stats_.successful_fallbacks.fetch_add(1);
    } else {
        stats_.failed_fallbacks.fetch_add(1);
    }
    
    LOG_INFO("Fallback recorded for {}: {} ({})", 
             endpoint, reason_to_string(reason), success ? "success" : "failure");
}

void FallbackManager::update_endpoint_stats(const std::string& endpoint, bool is_cpp, bool success, uint64_t response_time_ns) {
    std::lock_guard<std::mutex> lock(endpoint_stats_mutex_);
    
    auto& stats = endpoint_stats_[endpoint];
    
    if (is_cpp) {
        stats.cpp_requests++;
        if (!success) {
            stats.cpp_failures++;
        }
        // Update average response time (simple moving average)
        if (stats.cpp_requests == 1) {
            stats.avg_cpp_response_time_ns = response_time_ns;
        } else {
            stats.avg_cpp_response_time_ns = 
                (stats.avg_cpp_response_time_ns * (stats.cpp_requests - 1) + response_time_ns) / stats.cpp_requests;
        }
    } else {
        stats.fallback_requests++;
        if (!success) {
            stats.fallback_failures++;
        }
        // Update average fallback response time
        if (stats.fallback_requests == 1) {
            stats.avg_fallback_response_time_ns = response_time_ns;
        } else {
            stats.avg_fallback_response_time_ns = 
                (stats.avg_fallback_response_time_ns * (stats.fallback_requests - 1) + response_time_ns) / stats.fallback_requests;
        }
    }
}

std::string FallbackManager::make_http_request(const std::string& url, const std::string& data, 
                                              std::chrono::milliseconds timeout) {
    CURL* curl = curl_easy_init();
    if (!curl) {
        throw std::runtime_error("Failed to initialize CURL");
    }
    
    std::string response_data;
    
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_data);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, timeout.count());
    
    if (!data.empty()) {
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, data.c_str());
        curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, data.length());
    }
    
    CURLcode res = curl_easy_perform(curl);
    
    if (res != CURLE_OK) {
        curl_easy_cleanup(curl);
        throw std::runtime_error("CURL request failed: " + std::string(curl_easy_strerror(res)));
    }
    
    curl_easy_cleanup(curl);
    return response_data;
}

std::string FallbackManager::reason_to_string(FallbackReason reason) {
    switch (reason) {
        case FallbackReason::CIRCUIT_BREAKER_OPEN: return "CIRCUIT_BREAKER_OPEN";
        case FallbackReason::PERFORMANCE_DEGRADATION: return "PERFORMANCE_DEGRADATION";
        case FallbackReason::COMPONENT_FAILURE: return "COMPONENT_FAILURE";
        case FallbackReason::RESOURCE_EXHAUSTION: return "RESOURCE_EXHAUSTION";
        case FallbackReason::TIMEOUT: return "TIMEOUT";
        case FallbackReason::MANUAL_OVERRIDE: return "MANUAL_OVERRIDE";
        default: return "UNKNOWN";
    }
}

// NodeJSClient Implementation
NodeJSClient::NodeJSClient(const std::string& base_url) : base_url_(base_url) {
    // Ensure base URL ends with /
    if (!base_url_.empty() && base_url_.back() != '/') {
        base_url_ += '/';
    }
}

NodeJSClient::Response NodeJSClient::get(const std::string& path, std::chrono::milliseconds timeout) {
    return make_request("GET", path, "", timeout);
}

NodeJSClient::Response NodeJSClient::post(const std::string& path, const std::string& data, 
                                         std::chrono::milliseconds timeout) {
    return make_request("POST", path, data, timeout);
}

bool NodeJSClient::is_healthy(std::chrono::milliseconds timeout) {
    try {
        auto response = get("health", timeout);
        return response.status_code >= 200 && response.status_code < 300;
    } catch (const std::exception&) {
        return false;
    }
}

NodeJSClient::Response NodeJSClient::make_request(const std::string& method, const std::string& path, 
                                                 const std::string& data, std::chrono::milliseconds timeout) {
    CURL* curl = curl_easy_init();
    if (!curl) {
        throw std::runtime_error("Failed to initialize CURL");
    }
    
    std::string url = base_url_ + path;
    std::string response_data;
    Response response;
    
    auto start_time = std::chrono::steady_clock::now();
    
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_data);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, timeout.count());
    
    if (method == "POST") {
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, data.c_str());
        curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, data.length());
        
        struct curl_slist* headers = nullptr;
        headers = curl_slist_append(headers, "Content-Type: application/json");
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    }
    
    CURLcode res = curl_easy_perform(curl);
    
    auto end_time = std::chrono::steady_clock::now();
    response.response_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    if (res != CURLE_OK) {
        curl_easy_cleanup(curl);
        throw std::runtime_error("CURL request failed: " + std::string(curl_easy_strerror(res)));
    }
    
    long status_code;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &status_code);
    
    response.status_code = static_cast<int>(status_code);
    response.body = response_data;
    
    curl_easy_cleanup(curl);
    return response;
}

} // namespace common
} // namespace ultra