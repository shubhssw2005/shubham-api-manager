#pragma once

#include "common/types.hpp"
#include "network/tcp_connection.hpp"
#include <string>
#include <vector>
#include <memory>
#include <atomic>
#include <chrono>
#include <functional>

namespace ultra::network {

// Backend server representation
struct Backend {
    std::string host;
    u16 port;
    u32 weight = 100;
    
    enum class State {
        HEALTHY,
        UNHEALTHY,
        DRAINING,
        DISABLED
    } state = State::HEALTHY;
    
    // Health check statistics
    struct HealthStats {
        aligned_atomic<u64> total_checks{0};
        aligned_atomic<u64> successful_checks{0};
        aligned_atomic<u64> failed_checks{0};
        aligned_atomic<timestamp_t> last_check_time{};
        aligned_atomic<u32> consecutive_failures{0};
        aligned_atomic<u32> response_time_ms{0};
    } health_stats;
    
    // Load balancing statistics
    struct LoadStats {
        aligned_atomic<u64> active_connections{0};
        aligned_atomic<u64> total_requests{0};
        aligned_atomic<u64> failed_requests{0};
        aligned_atomic<u64> total_response_time_ms{0};
        aligned_atomic<timestamp_t> last_request_time{};
    } load_stats;
    
    Backend(std::string h, u16 p, u32 w = 100) 
        : host(std::move(h)), port(p), weight(w) {}
    
    std::string get_key() const {
        return host + ":" + std::to_string(port);
    }
    
    double get_average_response_time() const {
        u64 total_requests = load_stats.total_requests.load();
        if (total_requests == 0) return 0.0;
        return static_cast<double>(load_stats.total_response_time_ms.load()) / total_requests;
    }
    
    double get_success_rate() const {
        u64 total_requests = load_stats.total_requests.load();
        if (total_requests == 0) return 1.0;
        u64 successful = total_requests - load_stats.failed_requests.load();
        return static_cast<double>(successful) / total_requests;
    }
};

// Load balancing algorithms
enum class LoadBalancingAlgorithm {
    ROUND_ROBIN,
    WEIGHTED_ROUND_ROBIN,
    LEAST_CONNECTIONS,
    WEIGHTED_LEAST_CONNECTIONS,
    CONSISTENT_HASH,
    LEAST_RESPONSE_TIME,
    RANDOM,
    IP_HASH
};

// Consistent hashing implementation
class ConsistentHashRing {
public:
    struct VirtualNode {
        u64 hash;
        std::shared_ptr<Backend> backend;
        
        bool operator<(const VirtualNode& other) const {
            return hash < other.hash;
        }
    };
    
    explicit ConsistentHashRing(u32 virtual_nodes_per_backend = 160);
    
    void add_backend(std::shared_ptr<Backend> backend);
    void remove_backend(const std::string& backend_key);
    void update_backend_weight(const std::string& backend_key, u32 new_weight);
    
    std::shared_ptr<Backend> get_backend(const std::string& key) const;
    std::shared_ptr<Backend> get_backend_for_hash(u64 hash) const;
    
    std::vector<std::shared_ptr<Backend>> get_all_backends() const;
    size_t get_backend_count() const { return backends_.size(); }
    
private:
    u32 virtual_nodes_per_backend_;
    std::vector<VirtualNode> ring_;
    std::unordered_map<std::string, std::shared_ptr<Backend>> backends_;
    mutable std::shared_mutex mutex_;
    
    u64 compute_hash(const std::string& key) const;
    void rebuild_ring();
};

// Health check system
class HealthChecker {
public:
    enum class HealthCheckType {
        TCP_CONNECT,
        HTTP_GET,
        HTTP_POST,
        CUSTOM
    };
    
    struct HealthCheckConfig {
        HealthCheckType type = HealthCheckType::TCP_CONNECT;
        std::string path = "/health";
        std::string expected_response = "OK";
        u32 timeout_ms = 5000;
        u32 interval_ms = 30000;
        u32 unhealthy_threshold = 3;
        u32 healthy_threshold = 2;
        std::string custom_check_data;
        std::function<bool(const std::string&)> custom_validator;
    };
    
    explicit HealthChecker(const HealthCheckConfig& config);
    ~HealthChecker();
    
    void start();
    void stop();
    
    void add_backend(std::shared_ptr<Backend> backend);
    void remove_backend(const std::string& backend_key);
    
    // Custom health check function
    void set_custom_health_check(std::function<bool(std::shared_ptr<Backend>)> check_func);
    
    // Health check callbacks
    void set_backend_state_changed_callback(
        std::function<void(std::shared_ptr<Backend>, Backend::State, Backend::State)> callback);
    
private:
    HealthCheckConfig config_;
    std::vector<std::shared_ptr<Backend>> backends_;
    std::shared_mutex backends_mutex_;
    
    std::thread health_check_thread_;
    std::atomic<bool> running_{false};
    
    std::function<bool(std::shared_ptr<Backend>)> custom_health_check_;
    std::function<void(std::shared_ptr<Backend>, Backend::State, Backend::State)> state_changed_callback_;
    
    void health_check_loop();
    bool perform_health_check(std::shared_ptr<Backend> backend);
    bool tcp_health_check(std::shared_ptr<Backend> backend);
    bool http_health_check(std::shared_ptr<Backend> backend);
    
    void update_backend_state(std::shared_ptr<Backend> backend, bool check_passed);
};

// Main load balancer class
class LoadBalancer {
public:
    struct Config {
        LoadBalancingAlgorithm algorithm = LoadBalancingAlgorithm::CONSISTENT_HASH;
        u32 virtual_nodes_per_backend = 160;
        bool enable_health_checks = true;
        HealthChecker::HealthCheckConfig health_check_config;
        bool enable_session_affinity = false;
        std::string session_cookie_name = "SESSIONID";
        u32 session_timeout_ms = 1800000; // 30 minutes
    };
    
    explicit LoadBalancer(const Config& config);
    ~LoadBalancer();
    
    // Backend management
    void add_backend(const std::string& host, u16 port, u32 weight = 100);
    void remove_backend(const std::string& host, u16 port);
    void update_backend_weight(const std::string& host, u16 port, u32 weight);
    void set_backend_state(const std::string& host, u16 port, Backend::State state);
    
    // Load balancing
    std::shared_ptr<Backend> select_backend(const std::string& client_key = "") const;
    std::shared_ptr<Backend> select_backend_for_session(const std::string& session_id) const;
    
    // Connection management
    std::shared_ptr<TcpConnection> get_connection(const std::string& client_key = "");
    void return_connection(std::shared_ptr<TcpConnection> conn, std::shared_ptr<Backend> backend);
    
    // Statistics and monitoring
    struct LoadBalancerStats {
        aligned_atomic<u64> total_requests{0};
        aligned_atomic<u64> successful_requests{0};
        aligned_atomic<u64> failed_requests{0};
        aligned_atomic<u64> backend_failures{0};
        aligned_atomic<u64> session_hits{0};
        aligned_atomic<u64> session_misses{0};
    };
    
    LoadBalancerStats get_stats() const;
    std::vector<std::shared_ptr<Backend>> get_backends() const;
    std::vector<std::shared_ptr<Backend>> get_healthy_backends() const;
    
    // Session affinity management
    void create_session(const std::string& session_id, std::shared_ptr<Backend> backend);
    void remove_session(const std::string& session_id);
    std::shared_ptr<Backend> get_session_backend(const std::string& session_id) const;
    
private:
    Config config_;
    LoadBalancerStats stats_;
    
    // Backend storage and selection
    std::vector<std::shared_ptr<Backend>> backends_;
    std::unique_ptr<ConsistentHashRing> consistent_hash_ring_;
    mutable std::shared_mutex backends_mutex_;
    
    // Round-robin state
    mutable std::atomic<size_t> round_robin_index_{0};
    
    // Health checking
    std::unique_ptr<HealthChecker> health_checker_;
    
    // Session affinity
    struct SessionInfo {
        std::shared_ptr<Backend> backend;
        timestamp_t created_at;
        timestamp_t last_accessed;
    };
    
    std::unordered_map<std::string, SessionInfo> sessions_;
    mutable std::shared_mutex sessions_mutex_;
    std::thread session_cleanup_thread_;
    std::atomic<bool> session_cleanup_running_{false};
    
    // Connection pooling per backend
    std::unordered_map<std::string, std::unique_ptr<TcpConnectionPool>> backend_pools_;
    mutable std::shared_mutex pools_mutex_;
    
    // Load balancing algorithm implementations
    std::shared_ptr<Backend> select_round_robin() const;
    std::shared_ptr<Backend> select_weighted_round_robin() const;
    std::shared_ptr<Backend> select_least_connections() const;
    std::shared_ptr<Backend> select_weighted_least_connections() const;
    std::shared_ptr<Backend> select_consistent_hash(const std::string& key) const;
    std::shared_ptr<Backend> select_least_response_time() const;
    std::shared_ptr<Backend> select_random() const;
    std::shared_ptr<Backend> select_ip_hash(const std::string& ip) const;
    
    // Helper functions
    std::vector<std::shared_ptr<Backend>> get_healthy_backends_internal() const;
    void on_backend_state_changed(std::shared_ptr<Backend> backend, 
                                Backend::State old_state, Backend::State new_state);
    void session_cleanup_loop();
    void cleanup_expired_sessions();
    
    TcpConnectionPool* get_or_create_backend_pool(std::shared_ptr<Backend> backend);
    u64 compute_hash(const std::string& key) const;
};

// Advanced load balancer with circuit breaker pattern
class AdvancedLoadBalancer : public LoadBalancer {
public:
    struct CircuitBreakerConfig {
        u32 failure_threshold = 5;
        u32 success_threshold = 3;
        u32 timeout_ms = 60000; // 1 minute
        double failure_rate_threshold = 0.5; // 50%
        u32 min_requests_for_circuit_breaker = 10;
    };
    
    struct AdvancedConfig : public LoadBalancer::Config {
        CircuitBreakerConfig circuit_breaker_config;
        bool enable_circuit_breaker = true;
        bool enable_adaptive_load_balancing = true;
        u32 adaptive_window_ms = 60000; // 1 minute window
    };
    
    explicit AdvancedLoadBalancer(const AdvancedConfig& config);
    
    // Circuit breaker functionality
    bool is_circuit_open(std::shared_ptr<Backend> backend) const;
    void record_success(std::shared_ptr<Backend> backend);
    void record_failure(std::shared_ptr<Backend> backend);
    
    // Adaptive load balancing based on real-time metrics
    std::shared_ptr<Backend> select_adaptive_backend(const std::string& client_key = "") const;
    
private:
    enum class CircuitState {
        CLOSED,
        OPEN,
        HALF_OPEN
    };
    
    struct CircuitBreakerState {
        aligned_atomic<CircuitState> state{CircuitState::CLOSED};
        aligned_atomic<u32> failure_count{0};
        aligned_atomic<u32> success_count{0};
        aligned_atomic<timestamp_t> last_failure_time{};
        aligned_atomic<timestamp_t> state_changed_time{};
        
        // Sliding window for failure rate calculation
        std::vector<bool> recent_results; // true = success, false = failure
        std::atomic<size_t> result_index{0};
        mutable std::mutex results_mutex;
    };
    
    AdvancedConfig advanced_config_;
    std::unordered_map<std::string, std::unique_ptr<CircuitBreakerState>> circuit_breakers_;
    mutable std::shared_mutex circuit_breakers_mutex_;
    
    CircuitBreakerState* get_or_create_circuit_breaker(std::shared_ptr<Backend> backend);
    void update_circuit_breaker_state(std::shared_ptr<Backend> backend, CircuitBreakerState* cb_state);
    double calculate_failure_rate(CircuitBreakerState* cb_state) const;
    double calculate_backend_score(std::shared_ptr<Backend> backend) const;
};

} // namespace ultra::network