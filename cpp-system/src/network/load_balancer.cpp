#include "network/load_balancer.hpp"
#ifdef ULTRA_LOGGER_AVAILABLE
    #include "common/logger.hpp"
#else
    #include "common/simple_logger.hpp"
#endif
#include <algorithm>
#include <random>
#include <chrono>
#include <functional>

namespace ultra::network {

// ConsistentHashRing Implementation
ConsistentHashRing::ConsistentHashRing(u32 virtual_nodes_per_backend) 
    : virtual_nodes_per_backend_(virtual_nodes_per_backend) {
}

void ConsistentHashRing::add_backend(std::shared_ptr<Backend> backend) {
    std::unique_lock lock(mutex_);
    
    backends_[backend->get_key()] = backend;
    rebuild_ring();
}

void ConsistentHashRing::remove_backend(const std::string& backend_key) {
    std::unique_lock lock(mutex_);
    
    backends_.erase(backend_key);
    rebuild_ring();
}

void ConsistentHashRing::update_backend_weight(const std::string& backend_key, u32 new_weight) {
    std::unique_lock lock(mutex_);
    
    auto it = backends_.find(backend_key);
    if (it != backends_.end()) {
        it->second->weight = new_weight;
        rebuild_ring();
    }
}

std::shared_ptr<Backend> ConsistentHashRing::get_backend(const std::string& key) const {
    u64 hash = compute_hash(key);
    return get_backend_for_hash(hash);
}

std::shared_ptr<Backend> ConsistentHashRing::get_backend_for_hash(u64 hash) const {
    std::shared_lock lock(mutex_);
    
    if (ring_.empty()) {
        return nullptr;
    }
    
    // Binary search for the first virtual node with hash >= target hash
    auto it = std::lower_bound(ring_.begin(), ring_.end(), hash,
        [](const VirtualNode& node, u64 target_hash) {
            return node.hash < target_hash;
        });
    
    // Wrap around to the beginning if we've gone past the end
    if (it == ring_.end()) {
        it = ring_.begin();
    }
    
    return it->backend;
}

std::vector<std::shared_ptr<Backend>> ConsistentHashRing::get_all_backends() const {
    std::shared_lock lock(mutex_);
    
    std::vector<std::shared_ptr<Backend>> result;
    result.reserve(backends_.size());
    
    for (const auto& [key, backend] : backends_) {
        result.push_back(backend);
    }
    
    return result;
}

u64 ConsistentHashRing::compute_hash(const std::string& key) const {
    // FNV-1a hash function
    u64 hash = 14695981039346656037ULL;
    for (char c : key) {
        hash ^= static_cast<u64>(c);
        hash *= 1099511628211ULL;
    }
    return hash;
}

void ConsistentHashRing::rebuild_ring() {
    ring_.clear();
    
    for (const auto& [key, backend] : backends_) {
        if (backend->state != Backend::State::HEALTHY) {
            continue; // Skip unhealthy backends
        }
        
        // Create virtual nodes based on weight
        u32 num_virtual_nodes = (virtual_nodes_per_backend_ * backend->weight) / 100;
        num_virtual_nodes = std::max(1u, num_virtual_nodes); // At least 1 virtual node
        
        for (u32 i = 0; i < num_virtual_nodes; ++i) {
            std::string virtual_key = key + ":" + std::to_string(i);
            u64 hash = compute_hash(virtual_key);
            ring_.emplace_back(VirtualNode{hash, backend});
        }
    }
    
    // Sort ring by hash value
    std::sort(ring_.begin(), ring_.end());
}

// HealthChecker Implementation
HealthChecker::HealthChecker(const HealthCheckConfig& config) 
    : config_(config) {
}

HealthChecker::~HealthChecker() {
    stop();
}

void HealthChecker::start() {
    if (running_.exchange(true)) {
        return; // Already running
    }
    
    health_check_thread_ = std::thread(&HealthChecker::health_check_loop, this);
}

void HealthChecker::stop() {
    if (!running_.exchange(false)) {
        return; // Not running
    }
    
    if (health_check_thread_.joinable()) {
        health_check_thread_.join();
    }
}

void HealthChecker::add_backend(std::shared_ptr<Backend> backend) {
    std::unique_lock lock(backends_mutex_);
    backends_.push_back(backend);
}

void HealthChecker::remove_backend(const std::string& backend_key) {
    std::unique_lock lock(backends_mutex_);
    
    backends_.erase(
        std::remove_if(backends_.begin(), backends_.end(),
            [&backend_key](const std::weak_ptr<Backend>& weak_backend) {
                auto backend = weak_backend.lock();
                return !backend || backend->get_key() == backend_key;
            }),
        backends_.end());
}

void HealthChecker::health_check_loop() {
    while (running_) {
        auto start_time = std::chrono::steady_clock::now();
        
        // Get copy of backends to check
        std::vector<std::shared_ptr<Backend>> backends_to_check;
        {
            std::shared_lock lock(backends_mutex_);
            backends_to_check.reserve(backends_.size());
            
            for (auto it = backends_.begin(); it != backends_.end();) {
                auto backend = it->lock();
                if (backend) {
                    backends_to_check.push_back(backend);
                    ++it;
                } else {
                    // Remove expired weak_ptr
                    it = backends_.erase(it);
                }
            }
        }
        
        // Perform health checks
        for (auto& backend : backends_to_check) {
            bool check_passed = perform_health_check(backend);
            update_backend_state(backend, check_passed);
        }
        
        // Sleep until next check interval
        auto end_time = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        auto sleep_time = std::chrono::milliseconds(config_.interval_ms) - elapsed;
        
        if (sleep_time > std::chrono::milliseconds(0)) {
            std::this_thread::sleep_for(sleep_time);
        }
    }
}

bool HealthChecker::perform_health_check(std::shared_ptr<Backend> backend) {
    auto start_time = std::chrono::steady_clock::now();
    
    bool result = false;
    
    try {
        switch (config_.type) {
            case HealthCheckType::TCP_CONNECT:
                result = tcp_health_check(backend);
                break;
                
            case HealthCheckType::HTTP_GET:
            case HealthCheckType::HTTP_POST:
                result = http_health_check(backend);
                break;
                
            case HealthCheckType::CUSTOM:
                if (custom_health_check_) {
                    result = custom_health_check_(backend);
                } else {
                    result = tcp_health_check(backend); // Fallback
                }
                break;
        }
    } catch (const std::exception& e) {
        ULTRA_LOG_ERROR("Health check exception for {}: {}", backend->get_key(), e.what());
        result = false;
    }
    
    auto end_time = std::chrono::steady_clock::now();
    auto response_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    // Update statistics
    backend->health_stats.total_checks.fetch_add(1);
    backend->health_stats.last_check_time.store(
        std::chrono::duration_cast<std::chrono::nanoseconds>(end_time.time_since_epoch()).count());
    backend->health_stats.response_time_ms.store(response_time.count());
    
    if (result) {
        backend->health_stats.successful_checks.fetch_add(1);
    } else {
        backend->health_stats.failed_checks.fetch_add(1);
    }
    
    return result;
}

bool HealthChecker::tcp_health_check(std::shared_ptr<Backend> backend) {
    TcpConnection::Config conn_config;
    conn_config.connect_timeout_ms = config_.timeout_ms;
    
    TcpConnection connection;
    return connection.connect(backend->host, backend->port, conn_config);
}

bool HealthChecker::http_health_check(std::shared_ptr<Backend> backend) {
    // Simplified HTTP health check - in a real implementation,
    // this would make a proper HTTP request
    return tcp_health_check(backend);
}

void HealthChecker::update_backend_state(std::shared_ptr<Backend> backend, bool check_passed) {
    Backend::State old_state = backend->state;
    Backend::State new_state = old_state;
    
    if (check_passed) {
        backend->health_stats.consecutive_failures.store(0);
        
        if (old_state == Backend::State::UNHEALTHY) {
            // Check if we have enough consecutive successes to mark as healthy
            u32 successful_checks = backend->health_stats.successful_checks.load();
            u32 total_checks = backend->health_stats.total_checks.load();
            
            if (total_checks >= config_.healthy_threshold) {
                u32 recent_successes = 0;
                // In a real implementation, we'd track recent check results
                // For now, assume if we have any recent success, mark as healthy
                if (successful_checks > 0) {
                    new_state = Backend::State::HEALTHY;
                }
            }
        }
    } else {
        u32 consecutive_failures = backend->health_stats.consecutive_failures.fetch_add(1) + 1;
        
        if (consecutive_failures >= config_.unhealthy_threshold) {
            new_state = Backend::State::UNHEALTHY;
        }
    }
    
    if (new_state != old_state) {
        backend->state = new_state;
        
        if (state_changed_callback_) {
            state_changed_callback_(backend, old_state, new_state);
        }
        
        ULTRA_LOG_INFO("Backend {} state changed from {} to {}", 
                      backend->get_key(), 
                      static_cast<int>(old_state), 
                      static_cast<int>(new_state));
    }
}

// LoadBalancer Implementation
LoadBalancer::LoadBalancer(const Config& config) 
    : config_(config) {
    
    if (config_.algorithm == LoadBalancingAlgorithm::CONSISTENT_HASH) {
        consistent_hash_ring_ = std::make_unique<ConsistentHashRing>(config_.virtual_nodes_per_backend);
    }
    
    if (config_.enable_health_checks) {
        health_checker_ = std::make_unique<HealthChecker>(config_.health_check_config);
        health_checker_->set_backend_state_changed_callback(
            [this](auto backend, auto old_state, auto new_state) {
                on_backend_state_changed(backend, old_state, new_state);
            });
        health_checker_->start();
    }
    
    if (config_.enable_session_affinity) {
        session_cleanup_running_ = true;
        session_cleanup_thread_ = std::thread(&LoadBalancer::session_cleanup_loop, this);
    }
}

LoadBalancer::~LoadBalancer() {
    if (health_checker_) {
        health_checker_->stop();
    }
    
    if (session_cleanup_running_) {
        session_cleanup_running_ = false;
        if (session_cleanup_thread_.joinable()) {
            session_cleanup_thread_.join();
        }
    }
}

void LoadBalancer::add_backend(const std::string& host, u16 port, u32 weight) {
    auto backend = std::make_shared<Backend>(host, port, weight);
    
    {
        std::unique_lock lock(backends_mutex_);
        backends_.push_back(backend);
        
        if (consistent_hash_ring_) {
            consistent_hash_ring_->add_backend(backend);
        }
    }
    
    if (health_checker_) {
        health_checker_->add_backend(backend);
    }
    
    ULTRA_LOG_INFO("Added backend {}:{} with weight {}", host, port, weight);
}

void LoadBalancer::remove_backend(const std::string& host, u16 port) {
    std::string backend_key = host + ":" + std::to_string(port);
    
    {
        std::unique_lock lock(backends_mutex_);
        
        backends_.erase(
            std::remove_if(backends_.begin(), backends_.end(),
                [&backend_key](const std::shared_ptr<Backend>& backend) {
                    return backend->get_key() == backend_key;
                }),
            backends_.end());
        
        if (consistent_hash_ring_) {
            consistent_hash_ring_->remove_backend(backend_key);
        }
    }
    
    if (health_checker_) {
        health_checker_->remove_backend(backend_key);
    }
    
    ULTRA_LOG_INFO("Removed backend {}", backend_key);
}

std::shared_ptr<Backend> LoadBalancer::select_backend(const std::string& client_key) const {
    stats_.total_requests.fetch_add(1);
    
    std::shared_ptr<Backend> selected_backend;
    
    switch (config_.algorithm) {
        case LoadBalancingAlgorithm::ROUND_ROBIN:
            selected_backend = select_round_robin();
            break;
            
        case LoadBalancingAlgorithm::WEIGHTED_ROUND_ROBIN:
            selected_backend = select_weighted_round_robin();
            break;
            
        case LoadBalancingAlgorithm::LEAST_CONNECTIONS:
            selected_backend = select_least_connections();
            break;
            
        case LoadBalancingAlgorithm::WEIGHTED_LEAST_CONNECTIONS:
            selected_backend = select_weighted_least_connections();
            break;
            
        case LoadBalancingAlgorithm::CONSISTENT_HASH:
            selected_backend = select_consistent_hash(client_key);
            break;
            
        case LoadBalancingAlgorithm::LEAST_RESPONSE_TIME:
            selected_backend = select_least_response_time();
            break;
            
        case LoadBalancingAlgorithm::RANDOM:
            selected_backend = select_random();
            break;
            
        case LoadBalancingAlgorithm::IP_HASH:
            selected_backend = select_ip_hash(client_key);
            break;
    }
    
    if (selected_backend) {
        selected_backend->load_stats.total_requests.fetch_add(1);
        selected_backend->load_stats.last_request_time.store(
            std::chrono::high_resolution_clock::now().time_since_epoch().count());
        stats_.successful_requests.fetch_add(1);
    } else {
        stats_.failed_requests.fetch_add(1);
    }
    
    return selected_backend;
}

std::shared_ptr<Backend> LoadBalancer::select_round_robin() const {
    auto healthy_backends = get_healthy_backends_internal();
    if (healthy_backends.empty()) {
        return nullptr;
    }
    
    size_t index = round_robin_index_.fetch_add(1) % healthy_backends.size();
    return healthy_backends[index];
}

std::shared_ptr<Backend> LoadBalancer::select_consistent_hash(const std::string& key) const {
    if (!consistent_hash_ring_) {
        return select_round_robin(); // Fallback
    }
    
    return consistent_hash_ring_->get_backend(key);
}

std::shared_ptr<Backend> LoadBalancer::select_least_connections() const {
    auto healthy_backends = get_healthy_backends_internal();
    if (healthy_backends.empty()) {
        return nullptr;
    }
    
    auto min_backend = std::min_element(healthy_backends.begin(), healthy_backends.end(),
        [](const std::shared_ptr<Backend>& a, const std::shared_ptr<Backend>& b) {
            return a->load_stats.active_connections.load() < b->load_stats.active_connections.load();
        });
    
    return *min_backend;
}

std::shared_ptr<Backend> LoadBalancer::select_random() const {
    auto healthy_backends = get_healthy_backends_internal();
    if (healthy_backends.empty()) {
        return nullptr;
    }
    
    static thread_local std::random_device rd;
    static thread_local std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> dist(0, healthy_backends.size() - 1);
    
    return healthy_backends[dist(gen)];
}

std::vector<std::shared_ptr<Backend>> LoadBalancer::get_healthy_backends_internal() const {
    std::shared_lock lock(backends_mutex_);
    
    std::vector<std::shared_ptr<Backend>> healthy_backends;
    for (const auto& backend : backends_) {
        if (backend->state == Backend::State::HEALTHY) {
            healthy_backends.push_back(backend);
        }
    }
    
    return healthy_backends;
}

void LoadBalancer::on_backend_state_changed(std::shared_ptr<Backend> backend,
                                          Backend::State old_state, Backend::State new_state) {
    if (consistent_hash_ring_) {
        if (new_state == Backend::State::HEALTHY && old_state != Backend::State::HEALTHY) {
            consistent_hash_ring_->add_backend(backend);
        } else if (new_state != Backend::State::HEALTHY && old_state == Backend::State::HEALTHY) {
            consistent_hash_ring_->remove_backend(backend->get_key());
        }
    }
}

void LoadBalancer::session_cleanup_loop() {
    while (session_cleanup_running_) {
        cleanup_expired_sessions();
        std::this_thread::sleep_for(std::chrono::minutes(1)); // Cleanup every minute
    }
}

void LoadBalancer::cleanup_expired_sessions() {
    auto now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    
    std::unique_lock lock(sessions_mutex_);
    
    for (auto it = sessions_.begin(); it != sessions_.end();) {
        if (now - it->second.last_accessed > config_.session_timeout_ms * 1000000ULL) {
            it = sessions_.erase(it);
        } else {
            ++it;
        }
    }
}

// Placeholder implementations for other selection algorithms
std::shared_ptr<Backend> LoadBalancer::select_weighted_round_robin() const {
    return select_round_robin(); // Simplified implementation
}

std::shared_ptr<Backend> LoadBalancer::select_weighted_least_connections() const {
    return select_least_connections(); // Simplified implementation
}

std::shared_ptr<Backend> LoadBalancer::select_least_response_time() const {
    return select_least_connections(); // Simplified implementation
}

std::shared_ptr<Backend> LoadBalancer::select_ip_hash(const std::string& ip) const {
    return select_consistent_hash(ip);
}

LoadBalancer::LoadBalancerStats LoadBalancer::get_stats() const {
    return stats_;
}

std::vector<std::shared_ptr<Backend>> LoadBalancer::get_backends() const {
    std::shared_lock lock(backends_mutex_);
    return backends_;
}

std::vector<std::shared_ptr<Backend>> LoadBalancer::get_healthy_backends() const {
    return get_healthy_backends_internal();
}

} // namespace ultra::network