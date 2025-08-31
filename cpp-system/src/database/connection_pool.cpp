#include "database/database_connector.hpp"
#include "common/logger.hpp"
#include <algorithm>
#include <random>
#include <thread>

namespace ultra_cpp {
namespace database {

ConnectionPool::ConnectionPool(const Config& config) 
    : config_(config), next_connection_index_(0), lb_strategy_(LoadBalancingStrategy::ROUND_ROBIN) {
}

ConnectionPool::~ConnectionPool() {
    shutdown();
}

bool ConnectionPool::initialize() {
    LOG_INFO("Initializing connection pool with {}-{} connections", 
             config_.min_connections, config_.max_connections);
    
    // Create minimum number of connections
    for (uint32_t i = 0; i < config_.min_connections; ++i) {
        auto connection = create_connection();
        if (!connection || !connection->connect()) {
            LOG_ERROR("Failed to create initial connection {}/{}", i + 1, config_.min_connections);
            return false;
        }
        
        std::lock_guard<std::mutex> lock(pool_mutex_);
        connections_.push_back(connection);
        idle_connections_.push_back(connection);
    }
    
    // Start health check thread
    health_check_thread_ = std::thread(&ConnectionPool::health_check_worker, this);
    
    LOG_INFO("Connection pool initialized with {} connections", connections_.size());
    return true;
}

void ConnectionPool::shutdown() {
    LOG_INFO("Shutting down connection pool");
    
    shutdown_requested_.store(true);
    connection_available_.notify_all();
    
    if (health_check_thread_.joinable()) {
        health_check_thread_.join();
    }
    
    std::lock_guard<std::mutex> lock(pool_mutex_);
    
    // Close all connections
    for (auto& conn : connections_) {
        if (conn) {
            conn->disconnect();
        }
    }
    
    connections_.clear();
    idle_connections_.clear();
    
    LOG_INFO("Connection pool shutdown complete");
}

bool ConnectionPool::is_healthy() const noexcept {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    
    // Pool is healthy if we have at least one working connection
    for (const auto& conn : connections_) {
        if (conn && conn->is_connected()) {
            return true;
        }
    }
    
    return false;
}

std::optional<ConnectionPool::ConnectionHandle> ConnectionPool::acquire_connection(uint32_t timeout_ms) {
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    
    std::unique_lock<std::mutex> lock(pool_mutex_);
    
    while (!shutdown_requested_.load()) {
        // Try to get an idle connection first
        if (!idle_connections_.empty()) {
            auto connection = idle_connections_.back();
            idle_connections_.pop_back();
            
            // Validate the connection
            if (connection && connection->is_connected()) {
                stats_.active_connections++;
                stats_.idle_connections--;
                return ConnectionHandle(connection, this);
            } else {
                // Remove invalid connection
                auto it = std::find(connections_.begin(), connections_.end(), connection);
                if (it != connections_.end()) {
                    connections_.erase(it);
                    stats_.failed_connections++;
                }
                continue;
            }
        }
        
        // Try to create a new connection if under limit
        if (connections_.size() < config_.max_connections) {
            lock.unlock();
            auto new_connection = create_connection();
            lock.lock();
            
            if (new_connection && new_connection->connect()) {
                connections_.push_back(new_connection);
                stats_.active_connections++;
                stats_.total_connections++;
                return ConnectionHandle(new_connection, this);
            } else {
                stats_.failed_connections++;
            }
        }
        
        // Wait for a connection to become available
        if (std::chrono::steady_clock::now() >= deadline) {
            LOG_WARN("Connection acquisition timeout after {}ms", timeout_ms);
            return std::nullopt;
        }
        
        connection_available_.wait_until(lock, deadline);
    }
    
    return std::nullopt;
}

void ConnectionPool::release_connection(std::shared_ptr<DatabaseConnector> connection) {
    if (!connection) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(pool_mutex_);
    
    // Validate connection before returning to pool
    if (connection->is_connected() && validate_connection(connection)) {
        idle_connections_.push_back(connection);
        stats_.active_connections--;
        stats_.idle_connections++;
        connection_available_.notify_one();
    } else {
        // Remove invalid connection from pool
        auto it = std::find(connections_.begin(), connections_.end(), connection);
        if (it != connections_.end()) {
            connections_.erase(it);
            stats_.failed_connections++;
            stats_.total_connections--;
        }
        
        // Try to create a replacement connection if below minimum
        if (connections_.size() < config_.min_connections) {
            auto new_connection = create_connection();
            if (new_connection && new_connection->connect()) {
                connections_.push_back(new_connection);
                idle_connections_.push_back(new_connection);
                stats_.total_connections++;
                stats_.idle_connections++;
            }
        }
    }
}

DatabaseConnector::QueryResult ConnectionPool::execute_query(const std::string& query, uint32_t timeout_ms) {
    auto connection_handle = acquire_connection(timeout_ms);
    if (!connection_handle) {
        return {false, "Failed to acquire database connection", {}, 0, 0};
    }
    
    auto start_time = std::chrono::steady_clock::now();
    auto result = connection_handle->execute_query(query);
    auto end_time = std::chrono::steady_clock::now();
    
    // Update statistics
    stats_.total_queries++;
    if (!result.success) {
        stats_.failed_queries++;
    }
    
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    stats_.average_query_time_ms = (stats_.average_query_time_ms * (stats_.total_queries - 1) + duration_ms) / stats_.total_queries;
    
    return result;
}

DatabaseConnector::QueryResult ConnectionPool::execute_prepared(const std::string& statement_name,
                                                               const std::vector<std::string>& params,
                                                               uint32_t timeout_ms) {
    auto connection_handle = acquire_connection(timeout_ms);
    if (!connection_handle) {
        return {false, "Failed to acquire database connection", {}, 0, 0};
    }
    
    auto start_time = std::chrono::steady_clock::now();
    auto result = connection_handle->execute_prepared(statement_name, params);
    auto end_time = std::chrono::steady_clock::now();
    
    // Update statistics
    stats_.total_queries++;
    if (!result.success) {
        stats_.failed_queries++;
    }
    
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    stats_.average_query_time_ms = (stats_.average_query_time_ms * (stats_.total_queries - 1) + duration_ms) / stats_.total_queries;
    
    return result;
}

std::future<DatabaseConnector::QueryResult> ConnectionPool::execute_query_async(const std::string& query) {
    return std::async(std::launch::async, [this, query]() {
        return execute_query(query);
    });
}

std::future<DatabaseConnector::QueryResult> ConnectionPool::execute_prepared_async(
    const std::string& statement_name, const std::vector<std::string>& params) {
    
    return std::async(std::launch::async, [this, statement_name, params]() {
        return execute_prepared(statement_name, params);
    });
}

ConnectionPool::PoolStats ConnectionPool::get_stats() const noexcept {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    
    PoolStats current_stats = stats_;
    current_stats.total_connections = connections_.size();
    current_stats.idle_connections = idle_connections_.size();
    current_stats.queue_depth = 0; // Would need additional tracking for waiting requests
    
    return current_stats;
}

void ConnectionPool::set_load_balancing_strategy(LoadBalancingStrategy strategy) {
    lb_strategy_ = strategy;
    LOG_INFO("Load balancing strategy changed to {}", static_cast<int>(strategy));
}

std::shared_ptr<DatabaseConnector> ConnectionPool::create_connection() {
    auto connection = std::make_shared<DatabaseConnector>(config_.db_config);
    return connection;
}

void ConnectionPool::health_check_worker() {
    LOG_INFO("Health check worker started");
    
    while (!shutdown_requested_.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(config_.health_check_interval_ms));
        
        if (shutdown_requested_.load()) {
            break;
        }
        
        std::lock_guard<std::mutex> lock(pool_mutex_);
        
        // Check all connections
        auto it = connections_.begin();
        while (it != connections_.end()) {
            auto& connection = *it;
            
            if (!connection || !validate_connection(connection)) {
                LOG_WARN("Removing unhealthy connection from pool");
                
                // Remove from idle connections if present
                auto idle_it = std::find(idle_connections_.begin(), idle_connections_.end(), connection);
                if (idle_it != idle_connections_.end()) {
                    idle_connections_.erase(idle_it);
                    stats_.idle_connections--;
                }
                
                it = connections_.erase(it);
                stats_.failed_connections++;
                stats_.total_connections--;
            } else {
                ++it;
            }
        }
        
        // Ensure minimum connections
        while (connections_.size() < config_.min_connections) {
            auto new_connection = create_connection();
            if (new_connection && new_connection->connect()) {
                connections_.push_back(new_connection);
                idle_connections_.push_back(new_connection);
                stats_.total_connections++;
                stats_.idle_connections++;
                LOG_INFO("Added new connection to maintain minimum pool size");
            } else {
                LOG_ERROR("Failed to create replacement connection");
                break;
            }
        }
        
        // Cleanup idle connections that exceed idle timeout
        cleanup_idle_connections();
    }
    
    LOG_INFO("Health check worker stopped");
}

std::shared_ptr<DatabaseConnector> ConnectionPool::select_connection() {
    if (connections_.empty()) {
        return nullptr;
    }
    
    switch (lb_strategy_) {
        case LoadBalancingStrategy::ROUND_ROBIN: {
            uint32_t index = next_connection_index_.fetch_add(1) % connections_.size();
            return connections_[index];
        }
        
        case LoadBalancingStrategy::RANDOM: {
            static thread_local std::random_device rd;
            static thread_local std::mt19937 gen(rd());
            std::uniform_int_distribution<> dis(0, connections_.size() - 1);
            return connections_[dis(gen)];
        }
        
        case LoadBalancingStrategy::LEAST_CONNECTIONS:
            // For simplicity, fall back to round robin
            // In a full implementation, we'd track active connections per connection
            [[fallthrough]];
            
        case LoadBalancingStrategy::WEIGHTED_ROUND_ROBIN:
            // For simplicity, fall back to round robin
            // In a full implementation, we'd support connection weights
            [[fallthrough]];
            
        default: {
            uint32_t index = next_connection_index_.fetch_add(1) % connections_.size();
            return connections_[index];
        }
    }
}

void ConnectionPool::cleanup_idle_connections() {
    auto now = std::chrono::steady_clock::now();
    auto timeout = std::chrono::milliseconds(config_.connection_idle_timeout_ms);
    
    // This is a simplified cleanup - in a full implementation,
    // we'd track the last used time for each connection
    if (idle_connections_.size() > config_.min_connections) {
        size_t excess = idle_connections_.size() - config_.min_connections;
        for (size_t i = 0; i < excess; ++i) {
            auto connection = idle_connections_.back();
            idle_connections_.pop_back();
            
            auto it = std::find(connections_.begin(), connections_.end(), connection);
            if (it != connections_.end()) {
                connections_.erase(it);
                stats_.total_connections--;
                stats_.idle_connections--;
            }
        }
        
        LOG_DEBUG("Cleaned up {} idle connections", excess);
    }
}

bool ConnectionPool::validate_connection(std::shared_ptr<DatabaseConnector> conn) {
    return conn && conn->is_connected() && conn->ping();
}

// ConnectionHandle implementation
ConnectionPool::ConnectionHandle::ConnectionHandle(std::shared_ptr<DatabaseConnector> conn, ConnectionPool* pool)
    : connection_(conn), pool_(pool) {
}

ConnectionPool::ConnectionHandle::~ConnectionHandle() {
    if (connection_ && pool_) {
        pool_->release_connection(connection_);
    }
}

} // namespace database
} // namespace ultra_cpp