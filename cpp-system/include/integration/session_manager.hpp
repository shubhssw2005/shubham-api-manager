#pragma once

#include <string>
#include <memory>
#include <unordered_map>
#include <chrono>
#include <atomic>
#include <mutex>
#include <optional>
#include <functional>

namespace integration {

/**
 * Shared session storage with Redis for state consistency between C++ and Node.js
 */
class SessionManager {
public:
    struct Config {
        std::string redis_host = "localhost";
        uint16_t redis_port = 6379;
        std::string redis_password;
        uint32_t redis_db = 0;
        uint32_t connection_timeout_ms = 5000;
        uint32_t command_timeout_ms = 1000;
        uint32_t max_connections = 10;
        std::string key_prefix = "session:";
        uint32_t default_ttl_seconds = 3600; // 1 hour
        bool enable_compression = true;
        bool enable_encryption = false;
        std::string encryption_key;
    };

    struct SessionData {
        std::string session_id;
        std::string user_id;
        std::string tenant_id;
        std::unordered_map<std::string, std::string> attributes;
        std::chrono::system_clock::time_point created_at;
        std::chrono::system_clock::time_point last_accessed;
        std::chrono::system_clock::time_point expires_at;
        bool is_authenticated = false;
        std::string user_role;
        std::vector<std::string> permissions;
    };

    struct SessionStats {
        std::atomic<uint64_t> total_sessions{0};
        std::atomic<uint64_t> active_sessions{0};
        std::atomic<uint64_t> expired_sessions{0};
        std::atomic<uint64_t> cache_hits{0};
        std::atomic<uint64_t> cache_misses{0};
        std::atomic<uint64_t> redis_operations{0};
        std::atomic<uint64_t> redis_errors{0};
        std::atomic<double> avg_operation_time_ms{0.0};
    };

    explicit SessionManager(const Config& config);
    ~SessionManager();

    // Lifecycle management
    bool initialize();
    void shutdown();
    bool is_connected() const;

    // Session operations
    std::optional<SessionData> get_session(const std::string& session_id);
    bool create_session(const SessionData& session);
    bool update_session(const std::string& session_id, const SessionData& session);
    bool delete_session(const std::string& session_id);
    bool extend_session(const std::string& session_id, uint32_t additional_seconds = 0);
    
    // Batch operations
    std::vector<SessionData> get_sessions(const std::vector<std::string>& session_ids);
    bool delete_sessions(const std::vector<std::string>& session_ids);
    
    // User session management
    std::vector<std::string> get_user_sessions(const std::string& user_id);
    bool delete_user_sessions(const std::string& user_id);
    
    // Tenant session management
    std::vector<std::string> get_tenant_sessions(const std::string& tenant_id);
    bool delete_tenant_sessions(const std::string& tenant_id);
    
    // Session validation
    bool is_session_valid(const std::string& session_id);
    bool is_session_authenticated(const std::string& session_id);
    std::optional<std::string> get_session_user_id(const std::string& session_id);
    std::optional<std::string> get_session_tenant_id(const std::string& session_id);
    
    // Session attributes
    bool set_session_attribute(const std::string& session_id, 
                              const std::string& key, 
                              const std::string& value);
    std::optional<std::string> get_session_attribute(const std::string& session_id, 
                                                    const std::string& key);
    bool remove_session_attribute(const std::string& session_id, const std::string& key);
    
    // Cleanup operations
    uint64_t cleanup_expired_sessions();
    uint64_t cleanup_inactive_sessions(uint32_t inactive_threshold_seconds);
    
    // Statistics and monitoring
    SessionStats get_stats() const;
    void reset_stats();
    uint64_t get_active_session_count();
    uint64_t get_total_session_count();
    
    // Health check
    bool health_check();
    
    // Event callbacks
    using SessionEventCallback = std::function<void(const std::string&, const SessionData&)>;
    void set_session_created_callback(SessionEventCallback callback);
    void set_session_updated_callback(SessionEventCallback callback);
    void set_session_deleted_callback(SessionEventCallback callback);
    void set_session_expired_callback(SessionEventCallback callback);

private:
    Config config_;
    SessionStats stats_;
    
    // Redis connection management
    class RedisConnection;
    std::unique_ptr<RedisConnection> redis_;
    
    // Local cache for frequently accessed sessions
    class SessionCache;
    std::unique_ptr<SessionCache> cache_;
    
    // Event callbacks
    SessionEventCallback session_created_callback_;
    SessionEventCallback session_updated_callback_;
    SessionEventCallback session_deleted_callback_;
    SessionEventCallback session_expired_callback_;
    
    // Internal methods
    std::string serialize_session(const SessionData& session);
    std::optional<SessionData> deserialize_session(const std::string& data);
    std::string get_redis_key(const std::string& session_id);
    std::string get_user_index_key(const std::string& user_id);
    std::string get_tenant_index_key(const std::string& tenant_id);
    
    // Compression and encryption
    std::string compress_data(const std::string& data);
    std::string decompress_data(const std::string& compressed_data);
    std::string encrypt_data(const std::string& data);
    std::string decrypt_data(const std::string& encrypted_data);
    
    // Index management
    bool update_user_index(const std::string& user_id, const std::string& session_id, bool add);
    bool update_tenant_index(const std::string& tenant_id, const std::string& session_id, bool add);
    
    // Metrics tracking
    void record_operation(const std::string& operation, double duration_ms, bool success);
};

/**
 * Session middleware for HTTP requests
 */
class SessionMiddleware {
public:
    explicit SessionMiddleware(std::shared_ptr<SessionManager> session_manager);
    
    struct RequestContext {
        std::string session_id;
        std::optional<SessionManager::SessionData> session;
        bool is_authenticated = false;
        std::string user_id;
        std::string tenant_id;
        std::string user_role;
        std::vector<std::string> permissions;
    };
    
    // Extract session from HTTP request
    RequestContext extract_session_context(const std::unordered_map<std::string, std::string>& headers,
                                          const std::unordered_map<std::string, std::string>& cookies);
    
    // Validate session and permissions
    bool validate_session(const RequestContext& context);
    bool has_permission(const RequestContext& context, const std::string& permission);
    bool has_role(const RequestContext& context, const std::string& role);
    
    // Session creation for login
    std::string create_login_session(const std::string& user_id, 
                                   const std::string& tenant_id,
                                   const std::string& user_role,
                                   const std::vector<std::string>& permissions,
                                   uint32_t ttl_seconds = 0);
    
    // Session cleanup for logout
    bool logout_session(const std::string& session_id);
    
private:
    std::shared_ptr<SessionManager> session_manager_;
    
    std::string extract_session_id_from_cookie(const std::string& cookie_header);
    std::string extract_session_id_from_header(const std::string& auth_header);
    std::string generate_session_id();
};

} // namespace integration