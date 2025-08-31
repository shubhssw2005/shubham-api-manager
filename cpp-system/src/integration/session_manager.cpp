#include "integration/session_manager.hpp"
#include "common/logger.hpp"
#include <hiredis/hiredis.h>
#include <json/json.h>
#include <zlib.h>
#include <openssl/aes.h>
#include <openssl/rand.h>
#include <uuid/uuid.h>
#include <sstream>
#include <iomanip>
#include <algorithm>

namespace integration {

// Redis connection wrapper
class SessionManager::RedisConnection {
public:
    explicit RedisConnection(const Config& config) : config_(config), context_(nullptr) {}
    
    ~RedisConnection() {
        disconnect();
    }
    
    bool connect() {
        if (context_) {
            return true;
        }
        
        struct timeval timeout = {
            .tv_sec = config_.connection_timeout_ms / 1000,
            .tv_usec = (config_.connection_timeout_ms % 1000) * 1000
        };
        
        context_ = redisConnectWithTimeout(config_.redis_host.c_str(), 
                                         config_.redis_port, timeout);
        
        if (!context_ || context_->err) {
            if (context_) {
                LOG_ERROR("Redis connection error: {}", context_->errstr);
                redisFree(context_);
                context_ = nullptr;
            } else {
                LOG_ERROR("Redis connection allocation failed");
            }
            return false;
        }
        
        // Set command timeout
        timeout.tv_sec = config_.command_timeout_ms / 1000;
        timeout.tv_usec = (config_.command_timeout_ms % 1000) * 1000;
        redisSetTimeout(context_, timeout);
        
        // Authenticate if password is provided
        if (!config_.redis_password.empty()) {
            redisReply* reply = static_cast<redisReply*>(
                redisCommand(context_, "AUTH %s", config_.redis_password.c_str())
            );
            
            if (!reply || reply->type == REDIS_REPLY_ERROR) {
                LOG_ERROR("Redis authentication failed");
                if (reply) freeReplyObject(reply);
                disconnect();
                return false;
            }
            freeReplyObject(reply);
        }
        
        // Select database
        if (config_.redis_db != 0) {
            redisReply* reply = static_cast<redisReply*>(
                redisCommand(context_, "SELECT %d", config_.redis_db)
            );
            
            if (!reply || reply->type == REDIS_REPLY_ERROR) {
                LOG_ERROR("Redis database selection failed");
                if (reply) freeReplyObject(reply);
                disconnect();
                return false;
            }
            freeReplyObject(reply);
        }
        
        LOG_INFO("Connected to Redis at {}:{}", config_.redis_host, config_.redis_port);
        return true;
    }
    
    void disconnect() {
        if (context_) {
            redisFree(context_);
            context_ = nullptr;
        }
    }
    
    bool is_connected() const {
        return context_ != nullptr && context_->err == 0;
    }
    
    std::optional<std::string> get(const std::string& key) {
        if (!ensure_connected()) {
            return std::nullopt;
        }
        
        redisReply* reply = static_cast<redisReply*>(
            redisCommand(context_, "GET %s", key.c_str())
        );
        
        if (!reply) {
            LOG_ERROR("Redis GET command failed for key: {}", key);
            return std::nullopt;
        }
        
        std::optional<std::string> result;
        if (reply->type == REDIS_REPLY_STRING) {
            result = std::string(reply->str, reply->len);
        } else if (reply->type == REDIS_REPLY_NIL) {
            // Key doesn't exist, return nullopt
        } else {
            LOG_ERROR("Unexpected Redis reply type for GET: {}", reply->type);
        }
        
        freeReplyObject(reply);
        return result;
    }
    
    bool set(const std::string& key, const std::string& value, uint32_t ttl_seconds = 0) {
        if (!ensure_connected()) {
            return false;
        }
        
        redisReply* reply;
        if (ttl_seconds > 0) {
            reply = static_cast<redisReply*>(
                redisCommand(context_, "SETEX %s %d %b", 
                           key.c_str(), ttl_seconds, value.c_str(), value.length())
            );
        } else {
            reply = static_cast<redisReply*>(
                redisCommand(context_, "SET %s %b", 
                           key.c_str(), value.c_str(), value.length())
            );
        }
        
        if (!reply) {
            LOG_ERROR("Redis SET command failed for key: {}", key);
            return false;
        }
        
        bool success = (reply->type == REDIS_REPLY_STATUS && 
                       strcmp(reply->str, "OK") == 0);
        
        if (!success) {
            LOG_ERROR("Redis SET failed for key: {}, reply type: {}", key, reply->type);
        }
        
        freeReplyObject(reply);
        return success;
    }
    
    bool del(const std::string& key) {
        if (!ensure_connected()) {
            return false;
        }
        
        redisReply* reply = static_cast<redisReply*>(
            redisCommand(context_, "DEL %s", key.c_str())
        );
        
        if (!reply) {
            LOG_ERROR("Redis DEL command failed for key: {}", key);
            return false;
        }
        
        bool success = (reply->type == REDIS_REPLY_INTEGER);
        freeReplyObject(reply);
        return success;
    }
    
    bool expire(const std::string& key, uint32_t ttl_seconds) {
        if (!ensure_connected()) {
            return false;
        }
        
        redisReply* reply = static_cast<redisReply*>(
            redisCommand(context_, "EXPIRE %s %d", key.c_str(), ttl_seconds)
        );
        
        if (!reply) {
            LOG_ERROR("Redis EXPIRE command failed for key: {}", key);
            return false;
        }
        
        bool success = (reply->type == REDIS_REPLY_INTEGER && reply->integer == 1);
        freeReplyObject(reply);
        return success;
    }
    
    std::vector<std::string> smembers(const std::string& key) {
        std::vector<std::string> members;
        
        if (!ensure_connected()) {
            return members;
        }
        
        redisReply* reply = static_cast<redisReply*>(
            redisCommand(context_, "SMEMBERS %s", key.c_str())
        );
        
        if (!reply) {
            LOG_ERROR("Redis SMEMBERS command failed for key: {}", key);
            return members;
        }
        
        if (reply->type == REDIS_REPLY_ARRAY) {
            members.reserve(reply->elements);
            for (size_t i = 0; i < reply->elements; ++i) {
                if (reply->element[i]->type == REDIS_REPLY_STRING) {
                    members.emplace_back(reply->element[i]->str, reply->element[i]->len);
                }
            }
        }
        
        freeReplyObject(reply);
        return members;
    }
    
    bool sadd(const std::string& key, const std::string& member) {
        if (!ensure_connected()) {
            return false;
        }
        
        redisReply* reply = static_cast<redisReply*>(
            redisCommand(context_, "SADD %s %s", key.c_str(), member.c_str())
        );
        
        if (!reply) {
            LOG_ERROR("Redis SADD command failed for key: {}", key);
            return false;
        }
        
        bool success = (reply->type == REDIS_REPLY_INTEGER);
        freeReplyObject(reply);
        return success;
    }
    
    bool srem(const std::string& key, const std::string& member) {
        if (!ensure_connected()) {
            return false;
        }
        
        redisReply* reply = static_cast<redisReply*>(
            redisCommand(context_, "SREM %s %s", key.c_str(), member.c_str())
        );
        
        if (!reply) {
            LOG_ERROR("Redis SREM command failed for key: {}", key);
            return false;
        }
        
        bool success = (reply->type == REDIS_REPLY_INTEGER);
        freeReplyObject(reply);
        return success;
    }
    
    bool ping() {
        if (!ensure_connected()) {
            return false;
        }
        
        redisReply* reply = static_cast<redisReply*>(
            redisCommand(context_, "PING")
        );
        
        if (!reply) {
            return false;
        }
        
        bool success = (reply->type == REDIS_REPLY_STATUS && 
                       strcmp(reply->str, "PONG") == 0);
        
        freeReplyObject(reply);
        return success;
    }
    
private:
    Config config_;
    redisContext* context_;
    
    bool ensure_connected() {
        if (is_connected()) {
            return true;
        }
        
        LOG_WARN("Redis connection lost, attempting to reconnect...");
        disconnect();
        return connect();
    }
};

// Simple LRU cache for sessions
class SessionManager::SessionCache {
public:
    explicit SessionCache(size_t max_size) : max_size_(max_size) {}
    
    std::optional<SessionData> get(const std::string& session_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        auto it = cache_.find(session_id);
        if (it == cache_.end()) {
            return std::nullopt;
        }
        
        // Move to front (most recently used)
        lru_list_.splice(lru_list_.begin(), lru_list_, it->second.list_iter);
        
        return it->second.session;
    }
    
    void put(const std::string& session_id, const SessionData& session) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        auto it = cache_.find(session_id);
        if (it != cache_.end()) {
            // Update existing entry
            it->second.session = session;
            lru_list_.splice(lru_list_.begin(), lru_list_, it->second.list_iter);
            return;
        }
        
        // Add new entry
        if (cache_.size() >= max_size_) {
            // Remove least recently used
            auto lru_session_id = lru_list_.back();
            lru_list_.pop_back();
            cache_.erase(lru_session_id);
        }
        
        lru_list_.push_front(session_id);
        cache_[session_id] = {session, lru_list_.begin()};
    }
    
    void remove(const std::string& session_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        auto it = cache_.find(session_id);
        if (it != cache_.end()) {
            lru_list_.erase(it->second.list_iter);
            cache_.erase(it);
        }
    }
    
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        cache_.clear();
        lru_list_.clear();
    }
    
    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return cache_.size();
    }
    
private:
    struct CacheEntry {
        SessionData session;
        std::list<std::string>::iterator list_iter;
    };
    
    size_t max_size_;
    std::unordered_map<std::string, CacheEntry> cache_;
    std::list<std::string> lru_list_;
    mutable std::mutex mutex_;
};

SessionManager::SessionManager(const Config& config) 
    : config_(config), redis_(std::make_unique<RedisConnection>(config)),
      cache_(std::make_unique<SessionCache>(1000)) {
}

SessionManager::~SessionManager() {
    shutdown();
}

bool SessionManager::initialize() {
    if (!redis_->connect()) {
        LOG_ERROR("Failed to connect to Redis");
        return false;
    }
    
    LOG_INFO("SessionManager initialized successfully");
    return true;
}

void SessionManager::shutdown() {
    if (redis_) {
        redis_->disconnect();
    }
    
    if (cache_) {
        cache_->clear();
    }
    
    LOG_INFO("SessionManager shut down");
}

bool SessionManager::is_connected() const {
    return redis_ && redis_->is_connected();
}

std::optional<SessionManager::SessionData> SessionManager::get_session(const std::string& session_id) {
    auto start_time = std::chrono::steady_clock::now();
    
    // Try cache first
    auto cached_session = cache_->get(session_id);
    if (cached_session) {
        stats_.cache_hits.fetch_add(1);
        record_operation("get_session_cache", 
                        std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - start_time
                        ).count(), true);
        return cached_session;
    }
    
    stats_.cache_misses.fetch_add(1);
    
    // Get from Redis
    std::string redis_key = get_redis_key(session_id);
    auto redis_data = redis_->get(redis_key);
    
    if (!redis_data) {
        record_operation("get_session_redis", 
                        std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - start_time
                        ).count(), false);
        return std::nullopt;
    }
    
    // Deserialize session
    auto session = deserialize_session(*redis_data);
    if (!session) {
        LOG_ERROR("Failed to deserialize session: {}", session_id);
        record_operation("get_session_redis", 
                        std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - start_time
                        ).count(), false);
        return std::nullopt;
    }
    
    // Check if session is expired
    auto now = std::chrono::system_clock::now();
    if (session->expires_at < now) {
        // Session expired, remove it
        delete_session(session_id);
        stats_.expired_sessions.fetch_add(1);
        
        if (session_expired_callback_) {
            session_expired_callback_(session_id, *session);
        }
        
        record_operation("get_session_redis", 
                        std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - start_time
                        ).count(), false);
        return std::nullopt;
    }
    
    // Update last accessed time
    session->last_accessed = now;
    
    // Cache the session
    cache_->put(session_id, *session);
    
    record_operation("get_session_redis", 
                    std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - start_time
                    ).count(), true);
    
    return session;
}

bool SessionManager::create_session(const SessionData& session) {
    auto start_time = std::chrono::steady_clock::now();
    
    // Serialize session
    std::string serialized_data = serialize_session(session);
    if (serialized_data.empty()) {
        LOG_ERROR("Failed to serialize session: {}", session.session_id);
        return false;
    }
    
    // Calculate TTL
    auto now = std::chrono::system_clock::now();
    auto ttl_duration = session.expires_at - now;
    uint32_t ttl_seconds = static_cast<uint32_t>(
        std::chrono::duration_cast<std::chrono::seconds>(ttl_duration).count()
    );
    
    if (ttl_seconds == 0) {
        ttl_seconds = config_.default_ttl_seconds;
    }
    
    // Store in Redis
    std::string redis_key = get_redis_key(session.session_id);
    bool success = redis_->set(redis_key, serialized_data, ttl_seconds);
    
    if (success) {
        // Update indexes
        update_user_index(session.user_id, session.session_id, true);
        update_tenant_index(session.tenant_id, session.session_id, true);
        
        // Cache the session
        cache_->put(session.session_id, session);
        
        stats_.total_sessions.fetch_add(1);
        stats_.active_sessions.fetch_add(1);
        
        if (session_created_callback_) {
            session_created_callback_(session.session_id, session);
        }
        
        LOG_DEBUG("Created session: {} for user: {}", session.session_id, session.user_id);
    }
    
    record_operation("create_session", 
                    std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - start_time
                    ).count(), success);
    
    return success;
}

bool SessionManager::delete_session(const std::string& session_id) {
    auto start_time = std::chrono::steady_clock::now();
    
    // Get session data for cleanup
    auto session = get_session(session_id);
    
    // Remove from Redis
    std::string redis_key = get_redis_key(session_id);
    bool success = redis_->del(redis_key);
    
    if (success && session) {
        // Update indexes
        update_user_index(session->user_id, session_id, false);
        update_tenant_index(session->tenant_id, session_id, false);
        
        stats_.active_sessions.fetch_sub(1);
        
        if (session_deleted_callback_) {
            session_deleted_callback_(session_id, *session);
        }
        
        LOG_DEBUG("Deleted session: {}", session_id);
    }
    
    // Remove from cache
    cache_->remove(session_id);
    
    record_operation("delete_session", 
                    std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - start_time
                    ).count(), success);
    
    return success;
}

std::string SessionManager::serialize_session(const SessionData& session) {
    try {
        Json::Value json;
        json["session_id"] = session.session_id;
        json["user_id"] = session.user_id;
        json["tenant_id"] = session.tenant_id;
        json["is_authenticated"] = session.is_authenticated;
        json["user_role"] = session.user_role;
        
        // Serialize attributes
        Json::Value attributes(Json::objectValue);
        for (const auto& [key, value] : session.attributes) {
            attributes[key] = value;
        }
        json["attributes"] = attributes;
        
        // Serialize permissions
        Json::Value permissions(Json::arrayValue);
        for (const auto& permission : session.permissions) {
            permissions.append(permission);
        }
        json["permissions"] = permissions;
        
        // Serialize timestamps
        json["created_at"] = std::chrono::duration_cast<std::chrono::seconds>(
            session.created_at.time_since_epoch()
        ).count();
        json["last_accessed"] = std::chrono::duration_cast<std::chrono::seconds>(
            session.last_accessed.time_since_epoch()
        ).count();
        json["expires_at"] = std::chrono::duration_cast<std::chrono::seconds>(
            session.expires_at.time_since_epoch()
        ).count();
        
        Json::StreamWriterBuilder builder;
        builder["indentation"] = "";
        std::string serialized = Json::writeString(builder, json);
        
        // Apply compression if enabled
        if (config_.enable_compression) {
            serialized = compress_data(serialized);
        }
        
        // Apply encryption if enabled
        if (config_.enable_encryption && !config_.encryption_key.empty()) {
            serialized = encrypt_data(serialized);
        }
        
        return serialized;
        
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to serialize session {}: {}", session.session_id, e.what());
        return "";
    }
}

std::optional<SessionManager::SessionData> SessionManager::deserialize_session(const std::string& data) {
    try {
        std::string processed_data = data;
        
        // Apply decryption if enabled
        if (config_.enable_encryption && !config_.encryption_key.empty()) {
            processed_data = decrypt_data(processed_data);
        }
        
        // Apply decompression if enabled
        if (config_.enable_compression) {
            processed_data = decompress_data(processed_data);
        }
        
        Json::Value json;
        Json::CharReaderBuilder builder;
        std::string errors;
        std::istringstream stream(processed_data);
        
        if (!Json::parseFromStream(builder, stream, &json, &errors)) {
            LOG_ERROR("Failed to parse session JSON: {}", errors);
            return std::nullopt;
        }
        
        SessionData session;
        session.session_id = json["session_id"].asString();
        session.user_id = json["user_id"].asString();
        session.tenant_id = json["tenant_id"].asString();
        session.is_authenticated = json["is_authenticated"].asBool();
        session.user_role = json["user_role"].asString();
        
        // Deserialize attributes
        const Json::Value& attributes = json["attributes"];
        for (const auto& key : attributes.getMemberNames()) {
            session.attributes[key] = attributes[key].asString();
        }
        
        // Deserialize permissions
        const Json::Value& permissions = json["permissions"];
        for (const auto& permission : permissions) {
            session.permissions.push_back(permission.asString());
        }
        
        // Deserialize timestamps
        session.created_at = std::chrono::system_clock::from_time_t(
            json["created_at"].asInt64()
        );
        session.last_accessed = std::chrono::system_clock::from_time_t(
            json["last_accessed"].asInt64()
        );
        session.expires_at = std::chrono::system_clock::from_time_t(
            json["expires_at"].asInt64()
        );
        
        return session;
        
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to deserialize session: {}", e.what());
        return std::nullopt;
    }
}

std::string SessionManager::get_redis_key(const std::string& session_id) {
    return config_.key_prefix + session_id;
}

std::string SessionManager::get_user_index_key(const std::string& user_id) {
    return config_.key_prefix + "user:" + user_id;
}

std::string SessionManager::get_tenant_index_key(const std::string& tenant_id) {
    return config_.key_prefix + "tenant:" + tenant_id;
}

bool SessionManager::update_user_index(const std::string& user_id, 
                                      const std::string& session_id, 
                                      bool add) {
    std::string index_key = get_user_index_key(user_id);
    
    if (add) {
        return redis_->sadd(index_key, session_id);
    } else {
        return redis_->srem(index_key, session_id);
    }
}

bool SessionManager::update_tenant_index(const std::string& tenant_id, 
                                        const std::string& session_id, 
                                        bool add) {
    std::string index_key = get_tenant_index_key(tenant_id);
    
    if (add) {
        return redis_->sadd(index_key, session_id);
    } else {
        return redis_->srem(index_key, session_id);
    }
}

void SessionManager::record_operation(const std::string& operation, 
                                    double duration_ms, 
                                    bool success) {
    stats_.redis_operations.fetch_add(1);
    
    if (!success) {
        stats_.redis_errors.fetch_add(1);
    }
    
    // Update average operation time
    auto current_avg = stats_.avg_operation_time_ms.load();
    auto new_avg = (current_avg * 0.9) + (duration_ms * 0.1);
    stats_.avg_operation_time_ms.store(new_avg);
}

bool SessionManager::health_check() {
    return redis_ && redis_->ping();
}

SessionManager::SessionStats SessionManager::get_stats() const {
    return stats_;
}

// Compression helpers (simplified zlib usage)
std::string SessionManager::compress_data(const std::string& data) {
    // Simplified compression - in production, use proper zlib compression
    return data; // TODO: Implement actual compression
}

std::string SessionManager::decompress_data(const std::string& compressed_data) {
    // Simplified decompression - in production, use proper zlib decompression
    return compressed_data; // TODO: Implement actual decompression
}

// Encryption helpers (simplified AES usage)
std::string SessionManager::encrypt_data(const std::string& data) {
    // Simplified encryption - in production, use proper AES encryption
    return data; // TODO: Implement actual encryption
}

std::string SessionManager::decrypt_data(const std::string& encrypted_data) {
    // Simplified decryption - in production, use proper AES decryption
    return encrypted_data; // TODO: Implement actual decryption
}

} // namespace integration