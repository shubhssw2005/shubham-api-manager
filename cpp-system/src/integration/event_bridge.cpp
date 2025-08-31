#include "integration/event_bridge.hpp"
#include "common/logger.hpp"
#include <hiredis/hiredis.h>
#include <json/json.h>
#include <uuid/uuid.h>
#include <regex>
#include <sstream>
#include <iomanip>
#include <algorithm>

namespace integration {

// Redis Publisher for sending events
class EventBridge::RedisPublisher {
public:
    explicit RedisPublisher(const Config& config) : config_(config), context_(nullptr) {}
    
    ~RedisPublisher() {
        disconnect();
    }
    
    bool connect() {
        if (context_) {
            return true;
        }
        
        struct timeval timeout = {5, 0}; // 5 seconds
        context_ = redisConnectWithTimeout(config_.redis_host.c_str(), 
                                         config_.redis_port, timeout);
        
        if (!context_ || context_->err) {
            if (context_) {
                LOG_ERROR("Redis publisher connection error: {}", context_->errstr);
                redisFree(context_);
                context_ = nullptr;
            }
            return false;
        }
        
        // Authenticate if needed
        if (!config_.redis_password.empty()) {
            redisReply* reply = static_cast<redisReply*>(
                redisCommand(context_, "AUTH %s", config_.redis_password.c_str())
            );
            
            if (!reply || reply->type == REDIS_REPLY_ERROR) {
                LOG_ERROR("Redis publisher authentication failed");
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
                LOG_ERROR("Redis publisher database selection failed");
                if (reply) freeReplyObject(reply);
                disconnect();
                return false;
            }
            freeReplyObject(reply);
        }
        
        return true;
    }
    
    void disconnect() {
        if (context_) {
            redisFree(context_);
            context_ = nullptr;
        }
    }
    
    bool publish(const std::string& channel, const std::string& message) {
        if (!ensure_connected()) {
            return false;
        }
        
        redisReply* reply = static_cast<redisReply*>(
            redisCommand(context_, "PUBLISH %s %b", 
                       channel.c_str(), message.c_str(), message.length())
        );
        
        if (!reply) {
            LOG_ERROR("Redis PUBLISH command failed for channel: {}", channel);
            return false;
        }
        
        bool success = (reply->type == REDIS_REPLY_INTEGER);
        long long subscribers = reply->integer;
        
        freeReplyObject(reply);
        
        if (success) {
            LOG_DEBUG("Published event to channel: {} (subscribers: {})", channel, subscribers);
        }
        
        return success;
    }
    
    bool store_event(const std::string& key, const std::string& data, uint32_t ttl_seconds) {
        if (!ensure_connected()) {
            return false;
        }
        
        redisReply* reply = static_cast<redisReply*>(
            redisCommand(context_, "SETEX %s %d %b", 
                       key.c_str(), ttl_seconds, data.c_str(), data.length())
        );
        
        if (!reply) {
            LOG_ERROR("Redis SETEX command failed for key: {}", key);
            return false;
        }
        
        bool success = (reply->type == REDIS_REPLY_STATUS && 
                       strcmp(reply->str, "OK") == 0);
        
        freeReplyObject(reply);
        return success;
    }
    
private:
    Config config_;
    redisContext* context_;
    
    bool ensure_connected() {
        if (context_ && context_->err == 0) {
            return true;
        }
        
        disconnect();
        return connect();
    }
};

// Redis Subscriber for receiving events
class EventBridge::RedisSubscriber {
public:
    explicit RedisSubscriber(const Config& config, EventBridge* bridge) 
        : config_(config), bridge_(bridge), context_(nullptr), running_(false) {}
    
    ~RedisSubscriber() {
        stop();
        disconnect();
    }
    
    bool connect() {
        if (context_) {
            return true;
        }
        
        struct timeval timeout = {5, 0}; // 5 seconds
        context_ = redisConnectWithTimeout(config_.redis_host.c_str(), 
                                         config_.redis_port, timeout);
        
        if (!context_ || context_->err) {
            if (context_) {
                LOG_ERROR("Redis subscriber connection error: {}", context_->errstr);
                redisFree(context_);
                context_ = nullptr;
            }
            return false;
        }
        
        // Authenticate if needed
        if (!config_.redis_password.empty()) {
            redisReply* reply = static_cast<redisReply*>(
                redisCommand(context_, "AUTH %s", config_.redis_password.c_str())
            );
            
            if (!reply || reply->type == REDIS_REPLY_ERROR) {
                LOG_ERROR("Redis subscriber authentication failed");
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
                LOG_ERROR("Redis subscriber database selection failed");
                if (reply) freeReplyObject(reply);
                disconnect();
                return false;
            }
            freeReplyObject(reply);
        }
        
        return true;
    }
    
    void disconnect() {
        if (context_) {
            redisFree(context_);
            context_ = nullptr;
        }
    }
    
    bool subscribe(const std::string& pattern) {
        if (!ensure_connected()) {
            return false;
        }
        
        redisReply* reply = static_cast<redisReply*>(
            redisCommand(context_, "PSUBSCRIBE %s", pattern.c_str())
        );
        
        if (!reply) {
            LOG_ERROR("Redis PSUBSCRIBE command failed for pattern: {}", pattern);
            return false;
        }
        
        bool success = (reply->type == REDIS_REPLY_ARRAY);
        freeReplyObject(reply);
        
        if (success) {
            subscribed_patterns_.insert(pattern);
            LOG_INFO("Subscribed to pattern: {}", pattern);
        }
        
        return success;
    }
    
    void start_listening() {
        if (running_.load()) {
            return;
        }
        
        running_.store(true);
        listener_thread_ = std::thread(&RedisSubscriber::listen_loop, this);
    }
    
    void stop() {
        running_.store(false);
        
        if (listener_thread_.joinable()) {
            listener_thread_.join();
        }
    }
    
private:
    Config config_;
    EventBridge* bridge_;
    redisContext* context_;
    std::atomic<bool> running_;
    std::thread listener_thread_;
    std::set<std::string> subscribed_patterns_;
    
    bool ensure_connected() {
        if (context_ && context_->err == 0) {
            return true;
        }
        
        disconnect();
        return connect();
    }
    
    void listen_loop() {
        while (running_.load()) {
            if (!ensure_connected()) {
                std::this_thread::sleep_for(std::chrono::seconds(1));
                continue;
            }
            
            redisReply* reply = nullptr;
            int result = redisGetReply(context_, reinterpret_cast<void**>(&reply));
            
            if (result != REDIS_OK || !reply) {
                if (reply) freeReplyObject(reply);
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }
            
            // Process message
            if (reply->type == REDIS_REPLY_ARRAY && reply->elements >= 4) {
                // PMESSAGE format: [pmessage, pattern, channel, message]
                std::string message_type(reply->element[0]->str, reply->element[0]->len);
                
                if (message_type == "pmessage") {
                    std::string pattern(reply->element[1]->str, reply->element[1]->len);
                    std::string channel(reply->element[2]->str, reply->element[2]->len);
                    std::string message(reply->element[3]->str, reply->element[3]->len);
                    
                    // Forward to event bridge for processing
                    bridge_->process_received_event(channel, message);
                }
            }
            
            freeReplyObject(reply);
        }
    }
};

EventBridge::EventBridge(const Config& config) 
    : config_(config), 
      subscriber_(std::make_unique<RedisSubscriber>(config, this)),
      publisher_(std::make_unique<RedisPublisher>(config)) {
}

EventBridge::~EventBridge() {
    shutdown();
}

bool EventBridge::initialize() {
    if (running_.load()) {
        return true;
    }
    
    // Connect publisher
    if (!publisher_->connect()) {
        LOG_ERROR("Failed to connect Redis publisher");
        return false;
    }
    
    // Connect subscriber
    if (!subscriber_->connect()) {
        LOG_ERROR("Failed to connect Redis subscriber");
        return false;
    }
    
    // Subscribe to all events for this system
    std::string pattern = config_.event_channel_prefix + "*";
    if (!subscriber_->subscribe(pattern)) {
        LOG_ERROR("Failed to subscribe to event pattern: {}", pattern);
        return false;
    }
    
    // Start worker threads
    for (uint32_t i = 0; i < config_.worker_threads; ++i) {
        worker_threads_.emplace_back(&EventBridge::worker_loop, this);
    }
    
    // Start batch processing thread
    batch_thread_ = std::thread(&EventBridge::batch_processing_loop, this);
    last_batch_time_ = std::chrono::steady_clock::now();
    
    // Start subscriber listening
    subscriber_->start_listening();
    
    running_.store(true);
    
    LOG_INFO("EventBridge initialized successfully");
    return true;
}

void EventBridge::shutdown() {
    if (!running_.load()) {
        return;
    }
    
    running_.store(false);
    
    // Stop subscriber
    if (subscriber_) {
        subscriber_->stop();
    }
    
    // Wake up all worker threads
    queue_cv_.notify_all();
    
    // Wait for worker threads to finish
    for (auto& thread : worker_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    worker_threads_.clear();
    
    // Wait for batch thread to finish
    if (batch_thread_.joinable()) {
        batch_thread_.join();
    }
    
    // Disconnect Redis connections
    if (subscriber_) {
        subscriber_->disconnect();
    }
    
    if (publisher_) {
        publisher_->disconnect();
    }
    
    LOG_INFO("EventBridge shut down");
}

bool EventBridge::publish_event(const Event& event) {
    if (!running_.load()) {
        LOG_ERROR("EventBridge not running");
        return false;
    }
    
    if (!validate_event(event)) {
        LOG_ERROR("Invalid event: {}", event.id);
        return false;
    }
    
    try {
        // Serialize event
        std::string serialized_event = serialize_event(event);
        if (serialized_event.empty()) {
            LOG_ERROR("Failed to serialize event: {}", event.id);
            return false;
        }
        
        // Publish to Redis
        std::string channel = config_.event_channel_prefix + event.channel;
        bool published = publisher_->publish(channel, serialized_event);
        
        if (published) {
            // Store event for persistence if enabled
            if (config_.enable_event_persistence) {
                std::string event_key = get_event_key(event.id);
                publisher_->store_event(event_key, serialized_event, config_.event_ttl_seconds);
            }
            
            stats_.events_published.fetch_add(1);
            LOG_DEBUG("Published event: {} to channel: {}", event.id, event.channel);
        } else {
            stats_.events_failed.fetch_add(1);
            LOG_ERROR("Failed to publish event: {} to channel: {}", event.id, event.channel);
        }
        
        return published;
        
    } catch (const std::exception& e) {
        LOG_ERROR("Exception publishing event {}: {}", event.id, e.what());
        stats_.events_failed.fetch_add(1);
        return false;
    }
}

bool EventBridge::publish_event(EventType type, 
                               const std::string& channel,
                               const std::string& payload,
                               const std::string& target_system,
                               uint32_t priority) {
    Event event;
    event.id = EventBuilder::generate_event_id();
    event.type = type;
    event.source_system = config_.cpp_instance_id;
    event.target_system = target_system;
    event.channel = channel;
    event.payload = payload;
    event.timestamp = std::chrono::system_clock::now();
    event.priority = priority;
    event.expires_at = EventBuilder::get_default_expiry();
    
    return publish_event(event);
}

std::string EventBridge::subscribe(const std::string& channel_pattern, 
                                  EventHandler handler,
                                  EventFilter filter) {
    std::lock_guard<std::mutex> lock(subscriptions_mutex_);
    
    std::string subscription_id = EventBuilder::generate_event_id();
    
    auto subscription = std::make_unique<Subscription>();
    subscription->id = subscription_id;
    subscription->channel_pattern = channel_pattern;
    subscription->handler = std::move(handler);
    subscription->filter = std::move(filter);
    subscription->is_active = true;
    
    subscriptions_[subscription_id] = std::move(subscription);
    stats_.subscriptions_active.fetch_add(1);
    
    LOG_INFO("Created subscription: {} for pattern: {}", subscription_id, channel_pattern);
    return subscription_id;
}

bool EventBridge::unsubscribe(const std::string& subscription_id) {
    std::lock_guard<std::mutex> lock(subscriptions_mutex_);
    
    auto it = subscriptions_.find(subscription_id);
    if (it != subscriptions_.end()) {
        subscriptions_.erase(it);
        stats_.subscriptions_active.fetch_sub(1);
        LOG_INFO("Removed subscription: {}", subscription_id);
        return true;
    }
    
    return false;
}

void EventBridge::process_received_event(const std::string& channel, const std::string& message) {
    try {
        // Deserialize event
        auto event = deserialize_event(message);
        if (!event) {
            LOG_ERROR("Failed to deserialize event from channel: {}", channel);
            return;
        }
        
        // Skip events from our own system to avoid loops
        if (event->source_system == config_.cpp_instance_id) {
            return;
        }
        
        // Check if event is targeted to us or is a broadcast
        if (!event->target_system.empty() && 
            event->target_system != config_.cpp_instance_id) {
            return;
        }
        
        stats_.events_received.fetch_add(1);
        
        // Add to processing queue
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            
            if (event_queue_.size() >= config_.event_queue_size) {
                LOG_WARN("Event queue full, dropping event: {}", event->id);
                stats_.events_failed.fetch_add(1);
                return;
            }
            
            event_queue_.push(*event);
            stats_.queue_depth.store(event_queue_.size());
        }
        
        queue_cv_.notify_one();
        
    } catch (const std::exception& e) {
        LOG_ERROR("Exception processing received event: {}", e.what());
        stats_.events_failed.fetch_add(1);
    }
}

void EventBridge::worker_loop() {
    while (running_.load()) {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        
        queue_cv_.wait(lock, [this] {
            return !event_queue_.empty() || !running_.load();
        });
        
        if (!running_.load()) {
            break;
        }
        
        if (event_queue_.empty()) {
            continue;
        }
        
        Event event = event_queue_.front();
        event_queue_.pop();
        stats_.queue_depth.store(event_queue_.size());
        
        lock.unlock();
        
        // Process the event
        process_event(event);
    }
}

void EventBridge::process_event(const Event& event) {
    auto start_time = std::chrono::steady_clock::now();
    
    try {
        // Find matching subscriptions
        std::vector<std::shared_ptr<Subscription>> matching_subscriptions;
        
        {
            std::lock_guard<std::mutex> lock(subscriptions_mutex_);
            
            for (const auto& [id, subscription] : subscriptions_) {
                if (!subscription->is_active) {
                    continue;
                }
                
                // Check if channel matches pattern
                if (!matches_pattern(event.channel, subscription->channel_pattern)) {
                    continue;
                }
                
                // Apply filter if present
                if (subscription->filter && !subscription->filter(event)) {
                    subscription->events_filtered.fetch_add(1);
                    continue;
                }
                
                matching_subscriptions.push_back(
                    std::shared_ptr<Subscription>(subscription.get(), [](Subscription*){})
                );
            }
        }
        
        // Process event with matching subscriptions
        for (auto& subscription : matching_subscriptions) {
            try {
                subscription->handler(event);
                subscription->events_processed.fetch_add(1);
                
            } catch (const std::exception& e) {
                LOG_ERROR("Exception in event handler for subscription {}: {}", 
                         subscription->id, e.what());
                subscription->processing_errors.fetch_add(1);
                stats_.events_failed.fetch_add(1);
            }
        }
        
        stats_.events_processed.fetch_add(1);
        
        auto end_time = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        
        // Update average processing time
        auto current_avg = stats_.avg_processing_time_ms.load();
        auto new_avg = (current_avg * 0.9) + (duration * 0.1);
        stats_.avg_processing_time_ms.store(new_avg);
        
    } catch (const std::exception& e) {
        LOG_ERROR("Exception processing event {}: {}", event.id, e.what());
        stats_.events_failed.fetch_add(1);
    }
}

std::string EventBridge::serialize_event(const Event& event) {
    try {
        Json::Value json;
        json["id"] = event.id;
        json["type"] = static_cast<int>(event.type);
        json["source_system"] = event.source_system;
        json["target_system"] = event.target_system;
        json["channel"] = event.channel;
        json["payload"] = event.payload;
        json["priority"] = event.priority;
        json["retry_count"] = event.retry_count;
        json["max_retries"] = event.max_retries;
        
        // Serialize metadata
        Json::Value metadata(Json::objectValue);
        for (const auto& [key, value] : event.metadata) {
            metadata[key] = value;
        }
        json["metadata"] = metadata;
        
        // Serialize timestamps
        json["timestamp"] = std::chrono::duration_cast<std::chrono::seconds>(
            event.timestamp.time_since_epoch()
        ).count();
        json["expires_at"] = std::chrono::duration_cast<std::chrono::seconds>(
            event.expires_at.time_since_epoch()
        ).count();
        
        Json::StreamWriterBuilder builder;
        builder["indentation"] = "";
        std::string serialized = Json::writeString(builder, json);
        
        // Apply compression if enabled
        if (config_.enable_compression) {
            serialized = compress_payload(serialized);
        }
        
        return serialized;
        
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to serialize event {}: {}", event.id, e.what());
        return "";
    }
}

std::optional<EventBridge::Event> EventBridge::deserialize_event(const std::string& data) {
    try {
        std::string processed_data = data;
        
        // Apply decompression if enabled
        if (config_.enable_compression) {
            processed_data = decompress_payload(processed_data);
        }
        
        Json::Value json;
        Json::CharReaderBuilder builder;
        std::string errors;
        std::istringstream stream(processed_data);
        
        if (!Json::parseFromStream(builder, stream, &json, &errors)) {
            LOG_ERROR("Failed to parse event JSON: {}", errors);
            return std::nullopt;
        }
        
        Event event;
        event.id = json["id"].asString();
        event.type = static_cast<EventType>(json["type"].asInt());
        event.source_system = json["source_system"].asString();
        event.target_system = json["target_system"].asString();
        event.channel = json["channel"].asString();
        event.payload = json["payload"].asString();
        event.priority = json["priority"].asUInt();
        event.retry_count = json["retry_count"].asUInt();
        event.max_retries = json["max_retries"].asUInt();
        
        // Deserialize metadata
        const Json::Value& metadata = json["metadata"];
        for (const auto& key : metadata.getMemberNames()) {
            event.metadata[key] = metadata[key].asString();
        }
        
        // Deserialize timestamps
        event.timestamp = std::chrono::system_clock::from_time_t(
            json["timestamp"].asInt64()
        );
        event.expires_at = std::chrono::system_clock::from_time_t(
            json["expires_at"].asInt64()
        );
        
        return event;
        
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to deserialize event: {}", e.what());
        return std::nullopt;
    }
}

bool EventBridge::matches_pattern(const std::string& channel, const std::string& pattern) {
    // Simple pattern matching with * wildcard
    if (pattern == "*") {
        return true;
    }
    
    if (pattern.find('*') == std::string::npos) {
        return channel == pattern;
    }
    
    // Convert pattern to regex
    std::string regex_pattern = pattern;
    std::replace(regex_pattern.begin(), regex_pattern.end(), '*', '.');
    regex_pattern = "^" + regex_pattern + "$";
    
    try {
        std::regex regex(regex_pattern);
        return std::regex_match(channel, regex);
    } catch (const std::exception& e) {
        LOG_ERROR("Invalid pattern regex: {}", pattern);
        return false;
    }
}

bool EventBridge::validate_event(const Event& event) {
    if (event.id.empty()) {
        LOG_ERROR("Event ID is empty");
        return false;
    }
    
    if (event.channel.empty()) {
        LOG_ERROR("Event channel is empty");
        return false;
    }
    
    if (event.payload.length() > config_.max_event_size_bytes) {
        LOG_ERROR("Event payload too large: {} bytes", event.payload.length());
        return false;
    }
    
    if (event.priority < 1 || event.priority > 10) {
        LOG_ERROR("Invalid event priority: {}", event.priority);
        return false;
    }
    
    return true;
}

std::string EventBridge::get_event_key(const std::string& event_id) {
    return config_.event_channel_prefix + "stored:" + event_id;
}

// Compression helpers (simplified)
std::string EventBridge::compress_payload(const std::string& payload) {
    // TODO: Implement actual compression
    return payload;
}

std::string EventBridge::decompress_payload(const std::string& compressed_payload) {
    // TODO: Implement actual decompression
    return compressed_payload;
}

EventBridge::Stats EventBridge::get_stats() const {
    return stats_;
}

bool EventBridge::health_check() {
    return running_.load() && 
           publisher_ && 
           subscriber_;
}

// EventBuilder implementation
std::string EventBuilder::generate_event_id() {
    uuid_t uuid;
    uuid_generate_random(uuid);
    
    char uuid_str[37];
    uuid_unparse_lower(uuid, uuid_str);
    
    return std::string(uuid_str);
}

std::chrono::system_clock::time_point EventBuilder::get_default_expiry() {
    return std::chrono::system_clock::now() + std::chrono::hours(1);
}

EventBridge::Event EventBuilder::create_user_action_event(
    const std::string& user_id,
    const std::string& action,
    const std::string& resource,
    const std::unordered_map<std::string, std::string>& details) {
    
    Event event;
    event.id = generate_event_id();
    event.type = EventBridge::EventType::USER_ACTION;
    event.channel = "user.actions";
    event.timestamp = std::chrono::system_clock::now();
    event.expires_at = get_default_expiry();
    event.priority = 5;
    
    Json::Value payload;
    payload["user_id"] = user_id;
    payload["action"] = action;
    payload["resource"] = resource;
    payload["timestamp"] = std::chrono::duration_cast<std::chrono::seconds>(
        event.timestamp.time_since_epoch()
    ).count();
    
    Json::Value details_json(Json::objectValue);
    for (const auto& [key, value] : details) {
        details_json[key] = value;
    }
    payload["details"] = details_json;
    
    Json::StreamWriterBuilder builder;
    builder["indentation"] = "";
    event.payload = Json::writeString(builder, payload);
    
    return event;
}

} // namespace integration