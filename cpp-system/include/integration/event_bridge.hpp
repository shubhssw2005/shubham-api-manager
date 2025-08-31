#pragma once

#include <string>
#include <memory>
#include <functional>
#include <unordered_map>
#include <vector>
#include <atomic>
#include <mutex>
#include <thread>
#include <queue>
#include <condition_variable>
#include <chrono>

namespace integration {

/**
 * Event bridge for real-time communication between C++ and Node.js systems
 */
class EventBridge {
public:
    struct Config {
        std::string redis_host = "localhost";
        uint16_t redis_port = 6379;
        std::string redis_password;
        uint32_t redis_db = 1; // Different DB from sessions
        std::string event_channel_prefix = "events:";
        std::string cpp_instance_id = "cpp-system";
        std::string nodejs_instance_id = "nodejs-system";
        uint32_t max_event_size_bytes = 1024 * 1024; // 1MB
        uint32_t event_queue_size = 10000;
        uint32_t batch_size = 100;
        uint32_t batch_timeout_ms = 100;
        bool enable_event_persistence = true;
        uint32_t event_ttl_seconds = 3600; // 1 hour
        bool enable_compression = true;
        uint32_t worker_threads = 2;
    };

    enum class EventType {
        USER_ACTION,
        SYSTEM_EVENT,
        CACHE_INVALIDATION,
        SESSION_EVENT,
        MEDIA_PROCESSING,
        PERFORMANCE_METRIC,
        ERROR_EVENT,
        CUSTOM
    };

    struct Event {
        std::string id;
        EventType type;
        std::string source_system;
        std::string target_system; // Empty for broadcast
        std::string channel;
        std::string payload;
        std::unordered_map<std::string, std::string> metadata;
        std::chrono::system_clock::time_point timestamp;
        uint32_t priority = 5; // 1-10, 1 is highest
        uint32_t retry_count = 0;
        uint32_t max_retries = 3;
        std::chrono::system_clock::time_point expires_at;
    };

    using EventHandler = std::function<void(const Event&)>;
    using EventFilter = std::function<bool(const Event&)>;

    struct Subscription {
        std::string id;
        std::string channel_pattern;
        EventHandler handler;
        EventFilter filter;
        bool is_active = true;
        std::atomic<uint64_t> events_processed{0};
        std::atomic<uint64_t> events_filtered{0};
        std::atomic<uint64_t> processing_errors{0};
    };

    struct Stats {
        std::atomic<uint64_t> events_published{0};
        std::atomic<uint64_t> events_received{0};
        std::atomic<uint64_t> events_processed{0};
        std::atomic<uint64_t> events_failed{0};
        std::atomic<uint64_t> events_retried{0};
        std::atomic<uint64_t> events_expired{0};
        std::atomic<uint64_t> subscriptions_active{0};
        std::atomic<double> avg_processing_time_ms{0.0};
        std::atomic<uint64_t> queue_depth{0};
        std::atomic<uint64_t> redis_operations{0};
        std::atomic<uint64_t> redis_errors{0};
    };

    explicit EventBridge(const Config& config);
    ~EventBridge();

    // Lifecycle management
    bool initialize();
    void shutdown();
    bool is_running() const { return running_.load(); }

    // Event publishing
    bool publish_event(const Event& event);
    bool publish_event(EventType type, 
                      const std::string& channel,
                      const std::string& payload,
                      const std::string& target_system = "",
                      uint32_t priority = 5);
    
    // Batch publishing for high throughput
    bool publish_events(const std::vector<Event>& events);
    
    // Event subscription
    std::string subscribe(const std::string& channel_pattern, 
                         EventHandler handler,
                         EventFilter filter = nullptr);
    bool unsubscribe(const std::string& subscription_id);
    void unsubscribe_all();
    
    // Channel management
    std::vector<std::string> get_active_channels();
    uint64_t get_channel_message_count(const std::string& channel);
    
    // Event querying (for persistent events)
    std::vector<Event> get_events(const std::string& channel,
                                 std::chrono::system_clock::time_point since,
                                 uint32_t limit = 100);
    std::vector<Event> get_events_by_type(EventType type,
                                         std::chrono::system_clock::time_point since,
                                         uint32_t limit = 100);
    
    // Statistics and monitoring
    Stats get_stats() const;
    void reset_stats();
    std::vector<Subscription> get_active_subscriptions();
    
    // Health check
    bool health_check();
    
    // Event replay for recovery
    bool replay_events(const std::string& channel,
                      std::chrono::system_clock::time_point from,
                      std::chrono::system_clock::time_point to);

private:
    Config config_;
    std::atomic<bool> running_{false};
    Stats stats_;
    
    // Redis connection for pub/sub
    class RedisSubscriber;
    class RedisPublisher;
    std::unique_ptr<RedisSubscriber> subscriber_;
    std::unique_ptr<RedisPublisher> publisher_;
    
    // Event processing
    std::vector<std::thread> worker_threads_;
    std::queue<Event> event_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    
    // Subscriptions management
    std::unordered_map<std::string, std::unique_ptr<Subscription>> subscriptions_;
    std::mutex subscriptions_mutex_;
    
    // Event batching
    std::vector<Event> batch_buffer_;
    std::mutex batch_mutex_;
    std::thread batch_thread_;
    std::chrono::steady_clock::time_point last_batch_time_;
    
    // Internal methods
    void worker_loop();
    void batch_processing_loop();
    void process_event(const Event& event);
    void process_received_event(const std::string& channel, const std::string& message);
    
    // Event serialization
    std::string serialize_event(const Event& event);
    std::optional<Event> deserialize_event(const std::string& data);
    
    // Channel management
    std::string get_channel_key(const std::string& channel);
    std::string get_event_key(const std::string& event_id);
    
    // Pattern matching for subscriptions
    bool matches_pattern(const std::string& channel, const std::string& pattern);
    
    // Event validation
    bool validate_event(const Event& event);
    
    // Retry mechanism
    void schedule_retry(const Event& event);
    
    // Compression helpers
    std::string compress_payload(const std::string& payload);
    std::string decompress_payload(const std::string& compressed_payload);
    
    // Metrics tracking
    void record_operation(const std::string& operation, double duration_ms, bool success);
};

/**
 * Predefined event builders for common use cases
 */
class EventBuilder {
public:
    static EventBridge::Event create_user_action_event(
        const std::string& user_id,
        const std::string& action,
        const std::string& resource,
        const std::unordered_map<std::string, std::string>& details = {}
    );
    
    static EventBridge::Event create_cache_invalidation_event(
        const std::string& cache_key_pattern,
        const std::string& reason = "manual"
    );
    
    static EventBridge::Event create_session_event(
        const std::string& session_id,
        const std::string& event_type, // created, updated, deleted, expired
        const std::string& user_id
    );
    
    static EventBridge::Event create_media_processing_event(
        const std::string& media_id,
        const std::string& operation, // upload, process, transcode, delete
        const std::string& status, // started, completed, failed
        const std::unordered_map<std::string, std::string>& metadata = {}
    );
    
    static EventBridge::Event create_performance_metric_event(
        const std::string& metric_name,
        double value,
        const std::string& unit,
        const std::unordered_map<std::string, std::string>& tags = {}
    );
    
    static EventBridge::Event create_error_event(
        const std::string& error_type,
        const std::string& error_message,
        const std::string& component,
        const std::unordered_map<std::string, std::string>& context = {}
    );
    
    static EventBridge::Event create_system_event(
        const std::string& event_name,
        const std::string& description,
        const std::unordered_map<std::string, std::string>& data = {}
    );

private:
    static std::string generate_event_id();
    static std::chrono::system_clock::time_point get_default_expiry();
};

/**
 * Event filters for common filtering scenarios
 */
class EventFilters {
public:
    static EventBridge::EventFilter by_source_system(const std::string& system);
    static EventBridge::EventFilter by_event_type(EventBridge::EventType type);
    static EventBridge::EventFilter by_priority_range(uint32_t min_priority, uint32_t max_priority);
    static EventBridge::EventFilter by_metadata_key(const std::string& key, const std::string& value);
    static EventBridge::EventFilter by_age_limit(std::chrono::seconds max_age);
    static EventBridge::EventFilter combine_and(const std::vector<EventBridge::EventFilter>& filters);
    static EventBridge::EventFilter combine_or(const std::vector<EventBridge::EventFilter>& filters);
};

} // namespace integration