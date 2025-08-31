#pragma once

#include <string>
#include <vector>
#include <memory>
#include <atomic>
#include <chrono>
#include <unordered_map>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <fstream>
#include <openssl/sha.h>
#include <openssl/hmac.h>

namespace ultra_cpp::security {

enum class SecurityEventType {
    AUTHENTICATION_SUCCESS,
    AUTHENTICATION_FAILURE,
    AUTHORIZATION_FAILURE,
    RATE_LIMIT_EXCEEDED,
    SUSPICIOUS_REQUEST,
    SQL_INJECTION_ATTEMPT,
    XSS_ATTEMPT,
    PATH_TRAVERSAL_ATTEMPT,
    INVALID_TOKEN,
    TOKEN_EXPIRED,
    PRIVILEGE_ESCALATION,
    DATA_ACCESS,
    CONFIGURATION_CHANGE,
    SYSTEM_ERROR,
    CUSTOM_EVENT
};

enum class SecurityLevel {
    LOW,
    MEDIUM,
    HIGH,
    CRITICAL
};

struct SecurityEvent {
    uint64_t event_id;
    SecurityEventType event_type;
    SecurityLevel severity;
    std::chrono::system_clock::time_point timestamp;
    std::string user_id;
    std::string tenant_id;
    std::string source_ip;
    std::string user_agent;
    std::string resource;
    std::string action;
    std::string details;
    std::unordered_map<std::string, std::string> metadata;
    
    // Integrity fields
    std::string hash;
    std::string previous_hash;
    uint64_t sequence_number;
};

class TamperEvidenceEngine {
public:
    struct Config {
        std::string hmac_key;
        std::string hash_algorithm = "SHA256";
        bool enable_chain_validation = true;
        size_t max_chain_length = 1000000;
    };

    explicit TamperEvidenceEngine(const Config& config);
    
    // Hash computation
    std::string compute_event_hash(const SecurityEvent& event);
    std::string compute_chain_hash(const std::string& current_hash, 
                                  const std::string& previous_hash);
    
    // Chain validation
    bool validate_event_integrity(const SecurityEvent& event);
    bool validate_chain_integrity(const std::vector<SecurityEvent>& events);
    
    // HMAC operations
    std::string compute_hmac(const std::string& data);
    bool verify_hmac(const std::string& data, const std::string& hmac);
    
    // Key management
    void rotate_hmac_key(const std::string& new_key);
    std::string get_key_fingerprint() const;

private:
    Config config_;
    std::string current_hmac_key_;
    std::atomic<uint64_t> sequence_counter_{0};
    
    std::string serialize_event_for_hash(const SecurityEvent& event);
};

class AuditStorage {
public:
    struct StorageConfig {
        std::string storage_path = "/var/log/security/audit";
        std::string backup_path = "/var/log/security/backup";
        size_t max_file_size = 100 * 1024 * 1024;  // 100MB
        size_t max_files = 1000;
        bool compress_old_files = true;
        bool encrypt_storage = true;
        std::string encryption_key;
        std::chrono::hours retention_period{24 * 30};  // 30 days
    };

    explicit AuditStorage(const StorageConfig& config);
    ~AuditStorage();
    
    // Storage operations
    bool store_event(const SecurityEvent& event);
    bool store_events_batch(const std::vector<SecurityEvent>& events);
    
    // Retrieval operations
    std::vector<SecurityEvent> get_events_by_time_range(
        const std::chrono::system_clock::time_point& start,
        const std::chrono::system_clock::time_point& end);
    
    std::vector<SecurityEvent> get_events_by_user(const std::string& user_id);
    std::vector<SecurityEvent> get_events_by_type(SecurityEventType event_type);
    std::vector<SecurityEvent> get_events_by_severity(SecurityLevel severity);
    
    // Maintenance operations
    void rotate_log_files();
    void cleanup_old_files();
    void backup_current_logs();
    bool verify_storage_integrity();
    
    // Statistics
    struct StorageStats {
        std::atomic<uint64_t> events_stored{0};
        std::atomic<uint64_t> storage_errors{0};
        std::atomic<uint64_t> files_rotated{0};
        std::atomic<uint64_t> bytes_written{0};
        std::atomic<uint64_t> integrity_checks{0};
        std::atomic<uint64_t> integrity_failures{0};
    };
    
    StorageStats get_stats() const { return stats_; }

private:
    StorageConfig config_;
    mutable StorageStats stats_;
    std::ofstream current_file_;
    std::string current_filename_;
    std::atomic<size_t> current_file_size_{0};
    mutable std::mutex file_mutex_;
    
    // File management
    std::string generate_filename();
    bool open_new_file();
    void close_current_file();
    std::string encrypt_data(const std::string& data);
    std::string decrypt_data(const std::string& encrypted_data);
    
    // Serialization
    std::string serialize_event(const SecurityEvent& event);
    SecurityEvent deserialize_event(const std::string& data);
};

class AuditLogger {
public:
    struct Config {
        TamperEvidenceEngine::Config tamper_config;
        AuditStorage::StorageConfig storage_config;
        size_t buffer_size = 10000;
        std::chrono::milliseconds flush_interval{1000};
        size_t worker_threads = 2;
        bool async_logging = true;
        SecurityLevel min_log_level = SecurityLevel::LOW;
    };

    explicit AuditLogger(const Config& config);
    ~AuditLogger();
    
    // Event logging
    void log_event(const SecurityEvent& event);
    void log_authentication_success(const std::string& user_id, 
                                   const std::string& source_ip);
    void log_authentication_failure(const std::string& user_id, 
                                   const std::string& source_ip, 
                                   const std::string& reason);
    void log_authorization_failure(const std::string& user_id, 
                                  const std::string& resource, 
                                  const std::string& action);
    void log_rate_limit_exceeded(const std::string& tenant_id, 
                                const std::string& source_ip);
    void log_suspicious_request(const std::string& details, 
                               const std::string& source_ip);
    void log_injection_attempt(SecurityEventType type, 
                              const std::string& payload, 
                              const std::string& source_ip);
    void log_token_event(SecurityEventType type, 
                        const std::string& user_id, 
                        const std::string& token_id);
    void log_data_access(const std::string& user_id, 
                        const std::string& resource, 
                        const std::string& action);
    void log_configuration_change(const std::string& user_id, 
                                 const std::string& component, 
                                 const std::string& changes);
    void log_system_error(const std::string& component, 
                         const std::string& error_details);
    void log_custom_event(const std::string& event_name, 
                         const std::unordered_map<std::string, std::string>& metadata);
    
    // Batch operations
    void log_events_batch(const std::vector<SecurityEvent>& events);
    
    // Control operations
    void flush();
    void start();
    void stop();
    
    // Query operations
    std::vector<SecurityEvent> query_events(
        const std::chrono::system_clock::time_point& start,
        const std::chrono::system_clock::time_point& end,
        SecurityEventType event_type = SecurityEventType::CUSTOM_EVENT);
    
    // Integrity operations
    bool verify_log_integrity();
    std::string generate_integrity_report();
    
    // Configuration
    void update_config(const Config& config);
    void set_min_log_level(SecurityLevel level);
    
    // Statistics
    struct LoggerStats {
        std::atomic<uint64_t> events_logged{0};
        std::atomic<uint64_t> events_dropped{0};
        std::atomic<uint64_t> flush_operations{0};
        std::atomic<uint64_t> integrity_checks{0};
        std::atomic<uint64_t> avg_log_time_ns{0};
        std::atomic<uint64_t> buffer_overflows{0};
    };
    
    LoggerStats get_stats() const { return stats_; }

private:
    Config config_;
    mutable LoggerStats stats_;
    
    std::unique_ptr<TamperEvidenceEngine> tamper_engine_;
    std::unique_ptr<AuditStorage> storage_;
    
    // Async logging infrastructure
    std::queue<SecurityEvent> event_buffer_;
    mutable std::mutex buffer_mutex_;
    std::condition_variable buffer_condition_;
    std::vector<std::thread> worker_threads_;
    std::atomic<bool> shutdown_{false};
    
    // Event ID generation
    std::atomic<uint64_t> event_id_counter_{0};
    
    // Worker thread functions
    void worker_thread_func();
    void flush_worker();
    
    // Event processing
    SecurityEvent create_base_event(SecurityEventType type, SecurityLevel severity);
    void process_event(SecurityEvent& event);
    bool should_log_event(const SecurityEvent& event);
    
    // Context extraction
    std::string get_current_user_id();
    std::string get_current_tenant_id();
    std::string get_source_ip();
    std::string get_user_agent();
};

// High-performance audit logger for ultra-low latency scenarios
class FastAuditLogger {
public:
    struct Config {
        size_t ring_buffer_size = 1048576;  // 1M events
        std::string storage_path = "/var/log/security/fast_audit";
        bool memory_mapped_storage = true;
        size_t batch_size = 1000;
    };

    explicit FastAuditLogger(const Config& config);
    ~FastAuditLogger();
    
    // Ultra-fast logging (lock-free)
    bool log_event_fast(SecurityEventType type, 
                       uint64_t user_hash, 
                       uint64_t resource_hash,
                       uint32_t details_hash);
    
    // Batch flush to persistent storage
    void flush_to_storage();
    
    // Statistics
    struct FastStats {
        std::atomic<uint64_t> events_logged{0};
        std::atomic<uint64_t> events_dropped{0};
        std::atomic<uint64_t> avg_log_time_ns{0};
        std::atomic<uint64_t> ring_buffer_wraps{0};
    };
    
    FastStats get_stats() const { return stats_; }

private:
    struct alignas(64) FastEvent {  // Cache line aligned
        std::atomic<uint64_t> timestamp_ns{0};
        std::atomic<uint32_t> event_type{0};
        std::atomic<uint64_t> user_hash{0};
        std::atomic<uint64_t> resource_hash{0};
        std::atomic<uint32_t> details_hash{0};
        std::atomic<uint32_t> sequence{0};
    };

    Config config_;
    mutable FastStats stats_;
    
    std::unique_ptr<FastEvent[]> ring_buffer_;
    std::atomic<uint64_t> write_index_{0};
    std::atomic<uint64_t> read_index_{0};
    
    // Background flushing
    std::thread flush_thread_;
    std::atomic<bool> shutdown_{false};
    
    void flush_worker();
    uint64_t get_next_write_index();
};

} // namespace ultra_cpp::security