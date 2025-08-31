#pragma once

#include <string>
#include <unordered_map>
#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <shared_mutex>

namespace ultra_cpp::security {

class TokenBucket {
public:
    struct Config {
        uint64_t capacity;           // Maximum tokens in bucket
        uint64_t refill_rate;        // Tokens per second
        std::chrono::milliseconds refill_interval{1000}; // Refill frequency
    };

    explicit TokenBucket(const Config& config);

    // Try to consume tokens (thread-safe)
    bool try_consume(uint64_t tokens = 1);
    
    // Get current token count
    uint64_t available_tokens() const;
    
    // Get bucket configuration
    const Config& get_config() const { return config_; }

    // Reset bucket to full capacity
    void reset();

private:
    Config config_;
    std::atomic<uint64_t> tokens_;
    std::atomic<std::chrono::steady_clock::time_point> last_refill_;

    void refill();
};

class RateLimiter {
public:
    struct TenantConfig {
        std::string tenant_id;
        uint64_t requests_per_second;
        uint64_t burst_capacity;
        bool enabled = true;
    };

    struct GlobalConfig {
        uint64_t default_requests_per_second = 1000;
        uint64_t default_burst_capacity = 2000;
        uint64_t max_tenants = 10000;
        std::chrono::seconds cleanup_interval{300}; // 5 minutes
        std::chrono::seconds tenant_ttl{3600};      // 1 hour
    };

    explicit RateLimiter(const GlobalConfig& config);
    ~RateLimiter();

    // Tenant management
    bool add_tenant(const TenantConfig& tenant_config);
    bool update_tenant(const TenantConfig& tenant_config);
    void remove_tenant(const std::string& tenant_id);
    bool is_tenant_enabled(const std::string& tenant_id);

    // Rate limiting
    bool is_allowed(const std::string& tenant_id, uint64_t tokens = 1);
    bool is_allowed_with_key(const std::string& tenant_id, 
                            const std::string& key, 
                            uint64_t tokens = 1);

    // Bulk operations for performance
    struct BulkRequest {
        std::string tenant_id;
        std::string key;
        uint64_t tokens;
    };

    struct BulkResult {
        bool allowed;
        uint64_t remaining_tokens;
    };

    std::vector<BulkResult> check_bulk(const std::vector<BulkRequest>& requests);

    // Statistics and monitoring
    struct TenantStats {
        std::atomic<uint64_t> requests_allowed{0};
        std::atomic<uint64_t> requests_denied{0};
        std::atomic<uint64_t> total_requests{0};
        std::atomic<uint64_t> current_tokens{0};
        std::chrono::steady_clock::time_point last_access;
    };

    struct GlobalStats {
        std::atomic<uint64_t> total_tenants{0};
        std::atomic<uint64_t> active_tenants{0};
        std::atomic<uint64_t> total_requests{0};
        std::atomic<uint64_t> total_allowed{0};
        std::atomic<uint64_t> total_denied{0};
        std::atomic<uint64_t> cleanup_runs{0};
    };

    TenantStats get_tenant_stats(const std::string& tenant_id);
    GlobalStats get_global_stats() const { return global_stats_; }

    // Configuration updates
    void update_global_config(const GlobalConfig& config);
    GlobalConfig get_global_config() const { return global_config_; }

    // Cleanup and maintenance
    void cleanup_inactive_tenants();
    void reset_all_buckets();

private:
    struct TenantData {
        TenantConfig config;
        std::unique_ptr<TokenBucket> bucket;
        TenantStats stats;
        std::unordered_map<std::string, std::unique_ptr<TokenBucket>> key_buckets;
        mutable std::shared_mutex key_buckets_mutex;
    };

    GlobalConfig global_config_;
    mutable GlobalStats global_stats_;
    
    std::unordered_map<std::string, std::unique_ptr<TenantData>> tenants_;
    mutable std::shared_mutex tenants_mutex_;

    // Background cleanup
    std::atomic<bool> cleanup_running_{false};
    std::thread cleanup_thread_;

    void cleanup_worker();
    TokenBucket* get_or_create_key_bucket(TenantData& tenant_data, const std::string& key);
    TenantData* get_tenant_data(const std::string& tenant_id);
};

// High-performance rate limiter using lock-free data structures
class LockFreeRateLimiter {
public:
    struct Config {
        uint64_t max_tenants = 10000;
        uint64_t default_rate = 1000;  // requests per second
        uint64_t default_burst = 2000; // burst capacity
    };

    explicit LockFreeRateLimiter(const Config& config);
    ~LockFreeRateLimiter();

    bool is_allowed(uint64_t tenant_hash, uint64_t tokens = 1);
    
    struct Stats {
        std::atomic<uint64_t> requests_processed{0};
        std::atomic<uint64_t> requests_allowed{0};
        std::atomic<uint64_t> requests_denied{0};
        std::atomic<uint64_t> avg_processing_time_ns{0};
    };

    Stats get_stats() const { return stats_; }

private:
    struct alignas(64) TenantBucket {  // Cache line aligned
        std::atomic<uint64_t> tokens{0};
        std::atomic<uint64_t> last_refill_ns{0};
        std::atomic<uint64_t> rate{1000};
        std::atomic<uint64_t> capacity{2000};
        
        // Padding to prevent false sharing
        char padding[64 - 4 * sizeof(std::atomic<uint64_t>)];
    };

    Config config_;
    mutable Stats stats_;
    std::unique_ptr<TenantBucket[]> buckets_;
    
    uint64_t get_bucket_index(uint64_t tenant_hash) const;
    void refill_bucket(TenantBucket& bucket, uint64_t current_time_ns);
};

} // namespace ultra_cpp::security