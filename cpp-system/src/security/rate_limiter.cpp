#include "security/rate_limiter.hpp"
#include "common/logger.hpp"
#include <algorithm>
#include <thread>

namespace ultra_cpp::security {

// TokenBucket Implementation
TokenBucket::TokenBucket(const Config& config) 
    : config_(config), tokens_(config.capacity) {
    last_refill_.store(std::chrono::steady_clock::now(), std::memory_order_relaxed);
}

bool TokenBucket::try_consume(uint64_t tokens) {
    refill();
    
    uint64_t current_tokens = tokens_.load(std::memory_order_acquire);
    
    while (current_tokens >= tokens) {
        if (tokens_.compare_exchange_weak(current_tokens, current_tokens - tokens, 
                                         std::memory_order_acq_rel, 
                                         std::memory_order_acquire)) {
            return true;
        }
        // current_tokens is updated by compare_exchange_weak on failure
    }
    
    return false;
}

uint64_t TokenBucket::available_tokens() const {
    return tokens_.load(std::memory_order_acquire);
}

void TokenBucket::reset() {
    tokens_.store(config_.capacity, std::memory_order_release);
    last_refill_.store(std::chrono::steady_clock::now(), std::memory_order_release);
}

void TokenBucket::refill() {
    auto now = std::chrono::steady_clock::now();
    auto last_refill = last_refill_.load(std::memory_order_acquire);
    
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_refill);
    
    if (elapsed >= config_.refill_interval) {
        uint64_t tokens_to_add = (elapsed.count() * config_.refill_rate) / 1000;
        
        if (tokens_to_add > 0) {
            uint64_t current_tokens = tokens_.load(std::memory_order_acquire);
            uint64_t new_tokens = std::min(current_tokens + tokens_to_add, config_.capacity);
            
            // Try to update both tokens and last_refill atomically
            if (tokens_.compare_exchange_strong(current_tokens, new_tokens, 
                                              std::memory_order_acq_rel)) {
                last_refill_.store(now, std::memory_order_release);
            }
        }
    }
}

// RateLimiter Implementation
RateLimiter::RateLimiter(const GlobalConfig& config) 
    : global_config_(config), cleanup_running_(true) {
    
    // Start cleanup thread
    cleanup_thread_ = std::thread(&RateLimiter::cleanup_worker, this);
    
    LOG_INFO("RateLimiter initialized with {} max tenants, {} default RPS", 
             config.max_tenants, config.default_requests_per_second);
}

RateLimiter::~RateLimiter() {
    cleanup_running_.store(false, std::memory_order_release);
    
    if (cleanup_thread_.joinable()) {
        cleanup_thread_.join();
    }
    
    LOG_INFO("RateLimiter destroyed");
}

bool RateLimiter::add_tenant(const TenantConfig& tenant_config) {
    std::unique_lock lock(tenants_mutex_);
    
    if (tenants_.size() >= global_config_.max_tenants) {
        LOG_WARN("Cannot add tenant {}: max tenants limit reached", tenant_config.tenant_id);
        return false;
    }
    
    auto tenant_data = std::make_unique<TenantData>();
    tenant_data->config = tenant_config;
    
    TokenBucket::Config bucket_config{
        .capacity = tenant_config.burst_capacity,
        .refill_rate = tenant_config.requests_per_second
    };
    
    tenant_data->bucket = std::make_unique<TokenBucket>(bucket_config);
    tenant_data->stats.last_access = std::chrono::steady_clock::now();
    
    tenants_[tenant_config.tenant_id] = std::move(tenant_data);
    global_stats_.total_tenants.fetch_add(1, std::memory_order_relaxed);
    
    LOG_INFO("Added tenant {} with {} RPS, {} burst", 
             tenant_config.tenant_id, tenant_config.requests_per_second, 
             tenant_config.burst_capacity);
    
    return true;
}

bool RateLimiter::update_tenant(const TenantConfig& tenant_config) {
    std::unique_lock lock(tenants_mutex_);
    
    auto it = tenants_.find(tenant_config.tenant_id);
    if (it == tenants_.end()) {
        return false;
    }
    
    it->second->config = tenant_config;
    
    // Update bucket configuration
    TokenBucket::Config bucket_config{
        .capacity = tenant_config.burst_capacity,
        .refill_rate = tenant_config.requests_per_second
    };
    
    it->second->bucket = std::make_unique<TokenBucket>(bucket_config);
    
    LOG_INFO("Updated tenant {} with {} RPS, {} burst", 
             tenant_config.tenant_id, tenant_config.requests_per_second, 
             tenant_config.burst_capacity);
    
    return true;
}

void RateLimiter::remove_tenant(const std::string& tenant_id) {
    std::unique_lock lock(tenants_mutex_);
    
    auto it = tenants_.find(tenant_id);
    if (it != tenants_.end()) {
        tenants_.erase(it);
        global_stats_.total_tenants.fetch_sub(1, std::memory_order_relaxed);
        LOG_INFO("Removed tenant {}", tenant_id);
    }
}

bool RateLimiter::is_tenant_enabled(const std::string& tenant_id) {
    std::shared_lock lock(tenants_mutex_);
    
    auto it = tenants_.find(tenant_id);
    return it != tenants_.end() && it->second->config.enabled;
}

bool RateLimiter::is_allowed(const std::string& tenant_id, uint64_t tokens) {
    global_stats_.total_requests.fetch_add(1, std::memory_order_relaxed);
    
    TenantData* tenant_data = get_tenant_data(tenant_id);
    if (!tenant_data) {
        // Create default tenant if not exists
        TenantConfig default_config{
            .tenant_id = tenant_id,
            .requests_per_second = global_config_.default_requests_per_second,
            .burst_capacity = global_config_.default_burst_capacity,
            .enabled = true
        };
        
        if (add_tenant(default_config)) {
            tenant_data = get_tenant_data(tenant_id);
        }
        
        if (!tenant_data) {
            global_stats_.total_denied.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
    }
    
    if (!tenant_data->config.enabled) {
        global_stats_.total_denied.fetch_add(1, std::memory_order_relaxed);
        tenant_data->stats.requests_denied.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    
    tenant_data->stats.total_requests.fetch_add(1, std::memory_order_relaxed);
    tenant_data->stats.last_access = std::chrono::steady_clock::now();
    
    bool allowed = tenant_data->bucket->try_consume(tokens);
    
    if (allowed) {
        global_stats_.total_allowed.fetch_add(1, std::memory_order_relaxed);
        tenant_data->stats.requests_allowed.fetch_add(1, std::memory_order_relaxed);
    } else {
        global_stats_.total_denied.fetch_add(1, std::memory_order_relaxed);
        tenant_data->stats.requests_denied.fetch_add(1, std::memory_order_relaxed);
    }
    
    tenant_data->stats.current_tokens.store(
        tenant_data->bucket->available_tokens(), std::memory_order_relaxed);
    
    return allowed;
}

bool RateLimiter::is_allowed_with_key(const std::string& tenant_id, 
                                     const std::string& key, 
                                     uint64_t tokens) {
    TenantData* tenant_data = get_tenant_data(tenant_id);
    if (!tenant_data || !tenant_data->config.enabled) {
        return false;
    }
    
    TokenBucket* key_bucket = get_or_create_key_bucket(*tenant_data, key);
    if (!key_bucket) {
        return false;
    }
    
    return key_bucket->try_consume(tokens);
}

std::vector<RateLimiter::BulkResult> RateLimiter::check_bulk(
    const std::vector<BulkRequest>& requests) {
    
    std::vector<BulkResult> results;
    results.reserve(requests.size());
    
    for (const auto& request : requests) {
        bool allowed;
        uint64_t remaining = 0;
        
        if (request.key.empty()) {
            allowed = is_allowed(request.tenant_id, request.tokens);
        } else {
            allowed = is_allowed_with_key(request.tenant_id, request.key, request.tokens);
        }
        
        if (allowed) {
            TenantData* tenant_data = get_tenant_data(request.tenant_id);
            if (tenant_data) {
                remaining = tenant_data->bucket->available_tokens();
            }
        }
        
        results.push_back({allowed, remaining});
    }
    
    return results;
}

RateLimiter::TenantStats RateLimiter::get_tenant_stats(const std::string& tenant_id) {
    TenantData* tenant_data = get_tenant_data(tenant_id);
    if (tenant_data) {
        return tenant_data->stats;
    }
    
    return TenantStats{};
}

void RateLimiter::update_global_config(const GlobalConfig& config) {
    global_config_ = config;
    LOG_INFO("Updated global rate limiter configuration");
}

void RateLimiter::cleanup_inactive_tenants() {
    auto now = std::chrono::steady_clock::now();
    std::vector<std::string> to_remove;
    
    {
        std::shared_lock lock(tenants_mutex_);
        for (const auto& [tenant_id, tenant_data] : tenants_) {
            auto idle_time = std::chrono::duration_cast<std::chrono::seconds>(
                now - tenant_data->stats.last_access);
            
            if (idle_time > global_config_.tenant_ttl) {
                to_remove.push_back(tenant_id);
            }
        }
    }
    
    for (const auto& tenant_id : to_remove) {
        remove_tenant(tenant_id);
    }
    
    if (!to_remove.empty()) {
        LOG_INFO("Cleaned up {} inactive tenants", to_remove.size());
    }
    
    global_stats_.cleanup_runs.fetch_add(1, std::memory_order_relaxed);
}

void RateLimiter::reset_all_buckets() {
    std::shared_lock lock(tenants_mutex_);
    
    for (auto& [tenant_id, tenant_data] : tenants_) {
        tenant_data->bucket->reset();
        
        std::unique_lock key_lock(tenant_data->key_buckets_mutex);
        for (auto& [key, key_bucket] : tenant_data->key_buckets) {
            key_bucket->reset();
        }
    }
    
    LOG_INFO("Reset all rate limiter buckets");
}

void RateLimiter::cleanup_worker() {
    while (cleanup_running_.load(std::memory_order_acquire)) {
        std::this_thread::sleep_for(global_config_.cleanup_interval);
        
        if (cleanup_running_.load(std::memory_order_acquire)) {
            cleanup_inactive_tenants();
        }
    }
}

TokenBucket* RateLimiter::get_or_create_key_bucket(TenantData& tenant_data, 
                                                  const std::string& key) {
    {
        std::shared_lock lock(tenant_data.key_buckets_mutex);
        auto it = tenant_data.key_buckets.find(key);
        if (it != tenant_data.key_buckets.end()) {
            return it->second.get();
        }
    }
    
    // Create new key bucket
    std::unique_lock lock(tenant_data.key_buckets_mutex);
    
    // Double-check after acquiring exclusive lock
    auto it = tenant_data.key_buckets.find(key);
    if (it != tenant_data.key_buckets.end()) {
        return it->second.get();
    }
    
    TokenBucket::Config bucket_config{
        .capacity = tenant_data.config.burst_capacity,
        .refill_rate = tenant_data.config.requests_per_second
    };
    
    auto key_bucket = std::make_unique<TokenBucket>(bucket_config);
    TokenBucket* bucket_ptr = key_bucket.get();
    
    tenant_data.key_buckets[key] = std::move(key_bucket);
    
    return bucket_ptr;
}

RateLimiter::TenantData* RateLimiter::get_tenant_data(const std::string& tenant_id) {
    std::shared_lock lock(tenants_mutex_);
    
    auto it = tenants_.find(tenant_id);
    return (it != tenants_.end()) ? it->second.get() : nullptr;
}

// LockFreeRateLimiter Implementation
LockFreeRateLimiter::LockFreeRateLimiter(const Config& config) 
    : config_(config) {
    
    buckets_ = std::make_unique<TenantBucket[]>(config_.max_tenants);
    
    // Initialize all buckets
    for (uint64_t i = 0; i < config_.max_tenants; ++i) {
        buckets_[i].tokens.store(config_.default_burst, std::memory_order_relaxed);
        buckets_[i].rate.store(config_.default_rate, std::memory_order_relaxed);
        buckets_[i].capacity.store(config_.default_burst, std::memory_order_relaxed);
        buckets_[i].last_refill_ns.store(
            std::chrono::steady_clock::now().time_since_epoch().count(), 
            std::memory_order_relaxed);
    }
    
    LOG_INFO("LockFreeRateLimiter initialized with {} buckets", config_.max_tenants);
}

LockFreeRateLimiter::~LockFreeRateLimiter() = default;

bool LockFreeRateLimiter::is_allowed(uint64_t tenant_hash, uint64_t tokens) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    stats_.requests_processed.fetch_add(1, std::memory_order_relaxed);
    
    uint64_t bucket_index = get_bucket_index(tenant_hash);
    TenantBucket& bucket = buckets_[bucket_index];
    
    auto current_time_ns = std::chrono::steady_clock::now().time_since_epoch().count();
    refill_bucket(bucket, current_time_ns);
    
    // Try to consume tokens atomically
    uint64_t current_tokens = bucket.tokens.load(std::memory_order_acquire);
    
    while (current_tokens >= tokens) {
        if (bucket.tokens.compare_exchange_weak(current_tokens, current_tokens - tokens,
                                               std::memory_order_acq_rel,
                                               std::memory_order_acquire)) {
            stats_.requests_allowed.fetch_add(1, std::memory_order_relaxed);
            
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
                end_time - start_time);
            stats_.avg_processing_time_ns.store(duration.count(), std::memory_order_relaxed);
            
            return true;
        }
    }
    
    stats_.requests_denied.fetch_add(1, std::memory_order_relaxed);
    return false;
}

uint64_t LockFreeRateLimiter::get_bucket_index(uint64_t tenant_hash) const {
    return tenant_hash % config_.max_tenants;
}

void LockFreeRateLimiter::refill_bucket(TenantBucket& bucket, uint64_t current_time_ns) {
    uint64_t last_refill = bucket.last_refill_ns.load(std::memory_order_acquire);
    uint64_t elapsed_ns = current_time_ns - last_refill;
    
    // Refill every second (1,000,000,000 ns)
    if (elapsed_ns >= 1000000000ULL) {
        uint64_t rate = bucket.rate.load(std::memory_order_acquire);
        uint64_t capacity = bucket.capacity.load(std::memory_order_acquire);
        
        uint64_t tokens_to_add = (elapsed_ns * rate) / 1000000000ULL;
        
        if (tokens_to_add > 0) {
            uint64_t current_tokens = bucket.tokens.load(std::memory_order_acquire);
            uint64_t new_tokens = std::min(current_tokens + tokens_to_add, capacity);
            
            // Try to update tokens and last_refill
            if (bucket.tokens.compare_exchange_strong(current_tokens, new_tokens,
                                                     std::memory_order_acq_rel)) {
                bucket.last_refill_ns.store(current_time_ns, std::memory_order_release);
            }
        }
    }
}

} // namespace ultra_cpp::security