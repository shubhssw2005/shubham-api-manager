#pragma once

#include "common/types.hpp"
#include <optional>
#include <string>
#include <memory>
#include <vector>
#include <atomic>

namespace ultra::cache {

template<typename Key, typename Value>
class UltraCache {
public:
    struct Config {
        size_t capacity = 1000000;
        size_t shard_count = 64;
        std::string backing_file;
        bool enable_rdma = false;
        std::string rdma_device = "mlx5_0";
        
        // Eviction policy configuration
        enum class EvictionPolicy {
            LRU,
            LFU,
            RANDOM,
            TTL_BASED
        };
        EvictionPolicy eviction_policy = EvictionPolicy::LRU;
        
        // Cache warming configuration
        bool enable_predictive_loading = false;
        size_t prediction_window_size = 1000;
        double prediction_threshold = 0.7;
        
        // Performance tuning
        size_t prefetch_batch_size = 64;
        std::chrono::milliseconds warmup_interval{100};
    };
    
    explicit UltraCache(const Config& config);
    ~UltraCache();
    
    // Core cache operations
    std::optional<Value> get(const Key& key) noexcept;
    void put(const Key& key, const Value& value) noexcept;
    void remove(const Key& key) noexcept;
    void clear() noexcept;
    
    // Batch operations for better performance
    std::vector<std::optional<Value>> get_batch(const std::vector<Key>& keys) noexcept;
    void put_batch(const std::vector<std::pair<Key, Value>>& items) noexcept;
    
    // Statistics and monitoring
    struct Stats {
        aligned_atomic<u64> hits{0};
        aligned_atomic<u64> misses{0};
        aligned_atomic<u64> evictions{0};
        aligned_atomic<u64> total_operations{0};
        aligned_atomic<u64> cache_size{0};
        aligned_atomic<u64> memory_usage_bytes{0};
        
        // Performance metrics
        aligned_atomic<u64> avg_get_latency_ns{0};
        aligned_atomic<u64> avg_put_latency_ns{0};
        aligned_atomic<u64> max_get_latency_ns{0};
        aligned_atomic<u64> max_put_latency_ns{0};
        
        // Predictive loading stats
        aligned_atomic<u64> predictions_made{0};
        aligned_atomic<u64> predictions_hit{0};
        aligned_atomic<u64> warmup_operations{0};
    };
    
    Stats get_stats() const noexcept;
    void reset_stats() noexcept;
    
    // Cache warming and predictive loading
    void warm_cache(const std::vector<Key>& keys);
    void enable_predictive_loading(bool enable);
    void set_prediction_threshold(double threshold);
    
    // Cache management
    void resize(size_t new_capacity);
    void set_eviction_policy(typename Config::EvictionPolicy policy);
    double get_hit_ratio() const noexcept;
    size_t get_size() const noexcept;
    
    // RDMA cluster operations
    void join_cluster(const std::vector<std::string>& peer_addresses);
    void leave_cluster();
    bool is_cluster_member() const noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> pimpl_;
};

// Specialized cache types for common use cases
using StringCache = UltraCache<std::string, std::string>;
using BinaryCache = UltraCache<std::string, std::vector<u8>>;
using NumericCache = UltraCache<u64, std::string>;

} // namespace ultra::cache