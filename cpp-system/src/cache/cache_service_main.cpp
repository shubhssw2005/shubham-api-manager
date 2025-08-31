#include "cache/ultra_cache.hpp"
#include "common/types.hpp"
#include <iostream>
#include <string>
#include <thread>
#include <chrono>
#include <signal.h>
#include <atomic>

namespace {
    std::atomic<bool> g_running{true};
    
    void signal_handler(int signal) {
        std::cout << "Received signal " << signal << ", shutting down..." << std::endl;
        g_running.store(false, std::memory_order_release);
    }
}

int main(int argc, char* argv[]) {
    // Install signal handlers
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    std::cout << "Ultra-Fast Cache Service Starting..." << std::endl;
    
    // Configure cache
    ultra::cache::UltraCache<std::string, std::string>::Config config;
    config.capacity = 1000000;  // 1M entries
    config.shard_count = 64;    // 64 shards for good parallelism
    config.enable_rdma = false; // Disable RDMA for now
    config.enable_predictive_loading = true;
    config.eviction_policy = ultra::cache::UltraCache<std::string, std::string>::Config::EvictionPolicy::LRU;
    
    // Create cache instance
    auto cache = std::make_unique<ultra::cache::UltraCache<std::string, std::string>>(config);
    
    std::cout << "Cache initialized with capacity: " << config.capacity << std::endl;
    std::cout << "Shard count: " << config.shard_count << std::endl;
    std::cout << "RDMA enabled: " << (config.enable_rdma ? "yes" : "no") << std::endl;
    std::cout << "Predictive loading: " << (config.enable_predictive_loading ? "yes" : "no") << std::endl;
    
    // Enable predictive loading
    cache->enable_predictive_loading(true);
    
    // Populate cache with some test data
    std::cout << "Populating cache with test data..." << std::endl;
    
    for (int i = 0; i < 10000; ++i) {
        std::string key = "key_" + std::to_string(i);
        std::string value = "value_" + std::to_string(i) + "_data_content";
        cache->put(key, value);
        
        if (i % 1000 == 0) {
            std::cout << "Inserted " << i << " entries..." << std::endl;
        }
    }
    
    std::cout << "Cache populated. Starting performance monitoring..." << std::endl;
    
    // Performance monitoring loop
    auto last_stats_time = std::chrono::high_resolution_clock::now();
    auto last_stats = cache->get_stats();
    
    while (g_running.load(std::memory_order_relaxed)) {
        std::this_thread::sleep_for(std::chrono::seconds(5));
        
        auto current_time = std::chrono::high_resolution_clock::now();
        auto current_stats = cache->get_stats();
        
        auto time_diff = std::chrono::duration_cast<std::chrono::milliseconds>(
            current_time - last_stats_time).count();
        
        if (time_diff > 0) {
            // Calculate rates
            auto ops_diff = current_stats.total_operations.load() - last_stats.total_operations.load();
            auto hits_diff = current_stats.hits.load() - last_stats.hits.load();
            auto misses_diff = current_stats.misses.load() - last_stats.misses.load();
            auto evictions_diff = current_stats.evictions.load() - last_stats.evictions.load();
            
            double ops_per_sec = (ops_diff * 1000.0) / time_diff;
            double hit_ratio = (ops_diff > 0) ? (hits_diff * 100.0) / ops_diff : 0.0;
            
            std::cout << "\n=== Cache Performance Stats ===" << std::endl;
            std::cout << "Operations/sec: " << static_cast<uint64_t>(ops_per_sec) << std::endl;
            std::cout << "Hit ratio: " << hit_ratio << "%" << std::endl;
            std::cout << "Total hits: " << current_stats.hits.load() << std::endl;
            std::cout << "Total misses: " << current_stats.misses.load() << std::endl;
            std::cout << "Total evictions: " << current_stats.evictions.load() << std::endl;
            std::cout << "Cache size: " << current_stats.cache_size.load() << std::endl;
            std::cout << "Memory usage: " << current_stats.memory_usage_bytes.load() << " bytes" << std::endl;
            
            if (current_stats.avg_get_latency_ns.load() > 0) {
                std::cout << "Avg GET latency: " << current_stats.avg_get_latency_ns.load() << " ns" << std::endl;
                std::cout << "Max GET latency: " << current_stats.max_get_latency_ns.load() << " ns" << std::endl;
            }
            
            if (current_stats.avg_put_latency_ns.load() > 0) {
                std::cout << "Avg PUT latency: " << current_stats.avg_put_latency_ns.load() << " ns" << std::endl;
                std::cout << "Max PUT latency: " << current_stats.max_put_latency_ns.load() << " ns" << std::endl;
            }
            
            if (config.enable_predictive_loading) {
                std::cout << "Predictions made: " << current_stats.predictions_made.load() << std::endl;
                std::cout << "Predictions hit: " << current_stats.predictions_hit.load() << std::endl;
                std::cout << "Warmup operations: " << current_stats.warmup_operations.load() << std::endl;
            }
            
            std::cout << "==============================\n" << std::endl;
        }
        
        last_stats_time = current_time;
        last_stats = current_stats;
        
        // Perform some random cache operations to generate activity
        static int counter = 0;
        for (int i = 0; i < 100; ++i) {
            std::string key = "key_" + std::to_string((counter + i) % 10000);
            auto value = cache->get(key);
            
            // Occasionally update values
            if (i % 10 == 0) {
                std::string new_value = "updated_value_" + std::to_string(counter + i);
                cache->put(key, new_value);
            }
        }
        counter += 100;
    }
    
    std::cout << "Shutting down cache service..." << std::endl;
    
    // Print final statistics
    auto final_stats = cache->get_stats();
    std::cout << "\n=== Final Cache Statistics ===" << std::endl;
    std::cout << "Total operations: " << final_stats.total_operations.load() << std::endl;
    std::cout << "Total hits: " << final_stats.hits.load() << std::endl;
    std::cout << "Total misses: " << final_stats.misses.load() << std::endl;
    std::cout << "Total evictions: " << final_stats.evictions.load() << std::endl;
    std::cout << "Final cache size: " << final_stats.cache_size.load() << std::endl;
    
    double final_hit_ratio = (final_stats.total_operations.load() > 0) ? 
        (final_stats.hits.load() * 100.0) / final_stats.total_operations.load() : 0.0;
    std::cout << "Overall hit ratio: " << final_hit_ratio << "%" << std::endl;
    std::cout << "===============================" << std::endl;
    
    cache.reset();
    
    std::cout << "Cache service shutdown complete." << std::endl;
    return 0;
}