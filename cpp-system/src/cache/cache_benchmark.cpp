#include "cache/ultra_cache.hpp"
#include "common/types.hpp"
#include <iostream>
#include <chrono>
#include <thread>
#include <vector>
#include <random>
#include <string>
#include <iomanip>

using namespace ultra::cache;

struct BenchmarkConfig {
    size_t cache_capacity = 1000000;
    size_t shard_count = 64;
    size_t num_threads = std::thread::hardware_concurrency();
    size_t operations_per_thread = 100000;
    size_t key_space_size = 10000;
    double read_ratio = 0.8; // 80% reads, 20% writes
    bool enable_predictive_loading = true;
    size_t value_size = 64; // bytes
};

class CacheBenchmark {
public:
    explicit CacheBenchmark(const BenchmarkConfig& config) : config_(config) {
        // Configure cache
        typename UltraCache<std::string, std::string>::Config cache_config;
        cache_config.capacity = config.cache_capacity;
        cache_config.shard_count = config.shard_count;
        cache_config.enable_rdma = false;
        cache_config.enable_predictive_loading = config.enable_predictive_loading;
        cache_config.eviction_policy = UltraCache<std::string, std::string>::Config::EvictionPolicy::LRU;
        
        cache_ = std::make_unique<UltraCache<std::string, std::string>>(cache_config);
        
        // Pre-populate cache
        populate_cache();
    }
    
    void run_benchmark() {
        std::cout << "Starting Ultra Cache Benchmark..." << std::endl;
        std::cout << "Configuration:" << std::endl;
        std::cout << "  Cache capacity: " << config_.cache_capacity << std::endl;
        std::cout << "  Shard count: " << config_.shard_count << std::endl;
        std::cout << "  Threads: " << config_.num_threads << std::endl;
        std::cout << "  Operations per thread: " << config_.operations_per_thread << std::endl;
        std::cout << "  Key space size: " << config_.key_space_size << std::endl;
        std::cout << "  Read ratio: " << (config_.read_ratio * 100) << "%" << std::endl;
        std::cout << "  Value size: " << config_.value_size << " bytes" << std::endl;
        std::cout << std::endl;
        
        // Reset statistics
        cache_->reset_stats();
        
        // Run benchmark
        auto start_time = std::chrono::high_resolution_clock::now();
        
        std::vector<std::thread> threads;
        std::vector<ThreadStats> thread_stats(config_.num_threads);
        
        for (size_t i = 0; i < config_.num_threads; ++i) {
            threads.emplace_back(&CacheBenchmark::worker_thread, this, i, std::ref(thread_stats[i]));
        }
        
        for (auto& thread : threads) {
            thread.join();
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        
        // Calculate and display results
        display_results(start_time, end_time, thread_stats);
    }

private:
    struct ThreadStats {
        uint64_t operations = 0;
        uint64_t reads = 0;
        uint64_t writes = 0;
        uint64_t hits = 0;
        uint64_t misses = 0;
        uint64_t total_latency_ns = 0;
        uint64_t min_latency_ns = UINT64_MAX;
        uint64_t max_latency_ns = 0;
    };
    
    BenchmarkConfig config_;
    std::unique_ptr<UltraCache<std::string, std::string>> cache_;
    
    void populate_cache() {
        std::cout << "Pre-populating cache..." << std::endl;
        
        // Fill cache to ~50% capacity to ensure some hits
        size_t populate_count = config_.cache_capacity / 2;
        
        for (size_t i = 0; i < populate_count; ++i) {
            std::string key = generate_key(i % config_.key_space_size);
            std::string value = generate_value(i);
            cache_->put(key, value);
            
            if (i % 10000 == 0) {
                std::cout << "  Populated " << i << " entries..." << std::endl;
            }
        }
        
        std::cout << "Pre-population complete." << std::endl;
    }
    
    void worker_thread(size_t thread_id, ThreadStats& stats) {
        std::random_device rd;
        std::mt19937 gen(rd() + thread_id);
        std::uniform_real_distribution<> op_dist(0.0, 1.0);
        std::uniform_int_distribution<size_t> key_dist(0, config_.key_space_size - 1);
        
        for (size_t i = 0; i < config_.operations_per_thread; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            
            bool is_read = op_dist(gen) < config_.read_ratio;
            size_t key_idx = key_dist(gen);
            std::string key = generate_key(key_idx);
            
            if (is_read) {
                // Read operation
                auto result = cache_->get(key);
                stats.reads++;
                
                if (result.has_value()) {
                    stats.hits++;
                } else {
                    stats.misses++;
                }
            } else {
                // Write operation
                std::string value = generate_value(thread_id * config_.operations_per_thread + i);
                cache_->put(key, value);
                stats.writes++;
            }
            
            auto end = std::chrono::high_resolution_clock::now();
            auto latency_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
            
            stats.operations++;
            stats.total_latency_ns += latency_ns;
            stats.min_latency_ns = std::min(stats.min_latency_ns, static_cast<uint64_t>(latency_ns));
            stats.max_latency_ns = std::max(stats.max_latency_ns, static_cast<uint64_t>(latency_ns));
        }
    }
    
    std::string generate_key(size_t index) {
        return "benchmark_key_" + std::to_string(index);
    }
    
    std::string generate_value(size_t index) {
        std::string value = "benchmark_value_" + std::to_string(index) + "_";
        
        // Pad to desired size
        while (value.size() < config_.value_size) {
            value += "x";
        }
        
        return value.substr(0, config_.value_size);
    }
    
    void display_results(
        const std::chrono::high_resolution_clock::time_point& start_time,
        const std::chrono::high_resolution_clock::time_point& end_time,
        const std::vector<ThreadStats>& thread_stats) {
        
        auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        
        // Aggregate thread statistics
        ThreadStats total_stats;
        for (const auto& stats : thread_stats) {
            total_stats.operations += stats.operations;
            total_stats.reads += stats.reads;
            total_stats.writes += stats.writes;
            total_stats.hits += stats.hits;
            total_stats.misses += stats.misses;
            total_stats.total_latency_ns += stats.total_latency_ns;
            total_stats.min_latency_ns = std::min(total_stats.min_latency_ns, stats.min_latency_ns);
            total_stats.max_latency_ns = std::max(total_stats.max_latency_ns, stats.max_latency_ns);
        }
        
        // Get cache statistics
        auto cache_stats = cache_->get_stats();
        
        std::cout << "\n=== Benchmark Results ===" << std::endl;
        std::cout << std::fixed << std::setprecision(2);
        
        // Throughput metrics
        double ops_per_sec = (total_stats.operations * 1000.0) / duration_ms;
        double reads_per_sec = (total_stats.reads * 1000.0) / duration_ms;
        double writes_per_sec = (total_stats.writes * 1000.0) / duration_ms;
        
        std::cout << "Throughput:" << std::endl;
        std::cout << "  Total operations: " << total_stats.operations << std::endl;
        std::cout << "  Duration: " << duration_ms << " ms" << std::endl;
        std::cout << "  Operations/sec: " << ops_per_sec << std::endl;
        std::cout << "  Reads/sec: " << reads_per_sec << std::endl;
        std::cout << "  Writes/sec: " << writes_per_sec << std::endl;
        std::cout << std::endl;
        
        // Latency metrics
        double avg_latency_ns = static_cast<double>(total_stats.total_latency_ns) / total_stats.operations;
        
        std::cout << "Latency:" << std::endl;
        std::cout << "  Average: " << avg_latency_ns << " ns (" << (avg_latency_ns / 1000.0) << " μs)" << std::endl;
        std::cout << "  Minimum: " << total_stats.min_latency_ns << " ns" << std::endl;
        std::cout << "  Maximum: " << total_stats.max_latency_ns << " ns" << std::endl;
        std::cout << std::endl;
        
        // Hit ratio
        double hit_ratio = (total_stats.reads > 0) ? 
            (static_cast<double>(total_stats.hits) / total_stats.reads * 100.0) : 0.0;
        
        std::cout << "Cache Performance:" << std::endl;
        std::cout << "  Total reads: " << total_stats.reads << std::endl;
        std::cout << "  Cache hits: " << total_stats.hits << std::endl;
        std::cout << "  Cache misses: " << total_stats.misses << std::endl;
        std::cout << "  Hit ratio: " << hit_ratio << "%" << std::endl;
        std::cout << std::endl;
        
        // Cache statistics
        std::cout << "Cache Statistics:" << std::endl;
        std::cout << "  Cache size: " << cache_stats.cache_size.load() << std::endl;
        std::cout << "  Memory usage: " << cache_stats.memory_usage_bytes.load() << " bytes" << std::endl;
        std::cout << "  Total evictions: " << cache_stats.evictions.load() << std::endl;
        
        if (config_.enable_predictive_loading) {
            std::cout << "  Predictions made: " << cache_stats.predictions_made.load() << std::endl;
            std::cout << "  Predictions hit: " << cache_stats.predictions_hit.load() << std::endl;
            std::cout << "  Warmup operations: " << cache_stats.warmup_operations.load() << std::endl;
        }
        
        std::cout << std::endl;
        
        // Performance analysis
        std::cout << "Performance Analysis:" << std::endl;
        
        if (ops_per_sec > 1000000) {
            std::cout << "  ✓ Excellent throughput (>1M ops/sec)" << std::endl;
        } else if (ops_per_sec > 500000) {
            std::cout << "  ✓ Good throughput (>500K ops/sec)" << std::endl;
        } else {
            std::cout << "  ⚠ Low throughput (<500K ops/sec)" << std::endl;
        }
        
        if (avg_latency_ns < 1000) {
            std::cout << "  ✓ Excellent latency (<1μs)" << std::endl;
        } else if (avg_latency_ns < 10000) {
            std::cout << "  ✓ Good latency (<10μs)" << std::endl;
        } else {
            std::cout << "  ⚠ High latency (>10μs)" << std::endl;
        }
        
        if (hit_ratio > 80.0) {
            std::cout << "  ✓ Excellent hit ratio (>80%)" << std::endl;
        } else if (hit_ratio > 60.0) {
            std::cout << "  ✓ Good hit ratio (>60%)" << std::endl;
        } else {
            std::cout << "  ⚠ Low hit ratio (<60%)" << std::endl;
        }
        
        std::cout << "=========================" << std::endl;
    }
};

int main(int argc, char* argv[]) {
    BenchmarkConfig config;
    
    // Parse command line arguments (simple implementation)
    for (int i = 1; i < argc; i += 2) {
        if (i + 1 >= argc) break;
        
        std::string arg = argv[i];
        std::string value = argv[i + 1];
        
        if (arg == "--capacity") {
            config.cache_capacity = std::stoull(value);
        } else if (arg == "--shards") {
            config.shard_count = std::stoull(value);
        } else if (arg == "--threads") {
            config.num_threads = std::stoull(value);
        } else if (arg == "--operations") {
            config.operations_per_thread = std::stoull(value);
        } else if (arg == "--keyspace") {
            config.key_space_size = std::stoull(value);
        } else if (arg == "--read-ratio") {
            config.read_ratio = std::stod(value);
        } else if (arg == "--value-size") {
            config.value_size = std::stoull(value);
        }
    }
    
    try {
        CacheBenchmark benchmark(config);
        benchmark.run_benchmark();
    } catch (const std::exception& e) {
        std::cerr << "Benchmark failed: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}