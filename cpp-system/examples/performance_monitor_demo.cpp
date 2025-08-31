#include "performance-monitor/performance_monitor.hpp"
#include "common/logger.hpp"

#include <iostream>
#include <thread>
#include <random>
#include <chrono>

using namespace ultra::monitor;

// Simulate a high-performance API service
class HighPerformanceAPI {
public:
    explicit HighPerformanceAPI(PerformanceMonitor& monitor) : monitor_(monitor) {
        // Register SLOs for our API
        PerformanceMonitor::SLOConfig api_slo;
        api_slo.name = "api_request_latency";
        api_slo.target_percentile = 0.99;
        api_slo.target_latency_ns = 1000000;  // 1ms P99
        api_slo.evaluation_window = std::chrono::seconds(60);
        
        monitor_.register_slo(api_slo);
        
        PerformanceMonitor::SLOConfig cache_slo;
        cache_slo.name = "cache_access_latency";
        cache_slo.target_percentile = 0.999;
        cache_slo.target_latency_ns = 100000;  // 100μs P99.9
        cache_slo.evaluation_window = std::chrono::seconds(30);
        
        monitor_.register_slo(cache_slo);
        
        LOG_INFO("HighPerformanceAPI initialized with SLO monitoring");
    }
    
    void handle_request(const std::string& request_type) {
        ULTRA_TIMER(monitor_, "api_request_duration_seconds");
        
        // Increment request counter
        monitor_.increment_counter("api_requests_total");
        monitor_.increment_counter("api_requests_by_type", 1);
        
        // Simulate request processing
        auto processing_time = simulate_processing(request_type);
        
        // Record latency
        monitor_.record_timing("api_request_latency", processing_time);
        
        // Update system metrics
        update_system_metrics();
        
        LOG_DEBUG("Processed {} request in {}μs", request_type, processing_time / 1000);
    }
    
    void access_cache(const std::string& key) {
        ULTRA_TIMER(monitor_, "cache_access_duration_seconds");
        
        // Simulate cache access
        auto access_time = simulate_cache_access();
        
        // Record metrics
        if (access_time < 50000) {  // Cache hit if < 50μs
            monitor_.increment_counter("cache_hits_total");
        } else {
            monitor_.increment_counter("cache_misses_total");
        }
        
        monitor_.record_timing("cache_access_latency", access_time);
        
        LOG_DEBUG("Cache access for key {} took {}ns", key, access_time);
    }
    
private:
    PerformanceMonitor& monitor_;
    std::random_device rd_;
    std::mt19937 gen_{rd_()};
    
    uint64_t simulate_processing(const std::string& request_type) {
        // Simulate different processing times based on request type
        std::uniform_int_distribution<> dist;
        
        if (request_type == "fast") {
            dist = std::uniform_int_distribution<>(200000, 800000);  // 200-800μs
        } else if (request_type == "medium") {
            dist = std::uniform_int_distribution<>(500000, 1500000); // 500μs-1.5ms
        } else {  // slow
            dist = std::uniform_int_distribution<>(1000000, 5000000); // 1-5ms
        }
        
        auto processing_time = dist(gen_);
        
        // Actually sleep to simulate work
        std::this_thread::sleep_for(std::chrono::nanoseconds(processing_time / 10));
        
        return processing_time;
    }
    
    uint64_t simulate_cache_access() {
        std::uniform_int_distribution<> dist(10000, 200000);  // 10-200μs
        auto access_time = dist(gen_);
        
        // Simulate occasional cache misses (slower access)
        if (dist(gen_) % 10 == 0) {
            access_time *= 5;  // Cache miss penalty
        }
        
        return access_time;
    }
    
    void update_system_metrics() {
        // Simulate system resource usage
        static uint64_t counter = 0;
        counter++;
        
        // Memory usage (simulate gradual increase with periodic GC)
        double memory_usage = 1024 * 1024 * (100 + (counter % 200));
        if (counter % 500 == 0) {
            memory_usage *= 0.7;  // Simulate garbage collection
        }
        monitor_.set_gauge("memory_usage_bytes", memory_usage);
        
        // CPU usage (simulate varying load)
        double cpu_usage = 20.0 + 30.0 * std::sin(counter * 0.01);
        monitor_.set_gauge("cpu_usage_percent", cpu_usage);
        
        // Active connections
        monitor_.set_gauge("active_connections", 100 + (counter % 300));
        
        // Queue depth
        monitor_.set_gauge("request_queue_depth", counter % 50);
    }
};

int main() {
    // Initialize logging
    ultra::common::Logger::initialize(ultra::common::LogLevel::INFO);
    
    LOG_INFO("Performance Monitor Demo starting...");
    
    try {
        // Create performance monitor with custom configuration
        PerformanceMonitor::Config config;
        config.collection_interval = std::chrono::milliseconds(50);  // High frequency collection
        config.enable_hardware_counters = true;
        config.enable_slo_monitoring = true;
        config.prometheus_port = 9090;
        config.histogram_buckets = 64;
        
        PerformanceMonitor monitor(config);
        
        // Start monitoring
        monitor.start_collection();
        monitor.start_prometheus_server();
        
        LOG_INFO("Performance monitor started on port {}", config.prometheus_port);
        LOG_INFO("Visit http://localhost:9090/metrics to see Prometheus metrics");
        
        // Create our high-performance API
        HighPerformanceAPI api(monitor);
        
        // Simulate load with multiple worker threads
        std::vector<std::thread> workers;
        std::atomic<bool> running{true};
        
        // Request processing workers
        for (int i = 0; i < 4; ++i) {
            workers.emplace_back([&api, &running, i]() {
                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_int_distribution<> type_dist(0, 2);
                std::uniform_int_distribution<> delay_dist(1, 10);
                
                std::vector<std::string> request_types = {"fast", "medium", "slow"};
                
                while (running.load()) {
                    std::string request_type = request_types[type_dist(gen)];
                    api.handle_request(request_type);
                    
                    // Random delay between requests
                    std::this_thread::sleep_for(std::chrono::milliseconds(delay_dist(gen)));
                }
                
                LOG_INFO("Worker {} stopped", i);
            });
        }
        
        // Cache access worker
        workers.emplace_back([&api, &running]() {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> key_dist(1, 1000);
            
            while (running.load()) {
                std::string key = "key_" + std::to_string(key_dist(gen));
                api.access_cache(key);
                
                std::this_thread::sleep_for(std::chrono::microseconds(500));
            }
            
            LOG_INFO("Cache worker stopped");
        });
        
        // Run demo for 60 seconds
        LOG_INFO("Running performance demo for 60 seconds...");
        LOG_INFO("Generating realistic load patterns with SLO monitoring");
        
        std::this_thread::sleep_for(std::chrono::seconds(60));
        
        // Stop workers
        running.store(false);
        for (auto& worker : workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }
        
        // Print final statistics
        auto stats = monitor.get_performance_stats();
        LOG_INFO("Final performance statistics:");
        LOG_INFO("  Total metrics collected: {}", stats.total_metrics_collected);
        LOG_INFO("  Metrics per second: {:.1f}", stats.metrics_per_second);
        LOG_INFO("  Collection overhead: {}ns", stats.collection_overhead_ns);
        LOG_INFO("  Export overhead: {}ns", stats.export_overhead_ns);
        
        // Print hardware metrics if available
        auto hw_metrics = monitor.get_hardware_metrics();
        if (hw_metrics.cpu_cycles > 0) {
            LOG_INFO("Hardware performance metrics:");
            LOG_INFO("  Instructions per cycle: {:.2f}", hw_metrics.ipc);
            LOG_INFO("  Cache hit rate: {:.1f}%", hw_metrics.cache_hit_rate * 100);
            LOG_INFO("  CPU cycles: {}", hw_metrics.cpu_cycles);
            LOG_INFO("  Instructions: {}", hw_metrics.instructions);
            LOG_INFO("  Cache misses: {}", hw_metrics.cache_misses);
        }
        
        // Check for SLO violations
        monitor.check_slo_violations();
        
        LOG_INFO("Performance demo completed successfully");
        
    } catch (const std::exception& e) {
        LOG_ERROR("Demo failed: {}", e.what());
        return 1;
    }
    
    return 0;
}