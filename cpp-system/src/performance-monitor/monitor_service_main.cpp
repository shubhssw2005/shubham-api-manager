#include "performance-monitor/performance_monitor.hpp"
#include "common/logger.hpp"

#include <csignal>
#include <iostream>
#include <memory>
#include <thread>

namespace {
    std::unique_ptr<ultra::monitor::PerformanceMonitor> g_monitor;
    std::atomic<bool> g_shutdown_requested{false};
}

void signal_handler(int signal) {
    LOG_INFO("Received signal {}, initiating shutdown", signal);
    g_shutdown_requested.store(true, std::memory_order_release);
}

void setup_signal_handlers() {
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);
    std::signal(SIGHUP, signal_handler);
}

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]\n"
              << "Options:\n"
              << "  --port PORT          Prometheus server port (default: 9090)\n"
              << "  --endpoint PATH      Metrics endpoint path (default: /metrics)\n"
              << "  --interval MS        Collection interval in milliseconds (default: 100)\n"
              << "  --no-hardware        Disable hardware performance counters\n"
              << "  --no-slo             Disable SLO monitoring\n"
              << "  --buckets COUNT      Number of histogram buckets (default: 64)\n"
              << "  --help               Show this help message\n";
}

int main(int argc, char* argv[]) {
    // Initialize logging
    ultra::common::Logger::initialize(ultra::common::LogLevel::INFO);
    
    LOG_INFO("Ultra Low-Latency Performance Monitor Service starting...");
    
    // Parse command line arguments
    ultra::monitor::PerformanceMonitor::Config config;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--port" && i + 1 < argc) {
            config.prometheus_port = static_cast<uint16_t>(std::stoi(argv[++i]));
        } else if (arg == "--endpoint" && i + 1 < argc) {
            config.prometheus_endpoint = argv[++i];
        } else if (arg == "--interval" && i + 1 < argc) {
            config.collection_interval = std::chrono::milliseconds(std::stoi(argv[++i]));
        } else if (arg == "--no-hardware") {
            config.enable_hardware_counters = false;
        } else if (arg == "--no-slo") {
            config.enable_slo_monitoring = false;
        } else if (arg == "--buckets" && i + 1 < argc) {
            config.histogram_buckets = std::stoul(argv[++i]);
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }
    
    // Setup signal handlers for graceful shutdown
    setup_signal_handlers();
    
    try {
        // Create and configure performance monitor
        g_monitor = std::make_unique<ultra::monitor::PerformanceMonitor>(config);
        
        // Register some example SLOs
        ultra::monitor::PerformanceMonitor::SLOConfig api_latency_slo;
        api_latency_slo.name = "api_request_latency";
        api_latency_slo.target_percentile = 0.99;
        api_latency_slo.target_latency_ns = 1000000;  // 1ms
        api_latency_slo.evaluation_window = std::chrono::seconds(60);
        
        g_monitor->register_slo(api_latency_slo);
        
        ultra::monitor::PerformanceMonitor::SLOConfig cache_latency_slo;
        cache_latency_slo.name = "cache_access_latency";
        cache_latency_slo.target_percentile = 0.999;
        cache_latency_slo.target_latency_ns = 100000;  // 100μs
        cache_latency_slo.evaluation_window = std::chrono::seconds(30);
        
        g_monitor->register_slo(cache_latency_slo);
        
        // Start monitoring services
        g_monitor->start_collection();
        g_monitor->start_prometheus_server();
        
        LOG_INFO("Performance monitor started successfully");
        LOG_INFO("Prometheus metrics available at http://localhost:{}{}", 
                 config.prometheus_port, config.prometheus_endpoint);
        LOG_INFO("Hardware counters: {}", config.enable_hardware_counters ? "enabled" : "disabled");
        LOG_INFO("SLO monitoring: {}", config.enable_slo_monitoring ? "enabled" : "disabled");
        
        // Simulate some metrics for demonstration
        std::thread metrics_simulator([&]() {
            uint64_t counter = 0;
            while (!g_shutdown_requested.load(std::memory_order_acquire)) {
                // Simulate API requests
                g_monitor->increment_counter("api_requests_total");
                g_monitor->increment_counter("cache_hits_total", counter % 10 == 0 ? 0 : 1);
                g_monitor->increment_counter("cache_misses_total", counter % 10 == 0 ? 1 : 0);
                
                // Simulate latency measurements
                double api_latency = 0.0005 + (counter % 100) * 0.00001;  // 0.5-1.5ms
                double cache_latency = 0.00005 + (counter % 50) * 0.000001;  // 50-100μs
                
                g_monitor->observe_histogram("api_request_duration_seconds", api_latency);
                g_monitor->observe_histogram("cache_access_duration_seconds", cache_latency);
                
                // Simulate system metrics
                g_monitor->set_gauge("memory_usage_bytes", 1024 * 1024 * (100 + counter % 50));
                g_monitor->set_gauge("cpu_usage_percent", 10.0 + (counter % 20));
                g_monitor->set_gauge("active_connections", 100 + (counter % 200));
                
                counter++;
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        });
        
        // Main service loop
        while (!g_shutdown_requested.load(std::memory_order_acquire)) {
            // Check SLO violations periodically
            g_monitor->check_slo_violations();
            
            // Print performance statistics
            auto stats = g_monitor->get_performance_stats();
            if (stats.total_metrics_collected > 0) {
                LOG_INFO("Performance stats - Metrics/sec: {:.1f}, Collection overhead: {}ns, Export overhead: {}ns",
                         stats.metrics_per_second, stats.collection_overhead_ns, stats.export_overhead_ns);
            }
            
            // Print hardware metrics if available
            auto hw_metrics = g_monitor->get_hardware_metrics();
            if (hw_metrics.cpu_cycles > 0) {
                LOG_INFO("Hardware metrics - IPC: {:.2f}, Cache hit rate: {:.1f}%, Branch prediction: {:.1f}%",
                         hw_metrics.ipc, hw_metrics.cache_hit_rate * 100, 
                         (1.0 - static_cast<double>(hw_metrics.branch_mispredictions) / hw_metrics.instructions) * 100);
            }
            
            std::this_thread::sleep_for(std::chrono::seconds(10));
        }
        
        // Stop metrics simulator
        if (metrics_simulator.joinable()) {
            metrics_simulator.join();
        }
        
        // Graceful shutdown
        LOG_INFO("Shutting down performance monitor...");
        g_monitor->stop_prometheus_server();
        g_monitor->stop_collection();
        g_monitor.reset();
        
        LOG_INFO("Performance monitor service stopped successfully");
        
    } catch (const std::exception& e) {
        LOG_ERROR("Fatal error: {}", e.what());
        return 1;
    }
    
    return 0;
}