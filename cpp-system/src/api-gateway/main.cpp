#include "api-gateway/fast_api_gateway.hpp"
#include "common/types.hpp"
#include <iostream>
#include <signal.h>
#include <memory>
#include <chrono>
#include <thread>

using namespace ultra::api;

// Global gateway instance for signal handling
std::unique_ptr<FastAPIGateway> g_gateway;

void signal_handler(int signal) {
    std::cout << "Received signal " << signal << ", shutting down gracefully..." << std::endl;
    if (g_gateway) {
        g_gateway->stop();
    }
    exit(0);
}

void setup_signal_handlers() {
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
}

void register_sample_routes(FastAPIGateway& gateway) {
    // Health check endpoint
    gateway.register_fast_route("/health", [](const HttpRequest& req, HttpResponse& resp) {
        resp.status_code = 200;
        resp.headers.emplace_back("Content-Type", "application/json");
        resp.body = R"({"status":"healthy","timestamp":)" + 
                   std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(
                       std::chrono::system_clock::now().time_since_epoch()).count()) + "}";
    });
    
    // Fast cache lookup endpoint
    gateway.register_fast_route("/api/v1/cache/", [](const HttpRequest& req, HttpResponse& resp) {
        // Extract key from path
        std::string key = req.path.substr(15); // Remove "/api/v1/cache/"
        
        // Simulate ultra-fast cache lookup
        resp.status_code = 200;
        resp.headers.emplace_back("Content-Type", "application/json");
        resp.headers.emplace_back("X-Cache", "HIT");
        resp.body = R"({"key":")" + key + R"(","value":"cached_data","latency_ns":)" + 
                   std::to_string(std::chrono::high_resolution_clock::now().time_since_epoch().count()) + "}";
    });
    
    // Fast metrics endpoint
    gateway.register_fast_route("/metrics", [&gateway](const HttpRequest& req, HttpResponse& resp) {
        auto stats = gateway.get_stats();
        
        resp.status_code = 200;
        resp.headers.emplace_back("Content-Type", "text/plain");
        
        std::string metrics = 
            "# HELP ultra_requests_total Total number of requests processed\n"
            "# TYPE ultra_requests_total counter\n"
            "ultra_requests_total " + std::to_string(stats.requests_processed.load()) + "\n"
            
            "# HELP ultra_cache_hits_total Total number of cache hits\n"
            "# TYPE ultra_cache_hits_total counter\n"
            "ultra_cache_hits_total " + std::to_string(stats.cache_hits.load()) + "\n"
            
            "# HELP ultra_fallback_requests_total Total number of fallback requests\n"
            "# TYPE ultra_fallback_requests_total counter\n"
            "ultra_fallback_requests_total " + std::to_string(stats.fallback_requests.load()) + "\n"
            
            "# HELP ultra_avg_latency_ns Average request latency in nanoseconds\n"
            "# TYPE ultra_avg_latency_ns gauge\n";
        
        u64 total_requests = stats.requests_processed.load();
        if (total_requests > 0) {
            u64 avg_latency = stats.total_latency_ns.load() / total_requests;
            metrics += "ultra_avg_latency_ns " + std::to_string(avg_latency) + "\n";
        } else {
            metrics += "ultra_avg_latency_ns 0\n";
        }
        
        resp.body = metrics;
    });
    
    // Fast API status endpoint
    gateway.register_fast_route("/api/status", [](const HttpRequest& req, HttpResponse& resp) {
        resp.status_code = 200;
        resp.headers.emplace_back("Content-Type", "application/json");
        resp.body = R"({
            "service": "ultra-api-gateway",
            "version": "1.0.0",
            "uptime_ns": )" + std::to_string(std::chrono::high_resolution_clock::now().time_since_epoch().count()) + R"(,
            "features": ["dpdk", "simd", "lock_free", "zero_copy"]
        })";
    });
}

void print_startup_info(const FastAPIGateway::Config& config) {
    std::cout << "Ultra-Fast API Gateway Starting..." << std::endl;
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Port: " << config.port << std::endl;
    std::cout << "  Worker Threads: " << config.worker_threads << std::endl;
    std::cout << "  Memory Pool Size: " << (config.memory_pool_size / (1024 * 1024)) << " MB" << std::endl;
    std::cout << "  DPDK Enabled: " << (config.enable_dpdk ? "Yes" : "No") << std::endl;
    std::cout << "  DPDK Port Mask: 0x" << std::hex << config.dpdk_port_mask << std::dec << std::endl;
    std::cout << "  Fallback Upstream: " << config.fallback_upstream << std::endl;
    std::cout << std::endl;
}

void print_performance_stats(const FastAPIGateway& gateway) {
    while (true) {
        std::this_thread::sleep_for(std::chrono::seconds(10));
        
        auto stats = gateway.get_stats();
        u64 total_requests = stats.requests_processed.load();
        u64 cache_hits = stats.cache_hits.load();
        u64 fallback_requests = stats.fallback_requests.load();
        u64 total_latency = stats.total_latency_ns.load();
        
        std::cout << "Performance Stats:" << std::endl;
        std::cout << "  Total Requests: " << total_requests << std::endl;
        std::cout << "  Cache Hits: " << cache_hits << std::endl;
        std::cout << "  Fallback Requests: " << fallback_requests << std::endl;
        
        if (total_requests > 0) {
            u64 avg_latency_ns = total_latency / total_requests;
            double avg_latency_us = avg_latency_ns / 1000.0;
            std::cout << "  Average Latency: " << avg_latency_us << " μs" << std::endl;
            
            double cache_hit_rate = (double)cache_hits / total_requests * 100.0;
            std::cout << "  Cache Hit Rate: " << cache_hit_rate << "%" << std::endl;
        }
        
        std::cout << std::endl;
    }
}

int main(int argc, char* argv[]) {
    setup_signal_handlers();
    
    // Configure the API gateway
    FastAPIGateway::Config config;
    config.port = 8080;
    config.worker_threads = std::thread::hardware_concurrency();
    config.memory_pool_size = 1024 * 1024 * 1024; // 1GB
    config.fallback_upstream = "http://localhost:3005";
    config.enable_dpdk = true;
    config.dpdk_port_mask = 0x1;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--port" && i + 1 < argc) {
            config.port = static_cast<u16>(std::stoi(argv[++i]));
        } else if (arg == "--workers" && i + 1 < argc) {
            config.worker_threads = static_cast<size_t>(std::stoi(argv[++i]));
        } else if (arg == "--no-dpdk") {
            config.enable_dpdk = false;
        } else if (arg == "--fallback" && i + 1 < argc) {
            config.fallback_upstream = argv[++i];
        } else if (arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  --port <port>        Listen port (default: 8080)" << std::endl;
            std::cout << "  --workers <count>    Number of worker threads (default: CPU cores)" << std::endl;
            std::cout << "  --no-dpdk           Disable DPDK (use regular sockets)" << std::endl;
            std::cout << "  --fallback <url>     Fallback upstream URL (default: http://localhost:3005)" << std::endl;
            std::cout << "  --help              Show this help message" << std::endl;
            return 0;
        }
    }
    
    print_startup_info(config);
    
    try {
        // Create and configure the gateway
        g_gateway = std::make_unique<FastAPIGateway>(config);
        
        // Register sample routes
        register_sample_routes(*g_gateway);
        
        // Start the gateway
        std::cout << "Starting Ultra-Fast API Gateway..." << std::endl;
        g_gateway->start();
        
        std::cout << "Gateway started successfully!" << std::endl;
        std::cout << "Listening on port " << config.port << std::endl;
        std::cout << "Press Ctrl+C to stop" << std::endl;
        
        // Start performance monitoring thread
        std::thread stats_thread([&]() {
            print_performance_stats(*g_gateway);
        });
        stats_thread.detach();
        
        // Keep the main thread alive
        while (true) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}