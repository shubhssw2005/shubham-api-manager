#pragma once

#include "common/types.hpp"
#include <string>
#include <functional>
#include <thread>
#include <vector>
#include <memory>

namespace ultra::api {

struct HttpRequest {
    std::string method;
    std::string path;
    std::string query_string;
    std::vector<std::pair<std::string, std::string>> headers;
    std::string body;
    timestamp_t received_at;
};

struct HttpResponse {
    u16 status_code = 200;
    std::vector<std::pair<std::string, std::string>> headers;
    std::string body;
    timestamp_t sent_at;
};

using FastHandler = std::function<void(const HttpRequest&, HttpResponse&)>;

class FastAPIGateway {
public:
    struct Config {
        u16 port = 8080;
        size_t worker_threads = std::thread::hardware_concurrency();
        size_t memory_pool_size = 1024 * 1024 * 1024; // 1GB
        std::string fallback_upstream = "http://localhost:3005";
        bool enable_dpdk = true;
        u32 dpdk_port_mask = 0x1;
    };
    
    explicit FastAPIGateway(const Config& config);
    ~FastAPIGateway();
    
    void start();
    void stop();
    void register_fast_route(const std::string& path, FastHandler handler);
    
    struct Stats {
        aligned_atomic<u64> requests_processed{0};
        aligned_atomic<u64> total_latency_ns{0};
        aligned_atomic<u64> cache_hits{0};
        aligned_atomic<u64> fallback_requests{0};
    };
    
    Stats get_stats() const noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> pimpl_;
};

} // namespace ultra::api