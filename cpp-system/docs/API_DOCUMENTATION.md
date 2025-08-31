# Ultra Low-Latency C++ System API Documentation

## Overview

This document provides comprehensive API documentation for the Ultra Low-Latency C++ System components. The system is designed to achieve sub-millisecond response times for critical operations while maintaining seamless integration with existing Node.js infrastructure.

## Table of Contents

1. [Fast API Gateway](#fast-api-gateway)
2. [Ultra Cache](#ultra-cache)
3. [Stream Processor](#stream-processor)
4. [GPU Compute Engine](#gpu-compute-engine)
5. [Performance Monitor](#performance-monitor)
6. [Error Handling](#error-handling)
7. [Configuration Management](#configuration-management)

## Fast API Gateway

The Fast API Gateway provides ultra-low latency HTTP request processing using DPDK for kernel bypass networking.

### Class: FastAPIGateway

```cpp
#include "api-gateway/fast_api_gateway.hpp"

class FastAPIGateway {
public:
    struct Config {
        uint16_t port = 8080;
        size_t worker_threads = std::thread::hardware_concurrency();
        size_t memory_pool_size = 1024 * 1024 * 1024; // 1GB
        std::string fallback_upstream = "http://localhost:3005";
        bool enable_dpdk = true;
        std::string dpdk_args = "-l 0-3 -n 4";
    };
    
    explicit FastAPIGateway(const Config& config);
    ~FastAPIGateway();
    
    // Core operations
    void start();
    void stop();
    void register_fast_route(const std::string& path, FastHandler handler);
    void register_fallback_route(const std::string& path);
    
    // Statistics
    struct Stats {
        std::atomic<uint64_t> requests_processed{0};
        std::atomic<uint64_t> fast_path_hits{0};
        std::atomic<uint64_t> fallback_hits{0};
        std::atomic<uint64_t> avg_latency_ns{0};
    };
    
    Stats get_stats() const noexcept;
};
```

### Usage Example

```cpp
#include "api-gateway/fast_api_gateway.hpp"
#include "cache/ultra_cache.hpp"

int main() {
    // Configure the gateway
    FastAPIGateway::Config config;
    config.port = 8080;
    config.worker_threads = 4;
    config.memory_pool_size = 2ULL * 1024 * 1024 * 1024; // 2GB
    
    FastAPIGateway gateway(config);
    
    // Set up ultra cache for fast responses
    UltraCache<std::string, std::string> cache({
        .capacity = 1000000,
        .shard_count = 64
    });
    
    // Register fast route for cached data
    gateway.register_fast_route("/api/posts/{id}", 
        [&cache](const HttpRequest& req, HttpResponse& resp) {
            auto post_id = req.path_params.at("id");
            
            if (auto cached_post = cache.get(post_id)) {
                resp.set_body(*cached_post);
                resp.set_status(200);
                resp.set_header("Content-Type", "application/json");
                resp.set_header("X-Cache", "HIT");
            } else {
                // Cache miss - let Node.js handle it
                resp.set_status(404);
                resp.set_header("X-Fallback", "nodejs");
            }
        });
    
    // Register fallback for complex operations
    gateway.register_fallback_route("/api/posts");
    gateway.register_fallback_route("/api/auth/*");
    
    // Start the gateway
    gateway.start();
    
    // Monitor performance
    while (true) {
        std::this_thread::sleep_for(std::chrono::seconds(10));
        auto stats = gateway.get_stats();
        std::cout << "Requests: " << stats.requests_processed 
                  << ", Fast path: " << stats.fast_path_hits
                  << ", Avg latency: " << stats.avg_latency_ns << "ns\n";
    }
    
    return 0;
}
```

## Ultra Cache

Lock-free, high-performance in-memory cache with RCU semantics for nanosecond access times.

### Class: UltraCache

```cpp
#include "cache/ultra_cache.hpp"

template<typename Key, typename Value>
class UltraCache {
public:
    struct Config {
        size_t capacity = 1000000;
        size_t shard_count = 64;
        std::string backing_file;
        bool enable_rdma = false;
        std::chrono::seconds default_ttl{3600};
        double load_factor = 0.75;
    };
    
    explicit UltraCache(const Config& config);
    ~UltraCache();
    
    // Core operations
    std::optional<Value> get(const Key& key) noexcept;
    void put(const Key& key, const Value& value) noexcept;
    void put(const Key& key, const Value& value, std::chrono::seconds ttl) noexcept;
    void remove(const Key& key) noexcept;
    void clear() noexcept;
    
    // Batch operations
    std::vector<std::optional<Value>> get_batch(const std::vector<Key>& keys) noexcept;
    void put_batch(const std::vector<std::pair<Key, Value>>& items) noexcept;
    
    // Statistics and monitoring
    struct Stats {
        std::atomic<uint64_t> hits{0};
        std::atomic<uint64_t> misses{0};
        std::atomic<uint64_t> evictions{0};
        std::atomic<uint64_t> size{0};
        std::atomic<uint64_t> memory_usage{0};
    };
    
    Stats get_stats() const noexcept;
    double hit_ratio() const noexcept;
    void reset_stats() noexcept;
};
```

### Usage Example

```cpp
#include "cache/ultra_cache.hpp"
#include <string>
#include <chrono>

int main() {
    // Configure cache for blog posts
    UltraCache<std::string, std::string>::Config config;
    config.capacity = 100000;
    config.shard_count = 32;
    config.default_ttl = std::chrono::minutes(30);
    config.backing_file = "/tmp/blog_cache.mmap";
    
    UltraCache<std::string, std::string> post_cache(config);
    
    // Cache some blog posts
    std::string post_json = R"({
        "id": "post123",
        "title": "Ultra Fast Blogging",
        "content": "This post is served from ultra-fast cache...",
        "author": "speed_demon",
        "created_at": "2025-01-27T10:00:00Z"
    })";
    
    // Store with default TTL
    post_cache.put("post123", post_json);
    
    // Store with custom TTL (5 minutes for trending posts)
    post_cache.put("trending_post456", post_json, std::chrono::minutes(5));
    
    // Retrieve posts
    if (auto cached_post = post_cache.get("post123")) {
        std::cout << "Cache hit! Post: " << *cached_post << std::endl;
    } else {
        std::cout << "Cache miss - need to fetch from database" << std::endl;
    }
    
    // Batch operations for efficiency
    std::vector<std::string> post_ids = {"post123", "post456", "post789"};
    auto cached_posts = post_cache.get_batch(post_ids);
    
    for (size_t i = 0; i < post_ids.size(); ++i) {
        if (cached_posts[i]) {
            std::cout << "Found cached post: " << post_ids[i] << std::endl;
        }
    }
    
    // Monitor cache performance
    auto stats = post_cache.get_stats();
    std::cout << "Cache stats:\n"
              << "  Hits: " << stats.hits << "\n"
              << "  Misses: " << stats.misses << "\n"
              << "  Hit ratio: " << post_cache.hit_ratio() << "\n"
              << "  Memory usage: " << stats.memory_usage << " bytes\n";
    
    return 0;
}
```

## Stream Processor

Real-time event processing engine with microsecond latency and SIMD acceleration.

### Class: StreamProcessor

```cpp
#include "stream-processor/stream_processor.hpp"

class StreamProcessor {
public:
    struct Event {
        uint64_t timestamp_ns;
        uint32_t type;
        uint32_t tenant_id;
        uint32_t size;
        alignas(64) char data[];
        
        template<typename T>
        const T* get_data() const {
            return reinterpret_cast<const T*>(data);
        }
    };
    
    using EventHandler = std::function<void(const Event&)>;
    
    struct Config {
        size_t ring_buffer_size = 1024 * 1024;
        size_t worker_threads = std::thread::hardware_concurrency();
        bool enable_simd = true;
        std::chrono::microseconds batch_timeout{100};
    };
    
    explicit StreamProcessor(const Config& config);
    ~StreamProcessor();
    
    // Event handling
    void subscribe(uint32_t event_type, EventHandler handler);
    void unsubscribe(uint32_t event_type);
    void publish(const Event& event);
    void publish_batch(const std::vector<Event>& events);
    
    // Lifecycle
    void start_processing();
    void stop_processing();
    
    // Windowed operations
    void create_sliding_window(const std::string& name, 
                              std::chrono::milliseconds window_size,
                              std::chrono::milliseconds slide_interval);
    void create_tumbling_window(const std::string& name,
                               std::chrono::milliseconds window_size);
    
    // Metrics
    struct Metrics {
        std::atomic<uint64_t> events_processed{0};
        std::atomic<uint64_t> processing_latency_ns{0};
        std::atomic<uint64_t> queue_depth{0};
        std::atomic<uint64_t> dropped_events{0};
    };
    
    Metrics get_metrics() const;
};
```

### Usage Example

```cpp
#include "stream-processor/stream_processor.hpp"
#include <iostream>

// Event types
enum EventType : uint32_t {
    USER_LOGIN = 1,
    POST_VIEW = 2,
    POST_LIKE = 3,
    COMMENT_CREATED = 4
};

struct UserLoginEvent {
    uint32_t user_id;
    char ip_address[16];
    uint64_t session_id;
};

struct PostViewEvent {
    uint32_t user_id;
    uint32_t post_id;
    uint32_t view_duration_ms;
};

int main() {
    StreamProcessor::Config config;
    config.ring_buffer_size = 2 * 1024 * 1024; // 2M events
    config.worker_threads = 8;
    config.enable_simd = true;
    
    StreamProcessor processor(config);
    
    // Set up event handlers
    processor.subscribe(USER_LOGIN, [](const StreamProcessor::Event& event) {
        auto login_data = event.get_data<UserLoginEvent>();
        std::cout << "User " << login_data->user_id 
                  << " logged in from " << login_data->ip_address << std::endl;
        
        // Real-time fraud detection
        // ... implement fraud detection logic
    });
    
    processor.subscribe(POST_VIEW, [](const StreamProcessor::Event& event) {
        auto view_data = event.get_data<PostViewEvent>();
        std::cout << "Post " << view_data->post_id 
                  << " viewed by user " << view_data->user_id << std::endl;
        
        // Update real-time analytics
        // ... implement analytics logic
    });
    
    // Create sliding window for real-time metrics
    processor.create_sliding_window("user_activity", 
                                   std::chrono::minutes(5),
                                   std::chrono::seconds(10));
    
    // Start processing
    processor.start_processing();
    
    // Simulate events
    for (int i = 0; i < 1000; ++i) {
        // Create login event
        StreamProcessor::Event login_event;
        login_event.timestamp_ns = std::chrono::high_resolution_clock::now()
                                  .time_since_epoch().count();
        login_event.type = USER_LOGIN;
        login_event.tenant_id = 1;
        login_event.size = sizeof(UserLoginEvent);
        
        auto login_data = reinterpret_cast<UserLoginEvent*>(login_event.data);
        login_data->user_id = i + 1;
        strcpy(login_data->ip_address, "192.168.1.100");
        login_data->session_id = i + 1000;
        
        processor.publish(login_event);
        
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    
    // Monitor performance
    auto metrics = processor.get_metrics();
    std::cout << "Stream processor metrics:\n"
              << "  Events processed: " << metrics.events_processed << "\n"
              << "  Avg latency: " << metrics.processing_latency_ns << "ns\n"
              << "  Queue depth: " << metrics.queue_depth << "\n";
    
    processor.stop_processing();
    return 0;
}
```## GP
U Compute Engine

Hardware-accelerated computing for ML inference, image processing, and cryptographic operations.

### Class: GPUComputeEngine

```cpp
#include "gpu-compute/gpu_compute_engine.hpp"

class GPUComputeEngine {
public:
    struct Config {
        int device_id = 0;
        size_t memory_pool_size = 512 * 1024 * 1024; // 512MB
        bool enable_tensorrt = true;
        bool enable_mixed_precision = true;
        std::string model_cache_dir = "/tmp/gpu_models";
    };
    
    explicit GPUComputeEngine(const Config& config);
    ~GPUComputeEngine();
    
    // ML Inference
    std::vector<float> infer(const std::string& model_name, 
                           const std::vector<float>& input);
    std::future<std::vector<float>> infer_async(const std::string& model_name,
                                               const std::vector<float>& input);
    
    // Batch inference for better throughput
    std::vector<std::vector<float>> infer_batch(const std::string& model_name,
                                               const std::vector<std::vector<float>>& inputs);
    
    // Image Processing
    struct ImageData {
        uint32_t width;
        uint32_t height;
        uint32_t channels;
        std::vector<uint8_t> data;
    };
    
    void resize_image_batch(const std::vector<ImageData>& inputs,
                          std::vector<ImageData>& outputs,
                          int target_width, int target_height);
    
    void apply_filter_batch(const std::vector<ImageData>& inputs,
                           std::vector<ImageData>& outputs,
                           const std::string& filter_type);
    
    // Cryptographic Operations
    std::vector<uint8_t> compute_hash_batch(const std::vector<std::vector<uint8_t>>& data,
                                           const std::string& algorithm = "sha256");
    
    // Model Management
    void load_model(const std::string& name, const std::string& path);
    void unload_model(const std::string& name);
    std::vector<std::string> list_loaded_models() const;
    
    // Statistics
    struct Stats {
        std::atomic<uint64_t> operations_completed{0};
        std::atomic<uint64_t> gpu_utilization_percent{0};
        std::atomic<uint64_t> memory_used_bytes{0};
        std::atomic<uint64_t> inference_count{0};
        std::atomic<uint64_t> avg_inference_time_us{0};
    };
    
    Stats get_stats() const;
};
```

### Usage Example

```cpp
#include "gpu-compute/gpu_compute_engine.hpp"
#include <opencv2/opencv.hpp>

int main() {
    GPUComputeEngine::Config config;
    config.device_id = 0;
    config.memory_pool_size = 1024 * 1024 * 1024; // 1GB
    config.enable_tensorrt = true;
    
    GPUComputeEngine gpu_engine(config);
    
    // Load ML models for content analysis
    gpu_engine.load_model("content_classifier", "/models/content_classifier.onnx");
    gpu_engine.load_model("image_embedder", "/models/image_embedder.trt");
    
    // Example: Process uploaded images
    std::vector<GPUComputeEngine::ImageData> uploaded_images;
    
    // Load images (example with OpenCV)
    cv::Mat img = cv::imread("uploaded_image.jpg");
    GPUComputeEngine::ImageData image_data;
    image_data.width = img.cols;
    image_data.height = img.rows;
    image_data.channels = img.channels();
    image_data.data.assign(img.data, img.data + img.total() * img.elemSize());
    uploaded_images.push_back(image_data);
    
    // Resize images for ML processing
    std::vector<GPUComputeEngine::ImageData> resized_images;
    gpu_engine.resize_image_batch(uploaded_images, resized_images, 224, 224);
    
    // Convert to ML input format
    std::vector<float> ml_input;
    for (const auto& pixel : resized_images[0].data) {
        ml_input.push_back(static_cast<float>(pixel) / 255.0f);
    }
    
    // Run content classification
    auto classification_result = gpu_engine.infer("content_classifier", ml_input);
    
    std::cout << "Content classification scores:\n";
    std::vector<std::string> classes = {"safe", "adult", "violence", "spam"};
    for (size_t i = 0; i < classification_result.size() && i < classes.size(); ++i) {
        std::cout << "  " << classes[i] << ": " << classification_result[i] << std::endl;
    }
    
    // Generate image embeddings for similarity search
    auto embeddings = gpu_engine.infer("image_embedder", ml_input);
    std::cout << "Generated " << embeddings.size() << "-dimensional embedding\n";
    
    // Batch hash computation for integrity verification
    std::vector<std::vector<uint8_t>> file_data;
    file_data.push_back(resized_images[0].data);
    
    auto hashes = gpu_engine.compute_hash_batch(file_data, "sha256");
    std::cout << "File hash: ";
    for (auto byte : hashes) {
        std::cout << std::hex << static_cast<int>(byte);
    }
    std::cout << std::endl;
    
    // Monitor GPU performance
    auto stats = gpu_engine.get_stats();
    std::cout << "GPU Engine Stats:\n"
              << "  Operations: " << stats.operations_completed << "\n"
              << "  GPU Utilization: " << stats.gpu_utilization_percent << "%\n"
              << "  Memory Used: " << stats.memory_used_bytes / (1024*1024) << "MB\n"
              << "  Avg Inference Time: " << stats.avg_inference_time_us << "μs\n";
    
    return 0;
}
```

## Performance Monitor

Real-time performance monitoring with hardware-level insights and nanosecond precision.

### Class: PerformanceMonitor

```cpp
#include "performance-monitor/performance_monitor.hpp"

class PerformanceMonitor {
public:
    struct Config {
        std::chrono::milliseconds collection_interval{100};
        bool enable_hardware_counters = true;
        std::string prometheus_endpoint = "/metrics";
        uint16_t metrics_port = 9090;
        bool enable_flame_graphs = false;
    };
    
    explicit PerformanceMonitor(const Config& config);
    ~PerformanceMonitor();
    
    // Lifecycle
    void start_collection();
    void stop_collection();
    
    // Timing measurements
    class Timer {
    public:
        explicit Timer(const std::string& name);
        ~Timer();
        
        uint64_t elapsed_ns() const;
        void reset();
        
    private:
        std::string name_;
        uint64_t start_cycles_;
        bool active_;
    };
    
    // Manual measurements
    void record_latency(const std::string& operation, uint64_t latency_ns);
    void increment_counter(const std::string& name, uint64_t value = 1);
    void set_gauge(const std::string& name, double value);
    void observe_histogram(const std::string& name, double value);
    
    // Hardware performance counters
    struct HardwareCounters {
        uint64_t cpu_cycles;
        uint64_t instructions;
        uint64_t cache_references;
        uint64_t cache_misses;
        uint64_t branch_instructions;
        uint64_t branch_misses;
        uint64_t page_faults;
        uint64_t context_switches;
    };
    
    HardwareCounters get_hardware_counters() const;
    
    // SLO monitoring
    void define_slo(const std::string& name, double target_percentile, 
                   uint64_t target_latency_ns);
    bool check_slo_violation(const std::string& name) const;
    
    // Export formats
    std::string export_prometheus_metrics();
    std::string export_json_metrics();
    void save_flame_graph(const std::string& filename);
    
    // Statistics
    struct SystemStats {
        double cpu_utilization_percent;
        uint64_t memory_used_bytes;
        uint64_t memory_available_bytes;
        uint64_t network_rx_bytes;
        uint64_t network_tx_bytes;
        uint64_t disk_read_bytes;
        uint64_t disk_write_bytes;
    };
    
    SystemStats get_system_stats() const;
};
```

### Usage Example

```cpp
#include "performance-monitor/performance_monitor.hpp"
#include <thread>

int main() {
    PerformanceMonitor::Config config;
    config.collection_interval = std::chrono::milliseconds(50);
    config.enable_hardware_counters = true;
    config.metrics_port = 9090;
    
    PerformanceMonitor monitor(config);
    
    // Define SLOs
    monitor.define_slo("api_latency", 99.0, 1000000); // P99 < 1ms
    monitor.define_slo("cache_latency", 95.0, 100000); // P95 < 100μs
    
    monitor.start_collection();
    
    // Example: Monitor API endpoint performance
    auto process_request = [&monitor](const std::string& endpoint) {
        PerformanceMonitor::Timer timer("api_request_" + endpoint);
        
        // Simulate request processing
        std::this_thread::sleep_for(std::chrono::microseconds(500));
        
        // Manual latency recording
        monitor.record_latency("endpoint_" + endpoint, timer.elapsed_ns());
        monitor.increment_counter("requests_total");
        
        // Check for SLO violations
        if (monitor.check_slo_violation("api_latency")) {
            std::cout << "SLO violation detected for API latency!" << std::endl;
        }
    };
    
    // Simulate traffic
    for (int i = 0; i < 1000; ++i) {
        process_request("/api/posts");
        process_request("/api/users");
        
        if (i % 100 == 0) {
            // Monitor system health
            auto hw_counters = monitor.get_hardware_counters();
            auto sys_stats = monitor.get_system_stats();
            
            std::cout << "Hardware Counters:\n"
                      << "  CPU Cycles: " << hw_counters.cpu_cycles << "\n"
                      << "  Cache Misses: " << hw_counters.cache_misses << "\n"
                      << "  Branch Misses: " << hw_counters.branch_misses << "\n";
            
            std::cout << "System Stats:\n"
                      << "  CPU Utilization: " << sys_stats.cpu_utilization_percent << "%\n"
                      << "  Memory Used: " << sys_stats.memory_used_bytes / (1024*1024) << "MB\n";
            
            // Export metrics for external monitoring
            auto prometheus_metrics = monitor.export_prometheus_metrics();
            // Send to Prometheus or save to file
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    
    monitor.stop_collection();
    return 0;
}
```

## Error Handling

Comprehensive error handling with automatic recovery and graceful degradation.

### Class: ErrorHandler

```cpp
#include "common/error_handling.hpp"

class ErrorHandler {
public:
    enum class ErrorType {
        PERFORMANCE_DEGRADATION,
        HARDWARE_FAILURE,
        MEMORY_EXHAUSTION,
        NETWORK_FAILURE,
        GPU_ERROR,
        CACHE_CORRUPTION
    };
    
    enum class Severity {
        LOW,
        MEDIUM,
        HIGH,
        CRITICAL
    };
    
    struct ErrorContext {
        ErrorType type;
        Severity severity;
        std::string component;
        std::string message;
        std::chrono::system_clock::time_point timestamp;
        std::map<std::string, std::string> metadata;
    };
    
    using ErrorCallback = std::function<void(const ErrorContext&)>;
    using RecoveryAction = std::function<bool()>;
    
    static ErrorHandler& instance();
    
    // Error reporting
    void report_error(ErrorType type, Severity severity, 
                     const std::string& component, const std::string& message);
    void report_error(const ErrorContext& context);
    
    // Recovery registration
    void register_recovery_action(ErrorType type, RecoveryAction action);
    void register_error_callback(ErrorCallback callback);
    
    // Circuit breaker
    void enable_circuit_breaker(const std::string& component, 
                               uint32_t failure_threshold = 5,
                               std::chrono::seconds timeout = std::chrono::seconds(30));
    bool is_circuit_open(const std::string& component) const;
    
    // Graceful degradation
    void enable_graceful_degradation(const std::string& feature);
    bool is_feature_degraded(const std::string& feature) const;
    
    // Statistics
    struct ErrorStats {
        std::map<ErrorType, uint64_t> error_counts;
        std::map<std::string, uint64_t> component_errors;
        uint64_t total_errors;
        uint64_t recovery_attempts;
        uint64_t successful_recoveries;
    };
    
    ErrorStats get_error_stats() const;
};
```

### Usage Example

```cpp
#include "common/error_handling.hpp"

int main() {
    auto& error_handler = ErrorHandler::instance();
    
    // Register error callbacks for logging/alerting
    error_handler.register_error_callback([](const ErrorHandler::ErrorContext& ctx) {
        std::cout << "[ERROR] " << ctx.component << ": " << ctx.message << std::endl;
        
        if (ctx.severity == ErrorHandler::Severity::CRITICAL) {
            // Send alert to monitoring system
            // ... implement alerting logic
        }
    });
    
    // Register recovery actions
    error_handler.register_recovery_action(ErrorHandler::ErrorType::MEMORY_EXHAUSTION, 
        []() -> bool {
            std::cout << "Attempting memory recovery..." << std::endl;
            // Trigger emergency cache eviction
            // ... implement memory recovery
            return true;
        });
    
    error_handler.register_recovery_action(ErrorHandler::ErrorType::NETWORK_FAILURE,
        []() -> bool {
            std::cout << "Attempting network recovery..." << std::endl;
            // Restart network connections
            // ... implement network recovery
            return true;
        });
    
    // Enable circuit breakers for external services
    error_handler.enable_circuit_breaker("database", 3, std::chrono::seconds(60));
    error_handler.enable_circuit_breaker("cache_cluster", 5, std::chrono::seconds(30));
    
    // Example: Protected database operation
    auto safe_database_operation = [&error_handler]() -> bool {
        if (error_handler.is_circuit_open("database")) {
            std::cout << "Database circuit breaker is open - using fallback" << std::endl;
            return false;
        }
        
        try {
            // Simulate database operation
            // ... database code here
            
            return true;
        } catch (const std::exception& e) {
            error_handler.report_error(
                ErrorHandler::ErrorType::NETWORK_FAILURE,
                ErrorHandler::Severity::HIGH,
                "database",
                "Database connection failed: " + std::string(e.what())
            );
            return false;
        }
    };
    
    // Test error handling
    for (int i = 0; i < 10; ++i) {
        if (!safe_database_operation()) {
            std::cout << "Database operation failed, using cache fallback" << std::endl;
        }
        
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    
    // Check error statistics
    auto stats = error_handler.get_error_stats();
    std::cout << "Error Statistics:\n"
              << "  Total errors: " << stats.total_errors << "\n"
              << "  Recovery attempts: " << stats.recovery_attempts << "\n"
              << "  Successful recoveries: " << stats.successful_recoveries << "\n";
    
    return 0;
}
```

## Configuration Management

Hot-reloadable configuration system with real-time updates and validation.

### Class: ConfigManager

```cpp
#include "common/config_manager.hpp"

class ConfigManager {
public:
    using ConfigChangeCallback = std::function<void(const std::string& key, const std::string& old_value, const std::string& new_value)>;
    
    static ConfigManager& instance();
    
    // Configuration loading
    void load_from_file(const std::string& filename);
    void load_from_json(const std::string& json_content);
    void load_from_environment();
    
    // Value retrieval with type safety
    template<typename T>
    T get_config(const std::string& key, const T& default_value = T{}) const;
    
    // Specialized getters
    std::string get_string(const std::string& key, const std::string& default_value = "") const;
    int64_t get_int(const std::string& key, int64_t default_value = 0) const;
    double get_double(const std::string& key, double default_value = 0.0) const;
    bool get_bool(const std::string& key, bool default_value = false) const;
    
    // Configuration updates
    void set_config(const std::string& key, const std::string& value);
    void remove_config(const std::string& key);
    
    // Hot reloading
    void enable_hot_reload(const std::string& config_file);
    void disable_hot_reload();
    void reload_config();
    
    // Change notifications
    void watch_config_changes(const std::string& key_pattern, ConfigChangeCallback callback);
    void unwatch_config_changes(const std::string& key_pattern);
    
    // Validation
    void add_validator(const std::string& key, std::function<bool(const std::string&)> validator);
    bool validate_config() const;
    
    // Export/Import
    std::string export_to_json() const;
    std::vector<std::pair<std::string, std::string>> get_all_configs() const;
};
```

### Usage Example

```cpp
#include "common/config_manager.hpp"

int main() {
    auto& config = ConfigManager::instance();
    
    // Load configuration from multiple sources
    config.load_from_file("/etc/ultra-cpp/config.json");
    config.load_from_environment(); // Override with env vars
    
    // Enable hot reloading
    config.enable_hot_reload("/etc/ultra-cpp/config.json");
    
    // Set up configuration watchers
    config.watch_config_changes("cache.*", [](const std::string& key, 
                                             const std::string& old_val, 
                                             const std::string& new_val) {
        std::cout << "Cache config changed: " << key 
                  << " from '" << old_val << "' to '" << new_val << "'" << std::endl;
        // Reconfigure cache with new settings
    });
    
    config.watch_config_changes("performance.*", [](const std::string& key,
                                                   const std::string& old_val,
                                                   const std::string& new_val) {
        std::cout << "Performance config changed: " << key << std::endl;
        // Update performance monitoring settings
    });
    
    // Add validators
    config.add_validator("cache.size", [](const std::string& value) {
        try {
            auto size = std::stoull(value);
            return size > 0 && size <= 10ULL * 1024 * 1024 * 1024; // Max 10GB
        } catch (...) {
            return false;
        }
    });
    
    config.add_validator("server.port", [](const std::string& value) {
        try {
            auto port = std::stoi(value);
            return port > 0 && port < 65536;
        } catch (...) {
            return false;
        }
    });
    
    // Use configuration values
    auto server_port = config.get_int("server.port", 8080);
    auto cache_size = config.get_int("cache.size", 1024 * 1024 * 1024);
    auto enable_dpdk = config.get_bool("network.enable_dpdk", true);
    auto log_level = config.get_string("logging.level", "INFO");
    
    std::cout << "Configuration loaded:\n"
              << "  Server port: " << server_port << "\n"
              << "  Cache size: " << cache_size << " bytes\n"
              << "  DPDK enabled: " << (enable_dpdk ? "yes" : "no") << "\n"
              << "  Log level: " << log_level << "\n";
    
    // Validate all configuration
    if (!config.validate_config()) {
        std::cerr << "Configuration validation failed!" << std::endl;
        return 1;
    }
    
    // Simulate runtime - config changes will be automatically detected
    std::cout << "Running... (config changes will be detected automatically)" << std::endl;
    std::this_thread::sleep_for(std::chrono::minutes(5));
    
    return 0;
}
```

## Integration Patterns

### Node.js Integration

```cpp
// HTTP proxy for seamless fallback
class NodeJSProxy {
public:
    struct Config {
        std::string upstream_host = "localhost";
        uint16_t upstream_port = 3005;
        std::chrono::seconds timeout{5};
        size_t connection_pool_size = 100;
    };
    
    explicit NodeJSProxy(const Config& config);
    
    HttpResponse forward_request(const HttpRequest& request);
    std::future<HttpResponse> forward_request_async(const HttpRequest& request);
    
    bool is_healthy() const;
    void enable_circuit_breaker(uint32_t failure_threshold = 5);
};
```

### Shared State Management

```cpp
// Redis integration for shared session state
class SharedStateManager {
public:
    void set_session_data(const std::string& session_id, const std::string& data);
    std::optional<std::string> get_session_data(const std::string& session_id);
    void invalidate_session(const std::string& session_id);
    
    void publish_event(const std::string& channel, const std::string& event);
    void subscribe_to_events(const std::string& channel, 
                           std::function<void(const std::string&)> handler);
};
```

This comprehensive API documentation provides detailed examples and usage patterns for all major components of the Ultra Low-Latency C++ System, enabling developers to effectively integrate and utilize the high-performance capabilities while maintaining compatibility with existing Node.js infrastructure.