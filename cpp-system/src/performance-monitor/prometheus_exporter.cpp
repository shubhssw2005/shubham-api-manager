#include "performance-monitor/prometheus_exporter.hpp"
#include "performance-monitor/metrics_collector.hpp"
#include "common/logger.hpp"

#include <sstream>
#include <iomanip>
#include <thread>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>
#include <zlib.h>

namespace ultra {
namespace monitor {

// HTTP Server implementation
class MetricsHttpServer::ServerImpl {
public:
    explicit ServerImpl(const Config& config) : config_(config) {}
    
    bool start(RequestHandler handler) {
        handler_ = std::move(handler);
        
        // Create socket
        server_fd_ = socket(AF_INET, SOCK_STREAM, 0);
        if (server_fd_ < 0) {
            LOG_ERROR("Failed to create socket: {}", strerror(errno));
            return false;
        }

        // Set socket options
        int opt = 1;
        if (setsockopt(server_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
            LOG_WARN("Failed to set SO_REUSEADDR: {}", strerror(errno));
        }

        // Bind socket
        struct sockaddr_in address;
        address.sin_family = AF_INET;
        address.sin_addr.s_addr = INADDR_ANY;
        address.sin_port = htons(config_.port);

        if (bind(server_fd_, (struct sockaddr*)&address, sizeof(address)) < 0) {
            LOG_ERROR("Failed to bind socket to port {}: {}", config_.port, strerror(errno));
            close(server_fd_);
            return false;
        }

        // Listen for connections
        if (listen(server_fd_, config_.max_connections) < 0) {
            LOG_ERROR("Failed to listen on socket: {}", strerror(errno));
            close(server_fd_);
            return false;
        }

        running_ = true;
        
        // Start worker threads
        for (size_t i = 0; i < config_.thread_pool_size; ++i) {
            worker_threads_.emplace_back(&ServerImpl::worker_loop, this);
        }

        LOG_INFO("HTTP server started on port {}", config_.port);
        return true;
    }

    void stop() {
        running_ = false;
        
        if (server_fd_ >= 0) {
            close(server_fd_);
            server_fd_ = -1;
        }

        for (auto& thread : worker_threads_) {
            if (thread.joinable()) {
                thread.join();
            }
        }
        worker_threads_.clear();
        
        LOG_INFO("HTTP server stopped");
    }

    bool is_running() const { return running_; }

private:
    Config config_;
    RequestHandler handler_;
    int server_fd_ = -1;
    std::atomic<bool> running_{false};
    std::vector<std::thread> worker_threads_;

    void worker_loop() {
        while (running_) {
            struct sockaddr_in client_addr;
            socklen_t client_len = sizeof(client_addr);
            
            int client_fd = accept(server_fd_, (struct sockaddr*)&client_addr, &client_len);
            if (client_fd < 0) {
                if (running_) {
                    LOG_WARN("Failed to accept connection: {}", strerror(errno));
                }
                continue;
            }

            handle_request(client_fd);
            close(client_fd);
        }
    }

    void handle_request(int client_fd) {
        char buffer[4096];
        ssize_t bytes_read = recv(client_fd, buffer, sizeof(buffer) - 1, 0);
        if (bytes_read <= 0) {
            return;
        }
        buffer[bytes_read] = '\0';

        // Parse HTTP request (simplified)
        std::string request(buffer);
        std::string path = parse_path(request);
        auto headers = parse_headers(request);

        if (path == config_.endpoint) {
            // Handle metrics request
            std::string response_body = handler_(path, headers);
            send_http_response(client_fd, 200, "text/plain", response_body);
        } else {
            // 404 Not Found
            send_http_response(client_fd, 404, "text/plain", "Not Found");
        }
    }

    std::string parse_path(const std::string& request) {
        size_t start = request.find(' ') + 1;
        size_t end = request.find(' ', start);
        if (start != std::string::npos && end != std::string::npos) {
            return request.substr(start, end - start);
        }
        return "/";
    }

    std::unordered_map<std::string, std::string> parse_headers(const std::string& request) {
        std::unordered_map<std::string, std::string> headers;
        std::istringstream iss(request);
        std::string line;
        
        // Skip request line
        std::getline(iss, line);
        
        while (std::getline(iss, line) && !line.empty() && line != "\r") {
            size_t colon = line.find(':');
            if (colon != std::string::npos) {
                std::string key = line.substr(0, colon);
                std::string value = line.substr(colon + 1);
                // Trim whitespace
                value.erase(0, value.find_first_not_of(" \t\r"));
                value.erase(value.find_last_not_of(" \t\r") + 1);
                headers[key] = value;
            }
        }
        
        return headers;
    }

    void send_http_response(int client_fd, int status_code, const std::string& content_type, const std::string& body) {
        std::ostringstream response;
        response << "HTTP/1.1 " << status_code << " ";
        
        switch (status_code) {
            case 200: response << "OK"; break;
            case 404: response << "Not Found"; break;
            default: response << "Unknown"; break;
        }
        
        response << "\r\n";
        response << "Content-Type: " << content_type << "\r\n";
        response << "Content-Length: " << body.size() << "\r\n";
        response << "Connection: close\r\n";
        response << "\r\n";
        response << body;

        std::string response_str = response.str();
        send(client_fd, response_str.c_str(), response_str.size(), 0);
    }
};

// PrometheusExporter implementation
PrometheusExporter::PrometheusExporter(const Config& config, MetricsCollector& collector)
    : config_(config), collector_(collector) {
    
    if (config_.instance_id.empty()) {
        config_.instance_id = "ultra-cpp-" + std::to_string(getpid());
    }
    
    formatter_ = std::make_unique<MetricsFormatter>();
    
    LOG_INFO("PrometheusExporter initialized for instance: {}", config_.instance_id);
}

PrometheusExporter::~PrometheusExporter() {
    stop_server();
    
    // Clean up cache
    CacheEntry* entry = cached_metrics_.exchange(nullptr, std::memory_order_acq_rel);
    if (entry) {
        delete entry;
    }
}

bool PrometheusExporter::start_server() {
    if (running_.exchange(true, std::memory_order_acq_rel)) {
        LOG_WARN("Prometheus server already running");
        return true;
    }

    MetricsHttpServer::Config server_config;
    server_config.port = config_.port;
    server_config.endpoint = config_.endpoint;
    
    server_ = std::make_unique<MetricsHttpServer>(server_config);
    
    auto handler = [this](const std::string& path, const std::unordered_map<std::string, std::string>& headers) {
        auto start_time = std::chrono::steady_clock::now();
        
        std::string response = export_metrics();
        
        auto end_time = std::chrono::steady_clock::now();
        auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
        
        stats_.requests_served.fetch_add(1, std::memory_order_relaxed);
        stats_.bytes_exported.fetch_add(response.size(), std::memory_order_relaxed);
        stats_.export_duration_ns.store(duration_ns, std::memory_order_relaxed);
        
        return response;
    };

    if (!server_->start(handler)) {
        running_.store(false, std::memory_order_release);
        return false;
    }

    LOG_INFO("Prometheus server started on port {}{}", config_.port, config_.endpoint);
    return true;
}

void PrometheusExporter::stop_server() {
    if (!running_.exchange(false, std::memory_order_acq_rel)) {
        return;
    }

    if (server_) {
        server_->stop();
        server_.reset();
    }

    LOG_INFO("Prometheus server stopped");
}

std::string PrometheusExporter::export_metrics() {
    // Check cache first
    CacheEntry* cached = cached_metrics_.load(std::memory_order_acquire);
    if (cached && is_cache_valid(cached)) {
        stats_.cache_hits.fetch_add(1, std::memory_order_relaxed);
        return cached->data;
    }

    stats_.cache_misses.fetch_add(1, std::memory_order_relaxed);
    
    // Generate new metrics
    std::string metrics = format_prometheus_metrics();
    
    // Update cache
    auto new_entry = std::make_unique<CacheEntry>();
    new_entry->data = metrics;
    new_entry->timestamp = std::chrono::steady_clock::now();
    
    update_cache(std::move(new_entry));
    
    return metrics;
}

std::string PrometheusExporter::export_metrics_compressed() {
    std::string uncompressed = export_metrics();
    
    // Compress using zlib
    uLongf compressed_size = compressBound(uncompressed.size());
    std::vector<char> compressed_buffer(compressed_size);
    
    int result = compress(reinterpret_cast<Bytef*>(compressed_buffer.data()), &compressed_size,
                         reinterpret_cast<const Bytef*>(uncompressed.c_str()), uncompressed.size());
    
    if (result == Z_OK) {
        stats_.compression_ratio_percent.store(
            (compressed_size * 100) / uncompressed.size(), std::memory_order_relaxed);
        return std::string(compressed_buffer.data(), compressed_size);
    }
    
    LOG_WARN("Failed to compress metrics: {}", result);
    return uncompressed;
}

size_t PrometheusExporter::export_metrics_to_buffer(char* buffer, size_t buffer_size) {
    std::string metrics = export_metrics();
    size_t copy_size = std::min(metrics.size(), buffer_size - 1);
    std::memcpy(buffer, metrics.c_str(), copy_size);
    buffer[copy_size] = '\0';
    return copy_size;
}

size_t PrometheusExporter::export_compressed_to_buffer(char* buffer, size_t buffer_size) {
    std::string compressed = export_metrics_compressed();
    size_t copy_size = std::min(compressed.size(), buffer_size);
    std::memcpy(buffer, compressed.c_str(), copy_size);
    return copy_size;
}

void PrometheusExporter::add_global_label(const std::string& key, const std::string& value) {
    global_labels_[key] = value;
    
    // Invalidate cache
    cache_version_.fetch_add(1, std::memory_order_relaxed);
    
    LOG_INFO("Added global label: {}={}", key, value);
}

void PrometheusExporter::remove_global_label(const std::string& key) {
    global_labels_.erase(key);
    
    // Invalidate cache
    cache_version_.fetch_add(1, std::memory_order_relaxed);
    
    LOG_INFO("Removed global label: {}", key);
}

void PrometheusExporter::set_instance_metadata(const std::string& version, const std::string& build_info) {
    instance_metadata_ = "version=\"" + version + "\",build_info=\"" + build_info + "\"";
    
    // Invalidate cache
    cache_version_.fetch_add(1, std::memory_order_relaxed);
    
    LOG_INFO("Set instance metadata: version={}, build_info={}", version, build_info);
}

bool PrometheusExporter::is_cache_valid(const CacheEntry* entry) const {
    if (!entry) return false;
    
    auto now = std::chrono::steady_clock::now();
    auto age = std::chrono::duration_cast<std::chrono::seconds>(now - entry->timestamp);
    
    return age < config_.cache_ttl;
}

void PrometheusExporter::update_cache(std::unique_ptr<CacheEntry> new_entry) const {
    CacheEntry* old_entry = cached_metrics_.exchange(new_entry.release(), std::memory_order_acq_rel);
    if (old_entry) {
        delete old_entry;
    }
}

std::string PrometheusExporter::format_prometheus_metrics() {
    formatter_->reset();
    formatter_->set_global_labels(global_labels_);

    // Export counters
    auto counter_names = collector_.get_counter_names();
    for (const auto& name : counter_names) {
        uint64_t value = collector_.get_counter_value(name);
        formatter_->add_help(name, "Counter metric");
        formatter_->add_type(name, "counter");
        formatter_->add_counter(name, value);
    }

    // Export gauges
    auto gauge_names = collector_.get_gauge_names();
    for (const auto& name : gauge_names) {
        double value = collector_.get_gauge_value(name);
        formatter_->add_help(name, "Gauge metric");
        formatter_->add_type(name, "gauge");
        formatter_->add_gauge(name, value);
    }

    // Export histograms
    auto histogram_names = collector_.get_histogram_names();
    for (const auto& name : histogram_names) {
        const auto* histogram = collector_.get_histogram_data(name);
        if (histogram && !histogram->buckets.empty()) {
            formatter_->add_help(name, "Histogram metric");
            formatter_->add_type(name, "histogram");
            
            std::vector<std::pair<double, uint64_t>> buckets;
            for (const auto& bucket : histogram->buckets) {
                buckets.emplace_back(bucket.upper_bound, bucket.count.load(std::memory_order_relaxed));
            }
            
            uint64_t total_count = histogram->total_count.load(std::memory_order_relaxed);
            double sum = histogram->sum.load(std::memory_order_relaxed);
            
            formatter_->add_histogram(name, buckets, total_count, sum);
        }
    }

    // Add system metrics
    formatter_->add_help("ultra_exporter_requests_total", "Total number of metrics requests");
    formatter_->add_type("ultra_exporter_requests_total", "counter");
    formatter_->add_counter("ultra_exporter_requests_total", stats_.requests_served.load());

    formatter_->add_help("ultra_exporter_bytes_exported_total", "Total bytes exported");
    formatter_->add_type("ultra_exporter_bytes_exported_total", "counter");
    formatter_->add_counter("ultra_exporter_bytes_exported_total", stats_.bytes_exported.load());

    return formatter_->get_output();
}

size_t PrometheusExporter::compress_data(const char* input, size_t input_size, char* output, size_t output_size) {
    uLongf compressed_size = output_size;
    int result = compress(reinterpret_cast<Bytef*>(output), &compressed_size,
                         reinterpret_cast<const Bytef*>(input), input_size);
    
    return (result == Z_OK) ? compressed_size : 0;
}

// PrometheusFormatter implementation
PrometheusFormatter::PrometheusFormatter(size_t initial_buffer_size)
    : capacity_(initial_buffer_size), size_(0) {
    buffer_ = std::make_unique<char[]>(capacity_);
}

PrometheusFormatter::~PrometheusFormatter() = default;

void PrometheusFormatter::reset() {
    size_ = 0;
}

void PrometheusFormatter::add_help(const std::string& metric_name, const std::string& help_text) {
    append_string("# HELP ");
    append_string(metric_name);
    append_string(" ");
    append_string(help_text);
    append_string("\n");
}

void PrometheusFormatter::add_type(const std::string& metric_name, const std::string& type) {
    append_string("# TYPE ");
    append_string(metric_name);
    append_string(" ");
    append_string(type);
    append_string("\n");
}

void PrometheusFormatter::add_counter(const std::string& name, uint64_t value, 
                                    const std::unordered_map<std::string, std::string>& labels) {
    append_string(name);
    append_labels(labels);
    append_string(" ");
    append_number(value);
    append_string("\n");
}

void PrometheusFormatter::add_gauge(const std::string& name, double value,
                                  const std::unordered_map<std::string, std::string>& labels) {
    append_string(name);
    append_labels(labels);
    append_string(" ");
    append_double(value);
    append_string("\n");
}

void PrometheusFormatter::add_histogram(const std::string& name, 
                                      const std::vector<std::pair<double, uint64_t>>& buckets,
                                      uint64_t count, double sum,
                                      const std::unordered_map<std::string, std::string>& labels) {
    // Add buckets
    for (const auto& bucket : buckets) {
        append_string(name);
        append_string("_bucket");
        
        auto bucket_labels = labels;
        bucket_labels["le"] = std::to_string(bucket.first);
        append_labels(bucket_labels);
        
        append_string(" ");
        append_number(bucket.second);
        append_string("\n");
    }
    
    // Add count
    append_string(name);
    append_string("_count");
    append_labels(labels);
    append_string(" ");
    append_number(count);
    append_string("\n");
    
    // Add sum
    append_string(name);
    append_string("_sum");
    append_labels(labels);
    append_string(" ");
    append_double(sum);
    append_string("\n");
}

std::string PrometheusFormatter::get_output() const {
    return std::string(buffer_.get(), size_);
}

void PrometheusFormatter::reserve(size_t capacity) {
    if (capacity > capacity_) {
        grow_buffer(capacity);
    }
}

void PrometheusFormatter::set_global_labels(const std::unordered_map<std::string, std::string>& labels) {
    global_labels_ = labels;
}

void PrometheusFormatter::append_string(const std::string& str) {
    append_cstring(str.c_str(), str.size());
}

void PrometheusFormatter::append_cstring(const char* str, size_t len) {
    ensure_capacity(len);
    std::memcpy(buffer_.get() + size_, str, len);
    size_ += len;
}

void PrometheusFormatter::append_number(uint64_t value) {
    char temp[32];
    int len = snprintf(temp, sizeof(temp), "%" PRIu64, value);
    append_cstring(temp, len);
}

void PrometheusFormatter::append_double(double value, int precision) {
    char temp[32];
    int len = snprintf(temp, sizeof(temp), "%.*f", precision, value);
    append_cstring(temp, len);
}

void PrometheusFormatter::append_labels(const std::unordered_map<std::string, std::string>& labels) {
    auto combined_labels = global_labels_;
    combined_labels.insert(labels.begin(), labels.end());
    
    if (combined_labels.empty()) {
        return;
    }
    
    append_string("{");
    bool first = true;
    for (const auto& [key, value] : combined_labels) {
        if (!first) {
            append_string(",");
        }
        first = false;
        
        append_string(key);
        append_string("=\"");
        // TODO: Escape value properly
        append_string(value);
        append_string("\"");
    }
    append_string("}");
}

void PrometheusFormatter::ensure_capacity(size_t additional_size) {
    if (size_ + additional_size > capacity_) {
        grow_buffer(std::max(capacity_ * 2, size_ + additional_size));
    }
}

void PrometheusFormatter::grow_buffer(size_t new_capacity) {
    auto new_buffer = std::make_unique<char[]>(new_capacity);
    if (size_ > 0) {
        std::memcpy(new_buffer.get(), buffer_.get(), size_);
    }
    buffer_ = std::move(new_buffer);
    capacity_ = new_capacity;
}

// MetricsHttpServer implementation
MetricsHttpServer::MetricsHttpServer(const Config& config)
    : config_(config), impl_(std::make_unique<ServerImpl>(config)) {
}

MetricsHttpServer::~MetricsHttpServer() {
    stop();
}

bool MetricsHttpServer::start(RequestHandler handler) {
    if (running_.exchange(true, std::memory_order_acq_rel)) {
        return true;
    }
    
    return impl_->start(std::move(handler));
}

void MetricsHttpServer::stop() {
    if (!running_.exchange(false, std::memory_order_acq_rel)) {
        return;
    }
    
    impl_->stop();
}

bool MetricsHttpServer::is_running() const noexcept {
    return running_.load(std::memory_order_acquire);
}

} // namespace monitor
} // namespace ultra