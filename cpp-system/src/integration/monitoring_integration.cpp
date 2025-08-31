#include "integration/monitoring_integration.hpp"
#include "common/logger.hpp"
#include "performance-monitor/performance_monitor.hpp"
#include <curl/curl.h>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <fstream>
#include <sys/resource.h>
#include <sys/sysinfo.h>
#include <unistd.h>

namespace integration {

// Simple HTTP server for metrics endpoint
class MonitoringIntegration::MetricsHttpServer {
public:
    explicit MetricsHttpServer(MonitoringIntegration* monitoring, 
                              const std::string& bind_address, 
                              uint16_t port)
        : monitoring_(monitoring), bind_address_(bind_address), port_(port), running_(false) {}
    
    ~MetricsHttpServer() {
        stop();
    }
    
    bool start() {
        if (running_.load()) {
            return true;
        }
        
        // For simplicity, we'll create a basic HTTP server
        // In production, you might want to use a proper HTTP library
        server_thread_ = std::thread(&MetricsHttpServer::server_loop, this);
        running_.store(true);
        
        LOG_INFO("Metrics HTTP server started on {}:{}", bind_address_, port_);
        return true;
    }
    
    void stop() {
        if (!running_.load()) {
            return;
        }
        
        running_.store(false);
        
        if (server_thread_.joinable()) {
            server_thread_.join();
        }
        
        LOG_INFO("Metrics HTTP server stopped");
    }
    
private:
    MonitoringIntegration* monitoring_;
    std::string bind_address_;
    uint16_t port_;
    std::atomic<bool> running_;
    std::thread server_thread_;
    
    void server_loop() {
        // Simplified HTTP server implementation
        // In production, use a proper HTTP server library
        while (running_.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            // TODO: Implement actual HTTP server
        }
    }
};

// HTTP client for push gateway
class MonitoringIntegration::HttpClient {
public:
    HttpClient() {
        curl_global_init(CURL_GLOBAL_DEFAULT);
        curl_ = curl_easy_init();
    }
    
    ~HttpClient() {
        if (curl_) {
            curl_easy_cleanup(curl_);
        }
        curl_global_cleanup();
    }
    
    struct Response {
        long status_code = 0;
        std::string body;
        double total_time = 0.0;
    };
    
    Response post(const std::string& url, 
                 const std::string& data,
                 const std::unordered_map<std::string, std::string>& headers = {}) {
        Response response;
        
        if (!curl_) {
            return response;
        }
        
        curl_easy_reset(curl_);
        
        // Set URL
        curl_easy_setopt(curl_, CURLOPT_URL, url.c_str());
        
        // Set POST data
        curl_easy_setopt(curl_, CURLOPT_POST, 1L);
        curl_easy_setopt(curl_, CURLOPT_POSTFIELDS, data.c_str());
        curl_easy_setopt(curl_, CURLOPT_POSTFIELDSIZE, data.length());
        
        // Set headers
        struct curl_slist* header_list = nullptr;
        for (const auto& [key, value] : headers) {
            std::string header = key + ": " + value;
            header_list = curl_slist_append(header_list, header.c_str());
        }
        if (header_list) {
            curl_easy_setopt(curl_, CURLOPT_HTTPHEADER, header_list);
        }
        
        // Set timeout
        curl_easy_setopt(curl_, CURLOPT_TIMEOUT, 30L);
        
        // Set write callback
        curl_easy_setopt(curl_, CURLOPT_WRITEFUNCTION, write_callback);
        curl_easy_setopt(curl_, CURLOPT_WRITEDATA, &response.body);
        
        // Perform request
        CURLcode res = curl_easy_perform(curl_);
        
        // Clean up headers
        if (header_list) {
            curl_slist_free_all(header_list);
        }
        
        if (res == CURLE_OK) {
            curl_easy_getinfo(curl_, CURLINFO_RESPONSE_CODE, &response.status_code);
            curl_easy_getinfo(curl_, CURLINFO_TOTAL_TIME, &response.total_time);
        }
        
        return response;
    }
    
private:
    CURL* curl_;
    
    static size_t write_callback(void* contents, size_t size, size_t nmemb, std::string* userp) {
        size_t total_size = size * nmemb;
        userp->append(static_cast<char*>(contents), total_size);
        return total_size;
    }
};

MonitoringIntegration::MonitoringIntegration(const Config& config) 
    : config_(config), http_client_(std::make_unique<HttpClient>()) {
    
    // Generate instance ID if not provided
    if (config_.instance_id.empty()) {
        char hostname[256];
        if (gethostname(hostname, sizeof(hostname)) == 0) {
            config_.instance_id = std::string(hostname) + ":" + std::to_string(getpid());
        } else {
            config_.instance_id = "unknown:" + std::to_string(getpid());
        }
    }
    
    // Create HTTP server if enabled
    if (config_.enable_http_server) {
        http_server_ = std::make_unique<MetricsHttpServer>(
            this, config_.metrics_bind_address, config_.metrics_port
        );
    }
}

MonitoringIntegration::~MonitoringIntegration() {
    shutdown();
}

bool MonitoringIntegration::initialize() {
    if (running_.load()) {
        return true;
    }
    
    // Register standard metrics
    StandardMetrics::register_http_metrics(this);
    StandardMetrics::register_cache_metrics(this);
    StandardMetrics::register_database_metrics(this);
    StandardMetrics::register_system_metrics(this);
    StandardMetrics::register_session_metrics(this);
    StandardMetrics::register_event_metrics(this);
    StandardMetrics::register_performance_metrics(this);
    
    // Start HTTP server
    if (http_server_ && !http_server_->start()) {
        LOG_ERROR("Failed to start metrics HTTP server");
        return false;
    }
    
    // Start collection thread
    collection_thread_ = std::thread(&MonitoringIntegration::collection_loop, this);
    
    // Start push thread if push gateway is enabled
    if (config_.enable_push_gateway && !config_.pushgateway_url.empty()) {
        push_thread_ = std::thread(&MonitoringIntegration::push_loop, this);
    }
    
    running_.store(true);
    
    LOG_INFO("MonitoringIntegration initialized successfully");
    return true;
}

void MonitoringIntegration::shutdown() {
    if (!running_.load()) {
        return;
    }
    
    running_.store(false);
    
    // Stop HTTP server
    if (http_server_) {
        http_server_->stop();
    }
    
    // Wait for threads to finish
    if (collection_thread_.joinable()) {
        collection_thread_.join();
    }
    
    if (push_thread_.joinable()) {
        push_thread_.join();
    }
    
    LOG_INFO("MonitoringIntegration shut down");
}

bool MonitoringIntegration::register_counter(const std::string& name, 
                                           const std::string& help,
                                           const std::vector<std::string>& label_names) {
    std::unique_lock<std::shared_mutex> lock(metrics_mutex_);
    
    MetricDefinition def;
    def.name = format_metric_name(name);
    def.help = help;
    def.type = MetricType::COUNTER;
    def.label_names = label_names;
    
    metric_definitions_[def.name] = def;
    counter_values_[def.name] = {};
    
    LOG_DEBUG("Registered counter metric: {}", def.name);
    return true;
}

bool MonitoringIntegration::register_gauge(const std::string& name, 
                                         const std::string& help,
                                         const std::vector<std::string>& label_names) {
    std::unique_lock<std::shared_mutex> lock(metrics_mutex_);
    
    MetricDefinition def;
    def.name = format_metric_name(name);
    def.help = help;
    def.type = MetricType::GAUGE;
    def.label_names = label_names;
    
    metric_definitions_[def.name] = def;
    gauge_values_[def.name] = {};
    
    LOG_DEBUG("Registered gauge metric: {}", def.name);
    return true;
}

bool MonitoringIntegration::register_histogram(const std::string& name, 
                                             const std::string& help,
                                             const std::vector<double>& buckets,
                                             const std::vector<std::string>& label_names) {
    std::unique_lock<std::shared_mutex> lock(metrics_mutex_);
    
    MetricDefinition def;
    def.name = format_metric_name(name);
    def.help = help;
    def.type = MetricType::HISTOGRAM;
    def.label_names = label_names;
    def.histogram_buckets = buckets;
    
    metric_definitions_[def.name] = def;
    histogram_values_[def.name] = {};
    
    LOG_DEBUG("Registered histogram metric: {}", def.name);
    return true;
}

void MonitoringIntegration::increment_counter(const std::string& name, 
                                            double value,
                                            const std::unordered_map<std::string, std::string>& labels) {
    std::string formatted_name = format_metric_name(name);
    
    std::unique_lock<std::shared_mutex> lock(metrics_mutex_);
    
    auto it = counter_values_.find(formatted_name);
    if (it == counter_values_.end()) {
        LOG_WARN("Counter metric not registered: {}", formatted_name);
        return;
    }
    
    // Find existing metric with same labels or create new one
    bool found = false;
    for (auto& metric : it->second) {
        if (metric.labels == labels) {
            metric.value += value;
            metric.timestamp = std::chrono::system_clock::now();
            found = true;
            break;
        }
    }
    
    if (!found) {
        MetricValue metric;
        metric.value = value;
        metric.labels = labels;
        metric.timestamp = std::chrono::system_clock::now();
        it->second.push_back(metric);
    }
    
    stats_.metrics_collected.fetch_add(1);
}

void MonitoringIntegration::set_gauge(const std::string& name, 
                                    double value,
                                    const std::unordered_map<std::string, std::string>& labels) {
    std::string formatted_name = format_metric_name(name);
    
    std::unique_lock<std::shared_mutex> lock(metrics_mutex_);
    
    auto it = gauge_values_.find(formatted_name);
    if (it == gauge_values_.end()) {
        LOG_WARN("Gauge metric not registered: {}", formatted_name);
        return;
    }
    
    // Find existing metric with same labels or create new one
    bool found = false;
    for (auto& metric : it->second) {
        if (metric.labels == labels) {
            metric.value = value;
            metric.timestamp = std::chrono::system_clock::now();
            found = true;
            break;
        }
    }
    
    if (!found) {
        MetricValue metric;
        metric.value = value;
        metric.labels = labels;
        metric.timestamp = std::chrono::system_clock::now();
        it->second.push_back(metric);
    }
    
    stats_.metrics_collected.fetch_add(1);
}

void MonitoringIntegration::observe_histogram(const std::string& name, 
                                            double value,
                                            const std::unordered_map<std::string, std::string>& labels) {
    std::string formatted_name = format_metric_name(name);
    
    std::unique_lock<std::shared_mutex> lock(metrics_mutex_);
    
    auto it = histogram_values_.find(formatted_name);
    if (it == histogram_values_.end()) {
        LOG_WARN("Histogram metric not registered: {}", formatted_name);
        return;
    }
    
    auto def_it = metric_definitions_.find(formatted_name);
    if (def_it == metric_definitions_.end()) {
        return;
    }
    
    // Find existing histogram with same labels or create new one
    HistogramValue* histogram = nullptr;
    for (auto& hist : it->second) {
        if (hist.labels == labels) {
            histogram = &hist;
            break;
        }
    }
    
    if (!histogram) {
        HistogramValue new_hist;
        new_hist.labels = labels;
        new_hist.count = 0;
        new_hist.sum = 0.0;
        new_hist.timestamp = std::chrono::system_clock::now();
        
        // Initialize buckets
        for (double bucket : def_it->second.histogram_buckets) {
            new_hist.buckets[bucket] = 0;
        }
        new_hist.buckets[std::numeric_limits<double>::infinity()] = 0; // +Inf bucket
        
        it->second.push_back(new_hist);
        histogram = &it->second.back();
    }
    
    // Update histogram
    histogram->count++;
    histogram->sum += value;
    histogram->timestamp = std::chrono::system_clock::now();
    
    // Update buckets
    for (auto& [bucket_le, count] : histogram->buckets) {
        if (value <= bucket_le) {
            count++;
        }
    }
    
    stats_.metrics_collected.fetch_add(1);
}

std::string MonitoringIntegration::export_prometheus_format() {
    auto start_time = std::chrono::steady_clock::now();
    
    std::ostringstream output;
    
    std::shared_lock<std::shared_mutex> lock(metrics_mutex_);
    
    // Export counters
    for (const auto& [name, values] : counter_values_) {
        auto def_it = metric_definitions_.find(name);
        if (def_it == metric_definitions_.end()) continue;
        
        const auto& def = def_it->second;
        
        output << "# HELP " << name << " " << def.help << "\n";
        output << "# TYPE " << name << " counter\n";
        
        for (const auto& value : values) {
            output << name;
            if (!value.labels.empty()) {
                output << "{" << format_labels(value.labels) << "}";
            }
            output << " " << std::fixed << std::setprecision(6) << value.value << "\n";
        }
    }
    
    // Export gauges
    for (const auto& [name, values] : gauge_values_) {
        auto def_it = metric_definitions_.find(name);
        if (def_it == metric_definitions_.end()) continue;
        
        const auto& def = def_it->second;
        
        output << "# HELP " << name << " " << def.help << "\n";
        output << "# TYPE " << name << " gauge\n";
        
        for (const auto& value : values) {
            output << name;
            if (!value.labels.empty()) {
                output << "{" << format_labels(value.labels) << "}";
            }
            output << " " << std::fixed << std::setprecision(6) << value.value << "\n";
        }
    }
    
    // Export histograms
    for (const auto& [name, histograms] : histogram_values_) {
        auto def_it = metric_definitions_.find(name);
        if (def_it == metric_definitions_.end()) continue;
        
        const auto& def = def_it->second;
        
        output << "# HELP " << name << " " << def.help << "\n";
        output << "# TYPE " << name << " histogram\n";
        
        for (const auto& histogram : histograms) {
            std::string base_labels = format_labels(histogram.labels);
            
            // Export buckets
            for (const auto& [bucket_le, count] : histogram.buckets) {
                output << name << "_bucket{";
                if (!base_labels.empty()) {
                    output << base_labels << ",";
                }
                output << "le=\"";
                if (std::isinf(bucket_le)) {
                    output << "+Inf";
                } else {
                    output << std::fixed << std::setprecision(6) << bucket_le;
                }
                output << "\"} " << count << "\n";
            }
            
            // Export count and sum
            output << name << "_count";
            if (!histogram.labels.empty()) {
                output << "{" << base_labels << "}";
            }
            output << " " << histogram.count << "\n";
            
            output << name << "_sum";
            if (!histogram.labels.empty()) {
                output << "{" << base_labels << "}";
            }
            output << " " << std::fixed << std::setprecision(6) << histogram.sum << "\n";
        }
    }
    
    // Add static labels to all metrics if configured
    // This is a simplified implementation - in production, you'd want to
    // properly merge static labels with metric labels
    
    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    stats_.last_export_duration_ms.store(duration);
    stats_.metrics_exported.fetch_add(1);
    
    return output.str();
}

void MonitoringIntegration::collection_loop() {
    while (running_.load()) {
        auto start_time = std::chrono::steady_clock::now();
        
        try {
            collect_system_metrics();
            collect_performance_metrics();
            collect_application_metrics();
            
            // Run custom collectors
            {
                std::lock_guard<std::mutex> lock(collectors_mutex_);
                for (const auto& [name, collector] : custom_collectors_) {
                    try {
                        auto metrics = collector();
                        for (const auto& [metric_name, metric_value] : metrics) {
                            set_gauge(metric_name, metric_value.value, metric_value.labels);
                        }
                    } catch (const std::exception& e) {
                        LOG_ERROR("Custom collector {} failed: {}", name, e.what());
                    }
                }
            }
            
        } catch (const std::exception& e) {
            LOG_ERROR("Metrics collection failed: {}", e.what());
        }
        
        auto end_time = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        stats_.last_collection_duration_ms.store(duration);
        
        std::this_thread::sleep_for(std::chrono::milliseconds(config_.collection_interval_ms));
    }
}

void MonitoringIntegration::push_loop() {
    while (running_.load()) {
        try {
            if (push_to_gateway()) {
                stats_.push_successes.fetch_add(1);
            } else {
                stats_.push_failures.fetch_add(1);
            }
            stats_.push_attempts.fetch_add(1);
            
        } catch (const std::exception& e) {
            LOG_ERROR("Push to gateway failed: {}", e.what());
            stats_.push_failures.fetch_add(1);
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(config_.push_interval_ms));
    }
}

bool MonitoringIntegration::push_to_gateway() {
    if (config_.pushgateway_url.empty()) {
        return false;
    }
    
    std::string metrics_data = export_prometheus_format();
    if (metrics_data.empty()) {
        return false;
    }
    
    // Build push URL
    std::string url = config_.pushgateway_url + "/metrics/job/" + config_.job_name;
    url += "/instance/" + config_.instance_id;
    
    // Add static labels to URL
    for (const auto& [key, value] : config_.static_labels) {
        url += "/" + key + "/" + value;
    }
    
    std::unordered_map<std::string, std::string> headers = {
        {"Content-Type", "text/plain; version=0.0.4"}
    };
    
    auto response = http_client_->post(url, metrics_data, headers);
    
    if (response.status_code >= 200 && response.status_code < 300) {
        LOG_DEBUG("Successfully pushed metrics to gateway");
        return true;
    } else {
        LOG_ERROR("Failed to push metrics to gateway: HTTP {}", response.status_code);
        return false;
    }
}

void MonitoringIntegration::collect_system_metrics() {
    // CPU usage
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        double cpu_time = usage.ru_utime.tv_sec + usage.ru_utime.tv_usec / 1000000.0;
        cpu_time += usage.ru_stime.tv_sec + usage.ru_stime.tv_usec / 1000000.0;
        set_gauge(StandardMetrics::SYSTEM_CPU_USAGE, cpu_time * 100.0);
    }
    
    // Memory usage
    struct sysinfo info;
    if (sysinfo(&info) == 0) {
        uint64_t total_memory = info.totalram * info.mem_unit;
        uint64_t free_memory = info.freeram * info.mem_unit;
        uint64_t used_memory = total_memory - free_memory;
        
        set_gauge(StandardMetrics::SYSTEM_MEMORY_USAGE, static_cast<double>(used_memory));
    }
    
    // Process-specific memory
    std::ifstream status_file("/proc/self/status");
    std::string line;
    while (std::getline(status_file, line)) {
        if (line.substr(0, 6) == "VmRSS:") {
            std::istringstream iss(line);
            std::string key, value, unit;
            iss >> key >> value >> unit;
            
            double memory_kb = std::stod(value);
            set_gauge("process_memory_rss_bytes", memory_kb * 1024);
            break;
        }
    }
}

void MonitoringIntegration::collect_performance_metrics() {
    // This would integrate with the performance monitor
    // For now, we'll collect some basic metrics
    
    auto now = std::chrono::system_clock::now();
    auto timestamp = std::chrono::duration_cast<std::chrono::seconds>(
        now.time_since_epoch()
    ).count();
    
    set_gauge("system_uptime_seconds", static_cast<double>(timestamp));
}

void MonitoringIntegration::collect_application_metrics() {
    // Collect application-specific metrics
    // This would integrate with other components like SessionManager, EventBridge, etc.
    
    // For now, just collect some basic stats
    auto stats = get_stats();
    set_gauge("monitoring_metrics_collected_total", static_cast<double>(stats.metrics_collected.load()));
    set_gauge("monitoring_metrics_exported_total", static_cast<double>(stats.metrics_exported.load()));
    set_gauge("monitoring_collection_duration_ms", stats.last_collection_duration_ms.load());
    set_gauge("monitoring_export_duration_ms", stats.last_export_duration_ms.load());
}

std::string MonitoringIntegration::format_metric_name(const std::string& name) {
    // Ensure metric name follows Prometheus naming conventions
    std::string formatted = name;
    
    // Replace invalid characters with underscores
    std::replace_if(formatted.begin(), formatted.end(), 
                   [](char c) { return !std::isalnum(c) && c != '_' && c != ':'; }, '_');
    
    // Ensure it doesn't start with a digit
    if (!formatted.empty() && std::isdigit(formatted[0])) {
        formatted = "_" + formatted;
    }
    
    return formatted;
}

std::string MonitoringIntegration::format_labels(const std::unordered_map<std::string, std::string>& labels) {
    if (labels.empty()) {
        return "";
    }
    
    std::vector<std::string> label_pairs;
    for (const auto& [key, value] : labels) {
        label_pairs.push_back(key + "=\"" + escape_label_value(value) + "\"");
    }
    
    std::sort(label_pairs.begin(), label_pairs.end());
    
    std::ostringstream result;
    for (size_t i = 0; i < label_pairs.size(); ++i) {
        if (i > 0) result << ",";
        result << label_pairs[i];
    }
    
    return result.str();
}

std::string MonitoringIntegration::escape_label_value(const std::string& value) {
    std::string escaped;
    escaped.reserve(value.length() * 2);
    
    for (char c : value) {
        switch (c) {
            case '\\': escaped += "\\\\"; break;
            case '"': escaped += "\\\""; break;
            case '\n': escaped += "\\n"; break;
            case '\t': escaped += "\\t"; break;
            case '\r': escaped += "\\r"; break;
            default: escaped += c; break;
        }
    }
    
    return escaped;
}

MonitoringIntegration::MonitoringStats MonitoringIntegration::get_stats() const {
    return stats_;
}

bool MonitoringIntegration::health_check() {
    return running_.load();
}

// MetricTimer implementation
MetricTimer::MetricTimer(MonitoringIntegration* monitoring, 
                        const std::string& metric_name,
                        const std::unordered_map<std::string, std::string>& labels)
    : monitoring_(monitoring), metric_name_(metric_name), labels_(labels), 
      start_time_(std::chrono::steady_clock::now()), stopped_(false) {
}

MetricTimer::~MetricTimer() {
    if (!stopped_) {
        stop();
    }
}

void MetricTimer::stop() {
    if (stopped_) {
        return;
    }
    
    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration<double>(end_time - start_time_).count();
    
    monitoring_->observe_histogram(metric_name_, duration, labels_);
    stopped_ = true;
}

double MetricTimer::get_elapsed_ms() const {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(now - start_time_).count();
}

// StandardMetrics implementation
void StandardMetrics::register_http_metrics(MonitoringIntegration* monitoring) {
    monitoring->register_counter(HTTP_REQUESTS_TOTAL, 
                                "Total number of HTTP requests", 
                                {"method", "endpoint", "status"});
    
    monitoring->register_histogram(HTTP_REQUEST_DURATION, 
                                  "HTTP request duration in seconds",
                                  {0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0},
                                  {"method", "endpoint"});
    
    monitoring->register_histogram(HTTP_REQUEST_SIZE, 
                                  "HTTP request size in bytes",
                                  {100, 1000, 10000, 100000, 1000000, 10000000},
                                  {"method", "endpoint"});
    
    monitoring->register_histogram(HTTP_RESPONSE_SIZE, 
                                  "HTTP response size in bytes",
                                  {100, 1000, 10000, 100000, 1000000, 10000000},
                                  {"method", "endpoint"});
}

void StandardMetrics::register_cache_metrics(MonitoringIntegration* monitoring) {
    monitoring->register_counter(CACHE_OPERATIONS_TOTAL, 
                                "Total number of cache operations", 
                                {"operation", "result"});
    
    monitoring->register_gauge(CACHE_HIT_RATIO, 
                              "Cache hit ratio", 
                              {"cache_name"});
    
    monitoring->register_gauge(CACHE_SIZE_BYTES, 
                              "Cache size in bytes", 
                              {"cache_name"});
    
    monitoring->register_counter(CACHE_EVICTIONS_TOTAL, 
                                "Total number of cache evictions", 
                                {"cache_name", "reason"});
}

void StandardMetrics::register_system_metrics(MonitoringIntegration* monitoring) {
    monitoring->register_gauge(SYSTEM_CPU_USAGE, 
                              "System CPU usage percentage", 
                              {});
    
    monitoring->register_gauge(SYSTEM_MEMORY_USAGE, 
                              "System memory usage in bytes", 
                              {"type"});
    
    monitoring->register_counter(SYSTEM_NETWORK_BYTES, 
                                "System network bytes transferred", 
                                {"direction", "interface"});
}

} // namespace integration