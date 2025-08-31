#include "performance-monitor/slo_monitor.hpp"
#include "performance-monitor/metrics_collector.hpp"
#include "common/logger.hpp"

#include <algorithm>
#include <sstream>
#include <thread>
#include <curl/curl.h>

namespace ultra {
namespace monitor {

// AlertManager implementation
class AlertManager::HttpClient {
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
    
    bool post_json(const std::string& url, const std::string& json_data, const std::string& auth_token = "") {
        if (!curl_) return false;
        
        curl_easy_reset(curl_);
        curl_easy_setopt(curl_, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl_, CURLOPT_POSTFIELDS, json_data.c_str());
        curl_easy_setopt(curl_, CURLOPT_TIMEOUT, 10L);
        
        struct curl_slist* headers = nullptr;
        headers = curl_slist_append(headers, "Content-Type: application/json");
        
        if (!auth_token.empty()) {
            std::string auth_header = "Authorization: Bearer " + auth_token;
            headers = curl_slist_append(headers, auth_header.c_str());
        }
        
        curl_easy_setopt(curl_, CURLOPT_HTTPHEADER, headers);
        
        CURLcode res = curl_easy_perform(curl_);
        
        curl_slist_free_all(headers);
        
        return res == CURLE_OK;
    }
    
private:
    CURL* curl_ = nullptr;
};

class AlertManager::SmtpClient {
public:
    bool send_email(const EmailConfig& config, const std::string& subject, const std::string& message) {
        // Simplified SMTP implementation - in production you'd use a proper SMTP library
        LOG_INFO("Email alert: {} - {}", subject, message);
        return true;
    }
};

AlertManager::AlertManager() 
    : http_client_(std::make_unique<HttpClient>())
    , smtp_client_(std::make_unique<SmtpClient>()) {
}

AlertManager::~AlertManager() = default;

void AlertManager::configure_webhook(const WebhookConfig& config) {
    webhook_config_ = config;
    LOG_INFO("Webhook configured: {}", config.url);
}

void AlertManager::configure_slack(const SlackConfig& config) {
    slack_config_ = config;
    LOG_INFO("Slack configured: {}", config.channel);
}

void AlertManager::configure_email(const EmailConfig& config) {
    email_config_ = config;
    LOG_INFO("Email configured: {} recipients", config.to_addresses.size());
}

bool AlertManager::send_webhook_alert(const std::string& message, const std::string& severity) {
    if (webhook_config_.url.empty()) {
        return false;
    }
    
    auto start_time = std::chrono::steady_clock::now();
    
    std::string payload = format_webhook_payload(message, severity);
    bool success = deliver_with_retry([this, &payload]() {
        return http_client_->post_json(webhook_config_.url, payload, webhook_config_.auth_token);
    }, webhook_config_.max_retries);
    
    auto end_time = std::chrono::steady_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    
    if (success) {
        stats_.webhook_alerts_sent.fetch_add(1, std::memory_order_relaxed);
    } else {
        stats_.delivery_failures.fetch_add(1, std::memory_order_relaxed);
    }
    
    stats_.delivery_duration_ms.store(duration_ms, std::memory_order_relaxed);
    
    return success;
}

bool AlertManager::send_slack_alert(const std::string& message, const std::string& severity) {
    if (slack_config_.webhook_url.empty()) {
        return false;
    }
    
    auto start_time = std::chrono::steady_clock::now();
    
    std::string payload = format_slack_payload(message, severity);
    bool success = deliver_with_retry([this, &payload]() {
        return http_client_->post_json(slack_config_.webhook_url, payload);
    }, 3);
    
    auto end_time = std::chrono::steady_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    
    if (success) {
        stats_.slack_alerts_sent.fetch_add(1, std::memory_order_relaxed);
    } else {
        stats_.delivery_failures.fetch_add(1, std::memory_order_relaxed);
    }
    
    stats_.delivery_duration_ms.store(duration_ms, std::memory_order_relaxed);
    
    return success;
}

bool AlertManager::send_email_alert(const std::string& subject, const std::string& message) {
    if (email_config_.to_addresses.empty()) {
        return false;
    }
    
    auto start_time = std::chrono::steady_clock::now();
    
    bool success = smtp_client_->send_email(email_config_, subject, message);
    
    auto end_time = std::chrono::steady_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    
    if (success) {
        stats_.email_alerts_sent.fetch_add(1, std::memory_order_relaxed);
    } else {
        stats_.delivery_failures.fetch_add(1, std::memory_order_relaxed);
    }
    
    stats_.delivery_duration_ms.store(duration_ms, std::memory_order_relaxed);
    
    return success;
}

void AlertManager::send_alert_to_all_channels(const std::string& message, const std::string& severity) {
    std::vector<std::thread> delivery_threads;
    
    // Send to webhook
    if (!webhook_config_.url.empty()) {
        delivery_threads.emplace_back([this, message, severity]() {
            send_webhook_alert(message, severity);
        });
    }
    
    // Send to Slack
    if (!slack_config_.webhook_url.empty()) {
        delivery_threads.emplace_back([this, message, severity]() {
            send_slack_alert(message, severity);
        });
    }
    
    // Send email
    if (!email_config_.to_addresses.empty()) {
        delivery_threads.emplace_back([this, message]() {
            send_email_alert("SLO Alert", message);
        });
    }
    
    // Wait for all deliveries to complete
    for (auto& thread : delivery_threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

std::string AlertManager::format_slack_payload(const std::string& message, const std::string& severity) {
    std::ostringstream oss;
    oss << "{"
        << "\"channel\":\"" << slack_config_.channel << "\","
        << "\"username\":\"" << slack_config_.username << "\","
        << "\"icon_emoji\":\"" << slack_config_.icon_emoji << "\","
        << "\"text\":\"" << message << "\","
        << "\"attachments\":[{"
        << "\"color\":\"" << (severity == "critical" ? "danger" : "warning") << "\","
        << "\"fields\":[{"
        << "\"title\":\"Severity\","
        << "\"value\":\"" << severity << "\","
        << "\"short\":true"
        << "}]"
        << "}]"
        << "}";
    return oss.str();
}

std::string AlertManager::format_webhook_payload(const std::string& message, const std::string& severity) {
    std::ostringstream oss;
    oss << "{"
        << "\"message\":\"" << message << "\","
        << "\"severity\":\"" << severity << "\","
        << "\"timestamp\":\"" << std::chrono::duration_cast<std::chrono::seconds>(
               std::chrono::system_clock::now().time_since_epoch()).count() << "\","
        << "\"source\":\"ultra-cpp-slo-monitor\""
        << "}";
    return oss.str();
}

bool AlertManager::deliver_with_retry(std::function<bool()> delivery_func, int max_retries) {
    for (int attempt = 0; attempt <= max_retries; ++attempt) {
        if (delivery_func()) {
            return true;
        }
        
        if (attempt < max_retries) {
            // Exponential backoff
            std::this_thread::sleep_for(std::chrono::milliseconds(100 * (1 << attempt)));
        }
    }
    
    return false;
}

// SLOMonitor implementation
SLOMonitor::SLOMonitor(const Config& config, MetricsCollector& collector)
    : config_(config), collector_(collector), alert_manager_(std::make_unique<AlertManager>()) {
    
    // Configure alert manager
    if (config_.alert_config.enable_webhook && !config_.alert_config.webhook_url.empty()) {
        AlertManager::WebhookConfig webhook_config;
        webhook_config.url = config_.alert_config.webhook_url;
        alert_manager_->configure_webhook(webhook_config);
    }
    
    if (config_.alert_config.enable_slack && !config_.alert_config.slack_channel.empty()) {
        AlertManager::SlackConfig slack_config;
        slack_config.channel = config_.alert_config.slack_channel;
        alert_manager_->configure_slack(slack_config);
    }
    
    if (config_.alert_config.enable_email && !config_.alert_config.email_recipients.empty()) {
        AlertManager::EmailConfig email_config;
        // Parse email recipients (simplified)
        std::istringstream iss(config_.alert_config.email_recipients);
        std::string email;
        while (std::getline(iss, email, ',')) {
            email_config.to_addresses.push_back(email);
        }
        alert_manager_->configure_email(email_config);
    }
    
    LOG_INFO("SLOMonitor initialized with {}ms check interval", config_.check_interval.count());
}

SLOMonitor::~SLOMonitor() {
    stop_monitoring();
}

void SLOMonitor::register_slo(const SLODefinition& slo) {
    std::lock_guard<std::mutex> lock(slo_mutex_);
    
    auto slo_state = std::make_unique<SLOState>();
    slo_state->definition = slo;
    slo_state->status.name = slo.name;
    slo_state->status.is_healthy = true;
    slo_state->last_check = std::chrono::steady_clock::now();
    
    // Initialize error budget
    slo_state->error_budget.slo_name = slo.name;
    slo_state->error_budget.total_budget_percent = slo.error_budget_percent;
    slo_state->error_budget.consumed_budget_percent = 0.0;
    slo_state->error_budget.remaining_budget_percent = slo.error_budget_percent;
    slo_state->error_budget.allowed_violations = calculate_allowed_violations(slo, slo.evaluation_window);
    slo_state->error_budget.actual_violations = 0;
    slo_state->error_budget.budget_reset_time = std::chrono::steady_clock::now() + slo.evaluation_window;
    
    slos_[slo.name] = std::move(slo_state);
    
    LOG_INFO("Registered SLO: {} (P{} < {}ns, {}% error budget)", 
             slo.name, slo.target_percentile * 100, slo.target_latency_ns, slo.error_budget_percent);
}

void SLOMonitor::unregister_slo(const std::string& name) {
    std::lock_guard<std::mutex> lock(slo_mutex_);
    
    auto it = slos_.find(name);
    if (it != slos_.end()) {
        slos_.erase(it);
        LOG_INFO("Unregistered SLO: {}", name);
    }
}

void SLOMonitor::update_slo(const SLODefinition& slo) {
    std::lock_guard<std::mutex> lock(slo_mutex_);
    
    auto it = slos_.find(slo.name);
    if (it != slos_.end()) {
        it->second->definition = slo;
        LOG_INFO("Updated SLO: {}", slo.name);
    }
}

std::vector<SLOMonitor::SLODefinition> SLOMonitor::get_all_slos() const {
    std::lock_guard<std::mutex> lock(slo_mutex_);
    
    std::vector<SLODefinition> slos;
    slos.reserve(slos_.size());
    
    for (const auto& [name, state] : slos_) {
        slos.push_back(state->definition);
    }
    
    return slos;
}

void SLOMonitor::start_monitoring() {
    if (monitoring_.exchange(true, std::memory_order_acq_rel)) {
        LOG_WARN("SLO monitoring already started");
        return;
    }
    
    monitoring_thread_ = std::make_unique<std::thread>(&SLOMonitor::monitoring_loop, this);
    LOG_INFO("SLO monitoring started");
}

void SLOMonitor::stop_monitoring() {
    if (!monitoring_.exchange(false, std::memory_order_acq_rel)) {
        return;
    }
    
    if (monitoring_thread_ && monitoring_thread_->joinable()) {
        monitoring_thread_->join();
        monitoring_thread_.reset();
    }
    
    LOG_INFO("SLO monitoring stopped");
}

SLOMonitor::SLOStatus SLOMonitor::get_slo_status(const std::string& name) const {
    std::lock_guard<std::mutex> lock(slo_mutex_);
    
    auto it = slos_.find(name);
    if (it != slos_.end()) {
        return it->second->status;
    }
    
    return SLOStatus{};
}

std::vector<SLOMonitor::SLOStatus> SLOMonitor::get_all_slo_status() const {
    std::lock_guard<std::mutex> lock(slo_mutex_);
    
    std::vector<SLOStatus> statuses;
    statuses.reserve(slos_.size());
    
    for (const auto& [name, state] : slos_) {
        statuses.push_back(state->status);
    }
    
    return statuses;
}

SLOMonitor::ErrorBudget SLOMonitor::get_error_budget(const std::string& slo_name) const {
    std::lock_guard<std::mutex> lock(slo_mutex_);
    
    auto it = slos_.find(slo_name);
    if (it != slos_.end()) {
        return it->second->error_budget;
    }
    
    return ErrorBudget{};
}

std::vector<SLOMonitor::PredictiveAlert> SLOMonitor::get_predictive_alerts() const {
    std::vector<PredictiveAlert> alerts;
    
    if (!config_.enable_predictive_alerting) {
        return alerts;
    }
    
    std::lock_guard<std::mutex> lock(slo_mutex_);
    
    for (const auto& [name, state] : slos_) {
        double predicted_consumption = predict_budget_consumption(*state);
        
        if (predicted_consumption > config_.prediction_threshold) {
            PredictiveAlert alert;
            alert.slo_name = name;
            alert.predicted_budget_consumption_percent = predicted_consumption * 100;
            alert.confidence_level = 0.85;  // Simplified confidence
            alert.prediction_model = "linear_trend";
            
            // Estimate time to budget exhaustion
            double remaining_budget = state->error_budget.remaining_budget_percent / 100.0;
            double consumption_rate = predicted_consumption - (state->error_budget.consumed_budget_percent / 100.0);
            
            if (consumption_rate > 0) {
                double hours_to_exhaustion = remaining_budget / consumption_rate;
                alert.time_to_budget_exhaustion = std::chrono::seconds(static_cast<int64_t>(hours_to_exhaustion * 3600));
            }
            
            alerts.push_back(alert);
        }
    }
    
    return alerts;
}

void SLOMonitor::trigger_manual_alert(const std::string& slo_name, const std::string& message) {
    std::lock_guard<std::mutex> lock(slo_mutex_);
    
    auto it = slos_.find(slo_name);
    if (it != slos_.end()) {
        send_alert(*it->second, message);
        LOG_INFO("Manual alert triggered for SLO: {} - {}", slo_name, message);
    }
}

void SLOMonitor::silence_alerts(const std::string& slo_name, std::chrono::seconds duration) {
    std::lock_guard<std::mutex> lock(slo_mutex_);
    
    auto it = slos_.find(slo_name);
    if (it != slos_.end()) {
        it->second->is_silenced = true;
        it->second->silence_until = std::chrono::steady_clock::now() + duration;
        LOG_INFO("Silenced alerts for SLO: {} for {}s", slo_name, duration.count());
    }
}

void SLOMonitor::unsilence_alerts(const std::string& slo_name) {
    std::lock_guard<std::mutex> lock(slo_mutex_);
    
    auto it = slos_.find(slo_name);
    if (it != slos_.end()) {
        it->second->is_silenced = false;
        LOG_INFO("Unsilenced alerts for SLO: {}", slo_name);
    }
}

void SLOMonitor::monitoring_loop() {
    LOG_INFO("SLO monitoring loop started");
    
    while (monitoring_.load(std::memory_order_acquire)) {
        auto start_time = std::chrono::steady_clock::now();
        
        {
            std::lock_guard<std::mutex> lock(slo_mutex_);
            
            for (auto& [name, slo_state] : slos_) {
                if (slo_state->definition.enabled) {
                    check_slo(*slo_state);
                }
            }
        }
        
        // Update predictive alerts
        if (config_.enable_predictive_alerting) {
            update_predictions();
        }
        
        auto end_time = std::chrono::steady_clock::now();
        auto check_duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
        
        stats_.slo_checks_performed.fetch_add(slos_.size(), std::memory_order_relaxed);
        stats_.check_duration_ns.store(check_duration_ns, std::memory_order_relaxed);
        
        // Sleep until next check interval
        std::this_thread::sleep_for(config_.check_interval);
    }
    
    LOG_INFO("SLO monitoring loop stopped");
}

void SLOMonitor::check_slo(SLOState& slo_state) {
    auto now = std::chrono::steady_clock::now();
    
    // Calculate current percentile value
    double current_percentile = calculate_percentile_from_histogram(
        slo_state.definition.metric_name, slo_state.definition.target_percentile);
    
    // Update status
    slo_state.status.current_percentile_value = current_percentile;
    slo_state.status.target_percentile_value = slo_state.definition.target_latency_ns / 1e9;  // Convert to seconds
    
    // Check for violation
    bool violation_occurred = evaluate_slo_violation(slo_state, slo_state.status);
    
    if (violation_occurred) {
        slo_state.status.violations_in_window++;
        slo_state.status.last_violation = now;
        slo_state.violation_history.push_back(
            std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count());
        
        stats_.violations_detected.fetch_add(1, std::memory_order_relaxed);
    }
    
    // Update error budget
    update_error_budget(slo_state, violation_occurred);
    
    // Check if we should send an alert
    if (violation_occurred && should_send_alert(slo_state)) {
        std::string message = format_alert_message(slo_state, "SLO violation detected");
        send_alert(slo_state, message);
        slo_state.status.last_alert = now;
    }
    
    slo_state.last_check = now;
}

bool SLOMonitor::evaluate_slo_violation(const SLOState& slo_state, const SLOStatus& current_status) {
    // Convert target latency from nanoseconds to seconds for comparison
    double target_latency_seconds = slo_state.definition.target_latency_ns / 1e9;
    
    return current_status.current_percentile_value > target_latency_seconds;
}

void SLOMonitor::update_error_budget(SLOState& slo_state, bool violation_occurred) {
    if (violation_occurred) {
        slo_state.error_budget.actual_violations++;
    }
    
    // Calculate consumed budget percentage
    if (slo_state.error_budget.allowed_violations > 0) {
        slo_state.error_budget.consumed_budget_percent = 
            (static_cast<double>(slo_state.error_budget.actual_violations) / 
             slo_state.error_budget.allowed_violations) * slo_state.error_budget.total_budget_percent;
    }
    
    slo_state.error_budget.remaining_budget_percent = 
        slo_state.error_budget.total_budget_percent - slo_state.error_budget.consumed_budget_percent;
    
    // Update status
    slo_state.status.error_budget_remaining_percent = slo_state.error_budget.remaining_budget_percent;
    slo_state.status.is_healthy = slo_state.error_budget.remaining_budget_percent > 0;
}

void SLOMonitor::send_alert(const SLOState& slo_state, const std::string& message) {
    if (slo_state.is_silenced && std::chrono::steady_clock::now() < slo_state.silence_until) {
        return;  // Alerts are silenced
    }
    
    std::string severity = slo_state.error_budget.remaining_budget_percent < 10.0 ? "critical" : "warning";
    
    if (config_.alert_config.callback) {
        config_.alert_config.callback(slo_state.definition.name, message);
    }
    
    alert_manager_->send_alert_to_all_channels(message, severity);
    
    stats_.alerts_sent.fetch_add(1, std::memory_order_relaxed);
    
    LOG_WARN("SLO alert sent: {} - {}", slo_state.definition.name, message);
}

bool SLOMonitor::should_send_alert(const SLOState& slo_state) const {
    auto now = std::chrono::steady_clock::now();
    
    // Check cooldown period
    if (slo_state.status.last_alert != std::chrono::steady_clock::time_point{}) {
        auto time_since_last_alert = now - slo_state.status.last_alert;
        if (time_since_last_alert < slo_state.definition.alert_cooldown) {
            return false;
        }
    }
    
    return true;
}

void SLOMonitor::update_predictions() {
    // Simplified predictive analysis - in production you'd use more sophisticated models
    std::lock_guard<std::mutex> lock(slo_mutex_);
    
    for (auto& [name, slo_state] : slos_) {
        double predicted_consumption = predict_budget_consumption(*slo_state);
        
        if (predicted_consumption > config_.prediction_threshold) {
            std::string message = "Predicted SLO budget exhaustion: " + name + 
                                " (predicted: " + std::to_string(predicted_consumption * 100) + "%)";
            
            if (should_send_alert(*slo_state)) {
                send_alert(*slo_state, message);
            }
        }
    }
}

double SLOMonitor::predict_budget_consumption(const SLOState& slo_state) const {
    // Simple linear trend prediction based on recent violation history
    if (slo_state.violation_history.size() < 2) {
        return slo_state.error_budget.consumed_budget_percent / 100.0;
    }
    
    // Calculate violation rate over the last hour
    auto now = std::chrono::steady_clock::now();
    auto one_hour_ago = now - std::chrono::hours(1);
    auto one_hour_ago_seconds = std::chrono::duration_cast<std::chrono::seconds>(one_hour_ago.time_since_epoch()).count();
    
    size_t recent_violations = 0;
    for (auto timestamp : slo_state.violation_history) {
        if (timestamp >= one_hour_ago_seconds) {
            recent_violations++;
        }
    }
    
    // Project current rate forward
    double violations_per_hour = static_cast<double>(recent_violations);
    double hours_in_evaluation_window = std::chrono::duration_cast<std::chrono::hours>(
        slo_state.definition.evaluation_window).count();
    
    double projected_violations = violations_per_hour * hours_in_evaluation_window;
    double projected_consumption = projected_violations / slo_state.error_budget.allowed_violations;
    
    return std::min(1.0, projected_consumption);
}

uint64_t SLOMonitor::calculate_allowed_violations(const SLODefinition& slo, std::chrono::seconds window_duration) const {
    // Simplified calculation - assumes 1 request per second
    // In practice, you'd base this on actual request rate
    uint64_t total_requests = window_duration.count();  // 1 RPS assumption
    return static_cast<uint64_t>(total_requests * (slo.error_budget_percent / 100.0));
}

double SLOMonitor::calculate_percentile_from_histogram(const std::string& metric_name, double percentile) const {
    auto percentile_data = collector_.calculate_percentiles(metric_name);
    
    // Map percentile to the appropriate field
    if (percentile <= 0.5) return percentile_data.p50;
    if (percentile <= 0.95) return percentile_data.p95;
    if (percentile <= 0.99) return percentile_data.p99;
    if (percentile <= 0.999) return percentile_data.p999;
    return percentile_data.p9999;
}

std::string SLOMonitor::format_alert_message(const SLOState& slo_state, const std::string& violation_details) const {
    std::ostringstream oss;
    oss << "SLO Alert: " << slo_state.definition.name << "\n"
        << "Details: " << violation_details << "\n"
        << "Current P" << (slo_state.definition.target_percentile * 100) << ": " 
        << std::fixed << std::setprecision(3) << (slo_state.status.current_percentile_value * 1000) << "ms\n"
        << "Target: " << (slo_state.definition.target_latency_ns / 1e6) << "ms\n"
        << "Error Budget Remaining: " << std::fixed << std::setprecision(1) 
        << slo_state.error_budget.remaining_budget_percent << "%\n"
        << "Violations in Window: " << slo_state.status.violations_in_window;
    
    return oss.str();
}

} // namespace monitor
} // namespace ultra