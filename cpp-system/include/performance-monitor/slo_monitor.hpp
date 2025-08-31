#pragma once

#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

namespace ultra {
namespace monitor {

class MetricsCollector;

/**
 * Real-time SLO (Service Level Objective) monitoring with alerting
 * Provides microsecond-precision SLO violation detection and alerting
 */
class SLOMonitor {
public:
    struct SLODefinition {
        std::string name;
        std::string metric_name;
        double target_percentile = 0.99;  // P99
        uint64_t target_latency_ns = 1000000;  // 1ms
        std::chrono::seconds evaluation_window{60};
        std::chrono::seconds alert_cooldown{300};  // 5 minutes
        double error_budget_percent = 1.0;  // 1% error budget
        bool enabled = true;
    };

    struct AlertConfig {
        std::function<void(const std::string& slo_name, const std::string& message)> callback;
        std::string webhook_url;
        std::string slack_channel;
        std::string email_recipients;
        bool enable_webhook = false;
        bool enable_slack = false;
        bool enable_email = false;
    };

    struct Config {
        std::chrono::milliseconds check_interval{1000};  // 1 second
        size_t max_slos = 100;
        bool enable_predictive_alerting = true;
        double prediction_threshold = 0.8;  // Alert when 80% of budget consumed
        AlertConfig alert_config;
    };

    explicit SLOMonitor(const Config& config, MetricsCollector& collector);
    ~SLOMonitor();

    // SLO management
    void register_slo(const SLODefinition& slo);
    void unregister_slo(const std::string& name);
    void update_slo(const SLODefinition& slo);
    std::vector<SLODefinition> get_all_slos() const;

    // Monitoring control
    void start_monitoring();
    void stop_monitoring();
    bool is_monitoring() const noexcept { return monitoring_.load(std::memory_order_acquire); }

    // SLO status and metrics
    struct SLOStatus {
        std::string name;
        bool is_healthy = true;
        double current_percentile_value = 0.0;
        double target_percentile_value = 0.0;
        double error_budget_remaining_percent = 100.0;
        uint64_t violations_in_window = 0;
        uint64_t total_requests_in_window = 0;
        std::chrono::steady_clock::time_point last_violation;
        std::chrono::steady_clock::time_point last_alert;
        std::string status_message;
    };

    SLOStatus get_slo_status(const std::string& name) const;
    std::vector<SLOStatus> get_all_slo_status() const;

    // Error budget tracking
    struct ErrorBudget {
        std::string slo_name;
        double total_budget_percent;
        double consumed_budget_percent;
        double remaining_budget_percent;
        uint64_t allowed_violations;
        uint64_t actual_violations;
        std::chrono::steady_clock::time_point budget_reset_time;
        std::chrono::seconds time_to_budget_reset;
    };

    ErrorBudget get_error_budget(const std::string& slo_name) const;

    // Predictive alerting
    struct PredictiveAlert {
        std::string slo_name;
        double predicted_budget_consumption_percent;
        std::chrono::seconds time_to_budget_exhaustion;
        double confidence_level;
        std::string prediction_model;
    };

    std::vector<PredictiveAlert> get_predictive_alerts() const;

    // Alert management
    void trigger_manual_alert(const std::string& slo_name, const std::string& message);
    void silence_alerts(const std::string& slo_name, std::chrono::seconds duration);
    void unsilence_alerts(const std::string& slo_name);

    // Performance statistics
    struct MonitorStats {
        std::atomic<uint64_t> slo_checks_performed{0};
        std::atomic<uint64_t> violations_detected{0};
        std::atomic<uint64_t> alerts_sent{0};
        std::atomic<uint64_t> check_duration_ns{0};
        std::atomic<uint64_t> prediction_accuracy_percent{0};
    };

    const MonitorStats& get_stats() const noexcept { return stats_; }

private:
    Config config_;
    MetricsCollector& collector_;
    MonitorStats stats_;
    std::atomic<bool> monitoring_{false};

    // SLO storage and management
    struct SLOState {
        SLODefinition definition;
        SLOStatus status;
        ErrorBudget error_budget;
        std::vector<uint64_t> violation_history;
        std::chrono::steady_clock::time_point last_check;
        bool is_silenced = false;
        std::chrono::steady_clock::time_point silence_until;
    };

    mutable std::mutex slo_mutex_;
    std::unordered_map<std::string, std::unique_ptr<SLOState>> slos_;

    // Monitoring thread
    std::unique_ptr<std::thread> monitoring_thread_;
    void monitoring_loop();

    // SLO evaluation
    void check_slo(SLOState& slo_state);
    bool evaluate_slo_violation(const SLOState& slo_state, const SLOStatus& current_status);
    void update_error_budget(SLOState& slo_state, bool violation_occurred);

    // Alerting system
    class AlertManager;
    std::unique_ptr<AlertManager> alert_manager_;

    void send_alert(const SLOState& slo_state, const std::string& message);
    bool should_send_alert(const SLOState& slo_state) const;

    // Predictive analysis
    class PredictiveAnalyzer;
    std::unique_ptr<PredictiveAnalyzer> predictive_analyzer_;

    void update_predictions();
    double predict_budget_consumption(const SLOState& slo_state) const;

    // Utility methods
    uint64_t calculate_allowed_violations(const SLODefinition& slo, 
                                        std::chrono::seconds window_duration) const;
    double calculate_percentile_from_histogram(const std::string& metric_name, 
                                             double percentile) const;
    std::string format_alert_message(const SLOState& slo_state, 
                                   const std::string& violation_details) const;
};

/**
 * Alert delivery system with multiple channels
 */
class AlertManager {
public:
    struct WebhookConfig {
        std::string url;
        std::string auth_token;
        std::chrono::seconds timeout{10};
        int max_retries = 3;
    };

    struct SlackConfig {
        std::string webhook_url;
        std::string channel;
        std::string username = "SLO Monitor";
        std::string icon_emoji = ":warning:";
    };

    struct EmailConfig {
        std::string smtp_server;
        uint16_t smtp_port = 587;
        std::string username;
        std::string password;
        std::string from_address;
        std::vector<std::string> to_addresses;
        bool use_tls = true;
    };

    explicit AlertManager();
    ~AlertManager();

    // Configuration
    void configure_webhook(const WebhookConfig& config);
    void configure_slack(const SlackConfig& config);
    void configure_email(const EmailConfig& config);

    // Alert delivery
    bool send_webhook_alert(const std::string& message, const std::string& severity = "warning");
    bool send_slack_alert(const std::string& message, const std::string& severity = "warning");
    bool send_email_alert(const std::string& subject, const std::string& message);

    // Batch operations
    void send_alert_to_all_channels(const std::string& message, const std::string& severity = "warning");

    // Statistics
    struct AlertStats {
        std::atomic<uint64_t> webhook_alerts_sent{0};
        std::atomic<uint64_t> slack_alerts_sent{0};
        std::atomic<uint64_t> email_alerts_sent{0};
        std::atomic<uint64_t> delivery_failures{0};
        std::atomic<uint64_t> delivery_duration_ms{0};
    };

    const AlertStats& get_stats() const noexcept { return stats_; }

private:
    WebhookConfig webhook_config_;
    SlackConfig slack_config_;
    EmailConfig email_config_;
    AlertStats stats_;

    // HTTP client for webhook/Slack delivery
    class HttpClient;
    std::unique_ptr<HttpClient> http_client_;

    // SMTP client for email delivery
    class SmtpClient;
    std::unique_ptr<SmtpClient> smtp_client_;

    // Helper methods
    std::string format_slack_payload(const std::string& message, const std::string& severity);
    std::string format_webhook_payload(const std::string& message, const std::string& severity);
    bool deliver_with_retry(std::function<bool()> delivery_func, int max_retries);
};

} // namespace monitor
} // namespace ultra