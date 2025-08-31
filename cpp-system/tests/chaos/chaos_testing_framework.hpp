#pragma once

#include "../framework/test_framework.hpp"
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <chrono>
#include <atomic>
#include <thread>
#include <random>
#include <map>
#include <set>

namespace ultra::testing::chaos {

// Types of chaos experiments
enum class ChaosType {
    // Infrastructure failures
    NETWORK_PARTITION,
    NETWORK_LATENCY,
    NETWORK_PACKET_LOSS,
    DISK_IO_FAILURE,
    DISK_FULL,
    MEMORY_PRESSURE,
    CPU_STRESS,
    
    // Application failures
    PROCESS_KILL,
    THREAD_STARVATION,
    MEMORY_LEAK_INJECTION,
    EXCEPTION_INJECTION,
    TIMEOUT_INJECTION,
    
    // Resource exhaustion
    FILE_DESCRIPTOR_EXHAUSTION,
    CONNECTION_POOL_EXHAUSTION,
    CACHE_INVALIDATION,
    
    // Timing issues
    CLOCK_SKEW,
    RACE_CONDITION_TRIGGER,
    DEADLOCK_INJECTION
};

// Chaos experiment configuration
struct ChaosExperiment {
    std::string name;
    ChaosType type;
    double probability = 0.1;                    // Probability of triggering (0.0-1.0)
    std::chrono::milliseconds duration{5000};    // Duration of the chaos
    std::chrono::milliseconds interval{10000};   // Interval between chaos events
    
    // Target selection
    std::set<std::string> target_components;     // Which components to target
    std::set<int> target_threads;               // Which threads to target
    
    // Experiment-specific parameters
    std::map<std::string, std::string> parameters;
    
    // Custom failure injection function
    std::function<void()> custom_failure_action;
    
    // Recovery function
    std::function<void()> recovery_action;
    
    // Validation function to check system state
    std::function<bool()> validation_function;
};

// Chaos experiment results
struct ChaosExperimentResult {
    std::string experiment_name;
    ChaosType type;
    size_t total_injections = 0;
    size_t successful_injections = 0;
    size_t failed_injections = 0;
    size_t system_recoveries = 0;
    
    struct FailureImpact {
        std::chrono::milliseconds detection_time{0};
        std::chrono::milliseconds recovery_time{0};
        bool system_remained_stable = true;
        std::vector<std::string> observed_effects;
    };
    
    std::vector<FailureImpact> failure_impacts;
    
    struct SystemMetrics {
        double avg_response_time_ms = 0.0;
        double error_rate = 0.0;
        double throughput_degradation = 0.0;
        size_t memory_usage_peak_mb = 0;
        double cpu_usage_peak_percent = 0.0;
    } system_metrics;
    
    bool experiment_passed = false;
    std::string failure_reason;
};

// Network chaos injector
class NetworkChaosInjector {
public:
    struct NetworkFailure {
        std::string interface = "eth0";
        std::chrono::milliseconds latency{0};
        double packet_loss_rate = 0.0;
        double bandwidth_limit_mbps = 0.0;
        bool partition_enabled = false;
        std::vector<std::string> blocked_ips;
    };
    
    void inject_network_latency(const std::string& interface, 
                              std::chrono::milliseconds latency);
    void inject_packet_loss(const std::string& interface, double loss_rate);
    void inject_bandwidth_limit(const std::string& interface, double limit_mbps);
    void inject_network_partition(const std::vector<std::string>& blocked_ips);
    void restore_network(const std::string& interface);

private:
    std::map<std::string, NetworkFailure> active_failures_;
    void execute_tc_command(const std::string& command);
    void execute_iptables_command(const std::string& command);
};

// Resource chaos injector
class ResourceChaosInjector {
public:
    void inject_memory_pressure(size_t pressure_mb, std::chrono::milliseconds duration);
    void inject_cpu_stress(double cpu_percent, std::chrono::milliseconds duration);
    void inject_disk_io_failure(const std::string& path, std::chrono::milliseconds duration);
    void inject_disk_full(const std::string& path, size_t fill_size_mb);
    void inject_fd_exhaustion(size_t max_fds);
    
    void cleanup_all_injections();

private:
    std::vector<std::thread> stress_threads_;
    std::atomic<bool> stop_stress_{false};
    std::vector<std::string> temp_files_;
    
    void memory_stress_worker(size_t size_mb);
    void cpu_stress_worker(double target_percent);
    void disk_stress_worker(const std::string& path);
};

// Application chaos injector
class ApplicationChaosInjector {
public:
    void inject_exception(const std::string& component, 
                         const std::string& exception_type,
                         double probability);
    void inject_timeout(const std::string& component,
                       std::chrono::milliseconds timeout_duration,
                       double probability);
    void inject_memory_leak(size_t leak_size_mb, 
                          std::chrono::milliseconds interval);
    void inject_thread_starvation(const std::string& component,
                                std::chrono::milliseconds duration);
    
    // Hook into allocation/deallocation for memory chaos
    void enable_memory_chaos();
    void disable_memory_chaos();
    
    // Hook into thread creation for thread chaos
    void enable_thread_chaos();
    void disable_thread_chaos();

private:
    std::map<std::string, double> exception_probabilities_;
    std::map<std::string, std::chrono::milliseconds> timeout_injections_;
    std::atomic<bool> memory_chaos_enabled_{false};
    std::atomic<bool> thread_chaos_enabled_{false};
    std::vector<void*> leaked_memory_;
    std::mt19937 rng_{std::random_device{}()};
    
    bool should_inject_failure(const std::string& component);
};

// System monitor for chaos experiments
class ChaosSystemMonitor {
public:
    struct SystemState {
        double cpu_usage_percent = 0.0;
        size_t memory_usage_mb = 0;
        size_t disk_usage_mb = 0;
        size_t network_rx_bytes = 0;
        size_t network_tx_bytes = 0;
        size_t open_file_descriptors = 0;
        size_t thread_count = 0;
        
        // Application-specific metrics
        double response_time_ms = 0.0;
        double error_rate = 0.0;
        size_t active_connections = 0;
        size_t cache_hit_rate = 0;
    };
    
    void start_monitoring();
    void stop_monitoring();
    
    SystemState get_current_state() const;
    std::vector<SystemState> get_history() const;
    
    // Anomaly detection
    bool detect_anomaly(const SystemState& current, const SystemState& baseline) const;
    
    // Alert callbacks
    using AlertCallback = std::function<void(const std::string&, const SystemState&)>;
    void set_alert_callback(AlertCallback callback);

private:
    std::atomic<bool> monitoring_active_{false};
    std::thread monitoring_thread_;
    mutable std::mutex state_mutex_;
    std::vector<SystemState> state_history_;
    AlertCallback alert_callback_;
    
    void monitoring_loop();
    SystemState collect_system_metrics();
};

// Main chaos testing engine
class ChaosTestingEngine {
public:
    explicit ChaosTestingEngine();
    ~ChaosTestingEngine();
    
    // Experiment management
    void register_experiment(const ChaosExperiment& experiment);
    void remove_experiment(const std::string& name);
    
    // Execution control
    void start_chaos_testing();
    void stop_chaos_testing();
    void pause_chaos_testing();
    void resume_chaos_testing();
    
    // Run specific experiment
    ChaosExperimentResult run_experiment(const std::string& experiment_name,
                                       std::chrono::seconds duration);
    
    // Run experiment suite
    struct ChaosTestSuite {
        std::string name;
        std::vector<std::string> experiment_names;
        std::chrono::seconds total_duration{300}; // 5 minutes default
        bool stop_on_failure = false;
    };
    
    struct ChaosTestSuiteResult {
        std::string suite_name;
        std::vector<ChaosExperimentResult> experiment_results;
        size_t passed_experiments = 0;
        size_t failed_experiments = 0;
        bool overall_success = false;
        std::chrono::milliseconds total_duration{0};
    };
    
    ChaosTestSuiteResult run_test_suite(const ChaosTestSuite& suite);
    
    // Monitoring and reporting
    void set_system_monitor(std::shared_ptr<ChaosSystemMonitor> monitor);
    std::vector<ChaosExperimentResult> get_experiment_history() const;
    
    // Configuration
    void set_global_probability(double probability);
    void set_safety_limits(const std::map<std::string, double>& limits);
    void enable_auto_recovery(bool enabled);

private:
    std::map<std::string, ChaosExperiment> experiments_;
    std::unique_ptr<NetworkChaosInjector> network_injector_;
    std::unique_ptr<ResourceChaosInjector> resource_injector_;
    std::unique_ptr<ApplicationChaosInjector> app_injector_;
    std::shared_ptr<ChaosSystemMonitor> system_monitor_;
    
    std::atomic<bool> chaos_active_{false};
    std::atomic<bool> chaos_paused_{false};
    std::thread chaos_thread_;
    
    mutable std::mutex results_mutex_;
    std::vector<ChaosExperimentResult> experiment_history_;
    
    double global_probability_ = 1.0;
    std::map<std::string, double> safety_limits_;
    bool auto_recovery_enabled_ = true;
    
    void chaos_execution_loop();
    void execute_experiment(const ChaosExperiment& experiment);
    bool check_safety_limits() const;
    void emergency_recovery();
};

// Predefined chaos experiments
class ChaosExperimentLibrary {
public:
    // Network chaos experiments
    static ChaosExperiment network_partition_experiment();
    static ChaosExperiment network_latency_experiment();
    static ChaosExperiment packet_loss_experiment();
    
    // Resource chaos experiments
    static ChaosExperiment memory_pressure_experiment();
    static ChaosExperiment cpu_stress_experiment();
    static ChaosExperiment disk_io_failure_experiment();
    
    // Application chaos experiments
    static ChaosExperiment exception_injection_experiment();
    static ChaosExperiment timeout_injection_experiment();
    static ChaosExperiment memory_leak_experiment();
    
    // Complex scenarios
    static ChaosExperiment cascading_failure_experiment();
    static ChaosExperiment split_brain_experiment();
    static ChaosExperiment resource_exhaustion_experiment();
    
    // Load-specific experiments
    static ChaosExperiment high_load_failure_experiment();
    static ChaosExperiment cache_invalidation_storm_experiment();
    static ChaosExperiment connection_pool_exhaustion_experiment();
};

// Chaos testing utilities
class ChaosTestingUtils {
public:
    // System state validation
    static bool validate_system_stability(const ChaosSystemMonitor::SystemState& state);
    static bool validate_performance_degradation(
        const ChaosSystemMonitor::SystemState& baseline,
        const ChaosSystemMonitor::SystemState& current,
        double max_degradation_percent);
    
    // Report generation
    static void generate_chaos_report(const std::vector<ChaosExperimentResult>& results,
                                    const std::string& output_file);
    static void generate_timeline_report(const std::vector<ChaosExperimentResult>& results,
                                       const std::string& output_file);
    
    // Experiment analysis
    static double calculate_system_resilience_score(
        const std::vector<ChaosExperimentResult>& results);
    static std::vector<std::string> identify_weak_points(
        const std::vector<ChaosExperimentResult>& results);
    
    // Safety checks
    static bool is_safe_to_run_experiment(const ChaosExperiment& experiment,
                                        const ChaosSystemMonitor::SystemState& current_state);
};

} // namespace ultra::testing::chaos